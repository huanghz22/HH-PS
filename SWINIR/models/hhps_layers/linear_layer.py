import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, Tuple, Union

try:
    from .kernels.map_1d_2d import expand_1d_to_2d_hashed, backward_expand_1d_to_2d_hashed
    from .kernels.linear_kernels import linear_hashed_forward_kernel, linear_multi_hash_forward_kernel, linear_masked_multi_hash_forward_kernel
    from .types_my import Context, Device
    from .kernels import global_var

except:
    from kernels.map_1d_2d import expand_1d_to_2d_hashed, backward_expand_1d_to_2d_hashed
    from kernels.linear_kernels import linear_hashed_forward_kernel, linear_multi_hash_forward_kernel, linear_masked_multi_hash_forward_kernel
    from types_my import Context, Device
    from kernels import global_var



def linear_hashed_triton(
    input_tensor: torch.Tensor,
    shared_weight: torch.Tensor,
    weight_shape: tuple,
    bias: Optional[torch.Tensor] = None,
    hashw_block_size: int = 64,
    hash_seed: int = 0,
    split_ratio: float = 0.5,
    collision_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    input_shape = input_tensor.shape
    is_3d = input_tensor.dim() == 3
    
    if is_3d:
        batch_size, seq_len, in_features = input_shape
        input_2d = input_tensor.view(-1, in_features)
        M = batch_size * seq_len
    else:
        M, in_features = input_shape
        input_2d = input_tensor
    
    out_features, _ = weight_shape
    N, K = out_features, in_features
    
    output_2d = torch.empty(M, N, device=input_tensor.device, dtype=input_tensor.dtype)
    
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']),)
    
    fp16 = (input_tensor.dtype == torch.float16)

    if split_ratio == 0 or split_ratio == 1:
        # Single-hash mode
        linear_hashed_forward_kernel[grid](
            input_2d, shared_weight, output_2d,
            M, K, N,
            input_2d.stride(0), input_2d.stride(1),
            output_2d.stride(0), output_2d.stride(1),
            shared_weight.numel(),
            hashw_block_size, hash_seed,
            fp16,
        )
    elif split_ratio == 0.5:
        split_point = int(shared_weight.numel() * split_ratio)
        shared_weight_1 = shared_weight[:split_point]
        shared_weight_2 = shared_weight[split_point:]
        
        linear_multi_hash_forward_kernel[grid](
            input_2d, shared_weight_1, shared_weight_2, output_2d,
            M, K, N,
            input_2d.stride(0), input_2d.stride(1),
            output_2d.stride(0), output_2d.stride(1),
            shared_weight_1.numel(), shared_weight_2.numel(),
            hashw_block_size, hash_seed,
            fp16,
        )
    else:
        split_point = int(shared_weight.numel() * split_ratio)
        shared_weight_1 = shared_weight[:split_point]
        shared_weight_2 = shared_weight[split_point:]
        
        linear_masked_multi_hash_forward_kernel[grid](
            input_2d, shared_weight_1, shared_weight_2, collision_mask, output_2d,
            M, K, N,
            input_2d.stride(0), input_2d.stride(1),
            output_2d.stride(0), output_2d.stride(1),
            shared_weight_1.numel(), shared_weight_2.numel(),
            hashw_block_size, hash_seed,
            fp16,
        )
    output = output_2d.view(*input_shape[:-1], N) if is_3d else output_2d
    
    if bias is not None:
        output += bias
    
    return output

class HashLinear(nn.Module):
    def __init__(
        self,
        shared_weight: Tensor,
        in_features: int,
        out_features: int,
        bias: bool = True,
        hashw_block_size: int = 64,
        dtype: torch.dtype = torch.float32,
        hash_seed: Optional[int] = None,
        layer_factor: bool = True,
        split_ratio: float = 0.5,
        grad_type: str = 'avg',
        **kwargs,
    ):
        """
        Hash-based fully connected layer that achieves parameter compression through
        a shared parameter pool (memory vector) and hash mapping.

        Description:
            - Uses a 1D shared parameter pool (memory vector), mapping to 2D weight
              matrix via hash functions
            - Supports single-hash mode (split_ratio=0 or 1) and hybrid-hash mode
            - During inference, directly uses Triton linear kernels with integrated
              hash mapping logic

        Args:
            shared_weight (Tensor): 1D shared parameter pool (memory vector), shared
                across all hash layers
            in_features (int): Input feature dimension
            out_features (int): Output feature dimension
            bias (bool): Whether to add a learnable bias term, default is True
            hashw_block_size (int): Hash block size, controls the granularity of hash
                mapping, default is 64
            dtype (torch.dtype): Parameter data type, default is torch.float32
            hash_seed (Optional[int]): Hash seed for generating different hash mappings;
                automatically assigned if None
            layer_factor (bool): Whether to use a learnable layer scaling factor,
                default is True
            split_ratio (float): Weight pool split ratio, controls multi-hash mode
            grad_type (str): Gradient computation type, 'avg' means averaging gradients
                by collision count, default is 'avg'
        """
        super().__init__()
        assert 0 <= split_ratio <= 1, f'split_ratio must be between [0, 1], but got {split_ratio}'
        self.split_ratio = split_ratio
        if 0 < self.split_ratio < 0.5: 
            self.split_ratio = 1 - self.split_ratio

        self.in_features = in_features
        self.out_features = out_features
        self.hashw_block_size = hashw_block_size
        self.shared_weight = shared_weight
        self.weight_shape = (out_features, in_features)
        self.grad_type = grad_type  
        
        if hash_seed is None:
            self.hash_seed = global_var._global_layer_counter
            global_var._global_layer_counter += 100
        else:
            self.hash_seed = hash_seed

        need_collision_dist = (self.grad_type == 'avg' or
                             (self.split_ratio != 0 and self.split_ratio != 0.5 and self.split_ratio != 1))

        if need_collision_dist:
            if global_var._collision_dist is None:
                global_var._collision_dist = torch.zeros(shared_weight.numel(), device=shared_weight.device)

            c_virtual_weight = torch.ones(self.weight_shape, device=shared_weight.device)

            if self.split_ratio == 0 or self.split_ratio == 1:
                collision_shared_weight = backward_expand_1d_to_2d_hashed(
                    c_virtual_weight, shared_weight.numel(), hashw_block_size, self.hash_seed)
                global_var._collision_dist += collision_shared_weight
            else:
                split_point = int(shared_weight.numel() * self.split_ratio)
                split_point = max(0, min(split_point, shared_weight.numel()))

                collision_shared_weight_1 = backward_expand_1d_to_2d_hashed(
                    c_virtual_weight, split_point, hashw_block_size, self.hash_seed)
                collision_shared_weight_2 = backward_expand_1d_to_2d_hashed(
                    c_virtual_weight, shared_weight.numel() - split_point, hashw_block_size, self.hash_seed + 1)
                global_var._collision_dist[:split_point] += collision_shared_weight_1
                global_var._collision_dist[split_point:] += collision_shared_weight_2

        if bias:
            self.bias = nn.Parameter(torch.randn(out_features, device=shared_weight.device, dtype=dtype))
        else:
            self.register_parameter('bias', None)
        
        if layer_factor:
            self.layer_factor = nn.Parameter(torch.randn(1, device=shared_weight.device, dtype=dtype))
        else:
            self.register_parameter('layer_factor', None)

    def forward(self, input: Tensor) -> Tensor:
        if not hasattr(self, 'collision_m'):
            if self.split_ratio in [0, 0.5, 1]:
                setattr(self, 'collision_m', None)
                if self.grad_type == 'avg':
                    setattr(self, 'collision_dist', global_var._collision_dist)
                else:
                    setattr(self, 'collision_dist', None)
            else:
                setattr(self, 'collision_dist', global_var._collision_dist)
                split_point = int(self.shared_weight.numel() * self.split_ratio)
                collision_dist_part1 = global_var._collision_dist[:split_point]
                mask_ratio = int(self.shared_weight.numel() * (1 - self.split_ratio)) / split_point
                threshold = torch.quantile(collision_dist_part1.float(), 1.0 - mask_ratio)
                collision_m = (collision_dist_part1 > threshold).to(self.shared_weight.device)
                setattr(self, 'collision_m', collision_m)
        
        out = linear_hashed_triton(
            input, self.shared_weight, self.weight_shape, self.bias,
            self.hashw_block_size, self.hash_seed, self.split_ratio, self.collision_m)
            
        if self.layer_factor is not None:
            out = out * self.layer_factor
        return out 