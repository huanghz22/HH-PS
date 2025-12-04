import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, Tuple, Union
import time
import math
import numpy as np
try:
    from kernels.map_1d_4d import expand_1d_to_4d_hashed, backward_expand_1d_to_4d_hashed
    from kernels.conv2d_kernels import conv2d_hashed_forward_kernel, conv2d_multi_hash_forward_kernel, conv2d_masked_multi_hash_forward_kernel
    from types_my import Context, Device
    from kernels import global_var
    
except:
    from .kernels.map_1d_4d import expand_1d_to_4d_hashed, backward_expand_1d_to_4d_hashed
    from .kernels.conv2d_kernels import conv2d_hashed_forward_kernel, conv2d_multi_hash_forward_kernel, conv2d_masked_multi_hash_forward_kernel
    from .types_my import Context, Device
    from .kernels import global_var


def conv2d_hashed_triton(
    input_tensor: torch.Tensor,
    shared_weight: torch.Tensor,
    weight_shape: tuple,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int]] = 0,
    groups: int = 1,
    hashw_block_size: int = 1024,
    hash_seed: int = 0,
    split_ratio: float = 0.5,
    collision_m: Optional[torch.Tensor] = None,  
) -> torch.Tensor:
    batch_dim, in_channels, in_height, in_width = input_tensor.shape
    out_channels, _, kernel_size, kernel_size = weight_shape

    if isinstance(stride, int):
        stride_h, stride_w = stride, stride
    else:
        stride_h, stride_w = stride

    if isinstance(padding, int):
        pad_h, pad_w = padding, padding
    else:
        pad_h, pad_w = padding

    out_height = (in_height + 2 * pad_h - kernel_size) // stride_h + 1
    out_width = (in_width + 2 * pad_w - kernel_size) // stride_w + 1

    output = torch.empty(
        batch_dim, out_channels, out_height, out_width,
        device=input_tensor.device, dtype=input_tensor.dtype
    )

    def grid(meta):
        out_group_dim = out_channels // groups
        return (
            triton.cdiv(batch_dim * out_height * out_width, meta['BLOCK_SIZE_BATCH_HEIGHT_WIDTH']),
            triton.cdiv(out_group_dim, meta['BLOCK_SIZE_OUT_FEAT']),
            groups
        )

    input_batch_stride, input_in_feat_stride, input_height_stride, input_width_stride = input_tensor.stride()
    output_batch_stride, output_out_feat_stride, output_height_stride, output_width_stride = output.stride()

    if split_ratio == 0 or split_ratio == 1:
        conv2d_hashed_forward_kernel[grid](
            input_tensor, shared_weight, output,
            batch_dim, in_channels, in_height, in_width,
            out_channels, out_height, out_width,
            input_batch_stride, input_in_feat_stride, input_height_stride, input_width_stride,
            output_batch_stride, output_out_feat_stride, output_height_stride, output_width_stride,
            shared_weight_size=shared_weight.numel(),
            kernel_height=kernel_size, kernel_width=kernel_size,
            stride_height=stride_h, stride_width=stride_w,
            padding_height=pad_h, padding_width=pad_w,
            groups=groups,
            HASH_BLOCK_SIZE=hashw_block_size,
            HASH_SEED=hash_seed,
            fp16=(input_tensor.dtype == torch.float16),
        )
    elif split_ratio == 0.5:
        split_point = int(shared_weight.numel() * split_ratio)
        split_point = max(0, min(split_point, shared_weight.numel()))

        shared_weight_1 = shared_weight[:split_point]
        shared_weight_2 = shared_weight[split_point:]

        conv2d_multi_hash_forward_kernel[grid](
            input_tensor, shared_weight_1, shared_weight_2, output,
            batch_dim, in_channels, in_height, in_width,
            out_channels, out_height, out_width,
            input_batch_stride, input_in_feat_stride, input_height_stride, input_width_stride,
            output_batch_stride, output_out_feat_stride, output_height_stride, output_width_stride,
            shared_weight_size_1=shared_weight_1.numel(),
            shared_weight_size_2=shared_weight_2.numel(),
            kernel_height=kernel_size, kernel_width=kernel_size,
            stride_height=stride_h, stride_width=stride_w,
            padding_height=pad_h, padding_width=pad_w,
            groups=groups,
            HASH_BLOCK_SIZE=hashw_block_size,
            HASH_SEED=hash_seed,
            fp16=(input_tensor.dtype == torch.float16),
        )
    else:
        split_point = int(shared_weight.numel() * split_ratio)
        split_point = max(0, min(split_point, shared_weight.numel()))

        shared_weight_1 = shared_weight[:split_point]
        shared_weight_2 = shared_weight[split_point:]

        conv2d_masked_multi_hash_forward_kernel[grid](
            input_tensor, shared_weight_1, shared_weight_2, collision_m, output,
            batch_dim, in_channels, in_height, in_width,
            out_channels, out_height, out_width,
            input_batch_stride, input_in_feat_stride, input_height_stride, input_width_stride,
            output_batch_stride, output_out_feat_stride, output_height_stride, output_width_stride,
            shared_weight_size_1=shared_weight_1.numel(),
            shared_weight_size_2=shared_weight_2.numel(),
            kernel_height=kernel_size, kernel_width=kernel_size,
            stride_height=stride_h, stride_width=stride_w,
            padding_height=pad_h, padding_width=pad_w,
            groups=groups,
            HASH_BLOCK_SIZE=hashw_block_size,
            HASH_SEED=hash_seed,
            fp16=(input_tensor.dtype == torch.float16),
        )

    if bias is not None:
        output += bias.view(1, -1, 1, 1)

    return output

class HashConv2d(nn.Module):
    def __init__(
        self,
        shared_weight: Tensor,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        groups: int = 1,
        bias: bool = True,
        hashw_block_size: int = 64,
        dtype: torch.dtype = torch.float32,
        hash_seed: Optional[int] = None,
        layer_factor: bool = True,
        split_ratio: float = 0.5,
        grad_type: str = 'avg', 
    ):
        super().__init__()
        assert  split_ratio >= 0 and split_ratio < 1, f'split_ratio must be in [0, 1), got {split_ratio}'
        if split_ratio < 0.5 and split_ratio != 0:
            split_ratio = 1 - split_ratio   

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding if isinstance(padding, tuple) else (padding, padding)
        self.groups = groups
        self.hashw_block_size = hashw_block_size
        self.shared_weight = shared_weight
        self.weight_shape = (out_channels, in_channels // groups, *self.kernel_size)
        self.split_ratio = split_ratio
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
                collision_shared_weight = backward_expand_1d_to_4d_hashed(
                    c_virtual_weight, shared_weight.numel(), hashw_block_size, self.hash_seed)
                global_var._collision_dist += collision_shared_weight
            else:
                split_point = int(shared_weight.numel() * split_ratio)
                split_point = max(0, min(split_point, shared_weight.numel()))

                collision_shared_weight_1 = backward_expand_1d_to_4d_hashed(
                    c_virtual_weight, split_point, hashw_block_size, self.hash_seed)
                collision_shared_weight_2 = backward_expand_1d_to_4d_hashed(
                    c_virtual_weight, shared_weight.numel() - split_point, hashw_block_size, self.hash_seed + 1)

                global_var._collision_dist[:split_point] += collision_shared_weight_1
                global_var._collision_dist[split_point:] += collision_shared_weight_2

        if bias:
            self.bias = nn.Parameter(torch.randn(out_channels, device=shared_weight.device, dtype=dtype))
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

        out = conv2d_hashed_triton(
            input, self.shared_weight, self.weight_shape, self.bias,
            self.stride, self.padding, self.groups, self.hashw_block_size,
            self.hash_seed, self.split_ratio, self.collision_m)

        if self.layer_factor is not None:
            out = out * self.layer_factor
        return out
