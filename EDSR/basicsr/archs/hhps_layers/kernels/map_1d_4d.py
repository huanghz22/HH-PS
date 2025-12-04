import torch
import triton
import triton.language as tl
import time

BLOCK_SIZE = 1024

@triton.jit
def fast_hash_with_seed(n, seed):
    n = n ^ seed
    n = (n ^ (n >> 16)) * 0x45d9f3b
    n = (n ^ (n >> 15)) * 0x119de1
    n = n ^ (n >> 16)
    return n & 0x7FFFFFFF  # Ensure the result is positive

@triton.jit
def expand_1d_to_4d_hashed_kernel(
    shared_weight_ptr, weight_4d_ptr,
    shared_weight_size: tl.constexpr, total_elements_4d: tl.constexpr,
    HASH_BLOCK_SIZE: tl.constexpr, BLOCK_SIZE: tl.constexpr,
    HASH_SEED: tl.constexpr,  
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    linear_4d_indices = offsets
    
    weight_block_ids = linear_4d_indices // HASH_BLOCK_SIZE
    weight_offsets_in_block = linear_4d_indices % HASH_BLOCK_SIZE
    weight_hashes = fast_hash_with_seed(weight_block_ids, HASH_SEED)  
    max_start_offset = shared_weight_size - HASH_BLOCK_SIZE
    weight_block_starts = weight_hashes % max_start_offset
    source_1d_indices = weight_block_starts + weight_offsets_in_block

    mask = (source_1d_indices < shared_weight_size) & (linear_4d_indices < total_elements_4d)
    shared_data = tl.load(shared_weight_ptr + source_1d_indices, mask=mask, other=0.0)
    tl.store(weight_4d_ptr + linear_4d_indices, shared_data, mask=mask)

@triton.jit
def backward_expand_1d_to_4d_hashed_kernel(
    grad_4d_ptr,                
    grad_1d_ptr,               
    shared_weight_size: tl.constexpr,
    total_elements_4d: tl.constexpr,
    HASH_BLOCK_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    HASH_SEED: tl.constexpr,  
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    linear_4d_indices = offsets
    mask_4d = linear_4d_indices < total_elements_4d
    grad_4d_block = tl.load(grad_4d_ptr + linear_4d_indices, mask=mask_4d, other=0.0)

    weight_block_ids = linear_4d_indices // HASH_BLOCK_SIZE
    weight_offsets_in_block = linear_4d_indices % HASH_BLOCK_SIZE
    weight_hashes = fast_hash_with_seed(weight_block_ids, HASH_SEED)  
    max_start_offset = shared_weight_size - HASH_BLOCK_SIZE
    weight_block_starts = weight_hashes % max_start_offset
    dest_1d_indices = weight_block_starts + weight_offsets_in_block
    tl.atomic_add(grad_1d_ptr + dest_1d_indices, grad_4d_block, mask=mask_4d)

def expand_1d_to_4d_hashed(shared_weight: torch.Tensor, target_shape: tuple, hash_block_size: int = 1024, hash_seed: int = 0) -> torch.Tensor:
    assert shared_weight.is_cuda and shared_weight.dim() == 1
    total_elements_4d = torch.prod(torch.tensor(target_shape)).item()
    weight_4d = torch.empty(total_elements_4d, device=shared_weight.device, dtype=shared_weight.dtype)
    
    grid = lambda meta: (triton.cdiv(total_elements_4d, meta['BLOCK_SIZE']),)

    expand_1d_to_4d_hashed_kernel[grid](
        shared_weight, weight_4d,
        shared_weight_size=shared_weight.numel(),
        total_elements_4d=total_elements_4d,
        HASH_BLOCK_SIZE=hash_block_size, 
        BLOCK_SIZE=BLOCK_SIZE,  
        HASH_SEED=hash_seed,  
    )
    return weight_4d.reshape(target_shape)

def backward_expand_1d_to_4d_hashed(grad_4d: torch.Tensor, shared_weight_size: int, hash_block_size: int = 1024, hash_seed: int = 0) -> torch.Tensor:
    assert grad_4d.is_cuda
    total_elements_4d = grad_4d.numel()
    
    grad_1d = torch.zeros(shared_weight_size, device=grad_4d.device, dtype=grad_4d.dtype)
    grad_4d_flat = grad_4d.flatten()

    grid = lambda meta: (triton.cdiv(total_elements_4d, meta['BLOCK_SIZE']),)
    
    backward_expand_1d_to_4d_hashed_kernel[grid](
        grad_4d_flat, grad_1d,
        shared_weight_size=shared_weight_size,
        total_elements_4d=total_elements_4d,
        HASH_BLOCK_SIZE=hash_block_size, 
        BLOCK_SIZE=BLOCK_SIZE,  
        HASH_SEED=hash_seed, 
    )
    return grad_1d