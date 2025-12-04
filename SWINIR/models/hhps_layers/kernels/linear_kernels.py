import triton
import triton.language as tl
try:
    from .utils import allow_tf32
except (ImportError, ModuleNotFoundError):
    def allow_tf32():
        return True

def get_autotune_config():
    return [
        # Configs for larger matrices
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        # # Configs for larger K dimension
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
    ]

@triton.jit
def fast_hash_with_seed(n, seed):
    n = n ^ seed
    n = (n ^ (n >> 16)) * 0x45d9f3b
    n = (n ^ (n >> 15)) * 0x119de1
    n = n ^ (n >> 16)
    return n & 0x7FFFFFFF

@triton.autotune(configs=get_autotune_config(), key=['M', 'N', 'K', 'fp16'])
@triton.heuristics({'tf32': lambda _: allow_tf32()})
@triton.jit
def linear_masked_multi_hash_forward_kernel(
    input_ptr, shared_weight_ptr_1, shared_weight_ptr_2, collision_mask_ptr, output_ptr,
    M, K, N,  
    input_stride_m, input_stride_k,
    output_stride_m, output_stride_n,
    shared_weight_size_1: tl.constexpr, shared_weight_size_2: tl.constexpr,
    HASH_BLOCK_SIZE: tl.constexpr, HASH_SEED: tl.constexpr,
    fp16: tl.constexpr, tf32: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    input_ptrs = input_ptr + (offs_m[:, None] * input_stride_m + offs_k[None, :] * input_stride_k)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    max_start_offset_1 = tl.maximum(shared_weight_size_1 - HASH_BLOCK_SIZE, 1)
    max_start_offset_2 = tl.maximum(shared_weight_size_2 - HASH_BLOCK_SIZE, 1)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        input_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        input_block = tl.load(input_ptrs, mask=input_mask, other=0.0)
        
        k_block_offset = k * BLOCK_SIZE_K + offs_k
        weight_2d_indices = offs_n[:, None] * K + k_block_offset[None, :]
        
        weight_block_ids = weight_2d_indices // HASH_BLOCK_SIZE
        weight_offsets_in_block = weight_2d_indices % HASH_BLOCK_SIZE
        
        weight_hashes_1 = fast_hash_with_seed(weight_block_ids, HASH_SEED)
        weight_block_starts_1 = weight_hashes_1 % max_start_offset_1
        weight_1d_indices_1 = weight_block_starts_1 + weight_offsets_in_block
        
        weight_mask_base = (offs_n[:, None] < N) & (k_block_offset[None, :] < K)
        weight_mask_1 = weight_mask_base & (weight_1d_indices_1 < shared_weight_size_1)
        weight_block_1 = tl.load(shared_weight_ptr_1 + weight_1d_indices_1, mask=weight_mask_1, other=0.0)

        collision_mask_block = tl.load(collision_mask_ptr + weight_1d_indices_1, mask=weight_mask_1, other=0.0)
        
        weight_hashes_2 = fast_hash_with_seed(weight_block_ids, HASH_SEED + 1)
        weight_block_starts_2 = weight_hashes_2 % max_start_offset_2
        weight_1d_indices_2 = weight_block_starts_2 + weight_offsets_in_block
        
        weight_mask_2 = weight_mask_base & (weight_1d_indices_2 < shared_weight_size_2)
        weight_block_2 = tl.load(shared_weight_ptr_2 + weight_1d_indices_2, mask=weight_mask_2, other=0.0)
        
        weight_block = weight_block_1 + weight_block_2 * collision_mask_block
        
        if fp16:
            input_block = input_block.to(tl.float16)
            weight_block = weight_block.to(tl.float16)
        
        accumulator += tl.dot(input_block, tl.trans(weight_block), allow_tf32=tf32)
        input_ptrs += BLOCK_SIZE_K * input_stride_k
    
    output_ptrs = output_ptr + offs_m[:, None] * output_stride_m + offs_n[None, :] * output_stride_n
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=output_mask)


@triton.autotune(configs=get_autotune_config(), key=['M', 'N', 'K', 'fp16'])
@triton.heuristics({'tf32': lambda _: allow_tf32()})
@triton.jit
def linear_multi_hash_forward_kernel(
    input_ptr, shared_weight_ptr_1, shared_weight_ptr_2, output_ptr,
    M, K, N,
    input_stride_m, input_stride_k,
    output_stride_m, output_stride_n,
    shared_weight_size_1: tl.constexpr, shared_weight_size_2: tl.constexpr,
    HASH_BLOCK_SIZE: tl.constexpr, HASH_SEED: tl.constexpr,
    fp16: tl.constexpr, tf32: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    input_ptrs = input_ptr + (offs_m[:, None] * input_stride_m + offs_k[None, :] * input_stride_k)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    max_start_offset_1 = tl.maximum(shared_weight_size_1 - HASH_BLOCK_SIZE, 1)
    max_start_offset_2 = tl.maximum(shared_weight_size_2 - HASH_BLOCK_SIZE, 1)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        input_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        input_block = tl.load(input_ptrs, mask=input_mask, other=0.0)
        
        k_block_offset = k * BLOCK_SIZE_K + offs_k
        weight_2d_indices = offs_n[:, None] * K + k_block_offset[None, :]
        
        weight_block_ids = weight_2d_indices // HASH_BLOCK_SIZE
        weight_offsets_in_block = weight_2d_indices % HASH_BLOCK_SIZE
        
        weight_hashes_1 = fast_hash_with_seed(weight_block_ids, HASH_SEED)
        weight_block_starts_1 = weight_hashes_1 % max_start_offset_1
        weight_1d_indices_1 = weight_block_starts_1 + weight_offsets_in_block
        
        weight_mask_base = (offs_n[:, None] < N) & (k_block_offset[None, :] < K)
        weight_mask_1 = weight_mask_base & (weight_1d_indices_1 < shared_weight_size_1)
        weight_block_1 = tl.load(shared_weight_ptr_1 + weight_1d_indices_1, mask=weight_mask_1, other=0.0)
        
        weight_hashes_2 = fast_hash_with_seed(weight_block_ids, HASH_SEED + 1)
        weight_block_starts_2 = weight_hashes_2 % max_start_offset_2
        weight_1d_indices_2 = weight_block_starts_2 + weight_offsets_in_block
        
        weight_mask_2 = weight_mask_base & (weight_1d_indices_2 < shared_weight_size_2)
        weight_block_2 = tl.load(shared_weight_ptr_2 + weight_1d_indices_2, mask=weight_mask_2, other=0.0)
        
        weight_block = weight_block_1 + weight_block_2
        
        if fp16:
            input_block = input_block.to(tl.float16)
            weight_block = weight_block.to(tl.float16)
        
        accumulator += tl.dot(input_block, tl.trans(weight_block), allow_tf32=tf32)
        input_ptrs += BLOCK_SIZE_K * input_stride_k
    
    output_ptrs = output_ptr + offs_m[:, None] * output_stride_m + offs_n[None, :] * output_stride_n
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=output_mask)


@triton.autotune(configs=get_autotune_config(), key=['M', 'N', 'K', 'fp16'])
@triton.heuristics({'tf32': lambda _: allow_tf32()})
@triton.jit
def linear_hashed_forward_kernel(
    input_ptr, shared_weight_ptr, output_ptr,
    M, K, N,
    input_stride_m, input_stride_k,
    output_stride_m, output_stride_n,
    shared_weight_size: tl.constexpr,
    HASH_BLOCK_SIZE: tl.constexpr, HASH_SEED: tl.constexpr,
    fp16: tl.constexpr, tf32: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M))
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N))
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    input_ptrs = input_ptr + (offs_m[:, None] * input_stride_m + offs_k[None, :] * input_stride_k)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        input_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K - k * BLOCK_SIZE_K)
        input_block = tl.load(input_ptrs, mask=input_mask, other=0.0)
        
        k_block_offset = k * BLOCK_SIZE_K + offs_k
        weight_2d_indices = offs_n[:, None] * K + k_block_offset[None, :]
        
        weight_block_ids = weight_2d_indices // HASH_BLOCK_SIZE
        weight_offsets_in_block = weight_2d_indices % HASH_BLOCK_SIZE
        weight_hashes = fast_hash_with_seed(weight_block_ids, HASH_SEED)
        max_start_offset = tl.maximum(shared_weight_size - HASH_BLOCK_SIZE, 1)
        weight_block_starts = weight_hashes % max_start_offset
        weight_1d_indices = weight_block_starts + weight_offsets_in_block
        
        weight_mask = (offs_n[:, None] < N) & (k_block_offset[None, :] < K)
        weight_block = tl.load(shared_weight_ptr + weight_1d_indices, mask=weight_mask, other=0.0)
        
        if fp16:
            input_block = input_block.to(tl.float16)
            weight_block = weight_block.to(tl.float16)
        
        accumulator += tl.dot(input_block, tl.trans(weight_block), allow_tf32=tf32)
        input_ptrs += BLOCK_SIZE_K * input_stride_k
    
    output_ptrs = output_ptr + offs_m[:, None] * output_stride_m + offs_n[None, :] * output_stride_n
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, accumulator, mask=output_mask)
