import triton
import triton.language as tl
try:
    from .utils import allow_tf32, get_n_stages
except:
    from utils import allow_tf32, get_n_stages

def conv2d_forward_config(
    BLOCK_SIZE_BATCH_HEIGHT_WIDTH: int,
    BLOCK_SIZE_IN_FEAT: int,
    BLOCK_SIZE_OUT_FEAT: int,
    n_warps: int = 4,
    n_stages: int = 2,
    ) -> triton.Config:
    """
    triton.Config object for conv2d_forward_kernel
    given meta-parameters for auto-tuning.
    """
    return triton.Config({'BLOCK_SIZE_BATCH_HEIGHT_WIDTH': BLOCK_SIZE_BATCH_HEIGHT_WIDTH,
                          'BLOCK_SIZE_IN_FEAT': BLOCK_SIZE_IN_FEAT,
                          'BLOCK_SIZE_OUT_FEAT': BLOCK_SIZE_OUT_FEAT},
                          num_warps=n_warps, num_stages=get_n_stages(n_stages))


@triton.jit
def fast_hash_with_seed(n, seed):
    n = n ^ seed
    n = (n ^ (n >> 16)) * 0x45d9f3b
    n = (n ^ (n >> 15)) * 0x119de1
    n = n ^ (n >> 16)
    return n & 0x7FFFFFFF


@triton.autotune(
    configs=[
        # conv2d_forward_config(128, 32, 128, n_warps=8, n_stages=2),
        # conv2d_forward_config(256, 32, 64, n_warps=8, n_stages=2),
        # conv2d_forward_config(256, 32, 32, n_warps=4, n_stages=4),
        # conv2d_forward_config(256, 64, 32, n_warps=4, n_stages=4),
        # conv2d_forward_config(256, 32, 16, n_warps=2, n_stages=4),
        # conv2d_forward_config(64, 32, 128, n_warps=8, n_stages=4),
        # conv2d_forward_config(128, 32, 64, n_warps=4, n_stages=4),
        conv2d_forward_config(64, 32, 64, n_warps=4, n_stages=4),
        # conv2d_forward_config(128, 32, 16, n_warps=4, n_stages=4),
        # conv2d_forward_config(128, 128, 128, n_warps=8, n_stages=3),
        # conv2d_forward_config(256, 128, 64, n_warps=8, n_stages=3),
        # conv2d_forward_config(256, 128, 32, n_warps=4, n_stages=4),
        # conv2d_forward_config(64, 128, 128, n_warps=4, n_stages=4),
        # conv2d_forward_config(128, 128, 64, n_warps=4, n_stages=4),
        # conv2d_forward_config(128, 64, 32, n_warps=2, n_stages=4),
        # conv2d_forward_config(64, 64, 64, n_warps=2, n_stages=4),
    ],
    key=['batch_dim', 'in_feat_dim', 'in_height', 'in_width',
         'out_feat_dim', 'out_height', 'out_width',
         'kernel_height', 'kernel_width',
         'stride_height', 'stride_width',
         'padding_height', 'padding_width',
         'groups', 'fp16'],
)
@triton.heuristics({'tf32': lambda _: allow_tf32()})
@triton.jit
def conv2d_masked_multi_hash_forward_kernel(
    input_pointer, shared_weight_ptr_1, shared_weight_ptr_2, collision_mask_ptr, output_pointer,
    batch_dim, in_feat_dim, in_height, in_width,
    out_feat_dim, out_height, out_width,
    input_batch_stride, input_in_feat_stride, input_height_stride, input_width_stride,
    output_batch_stride, output_out_feat_stride, output_height_stride, output_width_stride,
    shared_weight_size_1: tl.constexpr,
    shared_weight_size_2: tl.constexpr,
    kernel_height: tl.constexpr, kernel_width: tl.constexpr,
    stride_height: tl.constexpr, stride_width: tl.constexpr,
    padding_height: tl.constexpr, padding_width: tl.constexpr,
    groups: tl.constexpr,
    HASH_BLOCK_SIZE: tl.constexpr,
    HASH_SEED: tl.constexpr,
    fp16: tl.constexpr, tf32: tl.constexpr,
    BLOCK_SIZE_BATCH_HEIGHT_WIDTH: tl.constexpr, BLOCK_SIZE_IN_FEAT: tl.constexpr,
    BLOCK_SIZE_OUT_FEAT: tl.constexpr,
    ):

    batch_height_width_pid = tl.program_id(0)
    out_feat_pid = tl.program_id(1)
    group_pid = tl.program_id(2)

    in_group_dim = in_feat_dim // groups
    out_group_dim = out_feat_dim // groups

    batch_height_width_offset = (batch_height_width_pid * BLOCK_SIZE_BATCH_HEIGHT_WIDTH +
                                 tl.arange(0, BLOCK_SIZE_BATCH_HEIGHT_WIDTH))
    batch_height_offset = batch_height_width_offset // out_width
    batch_offset = batch_height_offset // out_height

    output_feat_offset_block = tl.arange(0, BLOCK_SIZE_OUT_FEAT)
    output_feat_offset = out_feat_pid * BLOCK_SIZE_OUT_FEAT + output_feat_offset_block

    output_height_offset = batch_height_offset % out_height
    output_width_offset = batch_height_width_offset % out_width

    input_pointer += (input_batch_stride * batch_offset +
                      input_in_feat_stride * group_pid * in_group_dim)[:, None]

    accum = tl.zeros((BLOCK_SIZE_BATCH_HEIGHT_WIDTH, BLOCK_SIZE_OUT_FEAT), dtype=tl.float32)

    kernel_map_size = kernel_height * kernel_width
    in_channel_map_size = in_group_dim * kernel_map_size
    
    max_start_offset_1 = tl.maximum(shared_weight_size_1 - HASH_BLOCK_SIZE, 1)
    max_start_offset_2 = tl.maximum(shared_weight_size_2 - HASH_BLOCK_SIZE, 1)

    local_out_indices = output_feat_offset

    for h in range(kernel_height):
        for w in range(kernel_width):
            for c in range(0, in_group_dim, BLOCK_SIZE_IN_FEAT):
                # Load Input Block
                input_feat_offset = c + tl.arange(0, BLOCK_SIZE_IN_FEAT)
                input_height_offset = h - padding_height + stride_height * output_height_offset
                input_width_offset = w - padding_width + stride_width * output_width_offset

                curr_input_pointer = (input_pointer +
                                     (input_in_feat_stride * input_feat_offset)[None, :] +
                                     (input_height_stride * input_height_offset)[:, None] +
                                     (input_width_stride * input_width_offset)[:, None])

                input_mask = ((batch_offset < batch_dim)[:, None] &
                              (input_feat_offset < in_group_dim)[None, :] &
                              (0 <= input_height_offset)[:, None] & (input_height_offset < in_height)[:, None] &
                              (0 <= input_width_offset)[:, None] & (input_width_offset < in_width)[:, None])
                
                input_block = tl.load(curr_input_pointer, mask=input_mask, other=0.0)

                local_in_indices = input_feat_offset[:, None]
                group_offset = group_pid * (out_group_dim * in_group_dim * kernel_height * kernel_width)
                
                weight_4d_indices = (local_out_indices[None, :] * in_group_dim * kernel_map_size +
                                     local_in_indices * kernel_map_size +
                                     h * kernel_width + w +
                                     group_offset)

                weight_mask = ((input_feat_offset[:, None] < in_group_dim) &
                               (output_feat_offset[None, :] < out_group_dim))

                weight_block_ids = weight_4d_indices // HASH_BLOCK_SIZE
                weight_offsets_in_block = weight_4d_indices % HASH_BLOCK_SIZE
                weight_hashes_1 = fast_hash_with_seed(weight_block_ids, HASH_SEED)
                
                weight_block_starts_1 = weight_hashes_1 % max_start_offset_1
                weight_1d_indices_1 = weight_block_starts_1 + weight_offsets_in_block
                
                weight_mask_1 = weight_mask & (weight_1d_indices_1 < shared_weight_size_1)
                weight_block_1 = tl.load(shared_weight_ptr_1 + weight_1d_indices_1, mask=weight_mask_1, other=0.0)

                collision_mask_block = tl.load(collision_mask_ptr + weight_1d_indices_1, mask=weight_mask_1, other=0.0)
                weight_hashes_2 = fast_hash_with_seed(weight_block_ids, HASH_SEED + 1)
                
                weight_block_starts_2 = weight_hashes_2 % max_start_offset_2
                weight_1d_indices_2 = weight_block_starts_2 + weight_offsets_in_block
                
                weight_mask_2 = weight_mask & (weight_1d_indices_2 < shared_weight_size_2)
                weight_block_2 = tl.load(shared_weight_ptr_2 + weight_1d_indices_2, mask=weight_mask_2, other=0.0)


                weight_block = weight_block_1 + weight_block_2 * collision_mask_block
                
                if fp16:
                    input_block = input_block.to(tl.float16)
                    weight_block = weight_block.to(tl.float16)

                accum += tl.dot(input_block, weight_block, allow_tf32=tf32)
    
    # Store output
    global_out_indices = group_pid * out_group_dim + output_feat_offset
    
    output_pointer += ((output_batch_stride * batch_offset)[:, None] +
                       (output_out_feat_stride * global_out_indices)[None, :] +
                       (output_height_stride * output_height_offset)[:, None] +
                       (output_width_stride * output_width_offset)[:, None])
    output_mask = ((batch_offset < batch_dim)[:, None] &
                   (output_feat_offset < out_group_dim)[None, :] &
                   (output_height_offset < out_height)[:, None] &
                   (output_width_offset < out_width)[:, None])

    tl.store(output_pointer, accum, mask=output_mask)


@triton.autotune(
    configs=[
        # conv2d_forward_config(128, 32, 128, n_warps=8, n_stages=2),
        # conv2d_forward_config(256, 32, 64, n_warps=8, n_stages=2),
        # conv2d_forward_config(256, 32, 32, n_warps=4, n_stages=4),
        # conv2d_forward_config(256, 64, 32, n_warps=4, n_stages=4),
        # conv2d_forward_config(256, 32, 16, n_warps=2, n_stages=4),
        # conv2d_forward_config(64, 32, 128, n_warps=8, n_stages=4),
        # conv2d_forward_config(128, 32, 64, n_warps=4, n_stages=4),
        conv2d_forward_config(64, 32, 64, n_warps=4, n_stages=4),
        # conv2d_forward_config(128, 32, 16, n_warps=4, n_stages=4),
        # conv2d_forward_config(128, 128, 128, n_warps=8, n_stages=3),
        # conv2d_forward_config(256, 128, 64, n_warps=8, n_stages=3),
        # conv2d_forward_config(256, 128, 32, n_warps=4, n_stages=4),
        # conv2d_forward_config(64, 128, 128, n_warps=4, n_stages=4),
        # conv2d_forward_config(128, 128, 64, n_warps=4, n_stages=4),
        # conv2d_forward_config(128, 64, 32, n_warps=2, n_stages=4),
        # conv2d_forward_config(64, 64, 64, n_warps=2, n_stages=4),
    ],
    key=['batch_dim', 'in_feat_dim', 'in_height', 'in_width',
         'out_feat_dim', 'out_height', 'out_width',
         'kernel_height', 'kernel_width',
         'stride_height', 'stride_width',
         'padding_height', 'padding_width',
         'groups', 'fp16'],
)
@triton.heuristics({'tf32': lambda _: allow_tf32()})
@triton.jit
def conv2d_multi_hash_forward_kernel(
    input_pointer, shared_weight_ptr_1, shared_weight_ptr_2, output_pointer,
    batch_dim, in_feat_dim, in_height, in_width,
    out_feat_dim, out_height, out_width,
    input_batch_stride, input_in_feat_stride, input_height_stride, input_width_stride,
    output_batch_stride, output_out_feat_stride, output_height_stride, output_width_stride,
    shared_weight_size_1: tl.constexpr,
    shared_weight_size_2: tl.constexpr,
    kernel_height: tl.constexpr, kernel_width: tl.constexpr,
    stride_height: tl.constexpr, stride_width: tl.constexpr,
    padding_height: tl.constexpr, padding_width: tl.constexpr,
    groups: tl.constexpr,
    HASH_BLOCK_SIZE: tl.constexpr,
    HASH_SEED: tl.constexpr,
    fp16: tl.constexpr, tf32: tl.constexpr,
    BLOCK_SIZE_BATCH_HEIGHT_WIDTH: tl.constexpr, BLOCK_SIZE_IN_FEAT: tl.constexpr,
    BLOCK_SIZE_OUT_FEAT: tl.constexpr,
    ):
    """
    Multi-hash 2D convolution kernel that uses two separate hash mappings.
    """
    batch_height_width_pid = tl.program_id(0)
    out_feat_pid = tl.program_id(1)
    group_pid = tl.program_id(2)

    in_group_dim = in_feat_dim // groups
    out_group_dim = out_feat_dim // groups

    batch_height_width_offset = (batch_height_width_pid * BLOCK_SIZE_BATCH_HEIGHT_WIDTH +
                                 tl.arange(0, BLOCK_SIZE_BATCH_HEIGHT_WIDTH))
    batch_height_offset = batch_height_width_offset // out_width
    batch_offset = batch_height_offset // out_height

    output_feat_offset_block = tl.arange(0, BLOCK_SIZE_OUT_FEAT)
    output_feat_offset = out_feat_pid * BLOCK_SIZE_OUT_FEAT + output_feat_offset_block

    output_height_offset = batch_height_offset % out_height
    output_width_offset = batch_height_width_offset % out_width

    input_pointer += (input_batch_stride * batch_offset +
                      input_in_feat_stride * group_pid * in_group_dim)[:, None]

    accum = tl.zeros((BLOCK_SIZE_BATCH_HEIGHT_WIDTH, BLOCK_SIZE_OUT_FEAT), dtype=tl.float32)

    kernel_map_size = kernel_height * kernel_width
    in_channel_map_size = in_group_dim * kernel_map_size
    
    max_start_offset_1 = tl.maximum(shared_weight_size_1 - HASH_BLOCK_SIZE, 1)
    max_start_offset_2 = tl.maximum(shared_weight_size_2 - HASH_BLOCK_SIZE, 1)

    local_out_indices = output_feat_offset

    for h in range(kernel_height):
        for w in range(kernel_width):
            for c in range(0, in_group_dim, BLOCK_SIZE_IN_FEAT):
                # Load Input Block
                input_feat_offset = c + tl.arange(0, BLOCK_SIZE_IN_FEAT)
                input_height_offset = h - padding_height + stride_height * output_height_offset
                input_width_offset = w - padding_width + stride_width * output_width_offset

                curr_input_pointer = (input_pointer +
                                     (input_in_feat_stride * input_feat_offset)[None, :] +
                                     (input_height_stride * input_height_offset)[:, None] +
                                     (input_width_stride * input_width_offset)[:, None])

                input_mask = ((batch_offset < batch_dim)[:, None] &
                              (input_feat_offset < in_group_dim)[None, :] &
                              (0 <= input_height_offset)[:, None] & (input_height_offset < in_height)[:, None] &
                              (0 <= input_width_offset)[:, None] & (input_width_offset < in_width)[:, None])
                
                input_block = tl.load(curr_input_pointer, mask=input_mask, other=0.0)

                local_in_indices = input_feat_offset[:, None]
                group_offset = group_pid * (out_group_dim * in_group_dim * kernel_height * kernel_width)
                
                weight_4d_indices = (local_out_indices[None, :] * in_group_dim * kernel_map_size +
                                     local_in_indices * kernel_map_size +
                                     h * kernel_width + w +
                                     group_offset)

                weight_mask = ((input_feat_offset[:, None] < in_group_dim) &
                               (output_feat_offset[None, :] < out_group_dim))

                weight_block_ids = weight_4d_indices // HASH_BLOCK_SIZE
                weight_offsets_in_block = weight_4d_indices % HASH_BLOCK_SIZE
                weight_hashes_1 = fast_hash_with_seed(weight_block_ids, HASH_SEED)
                
                weight_block_starts_1 = weight_hashes_1 % max_start_offset_1
                weight_1d_indices_1 = weight_block_starts_1 + weight_offsets_in_block
                
                weight_mask_1 = weight_mask & (weight_1d_indices_1 < shared_weight_size_1)
                weight_block_1 = tl.load(shared_weight_ptr_1 + weight_1d_indices_1, mask=weight_mask_1, other=0.0)

                weight_hashes_2 = fast_hash_with_seed(weight_block_ids, HASH_SEED + 1)
                
                weight_block_starts_2 = weight_hashes_2 % max_start_offset_2
                weight_1d_indices_2 = weight_block_starts_2 + weight_offsets_in_block
                
                weight_mask_2 = weight_mask & (weight_1d_indices_2 < shared_weight_size_2)
                weight_block_2 = tl.load(shared_weight_ptr_2 + weight_1d_indices_2, mask=weight_mask_2, other=0.0)

                weight_block = weight_block_1 + weight_block_2
                                
                # GEMM
                if fp16:
                    input_block = input_block.to(tl.float16)
                    weight_block = weight_block.to(tl.float16)

                accum += tl.dot(input_block, weight_block, allow_tf32=tf32)

    # Store output
    global_out_indices = group_pid * out_group_dim + output_feat_offset
    
    output_pointer += ((output_batch_stride * batch_offset)[:, None] +
                       (output_out_feat_stride * global_out_indices)[None, :] +
                       (output_height_stride * output_height_offset)[:, None] +
                       (output_width_stride * output_width_offset)[:, None])
    output_mask = ((batch_offset < batch_dim)[:, None] &
                   (output_feat_offset < out_group_dim)[None, :] &
                   (output_height_offset < out_height)[:, None] &
                   (output_width_offset < out_width)[:, None])

    tl.store(output_pointer, accum, mask=output_mask)


@triton.autotune(
    configs=[
        # conv2d_forward_config(128, 32, 128, n_warps=8, n_stages=2),
        # conv2d_forward_config(256, 32, 64, n_warps=8, n_stages=2),
        # conv2d_forward_config(256, 32, 32, n_warps=4, n_stages=4),
        # conv2d_forward_config(256, 64, 32, n_warps=4, n_stages=4),
        # conv2d_forward_config(256, 32, 16, n_warps=2, n_stages=4),
        # conv2d_forward_config(64, 32, 128, n_warps=8, n_stages=4),
        # conv2d_forward_config(128, 32, 64, n_warps=4, n_stages=4),
        conv2d_forward_config(64, 32, 64, n_warps=4, n_stages=4),
        # conv2d_forward_config(128, 32, 16, n_warps=4, n_stages=4),
        # conv2d_forward_config(128, 128, 128, n_warps=8, n_stages=3),
        # conv2d_forward_config(256, 128, 64, n_warps=8, n_stages=3),
        # conv2d_forward_config(256, 128, 32, n_warps=4, n_stages=4),
        # conv2d_forward_config(64, 128, 128, n_warps=4, n_stages=4),
        # conv2d_forward_config(128, 128, 64, n_warps=4, n_stages=4),
        # conv2d_forward_config(128, 64, 32, n_warps=2, n_stages=4),
        # conv2d_forward_config(64, 64, 64, n_warps=2, n_stages=4),
    ],
    key=['batch_dim', 'in_feat_dim', 'in_height', 'in_width',
         'out_feat_dim', 'out_height', 'out_width',
         'kernel_height', 'kernel_width',
         'stride_height', 'stride_width',
         'padding_height', 'padding_width',
         'groups', 'fp16'],
)
@triton.heuristics({'tf32': lambda _: allow_tf32()})
@triton.jit
def conv2d_hashed_forward_kernel(
    input_pointer, shared_weight_ptr, output_pointer,
    batch_dim, in_feat_dim, in_height, in_width,
    out_feat_dim, out_height, out_width,
    input_batch_stride, input_in_feat_stride, input_height_stride, input_width_stride,
    output_batch_stride, output_out_feat_stride, output_height_stride, output_width_stride,
    shared_weight_size: tl.constexpr,
    kernel_height: tl.constexpr, kernel_width: tl.constexpr,
    stride_height: tl.constexpr, stride_width: tl.constexpr,
    padding_height: tl.constexpr, padding_width: tl.constexpr,
    groups: tl.constexpr,
    HASH_BLOCK_SIZE: tl.constexpr,
    HASH_SEED: tl.constexpr,
    fp16: tl.constexpr, tf32: tl.constexpr,
    BLOCK_SIZE_BATCH_HEIGHT_WIDTH: tl.constexpr, BLOCK_SIZE_IN_FEAT: tl.constexpr,
    BLOCK_SIZE_OUT_FEAT: tl.constexpr,
    ):
    """
    2D-convolves over the input using weights loaded from a hashed shared vector.
    """
    batch_height_width_pid = tl.program_id(0)
    out_feat_pid = tl.program_id(1)
    group_pid = tl.program_id(2)
    in_group_dim = in_feat_dim // groups
    out_group_dim = out_feat_dim // groups

    batch_height_width_offset = (batch_height_width_pid * BLOCK_SIZE_BATCH_HEIGHT_WIDTH +
                                 tl.arange(0, BLOCK_SIZE_BATCH_HEIGHT_WIDTH))
    batch_height_offset = batch_height_width_offset // out_width
    batch_offset = batch_height_offset // out_height

    output_feat_offset_block = tl.arange(0, BLOCK_SIZE_OUT_FEAT)
    output_feat_offset = out_feat_pid * BLOCK_SIZE_OUT_FEAT + output_feat_offset_block

    output_height_offset = batch_height_offset % out_height
    output_width_offset = batch_height_width_offset % out_width

    input_pointer += (input_batch_stride * batch_offset +
                      input_in_feat_stride * group_pid * in_group_dim)[:, None]
    accum = tl.zeros((BLOCK_SIZE_BATCH_HEIGHT_WIDTH, BLOCK_SIZE_OUT_FEAT), dtype=tl.float32)

    kernel_map_size = kernel_height * kernel_width
    in_channel_map_size = in_group_dim * kernel_map_size
    max_start_offset = shared_weight_size - HASH_BLOCK_SIZE

    local_out_indices = output_feat_offset

    for h in range(kernel_height):
        for w in range(kernel_width):
            for c in range(0, in_group_dim, BLOCK_SIZE_IN_FEAT):
                input_feat_offset = c + tl.arange(0, BLOCK_SIZE_IN_FEAT)
                input_height_offset = h - padding_height + stride_height * output_height_offset
                input_width_offset = w - padding_width + stride_width * output_width_offset

                curr_input_pointer = (input_pointer +
                                     (input_in_feat_stride * input_feat_offset)[None, :] +
                                     (input_height_stride * input_height_offset)[:, None] +
                                     (input_width_stride * input_width_offset)[:, None])

                input_mask = ((batch_offset < batch_dim)[:, None] &
                              (input_feat_offset < in_group_dim)[None, :] &
                              (0 <= input_height_offset)[:, None] & (input_height_offset < in_height)[:, None] &
                              (0 <= input_width_offset)[:, None] & (input_width_offset < in_width)[:, None])
                
                input_block = tl.load(curr_input_pointer, mask=input_mask, other=0.0)

                local_in_indices = input_feat_offset[:, None]
                group_offset = group_pid * (out_group_dim * in_group_dim * kernel_height * kernel_width)
                
                weight_4d_indices = (local_out_indices[None, :] * in_group_dim * kernel_map_size +
                                     local_in_indices * kernel_map_size +
                                     h * kernel_width + w +
                                     group_offset)

                weight_block_ids = weight_4d_indices // HASH_BLOCK_SIZE
                weight_offsets_in_block = weight_4d_indices % HASH_BLOCK_SIZE
                weight_hashes = fast_hash_with_seed(weight_block_ids, HASH_SEED)
                weight_block_starts = weight_hashes % max_start_offset
                weight_1d_indices = weight_block_starts + weight_offsets_in_block
                
                # Load weight block
                weight_mask = ((input_feat_offset[:, None] < in_group_dim) &
                               (output_feat_offset[None, :] < out_group_dim))
                
                weight_block = tl.load(shared_weight_ptr + weight_1d_indices, mask=weight_mask, other=0.0)
                                
                # GEMM
                if fp16:
                    input_block = input_block.to(tl.float16)
                    weight_block = weight_block.to(tl.float16)
                accum += tl.dot(input_block, weight_block, allow_tf32=tf32)

    # Store output
    global_out_indices = group_pid * out_group_dim + output_feat_offset
    output_pointer += ((output_batch_stride * batch_offset)[:, None] +
                       (output_out_feat_stride * global_out_indices)[None, :] +
                       (output_height_stride * output_height_offset)[:, None] +
                       (output_width_stride * output_width_offset)[:, None])
    output_mask = ((batch_offset < batch_dim)[:, None] &
                   (output_feat_offset < out_group_dim)[None, :] &
                   (output_height_offset < out_height)[:, None] &
                   (output_width_offset < out_width)[:, None])

    tl.store(output_pointer, accum, mask=output_mask)