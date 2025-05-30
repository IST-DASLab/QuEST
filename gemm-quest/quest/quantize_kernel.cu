#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>   // For __nv_bfloat16, __bfloat162float, __float2bfloat16_rn


// Generic template for type-specific conversion
template <typename T>
struct TypeTraits;


// Specialization for __half
template <>
struct TypeTraits<__half> {
    static __device__ float toFloat(__half val) {
        return __half2float(val);
    }

    static __device__ __half fromFloat(float val) {
        return __float2half_rn(val);
    }
};


// Specialization for __nv_bfloat16
template <>
struct TypeTraits<__nv_bfloat16> {
    static __device__ float toFloat(__nv_bfloat16 val) {
        return __bfloat162float(val);
    }

    static __device__ __nv_bfloat16 fromFloat(float val) {
        return __float2bfloat16_rn(val);
    }
};


template<typename InputDtype, int BlockSize>
__global__ void find_scale_int4_cuda_kernel(
    InputDtype* __restrict__ scale_ptr,
    const InputDtype* __restrict__ x_ptr,
    int group_size,
    int batch_size
)
{
    // Each block corresponds to one row in the batch dimension
    int batch_id = blockIdx.x;
    if (batch_id >= batch_size) {
        return;
    }

    // Thread-local accumulator in float32
    float sum_sq = 0.0f;

    // Stride over group_size in steps of blockDim.x
    for (int col = threadIdx.x; col < group_size; col += blockDim.x) {
        // Load bfloat16 and convert to float
        InputDtype x_bf16 = x_ptr[batch_id * group_size + col];
        float x_f32 = TypeTraits<InputDtype>::toFloat(x_bf16);
        // Accumulate sum of squares
        sum_sq += x_f32 * x_f32;
    }

    // Reduce partial sums within the block
    __shared__ float sdata[BlockSize];  // Adjust if blockDim.x < 512
    int tid = threadIdx.x;
    sdata[tid] = sum_sq;
    __syncthreads();

    // Tree reduction in shared memory
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    // The first thread in each block computes the final scale
    if (tid == 0) {
        // mean of squares
        float mean_sq = sdata[0] / static_cast<float>(group_size);
        float std_val = sqrtf(mean_sq);

        // scale factor for INT4 (2.513930578568423 * 2 / ((1 << 4) - 1)) + eps
        float factor = 2.513930578568423f * 2.f / 15.f;  // 15 = (1 << 4) - 1
        float scale_val_f = std_val * factor + 1e-8f;

        // convert back to bfloat16
        InputDtype scale_bf16 = TypeTraits<InputDtype>::fromFloat(scale_val_f);

        // store to output
        scale_ptr[batch_id] = scale_bf16;
    }
}


void find_scale_int4_cuda_bf16(
    void* scale_ptr,
    const void* x_ptr,
    int group_size,
    int batch_size,
    cudaStream_t stream
)
{
    using InputDtype = __nv_bfloat16;
    // E.g. 512 threads per block to match the Triton pattern
    const int BlockSize = 512;
    // One block per batch element
    const int gridSize = batch_size;  // Ensure batch_size isn't too large for 1D grid

    find_scale_int4_cuda_kernel<InputDtype, BlockSize><<<gridSize, BlockSize, 0, stream>>>(
        (InputDtype*) scale_ptr,
        (InputDtype*) x_ptr,
        group_size,
        batch_size
    );
}


template<typename InputDtype, int BlockSize>
__global__ void quantize_int4_cuda_kernel(
    InputDtype* __restrict__ x_ptr,
    const InputDtype* __restrict__ scale_ptr,
    int8_t* __restrict__ x_int_ptr,
    uint8_t* __restrict__ x_int_packed_ptr,
    int32_t* __restrict__ x_int_row_sum_ptr,
    int group_size,
    int batch_size,
    bool do_dequantize
)
{
    // Each block handles one "row" in the batch dimension
    int row = blockIdx.x;
    if (row >= batch_size) return;

    // Load scale for this row (bfloat16 -> float)
    float scale_f = TypeTraits<InputDtype>::toFloat(scale_ptr[row]);

    // Local partial sum of int8 values for row-sum (if enabled)
    int local_sum = 0;

    // We'll process columns in chunks of blockDim.x = 512
    for (int col_start = 0; col_start < group_size; col_start += blockDim.x) {
        int col = col_start + threadIdx.x;
        bool valid = (col < group_size);

        // 1) Load bfloat16 x -> float
        float x_f = 0.f;
        if (valid) {
            InputDtype x_bf16 = x_ptr[row * group_size + col];
            x_f = TypeTraits<InputDtype>::toFloat(x_bf16);
        }

        // 2) Quantize to int4 range: [-8, 7]
        float q_f = floorf(x_f / scale_f);
        q_f = fmaxf(q_f, -8.f);  // clamp min
        q_f = fminf(q_f,  7.f);  // clamp max
        int8_t q_i8 = static_cast<int8_t>(static_cast<int>(q_f));

        // 3) Optionally dequantize back into x (bfloat16)
        if (do_dequantize && valid) {
            float dq_f = (q_f * scale_f) + (scale_f * 0.5f);
            x_ptr[row * group_size + col] = TypeTraits<InputDtype>::fromFloat(dq_f);
        }

        // 4) Optionally store int8 in x_int
        if (x_int_ptr && valid) {
            x_int_ptr[row * group_size + col] = q_i8;
        }

        // 5) Accumulate row sum (in int32) if needed
        if (x_int_row_sum_ptr && valid) {
            local_sum += q_i8;
        }

        // 6) If we are packing 4-bit values, we need to stage them in shared memory
        //    so that we can pair up consecutive int8 values.
        __shared__ int8_t smem[BlockSize];  // blockDim.x = 512
        smem[threadIdx.x] = q_i8;
        __syncthreads();

        if (x_int_packed_ptr) {
            // We combine pairs: (low, high) -> single byte: (high << 4) | (low & 0xF)
            // Number of pairs in this chunk = blockDim.x/2 = 256 if blockDim.x=512
            int half_threads = blockDim.x >> 1;  // e.g., 256
            int pairs_in_this_chunk = (group_size - col_start) / 2;  // how many valid pairs in this chunk
            if (threadIdx.x < half_threads && threadIdx.x < pairs_in_this_chunk) {
                // Global index for the pair in the current chunk
                int pair_idx = col_start / 2 + threadIdx.x;

                // Indices in shared memory
                int i = 2 * threadIdx.x;       // "low" half
                int j = i + 1;                 // "high" half

                int8_t x_low  = smem[i];
                int8_t x_high = smem[j];

                // Pack two 4-bit values into one byte
                uint8_t packed = static_cast<uint8_t>(x_high) << 4;
                packed |= (static_cast<uint8_t>(x_low) & 0x0F);

                // Store into x_int_packed_ptr
                x_int_packed_ptr[row * (group_size / 2) + pair_idx] = packed;
            }
        }
        __syncthreads();
    }

    // Finally, reduce local_sum across threads if we want row_sum
    if (x_int_row_sum_ptr) {
        // We can do a block-wide reduction in shared memory
        __shared__ int sdata[BlockSize];
        sdata[threadIdx.x] = local_sum;
        __syncthreads();

        // Simple tree reduction
        for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
            if (threadIdx.x < offset) {
                sdata[threadIdx.x] += sdata[threadIdx.x + offset];
            }
            __syncthreads();
        }

        // Thread 0 writes out
        if (threadIdx.x == 0) {
            x_int_row_sum_ptr[row] = sdata[0];
        }
    }
}


void quantize_int4_cuda_bf16(
    void* x_ptr,
    const void* scale_ptr,
    void* x_int_ptr,
    void* x_int_packed_ptr,
    void* x_int_row_sum_ptr,
    int group_size,
    int batch_size,
    bool do_dequantize,
    cudaStream_t stream
)
{
    using InputDtype = __nv_bfloat16;
    const int BlockSize = 512;

    // One block per row, 512 threads per block (mirroring BLOCK_SIZE_B=1 and BLOCK_SIZE_G=512).
    dim3 gridDim(batch_size);
    dim3 blockDim(BlockSize);

    quantize_int4_cuda_kernel<InputDtype, BlockSize><<<gridDim, blockDim, 0, stream>>>(
        (InputDtype*) x_ptr,
        (InputDtype*) scale_ptr,
        (int8_t*) x_int_ptr,
        (uint8_t*) x_int_packed_ptr,
        (int32_t*) x_int_row_sum_ptr,
        group_size,
        batch_size,
        do_dequantize
    );
}


template<typename InputDtype, int BlockSize>
__global__ void add_mul_vv_cuda_kernel(
    InputDtype* __restrict__ mat_d,
    const int32_t* __restrict__ vec_a_add,
    const InputDtype* __restrict__ vec_a_mul,
    const int32_t* __restrict__ vec_b_add,
    const InputDtype* __restrict__ vec_b_mul,
    const int32_t* __restrict__ mat_c,
    int size_a,
    int size_b
)
{
    // Number of chunks along the columns dimension
    int num_col_chunks = (size_b + BlockSize - 1) / BlockSize;

    // We replicate the Triton grid logic: total blocks = size_a * num_col_chunks
    int global_block_id = blockIdx.x;             // 0..(size_a*num_col_chunks - 1)
    if (global_block_id >= size_a * num_col_chunks) return;

    // Decode which row and which chunk of columns this block processes
    int row       = global_block_id / num_col_chunks;  // in [0..size_a)
    int chunk_idx = global_block_id % num_col_chunks;  // in [0..(num_col_chunks-1)]

    // Each thread covers one column within the chunk
    int col = chunk_idx * BlockSize + threadIdx.x;
    if (row < size_a && col < size_b)
    {
        // 1) Convert inputs to float32
        int32_t a_add = vec_a_add[row];
        float   a_mul = TypeTraits<InputDtype>::toFloat(vec_a_mul[row]);  // bfloat16 -> float
        int32_t b_add = vec_b_add[col];
        float   b_mul = TypeTraits<InputDtype>::toFloat(vec_b_mul[col]);
        int32_t c_val = mat_c[row * size_b + col];

        // 2) Compute in float32
        float lhs = float(c_val * 2 + a_add + b_add);
        float rhs = 0.5f * a_mul * b_mul;
        float d_val = lhs * rhs;  // float32 result

        // 3) Convert result to bfloat16
        mat_d[row * size_b + col] = TypeTraits<InputDtype>::fromFloat(d_val);
    }
}


void add_mul_vv_cuda_bf16(
    void* mat_d,
    const void* vec_a_add,
    const void* vec_a_mul,
    const void* vec_b_add,
    const void* vec_b_mul,
    const void* mat_c,
    int size_a,
    int size_b,
    cudaStream_t stream
)
{
    using InputDtype = __nv_bfloat16;
    const int BlockSize = 512;
    // Grid size = size_a * ceil_div(size_b, BLOCK_SIZE_B)
    int num_col_chunks = (size_b + BlockSize - 1) / BlockSize;
    int grid_size = size_a * num_col_chunks;

    // We launch 1D grid, each block with BLOCK_SIZE_B threads
    dim3 grid(grid_size);
    dim3 block(BlockSize);

    add_mul_vv_cuda_kernel<InputDtype, BlockSize><<<grid, block, 0, stream>>>(
        (InputDtype*) mat_d,
        (int32_t*) vec_a_add,
        (InputDtype*) vec_a_mul,
        (int32_t*) vec_b_add,
        (InputDtype*) vec_b_mul,
        (int32_t*) mat_c,
        size_a,
        size_b
    );
}
