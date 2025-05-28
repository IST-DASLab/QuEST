#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>


int fused_matmul_dequantize_int4_int4t_bf16_cuda(
    void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const int,
    const int,
    const int,
    cudaStream_t
);


void fused_matmul_dequantize_int4_int4t_bf16(
    torch::Tensor &mat_d,
    const torch::Tensor &mat_a,
    const torch::Tensor &mat_b,
    const torch::Tensor &vec_a_add,
    const torch::Tensor &vec_b_add,
    const torch::Tensor &vec_a_mul,
    const torch::Tensor &vec_b_mul
) {
    int err = fused_matmul_dequantize_int4_int4t_bf16_cuda(
        mat_d.data_ptr(),
        mat_a.data_ptr(),
        mat_b.data_ptr(),
        vec_a_add.data_ptr(),
        vec_b_add.data_ptr(),
        vec_a_mul.data_ptr(),
        vec_b_mul.data_ptr(),
        mat_a.size(0),
        mat_b.size(1) * 2,
        mat_b.size(0),
        at::cuda::getCurrentCUDAStream(mat_d.device().index())
    );
}
