#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>


void find_scale_int4_cuda_bf16(
    void*,
    const void*,
    int,
    int,
    cudaStream_t
);


void find_scale_int4_bf16(
    torch::Tensor& scale,
    const torch::Tensor& x
) {
    find_scale_int4_cuda_bf16(
        scale.data_ptr(),
        x.data_ptr(),
        x.size(1),
        x.size(0),
        at::cuda::getCurrentCUDAStream(scale.device().index())
    );
}


void quantize_int4_cuda_bf16(
    void*,
    const void*,
    void*,
    void*,
    void*,
    int,
    int,
    bool,
    cudaStream_t
);


void quantize_int4_bf16(
    torch::Tensor& x,
    const torch::Tensor& scale,
    torch::Tensor& x_int,
    torch::Tensor& x_int_packed,
    torch::Tensor& x_int_row_sum,
    bool do_dequantize
) {
    void *x_int_ptr = nullptr, *x_int_packed_ptr = nullptr, *x_int_row_sum_ptr = nullptr;
    if (x_int.numel()) {
        x_int_ptr = x_int.data_ptr();
    }
    if (x_int_packed.numel()) {
        x_int_packed_ptr = x_int_packed.data_ptr();
    }
    if (x_int_row_sum.numel()) {
        x_int_row_sum_ptr = x_int_row_sum.data_ptr();
    }
    quantize_int4_cuda_bf16(
        x.data_ptr(),
        scale.data_ptr(),
        x_int_ptr,
        x_int_packed_ptr,
        x_int_row_sum_ptr,
        x.size(1),
        x.size(0),
        do_dequantize,
        at::cuda::getCurrentCUDAStream(x.device().index())
    );
}


void add_mul_vv_cuda_bf16(
    void*,
    const void*,
    const void*,
    const void*,
    const void*,
    const void*,
    int,
    int,
    cudaStream_t
);


void add_mul_vv_bf16(
    torch::Tensor& mat_d,
    const torch::Tensor& vec_a_add,
    const torch::Tensor& vec_a_mul,
    const torch::Tensor& vec_b_add,
    const torch::Tensor& vec_b_mul,
    const torch::Tensor& mat_c
) {
    add_mul_vv_cuda_bf16(
        mat_d.data_ptr(),
        vec_a_add.data_ptr(),
        vec_a_mul.data_ptr(),
        vec_b_add.data_ptr(),
        vec_b_mul.data_ptr(),
        mat_c.data_ptr(),
        mat_d.size(0),
        mat_d.size(1),
        at::cuda::getCurrentCUDAStream(mat_d.device().index())
    );
}
