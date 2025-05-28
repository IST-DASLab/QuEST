#include <torch/torch.h>


void fused_matmul_dequantize_int4_int4t_bf16(
    torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&
);


void matmul_int4_int4t_int32(
    torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&
);


void matmul_int4sp_int4t_int32(
    torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&
);


void reorder_meta48_int4(
    torch::Tensor&,
    const torch::Tensor&
);


void uncompress_meta48_int4(
    torch::Tensor&,
    const torch::Tensor&,
    const torch::Tensor&
);


void generate_random_meta48_int4(
    torch::Tensor&
);


void find_scale_int4_bf16(
    torch::Tensor&,
    const torch::Tensor&
);


void quantize_int4_bf16(
    torch::Tensor&,
    const torch::Tensor&,
    torch::Tensor&,
    torch::Tensor&,
    torch::Tensor&,
    bool
);


void add_mul_vv_bf16(
    torch::Tensor& mat_d,
    const torch::Tensor& vec_a_add,
    const torch::Tensor& vec_a_mul,
    const torch::Tensor& vec_b_add,
    const torch::Tensor& vec_b_mul,
    const torch::Tensor& mat_c
);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_matmul_dequantize_int4_int4t_bf16", &fused_matmul_dequantize_int4_int4t_bf16, "C = (A @ B.t() + a_add + b_add.t()) * a_mul * b_mul (INT4 x INT4 = BF16 fused matmul dequantize)");
    m.def("matmul_int4_int4t_int32", &matmul_int4_int4t_int32, "C += A @ B.t() (INT4 x INT4 = INT32 matmul)");
    m.def("matmul_int4sp_int4t_int32", &matmul_int4sp_int4t_int32, "C += A(E) @ B.t() (INT4 sparse x INT4 = INT32 matmul)");
    m.def("reorder_meta48_int4", &reorder_meta48_int4, "Reorder UINT32 meta data for sparse INT4 x INT4 = INT32 matmul");
    m.def("uncompress_meta48_int4", &uncompress_meta48_int4, "Uncompress sparse INT4 to dense INT4 using UINT32 meta data");
    m.def("generate_random_meta48_int4", &generate_random_meta48_int4, "Fill with uniform-distributed random UINT32 meta data, 0x489CDE");
    m.def("find_scale_int4_bf16", &find_scale_int4_bf16, "Find scale for INT4 quantization using BF16");
    m.def("quantize_int4_bf16", &quantize_int4_bf16, "Quantize BF16 to INT4 using scale");
    m.def("add_mul_vv_bf16", &add_mul_vv_bf16, "Dequantize INT4 to BF16 after matmul");
}
