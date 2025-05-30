#include "symmetric/gemm/device/gemm_dequant.h"


int fused_matmul_dequantize_int4_int4t_bf16_cuda(
    void* mat_d,
    const void* mat_a,
    const void* mat_b,
    const void* vec_a_add,
    const void* vec_b_add,
    const void* vec_a_mul,
    const void* vec_b_mul,
    const int size_m,
    const int size_k,
    const int size_n,
    cudaStream_t stream
) {
    using Gemm = cutlass::gemm::device::symmetric::GemmDequant<
        cutlass::int4b_t,                // ElementA
        cutlass::layout::RowMajor,       // LayoutA
        cutlass::int4b_t,                // ElementB
        cutlass::layout::ColumnMajor,    // LayoutB
        cutlass::half_t,                 // ElementGemmOutput
        cutlass::layout::RowMajor,       // LayoutGemmOutput
        cutlass::bfloat16_t,             // ElementOutput
        cutlass::layout::RowMajor,       // LayoutOutput
        int32_t,                         // ElementAccumulator
        cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
        cutlass::arch::Sm80,             // tag indicating target GPU compute architecture
        cutlass::gemm::GemmShape<128, 128, 128>,
        cutlass::gemm::GemmShape<64, 64, 128>,
        cutlass::gemm::GemmShape<16, 8, 64>
    >;

    typename Gemm::Arguments arguments{
        {size_m, size_n, size_k},
        {(cutlass::int4b_t *) mat_a, size_k},
        {(cutlass::int4b_t *) mat_b, size_k},
        {(cutlass::bfloat16_t *) mat_d, size_n},
        {(cutlass::bfloat16_t *) mat_d, size_n},
        {(cutlass::bfloat16_t *) vec_b_mul, size_n},
        {(cutlass::bfloat16_t *) vec_a_mul, size_m},
        {(int32_t *) vec_b_add, size_n},
        {(int32_t *) vec_a_add, size_m},
        Gemm::ElementC(0)
    };

    Gemm gemm_op;  // Create GEMM operation

    cutlass::Status status = gemm_op(arguments, nullptr, stream);  // Run the GEMM operation

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM operation failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
