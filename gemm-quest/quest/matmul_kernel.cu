#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_sparse.h>
#include <cutlass/util/host_reorder.h>
#include <cutlass/util/host_uncompress.h>
#include <cutlass/util/reference/host/tensor_fill.h>


int matmul_int4_int4t_int32_cuda(
    void* mat_c,
    const void* mat_a,
    const void* mat_b,
    const int size_m,
    const int size_k,
    const int size_n,
    cudaStream_t stream
) {
    using ElementOutput = int32_t;
    using ElementAccumulator = int32_t;

    using Gemm = cutlass::gemm::device::Gemm<
        cutlass::int4b_t,                         // ElementInputA,
        cutlass::layout::RowMajor,                // LayoutInputA,
        cutlass::int4b_t,                         // ElementInputB,
        cutlass::layout::ColumnMajor,             // LayoutInputB,
        ElementOutput,                            // ElementOutput,
        cutlass::layout::RowMajor,                // LayoutOutput,
        ElementAccumulator,                       // ElementAccumulator,
        cutlass::arch::OpClassTensorOp,           // MMAOp,
        cutlass::arch::Sm80,                      // SmArch,
        cutlass::gemm::GemmShape<128, 128, 128>,  // ShapeMMAThreadBlock,
        cutlass::gemm::GemmShape<64, 64, 128>,    // ShapeMMAWarp,
        cutlass::gemm::GemmShape<16, 8, 64>,      // ShapeMMAOp
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput,
            128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator,
            ElementAccumulator
        >,  // EpilogueOp
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,  // SwizzleThreadBlock
        3  // NumStages
    >;

    typename Gemm::Arguments arguments{
        {size_m, size_n, size_k},
        {(cutlass::int4b_t *) mat_a, size_k},  // A, lda
        {(cutlass::int4b_t *) mat_b, size_k},  // B, ldb
        {(ElementOutput *) mat_c, size_n},  // C, ldc
        {(ElementOutput *) mat_c, size_n},  // D, ldd
        {1, 0},  // alpha, beta
        1  // split_k_slices
    };

    Gemm gemm_op;  // Create GEMM operation

    cutlass::Status status = gemm_op(arguments, nullptr, stream);  // Run the GEMM operation

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM operation failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


int matmul_int4sp_int4t_int32_cuda(
    void* mat_c,
    const void* mat_a,
    const void* meta_e,
    const void* mat_b,
    const int size_m,
    const int size_k,
    const int size_n,
    cudaStream_t stream
) {
    using ElementOutput = int32_t;
    using ElementAccumulator = int32_t;

    using Gemm = cutlass::gemm::device::SparseGemm<
        cutlass::int4b_t,                         // ElementInputA,
        cutlass::layout::RowMajor,                // LayoutInputA,
        cutlass::int4b_t,                         // ElementInputB,
        cutlass::layout::ColumnMajor,             // LayoutInputB,
        ElementOutput,                            // ElementOutput,
        cutlass::layout::RowMajor,                // LayoutOutput,
        ElementAccumulator,                       // ElementAccumulator,
        cutlass::arch::OpClassTensorOp,           // MMAOp,
        cutlass::arch::Sm80,                      // SmArch,
        cutlass::gemm::GemmShape<128, 128, 256>,  // ShapeMMAThreadBlock,
        cutlass::gemm::GemmShape<64, 64, 256>,    // ShapeMMAWarp,
        cutlass::gemm::GemmShape<16, 8, 128>,     // ShapeMMAOp
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput,
            128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator,
            ElementAccumulator
        >,  // EpilogueOp
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,  // SwizzleThreadBlock
        3  // NumStages
    >;

    typename Gemm::Arguments arguments{
        {size_m, size_n, size_k},
        {(cutlass::int4b_t *) mat_a, size_k / Gemm::kSparse},  // A, lda (sparse)
        {(cutlass::int4b_t *) mat_b, size_k},  // B, ldb
        {(ElementOutput *) mat_c, size_n},  // C, ldc
        {(ElementOutput *) mat_c, size_n},  // D, ldd
        {(Gemm::ElementE *) meta_e, Gemm::LayoutE::packed(
            cutlass::make_Coord(size_m, size_k / Gemm::kSparse / Gemm::kElementsPerElementE)
        )},  // E, lde (sparse metadata)
        {1, 0},  // alpha, beta
        1  // split_k_slices
    };

    Gemm gemm_op;  // Create GEMM operation

    cutlass::Status status = gemm_op(arguments, nullptr, stream);  // Run the GEMM operation

    if (status != cutlass::Status::kSuccess) {
        std::cerr << "GEMM operation failed: " << cutlass::cutlassGetStatusString(status) << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


void reorder_meta48_int4_cpu(
    void* meta_e_out,
    const void* meta_e_in,
    const int size_m,
    const int size_k
) {
    int size_k_64 = size_k / (2 * 16 * 2);
    cutlass::TensorRef<uint32_t, cutlass::layout::RowMajor> tensor_e((uint32_t *) meta_e_in, size_k_64);
    cutlass::TensorRef<uint32_t, cutlass::layout::ColumnMajorInterleaved<2>> tensor_e_reordered(
        (uint32_t *) meta_e_out,
        cutlass::layout::ColumnMajorInterleaved<2>::packed(cutlass::make_Coord(size_m, size_k_64))
    );
    cutlass::reorder_meta(tensor_e_reordered, tensor_e, {size_m, 0, size_k_64});
}


void uncompress_meta48_int4_cpu(
    void* mat_a_out,
    const void* mat_a_in,
    const void* meta_e,
    const int size_m,
    const int size_k
) {
    cutlass::TensorRef<cutlass::int4b_t, cutlass::layout::RowMajor> tensor_a_uncompressed{(cutlass::int4b_t *) mat_a_out, size_k};
    cutlass::TensorRef<cutlass::int4b_t, cutlass::layout::RowMajor> tensor_a{(cutlass::int4b_t *) mat_a_in, size_k / 2};
    cutlass::TensorRef<uint32_t, cutlass::layout::RowMajor> tensor_e{(uint32_t *) meta_e, size_k / (2 * 16 * 2)};
    cutlass::uncompress(tensor_a_uncompressed, tensor_a, tensor_e, size_m, size_k);
}


void generate_random_meta48_int4_cpu(
    void* meta_e,
    const int size_m,
    const int size_k
) {
    const cutlass::MatrixCoord extent = cutlass::make_Coord(size_m, size_k / (2 * 16 * 2));
    cutlass::TensorView<uint32_t, cutlass::layout::RowMajor> tensor_e{(uint32_t *) meta_e, cutlass::layout::RowMajor::packed(extent), extent};
    cutlass::reference::host::TensorFillRandomSparseMeta(tensor_e, 1, 2);
}
