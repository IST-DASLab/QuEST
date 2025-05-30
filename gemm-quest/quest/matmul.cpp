#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>


int matmul_int4_int4t_int32_cuda(
    void*,
    const void*,
    const void*,
    const int,
    const int,
    const int,
    cudaStream_t
);


int matmul_int4sp_int4t_int32_cuda(
    void*,
    const void*,
    const void*,
    const void*,
    const int,
    const int,
    const int,
    cudaStream_t
);


void reorder_meta48_int4_cpu(
    void*,
    const void*,
    const int,
    const int
);


void uncompress_meta48_int4_cpu(
    void*,
    const void*,
    const void*,
    const int,
    const int
);


void generate_random_meta48_int4_cpu(
    void* meta_e,
    const int size_m,
    const int size_k
);


void matmul_int4_int4t_int32(
    torch::Tensor &mat_c,
    const torch::Tensor &mat_a,
    const torch::Tensor &mat_b
) {
    int err = matmul_int4_int4t_int32_cuda(
        mat_c.data_ptr(),
        mat_a.data_ptr(),
        mat_b.data_ptr(),
        mat_a.size(0),
        mat_b.size(1) * 2,
        mat_b.size(0),
        at::cuda::getCurrentCUDAStream(mat_c.device().index())
    );
}


void matmul_int4sp_int4t_int32(
    torch::Tensor &mat_c,
    const torch::Tensor &mat_a,
    const torch::Tensor &meta_e,
    const torch::Tensor &mat_b
) {
    int err = matmul_int4sp_int4t_int32_cuda(
        mat_c.data_ptr(),
        mat_a.data_ptr(),
        meta_e.data_ptr(),
        mat_b.data_ptr(),
        mat_a.size(0),
        mat_b.size(1) * 2,
        mat_b.size(0),
        at::cuda::getCurrentCUDAStream(mat_c.device().index())
    );
}


void reorder_meta48_int4(
    torch::Tensor &meta_e_out,
    const torch::Tensor &meta_e_in
) {
    reorder_meta48_int4_cpu(
        meta_e_out.data_ptr(),
        meta_e_in.data_ptr(),
        meta_e_in.size(0),
        meta_e_in.size(1) * 64
    );
}


void uncompress_meta48_int4(
    torch::Tensor &meta_a_out,
    const torch::Tensor &meta_a_in,
    const torch::Tensor &meta_e
) {
    uncompress_meta48_int4_cpu(
        meta_a_out.data_ptr(),
        meta_a_in.data_ptr(),
        meta_e.data_ptr(),
        meta_a_in.size(0),
        meta_a_in.size(1) * 4
    );
}


void generate_random_meta48_int4(
    torch::Tensor &meta_e
) {
    generate_random_meta48_int4_cpu(
        meta_e.data_ptr(),
        meta_e.size(0),
        meta_e.size(1) * 64
    );
}
