import torch

import quest_c


def fused_matmul_dequantize_int4_int4t_bf16(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        vec_a_add: torch.Tensor,
        vec_b_add: torch.Tensor,
        vec_a_mul: torch.Tensor,
        vec_b_mul: torch.Tensor,
        out: torch.Tensor = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(mat_a.size(0), mat_b.size(0), dtype=torch.bfloat16, device=mat_a.device)
    assert mat_a.dtype == torch.uint8 and mat_b.dtype == torch.uint8 and vec_a_add.dtype == torch.int32 and vec_b_add.dtype == torch.int32 and vec_a_mul.dtype == torch.bfloat16 and vec_b_mul.dtype == torch.bfloat16 and out.dtype == torch.bfloat16
    assert mat_a.is_contiguous() and mat_b.is_contiguous() and vec_a_add.is_contiguous() and vec_b_add.is_contiguous() and vec_a_mul.is_contiguous() and vec_b_mul.is_contiguous() and out.is_contiguous()
    quest_c.fused_matmul_dequantize_int4_int4t_bf16(out, mat_a, mat_b, vec_a_add, vec_b_add, vec_a_mul, vec_b_mul)
    return out


def matmul_int4_int4t_int32(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        out: torch.Tensor = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(mat_a.size(0), mat_b.size(0), dtype=torch.int32, device=mat_a.device)
    assert mat_a.dtype == torch.uint8 and mat_b.dtype == torch.uint8 and out.dtype == torch.int32
    assert mat_a.is_contiguous() and mat_b.is_contiguous() and out.is_contiguous()
    quest_c.matmul_int4_int4t_int32(out, mat_a, mat_b)
    return out


def matmul_int4sp_int4t_int32(
        mat_a: torch.Tensor,
        meta_e: torch.Tensor,
        mat_b: torch.Tensor,
        out: torch.Tensor = None,
) -> torch.Tensor:
    if out is None:
        out = torch.empty(mat_a.size(0), mat_b.size(0), dtype=torch.int32, device=mat_a.device)
    assert mat_a.dtype == torch.uint8 and meta_e.dtype == torch.uint32 and mat_b.dtype == torch.uint8 and out.dtype == torch.int32
    assert mat_a.is_contiguous() and meta_e.is_contiguous() and mat_b.is_contiguous() and out.is_contiguous()
    quest_c.matmul_int4sp_int4t_int32(out, mat_a, meta_e, mat_b)
    return out


def reorder_meta48_int4(
        meta_e_in: torch.Tensor,
) -> torch.Tensor:
    assert meta_e_in.dtype == torch.uint32
    assert meta_e_in.is_contiguous()
    meta_e_out = torch.empty_like(meta_e_in, device=torch.device('cpu'))
    quest_c.reorder_meta48_int4(meta_e_out, meta_e_in.cpu())
    return meta_e_out.to(device=meta_e_in.device)


def uncompress_meta48_int4(
        mat_a: torch.Tensor,
        meta_e: torch.Tensor,
        out: torch.Tensor = None,
) -> torch.Tensor:
    assert mat_a.dtype == torch.uint8 and meta_e.dtype == torch.uint32
    assert mat_a.is_contiguous() and meta_e.is_contiguous()
    out_ = torch.empty(mat_a.size(0), mat_a.size(1) * 2, dtype=torch.uint8, device=torch.device('cpu'))
    quest_c.uncompress_meta48_int4(out_, mat_a.cpu(), meta_e.cpu())
    if out is None:
        out = out_.to(mat_a.device)
    else:
        assert out.dtype == torch.uint8 and out.is_contiguous()
        out.copy_(out_)
    return out


def generate_random_meta48_int4(
        size_m: int,
        size_k: int,
        device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    meta_e = torch.empty(size_m, size_k // 64, dtype=torch.uint32, device=torch.device('cpu'))
    quest_c.generate_random_meta48_int4(meta_e)
    return meta_e.to(device=device)


def find_scale_int4(
        x: torch.Tensor,
        scale: torch.Tensor = None,
) -> torch.Tensor:
    if scale is None:
        scale = torch.empty(x.size(0), 1, dtype=x.dtype, device=x.device)
    assert x.dtype == torch.bfloat16 and scale.dtype == torch.bfloat16
    assert x.is_contiguous() and scale.is_contiguous()
    quest_c.find_scale_int4_bf16(scale, x)
    return scale


def quantize_int4(
        x: torch.Tensor,
        scale: torch.Tensor,
        x_int: torch.Tensor = None,
        x_int_packed: torch.Tensor = None,
        x_int_row_sum: torch.Tensor = None,
        do_dequantize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    meta_device: torch.device = torch.device('meta')
    x_int_: torch.Tensor = torch.empty(0, dtype=torch.int8, device=meta_device) if x_int is None else x_int
    x_int_packed_: torch.Tensor = torch.empty(0, dtype=torch.uint8, device=meta_device) if x_int_packed is None else x_int_packed
    x_int_row_sum_: torch.Tensor = torch.empty(0, dtype=torch.int32, device=meta_device) if x_int_row_sum is None else x_int_row_sum
    assert x.dtype == torch.bfloat16 and scale.dtype == torch.bfloat16 and x_int_.dtype == torch.int8 and x_int_packed_.dtype == torch.uint8 and x_int_row_sum_.dtype == torch.int32
    assert x.is_contiguous() and scale.is_contiguous() and x_int_.is_contiguous() and x_int_packed_.is_contiguous() and x_int_row_sum_.is_contiguous()
    quest_c.quantize_int4_bf16(x, scale, x_int_, x_int_packed_, x_int_row_sum_, do_dequantize)
    return x, x_int, x_int_packed, x_int_row_sum


def add_mul_vv(
        vec_a_add: torch.Tensor,
        vec_a_mul: torch.Tensor,
        vec_b_add: torch.Tensor,
        vec_b_mul: torch.Tensor,
        mat_c: torch.Tensor,
        mat_d: torch.Tensor = None,
) -> torch.Tensor:
    if mat_d is None:
        mat_d: torch.Tensor = torch.empty(mat_c.shape, dtype=vec_a_mul.dtype, device=mat_c.device)
    assert vec_a_add.dtype == torch.int32 and vec_a_mul.dtype == torch.bfloat16 and vec_b_add.dtype == torch.int32 and vec_b_mul.dtype == torch.bfloat16 and mat_c.dtype == torch.int32 and mat_d.dtype == torch.bfloat16
    assert vec_a_add.is_contiguous() and vec_a_mul.is_contiguous() and vec_b_add.is_contiguous() and vec_b_mul.is_contiguous() and mat_c.is_contiguous() and mat_d.is_contiguous()
    quest_c.add_mul_vv_bf16(mat_d, vec_a_add, vec_a_mul, vec_b_add, vec_b_mul, mat_c)
    return mat_d
