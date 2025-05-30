import time

import torch
from torch import nn
import torch.nn.functional as F
import triton
from triton import language as tl

from fast_hadamard_transform import hadamard_transform

import quest

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision('highest')

HADAMARD_MATRIX = hadamard_transform(x=torch.eye(128, dtype=torch.bfloat16, device="cuda"), scale=128**-0.5)


def preprocess_weight(
        weight: torch.Tensor,
        use_sparse: bool = False,
        use_hadamard: bool = False,
        backend: str = 'triton'
) -> tuple[torch.Tensor, torch.Tensor,torch.Tensor, torch.Tensor | None]:
    """
    Preprocess the weight matrix
    Args:
        weight: fp, (N, K)
        use_sparse: bool, whether to use 4:8 sparse pattern
        use_hadamard: bool, whether to apply hadamard transform to the input
        backend: str, choose an implementation

    Returns:
        ...

    """
    if use_hadamard:
        weight: torch.Tensor = hadamard_transform(x=weight.unflatten(dim=-1, sizes=(-1, 128)), scale=2. ** -3.5).flatten(start_dim=-2)  # fp, (N, K)
    weight: torch.Tensor = weight.contiguous()
    N, K = weight.shape
    if not use_sparse:
        w_scale: torch.Tensor = torch.empty(N, 1, dtype=weight.dtype, device=weight.device)  # fp, (N, 1)
        w_int_packed: torch.Tensor = torch.empty(N, K // 2, dtype=torch.uint8, device=weight.device)  # uint8, (N, K // 2)
        w_int_row_sum: torch.Tensor = torch.empty(N, 1, dtype=torch.int32, device=weight.device)  # int32, (N, 1)
        find_scale_int4(x=weight, scale=w_scale, backend=backend)  # fp, (N, 1)
        quantize_int4(x=weight, scale=w_scale, x_int=None, x_int_packed=w_int_packed, x_int_row_sum=w_int_row_sum, do_dequantize=False, backend=backend)
        w_add: torch.Tensor = w_int_row_sum + K // 2  # int32, (N, 1)
        return w_int_packed, w_scale, w_add, None
    else:
        _, w_scale, _, _, w_int_compressed, w_int_compressed_packed, _, _, _, meta_e = quantize_int4_sparse(x=weight.clone())
        w_sp_add: torch.Tensor = w_int_compressed.to(dtype=torch.int32).sum(dim=-1, keepdim=True).to(dtype=torch.int32) + K // 4  # int32, (N, 1)
        return w_int_compressed_packed, w_scale, w_sp_add, meta_e


def linear_forward(
        activation: torch.Tensor,
        w_int_packed: torch.Tensor,
        w_scale: torch.Tensor,
        w_add: torch.Tensor,
        w_meta_e: torch.Tensor = None,
        out: torch.Tensor = None,
        _a_scale: torch.Tensor = None,
        _a_int_packed: torch.Tensor = None,
        _a_int_row_sum: torch.Tensor = None,
        _c: torch.Tensor = None,
        use_hadamard: bool = False,
        backend: str = 'triton',
) -> torch.Tensor:
    """
    Compute out = activation @ weight.t() in quantized mode
    Args:
        activation: fp, (M, K)
        w_int_packed: uint8, dense: (N, K // 2), sparse: (N, K // 4)
        w_scale: fp, (N, 1)
        w_add: int32, (N, 1)
        w_meta_e: uint32, dense: None, sparse: (N, K // 64)
        out: fp, (M, N), inplace
        use_hadamard: bool, whether to apply hadamard transform to the input
        backend: str, choose an implementation

    Returns:
        out: fp, (M, N), inplace
    """
    M, K, N = activation.size(0), activation.size(1), w_int_packed.size(0)
    dtype: torch.dtype = activation.dtype
    device: torch.device = activation.device

    if use_hadamard:
        activation: torch.Tensor = F.linear(activation.view(-1, 128), HADAMARD_MATRIX).view_as(activation)  # fp, (M, K)

    if out is None:
        out: torch.Tensor = torch.empty(M, N, dtype=dtype, device=device)  # fp, (M, N)
    a_scale: torch.Tensor = torch.empty(M, 1, dtype=dtype, device=device) if _a_scale is None else _a_scale  # fp, (M, 1)
    a_int_packed: torch.Tensor = torch.empty(M, K // 2, dtype=torch.uint8, device=device) if _a_int_packed is None else _a_int_packed  # uint8, (M, K // 2)

    find_scale_int4(x=activation, scale=a_scale, backend=backend)  # fp, (M, 1)
    if w_meta_e is None:
        a_int_row_sum: torch.Tensor = torch.empty(M, 1, dtype=torch.int32, device=device) if _a_int_row_sum is None else _a_int_row_sum  # int32, (M, 1)
        quantize_int4(x=activation, scale=a_scale, x_int=None, x_int_packed=a_int_packed, x_int_row_sum=a_int_row_sum, do_dequantize=False, backend=backend)
        fused_matmul_dequantize_int4_bf16(mat_a=a_int_packed, mat_b=w_int_packed, vec_a_add=a_int_row_sum, vec_b_add=w_add, vec_a_mul=a_scale, vec_b_mul=w_scale, mat_d=out, _mat_c=None, debug_mode=False)  # fp, (M, N)
    else:
        ct: torch.Tensor = torch.empty(N, M, dtype=torch.int32, device=device)  # int32, (N, M)
        c1t: torch.Tensor = torch.empty(N, M, dtype=torch.int32, device=device)  # int32, (N, M)
        w1: torch.Tensor = torch.full((N, K // 4), 0x11, dtype=torch.uint8, device=device)  # uint8, (N, K // 4)
        quantize_int4(x=activation, scale=a_scale, x_int=None, x_int_packed=a_int_packed, x_int_row_sum=None, do_dequantize=False, backend=backend)
        quest.matmul_int4sp_int4t_int32(mat_a=w_int_packed, meta_e=w_meta_e, mat_b=a_int_packed, out=ct)  # int32, (N, M)
        quest.matmul_int4sp_int4t_int32(mat_a=w1, meta_e=w_meta_e, mat_b=a_int_packed, out=c1t)  # int32, (N, M)
        out_: torch.Tensor = (ct * 2 + c1t + w_add).t().to(dtype=torch.float32) * (.5 * a_scale.to(dtype=torch.float32) * w_scale.t().to(dtype=torch.float32))  # fp32, (M, N)
        out.copy_(out_)  # fp, (M, N)
    return out  # fp, (M, N)


def find_scale_int4(
        x: torch.Tensor,
        scale: torch.Tensor = None,
        backend: str = 'triton',
) -> torch.Tensor:
    """
    Find the optimal scale for quantization with respect to the Gaussian MSE loss

    Args:
        x: fp, (..., K)
        scale: fp, (..., 1), inplace
        backend: str, choose an implementation

    Returns:
        scale: fp, (..., 1), inplace

    """
    device: torch.device = x.device

    if scale is None:
        scale: torch.Tensor = torch.empty(*x.shape[:-1], 1, dtype=x.dtype, device=device)

    if backend == 'pytorch':
        return find_scale_int4_baseline(x, scale)
    elif backend == 'cuda':
        return quest.find_scale_int4(x, scale)
    elif backend == 'triton':
        pass
    else:
        raise NotImplementedError(backend)

    assert x.is_contiguous() and scale.is_contiguous()
    batch_size: int = torch.tensor(x.shape[:-1]).prod().item()
    previous_device: torch.device = torch.device(f'cuda:{torch.cuda.current_device()}')
    torch.cuda.set_device(device)
    grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_SIZE_B']), )
    find_scale_int4_triton_kernel[grid](
        x,
        scale,
        x.size(-1),
        batch_size,
        BLOCK_SIZE_G=512,
        BLOCK_SIZE_B=1,
    )
    torch.cuda.set_device(previous_device)
    return scale


@triton.jit
def find_scale_int4_triton_kernel(
        x_ptr,
        scale_ptr,
        group_size: int,
        batch_size: int,
        BLOCK_SIZE_G: tl.constexpr,  # C
        BLOCK_SIZE_B: tl.constexpr,  # R
):
    tl.multiple_of(group_size, 256)
    tl.multiple_of(batch_size, 32)

    accum_dtype = tl.float32

    pid = tl.program_id(axis=0)  # ()
    b_offsets = pid * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)  # (R)
    b_mask = b_offsets < batch_size  # (R)

    x_ptrs = x_ptr + b_offsets[:, None] * group_size + tl.arange(0, BLOCK_SIZE_G)  # (R, C)
    accum = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_G), dtype=accum_dtype)  # fp32, (R, C)
    for k in range(0, group_size, BLOCK_SIZE_G):
        x_mask = b_mask[:, None] & (tl.arange(0, BLOCK_SIZE_G) < group_size - k)  # (R, C)
        x = tl.load(x_ptrs, mask=x_mask, other=0.).cast(dtype=accum_dtype)  # fp32, (R, C)
        accum = tl.fma(x, x, accum)  # fp32, (R, C)
        x_ptrs += BLOCK_SIZE_G  # (R, C), advance the ptrs to the next block

    standard_deviation = tl.sqrt_rn(tl.div_rn(tl.sum(accum, axis=-1), tl.cast(group_size, accum_dtype)))  # fp32, (R)
    scale = tl.fma(standard_deviation, 2.513930578568423 * 2. / ((1 << 4) - 1), 1e-8)  # fp32, (R)
    tl.store(scale_ptr + b_offsets, scale, mask=b_mask)  # fp, (R)


def find_scale_int4_baseline(
        x: torch.Tensor,
        scale: torch.Tensor,
) -> torch.Tensor:
    # OPTIMAL_GAUSSIAN_SCALES = {
    #     1: 0.7978845587140913,
    #     2: 1.4935346200015913,
    #     3: 2.051068354131873,
    #     4: 2.513930578568423,
    #     5: 2.9160938834961225,
    #     6: 3.276597282593217,
    #     7: 3.6010497188221655,
    #     8: 3.884938678807525,
    # }
    x: torch.Tensor = x.to(dtype=torch.float32)  # fp32, (..., K)
    scale_: torch.Tensor = (x * x).mean(dim=-1, keepdim=True).sqrt() * (2.513930578568423 * 2. / ((1 << 4) - 1)) + 1e-8  # fp32, (..., 1)
    return scale.copy_(scale_)  # fp, (..., 1)


def quantize_int4(
        x: torch.Tensor,
        scale: torch.Tensor,
        x_int: torch.Tensor = None,
        x_int_packed: torch.Tensor = None,
        x_int_row_sum: torch.Tensor = None,
        do_dequantize: bool = False,
        backend: str = 'triton',
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    """
    Find the optimal scale for quantization with respect to the Gaussian MSE loss

    Args:
        x: fp, (..., K), inplace
        scale: fp, (..., 1)
        x_int: int8, (..., K), inplace
        x_int_packed: uint8, (..., K // 2), inplace
        x_int_row_sum: int32, (..., 1), inplace
        do_dequantize: bool, whether to write the dequantized data to x
        backend: str, choose an implementation

    Returns:
        x: fp, (..., K), inplace
        x_int: int8, (..., K), inplace
        x_int_packed: uint8, (..., K // 2), inplace
        x_int_row_sum: int32, (..., 1), inplace

    """
    if backend == 'pytorch':
        return quantize_int4_baseline(x, scale, x_int, x_int_packed, x_int_row_sum, do_dequantize)
    elif backend == 'cuda':
        return quest.quantize_int4(x, scale, x_int, x_int_packed, x_int_row_sum, do_dequantize)
    elif backend == 'triton':
        pass
    else:
        raise NotImplementedError(backend)

    assert x.is_contiguous() and scale.is_contiguous()
    batch_size: int = torch.tensor(x.shape[:-1]).prod().item()
    previous_device: torch.device = torch.device(f'cuda:{torch.cuda.current_device()}')
    torch.cuda.set_device(x.device)
    grid = lambda meta: (triton.cdiv(batch_size, meta['BLOCK_SIZE_B']), )
    quantize_int4_triton_kernel[grid](
        x,
        scale,
        x_int,
        x_int_packed,
        x_int_row_sum,
        x.size(-1),
        batch_size,
        WRITE_DEQUANTIZED=do_dequantize,
        WRITE_DENSE=x_int is not None,
        WRITE_PACKED=x_int_packed is not None,
        WRITE_ROW_SUM=x_int_row_sum is not None,
        BLOCK_SIZE_G=512,
        BLOCK_SIZE_B=1,
    )
    torch.cuda.set_device(previous_device)
    return x, x_int, x_int_packed, x_int_row_sum


@triton.jit
def quantize_int4_triton_kernel(
        x_ptr,
        scale_ptr,
        x_int_ptr,
        x_int_packed_ptr,
        x_int_row_sum_ptr,
        group_size: int,
        batch_size: int,
        WRITE_DEQUANTIZED: tl.constexpr,
        WRITE_DENSE: tl.constexpr,
        WRITE_PACKED: tl.constexpr,
        WRITE_ROW_SUM: tl.constexpr,
        BLOCK_SIZE_G: tl.constexpr,  # C
        BLOCK_SIZE_B: tl.constexpr,  # R
):
    tl.multiple_of(group_size, 256)
    tl.multiple_of(batch_size, 32)

    pid = tl.program_id(axis=0)  # ()
    b_offsets = pid * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)  # (R)
    b_mask = b_offsets < batch_size  # (R)

    x_offsets = b_offsets[:, None] * group_size + tl.arange(0, BLOCK_SIZE_G)  # (R, C)
    x_int_packed_offsets = b_offsets[:, None] * (group_size // 2) + tl.arange(0, BLOCK_SIZE_G // 2)  # (R, C // 2)

    accum = tl.zeros((BLOCK_SIZE_B, BLOCK_SIZE_G), dtype=tl.int32)  # int32, (R, C)

    for k in range(0, group_size, BLOCK_SIZE_G):
        scale = tl.load(scale_ptr + b_offsets, mask=b_mask)[:, None]  # fp, (R, 1)
        x_ptrs = x_ptr + x_offsets  # (R, C)
        x_mask = b_mask[:, None] & (tl.arange(0, BLOCK_SIZE_G) < group_size - k)  # (R, C)
        x = tl.load(x_ptrs, mask=x_mask, other=0.)  # fp, (R, C)
        x_q = tl.clamp(tl.floor(tl.div_rn(x.cast(dtype=tl.float32), scale.cast(dtype=tl.float32))), 16 // -2, 16 // 2 - 1)  # fp32, (R, C)
        x_int = x_q.cast(dtype=tl.int8)  # int8, (R, C)

        if WRITE_DEQUANTIZED:
            x_dq = tl.fma(x_q, scale, scale * .5)  # fp32, (R, C)
            tl.store(x_ptrs, x_dq.cast(scale.dtype, fp_downcast_rounding='rtne'), mask=x_mask)  # fp, (R, C)

        if WRITE_DENSE:
            tl.store(x_int_ptr + x_offsets, x_int, mask=x_mask)  # int8, (R, C)

        if WRITE_PACKED:
            x_low, x_high = x_int.reshape(BLOCK_SIZE_B, BLOCK_SIZE_G // 2, 2).split()  # int8, (R, C // 2)
            x_int_packed = (x_high << 4) | (x_low & 0xF)  # int8, (R, C // 2)
            x_packed_mask = b_mask[:, None] & (tl.arange(0, BLOCK_SIZE_G // 2) < (group_size - k) // 2)  # (R, C // 2)
            tl.store(x_int_packed_ptr + x_int_packed_offsets, x_int_packed, mask=x_packed_mask)  # uint8, (R, C // 2)

        if WRITE_ROW_SUM:
            accum += x_int  # fp32, (R, C)

        # advance the ptr offsets to the next block
        x_offsets += BLOCK_SIZE_G  # (R, C)
        x_int_packed_offsets += BLOCK_SIZE_G // 2  # (R, C // 2)

    if WRITE_ROW_SUM:
        row_sum = tl.sum(accum, axis=-1)  # int32, (R)
        tl.store(x_int_row_sum_ptr + b_offsets, row_sum, mask=b_mask)  # int32, (R)


def quantize_int4_baseline(
        x: torch.Tensor,
        scale: torch.Tensor,
        x_int: torch.Tensor = None,
        x_int_packed: torch.Tensor = None,
        x_int_row_sum: torch.Tensor = None,
        do_dequantize: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    x_q: torch.Tensor = (x.to(dtype=torch.float32) / scale.to(dtype=torch.float32)).floor().clamp(16 // -2, 16 // 2 - 1)  # fp32, (..., K)

    if do_dequantize:
        x_dq: torch.Tensor = x_q * scale + (scale * .5)  # fp32, (..., K)
        x.copy_(x_dq)  # fp, (..., K)

    write_dense: bool = x_int is not None
    write_packed: bool = x_int_packed is not None
    write_row_sum: bool = x_int_row_sum is not None

    if write_dense or write_packed or write_row_sum:
        x_int_: torch.Tensor = x_q.to(dtype=torch.int8)  # int8, (..., K)

        if write_dense:
            x_int.copy_(x_int_)  # int8, (..., K)

        if write_packed:
            x_int_packed_: torch.Tensor = (x_int_[..., 1::2] << 4) | (x_int_[..., ::2] & 0xF)  # int8, (..., K // 2)
            x_int_packed.copy_(x_int_packed_)  # uint8, (..., K)

        if write_row_sum:
            x_int_row_sum.copy_(x_int_.to(dtype=torch.int32).sum(dim=-1, keepdim=True))  # int32, (..., 1)

    return x, x_int, x_int_packed, x_int_row_sum


def quantize_int4_sparse(
        x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Quantize using 4:8 sparsity
    Args:
        x: fp, (..., K), inplace

    Returns:
        x: fp, (..., K), inplace
        ...

    """
    p: float = 2.  # norm to find sparse pairs
    _, nonzero_idx = torch.linalg.vector_norm(
        x.unflatten(dim=-1, sizes=(-1, 4, 2)), ord=p, dim=-1, keepdim=True,
    ).topk(
        k=2, dim=-2, largest=True, sorted=False,
    )  # int64, (..., K // 8, 2, 1)
    nonzero_idx, _ = nonzero_idx.sort(dim=-2, descending=False)  # int64, (..., K // 8, 2, 1)
    nonzero_mask: torch.Tensor = torch.zeros_like(x, dtype=torch.bool).unflatten(dim=-1, sizes=(-1, 4, 2)).scatter_(
        dim=-2,
        index=nonzero_idx.expand(*nonzero_idx.shape[:-1], 2),  # int64, (..., K //8 , 2, 2)
        src=torch.ones((), dtype=torch.bool, device=x.device).expand(*nonzero_idx.shape[:-1], 2),
        # int64, (..., K // 8, 2, 2)
    ).flatten(start_dim=-3)  # bool, (..., K)

    scale: torch.Tensor = torch.empty(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)  # fp, (..., 1)
    x_int: torch.Tensor = torch.empty(x.shape, dtype=torch.int8, device=x.device)  # int8, (..., K)
    scale = find_scale_int4(x=x, scale=scale, backend='pytorch')  # fp, (..., 1)
    _, x_int, _, _ = quantize_int4(x=x, scale=scale, x_int=x_int, x_int_packed=None, x_int_row_sum=None, do_dequantize=False, backend='pytorch')  # int8, (..., K)
    x_int *= nonzero_mask  # int8, (..., K)

    x_dq: torch.Tensor = (x_int.to(dtype=scale.dtype) + .5) * scale * nonzero_mask  # fp, (..., K)
    x.copy_(x_dq)  # fp, (..., K)

    x_int_packed: torch.Tensor = ((x_int[..., 1::2] << 4) | (x_int[..., ::2] & 0xF)).to(dtype=torch.uint8)  # uint8 = 2 * int4, (..., K // 2)
    x_int_compressed: torch.Tensor = x_int.unflatten(dim=-1, sizes=(-1, 4, 2)).take_along_dim(nonzero_idx, dim=-2).flatten(start_dim=-3)  # int8, (..., K // 2)
    x_int_compressed_packed: torch.Tensor = ((x_int_compressed[..., 1::2] << 4) | (x_int_compressed[..., ::2] & 0xF)).to(dtype=torch.uint8)  # uint8 = 2 * int4, (..., K // 4)

    meta_e: torch.Tensor = nonzero_idx[..., 0].to(dtype=torch.int32)  # int32, (..., K // 8, 2)
    meta_e = meta_e[..., 0] | (meta_e[..., 1] << 2)  # int32 = 2 * int2, (..., K // 8)
    meta_e = meta_e[..., ::2] | (meta_e[..., 1::2] << 4)  # int32 = 4 * int2, (..., K // 16)
    meta_e = meta_e[..., ::2] | (meta_e[..., 1::2] << 8)  # int32 = 8 * int2, (..., K // 32)
    meta_e = meta_e[..., ::2] | (meta_e[..., 1::2] << 16)  # int32 = 16 * int2, (..., K // 64)
    meta_e = meta_e.to(dtype=torch.uint32)  # uint32, (..., K // 64)
    meta_e_no_reorder: torch.Tensor = meta_e.clone()  # uint32, (..., K // 64)
    meta_e: torch.Tensor = quest.reorder_meta48_int4(meta_e_in=meta_e)  # uint32, (..., K // 64)

    return x, scale, x_int, x_int_packed, x_int_compressed, x_int_compressed_packed, nonzero_mask, nonzero_idx, meta_e_no_reorder, meta_e


def unpack_int4(
        x_int_packed: torch.Tensor,
        meta_e_no_reorder: torch.Tensor = None,
) -> torch.Tensor:
    """
    Unpack uint8 to int4
    Args:
        x_int_packed: uint8, dense: (..., K // 2), sparse: (..., K // 4)
        meta_e_no_reorder: uint32, sparse: (..., K // 64)

    Returns:
        x_int: int8, (..., K)

    """
    if meta_e_no_reorder is not None:
        x_int_packed = quest.uncompress_meta48_int4(
            mat_a=x_int_packed,
            meta_e=meta_e_no_reorder,
            out=None,
        )  # uint8, (..., K // 2)

    x_int: torch.Tensor = torch.empty(*x_int_packed.shape[:-1], x_int_packed.size(-1) * 2, dtype=torch.int8, device=x_int_packed.device)  # int8, (..., K)
    x_int[..., ::2] = x_int_packed & 0xF
    x_int[..., 1::2] = x_int_packed >> 4
    x_int: torch.Tensor = torch.where(x_int >= 8, x_int - 16, x_int)  # int8, (..., K)
    return x_int  # int8, (..., K)


def add_mul_vv(
        vec_a_add: torch.Tensor,
        vec_a_mul: torch.Tensor,
        vec_b_add: torch.Tensor,
        vec_b_mul: torch.Tensor,
        mat_c: torch.Tensor,
        mat_d: torch.Tensor = None,
        backend: str = 'triton',
) -> torch.Tensor:
    """
    Compute mat_d = (mat_c * 2 + vec_a_add + vec_b_add.t()) * (.5 * vec_a_mul * vec_b_mul.t())
    Args:
        vec_a_add: int32, (M, 1)
        vec_a_mul: fp, (M, 1)
        vec_b_add: int32, (N, 1)
        vec_b_mul: fp, (N, 1)
        mat_c: int32, (M, N)
        mat_d: fp, (M, N), inplace
        backend: str, choose an implementation

    Returns:
        mat_d: fp, (M, N), inplace

    """
    device: torch.device = mat_c.device
    size_a, size_b = mat_c.shape

    if mat_d is None:
        mat_d: torch.Tensor = torch.empty(size_a, size_b, dtype=vec_a_mul.dtype, device=device)

    if backend == 'pytorch':
        return add_mul_vv_baseline(vec_a_add, vec_a_mul, vec_b_add, vec_b_mul, mat_c, mat_d)
    elif backend == 'cuda':
        return quest.add_mul_vv(vec_a_add, vec_a_mul, vec_b_add, vec_b_mul, mat_c, mat_d)
    elif backend == 'triton':
        pass
    else:
        raise NotImplementedError(backend)

    previous_device: torch.device = torch.device(f'cuda:{torch.cuda.current_device()}')
    torch.cuda.set_device(device)
    grid = lambda meta: (size_a * triton.cdiv(size_b, meta['BLOCK_SIZE_B']), )
    add_mul_vv_triton_kernel[grid](
        vec_a_add,
        vec_a_mul,
        vec_b_add,
        vec_b_mul,
        mat_c,
        mat_d,
        size_a,
        size_b,
        BLOCK_SIZE_B=512,
    )
    torch.cuda.set_device(previous_device)
    return mat_d


@triton.jit
def add_mul_vv_triton_kernel(
        vec_a_add_ptr,
        vec_a_mul_ptr,
        vec_b_add_ptr,
        vec_b_mul_ptr,
        mat_c_ptr,
        mat_d_ptr,
        size_a: int,
        size_b: int,
        BLOCK_SIZE_B: tl.constexpr,
):
    tl.multiple_of(size_a, 32)
    tl.multiple_of(size_b, 32)

    pid = tl.program_id(axis=0)  # ()
    offset_a = pid % size_a  # ()
    offsets_b = pid // size_a * BLOCK_SIZE_B + tl.arange(0, BLOCK_SIZE_B)  # (B)
    mask = offsets_b < size_b  # (B)
    offsets_c = offset_a * size_b + offsets_b  # (B)

    a_add = tl.load(vec_a_add_ptr + offset_a)  # int32, ()
    a_mul = tl.load(vec_a_mul_ptr + offset_a)  # fp, ()
    b_add = tl.load(vec_b_add_ptr + offsets_b, mask=mask)  # int32, (B)
    b_mul = tl.load(vec_b_mul_ptr + offsets_b, mask=mask)  # fp, (B)
    c = tl.load(mat_c_ptr + offsets_c, mask=mask)  # int32, (B)
    d = (c * 2 + a_add + b_add).cast(dtype=tl.float32) * (.5 * a_mul.cast(dtype=tl.float32) * b_mul.cast(dtype=tl.float32))  # fp32, (B)
    tl.store(mat_d_ptr + offsets_c, d.cast(dtype=a_mul.dtype, fp_downcast_rounding='rtne'), mask=mask)  # fp, (B)


def add_mul_vv_baseline(
        vec_a_add: torch.Tensor,
        vec_a_mul: torch.Tensor,
        vec_b_add: torch.Tensor,
        vec_b_mul: torch.Tensor,
        mat_c: torch.Tensor,
        mat_d: torch.Tensor,
) -> torch.Tensor:
    accum_dtype: torch.dtype = torch.float32
    mat_d_: torch.Tensor = (mat_c * 2 + vec_a_add + vec_b_add.t()).to(dtype=accum_dtype) * (.5 * vec_a_mul.to(dtype=accum_dtype) * vec_b_mul.t().to(dtype=accum_dtype))  # fp32, (M, N)
    return mat_d.copy_(mat_d_)  # fp, (M, N)


def fused_matmul_dequantize_int4_bf16(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        vec_a_add: torch.Tensor,
        vec_b_add: torch.Tensor,
        vec_a_mul: torch.Tensor,
        vec_b_mul: torch.Tensor,
        mat_d: torch.Tensor,
        _mat_c: torch.Tensor = None,
        debug_mode: bool = True,
) -> torch.Tensor:
    """
    mat_c = A @ B.t() = matmul(mat_a, mat_b)
    mat_d = (mat_c * 2 + vec_a_add + vec_b_add.t()) * (.5 * vec_a_mul * vec_b_mul.t())
    Args:
        mat_a: uint8 = 2 * int4, (M, K // 2)
        mat_b: uint8 = 2 * int4, (N, K // 2)
        vec_a_add: int32, (M, 1)
        vec_a_mul: fp, (M, 1)
        vec_b_add: int32, (N, 1)
        vec_b_mul: fp, (N, 1)
        mat_d: fp, (M, N), inplace,
        _mat_c: int32, (M, N), inplace,
        debug_mode: bool

    Returns:
        mat_d: fp, (M, N), inplace

    """
    M, N = mat_a.size(0), mat_b.size(0)
    device: torch.device = mat_a.device
    if mat_d is None:
        mat_d: torch.Tensor = torch.empty(M, N, dtype=vec_a_mul.dtype, device=device)  # fp, (M, N)
    if debug_mode:
        mat_c: torch.Tensor = torch.empty(M, N, dtype=torch.int32, device=device) if _mat_c is None else _mat_c  # int32, (M, N)
        quest.matmul_int4_int4t_int32(mat_a=mat_a, mat_b=mat_b, out=mat_c)  # int32, (M, N)
        quest.add_mul_vv(vec_a_add=vec_a_add, vec_a_mul=vec_a_mul, vec_b_add=vec_b_add, vec_b_mul=vec_b_mul, mat_c=mat_c, mat_d=mat_d)  # fp, (M, N)
    else:
        quest.fused_matmul_dequantize_int4_int4t_bf16(mat_a=mat_a, mat_b=mat_b, vec_a_add=vec_a_add, vec_b_add=vec_b_add, vec_a_mul=vec_a_mul, vec_b_mul=vec_b_mul, out=mat_d)  # fp, (M, N)
    return mat_d  # fp, (M, N)


def _get_random_inputs(size_in: int, size_out: int, size_batch: int, dtype: torch.dtype, seed: int) -> tuple:
    """
    Generate random weights and activations

    Args:
        size_in: K
        size_out: N
        size_batch: M
        dtype: fp
        seed:

    Returns:
        weight: fp, (N, K)
        activation: fp, (M, K)
    """
    assert size_in % 256 == 0 and size_out % 32 == 0 and size_batch % 32 == 0
    torch.manual_seed(seed)
    device: torch.device = torch.device('cuda')
    weight: torch.Tensor = torch.randn(size_out, size_in, dtype=dtype, device=device)  # fp, (N, K)
    activation: torch.Tensor = torch.randn(size_batch, size_in, dtype=dtype, device=device)  # fp, (M, K)
    return weight, activation


def _triton_test() -> None:
    @triton.jit
    def test_fn(x_ptr):
        x = tl.cast(-15.9, dtype=tl.int8)
        tl.store(x_ptr, x)

    x = torch.zeros(1, dtype=torch.bfloat16, device=torch.device('cuda'))
    grid = lambda meta: (1, )
    test_fn[grid](x)
    print(x)
    exit(0)


def _unit_test(M: int = 32 * 5, K: int = 256 * 3, N: int = 32 * 7) -> None:
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = torch.device('cuda')
    weight, activation = _get_random_inputs(K, N, M, dtype=dtype, seed=0)

    w_sp_dq, w_scale, w_int, w_int_packed, w_int_compressed, w_int_compressed_packed, w_mask, w_mask_idx, w_meta_e_no_reorder, w_meta_e = quantize_int4_sparse(x=weight.clone())
    w_int_unpacked = unpack_int4(x_int_packed=w_int_packed, meta_e_no_reorder=None)
    assert w_int_unpacked.equal(w_int)
    w_int_unpacked = unpack_int4(x_int_packed=w_int_compressed_packed, meta_e_no_reorder=w_meta_e_no_reorder)
    assert w_int_unpacked.equal(w_int)

    a_scale_ref = find_scale_int4(x=activation, scale=None, backend='pytorch')
    a_scale = find_scale_int4(x=activation, scale=None, backend='triton')
    assert a_scale.equal(a_scale_ref)
    a_scale = find_scale_int4(x=activation, scale=None, backend='cuda')
    assert a_scale.equal(a_scale_ref)

    a_int_ref = torch.empty(M, K, dtype=torch.int8, device=device)
    a_int = torch.empty_like(a_int_ref)
    a_int_packed_ref = torch.empty(M, K // 2, dtype=torch.uint8, device=device)
    a_int_packed = torch.empty_like(a_int_packed_ref)
    a_int_row_sum_ref = torch.empty(M, 1, dtype=torch.int32, device=device)
    a_int_row_sum = torch.empty_like(a_int_row_sum_ref)

    a_dq_ref, _, _, _ = quantize_int4(x=activation.clone(), scale=a_scale, x_int=a_int_ref, x_int_packed=a_int_packed_ref, x_int_row_sum=a_int_row_sum_ref, do_dequantize=True, backend='pytorch')
    a_dq, _, _, _ = quantize_int4(x=activation.clone(), scale=a_scale, x_int=a_int, x_int_packed=a_int_packed, x_int_row_sum=a_int_row_sum, do_dequantize=True, backend='triton')
    assert a_dq.equal(a_dq_ref)
    assert a_int.equal(a_int_ref)
    assert a_int_packed.equal(a_int_packed_ref)
    assert a_int_row_sum.equal(a_int_row_sum_ref)
    a_dq, a_int, a_int_packed, a_int_row_sum = quantize_int4(x=activation.clone(), scale=a_scale, x_int=torch.empty_like(a_int), x_int_packed=torch.empty_like(a_int_packed), x_int_row_sum=torch.empty_like(a_int_row_sum), do_dequantize=True, backend='cuda')
    assert a_dq.equal(a_dq_ref)
    assert a_int.equal(a_int_ref)
    assert a_int_packed.equal(a_int_packed_ref)
    assert a_int_row_sum.equal(a_int_row_sum_ref)

    a_int_unpacked_ref = unpack_int4(x_int_packed=a_int_packed_ref, meta_e_no_reorder=None)
    a_int_unpacked = unpack_int4(x_int_packed=a_int_packed, meta_e_no_reorder=None)
    assert a_int_unpacked_ref.equal(a_int_ref)
    assert a_int_unpacked.equal(a_int)

    a_dq_fp64 = (a_int.to(dtype=torch.float64) + .5) * a_scale.to(dtype=torch.float64)  # fp64, (M, K)
    w_dq_fp64 = (w_int.to(dtype=torch.float64) + .5) * w_scale.to(dtype=torch.float64)  # fp64, (N, K)
    w_sp_dq_fp64 = w_dq_fp64 * w_mask  # fp64, (N, K)

    m_ref = a_dq_fp64 @ w_dq_fp64.t()  # fp64, (M, N)
    m = a_scale.to(dtype=torch.float64) * (
            2. * a_int.to(dtype=torch.float64) @ w_int.to(dtype=torch.float64).t()
            + a_int.to(dtype=torch.float64).sum(dim=-1, keepdim=True)
            + w_int.to(dtype=torch.float64).sum(dim=-1, keepdim=True).t()
            + .5 * K
    ) * (.5 * w_scale.to(dtype=torch.float64)).t()  # fp64, (M, N)
    assert m.equal(m_ref)

    c_ref = (a_int.to(dtype=torch.float64) @ w_int.to(dtype=torch.float64).t()).to(dtype=torch.int32)  # int32, (M, N)

    w_add = w_int.to(dtype=torch.int32).sum(dim=-1, keepdim=True).to(dtype=torch.int32) + K // 2  # int32, (N, 1)
    m = add_mul_vv(vec_a_add=a_int_row_sum, vec_a_mul=a_scale, vec_b_add=w_add, vec_b_mul=w_scale, mat_c=c_ref, mat_d=None, backend='pytorch')  # fp, (M, N)
    assert m.equal(m_ref.to(dtype=dtype))
    m = add_mul_vv(vec_a_add=a_int_row_sum, vec_a_mul=a_scale, vec_b_add=w_add, vec_b_mul=w_scale, mat_c=c_ref, mat_d=None, backend='triton')  # fp, (M, N)
    assert m.equal(m_ref.to(dtype=dtype))
    m = add_mul_vv(vec_a_add=a_int_row_sum, vec_a_mul=a_scale, vec_b_add=w_add, vec_b_mul=w_scale, mat_c=c_ref, mat_d=None, backend='cuda')  # fp, (M, N)
    assert m.equal(m_ref.to(dtype=dtype))

    m_sp_ref = a_dq_fp64 @ w_sp_dq_fp64.t()  # fp64, (M, N)
    m_sp = a_scale.to(dtype=torch.float64) * (
            2. * a_int.to(dtype=torch.float64) @ w_int.to(dtype=torch.float64).t()
            + a_int.to(dtype=torch.float64) @ w_mask.to(dtype=torch.float64).t()
            + (w_int * w_mask).to(dtype=torch.float64).sum(dim=-1, keepdim=True).t()
            + .25 * K
    ) * (.5 * w_scale.to(dtype=torch.float64)).t()  # fp64, (M, N)
    assert m_sp.equal(m_sp_ref)

    w_sp_add = w_int_compressed.to(dtype=torch.int32).sum(dim=-1, keepdim=True).to(dtype=torch.int32) + K // 4  # int32, (N, 1)
    m = a_scale.to(dtype=torch.float32) * (
            c_ref * 2 + w_sp_add.t() + a_int.to(dtype=torch.float32) @ w_mask.to(dtype=torch.float32).t()
    ) * (w_scale.to(dtype=torch.float32) * .5).t()  # fp32, (M, N)
    assert m.to(dtype=dtype).equal(m_sp_ref.to(dtype=dtype))

    random_meta_e = quest.generate_random_meta48_int4(size_m=N, size_k=K, device=device)  # uint32, (N, K // 64)

    c = quest.matmul_int4_int4t_int32(mat_a=a_int_packed, mat_b=w_int_packed, out=None)  # int32, (M, N)
    assert c.equal(c_ref)

    c_sp_inplace = torch.empty_like(c.t().contiguous())  # int32, (N, M)
    c_sp = quest.matmul_int4sp_int4t_int32(mat_a=w_int_compressed_packed, meta_e=w_meta_e, mat_b=a_int_packed, out=c_sp_inplace).t()  # int32, (M, N)
    assert c_sp.equal(c_ref)
    assert c_sp_inplace.t().equal(c_ref)

    mmm = linear_forward(activation=activation, w_int_packed=w_int_packed, w_scale=w_scale, w_add=w_add, use_hadamard=False, backend='triton')  # fp, (M, N)
    assert mmm.equal(m_ref.to(dtype=dtype))
    mmm_sp = linear_forward(activation, *preprocess_weight(weight=weight, use_sparse=True, use_hadamard=False, backend='triton'), use_hadamard=False, backend='triton')  # fp, (M, N)
    assert mmm_sp.equal(m_sp_ref.to(dtype=dtype))

    try:
        import quik
    except ImportError:
        print('Skip quik.')
    else:
        def int4_spmm_example():
            import numpy as np

            def pack_to_i4(X):
                def two_compl(x, bits):
                    return torch.where(x < 0, 2 ** bits + x, x)

                X_i8 = two_compl(X.to(dtype=torch.int8), 4).to(torch.uint8)
                X_i4 = X_i8[:, 0::2] | (X_i8[:, 1::2] << 4)
                return X_i4

            torch.manual_seed(1)
            np.random.seed(1)
            M, K, N = 256, 512, 128
            assert K % 16 == 0
            a = torch.randint(-3, 3, (M, K // 2), dtype=torch.int32).cuda()  # int32, (M, K // 2) = (256, 256)
            metadata = quik.matmul.int4GenRandomSparseMeta(M, K)  # int32 = 16 * int2, (M, K // 64) = (256, 8)

            e = quik.matmul.int4ReorderMeta(metadata).cuda()  # int32 = 16 * int2, (M * K // 64) = (2048)
            b = torch.randint(-3, 3, (N, K), dtype=torch.int32).cuda()  # int32, (N, K) = (128, 512)
            qa = pack_to_i4(a)  # uint8 = 2 * int4, (M, K // 4) = (256, 128)
            qb = pack_to_i4(b)  # uint8 = 2 * int4, (N, K // 2) = (128, 256)
            c = quik.matmul.int4SpMatmul(qa, qb, e)  # int32, (M, N) = (256, 128)
            qa_uncompressed = quik.matmul.int4Uncompress(qa.cpu(), metadata, M, K).cuda()  # uint8, (M, K // 2) = (256, 256)
            c_ref = quik.matmul.int4Matmul(qa_uncompressed, qb)  # int32, (M, N) = (256, 128)
            assert torch.equal(c, c_ref)

        int4_spmm_example()

        random_meta_e_quik = quik.matmul.int4GenRandomSparseMeta(N, K)  # int32 = 16 * int2, (N, K // 64)

        e_quik = quik.matmul.int4ReorderMeta(w_meta_e_no_reorder.to(torch.int32).cpu()).cuda()  # int32 = 16 * int2, (M * K // 64)
        assert w_meta_e.flatten().to(dtype=torch.int32).equal(e_quik)

        c_quik = quik.matmul.int4Matmul(a_int_packed, w_int_packed)  # int32, (M, N)
        assert c_quik.equal(c_ref)

        c_sp_quik = quik.matmul.int4SpMatmul(w_int_compressed_packed, a_int_packed, w_meta_e.to(dtype=torch.int32)).t()  # int32, (M, N)
        assert c_sp_quik.equal(c_ref)

    graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
    s: torch.cuda.Stream = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    graph_tensors: dict[str, torch.Tensor] = {
        'a_int_packed': torch.empty(M, K // 2, dtype=torch.uint8, device=device),
        'w_int_packed': torch.empty(N, K // 2, dtype=torch.uint8, device=device),
        'c': torch.empty(M, N, dtype=torch.int32, device=device),
    }
    with torch.cuda.stream(s):
        n_warmups: int = 5
        for _ in range(n_warmups):
            quest.matmul_int4_int4t_int32(mat_a=graph_tensors['a_int_packed'], mat_b=graph_tensors['w_int_packed'], out=graph_tensors['c'])
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(graph):
        quest.matmul_int4_int4t_int32(mat_a=graph_tensors['a_int_packed'], mat_b=graph_tensors['w_int_packed'], out=graph_tensors['c'])
    graph_tensors['a_int_packed'].copy_(a_int_packed)
    graph_tensors['w_int_packed'].copy_(w_int_packed)
    graph.replay()
    c = graph_tensors['c'].clone()
    assert c.equal(c_ref)

    print('Unit Test OK!')


@torch.inference_mode()
def do_bench(f, n_iter: int = 100, n_warmup: int = 10) -> float:
    for _ in range(n_warmup):
        f()
    torch.cuda.synchronize()
    ts = time.time()
    for _ in range(n_iter):
        f()
    torch.cuda.synchronize()
    te = time.time()
    return (te - ts) / n_iter


@torch.inference_mode()
def do_bench_network(
        f,
        M: int,
        K: int,
        N: int,
        dtype: torch.dtype,
        device: torch.device,
        n_layers: int = 20,
        n_warmup: int = 3,
) -> float:
    workspace: torch.Tensor = torch.empty(M * max(K, N), dtype=dtype, device=device)
    activation_in = workspace[:M * K].reshape(M, K)
    activation_out = workspace[:M * N].reshape(M, N)
    def network():
        for _ in range(n_layers):
            f(activation_in, activation_out)

    tf = do_bench(network, n_iter=1, n_warmup=n_warmup)

    graph: torch.cuda.CUDAGraph = torch.cuda.CUDAGraph()
    s: torch.cuda.Stream = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(n_warmup):
            network()
    torch.cuda.current_stream().wait_stream(s)
    with torch.cuda.graph(graph):
        network()
    g = lambda: graph.replay()
    tg = do_bench(g, n_iter=1, n_warmup=n_warmup)
    # if tf < tg:
    #     print('CUDA Graph is slower!', M, K, N, tf, tg)
    return min(tf, tg) / n_layers


def _basic_benchmark(M: int = 64, K: int = 2048, N: int = 2048) -> None:
    dtype: torch.dtype = torch.bfloat16
    device: torch.device = torch.device('cuda')
    weight = torch.empty(N, K, dtype=dtype, device=device)
    w_scale = torch.empty(N, 1, dtype=dtype, device=device)
    w_int_packed = torch.empty(N, K // 2, dtype=torch.uint8, device=device)
    w_int_compressed_packed = torch.empty(N, K // 4, dtype=torch.uint8, device=device)
    w_meta_e = torch.empty(N, K // 64, dtype=torch.uint32, device=device)
    w_meta_e_quik = torch.empty(N, K // 64, dtype=torch.int32, device=device)
    w_add = torch.empty(N, 1, dtype=torch.int32, device=device)
    w_sp_add = torch.empty_like(w_add)
    activation = torch.empty(M, K, dtype=dtype, device=device)
    a_scale = torch.empty(M, 1, dtype=dtype, device=device)
    a_int_packed = torch.empty(M, K // 2, dtype=torch.uint8, device=device)
    a_int_row_sum = torch.empty(M, 1, dtype=torch.int32, device=device)
    c = torch.empty(M, N, dtype=torch.int32, device=device)
    d = torch.empty(M, N, dtype=dtype, device=device)

    t = do_bench(lambda: find_scale_int4(x=activation, scale=a_scale, backend='pytorch'))
    print('scale_pytorch', t * 1e6, sep='\t')
    t = do_bench(lambda: find_scale_int4(x=activation, scale=a_scale, backend='triton'))
    print('scale_triton', t * 1e6, sep='\t')
    tt1 = t = do_bench(lambda: find_scale_int4(x=activation, scale=a_scale, backend='cuda'))
    print('scale_cuda', t * 1e6, sep='\t')

    t = do_bench(lambda: quantize_int4(x=activation, scale=a_scale, x_int=None, x_int_packed=a_int_packed, x_int_row_sum=a_int_row_sum, do_dequantize=False, backend='pytorch'))
    print('quantize_pytorch', t * 1e6, sep='\t')
    t = do_bench(lambda: quantize_int4(x=activation, scale=a_scale, x_int=None, x_int_packed=a_int_packed, x_int_row_sum=a_int_row_sum, do_dequantize=False, backend='triton'))
    print('quantize_triton', t * 1e6, sep='\t')
    tt2 = t = do_bench(lambda: quantize_int4(x=activation, scale=a_scale, x_int=None, x_int_packed=a_int_packed, x_int_row_sum=a_int_row_sum, do_dequantize=False, backend='cuda'))
    print('quantize_cuda', t * 1e6, sep='\t')

    t = do_bench(lambda: add_mul_vv(vec_a_add=a_int_row_sum, vec_a_mul=a_scale, vec_b_add=w_add, vec_b_mul=w_scale, mat_c=c, mat_d=d, backend='pytorch'))
    print('dequantize_pytorch', t * 1e6, sep='\t')
    t = do_bench(lambda: add_mul_vv(vec_a_add=a_int_row_sum, vec_a_mul=a_scale, vec_b_add=w_add, vec_b_mul=w_scale, mat_c=c, mat_d=d, backend='triton'))
    print('dequantize_triton', t * 1e6, sep='\t')
    tt4 = t = do_bench(lambda: add_mul_vv(vec_a_add=a_int_row_sum, vec_a_mul=a_scale, vec_b_add=w_add, vec_b_mul=w_scale, mat_c=c, mat_d=d, backend='cuda'))
    print('dequantize_cuda', t * 1e6, sep='\t')

    tt3 = t = do_bench(lambda: quest.matmul_int4_int4t_int32(mat_a=a_int_packed, mat_b=w_int_packed, out=c))
    print('matmul_quest', t * 1e6, sep='\t')
    t = do_bench(lambda: quest.matmul_int4sp_int4t_int32(mat_a=w_int_compressed_packed, meta_e=w_meta_e, mat_b=a_int_packed, out=c))
    print('matmul_sp_quest', t * 1e6, sep='\t')

    try:
        import quik
    except ImportError:
        print('Skip quik.')
    else:
        t = do_bench(lambda: quik.matmul.int4Matmul(a_int_packed, w_int_packed))
        print('matmul_quik', t * 1e6, sep='\t')

        t = do_bench(lambda: quik.matmul.int4SpMatmul(w_int_compressed_packed, a_int_packed, w_meta_e_quik))
        print('matmul_sp_quik', t * 1e6, sep='\t')

    tt = t = do_bench(lambda: linear_forward(activation=activation, w_int_packed=w_int_packed, w_scale=w_scale, w_add=w_add, w_meta_e=None, out=d, _a_scale=a_scale, _a_int_packed=a_int_packed, _a_int_row_sum=a_int_row_sum, _c=c, use_hadamard=False, backend='triton'))
    print('forward', t * 1e6, sep='\t')
    t = do_bench(lambda: linear_forward(activation=activation, w_int_packed=w_int_compressed_packed, w_scale=w_scale, w_add=w_sp_add, w_meta_e=w_meta_e, out=d, _a_scale=a_scale, _a_int_packed=a_int_packed, _a_int_row_sum=None, _c=None, use_hadamard=False, backend='triton'))
    print('forward_sp', t * 1e6, sep='\t')
    tt_ref = t = do_bench(lambda: activation @ weight.t())
    print('forward_16', t * 1e6, sep='\t')

    print(tt_ref * 1e6, tt * 1e6, (tt1 + tt2) * 1e6, tt3 * 1e6, tt4 * 1e6, (tt - (tt1 + tt2 + tt3 + tt4)) * 1e6, sep='\t')


if __name__ == '__main__':
    # _triton_test()
    _unit_test()
    _basic_benchmark()