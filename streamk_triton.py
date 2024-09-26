import torch
import triton
import triton.language as tl
import random
from bitblas import tvm as tvm
from tvm import tl as TL
torch.manual_seed(0)
header = """
__device__ void acquire_lock(int* lock) {
    while (atomicCAS(lock , 1, 1) != 1) {
        // Busy-wait (spin-wait) until the lock is acquired
    }
}
"""

# iterate, multiply and accumulate over K axis
@triton.jit()
def mac_loop(A, B, C,
             M, N, K,
             locks,
             stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
             iters_per_tile,
             start_iter, end_iter,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
             ACC_TYPE: tl.constexpr, GROUP_M: tl.constexpr):

    # where are we in the grid
    tile_id = start_iter // iters_per_tile
    pid_m, pid_n = tile_id // tl.cdiv(N, BLOCK_N), tile_id % tl.cdiv(N, BLOCK_N)

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak) + BLOCK_K * stride_ak * (start_iter % iters_per_tile)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn) + BLOCK_K * stride_bk * (start_iter % iters_per_tile)
    
    # A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    # B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)

    for current_iter in range(start_iter, end_iter):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b, out_dtype=ACC_TYPE)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    if end_iter % iters_per_tile == 0:  # last iteration of the tile always happens before its start on another SM
        C_ = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)  # compute inside the if/else to avoid spilling!
        tl.store(C_, acc)
        if start_iter % iters_per_tile != 0:  # only if tile has been partially processed
            tl.atomic_xchg(locks + tile_id, 1)
    else:
        while tl.atomic_cas(locks + tile_id, 1, 1) != 1:
            pass
        C_ = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)  # compute inside the if/else to avoid spilling!
        # tl.store(C_, acc)
        tl.atomic_add(C_, acc)


@triton.jit()
def first_wave(
    A, B, C,
    M, N, K,
    locks,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    total_full_tiles_streamk, total_partial_tiles_streamk, iters_per_tile,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    start_iter = pid * total_full_tiles_streamk + tl.minimum(pid, total_partial_tiles_streamk)
    last_iter = (pid + 1) * total_full_tiles_streamk + tl.minimum(pid + 1, total_partial_tiles_streamk)

    while start_iter < last_iter:
        end_iter = tl.minimum(start_iter + (iters_per_tile - start_iter % iters_per_tile), last_iter)
        mac_loop(A, B, C,
                 M, N, K,
                 locks,
                 stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 iters_per_tile,
                 start_iter, end_iter,
                 BLOCK_M, BLOCK_N, BLOCK_K, ACC_TYPE,
                 GROUP_M,
                 )
        
        start_iter = end_iter



# similar to the reference matmul kernel
@triton.jit()
def full_tiles(
    A, B, C,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    total_tiles_streamk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, ACC_TYPE: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # first wave has done more tiles than there are SMs, we adjust pid
    tile_id = tl.program_id(0) + total_tiles_streamk
    # if tl.program_id(0) == 0:
    #     print("tile_id: ", tile_id, "total_tiles_streamk: ", total_tiles_streamk, "pid: ", tl.program_id(0))
    # pid_m, pid_n = linear_tile(tile_id, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, GROUP_M)
    pid_m = tile_id // tl.cdiv(N, BLOCK_N)
    pid_n = tile_id % tl.cdiv(N, BLOCK_N)
    # do matrix multiplication
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)
    # pointers
    A = A + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=ACC_TYPE)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(A)
        b = tl.load(B)
        acc += tl.dot(a, b)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk
    acc = acc.to(tl.float32)  # restore C.dtype.element_ty
    # rematerialize rm and rn to save registers
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    C = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(C, acc)


m, n, k = 16, 16384, 8192  # some problem size to test

total_sm = 108

torch.random.manual_seed(0)
A = torch.randn(m, k, device="cuda", dtype=torch.float32)
B = torch.randn(k, n, device="cuda", dtype=torch.float32)

BLK_M = 16
BLK_N = 128
BLK_K = 32
two_tiles = False
M, K = A.shape
_, N = B.shape
# accumulator types
ACC_TYPE = tl.float32 if A.dtype in [torch.float32, torch.bfloat16, torch.float32] else tl.int32
# compute grid (work to do per SM on the first wave)
total_blocks_M = triton.cdiv(M, BLK_M)
total_blocks_N = triton.cdiv(N, BLK_N)
iters_per_tile = triton.cdiv(K, BLK_K)
GROUP_M = 8  # 0 to disable swizzling
total_tiles = total_blocks_M * total_blocks_N
total_programs_streamk = total_sm
if total_programs_streamk > 0:  # Stream-K
    # last wave may occupy less than total_programs_streamk SMs
    total_tiles_streamk = total_tiles % total_programs_streamk
    # for two-tile Stream-K + data-parallel from original paper
    if two_tiles and total_tiles - total_tiles_streamk > total_programs_streamk:
        total_tiles_streamk += total_programs_streamk
    # remaining tiles are computed using classical blocking
    total_blocking_tiles = total_tiles - total_tiles_streamk
    total_iters_streamk = total_tiles_streamk * iters_per_tile
    # iterations related to full waves
    total_full_tiles_streamk = total_iters_streamk // total_programs_streamk
    # iterations related to last (partial) wave
    total_partial_tiles_streamk = total_iters_streamk % total_programs_streamk
else:  # all tiles are computed using classical blocking
    total_blocking_tiles = total_tiles
    total_tiles_streamk = 0
    total_full_tiles_streamk = 0
    total_partial_tiles_streamk = 0
    total_iters_streamk = 0

print(f"{total_tiles_streamk=} ")
print(f"{total_blocking_tiles=} ")
print(f"{total_tiles=} ")
print(f"{total_iters_streamk=} ")
print(f"{total_full_tiles_streamk=} ")
print(f"{iters_per_tile=} ")

locks = torch.zeros((total_tiles_streamk,), device="cuda", dtype=torch.int32)

@tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
def tvm_callback_cuda_postproc(code, _):
    return code


C = torch.zeros((m, n), device="cuda", dtype=torch.float32)

print("total_tiles_streamk", total_tiles_streamk)

k2 = first_wave[(total_blocking_tiles,)](
    A,
    B,
    C,
    M,
    N,
    K,
    locks,
    A.stride(0),
    A.stride(1),
    B.stride(0),
    B.stride(1),
    C.stride(0),
    C.stride(1),
    total_full_tiles_streamk,
    total_partial_tiles_streamk,
    iters_per_tile,
    16,
    128,
    32,
    tl.float32,
    4
)

k1 = full_tiles[(total_blocking_tiles,)](
    A,
    B,
    C,
    M,
    N,
    K,
    A.stride(0),
    A.stride(1),
    B.stride(0),
    B.stride(1),
    C.stride(0),
    C.stride(1),
    total_tiles_streamk=total_tiles_streamk,
    BLOCK_M=BLK_M,
    BLOCK_N=BLK_N,
    BLOCK_K=BLK_K,
    ACC_TYPE=ACC_TYPE,
    GROUP_M=GROUP_M,
    num_stages=3,
    num_warps=4,
)

print("C: ", C)

b_c = torch.matmul(A, B)
torch.testing.assert_close(C, b_c, rtol=1e-2, atol=1e-2)
