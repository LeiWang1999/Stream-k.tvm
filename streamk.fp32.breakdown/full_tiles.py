import torch
import triton
import triton.language as tl
import random
from bitblas import tvm as tvm
from tvm import tl as TL



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

def tl_matmul(
    M,
    N,
    K,
    total_tiles_streamk,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    dtypeAB,
    dtypeC,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (M, K)
    B_shape = (K, N)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_K, block_N)

    import tvm.tl.language as T

    @T.prim_func
    def main(A: T.Buffer(A_shape, dtypeAB), B: T.Buffer(B_shape, dtypeAB), C: T.Buffer((M, N), dtypeC)):
        with T.Kernel(total_sm) as (pid):
            A_shared = T.alloc_shared(A_shared_shape, dtypeAB)
            B_shared = T.alloc_shared(B_shared_shape, dtypeAB)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            
            tile_id = pid + total_tiles_streamk
            pid_m = tile_id // T.ceildiv(N, block_N)
            pid_n = tile_id % T.ceildiv(N, block_N)
            T.clear(C_local)
            for k in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[pid_m * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, pid_n * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[pid_m * block_M, pid_n * block_N])

    return main


_tl_matmul = tl_matmul(m, n, k, total_tiles_streamk, 16, 128, 32, False, False, "float32", "float32", "float32", 2, 256)
print(_tl_matmul)
mod, params = TL.lower(_tl_matmul)
mod = TL.Profiler(mod, params, [], TL.TensorSupplyType.Integer)

C = torch.zeros((m, n), device="cuda", dtype=torch.float32)
print("total_tiles_streamk", total_tiles_streamk)
k2 = full_tiles[(total_blocking_tiles,)](
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

b_c = torch.zeros((m, n), device="cuda", dtype=torch.float32)
print("mod source:", mod.mod.imported_modules[0].get_source())

mod(A, B, b_c)

print("b_c: ", b_c)
