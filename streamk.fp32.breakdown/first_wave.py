import torch
import triton
import triton.language as tl
import random
from bitblas import tvm as tvm
from tvm import tl as TL

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
    def main(A: T.Buffer(A_shape, dtypeAB), B: T.Buffer(B_shape, dtypeAB), C: T.Buffer((M, N), dtypeC), locks: T.Buffer((total_tiles_streamk,), "int32")):
        with T.Kernel(total_sm, threads=threads) as pid:
            
            A_shared = T.alloc_shared(A_shared_shape, dtypeAB)
            B_shared = T.alloc_shared(B_shared_shape, dtypeAB)
            
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            start_iter = T.alloc_fragment((1, ), "int32", "local")
            end_iter = T.alloc_fragment((1, ), "int32", "local")

            start_iter[0] = pid * total_full_tiles_streamk + T.min(pid, total_partial_tiles_streamk)
            last_iter = (pid + 1) * total_full_tiles_streamk + T.min(pid + 1, total_partial_tiles_streamk)
            
            while start_iter[0] < last_iter:
                end_iter[0] = T.min(start_iter[0] + (iters_per_tile - (start_iter[0] % iters_per_tile)), last_iter)
                
                tile_id = start_iter[0] // iters_per_tile
                pid_m = tile_id // T.ceildiv(N, block_N)
                pid_n = tile_id % T.ceildiv(N, block_N)
                
                T.clear(C_local)
                for k in T.serial(0, end_iter[0] - start_iter[0], annotations={"pragma_import_c": header}):
                    T.copy(A[pid_m * block_M, (k + (start_iter[0] % iters_per_tile)) * block_K], A_shared)
                    T.copy(B[(k + (start_iter[0] % iters_per_tile)) * block_K, pid_n * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

                # last iteration of the tile always happens before its start on another SM
                if end_iter[0] % iters_per_tile == 0:
                    T.copy(C_local, C[pid_m * block_M, pid_n * block_N])
                    if start_iter[0] % iters_per_tile: 
                        # only if tile has been partially processed
                        T.call_extern("handle", "atomicExch", T.address_of(locks[tile_id]), 1)
                else:
                    T.call_extern("handle", "acquire_lock", T.address_of(locks[tile_id]))
                    for i, j in T.Parallel(block_M, block_N):
                        T.atomic_add(C[pid_m * block_M + i, pid_n * block_N + j], C_local[i, j])

                start_iter[0] = end_iter[0]
            

    return main

_tl_matmul = tl_matmul(m, n, k, total_tiles_streamk, 16, 128, 32, False, False, "float32", "float32", "float32", 2, 128)
print(_tl_matmul)

@tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
def tvm_callback_cuda_postproc(code, _):
    return code
    
mod, params = TL.lower(_tl_matmul)
mod = TL.Profiler(mod, params, [], TL.TensorSupplyType.Integer)

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

print("C: ", C)

b_c = torch.zeros((m, n), device="cuda", dtype=torch.float32)

print("mod source:", mod.mod.imported_modules[0].get_source())

locks = torch.zeros((total_tiles_streamk,), device="cuda", dtype=torch.int32)
mod(A, B, b_c, locks)

print("b_c: ", b_c)

torch.testing.assert_close(C, b_c, rtol=1e-2, atol=1e-2)

