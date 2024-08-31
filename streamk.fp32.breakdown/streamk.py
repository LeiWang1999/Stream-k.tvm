import torch
import torch.backends
import triton
import triton.language as tl
import random
from bitblas import tvm as tvm
from tvm import tl as TL
# disable tf32
torch.backends.cuda.matmul.allow_tf32 = False
header = """
__device__ void acquire_lock(int* lock) {
    while (atomicCAS(lock , 1, 1) != 1) {
        // Busy-wait (spin-wait) until the lock is acquired
    }
}
"""


m, n, k = 16, 16384, 8192  # some problem size to test

total_sm = 108

torch.random.manual_seed(0)
# uniform distribution from -1 to 1
A = torch.rand(m, k, device="cuda", dtype=torch.float32) * 2 - 1
B = torch.rand(k, n, device="cuda", dtype=torch.float32) * 2 - 1

BLK_M = 16
BLK_N = 128
BLK_K = 32
two_tiles = False
M, K = A.shape
_, N = B.shape
# compute grid (work to do per SM on the first wave)
total_blocks_M = triton.cdiv(M, BLK_M)
total_blocks_N = triton.cdiv(N, BLK_N)
iters_per_tile = triton.cdiv(K, BLK_K)
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


def tl_matmul_streamk(
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
    @T.macro
    def compute_first_wave(
        pid: T.int32,
        A_buf: T.Buffer,
        A_buf_shared: T.Buffer,
        B_buf: T.Buffer,
        B_buf_shared: T.Buffer,
        C: T.Buffer,
        C_local: T.Buffer,
        locks_buf: T.Buffer,
    ):
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
                T.copy(A_buf[pid_m * block_M, (k + (start_iter[0] % iters_per_tile)) * block_K], A_buf_shared)
                T.copy(B_buf[(k + (start_iter[0] % iters_per_tile)) * block_K, pid_n * block_N], B_buf_shared)
                T.gemm(A_buf_shared, B_buf_shared, C_local)

            # last iteration of the tile always happens before its start on another SM
            if end_iter[0] % iters_per_tile == 0:
                T.copy(C_local, C[pid_m * block_M, pid_n * block_N])
                if start_iter[0] % iters_per_tile: 
                    # only if tile has been partially processed
                    T.call_extern("handle", "atomicExch", T.address_of(locks_buf[tile_id]), 1)
            else:
                T.call_extern("handle", "acquire_lock", T.address_of(locks_buf[tile_id]))
                for i, j in T.Parallel(block_M, block_N):
                    T.atomic_add(C[pid_m * block_M + i, pid_n * block_N + j], C_local[i, j])

            start_iter[0] = end_iter[0] 
    
    @T.macro
    def compute_full_tiles(
        pid: T.int32,
        A_buf: T.Buffer,
        A_shared: T.Buffer,
        B_buf: T.Buffer,
        B_shared: T.Buffer,
        C: T.Buffer,
        C_local: T.Buffer,
    ):
        
        tile_id = pid + total_tiles_streamk
        pid_m = tile_id // T.ceildiv(N, block_N)
        pid_n = tile_id % T.ceildiv(N, block_N)
        T.clear(C_local)
        for k in T.serial(T.ceildiv(K, block_K)):
            T.copy(A_buf[pid_m * block_M, k * block_K], A_shared)
            T.copy(B_buf[k * block_K, pid_n * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[pid_m * block_M, pid_n * block_N])



    @T.prim_func
    def main(A: T.Buffer(A_shape, dtypeAB), B: T.Buffer(B_shape, dtypeAB), C: T.Buffer((M, N), dtypeC), locks: T.Buffer((total_tiles_streamk,), "int32")):
        with T.Kernel(total_sm, threads=threads) as pid:
            
            A_shared = T.alloc_shared(A_shared_shape, dtypeAB)
            B_shared = T.alloc_shared(B_shared_shape, dtypeAB)
            
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            compute_first_wave(pid, A, A_shared, B, B_shared, C, C_local, locks)
            compute_full_tiles(pid, A, A_shared, B, B_shared, C, C_local)
            
    return main



_tl_matmul_streamk= tl_matmul_streamk(
    m, n, k, total_tiles_streamk, 16, 128, 32, False, False, "float32", "float32", "float32", 2, 128
)

tl_mod, params = TL.lower(_tl_matmul_streamk)
tl_mod = TL.Profiler(tl_mod, params, [], TL.TensorSupplyType.Integer)

b_c = torch.zeros((m, n), device="cuda", dtype=torch.float32)

locks = torch.zeros((total_tiles_streamk,), device="cuda", dtype=torch.int32)
tl_mod(A, B, b_c, locks)

C = torch.matmul(A, B)

print(b_c)
print(C)
torch.testing.assert_close(C, b_c, rtol=1e-2, atol=1e-2)
