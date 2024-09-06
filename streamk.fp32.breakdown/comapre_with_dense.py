import torch
import torch.backends
import triton
import triton.language as tl
import random
from bitblas import tvm as tvm
from tvm import tl as TL
# disable tf32
torch.backends.cuda.matmul.allow_tf32 = False

m, n, k = 512, 512, 512  # some problem size to test

total_sm = 108

torch.random.manual_seed(0)
# uniform distribution from -1 to 1
A = torch.rand(m, k, device="cuda", dtype=torch.float32) * 2 - 1
B = torch.rand(k, n, device="cuda", dtype=torch.float32) * 2 - 1


streamk_programs = total_sm
BLOCK_SIZE_M = 32
BLOCK_SIZE_N = 64
BLOCK_SIZE_K = 32
two_tiles = False
M, K = A.shape
K, N = B.shape
# accumulator types
ACC_TYPE = tl.float32 if A.dtype in [torch.float32, torch.bfloat16, torch.float32] else tl.int32
# compute grid (work to do per SM on the first wave)
num_block_m = triton.cdiv(M, BLOCK_SIZE_M)
num_block_n = triton.cdiv(N, BLOCK_SIZE_N)
iters_per_tile = triton.cdiv(K, BLOCK_SIZE_K)
total_tiles = num_block_m * num_block_n

# Two-tile SK + DP
streamk_tiles = total_tiles % streamk_programs
if total_tiles - streamk_tiles > streamk_programs:  # (total_tiles // total_programs > 1)
    streamk_tiles += streamk_programs

blocking_tiles = total_tiles - streamk_tiles
streamk_iters = streamk_tiles * iters_per_tile

streamk_full_tiles = streamk_iters // streamk_programs
streamk_partial_tiles = streamk_iters % streamk_programs

print(f"{total_tiles=} ")
print(f"{iters_per_tile=} ")



def tl_first_wave(
    M,
    N,
    K,
    streamk_programs,
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
        with T.Kernel(streamk_programs, threads=threads) as pid:
            
            A_shared = T.alloc_shared(A_shared_shape, dtypeAB)
            B_shared = T.alloc_shared(B_shared_shape, dtypeAB)
            
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            start_iter = T.alloc_fragment((1, ), "int32", "local")
            end_iter = T.alloc_fragment((1, ), "int32", "local")

            start_iter[0] = pid * streamk_full_tiles + T.min(pid, streamk_partial_tiles)
            last_iter = (pid + 1) * streamk_full_tiles + T.min(pid + 1, streamk_partial_tiles)
            
            while start_iter[0] < last_iter:
                end_iter[0] = T.min(start_iter[0] + (iters_per_tile - (start_iter[0] % iters_per_tile)), last_iter)
                
                tile_id = start_iter[0] // iters_per_tile
                remain_iters = start_iter[0] % iters_per_tile
                pid_m = tile_id // T.ceildiv(N, block_N)
                pid_n = tile_id % T.ceildiv(N, block_N)
                
                T.clear(C_local)
                for k in T.serial(0, end_iter[0] - start_iter[0]):
                    T.copy(A[pid_m * block_M, (k + (start_iter[0] % iters_per_tile)) * block_K], A_shared)
                    T.copy(B[(k + (start_iter[0] % iters_per_tile)) * block_K, pid_n * block_N], B_shared)
                    T.gemm(A_shared, B_shared, C_local)

                # last iteration of the tile always happens before its start on another SM
                if remain_iters == 0 and (end_iter[0] % iters_per_tile == 0):
                    T.copy(C_local, C[pid_m * block_M, pid_n * block_N])
                else:
                    for i, j in T.Parallel(block_M, block_N):
                        T.atomic_add(C[pid_m * block_M + i, pid_n * block_N + j], C_local[i, j])

                start_iter[0] = end_iter[0]
            
    return main


_tl_first_wave = tl_first_wave(
    m, n, k, streamk_programs, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, False, False, "float32", "float32", "float32", 2, 128)

def tl_full_tiles(
    M,
    N,
    K,
    streamk_tiles,
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
        with T.Kernel(blocking_tiles) as (pid):
            A_shared = T.alloc_shared(A_shared_shape, dtypeAB)
            B_shared = T.alloc_shared(B_shared_shape, dtypeAB)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            # TODO(lei): There's a bug in tl that when blocking_tiles is zero.
            # The following code will still be executed with only one grid
            # if blocking_tiles > 0:
            tile_id = pid + streamk_tiles
            pid_m = tile_id // T.ceildiv(N, block_N)
            pid_n = tile_id % T.ceildiv(N, block_N)
            T.clear(C_local)
            for k in T.serial(T.ceildiv(K, block_K)):
                T.copy(A[pid_m * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, pid_n * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[pid_m * block_M, pid_n * block_N])

    return main


_tl_full_tiles = tl_full_tiles(m, n, k, streamk_tiles, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, False, False, "float32", "float32", "float32", 2, 128)

first_wave_mod, params = TL.lower(_tl_first_wave)
first_wave_mod = TL.Profiler(first_wave_mod, params, [], TL.TensorSupplyType.Integer)

full_tiles_mod, params = TL.lower(_tl_full_tiles)
full_tiles_mod = TL.Profiler(full_tiles_mod, params, [], TL.TensorSupplyType.Integer)

b_c = torch.zeros((m, n), device="cuda", dtype=torch.float32)

first_wave_mod(A, B, b_c)
full_tiles_mod(A, B, b_c)

C = torch.matmul(A, B)

print(b_c)
print(C)
torch.testing.assert_close(C, b_c, rtol=1e-2, atol=1e-2)
