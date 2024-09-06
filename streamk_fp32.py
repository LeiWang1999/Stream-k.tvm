import torch
import torch.backends
from bitblas import tvm as tvm
from tvm import tl as TL
import math


def cdiv(a, b):
    return math.ceil(a / b)


# disable tf32
torch.backends.cuda.matmul.allow_tf32 = False

m, n, k = 16, 16384, 8192  # some problem size to test

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
# compute grid (work to do per SM on the first wave)
num_block_m = cdiv(M, BLOCK_SIZE_M)
num_block_n = cdiv(N, BLOCK_SIZE_N)
iters_per_tile = cdiv(K, BLOCK_SIZE_K)
total_tiles = num_block_m * num_block_n

# Two-tile SK + DP
streamk_tiles = total_tiles % streamk_programs
if (
    total_tiles - streamk_tiles > streamk_programs
):  # (total_tiles // total_programs > 1)
    streamk_tiles += streamk_programs

blocking_tiles = total_tiles - streamk_tiles
streamk_iters = streamk_tiles * iters_per_tile

streamk_full_tiles = streamk_iters // streamk_programs
streamk_partial_tiles = streamk_iters % streamk_programs

print(f"{total_tiles=} ")
print(f"{iters_per_tile=} ")

sm_patition_factor = max(blocking_tiles // total_sm, 1)


def tl_matmul_streamk(
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

    @T.macro
    def compute_first_wave(
        pid: T.int32,
        A_buf: T.Buffer,
        A_buf_shared: T.Buffer,
        B_buf: T.Buffer,
        B_buf_shared: T.Buffer,
        C: T.Buffer,
        C_local: T.Buffer,
    ):
        start_iter = T.alloc_fragment((1,), "int32", "local")
        end_iter = T.alloc_fragment((1,), "int32", "local")

        start_iter[0] = pid * streamk_full_tiles + T.min(pid, streamk_partial_tiles)
        last_iter = (pid + 1) * streamk_full_tiles + T.min(
            pid + 1, streamk_partial_tiles
        )

        while start_iter[0] < last_iter:
            end_iter[0] = T.min(
                start_iter[0] + (iters_per_tile - (start_iter[0] % iters_per_tile)),
                last_iter,
            )

            tile_id = start_iter[0] // iters_per_tile
            remain_iters = start_iter[0] % iters_per_tile
            pid_m = tile_id // T.ceildiv(N, block_N)
            pid_n = tile_id % T.ceildiv(N, block_N)

            T.clear(C_local)
            for k in T.serial(0, end_iter[0] - start_iter[0]):
                T.copy(
                    A_buf[
                        pid_m * block_M,
                        (k + (start_iter[0] % iters_per_tile)) * block_K,
                    ],
                    A_buf_shared,
                )
                T.copy(
                    B_buf[
                        (k + (start_iter[0] % iters_per_tile)) * block_K,
                        pid_n * block_N,
                    ],
                    B_buf_shared,
                )
                T.gemm(A_buf_shared, B_buf_shared, C_local)

            # last iteration of the tile always happens before its start on another SM
            if remain_iters == 0 and (end_iter[0] % iters_per_tile == 0):
                T.copy(C_local, C[pid_m * block_M, pid_n * block_N])
            else:
                for i, j in T.Parallel(block_M, block_N):
                    T.atomic_add(
                        C[pid_m * block_M + i, pid_n * block_N + j], C_local[i, j]
                    )

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

        for p in T.serial(sm_patition_factor):
            tile_id = pid + streamk_tiles + p * total_sm
            pid_m = tile_id // T.ceildiv(N, block_N)
            pid_n = tile_id % T.ceildiv(N, block_N)
            T.clear(C_local)
            for k in T.serial(T.ceildiv(K, block_K)):
                T.copy(A_buf[pid_m * block_M, k * block_K], A_shared)
                T.copy(B_buf[k * block_K, pid_n * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[pid_m * block_M, pid_n * block_N])

    @T.prim_func
    def main(
        A: T.Buffer(A_shape, dtypeAB),
        B: T.Buffer(B_shape, dtypeAB),
        C: T.Buffer((M, N), dtypeC),
    ):
        with T.Kernel(streamk_programs, threads=threads) as pid:

            A_shared = T.alloc_shared(A_shared_shape, dtypeAB)
            B_shared = T.alloc_shared(B_shared_shape, dtypeAB)

            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            compute_first_wave(pid, A, A_shared, B, B_shared, C, C_local)

            if sm_patition_factor > 0:
                compute_full_tiles(pid, A, A_shared, B, B_shared, C, C_local)

    return main


_tl_matmul_streamk = tl_matmul_streamk(
    m,
    n,
    k,
    streamk_tiles,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    False,
    False,
    "float32",
    "float32",
    "float32",
    2,
    128,
)

tl_mod, params = TL.lower(_tl_matmul_streamk)
tl_mod = TL.Profiler(tl_mod, params, [], TL.TensorSupplyType.Integer)

b_c = torch.zeros((m, n), device="cuda", dtype=torch.float32)

tl_mod(A, B, b_c)

C = torch.matmul(A, B)

print(b_c)
print(C)
torch.testing.assert_close(C, b_c, rtol=1e-2, atol=1e-2)
