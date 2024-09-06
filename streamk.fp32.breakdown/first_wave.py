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

@triton.jit
def linear_tile(tile_id,
                # Matrix dimensions
                M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                # Meta-parameters
                BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                GROUP_SIZE_M: tl.constexpr):
    pid_m = tile_id // tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = tile_id % tl.cdiv(N, BLOCK_SIZE_N)
    return pid_m, pid_n

# Multiply-accumulate loop in GEMM Stream K tiles
@triton.jit
def mac_loop(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, stride_am: tl.constexpr, stride_ak: tl.constexpr,  #
        stride_bk: tl.constexpr, stride_bn: tl.constexpr,  #
        stride_cm: tl.constexpr, stride_cn: tl.constexpr,
        # Stream-K parameters
        iters_per_tile, start_iter, end_iter,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):

    tile_id = start_iter // iters_per_tile
    remain_iters = start_iter % iters_per_tile
    # if GROUP_SIZE_M > 0:
    #     # pid swizzle to get better L2 cache performance
    #     pid_m, pid_n = swizzle_tile(tile_id, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M)
    # else:
    pid_m, pid_n = linear_tile(tile_id, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M)

    a_ptr += BLOCK_SIZE_K * stride_ak * remain_iters
    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    order=(1, 0))
    b_ptr += BLOCK_SIZE_K * stride_bk * remain_iters
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                    order=(1, 0))

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for _ in range(start_iter, end_iter):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc += tl.dot(a, b)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    if remain_iters == 0 and end_iter % iters_per_tile == 0:
        c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                        offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
                                        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
        tl.store(c_block_ptr, acc, boundary_check=(0, 1))
    else:
        rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        c_ptr_ = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
        mask = (rm < M)[:, None] & (rn < N)[None, :]
        tl.atomic_add(c_ptr_, acc, mask=mask)


@triton.jit
def first_wave(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,  #
        stride_am: tl.constexpr, stride_ak: tl.constexpr,  #
        stride_bk: tl.constexpr, stride_bn: tl.constexpr,  #
        stride_cm: tl.constexpr, stride_cn: tl.constexpr,
        # Stream-K parameters
        full_tiles, partial_tiles, iters_per_tile,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):

    pid = tl.program_id(axis=0)
    start_iter = pid * full_tiles + tl.minimum(pid, partial_tiles)
    last_iter = (pid + 1) * full_tiles + tl.minimum(pid + 1, partial_tiles)

    while start_iter < last_iter:
        end_iter = start_iter + (iters_per_tile - start_iter % iters_per_tile)
        end_iter = tl.minimum(end_iter, last_iter)
        mac_loop(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                 iters_per_tile, start_iter, end_iter, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M)

        start_iter = end_iter



m, n, k = 512, 512, 512  # some problem size to test

total_sm = 108

torch.random.manual_seed(0)
A = torch.randn(m, k, device="cuda", dtype=torch.float32)
B = torch.randn(k, n, device="cuda", dtype=torch.float32)

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


def tl_matmul(
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
                for k in T.serial(0, end_iter[0] - start_iter[0], annotations={"pragma_import_c": header}):
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

_tl_matmul = tl_matmul(m, n, k, streamk_programs, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, False, False, "float32", "float32", "float32", 2, 128)
print(_tl_matmul)

@tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
def tvm_callback_cuda_postproc(code, _):
    return code
    
mod, params = TL.lower(_tl_matmul)
mod = TL.Profiler(mod, params, [], TL.TensorSupplyType.Integer)

C = torch.zeros((m, n), device="cuda", dtype=torch.float32)

print("streamk_programs", streamk_programs)

k2 = first_wave[(streamk_programs,)](
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
    streamk_full_tiles,
    streamk_partial_tiles,
    iters_per_tile,
    BLOCK_SIZE_M,
    BLOCK_SIZE_N,
    BLOCK_SIZE_K,
    4
)

print("C: ", C)

b_c = torch.zeros((m, n), device="cuda", dtype=torch.float32)

print("mod source:", mod.mod.imported_modules[0].get_source())

mod(A, B, b_c)

print("b_c: ", b_c)

torch.testing.assert_close(C, b_c, rtol=1e-2, atol=1e-2)

