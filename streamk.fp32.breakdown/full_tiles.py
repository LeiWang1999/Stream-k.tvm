import torch
import triton
import triton.language as tl
import random
from bitblas import tvm as tvm
from tvm import tl as TL


@triton.jit
def swizzle_tile(tile_id,
                 # Matrix dimensions
                 M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
                 # Meta-parameters
                 BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                 GROUP_SIZE_M: tl.constexpr):
    grid_m = tl.cdiv(M, BLOCK_SIZE_M)
    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
    width = GROUP_SIZE_M * grid_n
    group_id = tile_id // width
    group_size = tl.minimum(GROUP_SIZE_M, grid_m - group_id * GROUP_SIZE_M)
    pid_m = group_id * GROUP_SIZE_M + (tile_id % group_size)
    pid_n = (tile_id % width) // group_size
    return pid_m, pid_n


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

# similar to the reference matmul kernel
@triton.jit
def full_tiles(
        # Pointers to matrices
        a_ptr, b_ptr, c_ptr,
        # Matrix dimensions
        M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
        # The stride variables represent how much to increase the ptr by when moving by 1
        # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
        # by to get the element one row down (A has M rows).
        stride_am: tl.constexpr, stride_ak: tl.constexpr,  #
        stride_bk: tl.constexpr, stride_bn: tl.constexpr,  #
        stride_cm: tl.constexpr, stride_cn: tl.constexpr,
        # Stream-K parameters
        streamk_tiles,
        ACC_TYPE: tl.constexpr,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr, GROUP_SIZE_M: tl.constexpr):

    tile_id = tl.program_id(axis=0) + streamk_tiles
    # if GROUP_SIZE_M > 0:
    #     # pid swizzle to get better L2 cache performance
    #     pid_m, pid_n = swizzle_tile(tile_id, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M)
    # else:
    pid_m, pid_n = linear_tile(tile_id, M, N, K, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, GROUP_SIZE_M)

    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
                                    offsets=(pid_m * BLOCK_SIZE_M, 0), block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
                                    order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
                                    offsets=(0, pid_n * BLOCK_SIZE_N), block_shape=(BLOCK_SIZE_K, BLOCK_SIZE_N),
                                    order=(1, 0))

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=ACC_TYPE)

    for _ in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_block_ptr, boundary_check=(0, 1))
        b = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc += tl.dot(a, b, out_dtype=ACC_TYPE)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_K, 0))

    c_block_ptr = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
                                    offsets=(pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    tl.store(c_block_ptr, acc, boundary_check=(0, 1))

m, n, k = 512, 512, 512  # some problem size to test

total_sm = 108

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

print(f"{streamk_tiles=} ")
print(f"{blocking_tiles=} ")

def tl_matmul(
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


_tl_matmul = tl_matmul(m, n, k, streamk_tiles, BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K, False, False, "float32", "float32", "float32", 2, 128)
print(_tl_matmul)

# @tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
# def tvm_callback_cuda_postproc(code, _):
#     code = code.replace("float C_local[16];", r"""
#                      float C_local[16];
#                      printf("blockIdx.x: %d, blockIdx.y: %d, threadIdx.x: %d, threadIdx.y: %d\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y);       
# """)
#     return code

mod, params = TL.lower(_tl_matmul)
mod = TL.Profiler(mod, params, [], TL.TensorSupplyType.Integer)
# print(mod.mod.imported_modules[0].get_source())
C = torch.zeros((m, n), device="cuda", dtype=torch.float32)

k2 = full_tiles[(blocking_tiles,)](
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
    streamk_tiles=streamk_tiles,
    ACC_TYPE=ACC_TYPE,
    BLOCK_SIZE_M=BLOCK_SIZE_M,
    BLOCK_SIZE_N=BLOCK_SIZE_N,
    BLOCK_SIZE_K=BLOCK_SIZE_K,
    GROUP_SIZE_M=4,
)

print("C: ", C)

b_c = torch.zeros((m, n), device="cuda", dtype=torch.float32)

mod(A, B, b_c)

print("b_c: ", b_c)

torch.testing.assert_close(C, b_c, rtol=1e-2, atol=1e-2)