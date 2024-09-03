import torch
import torch.backends
import bitblas
from bitblas import tvm as tvm
from tvm import tl as TL

from bitblas.base.arch import CUDA
from bitblas.utils import auto_detect_nvidia_target

def mma_32x8_to_shared_16x16_layout(thread_id, local_id):
    row = 8 * (local_id % 4 // 2) + (thread_id // 4)
    col = 8 * (local_id // 4) + (thread_id % 4) * 2 + (local_id % 2)
    return row, col

index_map_rev = mma_32x8_to_shared_16x16_layout


arch = CUDA(auto_detect_nvidia_target())
intrin_info = bitblas.base.hint.IntrinInfo(
    in_dtype="float16",
    out_dtype="float32",
    trans_b=True,
)
config = bitblas.base.Hint.from_dict(
    {
        "arch": arch,
        "block": [64, 64],
        "warp": [32, 32],
        "rstep": [32],
        "pipeline_stage": 1,
        "use_async": False,
        "intrin_info": intrin_info,
        "shared_scope": "shared",
        "vectorize": {"b": 8, "a": 8},
    }
)


warp_row_tiles = config.warp[0]
warp_col_tiles = config.warp[1]
block_row_warps = config.block[0] // warp_row_tiles
block_col_warps = config.block[1] // warp_col_tiles
stage = config.pipeline_stage
use_async = config.use_async
chunk = config.rstep[0]
# tensor core intrinsic size
shared_scope = config.shared_scope
micro_size_x, micro_size_y, micro_size_k = 16, 16, 16

M = 256
N = 256
K = 256

A = torch.rand(M, K, device="cuda", dtype=torch.float16)
B = torch.rand(N, K, device="cuda", dtype=torch.float16)
C = torch.zeros(M, N, device="cuda", dtype=torch.float32)

def tl_matmul(
    M,
    N,
    K,
    dtypeAB,
    dtypeC,
    accum_dtype,
):

    block_M = block_row_warps * warp_row_tiles
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shape = (M, K)
    B_shape = (N, K)
    A_shared_shape = (block_M, block_K)
    B_shared_shape = (block_N, block_K)
    C_shared_shape = (block_M // micro_size_x, block_N // micro_size_y, micro_size_x, micro_size_y)

    import tvm.tl.language as T
    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size = (micro_size_x * micro_size_y) // warp_size

    @T.prim_func
    def main(A: T.Buffer(A_shape, dtypeAB), B: T.Buffer(B_shape, dtypeAB), C: T.Buffer((M, N), dtypeC)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):

            A_shared = T.alloc_shared(A_shared_shape, dtypeAB, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, dtypeAB, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, dtypeC, scope=shared_scope)
            A_local = T.alloc_fragment(((warp_row_tiles // micro_size_x) * local_size), dtypeAB, scope="local")
            B_local = T.alloc_fragment(((warp_col_tiles // micro_size_y) * local_size), dtypeAB, scope="local")
            C_local = T.alloc_fragment(((warp_row_tiles // micro_size_x) * (warp_col_tiles // micro_size_y) * local_size), accum_dtype, scope="local")
            thread_bindings = T.thread_binding(0, threads, "threadIdx.x")
            tx = thread_bindings % warp_size
            ty = (thread_bindings // warp_size) % block_row_warps
            tz = (thread_bindings // (warp_size * block_row_warps))

            for i in T.serial(((warp_row_tiles // micro_size_x) * (warp_col_tiles // micro_size_y) * local_size)):
                C_local[i] = 0

            for ko in T.Pipelined((K // block_K), num_stages=2):
                # TODO(lei): storage sync should be able to be injected automatically by TVM Pass
                T.tvm_storage_sync("shared")
                # Load A into shared memory
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]
                # Load B into shared memory
                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                # TODO(lei): storage sync should be able to be injected automatically by TVM Pass
                T.tvm_storage_sync("shared")

                for ki in T.serial(0, (block_K // micro_size_k)):
                    # Load A into fragment
                    for i in T.serial(warp_row_tiles // micro_size_x):
                        T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", A_local.data, i * local_size, T.address_of(A_shared[ty * warp_row_tiles + i * micro_size_x, ki * micro_size_k]), block_K * (tx % 16) + 8 * (tx // 16))

                    # Load B into fragment
                    for j in T.serial(warp_col_tiles // micro_size_y):
                        T.ptx_ldmatrix("float16", T.bool(False), 4, ".b16", B_local.data, j * local_size, T.address_of(B_shared[tz * warp_col_tiles + j * micro_size_y, ki * micro_size_k]), block_K * 8 * (tx // 16) + block_K * (tx % 8) + 8 * (tx % 16 // 8))

                    # Apply MMA
                    for i, j in T.grid((warp_row_tiles // micro_size_x), (warp_col_tiles // micro_size_y)):
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, i * local_size, B_local.data, j * local_size, C_local.data, i * (warp_col_tiles // micro_size_y) * local_size + j * local_size, T.bool(False))
                        T.ptx_mma("float32", "m16n8k16", "row", "col", "fp16", "fp16", "fp32", A_local.data, i * local_size, B_local.data, j * local_size + 4, C_local.data, i * (warp_col_tiles // micro_size_y) * local_size + j * local_size + 4, T.bool(False))

                # STS
                # MMA Store must be in simulated instead of TVM Intrins
                # As TVM Intrins is like a hack that the threadIdx.x should be always
                # equal to the warp_size
                for i, j in T.grid((warp_row_tiles // micro_size_x), (warp_col_tiles // micro_size_y)):
                    for local_id in T.serial(local_size):
                        row, col = T.meta_var(index_map_rev(tx, local_id))
                        C_shared[ty * (warp_row_tiles // micro_size_x) + i, tz * (warp_col_tiles // micro_size_y) + j, row, col] = C_local[i * (warp_col_tiles // micro_size_y * local_size) + j * local_size + local_id]

                for i, j in T.Parallel(block_M, block_N):
                    C[by * block_M + i, bx * block_N + j] = C_shared[i // micro_size_x, j // micro_size_y, i % micro_size_x, j % micro_size_y]

    return main


matmul = tl_matmul(
    M, N, K, "float16", "float32", "float32")

print(matmul)

@tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
def tvm_callback_cuda_postproc(code, _):
    return code

mod, params = TL.lower(matmul)
print(mod.imported_modules[0].get_source())
mod = TL.Profiler(mod, params, [], TL.TensorSupplyType.Integer)
mod(A, B, C)

print(C)

ref_c = torch.matmul(A, B.T).float()
print(ref_c)
torch.testing.assert_allclose(C, ref_c, rtol=1e-2, atol=1e-2)
