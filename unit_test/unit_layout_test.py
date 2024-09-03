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
N = 32

A = torch.rand(M, N, device="cuda", dtype=torch.float16)
B = torch.rand(M, N, device="cuda", dtype=torch.float16)

from tvm import tl
import tvm.tl.language as T


def make_test_layout(A_shared):
    def transform_func(i, j):
        spatial_i, spatial_j = i // 16, j // 32
        warp_i, warp_j = i % 16, j % 32 # 16x32
        new_warp_i = (warp_j // 16) * 16 + warp_i % 16 # 32
        new_warp_j = (warp_i // 16) * 16 + warp_j % 16 # 16
        # transform to 16x32
        new_warp_i, new_warp_j = (new_warp_i * 16 + new_warp_j) // 32, (new_warp_i * 16 + new_warp_j) % 32
        return [
            spatial_i * 16 + new_warp_i,
            spatial_j * 32 + new_warp_j,
        ]

    # atomicAdd can not be vectorized, so we need to reorder dq to match the 8x8 gemm fragment
    return T.Layout(A_shared.shape, transform_func)


def tl_add(
    M,
    N,
    dtypeAB,
):

    block_M = 32
    block_N = block_col_warps * warp_col_tiles
    block_K = chunk

    A_shape = (M, N)
    B_shape = (M, N)
    A_shared_shape = (block_M, block_K)

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size = (micro_size_x * micro_size_y) // warp_size

    
    @T.prim_func
    def main(
        A: T.Buffer(A_shape, dtypeAB),
        B: T.Buffer(B_shape, dtypeAB),
    ):
        with T.Kernel(M // block_M, threads=1) as bx:

            A_shared = T.alloc_shared((block_M, N), dtypeAB, scope=shared_scope)
            A_local = T.alloc_fragment(
                (block_M, N,),
                dtypeAB,
                scope="local",
            )
            T.annotate_layout(
                {A_shared: make_test_layout(A_shared)}
            )
            for i, j in T.grid(block_M, N):
                A_shared[i, j] = A[bx * block_M + i, j]

            for i, j in T.grid(block_M, N):
                A_local[i, j] = A_shared[i, j] + 1

            for i, j in T.grid(block_M, N):
                B[bx * block_M + i, j] = A_local[i, j]

    return main


matmul = tl_add(M, N, "float16")

print(matmul)


mod, params = TL.lower(matmul)
print(mod.imported_modules[0].get_source())
mod = TL.Profiler(mod, params, [], TL.TensorSupplyType.Integer)
mod(A, B)

print(B)

ref_c = torch.add(A, 1)
print(ref_c)
torch.testing.assert_allclose(B, ref_c, rtol=1e-2, atol=1e-2)
