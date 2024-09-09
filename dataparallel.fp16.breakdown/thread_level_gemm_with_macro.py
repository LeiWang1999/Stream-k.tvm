import torch
import torch.backends
import bitblas
from bitblas import tvm as tvm
from tvm import tl as TL
from bitblas.base.arch import CUDA
from bitblas.base.roller.rasterization import Rasterization2DColumn
from bitblas.utils import auto_detect_nvidia_target
from bitblas.tl.utils import get_swizzle_layout
from bitblas.tl.macro_generator import TensorCorePTXMacroGenerator

torch.manual_seed(0)

VERIFY_CORRECTNESS = True
in_dtype = "float16"
accum_dtype = "float16"
# accum_dtype = "float16"
# Support we're from a config file
arch = CUDA(auto_detect_nvidia_target())
intrin_info = bitblas.base.hint.IntrinInfo(
    in_dtype=in_dtype,
    out_dtype=accum_dtype,
    trans_b=True,
)
config = bitblas.base.Hint.from_dict(
    {
        "arch": arch,
        "block": [16, 64],
        "warp": [16, 16],
        "rstep": [32],
        "pipeline_stage": 2,
        "use_async": False,
        "intrin_info": intrin_info,
        "shared_scope": "shared.dyn",
        "vectorize": {"b": 8, "a": 8},
        "rasterization_plan": Rasterization2DColumn(10),
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
# for L2 cache
swizzle_panel_size = config.rasterization_plan.panel_width
device_func, invoke_func = config.rasterization_plan.get_code()

M = 16384
N = 16384
K = 16384
if VERIFY_CORRECTNESS:
    M = 256
    N = 512
    K = 32

A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
B = torch.rand(N, K, device="cuda", dtype=getattr(torch, in_dtype))
C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))


import tvm.tl.language as T


def make_swizzle_layout(shared_buf):
    dtype = shared_buf.dtype
    shape = shared_buf.shape

    def transform_func(i, j):
        new_warp_i, new_warp_j = get_swizzle_layout(i, j, shape[-1], dtype)
        return [new_warp_i, new_warp_j]

    return T.Layout(shape, transform_func)


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

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    ptx_macro_generator = TensorCorePTXMacroGenerator(
        a_dtype=dtypeAB, b_dtype=dtypeAB, accum_dtype=accum_dtype,
        a_transposed=False, b_transposed=True, block_row_warps=block_row_warps,
        block_col_warps=block_col_warps, warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles, chunk=chunk
    )

    @T.prim_func
    def main(
        A: T.Buffer(A_shape, dtypeAB),
        B: T.Buffer(B_shape, dtypeAB),
        C: T.Buffer((M, N), dtypeC),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads
        ) as (bx, by):

            A_shared = T.alloc_shared(A_shared_shape, dtypeAB, scope=shared_scope)
            B_shared = T.alloc_shared(B_shared_shape, dtypeAB, scope=shared_scope)
            C_shared = T.alloc_shared(C_shared_shape, dtypeC, scope=shared_scope)
            A_local = T.alloc_fragment((warp_rows * local_size), dtypeAB, scope="local")
            B_local = T.alloc_fragment((warp_cols * local_size), dtypeAB, scope="local")
            C_local = T.alloc_fragment((warp_rows * warp_cols * local_size), accum_dtype, scope="local")
            thread_bindings = T.thread_binding(0, threads, "threadIdx.x")

            T.annotate_layout(
                {
                    A_shared: make_swizzle_layout(A_shared),
                    B_shared: make_swizzle_layout(B_shared),
                }
            )

            T.use_swizzle(panel_size=10)

            T.clear(C_local)

            for ko in T.Pipelined((K // block_K), num_stages=stage):

                # Load A into shared memory
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                # Load B into shared memory
                for j, k in T.Parallel(block_N, block_K):
                    B_shared[j, k] = B[bx * block_N + j, ko * block_K + k]

                for ki in T.serial(0, (block_K // micro_size_k)):

                    # Load A into fragment
                    ptx_macro_generator.LDMATRIX_A(
                        ptx_macro_generator,
                        A_local,
                        A_shared,
                        ki,
                        thread_bindings=thread_bindings,
                    )

                    # Load B into fragment
                    ptx_macro_generator.LDMATRIX_B(
                        ptx_macro_generator,
                        B_local,
                        B_shared,
                        ki,
                        thread_bindings=thread_bindings,
                    )

                    ptx_macro_generator.MMA(
                        ptx_macro_generator,
                        A_local,
                        B_local,
                        C_local
                    )

            ptx_macro_generator.STMATRIX(
                ptx_macro_generator,
                C_local,
                C_shared,
                thread_bindings=thread_bindings,
            )

            for i, j in T.Parallel(block_M, block_N):
                C[by * block_M + i, bx * block_N + j] = C_shared[i // micro_size_x, j // micro_size_y, i % micro_size_x, j % micro_size_y]

    return main


matmul = tl_matmul(M, N, K, in_dtype, accum_dtype, accum_dtype)

print(matmul)


@tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
def tvm_callback_cuda_postproc(code, _):
    return code


mod, params = TL.lower(matmul)
print(mod.imported_modules[0].get_source())
mod = TL.Profiler(mod, params, [], TL.TensorSupplyType.Integer)

mod(A, B, C)

latency = mod.do_bench(mod.func, warmup = 25)
print(f"Latency: {latency}")


if not VERIFY_CORRECTNESS:
    exit()

print(C)
ref_c = torch.matmul(A, B.T).to(getattr(torch, accum_dtype))
print(ref_c)
torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)
