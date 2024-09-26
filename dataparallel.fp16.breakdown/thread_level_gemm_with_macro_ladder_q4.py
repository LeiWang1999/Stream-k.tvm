import torch
import torch.backends
import bitblas
from bitblas import tvm as tvm
from tvm import tl as TL
from tvm import DataType
from bitblas.base.arch import CUDA
from bitblas.base.roller.rasterization import Rasterization2DColumn
from bitblas.utils import auto_detect_nvidia_target
from bitblas.tl.utils import get_swizzle_layout
from bitblas.quantization import _tir_packed_to_unsigned_convert
from bitblas.tl.macro_generator import TensorCorePTXMacroGeneratorWithLadderTransform
from bitblas.gpu.intrin.lop3 import decode_i4_to_f16
torch.manual_seed(0)

num_bits = 4
num_elems_per_byte = 8 // num_bits
storage_dtype = "int8"

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

transform_a = 0
transform_b = 3

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
    N = 1024
    K = 512

import tvm.tl.language as T


def make_swizzle_layout(shared_buf, is_smooth=False):
    dtype = shared_buf.dtype
    shape = shared_buf.shape
    if is_smooth:
        return T.Layout(shape, lambda *args: args)

    can_swizzle = shape[-1] * DataType(dtype).bits == 512
    if not can_swizzle:
        print(f"shape is not swizzlable: {shape} {dtype}")
        return T.Layout(shape, lambda *args: args)

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

    is_smooth_a = transform_a >= 2
    can_swizzle = block_K * DataType(in_dtype).bits == 512
    apply_pad_a = not (is_smooth_a or can_swizzle)
    pad_factor = 8
    A_shape = (M, K)
    B_shape = (N // micro_size_y, K // micro_size_k, micro_size_y, micro_size_k // num_elems_per_byte)
    A_shared_shape = (block_M, (block_K + pad_factor) if apply_pad_a else block_K)
    B_shared_shape = (
        block_N // micro_size_y,
        block_K // micro_size_k,
        micro_size_y,
        micro_size_k // num_elems_per_byte,
    )
    C_shared_shape = (block_M // micro_size_x, block_N // micro_size_y, micro_size_x, micro_size_y)

    warp_size = 32
    threads = warp_size * (block_row_warps * block_col_warps)
    local_size = (micro_size_x * micro_size_y) // warp_size
    warp_rows = warp_row_tiles // micro_size_x
    warp_cols = warp_col_tiles // micro_size_y

    ptx_macro_generator = TensorCorePTXMacroGeneratorWithLadderTransform(
        a_dtype=dtypeAB, b_dtype=dtypeAB, accum_dtype=accum_dtype,
        a_transposed=False, b_transposed=True, block_row_warps=block_row_warps,
        block_col_warps=block_col_warps, warp_row_tiles=warp_row_tiles,
        warp_col_tiles=warp_col_tiles, chunk=chunk,
        transform_kind_b=transform_b, num_elems_per_byte=num_elems_per_byte
    )

    vec_load_qb = 16
    if block_N * (block_K) // num_elems_per_byte // threads < vec_load_qb:
        vec_load_qb = block_N * (block_K) // num_elems_per_byte // threads
    @T.prim_func
    def main(
        A: T.Buffer(A_shape, dtypeAB),
        B: T.Buffer(B_shape, storage_dtype),
        C: T.Buffer((M, N), dtypeC),
    ):
        with T.Kernel(
            T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads, prelude=decode_i4_to_f16
        ) as (bx, by):

            A_shared = T.alloc_shared(A_shared_shape, dtypeAB, scope=shared_scope)
            B_shared = T.alloc_shared(
                B_shared_shape, storage_dtype, scope=shared_scope
            )
            C_shared = T.alloc_shared(C_shared_shape, dtypeC, scope=shared_scope)
            A_local = T.alloc_fragment((warp_rows * local_size), dtypeAB, scope="local")
            B_local = T.alloc_fragment((warp_cols * local_size // num_elems_per_byte), storage_dtype, scope="local")
            B_dequantize_local = T.alloc_fragment((warp_cols * local_size), dtypeAB, scope="local")
            C_local = T.alloc_fragment((warp_rows * warp_cols * local_size), accum_dtype, scope="local")
            thread_bindings = T.thread_binding(0, threads, "threadIdx.x")

            T.annotate_layout(
                {
                    A_shared: make_swizzle_layout(A_shared),
                    B_shared: make_swizzle_layout(B_shared, is_smooth=True),
                }
            )

            T.use_swizzle(panel_size=10)

            T.clear(C_local)

            for ko in T.Pipelined((K // block_K), num_stages=stage):

                # Load A into shared memory
                for i, k in T.Parallel(block_M, block_K):
                    A_shared[i, k] = A[by * block_M + i, ko * block_K + k]

                # Load B into shared memory
                # for j, k, jj, kk in T.Parallel(
                #     (block_N // micro_size_y), block_K // micro_size_k, micro_size_y, micro_size_k // num_elems_per_byte
                # ):
                #     B_shared[j, k, jj, kk] = B[bx * (block_N // micro_size_y) + j, ko * (block_K // micro_size_k) + k, jj, kk]
                
                # TODO(lei): Layout Inference Pass is not efficient to handle the four dims int8 load
                for i in T.serial(block_N * block_K // num_elems_per_byte // (threads * vec_load_qb)):
                    for v in T.vectorized(0, vec_load_qb):
                        t = thread_bindings
                        idx = i * threads * vec_load_qb + t * vec_load_qb + v
                        vkk = idx % (micro_size_k // num_elems_per_byte)
                        vjj = (idx // (micro_size_k // num_elems_per_byte)) % micro_size_y
                        vk = (idx // (micro_size_k // num_elems_per_byte) // micro_size_y) % (block_K // micro_size_k)
                        vj = (idx // (micro_size_k // num_elems_per_byte) // micro_size_y // (block_K // micro_size_k)) % (block_N // micro_size_y)
                        B_shared[vj, vk, vjj, vkk] = B[bx * (block_N // micro_size_y) + vj, ko * (block_K // micro_size_k) + vk, vjj, vkk]

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

                    for j in T.serial(warp_cols):
                        local_size_b = ptx_macro_generator.local_size_b
                        T.call_extern('handle', 'decode_i4u_to_f16', T.address_of(B_local[j * local_size_b // num_elems_per_byte]), 
                                           T.address_of(B_dequantize_local[j * local_size_b]), 8)
        

                    ptx_macro_generator.MMA(
                        ptx_macro_generator,
                        A_local,
                        B_dequantize_local,
                        C_local
                    )

            ptx_macro_generator.STMATRIX(
                ptx_macro_generator,
                C_local,
                C_shared,
                thread_bindings=thread_bindings,
            )

            for i, j in T.Parallel(block_M, block_N):
                vj = j
                C[by * block_M + i, bx * block_N + vj] = C_shared[i // micro_size_x, vj // micro_size_y, i % micro_size_x, vj % micro_size_y]

    return main


matmul = tl_matmul(M, N, K, in_dtype, accum_dtype, accum_dtype)

print(matmul)


@tvm.register_func(func_name="tvm_callback_cuda_postproc", override=True)
def tvm_callback_cuda_postproc(code, _):
    return code


mod, params = TL.lower(matmul)
print(mod.imported_modules[0].get_source())
mod = TL.Profiler(mod, params, [], TL.TensorSupplyType.Integer)


latency = mod.do_bench(mod.func, warmup = 25)
print(f"Latency: {latency}")

ladder_permutate_config = bitblas.ops.LadderPermutateConfig(
    M=N,
    N=K,
    transform_kind=transform_b,
    transpose_matrix=True,
    dequantize_bits=num_bits,
    storage_dtype=storage_dtype,
)

ladder_permutate = bitblas.ops.LadderPermutate(
    ladder_permutate_config
)

lop3_permutate_config = bitblas.ops.LOP3PermutateConfig(
    M=N,
    N=K,
    datatype=in_dtype,
    dequantize_bits=num_bits,
    storage_dtype=storage_dtype,
)
lop3_permutate = bitblas.ops.LOP3Permutate(
    config=lop3_permutate_config,
    target=tvm.target.Target("llvm"),
)

A = torch.rand(M, K, device="cuda", dtype=getattr(torch, in_dtype))
qB = torch.randint(0, 127, (N, K // num_elems_per_byte), device="cuda", dtype=getattr(torch, storage_dtype))
C = torch.zeros(M, N, device="cuda", dtype=getattr(torch, accum_dtype))

B = (
    torch.zeros(qB.shape[0], qB.shape[1] * 8 // 4,
                dtype=torch.half).to(torch.half).to(A.device))
for i in range(B.shape[0]):
    for j in range(B.shape[1]):
        B[i][j] = ((qB[i][j // 2] >> (4 * (j % 2))) & 0xF).to(torch.half)

QLB = ladder_permutate(qB.cpu()).cuda()
QLB = lop3_permutate(QLB.cpu()).cuda()

if not VERIFY_CORRECTNESS:
    exit()

ref_c = torch.matmul(A, B.T).to(getattr(torch, accum_dtype))
print(ref_c)
mod(A, QLB, C)
print(C)
torch.testing.assert_close(C, ref_c, rtol=1e-2, atol=1e-2)
