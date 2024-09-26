import math
import torch
import torch.backends
from bitblas import tvm as tvm
from tvm import DataType
from bitblas.tl.utils import get_swizzle_layout
from bitblas.tl.macro_generator import TensorCorePTXMacroGeneratorWithLadderTransform
from bitblas.gpu.intrin.lop3 import decode_i4_to_f16

# disable tf32
torch.backends.cuda.matmul.allow_tf32 = False

def cdiv(a, b):
    return math.ceil(a / b)

torch.manual_seed(0)

VERIFY_CORRECTNESS = True
in_dtype = "float16"
accum_dtype = "float16"
# accum_dtype = "float16"
# Support we're from a config file
num_bits = 4
num_elems_per_byte = 8 // num_bits
storage_dtype = "int8"

micro_size_x, micro_size_y, micro_size_k = 16, 16, 16
m, n, k = 16, 16384, 16384

total_sm = 108

A = torch.rand(m, k, device="cuda", dtype=getattr(torch, in_dtype))
qB = torch.randint(0, 127, (n, k // num_elems_per_byte), device="cuda", dtype=getattr(torch, storage_dtype))
C = torch.zeros(m, n, device="cuda", dtype=getattr(torch, accum_dtype))

B = (
    torch.zeros(qB.shape[0], qB.shape[1] * 8 // 4,
                dtype=torch.half).to(torch.half).to(A.device))

qB_expanded = ((qB.unsqueeze(-1) >> (4 * torch.arange(2, device=qB.device))) & 0xF).to(torch.half)

B = qB_expanded.view(B.shape)
import tvm.tl.language as T


def get_configs():
    import itertools

    block_row_warps = [1]
    block_col_warps = [1, 2, 4, 8]
    warp_rows = [1]
    warp_cols = [1, 2, 4, 8]
    chunk = [2, 4, 8, 16, 32]
    # reduce_k = [1, 2, 4, 8]
    reduce_k = [2]
    num_stages = [2, 3, 4]
    splitk_factor = [2, 4, 8, 16]


    # block_row_warps = [1]
    # block_col_warps = [4]
    # warp_rows = [1]
    # warp_cols = [1]
    # chunk = [32]
    # reduce_k = [4]
    # num_stages = [2]
    # splitk_factor = [2]
    
    _configs = list(itertools.product(block_row_warps, block_col_warps, warp_rows, warp_cols, chunk, reduce_k, num_stages, splitk_factor))

    configs = [
        {'block_row_warps': brw, 'block_col_warps': bcw, 'warp_rows': wr, 'warp_cols': wc, 'chunk': c, 'reduce_k' : rk, 'num_stages': ns, 'splitk_factor': skf}
        for brw, bcw, wr, wc, c, rk, ns, skf in _configs
    ]
    return configs


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


def ref_program(*args):
    return torch.matmul(A, B.T, out=C)

def tl_matmul_streamk(
    M,
    N,
    K,
    dtypeAB,
    dtypeC,
    accum_dtype,
):

    from tvm import tl
    import tvm.tl.language as T
    from tvm.tl.autotuner import autotune, jit

    shared_scope = "shared.dyn"
    @autotune(configs=get_configs(), keys=['block_row_warps', 'block_col_warps', 'warp_rows', 'warp_cols', 'chunk', 'reduce_k', 'num_stages', 'splitk_factor'], warmup=5, rep=20)
    @jit(out_idx=[2], supply_type=tl.TensorSupplyType.Normal, ref_prog=ref_program, rtol=1, skip_check=True, profiler='tvm')
    def kernel(block_row_warps = None, block_col_warps = None, warp_rows=None, warp_cols = None, chunk=None, reduce_k=None, num_stages=None, splitk_factor=None):
        warp_row_tiles = warp_rows * micro_size_x
        warp_col_tiles = warp_cols * micro_size_y
        chunk = chunk * micro_size_k // reduce_k
        block_M = block_row_warps * warp_row_tiles
        block_N = block_col_warps * warp_col_tiles
        block_K = chunk * reduce_k

        transform_b = 3

        assert block_K >= 16

        is_smooth_a = False
        can_swizzle = block_K * DataType(in_dtype).bits == 512
        apply_pad_a = not (is_smooth_a or can_swizzle)
        pad_factor = 8

        is_smooth_b = transform_b >= 2

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

        ptx_macro_generator = TensorCorePTXMacroGeneratorWithLadderTransform(
            a_dtype=dtypeAB, b_dtype=dtypeAB, accum_dtype=accum_dtype,
            a_transposed=False, b_transposed=True, block_row_warps=block_row_warps,
            block_col_warps=block_col_warps, warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles, chunk=chunk, reduce_k=reduce_k,
            transform_kind_b=transform_b, num_elems_per_byte=num_elems_per_byte
        )

        vec_load_qb = 16
        if block_N * (block_K // reduce_k) // num_elems_per_byte // threads < vec_load_qb:
            vec_load_qb = block_N * (block_K // reduce_k) // num_elems_per_byte // threads

        @T.prim_func
        def main(
            A: T.Buffer(A_shape, dtypeAB),
            B: T.Buffer(B_shape, storage_dtype),
            C: T.Buffer((M, N), dtypeC),
        ):
            with T.Kernel(
                T.ceildiv(N, block_N), T.ceildiv(M, block_M), splitk_factor, threads=threads, prelude=decode_i4_to_f16
            ) as (bx, by, bz):

                A_shared = T.alloc_shared(A_shared_shape, dtypeAB, scope=shared_scope)
                B_shared = T.alloc_shared(
                    B_shared_shape, storage_dtype, scope=shared_scope
                )
                C_shared = T.alloc_shared(C_shared_shape, dtypeC, scope=shared_scope)
                A_local = T.alloc_fragment((warp_rows * local_size), dtypeAB, scope="local")
                B_local = T.alloc_fragment((warp_cols * local_size // num_elems_per_byte), storage_dtype, scope="local")
                B_dequantize_local = T.alloc_fragment((warp_cols * local_size), dtypeAB, scope="local")
                C_local = T.alloc_fragment((warp_rows * warp_cols * local_size), accum_dtype, scope="local")
                reduced_accum_res = T.alloc_fragment(
                    0, accum_dtype, scope="local"
                )
                thread_bindings = T.thread_binding(0, threads, "threadIdx.x")
                rk = T.thread_binding(0, reduce_k, "threadIdx.y")

                T.annotate_layout(
                    {
                        A_shared: make_swizzle_layout(A_shared),
                        B_shared: make_swizzle_layout(B_shared, is_smooth=True),
                    }
                )

                T.use_swizzle(panel_size=10)

                T.clear(C_local)

                for sko in T.Pipelined(((K // splitk_factor) // block_K), num_stages=num_stages):
                    ko = bz * ((K // splitk_factor) // block_K) + sko

                    # Load A into shared memory
                    for i, k in T.Parallel(block_M, (block_K // reduce_k)):
                        vk = rk * (block_K // reduce_k) + k
                        A_shared[i, vk] = A[by * block_M + i, ko * block_K + vk]
                    
                    for i in T.serial(block_N * (block_K // reduce_k) // num_elems_per_byte // (threads * vec_load_qb)):
                        for v in T.vectorized(0, vec_load_qb):
                            t = thread_bindings
                            idx = i * threads * vec_load_qb * reduce_k + rk * threads * vec_load_qb + t * vec_load_qb + v
                            vkk = idx % (micro_size_k // num_elems_per_byte)
                            vjj = (idx // (micro_size_k // num_elems_per_byte)) % micro_size_y
                            vk = (idx // (micro_size_k // num_elems_per_byte) // micro_size_y) % (block_K // micro_size_k)
                            vj = (idx // (micro_size_k // num_elems_per_byte) // micro_size_y // (block_K // micro_size_k)) % (block_N // micro_size_y)
                            B_shared[vj, vk, vjj, vkk] = B[bx * (block_N // micro_size_y) + vj, ko * (block_K // micro_size_k) + vk, vjj, vkk]

                    for ki in T.serial(0, (block_K // (micro_size_k * reduce_k))):

                        # Load A into fragment
                        ptx_macro_generator.LDMATRIX_A(
                            ptx_macro_generator,
                            A_local,
                            A_shared,
                            ki,
                            thread_bindings=thread_bindings,
                            rk=rk,
                        )

                        # Load B into fragment
                        ptx_macro_generator.LDMATRIX_B(
                            ptx_macro_generator,
                            B_local,
                            B_shared,
                            ki,
                            thread_bindings=thread_bindings,
                            rk=rk,
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

                if reduce_k > 1:
                    for n in T.serial(warp_rows * warp_cols * local_size):
                        T.attr(
                            T.comm_reducer(lambda x, y: x + y, [T.float16(0)]),
                            "reduce_scope",
                            T.reinterpret(T.uint64(0), dtype="handle"),
                        )
                        T.evaluate(
                            T.tvm_thread_allreduce(
                                T.uint32(1),
                                C_local[n],
                                True,
                                reduced_accum_res[0],
                                rk,
                                dtype="handle",
                            )
                        )
                        if rk == 0:
                            C_local[n] = reduced_accum_res[0]

                if rk == 0:
                    ptx_macro_generator.STMATRIX(
                        ptx_macro_generator,
                        C_local,
                        C_shared,
                        thread_bindings=thread_bindings,
                    )
                    
                for i, j in T.Parallel(block_M, (block_N // reduce_k) // 2):
                    vj = rk * (block_N // reduce_k) + j * 2
                    T.atomic_addx2(C[by * block_M + i, bx * block_N + vj], C_shared[i // micro_size_x, vj // micro_size_y, i % micro_size_x, vj % micro_size_y])

        return main
    return kernel()


_tl_matmul_streamk = tl_matmul_streamk(
    m, n, k, in_dtype, in_dtype, accum_dtype
)
print(_tl_matmul_streamk)

