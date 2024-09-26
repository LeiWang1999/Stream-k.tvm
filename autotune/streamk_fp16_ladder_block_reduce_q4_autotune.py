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
    reduce_k = [2, 4, 8]
    num_stages = [2, 3, 4]


    # block_row_warps = [1]
    # block_col_warps = [4]
    # warp_rows = [1]
    # warp_cols = [1]
    # chunk = [32]
    # reduce_k = [4]
    # num_stages = [2]
    
    _configs = list(itertools.product(block_row_warps, block_col_warps, warp_rows, warp_cols, chunk, reduce_k, num_stages))

    configs = [
        {'block_row_warps': brw, 'block_col_warps': bcw, 'warp_rows': wr, 'warp_cols': wc, 'chunk': c, 'reduce_k' : rk, 'num_stages': ns}
        for brw, bcw, wr, wc, c, rk, ns in _configs
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
    streamk_programs = total_sm

    from tvm import tl
    import tvm.tl.language as T
    from tvm.tl.autotuner import autotune, jit

    shared_scope = "shared.dyn"
    @autotune(configs=get_configs(), keys=['block_row_warps', 'block_col_warps', 'warp_rows', 'warp_cols', 'chunk', 'reduce_k', 'num_stages'], warmup=5, rep=20)
    @jit(out_idx=[2], supply_type=tl.TensorSupplyType.Normal, ref_prog=ref_program, rtol=1, skip_check=True, profiler='tvm')
    def kernel(block_row_warps = None, block_col_warps = None, warp_rows=None, warp_cols = None, chunk=None, reduce_k=None, num_stages=None):
        warp_row_tiles = warp_rows * micro_size_x
        warp_col_tiles = warp_cols * micro_size_y
        chunk = chunk * micro_size_k // reduce_k
        BLOCK_SIZE_M = block_row_warps * warp_row_tiles
        BLOCK_SIZE_N = block_col_warps * warp_col_tiles
        BLOCK_SIZE_K = chunk * reduce_k
        transform_a = 0
        transform_b = 3

        assert BLOCK_SIZE_K >= 16

        is_smooth_a = False
        can_swizzle = BLOCK_SIZE_K * DataType(in_dtype).bits == 512
        apply_pad_a = not (is_smooth_a or can_swizzle)
        pad_factor = 8

        is_smooth_b = transform_b >= 2

        warp_size = 32
        threads = warp_size * (block_row_warps * block_col_warps)
        local_size = (micro_size_x * micro_size_y) // warp_size

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

        sm_patition_factor = max(blocking_tiles // total_sm, 1)

        A_shape = (M, K)
        B_shape = (N // micro_size_y, K // micro_size_k, micro_size_y, micro_size_k // num_elems_per_byte)
        C_shape = (M, N)
        A_shared_shape = (
            BLOCK_SIZE_M, 
            (BLOCK_SIZE_K + pad_factor) if apply_pad_a else BLOCK_SIZE_K,
        )
        B_shared_shape = (
            BLOCK_SIZE_N // micro_size_y,
            BLOCK_SIZE_K // micro_size_k,
            micro_size_y,
            micro_size_k // num_elems_per_byte,
        )
        C_shared_shape = (
            BLOCK_SIZE_M // micro_size_x,
            BLOCK_SIZE_N // micro_size_y,
            micro_size_x,
            micro_size_y,
        )

        ptx_macro_generator = TensorCorePTXMacroGeneratorWithLadderTransform(
            a_dtype=dtypeAB,
            b_dtype=dtypeAB,
            accum_dtype=accum_dtype,
            a_transposed=False,
            b_transposed=True,
            block_row_warps=block_row_warps,
            block_col_warps=block_col_warps,
            warp_row_tiles=warp_row_tiles,
            warp_col_tiles=warp_col_tiles,
            chunk=chunk,
            reduce_k=reduce_k,
            transform_kind_b=transform_b,
            num_elems_per_byte=num_elems_per_byte,
        )

        vec_load_qb = 16
        if BLOCK_SIZE_N * (BLOCK_SIZE_K // reduce_k) // num_elems_per_byte // threads < vec_load_qb:
            vec_load_qb = BLOCK_SIZE_N * (BLOCK_SIZE_K // reduce_k) // num_elems_per_byte // threads

        @T.macro
        def compute_first_wave(
            pid: T.int32,
            A_buf: T.Buffer,
            A_buf_shared: T.Buffer,
            A_buf_local: T.Buffer,
            B_buf: T.Buffer,
            B_buf_shared: T.Buffer,
            B_buf_local: T.Buffer,
            B_buf_dequantize_local: T.Buffer,
            C_buf: T.Buffer,
            C_buf_shared: T.Buffer,
            C_buf_local: T.Buffer,
            reduced_buf: T.Buffer,
            start_iter,
            end_iter,
            thread_bindings,
            rk,
        ):

            start_iter[0] = pid * streamk_full_tiles + T.min(
                pid, streamk_partial_tiles
            )
            last_iter = (pid + 1) * streamk_full_tiles + T.min(
                pid + 1, streamk_partial_tiles
            )

            while start_iter[0] < last_iter:
                end_iter[0] = T.min(
                    start_iter[0]
                    + (iters_per_tile - (start_iter[0] % iters_per_tile)),
                    last_iter,
                )

                tile_id = start_iter[0] // iters_per_tile
                remain_iters = start_iter[0] % iters_per_tile
                pid_m = tile_id // T.ceildiv(N, BLOCK_SIZE_N)
                pid_n = tile_id % T.ceildiv(N, BLOCK_SIZE_N)

                T.clear(C_buf_local)

                for ko in T.Pipelined(
                    end_iter[0] - start_iter[0],
                    num_stages=num_stages,
                ):
                    # Load A into shared memory
                    for i, k in T.Parallel(BLOCK_SIZE_M, (BLOCK_SIZE_K // reduce_k)):
                        vk = rk * (BLOCK_SIZE_K // reduce_k) + k
                        A_buf_shared[i, vk] = A_buf[
                            pid_m * BLOCK_SIZE_M + i,
                            (ko + (start_iter[0] % iters_per_tile)) * BLOCK_SIZE_K
                            + vk,
                        ]

                    # Load B into shared memory
                    for i in T.serial(BLOCK_SIZE_N * (BLOCK_SIZE_K // reduce_k) // num_elems_per_byte // (threads * vec_load_qb)):
                        for v in T.vectorized(0, vec_load_qb):
                            t = thread_bindings
                            idx = i * threads * vec_load_qb * reduce_k + rk * threads * vec_load_qb + t * vec_load_qb + v
                            vkk = idx % (micro_size_k // num_elems_per_byte)
                            vjj = (idx // (micro_size_k // num_elems_per_byte)) % micro_size_y
                            vk = (idx // (micro_size_k // num_elems_per_byte) // micro_size_y) % (BLOCK_SIZE_K // micro_size_k)
                            vj = (idx // (micro_size_k // num_elems_per_byte) // micro_size_y // (BLOCK_SIZE_K // micro_size_k)) % (BLOCK_SIZE_N // micro_size_y)
                            B_buf_shared[vj, vk, vjj, vkk] = B_buf[pid_n * (BLOCK_SIZE_N // micro_size_y) + vj, (ko + (start_iter[0] % iters_per_tile)) * (BLOCK_SIZE_K // micro_size_k) + vk, vjj, vkk]

                    for ki in T.serial(
                        0, (BLOCK_SIZE_K // (micro_size_k * reduce_k))
                    ):

                        # Load A into fragment
                        ptx_macro_generator.LDMATRIX_A(
                            ptx_macro_generator,
                            A_buf_local,
                            A_buf_shared,
                            ki,
                            thread_bindings=thread_bindings,
                            rk=rk,
                        )

                        # Load B into fragment
                        ptx_macro_generator.LDMATRIX_B(
                            ptx_macro_generator,
                            B_buf_local,
                            B_buf_shared,
                            ki,
                            thread_bindings=thread_bindings,
                            rk=rk,
                        )

                        for j in T.serial(warp_cols):
                            local_size_b = ptx_macro_generator.local_size_b
                            T.call_extern('handle', 'decode_i4u_to_f16', T.address_of(B_buf_local[j * local_size_b // num_elems_per_byte]), 
                                            T.address_of(B_buf_dequantize_local[j * local_size_b]), 8)
                        # Compute
                        ptx_macro_generator.MMA(
                            ptx_macro_generator, A_buf_local, B_buf_dequantize_local, C_buf_local
                        )

                for n in T.serial(warp_rows * warp_cols * local_size):
                    T.attr(
                        T.comm_reducer(lambda x, y: x + y, [T.float16(0)]),
                        "reduce_scope",
                        T.reinterpret(T.uint64(0), dtype="handle"),
                    )
                    T.evaluate(
                        T.tvm_thread_allreduce(
                            T.uint32(1),
                            C_buf_local[n],
                            True,
                            reduced_buf[0],
                            rk,
                            dtype="handle",
                        )
                    )
                    if rk == 0:
                        C_buf_local[n] = reduced_buf[0]
                if rk == 0:
                    ptx_macro_generator.STMATRIX(
                        ptx_macro_generator,
                        C_buf_local,
                        C_buf_shared,
                        thread_bindings=thread_bindings,
                    )

                # last iteration of the tile always happens before its start on another SM
                if remain_iters == 0 and (end_iter[0] % iters_per_tile == 0):
                    for i, j in T.Parallel(BLOCK_SIZE_M, BLOCK_SIZE_N // reduce_k):
                        vj = rk * (BLOCK_SIZE_N // reduce_k) + j
                        C_buf[pid_m * BLOCK_SIZE_M + i, pid_n * BLOCK_SIZE_N + vj] = C_buf_shared[
                            i // micro_size_x, vj // micro_size_y, i % micro_size_x, vj % micro_size_y
                        ]
                else:
                    for i, j in T.Parallel(BLOCK_SIZE_M, (BLOCK_SIZE_N // reduce_k // 2)):
                        vj = rk * (BLOCK_SIZE_N // reduce_k) + j * 2
                        T.atomic_addx2(
                            C_buf[pid_m * BLOCK_SIZE_M + i, pid_n * BLOCK_SIZE_N + vj], 
                            C_buf_shared[
                                i // micro_size_x, vj // micro_size_y, i % micro_size_x, vj % micro_size_y
                            ]
                        )

                start_iter[0] = end_iter[0]

        @T.macro
        def compute_full_tiles(
            pid: T.int32,
            A_buf: T.Buffer,
            A_buf_shared: T.Buffer,
            A_buf_local: T.Buffer,
            B_buf: T.Buffer,
            B_buf_shared: T.Buffer,
            B_buf_local: T.Buffer,
            B_buf_dequantize_local: T.Buffer,
            C_buf: T.Buffer,
            C_buf_shared: T.Buffer,
            C_buf_local: T.Buffer,
            reduced_buf: T.Buffer,
            thread_bindings,
            rk,
        ):

            for p in T.serial(sm_patition_factor):
                tile_id = pid + streamk_tiles + p * total_sm
                pid_m = tile_id // T.ceildiv(N, BLOCK_SIZE_N)
                pid_n = tile_id % T.ceildiv(N, BLOCK_SIZE_N)

                T.clear(C_buf_local)

                for ko in T.Pipelined((K // BLOCK_SIZE_K), num_stages=num_stages):
                    # for ko in T.serial(
                    #     (K // BLOCK_SIZE_K),
                    # ):
                    # Load A into shared memory
                    for i, k in T.Parallel(BLOCK_SIZE_M, (BLOCK_SIZE_K // reduce_k)):
                        vk = rk * (BLOCK_SIZE_K // reduce_k) + k
                        A_buf_shared[i, vk] = A_buf[
                            pid_m * BLOCK_SIZE_M + i, ko * BLOCK_SIZE_K + vk
                        ]

                    # Load B into shared memory
                    for i in T.serial(BLOCK_SIZE_N * (BLOCK_SIZE_K // reduce_k) // num_elems_per_byte // (threads * vec_load_qb)):
                        for v in T.vectorized(0, vec_load_qb):
                            t = thread_bindings
                            idx = i * threads * vec_load_qb * reduce_k + rk * threads * vec_load_qb + t * vec_load_qb + v
                            vkk = idx % (micro_size_k // num_elems_per_byte)
                            vjj = (idx // (micro_size_k // num_elems_per_byte)) % micro_size_y
                            vk = (idx // (micro_size_k // num_elems_per_byte) // micro_size_y) % (BLOCK_SIZE_K // micro_size_k)
                            vj = (idx // (micro_size_k // num_elems_per_byte) // micro_size_y // (BLOCK_SIZE_K // micro_size_k)) % (BLOCK_SIZE_N // micro_size_y)
                            B_buf_shared[vj, vk, vjj, vkk] = B_buf[pid_n * (BLOCK_SIZE_N // micro_size_y) + vj, ko * (BLOCK_SIZE_K // micro_size_k) + vk, vjj, vkk]

                    for ki in T.serial(0, (BLOCK_SIZE_K // (micro_size_k * reduce_k))):

                        # Load A into fragment
                        ptx_macro_generator.LDMATRIX_A(
                            ptx_macro_generator,
                            A_buf_local,
                            A_buf_shared,
                            ki,
                            thread_bindings=thread_bindings,
                            rk=rk,
                        )

                        # Load B into fragment
                        ptx_macro_generator.LDMATRIX_B(
                            ptx_macro_generator,
                            B_buf_local,
                            B_buf_shared,
                            ki,
                            thread_bindings=thread_bindings,
                            rk=rk,
                        )
                        for j in T.serial(warp_cols):
                            local_size_b = ptx_macro_generator.local_size_b
                            T.call_extern('handle', 'decode_i4u_to_f16', T.address_of(B_buf_local[j * local_size_b // num_elems_per_byte]), 
                                            T.address_of(B_buf_dequantize_local[j * local_size_b]), 8)
                        ptx_macro_generator.MMA(
                            ptx_macro_generator,
                            A_buf_local,
                            B_buf_dequantize_local,
                            C_buf_local
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
                                C_buf_local[n],
                                True,
                                reduced_buf[0],
                                rk,
                                dtype="handle",
                            )
                        )
                        if rk == 0:
                            C_buf_local[n] = reduced_buf[0]

                if rk == 0:
                    ptx_macro_generator.STMATRIX(
                        ptx_macro_generator,
                        C_buf_local,
                        C_buf_shared,
                        thread_bindings=thread_bindings,
                    )

                for i, j in T.Parallel(BLOCK_SIZE_M, (BLOCK_SIZE_N // reduce_k)):
                    vj = rk * (BLOCK_SIZE_N // reduce_k) + j
                    C_buf[pid_m * BLOCK_SIZE_M + i, pid_n * BLOCK_SIZE_N + vj] = C_buf_shared[i // micro_size_x, vj // micro_size_y, i % micro_size_x, vj % micro_size_y]

        @T.prim_func
        def main(
            A: T.Buffer(A_shape, dtypeAB),
            B: T.Buffer(B_shape, storage_dtype),
            C: T.Buffer(C_shape, dtypeC),
        ):
            with T.Kernel(total_sm, threads=threads, prelude=decode_i4_to_f16) as pid:

                A_shared = T.alloc_shared(A_shared_shape, dtypeAB, scope=shared_scope)
                B_shared = T.alloc_shared(B_shared_shape, storage_dtype, scope=shared_scope)
                C_shared = T.alloc_shared(C_shared_shape, dtypeC, scope=shared_scope)
                A_shared_full_tiles = T.alloc_shared(A_shared_shape, dtypeAB, scope=shared_scope)
                B_shared_full_tiles = T.alloc_shared(B_shared_shape, storage_dtype, scope=shared_scope)
                C_shared_full_tiles = T.alloc_shared(C_shared_shape, dtypeC, scope=shared_scope)
                A_local = T.alloc_fragment((warp_rows * local_size), dtypeAB, scope="local")
                B_local = T.alloc_fragment((warp_cols * local_size // num_elems_per_byte), storage_dtype, scope="local")
                B_dequantize_local = T.alloc_fragment((warp_cols * local_size), dtypeAB, scope="local")
                C_local = T.alloc_fragment((warp_rows * warp_cols * local_size), accum_dtype, scope="local")
                reduced_accum_res = T.alloc_fragment(
                    0, accum_dtype, scope="local"
                )
                reduced_accum_res_full_tiles = T.alloc_fragment(
                    0, accum_dtype, scope="local"
                )
                start_iter = T.alloc_fragment((1,), "int32", "local")
                end_iter = T.alloc_fragment((1,), "int32", "local")

                T.annotate_layout(
                    {
                        A_shared: make_swizzle_layout(A_shared, is_smooth=is_smooth_a),
                        B_shared: make_swizzle_layout(B_shared, is_smooth=is_smooth_b),
                        A_shared_full_tiles: make_swizzle_layout(A_shared_full_tiles, is_smooth=is_smooth_a),
                        B_shared_full_tiles: make_swizzle_layout(B_shared_full_tiles, is_smooth=is_smooth_b),
                    }
                )

                T.use_swizzle(10)

                thread_bindings = T.thread_binding(0, threads, "threadIdx.x")
                rk = T.thread_binding(0, reduce_k, "threadIdx.y")

                compute_first_wave(
                    pid,
                    A,
                    A_shared,
                    A_local,
                    B,
                    B_shared,
                    B_local,
                    B_dequantize_local,
                    C,
                    C_shared,
                    C_local,
                    reduced_accum_res,
                    start_iter,
                    end_iter,
                    thread_bindings,
                    rk,
                )

                compute_full_tiles(
                    pid,
                    A,
                    A_shared_full_tiles,
                    A_local,
                    B,
                    B_shared_full_tiles,
                    B_local,
                    B_dequantize_local,
                    C,
                    C_shared_full_tiles,
                    C_local,
                    reduced_accum_res_full_tiles,
                    thread_bindings,
                    rk,
                )

        return main
    return kernel()


_tl_matmul_streamk = tl_matmul_streamk(
    m, n, k, in_dtype, in_dtype, accum_dtype
)
print(_tl_matmul_streamk)

