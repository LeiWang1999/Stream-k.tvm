import math
import torch
import torch.backends
import bitblas
from bitblas import tvm as tvm
from tvm import tl as TL
from bitblas.base.arch import CUDA
from tvm import DataType
from bitblas.tl.utils import get_swizzle_layout
from bitblas.tl.macro_generator import TensorCoreIntrinEmitter

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

micro_size_x, micro_size_y, micro_size_k = 16, 16, 16
m, n, k = 16, 16384, 16384

total_sm = 108

# uniform distribution from -1 to 1
A = torch.rand(m, k, device="cuda", dtype=torch.float16) * 2 - 1
B = torch.rand(n, k, device="cuda", dtype=torch.float16) * 2 - 1

import tvm.tl.language as T


def get_configs():
    import itertools
    block_row_warps = [1]
    block_col_warps = [1, 2, 4, 8]
    warp_rows = [1]
    warp_cols = [1, 2, 4, 8]
    chunk = [1, 2, 4, 8]
    reduce_k = [1, 2, 4, 8]
    num_stages = [1, 2, 3, 4]

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


def ref_program(A, B):

    return torch.matmul(A, B.T)

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
    @autotune(configs=get_configs(), keys=['block_row_warps', 'block_col_warps', 'warp_rows', 'warp_cols', 'chunk', 'reduce_k', 'num_stages'], warmup=10, rep=5)
    @jit(out_idx=[2], supply_type=tl.TensorSupplyType.Normal, ref_prog=ref_program, rtol=1, skip_check=True)
    def kernel(block_row_warps = None, block_col_warps = None, warp_rows=None, warp_cols = None, chunk=None, reduce_k=None, num_stages=None):
        warp_row_tiles = warp_rows * micro_size_x
        warp_col_tiles = warp_cols * micro_size_y
        chunk = chunk * micro_size_k
        BLOCK_SIZE_M = block_row_warps * warp_row_tiles
        BLOCK_SIZE_N = block_col_warps * warp_col_tiles
        BLOCK_SIZE_K = chunk * reduce_k

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

        is_smooth_a = False
        can_swizzle = BLOCK_SIZE_K * DataType(in_dtype).bits == 512
        apply_pad_a = not (is_smooth_a or can_swizzle)
        pad_factor = 8

        is_smooth_b = False
        apply_pad_b = not (can_swizzle)

        A_shape = (M, K)
        B_shape = (N, K)
        C_shape = (M, N)
        A_shared_shape = (BLOCK_SIZE_M, (BLOCK_SIZE_K + pad_factor) if apply_pad_a else BLOCK_SIZE_K)
        B_shared_shape = (BLOCK_SIZE_N, (BLOCK_SIZE_K + pad_factor) if apply_pad_b else BLOCK_SIZE_K)
        C_shared_shape = (
            BLOCK_SIZE_M // micro_size_x,
            BLOCK_SIZE_N // micro_size_y,
            micro_size_x,
            micro_size_y,
        )

        mma_emitter = TensorCoreIntrinEmitter(
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
        )


        @T.macro
        def compute_first_wave(
            pid: T.int32,
            A_buf: T.Buffer,
            A_buf_shared: T.Buffer,
            A_buf_local: T.Buffer,
            B_buf: T.Buffer,
            B_buf_shared: T.Buffer,
            B_buf_local: T.Buffer,
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
                    for j, k in T.Parallel(
                        BLOCK_SIZE_N,
                        (BLOCK_SIZE_K // reduce_k),
                    ):
                        vk = rk * (BLOCK_SIZE_K // reduce_k) + k
                        B_buf_shared[j, vk] = B_buf[
                            pid_n * BLOCK_SIZE_N + j,
                            (ko + (start_iter[0] % iters_per_tile))
                            * BLOCK_SIZE_K
                            + vk,
                        ]

                    for ki in T.serial(
                        0, (BLOCK_SIZE_K // (micro_size_k * reduce_k))
                    ):

                        # Load A into fragment
                        mma_emitter.ldmatrix_a(
                                A_buf_local,
                            A_buf_shared,
                            ki,
                            thread_bindings=thread_bindings,
                            rk=rk,
                        )

                        # Load B into fragment
                        mma_emitter.ldmatrix_b(
                                B_buf_local,
                            B_buf_shared,
                            ki,
                            thread_bindings=thread_bindings,
                            rk=rk,
                        )

                        # Compute
                        mma_emitter.mma(
                            mma_emitter, A_buf_local, B_buf_local, C_buf_local
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
                    mma_emitter.stmatrix(
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

                    # Load A into shared memory
                    for i, k in T.Parallel(BLOCK_SIZE_M, (BLOCK_SIZE_K // reduce_k)):
                        vk = rk * (BLOCK_SIZE_K // reduce_k) + k
                        A_buf_shared[i, vk] = A_buf[
                            pid_m * BLOCK_SIZE_M + i, ko * BLOCK_SIZE_K + vk
                        ]

                    # Load B into shared memory
                    for j, k in T.Parallel(
                        BLOCK_SIZE_N,
                        BLOCK_SIZE_K // reduce_k,
                    ):
                        vk = rk * (BLOCK_SIZE_K // reduce_k) + k
                        B_buf_shared[j, vk] = B_buf[
                            pid_n * BLOCK_SIZE_N + j,
                            ko * BLOCK_SIZE_K + vk,
                        ]

                    for ki in T.serial(0, (BLOCK_SIZE_K // (micro_size_k * reduce_k))):

                        # Load A into fragment
                        mma_emitter.ldmatrix_a(
                                A_buf_local,
                            A_buf_shared,
                            ki,
                            thread_bindings=thread_bindings,
                            rk=rk,
                        )

                        # Load B into fragment
                        mma_emitter.ldmatrix_b(
                                B_buf_local,
                            B_buf_shared,
                            ki,
                            thread_bindings=thread_bindings,
                            rk=rk,
                        )

                        mma_emitter.mma(
                                A_buf_local,
                            B_buf_local,
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
                    mma_emitter.stmatrix(
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
            B: T.Buffer(B_shape, dtypeAB),
            C: T.Buffer(C_shape, dtypeC),
        ):
            with T.Kernel(total_sm, threads=threads) as pid:

                A_shared = T.alloc_shared(A_shared_shape, dtypeAB, scope=shared_scope)
                B_shared = T.alloc_shared(B_shared_shape, dtypeAB, scope=shared_scope)
                A_shared_full_tiles = T.alloc_shared(A_shared_shape, dtypeAB, scope=shared_scope)
                B_shared_full_tiles = T.alloc_shared(B_shared_shape, dtypeAB, scope=shared_scope)
                C_shared = T.alloc_shared(C_shared_shape, dtypeC, scope=shared_scope)
                A_local = T.alloc_fragment((warp_rows * local_size), dtypeAB, scope="local")
                B_local = T.alloc_fragment((warp_cols * local_size), dtypeAB, scope="local")
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
                    }
                )

                # T.use_swizzle(10)

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
                    C,
                    C_shared,
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

# tl_mod, params = TL.lower(_tl_matmul_streamk)
# tl_mod = TL.Profiler(tl_mod, params, [], TL.TensorSupplyType.Integer)
# print(tl_mod.mod.imported_modules[0].get_source())
# b_c = torch.zeros((m, n), device="cuda", dtype=torch.float16)

# tl_mod(A, B, b_c)

# C = torch.matmul(A, B.T)

# print(b_c)
# print(C)

# tl_latency = tl_mod.do_bench(tl_mod.func, warmup=25)
# print(f"TL Latency: {tl_latency}")
# torch_latency = tl_mod.do_bench(lambda A, B, C: torch.matmul(A, B.T, out=C), warmup=25)
# print(f"PyTorch Latency: {torch_latency}")

# torch.testing.assert_close(C, b_c, rtol=1e-2, atol=1e-2, equal_nan=True)
