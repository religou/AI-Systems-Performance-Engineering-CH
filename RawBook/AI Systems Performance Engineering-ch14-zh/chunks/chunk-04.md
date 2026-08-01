```
def tiled_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    Tiled matrix multiplication using autotuned Triton kernel.

    The kernel will automatically select the best block size configuration
    for the given matrix dimensions (M, N, K).
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimensions must match: {K} != {K2}"
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    # Grid is computed based on max block size from autotuning configs
    # Triton's autotuner will pick the optimal block size at runtime
    MAX_BLOCK_M = 128  # From largest config
    MAX_BLOCK_N = 128
    grid = (triton.cdiv(M, MAX_BLOCK_M), triton.cdiv(N, MAX_BLOCK_N))
    # Launch with autotuning - Triton will select best config
    tiled_gemm_kernel[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C
```

Here, we see that by doing the entire K-loop inside one kernel launch, we avoid launching multiple kernels per output tile. On modern GPUs, this approach can increase utilization when K is large—at the cost of holding resources for longer in a single kernel.

You can combine this tiled approach with a persistent kernel. This would reuse the same thread block across multiple output tiles. The persistent kernel would use a 1-D grid and stride the tile index by tl.num_programs(0) as shown in this code block:

```
@triton.jit
def matmul_kernel_persistent(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # compile-time constants
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """
    Persistent thread GEMM with Triton tensor descriptors + autotuning.
    Key Blackwell Optimizations:
      1) Tensor descriptors -> TMA hardware acceleration
      2) Autotuning across multiple block-size configurations
      3) Persistent threads to amortize launch overhead
      4) TMEM accumulation (accumulators live in ~256 KB/SM TMEM on Blackwell)
    This is the PRODUCTION-READY version combining these best practices.
    """
    # 1-D persistent launch: each program processes multiple tiles
    pid   = tl.program_id(0)
    nprog = tl.num_programs(0)
    MT = tl.cdiv(M, BLOCK_M)
    NT = tl.cdiv(N, BLOCK_N)
    TILE_COUNT = MT * NT
    # --- Tensor descriptors (map to TMA on NVIDIA). Descriptor rules:
    #     leading strides must be multiples of 16 BYTES;
    #     last dimension contiguous.
    A_desc = tl.make_tensor_descriptor(
        A_ptr, shape=[M, K], strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    B_desc = tl.make_tensor_descriptor(
        B_ptr, shape=[K, N], strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    # Persistent stride over all output tiles handled by this program.
    tile_idx = pid
    while tile_idx < TILE_COUNT:
        pid_m = tile_idx // NT
        pid_n = tile_idx %  NT
        m0 = pid_m * BLOCK_M
        n0 = pid_n * BLOCK_N
        offs_m = m0 + tl.arange(0, BLOCK_M)
        offs_n = n0 + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        # FP32 accumulator (on Blackwell, accumulators live in TMEM,
        # not registers)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # ---- K loop (double-buffered ring-of-two using descriptors -> TMA) ----
        K_tiles = (K + BLOCK_K - 1) // BLOCK_K
        if K_tiles == 0:
            # Masked store of zeros when K == 0
            c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :]
                              * stride_cn)
            c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
            tl.store(c_ptrs, acc, mask=c_mask)
            tile_idx += nprog
            continue
        # Prefetch first CURRENT tiles
        k0 = 0
        if (m0 + BLOCK_M <= M) and (k0 + BLOCK_K <= K):
            a_cur = A_desc.load([m0, k0])
        else:
            row_offsets = offs_m[:, None] + tl.zeros((BLOCK_M, BLOCK_K),
                                                      dtype=offs_m.dtype)
            col_offsets = (k0 + offs_k)[None, :] + tl.zeros((BLOCK_M, BLOCK_K),
                                                             dtype=offs_k.dtype)
            a_cur = tl.load(A_desc, offsets=(row_offsets, col_offsets),
                            boundary_check=(0, 1), padding_option="zero")
        if (n0 + BLOCK_N <= N) and (k0 + BLOCK_K <= K):
            b_cur = B_desc.load([k0, n0])
        else:
            row_offsets = (k0 + offs_k)[:, None] + tl.zeros((BLOCK_K, BLOCK_N),
                                                             dtype=offs_k.dtype)
            col_offsets = offs_n[None, :] + tl.zeros((BLOCK_K, BLOCK_N),
                                                      dtype=offs_n.dtype)
            b_cur = tl.load(B_desc, offsets=(row_offsets, col_offsets),
                            boundary_check=(0, 1), padding_option="zero")

        # Pipeline: prefetch-NEXT -> compute-CURRENT -> swap
        for kt in tl.range(0, K_tiles, num_stages=NUM_STAGES,
                           warp_specialize=True):
            next_k = (kt + 1) * BLOCK_K
            if kt + 1 < K_tiles:
                if (m0 + BLOCK_M <= M) and (next_k + BLOCK_K <= K):
                    a_next = A_desc.load([m0, next_k])
                else:
                    row_offsets = offs_m[:, None] + tl.zeros((BLOCK_M, BLOCK_K),
                                                              dtype=offs_m.dtype)
                    col_offsets = (next_k + offs_k)[None, :]
                                   + tl.zeros((BLOCK_M, BLOCK_K),
                                   dtype=offs_k.dtype)
                    a_next = tl.load(A_desc, offsets=(row_offsets, col_offsets),
                                     boundary_check=(0, 1),
                                     padding_option="zero")
                if (n0 + BLOCK_N <= N) and (next_k + BLOCK_K <= K):
                    b_next = B_desc.load([next_k, n0])
                else:
                    row_offsets = (next_k + offs_k)[:, None]
                                   + tl.zeros((BLOCK_K, BLOCK_N),
                                   dtype=offs_k.dtype)
                    col_offsets = offs_n[None, :] + tl.zeros((BLOCK_K, BLOCK_N),
                                                              dtype=offs_n.dtype)
                    b_next = tl.load(B_desc, offsets=(row_offsets, col_offsets),
                                     boundary_check=(0, 1),
                                     padding_option="zero")
            # Compute on CURRENT tiles (UMMA on Blackwell; accumulators in TMEM)
            acc += tl.dot(a_cur, b_cur)
            # Swap in the prefetched NEXT tiles
            if kt + 1 < K_tiles:
                a_cur = a_next
                b_cur = b_next
        # ---- Store C with masking --------------------------------------------
        c_ptrs = C_ptr + (offs_m[:, None] * stride_cm
                          + offs_n[None, :] * stride_cn)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)
        # Advance to the next tile handled by this program (persistent stepping)
        tile_idx += nprog
```

This persistent approach aligns well with modern GPUs, which have relatively large register files, shared memory, and L2 cache. These can accommodate larger tile sizes (BLOCK_K). This means more of the K-loop can be unrolled per iteration.

A persistent kernel like this will usually outperform a sequence of smaller matmul kernels—especially when K is very large (e.g., > 1,024). The trade-off is that one SM is occupied longer. But if the kernel can fully utilize the SM, this is often ideal.

In the preceding code, you can experiment with increasing BLOCK_K on modern GPUs since they have increased on-chip memory and can handle more data per tile.

However, beyond a certain point, register pressure may increase and lead to register spilling. As such, it’s always important to profile and find the right balance for your workload and hardware.

### Software Pipelining and Double Buffering with Triton

This example shows how to implement double buffering with Triton. Double buffering, a two-stage form of software pipelining, overlaps memory loads and computations in a single loop.

On modern NVIDIA GPUs, asynchronous global-to-shared copies allow multiple in-flight stages of prefetch to overlap memory transfers with compute. This makes double buffering (and triple buffering, etc.) a valuable and important performance optimization technique.

Triton implements pipelining through a num_stages meta-parameter used by Triton loop iterators. This parameter is passed to the tl.range() loop iterators. When you pass num_stages>1, the iterators automatically pipeline the loop by issuing asynchronous copy operations for up to num_stages iterations in flight.

This will overlap memory loads with computations across those staged iterations. This process continues until all tiles are processed. Here is an implementation of tile-based double buffering (num_stages=2) in Triton:

```
@triton.jit
def pipelined_matmul(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # compile-time constants:
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # Program (CTA) ids for the MxN tiling
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # ---------------- Descriptor creation (maps to TMA on Blackwell) ----------
    # Requirements for descriptor/TMA on NVIDIA GPUs:
    #  - leading strides are multiples of 16 BYTES
    #  - last dimension contiguous
    #  - block_shape matches the tile you intend to move
    A_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    B_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    # Accumulator in FP32; on Blackwell this resides in TMEM, not registers.
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    # Number of K tiles
    K_tiles = (K + BLOCK_K - 1) // BLOCK_K
    if K_tiles == 0:
        # Nothing to do, store zeros (masked)
        c_ptrs = C_ptr + (offs_m[:, None]
                          * stride_cm + offs_n[None, :] * stride_cn)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)
        return
    # --------- Prefetch the first "current" tiles (fast path if fully in-bounds)
    k0 = 0
    if (m0 + BLOCK_M <= M) and (k0 + BLOCK_K <= K):
        a_cur = A_desc.load([m0, k0])
    else:
        # Boundary: compute row/col offsets and use boundary-checked load
        row_offsets = offs_m[:, None] + tl.zeros((BLOCK_M, BLOCK_K),
                                                  dtype=offs_m.dtype)
        col_offsets = (k0 + offs_k)[None, :] + tl.zeros((BLOCK_M, BLOCK_K),
                                                         dtype=offs_k.dtype)
        a_cur = tl.load(A_desc, offsets=(row_offsets, col_offsets),
                        boundary_check=(0, 1), padding_option="zero")
    if (n0 + BLOCK_N <= N) and (k0 + BLOCK_K <= K):
        b_cur = B_desc.load([k0, n0])
    else:
        row_offsets = (k0 + offs_k)[:, None] + tl.zeros((BLOCK_K, BLOCK_N),
                                                         dtype=offs_k.dtype)
        col_offsets = offs_n[None, :] + tl.zeros((BLOCK_K, BLOCK_N),
                                                  dtype=offs_n.dtype)
        b_cur = tl.load(B_desc, offsets=(row_offsets, col_offsets),
                        boundary_check=(0, 1), padding_option="zero")
    # ------------- K loop with software pipelining and TMA prefetch -----------
    # Put prefetch as early as possible in loop body; keep separate "current"
    # tile so loads for the next iteration can overlap with the dot-product of
    # current tile. Use warp_specialize to partition producer/consumer warps.
    for kt in tl.range(0, K_tiles, num_stages=NUM_STAGES, warp_specialize=True):
        next_k = (kt + 1) * BLOCK_K
        # Prefetch NEXT tiles early (TMA), if there is a next tile
        if kt + 1 < K_tiles:
            if (m0 + BLOCK_M <= M) and (next_k + BLOCK_K <= K):
                a_next = A_desc.load([m0, next_k])
            else:
                row_offsets = offs_m[:, None] + tl.zeros((BLOCK_M, BLOCK_K),
                                                          dtype=offs_m.dtype)
                col_offsets = (next_k + offs_k)[None, :]
                              + tl.zeros((BLOCK_M, BLOCK_K), dtype=offs_k.dtype)
                a_next = tl.load(A_desc, offsets=(row_offsets, col_offsets),
                                 boundary_check=(0, 1), padding_option="zero")
            if (n0 + BLOCK_N <= N) and (next_k + BLOCK_K <= K):
                b_next = B_desc.load([next_k, n0])
            else:
                row_offsets = (next_k + offs_k)[:, None]
                               + tl.zeros((BLOCK_K, BLOCK_N), dtype=offs_k.dtype)
                col_offsets = offs_n[None, :] + tl.zeros((BLOCK_K, BLOCK_N),
                                                          dtype=offs_n.dtype)
                b_next = tl.load(B_desc, offsets=(row_offsets, col_offsets),
                                 boundary_check=(0, 1), padding_option="zero")
        # Compute on the CURRENT tiles (UMMA; accumulators in TMEM on Blackwell)
        acc += tl.dot(a_cur, b_cur)
        # Swap in prefetched NEXT tiles
        if kt + 1 < K_tiles:
            a_cur = a_next
            b_cur = b_next
    # ------------------------------- Store C ----------------------------------
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)
```

Triton’s compiler sees num_stages, generates asynchronous loads/stores with TMA and TMEM using Triton tensor descriptors, and automatically manages the synchronization. The use of TMA tensor descriptors and TMEM reduces register pressure and simplifies expressing multidimensional transfers. The preceding loop performs prefetch-next → compute-current → swap. This preserves overlap with num_stages>1 and avoids silent overlap collapse when you overwrite the current tile too early. Specifically, while one tile is being loaded into shared memory, the previous tile’s computation is being performed.

You can also use warp specialization, in which Triton partitions producer and consumer warps to overlap global-to-shared copies with compute. This is automatically enabled when heuristics select it.

It’s important to keep the pipeline’s overlap intact. When you set num_stages > 1, make sure the tile you are currently computing stays live across loop iterations. You don’t want to overwrite or drop its reference until the math that consumes it is finished.

Consider a 2-stage ring buffer in which you compute the current buffer while asynchronously fetching the next buffer. You then swap the current and next buffers. In this scenario, you need to wait and consume the “current” buffer before releasing it. At the same time, you need to use commit/wait to guard the “next” buffer (tile) until the swap.

If the current tile doesn’t survive across iterations (e.g., you overwrite its buffer or let the reference die), the compiler is free to reorder, sink, or hoist the “next” iteration’s copies past the “current” uses. This will cause your load/compute overlap to quietly disappear.

And If you manage shared-memory block pointers/TMA yourself, make sure to use the explicit handshake: enqueue async copies → tl.commit() → (later) tl.wait() → consume the tile → advance pointers. If you call tl.wait() right before you read the tile—and only release/swap after consumption—you will preserve true overlap.

> On modern GPUs, Triton backs descriptor loads and stores with the Tensor Memory Accelerator (TMA). When you pipeline a loop (num_stages > 1), prefetch the next tile early and keep a separate variable for the current tile. Don’t overwrite or drop the “current” reference until the operation that consumes it is finished. Otherwise, the compiler may legally reorder the next iteration’s copies past current uses and your overlap will collapse. The simplest pattern is prefetch-next → compute-current → swap.

In short, software pipelining improves memory bandwidth utilization and hides latency. And if memory bandwidth isn’t the bottleneck, you can increase num_stages to 3 (e.g., triple buffering) to prefetch two tiles in flight and further improve performance at the cost of more shared memory usage for the additional buffers. However, on hardware with tighter shared-memory budgets—or for compute-heavy kernels—the extra stages might lead to a diminishing return, lower occupancy, and reduced overall performance.

### Profiling with Triton Proton Profiler

To deeply profile Triton kernel performance, use Triton’s Proton profiler. This is a separate profiling package that integrates with Triton and emits NVTX ranges visible in Nsight Systems timelines. These NVTX ranges instrument code regions, collect timings, and track key metrics. The NVTX markers then appear in Nsight Systems timelines. Use these NVTX regions to jump from Proton summaries into Nsight Compute for more focused kernel-level analysis.

In practice, you wrap critical sections of Triton code using the with proton.scope("name", metadata) Python context manager. Before running the workload, you activate the profiler with proton.start("matmul", hook="triton").

metadata in Proton is any user-provided dictionary of annotations or metrics, such as total FLOPS, memory byte counts, thread block indices, warp-level indices, or the number of recorded slots. These get recorded alongside timing data for richer performance analysis.

After execution, you finalize and fetch the profiling data. The output can be printed as a hierarchical table of timings. Here is an excerpt of a Proton profiling result that compares different matmul kernels for multiplying 8,192 × 8,192 matrices:

```
168.314 ms    16331.291  ROOT
├─ 174.920 ms   3928.623  cublas [M=8192,N=8192,K=512]
├─ 165.349 ms   4156.033  matmul_kernel [M=8192,N=8192,K=512]
├─ 159.352 ms   4312.421  matmul_kernel_persistent [M=8192,N=8192,K=512]
└─ 174.671 ms   3934.214  torch [M=8192,N=8192,K=512]
```

> In the preceding profiling output, the label K denotes the reduction dimension, or the shared inner size of the matrix multiply (e.g., an M × K matrix multiplied by a K × N matrix). This represents the length of the dot product that drives both compute cost and memory traffic.

In this profile report, we see the Triton persistent kernel outperforms cuBLAS, 159.352 ms versus 174.920 ms. Proton makes it easy to quantify such differences. It also computes derived metrics like effective TFLOPS and memory bandwidth. If you supply additional metadata such as the total FLOPS count, Proton will show you if you’re nearing the theoretical hardware TFLOPS limit for the precision you are using (e.g., BF16/FP16, FP8, FP4, etc.). For many matrix shapes and precisions, cuBLASLt or CUTLASS paths in PyTorch will match or exceed a custom Triton kernel.

NVIDIA’s Nsight Systems and Nsight Compute tools support Triton kernels as well. For instance, Nsight Systems shows kernel-launch names and any NVTX ranges emitted by PyTorch or Proton. These can be used to correlate Proton scopes with Nsight timelines. This way, you can use Proton’s output to pinpoint interesting kernels in Nsight Systems and then dive deeper into Nsight Compute for low-level analysis. You would use Nsight Compute to analyze register usage, achieved occupancy, etc. Using these tools together provides a more complete system performance analysis.

## PyTorch XLA Backend

While TorchInductor is the PyTorch default backend for GPUs and CPUs, PyTorch XLA is a separate backend-compilation option that targets Google Cloud TPUs and other accelerators. PyTorch XLA allows PyTorch models to run on these accelerators by mapping PyTorch operations into XLA’s graph IR and executing them using the target hardware’s runtime, as shown in Figure 14-6.

![Figure 14-6. OpenXLA, the basis of the PyTorch XLA compiler backend](AI%20Systems%20Performance%20Engineering-ch14_images/figure-14-6.png)

To activate the XLA backend, you can use torch.compile(..., backend="openxla"), which activates PyTorch XLA based on OpenXLA. This backend string is supported by the PyTorch XLA project and activates OpenXLA-based compilation. Similar to TorchDynamo and TorchInductor, XLA captures the graph of computations. However, it compiles whole programs ahead of time because XLA is designed to generate static graphs.

XLA is optimized for static shapes or bounded dynamic shapes. As such, when using XLA, dynamic shape support is a bit more limited. New shapes trigger whole-program recompilation, which is expensive for latency-sensitive inference. However, OpenXLA caches executables per shape signature, which can improve performance.

> You may need to pad or use fixed-size buckets for your inputs. This is because XLA will recompile for new shapes rather than handling them symbolically.

The XLA compiler will cache each compiled graph per unique input shape and signature. As such, performance will improve after a few warm-up steps similar to TorchInductor. The major difference is that XLA will not incrementally compile mid-run. The graph is built statically ahead of time. And if a new shape is encountered, it will trigger a new whole-graph compilation, which is very expensive and impacts latency-sensitive workloads like inference.

In short, if you’re running on a hardware device not currently supported by the TorchInductor backend, you can potentially use the XLA backend if the device supports XLA. Many of the same principles apply, such as minimizing graph breaks. You can also use some distributed strategies with XLA, such as data and model parallelism. While XLA isn’t commonly used with NVIDIA GPUs, it’s a powerful backend for non-NVIDIA hardware (e.g., Google TPUs using OpenXLA or other accelerators that support XLA IR). XLA benefits from many similar compilation techniques discussed in this chapter.

## Key Takeaways

We covered quite a lot in this chapter, and we dove super deep into the PyTorch compiler stack and OpenAI’s Triton language and compiler. The following are some key takeaways:

*Leverage* torch.compile *for easy speedups* Choose the compilation mode based on your needs (e.g., "default" for quicker startup, "max-autotune" for maximum performance). Always perform a few warm-up iterations to get past the initial compile. For short-running jobs or small models, the compile overhead might outweigh the speedup, so use it when you have enough work to amortize the one-time cost.

*Set performance flags early* Following are the flags you should enable for fast FP32 matmuls and SDPA variants including Flash Attention. Always validate accuracy for your model and keep these enabled for the best performance:

```
import torch
 # maps to TF32/BF16 fast paths
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True
# affects convs
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
```

*Minimize graph breaks* Inspect graph breaks using torch._dynamo.explain or TORCH_LOGS= "graph_breaks". Remove or refactor code that causes breaks (e.g., prints, data-dependent Python control flow, unsupported ops) to maximize the contiguous regions that can be compiled. Fewer, larger graphs generally mean better performance. If needed, use the new torch.cond API for conditional logic or move noncritical Python-side processing out of the model’s forward function. The goal is to present the compiler a long, purely tensor-in and tensor-out code path.

*Use dynamic shapes carefully* While not necessary, setting torch.compile(dynamic=True) upfront forces the compiler to consider all dimensions as dynamic. This way, one compiled model can handle a wide range of input shapes—reducing the need for padding. The compiler will do this dynamically, but setting dynamic=True forces the compiler to do this upfront. You can also mark only specific dimensions dynamic with mark_dynamic() instead of all dimensions. This way, you localize the flexibility to where it’s actually needed. This is great for variable sequence lengths, but make sure to measure the trade-offs. Enabling dynamic-shape support can disable CUDA Graphs and insert additional guards. If your shapes don’t vary too widely, a hybrid approach can work best. Specifically, you can bucket inputs by size—up to limit the number of distinct shapes and then enable dynamic shapes for the remaining variability. This hybrid approach avoids excessive recompilations while still reducing padding waste.

*Profile for recompilation guards* Use TORCH_LOGS="graph_breaks,guards,recompiles" to find which guard is triggering if you see multiple compilations. Common culprits are Python random values, changing tensor ranks, or varying device/dtype. Make those aspects static or mark them as safe with allow_in_graph if appropriate.

*Avoid recompilations if possible* In a well-tuned training loop, you should see zero recompilations after the initial few iterations. If you do see continued recompiling, investigate immediately, as it usually means something is changing on every iteration. This includes debug print statements with an incrementing counter, etc. Use the set_stance() API and guard-logging to catch these. Also make sure that you’re not unintentionally mixing devices by sending a CPU tensor on one iteration and then a GPU on the next. This will trigger a recompile.

*Tune and monitor memory usage* Compiled mode might use more memory for larger fused kernels and guard buffers. Monitor GPU memory. If you hit memory issues, consider using a smaller BLOCK_SIZE in Triton kernels or disabling certain fusions. Also ensure you free any large intermediate results promptly, as they might hang around longer in a compiled graph’s lifecycle. If you see out-of-memory errors during compilation, you might need to split your model or use lower max_autotune settings. PyTorch supports automatic checkpointing for large graphs to reduce memory pressure, but it’s not foolproof. You can also try compiling submodules (e.g., each layer or block) separately instead of compiling the entire model at once to limit memory usage during kernel generation.

*Combine the PyTorch compiler with distributed training wisely* When using DDP or FSDP, be aware of intentional graph breaks at communication points. They are expected and optimized by overlapping communication with computation. Wrap submodules in FSDP to get shard-wise compilation and memory savings. Keep an eye on any all-reduce-related warnings in dynamo.explain. PyTorch’s design tries to minimize performance degradation due to graph breaks. When using FSDP with torch.compile, you may see graph breaks for gradient prereduction and postreduction steps. These are expected and handled. Focus on the main forward and backward passes within each shard being compiled.

*Use* TORCH_LOGS="perf_hints" *to catch missed optimizations* This will tell you, for example, if CUDA Graphs weren’t used due to input mutation—or if an operation fell back to eager mode. These hints can guide you to potential improvements by telling you to avoid certain patterns or wait for future support. Often, a hint will directly suggest the workaround. For instance, the “input is mutated” hint implies that you should avoid in-place operations on the input before compiling.

*Debug with small inputs first* When developing custom kernels or testing compiled mode, use small tensor sizes to quickly catch correctness issues. Once it’s working, scale up. Use PyTorch’s torch._dynamo.config.verbose or even TORCH_LOGS="output_code" to inspect generated code for small cases. Be aware that performance measured on tiny inputs may not reflect behavior on real sizes. But, for debugging correctness and seeing what kernels are generated, this is a useful technique.

*Write custom kernels only for the true bottlenecks* Before embarking on Triton kernel development, profile your model with and without compiling to identify hotspots. Often, TorchInductor already fuses many things. Focus your efforts on areas where Inductor falls short—maybe a custom op or an atypical fusion. Do not prematurely optimize everything. Use the compiler for most—and hand-tune the rest. For example, if TorchInductor fails to fuse a particular sequence of operations that is performance-critical, such as a multistep custom activation, this might justify a separate Triton kernel. Just weigh the maintenance cost since each custom kernel is code, and you may need to update this code when you change hardware—or if PyTorch eventually adds native support. Often, filing an issue or feature request for TorchInductor to support a pattern is worthwhile if it benefits many other users.

*Follow Triton best practices* When writing Triton code, ensure memory accesses are coalesced, avoid bank conflicts in shared memory (pad if needed), mask tl.load() and tl.store() at boundaries, choose block and tile sizes that align with warp sizes (multiples of 32 threads), select tile sizes that fit into the L1/shared carve-out, and tune num_warps and num_stages settings with @triton.autotune([... triton.Config(num_warps=..., num_stages=...) ...], key=[...]).. Use Triton’s autotuner to find the best configuration if performance is critical. Start with num_warps ∈ {4,8} and num_stages ∈ {2,3,4}. Remember to check the Triton documentation and examples since many common patterns like FlashAttention have already been implemented and optimized by the community. At a minimum, these existing examples are a good starting point for your own implementation.

*Cache hint caution* Some forms of PTX loads (e.g., DeepSeek-style non-coherent + L1 no-allocate + 256B L2 prefetch) are undocumented and potentially undefined on certain hardware. This may break across driver versions and hardware generations. Prefer Triton’s eviction_policy= (e.g., evict_last) and cache_modifier= knobs to improve portability. Remember to profile when migrating to different CUDA drivers and GPU generations. Avoid introducing conflicting cache hints, as these are extremely difficult to debug.

*Keep an eye on new PyTorch releases* PyTorch is rapidly evolving its compiler. Each version brings expanded operator support, better performance, and fewer graph breaks. Upgrading can often give free speedups or fix issues. The same goes for Triton. Staying up-to-date ensures you benefit from the latest work. This is especially important for rapidly evolving GPU hardware.

## Conclusion

As complexity increases and the PyTorch ecosystem continues to expand, it’s important that you embrace performance profiling and iterative optimizations. You should use Nsight Systems, Nsight Compute, PyTorch profiler, or Triton profiler together to verify that you’re getting the GPU utilization and performance that you expect. If not, adjust your performance strategy.

And remember to maintain a holistic view during your iterative approach. When one bottleneck is removed, the next one will emerge. Iterative performance tuning can lead you from GPU kernel occupancy to CPU overhead to tuning the input pipeline. The compiler helps with a big piece of this puzzle, but you likely need to optimize data loading, I/O, and algorithmic choices to achieve optimal performance.