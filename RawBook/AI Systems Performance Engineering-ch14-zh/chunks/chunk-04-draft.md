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

这里可以看到，通过把整个 K 循环放在单次核函数启动内部完成，我们避免了为每个输出分块（tile）都启动多个核函数。在现代 GPU 上，当 K 很大时，这种做法可以提升利用率——代价是在单个核函数中占用资源的时间更长。

你可以把这种分块方式与持久化核函数（persistent kernel）结合起来。这样就能在多个输出分块之间复用同一个线程块。持久化核函数会使用一维网格（1-D grid），并以 tl.num_programs(0) 为步长来推进分块索引，如下面这段代码所示：

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

这种持久化方式非常契合现代 GPU，因为它们拥有相对较大的寄存器文件、共享内存（shared memory）和 L2 缓存。这些资源能够容纳更大的分块尺寸（BLOCK_K）。这意味着每次迭代可以展开更多的 K 循环。

像这样的持久化核函数通常会胜过一连串较小的矩阵乘核函数——尤其是当 K 非常大（例如 > 1,024）时。其权衡在于单个 SM 会被占用更长时间。但如果核函数能够充分利用该 SM，这往往是理想的。

在前面的代码中，你可以尝试在现代 GPU 上增大 BLOCK_K，因为它们拥有更大的片上内存，每个分块能够处理更多数据。

不过，超过某个临界点之后，寄存器压力可能会上升，并导致寄存器溢出（register spilling）。因此，务必进行剖析，为你的工作负载和硬件找到恰当的平衡点。

### 使用 Triton 实现软件流水线与双缓冲

这个示例展示了如何用 Triton 实现双缓冲（double buffering）。双缓冲是软件流水线（software pipelining）的一种两阶段形式，它在单个循环中把内存加载与计算重叠起来。

在现代 NVIDIA GPU 上，异步的全局内存到共享内存拷贝允许多个处于飞行中的预取阶段同时进行，从而把内存传输与计算重叠。这使得双缓冲（以及三缓冲等）成为一种有价值且重要的性能优化技术。

Triton 通过 Triton 循环迭代器所使用的 num_stages 元参数来实现流水线。该参数会传递给 tl.range() 循环迭代器。当你传入 num_stages>1 时，迭代器会自动对循环进行流水线化，为多达 num_stages 次迭代发出处于飞行中的异步拷贝操作。

这会在这些分阶段的迭代之间把内存加载与计算重叠。该过程会持续进行，直到所有分块都处理完毕。下面是一个在 Triton 中实现基于分块的双缓冲（num_stages=2）的实现：

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

Triton 的编译器会识别 num_stages，利用 Triton 张量描述符（tensor descriptor）借助 TMA 和 TMEM 生成异步加载/存储，并自动管理同步。使用 TMA 张量描述符和 TMEM 可以降低寄存器压力，并简化对多维传输的表达。前面的循环执行的是「预取下一个 → 计算当前 → 交换」。这在 num_stages>1 时保持了重叠，并避免了因过早覆盖当前分块而导致重叠悄然失效。具体来说，当一个分块正被加载进共享内存时，上一个分块的计算正在进行。

你还可以使用 warp 专门化（warp specialization），Triton 会借此划分生产者 warp 与消费者 warp，从而把全局内存到共享内存的拷贝与计算重叠起来。当启发式规则选择使用它时，这一功能会自动启用。

保持流水线的重叠不被破坏很重要。当你设置 num_stages > 1 时，要确保当前正在计算的分块在整个循环迭代期间保持存活。在消费它的运算完成之前，你不应覆盖或丢弃对它的引用。

考虑一个两阶段的环形缓冲区（ring buffer）：你在计算当前缓冲区的同时，异步获取下一个缓冲区。随后你交换当前缓冲区与下一个缓冲区。在这种情形下，你需要先等待并消费「当前」缓冲区，然后才能释放它。与此同时，你需要用提交/等待（commit/wait）来守护「下一个」缓冲区（分块），直到发生交换。

如果当前分块无法跨迭代存活（例如你覆盖了它的缓冲区，或让其引用失效），编译器就可以自由地把「下一次」迭代的拷贝重排、下沉或提升到「当前」使用之前。这会让你的加载/计算重叠悄然消失。

而如果你自己管理共享内存块指针/TMA，务必使用显式握手：把异步拷贝入队 → tl.commit() →（稍后）tl.wait() → 消费该分块 → 推进指针。如果你在读取分块之前才调用 tl.wait()，并且只在消费之后才释放/交换，你就能保持真正的重叠。

> 在现代 GPU 上，Triton 用张量内存加速器（Tensor Memory Accelerator，TMA）来支撑描述符的加载与存储。当你对循环进行流水线化（num_stages > 1）时，要尽早预取下一个分块，并为当前分块保留一个单独的变量。在消费它的操作完成之前，不要覆盖或丢弃「当前」引用。否则，编译器可能合法地把下一次迭代的拷贝重排到当前使用之前，你的重叠就会崩溃。最简单的模式是「预取下一个 → 计算当前 → 交换」。

简而言之，软件流水线能改善内存带宽利用率并隐藏延迟。而如果内存带宽不是瓶颈，你可以把 num_stages 增大到 3（例如三缓冲），从而预取两个处于飞行中的分块，以进一步提升性能，代价是为额外的缓冲区占用更多共享内存。然而，在共享内存预算更紧张的硬件上——或者对于计算密集型核函数——额外的阶段可能带来收益递减、更低的占用率（occupancy）以及更差的整体性能。

### 使用 Triton Proton 剖析器进行剖析

要深入剖析 Triton 核函数的性能，请使用 Triton 的 Proton 剖析器。这是一个独立的剖析软件包，与 Triton 集成，并发出可在 Nsight Systems 时间线中看到的 NVTX 区间。这些 NVTX 区间会对代码区域进行插桩、收集计时并跟踪关键指标。随后这些 NVTX 标记会出现在 Nsight Systems 的时间线中。利用这些 NVTX 区间，你可以从 Proton 的汇总结果跳转到 Nsight Compute，进行更聚焦的核函数级分析。

在实践中，你使用 with proton.scope("name", metadata) 这个 Python 上下文管理器来包裹 Triton 代码的关键区段。在运行工作负载之前，你用 proton.start("matmul", hook="triton") 激活剖析器。

Proton 中的 metadata 是任意由用户提供的注解或指标字典，例如总 FLOPS、内存字节数、线程块索引、warp 级索引，或记录的槽位数量。这些会与计时数据一起被记录下来，以支持更丰富的性能分析。

执行结束后，你完成收尾并取回剖析数据。输出可以打印为一张分层的计时表格。下面是一段 Proton 剖析结果的摘录，它比较了用于相乘 8,192 × 8,192 矩阵的不同 matmul 核函数：

```
168.314 ms    16331.291  ROOT
├─ 174.920 ms   3928.623  cublas [M=8192,N=8192,K=512]
├─ 165.349 ms   4156.033  matmul_kernel [M=8192,N=8192,K=512]
├─ 159.352 ms   4312.421  matmul_kernel_persistent [M=8192,N=8192,K=512]
└─ 174.671 ms   3934.214  torch [M=8192,N=8192,K=512]
```

> 在前面的剖析输出中，标签 K 表示归约维度，即矩阵乘法共享的内部尺寸（例如一个 M × K 矩阵乘以一个 K × N 矩阵）。它代表点积的长度，而点积同时驱动着计算成本与内存流量。

在这份剖析报告中，我们看到 Triton 持久化核函数胜过了 cuBLAS，159.352 ms 对 174.920 ms。Proton 使得量化这类差异变得容易。它还会计算派生指标，例如有效 TFLOPS 和内存带宽。如果你提供了额外的 metadata（例如总 FLOPS 计数），Proton 会告诉你是否正接近你所用精度（例如 BF16/FP16、FP8、FP4 等）的理论硬件 TFLOPS 上限。对于许多矩阵形状和精度，PyTorch 中的 cuBLASLt 或 CUTLASS 路径都会追平甚至超过自定义 Triton 核函数。

NVIDIA 的 Nsight Systems 和 Nsight Compute 工具同样支持 Triton 核函数。例如，Nsight Systems 会显示核函数启动的名称，以及由 PyTorch 或 Proton 发出的任何 NVTX 区间。这些可用于把 Proton 作用域与 Nsight 时间线关联起来。这样，你就可以用 Proton 的输出在 Nsight Systems 中定位有意思的核函数，然后深入到 Nsight Compute 中进行底层分析。你会用 Nsight Compute 来分析寄存器使用、实际占用率等。将这些工具配合使用，能提供更完整的系统性能分析。

## PyTorch XLA 后端

虽然 TorchInductor 是 PyTorch 面向 GPU 和 CPU 的默认后端，但 PyTorch XLA 是一个独立的后端编译选项，面向 Google Cloud TPU 及其他加速器。PyTorch XLA 通过把 PyTorch 运算映射为 XLA 的图 IR，并使用目标硬件的运行时来执行它们，从而让 PyTorch 模型能够在这些加速器上运行，如图 14-6 所示。

![图 14-6. OpenXLA，即 PyTorch XLA 编译器后端的基础](AI%20Systems%20Performance%20Engineering-ch14_images/figure-14-6.png)

要激活 XLA 后端，你可以使用 torch.compile(..., backend="openxla")，它会激活基于 OpenXLA 的 PyTorch XLA。这个后端字符串由 PyTorch XLA 项目支持，会启用基于 OpenXLA 的编译。与 TorchDynamo 和 TorchInductor 类似，XLA 会捕获计算图。不过，它会提前编译整个程序，因为 XLA 的设计目标是生成静态图。

XLA 针对静态形状或有界的动态形状进行了优化。因此，使用 XLA 时，对动态形状的支持会更受限一些。新形状会触发整个程序的重新编译，这对延迟敏感的推理来说代价高昂。不过，OpenXLA 会按形状签名缓存可执行文件，这可以改善性能。

> 你可能需要对输入进行填充，或使用固定大小的分桶。这是因为 XLA 会针对新形状重新编译，而不是以符号化方式处理它们。

XLA 编译器会为每个唯一的输入形状和签名缓存对应的已编译图。因此，与 TorchInductor 类似，经过几个预热步骤之后性能会有所提升。主要的区别在于，XLA 不会在运行过程中增量编译。图是提前静态构建的。如果遇到新的形状，就会触发一次全新的整图编译，这非常昂贵，并会影响像推理这类延迟敏感的工作负载。

简而言之，如果你运行在 TorchInductor 后端目前尚不支持的硬件设备上，只要该设备支持 XLA，你就有可能使用 XLA 后端。许多相同的原则依然适用，例如尽量减少图中断。你也可以在 XLA 上使用一些分布式策略，例如数据并行和模型并行。虽然 XLA 在 NVIDIA GPU 上并不常用，但对于非 NVIDIA 硬件（例如使用 OpenXLA 的 Google TPU，或其他支持 XLA IR 的加速器）而言，它是一个强大的后端。XLA 得益于本章讨论过的许多类似编译技术。

## 关键要点

本章我们覆盖了相当多的内容，并且超级深入地剖析了 PyTorch 编译器栈以及 OpenAI 的 Triton 语言与编译器。以下是一些关键要点：

_善用_ torch.compile _实现轻松加速_ 根据你的需求选择编译模式（例如用 "default" 换取更快的启动，用 "max-autotune" 换取最高性能）。始终执行几次预热迭代，以越过初次编译。对于运行时间很短的作业或小模型，编译开销可能超过所获得的加速，因此只有当你有足够多的工作量来摊薄这一次性成本时才使用它。

_尽早设置性能标志_ 下面这些标志用于启用快速的 FP32 矩阵乘以及包括 Flash Attention 在内的 SDPA 变体。始终为你的模型验证精度，并保持启用这些标志以获得最佳性能：

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

_尽量减少图中断_ 使用 torch.\_dynamo.explain 或 TORCH_LOGS="graph_breaks" 检查图中断。移除或重构会引发中断的代码（例如打印、依赖数据的 Python 控制流、不受支持的算子），以最大化可被编译的连续区域。更少、更大的图通常意味着更好的性能。如有需要，可使用新的 torch.cond API 来处理条件逻辑，或把非关键的 Python 侧处理移出模型的 forward 函数。目标是给编译器呈现一条长而纯粹的、张量进、张量出的代码路径。

_谨慎使用动态形状_ 虽然并非必需，但预先设置 torch.compile(dynamic=True) 会强制编译器把所有维度都视为动态。这样，一个已编译的模型就能处理很宽范围的输入形状——从而减少对填充的需求。编译器本来会动态地做这件事，但设置 dynamic=True 会强制它提前完成。你也可以用 mark_dynamic() 只把特定维度标记为动态，而不是标记所有维度。这样，你就把这种灵活性局限在真正需要的地方。这对于可变序列长度非常有用，但务必衡量其中的权衡。启用动态形状支持可能会禁用 CUDA Graphs 并插入额外的保护条件。如果你的形状变化不太大，混合方法可能效果最好。具体来说，你可以按大小对输入进行分桶——从而限制不同形状的数量，然后对剩余的可变性启用动态形状。这种混合方法既避免了过度的重新编译，又减少了填充带来的浪费。

_剖析重新编译的保护条件_ 如果你看到多次编译，使用 TORCH_LOGS="graph_breaks,guards,recompiles" 来查明是哪个保护条件被触发。常见的元凶是 Python 随机值、变化的张量秩，或不同的设备/dtype。把这些方面变成静态，或在合适的情况下用 allow_in_graph 把它们标记为安全。

_尽量避免重新编译_ 在一个调优良好的训练循环中，初始的几次迭代之后你应当看到零次重新编译。如果确实看到持续的重新编译，请立即排查，因为这通常意味着某些东西在每次迭代时都在变化。这包括带有递增计数器的调试打印语句等。使用 set_stance() API 和保护条件日志来捕捉这些问题。同时确保你不会无意中混用设备——在某次迭代发送一个 CPU 张量，在下一次迭代又发送一个 GPU 张量。这会触发一次重新编译。

_调优并监控内存使用_ 编译模式可能会为更大的融合核函数和保护条件缓冲区使用更多内存。请监控 GPU 内存。如果遇到内存问题，可以考虑在 Triton 核函数中使用更小的 BLOCK_SIZE，或禁用某些融合。也要确保及时释放任何大的中间结果，因为在编译图的生命周期中它们可能停留得更久。如果你在编译期间看到内存不足错误，可能需要拆分模型，或使用更低的 max_autotune 设置。PyTorch 支持对大图进行自动检查点以减轻内存压力，但这并非万无一失。你也可以尝试分别编译子模块（例如每一层或每个块），而不是一次性编译整个模型，以此限制核函数生成期间的内存使用。

_明智地将 PyTorch 编译器与分布式训练结合_ 使用 DDP 或 FSDP 时，要注意在通信点处存在有意为之的图中断。它们是预期之内的，并会通过把通信与计算重叠来加以优化。把子模块包裹进 FSDP，以获得分片级的编译和内存节省。留意 dynamo.explain 中任何与 all-reduce 相关的警告。PyTorch 的设计力图把图中断带来的性能下降降到最低。当把 FSDP 与 torch.compile 一起使用时，你可能会在梯度预归约和后归约步骤处看到图中断。这些是预期之内并会被妥善处理的。请专注于每个被编译分片内部的主要前向和反向传播。

_使用_ TORCH*LOGS="perf_hints" *捕捉错失的优化\_ 举例来说，它会告诉你 CUDA Graphs 是否因为输入被改写而未被使用——或者某个运算是否回退到了即时执行模式。这些提示可以引导你发现潜在的改进，办法是告诉你避免某些模式或等待未来的支持。提示往往会直接建议解决办法。例如，「input is mutated」提示意味着你应该避免在编译之前对输入执行原地操作。

_先用小输入调试_ 在开发自定义核函数或测试编译模式时，使用较小的张量尺寸来快速捕捉正确性问题。一旦它能正常工作，再扩大规模。使用 PyTorch 的 torch.\_dynamo.config.verbose，甚至 TORCH_LOGS="output_code"，来检查小规模用例所生成的代码。请注意，在极小输入上测得的性能可能无法反映真实尺寸下的行为。但对于调试正确性以及查看生成了哪些核函数，这是一种有用的技术。

_只为真正的瓶颈编写自定义核函数_ 在着手开发 Triton 核函数之前，先分别在编译和不编译的情况下剖析你的模型，以识别热点。通常，TorchInductor 已经融合了很多东西。把精力集中在 Inductor 力有不逮的地方——也许是某个自定义算子或一种非典型的融合。不要过早地优化一切。用编译器处理大部分——其余的手工调优。举例来说，如果 TorchInductor 未能融合某个对性能至关重要的特定运算序列（比如一个多步的自定义激活），这或许就足以证明单独编写一个 Triton 核函数的合理性。只要权衡好维护成本即可，因为每个自定义核函数都是代码，当你更换硬件时——或者当 PyTorch 最终加入原生支持时——你可能都需要更新这段代码。通常，如果某个模式能让许多其他用户受益，那么为 TorchInductor 提交一个 issue 或功能请求来支持它是值得的。

_遵循 Triton 最佳实践_ 编写 Triton 代码时，确保内存访问是合并（coalesced）的，避免共享内存中的存储体冲突（bank conflict）（必要时进行填充），在边界处对 tl.load() 和 tl.store() 加掩码，选择与 warp 大小对齐的块和分块尺寸（32 线程的倍数），选择能装进 L1/共享内存划分区的分块尺寸，并用 @triton.autotune([... triton.Config(num_warps=..., num_stages=...) ...], key=[...]) 来调优 num_warps 和 num_stages 设置。如果性能至关重要，使用 Triton 的自动调优器来找到最佳配置。从 num_warps ∈ {4,8} 和 num_stages ∈ {2,3,4} 开始尝试。记得查阅 Triton 的文档和示例，因为许多常见模式（例如 FlashAttention）社区已经实现并优化过了。至少，这些现成的示例是你自己实现的良好起点。

_缓存提示需谨慎_ 某些形式的 PTX 加载（例如 DeepSeek 风格的非一致 + L1 不分配 + 256B L2 预取）是未有文档记载的，在某些硬件上甚至可能是未定义的。这可能会在不同驱动版本和硬件代际之间失效。优先使用 Triton 的 eviction_policy=（例如 evict_last）和 cache_modifier= 旋钮来提升可移植性。在迁移到不同的 CUDA 驱动和 GPU 代际时，记得进行剖析。避免引入相互冲突的缓存提示，因为这类问题极难调试。

_关注 PyTorch 新版本_ PyTorch 正在快速演进其编译器。每个版本都会带来更广的算子支持、更好的性能以及更少的图中断。升级往往能带来免费的加速或修复问题。Triton 同理。保持更新可以确保你从最新的成果中受益。对于快速演进的 GPU 硬件而言，这一点尤为重要。

## 结语

随着复杂度上升、PyTorch 生态持续扩张，你必须拥抱性能剖析和迭代式优化。你应该把 Nsight Systems、Nsight Compute、PyTorch 剖析器或 Triton 剖析器配合使用，以验证你是否获得了预期的 GPU 利用率和性能。如果没有，就调整你的性能策略。

同时，记得在迭代过程中保持全局视角。当一个瓶颈被消除后，下一个就会浮现。迭代式的性能调优可能会把你从 GPU 核函数占用率带到 CPU 开销，再带到调优输入流水线。编译器帮你解决了这道难题中很大的一部分，但你很可能还需要优化数据加载、I/O 和算法选择，才能达到最优性能。
