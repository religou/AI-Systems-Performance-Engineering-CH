# 第 6 章　GPU 架构、CUDA 编程与最大化占用率

在本章中，我们将首先回顾单指令多线程（single instruction, multiple-threads，SIMT）执行模型，以及 warp、线程块（thread block）和网格（grid）如何将你基于 GPU 的算法映射到流式多处理器（streaming multiprocessor，SM）上。

我们会回顾现代 NVIDIA GPU 上的 SIMT 执行模型，包括 warp、线程块和网格如何映射到 SM。随后深入 CUDA 编程模式，讨论片上内存层级（寄存器堆、共享内存/L1、L2、HBM3e），并演示 GPU 的异步数据传输能力，包括 Tensor Memory Accelerator（TMA）以及作为 Tensor Core 运算累加器的 Tensor Memory（TMEM）。

我们还会引入 Roofline 分析，用以识别计算受限（compute-bound）与访存受限（memory-bound）的核函数。这将为把现代 GPU 系统推向其理论峰值吞吐上限打下基础。

## 理解 GPU 架构

与面向低延迟单线程性能优化的 CPU 不同，GPU 是面向吞吐优化的处理器，天生就是为并行运行数千个线程而构建的。CPU 与 GPU 之间一个简单的 CUDA 编程流程如图 6-1 所示。

![图 6-1. 简单的 CUDA 编程流程](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-1.png)

最初，主机（host）将数据加载到 CPU 内存中。然后它把数据从 CPU 复制到 GPU 内存。在用 GPU 内存中的数据调用 GPU 核函数（kernel）之后，CPU 再把结果从 GPU 内存复制回 CPU 内存。此时结果又回到 CPU 上，供进一步处理。

GPU 依靠大规模并行来隐藏诸如图 6-1 中描述的 CPU-GPU 数据传输之类的数据传输延迟。每个 GPU 由许多 SM 组成，它们大致类似于 CPU 核心，但为并行做了精简。在 Blackwell 上，每个 SM 最多可以跟踪 64 个 warp（32 线程一组）。

每个 GPU 都包含许多 SM——类似于 CPU 核心，但为吞吐做了优化。在现代 GPU 上，每个 SM 最多可并发跟踪 64 个 warp（2,048 个线程）。Blackwell GPU 每个 SM 拥有 64K 个 32 位寄存器（共 256 KB），以及每个 SM 合计 256 KB 的 L1 缓存/共享内存。这块 SRAM 中最多可有 228 KB（227 KB 可用）被配置为每个 SM 由用户管理的共享内存。任何单个线程块最多可以请求 227 KB 的动态共享内存（228 KB 中有 1 KB 由 CUDA 保留）。这些都有助于 SM 支撑 GPU 高度的线程级并行。

在一个 Blackwell SM 内部，多个 warp 调度器（warp scheduler）向可用流水线发射指令；四个独立的 warp 调度器允许每个周期最多有四个 warp 向可用流水线发射指令。此外，每个调度器都支持双发射（dual-issue），能够为每个 warp 发射两条独立指令（例如一条算术运算和一条内存操作）。注意，双发射必须来自同一个 warp——而不能跨 warp。

在最好的情况下，每个调度器在每个周期都能有一个 warp 并发发射一条指令，从而允许每个周期有四个 warp 并行执行。当利用指令混合时，这会进一步提升吞吐，如图 6-2 所示。

![图 6-2. Blackwell SM 包含四个独立的 warp 调度器，每个调度器每周期能发射一条 warp 指令，并可为每个调度器双发射一条算术运算和一条内存操作](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-2.png)

在这里，每个 SM 被细分为四个独立的调度分区——每个分区都有自己的 warp 调度器和分派逻辑。你可以把 SM 想象成共享片上资源的四个“迷你 SM”。这让硬件能够挑选就绪的 warp，并在每个时钟周期从多达四个不同的 warp 发射指令。

在这四个“迷你 SM”分区中的每一个内部，调度器每周期都能从同一个 warp 发射两条指令：一条算术指令（例如 INT32、FP32 或 Tensor Core）和一条内存指令（一次加载或存储）。这正是该调度器被称为*双发射*的原因。表 6-1 汇总了这些数字。

_表 6-1. 关键的 SM 调度器与指令发射上限（每个时钟周期）_

| 指标               | 数值                                  |
| ------------------ | ------------------------------------- |
| 调度器数量         | 四个                                  |
| 最多发射的 warp 数 | 四个（每个调度器一个）                |
| 最多算术运算数     | 四个（每个调度器的算术发射一个）      |
| 最多内存操作数     | 四个（每个调度器的加载/存储发射一个） |

> 注：所有指标表格中的数值均为示意性质，用于解释概念。关于不同 GPU 架构上的实际基准测试结果，请参见 GitHub 代码仓库。

因此在最好的情况下，你可以在每个周期跨四个 warp 双发射四条算术指令和四条内存指令。这将同时最大化计算与内存吞吐。这些数字既源于 SM 的四路分区，也源于它能够为每个分区挑选一个 warp 并在每个周期发射两条正交指令的能力。

特殊功能单元（Special Function Unit，SFU）与 INT32、FP32 和 Tensor Core 流水线并列存在。它们处理超越函数运算（例如正弦、余弦、倒数、平方根）。不过，它们并不属于双发射的算术与内存指令对。SFU 使用专用的 SFU 流水线，独立于主 INT32/FP32 和加载/存储（load/store，LD/ST）流水线运行。

由于 SFU 占用独立的流水线，并可在需要时并行执行，SM 能够继续发射算术和内存指令，而无需等待较慢的函数完成。这种分离进一步提升了指令级并行，并为混合运算的核函数带来更高的整体吞吐。它们让复杂的数学运算不至于拖住核心的计算和内存流水线。

因为有四个调度器——且每个调度器通常每周期能发射一条 warp 指令——所以当存在足够的独立工作和发射配对时，每个周期最多可有四个 warp 向前推进。例如，内存操作可以流经 SM 合计 16 条加载/存储（LD/ST）流水线（每个调度器四条 LD/ST 流水线）。这些流水线会向 L1/共享内存、L2 缓存或全局内存（在接下来的小节中讨论）读取或写入数据。

> 确切的 LD/ST 流水线数量与配对并不保证如此。请依靠性能剖析计数器来判断你的核函数是受限于内存发射还是计算发射。并参阅 NVIDIA 文档以了解你所用架构的具体细节。Blackwell 调优指南是一个不错的起点。

简而言之，GPU 擅长数据并行工作负载，包括大型矩阵乘法、卷积以及其他同一条指令作用于许多元素的运算。开发者可以直接用 CUDA C++ 编写核函数，或间接通过 PyTorch 这样的高层框架，以及像 OpenAI Triton 这样基于 Python 的领域专用 GPU 语言来编写。

在深入核函数开发和内存访问优化之前，我们先回顾一下支撑所有这些实践的 CUDA 线程层级和关键术语。

### 线程、warp、线程块与网格

CUDA 将并行工作组织成三级层级——线程、线程块（又称*协作线程阵列*〔cooperative thread array，CTA〕）和网格——以在可编程性与大规模吞吐之间取得平衡。在最底层，每个线程执行你的核函数代码。在现代 GPU 上，你把线程分组为最多各 1,024 个线程的线程块。当你启动核函数时，线程块组成一个网格，如图 6-3 所示。

![图 6-3. 线程、线程块（又称 CTA）与网格](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-3.png)

通过合理设定网格大小，你可以在不改变核函数逻辑的情况下扩展到数百万个线程。CUDA 的运行时（以及 PyTorch 这样的框架）会处理跨所有 SM 的调度与分发。图 6-4 展示了线程层级的另一个视角，包括基于 CPU 的主机——它调用运行在 GPU 设备上的 CUDA 核函数。

![图 6-4. 线程层级视图，包括基于 CPU 的主机，它启动运行在 GPU 设备上的核函数](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-4.png)

传统上，来自不同线程块的线程无法彼此直接协作。然而，现代 GPU 架构和 CUDA 版本支持线程块簇（thread block cluster）。线程块簇是一组能够跨 SM 相互通信的线程块。

具体来说，在一个线程块簇内，不同线程块中的线程可以访问彼此的共享内存，并使用硬件支持的、簇作用域的屏障（barrier）。这些能力允许进行更大规模的计算运算，包括矩阵乘法——它在当今庞大的 LLM 工作负载中非常常见。线程块簇在参与该簇的各个 SM 之间共享一个分布式共享内存（distributed shared memory，DSMEM）地址空间，如图 6-5 所示。

![图 6-5. 硬件支持的 DSMEM，用于包含多个线程块的线程块簇](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-5.png)

DSMEM 是一项硬件特性，它通过快速的片上互连将一个线程块簇内所有 SM 的共享内存 bank 连接起来。借助 DSMEM，这些 SM 共享一个合并的多 SM 分布式共享内存池。这种统一让不同块中的线程能够以片上速度读取、写入并原子更新彼此的共享缓冲区——而且无需占用全局内存带宽。

> 我们将在第 10 章讨论线程块簇和 DSMEM 等高级主题。它们是现代 GPU 处理中极其重要的新增能力——对于 AI 系统性能工程师而言也非常重要，需要理解。就本章而言，我们的重点仍然放在块内共享内存优化上。

在每个线程块内部，线程使用低延迟的片上共享内存来共享数据，并用 \_\_syncthreads() 进行同步。由于每次屏障都会带来开销，你应尽量减少同步点，如图 6-6 所示。

![图 6-6. 在两段代码之间对一个线程块内的所有线程进行同步](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-6.png)

目标是尽量减少同步点。不过，GPU 硬件会通过在多个 warp 之间快速切换，尝试隐藏诸如全局内存加载、缓存填充和流水线停顿等长延迟事件。

线程块被细分为 32 个线程的 warp，它们在 SIMT 模型下由 warp 调度器管理并以锁步（lockstep）方式执行。如图 6-7 所示。

![图 6-7. warp（32 线程）作为一个整体推进，其指令由 warp 调度器管理](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-7.png)

在 SM 上保持更多 warp 处于运行中，被称为*高占用率*。当你的 CUDA 代码能够实现高占用率时，就意味着当一个 warp 停顿时，另一个 warp 已准备好运行。这让 GPU 的计算单元保持忙碌。

然而，高占用率必须与每线程资源上限（如寄存器和共享内存）相权衡。将寄存器溢出（register spilling）到较慢的内存中会造成新的停顿。在剖析占用率的同时也剖析寄存器和共享内存的使用情况，有助于你选择既能最大化吞吐、又不会触发资源争用的块大小。

> 我们将在第 8 章讨论占用率调优，但在 SM、warp、线程等语境下，它是一个需要理解的关键概念。

线程块相互独立执行，且没有保证的执行顺序。这让 GPU 调度器能够将它们分派到所有 SM 上，充分利用硬件并行。这种网格—块—warp 层级保证了你的 CUDA 核函数无需修改即可运行在拥有更多 SM 和线程的未来 GPU 架构上。

吞吐还取决于 warp 的执行效率。同一个 warp 中的线程必须遵循相同的控制流路径并执行合并访问（coalesced access）的内存访问。如果某些线程发生分歧，使得一个分支走 if 路径而其他线程走 else 路径，那么该 warp 会串行化执行，逐一顺序处理每条分支路径。这被称为*warp 分歧*（warp divergence），如图 6-8 所示。

![图 6-8. SIMT warp 分歧（左）与一致性（右）的对比](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-8.png)

通过屏蔽非活跃的 lane 并运行额外的遍历来覆盖每条分支，warp 分歧会把整体执行时间乘以分支的数量。我们将在第 8 章更深入地探讨 warp 分歧——以及检测、剖析和缓解它的方法。

> 分歧只对单个 warp 内部的线程构成问题。不同的 warp 可以走不同的分支，而不会有性能损失。

### 选择每块线程数与每网格块数

GPU 性能的一个关键方面是选择一个与硬件 32 线程 warp 大小对齐的线程块大小。因此，你通常会挑选恰好是 32 的整数倍的线程块大小。例如，一个 256 线程的块（8 个 warp = 256 ÷ 32）能完全占满每个 warp，而一个 33 线程的块则需要两个 warp 槽，且只用到第二个 warp 中 1/32 的 lane。这会浪费并行机会，因为无论一个 warp 是在活跃运行 32 个线程还是仅 1 个线程，它都会占用一个调度器槽。

此外，不同的 GPU 世代有不同的硬件上限，包括每个 SM 的最大线程数和每个 SM 的寄存器数量。如果我们想保持良好性能，这自然会限制块的大小。例如，块太大可能需要太多寄存器，从而导致*寄存器溢出*并降低核函数的性能。

大块也可能需要太多共享内存，而共享内存在 GPU 硬件中是有限的。具体来说，Blackwell 每个 SM 仅提供 228 KB（227 KB 可用）的共享内存，供运行在该 SM 上的所有驻留线程块寻址。

这些硬件上限会影响一个 SM 上同时能激活多少个块/warp。这正是我们前面介绍过的占用率的度量。如果较小的块能让更多并发 warp 在 SM 上同时运行，它们可能带来更高的占用率。

理解你所用 GPU 世代的相对规模和硬件线程上限很重要，包括线程、线程块、warp 和 SM 的数量。图 6-9 展示了这些资源的相对规模，包括它们的上限。

![图 6-9. Blackwell GPU 上线程的相对规模与硬件上限](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-9.png)

表 6-2 汇总了 Blackwell B200 GPU 的这些上限。其余上限可在 NVIDIA 网站上查到。（其他 GPU 世代会有不同的上限，因此务必核对你系统的确切规格。）

_表 6-2. 线程级与块级上限（Blackwell B200）_

| 资源                     | 硬件上限   | 说明                                                                            |
| ------------------------ | ---------- | ------------------------------------------------------------------------------- |
| warp 大小                | 32 线程    | 基本的 SIMT 执行单元是 32 个线程（一个 warp）。始终使用 32 的整数倍以避免浪费。 |
| 每个线程块的最大线程数   | 1,024 线程 | blockDim.x _ blockDim.y _ blockDim.z ≤ 1024。                                   |
| 每个线程块的最大 warp 数 | 32 warp    | （1,024 线程 ÷ 每个 warp 32 线程）= 每块最多 32 个 warp。                       |

我们已经讨论过 32 线程的 warp 大小上限，它促使我们选择 32 线程整数倍的块维度，以构成“完整 warp”并避免利用不足的 warp。注意，每个块最多可有 1,024 个线程，相应地，一个块只能包含 32 个 warp。这些上限会影响你的占用率，因为一旦一个块被调度，每个 SM 只能同时容纳有限数量的 warp 和块。

此外，针对不同的 GPU 世代，还存在每 SM 的上限，或常被称为 _SM 驻留上限_。这些 Blackwell 的 SM 驻留上限汇总在表 6-3 中。

_表 6-3. SM 驻留资源上限（Blackwell B200）_

| 资源（每 SM）          | 硬件上限   | 说明                                                                                                                                                                                                                                                                   |
| ---------------------- | ---------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 每 SM 最大驻留 warp 数 | 64 warp    | 硬件最多可保持 64 个 warp 在运行中（64 × 32 线程 = 2,048 线程）。注：这一上限已经历多个世代保持不变，在 Blackwell 上依然成立。                                                                                                                                         |
| 每 SM 最大驻留线程数   | 2,048 线程 | 等于 64 warp × 每 warp 32 线程。如果每个块使用 1,024 个线程，那么一个 SM 上最多可同时驻留 2 个这样的块（64 warp）。使用更小的块（例如 256 线程）可让更多块驻留在 SM 上（最多 8 块 × 256 = 2,048 线程），这能提升占用率并帮助隐藏延迟——不过过多的微小块会增加调度开销。 |
| 每 SM 最大活跃块数     | 32 块      | 一个 SM 上最多可同时驻留 32 个线程块（如果块更小，则在此上限内可容纳更多）。                                                                                                                                                                                           |

在这里，我们看到 Blackwell 上每 SM 的最大并发 warp 数为 64。这在近几代 GPU 中没有变化，因此占用率方面的考量得以延续。一个 SM 上的最大活跃块数为 32，相应地，每 SM 的最大驻留线程数为 2,048 线程。CUDA 网格也有最大维度，如表 6-4 所示。

_表 6-4. CUDA 网格上限_

| 网格维度                 | 上限                                            | 说明                                                             |
| ------------------------ | ----------------------------------------------- | ---------------------------------------------------------------- |
| X、Y 或 Z 方向的最大块数 | X：2,147,483,647 块；Y：65,535 块；Z：65,535 块 | 一个 3D 网格最大可达 2,147,483,647 × 65,535 × 65,535 块。        |
| 最大并发网格（核函数）数 | 128 网格                                        | 一个设备上最多可并发执行 128 个核函数（即同时驻留 128 个网格）。 |

虽然了解理论上的网格上限是好事，但你通常会先受制于前面展示的线程/块/每 SM 上限。如果你确实需要在某一维度上超过 65,535 个块，可以启动一个 2D 或 3D 网格，将工作拆分到多次核函数启动（多次启动，multilaunch）中。我们会在后面的小节中给出这样的示例。实际上，很少会在触及其他资源上限之前先触及网格大小上限。

### CUDA GPU 向后与向前兼容模型

CUDA 的核心优势之一是它的向前与向后兼容模型。只要你在二进制文件中包含 PTX 以实现向前兼容，今天编译的核函数通常无需修改即可运行在未来的 GPU 世代上。如果你只针对单个架构发布 SASS（例如面向 Hopper 的 sm_90 或面向 Blackwell 的 sm_100）而不带 PTX，那么该二进制文件将无法在更新的架构上向前运行。像 sm_100f 或 compute_100f 这样的家族特定目标会将可移植性限制在同一特性家族的设备内。最佳做法是发布一个既包含通用 cubin/PTX、又包含所需家族特定 cubin（例如各种优化等）的 fatbin。

你可以通过在加载时强制进行 PTX JIT 编译来验证兼容性——设置 CUDA_FORCE_PTX_JIT=1 来即时编译 PTX 并缓存结果。如果你的二进制文件缺少 PTX，核函数启动将会失败。这会迫使你带上 PTX 支持重新构建。这种兼容模型是庞大 CUDA 生态系统的根基。它让你能从单一代码库同时面向旧硬件和最前沿硬件。

> 若要在当前与未来 GPU 世代之间真正保持向后与向前兼容，你应当使用通用目标进行编译——或显式包含 PTX。当你需要来自较新硬件特性的特定优化时，可以使用世代特定的目标。这样做时，务必为其他架构提供回退路径。

## CUDA 编程复习

在 CUDA C++ 中，你通过编写核函数来定义并行工作。这些是用 **global** 注解的特殊函数，运行在 GPU 设备上。当你从 CPU（主机）代码调用一个核函数时，你使用 <<< >>>（“chevron”/三尖括号）语法来指定应运行多少个线程——以及它们如何组织——需要两个配置参数：blocksPerGrid 表示线程块的数量，threadsPerBlock 表示每个块内的线程数量。

下面是一个简单示例，展示了一个 CUDA 核函数及核函数启动的关键组成部分。这个核函数只是把输入数组中的每个元素就地加倍，因此不会创建额外的内存——只有输入数组本身。在幕后，CUDA 会把 **global** 函数编译成 GPU 设备代码，可由数千或数百万个轻量级线程并行执行：

```
//-------------------------------------------------------
// Kernel: myKernel running on the device (GPU)
//   - input : device pointer to float array of length N
//   - N   : total number of elements in the input
//-------------------------------------------------------

__global__ void myKernel(float* input, int N) {
    // Compute a unique global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only process valid elements
    if (idx < N) {
        input[idx] *= 2.0f;
    }
}

// This code runs on the host (CPU)
int main() {
    // 1) Problem size: one million floats
    const int N = 1'000'000;
    float *h_input = nullptr;
    float *d_input = nullptr;

    // 1) Allocate input float array of size N on host
    cudaMallocHost(&h_input, N * sizeof(float));

    // 2) Initialize host data (for example, all ones)
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;
    }

    // 3) Allocate device memory for input on the device
    cudaMalloc(&d_input, N * sizeof(float));

    // 4) Copy data from the host to the device
    cudaMemcpy(d_input, h_input, N * sizeof(float),
      cudaMemcpyHostToDevice);

    // 5) Choose kernel launch parameters

    // Number of threads per block (multiple of 32)
    const int threadsPerBlock = 256;

    // Number of blocks per grid (3,907 for N = 1000000)
    const int blocksPerGrid = (N + threadsPerBlock - 1) /
      threadsPerBlock;

    // 6) Launch myKernel across blocksPerGrid blocks
    // Each block has threadsPerBlock number of threads
    // Pass a reference to the d_input device array
    myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input,
      N);
    // 7) Wait for the kernel to finish running on device
    cudaDeviceSynchronize();

    // 8) When finished, copy the results
    //    (stored in d_input) from the device back to
    //     host (stored in h_input)

    cudaMemcpy(h_input, d_input, N * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Cleanup: Free memory on the device and host
    cudaFree(d_input);
    cudaFreeHost(h_input);

    // return 0 for success!
    return 0;
```

> 这段代码尚未完全优化。我们会在本书后续过程中不断优化性能。但它给了你一个简单、完整的模板，可用于开始构建你自己的 CUDA 核函数。

在这里，我们传入核函数的输入参数 d_input 和 N，它们在核函数内部可访问以供处理。处理工作会并行地分摊到许多线程上。这是有意为之的设计。

完整的数据流如下：

1. 在主机上分配内存（h_input）。

2. 使用带 cudaMemcpyHostToDevice 的 cudaMemcpy 将数据从主机（h_input）复制到设备（d_input）。

3. 用 d_input 在设备上运行核函数。

4. 进行同步，以确保核函数已在设备上执行完毕。

5. 使用带 cudaMemcpyDeviceToHost 的 cudaMemcpy 将结果（d_input）从设备传回主机（h_input）。

6. 用 cudaFree 和 cudaFreeHost 清理设备和主机上的内存。

你可以在启动时用 <<< >>> 向核函数传入额外的、高级的、CUDA 特有的参数，包括共享内存大小（以及许多其他参数），但两个核心启动参数 blocksPerGrid 和 threadsPerBlock 是任何 CUDA 核函数调用的基础。在下一节中，我们将讨论如何最好地选择这些启动参数的取值。

你可能会疑惑，为什么我们必须传入输入数组的大小 N。这看起来是多余的，因为核函数应该能够检查数组的大小。然而，这正是 GPU CUDA 核函数与典型 CPU 函数之间的核心区别：CUDA 核函数被设计为运行在单个线程内部，与其他数千个线程一起，处理输入数据的一个分区。因此，N 定义了这个特定核函数将要处理的分区大小。

结合内置的核函数变量 blockDim（在本例中为 1，因为我们传入的是一维输入数组）、blockIdx 和 threadIdx，核函数计算出进入输入数组的具体 idx。这个唯一的 idx 让核函数能够干净且唯一地处理输入数组的每一个元素，并行地跨许多同时运行在许多不同 SM 上的线程进行。

注意其中的边界检查 if (idx < N)。之所以需要它，是为了避免越界访问（边界检查），因为 N 未必恰好是块大小的整数倍。例如，考虑这样一种情形：输入数组大小为 63，因此 N = 63。warp 调度器很可能会分配两个 warp（每个 32 线程）来处理输入数组中的 63 个元素。

第一个 warp 会同时运行 32 个核函数实例来处理元素 0–31，绝不会超过 N = 63。这很直接。与第一个 warp 并行运行的第二个 warp，本会期望处理元素 32–64。然而，它会在到达 N = 63 时停止。

如果没有 if (idx < N) 边界检查，第二个 warp 会尝试处理 idx = 64，并抛出非法内存访问错误（例如 cudaErrorIllegalAddress）。边界检查确保每个线程要么处理一个有效的输入元素，要么在其 idx 越界时立即退出。

CUDA 核函数在设备上异步执行，且没有每线程的异常机制；相反，任何非法操作（越界访问、未对齐访问等）都会为整个启动设置一个全局故障标志。主机驱动只有在你下一次调用同步或另一个 CUDA API 函数时才检查该标志，因此错误是延迟浮现的（例如以 cudaErrorIllegalAddress 或一个通用的启动失败的形式）。

这种设计让 GPU 的流水线和互连保持充分占用，但要求你在主机上显式同步并轮询错误——通常在核函数启动之后立即调用 cudaGetLastError() 和 cudaDeviceSynchronize()。这样，你就能在故障一发生时就将其捕获。

你会在很多 CUDA 核函数中看到边界检查。如果你没看到它，就应该弄清楚它为什么不在那里。它很可能以某种形式存在——或者 CUDA 核函数开发者能够以某种方式保证非法内存访问错误永远不会发生。

最后，我们来到实际的核函数逻辑。在计算出进入输入数组的唯一索引 idx 之后，这个核函数（在许多 SM 上并行地运行在数千个线程上，各自独立）将输入数组中索引 idx 处的值乘以 2。然后它就地更新输入数组中的该值。在这个具体的核函数中，除了 int 类型的临时变量 idx 之外，不需要额外的内存。

### 配置启动参数：每网格块数与每块线程数

如前所述，使用一个是 warp 大小（32）整数倍的块大小至关重要。256（8 个 warp）的 threadsPerBlock 大小是一个常见的起点，用以平衡占用率与资源使用。这会帮助我们在核函数执行期间避免部分填充的 warp、隐藏延迟，并平衡 SM 及其他硬件资源：

_32 线程的整数倍_

选择一个是 32 线程整数倍的块大小，有助于避免空的 warp 槽。否则，那些填充不满的 warp 会占用稀缺的调度器资源——却不贡献有用的工作。

_延迟隐藏_

需要每个 SM 有数百个线程来隐藏 DRAM 和指令延迟造成的停顿。如果你在一个容量为 2,048 线程的 SM 上启动，比如说，8 个各 256 线程的块，就能让流水线保持忙碌而不会过度订阅。

_占用率_

例如，使用 256 的 threadsPerBlock，你每个块只需要 8 个 warp。这往往能带来良好的占用率，同时不会耗尽每个块的寄存器或共享内存。

> 对于 Blackwell 这样的现代 GPU，可考虑每块 256–512 个线程，在遵守寄存器和共享内存上限的同时最大化占用率。

_资源均衡_

256 足够小，你很少会超过每块 1,024 线程的上限。它又足够大，以至于当其他 warp 中的线程停顿时，你不会让太多 warp 空闲。

从 threadsPerBlock=256 开始，你可以根据核函数的寄存器和共享内存需求——以及占用率特征——向上或向下调整（128、512 等）。

对于 blocksPerGrid，你可以基于 N 个输入元素的数量和 threadsPerBlock 的取值来确定它。例如，blocksPerGrid 通常被设为 (N + threadsPerBlock - 1) / threadsPerBlock 来向上取整，这样即使 N 不是 threadsPerBlock 的整数倍，你也能覆盖所有元素。这是一个常见选择，保证每个输入元素都被一个线程覆盖。下面的代码展示了这个计算：

```
//-------------------------------------------------------
// Kernel: myKernel running on the device (GPU)
//   - input : device pointer to float array of length N
//   - N   : total number of elements in the input
//-------------------------------------------------------
__global__ void myKernel(float* input, int N) {
    // Compute a unique global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only process valid elements
    if (idx < N) {
        input[idx] *= 2.0f;
    }
}

// This code runs on the host (CPU)
int main() {
    // 1) Problem size: one million floats
    const int N = 1'000'000;

    float* h_input = nullptr;
    cudaMallocHost(&h_input, N * sizeof(float));

    // Initialize host data (for example, all ones)
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;
    }

    // Allocate device memory for input on the device (d_)
    float* d_input = nullptr;
    cudaMalloc(&d_input, N * sizeof(float));

    // Copy data from the host to the device using cudaMemcpyHostToDevice
    cudaMemcpy(d_input, h_input, N * sizeof(float),
      cudaMemcpyHostToDevice);

    // 2) Tune launch parameters
    const int threadsPerBlock = 256; // multiple of 32
    const int blocksPerGrid = (N + threadsPerBlock - 1) /
      threadsPerBlock; // 3,907, in this case

    // Launch myKernel across blocksPerGrid number of blocks
    // Each block has threadsPerBlock number of threads
    // Pass a reference to the d_input device array
    myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, N);
    // Wait for the kernel to finish running on the device
    cudaDeviceSynchronize();

    // When finished, copy results (stored in d_input) from device to host
    // (stored in h_input) using cudaMemcpyDeviceToHost
    cudaMemcpy(h_input, d_input, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup: Free memory on the device and host
    cudaFree(d_input);
    cudaFreeHost(h_input);

    return 0; // return 0 for success!
```

这与前面是同一个核函数，但会基于 N 的大小动态计算 blocksPerGrid 和 threadsPerBlock。注意那个熟悉的 if (idx < N) 边界检查。它确保最后一个块中任何落在 N 之外的“多余”线程只会什么都不做——而不会引发非法内存地址错误。接下来，让我们探讨像 2D 图像和 3D 体数据这样的多维输入。

### 2D 与 3D 核函数输入

当你的输入数据天然存在于二维中（例如图像）时，你可以启动一个由 2D 块组成的 2D 网格。例如，下面是一个核函数，它使用一个 16 × 16 维的线程块（共 256 个线程）来处理一个二维的 1,024 × 1,024 矩阵：
