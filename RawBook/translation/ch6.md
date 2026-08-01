# 第 6 章　GPU 架构、CUDA 编程与最大化占用率

在本章中，我们将首先回顾单指令多线程（single instruction, multiple-threads，SIMT）执行模型，以及 warp、线程块（thread block）和网格（grid）如何将你基于 GPU 的算法映射到流式多处理器（streaming multiprocessor，SM）上。

我们会回顾现代 NVIDIA GPU 上的 SIMT 执行模型，包括 warp、线程块和网格如何映射到 SM。随后深入 CUDA 编程模式，讨论片上内存层级（寄存器堆、共享内存/L1、L2、HBM3e），并演示 GPU 的异步数据传输能力，包括 Tensor Memory Accelerator（TMA）以及作为 Tensor Core 运算累加器的 Tensor Memory（TMEM）。

我们还会引入 Roofline 分析，用以识别计算受限（compute-bound）与访存受限（memory-bound）的核函数。这将为把现代 GPU 系统推向其理论峰值吞吐上限打下基础。

## 理解 GPU 架构

与面向低延迟单线程性能优化的 CPU 不同，GPU 是面向吞吐优化的处理器，天生就是为并行运行数千个线程而构建的。CPU 与 GPU 之间一个简单的 CUDA 编程流程如图 6-1 所示。

![图 6-1. 简单的 CUDA 编程流程](../images/figure-6-1.png)

最初，主机（host）将数据加载到 CPU 内存中。然后它把数据从 CPU 复制到 GPU 内存。在用 GPU 内存中的数据调用 GPU 核函数（kernel）之后，CPU 再把结果从 GPU 内存复制回 CPU 内存。此时结果又回到 CPU 上，供进一步处理。

GPU 依靠大规模并行来隐藏诸如图 6-1 中描述的 CPU-GPU 数据传输之类的数据传输延迟。每个 GPU 由许多 SM 组成，它们大致类似于 CPU 核心，但为并行做了精简。在 Blackwell 上，每个 SM 最多可以跟踪 64 个 warp（32 线程一组）。

每个 GPU 都包含许多 SM——类似于 CPU 核心，但为吞吐做了优化。在现代 GPU 上，每个 SM 最多可并发跟踪 64 个 warp（2,048 个线程）。Blackwell GPU 每个 SM 拥有 64K 个 32 位寄存器（共 256 KB），以及每个 SM 合计 256 KB 的 L1 缓存/共享内存。这块 SRAM 中最多可有 228 KB（227 KB 可用）被配置为每个 SM 由用户管理的共享内存。任何单个线程块最多可以请求 227 KB 的动态共享内存（228 KB 中有 1 KB 由 CUDA 保留）。这些都有助于 SM 支撑 GPU 高度的线程级并行。

在一个 Blackwell SM 内部，多个 warp 调度器（warp scheduler）向可用流水线发射指令；四个独立的 warp 调度器允许每个周期最多有四个 warp 向可用流水线发射指令。此外，每个调度器都支持双发射（dual-issue），能够为每个 warp 发射两条独立指令（例如一条算术运算和一条内存操作）。注意，双发射必须来自同一个 warp——而不能跨 warp。

在最好的情况下，每个调度器在每个周期都能有一个 warp 并发发射一条指令，从而允许每个周期有四个 warp 并行执行。当利用指令混合时，这会进一步提升吞吐，如图 6-2 所示。

![图 6-2. Blackwell SM 包含四个独立的 warp 调度器，每个调度器每周期能发射一条 warp 指令，并可为每个调度器双发射一条算术运算和一条内存操作](../images/figure-6-2.png)

在这里，每个 SM 被细分为四个独立的调度分区——每个分区都有自己的 warp 调度器和分派逻辑。你可以把 SM 想象成共享片上资源的四个“迷你 SM”。这让硬件能够挑选就绪的 warp，并在每个时钟周期从多达四个不同的 warp 发射指令。

在这四个“迷你 SM”分区中的每一个内部，调度器每周期都能从同一个 warp 发射两条指令：一条算术指令（例如 INT32、FP32 或 Tensor Core）和一条内存指令（一次加载或存储）。这正是该调度器被称为*双发射*的原因。表 6-1 汇总了这些数字。

*表 6-1. 关键的 SM 调度器与指令发射上限（每个时钟周期）*

| 指标 | 数值 |
| --- | --- |
| 调度器数量 | 四个 |
| 最多发射的 warp 数 | 四个（每个调度器一个） |
| 最多算术运算数 | 四个（每个调度器的算术发射一个） |
| 最多内存操作数 | 四个（每个调度器的加载/存储发射一个） |

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

![图 6-3. 线程、线程块（又称 CTA）与网格](../images/figure-6-3.png)

通过合理设定网格大小，你可以在不改变核函数逻辑的情况下扩展到数百万个线程。CUDA 的运行时（以及 PyTorch 这样的框架）会处理跨所有 SM 的调度与分发。图 6-4 展示了线程层级的另一个视角，包括基于 CPU 的主机——它调用运行在 GPU 设备上的 CUDA 核函数。

![图 6-4. 线程层级视图，包括基于 CPU 的主机，它启动运行在 GPU 设备上的核函数](../images/figure-6-4.png)

传统上，来自不同线程块的线程无法彼此直接协作。然而，现代 GPU 架构和 CUDA 版本支持线程块簇（thread block cluster）。线程块簇是一组能够跨 SM 相互通信的线程块。

具体来说，在一个线程块簇内，不同线程块中的线程可以访问彼此的共享内存，并使用硬件支持的、簇作用域的屏障（barrier）。这些能力允许进行更大规模的计算运算，包括矩阵乘法——它在当今庞大的 LLM 工作负载中非常常见。线程块簇在参与该簇的各个 SM 之间共享一个分布式共享内存（distributed shared memory，DSMEM）地址空间，如图 6-5 所示。

![图 6-5. 硬件支持的 DSMEM，用于包含多个线程块的线程块簇](../images/figure-6-5.png)

DSMEM 是一项硬件特性，它通过快速的片上互连将一个线程块簇内所有 SM 的共享内存 bank 连接起来。借助 DSMEM，这些 SM 共享一个合并的多 SM 分布式共享内存池。这种统一让不同块中的线程能够以片上速度读取、写入并原子更新彼此的共享缓冲区——而且无需占用全局内存带宽。

> 我们将在第 10 章讨论线程块簇和 DSMEM 等高级主题。它们是现代 GPU 处理中极其重要的新增能力——对于 AI 系统性能工程师而言也非常重要，需要理解。就本章而言，我们的重点仍然放在块内共享内存优化上。

在每个线程块内部，线程使用低延迟的片上共享内存来共享数据，并用 __syncthreads() 进行同步。由于每次屏障都会带来开销，你应尽量减少同步点，如图 6-6 所示。

![图 6-6. 在两段代码之间对一个线程块内的所有线程进行同步](../images/figure-6-6.png)

目标是尽量减少同步点。不过，GPU 硬件会通过在多个 warp 之间快速切换，尝试隐藏诸如全局内存加载、缓存填充和流水线停顿等长延迟事件。

线程块被细分为 32 个线程的 warp，它们在 SIMT 模型下由 warp 调度器管理并以锁步（lockstep）方式执行。如图 6-7 所示。

![图 6-7. warp（32 线程）作为一个整体推进，其指令由 warp 调度器管理](../images/figure-6-7.png)

在 SM 上保持更多 warp 处于运行中，被称为*高占用率*。当你的 CUDA 代码能够实现高占用率时，就意味着当一个 warp 停顿时，另一个 warp 已准备好运行。这让 GPU 的计算单元保持忙碌。

然而，高占用率必须与每线程资源上限（如寄存器和共享内存）相权衡。将寄存器溢出（register spilling）到较慢的内存中会造成新的停顿。在剖析占用率的同时也剖析寄存器和共享内存的使用情况，有助于你选择既能最大化吞吐、又不会触发资源争用的块大小。

> 我们将在第 8 章讨论占用率调优，但在 SM、warp、线程等语境下，它是一个需要理解的关键概念。

线程块相互独立执行，且没有保证的执行顺序。这让 GPU 调度器能够将它们分派到所有 SM 上，充分利用硬件并行。这种网格—块—warp 层级保证了你的 CUDA 核函数无需修改即可运行在拥有更多 SM 和线程的未来 GPU 架构上。

吞吐还取决于 warp 的执行效率。同一个 warp 中的线程必须遵循相同的控制流路径并执行合并访问（coalesced access）的内存访问。如果某些线程发生分歧，使得一个分支走 if 路径而其他线程走 else 路径，那么该 warp 会串行化执行，逐一顺序处理每条分支路径。这被称为*warp 分歧*（warp divergence），如图 6-8 所示。

![图 6-8. SIMT warp 分歧（左）与一致性（右）的对比](../images/figure-6-8.png)

通过屏蔽非活跃的 lane 并运行额外的遍历来覆盖每条分支，warp 分歧会把整体执行时间乘以分支的数量。我们将在第 8 章更深入地探讨 warp 分歧——以及检测、剖析和缓解它的方法。

> 分歧只对单个 warp 内部的线程构成问题。不同的 warp 可以走不同的分支，而不会有性能损失。

### 选择每块线程数与每网格块数

GPU 性能的一个关键方面是选择一个与硬件 32 线程 warp 大小对齐的线程块大小。因此，你通常会挑选恰好是 32 的整数倍的线程块大小。例如，一个 256 线程的块（8 个 warp = 256 ÷ 32）能完全占满每个 warp，而一个 33 线程的块则需要两个 warp 槽，且只用到第二个 warp 中 1/32 的 lane。这会浪费并行机会，因为无论一个 warp 是在活跃运行 32 个线程还是仅 1 个线程，它都会占用一个调度器槽。

此外，不同的 GPU 世代有不同的硬件上限，包括每个 SM 的最大线程数和每个 SM 的寄存器数量。如果我们想保持良好性能，这自然会限制块的大小。例如，块太大可能需要太多寄存器，从而导致*寄存器溢出*并降低核函数的性能。

大块也可能需要太多共享内存，而共享内存在 GPU 硬件中是有限的。具体来说，Blackwell 每个 SM 仅提供 228 KB（227 KB 可用）的共享内存，供运行在该 SM 上的所有驻留线程块寻址。

这些硬件上限会影响一个 SM 上同时能激活多少个块/warp。这正是我们前面介绍过的占用率的度量。如果较小的块能让更多并发 warp 在 SM 上同时运行，它们可能带来更高的占用率。

理解你所用 GPU 世代的相对规模和硬件线程上限很重要，包括线程、线程块、warp 和 SM 的数量。图 6-9 展示了这些资源的相对规模，包括它们的上限。

![图 6-9. Blackwell GPU 上线程的相对规模与硬件上限](../images/figure-6-9.png)

表 6-2 汇总了 Blackwell B200 GPU 的这些上限。其余上限可在 NVIDIA 网站上查到。（其他 GPU 世代会有不同的上限，因此务必核对你系统的确切规格。）

*表 6-2. 线程级与块级上限（Blackwell B200）*

| 资源 | 硬件上限 | 说明 |
| --- | --- | --- |
| warp 大小 | 32 线程 | 基本的 SIMT 执行单元是 32 个线程（一个 warp）。始终使用 32 的整数倍以避免浪费。 |
| 每个线程块的最大线程数 | 1,024 线程 | blockDim.x * blockDim.y * blockDim.z ≤ 1024。 |
| 每个线程块的最大 warp 数 | 32 warp | （1,024 线程 ÷ 每个 warp 32 线程）= 每块最多 32 个 warp。 |

我们已经讨论过 32 线程的 warp 大小上限，它促使我们选择 32 线程整数倍的块维度，以构成“完整 warp”并避免利用不足的 warp。注意，每个块最多可有 1,024 个线程，相应地，一个块只能包含 32 个 warp。这些上限会影响你的占用率，因为一旦一个块被调度，每个 SM 只能同时容纳有限数量的 warp 和块。

此外，针对不同的 GPU 世代，还存在每 SM 的上限，或常被称为 *SM 驻留上限*。这些 Blackwell 的 SM 驻留上限汇总在表 6-3 中。

*表 6-3. SM 驻留资源上限（Blackwell B200）*

| 资源（每 SM） | 硬件上限 | 说明 |
| --- | --- | --- |
| 每 SM 最大驻留 warp 数 | 64 warp | 硬件最多可保持 64 个 warp 在运行中（64 × 32 线程 = 2,048 线程）。注：这一上限已经历多个世代保持不变，在 Blackwell 上依然成立。 |
| 每 SM 最大驻留线程数 | 2,048 线程 | 等于 64 warp × 每 warp 32 线程。如果每个块使用 1,024 个线程，那么一个 SM 上最多可同时驻留 2 个这样的块（64 warp）。使用更小的块（例如 256 线程）可让更多块驻留在 SM 上（最多 8 块 × 256 = 2,048 线程），这能提升占用率并帮助隐藏延迟——不过过多的微小块会增加调度开销。 |
| 每 SM 最大活跃块数 | 32 块 | 一个 SM 上最多可同时驻留 32 个线程块（如果块更小，则在此上限内可容纳更多）。 |

在这里，我们看到 Blackwell 上每 SM 的最大并发 warp 数为 64。这在近几代 GPU 中没有变化，因此占用率方面的考量得以延续。一个 SM 上的最大活跃块数为 32，相应地，每 SM 的最大驻留线程数为 2,048 线程。CUDA 网格也有最大维度，如表 6-4 所示。

*表 6-4. CUDA 网格上限*

| 网格维度 | 上限 | 说明 |
| --- | --- | --- |
| X、Y 或 Z 方向的最大块数 | X：2,147,483,647 块；Y：65,535 块；Z：65,535 块 | 一个 3D 网格最大可达 2,147,483,647 × 65,535 × 65,535 块。 |
| 最大并发网格（核函数）数 | 128 网格 | 一个设备上最多可并发执行 128 个核函数（即同时驻留 128 个网格）。 |

虽然了解理论上的网格上限是好事，但你通常会先受制于前面展示的线程/块/每 SM 上限。如果你确实需要在某一维度上超过 65,535 个块，可以启动一个 2D 或 3D 网格，将工作拆分到多次核函数启动（多次启动，multilaunch）中。我们会在后面的小节中给出这样的示例。实际上，很少会在触及其他资源上限之前先触及网格大小上限。

### CUDA GPU 向后与向前兼容模型

CUDA 的核心优势之一是它的向前与向后兼容模型。只要你在二进制文件中包含 PTX 以实现向前兼容，今天编译的核函数通常无需修改即可运行在未来的 GPU 世代上。如果你只针对单个架构发布 SASS（例如面向 Hopper 的 sm_90 或面向 Blackwell 的 sm_100）而不带 PTX，那么该二进制文件将无法在更新的架构上向前运行。像 sm_100f 或 compute_100f 这样的家族特定目标会将可移植性限制在同一特性家族的设备内。最佳做法是发布一个既包含通用 cubin/PTX、又包含所需家族特定 cubin（例如各种优化等）的 fatbin。

你可以通过在加载时强制进行 PTX JIT 编译来验证兼容性——设置 CUDA_FORCE_PTX_JIT=1 来即时编译 PTX 并缓存结果。如果你的二进制文件缺少 PTX，核函数启动将会失败。这会迫使你带上 PTX 支持重新构建。这种兼容模型是庞大 CUDA 生态系统的根基。它让你能从单一代码库同时面向旧硬件和最前沿硬件。

> 若要在当前与未来 GPU 世代之间真正保持向后与向前兼容，你应当使用通用目标进行编译——或显式包含 PTX。当你需要来自较新硬件特性的特定优化时，可以使用世代特定的目标。这样做时，务必为其他架构提供回退路径。

## CUDA 编程复习

在 CUDA C++ 中，你通过编写核函数来定义并行工作。这些是用 __global__ 注解的特殊函数，运行在 GPU 设备上。当你从 CPU（主机）代码调用一个核函数时，你使用 <<< >>>（“chevron”/三尖括号）语法来指定应运行多少个线程——以及它们如何组织——需要两个配置参数：blocksPerGrid 表示线程块的数量，threadsPerBlock 表示每个块内的线程数量。

下面是一个简单示例，展示了一个 CUDA 核函数及核函数启动的关键组成部分。这个核函数只是把输入数组中的每个元素就地加倍，因此不会创建额外的内存——只有输入数组本身。在幕后，CUDA 会把 __global__ 函数编译成 GPU 设备代码，可由数千或数百万个轻量级线程并行执行：

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

*32 线程的整数倍*

选择一个是 32 线程整数倍的块大小，有助于避免空的 warp 槽。否则，那些填充不满的 warp 会占用稀缺的调度器资源——却不贡献有用的工作。

*延迟隐藏*

需要每个 SM 有数百个线程来隐藏 DRAM 和指令延迟造成的停顿。如果你在一个容量为 2,048 线程的 SM 上启动，比如说，8 个各 256 线程的块，就能让流水线保持忙碌而不会过度订阅。

*占用率*

例如，使用 256 的 threadsPerBlock，你每个块只需要 8 个 warp。这往往能带来良好的占用率，同时不会耗尽每个块的寄存器或共享内存。

> 对于 Blackwell 这样的现代 GPU，可考虑每块 256–512 个线程，在遵守寄存器和共享内存上限的同时最大化占用率。

*资源均衡*

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

```
// 2d_kernel.cu

#include <cuda_runtime.h>
#include <iostream>

//-------------------------------------------------------
// Kernel: my2DKernel running on the device (GPU)
//   - input  : device pointer to float array of size width×height
//   - width  : number of columns
//   - height : number of rows
//-------------------------------------------------------
__global__ void my2DKernel(float* input, int width, int height) {
    // Compute 2D thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Only process valid pixels
    if (x < width && y < height) {
        int idx = y * width + x;
        input[idx] *= 2.0f;
    }
}

int main() {
    // Image dimensions
    const int width  = 1024;
    const int height = 1024;
    const int N      = width * height;

    // 1) Allocate and initialize host image
    float* h_image = nullptr;
    cudaMallocHost(&h_image, N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        h_image[i] = 1.0f;  // e.g., initialize all pixels to 1.0f
    }

    // 2) Allocate device image and copy data to device
    cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    float* d_image = nullptr;
    cudaMallocAsync(&d_image, N * sizeof(float), s);
    cudaMemcpyAsync(d_image, h_image, N * sizeof(float),
                    cudaMemcpyHostToDevice, s);

    my2DKernel<<<blocksPerGrid2D, threadsPerBlock2D,
                 0, s>>>(d_image, width, height);

    cudaMemcpyAsync(h_image, d_image, N * sizeof(float),
                    cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    cudaFreeAsync(d_image, s);
    cudaStreamDestroy(s);
```

这里再次给出完整的核函数（设备端）与调用（主机端）代码。同样的模式可以直接推广到 3D：只需对 blocksPerGrid 和 threadsPerBlock 都使用 dim3(x, y, z)，即可把体数据直接映射到 GPU 的线程层级上。

> 本书大多数情况下对 blocksPerGrid 和 threadsPerBlock 使用 1D 或 2D（分块）取值。在 1D 情形下，你可以把 blocksPerGrid 和 threadsPerBlock 定义为简单的常量，而不必使用 dim3。

### 异步内存分配与内存池

如前面示例所示，标准的 cudaMalloc/cudaFree 调用是同步的，且开销相对较大。它们需要一次完整的设备同步（相对较慢），并涉及 mmap/ioctl 等操作系统级调用来管理 GPU 内存。

这种操作系统级交互会引发内核态上下文切换和驱动开销，因此相较纯设备端操作而言相对较慢。为此，建议使用异步版本 cudaMallocAsync 和 cudaFreeAsync，以在 GPU 上实现更高效的内存分配。

默认情况下，CUDA 运行时维护一个全局的 GPU 内存池。当你异步释放内存时，它会回到内存池中，供后续分配复用。cudaMallocAsync 和 cudaFreeAsync 在底层就使用了 CUDA 内存池（memory pool）。

内存池会回收已释放的内存缓冲区，避免为分配新内存而反复进行操作系统调用。例如，在一个长时间运行的训练循环中，它通过复用先前已释放的块（而不是每次迭代都新建），有助于随时间推移减少内存碎片化。许多高性能库和运行时（如 PyTorch）默认启用了内存池。

事实上，PyTorch 使用一个自定义的内存缓存分配器（caching allocator），通过 PYTORCH_ALLOC_CONF（旧称 PYTORCH_CUDA_ALLOC_CONF）进行配置。PyTorch 的内存缓存分配器在思路上与 CUDA 的内存池类似：它复用 GPU 内存，避免在——例如长时间运行的训练循环的每次迭代中——为每个新创建的 PyTorch 张量都调用同步的 cudaMalloc 操作所带来的开销。

在需要频繁进行细粒度分配的 CUDA 应用中，使用基于内存池的异步例程——cudaMallocAsync 和 cudaFreeAsync——要比使用传统的同步 cudaMalloc/cudaFree 高效得多，后者会引发整设备同步，甚至操作系统级调用。要使用流序（stream-ordered）分配，先创建一个非阻塞流：

```
cudaStream_t stream1;
cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
```

> 使用显式 CUDA 流是重叠传输、核函数与内存操作的最佳实践。可以把每个流看作一个隔离的通道，它在自身的操作之间强制保持顺序。此外，建议用 cudaStreamCreateWithFlags(..., cudaStreamNonBlocking) 创建非阻塞流，以避免遗留的默认流屏障。我们将在第 11 章更详细地探讨多流重叠技术与最佳实践。

然后，每当你需要一个包含 N 个 float 的缓冲区时，就在该流上使用 cudaMallocAsync 和 cudaFreeAsync 进行分配与释放，如下所示：

```
float* d_buf = nullptr;
cudaMallocAsync(&d_buf, N * sizeof(float), stream1);

// ... launch kernels into stream1 that use d_buf ...
myKernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_buf, N);

// Free is deferred until all work in stream1 completes—
cudaFreeAsync(d_buf, stream1);
```

这些 API 从每个设备的内存池中分配，但会尊重你传入的流的顺序，因此释放会被推迟，直到该流的工作完成。而且由于 cudaFreeAsync 只等待 stream1 完成，所以既不需要开销高昂的全局 cudaDeviceSynchronize，也不会与其他流发生隐式同步。其结果是：当你的代码发起成千上万——甚至数百万——次分配/释放循环时，分配开销大幅降低，同时减少碎片化并平滑延迟尖峰。总体而言，相较传统的 cudaMalloc 和 cudaFree，这种模式减少了全局同步与碎片化。

你还可以进一步调节从设备内存池进行的流序分配的行为——例如，设置 cudaMemPoolAttrReleaseThreshold，提示内存池在尝试释放之前应保留多少预留内存。你也可以使用 cudaMemPoolTrimTo 主动归还内存。这些手段有助于在 GPU 总内存占用与碎片化之间取得平衡。

对于简单的一次性缓冲区，阻塞式的 cudaMalloc 和 cudaFree 或许就够用了。然而在更复杂、长时间运行、反复分配和释放内存的循环中，改用专用流上的 cudaMallocAsync 和 cudaFreeAsync 并利用其内存池，将带来更一致的性能和更高的吞吐量。

改用专用流上的 cudaMallocAsync 和 cudaFreeAsync 并利用其内存池，将带来更一致的性能和更高的吞吐量。你还可以用 cudaMemPoolSetAttribute 进一步调节内存池行为（例如调整 cudaMemPoolAttrReleaseThreshold），以 *调节释放阈值，在最小内存占用与低碎片化之间取得恰当的权衡。*

### 理解 GPU 内存层级

到目前为止，我们一直在较高层面上宽泛地讨论内存分配，且通常是从全局内存进行分配。这些分配来自某个流的内存池——包括默认的流 0 内存池。

然而实际上，GPU 提供了一个多级内存层级（memory hierarchy），有助于在容量与速度之间取得平衡。该层级包括寄存器、共享内存、缓存、全局内存，以及 Blackwell 及更新 GPU 上专用的 TMEM。TMEM（稍后详述）是一块专用的每 SM 约 256 KB 的片上内存，供 Blackwell 第五代 Tensor Core 指令（tcgen05.*）使用。它无法从 CUDA C++ 中直接以指针寻址。相反，数据搬运由 TMA 硬件（global memory ↔ SMEM）以及 tcgen05 Tensor Core 数据搬运指令（SMEM ↔ TMEM，隐式地使用张量描述符）来编排。

全局内存（HBM 或 DRAM）容量大、位于片外，且相对较慢。寄存器很小、位于片上，且极快。L1 缓存、L2 缓存与共享内存则介于两者之间。缓存和共享内存的好处在于，它们能隐藏访问大容量片外内存存储的相对较长的延迟。GPU 内存层级（含 CPU）的高层视图如图 6-10 所示。

![图 6-10. GPU 内存层级（含 CPU）](../images/figure-6-10.png)

TMEM 是一块专用的每 SM 256 KB 缓冲区，以每秒数十太字节的带宽透明地与 Tensor Core 通信。这减少了 Tensor Core 对全局内存的依赖。图 6-11 展示了 TMEM 与 SMEM 一起为 Tensor Core 提供服务，以计算 C = A × B 矩阵乘法。

![图 6-11. TMEM 与 SMEM 为 Tensor Core 提供服务以计算 C = A × B 矩阵乘法](../images/figure-6-11.png)

在这里，操作数 B 来自 SMEM。操作数 A 位于 TMEM（不过它也可能位于 SMEM）。累加器同样位于 TMEM。分块（tile）通过 TMA（例如 cuda::memcpy_async）从全局内存经 L2 缓存流向 SMEM。

操作数在 SMEM 与 TMEM 之间的移动，则是通过诸如 unified matrix-multiply-accumulate（UMMA）和 tcgen05.mma 之类的 Tensor Core 指令隐式完成的。

表 6-5 展示了 Blackwell GPU 各级内存及其特性。随后对内存层级的每一级进行说明。

*表 6-5. Blackwell 内存层级及其特性*

| 级别 | 作用范围 | 容量 | 延迟 | 带宽（近似） |
| --- | --- | --- | --- | --- |
| 寄存器 | 每线程（位于 SM 上） | 每 SM 64 K 个 32-bit 寄存器（每线程最多 255 个） | 接近寄存器延迟（寄存器读写为单周期，基本零开销） | 每 SM 数十 TB/s（寄存器堆端口） |
| 共享内存与 L1 缓存 | 每 SM | 228 KB（227 KB usable）共享内存 + 其余作为 L1/数据缓存 | ~20–30 周期（L1/共享内存基准） | 每 SM TB/s 级（无 bank 冲突时） |
| TMEM | 每 SM | 每 SM 256 KB SRAM，专供 Tensor Core | ~10 周期（SM 上的专用 SRAM） | 与 Tensor Core 之间 TB/s 级的通信 |
| 常量内存缓存 | 每 SM | 约 8 KB 缓存，服务于 64 KB 的 __constant__ 空间 | ~1 周期（warp 广播）。命中缓存且一个 warp 内所有线程访问同一地址时，凭借常量缓存与广播行为，速度可快如寄存器。发生分歧或未命中时则会串行化，或产生更高延迟 | TB/s 级（广播吞吐） |
| L2 缓存 | 全 GPU 范围（所有 SM） | 共 126 MB | ~200 周期 | 多 TB/s 总带宽 |
| 本地内存 | 每线程（溢出到 DRAM） | 近乎无限（由全局内存支撑） | 100s → 1,000 周期（类 DRAM） | ~8 TB/s（HBM3e） |
| 全局内存（HBM 或 DRAM） | 全设备范围（片外 DRAM） | 每块 Blackwell B200 GPU 最高 180 GB（每块 Blackwell B300 GPU 最高 ~288 GB） | 100s → 1,000 周期（全局内存延迟） | 总计 ~8 TB/s |

由此你可以看出，为什么最大化在寄存器、共享内存以及 L1/L2 缓存中的数据复用——并最小化对全局内存和本地内存（由全局内存支撑）的依赖——对于高吞吐的 GPU 核函数至关重要。下面对该层级的每一级再作一些说明：

*寄存器*

在 Blackwell 上，每个线程的旅程都从寄存器堆（register file）开始，这是一块微小的 SM 上 SRAM 阵列，用于保存每个线程的局部变量，且几乎不增加延迟。每个 SM 拥有 64 K 个 32-bit 寄存器（共 256 KB），但硬件对每个线程最多只暴露 255 个寄存器。

由于读写都在单个周期内完成，且几乎不与其他任何东西争用，寄存器带宽每 SM 可达每秒数十太字节。然而，如果你的核函数需要更多寄存器——无论是因为大量线程局部变量还是编译器临时变量——溢出部分就会溢入本地内存（映射到片外 DRAM），并带来数百乃至上千周期的延迟。这块本地内存如图 6-12 所示。

![图 6-12. 每线程的本地内存](../images/figure-6-12.png)

*共享内存与 L1 数据缓存*

再上一层是统一的 L1/数据缓存与共享内存块。这是每 SM 256 KB 的 SM 上 SRAM，你可以在用户管理的共享内存（每块最多 228 KB）之间动态划分它——在 Blackwell 这类具有统一 L1/纹理/共享内存的架构上，使用 cudaFuncSetAttribute() 配合 cudaFuncAttributePreferredSharedMemoryCarveout 来选择内存划分（carveout）。每块最大动态共享内存为 227 KB（CUDA 为每块保留 1KB），且每 SM 可分配的共享内存总量也受此上限约束。

这里的访问大约耗费 20–30 周期，但如果你设计线程块时避免了 bank 冲突，就能达到每秒太字节级的吞吐。线程块共享内存如图 6-13 所示。

![图 6-13. 线程块共享内存](../images/figure-6-13.png)

*TMEM*

TMEM 是每 SM 的一块专用片上内存（Blackwell 上为 256 KB），供 Tensor Core 专属的运算与指令使用，包括 unified matrix-multiply-accumulate 和 tcgen05（详见第 10 章）。它不是 CUDA C++ 中普通的指针可寻址空间。相反，传输由 Tensor Memory Accelerator（TMA）借助描述符来编排。这使开发者无需手动管理与 Tensor Core 之间的数据流。例如，某些算术操作数驻留在共享内存中，而累加器则驻留在 TMEM 中。随后由 TMA 负责在全局内存、共享内存与 TMEM 内存之间搬运数据以完成计算。

*常量内存缓存*

对于微小的只读表，Blackwell 提供了每 SM 约 8 KB 的常量内存缓存，位于 64 KB 的 __constant__ 空间之前。当一个 warp 中全部 32 个线程加载同一地址时，该缓存会在单个周期内广播该值。

分歧的读取则会跨 lane 串行化。它非常适合共享小型查找表，例如旋转位置编码、带线性偏置的注意力（Attention with Linear Biases，ALiBi）斜率、LayerNorm 的 γ/β 向量以及嵌入量化缩放因子。这些都能在每个线程间共享，而无需产生全局内存流量。

*L2 缓存*

在片上 SRAM 之外是 L2 缓存，这是一块 126 MB 的全 GPU 范围缓冲区，把所有 SM 与片外 HBM3e 粘合在一起。延迟接近 200 周期，聚合带宽达每秒数十太字节，L2 吸收来自 L1 的溢出。

有了 L2，数据可以由一个线程块取回，再由其他线程块复用，而无需再次访问 DRAM。为了最大化 L2 的收益，应把你的全局加载组织成 128 字节的合并访问事务，使其干净地映射到缓存行上。稍后我们会展示具体做法。

> 把你的全局加载组织成 128 字节对齐、合并的分段，使其干净地映射到缓存行上。这可以避免拆分事务，并最大化利用 L2 与 DRAM 带宽。

*全局内存（HBM 或 DRAM）*

全局内存层级——本地溢出空间与 HBM——位于片外。任何溢出的寄存器或超大的自动数组都驻留在本地内存中，尽管 HBM3e 带宽高达 ~8 TB/s，仍要付出完整的 DRAM 延迟（数百乃至一千多周期）。

对于 Blackwell，HBM3e 层级提供最高 180 GB 的全设备范围存储，总带宽 ~8 TB/s。然而其高延迟使它成为整条链路中最慢的一环。每设备的全局内存如图 6-14 所示。

![图 6-14. 每设备的全局内存，即 HBM](../images/figure-6-14.png)

借助 Nsight Compute 之类的工具跟踪溢出与缓存命中率，你可以让核函数尽可能贴近该内存层级的片上峰值运行。这些工具能帮助你有效地在寄存器、shared/L1、常量缓存与 L2 缓存之间编排数据。像 Blackwell 这样的现代 GPU 允许核函数开发者利用内存层级——使用 L2 缓存和统一的 L1/共享内存来缓冲并合并对 HBM 的访问，我们很快就会看到。

> Blackwell B200 对外呈现为一块单一 GPU，构建于统一的全局地址空间之上。然而，它由两个光罩受限裸片（reticle-limited die）组成，二者通过 10 TB/s 的芯片间互连连接。每个裸片连接到四个 HBM3e 堆栈，共计八个 HBM3e 堆栈。不过，从开发者的视角看，HBM 内存访问在这一合并地址空间中是一致的，但了解这一架构的底层细节仍然是值得的。

内存层级中各级的一致性点（point of coherency，PoC）取决于你的需求，以及线程进行通信所处的层级。它通常发生在以下层级：线程、线程块（又称 *CTA*）、线程块簇（又称 *CTA 簇*）、设备或系统，如图 6-15 所示。

![图 6-15. GPU 线程的内存一致性点](../images/figure-6-15.png)

总而言之，理解 GPU 的内存层级并恰当地针对每一级进行优化非常重要。这样做，你就能构建 CUDA 核函数以最大化数据局部性、隐藏内存访问延迟、提高占用率，并充分发挥 Blackwell 强大的并行计算能力，稍后我们会加以探讨。首先，让我们讨论 NVIDIA 的统一内存——鉴于 Grace Hopper 与 Grace Blackwell 的 CPU-GPU 统一超级芯片设计，理解它十分重要。

### 统一内存

统一内存（Unified Memory，又称 CUDA 托管内存（Managed Memory））为你提供一个横跨 CPU 与 GPU 的单一、一致的地址空间，因此你不再需要费心管理分开的主机与设备缓冲区，也不必发起显式的 cudaMemcpy 调用。在底层，CUDA 运行时为每一次 cudaMallocManaged() 分配都以页面作为支撑，这些页面能够在连接你的 CPU 与 GPU 的任何互连上按需迁移，如图 6-16 所示。

尽管访问统一内存对开发者极为友好，它却可能引发 CPU 与 GPU 之间不期而至的按需页迁移。这会带来隐藏的延迟与执行停顿。例如，如果一个 GPU 线程访问当前驻留在 CPU 内存中的数据，GPU 就会发生缺页，并在数据经由 NVLink-C2C 互连传输期间等待。统一内存的性能在很大程度上取决于底层硬件。

在传统 PCIe 或早期 NVLink 系统上，这类迁移以相对较低的带宽进行——常常使缺页触发的传输比手动 cudaMemcpy 还慢。但在 Grace Hopper 与 Grace Blackwell Superchip 上，NVLink-C2C 结构在 CPU 的 HBM 与 GPU 的 HBM3e 之间提供最高 ~900 GB/s 的带宽。因此，缺页驱动的迁移在速度上要接近设备原生水平得多——尽管它们仍带有非零的延迟。

尽管如此，核函数启动期间任何意外的缺页都会在运行时把所需页面搬到位的过程中使 GPU 停顿。为避免这些“意外”停顿，你可以事先用 cudaMemPrefetchAsync() 预取内存，如图 6-17 所示。

这会提示驱动在你启动核函数之前，把指定范围搬到目标 GPU（或 CPU）上，从而把代价高昂的首次访问（first-touch）迁移转化为可重叠的异步传输。你还可以给出内存建议（memory advice），如下面的代码所示：

```
cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, gpuId);
cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, gpuId);
```

在这里，你可以用 PreferredLocation 告诉驱动你将主要在何处使用数据，用 ReadMostly 表示数据在很大程度上是只读的，如图 6-18 所示。

![图 6-16. CPU-GPU 统一内存的自动页迁移](../images/figure-6-16.png)

![图 6-17. 用 cudaMemPrefetchAsync() 通过 NVLink-C2C 将数据从 CPU 流式传输到 GPU](../images/figure-6-17.png)

![图 6-18. 指定“首选位置”以告诉 CUDA 驱动数据的主要用途（例如，对基本只读的工作负载使用 ReadMostly）](../images/figure-6-18.png)

你还可以调用下面的代码，让第二块 GPU 映射这些页面，而不会在启动时触发迁移：

```
cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy,
   otherGpuId);
```

默认情况下，任何 CUDA 流或设备核函数都可能在一次托管分配上触发缺页。这会导致意外的迁移和隐式同步。如果你知道某个缓冲区在同一时间只会在一个流/GPU 上使用，把它附着到那个流，就能让迁移与其他流中的操作相重叠。调用下面的代码可将该内存范围绑定到指定的流：

```
cudaStreamAttachMemAsync(stream, ptr, 0,
    cudaMemAttachSingle);
```

在这种情况下，只有该流中的操作才会缺页并迁移其页面。这可以防止其他流意外地在它上面停顿。因此，把某个范围附着到特定的流会推迟其迁移，使之只与该流的工作相重叠。这就避免了跨流同步。

> 在没有 NVLink-C2C 的多 GPU 系统中，你还可以使用 cudaMemcpyPeerAsync()，或预取到某个特定设备，以把数据固定在最近的 NUMA 本地 GPU 内存中，从而避免缓慢的远程访问。

简而言之，显式预取托管内存并提供内存建议，能够消除统一内存带来的大多数“意外”停顿。数据在核函数运行时已经就位，而不是让 GPU 暂停以按需取数。

借助主动预取、有针对性的内存建议以及流附着等技术，统一内存可以在保留统一地址空间之简洁性的同时，交付非常接近手动 cudaMemcpy 的性能。

### 维持高占用率与 GPU 利用率

GPU 通过并发运行大量 warp 来维持性能，这样当一个 warp 因等待数据而停顿时，另一个 warp 就能运行。这种在 warp 之间快速切换的能力使 GPU 得以隐藏内存延迟。正如我们前面所述，一个 SM 的容量中被活跃 warp 实际占据的比例，被称为 *占用率*。

如果占用率很低（只有寥寥几个活跃 warp），当一个 warp 在等待内存时，SM 可能就会空闲。这会导致 SM 利用率低下。在 Blackwell 上，凭借其庞大的寄存器堆（每 SM 64K 个寄存器），实现高占用率会容易一些，它可以在不溢出的情况下支撑许多 warp。

> 正如你前面所见，一个 warp 中的每个线程最多可使用 255 个寄存器。务必使用你的剖析工具来检查实际占用率——并据此调整核函数的块大小和寄存器用量。

反过来，高占用率（每 SM 有许多活跃 warp）会让 GPU 计算单元保持繁忙，因为当一个 warp 在等待内存访问时，其他 warp 会切入 SM 并执行。这就掩盖了漫长的内存访问延迟。这通常被称为 *隐藏延迟*。

让我们展示一个能提升占用率、并最终提升 GPU 利用率、吞吐量与整体核函数性能的示例。这是 CUDA 性能优化最基本的规则之一：启动足够的并行工作以充分占满 GPU。

如果你的实际占用率（正在使用的硬件线程槽的比例）远低于 GPU 的上限且性能不佳，首要的补救办法就是增加并行度——使用更多的块或线程，使占用率在现代 GPU 上接近 80%–100% 的区间。

反之，如果占用率已经处于中到高的水平，但核函数受内存吞吐所限，把它推到 100% 可能也无济于事。你通常只需要恰好足够多的 warp 来隐藏延迟，超过之后瓶颈可能就在别处了（例如内存带宽）。

为说明占用率的影响，考虑一个非常简单的操作：把两个长度为 *N* 的向量相加（计算 C = A + B）。我们将考察两种核函数实现：addSequential 和 addParallel。addSequential 使用单个线程（或单个 warp）在循环中把全部 *N* 个元素相加。addParallel 使用许多线程，使加法在整个数组上并发完成。

在串行版本中，一个 GPU 线程串行地处理整个工作负载，如下所示：

```
#include <cuda_runtime.h>

const int N = 1'000'000;

// Single thread does all N additions
__global__ void addSequential(const float* A,
                              const float* B,
                                    float* C,
                              int N)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < N; ++i) {
            C[i] = A[i] + B[i];
        }
    }
}

int main()
{
    // Allocate and initialize host
    float* h_A = nullptr;
    float* h_B = nullptr;
    float* h_C = nullptr;
    cudaMallocHost(&h_A, N * sizeof(float));
    cudaMallocHost(&h_B, N * sizeof(float));
    cudaMallocHost(&h_C, N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        h_A[i] = float(i);
        h_B[i] = float(i * 2);
    }

    // Allocate device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch: one thread
    // Note: This kernel assumes <<<1,1>>>
    // (one block, one thread).
    // Do not change the launch config when running this example.
    addSequential<<<1,1>>>(d_A, d_B, d_C, N);

    // Ensure completion before exit
    cudaDeviceSynchronize();

    // Copy d_C => h_C (back to host)
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
```

在这个单线程版本中，GPU 庞大的资源大多处于闲置状态。只有一个 warp，甚至只有该 warp 中的一个线程在工作，而其他所有线程都闲着。结果是占用率极低，并最终导致性能低下。

还必须小心，避免在 PyTorch 这类高级库和框架中间接执行低效的 GPU 代码。例如，下面这段朴素的 PyTorch 代码错误地用一个 Python for 循环执行逐元素操作，一个接一个地在 GPU 上发起了 *N* 次独立的加法操作：

```
import torch

N = 1_000_000
A = torch.arange(N, dtype=torch.float32, device='cuda')
B = 2 * A
C = torch.empty_like(A)

# Ensure all previous work is done
torch.cuda.synchronize()

# Naive, Sequential GPU operations - DO NOT DO THIS
with torch.inference_mode(): # avoids unnecessary autograd graph construction
    # This launches N tiny GPU operations serially
    for i in range(N):
        C[i] = A[i] + B[i]

torch.cuda.synchronize()
```

这段代码实际上把 GPU 当成了一个标量的、非并行的处理器。它的占用率极低，与前面原生的 addSequential CUDA C++ 代码类似。

让我们优化这段 CUDA 核函数与 PyTorch 代码，实现向量加法操作的并行版本。图 6-19 展示了向量化加法操作的工作方式。

![图 6-19. 向量化加法在向量各元素间并行进行](../images/figure-6-19.png)

在下面的 CUDA C++ 代码中，我们启动足够多的线程以覆盖所有元素（<<< (N+255)/256, 256 >>>），使得每块 256 个线程在所需数量的块上并行处理 *N* 个元素：

```
#include <cuda_runtime.h>

const int N = 1'000'000;

// One thread per element
__global__ void addParallel(const float* __restrict__ A,
                            const float* __restrict__ B,
                                  float* __restrict__ C,
                            int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }

}

int main()
{
  // Allocate and initialize host (pinned for faster DMA)
  float* h_A = nullptr;
  float* h_B = nullptr;
  float* h_C = nullptr;
  cudaMallocHost(&h_A, N * sizeof(float));
  cudaMallocHost(&h_B, N * sizeof(float));
  cudaMallocHost(&h_C, N * sizeof(float));
  for (int i = 0; i < N; ++i) { h_A[i] = float(i); h_B[i] = float(2*i); }

  // Create a non-blocking stream and allocate device buffers
  cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  cudaMallocAsync(&d_A, N * sizeof(float), s);
  cudaMallocAsync(&d_B, N * sizeof(float), s);
  cudaMallocAsync(&d_C, N * sizeof(float), s);

  // Async HtoD copies on the same stream
  cudaMemcpyAsync(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_B, h_B, N*sizeof(float), cudaMemcpyHostToDevice, s);

  // Launch (same stream)
  int threads = 256;
  int blocks  = (N + threads - 1) / threads;
  addParallel<<<blocks, threads, 0, s>>>(d_A, d_B, d_C, N);

  // Async DtoH copy and stream sync
  cudaMemcpyAsync(h_C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);

  // Cleanup (stream-ordered free)
  cudaFreeAsync(d_A, s); cudaFreeAsync(d_B, s); cudaFreeAsync(d_C, s);
  cudaStreamDestroy(s);
  cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);
  return 0;
}
```

当 *N* 足够大时，GPU 利用率的差异非常显著。现在让我们优化这段 PyTorch 代码——它启动一个单一的向量化核函数（A + B），像前面优化过的 addParallel CUDA C++ 示例那样，在 GPU 上并发调动许多线程。下面是 PyTorch 代码的并行版本：

```
# add_parallel.py
import torch

N = 1_000_000
A = torch.arange(N, dtype=torch.float32, device='cuda')
B = 2 * A

torch.cuda.synchronize()

# Proper parallel approach using vectorized operation
# Launches a single GPU kernel that adds all elements in parallel
C = A + B

torch.cuda.synchronize()
```

> 在实践中，当你使用向量化的张量操作时，PyTorch 这类高级框架会做出正确的处理。只需注意：在 GPU 操作外围引入 Python 层的循环会使工作串行化，并对性能产生负面影响。如有可能应加以避免。除非你在编写某种新颖的东西，否则几乎总能找到一个经过优化的 PyTorch 原生实现——包括 PyTorch 编译器生成的代码。

为了量化使用并行实现相较串行实现的性能影响，我们可以用 Nsight Systems 和 Nsight Compute 来测量两种方法的核函数总执行时间、GPU 利用率、占用率以及 warp 执行效率指标。下面是 Nsight Systems（nsys）和 Nsight Compute（ncu）命令：

```
# Sequential add
nsys profile \
  --stats=true \
  -t cuda,nvtx \
  -o sequential_nsys_report \
  ./add_sequential.py

ncu \
 --section SpeedOfLight \
 --metrics
     sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.avg \
 --target-processes all \
 --print-summary per-gpu \
 -o sequential_ncu_report \
 ./add_sequential.py

# Parallel add
nsys profile \
  --stats=true \
  -t cuda,nvtx \
  -o parallel_nsys_report \
  ./add_parallel.py

ncu \
 --section SpeedOfLight \
 --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
 --target-processes all \
 --print-summary per-gpu \

 -o parallel_ncu_report \
 ./add_parallel.py
```

我们用 nsys 来揭示时间花在了哪里，以及 GPU 是被“饿着”还是被阻塞。然后用 ncu 来解释核函数为何呈现这样的表现——也许是由于占用率低，等等。

如果你只运行 nsys，可能会错过细粒度的核函数低效之处。而如果你只运行 ncu，则无从得知你的核函数是否被足够快地喂入了数据。表 6-6 展示了统一的结果。

*表 6-6. 串行与并行 CUDA 核函数的对比*

| 指标 | add_sequential | add_parallel |
| --- | --- | --- |
| 核函数执行时间（ms） | 48.21 | 2.17 |
| GPU 利用率 | 1.5% | 95% |
| 实际占用率 | 1.3% | 38.7% |
| warp 执行效率 | 3.1% | 100% |

> 其他剖析工具可能对这些指标使用不同的标签。例如，Nsight Systems 报告的是整体的“GPU Utilization”，而 Nsight Compute 提供的是每核函数的“SM Active %”指标——但两者都反映了 GPU 的 SM 被活跃 warp 占据的充分程度。

正如预期，从单线程、单 warp 的实现转向完全并行、多 warp 的实现，将平均占用率从 1.3% 提升到了 ~38.7%。这将运行时间缩短了约 22×，从 48.21 ms 降至 2.17 ms。

在串行的情形下，单个 SM 上只有一个 warp、且仅有一个线程在工作。这正是我们看到 GPU 利用率低至 1.5% 的原因；而在并行的情形下，众多 SM 都在运行多个活跃 warp。这使得 warp 执行效率从 3.1% 提升到 100%，因为在每条指令执行期间，warp 中全部 32 个线程都在做有用的工作。它把 GPU 利用率从 1.5% 提高到了 95%。

这个例子说明了为什么充足的并行度对 GPU 至关重要。无论每个线程有多快，你都需要大量线程才能发挥出 GPU 的吞吐潜力。

请记住，GPU 是一种面向吞吐优化的处理器，它既要与 CPU 交互以启动 CUDA 核函数，也要与内存子系统交互以从缓存、共享内存和全局内存中加载数据。因此，隐藏这些延迟能极大地提升 GPU 性能。

当核函数编写得当时，它会指示 GPU 把来自不同 warp 的内存加载与计算（例如加法）并行地交错执行。这有助于跨 warp 隐藏内存延迟。

尤其是在多个 warp 中运行的并行核函数，能从 warp 级延迟隐藏中获益。当一个 warp 正在等待内存加载时，另一个 warp 可以执行加法计算，而再有一个 warp 可以获取下一批数据，如此往复。我们将在后续章节探讨许多隐藏内存延迟的技术。

在串行核函数中，当一个 warp 在等待时，并没有其他 warp 可运行，因此硬件流水线常常闲置。其时间线是一长串操作，其间在等待内存时留有空闲间隙。而在并行版本中，这些间隙被其他 warp 的工作填满，于是 GPU 持续保持忙碌。二者的对比如图 6-20 所示。

![图 6-20. 并行与串行时间线对比](../images/figure-6-20.png)

在这里，串行时间线是一长串操作，其间在等待内存时留有空闲间隙。而在并行版本中，这些间隙被其他 warp 的工作填满，于是 GPU 持续保持忙碌。

关键的一点是，首先要确保有足够的并行工作来充分占用 GPU。高占用率——即有足够多的 warp 来掩盖延迟——能最大化吞吐并最小化空闲停顿。在我们的例子中，并行化把 GPU 利用率提升到了 ~95%。

一旦启动了足够的线程，下一步就是通过指令级并行以及其他针对每线程的改进，来优化每个 warp 的执行效率。但要注意：即使在 100% 占用率下，如果工作负载是访存受限的——也就是受限于缓慢的内存访问而非计算——性能仍可能受损。

一个广为人知的访存受限工作负载的例子，是 LLM 的“解码”（decode）阶段。在解码期间，LLM 需要把大量数据（模型权重或参数）从全局 HBM 内存搬运到 GPU 寄存器和共享内存中。

由于现代 LLM 包含数千亿个参数（再乘以，比如说，每个参数 8 比特即 1 字节），这些模型的体积可达数百 GB。将如此海量的数据搬入搬出 GPU，很容易就会把内存带宽占满。

> GPU 的 FLOPS 正在超越内存带宽的增长。例如，Blackwell 的 HBM3e 提供 ~8 TB/s，但算力和模型规模增长得更快。因此，优化内存搬运对于避免现代 AI 工作负载中的访存受限瓶颈绝对至关重要。

### 用启动边界调优占用率

在某些情况下，仅仅使用更多线程并不足够——尤其是当每个线程使用了大量资源（如寄存器和共享内存）时。我们可以通过 CUDA 的 __launch_bounds__ 核函数注解，引导编译器针对占用率进行优化。

这个注解让我们在编译期为核函数指定两个参数：我们将启动的每个线程块的最大线程数，以及我们希望在每个 SM 上保持常驻的最小线程块数。这些提示会影响编译器的寄存器分配和内联决策。示例如下：

```
__global__ __launch_bounds__(256, 16)
void myKernel(...) { /* ... */ }
```

在这里，__launch_bounds__(256, 16) 承诺该 CUDA 核函数启动时，每个线程块的线程数绝不会超过 256。它还请求编译器分配足够的寄存器并内联函数，使得至少 16 个每块 256 线程的线程块（即 4,096 个线程，16 blocks × 256 threads per block）能够同时常驻于该 SM 上。

> 请记住，在现代 GPU（例如 Blackwell）上，每个线程块最多只能有 1,024 个线程，每个 SM 最多只能有 2,048 个常驻线程。

在实践中，由于当前的 NVIDIA GPU 将每个 SM 限制为总共 2,048 个线程、每个线程块限制为 1,024 个线程，编译器会把你的请求削减到硬件上限——在本例中即每个 SM 2,048 个线程（8 blocks × 256 threads per block）。并且它会发出一条警告，因为 4,096 的请求（16 blocks × 256 threads per block）超出了该 SM 的容量。

> 警告内容类似于 “ptxas warning: Value of threads per SM…is out of range. .minnctapersm will be ignored.”

在实践中，使用 __launch_bounds__ 常常会促使编译器限制每线程的寄存器用量（有时还会限制展开或内联），以避免溢出并允许更高的占用率。我们本质上是在牺牲一点每线程性能，不再用尽最后一个寄存器、也不把循环展开到极致。换来的是通过让更多 warp 保持在飞状态，从而获得更稳定的 warp 吞吐。

*提高占用率必须与每线程资源相权衡。* 你要避免寄存器溢出（register spilling）——当你强行塞入过多线程、导致它们耗尽寄存器并溢出到本地内存时，就会发生溢出，从而引发缓慢的内存访问。

你也可以在运行时使用 CUDA 占用率 API 来确定最优的启动配置。例如，cudaOccupancyMaxPotentialBlockSize() 会在考虑给定核函数的寄存器和共享内存用量后，计算出能产生最高占用率的线程块大小。本质上，cudaOccupancyMaxPotentialBlockSize 可以为你自动调优线程块大小以获得最优占用率，如下所示：

```
int minGridSize = 0, bestBlockSize = 0;

// If your kernel uses dynamic shared memory (extern __shared__),
// set this correctly:
size_t dynSmemBytes = /* bytes per block (e.g., tiles * sizeof(T)) */ 0;

cudaOccupancyMaxPotentialBlockSize(
  &minGridSize, &bestBlockSize,
  myKernel,
  dynSmemBytes,      // must match your kernel's dynamic shared memory use
  /* blockSizeLimit = */ 0);

// Compute a grid that covers N, but don't go below the min grid
// that saturates occupancy
int gridSize = std::max(minGridSize, (N + bestBlockSize - 1) / bestBlockSize);

myKernel<<<gridSize, bestBlockSize, dynSmemBytes>>>(...);
```

这个 API 会计算：在给定核函数资源用量的情况下，每个线程块用多少线程可能优化占用率。随后我们就可以在核函数启动时使用 bestBlockSize（以及建议的网格大小）。需要重点指出的是，minGridSize 是在该设备上让该核函数占用率饱和的最小网格大小，它未必就是覆盖长度为 N 的输入所需的正确网格大小。请计算 gridSize = max(minGridSize, ceil_div(N, bestBlockSize))，并且如果核函数使用了 extern __shared__，就传入它实际的动态共享内存字节数。

> 通过在 ±1–2 个候选线程块大小上对核函数计时，来验证占用率 API 的建议。在现代 GPU 上，寄存器压力和 L2 行为实际上可能让一个略低于最大占用率的配置在实践中反而更快。

采用之后，编译器的启发式判断通常是不错的，但 __launch_bounds__ 和占用率计算器能在需要时给你显式的控制权。当你*知道*自己的核函数可以用一些每线程资源换取更多活跃 warp 时，就使用它们。这有助于防止因线程过“重”而导致 SM 占用不足。

> 寄存器与占用率之间的取舍很重要。每线程使用更少的寄存器——或用启动边界给它们设上限——可以让更多 warp 常驻，从而改善延迟隐藏。然而，寄存器用得太少会迫使编译器把数据溢出到本地内存，损害性能。找到那个最佳平衡点往往需要实验。Nsight Compute 的 “Registers Per Thread” 和 “Occupancy” 指标可以在这里为你提供指引。

## 用 NVIDIA Compute Sanitizer 调试功能正确性

由于 CUDA 应用每个核函数可能派生出成千上万个线程，传统的调试方法可能无法捕捉到隐蔽的内存缺陷和竞态条件。NVIDIA Compute Sanitizer 是随 CUDA Toolkit 一同提供的功能正确性套件，它通过在运行时对代码进行插桩，在开发早期就发现错误，从而应对这些挑战。这减少了调试的往复次数，并提升了整体代码可靠性。

Sanitizer 通过 compute-sanitizer 命令行工具调用，并支持 NVIDIA Tools Extension（NVTX）注解以进行更细粒度的分析。NVTX 应当被广泛用于正确性分析和性能分析。使用该命令行工具时，你可以用 --option value 指定选项，并加入诸如 --error-exitcode 的标志以在出错时使程序失败。你还可以施加过滤器，只对特定核函数进行检查，例如使用 --kernel-name 和 --kernel-name-exclude。你可以用 --nvtx yes 启用 NVTX，以帮助收窄分析范围，例如在内存泄漏报告中最小化误报：

```
compute-sanitizer [--tool toolname] [options] <application> [app_args]
```

> 建议借助 --error-exitcode，并配合核函数过滤器和 NVTX 区域注解，将 Compute Sanitizer 集成到你的持续集成（continuous integration，CI）流水线中，以捕捉正确性回归。

Compute Sanitizer 由四个主要工具组成：memcheck、racecheck、initcheck 和 synccheck。它们有助于检测你的 CUDA 代码中的越界内存访问、数据竞争、未初始化内存读取以及同步问题：

*Memcheck*

memcheck 工具能精确检测并归因于全局、本地和共享内存中的越界或未对齐访问；报告 GPU 硬件异常；还能识别设备侧的内存泄漏。它通过命令行开关支持额外的检查，例如用于堆分配的 --check-device-heap。

*Racecheck*

Racecheck 报告共享内存的数据冒险，包括写后写（Write-After-Write）、读后写（Write-After-Read）和写后读（Read-After-Write），这些都可能导致非确定性行为。Racecheck 帮助开发者验证 warp 和线程块内部线程间通信的正确性。

*Initcheck*

Initcheck 标记出对未初始化的设备全局内存的任何访问。这可能源于缺失的主机到设备拷贝，或被跳过的设备侧写入。该工具有助于避免由陈旧或垃圾数据引发的隐蔽缺陷。

*Synccheck*

Synccheck 检测同步原语的无效使用，例如不匹配的屏障。它识别出可能导致死锁以及跨线程状态不一致的线程顺序冒险。

简而言之，NVIDIA Compute Sanitizer 提供了一套工具，用于发现并解决 CUDA 应用中的内存、竞态、初始化和同步缺陷。当这些工具与 CI 系统集成时，可以帮助开发者及早发现正确性问题。如此一来，他们便能满怀信心地交付可靠、高性能的代码。

## Roofline 模型：计算受限还是访存受限的工作负载

Roofline 模型是一种有用的可视化方法，它绘出两条由硬件强加的性能上限：一条是处理器峰值浮点速率对应的水平线，另一条是峰值内存带宽决定的斜线。二者共同构成一个“屋顶线”（roofline）包络，揭示某个给定核函数究竟是受限于计算（计算受限，compute bound）还是受限于数据搬运（访存受限，memory bound）。

这两条线相交之处被称为*脊点*（ridge point）。它对应于“算术强度”（arithmetic intensity）的阈值，核函数在此阈值处从访存受限（脊点左侧）转变为计算受限（脊点右侧）。算术强度的度量方式是：在片外全局内存与 GPU 之间每传输一个字节所执行的 FLOPS 数。

我们来看一个简单的例子，以说明为什么算术强度很重要。假设一个核函数加载两个 32 位浮点数（共 8 字节），将它们相加（1 FLOP），然后写回一个 32 位浮点结果（4 字节）。在这种情况下，该算法为 12 字节的内存流量执行了 1 FLOP，得出的算术强度为 0.083 FLOPs/byte（1 FLOP/12 bytes ≈ 0.083 FLOPs per byte）。

将其与某 GPU 的脊点 10 FLOPs per byte（10 FLOPs = ~80 TFLOPs ÷ 8 TB/s）相比。这个浮点加法核函数的 0.083 脊点位于 Roofline 的最左侧（访存受限一侧），比阈值低了几个数量级。它比该阈值还低 100× 以上，因此无法让算术逻辑单元（ALU）保持忙碌。这个核函数处于访存受限区间，其性能由内存停顿而非计算所主导。图 6-21 展示了一个具有代表性的 Blackwell Roofline 模型，包括峰值算力（位于 ~80 FLOPs/sec 的水平线）和峰值内存带宽（对应 8 TB/s 的斜线）。

在这里，我们看到 Blackwell GPU 的脊点等于持续 FLOPs/sec 除以持续 HBM 带宽。这里它就是图中标在 10 FLOPs/byte 处的交点。我们示例核函数的算术强度为 0.083 FLOPs/byte，落在那条倾斜的内存带宽斜线的左侧。因此，这个核函数落在 Roofline 那条倾斜的内存带宽上限之上。这证实了它是访存受限的。

要让这个核函数不那么访存受限（从而更偏向计算受限），你可以通过让每字节数据完成更多工作来提高它的算术强度。这会把该核函数向右移动，从而把性能向上推向计算屋顶线。

![图 6-21. 某 Blackwell 级 GPU（~80 TFLOPs/sec FP32、~8 TB/s HBM3e）的 Roofline 模型，展示了我们核函数所处的点以及 ~10 FLOPs/byte 的算术强度脊点](../images/figure-6-21.png)

让核函数不那么访存受限的一个简单办法，是使用更低精度的数据。例如，如果你用 16 位浮点数（FP16）代替 32 位（FP32），你就能把每次运算传输的字节数减半，并立刻把 FLOPs/byte 强度翻倍。

现代 GPU 还支持专用的 8 位浮点（FP8）Tensor Core。Blackwell 还为某些 AI 工作负载引入了对 4 位浮点（FP4）Tensor Core 的原生支持。这些都进一步减少了每次运算的字节数，把 FLOPs/byte 强度提升得更多。

例如，Blackwell 支持 FP8 Tensor Core（每个值 1 字节），相较 FP16 使吞吐翻倍、内存占用减半。它还为诸如模型推理之类的部分工作负载支持 FP4（每个值半字节）。

单次 128 字节的内存事务可以承载 32 个 FP32、64 个 FP16、128 个 FP8 或 256 个 FP4 值。Blackwell 引入了硬件解压来加速压缩后的模型权重。例如，模型可以以压缩形式存储在 HBM 中（甚至压缩到超过 FP4 的程度），而硬件可以在读取时即时解压这些权重。这在读取这些权重时进一步有效提高了可用的内存带宽。

因此，对于像基于 transformer 的 token 生成这类访存受限的工作负载，Blackwell 具有架构上的优势。权重以压缩的 4 位或 2 位方案存储，在加载时由硬件解压，并转换为 FP16/FP32 以进行更高精度的聚合与计算。这展示了更低精度如何能减少传输的数据量、提高核函数的算术强度，并改善你工作负载的整体内存吞吐。

对于访存受限的工作负载，目标是把核函数的运算点在 Roofline 上向右推，以提高其算术强度，并向计算受限靠拢。通过更接近计算受限区间，你的核函数就能更好地利用 GPU 全部的浮点马力。

> 基于 transformer 的模型（例如 LLM）在不同阶段既可能是计算受限的，也可能是访存受限的。例如，注意力层（预填充阶段）通常是计算受限的，而矩阵乘法（解码阶段）往往是访存受限的。我们将在第 15–18 章深入探讨推理时对此展开更多讨论。

当核函数是访存受限时，Nsight Compute 会报告非常高的 DRAM 带宽利用率，同时报告较低的实际算力指标，例如较低的 ALU 利用率。这表明 warp 把大部分时间都花在了因内存访问而停顿上。

要深入了解正在发生什么，最好使用 Nsight Compute 来获取每核函数的计数器，包括延迟、缓存命中率和 warp 发射停顿。此外，现代版本的 Nsight Compute 具备范围重放（range replay，带指令级源码指标）、改进的源码关联导航以及启动栈大小指标。这些特性有助于更快地诊断依赖停顿、寄存器压力和启动配置的影响。

随后你可以用 Nsight Systems 获取一个整体的时间线视图，展示 GPU 空闲间隙、与 CPU 工作的重叠，以及 PCIe/NVLink 传输。二者结合能同时给你*为何*（是哪些停顿、哪些资源）和*何时*（这些停顿如何嵌入你应用的整体执行）。

关键在于借助 Nsight Compute 和 Nsight Systems 两者的指标，迭代地进行剖析并识别内存热点。你应当在可疑代码周围添加 NVTX 区域，放大观察时间线行为，并利用反馈进行优化。

例如，你可以用 NVTX 把区域标注为“内存拷贝”或“核函数执行”，并在 Nsight Systems 时间线上查看它们。如前所述，这对于确认主机-设备传输与计算的重叠极其有用。

例如，为了验证重叠，你可以用 NVTX 标记来标注数据传输和核函数调用两者的起止。Nsight Systems 会在时间线上显示这些 NVTX 区域，让重叠一目了然。借助异步内存拷贝（cudaMemcpyAsync），数据传输会与 GPU 上的核函数执行相重叠（见图 6-22），该图对比了同步和异步的内存传输。

![图 6-22. 同步（串行）与异步（重叠）数据传输及核函数计算](../images/figure-6-22.png)

如果你期望出现重叠，却看到拷贝和核函数是串行而非并行运行，那么问题多半类似于一次意外的默认流同步。否则，很可能是缺失了固定内存缓冲区，从而阻止了真正的重叠。

> 如果不使用固定（页锁定）内存，cudaMemcpyAsync 传输就无法与核函数执行相重叠。这是一个常见的性能问题。

当你怀疑某个核函数因数据不足而“饥饿”时，先在 Nsight Compute 和 Nsight Systems 下运行它。在 Nsight Compute 中你会看到全局加载效率（global load efficiency）指标下降。这标志着你的 DRAM 请求没有被足够快地满足。与此同时，Nsight Systems 时间线会揭示出核函数启动之间、GPU 等待数据传输时的空闲区段。

一旦你应用了本章的内存层级优化，那些空闲间隙将几乎全部消失，而 Nsight Compute 会显示内存管线利用率百分比向峰值攀升。你还会看到端到端核函数吞吐相应地跃升。

> 每次改动之后都要测量。剖析工具会确认某项优化究竟是否真的减少了内存停顿。

## 关键要点

在本章中，你学习了如何选择能优化占用率的启动参数、如何异步管理 GPU 内存，以及如何应用 Roofline 分析来区分计算受限与访存受限的核函数。以下是一些值得回顾的关键要点：

*SIMT 执行模型*

GPU 在单指令多线程（SIMT）模型下以 warp（32 线程）为单位执行线程，每个 warp 以锁步方式发射指令。高占用率——即保持大量 warp 处于在飞状态——能隐藏内存和流水线延迟。

*线程层级：线程 → 线程块 → 网格*

线程被分组为线程块（最多 1,024 个线程），而线程块又组成网格，从而无需修改代码即可扩展到数百万个线程。同步（__syncthreads() 或 cooperative groups）能在共享内存中实现数据复用，但会带来开销，因此要尽量减少屏障。

*占用率与资源上限*

将线程块大小选为 32 的倍数，以避免 warp 填不满并最大化调度器利用率。要留意每 SM 的上限。对于 Blackwell，每线程最大寄存器数为 255，每 SM 最大共享内存为 228 KB，最大常驻 warp 数为 64，最大常驻线程块数为 32。

*CUDA 核函数启动参数*

从 threadsPerBlock = 256（8 个 warp）开始，以兼顾占用率与资源使用；计算 blocksPerGrid = (N + threadsPerBlock – 1) / threadsPerBlock 以覆盖所有元素。根据剖析反馈（寄存器/寄存器溢出、共享内存用量、实际占用率）来调优这些数值。

*异步内存管理*

优先在专用流上使用 cudaMallocAsync/cudaFreeAsync，并利用 CUDA 内存池来避免全局同步和操作系统层面的开销。PyTorch 的缓存分配器遵循类似的模式来高效分配张量，并避免代价高昂的 cudaMalloc() 和 cudaFree() 调用。

*GPU 内存层级*

Registers → L1/shared → L2 → global (HBM3e) → host：每一层都以容量换取延迟/带宽。要在寄存器以及共享内存/L1 缓存中最大化数据复用。

*统一内存的考量*

CUDA 托管内存（统一内存）简化了编程，但可能引发隐式的页迁移；使用 cudaMemPrefetchAsync 和内存建议来避免意外的停顿。

*Roofline 模型分析*

算术强度（每字节的 FLOPS）决定了一个核函数是访存受限还是计算受限。使用更低精度（FP16/FP8/FP4 以及硬件解压）来提升 FLOPS/byte 比值，把核函数推向计算屋顶线。用 Nsight Compute（每核函数指标）和 Nsight Systems（时间线）进行剖析，以识别并消除内存停顿。在 Blackwell 上，将 TMEM 与统一矩阵乘累加（unified matrix-multiply-accumulate，UMMA）结合，再配合 FP8 和 FP4，可以把核函数从访存受限转变为计算受限。我们将在第 10 章更详细地介绍 UMMA。

## 结语

本章通过揭开 GPU 的 SIMT 模型、线程层级和多级内存系统的神秘面纱，为高性能 CUDA 开发打下了基础。请记住，占用率——即活跃 warp 数与理论 GPU 最大值之比——对于延迟隐藏很重要。

然而，最大化占用率并不能在每种情况下都保证最佳性能。如果线程具备充足的指令级并行（ILP）——或者如果其他资源才是瓶颈——GPU 往往能在中等甚至较低的占用率下实现非常高的吞吐。

虽然更高的占用率有助于隐藏延迟，但在某些场景下，减少活跃线程数反而能为其他线程腾出寄存器。这允许每个线程执行更多计算——并最终提升吞吐。请务必对不同的占用率水平进行基准测试，以便为你的工作负载和硬件找到最优设置。

有了这些基础知识和剖析技术，你现在已经准备好深入研究有针对性的优化了，例如避免 warp 分歧、利用 GPU 内存层级，以及异步预取内存。我们还将深入探讨 TMA，它负责处理批量内存传输，从而解放 GPU，让它专注于有用的工作并提高计算有效吞吐（goodput）。
