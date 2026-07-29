# 第 10 章 核内流水线化、warp 专门化与协作式线程块簇

在前几章中，我们介绍了一些基础优化，例如调优内存访问、最大化并行度、重叠计算与数据传输、提升占用率以及最小化 warp 停顿。这些手段帮助我们隐藏延迟并消除瓶颈。然而，现代 GPU 提供了先进的硬件特性与执行模型，让我们能够把这些基础优化技术推进得更远。

在本章中，我们将介绍一些更高级的 CUDA 技术，例如 warp 专门化的流水线、带网格级与簇级同步的协作组（cooperative groups）、在动态工作队列上循环的持久化核函数（persistent kernel），以及使用分布式共享内存（distributed shared memory，DSMEM 或 DSM）和 Tensor Memory Accelerator（TMA）多播（multicast）的线程块簇（thread block cluster，又称 *cooperative thread array cluster* [CTA]）。从宏观上看，线程块簇是一组保证并发运行的线程块。它们可以借助 DSMEM 对彼此的共享内存进行读、写和原子操作。

这些方法让我们无需主机干预即可重叠内存访问与计算操作。我们还能在片上（on-chip）跨线程块共享数据——并保持每个 SM 都得到充分利用。

理解了这些现代 GPU 执行模型后，你就为进入下一章做好了准备：在下一章中，我们将通过探索基于 CUDA streams 的核间流水线（inter-kernel pipeline）把这些优化进一步延伸。下一章会在本章通篇讨论的核内优化基础之上，构建核间流水线。

## 核内流水线化技术

*核内流水线化*（intra-kernel pipelining）指的是一组在单次核函数执行内部重叠内存操作与计算的技术。（在下一章，我们将探讨核间流水线化（inter-kernel pipelining），它在不同 streams 中运行的多个核函数之间重叠工作。）

其核心思想是把核函数组织为若干并发阶段，使得在加载或存储某份数据的同时，先前加载的数据正在被处理。这些阶段在不同的分块（tile）或数据块上并行运作。这样既提升了吞吐，又高效地隐藏了延迟。

传统上，GPU 依赖 warp 级多线程来隐藏延迟。当一个 warp 因内存加载而停顿时，其他 warp 继续执行计算。这正是执行模型中单指令多线程（SIMT）延迟隐藏的基础。

核内流水线化在此基础上更进一步，在同一 warp 或核函数内部重叠内存与计算。它借助细粒度协调来错开内存加载与计算——有时就在单个 warp 之内完成。

使用 CUDA Pipeline API 的核内流水线化能在完全不使用 __syncthreads() 的情况下重叠异步内存传输与计算。核内流水线化有两种常见方式：双缓冲（double buffering）与 warp 专门化（warp specialization）。

在双缓冲（两阶段）流水线方式中，所有线程以统一方式协作。在 warp 专门化流水线方式中，warp 被专门化为不同角色，如内存加载器、计算和内存存储器。选择哪种方式取决于你的工作负载与性能需求。表 10-1 总结了这两种 <cuda/pipeline> 变体。

表 10-1. 使用 CUDA Pipeline API 在现代 GPU 上进行核内流水线化的两种方式

| API 方案 | 最适用于 | 主要用途 |
| --- | --- | --- |
| Double-buffered pipeline | 基于循环的分块与双缓冲 | 在同一 warp 或块内重叠加载与计算 |
| Warp-specialized pipeline（例如三阶段：memory loader、compute、memory storer） | 具有多种不同 warp 角色（本例为 3 种）的持久化核函数 | 将 warp 分派到独立的角色/阶段，如内存加载、计算和内存存储 |

### 用 CUDA Pipeline API 做协作式分块与双缓冲

你可以使用 C++ Pipeline API 实现传统的双缓冲分块模式：实例化一个两阶段流水线来重叠内存加载与计算。具体做法是声明一个两阶段的 cuda::pipeline_shared_state <cuda::thread_scope_block, 2> 对象，它借助协作组（稍后讨论）被限定到某个特定的线程块。这本质上是一种生产者—消费者（producer-consumer）模式，如图 10-1 所示。

![图 10-1. 使用 CUDA Pipeline API 的两阶段生产者—消费者模式](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-1.png)

下面给出关键的 CUDA Pipeline API 调用。随后是一个使用该 API 实现的双缓冲、协作式分块核函数，用以演示与硬件特性契合的现代 CUDA 技术：

pipe.producer_acquire() 预留下一个流水线阶段以供写入

pipe.producer_commit() 表明本阶段先前发起的异步操作已就绪、可供消费

pipe.consumer_wait() 等待本阶段先前提交的操作完成，以避免循环中的竞态条件

pipe.consumer_release() 释放当前阶段以便复用

流水线中的两个阶段将全局内存加载与计算重叠起来。在第一个阶段 Stage 0 中，线程块里的一个 warp 为下一个分块发起异步预取（prefetch）到共享内存。该预取发起协作式 cuda::memcpy_async 拷贝，这些拷贝会下沉为逐线程的 cp.async 到共享内存。

当 Stage 0 在一个 warp 中生产（加载）数据时，线程块中其余的 warp 正在第二个阶段 1 中消费（计算）已加载的数据。这个简单的生产者—消费者实现用持续进行的计算隐藏了 DRAM 延迟，如图 10-2 所示。

![图 10-2. 使用生产者—消费者流水线以 (C) 计算隐藏全局 DRAM 加载 (L) 延迟](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-2.png)

这种模式可以提升 SM 利用率并缩短核函数时间，但你究竟是计算受限（compute bound）还是访存受限（memory bound），取决于具体运算、分块大小以及重叠效率。即便是像 FlashAttention-3 这样高度优化的注意力核函数，由于重叠与数据搬运方面的实际限制，也只报告约 75% 的峰值 FP16 FLOPs。

下面是一个使用 CUDA C++ Pipeline API 的两阶段双缓冲示例。该 API 启用了这里所用的细粒度生产者—消费者同步代码：

```
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <algorithm>
namespace cg = cooperative_groups;
#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif
#ifndef STAGES
#define STAGES 2          // 2 = double buffer, 3 = triple buffer, etc.
#endif
__device__  float computeTile(const float* __restrict__ A_sub,
                              const float* __restrict__ B_sub,
                              int tx, int ty) {
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        s += A_sub[ty * TILE_SIZE + k] * B_sub[k * TILE_SIZE + tx];
    }
    return s;
}
extern "C" __global__
void gemm_tiled_pipeline(const float* __restrict__ A_global, // [M x K]
                         const float* __restrict__ B_global, // [K x N]
                         float* __restrict__ C_global,       // [M x N]
                         int M, int N, int K) {
    cg::thread_block cta = cg::this_thread_block();
    // Shared memory layout: A[STAGES] then B[STAGES]
    extern __shared__ float shared_mem[];
    float* A_buf[STAGES];
    float* B_buf[STAGES];
    {
        float* p = shared_mem;
        for (int s = 0; s < STAGES; ++s) {
            A_buf[s] = p;
            p += TILE_SIZE * TILE_SIZE;
       }
        for (int s = 0; s < STAGES; ++s) {
            B_buf[s] = p;
            p += TILE_SIZE * TILE_SIZE;
       }
    }
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> state;
    auto pipe = cuda::make_pipeline(cta, &state);
    int tx = threadIdx.x, ty = threadIdx.y;
    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;
    float accum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    // Prologue: load first STAGES tiles (or fewer if short)
    for (int s = 0; s < std::min(STAGES, numTiles); ++s) {
        int aRow = block_row + ty;
        int aCol = s * TILE_SIZE + tx;
        int bRow = s * TILE_SIZE + ty;
        int bCol = block_col + tx;
        pipe.producer_acquire();
        if (aRow < M && aCol < K) {
            cuda::memcpy_async(cta, A_buf[s] + ty*TILE_SIZE + tx,
                               &A_global[aRow*K + aCol],
                               cuda::aligned_size_t<32>(sizeof(float)), pipe);
        } else {
            A_buf[s][ty*TILE_SIZE + tx] = 0.0f;
        }
        if (bRow < K && bCol < N) {
            cuda::memcpy_async(cta, B_buf[s] + ty*TILE_SIZE + tx,
                               &B_global[bRow*N + bCol],
                               cuda::aligned_size_t<32>(sizeof(float)), pipe);
        } else {
            B_buf[s][ty*TILE_SIZE + tx] = 0.0f;
        }
        pipe.producer_commit();
    }
    // Steady state
    for (int tile = 0; tile < numTiles; ++tile) {
        int s = tile % STAGES;
        // Block-scope wait
        pipe.consumer_wait();
        accum += computeTile(A_buf[s], B_buf[s], tx, ty);
        pipe.consumer_release();
        // Prefetch next tile into the same slot s (ring buffer)
        int nextTile = tile + STAGES;
        if (nextTile < numTiles) {
            int aRow = block_row + ty;
            int aCol = nextTile * TILE_SIZE + tx;
            int bRow = nextTile * TILE_SIZE + ty;
            int bCol = block_col + tx;
            pipe.producer_acquire();
            if (aRow < M && aCol < K) {
                cuda::memcpy_async(cta, A_buf[s] + ty*TILE_SIZE + tx,
                                   &A_global[aRow*K + aCol],
                                   cuda::aligned_size_t<32>{sizeof(float)}, pipe);
            } else {
                A_buf[s][ty*TILE_SIZE + tx] = 0.0f;
            }
            if (bRow < K && bCol < N) {
                cuda::memcpy_async(cta, B_buf[s] + ty*TILE_SIZE + tx,
                                   &B_global[bRow*N + bCol],
                                   cuda::aligned_size_t<32>{sizeof(float)}, pipe);
            } else {
                B_buf[s][ty*TILE_SIZE + tx] = 0.0f;
            }
            pipe.producer_commit();
        }
    }
    // Epilogue: final store (guard tails)
    int cRow = block_row + ty;
    int cCol = block_col + tx;
    if (cRow < M && cCol < N) {
        C_global[cRow * N + cCol] = accum;
    }
}
```

代码首先使用协作组（cooperative groups，CG，稍后讨论）获取当前线程块的句柄。随后立即实例化一个绑定到该块的两阶段 cuda::pipeline 对象。通过在任何异步操作之前创建流水线，流水线的内部屏障与内部同步机制就在数据搬运开始之前准备就绪了。

接着，核函数通过定义 extern __shared__ float shared_mem[] 为 A 和 B 的分块分配一整块连续的共享内存区域。它使用指针运算（float* A_buf[2] 与 float* B_buf[2]）把这块缓冲划分为四个子区域，两个给 A、两个给 B。这样便可在不额外进行动态分配的情况下实现真正的双缓冲。

在进入主循环之前，核函数使用绑定到以下流水线的异步拷贝，异步预取前 STAGES 个分块：producer_acquire() → memcpy_async() → producer_commit()。消费者 warp 使用 consumer_wait() 和 consumer_release()，从而在预取的分块就绪的那一刻恰好开始计算。

这一初始屏障替代了将来对 __syncthreads() 的需求，并确保流水线的 stage 0 与 stage 1 缓冲在生产者—消费者序列的第一次迭代时被正确填充。

在每次迭代内，核函数通过调用 pipe.producer_acquire() 预留下一个缓冲以加载后续分块。随后它发起两个 cuda::memcpy_async 操作——一个用于 A 分块，一个用于 B 分块。每次加载都通过 cuda::memcpy_async(..., pipe) 绑定到流水线对象，从而发起从全局内存到共享内存的异步拷贝。

当访问被合并且分块正确时，这些异步拷贝能与计算很好地重叠。如此一来，流水线始终有数据可用，得以执行最大量的有用工作。在排队 memcpy_async 调用之后，核函数立即用 pipe.producer_commit() 发出完成信号。该提交记录了拷贝的到达，并允许消费者 warp 等待那个特定阶段，而不必阻塞整个线程块。

与此同时，块中的其他 warp 调用 pipe.consumer_wait()。这会仅高效地停顿那些依赖当前缓冲数据的线程，直到生产者提交完成。等待完成后，每个线程调用设备函数 computeTile(...) 来执行其 TILE_SIZE x TILE_SIZE 的点积计算。结果在寄存器中被逐步累加。

在完成当前缓冲上的计算后，warp 调用 pipe.consumer_release() 以释放该阶段，供后续迭代复用。这种细粒度的释放避免了块级停顿，并最大化了计算与内存传输阶段之间的重叠。

在每次迭代的末尾，交换 curr 与 next，从而在两个双缓冲阶段之间循环。这样，刚刚完成计算的内存缓冲就成为下一次异步内存的目标。

这些生产者与消费者阶段会针对所有 K 维分块反复执行。在处理完最后一个分块后，每个线程的累加器中都包含了完整的 TILE_SIZE x TILE_SIZE 点积结果。随后该结果被写回到 C_global 的相应元素。

使用双缓冲时，建议确保每个异步拷贝（memcpy_async）之后都跟有恰当的 producer_commit() 与 consumer_wait() 调用以进行同步。这可以保证计算核函数只在数据被正确加载之后才使用它。对于简单的循环，现代 CUDA 编译器往往会自动执行这些优化。

> 建议使用 Nsight Compute 的异步拷贝指标来验证流水线是否按预期执行。尤其要关注 warp 占用率与共享内存 bank 冲突。目标是提升 SM Active 百分比并减少停顿周期。能否完全隐藏 DRAM 延迟取决于占用率、访问模式以及共享内存 bank 行为。

这个简单的双缓冲方案大约比朴素分块快 2×，如表 10-2 所示——该表对比了朴素分块实现与经过优化的双缓冲流水线实现。

表 10-2. 使用 CUDA Pipeline API 的朴素分块核函数与双缓冲核函数的性能对比

| 指标 | 朴素分块 | 两阶段双缓冲流水线（double_buffered_pipeline） |
| --- | --- | --- |
| 核函数执行时间 | 41.3 ms | 20.5 ms（相较 naive 约 2× 加速） |
| SM Active % | 68% | 92%（相较 naive +24%） |

> 注：所有指标表中的数值均为示意，用以解释概念。若需在不同 GPU 架构上的实际基准结果，请参见 GitHub 仓库。

在这次实验中，使用 CUDA C++ Pipeline API 的 gemm_tiled_pipeline 核函数相较朴素分块版本实现了 2× 加速。通过使用细粒度的 pipe.producer_commit() 与 pipe.consumer_wait() 原语，流水线始终保持充盈，SM Active % 从 68% 跃升 24% 至 92%。

### warp 专门化与生产者—消费者模型

warp 专门化通过把操作分派给使用不同硬件的 warp——例如数据搬运（如 TMA）与计算（如 Tensor Core）——来扩展双缓冲。这与让同一批 warp 既加载数据又做计算的做法形成对比，如图 10-3 所示。

![图 10-3. 非 warp 专门化的核函数，每个 warp 都混合执行数据加载与计算（来源：https://oreil.ly/WZDbM）](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-3.png)

这类专门化让每一组 warp 拥有各自的指令序列。因此，指令得以连续发射与执行，而不会被其他类型的操作打断。具体来说，warp 专门化让你可以指派一组“生产者”或“内存”warp 使用 cuda::memcpy_async 异步预取分块。然后所有其他“消费者”或“计算”warp 执行计算，如图 10-4 所示。

这里，四个 warp 被分派为生产者角色，其余八个 warp 被分派为消费者角色。与大多数生产者—消费者模式一样，你可以为生产者和消费者分派不同数量的 warp。

因为每个 warp 都有自己的调度器，GPU 可以在同一周期内，通过不同的 warp 调度器从不同的 warp 发射一条加载指令、一条数学指令和一条写指令。因此，一个具有多个调度器的 SM 可以在一个周期内从 Warp 0 发射一条内存指令、从 Warp 1 发射一条数学指令，依此类推。

![图 10-4. warp 专门化核函数，一组 warp 负责加载数据，其余所有 warp 负责计算（来源：https://oreil.ly/WZDbM）](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-4.png)

这实际上在跨 warp 的层面构成了一个线程块级的多发射场景。单 warp 双缓冲无法做到这一点，因为单个 warp 的调度器每周期只能发射一条指令。

warp 专门化的一个有趣模式是使用三种不同类型的 warp，如“loader”、“compute”和“storer” warp。loader warp 把分块推入流水线的队列。compute warp 在每个分块上运行计算核函数。而 storer warp 写出结果，如图 10-5 所示。

这种 warp 专门化流水线挤出了单 warp、双缓冲、顺序加载—计算循环无法消除的空闲周期。warp 专门化对数据传输与计算的高效重叠提升了 GPU 利用率——尤其对于长时间运行的循环与持久化核函数。在这些情形下，角色协调与数据交接的开销会被摊薄到许多次迭代之中。

一篇关于 warp 调度的论文证明，warp 专门化可以实现内存与计算近乎完美的重叠。在该案例中，GPU 核函数具有相互分离的内存阶段与计算阶段，以至于内存和计算轮流成为瓶颈。通过应用 warp 专门化，其工作负载被转变为这样一种状态：SM 的内存子系统与计算单元几乎在整段时间内都同时保持忙碌。

![图 10-5. 三角色 warp 专门化流水线配置，一组 warp 负责加载数据、另一组负责计算、再一组负责数据存储（来源：https://oreil.ly/xs7YN）](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-5.png)

性能分析显示，在使用 warp 专门化之前，他们的 L2 带宽利用率与 Tensor Core 利用率是不同相位的。应用 warp 专门化之后，L2 带宽与 Tensor Core 利用率变为同相位。这带来了高得多的有效吞吐，并表明 warp 专门化能挤出最后那点空闲时间——即便对于一条调优良好、但仍可能有余量未被利用的异步流水线也是如此。

另一种 warp 专门化模式是对三角色 warp 专门化流水线的一种改造。它像之前一样分派一组 warp 作为内存加载器，但随后使用两组消费者 warp，在计算和内存存储两种角色之间“乒乓”（ping-pong）切换。这种三角色 warp 专门化架构在 CUTLASS 中以 gemm_tma_warp-specialized_ping-pong 的形式暴露出来，如图 10-6 所示。

![图 10-6. 采用三角色 warp 专门化核函数的 ping-pong 架构（来源：https://oreil.ly/xs7YN）](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-6.png)

这里，消费者的 MMA 相互重叠，并包含少量 MMA 后的收尾处理，即*收尾*（epilogue）处理。这种逐 MMA 的收尾清理需要在启动下一次 MMA 之前完成。具体来说，收尾可以包括累加、缩放、写回全局内存，或把结果洗牌（shuffle）到另一个 warp。此外，收尾还可以执行一些内务处理，如推进分块指针、更新循环计数器，以及发出该分块已完成的信号，以便下一次 TMA 加载或 MMA 可以启动。

> 虽然这里没有展示，但在 MMA 操作之前也有一个等价的序言（prologue）处理步骤。在这种情形下，序言阶段会用几个 TMA 请求填充流水线，以便在 MMA 消费者能够在 Tensor Core 上开始有用工作之前把数据移入寄存器。

在实践中，warp 专门化在挤出最后一点性能方面极其有效。事实上，FlashAttention v3 将其部分加速归功于 warp 专门化流水线，这些流水线把 GEMM 与 softmax 计算——连同数据传输——重叠起来，以保持所有硬件单元忙碌。由于对计算与 TMA 驱动的数据搬运进行了激进的重叠，这帮助注意力计算达到了接近峰值的 FLOPS。

此外，PyTorch 编译器（详见第 13 章和第 14 章）生成的核函数会使用 warp 专门化，以调度分离的 warp 分别加载与计算数据。它还使用低成本屏障来同步这些 warp，与下一节详述的 CUDA Pipeline API 实现类似。PyTorch 编译器系统还与 CUTLASS 的 ping-pong GEMM 集成。torch.compile 和 Triton 都可能为受支持的操作生成 warp 专门化的核函数。然而，它们基于启发式规则有选择地应用 warp 专门化，并不会为每个算子都启用 warp 专门化。

> 在负载不均衡或需要隐藏延迟的场景中使用 warp 专门化——尤其当单个 warp 的计算不足以隐藏内存加载延迟时。然而，如果核函数很小——或极度访存受限——那么坚持使用更简单的双缓冲方案可能带来相近的收益，而无需额外的代码复杂度。

### 使用 CUDA Pipeline API 实现 warp 专门化

warp 专门化在 CUDA Pipeline API 之上构建，允许专门化的 warp 使用细粒度的生产者与消费者原语进行通信。这些调用避免了整块屏障，同时能与诸如 cuda::memcpy_async 之类的异步拷贝自然组合。

使用 CUDA Pipeline API 的生产者与消费者调用（如 pipe.producer_acquire()、pipe.producer_commit()、pipe.consumer_wait() 和 pipe.consumer_release()）的关键优势在于：它们只同步那些真正需要交接数据的特定 warp 或阶段。这与强制块中每个线程都等待的做法形成对比。

块级屏障会停顿每一个 warp——即便那些与生产者—消费者流水线无关的 warp 也不例外。该块中的所有执行都必须暂停，直到每个线程都到达屏障，如图 10-7 所示。

相比之下，Pipeline API 在内部维护逐阶段的状态。当一个生产者 warp 完成其异步拷贝并调用 pipe.producer_commit 时，只有那些调用 pipe.consumer_wait 的 warp 才会阻塞，直到数据就绪。块中的其他 warp 可以继续运行任何不依赖该阶段的工作。在实践中，CUDA Pipeline API 减少了空闲时间并降低了停顿的 warp 数，因为它消除了用屏障暂停整个块的需要。借助流水线，你能以比手写异步拷贝序列（例如 PTX cp.async + __syncthreads()）更细的粒度来协调生产者与消费者的交接。

你可以用 CUDA Pipeline API 以三角色模式实现 warp 专门化。一个 loader warp 为 compute warp 生产输入，compute warp 消费这些输入并生产结果，一个 storer warp 消费这些结果并写出。流水线对象是块作用域的，它在内部跟踪阶段顺序。

![图 10-7. 块级屏障阻止线程继续执行，直到它们完成同步并加载新数据](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-7.png)

下面是一个 warp 专门化的三角色核函数示例，它使用 loader warp（warp_id 0）、compute warp（warp_id 1）和 storer warp（warp_id 2）来计算数据分块：

```
// warp specialized roles using the CUDA Pipeline API
#include <cuda/pipeline>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
// smem_bytes = 3×TILE_SIZE^2×sizeof(float) (three roles, single-pipeline)
// 3 tiles * 112 * 112 * 4 byte per float =
// 150,528 bytes < 227,328 bytes (~227 KB)
// per-block dynamic SMEM limit on Blackwell
// Choose TILE_* so total shared memory per block is ≤ the device’s
// reported per-block shared memory limit.
// Query cudaDevAttrMaxSharedMemoryPerBlockOption and set
// cudaFuncAttributeMaxDynamicSharedMemorySize accordingly.
#define TILE_SIZE 112
// Example tile compute: C = A x B for one TILE_SIZE by TILE_SIZE tile
__device__ void compute_full_tile(const float* __restrict__ A_tile,
                                  const float* __restrict__ B_tile,
                                  float* __restrict__ C_tile,
                                  int lane_id) {
    for (int idx = lane_id; idx < TILE_SIZE * TILE_SIZE; idx += warpSize) {
        int row = idx / TILE_SIZE;
        int col = idx % TILE_SIZE;
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += A_tile[row * TILE_SIZE + k] * B_tile[k * TILE_SIZE + col];
        }
        C_tile[idx] = acc;
    }
}
extern "C"
__global__ void warp_specialized_pipeline_kernel(
        const float* __restrict__ A_global,
        const float* __restrict__ B_global,
        float* __restrict__ C_global,
        int numTiles) {
    thread_block cta = this_thread_block();
    // three square tiles in dynamic shared memory: A, B, and C
    extern __shared__ float shared_mem[];
    float* A_tile = shared_mem;
    float* B_tile = A_tile + TILE_SIZE * TILE_SIZE;
    float* C_tile = B_tile + TILE_SIZE * TILE_SIZE;
    // three stage pipeline shared by the block
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 3>
        pipe_state;
    auto pipe = cuda::make_pipeline(cta, &pipe_state);
    int warp_id  = threadIdx.x >> 5;
    int lane_id  = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    // grid wide warp indexing for persistent tiling
    int totalWarps = gridDim.x * warps_per_block;
    int global_warp = warp_id + blockIdx.x * warps_per_block;
    for (int tile = global_warp; tile < numTiles; tile += totalWarps) {
        size_t offset = static_cast<size_t>(tile) * TILE_SIZE * TILE_SIZE;
        if (warp_id == 0) {
            // loader produces A_tile and B_tile
            pipe.producer_acquire();

            auto bytes = TILE_SIZE * TILE_SIZE * sizeof(float);
            cuda::memcpy_async(cta, A_tile, A_global + offset,
                               cuda::aligned_size_t<32>{bytes}, pipe);
            cuda::memcpy_async(cta, B_tile, B_global + offset,
                               cuda::aligned_size_t<32>{bytes}, pipe);
            pipe.producer_commit();
        }
        if (warp_id == 1) {
            // wait for A_tile/B_tile
            pipe.consumer_wait();
            // compute C_tile from A_tile, B_tile
            compute_full_tile(A_tile, B_tile, C_tile, lane_id);
            // finished consuming A/B; free that stage
            pipe.consumer_release();
            // publish C_tile for the storer warp
            pipe.producer_acquire();
            pipe.producer_commit();
        }
        if (warp_id == 2) {
            // store consumes computed C_tile
            // Wait for the committed stage before consuming
            pipe.consumer_wait();
            for (int idx=lane_id; idx<TILE_SIZE*TILE_SIZE; idx+=warpSize) {
                C_global[offset + idx] = C_tile[idx];
            }
            pipe.consumer_release();
        }
    }
    // launch with dynamic shared memory size equal to
    // 3 * TILE_SIZE * TILE_SIZE * sizeof(float)
    // When launching with dynamic shared memory >48 KB
    // You need to set cudaFuncAttributeMaxDynamicSharedMemorySize
    // for the kernel
}
```

这里，每个 warp 都工作在一个不同的角色中。loader warp 调用 producer_acquire()，用 cuda::memcpy_async 执行两次协作式拷贝到 A_tile 和 B_tile，然后调用 producer_commit()。compute warp 调用 pipe.consumer_wait() 以观察到新提交的数据，并立即调用 consumer_release() 以释放该阶段供复用。

compute warp 随后通过调用 producer_acquire()、计算 C_tile 并调用 producer_commit()，成为下一次交接的生产者。storer warp 调用 consumer_wait() 以观察到已计算好的 C_tile，并在一个 warp 分条（warp-striped）循环中把它写入全局内存，然后调用 consumer_release()。这一序列使用单个块作用域的流水线，没有显式的阶段编号，也避免了任何块级 __syncthreads。

> 该核函数作为持久化核函数在许多分块上运行，以摊薄启动开销。稍后会详述持久化核函数。

简而言之，将 CUDA Pipeline API 与协作组一起使用，可以在完全不使用任何显式 __sync threads() 调用的情况下，实现细粒度、SM 级的生产者—消费者交接。表 10-3 对比了三种实现：一个朴素分块核函数、一个使用 double_buffered_pipeline 的两阶段双缓冲 GEMM，以及我们的 warp 专门化流水线核函数 warp_specialized_pipeline。

表 10-3. 三种实现的对比：朴素分块核函数、两阶段双缓冲 GEMM 和 warp 专门化流水线核函数

| 指标 | 朴素分块 | 两阶段双缓冲流水线（double_buffered_pipeline） | warp 专门化流水线（warp_specialized_pipeline） |
| --- | --- | --- | --- |
| 核函数执行时间 | 41.3 ms | 20.5 ms（相较 naive 2.01× 加速） | 18.4 ms（相较两阶段 10.2% 加速） |
| warp 执行效率 | 68% | 92%（相较 naive +24%） | 96%（相较两阶段 +4%） |
| 共享内存停顿延迟/warp 同步停顿 | 高 | 低 | 极小 |
| L2 吞吐 | 80 GB/s | 155 GB/s（相较 naive +94%） | 165 GB/s（相较两阶段 +6.45%） |
| 吞吐可扩展性 | 每个 SM 仅能扩展到 2–3 个 warp | 每个 SM 可良好扩展到约 6 个 warp | 近乎线性扩展至 SM 的 warp 上限（例如 64 warps） |
| DRAM 读取字节数与 SM 周期数之比 | 重叠差 | 重叠好 | 重叠极佳 |
| 指令数 | 1.7 B | 1.05 B（相较 naive –38%） | ~1.00 B（相较两阶段 –5%） |

这里，双缓冲流水线在 20.5 ms 完成 GEMM，而 warp 专门化版本仅用 18.4 ms 便完成。这 10.2% 的改进源于 warp 专门化核函数只停顿消费者 warp（如计算）。其他 warp（如 loader 和 storer）独立推进。在前面的两阶段双缓冲核函数中，每个线程都参与消费者阶段。因此，consumer_wait() 实际上会停顿整个线程块。

这种更细粒度的、逐 warp 的同步消除了隐式的整块等待，并允许全部三个 warp（loader、compute 和 storer）持续重叠。结果，平均 SM 利用率从双缓冲设计中的约 92% 上升到 warp 专门化版本中的约 96%——并且 warp 停顿周期降至接近零。

从可扩展性的角度看，Nsight Compute 显示，朴素分块核函数在每个 SM 仅有两到三个活跃 warp 之后就饱和了。这是因为每次分块加载都必须完成后，任何计算才能开始。

两阶段双缓冲核函数通过重叠加载与计算改进了这一点。该实现在共享内存或寄存器限制成为问题之前，能扩展到每个 SM 6 个 warp。

相比之下，只要你能为加载、计算和存储分派额外的 warp，warp_specialized_pipeline 就能几乎线性地扩展。例如在 Blackwell 上，你可以一直用满每个 SM 64 个驻留 warp 的架构上限。（实际驻留量取决于寄存器、共享内存和块大小。）

如表 10-3 所示，双缓冲方式与 warp 专门化方式都大幅超越了朴素分块核函数。double_buffer_pipeline 通过重叠分块加载与计算把运行时间减半，而 warp_specialized_pipeline 通过避免任何隐式的块级等待再增加了 10.2% 的加速。只有专职的“compute” warp 会发生停顿。

指令数从朴素版本的 17 亿降到两阶段流水线的 10.5 亿，减少了 38%，再进一步降到 warp 专门化核函数的约 10 亿，又减少了 4.76%。

L2 加载吞吐从朴素分块的 80 GB/s 攀升到两阶段方式的 155 GB/s（+94%），再到 warp 专门化核函数的 165 GB/s（相较两阶段 +6.45%）。这是因为这个 warp 专门化核函数专门用一个 warp 把每个分块一次性加载进共享内存——然后把这单份拷贝多播给所有计算 lane。这消除了任何残余的冗余 L2 读取。因此，在分块与双缓冲之后，流水线中几乎所有冗余都已被移除。

在实践中，两阶段（two-stage）双缓冲流水线非常适合均匀分块（tiling）的 GEMM 工作负载。它更简单的生产者/消费者模型能把大部分 DRAM 延迟隐藏在计算之下。而 warp 专门化方法则针对不规则或更深的流水线做了优化，例如融合注意力核函数。原因在于，每个 warp 都能持续执行分配给它的角色——加载、计算或存储——而不必迫使块内其余部分停顿。

### PyTorch、CUDA Pipeline API 与 warp 专门化

PyTorch 的公开 API 并不直接暴露 cuda::pipeline，但当你调用 torch.compile 时，编译器会生成核函数（用 OpenAI 的 Triton 语言实现），这些核函数所用的优化实现了与 <cuda/pipeline> 原语和 warp 专门化的生产者/消费者等价的功能。

因此，虽然你看不到显式生成的 cuda::pipeline 调用，但你会看到底层的指令与屏障。这些优化有助于提升占用率（occupancy）并增大数据传输/计算的重叠。换句话说，你无需自己编写任何 CUDA 代码，就能获得与手写 <cuda/pipeline> 实现相同的低延迟、高吞吐行为。

PyTorch 由 TorchInductor 生成的融合注意力核函数，使用了带生产者组与消费者组的 warp 专门化。例如，考虑 PyTorch 中如下所示的三个独立 GPU 核函数：

```
scores = torch.matmul(queries, keys.transpose(-2, -1))
probabilities = F.softmax(scores, dim=-1)
context = torch.matmul(probabilities, values)
```

你可以在 PyTorch 中用一行代码将其重写，如下所示：

```
context = torch.nn.functional.scaled_dot_product_attention(queries,
   keys, values)
```

在底层，加载 warp 将分块取入共享内存，计算 warp 处理这些分块，存储 warp 将结果写回。这消除了 matmul 与 softmax 阶段之间对全局内存的昂贵往返（round trip）。

总体而言，torch.compile 非常适合性能敏感且不规则的工作负载。像 Blackwell 这样的现代 GPU 拥有充裕的 SM 资源和高内存带宽。因此，warp 专门化之类的技术能最大限度地减少空闲周期，并允许在各 warp 之间实现近乎线性的扩展——至少在 SM 或内存带宽被完全占满之前如此。

> 即便是像 FlashAttention-3 这样高度优化的核函数，使用 warp 专门化重叠也只能达到峰值 FP16 FLOPS 的约 ~75%。这表明你不必达到 100% 的计算利用率，也能取得显著的优化里程碑。

通过将加载、计算和存储解耦为独立的阶段，流水线模型最大化了吞吐（throughput）与资源利用率，使其成为长时间运行核函数的首选方法，包括诸如 transformer 注意力、融合算子流水线以及自定义任务调度器等处理过程。这些核函数利用细粒度的 warp 间通信、深度流水线化以及线性的 warp 扩展来提供高性能。

## 持久化核函数与巨型核函数

*持久化核函数*（persistent kernel），也称为*持久化线程*（persistent threads），颠覆了通常“每个任务一个核函数”的做法。你不必启动许多各自带来显著开销的小核函数，而是可以启动单个长时间运行的核函数，其线程持续从全局内存或共享内存中的共享生产者—消费者队列拉取工作。

当持久化线程循环运行时，它们会在数据块到达时立即处理——通常借助内存拷贝或主机信号——而不退出核函数。这完全避免了重复的核函数启动开销。例如，一个持久化核函数可能用一个线程块的 Warp 0 将数据从全局内存（或 CPU 主机内存）拷贝到共享内存。与此同时，Warp 1 计算上一批数据。这是 GPU 上软件流水线化的一种形式。

例如，考虑有 1,000 个微小的、彼此独立的任务。传统上，人们可能会启动 1,000 个独立的核函数。每个核函数只短暂占用几个 SM，随即退出。

在实践中，GPU 会为每个微小核函数反复加速启动，之后再减速退出。这会在各次启动之间让大多数 SM 处于空闲，无法充分利用硬件。

改用持久化核函数后，你转而启动一个大网格，其设计目标是在整个工作负载期间都让 GPU 保持忙碌。例如，在一块拥有 132 个 SM 的 GPU 上，这可能意味着为每个 SM 启动一个块、每个块 256 个线程。总共就是 33,792 个线程。随后每个线程执行大致如下所示的代码：

```
__device__ int g_index;  // global counter for next task; initialize to 0 on host
__global__ void persistentKernel(Task* tasks, int totalTasks) {
    // Every thread loops, atomically grabbing next task index until none remain
    while (true) {
        int idx = atomicAdd(&g_index, 1);
        if (idx >= totalTasks) break;
        processTask(tasks[idx]);
    }
}
```

在启动这个核函数之前，主机会在设备内存中设置 g_index = 0。然后它会像下面这样调用该核函数：

```
cudaMemset(&g_index, 0, sizeof(int));
// one block per SM on a 132 SM GPU
int blocks         = 132;
int threadsPerBlock = 256;
persistentKernel<<<blocks, threadsPerBlock>>>(d_tasks,
    totalTasks);
cudaDeviceSynchronize();
```

现在，你不再为每个单独任务支付启动开销，而只需为启动 persistentKernel 支付一次开销。假设启动大约需要 0.02 ms，而 1,000 个任务中每个运行 0.1 ms，那么这些核函数会背靠背运行，总工作量约为 100 ms。

相比之下，背靠背运行 1,000 个微小核函数、每个启动耗时 0.02 ms，会在执行时间上额外增加 20 ms 的开销——这与全部 1,000 个任务共计 100 ms 的总运行时间是分开的。简而言之，将小任务整合进一个持久化循环，可以削减数十毫秒的启动开销。

使用持久化核函数，GPU 保持高度利用——几乎所有 SM 都在并发处理任务——因为有数万个线程可用于处理这 ~1,000 个任务。相反，顺序启动 1,000 个微小核函数会让 GPU 在任一时刻都有大部分处于未充分利用的状态。在我们的例子中，这相当于平均只使用了 GPU ~35% 的容量。

在这个持久化核函数场景中，每个 SM 运行一个 256 线程的块（在最多 2,048 线程中），因此每个 SM 都在工作。这样一来，尽管每个 SM 自身的占用率相对较低，仅为 12%（256 ÷ 2048 线程），却有近 100% 的 SM 处于活跃状态。

以这种方式使用持久化核函数，GPU 可以维持较高的实测 SM Active percent，但受寄存器和共享内存用量的制约。请记得始终用 Nsight Compute 加以验证。

在现代 GPU 上，持久化核函数尤其有效，因为其更大的共享内存容量和扩展的寄存器文件让每个线程能在片上保留更多中间状态。线程可以在其他 warp 进行计算时，使用 TMA 为即将到来的任务预取张量分块。因此，在某些线程处理一个任务的同时，其他线程使用 TMA 为即将到来的任务预取数据——而不会用内存传输指令拖累 SM 的计算流水线。

### 持久化核函数的常见工作负载

当你有许多小的或大小不均的任务、若分别处理会带来高启动开销时，持久化核函数便大放异彩。它们支持动态负载均衡。这让更快的线程能够继续循环、抓取更多工作。使用持久化核函数，任何 SM 都不会过早地陷入空闲。

这种模式常见于不规则工作负载，例如图遍历、自定义批处理变换，以及 LLM 推理中常见的逐 token 操作。在这些情形中，每个任务的耗时可能差异很大。

不过，持久化核函数也有缺点。首先，你必须使用原子操作显式地管理任务队列与同步。如果许多线程试图同时递增同一个计数器，这可能引入竞争。

调试单个巨型的持久化循环，比调试多个小核函数更为复杂。这是因为单个发散线程或一个意料之外的分支就可能导致整个核函数挂起。此外，一个持久化核函数可以无限期地独占 GPU。因此，如果需要并发运行其他工作负载，你必须谨慎地分配 stream 或划分资源。

简而言之，通过把 GPU 变成一个不断抓取并处理任务的动态“工作线程”池，持久化核函数相比朴素核函数可以大幅提高整体吞吐（例如 2–3×）。在现代 GPU 硬件上，这种方法消除了启动开销，最大化了 SM 占用率，并且——当与协作组或线程块簇（稍后介绍）结合时——能在多阶段流水线中始终把数据保持在片上。

越来越多的框架和库正在使用持久化核函数与巨型核函数，以避免容量浪费并提升推理等延迟敏感型工作负载的性能。关键在于消除重复启动，并在 GPU 上使用设备端任务队列（device-side queue），让 SM 始终被占满并执行有用的工作。

> 在撰写本文时，由于调度的复杂性，PyTorch 尚不会自动把整个多阶段工作负载融合进一个核函数。因此，要充分获得持久化核函数与巨型核函数的收益，需要自定义 CUDA 代码或专用编译器。尽管如此，对于多阶段算法，重构为持久化巨型核函数可以带来显著的性能提升——前提是你妥善处理各处同步并避免死锁（deadlock）。

### 面向推理的巨型核函数

此外，一种源自大规模推理的持久化核函数现代做法称为*巨型核函数*（megakernel）。巨型核函数将跨层——甚至跨 GPU——的整段操作序列融合进单个大型核函数。如图 10-8 所示，持久化巨型核函数已被证明通过消除重复的核函数启动开销，相比传统的逐层启动可将延迟降低 1.2× 到 6.7×。

![图 10-8. 巨型核函数相对于 vLLM 和 SGLang 的解码吞吐提升（来源：https://oreil.ly/2aZiF）](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-8.png)

### 持久化核函数与 warp 专门化

warp 专门化通常与持久化核函数一起使用，在这些场景中线程会在相对较长的时间内执行多次迭代。这样便可实现更深的流水线、更好的重叠，以及对长生命周期资源的高效利用。对于运行时间较短的核函数，持久化核函数与 warp 专门化所增加的代码复杂性可能得不偿失。

而持久化核函数调度的一个局限，是要为持久化核函数找到足够的 SM 来利用。如果太多 SM 被另一个核函数占用，可能就没有足够的资源来启动持久化核函数。这在尝试跨 SM 调度和负载均衡工作时会带来挑战。

为了便于实现持久化核函数（从而实现 warp 专门化），现代 GPU 支持*线程块簇*——由于线程块也称为协作线程阵列（cooperative thread array，CTA），因此它也被称为*协作线程阵列（CTA）簇*。我们会在后续小节讨论线程块（CTA）簇，简而言之，它们让你把多个线程块组合成占用 GPU 上多个邻近 SM 的“簇”。

## 协作组

协作组让你能够以任意粒度定义并同步线程组。例如，你可以用单个线程、warp、分块、块和簇来创建组，如图 10-9 所示。

![图 10-9. 使用协作组在不同粒度的线程之间进行同步](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-9.png)

协作组提供了安全、可复用的集合操作，如 sync、broadcast 和 reduce。这与使用临时凑合的同步屏障形成对比。通常，线程只能使用 __syncthreads() 在自己的块内同步——例如，并没有针对整个网格的内置全局屏障。

协作组为你提供核函数内部的细粒度同步。该 API 非常适合跨 warp、块和簇协调多阶段流水线。要在你的核函数中使用 Cooperative Groups API，只需包含 <cooperative_groups.h>，获取组对象，然后在这些组之间进行同步与协调。

该 API 包含诸如 cg::this_thread_block()、cg::tiled_partition() 和 cg::this_cluster() 之类的调用。随后你调用 group.sync()——或类似的集合操作——来协调这些组中的线程。

要以协作模式启动核函数，你使用 cudaLaunchCooperativeKernel()。在协作模式下，CUDA 会确保你启动的网格能够并发驻留——否则启动失败。因此，建议始终使用 cudaOccupancyMaxActiveBlocksPerMultiprocessor 来确定协作式网格（cooperative grid）的规模，并对网格大小进行钳制以避免启动失败。在核函数内部，你随后可以调用下面的代码来实现一个覆盖所有线程块的屏障：

```
cooperative_groups::this_grid().sync(); // or grid.sync()
```

在这种情况下，任何块中的任何线程都不能越过该点，直到每个块中的每个线程都到达该点为止。这个屏障让你能够把一个核函数拆分为顺序的多个阶段，而无需结束启动或将控制权交还给主机。

例如，考虑 LLM 中常见的 softmax 算法，它有两个阶段：先在整个数组上做归约以计算聚合和，然后是使用该聚合和的逐元素后续计算。传统上，你会启动一个核函数做归约，将结果拷回主机内存或全局内存，再启动第二个核函数从主机或全局内存消费该结果，然后计算 softmax。这需要大量相对缓慢的内存搬移。

有了协作组，你可以在一个核函数中完成这两个阶段：每个块计算它的部分和，由一个块把这些部分和聚合成最终结果，所有块调用 grid.sync() 等待聚合完成，然后所有线程进入第二阶段。随后每个线程会从寄存器或共享内存——而非从全局内存——读取聚合和。

CUDA 保证协作式启动中的每个块都同时驻留在 GPU 上。如果你请求的块数超过了能够并发容纳的数量，启动就会直接失败。

由于协作式核函数必须以 GPU 能够并发运行的网格大小来启动，grid.sync() 不会因等待不存在的块而挂起。换句话说，CUDA 运行时保证所有线程块都处于活跃状态并会到达 grid.sync() 屏障。如果核函数启动过大而无法一次性运行，它只会启动失败。因此，检查 cudaLaunchCooperativeKernel() 的返回状态非常重要。

例如，如果一块 GPU 有 132 个 SM，而你的核函数占用的资源使得每个 SM 能运行四个块，那么网格必须不大于 528 个块才能成功。如果超出该上限，协作式启动就会直接失败。请使用 cudaOccupancyMaxActiveBlocksPerMultiprocessor 为协作式核函数确定网格规模，并在假定已取得进展之前检查 cudaLaunchCooperativeKernel 的返回状态。

在 Blackwell 之前，开发者常将 cudaLaunchCooperativeKernel 与全局内存原子操作或标志位配合使用，来协调那些并不共享片上内存的多个块。这种方法虽然可行，但迫使中间结果被搬移到全局内存。这带来了额外的 HBM 流量。

在 Blackwell 上，线程块簇与 DSMEM 提供了一种效率高得多的替代方案。线程块簇可以在片上 SRAM 中共享数据——并且无需全局内存往返即可同步。我们将在本章后面介绍线程块簇与 DSMEM。

当你需要在单次核函数启动内实现一个真正的、覆盖所有线程块的屏障时，就应当使用协作式核函数。此外，当你希望把中间结果保留在快速内存（例如寄存器或共享内存）中、而不是反复写入和读取全局内存时，它们也很有用。建议你把网格大小限制在 GPU 的最大并发块容量以内——或选择更大的块——以减少总块数并避免核函数启动失败。

协作式核函数的缺点在于你的网格大小受容量限制。这可能迫使你使用更少、更大的块——或依赖线程块簇（本章后面介绍）。

尽管所有块都并发运行，协作式屏障仍要求每个块中的每个线程都调用 grid.sync()。如果任何单个线程跳过——或永远无法到达——grid.sync() 调用（例如由于一条发散的 if 语句），那么每个确实调用了 grid.sync() 的其他线程都将永远等待下去。这会导致死锁。

简而言之，协作组核函数让你能把整个 GPU 当作一个单一的协作资源来对待，以 grid.sync() 充当全局屏障。这对于需要全局同步与数据共享的多阶段算法非常理想。只是要记住，grid.sync() 是一种相对重量级的同步，其开销高于块级或簇级屏障。这是因为它必须在运行于多个 SM 上的所有线程块之间进行协调。因此，你应当谨慎使用 grid.sync()，并且至少要确保你的核函数在两次重量级同步调用之间执行了相当多的工作。

> 对于只需要有限的跨块协调的情形，使用全局内存原子操作或每块标志位可能比依赖整网格屏障更安全、更简单。不过，正如我们稍后将讨论的，它们的效率不及使用线程块簇与 DSMEM。

### 协作式网格同步与持久化核函数

如果你的工作负载偶尔需要在持久化循环内部使用跨线程块屏障来聚合部分结果，你可以通过调用 grid.sync() 把持久化核函数与协作组结合起来。这将提供一个覆盖整个网格的屏障，从而避免结束核函数并不得不重新启动它。以这种方式，一条由归约和其他全局步骤组成的多阶段流水线便可完全保持在设备上。

例如，考虑一个工作负载，它在 1,000 次迭代中反复执行两种不同的计算，且每种计算因跨块数据依赖而需要一个全局屏障。朴素的实现可能每次迭代启动两个独立的核函数，导致 2,000 次核函数启动。在单个 stream 中，第二个核函数会自动等待第一个——无需显式的主机端同步——但你仍要支付启动开销。相反，你可以把所有内容融合进一个协作式、持久化的核函数，如下所示：

```
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__device__ inline float someComputationA(float x) { return x * 2.0f; }
__device__ inline float someComputationB(float a, float b) { return a + b; }
__global__ void combinedKernel(float* __restrict__ dataA,
                               float* __restrict__ dataB,
                               int N,
                               int iterations) {
    cg::grid_group grid = cg::this_grid();                 // whole-grid group
    const int tid = blockIdx.x * blockDim.x + threadIdx.x; // thread's linear id
    const int stride = gridDim.x * blockDim.x;             // total grid threads
    for (int it = 0; it < iterations; ++it) {
        // Stage 1: update all elements of A using a grid-stride loop
        for (int i = tid; i < N; i += stride) {
            dataA[i] = someComputationA(dataA[i]);
        }
        grid.sync();  // global barrier; also provides memory visibility
        // Stage 2: use finished A to update B (again grid-stride)
        for (int i = tid; i < N; i += stride) {
            const float mid = dataA[i];
            dataB[i] = someComputationB(mid, dataB[i]);
        }
        grid.sync();  // barrier before next iteration
    }
}
```

在主机端，你只需调用这个合并了协作式与持久化特性的核函数，如下所示：

```
// ... allocate/copy dA, dB, set N, iterations, choose blockSize
cudaDeviceProp prop{};
cudaGetDeviceProperties(&prop, 0);
if (!prop.cooperativeLaunch) {
    // Fallback: run two kernels per iteration instead of grid.sync()
    // (see alternative below)
}
// Compute the maximum grid size allowed for a cooperative kernel
int maxBlocksPerSm = 0;
const int blockSize = 256;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSm,
                                              combinedKernel,
                                              blockSize, 0);
const int maxCoopGrid = maxBlocksPerSm * prop.multiProcessorCount;
// Choose grid size, but clamp to cooperative limit
int gridSize = (N + blockSize - 1) / blockSize;
if (gridSize > maxCoopGrid) gridSize = maxCoopGrid;
// Launch cooperatively
void* args[] = { &dA, &dB, &N, &iterations };
cudaLaunchCooperativeKernel((void*)combinedKernel, gridSize, blockSize, args);
cudaDeviceSynchronize();
```

这个单一核函数取代了原本需要的 2 × 1,000 = 2,000 次核函数启动，并避免了重复的启动开销。在循环内部，grid.sync() 确保了正确的次序（每个块都先完成 Stage 1，然后任何块才开始 Stage 2——并且在进入下一次迭代之前如此），而无需任何主机端同步。

逐线程和逐块的数据可以跨越 grid.sync() 保留在寄存器/共享内存中。对于跨块的数据交换，则使用全局内存（或者线程块簇与分布式共享内存，我们稍后会介绍）。与此同时，由于工作是在 GPU 上循环进行的，首次调用之后就不再有启动开销。

协作式核函数需要设备支持（cooperativeLaunch），并且必须用 cudaLaunchCooperativeKernel 启动。网格中的所有块都必须并发驻留。请使用 CUDA Occupancy API 来确定并钳制网格大小，使所有 CTA 都能共同驻留。否则启动将失败。一个例子是当 ((N + threads - 1) ÷ threads) 超过协作容量时。

> 使用 CUDA Occupancy API 来确定协作式网格的规模。

现代 GPU 可以并发运行数千个数量级的线程块（前提是每个块只使用适量的资源，包括寄存器和共享内存），因此有相当大的余量。不过，在采用这种实现之前，你仍应验证块大小和资源消耗。

在这个核函数示例中，核函数在两次迭代之间从不将控制权交还给主机。因此，对于外层循环它表现得像一个持久化核函数，但在内层阶段又强制了全局同步点。

这种组合模式对于 LLM 推理中的多步算法尤其强大，其中每一层可能都需要一次归约或归一化（Stage 1），随后是一次逐元素变换（Stage 2）。通过把所有内容都装进一个协作式持久化核函数，你消除了所有核间往返，并最大化了片上数据局部性。

### 何时结合使用持久化核函数与协作组

一种常见的最佳实践是，在资源允许的情况下，为每个 SM 启动一个线程块的持久化核函数。这样，每个 SM 都有一个驻留的线程块，反复处理来自工作队列的任务。这最大化了占用率，并让 SM 不至于陷入空闲。这些线程块随后会使用 grid.sync() 在各阶段之间协调。然而，在协作式策略与持久化策略之间——以及是否将二者结合——做决定时，请提出以下问题：

*你是否有多个需要全局同步的顺序阶段？* 如果是，就使用协作式核函数（或协作式持久化核函数），以便在各阶段之间调用 grid.sync()。

*你是否有许多因启动开销而受损的小任务或不规则任务？* 如果是，就使用持久化核函数，让你的线程在一个共享队列上循环，而无需返回主机。

*你是否能够承担为这一工作负载独占整个网格（或一个线程块簇）？* 如果是，协作式持久化核函数或许能带来最佳性能。

*你是否需要与其他工作共享 SM？* 如果是，就考虑使用线程块簇（下一节介绍），以便持久化或协作式核函数不为其他工作负载服务。

简而言之，当你使用单个核函数既在任务上持久化循环、又包含 grid.sync() 调用来在各阶段之间同步所有线程块时，你就消除了通常在独立核函数启动之间发生的暂停和额外的内存传输。

在现代 GPU 上，这意味着数据在整个计算过程中都保留在共享内存或寄存器里。这与在每个阶段之后把数据写回全局内存形成对比。其结果是，GPU 几乎始终忙于做有用的工作——达到接近其硬件峰值极限的性能。

一个重要的注意事项：协作式核函数会占用网格中的所有 SM，因此在这些 SM 上不能并发运行其他核函数。如果你需要协同调度（coscheduled）其他工作，例如在另一个 stream 上的异步数据预取或较低优先级的推理核函数，你可能需要把 GPU 划分为线程块簇，这将在下一节介绍。

## 线程块簇与分布式共享内存

*协作组*是一种软件层面的抽象，它提供了一个 API，让你能把核函数的线程切分为任意的集合以进行同步和数据搬移。这包括 warp、分块、线程块——甚至整个网格或多设备网格。

相比之下，线程块簇（或协作线程阵列 [CTA] 簇）是一种硬件层面的层次结构。它把 SM 的一个子集授予你的协作式网格——并把其余部分留给其他核函数使用。这缓解了单个核函数独占 GPU 的风险。GPU 保证这些线程块会被协同调度到同一个 GPU 处理集群（GPC）上，如图 10-10 所示。

![图 10-10. 多个线程块簇被保证协同调度到同一个 GPC 或 GPC 分区上](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-10.png)

GPC 是一组邻近 SM 的集合。GPU 把线程块调度到各个 GPC 上，类似于它把一个线程块的线程调度到同一个 SM 上。

> 在像 NVIDIA Blackwell B200/300 和 GB200/300 这样的多晶片模块上，实际上存在多个 GPC“分区”——每个晶片一个 GPC 分区。由于 Blackwell 是双晶片 GPU，它有两个 GPC 分区。请记住，Blackwell 的两个 GPU 晶片通过 NV-HBI 连接，并作为单个 CUDA 设备呈现，跨晶片具有完整的缓存一致性。L2 缓存在两个晶片之间也是一致的。因此，这些晶片构成了一个组合的逻辑 GPC，从而由架构为你处理各个独立的 GPC 分区。

GPU 提供分布式共享内存（将在下一节讨论），供线程块簇跨这些块使用。它还支持使用 Cluster Group API 的簇级屏障（cluster.sync()）。

这个簇级屏障让你只同步线程块的一个子集，而不阻塞整个 GPU。线程块簇让你能够启动一个协作式核函数，把网格细分为更小的块组，如图 10-11 所示。

![图 10-11. 对于这四个（2 × 2）线程簇，A 和 B 的每个分块被同时加载到两个线程块中（来源：https://oreil.ly/kEZsv）](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-11.png)

在每个组内，调用 cluster.sync() 提供一个局部屏障。这让簇内的块能够通过专用的片上资源共享数据，而不必独占每一个 SM。在现代 GPU 上，你可以使用 DSMEM，它允许线程块簇共享一块连续的片上 SRAM 区域。这在同一簇内的块之间实现了低延迟通信，并有原生硬件支持。

线程块簇把线程块的一个子集分组，并在每个簇内调用 cluster.sync() 以“局部”地在这些线程块之间同步。虽然线程块内的线程传统上可以使用共享内存进行协作，但在现代 GPU 上，它们现在还可以使用线程块簇和 DSMEM 相互协作。

换句话说，如果没有线程块簇，只有同一线程块内的线程才能在共享内存中共享数据。不同的线程块则必须使用全局内存和全局屏障（例如 grid.sync()）来协调，而这会让所有线程停顿并限制可扩展性。

而且，此外，正如上一节所述，不同线程块中的线程此前除了使用全局内存或用于粗粒度、覆盖整个网格的同步的 grid.sync() 之外，无法高效地共享状态和同步。遗憾的是，全局内存和 grid.sync() 都相对缓慢，因此可能成为瓶颈。

线程块簇由 GPU 硬件原生支持，包括*簇启动控制*（cluster launch control）。簇启动控制是一种硬件层面的机制，用于启动和调度持久化的线程块簇。具体而言，它允许一个持久化核函数（及其线程块簇）维持均衡的工作负载——即便某些 SM 已被占用。这为高效的 warp 专门化实现奠定了基础。

借助硬件支持的通信与同步构件，线程块簇可以以底层 PTX 指令和 CUDA intrinsic 的形式使用覆盖整个簇的屏障进行同步。因此，由于线程块之间有硬件支持的屏障同步，簇内的块进行屏障同步可以比使用 grid.sync() 的完整网格同步快得多。

### 线程块交错

在直接的网格启动中，线程块按严格的行主序或列主序处理分块。这可能导致较早的块驱逐掉稍后的块将会需要的数据——从而造成糟糕的复用和额外的内存流量。相反，你希望同一波（wave）内的分块 A 和 B 能从 L2 缓存中读出。

为了绕过这种低效，你可以使用线程块交错（thread block swizzling）。类似于用交错来优化内存访问并避免共享内存 bank 冲突，你可以使用线程块交错来避免以低效的行主序和列主序分配分块，如图 10-12 所示。

![图 10-12. 线程块交错，在单一波中从 L2 缓存读取分块 A 和 B](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-12.png)

线程块交错让同一波所需的 A 矩阵和 B 矩阵的分块都留在 L2 中以实现最大复用。当应用于持久化和分块的 GEMM 工作负载时，这类交错可以通过减少内存未命中和带宽压力，带来两位数的性能提升。线程块交错是一种简单却强大的模式重排技术，它把核函数的启动次序与缓存局部性对齐。

有了对线程块簇的硬件支持、相对大量的 SM 以及改进的流水线调度，现代 GPU 可以承载带有精细 warp 专门化流水线的大型持久化核函数，从而优化核函数性能。

简而言之，线程块簇建立在协作组模型之上，允许块的一个子集在不锁死整个 GPU 的情况下进行同步和状态共享。这让你能够在一次启动内构建多阶段、细粒度的流水线，而不锁死设备的其余部分。这就让剩余的 SM 得以自由执行独立的工作。

在线程块簇内部，有两种在线程块之间共享状态的关键机制：DSMEM 和暂存内存（scratch memory）。接下来我们逐一介绍。

### 分布式共享内存

分布式共享内存把 __shared__ 内存的概念从单个线程块扩展到横跨整个线程块簇。在传统核函数中，每个线程块都有自己私有的 __shared__ 区域，且其他线程块无法访问。

然而有了 DSMEM，同一簇内的多个块可以读写一个在逻辑上把它们所有本地 __shared__ 区域组合起来的共享内存空间。实际上，簇的共享内存被拼接成一个分布式的片上缓冲区。

通过把块间数据保留在片上内存中，DSMEM 显著提高了有效算术强度（arithmetic intensity），因为块之间可以交换中间结果或执行归约，而无需往返全局内存。当数据集对单个块的共享内存来说过大时，这尤其有价值。

与其退回到全局加载和存储，你可以启动一个线程块簇，让其中的每个块处理数据的一部分，然后使用 DSMEM 进行同步与共享。所有通信都以 SM 间（SM-to-SM）速度完成，从而避免了昂贵的 HBM 流量。

简而言之，DSMEM 是一种比使用全局内存进行块间通信更快、更结构化的替代方案。但它的作用域仅限于同一个簇内的块。DSMEM 非常适合以下场景：需要频繁、低延迟块间协调的持久化核函数、注意力机制，或 LLM 中的其他多阶段算法（每个块在继续之前都必须交换中间状态）——以及任何受益于将中间结果保留在片上、而非写回全局内存再重新加载的工作负载。

> 可移植的最大簇大小为 8 个块。某些 GPU 在通过 cudaFuncAttributeNonPortableClusterSizeAllowed 属性显式启用时，支持更大的非可移植簇大小（例如最多 16 个 CTA）。这会增大你的 DSMEM 占用量，代价是占用簇中更多的 SM。

### 暂存内存

暂存内存（scratch memory）是支撑 DSMEM 与线程块簇同步的底层硬件基础设施；DSMEM 是对你的 CUDA 代码隐式可见的共享数据缓冲区，而暂存内存则是 GPU 用来跟踪协调元数据（如屏障状态、组进度以及 DSMEM 访问标志）的一块独立的片上 SRAM 区域。

你不会从核函数中直接访问暂存内存，而是由 GPU 自动管理它。当你启动一个线程块簇时，硬件会分配一部分暂存内存，用来维护簇级屏障计数器（cluster.sync() 状态）、跟踪哪些块已到达 DSMEM 操作或同步点，并协调跨 SM 对分布式共享区域的安全访问。

由于暂存内存针对这些元数据操作做了优化，它使簇级屏障和 DSMEM 访问能够非常快速地完成。如果对于非常大的簇或复杂的同步模式，元数据大小超过了可用的暂存 SRAM，GPU 会透明地将部分状态溢出到本地内存中（由 L1/L2 缓存支撑）。这种溢出在保证正确性的同时，仍尽可能保留片上效率。

简而言之，DSMEM 是你用来在块之间共享数据的抽象，暂存内存则是幕后使这些 DSMEM 操作快速且可扩展的设施。二者结合，使一个线程块簇能够表现得像一个单一的逻辑单元，打破了线程块之间的传统壁垒。这为那些需要跨多个线程块紧密协调并行的工作负载提升了性能。

### 启动线程块簇

让我们说明如何在实践中使用线程块簇。首先，我们需要用一个此前未使用过的簇维度（cluster dimension）来启动核函数。

在 CUDA 中，这通过一个允许指定簇大小的扩展启动 API 来完成。例如，如果我们想要两个块组成的簇，也称为*线程块对*（thread block pair）或 *CTA pair*，可以配置一个特殊的启动属性，如下所示：

```
// Host code: launch a kernel with thread block clusters of size 2
cudaLaunchConfig_t config{};
config.gridDim = dim3(128, 1, 1);
config.blockDim = dim3(256, 1, 1);
config.dynamicSmemBytes = 0;
cudaLaunchAttribute attr{};
attr.id = cudaLaunchAttributeClusterDimension;
attr.val.clusterDim.x = 2;
attr.val.clusterDim.y = 1;
attr.val.clusterDim.z = 1;
config.attrs = &attr;
config.numAttrs = 1;
// Allow non-portable cluster sizes if you intend to use > 8 later
cudaFuncSetAttribute(MyClusterKernel,
                     cudaFuncAttributeNonPortableClusterSizeAllowed,
                     1);
cudaLaunchKernelEx(&config, MyClusterKernel, args, nullptr);
```

在这段主机代码中，我们使用 cudaLaunchKernelEx 来启动 MyClusterKernel，每个簇包含 2 个线程块（因此总共 64 个簇，因为 128 个块 ÷ 每簇 2 个）。clusterDim 被设为 2，意味着每个簇将包含 2 个线程块。对于我们想要簇协作核函数的场景，这取代了传统的 <<<gridDim, blockDim>>> 语法。在底层，CUDA 运行时确保这些配对的块被调度到能够通过 DSMEM 通信的位置。

> 请记住，每个线程块簇都需要在物理上装入你所指定的 SM 与资源。可移植的最大簇维度为 8 个线程块。要在受支持的 Blackwell 型号上启动 16 个块，你必须启用非可移植属性，并合理设置块大小，使簇能够常驻于 SM 上。在为线程块簇设定大小时请牢记这一点。

### 用协作组 API 协调线程块簇

要在簇中的多个线程块之间协调工作，你首先要使用 cooperative_groups::this_cluster() 获得该组块的句柄。这个簇句柄让你能够执行硬件加速的簇级屏障——并直接访问另一个块的共享内存。

这一切都无需离开核函数、也无需借助全局内存标志。下面是一个示例核函数，它将每个线程块的一个本地值累加到线程块 0 的共享内存中：

```
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__global__ void MyClusterKernel(/* args */) {
    // 1. Form a cluster group for all thread blocks in this thread block cluster
    cg::cluster_group cluster = cg::this_cluster();
    // 2. Allocate the same extern shared buffer in each block
    extern __shared__ int shared_buffer[];
    // 3. Each block learns its rank and the total cluster size
    int clusterRank = cluster.block_rank();
    int clusterSize = cluster.num_blocks();
    // 4. Initialize this block’s portion of shared memory (a simple local sum)
    int localSum = threadIdx.x;
    shared_buffer[threadIdx.x] = localSum;
    // 5. Barrier across all blocks in the cluster; no block proceeds until
    //    every block reaches this point and has written its shared_buffer.
    cluster.sync();
    // 6. Map a pointer to block 0’s shared_buffer so block can write into it
    // Pointers returned by cluster.map_shared_rank
    // refer to the remote CTA’s shared memory
    // and support remote atomics
    // and memory operations within
    // the cluster
    // Pair updates with cluster.sync at well defined point
    int* remote_buffer = cluster.map_shared_rank(shared_buffer, 0);
    if (clusterRank != 0) {
        // 7. Nonzero blocks atomically add their local shared_buffer[0] into
        //    rank 0’s buffer. This atomicAdd is routed on‐chip over DSMEM
        //    (not through DRAM.)
        atomicAdd(&remote_buffer[0], shared_buffer[0]);
    }
    // 8. Another cluster‐wide barrier ensures all atomic adds have completed
    cluster.sync();
    // 9. Finally, block 0 (rank 0, thread 0) can read the combined result
    if (clusterRank == 0 && threadIdx.x == 0) {
        printf("Combined sum in cluster[0]: %d\n", shared_buffer[0]);
    }
}
```

在这里，我们通过调用 cg::this_cluster() 获得一个簇句柄。它返回一个 cluster_group 对象，正好表示那些作为一个线程块簇一起启动的块。运行时保证该簇组中的每个线程块都同时常驻——否则启动将失败。

系统会隐式提供一致的共享内存分配。簇中的每个块都必须为 extern __shared__ int shared_buffer[] 声明相同的大小。DSMEM 硬件随后会在逻辑上将每个块的共享内存合并为一个虚拟地址空间。而且由于所有块都预留了相同的 shared_buffer 大小，指向线程块 0（即 rank 0）缓冲区的指针将能够跨 SM 正确协调。

> 使用线程块簇时，务必像全局内存合并一样组织每个线程块的 DSMEM 访问。换言之，让每个 warp 读写连续的、32 字节对齐的扇区。这样，硬件就能在没有 bank 冲突或意外串行化的情况下路由簇级 DSMEM 传输。在实践中，布局你的分块数据，使线程块 j 中的 warp i 始终访问一个唯一的、对齐的范围。这将避免内存 bank 争用，并使 DSMEM 数据传输保持全速运行。

在前面的示例代码中，我们还看到 cluster.sync() 被用作跨簇内线程块的簇级屏障。与只同步单个块内线程的 __syncthreads() 不同，cluster.sync() 同步该簇中每个块的所有线程。

这个屏障由硬件实现，通常比网格级同步具有更低的延迟，因为它只协调簇中的块。这意味着你可以频繁地同步块而影响极小——只要它们在同一个簇内。

而且不存在因块缺失而导致死锁的风险，因为 CUDA 强制要求网格正确启动并装入 GPU。因此，不会出现某些块到达屏障后永远挂起、等待那些从未启动的“缺失”块的情形。所有块都已存在，因此一旦每个块都调用它，屏障就会干净地完成。

在前面的代码块中，你看到跨块共享内存访问是通过 map_shared_rank() 完成的。在第一个屏障之后，每个块的 shared_buffer[] 都已初始化。为了获得指向块 0 共享内存的指针，其他块调用 int* remote_buffer = cluster.map_shared_rank(shared_buffer, 0)，其中指定了簇中的线程块 id（0）。

GPU 硬件会自动转换该指针，使任何加载或存储都经由片上 DSMEM 网络，而不是发往全局 DRAM。因此，对 remote_buffer 的任何写操作——无论是原子写还是普通写——都会直接在片上更新块 0 的共享内存。值得强调的是，远程 DSMEM 访问的性能特征不同于本地共享内存。远程加载和存储受益于合并的、对齐的 32 字节段。

例如，在前面的代码块中，当块 1...n 需要将它们的本地值累加到块 ID 0 的共享缓冲区时，它们调用 atomicAdd(&remote_buffer[0], shared_buffer[0])。而由于 remote_buffer 是通过 cluster.map_shared_rank() 获得的，这个 atomicAdd 会经由片上 DSMEM 网络直接进入块 0 的 SMEM。换言之，不存在到 DRAM 或 L2 缓存的往返。每次写入都以片上速度完成。

在前面的代码示例中，你看到一旦所有块都执行了它们的 atomicAdd，第二个 cluster.sync() 会确保块 0 在继续之前，在自己的 shared_buffer[0] 中看到完整的和。

简而言之，cooperative_groups::this_cluster() 加上 cluster.sync() 和 cluster.map_shared_rank() 为你提供了一种简单、高效的方式，在一个线程块簇中的多个线程块之间进行同步和数据共享。所有这些操作都在片上进行，避免了全局内存往返，并支持块之间的细粒度协作。这种线程块簇与 DSMEM 的结合提供的性能，远高于任何全局内存回退方案或手动原子标志方法。

### 线程块对

借助现代 NVIDIA GPU，你可以在单个 GPC 内的多个 SM 上，恰好协同调度簇中的两个线程块，即线程块对（thread block pair，即 *CTA pair*）。通过将线程块编组为一个簇（例如图 10-13 所示的 2 块簇），共享数据的核函数可以使用 TMA 将分块搬移到每个块的共享内存中。

![图 10-13. 线程块对由两个线程块组合而成](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-13.png)

单个线程块可能缺乏足够的寄存器或共享内存容量来独自处理一个非常大的分块（例如一个 256 × 256 的矩阵子块）。通过在一个 GPC 内相邻的 SM 上配对两个线程块并使用 DSMEM，这两个块可以分担一个大分块上的工作，同时通过一个统一的共享内存区域共享数据，如图 10-14 所示。

![图 10-14. 线程块对（即 CTA pair）为 A * B 矩阵乘法加载分块作为操作数（来源：https://oreil.ly/kEZsv）](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-14.png)

在这里，对中的每个线程块都可以将操作数分块的一部分（例如 128 × 16）加载到其片上 SMEM 中以进行矩阵乘法。此外，每个线程块在 Tensor Memory（TMEM）中持有累加器的一部分（例如 128 × 256）。这使得 CTA pair 中的两个线程块能够协作处理单个分块，如图 10-15 所示。

![图 10-15. 配合 Tensor Core 与 TMEM 的线程块对](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-15.png)

线程块对允许跨两个物理 SM 的更大矩阵乘法。使用一对线程块实际上使分块大小翻倍，因为每个 SM 处理分块数据的一半。

SM 硬件使用 DSMEM 在两个 SM 之间共享操作数数据。DSMEM 减少了重复加载，改善了数据复用，并提高了算术强度。图 10-16 展示了在一个线程块簇中使用 DSMEM 在 SM 之间进行的这种数据共享。

![图 10-16. 在一个线程块簇中使用 DSMEM 在 SM 之间共享数据](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-16.png)

CUDA 为这些多 SM 操作提供了增强的多播屏障与同步原语。这一对块使用轻量级的 cluster.sync() 调用彼此同步。这样，当一个 SM 用完一个分块时，CTA pair 中相邻的 SM 就可以消费它。

每个分块都被馈入这一对线程块，而无需冗余的全局内存事务。这更好地占用了原本闲置的寄存器、共享内存 bank 和 Tensor Core。而且线程块对通过为一个分块提供翻倍的线程和共享内存，实现了更高的 Tensor Core 利用率。

任何性能提升都会透明地发生。从程序员的角度看，你只需启动一个两块的簇，并把它当作一个被拆分到两个线程块上的大型线程块来对待。例如在 CUTLASS 中，这被暴露为一个 2-SM UMMA（Unified MMA）GEMM 操作，它使用 DSMEM 和簇屏障来进行 CTA 间的数据共享。你只需为单个 GEMM 分块请求两个 SM，CUTLASS 就会自动处理簇启动、DSMEM 设置和线程块间同步。

如果你请求一个单个线程块无法处理的分块大小（比如用于 FP16 Tensor Core 操作的 128 × 128），CUTLASS 会自动将两个线程块分配为一对。这些 SM 会分担工作，并在底层依靠 DSMEM + cluster.sync() 来共享分块。这样，你就可以用 DSMEM 对 UMMA 操作进行成对重叠。

简而言之，线程块对与 DSMEM 以多 SM 协作的形式提供并行性。它们在线程块之间提供快速的片上数据共享与同步。这消除了许多以往需要全局内存交接或额外核函数启动的场景。这使许多多线程块算法受益，并简化了它们的实现。

## 用线程块簇减少全局内存流量

从性能角度看，DSMEM 可以显著削减冗余的全局内存流量，并实现更高的有效带宽。例如，在一个分块化 GEMM 中，簇内的多个线程块共享 A 矩阵或 B 矩阵的若干块，其中一个块可以从全局内存加载一个分块，并使用 TMA 将该分块多播到簇中的其他块。

TMA 引擎支持一种多播复制模式，当线程块属于同一个簇时，它可以直接馈入 DSMEM。一次从全局内存发起的 TMA 传输可以同时将数据放入每个参与块的共享内存中。这避免了冗余的 DRAM 取数。

借助 TMA 多播，GPU 确保被 L2 缓存的数据在一趟传输中被广播到每个簇成员（线程块）的共享内存中。因此，你避免了对同一分块的重复全局内存加载。这提高了带宽利用率，并缩减了 DRAM 流量——尤其是在许多块需要相同输入时，如图 10-17 所示。

![图 10-17. 对于这四个（2 × 2）线程簇，每个分块只加载一次，并被多播到每个簇中所有 CTA 的共享内存中（来源：https://oreil.ly/kEZsv）](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-17.png)

在这里，TMA 引擎执行一次从全局内存到 DSMEM 的多播，将分块广播到簇中每个线程块的 SMEM，并消除冗余的 DRAM 读取。TMA 多播通过一个张量映射描述符（tensor-map descriptor）配置，并作为面向 shared::cluster 的 cp.async.bulk.tensor 发出。幸运的是，CUTLASS/cuTe 和 Triton 等更高层的库会替你生成这些多播操作、张量映射描述符和批量张量复制。具体来说，这些库可以发出相关的 PTX 指令，包括带有张量映射操作数的 cp.async.bulk.tensor。它们可以发出面向 shared::cluster 进行多播的 PTX 指令。在这些情况下，硬件会将分块传递到每个线程块的 SMEM。

另一个常见的用例是多块归约或扫描。与其让每个块把它的部分和或扫描结果写出到全局内存、然后再启动一个单独的核函数来合并它们，一个线程块簇中的线程块可以把它们的部分结果写入 DSMEM。

在同一个簇内，你可以使用共享内存和一个簇级屏障 cluster.sync()，对这些部分结果执行快速的片上归约或前缀和。只有最终结果需要写到全局内存。这极大地减少了全局内存的读写。

通过将线程块簇、DSMEM 和 TMA 多播结合起来，你可以构建多阶段、细粒度的流水线，使大多数数据交换保持在片上。无论你是在为 GEMM 共享分块、为归约累加部分和，还是执行多块扫描，这些机制都能让你最小化 HBM 往返并最大化算术强度。

> 块作用域的 cuda::pipeline 不会同步其他块；簇级分发使用 TMA 多播或配合 cluster.sync 的 DSMEM。

考虑两个线程块 CTA 0 和 CTA 1，它们都需要 A 矩阵的同一个分块。没有 DSMEM 时，每个 CTA 都会发起自己的全局内存加载，通过两次取回相同的数据而浪费了 DRAM 带宽。

而有了 DSMEM，CTA 0 只需将该分块加载一次到它的共享内存中，然后经由片上 DSMEM 网络将其多播，使 CTA 1 可以直接从共享内存中读取它。如果一个簇由两个以上的 CTA 组成，同一个分块将在簇中所有线程块之间共享，但这仍然只需要一次全局 HBM 加载。表 10-4 展示了使用两个线程块的簇与两个独立线程块的对比。

表 10-4. DSMEM 对两块工作负载的性能影响

| 指标 | 两个独立 CTA（无 DSMEM） | 配合 DSMEM 的 CTA pair（2 块簇） |
| --- | --- | --- |
| 全局加载事务 | 2×（每个线程块加载分块） | 1×（分块只加载一次） |
| L2 缓存命中率 | 50% | 85% |
| CTA 间数据复用 | N/A（无复用） | 显著（分块被 CTA 1 复用） |
| 每 CTA 有效 DRAM 带宽 | 300 GB/s | 150 GB/s（减少 50%） |
| 核函数时间（相对） | 1.0× | 0.6×（加速 40%） |

在这里，我们看到使用配合 DSMEM 的线程块对（即 *CTA pair*）时，核函数执行有 40% 的加速。这与带宽的节省也是一致的。每线程块的 DRAM 带宽从 300 GB/s 下降 50% 到 150 GB/s，因为每个分块只被取回一次。L2 命中率从 50% 跃升到 85%，因为 TMA 硬件在片上执行了一次多播复制——从而避免了从全局内存的多次加载。

简而言之，通过多播共享数据，线程块簇让多个块能够以片上速度复用数据。这为访存受限的工作负载带来了可观的加速。

需要指出的是，DSMEM 和 L2 是并行工作的。这为块间数据共享提供了两条“通道”。这种双路径设计将 DSMEM 超快的簇级通信与 L2 更广泛的缓存覆盖结合起来。换言之，对于簇内本地地址，DSMEM 访问会绕过 L2 缓存。取而代之，DSMEM 使用专用的 SM 间网络，如图 10-18 所示。

![图 10-18. DSMEM 使用两个线程块簇之间的一个 SM 间、簇内本地网络进行数据交换](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-18.png)

远程 DSMEM 访问使用线程块簇互连进行路由，并与全局内存流量相区别。例如，当一个线程块使用 DSMEM 拉取一个分块时，它使用低延迟的共享内存传输。这确保了一个线程块簇中线程块间的数据共享与片上共享内存一样快。

如果某个分块不在 DSMEM 中，就必须从全局内存取回（遵循可能命中 L2 的正常缓存层次结构）。这样，如果一个分块已经从 DSMEM 中被逐出，线程块仍可以从 L2 缓存中取回该分块数据，而不必一路回到全局 DRAM。

因此，内存停顿很少见，占用率保持在高位，整体执行速度显著快于未经优化的、两个独立线程块的非 DSMEM 实现。

## 用线程块簇设计高效算法

线程块簇为并行化那些以往需要全局内存通信或多次核函数启动的工作负载提供了新的策略。例如，设想一个大型矩阵乘法，其输出分块因共享内存限制而过大，无法由单个线程块处理。

在过去，你可能会把工作拆分到两个线程块上，但那样每个块就都不得不通过全局内存来交换部分结果，而这相对缓慢且低效。否则，你就得启动一个单独的归约核函数来合并这两个线程块的输出。

有了线程块簇和 DSMEM，这两个块就可以组成一个簇，并直接共享一块联合的片上共享内存区域，使用硬件支持的原语无缝地合并它们的结果。DSMEM 硬件允许一个 SM 通过一个快速网络对另一个 SM 的共享内存执行加载/存储/原子操作。

精心设计算法以有效使用线程块簇很重要。同步开销很低，但它仍然非零。执行非常细粒度的数据共享可能得不偿失。

线程块簇屏障的工作方式类似于第 7 章介绍的 warp 级内建函数（例如 __shfl_sync()），即每个参与者都必须一起到达同步点。当你调用 cluster.sync() 时，簇中的所有线程块都必须到达那一行，任何块才能继续。

如果一个块提前完成了它的工作，它就只是等待。如果某个块因为它的线程走了不同的分支等原因而始终未到达屏障，整个簇就会死锁。

换言之，正如 warp 内建函数要求一个 warp 中的所有线程执行相同的指令路径以避免发散，线程块簇要求所有块在每个 cluster.sync() 调用之前都遵循相同的控制流。

由于这种锁步（lockstep）要求，块之间的细粒度数据共享必须与开销和死锁风险相权衡。正确执行时同步本身是廉价的，但如果哪怕只有一个块绕过或延迟到达 cluster.sync()，性能就可能受影响——或者更糟，核函数可能会挂起。

你应该组织代码，使每个块都齐步到达每个屏障——正如使用 warp 级内建函数时一个 warp 中的每个线程都必须到达屏障一样。这对于充分利用线程块簇低延迟的片上通信而不陷入死锁至关重要。

通常，当每个块都有相当数量的工作、可以独立运行直到需要一个同步或数据交换点时，线程块簇表现最佳。一旦同步发生、数据在片上完成传输，线程块就可以继续处理。

线程块簇对块稀疏矩阵操作尤其有效——这类操作常见于 LLM 中的稀疏注意力、模型剪枝和压缩。在这些情况下，各个块处理不同的非零区域，并共享它们的边界数据。

线程块簇对于多阶段归约也很有用，例如 transformer 层中的 softmax 和归一化步骤。这些操作需要使用线程块簇中每个线程块的部分结果进行一个最终合并步骤。

更一般地，当大型 GEMM 超出单个块的资源时，它们会从线程块簇中受益。当然，GEMM 是现代 LLM 中常见的 transformer 注意力、多层感知机（MLP）层和嵌入查找的核心。

不过，更大的簇大小可能会降低整体占用率。例如，一个 16 块的簇可能会为一个任务独占 16 个 SM。这可能会让该次核函数启动所需的其他任务可用的 SM 更少。建议从小簇开始，即两块或四块的簇，除非特殊情况需要更大的簇。一如既往，你应该用你的具体工作负载做性能分析，以确认跨线程块簇共享片上资源的收益胜过潜在的并行性损失。

> 使用簇启动时，请用 Nsight Compute 的启动统计信息和标准占用率 API（如 cudaOccupancyMaxActiveBlocksPer Multiprocessor）来验证活跃块和簇常驻情况。对于非可移植簇大小，请设置 cudaFuncAttri buteNonPortableClusterSizeAllowed（或使用 cudaLaunchKernelEx 属性传递 cudaLaunchAttrib uteNonPortableClusterSizeAllowed）。否则启动可能会失败，或测得的占用率很低。

## warp 专门化与线程块簇结合

现在让我们用一个线程块簇配合 CUDA Pipeline API 来重新审视 warp 专门化。我们使用一个线程块作用域的流水线作为簇的主导块来暂存复制，并使用 DSMEM 执行簇级屏障。这样，簇中的每个块都消费相同的输入分块，而无需从全局内存重新加载它们。

角色是在每个块内部按 warp 分配的。主导块的加载器 warp 每个分块执行一次协作式复制到它的共享内存中。在一个簇级屏障发布这些分块之后，每个块使用一个计算 warp 通过 DSMEM 读取主导块的分块，并计算一个不相交的行带。每个块中的一个存储器 warp 将该行带写回全局内存。块作用域的流水线仅由主导块用于异步复制。其他块不会在该流水线上等待。代码如下：

```
// Warp specialization across a thread-block cluster
// using DSMEM and a block-scoped pipeline
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <algorithm>
namespace cg = cooperative_groups;
#define TILE_SIZE 128
#define TILE_ELEMS (TILE_SIZE * TILE_SIZE)
// Compute a band of rows of the TILE_SIZE×TILE_SIZE product from DSMEM sources.
// Each lane processes rows [row_begin, row_end) in a 32-way striped loop.
__device__ void compute_rows_from_ds(const float* __restrict__ A_src,
                                     const float* __restrict__ B_src,
                                     float* __restrict__ C_dst,
                                     int row_begin, int row_end,
                                     int lane_id) {
    for (int row = row_begin + lane_id; row < row_end; row += warpSize) {
        for (int col = 0; col < TILE_SIZE; ++col) {
            float acc = 0.0f;
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                acc += A_src[row * TILE_SIZE + k] * B_src[k * TILE_SIZE + col];
            }
            C_dst[row * TILE_SIZE + col] = acc;
        }
    }
}
extern "C"
__global__ void warp_specialized_cluster_pipeline(
    const float* __restrict__ A_global,
    const float* __restrict__ B_global,
    float* __restrict__ C_global,
    int numTiles) {
    thread_block cta = this_thread_block();
    cluster_group  cluster = this_cluster();
    extern __shared__ float shared_mem[];
    float* A_tile_local = shared_mem;
    float* B_tile_local = A_tile_local + TILE_ELEMS;
    float* C_tile_local = B_tile_local + TILE_ELEMS;
    // Block-scoped pipeline used only by the cluster leader
    // to stage asynchronous copies
    __shared__
    cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipe_state;
    auto pipe = cuda::make_pipeline(cta, &pipe_state);
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int cluster_rank      = cluster.block_rank();
    const dim3 cluster_dims     = cluster.dim_blocks();
    const int  blocks_in_cluster = cluster_dims.x * cluster_dims.y *
                                   cluster_dims.z;
    // 1D cluster arrangement along x;
    // each iteration processes one tile per cluster
    auto loader = cooperative_groups::tiled_partition<32>(cta);
    for (int tile = blockIdx.x / cluster_dims.x; tile < numTiles;
         tile += gridDim.x / cluster_dims.x) {
        const size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;
        // Leader block’s loader warp stages A and B once for the entire cluster
        if (cluster_rank == 0 && warp_id == 0) {
            pipe.producer_acquire();
            cuda::memcpy_async(loader, A_tile, A_global + offset,
                               cuda::aligned_size_t<32>{TILE_ELEMS*sizeof(float)},
                               pipe);
            cuda::memcpy_async(loader, B_tile, B_global + offset,
                               cuda::aligned_size_t<32>{TILE_ELEMS*sizeof(float)},
                               pipe);
            pipe.producer_commit();
            // Make loads visible to leader CTA before publishing cluster-wide
            // wait for committed stage before publishing
            pipe.consumer_wait();
            pipe.consumer_release();
        }
        // Publish the leader’s tiles to every block via DSMEM
        cluster.sync();
        const float* A_src = cluster.map_shared_rank(A_tile_local, 0);
        const float* B_src = cluster.map_shared_rank(B_tile_local, 0);
        // Divide rows among blocks in the cluster
        const int rows_per_block = (TILE_SIZE + blocks_in_cluster - 1)
                                   / blocks_in_cluster;
        const int row_begin = std::min(cluster_rank * rows_per_block, TILE_SIZE);
        const int row_end   = std::min(row_begin + rows_per_block, TILE_SIZE);
        // Compute warp produces block’s band of rows into local shared memory
        if (warp_id == 1) {
            pipe.producer_acquire();
            compute_rows_from_ds(A_src, B_src, C_tile_local, row_begin, row_end,
                                 lane_id);
            pipe.producer_commit(); // publish C_tile_local
        }
        // Storer warp writes this block’s rows back to global memory
        if (warp_id == 2) {
            pipe.consumer_wait(); // observe C_tile_local
            for (int row = row_begin + lane_id; row < row_end; row += warpSize) {
                for (int col = 0; col < TILE_SIZE; ++col) {
                    C_global[offset + row * TILE_SIZE + col] =
                        C_tile_local[row * TILE_SIZE + col];
                }
            }
            pipe.consumer_release();
        }
        // All blocks finish this tile before the leader reuses its buffers
        cluster.sync();
    }
    // dynamic shared memory size: 3 * TILE_ELEMS * sizeof(float)
}
```

在这里，主导块使用一个块作用域的流水线，将每个分块进行一对协作式复制到它的共享内存中。簇级屏障通过 DSMEM 使这些数据对所有簇成员可见。

这种“加载一次、通过 DSMEM 共享”的模式在一次簇级屏障之后，使用 DSMEM 与 map_shared_rank 共享主导块的分块。它并不像我们前面介绍的那样，用张量映射描述符（tensor-map descriptor）、cp.async.bulk.tensor 和 shared::cluster 来执行 TMA 多播。这是一个需要理解的重要区别：在 DSMEM 模式中，如果没有 cluster.sync()，跟随块可能读到过期的主导块 SMEM 数据。因此，在调用 map_shared_rank() 之前必须先执行 cluster.sync()。

> 下面这些原则有助于你在 TMA 多播与 DSMEM 共享之间做选择：如果每个块的分块复用率很高（例如每个块多次访问同一个分块），或者簇大小（C）较大（8-16 个 CTA），则通常 TMA 多播胜出，因为这种模式只写入一次、随后在本地多次读取。如果 SMEM 紧张（例如分块很大），或者簇大小（C）较小（2-4 个 CTA）且每个跟随块只访问该分块一次，则 DSMEM 共享通常是更好的选择，因为它占用的内存足迹更小。

每个块都借助 map_shared_rank 直接从主导块的共享内存中计算一段不同的行带，并将各自的结果写回全局内存。这消除了整个簇范围内重复的全局加载，同时在最关键的地方保留了流水线的重叠优势。

主导块的加载 warp 调用 producer_commit()，将其本地阶段发布给该块。其他块并不等待主导块的流水线。相反，它们在簇级屏障之后观察主导块的分块，然后从分布式共享内存中读取。

这使得分块的加载、计算与存储能够跨多个线程块交错进行，从而将 SM 推向更高的利用率。表 10-5 对比了单块 warp 专门化核函数与本节介绍的线程块簇（多块）warp 专门化核函数。

表 10-5. 朴素分块、两阶段双缓冲、warp 专门化以及线程块簇流水线核函数的对比。

| 指标 | 朴素分块 | 两阶段双缓冲（double_buffer_pipeline） | warp 专门化（warp_specialized_pipeline） | 线程块簇流水线（warp_specialized_cluster_pipeline） |
| --- | --- | --- | --- | --- |
| 核函数执行时间 | 41.3 ms | 20.5 ms（相比朴素快 +2.01×） | 18.4 ms（相比两阶段提速 +10.2%） | 17.2 ms（相比 warp 专门化提速 +6.5%） |
| warp 执行效率 | 68% | 92%（相比朴素 +24%） | 96%（相比两阶段 +4%） | 97%（相比 warp 专门化 +1%） |
| warp 状态停顿百分比（共享内存与屏障等待） | 高 | 低 | 极低 | 极低（进一步降低） |
| L2 吞吐 | 80 GB/s | 155 GB/s（相比朴素 +94%） | 165 GB/s（相比两阶段 +6.45%） | 170 GB/s（相比 warp 专门化 +3%） |
| 吞吐可扩展性 | 扩展到 2–3 warps/SM | 扩展到约 6 warps/SM | 近乎线性地扩展到 SM 的 warp 上限（Blackwell 上每个 SM 64 个驻留 warp） | 完整跨线程块扩展，直到受限于 Blackwell 上每个 SM 64 个驻留 warp 的 SM warp 上限 |
| DRAM 读取吞吐相对核函数时长 | 重叠很差 | 重叠很好 | 重叠极佳 | 重叠极佳（即使在线程块交接时也是如此） |
| 指令数 | 1.7 B | 1.05 B（相比朴素 –38%） | ~1.00 B（相比两阶段 –4.76%） | ~0.98 B（相比 warp 专门化 –2%） |

在这里我们看到，线程块簇实现在 warp 专门化的基础上进一步改进——它在一次协作式启动中，将加载、计算和存储的角色分布到多个线程块上。这提升了整体的 SM 利用率，同时减少了执行时间和冗余的内存流量。

具体来说，线程块簇流水线核函数实现（warp_specialized_cluster_pipeline）相比单线程块的 warp 专门化核函数又挤出了约 1.2 ms（6.5%）。这是因为线程块簇版本在所有线程块之间交错进行分块加载、计算与存储。

SM 利用率达到约 97%，因为某个线程块中任何空闲的 SM 都可以被另一个执行不同流水线阶段的线程块保持忙碌。得益于更好的共享内存复用和更少的冗余加载，L2 加载吞吐峰值约为 170 GB/s。因此，簇能够避免重复的全局读取。使用 DSMEM 时，主导块只加载一次分块，其他块在调用 cluster.sync() 之后通过 map_shared_rank() 读取它。使用 TMA 多播时，一次来自全局内存的张量拷贝可以直接广播进每个块的共享内存。

指令数降至约 0.98 B，因为线程块级的流水线化进一步减少了在冗余全局读取上浪费的周期。吞吐可扩展性也得以维持，因为只要你有足够的线程块和 warp，流水线就能让它们全部保持饱和，直到达到你的 GPU 的每 SM warp 硬件上限，即 Blackwell 上每个 SM 64 个 warp。

需要注意的是，warp 专门化的单线程块版本和线程块簇版本最终都会撞上同一个“每 SM warp 数”的天花板，但它们在如何为这些 warp 供给数据以及如何同步方面存在差异。正如我们在表 10-5 中看到的，这种差异转化为线程块簇实现可测量的性能优势。

具体来说，在单线程块的 warp 专门化核函数中，每个线程块恰好分配三个 warp：一个负责将 A 和 B 的分块大小的数据块加载进共享内存，一个负责在该数据块上计算，一个负责将结果存回全局内存。一旦这三个 warp 完成各自的角色，线程块就前进到它的下一个分块，如此循环往复。

即使每个 SM 都有三个用于加载、计算和存储的活跃 warp，你也并未使 SM 的 warp 上限饱和。现代 GPU 允许每个 SM 最多 64 个并发 warp，而实际驻留量则受寄存器、共享内存和每 SM 块数的限制。

在总活跃 warp 数相近的情况下，单块 warp 专门化设计与协作式网格可以表现出相当的占用率，但内存行为不同。在单块设计中，每个块都会获取它所访问的任何输入分块的自有副本。当多个块在大致相同的时间复用同一个分块时，设备就会执行重复读取，消耗 L2 容量和 HBM 带宽。

线程块簇改变了这种模式。簇中的块可以通过分布式共享内存共享数据，并且可以将一次全局分块的多播接收进簇中所有块的共享内存。可以在主导块中使用块作用域的流水线，配合簇级同步与 map_shared_rank；或者在更倾向于从全局内存广播时，配置为多播。这消除了簇内的重复加载，并降低了对 L2 和 HBM 的压力。

cuda::pipeline 的作用域限定在创建它的块。一个块中的 producer_commit() 不会释放其他块中的 consumer_wait()。要在簇内跨块协调，先发布数据（例如写入主导块的共享内存），然后使用 cluster.sync()。随后跟随块用 map_shared_rank() 访问主导块的共享内存。可选的 TMA 多播可以从全局内存直接广播进每个块的共享内存。

至此，每个块都通过 DSMEM（或 TMA 多播）观察到主导块的分块并继续执行。块作用域的流水线并不显式同步其他块。cluster.sync() 与 DSMEM 语义提供了簇范围的可见性。

当设计得当、能高效使用 TMA 多播和 DSMEM 时，线程块簇流水线会精确地只获取每个分块一次，并将其多播进每个线程块的共享内存。这与从每个线程块冗余地重新加载数据形成对比。因此，尽管每个 SM 仍然无法超过其 64 个 warp 的上限，全局的线程块簇协调确保这些 warp 在重复加载上花费的时间大大减少，而在实际计算上花费的时间大大增加。

此外，在单线程块版本中，每个块一完成当前分块的计算，就立即加载它的下一个分块。如果某个线程块比另一个稍早完成计算，它会立即为下一个分块发起自己的加载——即使该分块已经在被另一个块加载。这会导致冗余的内存流量。

然而，在线程块簇流水线版本中，如果 SM 0 上的某个计算 warp 提前完成，它不会立即去获取自己的下一个分块。相反，它会在 consumer_wait() 处停顿，直到全局加载器——可能运行在 SM 7 上——为所有线程块把该分块加载进共享内存。换句话说，计算 warp 等待簇的那一次共享加载，而不是发起自己的冗余拷贝。

通过暂停到分块确保可用为止，参与某个线程块簇的每个 SM 都避免了空转或执行重复加载。这种跨线程块簇的对齐平滑了加载时间的波动，让每个 SM 的计算 warp 都能忙于处理每簇仅加载一次的数据。这提升了整体吞吐。

设想这样一个问题：它如此之大、网格如此充满工作，以至于在单块方法下每个 SM 都已经有活跃的加载、计算和存储 warp。在这种情况下，两个核函数之间原始的 warp 数量和每 SM 占用率看起来可能相近。当多个块复用相同的输入分块时，簇版本通过消除簇内重复的全局读取、并利用簇范围的同步来对齐各阶段，从而带来帮助。

这通常在复用密集的工作负载上带来适度的吞吐提升，但增益取决于分块复用、对齐和资源限制。两种设计都仍受相同的架构限制约束，例如驻留 warp、寄存器和共享内存。

这里所有的核函数都使用 CUDA 流水线来进行生产者与消费者的交接，而不调用块级的 __syncthreads。朴素分块核函数不重叠内存与计算。两阶段双缓冲流水线将拷贝与计算重叠，在访存受限的情形下能显著缩短求解时间。

warp 专门化流水线为加载、计算和存储分配各自独立的 warp，以避免隐式的全块等待。线程块簇变体通过分布式共享内存或多播在簇内跨块共享分块，使得一个分块每簇只获取一次，而不是每块获取一次。这些模式提升了利用率并减少了冗余的内存流量。

## 关键要点

以下是本章的关键要点，本章聚焦于在现代 GPU 上榨取峰值性能。这些要点将使核函数不再因 DRAM、warp 调度器和全局屏障而空闲：

*用流水线深度隐藏延迟* 使用两阶段（cuda::pipeline_shared_state<cuda::thread_scope_block, 2>）分块，将异步加载与计算重叠；当计算多于内存时（例如 Blackwell 这样的现代 GPU），再增加一个阶段（cuda::pipeline_shared_state<cuda::thread_scope_block, 3>）。这将有助于消除空闲的 warp。

*用 warp 专门化平衡工作负载* 当计算阶段占主导时，为加载、计算和存储分配各自独立的 warp，从而在现代 GPU 硬件上确保接近峰值的 warp 效率。

*用持久化核函数消除启动开销* 在一个设备端工作队列上运行单个长期存活的核函数，并对多阶段算法使用 grid.sync()。这将减少主机—设备往返和整体的启动成本。

*用线程块簇和 DSMEM 实现片上共享* 将线程块（CTA）分组为簇，使它们共享一段连续的片上缓冲区。对于簇范围的广播，使用 cp.async.bulk.tensor 配合 TMA 多播描述符。使用 TMA 多播将分块一次性广播到每个线程块。这将提升 L2 命中率并削减 DRAM 带宽。

*特别关注屏障语义* cluster.sync() 和 grid.sync() 都要求每个参与的线程块到达同一个同步点。控制流不匹配——或者簇大小过大——可能导致死锁或启动失败。

*先剖析再调优* 使用 Nsight Compute 判断你的核函数是访存受限还是计算受限。如果是访存受限，从两阶段流水线开始。如果是计算受限，考虑 warp 专门化或线程块簇化。

*在手动调优之前先验证编译器生成的流水线* 剖析并检查来自 PyTorch 编译器和 NVCC 等编译器所生成的框架代码。如果编译后的代码使用了基于 cuda::memcpy_async、producer_commit() 和 consumer_wait() 的异步流水线，那么手动调优很可能不会带来太多提速。

## 结论

本章讨论的技术有助于系统性地隐藏延迟并消除冗余加载。这使 GPU 在核函数执行的整个期间保持接近峰值的利用率。

warp 专门化流水线将加载、计算和存储操作重叠。协作组屏障（grid.sync() 和 cluster.sync()）有助于在不进行主机往返的情况下协调多阶段工作。而持久化核函数则在一个设备端队列上循环，以消除启动开销。

一如既往，你应当从剖析开始。如果全局内存停顿占主导，像双缓冲这样的两阶段异步拷贝流水线通常就足够了。如果计算 warp 仍然停顿，则切换到多阶段的 warp 专门化流水线，使加载、计算和存储 warp 能够以最小的争用运行。唯一的争用将出现在内存子系统。

> 请记住，并发在达到硬件（例如内存带宽）饱和之前都是有益的。超过这一点后，并行任务就会争抢吞吐。

对于多阶段归约或不规则任务，用单个持久化核函数加 grid.sync() 替代多次启动，以保持占用率。而当线程块需要相同的数据时（例如多头注意力），你可以组建一个线程块簇，使 DSMEM 和 TMA 对每个分块只加载一次——并将数据多播到其他线程块——而无需反复访问全局内存。这些技术会把性能推向更接近 GPU 的峰值理论极限。

在你继续阅读后续章节时，牢记这些原则很重要，因为它们适用于更多的优化。具体来说，你应当在 warp 和线程块层面重叠工作、直接在设备端同步（而不是在主机端），并在片上共享数据。这些是调优超高性能 GPU 工作负载的关键机制。

在下一章，我们将保留这些核内构建块——cuda::pipeline 双缓冲、warp 专门化角色以及线程块簇——并展示如何通过 CUDA streams 来驱动它们。目标是隐藏核函数之间以及主机 ↔ 设备通信之间的延迟，而不仅仅是在单个核函数内部。具体而言，我们将复用本章的核函数，并用 cudaMemcpyAsync、cudaMallocAsync/cudaFreeAsync 以及基于事件的同步，在多流水线中运行它们。这将有助于把整个系统推向在你的 AI 系统中的众多 GPU 上实现峰值性能。
