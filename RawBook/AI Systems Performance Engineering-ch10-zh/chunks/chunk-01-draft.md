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

warp 专门化的一个有趣模式是使用三种不同类型的 warp，如“loader”、“compute”和“storer”warp。loader warp 把分块推入流水线的队列。compute warp 在每个分块上运行计算核函数。而 storer warp 写出结果，如图 10-5 所示。

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

如表 10-3 所示，双缓冲方式与 warp 专门化方式都大幅超越了朴素分块核函数。double_buffer_pipeline 通过重叠分块加载与计算把运行时间减半，而 warp_specialized_pipeline 通过避免任何隐式的块级等待再增加了 10.2% 的加速。只有专职的“compute”warp 会发生停顿。

指令数从朴素版本的 17 亿降到两阶段流水线的 10.5 亿，减少了 38%，再进一步降到 warp 专门化核函数的约 10 亿，又减少了 4.76%。

L2 加载吞吐从朴素分块的 80 GB/s 攀升到两阶段方式的 155 GB/s（+94%），再到 warp 专门化核函数的 165 GB/s（相较两阶段 +6.45%）。这是因为这个 warp 专门化核函数专门用一个 warp 把每个分块一次性加载进共享内存——然后把这单份拷贝多播给所有计算 lane。这消除了任何残余的冗余 L2 读取。因此，在分块与双缓冲之后，流水线中几乎所有冗余都已被移除。