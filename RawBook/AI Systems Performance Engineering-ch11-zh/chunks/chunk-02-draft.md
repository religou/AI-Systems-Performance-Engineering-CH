_显式流（用_ cudaStreamCreateWithFlags() _创建）_ 显式流是相互独立的队列，只有当你显式插入依赖（例如使用 cudaStreamWaitEvent()）时才会彼此同步。

下面给出一些最佳实践，帮助你把握默认流与显式流（非默认流）的用法：

_绝不要把性能关键的核函数放进 stream 0（传统默认流），除非你有意串行化所有 GPU 工作_ 哪怕只有一个走散的核函数或拷贝进入 stream 0，也会拖住所有其他活跃的流。举例来说，较老的 CUDA API 可能会隐式使用 stream 0。例如，当你在不指定显式流的情况下调用 CUDA 驱动 API 时，很可能就会用到传统流。这也是应当迁移到较新 API、或始终显式指定流的又一个理由。

_启用每线程默认流_ 如果你的应用使用多个 CPU 线程、每个线程都在向 GPU 排入工作，就应使用 PTDS。这可以避免这些线程的默认流之间出现隐藏的主机端屏障。在现代系统上，你也可以用 cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) 来创建一个不与 stream 0 同步的流。

_创建并管理显式的、非默认的流_ 显式流是一项最佳实践，因为它们允许核函数与内存拷贝重叠。始终向 <<<...>>>()、cudaMemcpyAsync() 或 cudaMallocAsync() 传入一个非默认的 cudaStream_t。这样可以保证不会有隐式的默认流同步来干扰你的流水线化工作流。

_使用_ cudaStreamWaitEvent() _来协调细粒度依赖，而不是_ cudaStreamSynchronize() 你应当使用 stream 事件，而不是在默认流上使用 cudaStreamSynchronize()。只在定义明确的全局节点（例如一个模型训练 epoch 的结尾）才调用 cudaStreamSynchronize()，以避免拖住不相关的流。

_对你的流标志保持显式_ 如果你要启用 PTDS，请在任何 CUDA 调用之前设置好设备标志。否则你仍处于传统默认流模式。任何对 stream 0 的涉及都会制造一道全局屏障。

简而言之，传统默认流（stream 0）始终扮演一道全局屏障。你排入其中的任何工作都会等待、并强制所有其他流完成；而只要 stream 0 有挂起的工作，所有其他流也都会等待。

要避免这些看不见的停顿，就不要把性能关键的核函数或拷贝放进 stream 0。相反，应使用 cudaStreamCreateWithFlags 创建你自己的具名流，并把一切都启动到这些流中，让它们能够独立运行。

如果你的程序有多个 CPU 线程——每个线程都发出各自的 GPU 工作——你还应启用每线程默认流。启用 PTDS 后，每个线程的“默认”流就不再与其他线程的默认流、也不再与 stream 0 同步。

这样一来，即使那些没有显式创建新流的代码，也不会意外阻塞任何其他线程的工作。在所有情况下，只要你想让两个操作重叠（例如一次拷贝和一个核函数），就应给它们各自独立的、显式的非默认流。如此便可避开 stream 0 的隐藏全局同步规则，让 GPU 以最大并行度运行。

## 用 events 与 callbacks 做细粒度同步

即便多个流与 DMA 引擎能够重叠，也总有一些时候，某个流的操作必须等待另一个流。设想一对生产者—消费者流。生产者 stream 0 正在加载并准备数据，供消费者 stream 1 处理。

在这种情形下，很容易想到在主机端同步这两个流、用一次完整的 cudaDeviceSynchronize() 阻塞 CPU。然而，这会阻塞 CPU，直到 GPU 上所有流的所有操作全部完成。这对性能非常不利。你也可以使用 cudaStreamSynchronize()（如图 11-5 所示），但这会一直阻塞，直到该流队列中的所有操作完成。

![图 11-5. 使用 cudaStreamSynchronize() 会阻塞 CPU，直到该 stream 中的所有操作都已同步](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-5.png)

相反，你可以使用 CUDA events，为流提供一种粒度细得多的同步机制。有了 CUDA events，你在产出数据的流中记录一个 cudaEvent_t，然后让消费者流去等待这个事件。

例如，你会在 stream 0 上启动一个生产者核函数，并在数据就绪可供消费时调用 cudaEventRecord(doneEvent, stream0)。随后在 stream 1 中，你会在启动消费者核函数之前调用 cudaStreamWaitEvent(stream1, doneEvent, 0)。这样一来，只有 stream 1 会停下来等待该事件被记录——主机线程和所有其他流都会继续执行，如图 11-6 所示。

![图 11-6. 使用 CUDA events 进行细粒度同步](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-6.png)

除了用 CUDA events 在流间协调之外，你还可以用它们在 GPU 设备与 CPU 主机之间通信。为此，你会在主机端用 cudaLaunchHostFunc() 注册一个主机回调。

假设你需要在某个 GPU 核函数于 stream 0 中一完成，就立刻在主机端回收一个自定义内存池。在这种情况下，你会在主机端用 cudaLaunchHostFunc() 注册一个回调，并指定你所关心的事件。然后在 stream 0 上启动该核函数。

当核函数完成时，它会记录该事件，主机随即运行回调函数、更新 CPU 上的内存池。这一切都无需轮询循环，也无需一次完整的 cudaDeviceSynchronize()。

一旦 GPU 工作完成，CUDA 运行时便会在一个主机线程上执行回调函数。这让 CPU 保持空闲，直到你恰好需要它的那一刻——从而避免浪费主机周期、也避免阻塞不相关的 GPU 工作。

你不应在用 cudaLaunchHostFunc() 启动的主机回调内部调用任何设备端 GPU API。举例来说，如果你在回调内部调用 cudaMemcpy()，就可能死锁，因为该回调运行在由 CUDA 运行时管理的主机线程上。在其中调用 CUDA API 会导致死锁，因为设备可能正在等待该回调完成。

> 回调应当仅限于 CPU 端任务，例如释放或回收主机内存。这条建议可以避免这样一种环形死锁：回调试图在 GPU 上启动新工作，而与此同时 GPU 正在等待该回调完成。

## 用 CUDA events 做跨流同步

当多个流并行运行时，我们常常需要在它们之间进行协调与同步。CUDA events 是跨流同步（cross-stream synchronization）的主要机制，既不会拖住 CPU，也不会拖住整个设备。

事件就像一个标记，由某个流在特定节点记录下来。其他流、甚至主机，都可以等待这个标记，从而得知某个事件何时发生（例如某个核函数已完成）。与会一直阻塞到所有流完成的完整 cudaDeviceSynchronize() 不同，事件允许在流之间进行细粒度的排序。

流可用于确保每个 transformer 层的数据在计算之前都已就位。它们还能改进流水线并行——确保 GPU 0 已产出某个张量之后，GPU 1 才去消费它。而这一切发生的同时，其他独立工作都不会闲置。

例如，设想一组四个流（Streams A–D），它们对操作进行流水线化，并在某些情况下彼此依赖。我们可以用 CUDA events 来强制这些依赖（见图 11-7）。

为确保 stream B 在 stream A 产出数据之前不会开始处理，我们可以在 stream A 中记录一个事件，让 stream B 先等待它再继续。类似地，stream D 可以如图所示等待 stream B。通过这样串联事件，我们在流之间维持正确的排序，同时仍能并发地运行不同的批次。

这种基于事件的同步在深度学习框架中被大量用于让梯度计算与 all-reduce 通信重叠。计算流在梯度就绪时记录一个事件，通信流等待该事件以启动一次 NCCL 操作，从而让通信与剩余计算重叠。

简而言之，CUDA events 轻量且针对设备信号做了优化。当某个流到达其命令队列中的特定节点时便会记录一个事件。而其他流可以高效地轮询/等待它。它们让我们能够在流之间编排复杂的依赖图，而无需强制全局等待。它们还提供了在多流 LLM 工作负载中实现流水线化执行所必需的控制力。

![图 11-7. 用 events 同步 CUDA streams（source: https://oreil.ly/MynOA）](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-7.png)

> NVIDIA 持续改进事件的时间粒度、并降低事件记录开销。因此，事件也是性能剖析的良好选择，因为它们会在 GPU 时间线上记录事件时间戳、可用于测量执行时间等。

## 把核内 warp 专门化与核间 CUDA streams 结合的流水线（Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)）

第 10 章演示了如何用 warp 专门化和 CUDA Pipeline API 在单个核函数内隐藏内存延迟。在那一节里，我们启动了一个单一的、常驻的、warp 专门化的核函数，其中每个线程块被拆分为三种不同类型的 warp：loader、compute 和 storer warp。这些 warp 共享一块连续的共享内存，以及一个两阶段（双缓冲）的 CUDA Pipeline（<cuda::pipeline>）对象来协调各自的工作。

让我们保留那个 warp 专门化的设备核函数，现在改用各自独立的 CUDA streams 驱动多个批次通过它。其结果是一个如下的两级流水线：

_核内重叠（每个块内部）_ loader ↔ compute ↔ storer 使用各自独立的 pipeline 并发运行。

_核间重叠（跨批次之间）_ 当 stream 0 在批次 _t_ 上计算时，stream 1 DMA 加载批次 t+1，而 stream 2 返回批次 t–1 的结果。这用到了非阻塞流、固定主机内存和流序内存分配器，以便分配/拷贝不会串行化其他工作。

如果多个块需要同一个 tile，第 10 章描述的线程块簇 + DSMEM 路径可以消除跨块的冗余全局加载。不过，协作式/簇启动会与其他协作式核函数串行化。这里所用的流模式针对的是主机 ↔ 设备的重叠，并适用于非协作式核函数。

> 如果你的瓶颈是设备上的 tile 复用，就使用线程块簇。如果瓶颈是主机 ↔ 设备的通信与批处理，就使用 CUDA streams。我们稍后会展示如何把二者结合，但这是初步决策的一个良好起点。

下一个核函数每个块使用三个 warp（0 = loader，1 = compute，2 = storer）和两个 pipeline 来乒乓推进各阶段。因此，它需要 6 _ TILE_SIZE _ sizeof(float) 的动态共享内存（[A0|B0|C0|A1|B1|C1]）。值得注意的是，我们改为使用两个独立的、块作用域的 pipeline，每个深度为 2。一个用于 loader → compute（pipe_lc）的交接，另一个用于 compute → storer（pipe_cs）。

此外，我们使用双缓冲共享内存，这样核函数就能在计算 tile _i_ 的同时加载 tile i+1、并存储 tile i–1。这就是为什么在接下来的代码里你会看到两个 cuda::pipeline 对象，以及共享内存中的“[A|B|C] × 2 stages”。这两个 pipeline 是一个核内选择，用于加深每个核函数内部的重叠。（注：你可以对任一种核函数风格使用流。）代码如下：

```
#include <cooperative_groups.h>     // thread_block, etc.
#include <cuda/pipeline>            // CUDA Pipeline API
namespace cg = cooperative_groups;
// shmem bytes = 2(stages)×3(buffers)×TILE_SIZE×sizeof(float)=6×TILE_SIZE×4
//             = 6 * 1024 * 4 = 24,576 B (24 KB) << 227 KB SMEM
// We keep this at 1024 (versus going higher) as a safe starting point
// (good occupancy balance)
#define TILE_SIZE 1024  // one tile = 1,024 floats per buffer (1-Dimension)
    // Alignment / size guards for vectorized copies
    static_assert((TILE_SIZE % (32 * 4)) == 0,
                  "TILE_SIZE must be multiple of 128 for float4 vectorization");
    // If you cannot guarantee 16B alignment or sizes, handle
    //  the tail/ragged edges with a fallback 4B loop.
// Three warps per block: 0 loads, 1 computes, 2 stores.
// Two block-scoped pipelines (each depth=2) implement ping-pong across tiles.
__global__ void warp_specialized_two_pipelines(
    const float* __restrict__ A_global,
    const float* __restrict__ B_global,
    float*       __restrict__ C_global,
    int          numTiles)
{
    thread_block cta = this_thread_block();
    // Stage s∈{0,1}: [A_s | B_s | C_s], each length TILE_SIZE
    extern __shared__ float shared_mem[];
    // loader -> compute pipeline (2 in-flight stages)
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> state_lc;
    auto pipe_lc = cuda::make_pipeline(cta, &state_lc);
    // compute -> storer pipeline (2 in-flight stages)
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> state_cs;
    auto pipe_cs = cuda::make_pipeline(cta, &state_cs);
    const int warp_id = threadIdx.x >> 5;   // 0,1,2
    const int lane_id = threadIdx.x & 31;
    // Prime first tile handled by this block
    const int first = blockIdx.x;
    if (warp_id == 0 && first < numTiles) {
        const int stage0 = first & 1;
        float* A0 = shared_mem + stage0 * 3 * TILE_SIZE;
        float* B0 = A0 + TILE_SIZE;
pipe_lc.producer_acquire();
#pragma unroll
for (int chunk = 0; chunk < TILE_SIZE; chunk += 32 * 4) {
    cuda::memcpy_async(
        cta,
        reinterpret_cast<float4*>(A0 + chunk) + lane_id,
        reinterpret_cast<const float4*>(A_global + size_t(first) *
          TILE_SIZE + chunk) + lane_id,
        sizeof(float4),
        pipe_lc);
    cuda::memcpy_async(
        cta,
        reinterpret_cast<float4*>(B0 + chunk) + lane_id,
        reinterpret_cast<const float4*>(B_global + size_t(first) *
          TILE_SIZE + chunk) + lane_id,
        sizeof(float4),
        pipe_lc);
}
pipe_lc.producer_commit();
    }
    // Walk a strided set of tiles; ping-pong stage by tile parity.
    for (int tile = blockIdx.x; tile < numTiles; tile += gridDim.x) {
        const size_t offset = size_t(tile) * TILE_SIZE;
        const int    stage  = tile & 1;
        float* A_buf = shared_mem + stage * 3 * TILE_SIZE;
        float* B_buf = A_buf      + TILE_SIZE;
        float* C_buf = B_buf      + TILE_SIZE;
        // Loader prefetches next tile while compute/store work on current.
        if (warp_id == 0) {
            const int next = tile + gridDim.x;
            if (next < numTiles) {
                const int next_stage = next & 1;
                float* A_next = shared_mem + next_stage * 3 * TILE_SIZE;
                float* B_next = A_next + TILE_SIZE;
                pipe_lc.producer_acquire();
                #pragma unroll
                for (int chunk = 0; chunk < TILE_SIZE; chunk += 32 * 4) {
                    cuda::memcpy_async(
            cta,
                        reinterpret_cast<float4*>(A_next + chunk) + lane_id,
                        reinterpret_cast<const float4*>(A_global + size_t(next) *
                          TILE_SIZE + chunk) + lane_id,
                        sizeof(float4),
                        pipe_lc);
                    cuda::memcpy_async(
                        cta,
                        reinterpret_cast<float4*>(B_next + chunk) + lane_id,
                        reinterpret_cast<const float4*>(B_global + size_t(next) *
                          TILE_SIZE + chunk) + lane_id,
                        sizeof(float4),
                        pipe_lc);
                }
                pipe_lc.producer_commit();
            }
        }
        // Compute consumes loader output and signals storer
        if (warp_id == 1) {
            pipe_lc.consumer_wait();
            #pragma unroll
            for (int chunk = 0; chunk < TILE_SIZE; chunk += 32) {
                C_buf[chunk+lane_id] = A_buf[chunk+lane_id]+B_buf[chunk+lane_id];
            }
            pipe_lc.consumer_release();
            pipe_cs.producer_acquire();
            pipe_cs.producer_commit();
        }
        // Storer waits on compute and writes back
        if (warp_id == 2) {
            pipe_cs.consumer_wait();
            #pragma unroll
            for (int chunk = 0; chunk < TILE_SIZE; chunk += 32) {
                C_global[offset + chunk + lane_id] = C_buf[chunk + lane_id];
            }
            pipe_cs.consumer_release();
        }
    }
    // Launch requirements: blockDim.x = 96 (3 warps), gridDim as needed,
    // dynamic shared memory = 6 * TILE_SIZE * sizeof(float).
}
```

在此，我们要指出一些值得关注的性能要点。首先，我们要求 A0、B0、A_global + base 和 B_global + base 满足 16 字节对齐。如果某条路径不是 16 字节对齐的，我们就对未对齐的序言/收尾退回到 4 字节的循环。

> 每 lane 16 字节（float4/int4）符合现代 GPU 的最佳实践，可在 warp 层级达成 128 字节的合并访问。这有助于降低流水线开销。

其次，我们假设 TILE_SIZE 是 128（32 lanes × 4 floats）的倍数。如果不是，就用标量循环处理尾部。此外，我们对 float4 使用 cuda::memcpy_async。这既保留了 pipeline 的异步语义，又把每个 tile 的拷贝计数从每 lane 4× 降到 1×。

下面是主机端驱动程序，它在非阻塞流之间轮转各批次，并使用固定主机内存和流序内存分配器（cudaMallocAsync ÷ cudaFreeAsync）。它会启动前一段代码中的 warp_specialized_two_pipelines：

```
#include <cstdio>
#include <cuda_runtime.h>
#define TILE_SIZE   1024
#define NUM_STREAMS 2
#define BATCHES     8
// Device kernel
__global__ void warp_specialized_two_pipelines(
    const float* __restrict__ A_global,
    const float* __restrict__ B_global,
    float*       __restrict__ C_global,
    int          numTiles);
int main() {
    // Create nonblocking streams (do NOT use legacy default stream)
    cudaStream_t s[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);
    // Pinned host buffers so cudaMemcpyAsync truly overlaps
    float *hA = nullptr, *hB = nullptr, *hC = nullptr;
    const size_t bytesPerBatch = TILE_SIZE * sizeof(float);
    cudaMallocHost(&hA, BATCHES * bytesPerBatch);
    cudaMallocHost(&hB, BATCHES * bytesPerBatch);
    cudaMallocHost(&hC, BATCHES * bytesPerBatch);
    // Initialize inputs
    for (int b = 0; b < BATCHES; ++b) {
        for (int i = 0; i < TILE_SIZE; ++i) {
            hA[b * TILE_SIZE + i] = float(i);
            hB[b * TILE_SIZE + i] = 1.0f;
        }
    }
    // Enqueue batches in a round-robin across streams
    for (int b = 0; b < BATCHES; ++b) {
        cudaStream_t st = s[b % NUM_STREAMS];
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cudaMallocAsync(&dA, bytesPerBatch, st);   // stream-ordered allocator
        cudaMallocAsync(&dB, bytesPerBatch, st);
        cudaMallocAsync(&dC, bytesPerBatch, st);
        cudaMemcpyAsync(dA, hA + size_t(b) * TILE_SIZE, bytesPerBatch,
                        cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(dB, hB + size_t(b) * TILE_SIZE, bytesPerBatch,
                        cudaMemcpyHostToDevice, st);
        const dim3 block(96);         // 3 warps: loader/compute/storer
        const dim3 grid(1);
        const size_t shmem = 6 * TILE_SIZE * sizeof(float); //[A0|B0|C0|A1|B1|C1]
        // Each batch is one tile in this 1-D example
        warp_specialized_two_pipelines<<<grid, block, shmem, st>>>(
            dA, dB, dC, /*numTiles=*/1);
        cudaMemcpyAsync(hC + size_t(b) * TILE_SIZE, dC, bytesPerBatch,
                        cudaMemcpyDeviceToHost, st);
        cudaFreeAsync(dA, st);        // stream-ordered free
        cudaFreeAsync(dB, st);
        cudaFreeAsync(dC, st);
    }
    // Clean up
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(s[i]);
        cudaStreamDestroy(s[i]);
    }
    cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC);
    return 0;
}
```

在这里，每个流承载着自己的一串序列：allocate → H2D → kernel → D2H → free。因为分配与拷贝是按流顺序排入的，且主机缓冲区是固定的，所以 GPU 的拷贝引擎可以让 H2D/D2H 与另一个流的 SM 计算重叠。这就是核间层，它与第 10 章引入的、由 warp 专门化所产生的核内重叠互为补充。

具体来说，这个例子展示了共享内存持有两组、每组三个长度均为 TILE_SIZE 的缓冲区，从而在准备下一个 tile 的同时可以使用其中一组。一个两阶段的 <cuda::pipeline> 在各 tile 之间提供双缓冲，并强制正确的排序，使 loader、compute 和 storer warp 能够在不同的 tile 上重叠。核函数在循环之前先给第一个 tile 预热，然后在计算 tile i 的同时预取 tile i 加 gridDim.x。

单独来看，这个 warp 专门化核函数通过在每个 tile 内重叠三种角色来隐藏内存延迟。然而，单个 warp 专门化核函数一次只能处理一批 tile。

为了让 GPU 忙于处理多个批次——并跨这些批次重叠 H2D 传输、核函数计算与 D2H 传输，我们在各自独立的 CUDA streams 中启动这同一个 warp 专门化核函数的多个实例。这将为每个批次使用流序内存分配器。

这第二层流水线化，即核间并发，让我们能够把接连而来的小批量（mini-batch）一波波送过 GPU。这样一来，当一个批次的核函数在 stream 0 中计算时，另一个批次的数据仍在 stream 1 中抵达。而与此同时，前一个批次的结果可能正在 stream 2 中流回主机。

在实践中，我们会挑选少量的流，两个或三个，并循环使用它们。每个流用 cudaMallocAsync 分配自己的设备缓冲区，用 cudaMemcpyAsync 异步拷贝输入数据，启动 warp 专门化核函数来处理这些缓冲区，把结果异步拷回主机，然后用 cudaFreeAsync 释放缓冲区。

因为这些操作中的每一个都被排入了特定的流，所以它们都可以与其他流中的等价操作重叠。在带有多个拷贝引擎和流序内存分配器的现代 GPU 上，这一模式能够通过同时压满芯片的每一个部分，来大幅提升利用率。实际能达成的重叠受限于拷贝引擎数量（查询 cudaDeviceProp::asyncEngineCount）、带宽以及核函数占用率。

一旦某个核函数中的 loader warp 正在等待一次全局内存取数，下一批次的 H2D 拷贝就已在拷贝引擎上在途（in-flight）。一旦前一批次的 storer warp 正在回写全局内存，另一个流中的分配器就可以为下一批次抓取内存，而无需强制一次全局同步。

本质上，第 10 章的 warp 专门化例子教会了我们如何让单个核函数通过在一个线程块内重叠 load/compute/store 来隐藏其内存延迟。本章的多流例子在此基础上更进一步，展示了如何通过在整条流水线上重叠主机 ↔ 设备传输、计算和设备 ↔ 主机传输，把许多这样的核函数——每个处理一个不同的批次——同时送入 GPU。

现代 GPU 拥有多个拷贝引擎，并执行异步内存分配。它们与 Tensor Core 一道并行工作，构成一种两级流水线化策略——核内的 warp 专门化与核间的流。这些机制让我们的核函数能够为 LLM 工作负载逼近硬件的峰值利用率。

## warp 专门化 + 线程块簇 + CUDA streams（Warp Specialization with Thread Block Clusters and CUDA Streams）

现在我们要把本章——以及此前各章——的所有内容整合起来，通过多个流驱动多个在途（in-flight）的小批量，而每个流都启动一个协作式的、线程块簇化的、warp 专门化的核函数。这代表了在最新 GPU 硬件上进行 CUDA 性能优化的巅峰。

> 尽管我们出于完整性考虑覆盖这一主题，但需要指出的是，由于其复杂性，这种技术组合在非常专门化的研究项目和对超低延迟敏感的推理引擎之外很少见到。不过它仍然值得讲述，因为它把我们目前学到的许多概念融进了同一个代码示例。

第 10 章展示了在单个线程块内部的 warp 专门化，也展示了用 DSMEM 把 warp 专门化与线程块簇结合起来。这样一来，一个主导线程块只加载一次 tile——而线程块簇中的每个块都从这份共享的片上副本进行计算。这消除了重复的全局加载。

这个例子复用了完全相同的模式：主导块用一个块作用域的 pipeline 来暂存这些拷贝。所有块都通过 cluster.map_shared_rank 进行读取，因此它继承了数据复用的收益。

在前面那个带 CUDA streams 的 warp 专门化例子中，每个线程块都在自己的共享内存区域内负责全部三个流水线阶段——loader、compute 和 storer。现在，让我们把那个较早的实现扩展为使用一个线程块簇 pipeline。这样一来，loader、compute 和 storer warp 就分布在整个线程块簇上。这与之前那个局限于单个线程块的实现形成对比。

线程块簇加 warp 专门化并配合 CUDA streams 的例子代码如下。在这个例子中，我们取 NUM_STREAMS = 2，以便主机可以在各自独立的 CUDA streams 中排入两次独立的启动。我们复用第 10 章“warp 专门化与线程块簇”一节中相同的 warp_specialized_cluster_pipeline 实现，并加入 CUDA streams：

```
// Warp specialization across a thread block cluster using DSMEM
// and a block scoped pipeline, launched from multiple CUDA streams
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <algorithm>
namespace cg = cooperative_groups;
#define TILE_SIZE   128   // 3×TILE_ELEMS×4 bytes=196,608 bytes < 227 KB SMEM
#define TILE_ELEMS  (TILE_SIZE * TILE_SIZE)
#define NUM_STREAMS 2
#define CLUSTER_BLOCKS 4  // blocks per cluster along x (tune to device)
// ---- Device helpers ----
__device__ void compute_rows_from_ds(const float* __restrict__ A_src,
                                     const float* __restrict__ B_src,
                                     float*       __restrict__ C_dst,
                                     int row_begin, int row_end, int lane_id)
{
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
// Clustered, warp-specialized kernel (leader loads once, others consume DSMEM)
extern "C"
__global__ void warp_specialized_cluster_pipeline(
    const float* __restrict__ A_global,
    const float* __restrict__ B_global,
    float*       __restrict__ C_global,
    int numTiles)
{
    thread_block cta      = this_thread_block();
    cluster_group cluster = this_cluster();
    extern __shared__ float smem[];
    float* A_tile_local = smem;
    float* B_tile_local = A_tile_local + TILE_ELEMS;
    float* C_tile_local = B_tile_local + TILE_ELEMS;
    // Leader uses a block-scoped pipeline to stage A/B exactly once per tile
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2>
        pipe_state;
    auto pipe = cuda::make_pipeline(cta, &pipe_state);
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int cluster_rank       = cluster.block_rank();
    const dim3 cluster_dims      = cluster.dim_blocks();
    const int blocks_in_cluster  = cluster_dims.x*cluster_dims.y*cluster_dims.z;
    // Each iteration handles one tile per cluster (1-D cluster along x)
    auto loader = cg::tiled_partition<32>(cta);
    for (int tile = blockIdx.x / cluster_dims.x; tile < numTiles;
         tile += gridDim.x / cluster_dims.x) {
        const size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;
        // Leader block’s loader warp stages A and B once for the entire cluster
        if (cluster_rank == 0 && warp_id == 0) {
            pipe.producer_acquire();
            cuda::memcpy_async(loader, A_tile_local, A_global + offset,
                               cuda::aligned_size_t<32>{TILE_ELEMS
                               * sizeof(float)}, pipe);
            cuda::memcpy_async(loader, B_tile_local, B_global + offset,
                               cuda::aligned_size_t<32>{TILE_ELEMS
                               * sizeof(float)} pipe);
            pipe.producer_commit();
            // Make visible inside leader before publishing
            pipe.consumer_wait();
            pipe.consumer_release();
        }
        // Publish to all blocks in the cluster
        cluster.sync();
        const float* A_src = cluster.map_shared_rank(A_tile_local, 0);
        const float* B_src = cluster.map_shared_rank(B_tile_local, 0);
        // Divide rows among blocks in the cluster
        const int rows_per_block = (TILE_SIZE + blocks_in_cluster - 1)
                                    / blocks_in_cluster;
        const int row_begin = std::min(cluster_rank * rows_per_block, TILE_SIZE);
        const int row_end   = std::min(row_begin + rows_per_block, TILE_SIZE);
        // Compute warp produces this block’s band into local SMEM
        if (warp_id == 1) {
            compute_rows_from_ds(A_src, B_src, C_tile_local,
                                 row_begin, row_end, lane_id);
        }
        // Ensure the storer sees computed rows
        cta.sync();
        // Storer warp writes band back to global
        if (warp_id == 2) {
            for (int row = row_begin + lane_id; row < row_end; row += warpSize){
                for (int col = 0; col < TILE_SIZE; ++col) {
                    C_global[offset + row * TILE_SIZE + col] =
                        C_tile_local[row * TILE_SIZE + col];
                }
            }
        }
        // Done with this tile; allow leader to reuse buffers
        cluster.sync();
    }
    // Dynamic shared memory size: 3 * TILE_ELEMS * sizeof(float)
}
// ---- Host driver: stream-staged batches + clustered kernel launch ----
void launch_warp_specialized_cluster_pipeline_multistream(
    const float* h_A,        // Host input A: length = numBatches * batchLength
    const float* h_B,        // Host input B: length = numBatches * batchLength
    float*       h_C,        // Host output C: length = numBatches * batchLength
    int          batchLength,// elems per batch; must be multiple of TILE_ELEMS
    int          numBatches)
{
    // Basic validation
    if (batchLength % TILE_ELEMS != 0) {
        fprintf(stderr, "batchLength must be a multiple of TILE_ELEMS (%d)\n",
                TILE_ELEMS);
        return;
    }
    const int numTiles = batchLength / TILE_ELEMS;
    // Nonblocking streams avoid legacy default-stream barriers
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    // Size and launch geometry
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    // grid.x must be a multiple of CLUSTER_BLOCKS
    const int blocksPerGrid = prop.multiProcessorCount * CLUSTER_BLOCKS;
    const dim3 blockDim(96);  // 3 warps: loader/compute/storer
    const size_t shmemBytes = 3ull * TILE_ELEMS * sizeof(float);
    // Cluster attributes
    cudaLaunchAttribute attr[2]{};
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim = dim3(CLUSTER_BLOCKS, 1, 1);
    attr[1].id = cudaLaunchAttributeNonPortableClusterSizeAllowed;
    attr[1].val.nonPortableClusterSizeAllowed = 1;
    // Enqueue batches round-robin across streams
    for (int b = 0; b < numBatches; ++b) {
        cudaStream_t st = streams[b % NUM_STREAMS];
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        const size_t bytes = static_cast<size_t>(batchLength) * sizeof(float);
        // Stream-ordered allocator avoids global sync
        cudaMallocAsync(&dA, bytes, st);
        cudaMallocAsync(&dB, bytes, st);
        cudaMallocAsync(&dC, bytes, st);
        // H2D (host pointers should be pinned for true overlap)
        cudaMemcpyAsync(dA, h_A + static_cast<size_t>(b) * batchLength, bytes,
                        cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(dB, h_B + static_cast<size_t>(b) * batchLength, bytes,
                        cudaMemcpyHostToDevice, st);
        // Clustered launch config
        void* args[] = { &dA, &dB, &dC, (void*)&numTiles };
        cudaLaunchConfig_t cfg{};
        cfg.gridDim  = dim3(blocksPerGrid);
        cfg.blockDim = blockDim;
        cfg.dynamicSmemBytes = shmemBytes;
        cfg.stream = st;
        cfg.attrs = attr;
        cfg.numAttrs = 2;
        // Launch the clustered kernel (cooperative/cluster launch)
        cudaKernel_t k;
        cudaGetKernel(&k, warp_specialized_cluster_pipeline);
        void* fptr = reinterpret_cast<void*>(k);
        cudaLaunchKernelExC(&cfg, fptr, args);
        // D2H and cleanup
        cudaMemcpyAsync(h_C + static_cast<size_t>(b) * batchLength, dC, bytes,
                        cudaMemcpyDeviceToHost, st);
        cudaFreeAsync(dA, st);
        cudaFreeAsync(dB, st);
        cudaFreeAsync(dC, st);
    }
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
}
```

> 这个例子使用了一次用 cudaLaunchKernelExC 配置的簇启动。协作式启动（包括簇启动）需要足够的资源，才能按启动约束让其各块保持常驻。这往往会限制并发，但在资源允许时，协作式核函数仍可与其他工作交错进行。并发并不被保证，因为它依赖于拓扑与启动方式。（注：在资源限制与启动约束的前提下，其他流中的数据传输与核函数仍可重叠。）
