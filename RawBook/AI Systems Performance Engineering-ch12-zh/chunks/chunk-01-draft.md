# 第 12 章 动态调度、CUDA Graphs 与设备端发起的核函数编排

到目前为止，我们已经在单个核函数层面释放了计算与访存吞吐。现在，是时候对这些核函数进行编排，让 GPU 永不空闲。

在本章中，我们将从在主机上调度转向在设备本身上调度。我们会探索由快速的 L2 缓存原子操作驱动的动态工作队列，折叠掉重复的核函数启动，并用 CUDA Graphs 来批处理固定的流水线、把 CPU 握手降到最少。

接着，我们会用设备侧的图启动与动态并行（dynamic parallelism），把编排推得更远。它们让 GPU 自己决定接下来运行什么，而无需回调 CPU。

最后，我们会进入多 GPU 环境，重叠点对点拷贝、NCCL 集合操作、CUDA-aware MPI 与 NVSHMEM 的单边 put/get 操作。这样一来，成群的 GPU 就表现得像一个巨大的、共享内存的协处理器。例如，NVIDIA 的 DGX GB200 NVL72 系统把 36 颗 Grace CPU 和 72 颗 Blackwell GPU 连接进单一 NVLink 域，具备统一寻址，且该域内 CPU 与 GPU 合计的统一内存最高可达 30 TB。它可以在这个 72-GPU 域内，通过 NVLink fabric 实现跨设备的远程 HBM 访问。更大的 NVLink 网络拓扑可以延伸到单个机架之外。

在这一过程中，我们会把每种技术都与 roofline 分析联系起来，帮助你选对工具——streams、graphs、原子操作或动态核函数——来提高核函数的运算强度（operational intensity）。这将有助于提升整个工作负载的总体性能。

读完本章，你将理解动态的、设备侧的以及基于图的核函数编排技术，它们能在多 GPU 集群中让每一个 SM 都保持满负荷。

## 用原子工作队列做动态调度

线程之间工作分配不均，会让一些 SM 空闲，而另一些仍在忙碌。这浪费了计算资源，也降低了整体吞吐。

当不同线程或线程块由于依赖输入的循环或条件工作负载而处理数量不等的工作时，往往会出现不均衡。有些线程块很快完成，让它们的 SM 空闲下来，而另一些 SM 则继续执行运行更久的线程块。在拥有数百个 SM 的现代 GPU 上，如果工作分配不均，空闲时段可能让许多 SM 闲置。这会显著损害性能。

等到运行时间最长的那部分工作完成时，GPU 的一部分已经空闲了一段时间。这降低了实际达成的占用率（occupancy），因为许多周期在没有活跃 warp 的情况下运行。记住，你可以用 Nsight Systems 做剖析，把这些空闲间隙（idle gap）清晰地展示在 GPU 时间线上。

你还可以把活跃 SM 周期与总的已流逝 SM 周期做对比，以衡量利用不足的程度。Nsight Compute 把它作为单一指标提供，代表至少有一个 warp 处于活跃状态的时间占比。较低的活跃-流逝比表明许多周期在没有活跃 warp 的情况下运行。换句话说，GPU 经常处于空闲。

除了 Nsight Systems，你还可以用 Nsight Compute 来检查实际达成的占用率（每个 SM 平均活跃 warp 占硬件上限的比例）或 SM Active 周期百分比（至少有一个 warp 活跃的时间占比），以量化这种利用不足。

> 若要把时间线上的间隙与具体的代码段关联起来，可在关键的 GPU 工作周围插入 NVTX range 标记。

接下来，我们将讨论如何实现原子队列，以便在核函数内部动态分配工作。它们对于在所有 SM 间均衡任意工作负载、避免线程空闲非常重要。在此之前，我们需要先引入原子计数器（atomic counter）。

### 原子计数器

原子计数器是原子队列的基础，而原子队列可以实现动态工作分配。

在现代 GPU 上，全局原子操作在设备端的 L2 缓存中被服务并串行化。当目标缓存行常驻时，这比往返 DRAM 的延迟更低。原子计数器仍会带来延迟，并在争用（contention）下串行化。但无争用的 atomicAdd 操作由于保持在片上，因而极其快速。图 12-1 展示了两个线程对一个原子变量做递增的示例。

![图 12-1. 在直方图计算场景下，跨多个线程的超快片上原子内存加操作](AI%20Systems%20Performance%20Engineering-ch12_images/figure-12-1.png)

然而，atomicAdd 并非没有代价。它仍然有延迟，并且在争用下，会串行化那些等待同一内存地址的线程。因此，L2 也需要把这些更新串行化。这会制造一个热点，需要加以优化。Nsight Compute 可以帮你量化其成本。

在 Nsight Compute 的 Memory Workload Analysis 部分，你会看到 atomic_transactions 和 atomic_transactions_per_request。原子事务计数器代表 L2 缓存原子事务的总数，包括由争用引起的任何重放。原子事务每请求数（atomic transactions per request）这个指标，或称争用率，代表每条 atomicAdd 指令平均产生的 L2 事务数。

当每条 atomicAdd 恰好触发一次 L2 事务时，你的 atomic_transactions_per_request 会徘徊在 1.0 附近，这意味着它只付出了最低限度的代价。如果这个比率攀升到 1.0 以上，就说明线程在停顿并重试原子更新，而不是在做有用的工作。每一次重试都表明存在争用。

这里的优化是通过批量抓取工作来摊薄原子操作的成本。因此，与其让每个线程或每个 warp 各自执行一次 atomicAdd，不如让每次原子操作批量领取一组任务。下面是批大小为 32 时的优化前后对比：

```
// Before batching
int idx = atomicAdd(&queue_head, 1);
if (idx < N) process(data[idx]);
// After batching
const int batchSize = 32;
int start = atomicAdd(&queue_head, batchSize);
for (int i = start; i < start + batchSize && i < N; ++i) {
    process(data[i]);
}
```

现在，单次原子更新就把一整片工作——本例中是 32 个工作项——授予一个 warp（或线程块），之后才会再次触及计数器。你仍然每批付出一次 L2 事务，但在这两次之间做了 32 倍的有用工作。

在实践中，每个 warp 中只有一个线程执行这次 atomicAdd 来获取下一批的起始索引。该线程随后把它广播给 warp 中的其余线程（例如用 \_\_shfl_sync）。整个 warp 接着并行处理这 32 个工作项。这样每个 warp 只产生一次原子操作，而不是每个线程一次，从而大幅降低争用。

在 Nsight Compute 中，你会看到 atomic_transactions 骤降，你的事务每请求数回落到 1.0 附近。这证明你已经用持续的计算换掉了昂贵的争用。

在现代 GPU 上，L2 缓存原子操作异常快速，得益于很高的 L2 带宽，即便是 8 或 16 这样不大的批大小，也能消除大部分争用。话虽如此，务必要验证你没有只是把瓶颈转移到了别处。

为验证这项优化没有对其他性能指标产生负面影响，请使用 Nsight Compute 的 Warp Stall Reasons 和 Register Pressure 报告，确认你融合后的循环现在不会受制于寄存器溢出或共享内存 bank 冲突。

> 如果在这些优化之后原子操作依然很热，可以考虑其他设计，例如按块计数器（per-block counter）或对工作分配做分层归约。

简而言之，通过按原子操作批量领取工作，你能让 GPU 的众多 warp 忙于真正的计算。这与在单个跟不上节奏的计数器前排队形成鲜明对比。

### 原子队列

现在，让我们用一个全局原子计数器来协调一个动态工作队列。目标是用原子计数器和 atomicAdd 在所有 SM 间均衡任意工作负载，让任何线程或 warp 都不会闲置。图 12-2 展示了这种动态工作队列的一个示例。

![图 12-2. 把原子计数器与 atomicAdd 用作动态工作队列，以在各 SM 与 warp 之间均衡工作负载](AI%20Systems%20Performance%20Engineering-ch12_images/figure-12-2.png)

在接下来的代码示例（computeKernel）中，每个线程根据 idx % 256 计算不同数量的迭代。idx % 256 值较小的线程做的工作很少，而 idx % 256 值较大的线程会做大量工作。由于这种不均衡，线程在不同时间完成，一些 SM 会空闲下来，等待运行时间最长的线程完成。下面是使用每线程静态、不均衡工作负载的代码：

```
// uneven_static.cu
#include <cuda_runtime.h>
#include <cmath>
__global__ void computeKernel(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Each thread does a variable amount of work based on idx
        int work = idx % 256;
        float result = 0.0f;
        for (int i = 0; i < work; ++i) {
            result += sinf(input[idx]) * cosf(input[idx]);
        }
        output[idx] = result;
    }
}
int main() {
    const int N = 1<<20;
    float* h_in = nullptr;
    float* h_out = nullptr;
    cudaMallocHost(&h_in, N * sizeof(float));
    cudaMallocHost(&h_out, N * sizeof(float));
    for (int i = 0; i < N; ++i) h_in[i] = float(i) / N;
    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(256), grid((N + 255) / 256);
    computeKernel<<<grid, block>>>(d_in, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    return 0;
}
```

> PyTorch 并没有用于设备侧动态工作分配的高层 API，因此我们需要用一个自定义 CUDA 核函数来实现它。为简洁起见，此处从略。

在接下来展示的优化后动态任务派发版本中，一个单独的全局计数器（位于设备内存中）被用作 warp 级的工作队列。我们用批量原子操作把这个计数器变成一个持久的 warp 级工作队列。这样，提前完成的 warp 会立刻抓取下一批工作，而不是空转：

```
// uneven_dynamic.cu
#include <cuda_runtime.h>
__device__ unsigned int globalIndex = 0;
// Warp-batched dynamic queue: 1 atomic per active warp
__global__ void computeKernelDynamicBatch(const float* input,
                                          float* output,
                                          int N) {
  // lane id in [0,31]
  int lane = threadIdx.x & (warpSize - 1);
  while (true) {
    // Elect an active leader each iteration (handles divergence safely)
    unsigned mask = __activemask();
    int leader = __ffs(mask) - 1;
    // Warp leader atomically grabs a contiguous batch for the whole warp
    unsigned int base = 0;
    if (lane == leader) {
      base = atomicAdd(&globalIndex, warpSize);
    }
    // Broadcast starting index to all active lanes in the warp
    base = __shfl_sync(mask, base, leader);
    unsigned int idx = base + lane;
    if (idx >= (unsigned)N) break;  // dynamic termination
    // Hoist invariants out of the variable trip-count loop
    // Note: You can also use __sincosf on Blackwell
    float s = sinf(input[idx]);
    float c = cosf(input[idx]);
    int   work = idx % 256;
    float result = 0.0f;
    #pragma unroll 1
    for (int i = 0; i < work; ++i) {
      result += s * c;
    }
    output[idx] = result;
    // loop continues until counter >= N
  }
}
int main() {
    const int N = 1 << 20;
    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    // Host buffers (pinned) for a realistic data path
    float *h_in = nullptr, *h_out = nullptr;
    cudaMallocHost(&h_in,  N * sizeof(float));
    cudaMallocHost(&h_out, N * sizeof(float));
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>(i % 1000);
    }
    // Copy inputs to device
    cudaMemcpy(d_in, h_in, N * sizeof(float),
        cudaMemcpyHostToDevice);
    // Reset global counter
    unsigned int zero = 0;
    // If you call this kernel repeatedly (e.g., in a loop),
    // reset 'globalIndex' to 0 before each launch.
    cudaMemcpyToSymbol(globalIndex, &zero,
        sizeof(unsigned int));
    // Launch with 256 threads per block
    dim3 block(256), grid((N + 255) / 256);
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    computeKernelDynamicBatch<<<grid, block, 0, stream>>>(d_in, d_out, N);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaDeviceSynchronize();
    // Copy results back and clean up
    cudaMemcpy(h_out, d_out, N * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    return 0;
}
```

每个 warp 从全局队列中原子地领取下一批大小为 warpSize（32 个线程）的任务，并在循环中处理它们。这确保了任何 SM 都不会空闲。这段代码实现了用单个全局原子工作队列完成的动态工作分配。

在这里，每个 warp 反复从那个全局计数器中拉取下一批的 base 索引。每个 warp 的第一个线程（if (lane==0)），称为 warp leader，执行一次原子加，用 base=atomicAdd(&globalIndex, warpSize) 获取这一连续块的起始索引。随后它用 **shfl_sync(**activemask(), base, 0) 把这个 base 广播给 warp 中的其余线程，正如本章前面所述。

换句话说，不再是每个线程被绑定到一个固定的元素索引，现在每个 warp 都从一个共享计数器中抓取一个连续的任务块。然后它用 idx = base + lane 进行计算。

该 warp 中的所有线程都为其动态获取的索引执行相同的 sin/cos 循环。因此，工作不再是按线程预先分配的。相反，工作是在运行时通过全局原子队列拉取并均衡的。

记住，if (idx >= N) 的边界检查会在没有更多工作时让 warp 退出。这防止了越界内存访问。除此之外，warp 中的每个线程都会执行与静态版本中完全相同的 sin/cos 循环。

在一个简单的微基准测试中，N = 1 << 20 且 work = idx % 256，静态分配的核函数耗时约 200 ms，而动态队列版本大约在 100 ms 内运行完成。这 2× 的加速是消除 SM 空闲时间并减少原子争用的结果。Nsight Compute 把活跃 SM 周期定义为至少有一个活跃 warp 的已流逝周期占比。

加速幅度会随工作不均衡程度而变化，但只要你的剖析显示出 warp 空闲停顿、实际占用率偏低，或时间线上因每任务运行时长不均而出现可见的间隙，动态工作分配就是一项值得探索的优化。在这些场景中，尤其是在中等不均衡的情况下，你通常能获得 10%–20% 的加速。

> 在极端不均衡的情况下，你只需把静态索引替换为原子驱动的工作队列，就能获得这 2× 的加速。对于轻度不均衡，原子操作和 shuffle 的开销可能会抵消收益。

简而言之，动态工作分配确保了近乎均匀的 SM 利用率，因为每个 warp 都持续抓取并处理新任务，直到计数器超过 N。这与许多 warp 远早于最慢的那些 warp 完成、从而让硬件资源闲置的情形形成对比。

## CUDA Graphs

当你的流水线由多个核函数、拷贝、stream-event 记录和回调组成时，每次迭代都在主机上逐个启动它们，仍然会带来 CPU 开销。CUDA Graphs 让你把整套工作流一次性捕获下来，并以基本为零的 CPU 开销反复重放。图 12-3 对比了不使用（上）与使用（下）CUDA Graphs 时的核函数启动。

![图 12-3. 不使用（上）与使用（下）CUDA Graphs 时的核函数启动时间线](AI%20Systems%20Performance%20Engineering-ch12_images/figure-12-3.png)

为什么要用 CUDA Graphs？首先，它们削减了启动开销。多个小核函数或拷贝基本上可以用一次 CPU 调用启动。其次，它们让 GPU 上的调度更好。工作作为一批提交，因此 CUDA 驱动有可能减少操作之间的一些内部延迟。

此外，使用 CUDA Graphs 时，依赖关系是预先已知的，因此 CPU 在其间同步的需求更少。在内存传输的场景中，CUDA Graph 确保异步拷贝与核函数执行被正确地关联为依赖，而无需任何手动同步。它并不会比普通 streams 天然地更多地重叠拷贝与核函数，但 CUDA Graph 会让它们的执行更顺畅。

### PyTorch、推理引擎与 CUDA Graphs

像 PyTorch 这样的 AI 框架会在底层利用 CUDA Graphs 来处理深度学习模型中的静态部分。具体来说，PyTorch 支持一个 torch.cuda.Graph 上下文，用于捕获一段操作序列。此外，PyTorch 持续优化其内部实现，把 CUDA Graphs 用于代码中可预测的部分。

像 vLLM 和 NVIDIA 的 TensorRT-LLM 这样的高性能推理引擎，也可以通过把模型的执行捕获为一组针对不同序列长度范围和输入批大小的预定义图，来利用 CUDA Graphs。当启用图捕获（graph capture）时，这些系统常常会对输入进行分桶或填充，以匹配所支持的图批大小，从而让捕获的图能以固定形状重放。这可以显著降低大规模生产推理工作负载的延迟。

例如，你会在启动或模型加载时为每个批大小各捕获一个 CUDA Graph。然后，在运行时，你会启动与传入请求批大小相匹配的那个预捕获图。

> PyTorch 编译器的 mode='reduce-overhead' 可能会把符合条件的代码段包进 CUDA Graphs 以减少启动开销，但需满足捕获要求，例如静态张量地址和仅限 CUDA 的区域。它并不保证对所有代码路径都进行图化。而且它可能因池化缓冲区而增加内存占用。请始终通过剖析来确认在你的模型上确有收益。

### CUDA Graphs 的内存池

一个重要的考量是使用 CUDA Graphs 时的内存管理。CUDA Graph 内部的内存操作遵循与 CUDA streams 中相同的规则。如果你在捕获内部分配 GPU 内存，那次分配就会成为图执行的一部分。

你通常应当避免在图内部分配 GPU 内存，而是在图之外预先分配内存。许多框架（如 PyTorch）在使用 CUDA Graphs 时采用静态内存池（static memory pool），如图 12-4 所示。使用静态内存池可以避免内存分配成为捕获到的图序列的一部分。

![图 12-4. PyTorch 为 CUDA Graphs 使用静态内存池](AI%20Systems%20Performance%20Engineering-ch12_images/figure-12-4.png)

虽然 CUDA Graphs 不会让单次内存拷贝或核函数执行变得更快，但它们可以在图内部自动重叠相互独立的数据传输与计算——类似于 CUDA streams。这消除了每次迭代的 CPU 调度，并且由于依赖图预先已知，这才得以实现。

### 用 CUDA Stream 捕获 CUDA Graph

要捕获一个图，你需要在一个 stream 上调用 cudaStreamBeginCapture()，把你所有的内存传输（cudaMemcpyAsync()）、核函数启动、事件（cudaEventRecord()）和回调（cudaLaunchHostFunc()）入队，然后调用 cudaStreamEndCapture() 来创建一个 CUDA Graph 定义（cudaGraph_t）。

之后，CUDA 驱动就可以在每次迭代用 cudaGraphLaunch() 启动这个 CUDA Graph。因为 CUDA 驱动预先知道整个依赖图，它会直接在 GPU 上重放预先构建好的 stream 序列。这带来的启动开销极小。

> cudaGraphExecUpdate 将在下一节讨论，它允许对已捕获的图做有限的改动，以应对迭代之间尺寸、维度或指针发生变化的场景。当输入尺寸变化时这很有用，因为你可以直接更新图的节点参数，而不必为每个新的输入尺寸重新捕获一整张新图。

即便你的流水线只有一部分是重复的，CUDA Graphs 也能捕获那一部分。例如，如果你总是执行一次 Host → Device 拷贝，接着两个核函数，再接着一次 Device → Host 拷贝，你就可以只捕获那一段子图，并用一次函数调用来重放它。

要重放这个 CUDA Graph，你提供 stream 句柄，GPU 便会在没有额外 CPU 指令的情况下执行这一系列操作。这与核间并发直接相关，因为 CUDA Graphs 让你能够通过混合异步拷贝、细粒度事件屏障和核函数来维持复杂的重叠行为——同时彻底移除 CPU 这个瓶颈。

通常，你会做一次重放的试运行以确保正确性。你会通过创建一个 cudaGraphExec_t 可执行图来实例化该图，然后用一次图重放调用来启动它。当你启动这个已捕获的图时，运行时会在 GPU 上按正确顺序执行所有操作。

为展示 CUDA Graphs 的用法，考虑一个简单的核函数序列。这里我们展示一段用 C++ 和 PyTorch 捕获并启动 CUDA Graph 的代码片段：

```
cudaStream_t stream;
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
cudaGraph_t graph;
cudaGraphExec_t instance;
cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
// Enqueue operations on 'stream' as usual
kernelA<<<grid, block, 0, stream>>>(d_X);
kernelB<<<grid, block, 0, stream>>>(d_Y);
kernelC<<<grid, block, 0, stream>>>(d_Z);
cudaStreamEndCapture(stream, &graph);
cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
// Now 'instance' can be launched in a loop
for (int iter = 0; iter < 100; ++iter) {
    cudaGraphLaunch(instance, stream);
    // No per-kernel sync needed; graph ensures dependencies
}
cudaStreamSynchronize(stream);
// (Destroy graph and instance when done)
cudaGraphExecDestroy(instance);
cudaGraphDestroy(graph);
```

在这段伪代码中，我们在一个 CUDA stream 上开始捕获，在该 stream 上依次启动三个核函数（A、B 和 C），然后结束捕获以获得一个图。接着我们实例化这个图。

现在，我们可以通过调用 cudaGraphLaunch(instance, stream) 任意多次来重放整个序列 A → B → C。我们每次迭代只付出一次启动的成本，而不是三次单独的启动。而且 GPU 会按照记录时的相同顺序执行核函数 A、B 和 C，在本例中即背靠背地执行。我们会在循环之后进行同步，以确保所有迭代都已完成。

如前所述，像 PyTorch 这样的高层 AI 框架支持 CUDA Graphs。PyTorch 为 Python 开发者提供了接近原生的 CUDA 性能，而无需深入了解 CUDA C++。这里，我们展示 PyTorch 的 torch.cuda.Graph 上下文管理器，用于捕获并重放操作：

```
import torch, time
X = torch.randn(1<<20, device='cuda')
# Define operations for reference
def opA(x): return x * 1.1
def opB(x): return x + 2.0
def opC(x): return x.sqrt()
# Persistent buffers for pointer stability
static_x = torch.empty_like(X)
static_y = torch.empty_like(X)
static_z = torch.empty_like(X)
static_w = torch.empty_like(X)
# Warm up once to initialize CUDA kernels and caches
_ = opC(opB(opA(X)))
torch.cuda.synchronize()
# Seed the static input before capture
static_x.copy_(X)
# Capture the graph
g = torch.cuda.CUDAGraph()
stream = torch.cuda.Stream()
torch.cuda.synchronize()
with torch.cuda.graph(g, stream=stream):
    # Record A then B then C using out parameters to avoid allocations
    torch.mul(static_x, 1.1, out=static_y)
    torch.add(static_y, 2.0, out=static_z)
    torch.sqrt(static_z, out=static_w)
# Replay the captured graph 100 times
for i in range(100):
    # If inputs change, copy new values into static_x before replay
    # static_x.copy_(new_X)
    g.replay()
```

在生产环境中，你应当分配持久的输入和输出缓冲区，让捕获到的核函数看到固定的内存地址。例如，在捕获之前创建 static*y = torch.empty_like(X)，然后在图内部写 static_y.copy*(opA(X))。这样可以避免在捕获期间进行分配，并满足 CUDA Graph 指针的稳定性规则。PyTorch CUDA Graphs 要求重放使用与捕获张量相同的内存地址。

在这个 PyTorch 示例中，我们定义了操作 opA、opB 和 opC。在实践中，它们可以是神经网络层或任何 GPU 操作。我们运行一次预热（warm up）遍历 opC(opB(opA(X)))，以确保所有核函数、内存分配和库上下文（例如 cuBLAS/cuDNN）都预先初始化好。这是必要的，因为 CUDA Graphs 捕获不会记录这些惰性初始化步骤。

> 跳过预热遍历可能导致你的图捕获失败，或在惰性初始化发生时引入意料之外的停顿。

我们首先通过运行一次序列来预热 GPU，以初始化所有核函数和库。然后，我们通过在一个全新的 CUDA stream 上，用 with torch.cuda.graph(g, stream=stream): 这个 Python 上下文管理器代码块把前向（A）、变换（B）和后向（C）操作包起来，将它们捕获进单个 torch.cuda.CUDAGraph。捕获之后，调用 g.replay() 100 次，就能以每次迭代一次主机调用来启动整个 A → B → C 流水线。结果汇总于表 12-1。

表 12-1. CUDA Graphs 对每次迭代开销的影响

| 指标                         | 使用 CUDA Graphs 之前           | 使用 CUDA Graphs 之后              |
| ---------------------------- | ------------------------------- | ---------------------------------- |
| 每 100 次迭代的 CPU 启动调用 | 300 separate kernel launches    | 100 graph replays（每次迭代 1 次） |
| 主机同步调用                 | 300 cudaDeviceSynchronize calls | 0                                  |
| 核函数之间的平均 GPU 空闲    | 每次迭代 ~3 µs 间隙             | 0 µs（连续背靠背执行）             |
| 端到端每次迭代延迟           | ~1.00 ms                        | ~0.75 ms（25% faster）             |

> 注：所有指标表格中的数值均为示意性质，用于解释概念。有关不同 GPU 架构上的实际基准测试结果，请参见 GitHub 仓库。

这里我们看到，CUDA Graphs 消除了每次迭代的 CPU 调度和主机-设备握手。这是因为每次迭代的 GPU 工作被批量合并进单个 g.replay() 调用，而不是三次单独的核函数启动。结果，迭代执行快了 25%，因为 CPU 只是发出轻量级的重放命令，并对 GPU 保持完全异步。

使用 CUDA Graphs 时有一些常见的陷阱，妥善处理它们很重要。例如，如果你的工作负载尺寸发生变化，一个已捕获的图可能就不再有效。这将需要重新捕获，或调用 cudaGraphExecUpdate，我们会在下一节介绍。

某些 CUDA API 调用——如分配内存和主机-设备同步原语——通常不应包含在图捕获中。虽然现代版本的 CUDA Graphs 支持在已捕获的图内部进行有限的内存管理操作，但仍建议在捕获之前完成所有内存分配。你还必须确保图中使用的数据保持在相同的内存地址。

> 内存必须在图执行期间保持在相同内存地址，这一要求正是 PyTorch 这类框架在使用 CUDA Graphs 时采用静态内存池的一个主要原因。例如，PyTorch 提供了 torch.cuda.graph_pool_handle() API，用于创建一个专用内存池以进行指针稳定（pointer-stable）的 CUDA 图捕获。使用一个单独的分配器池可确保张量地址在多次捕获与重放之间保持固定。这满足了指针稳定性（pointer stability）的要求。在迭代之间，通过把数据拷贝进静态张量来更新输入。不要在每次迭代都重新分配张量。

你还应当避免在 CUDA Graphs 捕获内部包含任何主机侧回调或不受支持的操作。这包括诸如 print()、随机数生成器（RNG）调用、嵌套捕获和新的内存分配之类的操作。这是因为图必须记录一段纯粹的、确定性的 GPU 工作序列。

此外，捕获中使用的所有张量都必须已经以固定形状分配在固定地址上。在捕获期间调整尺寸或调用 cudaMalloc 会破坏这个图。

### 动态图更新

一旦你记录了一个 CUDA Graph，就不必仅仅因为某些启动参数发生变化而把它丢弃。与其重新捕获，不如调用图更新 API，直接在现有图中更新 grid/block 维度、指针地址或核函数参数。图更新 API 包括 cudaGraphExecUpdate 以及更底层的 cudaGraphExecKernelNodeSetParams。

cudaGraphExecUpdate 让你能用一个相同形状的新核函数节点替换掉某个核函数节点。例如，你可以换入一个不同的融合核函数实现——只要它形状相同即可。CUDA 运行时会校验你的改动，并让你立即重放修改后的图。这避免了一次完整捕获的成本。

> 截至本文撰写时，你还不能任意添加或删除节点。如果某次更新违反了现有图所能处理的范围，运行时会返回一个错误。在这种情况下，你必须捕获一张新图。

例如，考虑我们前面那个使用最大批大小的三核函数图 A → B → C。在每次推理循环中，你只需更新核函数 B 的启动维度以匹配当前批，然后重放同一个图。这为半静态工作负载降低了开销——在这类负载中，整体流水线是固定的，但少数几个参数可能会变化。

在实践中，一个典型的工作流分三步。首先，用预期的最大尺寸（例如最大的批）捕获一个模板图。举例来说，你可能用最大批大小 128 来捕获你的图。之后，如果来了一个批大小为 64 的请求，你就调用 cudaGraphExecUpdate 把启动参数调整为 64——或许还会把内存指针更新为一个更小的缓冲区。

使用 cudaGraphExecUpdate 可以让你在核函数参数、grid/block 维度或内存地址发生变化时，无需重建图。而且它只需几微秒，因此你保住了 CUDA Graph 重放那种低于 100 µs 的快速启动开销。此外，你还保有在运行时调整关键参数的灵活性。请注意，不兼容的改动会返回一个错误状态并需要重新捕获。

如果你确实需要改变图的结构——比如指定不同数量的核函数——你可以退回到“重新捕获再更新”的工作流。在这种情况下，你用 cudaStreamBeginCapture 和 cudaStreamEndCapture 把代码的一次迭代包起来，以构建新图。然后在后续运行中用更轻量的 cudaGraphExecUpdate 做小幅调整。

实际上，动态图更新让你能够完全在 GPU 上创建“参数化”或有条件的执行路径。每当你有一个高频的 GPU 工作循环、但只有少数几个变化的参数（如批大小）时，你都可以捕获一次、快速更新，并同时享有最小的 CPU 开销以及你的使用场景所需的适应性。

### 设备端发起的 CUDA Graph 启动

既然你已经理解了如何以低开销从 CPU 启动并适配一个已捕获的流水线，下一步就是把 CPU 从启动决策中彻底移除。借助设备端发起的（device-initiated）CUDA Graph 启动，一个正在运行的 GPU 核函数可以直接在设备上触发一张预先记录好的图，从而完全绕开主机。

要启用设备侧启动，先照常在主机上捕获图。然后用 cudaGraphInstantiate 实例化它，并传入 cudaGraphInstantiateFlagDeviceLaunch。实例化之后，在任何设备侧启动之前，先在一个主机 stream 上用 cudaGraphUpload 上传这个可执行图。

接着，你用 cudaGraphUpload 把图上传到 GPU 内存。这必须在 GPU 能够启动该图之前完成。（在未上传图的情况下尝试设备端启动会导致错误。）
