与第 10 章一样，我们在启动核函数时指定一个簇维度，使各个块加入同一个 cluster_group，从而可以访问彼此的分布式共享内存。主导块（leader block）通过块作用域的 cuda::thread_scope_block 暂存其拷贝，而所有块都使用 cluster.map_shared_rank 读取主导块的 tile。

借助簇作用域与分布式共享内存，加载阶段在主导块中每个簇只运行一次，而计算与存储则在簇内各个块上并发执行。与之前一样，每个 warp 的 warp_id 决定其角色，warp 以持久化方式循环遍历所有 tile，在同步的轮转中依次执行取数、计算与存储。

通过加入 CUDA streams，并在各自独立的 CUDA streams 中启动本核函数的独立副本（NUM_STREAMS = 2），我们让 GPU 持续忙于处理多批输入数据。在每个 stream 中，我们执行以下步骤：

1. 用 cudaMallocAsync 为每个线程块分配按批次划分的设备缓冲区。

2. 用 cudaMemcpyAsync 将输入从主机暂存到设备。

3. 用 cudaLaunchKernelExC 启动带簇的 warp 专门化核函数。

4. 用 cudaMemcpyAsync 将输出拷回主机。

5. 用 cudaFreeAsync 释放设备缓冲区。

由于每个 stream 都会将自己的协作式启动连同异步拷贝和释放一并入队，即便一次只运行一个协作式核函数，主机也能让多个 mini-batch 保持在途（in flight）。请记住，GPU 会将协作式核函数启动串行化，因为每个协作式核函数会同时占满所有线程块（这一限制会在代码示例之后再次强调）。

使用多个 stream（NUM*STREAMS = 2）仍然让我们能够把主机端的分配与拷贝和上一个核函数的执行重叠起来。例如，当 stream 0 的线程块簇处理 tile \_n* 时，stream 1 可以异步分配缓冲区（cudaMallocAsync）并把 tile n+1 拷入设备内存，而 stream 2 则可以把 tile n–1 的结果写回主机。

> 协作式核函数必须占满它在 GPU 上所需的每一个线程块槽位（CTA）。这会阻止任何其他协作式启动同时运行，因为所有这些 CTA 资源都已被占用。

在实践中，stream 0 发出其协作式启动并开始执行，而 stream 1 可以立即将自己的启动入队。但第二个启动会一直处于挂起状态，直到 stream 0 的核函数结束。

不过，一旦 stream 0 完成，stream 1 的启动会立即开始。这是因为它的输入早已由 cudaMemcpyAsync 暂存完毕——而它的缓冲区也已由 cudaMallocAsync 分配好。

将线程块簇、核内 warp 专门化（三阶段的 cuda::thread_scope_cluster）与核间 CUDA streams 结合起来，我们便构建出跨越多个线程块的两层流水线。这种双层流水线能把硬件推向峰值。

在第一层，线程块簇流水线让 loader、compute 与 storer warp 跨所有线程块协作。线程块簇可以利用 TMA 多播（multicast），把一次全局内存 tile 传输复制到簇内每个块的共享内存中。多播局限于线程块簇内部。本质上，线程块簇对每个 tile 只取一次。在第二层，多个 stream 让主机端的分配、拷贝和核函数启动彼此相互隐藏。

简而言之，这些组合技术确保内存延迟在网格级和主机到设备级都被隐藏起来。这会把现代 GPU 上的硬件利用率推向接近峰值——并为 LLM 工作负载最大化吞吐。

> 在许多场景中，像 DRAM 带宽这样的内存层级瓶颈，以及像 NCCL all-reduce 这样的模型并行通信，会主导整个工作负载。因此，一旦你的核函数已接近饱和 HBM 带宽，并且你的 stream 已经隐藏了 CPU → GPU 延迟，那么在此之上再叠加 warp 专门化、进而再叠加线程块簇所能多换来的那几个百分点的 SM 利用率，往往难以抵消其高昂的工程成本。在实践中，大多数真实（real-world）LLM 训练与推理工作负载，从前面介绍的更简单设计（例如双缓冲核函数，或配合 CUDA streams 的两阶段流水线）中获得的收益已然足够。

## 多 GPU 计算与数据传输的重叠（借助 CUDA Streams）

在跨多个 GPU 训练或服务 LLM 时，CUDA streams 让你能够重叠本地计算、点对点传输、集合通信、数据准备与内存管理，从而让任何设备都不空闲。例如，假设你把一个 transformer 模型切分到两个 GPU 上，让 GPU 0 处理第 0–3 层，GPU B 处理第 4–7 层。在 GPU A 上，你可能会这样写：

```
// Stream 0 on GPU A: compute layers 0–3
myTransformerLayers0to3<<<gridA, blockA, 0, stream0_A>>>(
    inputActivationsA, outputActivationsA);
cudaEvent_t eventA, eventFromA;
cudaEventCreateWithFlags(eventA,
    cudaEventDisableTiming); // lowers overhead for sync-only events
cudaEventCreateWithFlags(eventFromA,
    cudaEventDisableTiming);  // lowers overhead for sync-only events
cudaEventRecord(eventA, stream0_A);
```

与此同时，在 CPU 上，你已经解码或准备好了下一个微批量（N+1），并发出了一次 cudaMemcpyAsync，将其拷入一块固定主机缓冲区。这样一来，当 GPU A 的 stream 0 完成批次 _N_ 的前向传播时，批次 N+1 的 CPU 到 GPU 拷贝就能立即开始，无需等待。

同时，你调用 cudaStreamWaitEvent(stream1_A, eventA, 0)，以确保 stream1_A 一直等待到 outputActivationsA 就绪后，才发起直接经由 NVLink/PCIe 拷入 GPU B 内存的操作，如下所示：

```
// Stream 1 on GPU A: wait for layer computation, then copy to GPU B
cudaStreamWaitEvent(stream1_A, eventA, 0);
cudaMemcpyPeerAsync(
    destActivationsB, /*dstGPU=*/1,
    outputActivationsA, /*srcGPU=*/0,
    bytes, stream1_A);
```

一旦你记录了 eventA，GPU A 的 stream0 就能立即启动批次 N+1 的下一个前向核函数，并用 cudaMemcpyAsync 把它的输入从固定主机缓冲区拷入设备内存而不发生阻塞。在 GPU B 上，你在拷贝开始或结束时记录一个相匹配的事件：

```
// Stream 1 on GPU A: after peer copy starts,
// record eventFromA
cudaEventRecord(eventFromA, stream1_A);
```

然后在 GPU B 上，你等待数据到达并运行第 4–7 层，如下所示：

```
// Stream 1 on GPU B: wait for data arrival, then run layers 4–7
cudaStreamWaitEvent(stream1_B, eventFromA, 0);
myTransformerLayers4to7<<<gridB, blockB, 0, stream1_B>>>(
    destActivationsB, nextActivationsB);
```

这种显式的事件—等待模式保证 GPU B 的第 4–7 层计算只在点对点拷贝完成之后才开始。与此同时，GPU B 的 stream0_B 可以并行地预取权重或执行其他准备工作。

这里使用的 P2P 传输运行在 GPU A 的 DMA 拷贝引擎上，不占用 GPU A 的任何 SM。这样，GPU A 的计算 stream 就能立即推进到批次 N+1，而无需去管理这次数据传输。

其结果是一个三路重叠：GPU A 的 SM 可以在 stream0*A 中立即开始批次 N+1，GPU A 的对端 DMA 引擎可以在 stream1_A 中把批次 \_N* 的激活值传送给 GPU B，而 GPU B 的 SM 可以在 stream0*B 中运行批次 \_N*，或在其 stream1*B 中开始批次 \_N*。通过把工作划分到不同的 stream 中，并在主机上使用固定内存，我们把 P2P 与 H2D 延迟隐藏在持续进行的计算和数据准备之后。

当需要在众多 GPU 之间同步梯度或广播参数时，NCCL 会使用低占用率的设备核函数在大规模下处理通信，这些核函数驱动 GPUDirect P2P 或通过 NVLink、PCIe 或 InfiniBand 进行 RDMA。在底层，NCCL 会把张量拆分成多个连续的分块。

这一设计表明，每个 GPU 都可以拥有多个 stream，包括一条用于主工作的计算流（compute stream）、一条用于接收数据的接收流，以及一条用于归约的通信流（communication stream）。为每种角色使用专用 stream——并给通信流更高的优先级——能够实现最优重叠。事实上，像 PyTorch 这样的框架通常会在这些专用 stream 上运行 NCCL 集合操作，并为网络传输赋予更高的优先级。

> PyTorch 和 NCCL 都使用专用的高优先级 stream，把通信与计算密集型操作交错进行。这样一来，它们就不会被排在大型计算核函数之后而延迟。

NCCL 会根据 NVLink 或 PCIe 拓扑选择环形（ring）或树形算法。考虑一个四 GPU 环形结构，如图 11-8 所示，其中有四个分块（1–4）。

![图 11-8. 四 GPU 环形结构中的分块 all-reduce](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-8.png)

在这种环形 all-reduce 中，每个 GPU 把分块 _i_ 发送给它的一个邻居（k → k+1），同时从它的另一个邻居（k−1 → k）接收分块 i−1，并使用设备核函数在 NVLink 或 PCIe 上搬运和归约这些分块。

通过对这些分块的发送与接收进行流水线化，NCCL 让 NVLink 保持完全饱和。当分块 _i_ 从 GPU 0 → GPU 1 传输时，分块 i−1 从 GPU 1 → GPU 2 移动，依此类推。这将空闲间隙降到最低。在代码中，你会看到类似下面这样的写法：

```
// In a high-priority communication stream on each GPU:
cudaEventRecord(eventK, computeStream);
cudaStreamWaitEvent(commStream, eventK, 0);
ncclAllReduce( // this is asynchronous
    gradBuffer, gradBuffer, numElements,
    ncclFloat, ncclSum, comm, commStream);
```

由于 NCCL 只使用少量 SM 线程块，它以低占用率的方式在 SM 上编排跨 NVLink 或 NVSwitch 的分块发送/接收 + 归约。与此同时，例如正在运行第 k+1 层反向传播的主计算 stream 可以继续运行。当 all-reduce 完成时，会记录事件，如下所示：

```
cudaEventRecord(eventAllReduceDone, commStream); // signals collective completion
cudaStreamWaitEvent(computeStream, eventAllReduceDone, 0);
// Now apply optimizer updates on computeStream...
```

请记住，像 all-reduce 这样的 NCCL 集合操作是异步的。它们会立即返回控制权。通过在调用之后记录一个事件，并在计算 stream 中等待它，你就能确保计算 stream 直到归约真正完成后才恢复。

> NCCL 的核函数经过刻意优化，能够在低占用率下、每个 GPU 只用少量 SM 资源的情况下达到最大带宽。在实践中，得益于为互连（interconnect）调优过的低占用率核函数，NCCL 每个 GPU 只用少量线程块就能饱和 NVLink 或 PCIe 带宽。此外，如果你的网络硬件支持，你还可以把其中一些集合聚合操作卸载到 NVIDIA SHARP。这将释放出更多 SM 资源来执行更有用的计算工作。

NIXL 为跨 NVLink、RDMA 与存储后端的点对点及分离式传输提供了统一的高吞吐 API，因此你通常会使用 NIXL 的 API，而不是自己去调用 cudaMemcpyPeerAsync。NIXL 能够跨不同的内存层级和互连快速且异步地搬运数据。当你用 nixlCommStream 调用一个 NIXL 操作时，数据会被分块，并使用可用的最快传输机制（例如 GPUDirect RDMA 或 NVLink）在网络上流水线化传输。

与 NCCL 一样，NIXL 也可以为分块的点对点及集合数据传输使用专用的高优先级 stream。这可以减少排队延迟，使它们的拷贝命令在发出后不久便命中 GPU 的 DMA 引擎——很可能抢占优先级更低的工作。

把这些传输 stream 标记为高优先级，并不意味着它们会预先占用 SM 资源。它只是保证当拷贝命令准备就绪时，它们会先于其他低优先级操作被发出。

在实践中，拷贝引擎随后会利用你的计算核函数尚未占用的那部分 DRAM 到 DRAM 带宽。因此，它们实际上只用空闲的内存织构（memory-fabric）带宽在互连上搬运数据。

这一设计最大化了重叠：当一个 stream 在拷贝引擎上驱动 cudaMemcpyPeerAsync 传输或 NCCL 归约核函数时，另一个 stream 的 SM 核函数可以继续处理前向/反向工作。而第三个 stream 可以执行异步分配或事件等待。这让所有硬件单元都保持忙碌且互不争用。

用 cudaMemcpyPeerAsync 启动的点对点传输完全运行在 GPU 的拷贝引擎上，因此只消耗 DRAM 到 DRAM 带宽，而不占用任何 SM 周期。而像 all-reduce 这样的集合操作则被实现为设备核函数，它们使用少量 SM，同时把互连带宽推向峰值。

与此同时，用于激活值或混合精度梯度的临时缓冲区会被异步地分配和释放，以避免全局停顿。例如，在每个 GPU 的计算 stream 内部，你会调用以下代码：

```
cudaMallocAsync(&tempBuffer, tempBytes, computeStream);
// ...use tempBuffer in kernels...
cudaFreeAsync(tempBuffer, computeStream);
```

由于流序内存分配器记录这些操作时不会强制进行设备级同步，CUDA 在分配或释放内存时不会暂停其他 stream，例如 P2P、NCCL 和 NIXL stream。这确保了缓冲区管理绝不会打断重叠进行的计算与通信流水线。

> 这正是我们前面讨论过的流序内存分配器——只不过现在应用到了多个 GPU 上。使用它有助于防止内存管理成为你分布式工作负载的瓶颈。

在你把一次迭代（例如训练或推理迭代）的前向传播、反向传播、P2P 拷贝、NCCL all-reduce、内存释放以及事件等待操作入队之后，你可以把这条执行链捕获进单个 CUDA Graph。

具体来说，当你调用 cudaStreamBeginCapture 时，你入队到该 stream 的每一个操作（例如核函数启动、计算、通信、数据传输、事件等）都会作为节点插入到图中。这可以包括 ncclAllReduce()、cudaMemcpyAsync、cudaMemcpyPeerAsync、cudaMallocAsync、cudaFreeAsync、cudaEventRecord、cudaStreamWaitEvent——以及它们之间的依赖关系。你用 cudaStreamEndCapture(stream, &graph) 结束捕获。代码大致如下：

```
cudaStreamBeginCapture(streamA, cudaStreamCaptureModeGlobal);
computeGradientsLayer1<<<... , 0, streamA>>>(...);
ncclAllReduce(..., comm, streamB);
computeGradientsLayer2<<<... , 0, streamA>>>(...);
ncclAllReduce(..., comm, streamB);
cudaStreamEndCapture(streamA, &graph);
```

在捕获提交到多个 stream 的工作时，请使用 cudaStreamCaptureMode Global，这样同一线程中入队到任何参与 stream 的操作都会被记录进同一个捕获会话——并且跨流依赖也会被保留。否则，只有入队到传给 cudaStream BeginCapture 的那个 stream 的操作才会被捕获。

随后你可以用 cudaGraphLaunch() 实例化并启动该图。这将以接近零的 CPU 开销重放整个 DAG。通过把整个多 GPU 迭代一次性捕获下来，你消除了每个操作的入队与启动开销。

此外，CUDA Graphs 支持条件节点（例如梯度裁剪），因此不常发生的分支也保留在图的逻辑内部。我们会在下一章更详细地介绍 CUDA Graphs，但理解它们与 CUDA streams 的关系很重要，包括 cudaStreamBeginCapture 和 cudaStreamEndCapture。

> 像 PyTorch 这样的现代框架把这些复杂性大多隐藏了起来。例如，PyTorch 的 DistributedDataParallel 会自动把数据传输、计算与通信调度到各自独立的 stream 中。它还使用 CUDA Graphs 来降低每次迭代的开销。尽管它可以使用 CUDA Graphs 来降低开销，但你必须通过使用捕获安全（capture-safe）的代码和 API 显式启用这一点，因为它并非自动启用。

虽然你并不需要显式使用这些 CUDA stream 的开始与结束 API 来构建 CUDA Graph，但当你希望在把操作入队到 stream 的同时捕获图时，它们会很方便（我们会在下一章更详细地介绍 CUDA Graphs）。

CUDA streams 通过在 CPU、DMA 引擎与 SM 之间重叠工作，让多 GPU 系统中的流水线得以精细调优。在 CPU 一侧，把数据用 cudaMemcpyAsync 拷入固定主机缓冲区，可以在 GPU 执行当前工作负载的同时为下一批次暂存数据，从而确保批次 N+1 的输入在批次 _N_ 完成之前就已就绪。与此同时，GPU 之间的点对点传输（使用 cudaMemcpyPeerAsync）可以被调度到专用 stream 上，并用 cudaStreamWaitEvent 同步，以便在不停顿正在进行的计算的前提下交接激活值或梯度。

如前所述，像 NCCL 或 NIXL 这样的集合通信库是使用独立 stream 的低占用率通信核函数。这样一来，当一个 GPU 在归约梯度或广播参数时，同一个 GPU 上的其他 stream 可以继续执行本地计算核函数（例如后续各层的前向或反向传播）。此外，按 stream 顺序使用 cudaMallocAsync 和 cudaFreeAsync，可以避免为临时缓冲区做全局同步，让分配与释放能够与计算和通信并发进行。

把整个迭代（例如 CPU 暂存、P2P 传输、计算核函数、NCCL 调用、分配、释放以及事件/等待）捕获进一个 CUDA Graph，可以进一步降低 CPU 开销。图一旦实例化，调用 cudaGraphLaunch() 就会一次性重放所记录的 DAG。这消除了每次调用的入队开销，并自动保留所有依赖关系。

综合起来，这些技术确保每个 GPU 的 SM 流水线、拷贝引擎和互连链路都保持忙碌。当一个 stream 运行矩阵乘核函数时，另一个执行点对点拷贝，第三个执行集合通信，而 CPU 则为下一个微批量暂存数据。

简而言之，你可以用 CUDA streams 来编排跨多个层次的工作，并重叠计算、内存操作与数据传输。这种方法通过把点对点和集合通信延迟隐藏在活跃计算之后，把每一个硬件单元都推向极限。总体而言，CUDA streams 为许多 AI 工作负载（包括 LLM 训练与推理）最大化了吞吐、利用率和效率。

## 程序化依赖启动

另一种核间流水线化与通信的方式称为*程序化依赖启动*（Programmatic Dependent Launch，PDL）。PDL 让一个核函数能够在设备上直接触发另一个核函数的执行，使用同一条 CUDA stream——并且无需 CPU 参与。例如，作为主核函数（primary kernel）的 Kernel A 可以触发作为次级核函数（secondary kernel）的 Kernel B，而后者正在等待来自 Kernel A 的信号。

这一触发甚至可以在 Kernel A 执行完成之前发生。为此，它使用 cudaTriggerProgrammaticLaunchCompletion()，如图 11-9 所示。这里，Kernel B 通过调用 cudaGridDependencySynchronize() 等待 Kernel A。

![图 11-9. 使用 PDL 从 Kernel A 启动 Kernel B——让 Kernel B 的执行与 Kernel A 的收尾（epilogue）部分重叠（在同一 CUDA stream 中）](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-9.png)

利用 PDL 提供的构件，Kernel A 可以在其*收尾*（epilogue，即收官）阶段就发信号让 Kernel B 执行。这样，Kernel A 就能与 Kernel B 并行执行一小段时间。需要重点注意的是，在 Kernel A 尚未产出并同步好 Kernel B 所需的全部数据（例如 L2/共享内存/全局内存）之前，Kernel A 不应触发 Kernel B 执行。

> 依赖核函数的数据依赖应在其继续处理之前，在 L2、共享内存、全局内存等处可见。

下面的代码展示了 Kernel A 使用 cudaTriggerProgrammaticLaunchCompletion() 向 Kernel B 发信号，表明其主要工作已经完成。这同时也通知 Kernel B：所有必要的全局内存刷新都已发生——可以安全地继续：

```
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
// Kernel A must trigger the PDL flag when it's
//    safe to launch Kernel B
__global__ void kernel_A(int *d_ptr) {
    // Perform work that produces data used by
    //   Kernel B

    // Signals that Kernel A's global-memory
    //   flushes are complete
    // This enables dependent Kernel B's launch
    cudaTriggerProgrammaticLaunchCompletion();
    // ... any further work that can overlap with  ...
    ...
}
// Kernel B must wait for Kernel A to write its memory
//    to global memory and become visible to Kernel B
__global__ void kernel_B(int *d_ptr) {
    // Wait on Kernel A to complete.
    // This ensures the Kernel B waits for the
    //   memory flush before accessing shared data
    cudaGridDependencySynchronize();
    // ... dependent work on d_ptr ...
    ...
}
int main() {
    // 1) Allocate device buffer
    int *d_ptr = nullptr;
    // Allocate an int (example)
    cudaMalloc((void**)&d_ptr, sizeof(int));
    // 2) Create a nonblocking stream for maximum overlap
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream,
        cudaStreamNonBlocking);  // Nonblocking
    // 3) Define grid/block sizes
    dim3 gridDim(128), blockDim(256);
    // 4) Launch Kernel A asynchronously
    kernel_A<<<gridDim, blockDim, 0,
        stream>>>(d_ptr);    // Async launch
    // 5) Configure PDL for Kernel B
    cudaLaunchConfig_t launch_cfg{};
    launch_cfg.gridDim           = gridDim;
    launch_cfg.blockDim          = blockDim;
    launch_cfg.dynamicSmemBytes  = 0;
    launch_cfg.stream            = stream;
    // Sets the PDL flag so cudaLaunchKernelExC overlaps
    //   with Kernel A's epilogue
    static cudaLaunchAttribute attrs[1];
    attrs[0].id  = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed =
        1;
    launch_cfg.attrs    = attrs;
    launch_cfg.numAttrs = 1;
    // 6) Pack the pointer argument
    void* kernelArgs[] = { &d_ptr };
    // 7) Launch Kernel B kernel early using PDL
    // Lookup device pointer for secondary_kernel
    cudaKernel_t kB;
    cudaGetKernel(&kB, kernel_B);
    void* funcPtr_kernel_B = reinterpret_cast<void*>(kB);
    cudaLaunchKernelExC(&launch_cfg, funcPtr_kernel_B, kernelArgs);
    // 8) Wait until all work in the stream completes
    cudaStreamSynchronize(stream);
    // 9) Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_ptr);
    return 0;
}
```

这里，Kernel B 一直在 cudaGridDependencySynchronize() 上等待，直到它收到来自 Kernel A 的这一程序化启动完成信号。一旦交接发生，这两个核函数就可以重叠执行。通过把 Kernel A 中的触发、Kernel B 中的同步以及主机端的启动属性配对起来，这段代码在 Kernel A 与 Kernel B 之间实现了尽可能多的重叠。

正如本例所示，使用 PDL 时你需要创建一个 cudaLaunchConfig_t，并在 cudaLaunchKernelExC() 这一 CUDA 调用中使用特殊属性。具体来说，在主机端，PDL 是通过为 Kernel B 的启动配置一个 cudaLaunchConfig_t，并启用 cudaLaunchAttributeProgrammaticStreamSerialization 属性来开启的，以允许提前的、重叠的分派。

用 cudaLaunchAttributeProgrammaticStream Serialization 调用 cudaLaunchKernelExC() 会告诉 CUDA 运行时：即便 Kernel A 尚未完全完成，把 Kernel B 入队也是安全的。随后用 cudaLaunchKernelExC() 携带这些扩展属性执行实际启动，它依赖触发机制来完成交接。

## 综合 PDL、线程块簇与 warp 专门化

让我们把三种正交的技术——PDL、warp 专门化流水线以及线程块簇——汇聚到一个执行模型中。

PDL 提供了让一个核函数发信号表明其序言（prologue）完成、并触发一个依赖核函数执行的机制。随后它会逐渐收尾，而依赖核函数则逐渐启动并执行。

warp 专门化把每个线程块细分为生产者 warp 与消费者 warp。具体来说，生产者 warp 使用 TMA 异步传输 tile，而计算 warp 执行乘加累加（matrix-multiply-accumulate，MMA）操作。

而线程块簇保证多个线程块运行在邻近的 SM 组上。这有利于多 SM 协作以及面向大规模工作负载的共享内存多播。

综合来看，这种核间与核内技术的组合隐藏了 DRAM 延迟，减少了核函数边界开销，并最大化了 Tensor Core 利用率。它们构建出一个具有以下特征的流水线：

_核内流水线化_ warp 专门化确保在每个线程块内部，数据传输（生产者 warp）与计算（消费者 warp）通过多阶段流水线完全重叠。

_核间重叠_ PDL 允许一个依赖 GEMM 核函数（它可能作用于神经网络中的下一层）的序言，在主核函数完成数据（权重）预取后立即开始——而无需等待整个线程块拆除。

_块间协作_ 线程块簇让多组线程块得以协调预取（例如多播），并在 SM 之间执行动态负载均衡。这样，生产者与消费者任务都能在整个簇范围内均匀分布。

例如，你可以在同一个核函数内把模型权重的搬运（数据传输）与 GEMM（计算）重叠起来，使得一个 tile 在被计算的同时，下一个 tile 正在途中。一个 warp 专门化的多阶段软件流水线（阶段 0… N）可以使用 PDL 和 _mbarrier_ 原语来协调这些角色，如图 11-10 所示。

![图 11-10. 结合 PDL、线程块簇与 TMA 多播的 warp 专门化多阶段流水线，用于实现高性能的核间与核内 GEMM](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-10.png)

下面是一个 CUDA C++ 示例，展示如何用一个简单的 TMA 风格异步拷贝 + 计算流水线，把 PDL、线程块簇与 warp 专门化结合起来。具体来说，主核函数 primary_gemm 使用 cudaTriggerProgrammatic LaunchCompletion() 向次级核函数发信号，表明所有内存刷新都已完成、数据已准备好被消费。于是，次级核函数 secondary_gemm 现在可以安全地从 cudaGridDependencySynchronize() 处继续，如下所示：

```
#include <cstdio>
#include <cuda_runtime.h>
// Cooperative Groups for clusters/barriers
#include <cooperative_groups.h>
// C++ barrier for TMA-like sync
#include <cuda/barrier>
// Async copy API
#include <cuda/pipeline>
namespace cg = cooperative_groups;
// Tile size for our toy GEMM
constexpr int TILE_M = 128;
constexpr int TILE_K = 128;
constexpr int TILE_N = 128;
// A very simple “producer/consumer” pipeline within each CTA
__global__ __cluster_dims__(2,1,1)    // Compile-time cluster of 2 CTAs
void primary_gemm(const float* __restrict__ A,
                  const float* __restrict__ B,
                        float* __restrict__ C,
                  int M, int N, int K)
{
    // Identify thread-block cluster & within-block group
    cg::thread_block_cluster cluster = cg::this_thread_block_cluster();
    cg::thread_block       cta     = cg::this_thread_block();
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warpId = tid / warpSize;
    const int numProducerWarps = 1;
    // Shared-memory tile buffers
    __shared__ float tileA[TILE_M * TILE_K];
    __shared__ float tileB[TILE_K * TILE_N];
    __shared__
    cuda::pipeline_shared_state<cuda::thread_scope_block, 1> pipe_state;
    auto pipe = cuda::make_pipeline(cta, &pipe_state);
    if (warpId < numProducerWarps) {
    pipe.producer_acquire();
    cuda::memcpy_async(cta, tileA, A, cuda::aligned_size_t<32>{TILE_M
                       * TILE_K * sizeof(float)}, pipe);
    cuda::memcpy_async(cta, tileB, B, cuda::aligned_size_t<32>{TILE_K
                       * TILE_N * sizeof(float)}, pipe);
    pipe.producer_commit();
    }
    cta.sync();
    pipe.consumer_wait();
    pipe.consumer_release();
// ... perform “compute” on the tile ...
// (e.g., a few fused multiply-adds)
do_compute();
    // Inter-CTA cluster-scope sync for load balancing
    cluster.sync();
    // Signal to dependent kernel that prologue is done
    cudaTriggerProgrammaticLaunchCompletion();
    // ... perform remaining epilogue work ...
    // ...
}
__global__ __cluster_dims__(2,1,1)
void secondary_gemm(const float* __restrict__ A,
                    const float* __restrict__ B,
                          float* __restrict__ C,
                    int M, int N, int K)
{
    // Wait for primary’s PDL signal before starting
    cudaGridDependencySynchronize();
    // Similar warp-specialized pipeline as above...
    //  (Omitted for brevity. Duplicate of primary logic,
    //  but reading from different offsets to compute
    //  next GEMM tile.)
}
// cudaLaunchKernelExC, cudaGetKernel
#include <cuda_runtime_api.h>
// cudaLaunchConfig_t
#include <cuda/launch_config.h>
int main()
{
    // Problem dimensions (must be multiples of TILE_)
    int M = TILE_M, N = TILE_N, K = TILE_K;
    // Allocate and initialize matrices A, B, C on device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));
    // (Initialize d_A, d_B via cudaMemcpy or kernels...)
    // Create a nonblocking stream for overlap
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream,
        cudaStreamNonBlocking);
    // Launch primary GEMM
    dim3 gridDim(M/TILE_M, N/TILE_N), blockDim(256);
    primary_gemm<<<gridDim, blockDim, 0, stream>>>(d_A,
        d_B, d_C, M, N, K);
    // Configure PDL attributes for secondary launch
    cudaLaunchConfig_t launch_cfg = {};
    launch_cfg.gridDim          = gridDim;
    launch_cfg.blockDim         = blockDim;
    launch_cfg.dynamicSmemBytes = 0;
    launch_cfg.stream           = stream;
    static cudaLaunchAttribute attrs[1];
    attrs[0].id  = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    launch_cfg.attrs    = attrs;
    launch_cfg.numAttrs = 1;
    // Prepare arguments and get function pointer
    //    for secondary_gemm
    void* kernelArgs[] = {&d_A, &d_B, &d_C, &M, &N, &K};
    cudaKernel_t k;
    cudaGetKernel(&k, secondary_gemm);
    void* funcPtr = reinterpret_cast<void*>(k);
    // Early enqueue of secondary GEMM via PDL
    cudaLaunchKernelExC(&launch_cfg, funcPtr, kernelArgs);
    // Wait for everything to finish
    cudaStreamSynchronize(stream);
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(stream);
    return 0;
}
```

在主机上，你把主核函数（primary_gemm）启动到一条非阻塞流中，并创建一个带有 ProgrammaticStreamSerialization 属性的 cudaLaunchConfig_t。你用它来为次级核函数（secondary_gemm）调用 cudaLaunchKernelExC。

在 primary_gemm 内部，一次对 cudaTriggerProgrammaticLaunchCompletion() 的调用标志着内存刷新的完成，并允许依赖核函数的处理开始——甚至在主核函数完全拆除之前。

依赖核函数随后调用 cudaGridDependencySynchronize()。这会等待来自主核函数的信号和必要的内存刷新，从而使它能够与主核函数的收尾并行地开始执行。

在每个线程块内部，我们使用 warp 专门化来重叠数据搬运与计算。通过把每个块拆分为单个“生产者” warp 和多个“消费者” warp，生产者发出 cuda::memcpy_async 调用把数据拷入共享内存。这利用 TMA 多播把单次 DMA 传输广播给簇内所有线程块，如图 11-11 所示。

![图 11-11. TMA 多播：主导 CTA 将 cp.async.bulk.tensor 发射到 DSMEM（簇共享内存）；跟随 CTA 使用 cluster.map_shared_rank 消费这些 tile；cuda::memcpy_async 驱动 TMA](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-11.png)

当生产者 warp 用 TMA 多播加载数据时，消费者会在一个 C++ 块作用域屏障（cuda::barrier<cuda::thread_scope_block>）上自旋，然后再执行其矩阵乘步骤（do_compute()）。这让每个 tile 的加载与其乘加融合（FMA）操作能够在一个细粒度的多阶段软件流水线中交错进行，从而在核函数内部隐藏全局内存延迟。

为了在 SM 之间协调工作，我们用 **cluster_dims**(2,1,1) 标注这两个核函数。这会把成对的线程块分组到邻近的 SM 上。一次对 cluster.sync()（协作组对 PTX 的 mbarrier 指令的封装）的调用充当簇级屏障和共享内存栅栏。这样，簇内所有线程块都能看到相同的 TMA 加载数据，并能动态地重新平衡剩余的 tile。这种块间协作避免了 SM 空闲，并增强了 warp 专门化流水线的收益。

> 生产级核函数通常对二维 tile 使用 Tensor Memory Accelerator（TMA）的 cp.async.bulk.tensor，并在多个线程块需要同一 tile 时跨线程块簇进行多播。当在 Blackwell 上做分块拷贝时，可考虑使用基于描述符的 TMA 归约操作（例如 cp.reduce.async.bulk(.tensor)）来做即时归约。当把小型归约融合进数据搬运流水线时，优先使用基于描述符的加载/存储加上 TMA 归约操作。这将降低寄存器压力并改善重叠。

最后，让我们再回顾一下这条高性能 GEMM 流水线的三个特征：

_核内流水线化_ 在每个线程块内部通过 warp 专门化实现，把工作细分开来，使生产者 warp 发出 cuda::memcpy_async 的 tile，而消费者 warp 在块作用域屏障上自旋。这让数据传输与计算操作完全重叠。
