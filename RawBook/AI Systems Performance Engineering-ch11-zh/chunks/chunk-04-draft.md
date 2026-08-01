_核间流水线化（Inter-kernel pipelining）_ 使用 PDL，它让下一个 GEMM 核函数的处理能够在主核函数被完全拆除之前就开始（借助 cudaTriggerProgrammaticLaunchCompletion() → cudaGridDependencySynchronize() 机制）。这掩盖了核函数启动开销。

_块间协作（Interblock cooperation）_ 通过用 **cluster_dims** 注解核函数，这段代码创建了被协同调度到邻近 SM 上的线程块对。再配合调用 cluster.sync()（基于 mbarrier 的簇屏障），这些线程块共享 TMA 多播数据，并在整个网格上动态负载均衡工作。

简而言之，通过把 warp 级流水线化与 TMA、簇级同步与多播，以及使用 PDL 的核间重叠层层叠加，你就构建出一条高度重叠的流水线，它隐藏 DRAM 延迟、掩盖核函数启动开销，并最大化 Tensor Core 利用率。其结果是在每个阶段都得到高性能 GEMM：在 warp 内拷贝中、在跨线程块屏障中，以及在核函数交接期间。它们协同运作，让硬件持续繁忙、避免停顿。

## 关键要点

本章涵盖了一些与 CUDA streams、流序内存分配器（stream-ordered memory allocator）、基于事件的同步、核间流水线化、线程块簇（thread block cluster）以及 PDL 相关的进阶主题。它们有助于为推理和训练工作负载构建高效的流水线。以下是一些关键要点：

_显式流与默认流（Explicit versus default streams）_ 避免使用传统默认流（legacy default stream）（stream 0），它会串行化所有工作并充当全局屏障。取而代之，创建显式的非阻塞流（cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)），使核函数与拷贝能够并发运行，而不会产生隐藏的同步。

_流序内存分配器（Stream-ordered memory allocator）_ 使用 cudaMallocAsync 和 cudaFreeAsync 在特定 stream 内分配/释放设备内存。这个非阻塞分配器把请求记录到该 stream 的队列中，避免全局设备同步，并让分配能够与在途（in-flight）的核函数和拷贝相重叠。

_重叠 H2D、计算与 D2H（Overlapping H2D, compute, and D2H）_ 通过在不同 stream 上入队异步的主机到设备拷贝（cudaMemcpyAsync）、核函数启动以及设备到主机拷贝，你可以实现三路重叠。当一个 stream 运行核函数时，另一个 stream 可以把下一批数据拷贝到设备（H2D），而第三个 stream 则可以把结果拷回主机（D2H）。这隐藏了延迟并减少了空闲时段。

_用于细粒度同步的 CUDA 事件（CUDA events for fine-grained synchronization）_ 使用 cudaEventRecord 和 cudaStreamWaitEvent 来协调跨 stream 的生产者—消费者依赖，而不必让整个 GPU 或 CPU 停顿。事件让消费者 stream 能够精确地等待，直到生产者 stream 完成一次拷贝或核函数，从而保持最大并发度。

_核间流水线化（Inter-kernel pipelining）_ 把核函数内部的 warp 专门化（多角色）、两阶段（双缓冲）流水线与多 stream 启动结合起来。在不同的 stream 中启动一个 warp 专门化核函数（loader → compute → storer）的多个实例，向 GPU 持续投喂相继的小批量（mini-batch）。这将核内内存/计算重叠与核间并发结合在了一起。

_线程块簇配合 streams（Thread block clusters with streams）_ 把核内 warp 专门化流水线扩展为覆盖整个网格的线程块簇（协作式启动），可让 loader/compute/storer warp 跨越多个块协同工作。在多个 stream 中启动这些协作式核函数，使得后续批次的主机端分配与拷贝能够在某个协作式核函数执行期间进行。

_核内信号与借助 PDL 的重叠（In-kernel signaling and overlap with PDL）_ 核函数 A 在其数据写入被刷新后调用 cudaTriggerProgrammaticLaunchCompletion()，核函数 B 则使用 cudaGridDependencySynchronize() 来等待该信号。这让核函数 B 的序言（prologue）能够开始，并与 A 的收尾（epilogue）相重叠——无需 CPU 介入。

_主机端 PDL 设置（Host-side PDL setup）_ 主机通过 cudaLaunchKernelExC() 配置核函数 B 的启动，并用一个设置了 cudaLaunchAttributeProgrammaticStreamSerialization 的 cudaLaunchConfig_t。这让驱动能够提前入队 B 并最大化核间重叠。PDL 在主核函数中使用 cudaTriggerProgrammaticLaunchCompletion，在依赖核函数中使用 cudaGridDependencySynchronize，并在主机端使用 cudaLaunchAttributeProgrammaticStreamSerialization 启动属性。

## 结论

总而言之，基于 CUDA streams 的核间并发已经从一种手动优化演变为现代 AI 框架与 GPU 运行时所使用的自动特性。通过理解诸如 streams、资源占用（resource occupancy）与同步点等核心原理，开发者可以最大化 GPU 利用率。

核间并发对于在现代工作负载中最大化 GPU 利用率至关重要，它通过在单个 GPU 内部以及跨多个 GPU 之间重叠核函数执行与数据传输来实现这一点。随着硬件不断增加更多并行性——而软件抽象掉越来越多的调度复杂性——理解如何最大化并发，是从你的高性能 AI 系统硬件中榨取最大价值的关键一环。

本章演示了如何在多个 CUDA stream 之间编排核函数、内存操作与分配，从而让所有 GPU 硬件单元都保持活跃运行。这些硬件单元包括计算流水线、DMA 引擎与互连（interconnect）。CUDA streams 充当基础机制，把核函数、内存操作与分配入队到相互独立的队列中，让 GPU 的计算引擎与 DMA 引擎能够同时运行。

通过避开默认流的隐藏屏障、利用流序分配器、用事件做精确同步，以及把核内 warp 专门化与核间多 stream 流水线相结合（并扩展到第 10 章的线程块簇），你即使面对复杂的 LLM 工作负载，也能达到接近峰值的利用率。

在多 GPU 场景中，跨不同 stream 重叠点对点（peer-to-peer，P2P）传输、集合通信（collective communication）与计算，可进一步压缩空闲时间。我们回顾了第 10 章，展示了如何用 CUDA Graphs 捕获这些 stream 工作流，以降低重复迭代时的 CPU 开销。

在下一章，我们将进一步深入，并在这些原理之上，引入动态核函数编排以及借助动态并行（dynamic parallelism）与 CUDA Graphs 的元调度。我们将在运行时协调整条核函数与数据搬运的流水线——以适应不断变化的工作负载。这种动态资源均衡与设备端编排，将把我们推向大规模 AI 系统性能优化的下一个层次。
