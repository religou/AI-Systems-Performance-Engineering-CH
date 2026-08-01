与此同时，GPU/PE 1 的核函数在 nvshmem_int_wait_until() 上自旋，一旦标志被置位便读取有效载荷。这不需要 CPU 介入，也不需要额外的拷贝——只是通过 NVLink 进行硬件加速的 GPU 间传输。

由于 NVSHMEM 通信使用的是通过 NVLink 或 PCIe 的 GPU 发起、单边（one-sided）操作，它消除了主机端暂存（host staging）。因此，NVSHMEM 通信能够达到接近峰值的线速。这是因为 NVSHMEM 的单边操作既绕过了 CPU，也绕过了核函数启动的软件开销。本质上，NVSHMEM 把过去多步骤的通信变成了单次硬件事务。

当然，能力越大，责任越大。因为 NVSHMEM 本质上就是 GPU 级别的共享内存编程，你必须小心地管理同步、避免竞态。此外，过度使用全局屏障也会让所有 GPU 被最慢的对端拖住。

实践中，避免过度同步。尽可能使用 NVSHMEM 的细粒度（fine-grained）信号或点对点同步，而不是总是调用 nvshmem_barrier_all()。

NVSHMEM 的现代实现为这些同步例程带来了效率提升。然而，它们仍然是同步点，一旦误用就可能成为瓶颈。NVSHMEM 提供了细粒度原语，例如用于等待设备变量的 nvshmem_wait_until，以及诸如 nvshmem_signal_fetch、nvshmem_signal_wait_until 这样的信号操作，或用于仅需一部分设备协调时进行点对点同步的 nvshmemx_signal_op 变体。发送方与接收方 GPU 之间 NVSHMEM 共享数据并用信号进行同步的底层细节，如图 12-14 所示。

![图 12-14. NVSHMEM 单边通信示例](AI%20Systems%20Performance%20Engineering-ch12_images/figure-12-14.png)

当工作负载不规则或依赖数据时，例如图算法、动态负载均衡和离散事件仿真，NVSHMEM 便大放异彩。在这些场景中，静态图与集合（collective）并不够用。

使用 NVSHMEM 的核函数可以像任何其他核函数一样被捕获进 CUDA Graphs。NVSHMEM 的设备端操作发生在这些核函数内部，并不是独立的图节点。然而，NVSHMEM 真正的强项在于让核函数即时地自适应与协调，而无需图所使用的固定通信脚本。

简而言之，NVSHMEM 把一组 GPU 变成一个共享内存域，从而实现纯设备端的核函数启动、数据传输与同步，其延迟和吞吐远超任何以 CPU 为中介的方式。

设想一条两阶段的 Transformer 推理流水线（注意力 + 多层感知机），其流程如下：GPU 0 计算注意力，然后 NVSHMEM 将其激活值 put 到 GPU 1 并发出信号。运行着持久化核函数（persistent kernel）的 GPU 1 看到该标志后，开始 MLP 阶段。

因为 GPU 0 会立即转去处理 batch 2，而 GPU 1 仍在处理 batch 1，经过几次迭代之后，两台设备便完美地协同工作。每一次交接都被隐藏在活跃的计算 warp 之后。这带来接近 100% 的利用率，而不会引入主机端停顿。

当你需要灵活的负载均衡时，NVSHMEM 的原子操作让每个 PE 都能动态地抢占工作。PE 是一个操作系统进程，是并行 NVSHMEM 应用的组成部分。

这里的代码展示了每个 GPU 从一个全局计数器拉取下一个索引、处理该分块、然后循环。这在设备上完全实现了真正的工作窃取（work stealing）——无需主机协调：

```
__global__ void work_steal_kernel(/*...*/, int *queue_head, Task *tasks) {
    while (true) {
        // Atomically claim the next task index
        int idx = nvshmem_int_atomic_inc(queue_head);
        if (idx >= N_tasks) break;
        // Process tasks[idx]...
    }
}
```

对于需要尽可能低抖动的场景，例如一次紧凑的多 GPU 集合操作或一次同步的模型并行步骤，你可以启动一个使用 NVSHMEM 并横跨所有 GPU 的协作核函数，方法是用 nvshmemx_collective_launch() 在 GPU 0 和 GPU 1 上同时启动发送方与接收方核函数。这使得它们无需任何主机介入即可使用 NVSHMEM 进行协调。然后，你可以使用 NVSHMEM 的设备端屏障，如下所示：

```
__global__ void synchronized_step_kernel(/*...*/) {
    nvshmem_barrier_all();
    // All GPUs proceed in lockstep here
    // ...
}
```

这里，每个 PE 一起进入 nvshmem_barrier_all()，然后同时继续。这保证了整个集群完美对齐的执行。

> 所有使用 NVSHMEM 设备级同步或集合操作的核函数都必须用 nvshmemx_collective_launch() 启动。这确保该核函数在作业中的所有 PE（GPU）上并发运行。

### 用 NCCL 与 CUDA Graphs 捕获多 GPU 集合操作

当你需要批量的集合操作，如广播、归约和 all-to-all 传输时，NVIDIA 的 NCCL 库是多 GPU 系统上的首选。传统做法是，每个 GPU 从主机端启动一个 ncclAllReduce 或类似的集合操作。然后它会等待（同步），才继续进入下一个计算阶段。这种顺序化的主机编排在前向传递和反向传递之间增加了开销与空闲时间。

不过，NCCL 调用也可以像核函数一样被录制进 CUDA Graphs。这让你能把前向核函数、all-reduce 和反向核函数“烘焙”进单个图中，并在每次迭代重放它：

```
cudaStreamBeginCapture(captureStream,
    cudaStreamCaptureModeGlobal);
forwardKernel<<<...>>>(...);
ncclAllReduce(sendBuf, recvBuf, count, ncclFloat,
    ncclSum, comm, captureStream);
backwardKernel<<<...>>>(...);
cudaStreamEndCapture(captureStream, &graph);
// Instantiate and upload before launching
cudaGraphExec_t graphExec;
cudaGraphInstantiate(&graphExec, graph, ...);
cudaGraphUpload(graphExec, captureStream);
// Each training step:
cudaGraphLaunch(graphExec, captureStream);
```

注意所有操作——包括 NCCL——都使用同一个捕获流。借助图，NCCL 调用会像核函数一样成为图节点。因为每个进程都在重放一个完全相同的图，NCCL 的内部逻辑会找到对端并执行 all-reduce，而无需额外的主机协调。这种图捕获的 all-reduce 在大型集群上尤其强大，因为它消除了每次迭代的启动抖动，并让所有 GPU 忙于重叠计算与网络操作。

因为每个 GPU 都启动同一个图（包括其集合节点），NCCL 在内部跨各 rank 进行会合，而无需额外的主机介入。主机每次迭代的工作量降到单次 cudaGraphLaunch，从而减少 CPU 开销与启动抖动。

在减轻 CPU 负载之外，把 all-reduce 捕获进图还能实现跨多个 GPU 和计算节点的通信与计算的真正重叠。假设你把梯度计算拆成两遍，第 1 层到第 _L_/2 层，以及第 (_L_/2 + 1) 层到第 _L_ 层，并把它们映射到不同的流，如下所示：

```
// Pseudocode in capture:
computeGradientsLayer1<<<...>>>(..., streamA);
ncclAllReduce(..., comm, streamB);       // in streamB, overlaps with streamA
computeGradientsLayer2<<<...>>>(..., streamA);
ncclAllReduce(..., comm, streamB);
```

由于这些节点是带着各自独有的流分配与依赖关系被捕获的，CUDA 驱动可以把 streamB 上 NCCL 的网络传输与 streamA 上的独立工作重叠起来。这种分桶式 all-reduce（bucketed-all-reduce）模式能稳定地把通信延迟隐藏在计算之后，改善多 GPU 扩展性。

> 当所有 rank 都用相同的通信子（communicator）捕获并重放相同的序列时，NCCL 集合操作与图捕获是兼容的。然而，所有 rank 必须用相同的通信子捕获并重放相同的 NCCL 序列。跨图重放使用不匹配的通信子会有死锁风险（这还算好，相对容易调试）或产生错误结果（最糟的情况，静默失败，且难以调试）。此外，建议捕获初步的预热集合操作，提前运行它们以初始化通信子，并复用已实例化的图，以达到最小的稳态延迟。

实践中，这种分桶式 all-reduce 做法在大模型训练中是标准做法。通过把一块块的梯度归约与后续层的计算重叠，你几乎隐藏了所有网络时间。图 12-15 展示了一个在独立进程（进程 1 和进程 2）中运行的、带 DDP 的分桶式 all-reduce 示例。

![图 12-15. 将 all-reduce 与计算重叠](AI%20Systems%20Performance%20Engineering-ch12_images/figure-12-15.png)

像 PyTorch DDP 这样的现代库会自动实现这种方法的变体。但捕获一个 CUDA Graph 可以进一步减少 CPU 开销，并提供更确定的性能。

在多 GPU 环境中使用 CUDA Graphs 时要记住几点考虑。首先，所有参与的 GPU 都必须以完全相同的顺序录制并重放集合操作，以避免死锁——这很像 MPI 的集合规则，其中所有 rank 必须以相同的顺序进入集合操作。

其次，虽然 CUDA Graphs 会固定并复用 GPU 缓冲区，但要确保梯度和通信缓冲区的分配在捕获之前完成。此外，如前所述，如果你需要修改图中的参数，例如 batch size，你可以用 cudaGraphExecUpdate 来打补丁式地修改这些参数，而无需完整重新捕获。

实践中，把 NCCL 加计算一起捕获进图，可以削减每步的 CPU 时间，并加速横跨众多 GPU 的大模型训练。在数十万 GPU 的超大规模下，这些节省会叠加累积——在整个集群上带来更紧密的同步与更高的利用率。

> NCCL 与 CUDA Graphs 为我们提供了一种把集合通信与计算一并调度的高效方式。然而，并非所有多 GPU 通信都是集合式的——有时我们需要在 GPU 之间进行更细粒度或异步的数据共享。而这正是前面介绍的 NVSHMEM 能派上用场的地方。

### N-GPU 扩展模式

无论你使用简单的对等拷贝、NCCL 环，还是 NVSHMEM 的单边原子操作，扩展到多 GPU 的模式始终相同。系统应当一次性分派核函数、把数据传输与计算做流水线化与重叠，并线性扩展——尤其是要让主机 CPU 离开关键路径。

例如，4 个 GPU 在理想情况下应当表现得像一个快 4× 的单 GPU——前提是你能让所有 GPU 并行地持续有数据可用（见图 12-16），并让主机置身事外。如果你能做到这一点，通过恰当地重叠通信与计算，你应当获得接近线性的加速。若没有重叠，一旦通信时间等于计算时间，扩展就会趋于平台期。

随着 GPU 数量增加，你需要对数据传输与计算采用更激进的流水线化——并减少 CPU 端的编排与同步。因此，你应当用异步拷贝、图中的 NCCL 集合操作以及 NVSHMEM 的 PGAS 原语，把更多编排卸载到设备上。这把更多责任转移到了软件。

![图 12-16. 每个 GPU 并行计算，同时并发交换数据——没有空闲的 GPU，也没有停滞的数据传输](AI%20Systems%20Performance%20Engineering-ch12_images/figure-12-16.png)

运用这些技术，你可以消除 CPU 瓶颈、把高速互连打满、把计算 FLOPS 榨干，构建真正低延迟的多 GPU 流水线。接下来，让我们在动态与设备端调度和编排的语境下重新审视 Roofline 模型。

## Roofline 引导的调度与编排决策

在过去的几章里，我们收集了一套扎实的编排技术，包括 CUDA streams、核函数融合、持久化核函数、CUDA Graphs、动态并行等等。Roofline 模型能帮助判断哪种工具最可能为你的情形带来最大收益。

Roofline 的核心可归结为运算的算术强度（arithmetic intensity），即执行的 FLOPS 与移动的字节数之比。它由两个硬件“天花板”构成：内存屋顶（memory roof，斜线）表示在受带宽限制时的峰值吞吐，计算屋顶（compute roof，平线）标出在受 ALU 限制时的峰值算术速率，如图 12-17 所示。

如果你的核函数靠近内存屋顶（例如低 FLOPS/byte），因而是访存受限（memory bound）的，那么最佳优化就是那些把内存传输与计算隐藏或重叠起来的手段。这意味着你应当用 CUDA streams 做异步拷贝——甚至并发运行多个访存受限的核函数。这样，你可以更好地打满内存系统的不同部分。

![图 12-17. 带两个硬件天花板的算术强度：访存受限（例如数据变换操作）与计算受限（例如矩阵乘法）](AI%20Systems%20Performance%20Engineering-ch12_images/figure-12-17.png)

> 核函数融合对访存受限的工作负载只有适度帮助。它能省去若干次中间的全局内存往返。但真正的收益来自于掩盖延迟，以及让更多的加载/存储在途并发。

相反，一个位于计算屋顶之下的高强度核函数需要让它的 ALU 保持忙碌。这里核函数融合便大放异彩，它把分开的 add+scale（前面我们的融合示例）合并成一遍，以提高每字节的 FLOPS、把点在 Roofline 图上向右移动，并把核函数推向更高比例的峰值 FLOP/s。

同样地，持久化核函数、线程块簇（thread block cluster）以及设备端发起的 CUDA Graphs 并不改变你的强度数值，但它们减少了反复启动所造成的空闲间隙（idle gap）。这把你核函数的性能推向那条平坦的计算天花板。

许多真实工作负载落在两者之间。它们既不是强访存受限，也不是强计算受限。在这些情况下，并发是你的好帮手。通过并行启动若干中等强度的核函数——无论是用流、并发的图，还是多个持久化核函数——你把它们组合起来，使得聚合吞吐点在两个坐标轴上都处于更高位置。这更好地利用了设备的资源。

彻底的 Roofline 分析需要严谨的测量。用 Nsight Compute 统计 FLOPS 与传输的字节数，把你核函数的点画出来，看看它离每条 Roofline 有多远。

如果工作负载是访存受限的，就动用流、重叠，也许还可以降低精度（FP16、FP8、FP4），以减小算术强度方程中的分母（例如传输的字节数）。

而如果你的核函数是计算受限，却因为启动开销或空闲期而未触及峰值 FLOPS，就把重点放在减少启动开销上。正如我们所学，你可以通过把操作融合进单个核函数、使用持久化核函数、捕获 CUDA Graphs，或执行设备端启动来做到这一点。这会让 ALU 一直有活干。

如果你的核函数远低于两条屋顶，既非完全访存受限也非完全计算受限，那就尝试提高并发。并行运行多个核函数与流。这会更好地利用你所有的系统资源。

有了这套定量的指导，你就能为你的核函数挑选正确的编排策略，而不是把所有招数一股脑全试一遍。只要记住验证每次优化都让你更接近硬件的真实潜力。在应用优化之后，务必测量。

Roofline 模型引导预期，但真实的性能测量——包括计算吞吐、实测占用率、内存吞吐等——才会道出完整的故事。把 Roofline 分析与迭代式、持续性的性能剖析相结合，将验证你所选的优化策略是否真的有效。

## 关键要点

达到峰值 GPU 性能，关键在于以最小的开销把计算与数据移动交织在一起。高效的编排让复杂工作负载在 CPU 与 GPU 之间流畅运转，确保任何一方都不拖住另一方。以下是本章的一些关键要点：

_用 L2 缓存原子队列做动态调度_ 现代 GPU 上的 L2 缓存原子操作异常快。用快速的 L2 缓存原子操作配合批量自增，在 GPU 上平衡不规则工作负载。这种批量式的工作分配减少了争用（contention），并通过消除 warp 空闲间隙让 warp 保持忙碌。在极端不均衡的情况下它能显著提升吞吐，最高可达 ~2×，但通常在 10% 到 30% 之间。由于 L2 带宽很高，即便是 8 或 16 这样不大的 batch size 也能消除大部分争用。

_用 CUDA Graphs 处理固定流水线_ 把一系列 GPU 操作录制一次，然后每次迭代用单次主机调用重放它。这减少了每次迭代的 CPU 调度开销，通常能降低 20%–30% 的延迟（规模越大，降幅越大）。要确保你在 GPU 上实现了相互依赖操作的最大重叠。

_用 CUDA Graphs 做低开销启动_ 把一系列异步拷贝、核函数启动、事件记录和分配捕获进一个 CUDA Graph（cudaStreamBeginCapture/cudaStreamEndCapture）。用 cudaGraphLaunch 重放该图，可在保留所有流间依赖的同时消除每次调用的 CPU 入队开销，进一步减少运行时瓶颈。

_设备端编排_ 通过尾部启动（tail-launch）一个预先录制的 CUDA Graph，或使用动态并行来派生子核函数，从 GPU 自身启动工作。这彻底消除了 CPU 调度间隙，并让 GPU 端到端保持忙碌，无需任何主机介入。

_多 GPU 重叠_ 始终把通信与计算重叠。用不同的流把 GPU 对等传输（cudaMemcpyPeerAsync）、NCCL 集合操作、CUDA-aware MPI（RDMA）或 NVSHMEM 单边操作做成流水线。这把通信延迟隐藏在有用的工作之后，并且在有利的计算与通信比以及充分重叠的条件下——甚至跨集群节点——当重叠足够时，可以逼近多 GPU 的线性扩展。

_Roofline 引导的抉择_ 让 Roofline 图来驱动你的策略。如果你的核函数是访存受限，就把重点放在重叠以及用异步内存拷贝和 FP8/FP4 之类的混合精度来减少数据移动上。如果它是计算受限却因开销而表现不足，就使用核函数融合、持久化核函数和 CUDA Graphs 之类的启动削减技术去逼近计算天花板。对于介于两者之间的核函数，通过并行运行多个操作来提高并发，以利用所有硬件单元。始终用性能剖析来验证所选优化确实产生了效果。

通过把这些技术交织在一起——动态分派、协作核函数、图捕获/重放以及 GPU 原生的内存共享——你打造出的流水线能为超大规模 AI 工作负载打满 GPU 集群的每一个部分。

## 结论

在本章中，我们超越了单核函数优化，探索了端到端的编排技术。我们讲解了如何用动态并行完全在设备上启动工作、如何把复杂的工作流捕获进 CUDA Graphs，以及如何用 NCCL 和 NVSHMEM 协调众多 GPU。每种技术都共享同一个目标：让每台引擎都有活可干、隐藏延迟、并压平主机-设备之间的间隙，从而让你的硬件全速运转。

NVIDIA 的现代 GPU 平台比以往任何时候都更加模糊了 CPU 与 GPU 之间的界线。例如，Grace Blackwell 和 Vera Rubin Superchip 用带有巨大带宽的一致性 NVLink 把 CPU 与多个 GPU 连接起来。

但即便硬件在降低 CPU-GPU 之间的壁垒，充分发挥这种高性能硬件的责任仍然落在软件身上。本章中的方法，无论是用 C++ 中的 CUDA 还是更高层的库 API，都是我们利用这些进步的途径。

在下一章，我们将看到 PyTorch 如何整合这些理念中的许多——包括流、图、异步操作和优化过的核函数——让你只用几行 Python 就能获得这种性能。让我们深入 PyTorch 生态系统，理解它为何在实现高性能 AI 工作负载方面如此受欢迎。
