NVIDIA 的多节点 GB200 与 GB300 NVL72 机架方案通过 NVLink Switch 将多达 72 个 GPU 连接进单一 NVLink 域，每跳延迟极低——大约仅数百纳秒。GB200 NVL72 架构可为机架内所有 GPU 提供高达约 130 TB/s 的全互联（all-to-all）带宽，且延迟低于微秒级。如果你的集群包含这类基础设施，请确保作业被调度到同一个 NVLink 域内，以充分利用这条超高速互连。这可以大幅降低对节点间较慢的 InfiniBand 与 Ethernet 通信的依赖。所幸，现代 InfiniBand 交换机（如 NVIDIA 的 Quantum 系列）每条链路可提供高达 800 Gb/s 的带宽，并具备在网计算特性。不过，NVLink 巨大的机架内带宽与 < 1µs 的延迟仍是首选。请尽可能让流量保持在 NVLink/NVSwitch 上。

**尽可能使用 RDMA**

如果运行在 InfiniBand 或支持 RoCE 的硬件上，请确认你的通信库（如 NCCL）确实在使用 RDMA。若条件允许，NCCL 会自动使用 GPUDirect RDMA。但如果 RDMA 配置错误或不受支持，NCCL 可能会悄无声息地回退到 TCP。一个警示信号是：如果你注意到在执行 all-reduce 操作时 GPU 利用率下降、CPU 利用率飙升，这表明 CPU 正在为通信复制数据。

**如有条件，用多张网卡聚合带宽**

一些服务器配有多个网络接口（网卡，network interface card，NIC）。NCCL 可以将流量分条（striping）到多张网卡上（称为多轨，multi-rail）以提升带宽。但你可能需要设置一些环境变量，如 NCCL_NSOCKS_PERTHREAD 和 NCCL_SOCKET_NTHREADS 来对此进行优化。稍后我们会更详细地讨论这些。只要确保每张网卡处于不同的子网上，并且 NCCL 能发现二者即可。经过正确配置后，例如并行使用两张 800 Gbps 网卡，可为 NCCL 流量提供 1.6 Tbps 的聚合带宽。而四条这样的网卡链路（例如两张双端口网卡）可达到约 3.2 Tbps。

**在可用时启用优化的“直连网卡”（direct NIC）模式**

优先采用高带宽、多轨的网卡配置，为每个 GPU 或每一小组 GPU 提供充足的专用网络带宽。在物理层面，网卡通过 PCIe 连接到主机 CPU 或 DPU。在现代 GPU 系统上，NCCL 支持通过 InfiniBand GPUDirect Async（IBGDA）和直连网卡路径实现 GPU 发起的网络通信，如图 4-4 所示。这让 GPU 无需 CPU 介入即可驱动全带宽 RDMA。

![图 4-4. 通过 GPU 与网卡之间的直连绕过 CPU 瓶颈](AI%20Systems%20Performance%20Engineering-ch4_images/figure-4-4.png)

**排查配置错误**

一个常见陷阱是网络配置不匹配，导致回退到较慢的路径。例如，如果 RDMA 因配置错误而无法工作，NCCL 可能会在 100 Gbps Ethernet 网络上使用 TCP，却由于内核开销只能得到其中一小部分带宽。更糟的是，如果集群的高速网络被错误识别，流量可能会走一条仅有 10 Gbps Ethernet 的较慢管理网络，而用户却毫无察觉。像 NCCL 的调试输出以及网络接口计数器（ibstat、ifstat）这样的工具，可以帮助你验证哪个接口被更多地使用。对于具备 200–400 Gbps 大带宽路径的现代系统来说，跌落到 10 Gbps 会造成严重瓶颈。

### 多节点通信陷阱

将训练扩展到集群中的多个节点会引入一类新的陷阱。这里我们重点介绍几个常见问题，并用具体示例演示如何修复它们。

陷阱 #1：使用受 CPU 限制的 Gloo 后端而非 NCCL

PyTorch 的分布式框架支持多种通信后端。对于多 GPU 训练，NCCL 是 NVIDIA GPU 的首选后端，但还有一个名为 Gloo 的回退后端，它使用 CPU 和 TCP 套接字。如果有人错误地为 GPU 训练用 Gloo 初始化 ProcessGroup——或者 NCCL 初始化失败并回退到 Gloo，训练仍能正确运行，但所有跨 GPU 通信都会走 CPU 和 Ethernet 协议栈。这会导致性能极其低下。

不幸的是，意外陷入这种错误配置相当常见，因为代码看上去运行正常、不会崩溃。它只是慢了一个数量级，需要借助性能剖析器或仔细的日志分析才能发现。总之，对于多 GPU 训练，务必指定 NCCL 以利用 RDMA 通信。所幸这正是 PyTorch 的默认值。否则，回退到 Gloo 会悄无声息地限制你的性能（甚至完全报错）。

让我们用代码来说明这一点。我们将模拟两个进程，分别运行在两个通过 800 Gb/s（100 GB/s）InfiniBand 互连相连的不同“节点”上的 gpu/rank 0 与 gpu/rank 1 上，并对一个张量执行一次大规模 all-reduce。首先，我们有意为 PyTorch 分布式通信使用 Gloo 后端（受 CPU 限制），如下所示：

```
# dist_allreduce.py

#!/usr/bin/env python

import os
import argparse
import torch
import torch.distributed as dist

def main():
    parser = argparse.ArgumentParser(description="Multi-node Gloo all-reduce")
    parser.add_argument(
        "--data-size",
        type=int,
        default=1024 * 1024 * 100,  # 100M floats ≈ 400 MB
        help="Number of elements in the tensor",
    )
    args = parser.parse_args()

    # Initialize the default ProcessGroup over env://
    # (uses MASTER_ADDR, MASTER_PORT, etc.)
    dist.init_process_group(backend="gloo", init_method="env://")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Allocate a large CPU tensor (Gloo is CPU-bound)
    tensor = torch.ones(args.data_size,
       dtype=torch.float32, device="cpu")

    # Warm up and barrier
    dist.barrier()

    # All-reduce (sum) across all ranks
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    # Barrier
    dist.barrier()

    ...

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

在这段代码中，我们有意选择了 backend="gloo"。由于 Gloo 受 CPU 限制，每个 rank 的张量（400 MB）都被分配在 CPU 上（device="cpu"）。使用 Gloo 时，集合操作在 CPU 内存上进行，并通过 TCP 通信。如果你改用 GPU 张量配合 Gloo，PyTorch 会经由 CPU 中转，并受制于这条路径。无论哪种方式，结果都远慢于在 GPU 上使用 NCCL。相比于让 GPU 通过 RDMA 直接对话，这种方式效率低下。

在前面的代码中，我们有意选择了 backend="gloo"。如果你用 Gloo 进程组尝试 dist.all_reduce()，运行会视构建情况而定，要么回退到经主机中转的路径，要么直接失败。这就失去了测量 GPU 集合操作的意义。

在带有正确计时的情况下运行这段代码会得到如下结果：

```
Rank0: All-reduce of 400.0 MB took 200.00 ms (2 GB/s)
```

对于 400 MB 的数据，200 ms 相当慢，因为这只是 2 GB/s 的聚合吞吐——远低于预期，也远低于本例中所用 800 Gb/s InfiniBand 硬件应有的 100 GB/s。这表明 CPU 路径上产生了大量额外开销。我们通过剖析 CPU 利用率证实了这一点——在这种情况下，CPU 利用率接近 100%。

让我们把后端改为 NCCL，看看差异。我们只需设置 back end='nccl'，并确保环境配置允许 GPU 直连通信。假设 NCCL 已正确配置，这段代码将使用 GPU 到 GPU 的直接通信。改进是惊人的。带计时运行更新后的代码会显示类似如下的结果：

```
Rank0: All-reduce of 400.0 MB took 4.00 ms (100 GB/s)
```

这里，我们看到 400 MB 只用了 4 ms，即 100 GB/s。这比之前快了两个数量级——并且运行在我们 800 Gb/s InfiniBand 硬件的线速极限上。这远好于之前用 Gloo 得到的 2 GB/s。这说明了在 GPU 多节点通信中使用 NCCL 是多么关键。使用错误的后端会显著降低性能。简而言之，请使用 backend="nccl"，以便集合操作在 GPU 上运行，并在可用时使用 GPUDirect RDMA。

> 你可以在 PyTorch 中调用 torch.distributed.get_backend() 来验证所用的后端。

从 GPU 的角度看，NCCL 的 all-reduce 集合操作是由 GPU 通过对其他 GPU 内存的直接内存访问完成的。因此，GPU 会一直忙于通信：要么是 SM 在执行某些网络复制内核，要么是 GPU 的 DMA 引擎处于活动状态。相比之下，使用 Gloo 时，GPU 在通信期间基本处于空闲状态，因为其中涉及 CPU，GPU 必须等待数据经由 CPU 的内存缓冲区通过 TCP（而非 InfiniBand）传输。

> 在拥有多张网卡的生产集群环境中，你应当显式设置 NCCL_SOCKET_IFNAME=ib0，以便 NCCL 的初始 TCP 握手在 InfiniBand 主机通道适配器（host channel adapter，HCA）上进行。这可以确保它正确地引导启动（bootstrap），随后在最快的路径上切换到 GPUDirect RDMA。否则，你可能会看到连接失败，或更糟——悄无声息地回退到一个慢得多的接口。请确保所有节点都能通过所选的互连彼此可达。

陷阱 #2：NCCL 版本不匹配

如果你让 PyTorch 自带的 NCCL（例如 torch.cuda.nccl.version() == ()）与系统安装的另一个版本的 libnccl 一起运行，系统会被挂起。或者更糟——它会悄无声息地回退到较慢的实现。这可能很难发现。请通过匹配 nvidia-nccl-cu\* 软件包，或针对系统 NCCL 重新编译 PyTorch，来确保版本对齐，从而避免这些兼容性陷阱。

陷阱 #3：NCCL 引导启动期间的 TCP 端口耗尽

NCCL 使用临时（ephemeral）TCP 端口进行其带外（out-of-band）建立，如果你操作系统的 net.ipv4.ip_local_port_range 过窄，可能会耗尽可用端口，导致握手失败或停滞。建议你在 /proc/sys/net/ipv4/ip_local_port_range 中拓宽端口范围（例如 50000 51000），以避免隐藏的引导启动失败。请注意，现代 NCCL 版本已改进了引导启动处理，但在大型集群上，主动设置一个较宽的端口范围仍是最佳做法。

陷阱 #4：网络带宽不足或网卡配置错误

另一个多节点陷阱很简单，就是相对于需要同步的数据量而言网络带宽不够——或者没有用上所有可用接口。随着 GPU 集群规模扩大，这个陷阱会越来越常见。例如，用 Blackwell GPU 很容易就把每节点单条 400 Gbps 链路打满。

在这些不理想的条件下剖析你的工作负载时，你会观察到扩展到多个节点会显著拖慢训练。换句话说，“每 GPU 吞吐量”会下降。这种情况下，请检查网络链路。

监控网络吞吐量往往很有用，例如可以用 nvidia-smi dmon 来收集 NVLink/PCIe/Network 统计信息。你也可以使用内置工具，如用 ethtool -S <iface> 或 ip -s link show <iface> 查看字节/数据包计数器，或启动交互式监视器（如 iftop 或 nload）来实时观察网卡吞吐量。

如果条件允许，你还可以尝试利用多个接口。例如，如果你正把一条 800 Gbps（100 GB/s）InfiniBand 链路打满，而作业还需要更多网络吞吐，可以考虑启用 NCCL 的多网卡支持——前提是你有多张网卡。请确保调好 NCCL_NSOCKS_PERTHREAD 和 NCCL_SOCKET_NTHREADS，因为它们控制着 NCCL 用于网络传输的并行连接数和线程数。在有多张网卡的情况下，将这些环境变量的值从其依赖于平台的默认值上调，有助于同时利用两张网卡。

例如，如果你有两张网卡，可以设置 NCCL_NSOCKS_PERTHREAD=2 并保持 NCCL_SOCKET_NTHREADS=2（因为 2 线程 × 2 套接字 = 每进程共 4 个连接）。不过，不要随意增大这些值。请记住，按照 NVIDIA 的指导，线程数与套接字数的乘积不应超过 64，因为线程越多意味着 CPU 占用越高。

> 请逐步增大这些与线程相关的设置（例如 2 → 4 → 8），并持续测量吞吐量。线程过多会争抢资源，可能导致收益递减。

陷阱 #5：掉队节点（straggler）或掉队进程

在多节点训练中，最慢的那个节点或 GPU 将决定整体节奏，因为同步需要等待每个节点和 GPU 响应。如果某台机器的网络链路较慢，或被其他任务拖累，它会拖慢整个作业。

为避免掉队节点，尽可能为每个训练作业使用同构硬件和专用的集群资源很重要。这样一来，你的环境是可预测的。例如，如果你运行在云环境中，混用不同的实例类型或使用不同的交换网络（switch fabric）会引入变异性。

使用像 NVIDIA 的 DCGM 或每个节点上的 InfiniBand 计数器这类监控工具，可以帮助发现某个节点是否因网卡链路抖动（link flapping）或 GPU 热降频（thermal throttling）而性能下降。使用集合操作剖析工具（如 PyTorch 的 torch.distributed.monitored_barrier）来识别某个特定 rank 是否持续滞后，也很有用，如下所示：

```
# barrier_straggler

import torch
import torch.distributed as dist
import os
import datetime

def run(rank, world_size):
    dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # ... your forward/backward work here ...

    # Before syncing at end of iteration, use a monitored barrier:
    try:
        # Wait up to 30 seconds for all ranks
        # if one lags, you’ll get a timeout on that rank
        dist.monitored_barrier(timeout=datetime.timedelta(seconds=30))
    except RuntimeError as e:
        print(f"Rank {rank} timed out at barrier: {e}")

    # Now proceed knowing all ranks are roughly in sync
    dist.destroy_process_group()
```

这里，dist.monitored_barrier(timeout=datetime.timedelta(seconds=30)) 会对任何未能在 30 秒内到达的 GPU 抛出错误。这将帮助你精确定位掉队节点。把它与 NCCL_DEBUG=INFO 和 NCCL_ASYNC_ERROR_HANDLING=1 结合使用，可以同时获得 PyTorch 与 NCCL 的日志，从而了解是哪个 rank 或链路变慢。

应当使用现代监控工具（如 PyTorch 的 torch.distributed.monitored_barrier 和 NCCL 的异步错误处理）来快速检测这类问题。

陷阱 #6：UCX/RDMA 下的 GPU 内存碎片化

PyTorch 的缓存分配器（caching allocator）会跨迭代持有 GPU 内存。在使用 UCX/RDMA 的分布式场景中，这些长期存活的分配可能会耗尽注册池（registration pool）或造成内存碎片化，从而导致偶发的分配失败或性能悬崖。监控 torch.cuda.memory_reserved() 与 memory_allocated() 有助于暴露这些边缘情况：

```
import torch
import torch.distributed as dist
import os
import time

def log_mem(iteration):
    reserved = torch.cuda.memory_reserved()
    allocated = torch.cuda.memory_allocated()
    print(f"[Iter {iteration:02d}] Reserved: {reserved/1e9:.3f} GB, "
          f"Allocated: {allocated/1e9:.3f} GB")

def run(rank, world_size):
    # Standard DDP / UCX init
    dist.init_process_group(backend="nccl", init_method="env://")
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)

    # Pre-allocate a big buffer that UCX will register once and hold
    big_buffer = torch.empty(int(2e8), device=rank)  # ~0.8 GB
    log_mem(0)

    for i in range(1, 11):
        # Simulate per-iteration tensor allocations of varying size
        small = torch.randn(int(1e7), device=rank)  # ~40 MB
        medium = torch.randn(int(5e7), device=rank) # ~200 MB

        # Free them explicitly to return to allocator cache
        del small, medium
        torch.cuda.synchronize()

        # Log memory after freeing
        log_mem(i)
        # Barrier so all ranks print in sync
        dist.barrier()
        time.sleep(0.1)

    dist.destroy_process_group()

if __name__ == "__main__":
    run(0, 1)
```

这段输出代码展示了随着缓存分配器持有已释放的内存块，保留（reserved）内存如何在每次迭代中增长。这会加速未来的分配。然而，它永远不会把这些内存归还给操作系统或 UCX 注册池。由于没有存活的张量残留，已分配（allocated）内存在每次释放后都回落到零：

```
[Iter 00] Reserved: 0.800 GB, Allocated: 0.800 GB
[Iter 01] Reserved: 1.040 GB, Allocated: 0.000 GB
[Iter 02] Reserved: 1.240 GB, Allocated: 0.000 GB
...
[Iter 10] Reserved: 1.240 GB, Allocated: 0.000 GB
```

要缓解这种情况，请务必升级到最新的 CUDA 运行时。你也可以尝试把 torch.cuda.empty_cache() 作为从碎片化问题中恢复的最后手段。不过，empty_cache() 并非长期解决方案。一些长期方案包括：调优分配器、追查问题根源并加以修复。

我们把多节点陷阱总结如下：始终为 GPU 通信使用 NCCL；确保 RDMA/高速网络处于活动状态；用尽所有可用带宽（尽可能使用多张网卡）；使用随 CUDA 一同发布的最新 NCCL 以继承最新修复；并警惕任何会导致回退到较慢通信的配置。在接下来的几节中，我们将聚焦 NCCL 的具体细节，以及如何优化节点内（intranode）与节点间（internode）的 GPU 通信。

## 用于分布式多 GPU 通信的 NCCL

NVIDIA NCCL 是一个多对多通信库，用于被称为集合操作（collectives）的操作，供成组的 GPU 共享数据。NCCL 支撑着 NVIDIA 生态系统中大多数多 GPU 训练工作负载。

NCCL 为 all-reduce、all-gather、broadcast 和 reduce-scatter 等集合通信操作提供了优化实现，可从数个 GPU 扩展到数千个，乃至未来的数百万个。在跨多个 GPU 执行模型训练和推理时，模型权重、梯度和激活值等数据必须快速交换，才能让 GPU 保持繁忙。NCCL 正是高效编排这些交换的库。

在分布式训练期间，每个 GPU 在其那部分数据上计算梯度。随后 NCCL 被用于对所有 GPU 上的这些梯度执行一次 all-reduce，使每个 GPU 都用平均后的梯度来更新模型权重。

在分布式推理期间，GPU 需要交换激活值和其他中间结果。

在推理场景中，一些框架使用 NCCL 的 send() 和 recv()，但许多部署更青睐通过 UCX 暴露的传输，或像 NIXL 这样的专用库，以获得更低的尾延迟和更好的重叠。

NCCL 针对 NVIDIA GPU 进行了优化，并支持在多种互连上通信，如 PCIe、NVLink、NVSwitch、InfiniBand 和 TCP 套接字。它会在任意两个 GPU 之间自动选择可用的最快路径。

与 NCCL 互补的是更新的 NVIDIA Inference Xfer Library（NIXL），它针对推理以及像 KV 缓存搬移这类点对点传输进行了优化。

NIXL 提供可插拔的存储后端，包括 POSIX 文件和 GPUDirect Storage（GDS）。对对象存储（如 Amazon S3）的支持由其对象存储插件提供，并取决于具体部署。这些插件会在适当时机把 KV 缓存片段在内存层级与存储后端之间搬移。稍后我们会深入介绍 NIXL——以及在后续聚焦推理工作负载调优的章节中进一步展开。

> 如后文所述，由于延迟更低，如今许多推理部署更青睐用 NIXL 来做这类点对点传输。不过，NCCL 的 send() 和 recv() 同样仍然可用。只是它们在最小化延迟方面不如 NIXL 优化得那么好。实践中，大规模推理工作负载因 NIXL 更低的开销和延迟而更青睐用它来做一对一传输，而 NCCL send/recv 则在需要定制集成时才更少见地被使用。

### NCCL 中的拓扑感知

拓扑感知在 NCCL 的性能中扮演着重要角色。NCCL 会检测 GPU 在物理上是如何连接的，并据此优化其通信模式。例如，在一个由全互连 NVLink 和 NVSwitch 构成的系统中，每个 GPU 都会使用这些高速互连与其他每个 GPU 通信。

虽然 NCCL 可以使用像环形 all-reduce（ring all-reduce）这样的简单模式对每条链路一视同仁地通信，但它会自动采用拓扑感知的分层通信模式来最大化通信性能。例如，对于有多个 NUMA 节点域的系统，NCCL 可能会先做一次节点内归约，再做一次跨节点归约，然后做一次节点内广播，这实际上就是一次分层 all-reduce。目标是让尽可能多的流量走最快的互连。

具体来说，考虑这样一种拓扑：GPU 0–1 共享一条 NVLink 连接，GPU 2–3 共享另一条 NVLink 连接。然而，0–1 这一对与 2–3 这一对之间的任何通信都必须走较慢的 PCIe 互连。在这种情况下，NCCL 的分层算法会先在每一对 NVLink 相连的 GPU 上执行归约集合操作，然后由每对中的一个 GPU 通过 PCIe 链路做一次归约交换，接着再在每一对内部分发数据。这样一来，缓慢的 PCIe 链路只处理一小部分数据。

当 NCCL 检测到这类拓扑时，通常会选择性能最优的方法。不过在某些情况下，自动检测可能不会启用，例如各拓扑选项性能相当时——或者消息尺寸很小时，等等。可以用环境变量 NCCL_ALGO（例如 NCCL_ALGO=NVLS,NVLSTree,Tree,Ring,PAT 等）来覆盖 NCCL 的算法选择，但一般而言，NCCL 能够根据拓扑很好地自动选择最佳路径。手动覆盖通常只用于特定场景，如故障排查、研究实验等。

为了说明拓扑对 NCCL 的影响，考虑这样一个场景：一个系统上有四个 GPU，配有两个 PCIe 交换机（GPU 0–1 在一个交换机上，2–3 在另一个上）。若采用朴素的做法，在全部四个 GPU 上用单个环执行 all-reduce，最终会有大量数据经过 PCIe 交换机间链路传输。这将成为一个重大的通信瓶颈。相比之下，分层的做法——用 0–1 和 2–3 两个独立的环，再配合每个环中各取一个 GPU 之间的交换——会降低通信压力。实践中，如果 NCCL 检测到拓扑中较慢的链路，它会在底层选择这种通信模式。

用剖析工具做一个快速实验就能揭示 NCCL 是否做了拓扑优化。使用 Nsight Systems 或 NCCL 追踪，当采用分层时你会看到多个 NCCL CUDA 内核。例如，你会看到一些组内 all-reduce 内核和一些组间内核。我们还会看到性能差异。例如，未经优化、不感知拓扑的算法每 GPU 只能达到数十 GB/s 量级，因为 all-reduce 的其中一个阶段走了较慢的 PCIe 链路；而拓扑感知的算法通过充分利用 NVLink 并最小化 PCIe 使用，每 GPU 可达到数百 GB/s。

考虑一种朴素做法：GPU 在 PCIe 互连上等待，测得的 SM 利用率仅为 60%，且在无重叠的情况下单次迭代总时间为 100 ms。在这种情况下，许多 warp 因内存访问而停顿，因为它们在等待经由缓慢 PCIe 互连传输的数据。然而在拓扑感知的做法中，SM 利用率跃升至 90%，单次迭代总时间下降了 30%，从 100 ms 降到 70 ms——表明内存停顿周期大大减少。这说明 GPU 得到了充分得多的利用、等待数据的时间更少，因为传输是经由 NVLink 而非 PCIe 进行的，如表 4-2 所示。

_表 4-2. 应用拓扑感知 NCCL 与计算–通信重叠优化前后的关键 GPU 性能指标_

| 指标             | 之前（无重叠）   | 之后（有重叠）  |
| ---------------- | ---------------- | --------------- |
| SM 繁忙度        | 60%              | 90%             |
| 内存停顿 warp 数 | 高               | 低得多          |
| 迭代时间         | 100 ms（无重叠） | 70 ms（有重叠） |

总之，拓扑感知能够决定多 GPU 系统扩展性能的成败。一条经验法则是：尽可能让通信保持在可用的最快互连上——对于节点内通信而言，很可能就是 NVLink/NVSwitch——并尽量减少经由 PCIe 和跨 NUMA 节点链路等较慢路径的传输。

如果你的多 GPU 作业在单节点上扩展不佳，请先确认流量没有被强制走到较慢的路径上。并且要记住，GPU 的直连 NVLink 通道数量是固定的。例如，GB200/GB300 NVL72 系统中的每个 Blackwell GPU 支持 18 条 NVLink 5 链路，每条约 100 GB/s，合计约 1.8 TB/s 的 GPU 到 GPU 双向聚合带宽。这是上一代 900 GB/s 的两倍。

在没有直接配对的设备之间通信，可能会让你退回到更少的通道——甚至退回到 PCIe。如果数据需要跨越 NUMA 域，你的吞吐量会显著下降。在一个 NVL72 机架中，全部 72 个 Blackwell GPU 都属于同一个 NVLink Switch 域。在一个 NVL72 机架内，任意 GPU 都能以全对分带宽（full bisection bandwidth）在单个 NVSwitch 级内到达任意其他 GPU。该 GPU 域提供了带 NVLS 支持的、均匀的全互联连通性。

NCCL 的分层算法应当会自动挑选带宽最高的路由，但你应在剖析中确认这一点。如果你仍然撞上带宽墙，请把作业约束到一个紧密相连的 GPU 子集上。

例如，同一 NUMA 节点上或同一 NVSwitch 岛（island）内的四个 GPU，要远好于跨越较慢链路分散到全部八个 GPU。经由那些受限或间接链路的额外同步开销，往往会盖过使用额外设备所带来的收益。

还值得一提的是，NVIDIA 超级芯片（superchip）模糊了 CPU 与 GPU 内存之间的界线。例如，它们在 Grace Blackwell Superchip 中提供了 CPU 与 GPU 之间约 900 GB/s 的 NVLink-C2C 这类极快的 CPU-GPU 互连。这让 CPU 内存能够充当 GPU 内存的高速扩展。

这意味着，即便 all-reduce 的某一部分涉及 CPU 或系统内存，它也可能仍然和老一代的 GPU-GPU 链路一样快。关键要点是：通过使用可用的最快路径来优化你的节点内通信——并尽量减少对慢速路径的使用。

> 像 NVIDIA Nsight Systems——或 NCCL 自带的、启用 NCCL_DEBUG=INFO 与 NCCL_TOPO_DUMP_FILE=<path> 的追踪——这类工具会显示 NVLink 路径是否被充分利用。

### NCCL 通信算法

在内部，NCCL 可以根据数据大小、GPU 数量和拓扑采用不同的通信算法。NCCL 用于集合操作的主要算法包括 Ring、Tree、CollTree、CollNet 和 Parallel Aggregated Tree（PAT）等。下面我们逐一深入了解：

**Ring**

在环形 all-reduce 中，GPU 在逻辑上被排成一个环。每个 GPU 以流水线方式向它的一个邻居发送数据，并从它的另一个邻居接收数据。对于 all-reduce，每一块数据都会绕环流转，累加部分和。环形算法有一个很好的特性：它能完美地均衡网络负载，使每个 GPU 发送和接收的数据量都完全相同。它是带宽最优的，因为在一次 all-reduce 集合操作中，每条链路传输 2 × (data_size ÷ num_gpus) 字节。缺点是延迟。总时间随 GPU 数量增长，因为数据必须遍历所有跳。Ring 通常非常适合大消息，因为当你发送非常大的消息时，实际搬移字节所花的时间远远盖过真正启动传输的开销。这被称为带宽主导（bandwidth-dominated）的工作负载。

**Tree 与 NVLSTree**

在 Tree 算法中，归约和广播使用生成树（spanning tree）算法以树形结构完成。一次 all-reduce 实际上是一次 reduce-scatter 后接一次 broadcast。对于 N 个 GPU，树可以在 O(log N) 步内完成一次 all-reduce——而环则需要 O(N)。因此，Tree 算法为较小的消息提供了更低的延迟。不过，对于大消息它可能无法充分利用所有链路，因为并非所有 GPU 都始终在传输。例如，有些 GPU 是树的叶子节点，只沿树向上发送一次。NCCL 的 Tree 算法经过优化，常用于较小的消息尺寸——此时总时间由传输启动延迟主导。这被称为延迟主导（latency-dominated）的工作负载——与环形算法针对大消息的带宽主导用例相对。使用 NVLSTree 将启用 NVLink SHARP 卸载。

**CollTree（分层树形集合操作）**

CollTree 构建一棵两级树，以在保持高本地带宽的同时降低延迟。在每个快速的本地域内（例如一个节点上的所有 GPU，或一个 NVSwitch 岛内的所有 GPU），它形成一棵本地树，并执行一次 reduce-scatter 后接一次 broadcast。随后，每个本地组各派一个领导者（leader）通过 RDMA 参与到组间的第二级树中。这两级是流水线化的，使得完成了本地阶段的张量块可以立即进入跨组阶段。与扁平的环相比，这将跨节点步数降到 O(log N)，并缩短了中小消息的延迟。同时，它在节点内部仍然使用全带宽。在 NVLink 域上，本地树经由 NVSwitch 路由，并可在条件允许时使用 NVLink SHARP 做交换机内聚合。在 InfiniBand 网络上，当启用 NCCL SHARP 插件时，组间树可以卸载到 SHARP。对于非常大的消息，Ring 或 Parallel Aggregated Tree（即 PAT，本列表后文讨论）可能达到更高的峰值吞吐，而当跨节点延迟占主导时则更青睐 CollTree。

**CollNet（跨节点的分层集合操作）**

CollNet，也称为树并行（tree parallelism），结合了两种集合策略，以在不同尺度上优化通信。首先，它把共享一条快速本地互连的 GPU 分为一组，例如单个节点内或一个 NVSwitch 岛内的所有 GPU。接着 CollNet 应用一种高吞吐算法（如环或本地树）在每组 GPU 内部聚合数据。每组中一个指定的领导者 GPU 参与到组间的第二级树归约中。这将跨组通信轮次的数量降到最低。通过在全局树交换之上叠加一层本地归约，CollNet 同时为节点间传输提供低延迟、为节点内流量提供高带宽。这使它在降低超大型多节点 GPU 集群的网络负载方面尤其有效。

**Parallel aggregated tree（PAT）**
