# 第 4 章　分布式网络通信调优

在当今的 AI 版图中，GPU、存储与网络接口之间无缝、低延迟的数据搬运已成为刚需。本章将介绍用于训练的 NVIDIA Magnum IO（例如 NCCL、GPUDirect RDMA、GDS），以及用于分离式推理（disaggregated inference）的 NIXL（NVIDIA Inference Xfer Library）。我们将结合现代 GPU 与诸如 NVL72 这样的集群来讨论这些技术。你将了解到，这些库——以及其底层所支持的硬件——如何共同构成超大规模 AI 系统所需的关键“基础互连织物”（fabric）。

在大规模系统中，即便是最快的 GPU，也可能因通信低效以及来自内存和磁盘的数据传输不畅而受制。我们将讨论加速数据传输的策略、恰当的数据分片（sharding）技巧、如何直接与高速存储子系统协作，以及在 GPU 上重叠通信与计算的高级模式。通信与计算重叠是一种常见模式，在我们探索 AI 系统性能工程的旅程中会反复出现。

我们将借助 NVIDIA IO 加速平台 Magnum IO 的各个组件来探讨通信与计算重叠的重要性，这些组件包括 NCCL、GPUDirect RDMA 与 GPUDirect Storage（GDS）。我们会演示如何利用这些库来降低通信延迟、减少 CPU 开销，并在多节点、多 GPU 的 AI 系统各个层面上最大化吞吐量。

像 PyTorch 这样的高层 AI 框架可以利用这些底层库来实现计算与通信的重叠。将这些技术集成到你的 AI 系统中，代表着一种整体性的思路：既为训练超大规模模型加速通信与数据流水线，也为扩展分布式推理服务器以支撑面向数十亿用户的应用提供动力。

所有这些优化确保了每个组件都被调优至峰值性能。性能工程师需要精心配置和调优网络与存储织物，以维持较高的 GPU 利用率与“有效吞吐”（goodput，即有用的吞吐量）。

## 通信与计算重叠（流水线化）

通信与计算重叠，即流水线化（pipelining），在大规模构建高效的训练与推理系统中扮演着关键角色。在这类环境里，让 GPU 保持忙碌、减少等待数据的时间至关重要。

其核心思路是：让数据传输与正在进行的计算并发发生，从而当一项任务完成时，下一阶段所需的结果要么已在传输途中，要么已经送达。像 PyTorch 这样的现代框架支持异步操作，使得集合通信（collective communication，例如对梯度做 all-reduce）可以与计算任务并行运行。这减少了 GPU 的空闲时间（见图 4-1），并提升了整体系统吞吐量。

![图 4-1. 在多个 CUDA 流 0–3 上将主机到设备（host-to-device，H2D）与设备到主机（device-to-host，D2H）通信与计算重叠](AI%20Systems%20Performance%20Engineering-ch4_images/figure-4-1.png)

基于 CUDA 的库充分利用了多个 CUDA 流的能力。当一个流执行计算密集的矩阵乘法时，另一个流则处理诸如聚合梯度之类的通信任务。当神经网络的每一层完成其计算后，前一层的输出已在传输途中，等待聚合或进一步处理。这种重叠确保系统在产生结果时不必进行不必要的等待，并维持稳定的数据流动。

增大两次通信事件之间所执行的计算量，可以进一步降低通信的相对开销。当系统处理更大批量的数据时，它会在需要停下来交换信息之前执行更多计算。

以分布式训练为例，这体现为梯度累积（gradient accumulation）：将来自若干个小批量（minibatch）的更新合并为单次同步步骤。通过降低通信事件的频率，系统减小了每次交换的开销，从而提升整体吞吐量。

另一种有助于计算与通信无缝重叠的技术是压缩（compression）。压缩减少了需要传输的数据量。例如，如果模型在发送梯度之前先对其进行压缩，网络所搬运的数据量就更小。这缩短了传输时间并缓解了拥塞。

更短的传输时间意味着通信阶段对计算阶段的干扰更小。虽然压缩本身并不直接产生重叠，但它缩短了数据在网络中移动的时间窗口，从而让计算工作能够更有效地并行推进。

现代深度学习框架还会将大张量通信拆分成更小的桶（bucket），以促成重叠。像 PyTorch 这样的框架会自动将大梯度或激活张量切分为若干个桶，一旦某个桶就绪便立即发送。举例来说，与其等待整层的梯度全部就绪，不如让其中一部分立即开始各自的 all-reduce。

通过调节桶大小并恰当地调度这些传输，就能获得更高程度的重叠，避免通信延迟拖住计算流水线。诸如 PyTorch profiler 和 NVIDIA Nsight Systems 之类的工具能帮助你洞察计算与通信是否发生了重叠，使工程师得以调整这些参数以达到最高效率。

将更大的批大小、梯度累积、异步传输、压缩与分桶（bucketing）整合成一套连贯的策略，大型分布式 AI 模型便能够克服网络限制并减少空闲时间。这种设计在实现高吞吐量与最优 GPU 利用率的同时，最小化了同步事件。

这些优化的结果是：一个训练系统能够缩短整体训练与推理时间，并更高效地利用可用的硬件资源。工程师无须再重复造轮子去编写底层网络例程，而可以专注于创新模型架构、调优更高层的参数，而不必去编写繁复的数据传输机制。

### 异步流执行

实现重叠从根本上依赖于异步执行。GPU 支持多个流（stream），也就是多个操作队列，只要它们面向不同的资源，就可以并发或重叠执行。一个流可以处理诸如矩阵乘法之类的计算内核，而另一个流则处理诸如数据拷贝和 all-reduce 调用之类的通信。

通过将工作分派到不同的流并使用非阻塞（nonblocking）操作，通信便可以在后台进行。例如，可以把一次 all-reduce 操作发起到一个单独的流上，而不必等待其完成。

与此同时，默认流继续对相互独立的数据进行后续计算。这要求诸如 NCCL 之类的通信库使用能够立即返回控制权的非阻塞调用。这样，程序员便可在需要时确保正确的同步。

在实践中，AI 框架隐藏了其中大部分复杂性。PyTorch 的 DistributedDataParallel 会自动在反向传播上安装钩子（hook），使得每个梯度桶在一个专用的通信 CUDA 流上触发一次异步的 NCCL all-reduce，而默认 CUDA 流则继续为后续层计算梯度。

> 我们将在后续章节中更深入地探讨 CUDA 流，这里只需知道：它们对于重叠通信与计算——以及避免代码中不必要的同步点——非常有用。

这种通过 CUDA 流将计算与通信交错进行的方式，形成了一波接一波、层层递进的工作流水线，从而隐藏通信延迟并让 GPU 始终保持忙碌。为维持恰当的重叠，应避免因 torch.cuda.synchronize() 而产生不必要的同步点，也应避免因用 torch.Tensor.item() 将张量移动到 CPU 而无意中触发一次完整的设备同步。如果你确实需要测量整体迭代时间，可在迭代的最末尾放置单次同步，以等待所有未完成的 GPU 工作结束，而不会打断正在进行的流水线。

### 降低通信频率与数据量

如前所述，在每一次通信步骤中执行更多的工作，可以增加重叠并提升效率。模型训练期间的梯度累积就是这样一种技术。与其为每一个小批量都做梯度 all-reduce，不如在若干个小批量上累积梯度、在本地求和，然后只做一次 all-reduce。这实际上是用存储未归约梯度所需的内存，换取更少的同步点。

设想累积四个小批量，你就把 all-reduce 的频率削减为原来的 1/4（即降低 4×）。这让更多计算得以在两次同步之间进行。其代价是有效批大小（effective batch size）增大，这可能影响模型收敛（convergence）与内存占用。你通常能找到一个最佳平衡点，在通信频率、内存占用与收敛之间取得良好平衡。

另一种降低通信数据量的方法是对所交换的数据进行压缩或量化（quantizing）。诸如梯度压缩之类的技术，可以在不显著影响模型质量的前提下减少每次通信中发送的数据量。要发送的数据越少，传输越快，也就有更多机会把这些传输隐藏在计算之下。这种做法的极端形式是稀疏化（sparsification）。使用稀疏性时，你只发送一小部分梯度。不过，这通常需要对算法进行改动才能保持精度。

分桶——正如 PyTorch 的 Distributed Data Parallel（DDP）通信机制中所实现的那样——也能通过将许多小张量归并为更大的消息来降低单次调用的开销。然而，桶大小是一种权衡。非常大的桶能最大化带宽利用率，但会延迟通信的启动，因为你需要等待更多梯度累积起来才能发起 all-reduce。非常小的桶能更早开始传输，但会因大量小的 NCCL 调用而带来更多开销。

在本书写作之时，PyTorch DDP 的默认桶大小为 25 MB。这是一个在多数情况下都能良好重叠的折中值。不过，如果你的模型含有非常大的层，或许可以增大该值以降低开销；如果你的模型含有许多小层，那么更小的桶反而可能带来好处，让通信更早开始。归根结底，要实现最大重叠，可能需要对不同的桶大小进行性能剖析，看哪一个能带来最佳的迭代时间。

### 在实践中实现最大重叠

为了直观感受重叠通信与计算的好处，我们来跑一个例子，对比两种场景：一种是在所有计算完成之后同步地执行梯度通信、没有任何重叠；另一种则是像 DDP 那样将通信与计算重叠。我们将用两块 GPU 模拟一个简单的分布式训练步骤来说明二者的差异。

假设我们不使用 DDP 内置的重叠，而是手动实现分布式训练：每个进程在本地计算出全部梯度，然后在反向传播结束时对这些梯度执行一次 all-reduce。由于通信只在所有计算完成之后才发生，这将模拟出未经优化、无重叠的场景。我们可以借助 PyTorch 的分布式原语来模拟这一过程：禁用 DDP 的钩子，并在 loss.backward() 之后显式调用 dist.all_reduce。

在下面的代码中，我们在单个节点上于 gpu/rank 0 与 gpu/rank 1 启动两个进程，这里使用一个简单的模型。我们将执行一次前向和反向传播，然后在两个进程间手动对梯度求平均：

```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

# 一个含多层的简单模型，用于产生多个梯度
class MultiLayerNet(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_no_overlap(rank, world_size):
    dist.init_process_group("nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    # 每个 rank 各自合成自己的数据
    # （避免通过 spawn 发送大张量）
    batch_size = 256
    data = torch.randn(batch_size, 1024, device=rank)
    target = torch.randn(batch_size, 1, device=rank)

    model = MultiLayerNet(data.size(1)).to(rank)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # 前向 + 反向（手动，无重叠）
    output = model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()

    # 反向传播之后进行同步的梯度 all-reduce
    for p in model.parameters():
        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
        p.grad /= world_size

    optimizer.step()
    dist.destroy_process_group()

if __name__ == "__main__":
    import torch.multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    world_size = min(2, torch.cuda.device_count() or 1)
    if world_size > 1:
        mp.spawn(train_no_overlap, args=(world_size,), nprocs=world_size,
                 join=True)
    else:
        train_no_overlap(0, 1)
```

在这段代码中，每个进程独立地为 MultiLayerNet 计算梯度。在 loss.backward() 之后，我们显式地对每个参数的梯度执行一次 all-reduce 来求平均。这本质上就是 DDP 在内部所做的事情，但这里我们是在整个反向传播完成之后才等待并执行——而不是在反向传播过程中并发地进行 all-reduce。

如果我们对这次迭代计时，all-reduce 操作会直接叠加到迭代时间上，因为它没有与任何其他步骤重叠。举例来说，假设前向和反向计算合计耗时 10 ms，而梯度 all-reduce 耗时 12 ms。那么在这种方式下，总迭代时间大约为 22 ms。相比之下，一个完全重叠的实现或许能把总时间做到接近这两个值中的最大者，在我们的例子中即 12 ms。这之所以可能，是因为 all-reduce 通信几乎可以被完全隐藏在计算之下。

在实践中，如果我们用 Nsight Systems 之类的工具剖析这个无重叠的例子，会看到 fc1、fc2 和 fc3 的所有反向计算内核先行执行，只有在它们全部完成之后，我们才会看到针对每个梯度的 NCCL all-reduce 内核。这里存在一个明显的分界：先计算，后通信。在通信阶段，除了 NCCL 的工作之外 GPU 处于空闲状态，因为没有更多的计算在进行。

现在，让我们改用 PyTorch 的 DDP 以带重叠的方式执行相同的操作。DDP 会挂钩到反向传播中，将梯度归约与反向计算重叠。代码是类似的，只是我们简单地用 DistributedDataParallel 包裹模型，并让它来处理同步：

```
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

class MultiLayerNet(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fc1 = nn.Linear(size, size)
        self.fc2 = nn.Linear(size, size)
        self.fc3 = nn.Linear(size, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_ddp(rank, world_size):
    rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method="env://",
                            world_size=world_size, rank=rank)
    torch.cuda.set_device(rank)

    model = MultiLayerNet(1024).to(rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # 每个 rank 生成自己的数据
    batch_size = 256
    data = torch.randn(batch_size, 1024, device=rank)
    target = torch.randn(batch_size, 1, device=rank)

    output = ddp_model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()  # DDP 在后台流上重叠梯度 all-reduce
    optimizer.step()
    dist.destroy_process_group()

def main():
    world_size = min(2, torch.cuda.device_count() or 1)
    mp.set_start_method("spawn", force=True)
    if world_size > 1:
        mp.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
    else:
        print("Only one GPU present; running DDP demo with world_size=1")
        train_ddp(0, 1)

if __name__ == "__main__":
    main()
```

采用 DDP 的重叠方式后，代码更为简洁，因为我们依靠 DistributedDataParallel 来处理梯度同步，而不必自己编写。当 loss.backward() 被调用时，DDP 内部的 reducer 会将梯度切分成桶，并在每个桶就绪时立即在一个单独的 CUDA 流上发起 NCCL all-reduce 操作。例如，它可能在 fc3.weight 和 fc3.bias 的梯度刚被计算出来时就立即对它们做 all-reduce，因为这些梯度来自最后一层、在反向传播中最先算出；而来自较早层的 fc1 和 fc2 梯度，则会在我们到达反向传播末尾时早已完成 all-reduce。

如果模型非常小、所有梯度都能装进一个桶，DDP 或许只会在末尾做一次 all-reduce，那样重叠就不多。但对于更大的模型和更大的批量，会存在多个桶以及显著的重叠——从而展现出这项技术更为可观的性能收益。

对 DDP 情形进行剖析，会看到 NCCL all-reduce 内核与反向计算内核相互交错。因此，我们看到的不再是清晰的两阶段划分，而是时间线上呈锯齿状的模式：先有一些计算，接着有一些 NCCL 通信，再是计算，如此往复。

良好重叠的关键标志是：总迭代时间低于计算时间与通信时间之和，并且 GPU 很少因等待通信而空闲，因为 all-reduce 就发生在计算过程中，如表 4-1 所示。

_表 4-1. 重叠带来的收益_

| 指标                                                       | 无重叠（手动同步）                         | 重叠（DDP）                      | 说明                                         |
| ---------------------------------------------------------- | ------------------------------------------ | -------------------------------- | -------------------------------------------- |
| 反向传播 + 通信总时间                                      | 100%（基线）                               | 约为基线的 ~70%                  | 例如，得益于重叠，每次迭代快约 30%（示意值） |
| 通信开始时间                                               | 反向传播完成之后                           | 反向传播期间                     | 在 DDP 中，通信在反向传播中途即开始          |
| 通信阶段 GPU 空闲                                          | 是——反向传播后，GPU 在 all-reduce 期间等待 | 极少——通信在其他层仍在计算时进行 | DDP 隐藏了大部分延迟                         |
| SM（streaming multiprocessor，流式多处理器，即 GPU）利用率 | 较低（通信期间部分周期 SM 空闲）           | 较高（持续活动）                 | 重叠使 GPU 更持续地保持忙碌                  |
| 实现的重叠（通信被计算覆盖的百分比）                       | 0%（串行执行）                             | ~50%（或更多）                   | 粗略估计：更大的模型或批量可实现更多重叠     |

> 注：所有指标表中的数值均为示意性质，用于说明概念。若需了解不同 GPU 架构上的实际基准测试结果，请参见 GitHub 仓库。

在本例中，将通信与计算重叠为我们的示例工作负载带来了约 30% 的迭代时间改善。在更大的训练作业中，收益会更加可观，因为更大的模型带来了更多潜在的通信瓶颈。PyTorch 的 DistributedDataParallel 默认使用 25 MiB 的桶。如此一来，它便会在每个桶就绪时立即为其发起一次 all-reduce。调节 bucket_cap_mb 有助于为你的特定模型拓扑增加重叠，但更大的桶会增加最后一个桶的延迟。

要点在于：一个调优良好的 DDP 应当能把大部分梯度通信与计算重叠。通常唯一无法重叠的部分就是尾部，也就是最后一个梯度桶——如果它恰好在最后一次计算之后才完成的话。目前仍有研究工作在尝试将优化器步骤也与通信重叠，或使用张量切分之类的技术来实现进一步的重叠；而 PyTorch DDP 的默认重叠策略常被称为无等待反向传播（wait-free backpropagation，WFBP），它将梯度分桶，并在每个桶就绪时立即发起归约。

值得注意的是，某些编码模式会无意中消除重叠。举例来说，如果你执行了任何在反向传播与下一次迭代之间强制同步的操作，就会导致计算停滞，直到所有通信完成。

> 在你确定所有异步 GPU 工作都已结束之前，应避免那些无意中将张量从 GPU 移动到 CPU 的操作（例如对张量调用 .item()）。否则，你会强制一次同步、使计算停滞，并拖慢你的训练或推理工作负载。这种情况通常发生在为调试而添加 print() 或 log() 语句时。它们对性能可能是灾难性的。

如有可能，你应把此类操作移到一个单独的流上。此外，手动调用 torch.cuda.synchronize() 应尽量减少，仅在需要精确基准测试时——或为了正确性而必需时——才使用。否则，它们会使 GPU 工作串行化，对性能造成负面影响。DDP 的设计以及 PyTorch 的各项操作本身已是异步的，并能正确处理依赖关系。在用户代码中很少需要显式同步。

总而言之，重叠计算与通信是分布式性能工程中最有效的技术之一。若能恰当运用，它可以通过把通信延迟隐藏在有用的工作之下，显著缩短训练时间。

在接下来的几节中，我们将探讨使这一切成为可能的软件与硬件基础设施，以及如何确保在现代系统上获得最大的重叠。接下来，我们将概览 NVIDIA 的 Magnum IO 栈，其中包含了使计算与通信得以重叠的各项技术。

## NVIDIA Magnum IO 优化栈

Magnum IO 是 NVIDIA 全局性的 I/O 加速平台，它汇集了一系列技术，用于加速数据在 GPU、CPU、存储与网络接口之间的搬运、访问与管理。Magnum IO 架构有四个关键组成部分，分别涵盖存储、网络、在网计算与 I/O 管理，如图 4-2 所示。

![图 4-2. NVIDIA Magnum IO 加速平台的四个组成部分](AI%20Systems%20Performance%20Engineering-ch4_images/figure-4-2.png)

以下是对图 4-2 中四个组成部分的说明：

**存储 I/O**

这一部分由诸如 NVIDIA GPUDirect Storage（GDS）和 BlueField SNAP 之类的技术实现。它们让 GPU 能够直接访问包括 NVMe SSD 在内的存储，而无须经由主机 CPU 内存进行不必要的拷贝。我们将在第 5 章更深入地探讨 GDS。

**网络 I/O**

这一部分包括 GPUDirect RDMA、NCCL、NVSHMEM、UCX 与 HPC-X（MPI/SHMEM 软件套件）等技术，用于在跨节点的 GPU 之间实现直接、高速的数据传输，并在节点间（internode）通信中绕过 CPU。

**在网计算**

SHARP 在 Quantum 系列 InfiniBand 交换机内部执行在网归约。归约运算在交换机芯片中完成。BlueField DPU 卸载网络处理，并可承载诸如子网管理器（Subnet Manager）和 SHARP 聚合管理器（Aggregation Manager）之类的控制服务。当启用 NCCL RDMA SHARP 插件、且织物具备 SHARP 固件和一个活跃的聚合管理器时，符合条件的集合操作便可被卸载到 IB 交换机上，从而减少主机与 GPU 的开销。稍后我们会更详细地介绍 NVIDIA SHARP。

> 基于以太网的 GPU 集群依赖诸如 RoCEv2 之类的技术来实现 RDMA，但通常缺乏像 SHARP 这样的特性。这也是许多超大规模 AI 系统选用 InfiniBand 或类似高性能互连而非以太网的原因之一。SHARP 能带来显著的性能提升，在可用时应当加以利用。

**I/O 管理**

诸如 NVIDIA NetQ 和 Unified Fabric Manager（UFM）之类的工具归入此类，它们为数据中心的 I/O 织物提供实时遥测、诊断与生命周期管理。

这些组件通过灵活的 API 和 SDK 协同工作，隐藏了大量底层复杂性。Magnum IO 的目标是为大规模 AI 与数据分析工作负载最大化端到端吞吐量。Magnum IO 也在随新硬件不断演进。

Magnum IO 现已集成对 NVLink Switch 网络域的支持，从而在织物规模上实现机架内的 GPU 通信。它还利用了 InfiniBand（Quantum-2 与 Quantum-X800 系列）和以太网（Spectrum-X）的进展，以进一步降低通信开销。

在本章中，我们将深入探讨其中的若干组件，例如用于网络传输的 RDMA、用于集合操作的 NCCL、用于推理数据搬运的 NIXL，以及用于高效数据加载的 GDS。此外，我们还会看到如何对它们各自进行剖析与调优。让我们直接开始吧！

## 使用 RDMA 实现高速、低开销的数据传输

RDMA 是一项为低延迟、高吞吐数据传输而优化的技术。RDMA 的工作方式是允许设备之间进行直接的内存到内存通信，而无须让 CPU 承担不必要的数据拷贝操作。简而言之，RDMA 绕过了传统内核网络栈的大部分环节，允许网卡（network interface card，NIC）直接读写应用程序内存。这避免了每个数据包都需要 CPU 参与，并减少了上下文切换与缓冲区拷贝。在可用且经过验证的地方，应优先选择 RDMA 路径。并且要始终通过日志和微基准测试来确认（并持续重新确认）RDMA 数据路径确实处于活跃状态。

在诸如 Docker 和 Kubernetes 之类的容器环境中，要确保容器能够直接访问主机的 InfiniBand 设备（例如 /dev/infiniband）。否则，NCCL 可能会悄无声息地回退到 TCP 套接字，而不使用 GPUDirect RDMA——而且没有任何明显的错误来提示这种性能退化。其结果是吞吐量从数十 GB/s 骤降到仅有几 Gb/s，却没有任何明显的错误信息。

一个相关的容器陷阱出现在容器的 GID 分配与主机不匹配时，例如某些 “rdma-shared” 的 Docker 镜像便是如此。这会导致无法进行 GPUDirect 注册，转而使用由 CPU 驱动的 RDMA 拷贝，而非真正基于 GPU 的 RDMA。

> 务必核实它确实是真正的 GPUDirect RDMA。用 lsmod | grep nvidia_peermem 确认内核模块已加载，并检查 dmesg 中的初始化信息。作为端到端的检查，可用 NCCL_DEBUG=INFO 运行 NCCL 以确认 NET/IB 路径，并使用带 --use_cuda 的 RDMA perftests 来验证 GPU 到 GPU 的传输。核实有助于防止隐蔽的性能退化。

在剖析器中，被削减的吞吐量会表现为吞吐量下降一个数量级。但你必须意识到这种可能性，并持续监控你的系统，以防这类微妙的回退。

NVIDIA 面向 GPU 的 RDMA 实现称为 GPUDirect RDMA。GPUDirect RDMA 让具备 RDMA 能力的网卡——例如 InfiniBand 和 RDMA over Converged Ethernet（RoCE）——能够跨两台服务器对 GPU 的设备内存执行直接内存访问（direct memory access，DMA），完全绕过主机 CPU 和系统 RAM。图 4-3 展示了一次使用 RoCE 的数据传输。

![图 4-3. 使用 RoCE 的 GPU 到 GPU 直接数据传输](AI%20Systems%20Performance%20Engineering-ch4_images/figure-4-3.png)

通过把 GPU 缓冲区在网卡上注册，GPUDirect RDMA 使得远程 GPU 之间能够进行单边（one-sided）的 RDMA 读写。这在多节点训练中同时最小化了延迟与 CPU 开销。

RDMA 天生受到 InfiniBand 的支持，也可通过 RoCE 在某些高速以太网网络上得到支持。使用 RoCE 时，只要网络设备支持 RDMA 并为此进行了恰当配置，你就能在以太网上获得类似 RDMA 的零拷贝传输。在 RoCE 上使用 RDMA 通常要求系统经过恰当配置并安装必要的驱动，包括用于 InfiniBand/RoCE 的 NVIDIA OFED。

使用 RDMA 与使用标准 TCP/IP 网络之间的性能差异可能非常巨大。例如，现代 InfiniBand 链路对一条小消息的延迟可能只有几微秒，而以太网上的标准 TCP 可能会带来高出 5–10× 的延迟。

对于受网络带宽限制的大传输，InfiniBand 上的 RDMA 能够维持数百 Gbps 量级的极高吞吐量。相比之下，典型的 TCP/IP 网络可能受制于内核开销与网卡速率——除非使用具备 RDMA 能力的 200–400 Gbps 以太网，否则往往只有 100 Gbps 或更低。TCP/IP 网络还会带来更多开销。

在分布式深度学习中，大消息的吞吐量往往比极小消息的延迟更为重要，因为梯度本身就很大。高带宽与低延迟二者兼备，才能让协议保持高效——并让 GPU 保持忙碌。

如果你只有以太网可用，就应当采用尽可能高带宽、低延迟的配置。例如，带 RDMA（RoCE）的 200+ Gbps 以太网，处理 all-reduce 流量的表现将远优于基础的 10–25 Gbps TCP 网络。至少，要确保你使用了巨型帧（jumbo frame），例如 MTU 9000。为你的集群网络启用这一配置后，数据传输将发送更少的大数据包，而非许多小数据包。这减少了 CPU 开销并提升了效率，其原理类似于更大的磁盘块大小如何提升顺序磁盘吞吐量。

同样，出于类似原因，调优 TCP 栈也很重要。你应当核实诸如 net.core.rmem_max/wmem_max 之类的 Linux sysctl 参数，以及自动调节范围 net.ipv4.tcp_rmem/tcp_wmem，都设置得足够高，以充分利用高带宽链路。

当然，使用现代的 TCP 拥塞控制算法来提升高延迟链路上的吞吐量也很重要。在一个精心设计、无外部互联网流量的专用集群网络上，默认的 CUBIC 拥塞控制通常表现尚可，因为它本就是为避免拥塞而设计的。

> 对于任何高延迟带宽积的链路，可考虑使用现代 TCP 拥塞算法，例如 Bottleneck Bandwidth and Round-trip propagation time（BBR），并调整缓冲区大小以确保充分利用。务必验证默认设置没有限制吞吐量。使用诸如 sysctl net.ipv4.tcp_congestion_control 之类的工具来查看和调节该设置。

在云或混合环境中，要留意你是否真正拥有一条受控的高速连接。例如，如果你使用带 Elastic Fabric Adapter（EFA）的 AWS EC2 实例，那么在部署于同一个 “placement group” 的实例之间，你能获得类似 InfiniBand 级别的 RDMA。但如果你试图运行一个横跨本地数据中心与云、又没有直连的多节点训练或推理作业，你的流量很可能会穿越公共互联网。这会引入不可预测的延迟与拥塞。

> 务必确保你的多节点部署位于一个经过恰当配置、高性能、低拥塞的网络上。要直接与你的云服务商协作，弄清网络架构中的每一跳。

即便在使用 RDMA 时，CPU 也并未完全置身事外。主机仍需建立 RDMA 传输并处理通信完成事件。因此，恰当的 CPU 亲和性（affinity）很重要。记得把网络中断处理——或轮询线程——固定到与网卡处于同一 NUMA 节点的 CPU 核心上，理想情况下也应与 GPU 处于同一节点。例如，如果一个 InfiniBand 主机通道适配器位于 NUMA 节点 0，就把它的中断 CPU 亲和性核心绑定到节点 0。这减少了控制操作的跨 NUMA 流量与延迟。

### 多节点连通性调优

对于使用 GPU 的分布式多节点训练，确保网络不成为瓶颈至关重要。这既涉及使用前面所述的恰当通信与网络技术，也涉及对这些技术进行恰当的配置。以下是一些值得采纳的建议和应当避免的陷阱：

**理解拓扑**

使用 nvidia-smi topo -m 可以获得基本的 GPU 互连视图；但对于基于 NVSwitch 和 NVLink 的系统，建议同时使用 nvidia-smi nvlink 或 Nsight Systems 来理解多跳交换机织物的连通性。

**尽可能利用 NVLink Switch 域**
