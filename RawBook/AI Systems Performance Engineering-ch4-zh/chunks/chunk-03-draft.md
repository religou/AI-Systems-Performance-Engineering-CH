PAT 是 NCCL 将 ring 与 tree 算法流水线化后的混合算法。一旦张量的某个分段在其 GPU 树上完成归约，下一个分段就会以错峰、轮转（round-robin）的方式同时开始自己的树形归约。这种连续归约阶段的重叠让 PAT 能持续把链路打满，达到接近纯 ring all-reduce 的带宽；同时，它把每个分段的传输启动延迟限定在 O(log N)，与 tree 算法相近。实践中，PAT 会把一条大消息切分成多个块（chunk），先对第 1 块启动基于 tree 的 reduce-scatter，随后立即对第 2 块发起相同操作，以此类推。这种交错方式保证始终有传输在途，从而为大数据传输带来接近 ring 级别的吞吐量，同时为较小分段保留 tree 级别的延迟优势。这是一种“两全其美”的方案。

与选择任何通信算法一样，NCCL 算法的选择通常取决于消息大小与拓扑。小消息（数十 MB 量级）更适合 tree 算法，因为步骤更少；大消息则更适合 ring 算法，因为它们能提供更好的带宽利用率。

默认情况下，NCCL 会针对给定的集合操作、消息大小与拓扑自动选择最优算法。NCCL 支持对称内存优化和低延迟内核，可在 NVLink 互连的系统上降低小消息和中等消息的 all-reduce 延迟。在某些情况下，实测这种降低在 NVLink 互连系统上对小消息和中等消息可达约 7.6×。这些改进，连同 PAT 等算法，进一步降低了像 NVL72 这类系统上的通信开销。

NCCL 的通信策略持续朝拓扑感知方向演进。例如，NCCL 可以利用 NVSwitch 的硬件多播（multicast）在一个 NVLink 域内实现单跳广播。这非常适合需要一次性把相同数据（例如更新后的模型权重）发送给所有 GPU 的场景。

NCCL 还会利用最新的硬件进展。例如，它可以在 InfiniBand 上使用 SHARP，在基于 NVSwitch 的互连结构上使用 NVLink SHARP（NVLS）。当配置了 NCCL-SHARP 插件后，这会在受支持的系统上加速 all-gather 和 reduce-scatter 等集合操作。

此外，如前所述，NCCL 实现了 PAT 算法，它把类 ring 的吞吐量与类 tree 的延迟结合在一起。该算法把大消息切分成块，检视物理 GPU 与交换机的拓扑布局，并利用这些信息在不同的 CUDA 流上恰当地交错 reduce-scatter 与 all-gather 阶段。当硬件与网络支持时，这能充分利用网络拓扑来兼得两者之长：类 tree 的延迟与类 ring 的带宽。

默认情况下，NCCL 的通信器初始化会自动检视消息大小、互连拓扑与 GPU 代次，从而为每种集合操作和拓扑自动挑选最快的算法与协议组合。不过，如果对工作负载进行剖析后发现通信并非最优——例如跨节点延迟意外偏高——你可以通过设置 `NCCL_ALGO` 环境变量（例如 `NCCL_ALGO=NVLSTree,PAT`）逐案覆盖通信算法。这会强制 NCCL 在该通信器上使用特定算法。如果在代码中设置该变量，务必在调用 `ncclCommInitRank()` 之前完成设置。

### 数据并行分布式策略

在实践中，对拥有数十亿乃至数万亿参数模型的大规模训练和推理，需要组合多种并行策略，包括数据并行、张量模型并行（tensor model parallel）、流水线并行（pipeline parallel）、专家并行（expert parallel）、上下文并行（context parallel）等。这些策略是让训练集群线性扩展、并避免因过高开销而浪费 GPU 资源所必需的。

关键在于在每一层级都让通信与计算重叠。这包括用 NCCL 做 all-reduce、用 NIXL 做一对一传输。借助这些机制，你可以高效地扩展到成千上万乃至数百万块 GPU。

> 在超大规模场景下，梯度累积（gradient accumulation）和激活检查点（activation checkpointing）等其他技术同样至关重要，它们能在不牺牲吞吐量的前提下控制内存占用。

当在单节点上扩展到多块 GPU 时，PyTorch 在框架层面同时提供了数据并行（拆分数据）和模型并行（拆分模型）两种方式。我们会在后续章节更详细地讨论它们，但现在先从系统性能的角度比较两种最基础的数据并行策略：`nn.DataParallel`（DP）和 `torch.distributed.DistributedDataParallel`（DDP）。理解它们的差异很重要，因为选错会严重影响性能：

**数据并行（data parallelism，DP）**

DP 是一套易用的 API，由单个进程（即单个 Python 线程）控制多块 GPU。该模块会自动把每个输入批次拆分到可用的 GPU 上，然后在每个分片上执行前向传播，把输出收集回主 GPU，并计算聚合后的损失。最后，在反向传播阶段，它把梯度收集回主 GPU，取平均，再广播回其他 GPU。

在 `DataParallel` 中，整个训练循环都是单进程的。由于无需启动多个进程，这使它更易于集成。尽管 DP 使用了多线程，但在不同设备上发起操作时它仍受限于 Python 的 GIL（Global Interpreter Lock，全局解释器锁）。因此，DP 在超过 2–4 块 GPU 后扩展性很差，因为单个 Python 线程成了瓶颈，GPU 利用率随之下降。此外，DP 中的梯度收集步骤是同步的，且不与计算重叠。这意味着它的行为类似于我们前面描述的“无重叠”场景。

**完全分片数据并行（FSDP）**

FSDP 通过在各 GPU 之间对激活、梯度和参数进行分片，避免保存完整的模型副本，从而大幅降低内存开销。对于超大规模模型，FSDP 常与张量并行、流水线并行等其他并行策略组合使用。

> 我们会在第 13 章讨论 FSDP——以及专家并行等其他超大规模训练技术。本节将聚焦 DP 和 DDP，为后续章节讨论的更高复杂度打下基础。

**分布式数据并行（Distributed Data Parallel，DDP）**

DDP 为每块 GPU 设备使用一个进程，并依赖 NCCL 来通信梯度。与大多数简单的数据并行策略一样（FSDP 是例外），每个进程都持有自己的模型副本。

在反向传播阶段，梯度会直接在各 GPU 之间交换，即执行 all-reduce。这种 all-reduce 通信通常会与反向计算重叠，正如我们前面讨论的那样，这是理想的。

DDP 通过使用独立进程完全规避了 GIL 问题，并由 NCCL 高效的 C++ 内核负责通信。其结果是，在多 GPU 训练中 DDP 几乎总是优于 DP。事实上，PyTorch 开发者建议在任何正式的多 GPU 工作中使用 `DistributedDataParallel` 而非 `DataParallel`，因为 DP 的 Python 线程受制于令人头疼的 GIL，往往会成为瓶颈。

让我们回顾前面的例子，但这次是在单节点上比较 DP 和 DDP 的语境下。我们会使用一个简单模型，分别用 DP 和 DDP 测量单个训练步。

在这个场景中，我们用 `DataParallel` 包装一个模型，让 PyTorch 拆分每个输入批次并使用两块 GPU。我们会对单次训练迭代计时：

```
# before_dataparallel.py

import torch
import torch.nn as nn
import torch.optim as optim

# Dummy model and dataset
class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

# Setup model and data
input_size = 1024
hidden_size = 256

model = SimpleNet(input_size, hidden_size)

model.cuda()  # move model to GPU 0, it will also replicate to GPU 1

model = nn.DataParallel(model)    # utilize 2 GPUs (0 and 1 by default)

optimizer = optim.SGD(model.parameters(), lr=0.01)
data = torch.randn(512, input_size).cuda()   # batch of 512 on GPU0
target = torch.randn(512, 1).cuda()          # target on GPU0

# Run a single training step
output = model(data)                      # forward (DP splits data internally)
loss = nn.functional.mse_loss(output, target)
loss.backward()                           # backward (DP gathers grads to GPU0)
optimizer.step()
```

这里，`nn.DataParallel` 的前向传播把形状为 [512,1024] 的输入张量（矩阵）拆分成两个大小为 [256,1024] 的张量，一个发往 GPU 0，一个发往 GPU 1。在前面的代码中，通过 `model.cuda()`，GPU 0 和 GPU 1 都持有同一模型的副本。

> 严格来说，模型最初只在 GPU 0 上，但当 `forward` 第一次被调用时，DP 会在执行计算前自动把模型复制到 GPU 1——以及任何其他参与的 GPU。

随后，`DataParallel` 会用每块 GPU 一个线程来入队各副本的计算，在 GPU 0 和 GPU 1 上并行发起前向传播。然而，由于这些线程共享同一个 Python 进程并争夺我们的宿敌——GIL，内核发起（kernel-launch）调用在 Python 中是顺序执行的。所幸，已入队的 GPU 工作会在设备上并发运行。

前向传播完成后，`DataParallel` 会把每块 GPU 的输出收集到主设备（GPU 0）上计算损失。在反向传播阶段，它同样把所有副本的梯度收集并求和到 GPU 0 上，再把 GPU 0 上聚合好的梯度广播回其余 GPU（此例中为 GPU 1），然后执行优化器步骤。

在这个 `DataParallel` 的例子中，有几点值得强调。GPU 0 承担了收集并求和所有梯度的额外负担。而且，每次梯度归约（GPU 1 → GPU 0 再返回）都是同步执行的——会阻塞反向传播中的后续工作。这种设计带来两大代价。第一，Python 控制线程必须在每块设备上串行地发起内核，这增加了 CPU 侧的开销。第二，由于梯度聚合没有与进行中的计算重叠，GPU 0 可能成为性能瓶颈。下面我们来比较 `DistributedDataParallel` 如何解决这些问题。

切换到 `DistributedDataParallel` 后，我们会（例如）用 `torch.multiprocessing.spawn` 为每块 GPU 启动一个进程。每个进程持有自己完整的模型副本，处理批次的一个独立切片。在下一个例子中，批大小为 256。运行两个进程时，有效总批大小仍为 512，与 `DataParallel` 的设置一致，以便公平比较：

```
# after_ddp.py

import os, time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, 1)
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

def train_ddp(rank, world_size):
    rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl",
       init_method="env://",
       world_size=world_size, rank=rank)
    model = SimpleNet(input_size=1024,
       hidden_size=256)

    model.cuda(rank)

    ddp_model =
       nn.parallel.DistributedDataParallel(model,
         device_ids=[rank])
    optimizer = optim.SGD(ddp_model.parameters(),
       lr=0.01)
    # Each process gets its own portion of data
    batch_size = 256
    data = torch.randn(batch_size, 1024).cuda(rank)
    target = torch.randn(batch_size, 1).cuda(rank)

    # Run one training iteration
    output = ddp_model(data)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
```

你可以使用 torchrun 启动器（或你集群的 MPI/SLURM/Kubernetes 集成）来运行它，并设置 `MASTER_ADDR` 和 `MASTER_PORT` 环境变量。运行下面的脚本，你会看到类似如下的输出（注意：这些结果与工作负载和硬件相关，不一定与你的结果一致）：

```
# Environment (common gotchas)
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=ib0      # use your HCA (e.g., ib0, ib1)
# Optional: for multi-rail IB, set NIC ordering
# so ranks use distinct rails.

# Local bring-up, 2 GPUs
torchrun --standalone --nproc-per-node=2 after_ddp.py

# SLURM (example)
srun --ntasks=$WORLD_SIZE --gpus-per-task=1 --nodes=$NNODES \
     --cpus-per-task=8 python after_ddp.py

### Output: ###

DDP step took 30.00 ms
```

虽然你的具体数字会有所不同，但 DDP 迭代明显快于 DP。在这个例子中，尽管两者处理的总数据量相同，DDP 仍比 DP 快 33%。

这一提升源于多个因素。首先，每个进程处理一半的批次且不存在 Python GIL 争用，因此它们真正做到了并行运行。其次，梯度通过 NCCL 进行 all-reduce，从而让通信与反向计算重叠。此外，不再需要把梯度额外拷贝到单块聚合 GPU（例如 GPU 0）再拷回。而且，每块 GPU 的梯度都是就地直接交换并求平均的。最后，通信工作分摊到所有参与这次 all-reduce 集合操作的 GPU 上，而不是压在负责聚合数据的单块 GPU 身上——此外还有诸多好处。

在我们的例子中，DDP 用了 30 ms 完成，而 DP 用了 45 ms。在更大的模型中，这一差距会更大——尤其当你扩展到数千块 GPU 时。DP 在扩展到 2–4 块 GPU 以上时甚至可能出现超线性劣化，因为主线程会不堪重负。而 DDP 往往能很好地随 GPU 数量扩展，主要受限于通信带宽，而非 CPU 或单 GPU 开销。

总而言之，在多 GPU 训练中应始终优先选择 `DistributedDataParallel` 而非 `DataParallel`——就这么简单。PyTorch 团队明确推荐这样做，正是因为它在性能和可扩展性上的优势。DP 唯一可以接受的场景，是在 2 块 GPU 上做快速原型验证时——此时更看重易用性，而性能是次要的。

如果你在意训练吞吐量，从 DataParallel 切换到 `DistributedDataParallel` 只需几行代码。即使在单节点多 GPU 上你也可以这样做。通过为每块 GPU 派生一个进程（例如 `torch.multiprocessing.spawn`），每个进程都持有自己的模型副本和数据分区。

在底层，DDP 使用异步的 NCCL all-reduce 来让梯度通信与计算重叠，并完全避免 Python GIL 争用。这带来了好得多的 GPU 利用率和更快的端到端迭代时间。正因如此，在多 GPU 系统上注重性能的工程师几乎总是青睐 `DistributedDataParallel` 而非 `DataParallel`。

### NCCL 通信器生命周期与环境变量陷阱

尽管 NCCL 抽象掉了大多数底层细节，但我们在代码中如何使用 NCCL 仍会影响性能。此外，NCCL 有许多控制其行为的环境变量。错误配置这些变量会降低性能，甚至导致挂起（hang）。

在本节中，我们介绍与 NCCL 通信器和环境设置相关的常见陷阱，并展示如何诊断和避免这些问题。

陷阱 #1：过于频繁地创建 NCCL 通信器

一个 NCCL 通信器（communicator）代表一组能够进行集合通信的 GPU（即 rank）。用 C++ 的 `ncclCommInitRank` 或 PyTorch 的 `torch.distributed.init_process_group` 创建通信器是一项开销很大的操作。初始化过程要求所有 rank 相互交换信息，包括唯一 ID、网络地址等，同时还要建立 ring/tree 并分配缓冲区。

如果你的代码反复初始化 NCCL 通信器，每次都会付出高昂的代价。设想一个有 32 块 GPU 的系统。如果你创建 32 个独立的 NCCL 通信器（每个 rank 一个），这可能需要 2–3 分钟，而不是 2–3 秒（甚至更快）。通信器初始化随 rank 数量的扩展往往差于线性，因为它常常需要在众多 GPU 之间进行全互联（all-to-all）握手与协调。

在 PyTorch 的 DDP 中，这一点已为你处理好。你只需在程序开始时调用一次 `init_process_group`，DDP 就会为所有进程创建一个通信器。随后每次迭代的所有集合操作都会复用它。

为说明每次迭代都创建 NCCL 通信器的代价，这里给出一个 PyTorch 例子，其中有人在训练的每次迭代都幼稚地初始化并销毁一个 NCCL 进程组：

```
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, world_size):
    rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.cuda.device(rank)
    for i in range(5):  # simulate 5 iterations
        // This naive approach re-initializes NCCL each
        // iteration. THIS IS EXTREMELY SLOW AND NOT RECOMMENDED!!!
        dist.init_process_group("nccl", init_method="env://",
                                 world_size=world_size, rank=rank)
        # do a tiny all-reduce to simulate some work
        tensor = torch.ones(1).cuda(rank)
        dist.all_reduce(tensor)
        if rank == 0:
            print(f"Iter {i} done")
        dist.destroy_process_group()
```

如果你运行这段代码，会发现它极其缓慢——尽管 all-reduce 本身微不足道——因为大部分时间都花在了 `init_process_group` 和 `destroy_process_group` 上。在 rank 更多的真实场景中，这一代价会成倍放大。

由于 `init_process_group` 调用被设计为在启动时只调用一次，你应避免任何在每次迭代都重新初始化它的设计。修复方法是在循环之外初始化一次，如下所示：

```
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os

def run(rank, world_size):
    # Pin GPU
    rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.cuda.device(rank)

    # Initialize NCCL communicator once
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://127.0.0.1:45678",
        world_size=world_size,
        rank=rank
    )

    # Simulate 5 training iterations
    for i in range(5):
        tensor = torch.ones(1, device=rank)
        dist.all_reduce(tensor)

    # Cleanup once at the end
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size)
```

把通信器的建立与销毁移出循环后，你就消除了每次迭代产生的 48 ms 初始化开销。这将总迭代时间降低了 98% 以上，如表 4-3 所示。

_表 4-3. 避免重复的通信器 init/destroy 对每次迭代时间的影响_

| 指标                          | 之前（每次迭代） | 之后（每次迭代） |
| ----------------------------- | ---------------- | ---------------- |
| init_process_group + destroy  | 48.0 ms          | 0 ms             |
| dist.all_reduce（1 元素张量） | 0.5 ms           | 0.5 ms           |
| 总迭代时间                    | 48.5 ms          | 0.5 ms           |

微小的 all-reduce 本身仍为 0.5 ms，但此前它完全被频繁的初始化开销所掩盖。在 rank 更多的真实多节点场景中，节省会成倍放大。只初始化一次是一项明确的性能最佳实践。

陷阱 #2：不要在每次迭代都创建和销毁 NCCL 通信器

由于 NCCL 通信器初始化如此昂贵，在为模型并行和流水线并行定义进程子组时，要小心不要意外地创建新的 NCCL 通信器。相反，应在开始时用 PyTorch 的 `torch.distributed.new_group()` 一次性创建子通信器并复用它们。绝不要在每次迭代都创建和销毁通信器。

如果你确实需要创建多个通信器——例如存在动态运行时成员关系的场景或分阶段初始化——NCCL 提供了一套 C++ API，可用 `ncclGroupStart()`、`ncclCommInitRank(...)` 和 `ncclGroupEnd()` 一起初始化多个通信器。这会大幅降低开销。

> 截至本文写作时，PyTorch 尚不支持在运行时进行完全动态的成员变更而不彻底拆除通信器。所有 rank 必须以锁步（lockstep）方式调用创建和销毁操作，以防止挂起。

陷阱 #3：避免用环境变量过度调优或禁用 NCCL 特性

NCCL 有许多环境变量，如 `NCCL_BUFFSIZE`、`NCCL_NSOCKS_PERTHREAD`、`NCCL_P2P_LEVEL`、`NCCL_SHM_DISABLE` 等。除非有特定理由，通常最好保持它们的默认值。更好的做法是，把它们显式设置为当前的默认值，以做到明确、而不依赖默认！务必查阅版本发布说明并据此调整取值。

> 虽然可以调大 `NCCL_BUFFSIZE` 来提升大型 all-reduce 操作的带宽，但必须谨慎设定其大小。设置过高会造成 GPU 内存压力，或迫使较小的模型逐出其工作集。可以从 4 MB 起步并逐步增大。增大该值时要监控 GPU 内存使用情况。

一个常见错误是在调试时禁用某些特性，却忘了在生产环境中重新启用它们。例如，在生产环境中绝不要让点对点（point-to-point，P2P）或共享内存传输处于关闭状态。

用 `NCCL_P2P_DISABLE=1` 禁用直接的 P2P GPU 拷贝在排障时或许有助于隔离问题，但如果一直启用，会大幅降低节点内（intranode）性能。这是因为它会强制所有节点内流量经过 CPU 主机暂存的中间缓冲区，而不是走 GPU 直连的 NVLink 链路。这增加了额外的跳数和 CPU 工作，可能把延迟从几微秒抬高到数十微秒，并把带宽从数百 GB/s 砍到数十 GB/s。

> 仅在诊断问题时使用 `NCCL_P2P_DISABLE=1`。调试完成后，记得用 `NCCL_P2P_DISABLE=0` 重新启用 P2P（或干脆移除该环境变量）。

同样，让共享内存交换处于禁用状态（`NCCL_SHM_DISABLE=1`）会迫使 NCCL 在节点内通信时不使用共享内存。这会退回到经由网络或主机中转的拷贝，带来额外的内核驱动开销和上下文切换，从而进一步增加延迟并抑制吞吐量。

> 只在调试期间短暂改动性能关键的环境变量。别忘了在恢复正常运行之前把它们设置回生产配置。

另一个变量是 `NCCL_DEBUG`。设置 `NCCL_DEBUG=INFO` 或 `DEBUG` 有助于记录 NCCL 操作日志，因为日志可能提示某些问题，例如回退到以太网。但额外的日志记录确实会带来开销。不要在生产环境以 `DEBUG` 级别运行；在需要时才用。然而，出于性能考虑，你可能想把该设置降到 `WARN`（默认）甚至只保留 `VERSION`。后者会静默除 NCCL 版本外的一切，但当真正需要排障时——而这一天必将到来！——会让调试变得困难。为调优性能，一些有用的变量包括以下这些：

`NCCL_NSOCKS_PERTHREAD` 和 `NCCL_SOCKET_NTHREADS`。如前所述，如果你有多块网卡或极高的网络带宽，增大它们可能有帮助。假设你有两块网卡，你或许会设置 `NCCL_NSOCKS_PERTHREAD=2`，让每个线程处理两个套接字，总共允许四条连接。默认值因平台和构建而异，但关键在于：按 NVIDIA 的指导，`NCCL_NSOCKS_PERTHREAD` 与 `NCCL_SOCKET_NTHREADS` 的乘积不得超过 64。

> 这个 64 的上限是 NCCL 内建的、每个通信器所允许的 TCP 套接字连接总数的最大值。它被定义为 `NCCL_SOCKET_NTHREADS` \* `NCCL_NSOCKS_PERTHREAD` 的乘积，旨在限制 CPU 和网络资源的使用——并避免超出操作系统与硬件的限制。

`NCCL_MIN_NCHANNELS` 和 `NCCL_MAX_NCHANNELS`。这两个变量控制 NCCL 可使用的子环（subring）数量，因为 NCCL 会在可能时把数据拆分到多个 ring 上，以并行利用多条 NVLink。建议保留这些值的默认设置。在带有 NVSwitch 的 GPU 系统上，NCCL 会根据拓扑和消息大小自动调优通道（channel）数量。此外，NCCL 会创建与通道数相同数量的子环，以匹配并发硬件链路的数量。每个通道对应一个用于通信的 CUDA block。因此，更多的通道会占用更多 GPU 资源。

`NCCL_TOPO_FILE`。你可以用一个拓扑文件设置该变量，为系统提供指引，帮助 NCCL 做出明智决策。这在复杂网络或云环境中很有用，因为 NCCL 可能无法正确检测拓扑。要捕获 NCCL 在运行时检测到的拓扑，可把 `NCCL_TOPO_DUMP_FILE` 设置为一个输出路径并检视生成的文件。

`NCCL_MNNVL_ENABLE`。启用 NVIDIA 多节点 NVLink（MNNVL）。它面向支持多节点 NVLink 交换机的系统（例如 NVL72 GB200/GB300）的高速通信。

`NCCL_SHARP_DISABLE`。该设置控制是否使用 SHARP 进行在网聚合。我们稍后会讨论 SHARP。默认情况下，如果 SHARP 可用且作业配置为使用它，NCCL 就会启用它。你可以通过设置 `NCCL_SHARP_DISABLE=1` 显式禁用 SHARP，以进行 A/B 测试和排障。

总之，除非有证据表明某个可调项会有帮助，否则应使用环境默认值。而如果你确实更改了这些值，务必把它们记录下来，并在升级到更新的硬件和 NCCL 版本时持续监控其效果是否仍然有益。

陷阱 #4：核实 NCCL 线程的 CPU-GPU NUMA 节点亲和性

NCCL 会启动后台 CPU 线程用于网络轮询和内核派发。当你用 `torch.multiprocessing` 或消息传递接口（Message Passing Interface，MPI）启动多个进程时，每个进程会继承一个 CPU 亲和性掩码；如果用 `taskset` 或 `numactl` 之类工具进行了绑定，该掩码可能指向全部核心，也可能只指向一个子集。

NCCL 通常会把它的线程分配到离所服务 GPU 最近的核心上，但如果进程被固定到一小组核心上，它可能把所有 NCCL 线程都挤到单个核心上，从而遭受糟糕的调度和低吞吐量。为防止这种情况，可设置环境变量 `NCCL_IGNORE_CPU_AFFINITY=1`，让 NCCL 忽略继承来的 CPU 亲和性掩码，自由地把工作线程分散到本地 NUMA 域内的各个核心上。推荐的做法是：把每个 GPU 进程绑定到其 NUMA 域对应的 CPU 核心上，然后设置 `NCCL_IGNORE_CPU_AFFINITY=1`，这样 NCCL 就能在这些核心内部微调线程放置。

设想一个有两个 NUMA 节点域和八块 GPU 的计算节点。如果 GPU 0–3 连接到第一个 CPU，设备 4–7 连接到第二个 CPU，你就应把 rank 0–3 绑定到第一组 CPU 核心，把 rank 4–7 绑定到第二组 CPU 核心。接下来，你会设置 `NCCL_IGNORE_CPU_AFFINITY=1` 来忽略继承来的 CPU 亲和性掩码。

> 实践中，使用 `numactl` 或设置 `CUDA_DEVICE_ORDER` 和 `CUDA_VISIBLE_DEVICES` 有助于强制这种绑定。PyTorch 的启动工具会自动处理其中大部分工作，但最好还是核实一下。

你还可以指定一个显式的拓扑文件，以进一步降低延迟并提升吞吐量。如果你不愿手动固定进程，可以依赖 MPI 运行时绑定或作业调度器选项（例如 SLURM 的 `--cpu-bind`）来确保每个 rank 落在正确的核心上。

陷阱 #5：抵制忽视 NCCL 警告和错误的诱惑

如果审慎地启用了日志，NCCL 会打印许多日志。在这些日志中，可能会有诸如回退到较慢的 PCIe 带宽之类的警告。这些都是需要重视的重要警告。

如果你看到诸如“unable to enable P2P, falling back to copy”的日志，千万不要忽视！它们往往指示了次优的状况。看到这条警告，意味着 NCCL 无法在两块 GPU 之间建立直接的 GPU P2P，或许是因为它们位于不同的 PCIe root complex 且不受支持。这意味着数据传输会慢得多，因为它们必须经由主机 CPU 内存缓冲区中转。

这类警告可能促使你重新安排在某个进程中使用哪些 GPU。解决办法是确保需要相互通信的 GPU 位于同一 NUMA 节点上，或采用不同的配对方案。另一个例子是 `NCCL INFO NET/Socket: using Ethernet interface eth0` 警告，它告诉你选中了哪个接口。如果那不是性能最高的互连，你可能需要显式设置 `NCCL_SOCKET_IFNAME`。例如，你可以设置 `NCCL_SOCKET_IFNAME=ib0`，让引导握手（bootstrap handshake）使用预期的互连结构。你应当追查为什么它没有自动设置为最快的接口。这很可能是一个更大的问题。

陷阱 #6：NCCL 通信器挂起、报错或完全关闭

偶尔地，如果一个进程崩溃，或某个 GPU rank 遇到错误，NCCL 通信器可能会让其他 rank 挂起，因为集合操作无法完成。遗憾的是，鉴于大规模场景下 GPU 故障的频率相对较高，这在大规模集群中相当常见，正如 Meta 在表 4-4 中所描述的那样。

_表 4-4. Llama 3 405B 预训练 54 天期间意外中断的根因归类（source: https://oreil.ly/z8QKu）_

| 组件               | 类别       | 中断次数 | 中断占比 |
| ------------------ | ---------- | -------- | -------- |
| GPU 故障           | GPU        | 148      | 30.1%    |
| GPU HBM3 内存      | GPU        | 72       | 17.2%    |
| 软件缺陷           | 依赖项     | 54       | 12.9%    |
| 网络交换机/线缆    | 网络       | 35       | 8.4%     |
| 主机维护           | 计划外维护 | 32       | 7.6%     |
| GPU SRAM 内存      | GPU        | 19       | 4.5%     |
| GPU 系统处理器     | GPU        | 17       | 4.1%     |
| NIC                | 主机       | 7        | 1.7%     |
| NCCL 看门狗超时    | 未知       | 7        | 1.7%     |
| 静默数据损坏       | GPU        | 6        | 1.4%     |
| GPU 热界面与传感器 | GPU        | 6        | 1.4%     |
| SSD                | 主机       | 3        | 0.7%     |
| 电源               | 主机       | 3        | 0.7%     |
| 服务器机箱         | 主机       | 2        | 0.5%     |
| IO 扩展板          | 主机       | 2        | 0.5%     |
| 依赖项             | 依赖项     | 2        | 0.5%     |
| CPU                | 主机       | 2        | 0.5%     |
| 系统内存           | 主机       | 2        | 0.5%     |

启用 `NCCL_ASYNC_ERROR_HANDLING=1` 可以提升韧性，让 NCCL 在出错时异步中止，但这可能带来轻微的开销。近期版本的 PyTorch 在你使用 `init_process_group` 时会默认设置该项。不过，为了清晰和可复现，最好仍显式地设置这个值。

> 绝不要依赖默认值。永远保持显式！默认值有时会在不同版本之间发生变化——而当它们变化时极难调试。在初始化时显式设置这些值可以避免依赖版本的行为差异。

要把 NCCL 当作一台“开箱即用”的高性能引擎来看待，这一点很重要。但同时要留心你如何初始化和使用 NCCL：只初始化一次、恰当地固定 CPU、在需要时用环境变量调整亲和性，并对环境变量及其默认值保持谨慎。
