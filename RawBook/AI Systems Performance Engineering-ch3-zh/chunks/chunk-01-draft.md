# 第 3 章 面向 GPU 环境的操作系统、Docker 与 Kubernetes 调优

即便拥有高度优化的 GPU 代码与库，系统层面的瓶颈仍可能在大规模 AI 训练中拖累性能。最快的 GPU，也只能发挥出喂给它数据与指令的环境所允许的水平。本章将探讨如何调优操作系统与容器运行时，让 GPU 得以充分释放潜能。

我们首先探讨作为基础的 GPU 软件栈，随后深入若干关键的 CPU 与内存优化，例如 NUMA 亲和性（affinity）与大页（hugepages）。这些手段能确保数据从存储经由 CPU 高效流向 GPU。与此同时，我们还会讨论一些至关重要的 GPU 驱动设置，如持久化模式（persistence mode）、多进程服务（Multi-Process Service，MPS）以及多实例 GPU（Multi-Instance GPU，MIG）切分。它们通过降低开销、高效同步资源来帮助维持 GPU 的最大利用率。

借助 NVIDIA Container Toolkit、Container Runtime、Kubernetes 拓扑管理器（Topology Manager）以及 Kubernetes GPU Operator 等方案，你可以为 GPU 环境构建一套统一且高度优化的软件栈。这些方案能够在单节点与多节点 GPU 环境中实现高效的资源分配与工作负载调度，并确保 GPU 的能力得到充分利用。

在这一过程中，你会逐步建立起对“这些优化为何重要”的直觉。本质上，它们旨在最小化延迟、最大化吞吐量，并确保 GPU 持续获得数据供给、始终运行在性能峰值。最终成果是一套稳健、可扩展的系统，能够为训练与推理工作负载带来显著的性能提升，以及很高的有效吞吐（goodput）占比。

## 操作系统

操作系统（OS）是一切之上运行的基础。GPU 服务器通常运行某种 Linux 发行版，例如 Ubuntu Server LTS 或 Red Hat，并搭配支持最新 GPU 硬件的更新内核。NVIDIA 驱动会安装内核模块，创建诸如 `/dev/nvidia0`、`/dev/nvidia1`、`/dev/nvidia2` 之类的设备文件——每个 GPU 对应一个。驱动还会创建 `/dev/nvidiactl` 用于驱动控制操作，`/dev/nvidia-uvm` 用于统一虚拟内存，以及 `/dev/nvidia-modeset` 用于模式设置与缓冲区管理。

操作系统负责管理 CPU 调度、内存、网络与存储——所有这些都应针对高 GPU 吞吐量进行调优。因此，操作系统应被配置为不干扰 GPU 任务。例如，GPU 节点应禁用交换（swap），或将 `vm.swappiness` 设为 0，以避免任何由 OS 发起的、可能干扰 GPU 工作负载的内存交换。作为性能工程师，我们的部分职责就是调整这些 OS 设置，让 GPU 具备发挥最大性能的条件。

一台以 GPU 为核心的服务器可能还需要运行一些额外的守护进程（daemon）或后台进程，例如 NVIDIA Persistence Daemon，用于让 GPU 驱动与硬件上下文始终保持加载与就绪状态——即使当前没有任何 GPU 作业在运行。此外，Fabric Manager 负责管理 GPU 互连拓扑，而 NVIDIA Data Center GPU Manager（DCGM）则负责监控 GPU 系统健康指标。

## NVIDIA 软件栈

运行一个多 petaFLOP 级的 GPU 集群，远不止编写高层的 PyTorch、TensorFlow 或 JAX 代码那么简单。GPU 运行的背后有一整套软件栈支撑，其中每一层都可能影响性能。图 3-1 展示了一组用于开发并将现代 LLM 工作负载投入生产的常见框架、库、编译器、运行时与工具，包括 PyTorch、cuDNN、cuBLAS、CUTLASS、CUDA C++、`nvcc` 以及 CUDA Runtime API（例如 CUDA 工具、驱动等）。

此外，NVIDIA GPU 与 CUDA 生态也拥抱 Python 库，允许你使用诸如 OpenAI 的 Triton 领域特定语言（domain-specific language，DSL）与 NVIDIA 的 Warp 框架，以及 NVIDIA 的 CUDA Python、cuTile 和 CUTLASS 库，用 Python 编写 CUDA 内核。

![图 3-1. 一组用于开发并将现代 LLM 工作负载投入生产的常见框架、库、编译器、运行时与工具](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-1.png)

_图 3-1. 一组用于开发并将现代 LLM 工作负载投入生产的常见框架、库、编译器、运行时与工具_

### GPU 驱动

处于最底层的是 NVIDIA GPU 驱动，它在 Linux 操作系统与 GPU 硬件之间充当接口。驱动负责管理底层 GPU 操作，包括设备端的内存分配、GPU 核心上的任务调度，以及为多租户（multitenant）使用而对 GPU 进行切分。

GPU 驱动负责开启 GPU 的各项特性，并持续为硬件供给工作。保持 NVIDIA 驱动为最新版本非常重要。新的驱动版本往往能解锁性能提升，并支持最新的 GPU 架构与 CUDA 特性。

诸如 `nvidia-smi` 之类的工具随驱动一并提供，可用于监控温度、测量利用率、查询纠错码（error-correcting code，ECC）显存状态，以及启用持久化模式等不同的 GPU 模式。

### CUDA 工具包与运行时

在驱动之上是 CUDA 运行时以及一组称为 CUDA 工具包（CUDA Toolkit）的库。工具包中包含用于编译 CUDA C++ 内核的 CUDA 编译器 `nvcc`。编译完成后，CUDA 程序会链接到 CUDA 运行时（`cudart`）。CUDA 运行时直接与 NVIDIA 驱动通信，以在 GPU 上启动工作并分配内存。

此外，CUDA 工具包还提供了许多经过优化的库：用于神经网络原语的 cuDNN、用于线性代数的 cuBLAS、用于多 GPU 通信的 NCCL，等等。因此，使用支持你 GPU 计算能力（compute capability，CC）的最新 CUDA 工具包版本至关重要，因为最新的工具包拥有针对你 GPU 的最新编译器优化与专用库。我们将在后续章节中更详细地介绍 CUDA 编译器与编程模型，以及 CUDA（和 PyTorch）优化。

### 跨 GPU 硬件世代的 CUDA 向前与向后兼容

NVIDIA GPU 编程模型的一个重要特性是它跨硬件世代的兼容性。当你编译 CUDA 代码时，生成的二进制文件既包含虚拟的（或称中间的）PTX 代码，也包含物理设备代码（例如 ARM、x86、GPU 指令），如图 3-2 所示。

![图 3-2. 使用 `nvcc` 将 CUDA 程序编译为 PTX——并最终编译为面向 GPU 目标设备的底层指令](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-2.png)

_图 3-2. 使用 `nvcc` 将 CUDA 程序编译为 PTX——并最终编译为面向 GPU 目标设备的底层指令_

这使得较新的 GPU 能够即时（just-in-time，JIT）编译 PTX，从而让你的程序运行在未来的架构上——同时也让较新的 GPU 能够执行为先前架构生成的旧二进制代码。这种兼容性是通过 NVIDIA 的 fatbinary 模型实现的：其中既包含用于面向未来的 PTX，也包含面向已知架构的 CUBIN（即特定于架构的 CUDA 设备代码二进制）。

CUBIN 是 `nvcc` 使用 `-cubin` 选项生成的二进制文件。它包含针对某一给定 NVIDIA 架构编译好的 GPU 流式汇编（streaming assembler，SASS）指令，并被打包进 fatbinary，供 CUDA 驱动在运行时加载。与作为中间、向前兼容表示形式的 PTX 不同，CUBIN 二进制文件允许在已知的 GPU 架构上直接执行。当 CUBIN 与 PTX 一同被包含在 fatbinary 中时，它既支持为未来 GPU 即时编译 PTX，也支持在较新硬件上运行较旧的 CUBIN 代码。

简而言之，当嵌入了 PTX 时，CUDA 提供向前兼容性，因为驱动可以在运行时为较新架构即时编译 PTX。CUBIN 对象是特定于架构的，对未来的 GPU 架构并不向前兼容，因此你应当包含 PTX，或者交付同时包含面向当前架构的 SASS 与用于向前兼容的 PTX 的 fat 二进制（即 “fatbinaries” 或简称 “fatbins”）。

### C++ 与 Python CUDA 库

尽管大多数 CUDA 工具包库都是 C++ 的，NVIDIA 当前面向 Python 的选项包括：CUDA Python（例如底层驱动与运行时访问）；用于数组编程的 cuPyNumeric、CuTe DSL、cuTile 和 CuPy；以及用于以 Python 编写 GPU 内核的 NVIDIA Warp。CUTLASS 是一个 C++ 模板库，被 cuBLAS 等库在底层使用，而非一个 Python 库。

虽然大多数 CUDA 工具包库都基于 C++，但 NVIDIA 正推出越来越多基于 Python 的库，它们以 “Cu” 为前缀，并构建在 C++ 工具包之上。例如，cuTile 与 cuPyNumeric 是 2025 年初推出的 Python 库，旨在降低 Python 开发者使用 CUDA 为 NVIDIA GPU 构建应用的门槛。

cuTile 是一个 Python 库，通过将大矩阵拆分为更小、更易管理的子矩阵（称为 tile）来简化在 GPU 上处理大矩阵的工作。它提供了一种高层的、基于 tile 的抽象，使得执行分块计算、优化内存访问模式以及高效调度 GPU 内核变得更加容易。

通过将大矩阵划分为 tile，cuTile 帮助开发者充分利用 GPU 的并行性，而无需手动管理底层细节。这种方式能够改善缓存使用，并在需要密集矩阵计算的应用中带来整体上更好的性能。

cuPyNumeric 是流行的 `numpy` Python 库的一个即插即用替代品（`import cupynumeric as np`），可利用 GPU。它提供了与 NumPy 几乎相同的函数、方法与行为，因此开发者通常只需对代码做极少改动即可切换过来。在底层，cuPyNumeric 借助 CUDA 在 GPU 上并行执行操作。这为大规模数值计算、矩阵运算与数据分析等计算密集型任务带来了显著的性能提升。

通过将工作卸载到 GPU，cuPyNumeric 加速了计算，并提升了处理海量数据集的应用的效率。它的目标是降低 Python 开发者驾驭 GPU 算力的门槛，而无需学习一套全新的接口，从而使其成为面向高性能计算、可有力替代 NumPy 的即插即用选择。

另一个值得注意的基于 Python 的编程模型是 OpenAI 的开源 Triton 语言与编译器。Triton 是一种 Python DSL，允许用 Python 编写自定义 GPU 内核。虽然它并非 NVIDIA 的库，但 Triton 通过让开发者直接用 Python 编写高性能内核，对 CUDA 形成了补充。

我们将在后续章节中介绍 Triton 及各种基于 Triton 的优化，这里只需知道：在许多情况下，Triton 减少了手写 CUDA C++ 的需求。而且它已集成到 PyTorch 的编译器后端中，可自动优化并融合 GPU 操作以获得更好的性能。下面我们把讨论转向 PyTorch。

### PyTorch 与更高层的 AI 框架

一些流行的、构建于 CUDA 之上的基于 Python 的框架包括 PyTorch、TensorFlow、JAX 和 Keras。这些框架为深度学习提供高层接口，同时充分利用 NVIDIA GPU 的算力。本书主要聚焦于 PyTorch 的编译与图优化特性，包括 `torch.compile` 栈。

PyTorch 编译器栈由 TorchDynamo、AOT Autograd，以及诸如 TorchInductor 或加速线性代数（Accelerated Linear Algebra，XLA）之类的后端组成，它们能够自动捕获并优化你的模型。TorchInductor 是最常用的后端，它在底层使用 OpenAI 的 Triton。正如我们将在第 14 章介绍的那样，Triton 会融合内核，并针对你特定的 GPU 与系统环境进行内核自动调优。

当你使用 GPU 对 PyTorch 张量执行操作时，它们看似只经过一次 Python 调用便从 CPU 移动到了 GPU。然而，这一次调用实际上被转换为对 CUDA 运行时的一系列调用，并借助了各种 CUDA 库，如图 3-3 所示。

![图 3-3. 从 PyTorch 代码到 GPU 设备的流程](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-3.png)

_图 3-3. 从 PyTorch 代码到 GPU 设备的流程_

例如，当你执行矩阵乘法时，PyTorch 会把这些任务委托给 cuBLAS 之类的库。cuBLAS 是 CUDA 工具包的一部分，并针对 GPU 执行进行了优化。在幕后，PyTorch 会确保前向与后向传播等操作都使用底层、经过优化的 CUDA 函数与库来执行。

简而言之，PyTorch 抽象掉了直接进行 CUDA 编程的复杂性，让你能够编写直观的 Python 代码，而这些代码最终会调用高度优化的 CUDA 例程，从而兼得开发的便捷与高性能。我们将在第 4、5 章讨论 CUDA 编程与优化，并在第 9 章讨论 PyTorch 优化。

所有这些组件——OS、GPU 驱动、CUDA 工具包、CUDA 库以及 PyTorch——必须协同工作，才能打造出理想的基于 GPU 的开发环境。当一名研究者提交训练作业时，调度器会预留节点，操作系统会借助 NVIDIA 驱动提供 GPU 设备与内存分配，容器则提供正确的软件环境（包括经过优化、硬件感知的 CUDA 库）。用户代码（例如 PyTorch、TensorFlow、JAX）使用这些 CUDA 库，而这些库最终会与驱动和硬件通信。

本章所述的各项优化，旨在使这一软件栈的每一层都尽可能高效。它们将帮助 GPU 始终忙于真正有用的训练与推理工作——而不是让 GPU 等待 CPU、等待内存或磁盘 I/O，抑或等待其他 GPU 同步。

一个调优良好的系统，能够确保拆分到数十个 GPU 上的模型不会被 I/O 或 OS 开销所拖累。相比模型优化，系统层面的调优常常被忽视，然而系统层面的优化却能带来可观的性能提升。在某些情况下，只需对 OS 层配置做些许微调，就能获得两位数百分比的改进。在大型 AI 项目的规模下，这可以在计算时间上节省数万乃至数十万美元。

## 面向 GPU 环境配置 CPU 与操作系统

GPU 无法达到充分利用率的最常见原因之一，就是 CPU 没能持续为其供给有用的工作。在典型的训练循环中，CPU 负责准备下一批数据，包括从磁盘加载数据、对数据进行分词、变换等。此外，CPU 还负责派发 GPU 内核，并在线程与进程之间进行协调。

如果这些主机侧任务很慢——或者操作系统对它们的调度很糟糕——那么昂贵的 GPU 可能会陷入空闲，无所事事地空转，等待下一个任务或下一批数据。为避免这种情况，我们需要优化 CPU 与操作系统处理 GPU 工作负载的方式。

这些优化包括：设置 CPU 亲和性以避免跨 NUMA 节点的流量，从而让恰当的核心处理恰当的数据；采用相应的内存分配策略以规避 NUMA 惩罚；以及应用 OS 层的改动以消除不必要的延迟。如此一来，GPU 便永远不会因缺数据而“挨饿”。其中一部分工作，是把后台守护进程与 OS 任务隔离到它们自己的核心上——远离那些为 GPU 供给数据的核心，这一点我们接下来会讨论。

### NUMA 感知与 CPU 绑定

现代服务器 CPU 拥有数十个核心，且常被划分为多个 NUMA（非统一内存访问）节点。NUMA 节点是一组在物理上彼此靠近的 CPU、GPU、网卡（network interface controllers，NIC）与内存的逻辑分组。了解系统的 NUMA 架构对性能调优十分重要。访问单个 NUMA 节点内的资源，要比访问其他 NUMA 节点的资源更快。

例如，如果一个运行在 NUMA 节点 0 上的 CPU 进程需要访问位于 NUMA 节点 1 的 GPU，它就需要跨节点间链路发送数据，从而带来更高的延迟。事实上，当访问跨越到其他 NUMA 节点时，内存访问延迟可能几乎翻倍。

> 在诸如 GH200 与 GB200 之类基于 Grace 的超级芯片上，CPU 与 GPU 通过 NVLink-C2C 相连，可在 Grace 及其配对的加速器之间提供高达约 900 GB/s 的一致性（coherent）CPU-到-GPU 内存访问。Linux 仍将 CPU DRAM 视为 CPU NUMA 内存，将 GPU HBM 视为设备内存。因此，即使一致性降低了软件开销，你仍应继续将 CPU 线程绑定到本地的 Grace CPU 并尊重数据局部性。

在许多双路（dual-socket）系统上，远程内存访问延迟可能显著高于本地内存访问。在一次实验中，本地 NUMA 节点的内存访问延迟约为 80 ns，而远程（跨节点）内存访问延迟约为 139 ns。这大约是 75% 的延迟增加，本地与远程 NUMA 节点内存访问在访问速度上的差距巨大。

通过将进程绑定到与其 GPU 处于同一 NUMA 节点的 CPU 上，我们就能避免这一额外开销。例如，你可以使用 `numactl --cpunodebind=<node> --membind=<node>`，将 CPU 线程与内存分配都绑定到 GPU 的本地 NUMA 节点。稍后你会进一步了解这一点。其核心思想是：让 CPU 的执行与内存访问都保持在它所服务的 GPU 的本地范围内。

> 虽然 Linux 内置了基本的 NUMA 平衡机制，但对于性能关键的 AI 工作负载而言，它通常并不足够。默认情况下，进程可能会在 NUMA 节点之间迁移，这会因远程内存访问而带来额外延迟。因此，显式地将进程与内存绑定到与本地 GPU 相同的 NUMA 节点非常重要。你可以使用 `numactl`、`taskset` 或 cgroups 来做到这一点，稍后我们会展示。

要显式指定 NUMA 亲和性，你需要将进程或线程“钉”（pin）到与 GPU 连接同一 NUMA 节点的特定 CPU 上。这种 CPU 亲和性称为 CPU 绑定（CPU pinning）。假设你在一个节点中有八个 GPU，其中四个连接到 NUMA 节点 0，另外四个连接到 NUMA 节点 1。

如果你启动八个训练进程，每个 GPU 对应一个，你就应当把每个训练进程绑定到与相应 GPU 连接同一 NUMA 节点的一个 CPU 核心——或一组 CPU 核心——上。在这种情况下，GPU 0–3 连接到 NUMA 节点 0，而 GPU 4–7 连接到 NUMA 节点 1 的核心，如图 3-4 所示。

![图 3-4. 一个节点中的八个 GPU，其中四个连接到 NUMA 节点 0，另外四个连接到 NUMA 节点 1](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-4.png)

_图 3-4. 一个节点中的八个 GPU，其中四个连接到 NUMA 节点 0，另外四个连接到 NUMA 节点 1_

这样一来，当一个 CPU 进程想要向 GPU 4 供给数据时，它就应当运行在连接到 NUMA 节点 1 的 CPU 上，因为 GPU 4 连接到的正是 NUMA 节点 1。Linux 提供了实现这一点的工具，包括 `numactl --cpunodebind=<node> --membind=<node>`，它会启动一个被钉到给定 NUMA 节点的进程。

你也可以使用 `taskset` 将进程钉到特定的核心 ID 上。下面是一个使用 `numactl` 将 `train.py` 脚本绑定到与 GPU 4 处于同一 NUMA 节点 1 的 CPU 上的示例：

```
numactl --cpunodebind=1 --membind=1 \
    python train.py --gpu 4
```

这里假设我们已知 NUMA 节点 ID，并且我们只将脚本绑定到一个 GPU。将 `train.py` 绑定到位于未知 NUMA 节点上的多个 GPU 则要复杂一些。下面的脚本使用 `nvidia-smi topo` 动态查询拓扑，并将脚本绑定到使用本地 NUMA 节点的各个 GPU：

```
#!/bin/bash
for GPU in 0 1 2 3; do
  # Query NUMA node for this GPU
  NODE=$(nvidia-smi topo -m -i $GPU \
         | awk '/NUMA Affinity/ {print $NF}')
  # Launch the training process pinned to that NUMA node
  numactl --cpunodebind=$NODE --membind=$NODE \
    bash -c "CUDA_VISIBLE_DEVICES=$GPU python train.py --gpu $GPU"
done
```

在这里，我们使用 `topo -m` 同时获取 CPU 与 NUMA 亲和性。随后我们从 NUMA Affinity 列中提取单个节点 ID。最后，我们将 `--cpunodebind` 和 `--membind` 都绑定到该节点，以确保你进程的线程与内存分配都保持在 GPU 的 NUMA 域本地。

许多深度学习框架也允许你以编程方式设置线程亲和性。例如，PyTorch 的 DataLoader 暴露了 `worker_init_fn`，让你可以在初始化期间为每个 worker 进程设置 CPU 亲和性，如下所示：

```
import os
import re
import glob
import subprocess
import psutil
import ctypes
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from functools import partial
# Optional: NVML is preferred for GPU↔ NUMA mapping
try:
    import pynvml as nvml  # pip install nvidia-ml-py3
    _HAS_NVML = True
except Exception:
    _HAS_NVML = False
# --- libnuma for memory binding
_libnuma = ctypes.CDLL("libnuma.so")
if _libnuma.numa_available() < 0:
    raise RuntimeError("NUMA not available on this system")
_libnuma.numa_run_on_node.argtypes = [ctypes.c_int]
_libnuma.numa_set_preferred.argtypes = [ctypes.c_int]
def parse_physical_cpu_list(phys_str: str):
    """Parse '0-3,8-11' -> [0,1,2,3,8,9,10,11]."""
    cpus = []
    if not phys_str:
        return cpus
    for part in phys_str.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            start, end = map(int, part.split('-'))
            cpus.extend(range(start, end + 1))
        else:
            cpus.append(int(part))
    return cpus
def get_numa_cpus_for_node(node: int):
    """Read /sys/devices/system/node/node{node}/cpulist."""
    path = f"/sys/devices/system/node/node{node}/cpulist"
    with open(path, "r") as f:
        return parse_physical_cpu_list(f.read().strip())
def get_numa_cpus_and_memory():
    """Return (current_cpu_mask, preferred_node) from numactl --show."""
    out = subprocess.run(["numactl", "--show"],
        capture_output=True, text=True).stdout
    phys = re.search(r"physcpubind:\s*([\d,\-\s]+)", out).group(1)
    cpus = parse_physical_cpu_list(phys)
    node = int(re.search(r"preferred node:\s*(-?\d+)", out).group(1))
    return cpus, node
def get_gpu_numa_node(device: int) -> int:
    """
    Determine NUMA node for a GPU (prefer NVML; fall back to sysfs;
    final fallback to current preferred node).
    """
    # NVML path (preferred)
    if _HAS_NVML:
        try:
            nvml.nvmlInit()
            props = torch.cuda.get_device_properties(device)
            pci = props.pci_bus_id  # '0000:03:00.0' or '00000000:03:00.0'
            # Normalize to 8-hex-digit domain if needed for NVML
            try:
                domain, bus, devfn = pci.split(':')
                if len(domain) < 8:
                    domain = domain.rjust(8, '0')
                pci8 = f"{domain}:{bus}:{devfn}"
            except ValueError:
                pci8 = pci
            try:
                handle = nvml.nvmlDeviceGetHandleByPciBusId_v2(pci8)
            except AttributeError:
                handle = nvml.nvmlDeviceGetHandleByPciBusId(pci8)
            # Direct NUMA ID if driver exposes it
            try:
                numa_id = nvml.nvmlDeviceGetNUMANodeId(handle)
                if isinstance(numa_id, int) and numa_id >= 0:
                    return numa_id
            except Exception:
                pass
            # Derive from NVML CPU affinity
            cpu_count = psutil.cpu_count(logical=True)
            elems = (cpu_count + 63) // 64
            mask = nvml.nvmlDeviceGetCpuAffinity(handle, elems)
            cpus = []
            for i, m in enumerate(mask):
                m = int(m)
                for b in range(64):
                    if m & (1 << b):
                        cpu_id = i * 64 + b
                        if cpu_id < cpu_count:
                            cpus.append(cpu_id)
            # Build CPU→NUMA map from sysfs and choose majority node
            cpu2node = {}
            for node_path in sorted(glob.glob("/sys/devices/system/node/node*")):
                node_id = int(os.path.basename(node_path).replace("node", ""))
                with open(os.path.join(node_path, "cpulist"), "r") as f:
                    for c in parse_physical_cpu_list(f.read().strip()):
                        cpu2node[c] = node_id
            counts = {}
            for c in cpus:
                n = cpu2node.get(c)
                if n is not None:
                    counts[n] = counts.get(n, 0) + 1
            if counts:
                return max(counts.items(), key=lambda kv: kv[1])[0]
        except Exception:
            pass
    # sysfs fallback
    try:
        props = torch.cuda.get_device_properties(device)
        pci = props.pci_bus_id
        sysfs_path = f"/sys/bus/pci/devices/{pci}/numa_node"
        with open(sysfs_path, "r") as f:
            val = int(f.read().strip())
            return val if val >= 0 else 0
    except Exception:
        pass
    # last resort: current preferred node
    _, node = get_numa_cpus_and_memory()
    return node if node >= 0 else 0
def set_numa_affinity(node: int):
    """Bind current process to CPUs and memory of the given NUMA node."""
    cpus = get_numa_cpus_for_node(node)  # IMPORTANT: CPUs of target node
    psutil.Process(os.getpid()).cpu_affinity(cpus)
    _libnuma.numa_run_on_node(node)
    _libnuma.numa_set_preferred(node)
    print(f"PID={os.getpid()} bound to NUMA node {node} (CPUs={cpus})")
    return cpus
def _worker_init_fn(worker_id: int, node: int, cpus: list):
    """Reapply binding in each DataLoader worker (no CUDA calls here)."""
    psutil.Process(os.getpid()).cpu_affinity(cpus)
    _libnuma.numa_run_on_node(node)
    _libnuma.numa_set_preferred(node)
    print(f"Worker {worker_id} (PID={os.getpid()}) bound to NUMA node {node}")
 # ----- Example usage below -----
class MyDataset(Dataset):
    def __len__(self): return 1024
    def __getitem__(self, idx): return torch.randn(224*224*3, device='cpu')
def main():
    # DDP setup
    dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.cuda.current_device()
    # Determine GPU's NUMA node and bind this process
    gpu_node = get_gpu_numa_node(device)
    cpus = set_numa_affinity(gpu_node)
    # Build dataloader with closure-based worker_init_fn
    dataset = MyDataset()
    init_fn = partial(_worker_init_fn, node=gpu_node, cpus=cpus)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,  # reduces worker respawn churn
        worker_init_fn=init_fn,
        prefetch_factor=2,
    )
    # Model and DDP
    model = torch.nn.Linear(224*224*3, 10, bias=True).to("cuda")
    ddp_model = DDP(model, device_ids=[device], static_graph=True)
    for batch in dataloader:
        batch = batch.to("cuda", non_blocking=True)
        out = ddp_model(batch)
        # ... loss, backward, optimizer ...
if __name__ == "__main__":
    main()
```

该脚本将主训练进程与每个 DataLoader worker 进程都绑定到 GPU 的本地 NUMA 节点，以防止跨 NUMA 的内存访问。在 DataLoader 中，我们传入了一个基于闭包的 `worker_init_fn`，它会在每个 worker 内部重新应用预先计算好的 NUMA 绑定。而且我们在 worker 中做这件事时，不会触碰任何 CUDA API。

在启动阶段，进程使用 NVML 将当前 GPU 映射到其 NUMA 节点与 CPU 亲和性掩码。在可用时，我们通过 `nvmlDeviceGetNUMANodeId` 直接读取节点。否则，我们从 GPU 的 CPU 亲和性掩码（`nvmlDeviceGetCpuAffinity`）推导出它。如果 NVML 不可用或不暴露该节点，我们就回退到内核在 `/sys/bus/pci/devices/<PCI_ID>/numa_node` 处的 sysfs 条目。作为最后手段，我们使用进程当前的首选（preferred）节点。

随后，我们从 `/sys/devices/system/node/node<N>/cpulist` 计算出该节点的 CPU 列表，并用 `psutil` 将 CPU 亲和性应用到这些核心上。我们还使用 `libnuma`（`numa_run_on_node + numa_set_preferred`）将所有未来的分配都绑定到该节点。

由于某些启动器、容器运行时或内核并不能可靠地将 NUMA 策略传播给子进程，我们会在每个 fork 出来的 worker 中显式地重新应用并验证绑定。仅仅依赖继承并不安全。

> 记得设置 `pin_memory=True`，并在 H2D（主机到设备）拷贝上使用 `non_blocking=True`，这样页锁定的主机缓冲区就会保持在正确的 NUMA 节点上。优先使用 `persistent_workers=True`，以避免重新 fork worker 并在各个 epoch 之间丢失它们的亲和性。另外，不要在 `worker_init_fn` 中调用 `torch.cuda.*`；相反，应通过闭包或环境变量传入 GPU 索引。

其结果是：数据准备与批次加载完全发生在本地内存中。如此一来，你的 GPU 会保持忙碌，永远不需要为一次远程 NUMA 跳转而暂停。有了这段代码，在任何安装了 `libnuma` 与 `numactl` 的 Linux 服务器上，你都能获得稳健的、拓扑感知的亲和性。

默认情况下，`numactl` 会将其 CPU 与内存策略应用到某个进程，并且据文档所述会将该策略继承给所有 fork 出来的子进程。然而在实践中，由 Python 框架派生的线程或 exec 出来的子进程，并不总能在每个内核或 Linux 发行版上采用相同的设置。当使用由框架管理的 worker 进程时，你应当在每个 worker 内部显式地重新确立 CPU 与内存策略。

> 对于像 Grace Blackwell（以及 Vera Rubin）这样的超级芯片架构，CPU 与 GPU 通过 NVLink-C2C 保持一致性。然而，Linux 仍然将 CPU DRAM 与 GPU HBM 建模为彼此独立的内存池。将 CPU 线程绑定到本地 CPU NUMA 节点，对于局部性而言仍然有益。

在实践中，绑定可以消除不可预测的 CPU 调度行为。它能确保像“为你的 GPU 加载数据的线程”这样的关键线程，不会在训练或推理进行到一半时突然被操作系统迁移到另一个 NUMA 节点的核心上。实践中，仅仅通过消除跨 NUMA 流量与 CPU 核心迁移，就有可能看到 5%–10% 的训练吞吐量提升。这往往还能降低性能抖动（jitter）与波动。

许多高性能 AI 系统会评估 CPU 同步多线程（simultaneous multithreading，SMT，也常被称为超线程 / hyperthreading）——有时为了获得更可预测的单核性能而将其禁用，但其收益取决于具体工作负载。这些系统还可能通过设置 `isolcpus` 内核参数，将少数几个核心从通用调度器中隔离出来，专门保留给 OS 后台任务。你也可以使用 Kubernetes 的 CPU 隔离来处理系统守护进程。这确保了其余核心能够完全专用于训练与推理线程，去做真正有用的工作。

需要指出的是，对于像 NVIDIA Grace Blackwell 这样的集成式 CPU-GPU 超级芯片，许多关于 CPU-到-GPU 数据传输的传统顾虑都得到了缓解，因为 CPU 与 GPU 通过 NVLink-C2C 暴露出一个一致性的共享虚拟地址空间，而 CPU DRAM 与 GPU HBM 仍是彼此独立的内存池。这意味着跨 NUMA 延迟之类的问题被最小化，数据可以在 CPU 与 GPU 之间更直接地流动。

> NVIDIA 通过把 CPU 与 GPU 合并到单一超级芯片（如 Grace Blackwell 架构）上来攻克 CPU-到-GPU 瓶颈，这并非偶然。在这一设计中，CPU 与 GPU 甚至通过高达 900 GB/s 的 NVLink-C2C 共享一块统一、一致的内存，从而最小化数据传输开销。可以预期，NVIDIA 将继续以更多此类硬件创新来应对系统瓶颈，并与软件和算法的需求协同设计。
