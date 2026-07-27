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

即便采用了紧耦合的 CPU-GPU 超级芯片架构，通过确保硬件与软件配置得当、让集成系统以峰值效率运行来优化整个软件栈依然很重要。即使在这些紧耦合架构中，你也希望尽量消除数据处理中一切不必要的延迟，从而让 GPU 保持满负荷利用。正如你将在接下来几节中看到的，这包括配置大页、使用高效预取以及固定内存。

### NUMA 友好的内存分配与内存固定

默认情况下，进程会从其当前运行所在 CPU 的 NUMA 节点分配内存。因此如果你把一个进程绑定到 NUMA 节点 0，它的内存自然会来自 NUMA 节点 0 的本地 RAM，这是理想状态。然而，如果 OS 调度器迁移了线程——或者某些内存是在你完成绑定之前就已分配的——你可能会陷入这样一种非理想情形：运行在 NUMA 节点 0 上的进程却在使用来自 NUMA 节点 1 的内存。在这种情况下，每一次内存访问都必须跳转到另一个 NUMA 节点，从而抵消了 CPU 绑定带来的收益。

为避免这种情况，`numactl --membind` 选项会强制从指定的 NUMA 节点分配内存，正如前面某一节所提到的。在代码中，也有一些 NUMA API 甚至环境变量可以影响这项配置。总的原则是：让内存靠近 CPU，而 CPU 又靠近 GPU。这样一来，从内存到 CPU 再到 GPU 的整条数据搬运链路都处于单个 NUMA 节点之内。下面还是之前那个例子，但加上了 `--membind=1`，以强制从包含 NUMA 节点 1 的首选 NUMA 节点分配内存：

```
numactl --cpunodebind=1 --membind=1 python train.py --gpu 5 &
```

需要重点指出的是，当你在 `numactl` 下启动一个进程时，它的 CPU 策略（`--cpunodebind`）和内存策略（`--membind`）都会应用于该进程，并被其所有子进程继承。因此，你的训练脚本 fork 出来的任何工作子进程都会自动使用相同的 NUMA 内存绑定。但是，它们必须以基于 fork 的模型创建。如果你改用 spawn 启动方式，或者以其他方式 `exec` 一个新程序，那些子进程就不会继承父进程的内存策略。

此外，固定内存（pinned memory，也称页锁定内存）对于高效且直接的 GPU 访问至关重要。当内存被固定后，OS 不会将其换出或移动。这带来了更快的直接内存访问（direct memory access，DMA）传输。由于 GPU 或 NIC 可以直接执行 DMA，从固定的主机内存拷贝数据到 GPU，可能比从普通可分页内存拷贝快 2–3×。

> 你可以使用已安装的 CUDA 实用工具中的 `bandwidthTest --memory=<pinned or pageable>` 来测试 CPU 内存与 GPU 内存之间的数据传输带宽。

事实上，这正是 NVIDIA GPUDirect 技术（如 GPUDirect RDMA）的基础，它让 InfiniBand 等 NIC 能够与 GPU 内存直接交换数据。类似地，GPUDirect Storage（GDS）让 NVMe 驱动器无需额外的 CPU 开销即可将数据流式传输进 GPU 内存。

深度学习框架提供了在数据加载器中使用固定内存的选项。例如，PyTorch 的 `DataLoader` 有一个标志 `pin_memory=True`，当其为 true 时，意味着加载的批次将被放入固定 RAM 中，如图 3-5 所示。

![Figure 3-5. Pinned memory (aka page-locked or nonpageable) is a type of memory that cannot be swapped out to disk](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-5.png)

_图 3-5. 固定内存（又称页锁定或不可分页内存）是一种无法被换出到磁盘的内存。_

内存固定可以加速 `tensor.to(device)` 操作，因为 CUDA 驱动无需即时地去固定页面。当你使用较大的批大小或每次迭代读取大量数据时，它尤其有益。许多从业者已经注意到，仅仅在 PyTorch 中打开 `pin_memory=True`，就能通过减少数据传输瓶颈、提高主机到设备的传输吞吐量，将性能提升多达 10%–20%。

简而言之，你应确保数据加载器使用固定内存（例如在 PyTorch `DataLoader` 中设置 `pin_memory=True`），并对受支持的硬件启用 GPUDirect RDMA 与 GDS。这会降低数据传输延迟。

需要重点指出的是，OS 对用户可以锁定（固定）的内存量设有上限。

> 这通过 `ulimit -l <max locked memory>` 命令设置。在容器化环境中，你可以相应地调整容器的安全上下文和 Docker 的 `--ulimit memlock` 设置。这样，容器就能锁定足够的内存。

> 如果你打算使用大型的固定缓冲区，请确保 `ulimit` 值足够高——或将其设为无限制。否则分配可能失败。对于大型 AI 工作负载和高性能计算（high-performance computing，HPC）应用，通常会将其设为无限制。

### 透明大页

除了固定内存并将其绑定到 NUMA 节点之外，我们还应谈谈透明大页（transparent hugepages，THP）。Linux 内存管理通常使用 4 KB 的页面，但当进程使用数十甚至数百 GB 内存时（如深度学习数据集、预取批次、模型参数等情形），管理数以百万计的微小页面效率低下。

大页——2 MB 甚至 1 GB 的页面——通过增大内存块，可以降低虚拟内存管理的开销。主要好处是更少的缺页（page fault）以及对转换后备缓冲（translation lookaside buffer，TLB）更小的压力。

TLB 是 CPU 用来将虚拟地址映射到物理地址的一种缓存。页面更少、更大，意味着在相同数量的条目下 TLB 能覆盖更多内存，从而减少未命中。

大页通常带来适度的收益——往往在 ~3%–5% 吞吐量提升这个量级。它们通过减少缺页开销和 TLB 压力来实现这一点。在大多数系统上启用 THP 是一个轻松的收益，因为内核会自动用 2 MB 页面来支撑大额分配。在拥有超大内存池的场景（例如为 I/O 预分配的固定缓冲区）中，你还可以考虑使用 `vm.nr_hugepages` 或 `hugetlbfs` 的显式大页，以获得更确定的性能。

> 请记住，当使用大型的固定内存区域时，你应将 `ulimit -l` 设置（最大锁定内存）调高到一个较大的值或设为 `unlimited`。如果该上限太低，你固定内存的尝试可能失败，导致回退到可交换内存——或出现内存不足（out-of-memory，OOM）错误。

需要重点指出的是，THP 的后台压缩可能引入不可预测的停顿，这对延迟敏感的 LLM 推理工作负载是灾难性的。Linux 默认配置为使用 THP，只要有可能就自动分配 2 MB 页面。这通常已经足够，但仍值得针对你的工作负载进行测试。

你可以禁用 THP，但那样就需要手动分配和控制大页。这会引入额外的复杂性，但对于推理这类低延迟工作负载可能是必要的。禁用 THP 后，你的系统将避免由内核驱动的碎片整理所导致的停顿。

> 现代的普遍共识是：对大多数以吞吐量为重的、基于 GPU 的训练工作负载启用 THP；而对于推理这类以延迟为重的工作负载，则完全禁用 THP（`transparent_hugepage=never`）——或使用 `madvise`。对于许多 rank（GPU）同时分配内存的分布式训练工作负载，同样如此。

除了 CPU/内存绑定与大页之外，还有一些其他值得一提的 OS 层调整。这包括线程调度、虚拟内存管理、文件系统缓存以及 CPU 频率设置，我们将在接下来几节中介绍。

### 调度器与中断亲和性

在繁忙的系统上，你希望确保重要线程（如数据流水线线程）不会被频繁中断。Linux 默认使用完全公平调度器（Completely Fair Scheduler，CFS），它在大多数情况下表现良好。

但如果你有一个对延迟极其敏感、专门向 GPU 供给数据的线程，你可以考虑对该线程使用实时的先进先出（first in, first out，FIFO）或轮转（round-robin，RR）优先级调度。这将确保这个高优先级线程运行时不会被普通优先级的线程抢占。

不过，使用时要谨慎，因为如果管理不当，实时线程可能会使其他进程陷入饥饿。但在实践中，如果你已经将线程绑定到专用核心上，通常就不需要去折腾实时线程优先级了，但仍值得留意。

另一个选择是隔离核心或创建独立的 CPU 分区，以进一步减少对这些专用计算资源的干扰。为此，你可以使用 `cset`、诸如 `isolcpus` 和 `nohz_full` 的内核参数，或 cgroup 的 `cpuset` 隔离。有了隔离，OS 调度器就会把那些 CPU 核心留给你随意使用。

> 在生产环境中强烈推荐使用 cgroup 的 CPU 与内存亲和性。借助它们，每个 AI 工作负载都被隔离在各自的物理核心和内存区域上。这将防止跨工作负载争用和 NUMA 惩罚。应使用诸如 `cpuset` cgroup 或容器运行时（`docker --cpuset-cpus`）之类的工具来强制执行这一点。

你可以将每个设备的硬件中断分配到同一 NUMA 节点上的核心。这将防止跨节点的中断处理，否则会带来额外延迟并逐出远端节点上有用的缓存行。例如，如果你位于 NUMA 节点 0 的 GPU 或 NIC 触发了一个中断，你就应将其绑定到节点 0 上的某个核心，以便没有其他节点来处理它。若没有这种绑定，位于不同 NUMA 节点的 CPU 可能会处理该中断。这将强制产生缓存一致性流量和跨节点通信。

在实践中，对性能敏感的系统常常会禁用默认的 `irqbalance` 守护进程，或以定制规则运行它。另一个选择是使用 `/proc/irq/*/smp_affinity` 手动设置每个中断的亲和性掩码。通过把每个 GPU 和 NIC 的中断都固定到最近的核心，你就能保证这些设备中断总是在最优的 NUMA 节点上得到处理。

简而言之，专用核心、恰当的调度优先级以及 NUMA 感知的硬件中断绑定三者相结合，有助于把向 GPU 供给数据的数据加载线程的抖动（jitter）降到最低。

### 虚拟内存与交换

这不言自明，但你应始终尽量避免内存交换。如果你进程内存的任何部分被换出到磁盘，你将会看到灾难性的、数个数量级的性能下降。GPU 程序往往会分配大量主机内存用于数据缓存。如果 OS 决定把某些数据换出内存到磁盘，那么当 GPU 需要访问那些数据时就会遭遇巨大的延迟。

我们建议设置 `vm.swappiness=0`，它告诉 Linux 除非在极端内存压力下否则避免交换。它实际上通过 cgroup 限制来隔离你训练作业的内存，以防止任何交换。

> 你应通过 Docker 或 Kubernetes 使用 cgroups v2 将内存和 CPU 固定给 AI 进程。这将在容器化环境中强制实施 NUMA 亲和性和禁止交换策略。

你也可以使用 `sudo swapoff -a` 临时禁用所有交换设备和文件，直到下次重启为止。只要确保你有足够的 RAM 支撑工作负载——或者设置上限以防止过量分配（overcommit）。否则，OOM killer 可能会终结该进程。使用 `vmstat` 或 `free -m` 监控交换使用情况，以确保交换保持为零。

另一个相关设置是 `ulimit -l`，正如前面针对固定内存所提到的。如果你想防止内存被交换，就应把该上限设得高一些，否则你可能会遭遇过度的内存交换。同样地，对于占用大量内存的大型 AI 工作负载，通常会将该上限设为无限制。

### 文件系统缓存与回写

针对大型训练作业的一个最佳实践是：频繁地向磁盘写入检查点（checkpoint），以便在作业失败时能够从一个已知良好的检查点重启。然而在写检查点期间，海量的数据突发可能会填满 OS 页缓存（page cache）并引发停顿。

对于存储，你可以调整 `vm.dirty_ratio` 和 `vm.dirty_background_ratio` 来调优用于缓冲写入的页缓存大小。例如，对于数 GB 大小的检查点，使用更高的脏页比率能让 OS 在刷写到磁盘之前在 RAM 中批量缓存更多数据。这将平滑大型检查点写入，减少训练循环中的停顿。

另一个选择是在一个独立线程中执行写检查点。PyTorch 中一个更新的选项是从集群中的各节点写出分布式检查点分片。在这种情况下，当作业失败重启后加载检查点时，这些检查点分片将被合并起来。

在延迟敏感的训练工作流中，最好完全绕过页缓存。例如，用 `O_DIRECT` 打开检查点文件，或使用 Linux 的 `io_uring` 进行异步 I/O，以避免页缓存停顿。每次写完检查点后，调用 `posix_fadvise` `(fd, 0, 0, POSIX_FADV_DONTNEED)` 立即把那些页面从缓存中丢弃，防止在后续迭代中造成内存压力。

### CPU 频率与 C-state

默认情况下，许多计算节点会让 CPU 运行在省电模式下，即在 CPU 空闲时对其降频或让其进入睡眠。这有助于节能、降低发热并降低成本。在模型训练期间，当 GPU 正在处理其数据集的最后几个批次时，CPU 可能并不总是被 100% 利用。然而，当有新任务到来、系统再次唤醒 CPU 时，这些电源管理特性可能会带来额外延迟。

为获得最大且一致的性能，AI 系统通常会将 CPU 频率调速器（governor）配置为 “performance” 模式，使 CPU 始终保持在最高频率。

> 这可以通过 `cpupower frequency-set -g performance` 完成，也可以在基本输入/输出系统（Basic Input/Output System，BIOS）中完成。

同样地，禁用较深的 C-state 可以防止核心进入低功耗睡眠状态。CPU 的 C-state 是由系统 ACPI 规范定义的省电模式。当一个 CPU 核心空闲时，它可以进入某个 C-state 以节能。C-state 越深，节省的功耗越多，但当任务到来时核心唤醒所需的时间也可能越长。禁用较深的 C-state 可以消除过度的延迟尖峰。C0 表示活动状态；C0 以上的任何状态都代表更深层次的睡眠。

> 在实践中，许多服务器 BIOS/UEFI（统一可扩展固件接口，Unified Extensible Firmware Interface）都提供高性能配置文件，会自动将 CPU 调速器设为 “Performance” 并禁用较深的 C-state。

本质上，我们可以用略高一些的功耗换取更灵敏的 CPU 行为。在 GPU 才是耗电大户的训练场景中，只要能让 GPU 保持满负荷，CPU 多用一点功耗通常是可以接受的。例如，如果一个数据加载线程在等待数据时进入睡眠，而 CPU 进入了较深的 C6 状态，那么 CPU 的相当一部分会被断电以最大化节能。

如果 CPU 进入较深的睡眠状态，它可能需要几微秒才能唤醒。虽然这算不上很长的时间，但许多个微秒累加起来，如果管理不当就可能造成 GPU 气泡（bubble）。气泡是指 GPU 等待 CPU 恢复数据处理的时间段。通过让 CPU 保持就绪，我们减少了此类卡顿。许多服务器 BIOS 都有一个设置用来禁用 C-state——或至少限制它们。

> 你应始终关闭系统中任何可能引入不可预测延迟的东西，比如过多的上下文切换、CPU 频率缩放以及内存到磁盘的交换。其结果应当是：你的 CPU 能以 GPU 消耗数据的速度向 GPU 供给数据，而不会出现 OS 把任务调度到错误的核心上、或在错误的时刻抢走 CPU 周期的情况。

### 调优主机 CPU 内存分配器

在一台调优良好的 GPU 服务器上，由于 GPU 承担了大部分计算，CPU 的利用率可能并不很高。然而，CPU 利用率应保持稳定并与 GPU 活动同步。在当前批次正被 GPU 处理时，CPU 必须持续忙于准备每一个进来的批次。

恰当的 CPU 到 GPU 交接对于维持高 GPU 利用率至关重要。通过调优你主机的内存分配器（`jemalloc` 或 `tcmalloc`），你可以消除数据准备过程中不可预测的停顿。这将使 GPU 保持在峰值运行——除了有意为之的同步点之外。

调优之后，你应看到每个 GPU 的利用率都在接近 100% 处徘徊，只在必需的同步屏障处才会下降。GPU 绝不应因 CPU 侧的延迟而停下来等待数据。使用 `jemalloc`，你可以把分配分片到按 CPU 划分的 arena（`narenas`）中，启用 `background_thread` 进行离路径（off-path）清理，并延长 `dirty_decay_ms`/`muzzy_decay_ms`，从而使被释放的页面不会立即归还给 OS。这将把锁争用和碎片化降到最低。

你可以通过 `MALLOC_CONF` 环境变量来调优 `jemalloc`，如下所示：

```
export MALLOC_CONF="narenas:8,dirty_decay_ms:10000,muzzy_decay_ms:10000
,background_thread:true"
```

类似地，`tcmalloc` 也能从调优 `TCMALLOC_MAX_TOTAL_THREAD_` `CACHE_BYTES` 和 `TCMALLOC_RELEASE_RATE` 环境变量中获益。它们将提供更大的按线程缓存，使小额分配得以避开全局锁和系统调用——让 CPU 线程保持就绪，以低而可预测的延迟向 GPU 供给数据。你可以这样做：

```
export TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES=$((512*1024*1024))
export TCMALLOC_RELEASE_RATE=16
```

简而言之，优化分配器可以减少分配器开销和碎片化。这将使 CPU 线程始终保持快速，并避免向 GPU 供给数据时出现意外停顿。请针对你特定的工作负载和环境试验并调优这些环境变量。

## 面向性能的 GPU 驱动与运行时设置

我们已经优化了 CPU 侧，但 GPU 驱动和运行时也有一些重要设置会影响性能——尤其是在多 GPU 和多用户场景中。NVIDIA GPU 有一些旋钮，若调优得当，可以减少开销并改善多个工作负载共享 GPU 的方式。

接下来，我们将介绍 GPU 持久化模式、MPS 的分区、MIG，以及诸如时钟设置、ECC 内存和内存不足行为等其他一些考量。

### GPU 持久化模式

默认情况下，如果没有应用程序在使用某块 GPU，驱动可能会把该 GPU 置入更低功耗的状态，并卸载部分驱动上下文。下一次有应用程序出现并想使用该 GPU 时，就需要付出初始化它的代价。驱动把一切启动起来可能需要一两秒的量级。

对于会周期性地释放并重新获取 GPU 的工作负载，GPU 初始化开销会对性能产生负面影响。举例来说，设想一个作业频繁启停的训练集群。或者一个低流量的推理集群，每当有新的推理请求到来时都必须唤醒 GPU。在这两种情况下，该开销都会降低整体工作负载性能。

持久化模式（persistence mode）通过运行 `nvidia-persistenced` 守护进程来启用。它使 GPU 驱动保持加载、硬件保持就绪状态，即便没有应用程序处于活动状态也是如此。它请求系统在 GPU 空闲时不要将其完全断电，从而防止电源门控（power gating）。持久化让 GPU 保持唤醒，使下一个作业拥有零启动延迟。这对于长时间运行且延迟敏感的工作负载通常是推荐的。你可以使用以下命令在启动时启用持久化守护进程：

```
systemctl enable nvidia-persistenced
```

> 在 Kubernetes 环境中，可以配置 NVIDIA GPU Operator，使其在所有 GPU 上自动启用持久化模式。

在 AI 集群中，常见的做法就是在服务器启动时对所有 GPU 启用持久化模式。这样，当作业开始时，GPU 已经初始化完毕，可以立即开始处理。它不会让你实际的计算变得更快，因为它并不加速数学运算，但它削减了作业启动延迟并防止冷启动延迟。

GPU 持久化模式也有助于交互式使用，因为若没有持久化，你在空闲一段时间后发起的第一个 CUDA 调用可能会因驱动重新初始化 GPU 而卡住。开启持久化后，该调用会迅速返回。

持久化唯一的缺点是空闲时功耗略高，因为 GPU 保持在更高的就绪状态。但对大多数数据中心 GPU 而言，为了更好的性能一致性，这是一个可以接受的权衡。一旦具有 `sudo` 权限的管理员设置了 GPU 持久化模式，你就可以享受其好处，然后转而着手其他优化。

### MPS

通常，当多个进程共享单块 GPU 时，GPU 的调度器会在它们之间进行时间分片。例如，如果两个 Python 进程各自都有一些内核（kernel）要在同一块 GPU 上运行，GPU 可能会执行一个进程的内核，然后再执行另一个进程的内核，如此往复。如果这些内核很短且它们之间存在空闲间隙，GPU 最终可能会利用不足，因为它在做“乒乓式”的上下文切换，而没有把工作重叠起来。

NVIDIA 的 MPS 是这样一项特性：它创建了一种“伞”，让多个进程可以在 GPU 上并发运行而无需严格的时间分片。有了 MPS，只要 GPU 资源（流式多处理器 [streaming multiprocessors，SM]、Tensor Core 等）可用，GPU 就能同时执行来自不同进程的内核。MPS 本质上把这些进程的上下文合并到一个调度器上下文中。这样一来，你就不必为在各个独立进程之间切换和空转付出全部代价。

MPS 何时有用？对于模型训练，如果你通常是每块 GPU 运行一个进程，那你可能用不到 MPS。但如果你遇到诸如在一块大 GPU 上运行许多推理作业的场景，MPS 就是一个改变游戏规则的东西。设想你有一块强大的 GPU 或 GPU 集群，但你的推理作业——或一组多个推理作业——并没有把它完全用起来。

举例来说，设想在一块 40 GB 的 GPU 上运行四个独立的推理作业，每个使用 5–10 GB 且仅占用 30% 的 GPU 算力。默认情况下，每个推理作业获得一个时间片，因此在任一时刻，GPU 上实际运行的只有一个作业的工作。这使得 GPU 平均有 70% 处于空闲。

如果你为这些推理作业启用 MPS，GPU 就能交错它们的工作，这样在一个作业等待内存时，另一个作业的内核可能会填满 GPU，如此等等。其结果是整体 GPU 利用率更高。在实践中，如果两个进程各使用 GPU 的 40%，启用 MPS 后你可能会看到 GPU 以 80%–90% 的利用率同时服务这两者。

举例来说，两个各自单独运行需要一小时的训练进程——在同一块 GPU 上顺序运行——借助 MPS 可以一起运行，并在略超过一小时的总时间内并行完成，而不是顺序运行的两小时。举例来说，两个各自单独运行需要一小时的训练进程——在同一块 GPU 上顺序运行——可以借助 MPS 一起运行。在这种情况下，它们会在略超过一小时的总时间内并行完成，而不是顺序运行的两小时。当来自并发客户端的内核和内存带宽彼此互补时，MPS 带来的加速可以接近翻倍。为便于形象化理解，设想进程 A 和进程 B 在没有 MPS 的情况下各自周期性地启动内核。GPU 的调度可能看起来像 A-B-A-B，其间夹着各自等待时的间隙，如图 3-6 所示。

![Figure 3-6. GPU alternates between running Process A’s kernels and Process B’s kernels and creates idle gaps in which one process is waiting while the other is active](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-6.png)

_图 3-6. GPU 在运行进程 A 的内核和进程 B 的内核之间来回交替，制造出空闲间隙——其间一个进程在等待，而另一个处于活动状态。_

有了 MPS，调度会将 A 和 B 重叠起来，这样每当 A 没有在使用 GPU 的某些部分时，B 的工作就能同时使用它们，反之亦然。这种重叠消除了空闲间隙，如图 3-7 所示。

![Figure 3-7. Reducing idle gaps for processes A and B using MPS](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-7.png)

_图 3-7. 使用 MPS 减少进程 A 和 B 的空闲间隙。_

设置 MPS 需要运行一个 MPS 控制守护进程（`nvidia-cuda-mps-` `control`），它随后会启动一个 MPS 服务器进程来居中协调 GPU 访问。在现代 GPU 上，MPS 更为精简，因为客户端（即各进程）可以直接与硬件通信，而来自计算节点本身的干扰极小。

通常，你会在一个节点上启动 MPS 服务器——往往是每块 GPU 一个或每个用户一个——然后用一个把作业连接到 MPS 的环境变量来运行你的 GPU 作业。该服务器下的所有作业都将并发共享这块 GPU。

MPS 的另一项特性是能够为每个客户端设置活动线程百分比。这会限制一个客户端可以使用多少 SM（本质上就是 GPU 核心）。如果你想保证服务质量（quality of service，QoS）——例如让两个作业各自最多获得 GPU 执行资源的 50%——这会很有用。在这种情况下，你可以设置 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50`，将一个客户端限制在大约 50% 的 SM 执行能力上。如果没有显式设置，这些作业就会彼此竞争，能用到多少 GPU 资源就用多少。

请注意，MPS 并不对 GPU 内存进行分区，因此所有进程都将共享完整的 GPU 内存空间。MPS 主要关乎计算共享与调度。问题在于，某个进程可能请求海量的 GPU RAM，在 GPU 上引发 OOM 错误，进而导致在该 GPU 上运行的所有其他进程被终止。这极具破坏性。此外，如果某个程序单凭自身就把 GPU 打满到 100%，MPS 也不会神奇地让它更快，因为你无法超过 100% 的利用率。只有当各个作业留出一些余量供其他作业填补时，MPS 才有益处。

MPS 的另一个限制是，默认情况下，所有 MPS 客户端必须以同一个 Unix 用户身份运行，因为它们共享一个上下文。在多用户集群中，这意味着 MPS 通常在调度器层面设置，使得任一时刻只有一个用户的作业共享一块 GPU。否则，你可以配置一个供所有用户共享的系统级 MPS，但要明白这些作业从安全角度看是没有相互隔离的。

现代 NVIDIA 驱动支持多用户 MPS，使得来自不同 Unix 用户的进程可以共享单个 MPS 服务器。这提升了可用性，但并不提供内存隔离。当需要强隔离时，优先选择 MIG。MPS 的一个具体替代方案是 Kubernetes 中用于对 GPU 进行时间分片的一项特性。Kubernetes 上的时间分片允许设备插件按时间在同一块 GPU 上调度不同的 pod。例如，如果你把一块 GPU 配置为时间分片复制因子为四，那么该 GPU 上的四个 pod 就可以各自获得一份时间份额。

Kubernetes 时间分片算是一种不需要 MPS 的自动化分时算法。然而，它并不重叠执行。相反，它只是比默认驱动切换得更快。对于你宁愿以一些空闲时间为代价换取隔离的交互式工作负载，时间分片可能有用。对于高吞吐量作业，如接下来所讨论的，用 MPS 重叠执行、或用 MIG 切分 GPU，通常比细粒度的时间分片更好。

### MIG

借助 MIG，现代 GPU 可以在硬件层面被切分为多个实例。MIG 是一种虚拟化，但由硬件实现。因此，其开销非常低——也许只有百分之几——代价是损失了一部分灵活性。

如果某个实例处于空闲状态，它也无法把自己的资源借给另一个实例，因为它们是硬性分区的。MIG 允许把一块 GPU 切分成多达七个更小的逻辑 GPU——每个都拥有自己专属的一部分内存和计算单元（即 SM），如图 3-8 所示。

![Figure 3-8. Seven MIG slices on a modern GPU](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-8.png)

_图 3-8. 现代 GPU 上的七个 MIG 切片。_

按照惯例，NVIDIA 的 MIG 配置文件（profile）命名使用前缀 `<X>g` 来表示现代 GPU 上计算切片的数量，取值介于 1（最小）到 7（最大）之间。每个切片编号代表分配给该分区的若干 SM 组。每个 SM 组大致是全部 SM 总数的 1/7 切片。

如果一块 GPU 有 132 个 SM，那么每个 1/7 切片就代表一组 132 SM × 1/7 = ~19 个 SM。因此，`1g` 代表约 19 个 SM，`2g` 代表约 38 个 SM，一直到 7g，代表全部约 132 个 SM。

与之相对（也有些令人困惑）的是，后缀 `<Y>gb` 则指定为该配置文件预留的 HBM GPU 显存的确切大小（以 GB 为单位）。MIG 配置文件的取值对每一代、每种型号的 GPU 都是固定的，并在 NVIDIA 文档中列出。对于 Blackwell B200，部分 MIG 配置文件取值如表 3-1 所示。

表 3-1. Blackwell B200 的 MIG 配置文件（来源：https://oreil.ly/FsPEx）

该表还列出了每种配置文件（profile）的硬件单元数量、拷贝引擎（copy engine）数量以及 L2 缓存占比。这些固定的配置文件与 GPU 的硬件内存控制器对齐，使得每个显存切片都映射到连续的 HBM 通道。

这种两段式命名方案将计算容量（SM 组的数量）与显存容量（总 GB 数）区分开来。管理员可以选择多种 MIG 配置文件的组合。所分配的 SM 与 HBM 之和无需精确匹配整块 GPU 的容量。不过，某些组合会受到硬件分区方式的约束。例如，管理员无法凭空创造出新的切片规格。

管理员可以使用 `nvidia-smi -mig` 之类的工具，或使用 NVIDIA Kubernetes GPU Operator 的 `nvidia.com/mig.config` config map，在每块 GPU 上仅启用或禁用受支持的 MIG 配置文件（例如 `1g.23gb`、`2g.45gb`、`4g.90gb` 等）。重新配置 MIG 需要先腾空（drain）工作负载，再调用 MIG 的动态重配置能力来应用变更。

一旦 GPU 进入 MIG 模式，现代 GPU 便可动态地创建和销毁 MIG 分区，而无需重启整个系统。你可以在腾空现有工作负载后即时调整 MIG 实例，但要在某块 GPU 上启用或禁用 MIG 模式本身，则需要重置该 GPU。

从软件的视角看，每个 MIG 实例都像一块独立的 GPU，因为它拥有自己的显存、自己的 SM，甚至独立的引擎上下文。MIG 的好处在于为每个作业提供强隔离和有保障的资源。如果你有多个用户或多个服务，各自只需要（比如说）10 GB 的逻辑 GPU 显存，你就可以把它们打包到同一块物理 GPU 上，而它们之间不会互相干扰彼此的显存或计算。

作业可以专门请求 MIG 设备，但你必须谨慎调度，确保用满所有切片。举例来说，如果你配置了 7 个切片，而某个作业只占用 1 个切片，那么其余 6 个切片就应当被其他作业填满，否则你就会让大量资源闲置。例如，你可以把集群中的某些节点配置为使用 MIG 来运行小型推理作业，而把另一些节点配置为运行大型训练作业的非 MIG 工作负载。

一个重要的运维注意事项是：要使用 MIG，你通常需要在系统层面（至少在节点层面）进行配置。必须将 GPU 置入 MIG 模式、创建切片并重置 GPU。完成这些步骤后，这些切片会作为独立设备呈现给系统——每个切片都有自己唯一的设备 ID。

如果由于前期规划不当而没有用满所有可用的 MIG 切片，你就会因为让它们处于碎片化和未使用状态而浪费资源。重要的是要提前规划分区大小以匹配你的工作负载——并在工作负载发生变化时调整分区大小。你需要重置 GPU 才能使变更生效。

对于跨越多块 GPU 的大规模模型训练作业和推理服务器而言，MIG 通常并无用处，因为我们希望访问全部 GPU。另一方面，对于可以在较小 GPU 分区上运行的多租户小模型推理服务器而言，MIG 及其隔离特性可能会很有用。

> 在撰写本书时，当 GPU 处于 MIG 模式时，GPU 到 GPU 的点对点（peer-to-peer）通信（包括 NVLink）会被禁用。这同时适用于跨 GPU 的 NVLink 和 PCIe P2P。MIG 实例无法与其他 GPU 进行 P2P。跨 MIG 实例的 CUDA IPC 也受到限制。这会降低分布式训练的吞吐量。在启用 MIG 之前，请确认你的训练或推理拓扑不依赖 GPU 点对点路径。大规模训练作业（以及稀疏 MoE 专家推理系统）需要海量的 GPU 到 GPU 通信，通常不适合使用 MIG。GPU MIG 实例之间的通信必须经由主机或网络 fabric 传输。

简而言之，只有当你需要在同一块 GPU 上运行多个相互独立、且需要强隔离的作业时，才启用 MIG。不要在跨 GPU 的大规模分布式训练或推理中使用 MIG，因为在那种场景下你希望获得 GPU 的全部算力及其高速互连。

在我们讨论的大型基于 transformer 的模型训练与推理场景中，我们会关闭 MIG。不过，了解这个特性的存在是有益的。也许某个集群可以动态切换模式：白天有大量小型训练或推理实验运行时启用 MIG，夜间关闭 MIG 以运行使用整块 GPU 的大型训练作业。

> 对于 1/7 的 GPU 切片（该切片共 23 GB），Kubernetes 设备插件会将 MIG 设备列为形如 `nvidia.com/mig-1g.23gb` 的资源。

### GPU 时钟频率与 ECC

NVIDIA GPU 具有一项名为 GPU Boost 的功能，它会在功耗和热量限制范围内自动调整核心时钟。大多数情况下，你应该让 GPU 自行处理。但有些用户喜欢锁定时钟以获得一致性，让 GPU 始终运行在固定的最高频率下。这样一来，多次运行之间的性能就会保持稳定，不受功耗或温度波动的影响。

在进行基准测试时，固定时钟极为重要，因为后续的运行可能会因过热而被降频。如果不考虑这一点，你可能会错误地解读后续运行的糟糕结果，因为这些后续运行的 GPU 可能由于先前运行产生的过多热量而被降频。

具体来说，NVIDIA 的 GPU Boost 会上下调整核心时钟以保持在功耗/热量限制范围内。使用 `nvidia-smi -lgc` 将核心时钟锁定在最高稳定频率，并使用 `-ac` 锁定显存时钟。这样可以确保 GPU 以恒定频率运行——并防止 GPU 的 Boost 默认行为在后续运行中降低时钟频率。

> 这主要与基准测试相关，用于获得确定且可复现的结果。对于日常训练和推理，建议保持自动 Boost 开启，除非你发现明显的性能波动和 GPU 降频。

如果你在追求最后一点确定性和一致性，那么锁定时钟是需要留意的事情。不过通常来说，让 GPU 保持默认的自动 Boost 模式就足够了。

有些团队会有意为 GPU 降频以减少发热——尤其是当他们运行非常长的作业、又不想随时间推移最终遭遇热降频时。数据中心级 GPU 通常有足够的温度余量——以及适当的风冷和液冷——因此你无需这样做，但了解这也是一种可选方案是有益的。

另一种方法是使用 `nvidia-smi -pl` 将功耗上限设置为略低于 GPU 的最大热设计功耗（thermal design power，TDP）。TDP 是 GPU 在持续负载下所能产生的最大热量，以瓦特为单位度量。它决定了为防止过热而必须散发的热量。

如果你把功耗上限设置在 TDP 以下，GPU Boost 会自动将时钟调整到热降频点以下。这可以减少峰值发热、防止降频，同时对性能的影响微乎其微。

GPU 上的 ECC 显存是另一个需要考量的因素。ECC 可确保：如果发生了（例如由宇宙射线引起的）单比特内存错误，该内存可被即时纠正。而如果发生了双比特错误，该错误会被检测到，并向调用代码抛出错误。在 NVIDIA 数据中心级 GPU 上，ECC 通常默认启用。

禁用 ECC 可以释放少量显存，因为 ECC 需要额外的比特位来做错误校验。这可能会通过减少即时错误校验带来的开销而带来微小的性能提升，但通常只有几个百分点。然而，关闭 ECC 也会移除关键的内存错误保护，可能导致系统不稳定或未被察觉的数据损坏。

对于 NVIDIA 的数据中心级 GPU（包括 Hopper 和 Blackwell），ECC 默认启用，并且应保持启用，以确保可靠的、经过纠错的计算和数据完整性。对于在超大模型上运行的长时间训练或推理作业，一个内存错误就可能彻底使作业崩溃，或者更糟——在没有任何警告的情况下悄悄损坏你的模型。

建议在任何严肃的 AI 工作负载中始终保持 ECC 开启。唯一可能考虑关闭它的场合，是在研究环境中，你愿意承担这一风险，因为你需要那一点额外的显存才能让模型塞进你显存有限的 GPU 集群里。

切换 ECC 模式需要重置 GPU，并且很可能需要重启当前正在该 GPU 上运行的作业。因此这不是一个你想频繁切换的开关。为了稳定性和可靠性，请保持 ECC 开启。这份安心远胜于关闭 ECC 所带来的微不足道的加速。

### GPU 显存超额订阅、碎片化与内存不足处理

与 CPU 内存不同，默认情况下并不存在 GPU“交换（swap）”内存这种东西。如果你试图分配比可用显存更多的 GPU 显存，你会收到一个不友好的 OOM 错误，以及一个更不友好的进程崩溃。有几种机制可以缓解这一问题：允许显存动态增长、跨 CPU 与 GPU 采用统一内存，以及利用内存池和缓存分配器。

默认情况下，某些框架（例如 TensorFlow）会在启动时抢占所有可用的 GPU 显存，以避免碎片化并提升性能。如果你不了解这一点，在共享 GPU 的场景下可能会非常糟糕。而 PyTorch 默认只按需分配 GPU 显存。

> TensorFlow 提供了一个选项（`TF_FORCE_GPU_ALLOW_GROWTH=true`），让它从占用少量显存开始，

并按需动态增长 GPU 显存用量——与 PyTorch 类似。然而，无论是 PyTorch 还是 TensorFlow，都不允许你分配超过 GPU 实际拥有的显存。但这种惰性分配在多租户场景下表现得更友好，因为两个进程不会都从一开始就试图同时抢占最大可用的 GPU 显存。

CUDA 的统一内存（Unified Memory）系统让你无需预先定义内存驻留在 CPU 还是 GPU 上即可进行分配。CUDA 运行时会按需搬移页面。像 Hopper 和 Blackwell 这样的现代 NVIDIA GPU 内置了硬件支持，可通过页迁移引擎（Page Migration Engine，PME）实现按需分页。

当 GPU 的可用显存不足时，PME 会自动在 GPU 显存与主机 CPU 内存之间迁移内存页。然而，尽管 PME 提供了灵活性，但与为工作负载配备足够的 GPU 显存相比，依赖它可能会带来性能损失。

不过，这种 GPU 到 CPU 的内存卸载可能会很慢，因为正如我们在第 2 章所学，CPU 内存 I/O 慢于 GPU 高带宽内存（HBM）I/O。这一机制主要是为那些试图运行放不进 GPU 内存的模型的从业者提供便利。

对于性能关键型工作负载，你通常应尽可能避免依赖统一内存的超额订阅。它的存在是作为一张安全网，避免直接让脚本崩溃，但当 GPU 显存被超额订阅时，你的作业会运行得更慢。

像 PyTorch 这样的库使用缓存分配器（caching allocator），因此当你释放 GPU 显存时，它并不会立即将显存归还给操作系统。相反，它会保留这块显存以便将来复用。这可以避免内存碎片化，以及反复请求操作系统分配同一块内存所带来的开销。

你可以使用诸如 `PYTORCH_ALLOC_CONF`（原为 `PYTORCH_CUDA_ALLOC_CONF`）之类的环境变量来配置 PyTorch 的分配器，以设置最大内存池大小。我们将在后面的章节中介绍针对 PyTorch 显存分配机制的优化。

如果你遇到 GPU OOM 错误（你迟早一定会遇到），它很可能是由内存碎片化或过度的内存缓存引起的。你可以尝试使用 PyTorch 的 `torch.cuda.empty_cache()` 来清空缓存，但这几乎总是意味着你的工作负载确实需要那么多显存。

PyTorch 还提供了 `torch.cuda.memory_stats()` 和 `torch.cuda.memory_summary()` 之类的工具，通过显示已分配显存与已预留显存来帮助诊断碎片化问题。NVIDIA 的 Nsight Systems 也能展示 GPU 显存使用模式，帮助识别内存泄漏、与泄漏相关的长期存活分配、CPU-GPU 互连活动，以及 GPUDirect Storage 时间线追踪。此外，Nsight Compute 分析器提供低层次的内核分析，包括占用率（occupancy）、吞吐量和 NVLink 使用情况。我们将在后续章节中一一介绍这些内容。

Docker 提供了 `--gpus` 标志来选择并向容器暴露 GPU，但它不支持设置 GPU 显存上限。如果你需要对 GPU 显存或计算进行硬性隔离，请使用 MIG 来对设备分区，或使用带有活动线程百分比的多进程服务（MPS）来实现公平共享。当你需要严格分区时，可在 Kubernetes 中使用诸如 `nvidia.com/mig-2g.45gb` 之类的 MIG 资源来配置限制。

在多租户节点上，这种做法有助于隔离作业。而在每块 GPU 只运行单个作业的情况下，通常不会设置显存上限，因为你希望让作业尽可能多地占用 GPU 显存。

总体而言，GPU 显存耗尽是可以在应用层管理的问题。例如，你可以减小数据批大小、模型权重精度，甚至在可行的情况下减少模型参数量。

一项最佳实践是在模型训练和推理期间，使用 `nvidia-smi` 或 NVML API 监控 GPU 显存使用情况。如果你接近显存上限，可以考虑一些变通办法，例如减小批大小、在训练中使用激活检查点（activation checkpointing），或其他降低显存占用的技术。

此外，你应确保 CPU 内存没有被交换出去，因为这会间接损害你的 GPU 利用率和有效吞吐量（goodput）：每当 GPU 试图从 CPU 主机取回某些数据，而该主机内存页却已被交换到磁盘时，你的性能就会被慢得多的磁盘 I/O 所拖累。因此，将这些减少内存占用的最佳实践与前文关于固定内存、提高 `ulimit`、禁用 swappiness 等的建议结合起来非常重要。

简而言之，建议始终保持 GPU 驱动处于加载状态，而不要在作业之间卸载 GPU 驱动。这与 GPU 持久化模式（persistence mode）类似，但作用层次更深。有些集群被配置为在没有作业运行时卸载驱动，以释放操作系统内核内存并出于安全考虑。然而，如果你这样做，下一个作业就必须承担重新加载 GPU 驱动的开销，而且如果使用了 MIG，还要重新配置 MIG 切片。

> 建议让驱动和任何 MIG 配置在作业之间保持持久化。你唯一想要卸载 GPU 驱动的时机，是排查故障或升级驱动的时候。因此，集群管理员通常会把系统设置为：机器一旦启动，NVIDIA 驱动模块就始终存在。

## 面向 GPU 的容器运行时优化

许多 AI 系统使用编排工具和容器运行时来管理软件环境。Kubernetes 和 Docker 在 AI 基础设施中很受欢迎。使用容器可以确保所有依赖（包括 CUDA 和各种库的版本）保持一致。这就避免了“但它在我机器上能跑”的问题。容器会引入一点复杂性和极小的开销，但只要配置得当，你就能用容器为 GPU 工作负载获得接近裸机的性能。

运行在节点上的容器并不是传统意义上的虚拟机（virtual machine，VM）。与虚拟机不同，容器共享宿主机的操作系统内核，因此 CPU 和内存操作能以接近原生的速度执行。而借助 NVIDIA Container Toolkit，在 Docker 容器内部访问 GPU 是直接的，不会产生开销。

> 对于运行最新 NVIDIA Container Toolkit 的现代 GPU，在配置得当的环境中，容器内的 GPU 性能与在容器之外的裸机宿主上直接运行代码的性能几乎相同（差异 < 2%）。事实上，MLPerf Inference v5.0 的结果中就使用了 Red Hat OpenShift 和 Kubernetes，这表明现代容器和编排配置不会损害效率或延迟。

### NVIDIA Container Toolkit 与 CUDA 兼容性

在将容器与 GPU 一起使用时，一个挑战是确保容器内部的 CUDA 库与宿主机上的驱动相匹配。NVIDIA 通过其 Container Toolkit 和基础 Docker 镜像解决了这个问题。宿主机提供 NVIDIA 驱动——请记住，它与内核和硬件紧密集成。在容器内部，你通常会找到某个特定版本的 CUDA 运行时库。

一般规则是：宿主机的 NVIDIA 驱动版本必须至少与容器内 CUDA 版本所要求的最低驱动版本一样新。对于 CUDA 13.x，所需的最低 Linux 宿主机驱动分支为 R580 或更新版本。对于 CUDA 12.x，所需的最低 Linux 宿主机驱动分支为 R525 或更新版本。用较新的 CUDA 运行时搭配较旧的驱动，会导致 CUDA 初始化失败。

> 每个新的 CUDA 版本都要求一个最低的 NVIDIA 驱动版本。请始终查阅 NVIDIA 官方兼容性矩阵，并在更新 CUDA 工具包时升级宿主机驱动。

对于 Docker 和 Kubernetes 环境，最简单的方法是使用来自 NVIDIA GPU Cloud（NGC）或 DockerHub 镜像仓库的 NVIDIA 官方基础 Docker 镜像。这些镜像（例如 `nvcr.io/nvidia/pytorch` 或类似镜像）打包了正确版本的 CUDA 运行时、cuDNN、NCCL 等。此外，这些 Docker 镜像会根据 CUDA 版本列出所需的最低 CUDA 驱动。这样，你无需为依赖问题头疼，就能获得对最新硬件的支持。

### NVIDIA Container Runtime

另一种方案是，NVIDIA 的容器运行时实际上可以在运行时将宿主机的驱动库注入到容器中，因此你甚至无需在镜像内部附带 NVIDIA 驱动。相反，你只需依赖宿主机的驱动。同样，这之所以可行，是因为容器并不像传统虚拟机那样完全隔离。Docker 容器被允许使用宿主机的设备、卷和库。

在容器内部，你的应用使用来自容器镜像的 CUDA 运行时库（例如 `libcudart.so`），而 NVIDIA Container Toolkit 会在容器启动时注入宿主机的驱动库，例如 `libcuda.so` 和 `libnvidia-ml.so`。宿主机的驱动库直接在宿主机上被调用，因此一切都能正常工作。

只要宿主机驱动满足镜像中 CUDA 工具包所要求的最低版本，CUDA 运行时库（在容器中）与 NVIDIA Container Toolkit（在宿主机上）之间的这种拆分就是受支持的。如果你版本不匹配，试图在容器中使用较新的 CUDA 版本、而宿主机上却是较旧的驱动，你很可能会遇到错误。匹配 CUDA 与驱动版本非常重要。

关键要点是：将容器用于 GPU 时，并不涉及任何 hypervisor 或虚拟化层。容器直接共享宿主机的内核和驱动，因此当一个内核在 GPU 上启动时，就仿佛是从宿主机启动的一样。

换句话说，你不会因为基于 Docker 的虚拟化而损失性能——除非你使用了诸如 VMware 或单根输入/输出虚拟化（Single Root Input/Output Virtualization，SR-IOV）虚拟 GPU 之类的东西，那是一种需要一些调优的特殊场景。有了 Docker 加 NVIDIA，其性能基本等同于裸机。

> NVIDIA Container Toolkit 不仅适用于 Docker，也适用于 containerd 和 Podman。这一点与使用 containerd 作为默认容器运行时的现代 Kubernetes 环境相关。

### 避免容器叠加文件系统的开销

在 Docker 容器中运行与直接在宿主机上运行的主要差异，可能在于 I/O。容器通常使用联合文件系统（union filesystem），它以透明的方式将多个底层文件系统（如宿主机文件系统和容器文件系统）叠加成单一的、统一的视图。

在诸如 OverlayFS 这样的联合文件系统中，来自多个来源的文件和目录看起来就像属于同一个文件系统。这一机制对容器尤其有用：来自基础镜像层的只读文件系统与一个可写的容器层结合在一起。

不过，使用叠加文件系统（overlay filesystem）会有一些开销。这种额外延迟产生的原因是：文件系统必须检查多个底层的层——既包括只读层也包括可写层——以确定应返回文件的哪个版本。与从单一、简单的文件系统读取相比，额外的元数据查找以及合并这些层的逻辑会带来少量开销。

此外，向叠加所使用的写时复制（copy-on-write，CoW）机制写入时也会有开销。CoW 意味着，当你修改只读层（例如基础镜像）中的某个文件时，必须先将该文件复制到可写层。写入随后发生在这个被复制到可写层的文件上——而不是原始的只读文件上。如前所述，读取一个被修改过的文件时，需要同时查看只读层和可写层，以确定应返回哪个才是正确的版本。

模型训练在读取数据集、加载模型和写入模型检查点时，往往涉及繁重的 I/O 操作。为了绕过这一问题，你可以使用绑定挂载（bind mount）将宿主机目录——或网络文件系统——挂载进容器。

绑定挂载会绕过叠加层，因此其性能与直接在宿主机上进行磁盘 I/O 相近。如果宿主机文件系统是诸如 NVMe SSD 或 NFS 挂载之类的存储，你就能获得该底层存储设备的完整性能。我们特意不把多 TB 级的数据集打包进镜像。相反，我们通过挂载把数据引入进来。

例如，如果你的训练数据位于宿主机的 `/data/dataset`，你会运行

> 带 `-v /data/dataset:/mnt/dataset:ro` 的容器，其中 ro 表示只读

挂载。然后你的训练脚本从 `/mnt/dataset` 读取数据。这样，你就是直接从宿主机文件系统读取。

事实上，避免对容器的可写层进行繁重的数据读写是一项最佳实践。相反，应将你的数据目录和输出目录从宿主机挂载进容器。你要确保 I/O 不会被容器 CoW 机制的开销所拖累。

### 缩减镜像大小以加快容器启动

如果镜像非常庞大且需要通过网络拉取，容器的启动时间可能会慢上不少。但在典型的长时间运行的训练循环中，与数小时、数天乃至数月的训练时间相比，几分钟的启动时间可以忽略不计。尽管如此，通过不包含不必要的构建工具或临时构建文件来让镜像保持合理精简，仍然是值得的。这既节省磁盘空间，又能改善容器启动时间。

一些 HPC 中心更青睐 Singularity（Apptainer）而非 Docker，因为它可以在用户空间中运行镜像，而不需要 root 守护进程。它还直接使用宿主机文件系统，往往除了操作系统本身已有的开销之外，几乎没有额外开销。

无论是 Docker 还是 Apptainer（原 Singularity），研究和基准测试都表明，一旦配置得当，无论是在容器中还是直接在宿主机上运行，这些容器方案测得的差异都只有几个百分点。本质上讲，如果有人给你一份 GPU 利用率和吞吐量的日志，你单凭该日志几乎无法分辨作业是否运行在容器中。

## 用 Kubernetes 实现拓扑感知的容器编排与网络

Kubernetes（也称为 K8s）是一款用于 AI 训练和推理的流行容器编排器。用于 Kubernetes 的 NVIDIA 设备插件是一个轻量级组件，它向调度器通告 GPU 硬件（`/dev/nvidia0`、`/dev/nvidiactl` 等）。当你在 `resources.limits` 下（如果你想同时显式设置，也可以在 `resources.requests` 下）请求 `nvidia.com/gpu` 时，它会把这些设备节点挂载进你的 Pod。这样，当你在 Kubernetes 上部署带有该设备插件的容器时，Kubernetes 就会负责把 GPU 提供给容器。该设备插件也是拓扑感知的。这意味着它可以倾向于为某个给定的 Pod 分配来自同一个 NVLink Switch 或同一个 NUMA 节点的多块 GPU。

NVIDIA Kubernetes GPU Operator 自动化了所有 NVIDIA 软件的安装和生命周期管理，包括驱动库、前文提到的 NVIDIA Kubernetes 设备插件以及 NVIDIA Container Toolkit。它还负责使用 NVIDIA 的 GPU Feature Discovery 进行节点标注，为每块 GPU 打上其 NUMA 节点和 NVLink/NVSwitch ID 的标签。随后，调度器便可以利用这些标签智能地把 GPU 分配给作业。GPU Operator 还使用 DCGM 实现 GPU 监控。

在使用 Kubernetes 编排基于 GPU 的容器时，你希望它以一种能感知硬件拓扑（包括 NUMA 节点和网络带宽配置）的方式为容器分配资源。然而，默认情况下 Kubernetes 并不具备拓扑感知能力。它把每块 GPU 视为一种资源，却不知道 GPU 0 和 GPU 1 是否位于同一个 NUMA 节点，或它们是否使用同一条 NVLink 互连。这可能会造成很大的差异。

设想一台配有两组各 4 块 GPU 的 8-GPU 服务器——每组内部由 NVLink 连接。如果你向 Kubernetes 为某个作业请求 4 块 GPU，理想情况下 K8s 应该给你四块全部通过 NVLink 互连的 GPU，因为它们能更快地共享数据。然而，如果 Kubernetes 随意挑选了分散在系统各处的四块 GPU，你的作业可能会被分配到来自一个 NVLink 域的两块 GPU 和来自另一个 NVLink 域的两块 GPU。

在不考虑拓扑的情况下分配 GPU，会把慢速互连（例如 InfiniBand 或 Ethernet）引入 GPU 到 GPU 的路径中。这可能会把你的 GPU 间带宽砍掉一半。在这种情况下，理想的做法是让 Kubernetes 分配四块全部由 NVLink 互连、且使用同一个 NUMA 节点的 GPU，而不是四块横跨不同机架和 NUMA 域的 GPU。

举例来说，考虑 NVIDIA 基于 NVLink 5 互连构建的 NVL72 机架，它将 72 块 GPU 连接成一个单一的高带宽域，机架内部的合计吞吐量约为 130 TB/s（72 块 GPU × 每块 1.8 TB/s）。在这种配置中，如果 Kubernetes 调度器不具备拓扑感知能力，它可能会把一个多 GPU 作业放置在不同的 NVLink 域之间——甚至放到 NVL72 组之外。在不尊重系统拓扑的情况下把 GPU 分配给作业，会抵消 NVL72 巨大的机架内带宽所带来的好处。

为了避免资源争用，你应该尽量要么预留你所需的资源，要么为你的作业请求整个节点。对于容器/Pod 的放置，你应使用 Kubernetes Topology Manager 组件，将 Pod 与 CPU 亲和性和 NUMA 节点对齐，把容器的 CPU 绑定到与该容器所分配 GPU 相同的 NUMA 节点上。接下来我们就讨论这一点。

### 使用 Kubernetes Topology Manager 编排容器

Kubernetes Topology Manager 可以提供详细的拓扑信息。例如，它可以检测到 GPU 0 连接到 NUMA 节点 0、NVLink 域 A 和 PCIe 总线 Z。随后 Kubernetes 调度器便可利用这些信息，以一种利于高效处理和通信的最优方式，把容器分配到 GPU 上。

拓扑感知的 GPU 调度仍在不断成熟。在许多集群中，管理员会使用 Kubernetes 标签显式地标注节点，以刻画 GPU 和系统的拓扑。这些标签确保多 GPU 的 Pod 落到那些 GPU 共享同一条 NVLink 互连、或位于同一个 NUMA 域内的服务器上。

就我们的目的而言，如果你在 Kubernetes 中运行多 GPU 作业，请务必启用拓扑感知调度。这通常需要将 `--topology-manager-policy` 配置为 `best-effort`、`restricted`，或在某些情况下配置为 `single-numa-node`。这种策略配置有助于多 GPU 以及 CPU + GPU 的工作负载通过避免远程内存访问来降低延迟。它与操作系统层面的 NUMA 调优相辅相成。

另外，务必使用上一节提到的最新 NVIDIA GPU 设备插件和 NVIDIA Kubernetes GPU Operator，因为它们是拓扑感知的，并支持把多 GPU 的 Pod 打包到连接同一个 NUMA 节点的 GPU 上。它们通过最小化跨 NUMA 节点通信、并降低多节点 GPU 工作负载中的延迟，来帮助优化性能。

在 NVLink-5 NVL72 系统上，单个机架级的 NVLink 域可提供高达 130 TB/s 的 GPU 到 GPU 双向聚合带宽，相当于每块 GPU 约 1.8 TB/s。在调度以集合通信为主的训练时，应优先选择能让流量在跨越较慢的网络 fabric 之前，先留在高速 NVLink 域内部的放置方案。

### 使用 Kubernetes 与 SLURM 进行作业调度

在多节点部署中，作业调度器对于最大化所有节点上的资源利用率至关重要。通常，训练集群使用简单 Linux 资源管理工具（Simple Linux Utility for Resource Management，SLURM），而推理集群则通常偏爱 Kubernetes。不过，已经出现了将 SLURM 与 Kubernetes 集成的混合方案。开源的 Slinky 项目就是一个示例方案，用于简化跨训练与推理工作负载的集群管理。

这些系统负责把 GPU 分配给作业，并协调跨节点的进程启动。如果一个训练作业请求 8 个节点、每节点 8 块 GPU，调度器会识别出符合条件的节点，并使用诸如 `mpirun` 之类的工具或诸如 Docker 之类的容器运行时来启动作业。这样，每个进程都能感知到该作业中所有可用的 GPU。许多集群还依赖经过充分测试的 Docker 仓库（如 NVIDIA 的 NGC Docker 仓库），以保证所有节点上有一致的软件环境——包括 GPU 驱动、CUDA 工具包、PyTorch 库和其他 Python 包。

在 SLURM 上也存在类似的问题。SLURM 有针对 GPU 的“通用资源（generic resources）”概念，你可以定义某些 GPU 挂接到某些 NUMA 节点或 NVLink/NVSwitch 上。然后在你的作业请求中，你就可以申请那些（比如说）连接到同一个 NUMA 节点的 GPU。

如果没有正确设置，调度器可能会把所有 GPU 都当作完全相同，从而为你的多 GPU 容器请求给出并不理想的分配。恰当的配置可以避免不必要的跨 NUMA 节点、跨 NVLink 的 GPU 通信开销。

SLURM 同样支持把 MIG 分区作为独立资源来调度。这对于把多个作业打包到同一块 GPU 上很有用，其原理类似于 Kubernetes 通过 Kubernetes 设备插件调度 GPU 切片。接下来，我们将讨论如何在 Kubernetes 中使用 MIG 切片。

### 用 MIG 切分 GPU

当你启用前文介绍过的 NVIDIA MIG 模式时，单块物理 GPU 会被切分成更小、固定且硬件隔离的分区，称为 MIG 实例。下面是一个 Kubernetes pod 配置示例，申请两个 `nvidia.com/mig-2g.45gb` MIG 切片（该配置假定 NVIDIA Kubernetes 设备插件已配置为能够识别每个节点上的 MIG 设备）：

```
resources:
  limits:
    nvidia.com/mig-2g.45gb: "2"
```

这里，该配置要求在某个节点上运行一个 pod，且该节点的一块 GPU 上至少有两个空闲的 `2g` `.45gb` 实例；换句话说，需要 2 个切片，每个切片占 2/7 的 SM（`2g`）。如果一块 GPU 总共有 132 个 SM，那么每个切片为 2/7 × 132 SM = ~38 SM。乘以 2，pod 总共分配到约 76 个 SM。总显存分配为 45 GB 的 GPU 显存。

请注意，调度器无法把这些切片跨 GPU 或跨节点拆分。只有当某个单一节点能够同时提供这两个分区时，Kubernetes 才会调度该 pod。这是因为 pod 不能跨越多个节点。如果没有任何单一节点拥有两个空闲的 `2g.45gb` 切片（即前面算出的总共 76 个 SM 和 45 GB GPU 显存），该 pod 就会一直停留在 Kubernetes 的 `Pending`（未调度）状态，因而无法运行——即便其他节点合计起来拥有足够的 MIG 容量也是如此。

这一约束凸显了根据典型工作负载需求来规划 MIG 尺寸的重要性。例如，如果很多作业都申请 `2g.45gb` 切片，你或许应当把每块 GPU 配置为承载三个 `2g.45gb` 实例——在其可能的七个切片之中——这样一来，就能有两个这样的实例共同驻留在同一块 GPU 上，供单个 pod 使用。

> 这一单节点约束可能导致 pod 永远无法运行——即使把集群中不同节点上的 MIG 资源合并起来能够满足需求也无济于事。只有当所请求的 MIG 资源在单一节点上可用时，请求才能被满足。

MIG 在运维上的一个缺点是：在 MIG 模式与普通（非 MIG）模式之间切换某块 GPU 需要重置该 GPU——或重启计算节点。因此，调度器难以按作业动态地完成这一切换。不过，通常你会提前创建好 MIG 分区，并让该配置持续运行一段时间。

在 Kubernetes 环境中，NVIDIA Kubernetes GPU Operator 的 MIG Manager 可以自动配置并保留各节点上的 MIG 分区。这样，MIG 切片就能在重启和驱动重新加载后依然保持有效。

你可以给一个 K8s 节点打上“mig-enabled”标签，给另一个打上“mig-disabled”标签，让调度器据此放置作业/pod。这更多是一个运维细节，但值得了解的是：MIG 是真正的静态分区——而非动态调度器的产物。

> 在使用 MIG 时，建议启用持久化模式（persistence mode），这样即使没有作业在运行，MIG 配置也能在 GPU 上保持有效。如此一来，GPU 就不必在每次运行周期性作业之前反复重建切片。

### 优化 Kubernetes 的网络通信

当你用 Kubernetes 上的容器运行多节点 GPU 工作负载时，这些 pod 之间需要相互通信。在 Kubernetes 中，默认情况下每个 pod 都有自己的 IP，不同节点上的 pod 之间可能存在叠加网络（overlay）或网络地址转换（network-address translation，NAT）。这会带来一些复杂性和额外开销。

对于 GPU 集群，往往最简单的方案是为这些性能敏感的作业使用主机网络（host networking）。这意味着容器的网络不再被隔离，而是直接使用主机的网络接口。要在 Kubernetes 中启用这一点，你需要在 pod 规格中设置 `host` `Network: true`。在 Docker 中，你可以这样运行：

> `--network=host`。

使用主机网络能让容器完全像主机一样访问 InfiniBand 互连——没有任何额外的转换层或防火墙层。这对 MPI 作业尤其有用，因为它省去了为每个 MPI rank 配置端口映射的麻烦。

然而，如果由于安全策略而无法使用主机网络，你就必须确保你的 Kubernetes 容器网络接口（container network interface，CNI）以及任何叠加网络都能处理所需的流量。在这种情况下，你可能需要开放特定端口以支持 NCCL 的握手与数据交换，并借助 `NCCL_PORT_RANGE` 和 `NCCL_SOCKET_IFNAME` 等环境变量来帮助建立连接。

在叠加网络上运行时，至关重要的是保持低延迟并让操作在内核空间中运行。同时，要确保没有任何用户空间代理限制节点间的流量。这些因素都会显著影响性能。

在 Kubernetes 环境中若想启用 RDMA，可以考虑安装 Mellanox 提供的 Kubernetes RDMA 设备插件。该插件会在 pod 接口上暴露 InfiniBand 和 GPUDirect RDMA 端点，从而实现低延迟、零拷贝的网络通信。

> 如果你使用 InfiniBand 或 RoCE 网络，请记得在 NVIDIA 驱动中启用 GPUDirect RDMA（前提是你的 NIC 支持）。这样 GPU 就能直接与 NIC 交换数据——在节点间通信中绕过 CPU。这对于在多节点环境中维持高性能至关重要。

### 降低 Kubernetes 编排抖动

运行像 Kubernetes 这样的编排器意味着每个节点上都会有一些后台进程在运行（例如 Kubernetes 的“kubelet”）、容器运行时守护进程，以及（最好还有）监控代理。虽然这些服务会消耗 CPU 和内存，但其消耗量约为单个核心的百分之几。因此，它们不会从基于 GPU 的训练作业中偷走明显的时间，而训练作业正是用这些核心来做数据加载和预处理的。

然而，如果训练作业运行在一个同时也在运行推理工作负载的节点上，你可能会在执行时序和吞吐量上遇到一些抖动（jitter），即不可预测的波动。不过，这在任何多租户场景中都很常见。如果同一台机器上的另一个容器意外地占用了大量 CPU 或 I/O，它就会通过争抢相同资源来影响你的容器——无论是训练还是推理。

> 从系统的角度来看，同构工作负载（例如全部为训练或全部为推理）比训练与推理混杂的异构组合要容易调试和调优得多。

### 改进资源保障

为了防范资源争用，Kubernetes 允许你为 pod 定义资源请求（requests）和限制（limits）。例如，你可以指定训练作业需要 16 个 CPU 核心和 64 GB 内存。随后，Kubernetes 会把这些资源专门预留给你的作业，并避免在相同的 CPU 上调度其他 pod。

这些限制通过 Linux cgroups 强制执行，因此如果你的容器超出其分配额度，它可能会被限流，甚至被 OOM killer 终止。常见的做法是使用资源请求——并可选地使用 CPU Manager 功能来绑定核心——以确保性能关键的作业能够独占所需的 CPU 资源，从而使其他进程无法从你预留的核心中偷走 CPU 时间。

抖动的另一个来源是后台内核线程和中断，正如我们在第 2 章讨论中断请求（IRQ）亲和性时所提到的。与 Kubernetes 的情况类似，如果其他 pod 与你的作业使用相同的网络或磁盘，这些 pod 可能会在承载你作业的计算节点上引发大量中断和额外的内核工作。这会造成抖动，并影响你作业的性能。

在理想情况下，一个 GPU 节点应完全专用于你的作业。但如果做不到，你就应确保使用 Linux cgroup 控制器对该节点的 I/O 和 CPU 进行细致的分区，以免其他工作负载造成干扰。

好在 Kubernetes 支持 CPU 隔离，它能确保 pod 获得其所请求的专用 CPU 核心和内存——并防止其他 pod 被调度到与你相同的 CPU 核心上。这样就避免了上下文切换和资源争用带来的额外开销。

> 在实践中，性能敏感的 Kubernetes 作业应当请求某个节点的全部 CPU 和 GPU，以免任何其他东西干扰或争抢该作业的资源。这说起来容易做起来难，但从性能与一致性的角度看，这是最理想的作业配置。

### 内存隔离与避免 OOM Killer

如果不加以恰当限制，内存干扰同样会发生。Kubernetes 提供一流的内存隔离支持（通过 Linux cgroups）。然而，一个贪婪的容器如果不受约束，就可能在主机上分配过多内存，进而导致主机把部分内存交换到磁盘。

如果一个不受限制的容器在主机上使用了过多内存，臭名昭著的 Linux “OOM killer” 就会开始杀死进程——甚至可能杀掉你的 Kubernetes 作业——即便占用过多内存的并不是你的作业。

OOM killer 在决定杀掉哪些 pod 时会使用启发式规则。有时它会决定杀掉正在运行的最大的 pod，而那很可能就是你那个在 CPU 内存中持有大量数据以喂给 GPU 的大型训练或推理作业。为避免这种情况，你可以有意不给训练或推理容器设置严格的内存限制。这样，它们在需要时就能使用所有可用内存。

借助恰当的监控与告警，你可以确保作业不会试图超出你预期地过度分配。如果你确实要设置内存限制，请确保它高于你实际预期的用量。这样能留出一点余量，避免在长时间运行的训练作业进行到第三天时被 OOM killer 杀掉。

> 在 Kubernetes 中，没有设置请求/限制的 Pod 会被视为 `BestEffort`，最容易被驱逐。要获得 `Guaranteed` QoS，每个容器都必须为 CPU 和内存都设置 `requests == limits`。仅仅设置一个较高的限制只会得到 `Burstable` QoS，而非 `Guaranteed`。

### 应对 I/O 隔离

遗憾的是，截至本文撰写时，Kubernetes 并未开箱即用地提供原生的一流 I/O 隔离。虽然 Linux 确实支持通过 cgroup 控制器进行 I/O 控制，但 Kubernetes 本身并不会像对待 CPU 和内存那样自动强制执行 I/O 限制。

如果你需要确保 GPU 节点上的重 I/O 工作负载彼此不相互干扰，你可能需要在节点层面手动配置 I/O 控制。这可能涉及调整 cgroup v2 的 I/O 控制器，或使用其他 OS 层面的配置来分区 I/O 资源。简而言之，尽管 Kubernetes 通过调度和资源请求来防止 CPU 争用，但 I/O 隔离通常需要对底层 Linux 系统进行额外的手动调优。

需要重点指出的是，在容器内部，某些系统设置是从主机继承而来的。例如，如果主机把 CPU 频率调节设为性能模式（performance mode），容器就会继承该设置。但如果容器运行在诸如云实例这样的虚拟化环境中，你或许无法更改这些设置。

始终确保对主机进行调优是个好主意，因为容器无法更改诸如大页设置或 CPU governor 限制之类的内核参数。通常，集群管理员会通过基础 OS 镜像来设置这些参数和配置。或者，在 Kubernetes 环境中，他们可能会使用类似 NVIDIA GPU Operator 的工具，在每个节点上设置持久化模式以及其他 `sysctl` 旋钮。

## 关键要点

以下是本章的一系列关键要点，涵盖了操作系统、驱动、GPU、CPU 与容器各层的优化：

- **数据与计算的局部性至关重要。** 确保数据的存储与处理尽可能靠近计算单元。使用本地高速存储（如 NVMe SSD 缓存）来最小化延迟，并减少对远程文件系统或网络 I/O 的依赖。

- **实施 NUMA 感知的配置和 CPU 亲和性。** 通过让进程和内存分配对齐到同一个 NUMA 节点内，来优化 CPU 到 GPU 的数据流。用 `numactl` 和 `taskset` 之类的工具绑定 CPU，可防止跨节点内存访问。这将带来更低的延迟和更高的吞吐量。

- **最大化 GPU 驱动与运行时的效率。** 精细调优 GPU 驱动设置，例如启用持久化模式以让 GPU 保持就绪状态。可考虑诸如多进程服务（Multi-Process Service，MPS）之类的功能，以便在单块 GPU 上重叠来自多个进程的工作。对于多租户环境，可探索使用 MIG 分区来有效隔离工作负载。

- **高效地预取与批处理数据。** 通过提前预取数据、并把小的 I/O 操作批处理成更大、更高效的读取，来持续喂饱 GPU。可利用诸如 PyTorch DataLoader 的 `prefetch_factor`（配合 `num_workers`）之类的预取机制，提前加载多个批次。

- **在数据加载时固定内存。** 把数据预取与内存固定结合起来，通过 PyTorch DataLoader 的 `pin_memory=True` 使用固定的 CPU 内存（页锁定、不可交换到磁盘），实现向 GPU 更快、异步的数据传输。如此一来，数据加载与模型执行便可重叠，空闲时间得以减少，CPU 与 GPU 资源都能被持续利用。

- **优化内存传输。** 利用固定的页锁定内存以及大页等技术，来加速主机与 GPU 之间的数据传输。这有助于降低拷贝开销，并让异步传输与计算重叠。

- **让通信与计算重叠。** 通过把梯度同步、数据暂存等内存操作与正在进行的 GPU 计算重叠起来，减少等待数据传输的时间。这种重叠有助于维持高 GPU 利用率和更好的整体系统效率。

- **调优并扩展网络栈。** 在多节点环境中，使用支持 RDMA 的网络（如 InfiniBand/以太网），并调优 TCP buffers、MTU、中断亲和性等网络设置，以在分布式训练和推理期间维持高吞吐量。

- **使用容器化与编排来保证一致性。** 使用诸如搭配 NVIDIA Container Toolkit 的 Docker 之类的容器运行时，以及诸如搭配 NVIDIA GPU Operator 和设备插件的 Kubernetes 之类的编排平台，使整个软件栈——包括驱动、CUDA 库和应用代码——在各节点间保持一致。这些方案有助于对齐 CPU–GPU 亲和性，并基于硬件拓扑管理资源分配。

- **消除容器运行时开销。** 虽然容器提升了可复现性和部署的便利性，但要确保 CPU 与 GPU 亲和性、主机网络以及资源隔离都得到正确配置，以将任何容器开销降到最低。

- **采用编排与调度的最佳实践。** 像 Kubernetes 这样健壮的容器编排器是确保高效资源分配的关键组件。高级调度技术——例如 Kubernetes 拓扑管理器（Topology Manager）——有助于确保具有高速互连的 GPU 被聚拢在一起。

- **通过动态适应与扩展来力求灵活。** 编排层负责分发工作，并在各节点间动态管理工作负载的切分。这种灵活性对于扩大训练任务规模，以及在数据负载和请求模式差异巨大的推理场景中确保高效运行，都至关重要。

持续且渐进地调优 系统级优化并非一劳永逸。要定期监控性能指标；随着工作负载的演变而调整 CPU 亲和性、批大小和预取设置；并把这些小的改进累积起来，以获得可观的性能提升。

在整个栈中减少瓶颈 最终目标是确保所有组件——从操作系统和 CPU，到 GPU 驱动和运行时——都能协调一致地工作。消除某一层中的瓶颈（例如 CPU 内存分配或驱动初始化），就能释放 GPU 的全部潜力，而这将直接转化为更快的训练、更低的成本和更高效的资源使用。

这些策略共同作用，最大限度地减少数据传输的摩擦、缩短等待时间，并确保你的硬件被充分利用，以实现高效的训练和推理。

## 结论

本章表明，即便是最先进的 GPU，也可能因其周边环境中的低效而受到掣肘。一个调优良好的操作系统、容器运行时、集群编排器和软件栈，构成了高性能 AI 系统的骨架。通过 NUMA 感知的绑定与本地存储方案让数据与计算对齐、让通信与计算重叠，并对主机系统和 GPU 驱动都进行精细调优，你就能降低延迟、提升吞吐量。

不妨把你的整个系统想象成一辆精密工程打造的跑车，其中每个组件（CPU、内存、GPU、网络、容器、编排器和编程栈）都必须无缝协作，才能交付最大性能。诸如启用持久化模式或优化 CPU 调度之类的小调整，单独看似乎微不足道，但当它们被组合起来并在大型 GPU 集群上扩展开来时，就能在时间和成本上带来可观的节省。这些优化确保 GPU 在训练大规模 transformer 模型和运行复杂推理管线时，始终运行在接近峰值效率的状态。

随着这一领域的演进和模型的持续增长，系统级调优的重要性只会与日俱增。本章讨论的技术让性能工程师和系统架构师能够榨取硬件的每一分潜力。这带来了更快的迭代周期和更具成本效益的 AI 部署。归根结底，一个深度优化的系统能够加速研究，并让前沿 AI 应用惠及更广泛的受众。

最后请记住，尽管硬件与软件栈看起来像是一大堆无从管理、相互关联的旋钮和开关，但小的调整却能转化为时间和成本上的显著节省。通过持续监控性能指标并对栈的每一层进行渐进式的精细调优，你就能把潜在的瓶颈转化为提升效率的机会。让数据来引导你，你就能释放 AI 系统的全部潜力。
