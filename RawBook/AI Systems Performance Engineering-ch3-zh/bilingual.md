# Chapter 3. OS, Docker, and Kubernetes Tuning for GPU-Based Environments

# 第 3 章 面向 GPU 环境的操作系统、Docker 与 Kubernetes 调优

Even with highly optimized GPU code and libraries, system-level bottlenecks can limit performance in large-scale AI training. The fastest GPU is only as good as the environment feeding it data and instructions. In this chapter, we explore how to tune the operating system and container runtime to let GPUs reach their full potential.

即便拥有高度优化的 GPU 代码与库，系统层面的瓶颈仍可能在大规模 AI 训练中拖累性能。最快的 GPU，也只能发挥出喂给它数据与指令的环境所允许的水平。本章将探讨如何调优操作系统与容器运行时，让 GPU 得以充分释放潜能。

We begin by exploring the foundational GPU software stack. We then dive into key CPU and memory optimizations such as NUMA affinity and hugepages. These ensure that data flows efficiently from storage through the CPU to the GPU. In parallel, we discuss critical GPU driver settings like persistence mode, Multi-Process Service (MPS), and Multi-Instance GPU (MIG) partitions. These help maintain maximum GPU utilization by reducing overhead and synchronizing resources effectively.

我们首先探讨作为基础的 GPU 软件栈，随后深入若干关键的 CPU 与内存优化，例如 NUMA 亲和性（affinity）与大页（hugepages）。这些手段能确保数据从存储经由 CPU 高效流向 GPU。与此同时，我们还会讨论一些至关重要的 GPU 驱动设置，如持久化模式（persistence mode）、多进程服务（Multi-Process Service，MPS）以及多实例 GPU（Multi-Instance GPU，MIG）切分。它们通过降低开销、高效同步资源来帮助维持 GPU 的最大利用率。

Using solutions like the NVIDIA Container Toolkit, Container Runtime, Kubernetes Topology Manager, and Kubernetes GPU Operator, you can create a unified and highly optimized software stack for GPU environments. These solutions enable efficient resource allocation and workload scheduling across single-node and multinode GPU environments—and ensure GPU capabilities are fully utilized.

借助 NVIDIA Container Toolkit、Container Runtime、Kubernetes 拓扑管理器（Topology Manager）以及 Kubernetes GPU Operator 等方案，你可以为 GPU 环境构建一套统一且高度优化的软件栈。这些方案能够在单节点与多节点 GPU 环境中实现高效的资源分配与工作负载调度，并确保 GPU 的能力得到充分利用。

Along the way, you’ll build intuition for why these optimizations matter. In essence, they minimize latency, maximize throughput, and ensure your GPUs are constantly fed with data and operating at their peak performance. The result is a robust, scalable system that delivers significant performance gains—and a high goodput percentage—for both training and inference workloads.

在这一过程中，你会逐步建立起对“这些优化为何重要”的直觉。本质上，它们旨在最小化延迟、最大化吞吐量，并确保 GPU 持续获得数据供给、始终运行在性能峰值。最终成果是一套稳健、可扩展的系统，能够为训练与推理工作负载带来显著的性能提升，以及很高的有效吞吐（goodput）占比。

## Operating System

## 操作系统

The operating system (OS) is the foundation that everything runs on. GPU servers typically run a Linux distribution such as Ubuntu Server LTS or Red Hat with an updated kernel that supports the latest GPU hardware. The NVIDIA driver installs kernel modules that create device files like `/dev/nvidia0`, `/dev/nvidia1`, and `/dev/` `nvidia2`—one for each GPU. The driver also creates `/dev/nvidiactl` for driver control operations, `/dev/nvidia-uvm` for unified virtual memory, and `/dev/nvidia-` `modeset` for mode-setting and buffer management.

操作系统（OS）是一切之上运行的基础。GPU 服务器通常运行某种 Linux 发行版，例如 Ubuntu Server LTS 或 Red Hat，并搭配支持最新 GPU 硬件的更新内核。NVIDIA 驱动会安装内核模块，创建诸如 `/dev/nvidia0`、`/dev/nvidia1`、`/dev/nvidia2` 之类的设备文件——每个 GPU 对应一个。驱动还会创建 `/dev/nvidiactl` 用于驱动控制操作，`/dev/nvidia-uvm` 用于统一虚拟内存，以及 `/dev/nvidia-modeset` 用于模式设置与缓冲区管理。

The OS manages CPU scheduling, memory, networking, and storage—all of which should be tuned for high GPU throughput. As such, the OS should be configured to avoid interfering with GPU tasks. For example, GPU nodes should disable swapping or set `vm.swappiness` to 0 to avoid any OS-initiated memory swapping that could interfere with GPU workloads. Part of our job as performance engineers is to adjust these OS settings to set the GPUs up for maximum performance.

操作系统负责管理 CPU 调度、内存、网络与存储——所有这些都应针对高 GPU 吞吐量进行调优。因此，操作系统应被配置为不干扰 GPU 任务。例如，GPU 节点应禁用交换（swap），或将 `vm.swappiness` 设为 0，以避免任何由 OS 发起的、可能干扰 GPU 工作负载的内存交换。作为性能工程师，我们的部分职责就是调整这些 OS 设置，让 GPU 具备发挥最大性能的条件。

A GPU-focused server might want to run additional daemons, or background processes, such as the NVIDIA Persistence Daemon to keep the GPU driver and hardware context loaded and ready—even when no GPU jobs are running. In addition, the Fabric Manager manages GPU interconnect topology. And the NVIDIA Data Center GPU Manager (DCGM) monitors GPU system health metrics.

一台以 GPU 为核心的服务器可能还需要运行一些额外的守护进程（daemon）或后台进程，例如 NVIDIA Persistence Daemon，用于让 GPU 驱动与硬件上下文始终保持加载与就绪状态——即使当前没有任何 GPU 作业在运行。此外，Fabric Manager 负责管理 GPU 互连拓扑，而 NVIDIA Data Center GPU Manager（DCGM）则负责监控 GPU 系统健康指标。

## NVIDIA Software Stack

## NVIDIA 软件栈

Running a multi-petaFLOP GPU cluster involves more than just writing high-level PyTorch, TensorFlow, or JAX code. There is a whole software stack underpinning GPU operations, and each layer can affect performance. Figure 3-1 shows a common set of frameworks, libraries, compilers, runtimes, and tools used to develop and productionize modern LLM workloads, including PyTorch, cuDNN, cuBLAS, CUTLASS, CUDA C++, `nvcc`, and the CUDA Runtime API (e.g., CUDA tools, driver, etc.).

运行一个多 petaFLOP 级的 GPU 集群，远不止编写高层的 PyTorch、TensorFlow 或 JAX 代码那么简单。GPU 运行的背后有一整套软件栈支撑，其中每一层都可能影响性能。图 3-1 展示了一组用于开发并将现代 LLM 工作负载投入生产的常见框架、库、编译器、运行时与工具，包括 PyTorch、cuDNN、cuBLAS、CUTLASS、CUDA C++、`nvcc` 以及 CUDA Runtime API（例如 CUDA 工具、驱动等）。

In addition, the NVIDIA GPU and CUDA ecosystem embraces Python libraries and allows you to create CUDA kernels in Python using frameworks like OpenAI’s Triton domain-specific language (DSL) and NVIDIA’s Warp framework—as well as NVIDIA’s CUDA Python, cuTile, and CUTLASS libraries.

此外，NVIDIA GPU 与 CUDA 生态也拥抱 Python 库，允许你使用诸如 OpenAI 的 Triton 领域特定语言（domain-specific language，DSL）与 NVIDIA 的 Warp 框架，以及 NVIDIA 的 CUDA Python、cuTile 和 CUTLASS 库，用 Python 编写 CUDA 内核。

![Figure 3-1. Common set of frameworks, libraries, compilers, runtimes, and tools used to develop and productionize modern LLM workloads](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-1.png)

![图 3-1. 一组用于开发并将现代 LLM 工作负载投入生产的常见框架、库、编译器、运行时与工具](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-1.png)

_Figure 3-1. Common set of frameworks, libraries, compilers, runtimes, and tools used to develop and productionize modern LLM workloads_

_图 3-1. 一组用于开发并将现代 LLM 工作负载投入生产的常见框架、库、编译器、运行时与工具_

### GPU Driver

### GPU 驱动

At the base is the NVIDIA GPU driver, which interfaces between the Linux OS and the GPU hardware. The driver manages low-level GPU operations, including memory allocation on the device, task scheduling on GPU cores, and partitioning the GPU for multitenant usage.

处于最底层的是 NVIDIA GPU 驱动，它在 Linux 操作系统与 GPU 硬件之间充当接口。驱动负责管理底层 GPU 操作，包括设备端的内存分配、GPU 核心上的任务调度，以及为多租户（multitenant）使用而对 GPU 进行切分。

The GPU driver turns on the GPUs’ features and keeps the hardware fed with work. It’s important to keep the NVIDIA driver up-to-date. New driver releases often unlock performance improvements and support the latest GPU architectures and CUDA features.

GPU 驱动负责开启 GPU 的各项特性，并持续为硬件供给工作。保持 NVIDIA 驱动为最新版本非常重要。新的驱动版本往往能解锁性能提升，并支持最新的 GPU 架构与 CUDA 特性。

Tools such as `nvidia-smi` come with the driver and allow you to monitor temperatures, measure utilization, query error-correcting code (ECC) memory status, and enable different GPU modes like persistence mode.

诸如 `nvidia-smi` 之类的工具随驱动一并提供，可用于监控温度、测量利用率、查询纠错码（error-correcting code，ECC）显存状态，以及启用持久化模式等不同的 GPU 模式。

### CUDA Toolkit and Runtime

### CUDA 工具包与运行时

On top of the driver sits the CUDA Runtime and libraries called the CUDA Toolkit. The toolkit includes the CUDA compiler, `nvcc`, used to compile CUDA C++ kernels. When compiled, CUDA programs link against the CUDA runtime (`cudart`). The CUDA runtime communicates directly with the NVIDIA driver to launch work and allocate memory on the GPU.

在驱动之上是 CUDA 运行时以及一组称为 CUDA 工具包（CUDA Toolkit）的库。工具包中包含用于编译 CUDA C++ 内核的 CUDA 编译器 `nvcc`。编译完成后，CUDA 程序会链接到 CUDA 运行时（`cudart`）。CUDA 运行时直接与 NVIDIA 驱动通信，以在 GPU 上启动工作并分配内存。

Additionally, the CUDA Toolkit provides many optimized libraries: cuDNN for neural network primitives, cuBLAS for linear algebra, NCCL for multi-GPU communication, etc. As such, it’s critical to use the latest CUDA Toolkit version that supports your GPU’s compute capability (CC) since an up-to-date toolkit has the latest compiler optimizations and libraries specific to your GPU. We will cover the CUDA compiler and programming model—as well as CUDA (and PyTorch) optimizations—in more detail in the upcoming chapters.

此外，CUDA 工具包还提供了许多经过优化的库：用于神经网络原语的 cuDNN、用于线性代数的 cuBLAS、用于多 GPU 通信的 NCCL，等等。因此，使用支持你 GPU 计算能力（compute capability，CC）的最新 CUDA 工具包版本至关重要，因为最新的工具包拥有针对你 GPU 的最新编译器优化与专用库。我们将在后续章节中更详细地介绍 CUDA 编译器与编程模型，以及 CUDA（和 PyTorch）优化。

### CUDA Forward and Backward Compatibility Across GPU Hardware Generations

### 跨 GPU 硬件世代的 CUDA 向前与向后兼容

An important feature of NVIDIA’s GPU programming model is its compatibility across hardware generations. When you compile CUDA code, the resulting binary includes virtual, or intermediate, PTX code as well as physical device code (e.g., ARM, x86, GPU instructions), as shown in Figure 3-2.

NVIDIA GPU 编程模型的一个重要特性是它跨硬件世代的兼容性。当你编译 CUDA 代码时，生成的二进制文件既包含虚拟的（或称中间的）PTX 代码，也包含物理设备代码（例如 ARM、x86、GPU 指令），如图 3-2 所示。

![Figure 3-2. Using `nvcc` to compile a CUDA program into PTX—and ultimately the low-level instructions for the GPU target device](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-2.png)

![图 3-2. 使用 `nvcc` 将 CUDA 程序编译为 PTX——并最终编译为面向 GPU 目标设备的底层指令](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-2.png)

_Figure 3-2. Using `nvcc` to compile a CUDA program into PTX—and ultimately the low-level instructions for the GPU target device_

_图 3-2. 使用 `nvcc` 将 CUDA 程序编译为 PTX——并最终编译为面向 GPU 目标设备的底层指令_

This allows newer GPUs to just-in-time (JIT) compile the PTX so your program runs on future architectures—and allows newer GPUs to execute older binary code for prior architectures. This compatibility is achieved through NVIDIA’s fatbinary model, which contains PTX for future-proofing and CUBIN, or architecture-specific CUDA device code binaries, for known architectures.

这使得较新的 GPU 能够即时（just-in-time，JIT）编译 PTX，从而让你的程序运行在未来的架构上——同时也让较新的 GPU 能够执行为先前架构生成的旧二进制代码。这种兼容性是通过 NVIDIA 的 fatbinary 模型实现的：其中既包含用于面向未来的 PTX，也包含面向已知架构的 CUBIN（即特定于架构的 CUDA 设备代码二进制）。

CUBIN is the binary produced by `nvcc` using the `-cubin` option. It contains compiled GPU streaming assembler (SASS) instructions for a given NVIDIA architecture. It’s packaged into a fatbinary for loading by the CUDA driver at runtime. Unlike PTX, which is an intermediate, forward-compatible representation, CUBIN binary files allow direct execution on known GPU architectures. When included alongside PTX in a fatbinary, CUBIN supports both JIT-compiling PTX for future GPUs and running older CUBIN code on newer hardware.

CUBIN 是 `nvcc` 使用 `-cubin` 选项生成的二进制文件。它包含针对某一给定 NVIDIA 架构编译好的 GPU 流式汇编（streaming assembler，SASS）指令，并被打包进 fatbinary，供 CUDA 驱动在运行时加载。与作为中间、向前兼容表示形式的 PTX 不同，CUBIN 二进制文件允许在已知的 GPU 架构上直接执行。当 CUBIN 与 PTX 一同被包含在 fatbinary 中时，它既支持为未来 GPU 即时编译 PTX，也支持在较新硬件上运行较旧的 CUBIN 代码。

In short, CUDA provides forward compatibility when PTX is embedded because the driver can JIT-compile PTX for newer architectures at runtime. CUBIN objects are architecture-specific and are not forward-compatible to future GPU architectures, so you should include PTX or ship fat binaries (aka “fatbinaries” or just “fatbins”) that contain both SASS for the current architectures and PTX for forward compatibility.

简而言之，当嵌入了 PTX 时，CUDA 提供向前兼容性，因为驱动可以在运行时为较新架构即时编译 PTX。CUBIN 对象是特定于架构的，对未来的 GPU 架构并不向前兼容，因此你应当包含 PTX，或者交付同时包含面向当前架构的 SASS 与用于向前兼容的 PTX 的 fat 二进制（即 “fatbinaries” 或简称 “fatbins”）。

### C++ and Python CUDA Libraries

### C++ 与 Python CUDA 库

While most CUDA toolkit libraries are C++, NVIDIA’s current Python-facing options include CUDA Python (e.g., low-level driver and runtime access); cuPyNumeric, CuTe DSL, cuTile, and CuPy for array programming; and NVIDIA Warp for authoring GPU kernels in Python. CUTLASS is a C++ templated library used under the hood by libraries such as cuBLAS rather than a Python library.

尽管大多数 CUDA 工具包库都是 C++ 的，NVIDIA 当前面向 Python 的选项包括：CUDA Python（例如底层驱动与运行时访问）；用于数组编程的 cuPyNumeric、CuTe DSL、cuTile 和 CuPy；以及用于以 Python 编写 GPU 内核的 NVIDIA Warp。CUTLASS 是一个 C++ 模板库，被 cuBLAS 等库在底层使用，而非一个 Python 库。

While most of the CUDA Toolkit libraries are C++ based, more and more Python-based libraries are emerging from NVIDIA that are prefixed with “Cu” and built upon the C++ toolkit. For instance, cuTile and cuPyNumeric are Python libraries launched in early 2025. They are targeted at lowering the barrier to entry for Python developers to build applications for NVIDIA GPUs using CUDA.

虽然大多数 CUDA 工具包库都基于 C++，但 NVIDIA 正推出越来越多基于 Python 的库，它们以 “Cu” 为前缀，并构建在 C++ 工具包之上。例如，cuTile 与 cuPyNumeric 是 2025 年初推出的 Python 库，旨在降低 Python 开发者使用 CUDA 为 NVIDIA GPU 构建应用的门槛。

cuTile is a Python library designed to simplify working with large matrices on GPUs by breaking them into smaller, more manageable submatrices called tiles. It provides a high-level, tile-based abstraction that makes it easier to perform block-wise computations, optimize memory access patterns, and efficiently schedule GPU kernels.

cuTile 是一个 Python 库，通过将大矩阵拆分为更小、更易管理的子矩阵（称为 tile）来简化在 GPU 上处理大矩阵的工作。它提供了一种高层的、基于 tile 的抽象，使得执行分块计算、优化内存访问模式以及高效调度 GPU 内核变得更加容易。

By dividing a large matrix into tiles, cuTile helps developers take full advantage of the GPU’s parallelism without needing to manage low-level details manually. This approach can lead to improved cache usage and overall better performance in applications that require intensive matrix computations.

通过将大矩阵划分为 tile，cuTile 帮助开发者充分利用 GPU 的并行性，而无需手动管理底层细节。这种方式能够改善缓存使用，并在需要密集矩阵计算的应用中带来整体上更好的性能。

cuPyNumeric is a drop-in replacement (`import cupynumeric as np`) for the popular `numpy` Python library that utilizes the GPU. It provides nearly the same functions, methods, and behaviors as NumPy, so developers can often switch to it with minimal changes to their code. Under the hood, cuPyNumeric leverages CUDA to perform operations in parallel on the GPU. This leads to significant performance gains for compute-intensive tasks such as large-scale numerical computations, matrix operations, and data analysis.

cuPyNumeric 是流行的 `numpy` Python 库的一个即插即用替代品（`import cupynumeric as np`），可利用 GPU。它提供了与 NumPy 几乎相同的函数、方法与行为，因此开发者通常只需对代码做极少改动即可切换过来。在底层，cuPyNumeric 借助 CUDA 在 GPU 上并行执行操作。这为大规模数值计算、矩阵运算与数据分析等计算密集型任务带来了显著的性能提升。

By offloading work to the GPU, cuPyNumeric accelerates computation and improves efficiency for applications handling massive datasets. Its goal is to lower the barrier for Python developers to harness GPU power without having to learn a completely new interface, making it a powerful drop-in alternative to NumPy for high-performance computing.

通过将工作卸载到 GPU，cuPyNumeric 加速了计算，并提升了处理海量数据集的应用的效率。它的目标是降低 Python 开发者驾驭 GPU 算力的门槛，而无需学习一套全新的接口，从而使其成为面向高性能计算、可有力替代 NumPy 的即插即用选择。

Another notable Python-based programming model is OpenAI’s open source Triton language and compiler. Triton is a Python DSL that allows writing custom GPU kernels in Python. While not an NVIDIA library, Triton complements CUDA by allowing developers to write high-performance kernels directly in Python.

另一个值得注意的基于 Python 的编程模型是 OpenAI 的开源 Triton 语言与编译器。Triton 是一种 Python DSL，允许用 Python 编写自定义 GPU 内核。虽然它并非 NVIDIA 的库，但 Triton 通过让开发者直接用 Python 编写高性能内核，对 CUDA 形成了补充。

We cover Triton and various Triton-based optimizations in a later chapter, but just know that Triton reduces the need for handwritten CUDA C++ in many cases. And it’s integrated into PyTorch’s compiler backend to automatically optimize and fuse GPU operations for better performance. Let’s now turn the discussion to PyTorch.

我们将在后续章节中介绍 Triton 及各种基于 Triton 的优化，这里只需知道：在许多情况下，Triton 减少了手写 CUDA C++ 的需求。而且它已集成到 PyTorch 的编译器后端中，可自动优化并融合 GPU 操作以获得更好的性能。下面我们把讨论转向 PyTorch。

### PyTorch and Higher-Level AI Frameworks

### PyTorch 与更高层的 AI 框架

Some popular Python-based frameworks built on CUDA are PyTorch, TensorFlow, JAX, and Keras. These frameworks provide high-level interfaces for deep learning while leveraging the power of NVIDIA GPUs. This book primarily focuses on PyTorch’s compilation and graph optimization features, including the `torch.compile` stack.

一些流行的、构建于 CUDA 之上的基于 Python 的框架包括 PyTorch、TensorFlow、JAX 和 Keras。这些框架为深度学习提供高层接口，同时充分利用 NVIDIA GPU 的算力。本书主要聚焦于 PyTorch 的编译与图优化特性，包括 `torch.compile` 栈。

The PyTorch compiler stack consists of TorchDynamo, AOT Autograd, and a backend like TorchInductor or Accelerated Linear Algebra (XLA), which automatically capture and optimize your models. TorchInductor is the most common backend, and it uses OpenAI’s Triton under the hood. Triton fuses kernels and performs kernel autotuning for your specific GPU and system environment, as we’ll cover in Chapter 14.

PyTorch 编译器栈由 TorchDynamo、AOT Autograd，以及诸如 TorchInductor 或加速线性代数（Accelerated Linear Algebra，XLA）之类的后端组成，它们能够自动捕获并优化你的模型。TorchInductor 是最常用的后端，它在底层使用 OpenAI 的 Triton。正如我们将在第 14 章介绍的那样，Triton 会融合内核，并针对你特定的 GPU 与系统环境进行内核自动调优。

When you perform operations on PyTorch tensors using GPUs, they are moved from the CPU to the GPU in what appears to be a single Python call. However, this single call is actually translated into a series of calls to the CUDA runtime utilizing various CUDA libraries, as shown in Figure 3-3.

当你使用 GPU 对 PyTorch 张量执行操作时，它们看似只经过一次 Python 调用便从 CPU 移动到了 GPU。然而，这一次调用实际上被转换为对 CUDA 运行时的一系列调用，并借助了各种 CUDA 库，如图 3-3 所示。

![Figure 3-3. Flow from PyTorch code to GPU device](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-3.png)

![图 3-3. 从 PyTorch 代码到 GPU 设备的流程](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-3.png)

_Figure 3-3. Flow from PyTorch code to GPU device_

_图 3-3. 从 PyTorch 代码到 GPU 设备的流程_

When you perform matrix multiplications, for example, PyTorch delegates these tasks to libraries such as cuBLAS. cuBLAS is part of the CUDA Toolkit and optimized for GPU execution. Behind the scenes, PyTorch ensures that operations like forward and backward passes are executed using low-level, optimized CUDA functions and libraries.

例如，当你执行矩阵乘法时，PyTorch 会把这些任务委托给 cuBLAS 之类的库。cuBLAS 是 CUDA 工具包的一部分，并针对 GPU 执行进行了优化。在幕后，PyTorch 会确保前向与后向传播等操作都使用底层、经过优化的 CUDA 函数与库来执行。

In short, PyTorch abstracts away the complexity of direct CUDA programming, allowing you to write intuitive Python code that ultimately calls highly optimized CUDA routines, delivering both ease of development and high performance. We will discuss CUDA programming and optimizations in Chapters 4 and 5—as well as PyTorch optimizations in Chapter 9.

简而言之，PyTorch 抽象掉了直接进行 CUDA 编程的复杂性，让你能够编写直观的 Python 代码，而这些代码最终会调用高度优化的 CUDA 例程，从而兼得开发的便捷与高性能。我们将在第 4、5 章讨论 CUDA 编程与优化，并在第 9 章讨论 PyTorch 优化。

All of these components—OS, GPU Driver, CUDA Toolkit, CUDA libraries, and PyTorch—must work together to create the ideal GPU-based development environment. When a researcher submits a training job, the scheduler reserves nodes, the OS provides the GPU devices and memory allocations using the NVIDIA driver, and the container provides the correct software environment (including the optimized, hardware-aware CUDA libraries). The user code (e.g., PyTorch, TensorFlow, JAX) uses these CUDA libraries, which ultimately communicate with the driver and hardware.

所有这些组件——OS、GPU 驱动、CUDA 工具包、CUDA 库以及 PyTorch——必须协同工作，才能打造出理想的基于 GPU 的开发环境。当一名研究者提交训练作业时，调度器会预留节点，操作系统会借助 NVIDIA 驱动提供 GPU 设备与内存分配，容器则提供正确的软件环境（包括经过优化、硬件感知的 CUDA 库）。用户代码（例如 PyTorch、TensorFlow、JAX）使用这些 CUDA 库，而这些库最终会与驱动和硬件通信。

The optimizations described in this chapter are designed to make each layer of this stack as efficient as possible. They will help the GPUs stay busy with actual useful training and inference work—instead of the GPU waiting on the CPU, waiting for memory or disk I/O, or waiting on other GPUs to synchronize.

本章所述的各项优化，旨在使这一软件栈的每一层都尽可能高效。它们将帮助 GPU 始终忙于真正有用的训练与推理工作——而不是让 GPU 等待 CPU、等待内存或磁盘 I/O，抑或等待其他 GPU 同步。

A well-tuned system ensures that models split across dozens of GPUs are not bottlenecked by I/O or OS overhead. System-level tuning is often overlooked in favor of model optimizations, but system-level optimizations can yield substantial performance gains. In some cases, you can get double-digit percentage improvements with small tweaks to your OS-level configuration. At the scale of a big AI project, this can save tens or hundreds of thousands of dollars in compute time.

一个调优良好的系统，能够确保拆分到数十个 GPU 上的模型不会被 I/O 或 OS 开销所拖累。相比模型优化，系统层面的调优常常被忽视，然而系统层面的优化却能带来可观的性能提升。在某些情况下，只需对 OS 层配置做些许微调，就能获得两位数百分比的改进。在大型 AI 项目的规模下，这可以在计算时间上节省数万乃至数十万美元。

## Configuring the CPUs and OS for GPU Environments

## 面向 GPU 环境配置 CPU 与操作系统

One of the most common reasons that GPUs don’t reach full utilization is that the CPU isn’t keeping them fed with useful work. In a typical training loop, the CPU is responsible for preparing the next batch of data, including loading the data from disk, tokenizing the data, transforming it, etc. In addition, the CPU is responsible for dispatching GPU kernels and coordinating between threads and processes.

GPU 无法达到充分利用率的最常见原因之一，就是 CPU 没能持续为其供给有用的工作。在典型的训练循环中，CPU 负责准备下一批数据，包括从磁盘加载数据、对数据进行分词、变换等。此外，CPU 还负责派发 GPU 内核，并在线程与进程之间进行协调。

If these host-side tasks are slow—or if the OS schedules them poorly—the expensive GPU can find itself idle, twiddling its transistors and waiting for the next task or batch of data. To avoid this, we need to optimize how the CPU and OS handle GPU workloads.

如果这些主机侧任务很慢——或者操作系统对它们的调度很糟糕——那么昂贵的 GPU 可能会陷入空闲，无所事事地空转，等待下一个任务或下一批数据。为避免这种情况，我们需要优化 CPU 与操作系统处理 GPU 工作负载的方式。

These optimizations include setting the CPU affinity to avoid cross-NUMA-node traffic so the right cores handle the right data, using memory-allocation strategies to avoid NUMA penalties and applying OS-level changes to eliminate unnecessary latency. This way, the GPU is never starved for data. Part of this involves isolating background daemons and OS tasks on their own cores—and away from the cores that feed the GPUs, which we’ll discuss next.

这些优化包括：设置 CPU 亲和性以避免跨 NUMA 节点的流量，从而让恰当的核心处理恰当的数据；采用相应的内存分配策略以规避 NUMA 惩罚；以及应用 OS 层的改动以消除不必要的延迟。如此一来，GPU 便永远不会因缺数据而“挨饿”。其中一部分工作，是把后台守护进程与 OS 任务隔离到它们自己的核心上——远离那些为 GPU 供给数据的核心，这一点我们接下来会讨论。

### NUMA Awareness and CPU Pinning

### NUMA 感知与 CPU 绑定

Modern server CPUs have dozens of cores and are often split into multiple NUMA nodes. A NUMA node is a logical grouping of CPUs, GPUs, network interface controllers (NICs), and memory that are physically close to one another. Being aware of the system’s NUMA architecture is important for performance tuning. Accessing resources within a single NUMA node is faster than accessing resources in other NUMA nodes.

现代服务器 CPU 拥有数十个核心，且常被划分为多个 NUMA（非统一内存访问）节点。NUMA 节点是一组在物理上彼此靠近的 CPU、GPU、网卡（network interface controllers，NIC）与内存的逻辑分组。了解系统的 NUMA 架构对性能调优十分重要。访问单个 NUMA 节点内的资源，要比访问其他 NUMA 节点的资源更快。

For example, if a process running on a CPU in NUMA node 0 needs to access a GPU in NUMA node 1, it will need to send data across an internode link, which will incur higher latency. In fact, memory access latency can nearly double when crossing to the other NUMA nodes.

例如，如果一个运行在 NUMA 节点 0 上的 CPU 进程需要访问位于 NUMA 节点 1 的 GPU，它就需要跨节点间链路发送数据，从而带来更高的延迟。事实上，当访问跨越到其他 NUMA 节点时，内存访问延迟可能几乎翻倍。

> On Grace-based superchips such as GH200 and GB200, the CPU and GPU are linked by NVLink-C2C, which provides coherent CPU-to-GPU memory access at up to ~900 GB/s between Grace and its paired accelerator. Linux still treats CPU DRAM as CPU NUMA memory and GPU HBM as device memory. As such, you should continue to bind CPU threads to the local Grace CPU and respect data locality, even though coherence reduces software overheads.

> 在诸如 GH200 与 GB200 之类基于 Grace 的超级芯片上，CPU 与 GPU 通过 NVLink-C2C 相连，可在 Grace 及其配对的加速器之间提供高达约 900 GB/s 的一致性（coherent）CPU-到-GPU 内存访问。Linux 仍将 CPU DRAM 视为 CPU NUMA 内存，将 GPU HBM 视为设备内存。因此，即使一致性降低了软件开销，你仍应继续将 CPU 线程绑定到本地的 Grace CPU 并尊重数据局部性。

On many dual-socket systems, remote memory access latency can be significantly higher than local memory access. In one experiment, local NUMA node memory access latency is ~80 ns compared to remote (cross-node) memory access latency of ~139 ns. This is roughly a 75% increase in latency, which is a huge difference in access speed between local and remote NUMA node memory access.

在许多双路（dual-socket）系统上，远程内存访问延迟可能显著高于本地内存访问。在一次实验中，本地 NUMA 节点的内存访问延迟约为 80 ns，而远程（跨节点）内存访问延迟约为 139 ns。这大约是 75% 的延迟增加，本地与远程 NUMA 节点内存访问在访问速度上的差距巨大。

By binding a process to a CPU on the same NUMA node as its GPU, we can avoid this extra overhead. For instance, you can use `numactl --cpunodebind=<node>` `--membind=<node>` to bind both CPU threads and memory allocations to the GPU’s local NUMA node. You’ll learn more about this in a bit. The key idea is to keep CPU execution and memory access local to the GPU that it’s serving.

通过将进程绑定到与其 GPU 处于同一 NUMA 节点的 CPU 上，我们就能避免这一额外开销。例如，你可以使用 `numactl --cpunodebind=<node> --membind=<node>`，将 CPU 线程与内存分配都绑定到 GPU 的本地 NUMA 节点。稍后你会进一步了解这一点。其核心思想是：让 CPU 的执行与内存访问都保持在它所服务的 GPU 的本地范围内。

> While Linux includes basic NUMA balancing, it’s usually not sufficient for performance-critical AI workloads. By default, processes may be migrated across NUMA nodes. This will lead to additional latency caused by remote memory accesses. As such, it’s important to explicitly bind processes and memory to the same NUMA node as the local GPU. You can do this using `numactl`, `taskset`, or cgroups, as we’ll show in a bit.

> 虽然 Linux 内置了基本的 NUMA 平衡机制，但对于性能关键的 AI 工作负载而言，它通常并不足够。默认情况下，进程可能会在 NUMA 节点之间迁移，这会因远程内存访问而带来额外延迟。因此，显式地将进程与内存绑定到与本地 GPU 相同的 NUMA 节点非常重要。你可以使用 `numactl`、`taskset` 或 cgroups 来做到这一点，稍后我们会展示。

To explicitly specify NUMA-affinity, you need to “pin” processes or threads to specific CPUs that are connected to the same NUMA node as the GPU. This type of CPU affinity is called CPU pinning. Suppose you have eight GPUs in a node, with four GPUs connected to NUMA node 0 and the other four to NUMA node 1.

要显式指定 NUMA 亲和性，你需要将进程或线程“钉”（pin）到与 GPU 连接同一 NUMA 节点的特定 CPU 上。这种 CPU 亲和性称为 CPU 绑定（CPU pinning）。假设你在一个节点中有八个 GPU，其中四个连接到 NUMA 节点 0，另外四个连接到 NUMA 节点 1。

If you launch eight training processes, one per GPU, you should bind each training process to a CPU core—or set of CPU cores—connected to the same NUMA node as the GPUs. In this case, GPUs 0–3 are connected to NUMA node 0 and GPUs 4–7s are connected to NUMA node 1’s cores, as shown in Figure 3-4.

如果你启动八个训练进程，每个 GPU 对应一个，你就应当把每个训练进程绑定到与相应 GPU 连接同一 NUMA 节点的一个 CPU 核心——或一组 CPU 核心——上。在这种情况下，GPU 0–3 连接到 NUMA 节点 0，而 GPU 4–7 连接到 NUMA 节点 1 的核心，如图 3-4 所示。

![Figure 3-4. Eight GPUs in a node, with four GPUs connected to NUMA node 0 and the other four to NUMA node 1](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-4.png)

![图 3-4. 一个节点中的八个 GPU，其中四个连接到 NUMA 节点 0，另外四个连接到 NUMA 节点 1](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-4.png)

_Figure 3-4. Eight GPUs in a node, with four GPUs connected to NUMA node 0 and the other four to NUMA node 1_

_图 3-4. 一个节点中的八个 GPU，其中四个连接到 NUMA 节点 0，另外四个连接到 NUMA 节点 1_

This way, when a CPU process wants to feed data to GPU 4, it should be running on a CPU connected to NUMA node 1 since GPU 4 is connected to NUMA node 1. Linux

> provides tools to do this, including `numactl --cpunodebind=<node> --membind=`

`<node>`, which launches a process pinned to the given NUMA node.

这样一来，当一个 CPU 进程想要向 GPU 4 供给数据时，它就应当运行在连接到 NUMA 节点 1 的 CPU 上，因为 GPU 4 连接到的正是 NUMA 节点 1。Linux 提供了实现这一点的工具，包括 `numactl --cpunodebind=<node> --membind=<node>`，它会启动一个被钉到给定 NUMA 节点的进程。

You can also use `taskset` to pin processes to specific core IDs. Here is an example using `numactl` to bind the `train.py` script to a CPU running in the same NUMA node 1 as GPU 4:

你也可以使用 `taskset` 将进程钉到特定的核心 ID 上。下面是一个使用 `numactl` 将 `train.py` 脚本绑定到与 GPU 4 处于同一 NUMA 节点 1 的 CPU 上的示例：

```
numactl --cpunodebind=1 --membind=1 \
    python train.py --gpu 4
```

```
numactl --cpunodebind=1 --membind=1 \
    python train.py --gpu 4
```

This assumes we know the NUMA node ID and that we are binding the script to only one GPU. Binding the `train.py` to multiple GPUs to an unknown NUMA node is a bit more complicated. The following script dynamically queries the topology using `nvidia-smi topo` and binds the script to GPUs using the local NUMA node:

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

Here, we use `topo -m` to get both CPU and NUMA affinities. We then extract the single-node ID from the NUMA Affinity column. Finally, we bind both `--cpunodebind` and `--membind` to that node to ensure your process’s threads and memory allocations stay local to the GPU’s NUMA domain.

在这里，我们使用 `topo -m` 同时获取 CPU 与 NUMA 亲和性。随后我们从 NUMA Affinity 列中提取单个节点 ID。最后，我们将 `--cpunodebind` 和 `--membind` 都绑定到该节点，以确保你进程的线程与内存分配都保持在 GPU 的 NUMA 域本地。

Many deep learning frameworks also let you set thread affinities programmatically. For instance, PyTorch’s DataLoader exposes `worker_init_fn` so you can set CPU affinity for each worker process during initialization, as shown here:

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

This script binds the main training process and each DataLoader worker process to the GPU’s local NUMA node to prevent cross-NUMA memory access. In the DataLoader, we pass a closure-based `worker_init_fn` that reapplies the precomputed NUMA binding inside each worker. And we do this without touching any CUDA APIs in the worker.

该脚本将主训练进程与每个 DataLoader worker 进程都绑定到 GPU 的本地 NUMA 节点，以防止跨 NUMA 的内存访问。在 DataLoader 中，我们传入了一个基于闭包的 `worker_init_fn`，它会在每个 worker 内部重新应用预先计算好的 NUMA 绑定。而且我们在 worker 中做这件事时，不会触碰任何 CUDA API。

At startup, the process uses NVML to map the current GPU to its NUMA node and CPU-affinity mask. When available, we read the node directly via `nvmlDeviceGet` `NUMANodeId`. Otherwise, we derive it from the GPU’s CPU-affinity mask (`nvmlDevice` `GetCpuAffinity`). If NVML is unavailable or does not expose the node, we fall back

> to the kernel’s sysfs entry at `/sys/bus/pci/devices/<PCI_ID>/numa_node`. As a last

resort, we use the process’s current preferred node.

在启动阶段，进程使用 NVML 将当前 GPU 映射到其 NUMA 节点与 CPU 亲和性掩码。在可用时，我们通过 `nvmlDeviceGetNUMANodeId` 直接读取节点。否则，我们从 GPU 的 CPU 亲和性掩码（`nvmlDeviceGetCpuAffinity`）推导出它。如果 NVML 不可用或不暴露该节点，我们就回退到内核在 `/sys/bus/pci/devices/<PCI_ID>/numa_node` 处的 sysfs 条目。作为最后手段，我们使用进程当前的首选（preferred）节点。

We then compute the CPU list for that node from `/sys/devices/system/node/` `node<N>/cpulist` and apply CPU affinity to those cores with `psutil`. We also bind all future allocations to that node using `libnuma` (`numa_run_on_node + numa_set_`

> `preferred`).

随后，我们从 `/sys/devices/system/node/node<N>/cpulist` 计算出该节点的 CPU 列表，并用 `psutil` 将 CPU 亲和性应用到这些核心上。我们还使用 `libnuma`（`numa_run_on_node + numa_set_preferred`）将所有未来的分配都绑定到该节点。

Because some launchers, container runtimes, or kernels do not reliably propagate NUMA policy to children, we explicitly reapply and verify the binding in every forked worker. It’s not safe to rely on inheritance alone.

由于某些启动器、容器运行时或内核并不能可靠地将 NUMA 策略传播给子进程，我们会在每个 fork 出来的 worker 中显式地重新应用并验证绑定。仅仅依赖继承并不安全。

> Remember to set `pin_memory=True` and use `non_blocking=True` on H2D copies so

that page-locked host buffers stay on the correct NUMA node. Prefer `persis` `tent_workers=True` to avoid re-forking workers and losing their affinity between epochs. And do not call `torch.cuda.*` in `worker_init_fn`. Instead, pass the GPU index using a closure or environment variable.

> 记得设置 `pin_memory=True`，并在 H2D（主机到设备）拷贝上使用 `non_blocking=True`，这样页锁定的主机缓冲区就会保持在正确的 NUMA 节点上。优先使用 `persistent_workers=True`，以避免重新 fork worker 并在各个 epoch 之间丢失它们的亲和性。另外，不要在 `worker_init_fn` 中调用 `torch.cuda.*`；相反，应通过闭包或环境变量传入 GPU 索引。

The result is that data preparation and batch loading happen entirely in local memory. This way, your GPUs stay busy and never need to pause for a remote-NUMA hop. With this code, you get robust, topology-aware affinity on any Linux server with `libnuma` and `numactl` installed.

其结果是：数据准备与批次加载完全发生在本地内存中。如此一来，你的 GPU 会保持忙碌，永远不需要为一次远程 NUMA 跳转而暂停。有了这段代码，在任何安装了 `libnuma` 与 `numactl` 的 Linux 服务器上，你都能获得稳健的、拓扑感知的亲和性。

By default, `numactl` applies its CPU and memory policy to a process and is documented to inherit that policy to all forked children. In practice, however, threads spawned by Python frameworks or exec’d subprocesses don’t always pick up the same settings on every kernel or Linux distribution. When using framework-managed worker processes, you should explicitly reassert the CPU and memory policy inside of each worker.

默认情况下，`numactl` 会将其 CPU 与内存策略应用到某个进程，并且据文档所述会将该策略继承给所有 fork 出来的子进程。然而在实践中，由 Python 框架派生的线程或 exec 出来的子进程，并不总能在每个内核或 Linux 发行版上采用相同的设置。当使用由框架管理的 worker 进程时，你应当在每个 worker 内部显式地重新确立 CPU 与内存策略。

> With a superchip architecture like Grace Blackwell (and Vera Rubin), the CPU and GPU are coherent using NVLink-C2C. However, Linux still models CPU DRAM and GPU HBM as separate pools. Binding CPU threads to the local CPU NUMA node still remains beneficial for locality.

> 对于像 Grace Blackwell（以及 Vera Rubin）这样的超级芯片架构，CPU 与 GPU 通过 NVLink-C2C 保持一致性。然而，Linux 仍然将 CPU DRAM 与 GPU HBM 建模为彼此独立的内存池。将 CPU 线程绑定到本地 CPU NUMA 节点，对于局部性而言仍然有益。

In practice, pinning can eliminate unpredictable CPU scheduling behavior. It ensures that a critical thread such as a data-loading thread for your GPU doesn’t suddenly get migrated by the OS to a core on a different NUMA node in the middle of training or inferencing. In practice, it’s possible to see 5%–10% training throughput improvements just by eliminating cross-NUMA traffic and CPU core migrations. This also tends to reduce performance jitter and variance.

在实践中，绑定可以消除不可预测的 CPU 调度行为。它能确保像“为你的 GPU 加载数据的线程”这样的关键线程，不会在训练或推理进行到一半时突然被操作系统迁移到另一个 NUMA 节点的核心上。实践中，仅仅通过消除跨 NUMA 流量与 CPU 核心迁移，就有可能看到 5%–10% 的训练吞吐量提升。这往往还能降低性能抖动（jitter）与波动。

Many high-performance AI systems evaluate CPU simultaneous multithreading (SMT), or hyperthreading as it’s often called—and sometimes disable it for more predictable per-core performance, but the benefit is workload-dependent. These systems may also reserve a handful of cores exclusively for OS background tasks by setting the `isolcpus` kernel parameter to isolate them from the general scheduler. You can also use Kubernetes CPU isolation for system daemons. This ensures that the remaining cores are dedicated entirely to training and inference threads and doing useful work.

许多高性能 AI 系统会评估 CPU 同步多线程（simultaneous multithreading，SMT，也常被称为超线程 / hyperthreading）——有时为了获得更可预测的单核性能而将其禁用，但其收益取决于具体工作负载。这些系统还可能通过设置 `isolcpus` 内核参数，将少数几个核心从通用调度器中隔离出来，专门保留给 OS 后台任务。你也可以使用 Kubernetes 的 CPU 隔离来处理系统守护进程。这确保了其余核心能够完全专用于训练与推理线程，去做真正有用的工作。

It’s important to note that for integrated CPU-GPU superchips like NVIDIA’s Grace Blackwell, many of the traditional concerns about CPU-to-GPU data transfer are alleviated because the CPU and GPU expose a coherent shared virtual address space over NVLink-C2C, while CPU DRAM and GPU HBM remain distinct memory pools. This means that issues like cross-NUMA delays are minimized, and the data can flow more directly between the CPU and GPU.

需要指出的是，对于像 NVIDIA Grace Blackwell 这样的集成式 CPU-GPU 超级芯片，许多关于 CPU-到-GPU 数据传输的传统顾虑都得到了缓解，因为 CPU 与 GPU 通过 NVLink-C2C 暴露出一个一致性的共享虚拟地址空间，而 CPU DRAM 与 GPU HBM 仍是彼此独立的内存池。这意味着跨 NUMA 延迟之类的问题被最小化，数据可以在 CPU 与 GPU 之间更直接地流动。

> It’s not a coincidence that NVIDIA tackled the CPU-to-GPU bottleneck by combining the CPU and GPU onto a single superchip such as the Grace Blackwell architecture. In this design, the CPU and GPU even share a unified, coherent memory using NVLink-C2C at up to 900 GB/s, which minimizes data transfer overhead. Expect NVIDIA to continue addressing system bottlenecks with more of these types of hardware innovations codesigned with the needs of software and algorithms.

> NVIDIA 通过把 CPU 与 GPU 合并到单一超级芯片（如 Grace Blackwell 架构）上来攻克 CPU-到-GPU 瓶颈，这并非偶然。在这一设计中，CPU 与 GPU 甚至通过高达 900 GB/s 的 NVLink-C2C 共享一块统一、一致的内存，从而最小化数据传输开销。可以预期，NVIDIA 将继续以更多此类硬件创新来应对系统瓶颈，并与软件和算法的需求协同设计。

Even with the tightly coupled CPU-GPU superchip architecture, it’s still important to optimize the stack by ensuring that the hardware and software are configured properly so that the integrated system operates at peak efficiency. Even in these tightly coupled architectures, you want to minimize any unnecessary delays in data handling to keep the GPU fully utilized. This includes configuring hugepages, using efficiency prefetching, and pinning memory, as you will see in the next sections.

即便采用了紧耦合的 CPU-GPU 超级芯片架构，通过确保硬件与软件配置得当、让集成系统以峰值效率运行来优化整个软件栈依然很重要。即使在这些紧耦合架构中，你也希望尽量消除数据处理中一切不必要的延迟，从而让 GPU 保持满负荷利用。正如你将在接下来几节中看到的，这包括配置大页、使用高效预取以及固定内存。

### NUMA-Friendly Memory Allocation and Memory Pinning

### NUMA 友好的内存分配与内存固定

By default, a process will allocate memory from the NUMA node of the CPU it’s currently running on. So if you pin a process to NUMA node 0, its memory will naturally come from NUMA node 0’s local RAM, which is ideal. However, if the OS scheduler migrates threads—or if some memory was allocated before you did the pinning—you could end up with the nonideal scenario in which a process running in NUMA node 0 is using memory from NUMA node 1. In this case, every memory access has to hop to the other NUMA node, negating the benefit of CPU pinning.

默认情况下，进程会从其当前运行所在 CPU 的 NUMA 节点分配内存。因此如果你把一个进程绑定到 NUMA 节点 0，它的内存自然会来自 NUMA 节点 0 的本地 RAM，这是理想状态。然而，如果 OS 调度器迁移了线程——或者某些内存是在你完成绑定之前就已分配的——你可能会陷入这样一种非理想情形：运行在 NUMA 节点 0 上的进程却在使用来自 NUMA 节点 1 的内存。在这种情况下，每一次内存访问都必须跳转到另一个 NUMA 节点，从而抵消了 CPU 绑定带来的收益。

To avoid this, the `numactl --membind` option forces memory allocation from a specific NUMA node, as mentioned in an earlier section. In code, there are also NUMA APIs or even environment variables that can influence this configuration. The general rule is to keep memory close to the CPU, which is close to the GPU. That way the chain of data movement from memory to CPU to GPU is all within a single NUMA node. Here is the same example as before but with `--membind=1` to force memory allocation from the preferred NUMA node that includes NUMA node 1:

为避免这种情况，`numactl --membind` 选项会强制从指定的 NUMA 节点分配内存，正如前面某一节所提到的。在代码中，也有一些 NUMA API 甚至环境变量可以影响这项配置。总的原则是：让内存靠近 CPU，而 CPU 又靠近 GPU。这样一来，从内存到 CPU 再到 GPU 的整条数据搬运链路都处于单个 NUMA 节点之内。下面还是之前那个例子，但加上了 `--membind=1`，以强制从包含 NUMA 节点 1 的首选 NUMA 节点分配内存：

```
numactl --cpunodebind=1 --membind=1 python train.py --gpu 5 &
```

```
numactl --cpunodebind=1 --membind=1 python train.py --gpu 5 &
```

It’s important to note that when you launch a process under `numactl`, both its CPU (`--cpunodebind`) and memory policies (`--membind`) are applied to that process and inherited by all of its child processes. As such, any worker subprocesses forked by your training script will automatically use the same NUMA memory binding. However, they must be created using a fork-based model. If you switch to a spawn start method, or otherwise `exec` a new program, those child processes do not inherit the parent’s memory policy.

需要重点指出的是，当你在 `numactl` 下启动一个进程时，它的 CPU 策略（`--cpunodebind`）和内存策略（`--membind`）都会应用于该进程，并被其所有子进程继承。因此，你的训练脚本 fork 出来的任何工作子进程都会自动使用相同的 NUMA 内存绑定。但是，它们必须以基于 fork 的模型创建。如果你改用 spawn 启动方式，或者以其他方式 `exec` 一个新程序，那些子进程就不会继承父进程的内存策略。

In addition, pinned memory, also called page-locked memory, is essential for efficient and direct GPU access. When memory is pinned, the OS won’t swap or move it. This leads to faster direct memory access (DMA) transfers. Copying data from pinned host memory to GPU can be 2–3× faster than from regular pageable memory since the GPU or NIC can perform DMA directly.

此外，固定内存（pinned memory，也称页锁定内存）对于高效且直接的 GPU 访问至关重要。当内存被固定后，OS 不会将其换出或移动。这带来了更快的直接内存访问（direct memory access，DMA）传输。由于 GPU 或 NIC 可以直接执行 DMA，从固定的主机内存拷贝数据到 GPU，可能比从普通可分页内存拷贝快 2–3×。

> You can test the data-transfer bandwidth between CPU memory and GPU memory using `bandwidthTest --memory=<pinned or pageable>` from the installed CUDA utilities.

> 你可以使用已安装的 CUDA 实用工具中的 `bandwidthTest --memory=<pinned or pageable>` 来测试 CPU 内存与 GPU 内存之间的数据传输带宽。

In fact, this is the basis of NVIDIA’s GPUDirect technologies such as GPUDirect RDMA, which allows NICs like InfiniBand to directly exchange data with GPU memory. Similarly, GPUDirect Storage (GDS) allows NVMe drives to stream data into GPU memory without extra CPU overhead.

事实上，这正是 NVIDIA GPUDirect 技术（如 GPUDirect RDMA）的基础，它让 InfiniBand 等 NIC 能够与 GPU 内存直接交换数据。类似地，GPUDirect Storage（GDS）让 NVMe 驱动器无需额外的 CPU 开销即可将数据流式传输进 GPU 内存。

Deep learning frameworks provide options to use pinned memory for data loaders. For example, PyTorch’s `DataLoader` has a flag `pin_memory=True`, which, when true, means the batches loaded will be placed in pinned RAM, as shown in Figure 3-5.

深度学习框架提供了在数据加载器中使用固定内存的选项。例如，PyTorch 的 `DataLoader` 有一个标志 `pin_memory=True`，当其为 true 时，意味着加载的批次将被放入固定 RAM 中，如图 3-5 所示。

![Figure 3-5. Pinned memory (aka page-locked or nonpageable) is a type of memory that cannot be swapped out to disk](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-5.png)

![Figure 3-5. Pinned memory (aka page-locked or nonpageable) is a type of memory that cannot be swapped out to disk](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-5.png)

_Figure 3-5. Pinned memory (aka page-locked or nonpageable) is a type of memory that cannot be swapped out to disk_

_图 3-5. 固定内存（又称页锁定或不可分页内存）是一种无法被换出到磁盘的内存。_

Memory pinning speeds up the `tensor.to(device)` operations because the CUDA driver doesn’t have to pin pages on the fly. It’s especially beneficial when you are using large batch sizes or reading a lot of data in each iteration. Many practitioners have noticed that just turning on `pin_memory=True` in PyTorch can improve performance up to 10%–20% by reducing data transfer bottlenecks and increasing host-to-device transfer throughput.

内存固定可以加速 `tensor.to(device)` 操作，因为 CUDA 驱动无需即时地去固定页面。当你使用较大的批大小或每次迭代读取大量数据时，它尤其有益。许多从业者已经注意到，仅仅在 PyTorch 中打开 `pin_memory=True`，就能通过减少数据传输瓶颈、提高主机到设备的传输吞吐量，将性能提升多达 10%–20%。

In short, you should make sure that your data loader uses pinned memory (e.g., `pin_memory=True` in PyTorch `DataLoader`) and that GPUDirect RDMA and GDS are enabled for supported hardware. This will reduce data transfer latency.

简而言之，你应确保数据加载器使用固定内存（例如在 PyTorch `DataLoader` 中设置 `pin_memory=True`），并对受支持的硬件启用 GPUDirect RDMA 与 GDS。这会降低数据传输延迟。

It’s important to note that the OS has a limit on how much memory a user can lock

需要重点指出的是，OS 对用户可以锁定（固定）的内存量设有上限。

> (pin). This is set with the `ulimit -l <max locked memory>` command. In container-

ized environments, you can adjust the container’s security context and Docker `--ulimit memlock` setting accordingly. This way, the container can lock sufficient memory.

> 这通过 `ulimit -l <max locked memory>` 命令设置。在容器化环境中，你可以相应地调整容器的安全上下文和 Docker 的 `--ulimit memlock` 设置。这样，容器就能锁定足够的内存。

> If you plan to use large, pinned buffers, ensure the `ulimit` value is high—or set it to unlimited. Otherwise the allocation might fail. Typically, one sets it to unlimited for large AI workloads and high-performance computing (HPC) applications.

> 如果你打算使用大型的固定缓冲区，请确保 `ulimit` 值足够高——或将其设为无限制。否则分配可能失败。对于大型 AI 工作负载和高性能计算（high-performance computing，HPC）应用，通常会将其设为无限制。

### Transparent Hugepages

### 透明大页

In addition to pinning memory and binding it to NUMA nodes, we should talk about transparent hugepages (THPs). Linux memory management typically uses 4 KB pages, but managing millions of tiny pages is inefficient when you have processes using tens or hundreds of gigabytes of memory, as in the case of deep learning datasets, prefetched batches, model parameters, etc.

除了固定内存并将其绑定到 NUMA 节点之外，我们还应谈谈透明大页（transparent hugepages，THP）。Linux 内存管理通常使用 4 KB 的页面，但当进程使用数十甚至数百 GB 内存时（如深度学习数据集、预取批次、模型参数等情形），管理数以百万计的微小页面效率低下。

Hugepages—2 MB or even 1 GB pages—can reduce the overhead of virtual memory management by making memory chunks bigger. The main benefits are fewer page faults and less pressure on the translation lookaside Buffer (TLB).

大页——2 MB 甚至 1 GB 的页面——通过增大内存块，可以降低虚拟内存管理的开销。主要好处是更少的缺页（page fault）以及对转换后备缓冲（translation lookaside buffer，TLB）更小的压力。

The TLB is a cache that the CPU uses to map virtual addresses to physical ones. Fewer, larger pages means the TLB can cover more memory with the same number of entries, reducing misses.

TLB 是 CPU 用来将虚拟地址映射到物理地址的一种缓存。页面更少、更大，意味着在相同数量的条目下 TLB 能覆盖更多内存，从而减少未命中。

Hugepages typically produce modest gains—often on the order of ~3%–5% throughput improvement. They do this by reducing page-fault overhead and TLB pressure. Enabling THP is a simple win on most systems since the kernel will automatically back large allocations with 2 MB pages. In scenarios with very large memory pools (e.g., preallocated pinned buffers for I/O), you may also consider explicit hugepages using `vm.nr_hugepages` or `hugetlbfs` for more deterministic performance.

大页通常带来适度的收益——往往在 ~3%–5% 吞吐量提升这个量级。它们通过减少缺页开销和 TLB 压力来实现这一点。在大多数系统上启用 THP 是一个轻松的收益，因为内核会自动用 2 MB 页面来支撑大额分配。在拥有超大内存池的场景（例如为 I/O 预分配的固定缓冲区）中，你还可以考虑使用 `vm.nr_hugepages` 或 `hugetlbfs` 的显式大页，以获得更确定的性能。

> Remember that, when using large, pinned memory regions, you should raise the `ulimit -l` setting (max locked memory) to a high value or `unlimited`. If this limit is too low, your attempt to pin memory can fail, leading to fallback on swappable memory—or out-of-memory (OOM) errors.

> 请记住，当使用大型的固定内存区域时，你应将 `ulimit -l` 设置（最大锁定内存）调高到一个较大的值或设为 `unlimited`。如果该上限太低，你固定内存的尝试可能失败，导致回退到可交换内存——或出现内存不足（out-of-memory，OOM）错误。

It’s important to note that THP’s background compaction can introduce unpredictable pauses that are disastrous for latency-sensitive LLM inference workloads. Linux is configured by default to use THP to automatically allocate 2 MB pages whenever possible. This is often sufficient, but it’s worth testing for your workload.

需要重点指出的是，THP 的后台压缩可能引入不可预测的停顿，这对延迟敏感的 LLM 推理工作负载是灾难性的。Linux 默认配置为使用 THP，只要有可能就自动分配 2 MB 页面。这通常已经足够，但仍值得针对你的工作负载进行测试。

You can disable THP, but you will need to manually allocate and control hugepages. This will incur extra complexity, but it might be needed for low-latency workloads like inference. With THP disabled, your system will avoid stalls caused by kernel-driven defragmentations.

你可以禁用 THP，但那样就需要手动分配和控制大页。这会引入额外的复杂性，但对于推理这类低延迟工作负载可能是必要的。禁用 THP 后，你的系统将避免由内核驱动的碎片整理所导致的停顿。

> The modern consensus is to enable THP for most GPU-based training workloads in which throughput is important and to disable THP completely (`transparent_hugepage=never`)—or use `madvise`—for workloads like inference in which latency is important. This is also true for distributed training workloads in which many ranks (GPUs) allocate memory simultaneously.

> 现代的普遍共识是：对大多数以吞吐量为重的、基于 GPU 的训练工作负载启用 THP；而对于推理这类以延迟为重的工作负载，则完全禁用 THP（`transparent_hugepage=never`）——或使用 `madvise`。对于许多 rank（GPU）同时分配内存的分布式训练工作负载，同样如此。

Beyond CPU/memory pinning and hugepages, there are a few other OS-level tweaks worth mentioning. These include thread scheduling, virtual memory management, filesystem caching, and CPU frequency settings, which we’ll cover in the next few sections.

除了 CPU/内存绑定与大页之外，还有一些其他值得一提的 OS 层调整。这包括线程调度、虚拟内存管理、文件系统缓存以及 CPU 频率设置，我们将在接下来几节中介绍。

### Scheduler and Interrupt Affinity

### 调度器与中断亲和性

On a busy system, you want to make sure that important threads such as data-pipeline threads aren’t interrupted frequently. Linux by default uses the Completely Fair Scheduler (CFS) that works well for most cases.

在繁忙的系统上，你希望确保重要线程（如数据流水线线程）不会被频繁中断。Linux 默认使用完全公平调度器（Completely Fair Scheduler，CFS），它在大多数情况下表现良好。

But if you have a very latency-sensitive thread that feeds the GPU with data, for example, you could consider using real-time first in, first out (FIFO) or round-robin (RR) priority scheduling for that thread. This would make sure that the high-priority thread runs without being preempted by normal-priority threads.

但如果你有一个对延迟极其敏感、专门向 GPU 供给数据的线程，你可以考虑对该线程使用实时的先进先出（first in, first out，FIFO）或轮转（round-robin，RR）优先级调度。这将确保这个高优先级线程运行时不会被普通优先级的线程抢占。

However, use this with caution, as real-time threads can starve other processes if not managed properly. In practice, however, if you’ve pinned your threads to dedicated cores, you often don’t need to mess with real-time thread priorities, but it’s worth keeping an eye on.

不过，使用时要谨慎，因为如果管理不当，实时线程可能会使其他进程陷入饥饿。但在实践中，如果你已经将线程绑定到专用核心上，通常就不需要去折腾实时线程优先级了，但仍值得留意。

Another option is to isolate cores or create separate CPU partitions to further reduce interruptions on these dedicated compute resources. To do this, you can use `cset`, kernel parameters like `isolcpus` and `nohz_full`, or cgroup `cpuset` isolation. With isolation, the OS scheduler leaves those CPU cores for you to use as you wish.

另一个选择是隔离核心或创建独立的 CPU 分区，以进一步减少对这些专用计算资源的干扰。为此，你可以使用 `cset`、诸如 `isolcpus` 和 `nohz_full` 的内核参数，或 cgroup 的 `cpuset` 隔离。有了隔离，OS 调度器就会把那些 CPU 核心留给你随意使用。

> cgroup CPU and memory affinity is strongly recommended in production environments. Using these, each AI workload is isolated on its own physical cores and memory regions. This will prevent cross-workload contention and NUMA penalties. Tools like `cpuset` cgroups or container runtimes (`docker --cpuset-cpus`) should be used to enforce this.

> 在生产环境中强烈推荐使用 cgroup 的 CPU 与内存亲和性。借助它们，每个 AI 工作负载都被隔离在各自的物理核心和内存区域上。这将防止跨工作负载争用和 NUMA 惩罚。应使用诸如 `cpuset` cgroup 或容器运行时（`docker --cpuset-cpus`）之类的工具来强制执行这一点。

You can assign each device’s hardware interrupts to cores on the same NUMA node. This will prevent cross-node interrupt handling that would otherwise incur extra latency and evict useful cache lines on a remote node. For example, if your GPU or NIC on NUMA node 0 raises an interrupt, you’d bind it to a core on node 0 so that no other node handles it. Without this binding, a CPU on a different NUMA node might process the interrupt. This would force cache coherency traffic and cross-node communication.

你可以将每个设备的硬件中断分配到同一 NUMA 节点上的核心。这将防止跨节点的中断处理，否则会带来额外延迟并逐出远端节点上有用的缓存行。例如，如果你位于 NUMA 节点 0 的 GPU 或 NIC 触发了一个中断，你就应将其绑定到节点 0 上的某个核心，以便没有其他节点来处理它。若没有这种绑定，位于不同 NUMA 节点的 CPU 可能会处理该中断。这将强制产生缓存一致性流量和跨节点通信。

In practice, performance-sensitive systems often disable the default `irqbalance` daemon or run it with bespoke rules. The other option is to manually set each interrupt’s affinity mask using `/proc/irq/*/smp_affinity`. By pinning every GPU and NIC interrupt to the nearest cores, you guarantee that those device interrupts are always serviced on the optimal NUMA node.

在实践中，对性能敏感的系统常常会禁用默认的 `irqbalance` 守护进程，或以定制规则运行它。另一个选择是使用 `/proc/irq/*/smp_affinity` 手动设置每个中断的亲和性掩码。通过把每个 GPU 和 NIC 的中断都固定到最近的核心，你就能保证这些设备中断总是在最优的 NUMA 节点上得到处理。

In short, the combination of dedicated cores, appropriate scheduling priorities, and NUMA-aware hardware interrupt bindings can help minimize jitter for data loading threads that are feeding the GPUs.

简而言之，专用核心、恰当的调度优先级以及 NUMA 感知的硬件中断绑定三者相结合，有助于把向 GPU 供给数据的数据加载线程的抖动（jitter）降到最低。

### Virtual Memory and Swapping

### 虚拟内存与交换

It goes without saying, but you should always try to avoid memory swapping. If any part of your process’s memory gets swapped to disk, you will see a catastrophic, multiple-orders-of-magnitude slowdown. GPU programs tend to allocate a lot of host memory for data caching. If the OS decides to swap some data out of memory and onto disk, the GPU will experience huge delays when it needs to access that data.

这不言自明，但你应始终尽量避免内存交换。如果你进程内存的任何部分被换出到磁盘，你将会看到灾难性的、数个数量级的性能下降。GPU 程序往往会分配大量主机内存用于数据缓存。如果 OS 决定把某些数据换出内存到磁盘，那么当 GPU 需要访问那些数据时就会遭遇巨大的延迟。

We recommend setting `vm.swappiness=0`, which tells Linux to avoid swapping except under extreme memory pressure. It effectively isolates your training job’s memory with cgroup limits to prevent any swapping.

我们建议设置 `vm.swappiness=0`，它告诉 Linux 除非在极端内存压力下否则避免交换。它实际上通过 cgroup 限制来隔离你训练作业的内存，以防止任何交换。

> You should use cgroups v2 through Docker or Kubernetes to pin memory and CPUs to the AI process. This will enforce NUMA affinity and no-swap policies in containerized environments.

> 你应通过 Docker 或 Kubernetes 使用 cgroups v2 将内存和 CPU 固定给 AI 进程。这将在容器化环境中强制实施 NUMA 亲和性和禁止交换策略。

You can also use `sudo swapoff -a` to temporarily disable all swap devices and files until the next reboot. Just make sure you have enough RAM for your workload—or put limits to prevent overcommit. Otherwise, the OOM killer may reap the process. Monitor swap usage using `vmstat` or `free -m` to make sure swap stays at zero.

你也可以使用 `sudo swapoff -a` 临时禁用所有交换设备和文件，直到下次重启为止。只要确保你有足够的 RAM 支撑工作负载——或者设置上限以防止过量分配（overcommit）。否则，OOM killer 可能会终结该进程。使用 `vmstat` 或 `free -m` 监控交换使用情况，以确保交换保持为零。

Another related setting is `ulimit -l`, as mentioned earlier for pinned memory. If you want to prevent memory from swapping, you should set that limit high or you may experience excessive memory swapping. Again, typically one sets this limit to unlimited for large AI workloads that utilize a lot of memory.

另一个相关设置是 `ulimit -l`，正如前面针对固定内存所提到的。如果你想防止内存被交换，就应把该上限设得高一些，否则你可能会遭遇过度的内存交换。同样地，对于占用大量内存的大型 AI 工作负载，通常会将该上限设为无限制。

### Filesystem Caching and Write-Back

### 文件系统缓存与回写

A best practice for large training jobs is to write frequent checkpoints to disk in case you need to restart a failed job from a known good checkpoint. During checkpointing, however, huge bursts of data might fill up the OS page cache and cause stalls.

针对大型训练作业的一个最佳实践是：频繁地向磁盘写入检查点（checkpoint），以便在作业失败时能够从一个已知良好的检查点重启。然而在写检查点期间，海量的数据突发可能会填满 OS 页缓存（page cache）并引发停顿。

For storage, you can adjust `vm.dirty_ratio` and `vm.dirty_background_ratio` to tune the page-cache size for buffering writes. For example, with multi-GB checkpoints, using a higher dirty ratio lets the OS batch more data in RAM before flushing to disk. This will smooth out large checkpoint writes and reduce stalls in your training loop.

对于存储，你可以调整 `vm.dirty_ratio` 和 `vm.dirty_background_ratio` 来调优用于缓冲写入的页缓存大小。例如，对于数 GB 大小的检查点，使用更高的脏页比率能让 OS 在刷写到磁盘之前在 RAM 中批量缓存更多数据。这将平滑大型检查点写入，减少训练循环中的停顿。

Another option is to perform checkpointing in a separate thread. A more recent option in PyTorch is to write distributed checkpoint partitions from nodes across the cluster. In this case, the checkpoint partitions will be combined when the checkpoint is loaded after a failed-job restart.

另一个选择是在一个独立线程中执行写检查点。PyTorch 中一个更新的选项是从集群中的各节点写出分布式检查点分片。在这种情况下，当作业失败重启后加载检查点时，这些检查点分片将被合并起来。

In latency-sensitive training workflows, it’s best to bypass the page cache entirely. For example, open checkpoint files with `O_DIRECT` or use Linux’s `io_uring` for asynchronous I/O to avoid page-cache stalls. After writing each checkpoint, call `posix_fadvise` `(fd, 0, 0, POSIX_FADV_DONTNEED)` to immediately drop those pages from cache and prevent memory pressure on subsequent iterations.

在延迟敏感的训练工作流中，最好完全绕过页缓存。例如，用 `O_DIRECT` 打开检查点文件，或使用 Linux 的 `io_uring` 进行异步 I/O，以避免页缓存停顿。每次写完检查点后，调用 `posix_fadvise` `(fd, 0, 0, POSIX_FADV_DONTNEED)` 立即把那些页面从缓存中丢弃，防止在后续迭代中造成内存压力。

### CPU Frequency and C-states

### CPU 频率与 C-state

By default, many compute nodes will run CPUs in a power-saving mode, which either downclocks a CPU or puts it to sleep when it’s idle. This helps save energy and reduce heat and lowers the cost. During model training, the CPUs might not always be 100% utilized as the GPUs are churning through the final batches of their dataset. However, these power management features could cause extra latency when the system wakes the CPUs up again when new work arrives.

默认情况下，许多计算节点会让 CPU 运行在省电模式下，即在 CPU 空闲时对其降频或让其进入睡眠。这有助于节能、降低发热并降低成本。在模型训练期间，当 GPU 正在处理其数据集的最后几个批次时，CPU 可能并不总是被 100% 利用。然而，当有新任务到来、系统再次唤醒 CPU 时，这些电源管理特性可能会带来额外延迟。

For maximum and consistent performance, AI systems often configure the CPU frequency governor to “performance” mode, which keeps the CPU at max frequency all

为获得最大且一致的性能，AI 系统通常会将 CPU 频率调速器（governor）配置为 “performance” 模式，使 CPU 始终保持在最高频率。

> the time. This can be done using `cpupower frequency-set -g performance` or in

the Basic Input/Output System (BIOS).

> 这可以通过 `cpupower frequency-set -g performance` 完成，也可以在基本输入/输出系统（Basic Input/Output System，BIOS）中完成。

Likewise, disabling deep C-states can keep cores from going into a low-power sleep state. CPU C-states are power-saving modes defined by the system’s ACPI specification. When a CPU core is idle, it can enter a C-state to save energy. The deeper the C-state, the more power is saved but the longer it may take for the core to wake up when work arrives. Disabling deeper C-states can remove excessive latency spikes. C0 is active; everything above C0 represents a deeper state of sleep.

同样地，禁用较深的 C-state 可以防止核心进入低功耗睡眠状态。CPU 的 C-state 是由系统 ACPI 规范定义的省电模式。当一个 CPU 核心空闲时，它可以进入某个 C-state 以节能。C-state 越深，节省的功耗越多，但当任务到来时核心唤醒所需的时间也可能越长。禁用较深的 C-state 可以消除过度的延迟尖峰。C0 表示活动状态；C0 以上的任何状态都代表更深层次的睡眠。

> In practice, many server BIOS/UEFI (Unified Extensible Firmware Interface) offer a high-performance profile that automatically sets the CPU governor to “Performance” and disables deep C-states.

> 在实践中，许多服务器 BIOS/UEFI（统一可扩展固件接口，Unified Extensible Firmware Interface）都提供高性能配置文件，会自动将 CPU 调速器设为 “Performance” 并禁用较深的 C-state。

Essentially, we can trade a bit of extra power draw for more responsive CPU behavior. In a training scenario where GPUs are the big power consumers, a bit more CPU power usage is usually fine if it keeps the GPUs fed. For example, if a data loader thread sleeps while waiting for data and the CPU goes into the deep C6 state, significant portions of the CPU are powered down to maximize energy savings.

本质上，我们可以用略高一些的功耗换取更灵敏的 CPU 行为。在 GPU 才是耗电大户的训练场景中，只要能让 GPU 保持满负荷，CPU 多用一点功耗通常是可以接受的。例如，如果一个数据加载线程在等待数据时进入睡眠，而 CPU 进入了较深的 C6 状态，那么 CPU 的相当一部分会被断电以最大化节能。

If the CPU enters a deeper sleep state, it might take a few microseconds to wake up. While this is not a long time, many microseconds can add up and can cause GPU bubbles if not managed properly. Bubbles are periods of time when the GPU is waiting for the CPU to resume data processing. By keeping the CPU ready, we reduce such hiccups. Many BIOSes for servers have a setting to disable C-states—or at least limit them.

如果 CPU 进入较深的睡眠状态，它可能需要几微秒才能唤醒。虽然这算不上很长的时间，但许多个微秒累加起来，如果管理不当就可能造成 GPU 气泡（bubble）。气泡是指 GPU 等待 CPU 恢复数据处理的时间段。通过让 CPU 保持就绪，我们减少了此类卡顿。许多服务器 BIOS 都有一个设置用来禁用 C-state——或至少限制它们。

> You should always turn off anything in your system that might introduce unpredictable latency, such as excess context switching, CPU frequency scaling, and memory-to-disk swapping. The result should be that your CPUs deliver data to the GPUs as fast as the GPUs can consume it, without the OS scheduling things on the wrong core or taking CPU cycles away at the wrong time.

> 你应始终关闭系统中任何可能引入不可预测延迟的东西，比如过多的上下文切换、CPU 频率缩放以及内存到磁盘的交换。其结果应当是：你的 CPU 能以 GPU 消耗数据的速度向 GPU 供给数据，而不会出现 OS 把任务调度到错误的核心上、或在错误的时刻抢走 CPU 周期的情况。

### Tune Host CPU Memory Allocator

### 调优主机 CPU 内存分配器

On a well-tuned GPU server, CPU usage may not be very high since GPUs handle most of the computation. However, CPU usage should remain steady and in lockstep with GPU activity. The CPUs must stay busy preparing each incoming batch while the current batch is being processed by the GPU.

在一台调优良好的 GPU 服务器上，由于 GPU 承担了大部分计算，CPU 的利用率可能并不很高。然而，CPU 利用率应保持稳定并与 GPU 活动同步。在当前批次正被 GPU 处理时，CPU 必须持续忙于准备每一个进来的批次。

Proper CPU-to-GPU handoff is crucial for sustaining high GPU utilization. By tuning your host’s memory allocator (`jemalloc` or `tcmalloc`), you can eliminate unpredictable pauses in data preparation. This will keep GPUs running at their peak—except for intentional synchronization points.

恰当的 CPU 到 GPU 交接对于维持高 GPU 利用率至关重要。通过调优你主机的内存分配器（`jemalloc` 或 `tcmalloc`），你可以消除数据准备过程中不可预测的停顿。这将使 GPU 保持在峰值运行——除了有意为之的同步点之外。

After tuning, you should see each GPU’s utilization hover near 100% and drop only at required synchronization barriers. The GPUs should never stall for data due to CPU-side delays. With `jemalloc`, you can shard allocations into per-CPU arenas (`narenas`), enable `background_thread` for off-path purging, and lengthen `dirty_decay_ms`/`muzzy_decay_ms` so that freed pages aren’t immediately returned to the OS. This will minimize lock contention and fragmentation.

调优之后，你应看到每个 GPU 的利用率都在接近 100% 处徘徊，只在必需的同步屏障处才会下降。GPU 绝不应因 CPU 侧的延迟而停下来等待数据。使用 `jemalloc`，你可以把分配分片到按 CPU 划分的 arena（`narenas`）中，启用 `background_thread` 进行离路径（off-path）清理，并延长 `dirty_decay_ms`/`muzzy_decay_ms`，从而使被释放的页面不会立即归还给 OS。这将把锁争用和碎片化降到最低。

You can tune `jemalloc` with the `MALLOC_CONF` environment variable as follows:

你可以通过 `MALLOC_CONF` 环境变量来调优 `jemalloc`，如下所示：

```
export MALLOC_CONF="narenas:8,dirty_decay_ms:10000,muzzy_decay_ms:10000
,background_thread:true"
```

```
export MALLOC_CONF="narenas:8,dirty_decay_ms:10000,muzzy_decay_ms:10000
,background_thread:true"
```

Similarly, `tcmalloc` benefits from tuning the `TCMALLOC_MAX_TOTAL_THREAD_` `CACHE_BYTES` and `TCMALLOC_RELEASE_RATE` environment variables. These will provide larger per-thread caches so that small allocations avoid global locks and syscalls—keeping CPU threads ready to feed the GPU with low, predictable latency. You can do this as follows:

类似地，`tcmalloc` 也能从调优 `TCMALLOC_MAX_TOTAL_THREAD_` `CACHE_BYTES` 和 `TCMALLOC_RELEASE_RATE` 环境变量中获益。它们将提供更大的按线程缓存，使小额分配得以避开全局锁和系统调用——让 CPU 线程保持就绪，以低而可预测的延迟向 GPU 供给数据。你可以这样做：

```
export TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES=$((512*1024*1024))
export TCMALLOC_RELEASE_RATE=16
```

```
export TCMALLOC_MAX_TOTAL_THREAD_CACHE_BYTES=$((512*1024*1024))
export TCMALLOC_RELEASE_RATE=16
```

In short, optimizing the allocator can reduce allocator overhead and fragmentation. This will keep CPU threads consistently fast and avoid unexpected stalls feeding the GPU. Experiment with these environment variables and tune them for your specific workload and environment.

简而言之，优化分配器可以减少分配器开销和碎片化。这将使 CPU 线程始终保持快速，并避免向 GPU 供给数据时出现意外停顿。请针对你特定的工作负载和环境试验并调优这些环境变量。

## GPU Driver and Runtime Settings for Performance

## 面向性能的 GPU 驱动与运行时设置

We’ve optimized the CPU side, but there are also important settings for the GPU driver and runtime that can affect performance—especially in multi-GPU and multiuser scenarios. NVIDIA GPUs have a few knobs that, when tuned properly, can reduce overhead and improve how multiple workloads share a GPU.

我们已经优化了 CPU 侧，但 GPU 驱动和运行时也有一些重要设置会影响性能——尤其是在多 GPU 和多用户场景中。NVIDIA GPU 有一些旋钮，若调优得当，可以减少开销并改善多个工作负载共享 GPU 的方式。

Next, we’ll cover GPU persistence mode, the partitions of MPS, MIG, and a few other considerations like clock settings, ECC memory, and out-of-memory behavior.

接下来，我们将介绍 GPU 持久化模式、MPS 的分区、MIG，以及诸如时钟设置、ECC 内存和内存不足行为等其他一些考量。

### GPU Persistence Mode

### GPU 持久化模式

By default, if no application is using a GPU, the driver may put the GPU into a lower-power state and unload some of the driver’s context. The next time an application comes along and wants to use the GPU, there’s a cost to initialize it. This can take on the order of a second or two for the driver to spin everything up.

默认情况下，如果没有应用程序在使用某块 GPU，驱动可能会把该 GPU 置入更低功耗的状态，并卸载部分驱动上下文。下一次有应用程序出现并想使用该 GPU 时，就需要付出初始化它的代价。驱动把一切启动起来可能需要一两秒的量级。

GPU initialization overhead can negatively impact performance for workloads that periodically release and reacquire the GPU. For instance, consider a training cluster where jobs are starting and stopping frequently. Or a low-volume inference cluster that has to wake up the GPU every time a new inference request arrives. In both of these cases, the overhead will reduce overall workload performance.

对于会周期性地释放并重新获取 GPU 的工作负载，GPU 初始化开销会对性能产生负面影响。举例来说，设想一个作业频繁启停的训练集群。或者一个低流量的推理集群，每当有新的推理请求到来时都必须唤醒 GPU。在这两种情况下，该开销都会降低整体工作负载性能。

Persistence mode is enabled by running the `nvidia-persistenced` daemon. This keeps the GPU driver loaded and the hardware in a ready state even when no application is active. This requests that the system not fully power down the GPU when idle, which prevents power gating. Persistence keeps the GPU awake so that the next job has zero startup delay. This is generally recommended for long-running and latency-sensitive workloads. You can enable the persistence daemon at boot time using the following command:

持久化模式（persistence mode）通过运行 `nvidia-persistenced` 守护进程来启用。它使 GPU 驱动保持加载、硬件保持就绪状态，即便没有应用程序处于活动状态也是如此。它请求系统在 GPU 空闲时不要将其完全断电，从而防止电源门控（power gating）。持久化让 GPU 保持唤醒，使下一个作业拥有零启动延迟。这对于长时间运行且延迟敏感的工作负载通常是推荐的。你可以使用以下命令在启动时启用持久化守护进程：

```
systemctl enable nvidia-persistenced
```

```
systemctl enable nvidia-persistenced
```

> In Kubernetes environments, the NVIDIA GPU Operator can be configured to enable persistence mode on all GPUs automatically.

> 在 Kubernetes 环境中，可以配置 NVIDIA GPU Operator，使其在所有 GPU 上自动启用持久化模式。

On AI clusters, it’s common to just enable persistence mode on all GPUs at server boot time. This way, when a job begins, the GPUs are already initialized and can start processing immediately. It won’t make your actual compute any faster, as it doesn’t speed up the math operations, but it shaves off job-startup latency and prevents cold start delays.

在 AI 集群中，常见的做法就是在服务器启动时对所有 GPU 启用持久化模式。这样，当作业开始时，GPU 已经初始化完毕，可以立即开始处理。它不会让你实际的计算变得更快，因为它并不加速数学运算，但它削减了作业启动延迟并防止冷启动延迟。

GPU persistence mode also helps with interactive usage, as without persistence, the first CUDA call you make after some idle time might stall while the driver reinitializes the GPU. With persistence on, that call returns quickly.

GPU 持久化模式也有助于交互式使用，因为若没有持久化，你在空闲一段时间后发起的第一个 CUDA 调用可能会因驱动重新初始化 GPU 而卡住。开启持久化后，该调用会迅速返回。

The only downside of persistence is a slightly higher power draw when idle since the GPU stays in a higher readiness state. But, for most data center GPUs, this is an acceptable trade-off for better performance consistency. Once GPU persistence mode is set by an admin with `sudo` access, you can enjoy the benefits and move on to tackle other optimizations.

持久化唯一的缺点是空闲时功耗略高，因为 GPU 保持在更高的就绪状态。但对大多数数据中心 GPU 而言，为了更好的性能一致性，这是一个可以接受的权衡。一旦具有 `sudo` 权限的管理员设置了 GPU 持久化模式，你就可以享受其好处，然后转而着手其他优化。

### MPS

### MPS

Normally, when multiple processes share a single GPU, the GPU’s scheduler time-slices between them. For example, if two Python processes each have some kernels to run on the same GPU, the GPU might execute one process’s kernel, then the other process’s kernel, and so on. If those kernels are short and there’s an idle gap between them, the GPU can end up underutilized as it’s doing “ping-pong” context switches and not overlapping the work.

通常，当多个进程共享单块 GPU 时，GPU 的调度器会在它们之间进行时间分片。例如，如果两个 Python 进程各自都有一些内核（kernel）要在同一块 GPU 上运行，GPU 可能会执行一个进程的内核，然后再执行另一个进程的内核，如此往复。如果这些内核很短且它们之间存在空闲间隙，GPU 最终可能会利用不足，因为它在做“乒乓式”的上下文切换，而没有把工作重叠起来。

NVIDIA’s MPS is a feature that creates a sort of umbrella under which multiple processes can run on the GPU concurrently and without strict time-slicing. With MPS, the GPU can execute kernels from different processes at the same time as long as the GPU resources (streaming multiprocessors [SMs], Tensor Cores, etc.) are available. MPS essentially merges the contexts of the processes into one scheduler context. This way, you don’t pay the full cost of switching and idling between independent processes.

NVIDIA 的 MPS 是这样一项特性：它创建了一种“伞”，让多个进程可以在 GPU 上并发运行而无需严格的时间分片。有了 MPS，只要 GPU 资源（流式多处理器 [streaming multiprocessors，SM]、Tensor Core 等）可用，GPU 就能同时执行来自不同进程的内核。MPS 本质上把这些进程的上下文合并到一个调度器上下文中。这样一来，你就不必为在各个独立进程之间切换和空转付出全部代价。

When is MPS useful? For model training, if you normally run one process per GPU, you might not use MPS. But if you have scenarios like running many inference jobs on one big GPU, MPS is a game changer. Imagine you have a powerful GPU or GPU cluster, but your inference job—or set of multiple inference jobs—doesn’t fully use it.

MPS 何时有用？对于模型训练，如果你通常是每块 GPU 运行一个进程，那你可能用不到 MPS。但如果你遇到诸如在一块大 GPU 上运行许多推理作业的场景，MPS 就是一个改变游戏规则的东西。设想你有一块强大的 GPU 或 GPU 集群，但你的推理作业——或一组多个推理作业——并没有把它完全用起来。

For instance, consider running four separate inference jobs on one 40 GB GPU, each using 5–10 GB and only 30% of GPU compute. By default, each inference job gets a time-slice, so at any moment, only one job’s work is actually running on the GPU. That leaves the GPU 70% idle on average.

举例来说，设想在一块 40 GB 的 GPU 上运行四个独立的推理作业，每个使用 5–10 GB 且仅占用 30% 的 GPU 算力。默认情况下，每个推理作业获得一个时间片，因此在任一时刻，GPU 上实际运行的只有一个作业的工作。这使得 GPU 平均有 70% 处于空闲。

If you enable MPS for these inference jobs, the GPUs can interleave their work so that while one job is waiting on memory, another job’s kernel might fill the GPU, etc. The result is higher overall GPU utilization. In practice, if two processes each use 40% of a GPU, with MPS you might see the GPU at 80%–90% utilization serving both.

如果你为这些推理作业启用 MPS，GPU 就能交错它们的工作，这样在一个作业等待内存时，另一个作业的内核可能会填满 GPU，如此等等。其结果是整体 GPU 利用率更高。在实践中，如果两个进程各使用 GPU 的 40%，启用 MPS 后你可能会看到 GPU 以 80%–90% 的利用率同时服务这两者。

For instance, two training processes that each would take one hour on their own—on the same GPU, running sequentially—can run together with MPS and finish in a bit over one hour total in parallel instead of two hours sequentially. For instance, two training processes that each would take one hour on their own—on the same GPU, running sequentially—can run together with MPS. In this case, they would finish in a bit more than one hour total in parallel instead of two hours sequentially. The speedup from MPS can approach a near-doubling when kernels and memory bandwidth from concurrent clients complement one another. To visualize, imagine Process A and Process B each launching kernels periodically without MPS. The GPU schedule might look like A-B-A-B with gaps in between while each one waits, as shown in Figure 3-6.

举例来说，两个各自单独运行需要一小时的训练进程——在同一块 GPU 上顺序运行——借助 MPS 可以一起运行，并在略超过一小时的总时间内并行完成，而不是顺序运行的两小时。举例来说，两个各自单独运行需要一小时的训练进程——在同一块 GPU 上顺序运行——可以借助 MPS 一起运行。在这种情况下，它们会在略超过一小时的总时间内并行完成，而不是顺序运行的两小时。当来自并发客户端的内核和内存带宽彼此互补时，MPS 带来的加速可以接近翻倍。为便于形象化理解，设想进程 A 和进程 B 在没有 MPS 的情况下各自周期性地启动内核。GPU 的调度可能看起来像 A-B-A-B，其间夹着各自等待时的间隙，如图 3-6 所示。

![Figure 3-6. GPU alternates between running Process A’s kernels and Process B’s kernels and creates idle gaps in which one process is waiting while the other is active](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-6.png)

![Figure 3-6. GPU alternates between running Process A’s kernels and Process B’s kernels and creates idle gaps in which one process is waiting while the other is active](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-6.png)

_Figure 3-6. GPU alternates between running Process A’s kernels and Process B’s kernels and creates idle gaps in which one process is waiting while the other is active_

_图 3-6. GPU 在运行进程 A 的内核和进程 B 的内核之间来回交替，制造出空闲间隙——其间一个进程在等待，而另一个处于活动状态。_

With MPS, the schedule overlaps A and B so that whenever A isn’t using some parts of the GPU, B’s work can use them simultaneously, and vice versa. This overlapping eliminates idle gaps, as shown in Figure 3-7.

有了 MPS，调度会将 A 和 B 重叠起来，这样每当 A 没有在使用 GPU 的某些部分时，B 的工作就能同时使用它们，反之亦然。这种重叠消除了空闲间隙，如图 3-7 所示。

![Figure 3-7. Reducing idle gaps for processes A and B using MPS](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-7.png)

![Figure 3-7. Reducing idle gaps for processes A and B using MPS](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-7.png)

_Figure 3-7. Reducing idle gaps for processes A and B using MPS_

_图 3-7. 使用 MPS 减少进程 A 和 B 的空闲间隙。_

Setting up MPS involves running an MPS control daemon (`nvidia-cuda-mps-` `control`), which then launches an MPS server process that brokers GPU access. On modern GPUs, MPS is more streamlined as clients (the processes) can talk directly to the hardware with minimal interference from the compute node itself.

设置 MPS 需要运行一个 MPS 控制守护进程（`nvidia-cuda-mps-` `control`），它随后会启动一个 MPS 服务器进程来居中协调 GPU 访问。在现代 GPU 上，MPS 更为精简，因为客户端（即各进程）可以直接与硬件通信，而来自计算节点本身的干扰极小。

Typically, you start the MPS server on a node—often one per GPU or one per user—and then run your GPU jobs with an environment variable that connects them to MPS. All jobs under that server will share the GPU concurrently.

通常，你会在一个节点上启动 MPS 服务器——往往是每块 GPU 一个或每个用户一个——然后用一个把作业连接到 MPS 的环境变量来运行你的 GPU 作业。该服务器下的所有作业都将并发共享这块 GPU。

Another feature of MPS is the ability to set an active thread percentage per client. This limits how many SMs (GPU cores, essentially) a client can use. This can be useful if you want to guarantee quality of service (QoS) where two jobs, for example, each get at most 50% of the GPU’s execution resources. In this case, you can set `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50` to cap a client to about 50% of SM execution capacity. If not explicitly set, the jobs will just compete and use whatever GPU resources they can.

MPS 的另一项特性是能够为每个客户端设置活动线程百分比。这会限制一个客户端可以使用多少 SM（本质上就是 GPU 核心）。如果你想保证服务质量（quality of service，QoS）——例如让两个作业各自最多获得 GPU 执行资源的 50%——这会很有用。在这种情况下，你可以设置 `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=50`，将一个客户端限制在大约 50% 的 SM 执行能力上。如果没有显式设置，这些作业就会彼此竞争，能用到多少 GPU 资源就用多少。

Note that MPS does not partition GPU memory, so all processes will share the full GPU memory space. MPS is mainly about compute sharing and scheduling. The issue is that one process could request a massive amount of GPU RAM, cause an OOM error on the GPU, and result in terminating all of the other processes running on the GPU. This is very disruptive. Also, if one program saturates the GPU 100% on its own, MPS won’t magically make it go faster, as you can’t exceed 100% utilization. It’s beneficial only when individual jobs leave some slack that others can fill.

请注意，MPS 并不对 GPU 内存进行分区，因此所有进程都将共享完整的 GPU 内存空间。MPS 主要关乎计算共享与调度。问题在于，某个进程可能请求海量的 GPU RAM，在 GPU 上引发 OOM 错误，进而导致在该 GPU 上运行的所有其他进程被终止。这极具破坏性。此外，如果某个程序单凭自身就把 GPU 打满到 100%，MPS 也不会神奇地让它更快，因为你无法超过 100% 的利用率。只有当各个作业留出一些余量供其他作业填补时，MPS 才有益处。

Another limitation of MPS is that, by default, all MPS clients must run as the same Unix user since they share a context. In multiuser clusters, this means MPS is usually set up at the scheduler level such that only one user’s jobs share a GPU at a time. Otherwise, you can configure a system-wide MPS that’s shared by all users, but understand that the jobs are not isolated from a security standpoint.

MPS 的另一个限制是，默认情况下，所有 MPS 客户端必须以同一个 Unix 用户身份运行，因为它们共享一个上下文。在多用户集群中，这意味着 MPS 通常在调度器层面设置，使得任一时刻只有一个用户的作业共享一块 GPU。否则，你可以配置一个供所有用户共享的系统级 MPS，但要明白这些作业从安全角度看是没有相互隔离的。

Modern NVIDIA drivers support multiuser MPS so that processes from different Unix users can share a single MPS server. This improves usability but does not provide memory isolation. Prefer MIG when strong isolation is required. One specific alternative to MPS is a feature for time-slicing GPUs in Kubernetes. Time-slicing on Kubernetes allows the device plugin to schedule different pods on the same GPU by time. For instance, if you configure a single GPU with a time-slicing replication factor of four, four pods on that GPU can each receive a time share.

现代 NVIDIA 驱动支持多用户 MPS，使得来自不同 Unix 用户的进程可以共享单个 MPS 服务器。这提升了可用性，但并不提供内存隔离。当需要强隔离时，优先选择 MIG。MPS 的一个具体替代方案是 Kubernetes 中用于对 GPU 进行时间分片的一项特性。Kubernetes 上的时间分片允许设备插件按时间在同一块 GPU 上调度不同的 pod。例如，如果你把一块 GPU 配置为时间分片复制因子为四，那么该 GPU 上的四个 pod 就可以各自获得一份时间份额。

Kubernetes time-slicing is sort of an automated time-sharing algorithm that doesn’t require MPS. However, this doesn’t overlap execution. Instead, it just switches more rapidly than the default driver would. Time-slicing may be useful for interactive workloads where you prefer isolation at the cost of some idle time. For high-throughput jobs, overlapping with MPS or splitting the GPU with a MIG is usually better than fine-grained time-slicing, as discussed next.

Kubernetes 时间分片算是一种不需要 MPS 的自动化分时算法。然而，它并不重叠执行。相反，它只是比默认驱动切换得更快。对于你宁愿以一些空闲时间为代价换取隔离的交互式工作负载，时间分片可能有用。对于高吞吐量作业，如接下来所讨论的，用 MPS 重叠执行、或用 MIG 切分 GPU，通常比细粒度的时间分片更好。

### MIG

### MIG

Modern GPUs can be partitioned at the hardware level into multiple instances using MIG. MIG is a form of virtualization but done in hardware. This way, the overhead is very low—maybe a few percent—due to the loss of some flexibility.

借助 MIG，现代 GPU 可以在硬件层面被切分为多个实例。MIG 是一种虚拟化，但由硬件实现。因此，其开销非常低——也许只有百分之几——代价是损失了一部分灵活性。

If one instance is idle, it can’t lend its resources to another, as they are hard partitioned. MIG allows a GPU to be sliced into as many as seven smaller logical GPUs—each with its own dedicated portion of memory and compute units, or SMs, as shown in Figure 3-8.

如果某个实例处于空闲状态，它也无法把自己的资源借给另一个实例，因为它们是硬性分区的。MIG 允许把一块 GPU 切分成多达七个更小的逻辑 GPU——每个都拥有自己专属的一部分内存和计算单元（即 SM），如图 3-8 所示。

![Figure 3-8. Seven MIG slices on a modern GPU](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-8.png)

![Figure 3-8. Seven MIG slices on a modern GPU](AI%20Systems%20Performance%20Engineering-ch3_images/figure-3-8.png)

_Figure 3-8. Seven MIG slices on a modern GPU_

_图 3-8. 现代 GPU 上的七个 MIG 切片。_

By convention, NVIDIA’s MIG profile naming uses the prefix `<X>g to` denote the number of compute slices between 1 (min) and 7 (max) on modern GPUs. Each slice number represents a number of SM groups allocated to that partition. Each SM group is roughly a 1/7 slice of the total number of SMs.

按照惯例，NVIDIA 的 MIG 配置文件（profile）命名使用前缀 `<X>g` 来表示现代 GPU 上计算切片的数量，取值介于 1（最小）到 7（最大）之间。每个切片编号代表分配给该分区的若干 SM 组。每个 SM 组大致是全部 SM 总数的 1/7 切片。

If a GPU has 132 SMs, each 1/7 slice represents 132 SMs × 1/7 = ~19 SMs in a group. As such, `1g` represents ~19 SMs, `2g` represents ~38 SMs, all the way up to 7g, which represents the total of ~132 SMs.

如果一块 GPU 有 132 个 SM，那么每个 1/7 切片就代表一组 132 SM × 1/7 = ~19 个 SM。因此，`1g` 代表约 19 个 SM，`2g` 代表约 38 个 SM，一直到 7g，代表全部约 132 个 SM。

In contrast, and somewhat confusingly, the suffix `<Y>gb` specifies the exact amount of HBM GPU RAM in gigabytes that is reserved for that profile. The MIG profile values are fixed for each GPU generation and type and listed in the NVIDIA documentation. For the Blackwell B200, some of the MIG profile values are shown in Table 3-1.

与之相对（也有些令人困惑）的是，后缀 `<Y>gb` 则指定为该配置文件预留的 HBM GPU 显存的确切大小（以 GB 为单位）。MIG 配置文件的取值对每一代、每种型号的 GPU 都是固定的，并在 NVIDIA 文档中列出。对于 Blackwell B200，部分 MIG 配置文件取值如表 3-1 所示。

Table 3-1. MIG Profiles for Blackwell B200 (source: https://oreil.ly/FsPEx)

表 3-1. Blackwell B200 的 MIG 配置文件（来源：https://oreil.ly/FsPEx）

The table also shows the number of hardware units, copy engines, and L2 cache fractions for each profile. These fixed profiles align with the GPU’s hardware memory controllers such that each memory slice maps to contiguous HBM channels.

该表还列出了每种配置文件（profile）的硬件单元数量、拷贝引擎（copy engine）数量以及 L2 缓存占比。这些固定的配置文件与 GPU 的硬件内存控制器对齐，使得每个显存切片都映射到连续的 HBM 通道。

This two-part scheme separates compute capacity (number of SM groups) from memory capacity (total GB). Administrators can choose combinations of MIG profiles. The sum of allocated SMs and HBM does not need to exactly match the full GPU capacity. However, certain combinations are constrained by hardware partitioning. They cannot invent new slice sizes, for instance.

这种两段式命名方案将计算容量（SM 组的数量）与显存容量（总 GB 数）区分开来。管理员可以选择多种 MIG 配置文件的组合。所分配的 SM 与 HBM 之和无需精确匹配整块 GPU 的容量。不过，某些组合会受到硬件分区方式的约束。例如，管理员无法凭空创造出新的切片规格。

Administrators can enable or disable only the supported MIG profiles (e.g., `1g.23gb`, `2g.45gb`, `4g.90gb`, etc.) on each GPU using tools like `nvidia-smi -mig` or using the NVIDIA Kubernetes GPU Operator’s `nvidia.com/mig.config` config map. Reconfiguring MIG requires draining workloads and invoking MIG’s dynamic reconfiguration capability to apply the changes.

管理员可以使用 `nvidia-smi -mig` 之类的工具，或使用 NVIDIA Kubernetes GPU Operator 的 `nvidia.com/mig.config` config map，在每块 GPU 上仅启用或禁用受支持的 MIG 配置文件（例如 `1g.23gb`、`2g.45gb`、`4g.90gb` 等）。重新配置 MIG 需要先腾空（drain）工作负载，再调用 MIG 的动态重配置能力来应用变更。

Once a GPU is in MIG mode, modern GPUs can create and destroy MIG partitions dynamically without rebooting the entire system. You can adjust MIG instances on the fly after draining existing workloads, but to enable or disable MIG mode itself on a GPU, a reset of that GPU is needed.

一旦 GPU 进入 MIG 模式，现代 GPU 便可动态地创建和销毁 MIG 分区，而无需重启整个系统。你可以在腾空现有工作负载后即时调整 MIG 实例，但要在某块 GPU 上启用或禁用 MIG 模式本身，则需要重置该 GPU。

Each MIG instance acts like a separate GPU from the perspective of software since it has its own memory, its own SMs, and even separate engine contexts. The benefit of MIG is strong isolation and guaranteed resources for each job. If you have multiple users or multiple services that need only, say, 10 GB of logical GPU memory each, you can pack them onto one physical GPU without them interfering with one another’s memory or compute.

从软件的视角看，每个 MIG 实例都像一块独立的 GPU，因为它拥有自己的显存、自己的 SM，甚至独立的引擎上下文。MIG 的好处在于为每个作业提供强隔离和有保障的资源。如果你有多个用户或多个服务，各自只需要（比如说）10 GB 的逻辑 GPU 显存，你就可以把它们打包到同一块物理 GPU 上，而它们之间不会互相干扰彼此的显存或计算。

Jobs can request MIG devices specifically, but you have to be careful to schedule in a way that uses all slices. For instance, if you have a 7-slice setup and a job takes only 1 slice, the other 6 should be packed with other jobs, or you’re leaving a lot idle. It’s possible to configure certain nodes in your cluster to use MIG for small inference jobs, for example, and configure other nodes for non-MIG workloads for large training jobs.

作业可以专门请求 MIG 设备，但你必须谨慎调度，确保用满所有切片。举例来说，如果你配置了 7 个切片，而某个作业只占用 1 个切片，那么其余 6 个切片就应当被其他作业填满，否则你就会让大量资源闲置。例如，你可以把集群中的某些节点配置为使用 MIG 来运行小型推理作业，而把另一些节点配置为运行大型训练作业的非 MIG 工作负载。

One important operational note is that, in order to use MIG, you typically configure it at the system level—or at least at the node level. The GPU has to be put into MIG mode, the slices created, and the GPU reset. Once these happen, the slices appear as separate devices to the system—each with their own unique device ID.

一个重要的运维注意事项是：要使用 MIG，你通常需要在系统层面（至少在节点层面）进行配置。必须将 GPU 置入 MIG 模式、创建切片并重置 GPU。完成这些步骤后，这些切片会作为独立设备呈现给系统——每个切片都有自己唯一的设备 ID。

If you’re not using all of the available MIG slices because of poor upfront planning, you’ll end up wasting resources by leaving them fragmented and unused. It’s important to plan partition sizes upfront to match your workloads—and adjust the partition sizes when the workload changes. You will need to reset the GPUs to pick up the change.

如果由于前期规划不当而没有用满所有可用的 MIG 切片，你就会因为让它们处于碎片化和未使用状态而浪费资源。重要的是要提前规划分区大小以匹配你的工作负载——并在工作负载发生变化时调整分区大小。你需要重置 GPU 才能使变更生效。

For large-scale model training jobs and inference servers that span many GPUs, MIG is typically not useful since we want access to the full set of GPUs. On the other hand, for multitenant, small-model inference servers that can run smaller GPU partitions, MIG and its isolation features could be useful.

对于跨越多块 GPU 的大规模模型训练作业和推理服务器而言，MIG 通常并无用处，因为我们希望访问全部 GPU。另一方面，对于可以在较小 GPU 分区上运行的多租户小模型推理服务器而言，MIG 及其隔离特性可能会很有用。

> As of this writing, when a GPU is in MIG mode, GPU-to-GPU peer-to-peer communication (including NVLink) is disabled. This applies to both NVLink and PCIe P2P across GPUs. MIG instances cannot engage in P2P with other GPUs. CUDA IPC across MIG instances is also limited. This can reduce distributed training throughput. Confirm that your training or inference topology does not rely on GPU peer paths before enabling MIG. Large-scale training jobs (and sparse MoE expert inference systems) require massive GPU-to-GPU communication and are typically not good candidates for MIG. Communication between GPU MIG instances must travel through either the host or network fabric.

> 在撰写本书时，当 GPU 处于 MIG 模式时，GPU 到 GPU 的点对点（peer-to-peer）通信（包括 NVLink）会被禁用。这同时适用于跨 GPU 的 NVLink 和 PCIe P2P。MIG 实例无法与其他 GPU 进行 P2P。跨 MIG 实例的 CUDA IPC 也受到限制。这会降低分布式训练的吞吐量。在启用 MIG 之前，请确认你的训练或推理拓扑不依赖 GPU 点对点路径。大规模训练作业（以及稀疏 MoE 专家推理系统）需要海量的 GPU 到 GPU 通信，通常不适合使用 MIG。GPU MIG 实例之间的通信必须经由主机或网络 fabric 传输。

In short, enable MIG only when you need to run multiple independent jobs on the same GPU with strong isolation. Do not use MIG for large-scale distributed training or inferencing that spans GPUs, as you want access to the full power of the GPUs and their fast interconnects.

简而言之，只有当你需要在同一块 GPU 上运行多个相互独立、且需要强隔离的作业时，才启用 MIG。不要在跨 GPU 的大规模分布式训练或推理中使用 MIG，因为在那种场景下你希望获得 GPU 的全部算力及其高速互连。

In our context of large transformer-based model training and inferencing, we will leave MIG off. But it’s good to know that this feature exists. Perhaps a cluster might dynamically switch modes and run MIG during the day when lots of small training or inferencing experiments are happening, then turn MIG off at night to run big training jobs that use whole GPUs.

在我们讨论的大型基于 transformer 的模型训练与推理场景中，我们会关闭 MIG。不过，了解这个特性的存在是有益的。也许某个集群可以动态切换模式：白天有大量小型训练或推理实验运行时启用 MIG，夜间关闭 MIG 以运行使用整块 GPU 的大型训练作业。

> The Kubernetes device plugin will list MIG devices as resources like `nvidia.com/mig-1g.23gb` in the case of a 1/7 GPU slice with a total 23 GB for the slice.

> 对于 1/7 的 GPU 切片（该切片共 23 GB），Kubernetes 设备插件会将 MIG 设备列为形如 `nvidia.com/mig-1g.23gb` 的资源。

### GPU Clock Speeds and ECC

### GPU 时钟频率与 ECC

NVIDIA GPUs have something called GPU Boost, which automatically adjusts the core clock within power and thermal limits. Most of the time, you should let the GPU just do its thing. But some users like to lock the clocks for consistency so that the GPU always runs at a fixed maximum frequency. This way, run-to-run performance is stable and not subject to variations in power or temperature.

NVIDIA GPU 具有一项名为 GPU Boost 的功能，它会在功耗和热量限制范围内自动调整核心时钟。大多数情况下，你应该让 GPU 自行处理。但有些用户喜欢锁定时钟以获得一致性，让 GPU 始终运行在固定的最高频率下。这样一来，多次运行之间的性能就会保持稳定，不受功耗或温度波动的影响。

Fixing the clock is extremely important when performing benchmarks since later runs may be throttled due to excessive heat. If you do not account for this, you may inadvertently interpret the poor results of later runs incorrectly since the GPUs of these subsequent runs may be throttled due to excessive heat caused by previous runs.

在进行基准测试时，固定时钟极为重要，因为后续的运行可能会因过热而被降频。如果不考虑这一点，你可能会错误地解读后续运行的糟糕结果，因为这些后续运行的 GPU 可能由于先前运行产生的过多热量而被降频。

Specifically, NVIDIA’s GPU Boost will vary the core clock up or down to stay within power/thermal limits. Locking the clock at the max stable frequency using `nvidia-` `smi -lgc` to lock the core clock and `-ac` to lock the memory clock. This will make sure the GPU runs at a constant frequency—and prevents the GPU’s Boost default functionality from downclocking in later runs.

具体来说，NVIDIA 的 GPU Boost 会上下调整核心时钟以保持在功耗/热量限制范围内。使用 `nvidia-smi -lgc` 将核心时钟锁定在最高稳定频率，并使用 `-ac` 锁定显存时钟。这样可以确保 GPU 以恒定频率运行——并防止 GPU 的 Boost 默认行为在后续运行中降低时钟频率。

> This is mostly relevant during benchmarking to achieve deterministic and reproducible results. For everyday training and inferencing, it’s recommended to leave auto-boost on unless you notice significant performance variance and GPU throttling.

> 这主要与基准测试相关，用于获得确定且可复现的结果。对于日常训练和推理，建议保持自动 Boost 开启，除非你发现明显的性能波动和 GPU 降频。

Locking the clocks is something to be aware of if you’re chasing the last bit of determinism and consistency. Typically, however, leaving the GPU in the default auto-boost mode is fine.

如果你在追求最后一点确定性和一致性，那么锁定时钟是需要留意的事情。不过通常来说，让 GPU 保持默认的自动 Boost 模式就足够了。

Some teams purposely underclock the GPUs to reduce heat—especially if they are running very long jobs and don’t want to suffer the eventual thermal slowdown over time. Data center GPUs typically have enough temperature headroom—as well as proper air and liquid cooling—so that you don’t need to do this, but it’s good to know that it’s an option.

有些团队会有意为 GPU 降频以减少发热——尤其是当他们运行非常长的作业、又不想随时间推移最终遭遇热降频时。数据中心级 GPU 通常有足够的温度余量——以及适当的风冷和液冷——因此你无需这样做，但了解这也是一种可选方案是有益的。

Another approach is to use `nvidia-smi -pl` to set a power limit slightly below the maximum thermal design power (TDP) of the GPU. TDP is the maximum amount of heat, measured in watts, that a GPU can generate under sustained load. This dictates the amount of heat that must be dissipated to prevent overheating.

另一种方法是使用 `nvidia-smi -pl` 将功耗上限设置为略低于 GPU 的最大热设计功耗（thermal design power，TDP）。TDP 是 GPU 在持续负载下所能产生的最大热量，以瓦特为单位度量。它决定了为防止过热而必须散发的热量。

If you set the power limit below TDP, the GPU Boost will auto-adjust clocks below the thermal throttle point. This can reduce peak heat generation, prevent throttling, and incur minimal performance impact.

如果你把功耗上限设置在 TDP 以下，GPU Boost 会自动将时钟调整到热降频点以下。这可以减少峰值发热、防止降频，同时对性能的影响微乎其微。

ECC memory on GPUs is another consideration. ECC ensures that if there’s a single-bit memory error caused by cosmic rays, for example, the memory can be corrected on the fly. And if there’s a double-bit error, the error is detected and will throw an error to the calling code. ECC is usually enabled by default on NVIDIA data center GPUs.

GPU 上的 ECC 显存是另一个需要考量的因素。ECC 可确保：如果发生了（例如由宇宙射线引起的）单比特内存错误，该内存可被即时纠正。而如果发生了双比特错误，该错误会被检测到，并向调用代码抛出错误。在 NVIDIA 数据中心级 GPU 上，ECC 通常默认启用。

Disabling ECC can free up a small amount of memory since ECC requires extra bits for error checking. This might yield a marginal performance gain by reducing the overhead associated with on-the-fly error checking, but typically just a few percent. However, turning off ECC also removes critical memory-error protection, which can lead to system instability or undetected data corruption.

禁用 ECC 可以释放少量显存，因为 ECC 需要额外的比特位来做错误校验。这可能会通过减少即时错误校验带来的开销而带来微小的性能提升，但通常只有几个百分点。然而，关闭 ECC 也会移除关键的内存错误保护，可能导致系统不稳定或未被察觉的数据损坏。

For NVIDIA’s data center GPUs, including Hopper and Blackwell, ECC comes enabled by default and is intended to remain enabled to ensure reliable, error-corrected computation and data integrity. For long training or inference jobs on huge models, a single memory error could crash the job completely or, even worse, silently corrupt your model without a warning.

对于 NVIDIA 的数据中心级 GPU（包括 Hopper 和 Blackwell），ECC 默认启用，并且应保持启用，以确保可靠的、经过纠错的计算和数据完整性。对于在超大模型上运行的长时间训练或推理作业，一个内存错误就可能彻底使作业崩溃，或者更糟——在没有任何警告的情况下悄悄损坏你的模型。

It’s recommended to always keep ECC on for any serious AI workload. The only time you’d possibly consider turning it off is in a research setting where you are fine with taking the risk because you need that extra sliver of memory for your model to fit into your limited-memory GPU cluster.

建议在任何严肃的 AI 工作负载中始终保持 ECC 开启。唯一可能考虑关闭它的场合，是在研究环境中，你愿意承担这一风险，因为你需要那一点额外的显存才能让模型塞进你显存有限的 GPU 集群里。

Toggling ECC mode requires resetting the GPU and likely restarting jobs that are currently running on that GPU. So it’s not a toggle that you want to switch frequently. Keep ECC on for stability and reliability. The peace of mind outweighs the negligible speedup of turning ECC off.

切换 ECC 模式需要重置 GPU，并且很可能需要重启当前正在该 GPU 上运行的作业。因此这不是一个你想频繁切换的开关。为了稳定性和可靠性，请保持 ECC 开启。这份安心远胜于关闭 ECC 所带来的微不足道的加速。

### GPU Memory Oversubscription, Fragmentation, and Out-of-Memory Handling

### GPU 显存超额订阅、碎片化与内存不足处理

Unlike CPU RAM, by default there is no such thing as GPU “swap” memory. If you try to allocate more GPU memory than available, you will get an unfriendly OOM error along with an even-unfriendlier process crash. There are a couple of mechanisms to mitigate this issue: allow memory to grow dynamically, embrace unified memory across CPU and GPU, and utilize memory pools and caching allocators.

与 CPU 内存不同，默认情况下并不存在 GPU“交换（swap）”内存这种东西。如果你试图分配比可用显存更多的 GPU 显存，你会收到一个不友好的 OOM 错误，以及一个更不友好的进程崩溃。有几种机制可以缓解这一问题：允许显存动态增长、跨 CPU 与 GPU 采用统一内存，以及利用内存池和缓存分配器。

By default, some frameworks (e.g., TensorFlow) grab all of the available GPU memory at startup to avoid fragmentation and improve performance. If you don’t know this, it can be very bad in scenarios where you are sharing the GPU. PyTorch, by default, allocates GPU memory only as needed.

默认情况下，某些框架（例如 TensorFlow）会在启动时抢占所有可用的 GPU 显存，以避免碎片化并提升性能。如果你不了解这一点，在共享 GPU 的场景下可能会非常糟糕。而 PyTorch 默认只按需分配 GPU 显存。

> TensorFlow has an option (`TF_FORCE_GPU_ALLOW_GROWTH=true`) to make it start small

> TensorFlow 提供了一个选项（`TF_FORCE_GPU_ALLOW_GROWTH=true`），让它从占用少量显存开始，

and dynamically grow the GPU memory usage as needed—similar to PyTorch. However, neither PyTorch nor TensorFlow lets you allocate more memory than the GPU has available. But this lazy allocation plays nicer in multitenant scenarios because two processes won’t both try to simultaneously allocate the maximum available GPU memory from the start.

并按需动态增长 GPU 显存用量——与 PyTorch 类似。然而，无论是 PyTorch 还是 TensorFlow，都不允许你分配超过 GPU 实际拥有的显存。但这种惰性分配在多租户场景下表现得更友好，因为两个进程不会都从一开始就试图同时抢占最大可用的 GPU 显存。

CUDA’s Unified Memory system lets you allocate memory without predefining whether it resides on the CPU or GPU. The CUDA Runtime handles moving pages as needed. Modern NVIDIA GPUs like Hopper and Blackwell include hardware support for on-demand paging using the Page Migration Engine (PME).

CUDA 的统一内存（Unified Memory）系统让你无需预先定义内存驻留在 CPU 还是 GPU 上即可进行分配。CUDA 运行时会按需搬移页面。像 Hopper 和 Blackwell 这样的现代 NVIDIA GPU 内置了硬件支持，可通过页迁移引擎（Page Migration Engine，PME）实现按需分页。

PME automatically migrates memory pages between GPU memory and host CPU RAM when the GPU runs low on available memory. However, while PME provides flexibility, relying on it can introduce performance penalties compared to having enough GPU memory for your workload.

当 GPU 的可用显存不足时，PME 会自动在 GPU 显存与主机 CPU 内存之间迁移内存页。然而，尽管 PME 提供了灵活性，但与为工作负载配备足够的 GPU 显存相比，依赖它可能会带来性能损失。

This GPU-to-CPU memory offloading can be slow, however, since CPU memory I/O is slower than GPU high-bandwidth memory (HBM) I/O, as we learned in Chapter 2. This mechanism is mostly a convenience for practitioners trying to run models that don’t fit into GPU RAM.

不过，这种 GPU 到 CPU 的内存卸载可能会很慢，因为正如我们在第 2 章所学，CPU 内存 I/O 慢于 GPU 高带宽内存（HBM）I/O。这一机制主要是为那些试图运行放不进 GPU 内存的模型的从业者提供便利。

For performance-critical workloads, you generally want to avoid relying on unified memory oversubscription where possible. It’s there as a safety net instead of outright crashing your script, but your job will run slower when GPU memory is oversubscribed.

对于性能关键型工作负载，你通常应尽可能避免依赖统一内存的超额订阅。它的存在是作为一张安全网，避免直接让脚本崩溃，但当 GPU 显存被超额订阅时，你的作业会运行得更慢。

Libraries like PyTorch use a caching allocator so that when you free GPU memory, it doesn’t return the memory to the OS immediately. Instead, it keeps it to reuse for future allocations. This avoids memory fragmentation and the overhead of asking the OS to repeatedly allocate the same block of memory.

像 PyTorch 这样的库使用缓存分配器（caching allocator），因此当你释放 GPU 显存时，它并不会立即将显存归还给操作系统。相反，它会保留这块显存以便将来复用。这可以避免内存碎片化，以及反复请求操作系统分配同一块内存所带来的开销。

You can configure PyTorch’s allocator using environment variables like `PYTORCH_ALLOC_CONF` (formerly `PYTORCH_CUDA_ALLOC_CONF`) to set a max pool size. We’ll cover optimizations to PyTorch’s memory-allocation mechanism in a later chapter.

你可以使用诸如 `PYTORCH_ALLOC_CONF`（原为 `PYTORCH_CUDA_ALLOC_CONF`）之类的环境变量来配置 PyTorch 的分配器，以设置最大内存池大小。我们将在后面的章节中介绍针对 PyTorch 显存分配机制的优化。

If you run into the GPU OOM error, which you surely will at some point, it’s likely caused by memory fragmentation or excessive memory caching. You can try to clear the cache using PyTorch’s `torch.cuda.empty_cache()`, but it almost always means your workload legitimately needs that much memory.

如果你遇到 GPU OOM 错误（你迟早一定会遇到），它很可能是由内存碎片化或过度的内存缓存引起的。你可以尝试使用 PyTorch 的 `torch.cuda.empty_cache()` 来清空缓存，但这几乎总是意味着你的工作负载确实需要那么多显存。

PyTorch also provides tools like `torch.cuda.memory_stats()` and `torch.cuda.mem` `ory_summary()` to help diagnose fragmentation by showing allocated versus reserved memory. NVIDIA’s Nsight Systems also shows GPU memory usage patterns to help identify memory leaks, long-lived allocations that correlate with leaks, CPU-GPU interconnect activity, and GPUDirect Storage timeline tracing. Additionally, the Nsight Compute profiler provides low-level kernel analysis, including occupancy, throughput, and NVLink usage. We’ll cover all of these in the upcoming chapters.

PyTorch 还提供了 `torch.cuda.memory_stats()` 和 `torch.cuda.memory_summary()` 之类的工具，通过显示已分配显存与已预留显存来帮助诊断碎片化问题。NVIDIA 的 Nsight Systems 也能展示 GPU 显存使用模式，帮助识别内存泄漏、与泄漏相关的长期存活分配、CPU-GPU 互连活动，以及 GPUDirect Storage 时间线追踪。此外，Nsight Compute 分析器提供低层次的内核分析，包括占用率（occupancy）、吞吐量和 NVLink 使用情况。我们将在后续章节中一一介绍这些内容。

Docker provides the `--gpus` flag to select and expose GPUs to a container, but it does not support setting a GPU memory limit. If you need hard isolation for GPU memory or compute, use MIG to partition the device or use Multi-Process Service (MPS) with active thread percentage for fair sharing. Configure limits in Kubernetes using MIG resources like `nvidia.com/mig-2g.45gb` when you require strict partitioning.

Docker 提供了 `--gpus` 标志来选择并向容器暴露 GPU，但它不支持设置 GPU 显存上限。如果你需要对 GPU 显存或计算进行硬性隔离，请使用 MIG 来对设备分区，或使用带有活动线程百分比的多进程服务（MPS）来实现公平共享。当你需要严格分区时，可在 Kubernetes 中使用诸如 `nvidia.com/mig-2g.45gb` 之类的 MIG 资源来配置限制。

In multitenant nodes, this could be useful to isolate jobs. In a single-job-per-GPU situation, it’s not common to set a memory limit, as you want to let the job use as much of the GPU memory as it can get.

在多租户节点上，这种做法有助于隔离作业。而在每块 GPU 只运行单个作业的情况下，通常不会设置显存上限，因为你希望让作业尽可能多地占用 GPU 显存。

In general, running out of GPU memory is something you can manage at the application level. For instance, you can reduce the data batch size, model weight precision, or even the model parameter count, if that’s an option.

总体而言，GPU 显存耗尽是可以在应用层管理的问题。例如，你可以减小数据批大小、模型权重精度，甚至在可行的情况下减少模型参数量。

A best practice is to monitor GPU memory usage with `nvidia-smi` or NVML APIs during model training and inferencing. If you’re close to the memory limit, consider workarounds like reducing batch size, using activation checkpointing for training, or other techniques to lower memory usage.

一项最佳实践是在模型训练和推理期间，使用 `nvidia-smi` 或 NVML API 监控 GPU 显存使用情况。如果你接近显存上限，可以考虑一些变通办法，例如减小批大小、在训练中使用激活检查点（activation checkpointing），或其他降低显存占用的技术。

Also, you should ensure that your CPU memory isn’t being swapped, as this would indirectly hurt your GPU utilization and goodput because each time your GPU tries to fetch something from the CPU host, but the host memory page has been swapped to disk, your performance will be bottlenecked by the much slower disk I/O. So it’s important to combine these memory-reduction best practices with the earlier advice about pinning memory, increasing the `ulimit`, and disabling swappiness, etc.

此外，你应确保 CPU 内存没有被交换出去，因为这会间接损害你的 GPU 利用率和有效吞吐量（goodput）：每当 GPU 试图从 CPU 主机取回某些数据，而该主机内存页却已被交换到磁盘时，你的性能就会被慢得多的磁盘 I/O 所拖累。因此，将这些减少内存占用的最佳实践与前文关于固定内存、提高 `ulimit`、禁用 swappiness 等的建议结合起来非常重要。

In short, it’s recommended to always keep the GPU driver loaded instead of unloading the GPU driver between jobs. This is similar to GPU persistence mode but at a deeper level. Some clusters are configured to unload the driver when no jobs are running in order to free OS kernel memory and for security. However, if you do that, the next job has to pay the cost of reloading the GPU driver and, if MIG is used, reconfiguring MIG slices.

简而言之，建议始终保持 GPU 驱动处于加载状态，而不要在作业之间卸载 GPU 驱动。这与 GPU 持久化模式（persistence mode）类似，但作用层次更深。有些集群被配置为在没有作业运行时卸载驱动，以释放操作系统内核内存并出于安全考虑。然而，如果你这样做，下一个作业就必须承担重新加载 GPU 驱动的开销，而且如果使用了 MIG，还要重新配置 MIG 切片。

> It’s recommended to keep the driver and any MIG configuration persistent across jobs. The only time you want to unload the GPU driver is for troubleshooting or upgrading the driver. As such, cluster admins often set up the system so that the NVIDIA driver modules are always present once the machine boots.

> 建议让驱动和任何 MIG 配置在作业之间保持持久化。你唯一想要卸载 GPU 驱动的时机，是排查故障或升级驱动的时候。因此，集群管理员通常会把系统设置为：机器一旦启动，NVIDIA 驱动模块就始终存在。

## Container Runtime Optimizations for GPUs

## 面向 GPU 的容器运行时优化

Many AI systems use orchestration tools and container runtimes to manage the software environment. Kubernetes and Docker are popular in AI infrastructure. Using containers ensures that all dependencies, including CUDA and library versions, are consistent. This avoids the “but it works on my machine” problem. Containers introduce a bit of complexity and a tiny amount of overhead, but with the right configuration, you can get near bare-metal performance for GPU workloads using containers.

许多 AI 系统使用编排工具和容器运行时来管理软件环境。Kubernetes 和 Docker 在 AI 基础设施中很受欢迎。使用容器可以确保所有依赖（包括 CUDA 和各种库的版本）保持一致。这就避免了“但它在我机器上能跑”的问题。容器会引入一点复杂性和极小的开销，但只要配置得当，你就能用容器为 GPU 工作负载获得接近裸机的性能。

A container running on a node is not a traditional virtual machine (VM). In contrast to VMs, containers share the host OS kernel so that CPU and memory operations perform at near-native speed. And with the NVIDIA Container Toolkit, GPU access from within a Docker container is direct and does not incur overhead.

运行在节点上的容器并不是传统意义上的虚拟机（virtual machine，VM）。与虚拟机不同，容器共享宿主机的操作系统内核，因此 CPU 和内存操作能以接近原生的速度执行。而借助 NVIDIA Container Toolkit，在 Docker 容器内部访问 GPU 是直接的，不会产生开销。

> For modern GPUs running with the latest NVIDIA Container Toolkit, GPU performance within a properly configured environment is virtually identical (< 2% difference) to running the code directly on the bare-metal host outside of the container. In fact, Red Hat OpenShift and Kubernetes were used in the MLPerf Inference v5.0 results, which demonstrates that modern containers and orchestration configuration do not compromise efficiency or latency.

> 对于运行最新 NVIDIA Container Toolkit 的现代 GPU，在配置得当的环境中，容器内的 GPU 性能与在容器之外的裸机宿主上直接运行代码的性能几乎相同（差异 < 2%）。事实上，MLPerf Inference v5.0 的结果中就使用了 Red Hat OpenShift 和 Kubernetes，这表明现代容器和编排配置不会损害效率或延迟。

### NVIDIA Container Toolkit and CUDA Compatibility

### NVIDIA Container Toolkit 与 CUDA 兼容性

One challenge when using containers with GPUs is making sure that the CUDA libraries inside the container match the driver on the host. NVIDIA solves this through their Container Toolkit and base Docker images. The host provides the NVIDIA driver, which, remember, is tightly integrated with the kernel and hardware. Inside the container, you typically find the CUDA runtime libraries of a certain version.

在将容器与 GPU 一起使用时，一个挑战是确保容器内部的 CUDA 库与宿主机上的驱动相匹配。NVIDIA 通过其 Container Toolkit 和基础 Docker 镜像解决了这个问题。宿主机提供 NVIDIA 驱动——请记住，它与内核和硬件紧密集成。在容器内部，你通常会找到某个特定版本的 CUDA 运行时库。

The general rule is that the host’s NVIDIA driver version must be at least as recent as the minimum driver version required by the CUDA version inside the container. For CUDA 13.x, the minimum required Linux host driver branch is R580 or newer. For CUDA 12.x, the minimum required Linux host driver branch is R525 or newer. Using a newer CUDA runtime with an older driver will cause the CUDA initialization to fail.

一般规则是：宿主机的 NVIDIA 驱动版本必须至少与容器内 CUDA 版本所要求的最低驱动版本一样新。对于 CUDA 13.x，所需的最低 Linux 宿主机驱动分支为 R580 或更新版本。对于 CUDA 12.x，所需的最低 Linux 宿主机驱动分支为 R525 或更新版本。用较新的 CUDA 运行时搭配较旧的驱动，会导致 CUDA 初始化失败。

> Each new CUDA version requires a minimum NVIDIA driver version. Always consult NVIDIA’s official compatibility matrix and upgrade the host driver when you update the CUDA toolkit.

> 每个新的 CUDA 版本都要求一个最低的 NVIDIA 驱动版本。请始终查阅 NVIDIA 官方兼容性矩阵，并在更新 CUDA 工具包时升级宿主机驱动。

For Docker and Kubernetes environments, the simplest approach is to use NVIDIA’s official base Docker images from the NVIDIA GPU Cloud (NGC) or DockerHub image repositories. These images (e.g., `nvcr.io/nvidia/pytorch` or similar) bundle the proper versions of the CUDA runtime, cuDNN, NCCL, etc. In addition, these Docker images list the minimum required CUDA driver, depending on the CUDA version. This way, you get support for the latest hardware without dependency headaches.

对于 Docker 和 Kubernetes 环境，最简单的方法是使用来自 NVIDIA GPU Cloud（NGC）或 DockerHub 镜像仓库的 NVIDIA 官方基础 Docker 镜像。这些镜像（例如 `nvcr.io/nvidia/pytorch` 或类似镜像）打包了正确版本的 CUDA 运行时、cuDNN、NCCL 等。此外，这些 Docker 镜像会根据 CUDA 版本列出所需的最低 CUDA 驱动。这样，你无需为依赖问题头疼，就能获得对最新硬件的支持。

### NVIDIA Container Runtime

### NVIDIA Container Runtime

Alternatively, NVIDIA’s container runtime can actually inject the host driver libraries into the container at runtime, so you don’t even need to ship the NVIDIA driver inside the image. Instead, you just rely on the host’s driver. Again, this works because the container isn’t fully isolated like a traditional VM. Docker containers are allowed to use host devices, volumes, and libraries.

另一种方案是，NVIDIA 的容器运行时实际上可以在运行时将宿主机的驱动库注入到容器中，因此你甚至无需在镜像内部附带 NVIDIA 驱动。相反，你只需依赖宿主机的驱动。同样，这之所以可行，是因为容器并不像传统虚拟机那样完全隔离。Docker 容器被允许使用宿主机的设备、卷和库。

Inside the container, your application uses the CUDA runtime libraries, such as `libcudart.so` from the container image, while the NVIDIA Container Toolkit injects the host’s driver libraries such as `libcuda.so` and `libnvidia-ml.so` at container start. The host driver libraries are invoked directly on the host so that everything just works.

在容器内部，你的应用使用来自容器镜像的 CUDA 运行时库（例如 `libcudart.so`），而 NVIDIA Container Toolkit 会在容器启动时注入宿主机的驱动库，例如 `libcuda.so` 和 `libnvidia-ml.so`。宿主机的驱动库直接在宿主机上被调用，因此一切都能正常工作。

The split between CUDA runtime libraries (container) and NVIDIA Container Toolkit (host) is supported as long as the host driver meets the minimum version required by the CUDA Toolkit in the image. If you were to mismatch and try to use a newer CUDA version in the container with an old driver on the host, you’d likely get an error. It’s important to match the CUDA and driver versions.

只要宿主机驱动满足镜像中 CUDA 工具包所要求的最低版本，CUDA 运行时库（在容器中）与 NVIDIA Container Toolkit（在宿主机上）之间的这种拆分就是受支持的。如果你版本不匹配，试图在容器中使用较新的 CUDA 版本、而宿主机上却是较旧的驱动，你很可能会遇到错误。匹配 CUDA 与驱动版本非常重要。

The key takeaway is that there is no hypervisor or virtualization layer involved when using containers for GPUs. The container is sharing the host kernel and driver directly, so when a kernel launches on the GPU, it’s as if it launched from the host.

关键要点是：将容器用于 GPU 时，并不涉及任何 hypervisor 或虚拟化层。容器直接共享宿主机的内核和驱动，因此当一个内核在 GPU 上启动时，就仿佛是从宿主机启动的一样。

In other words, you aren’t losing performance to Docker-based virtualization—unless you are using something like VMware or Single Root Input/Output Virtualization (SR-IOV) virtual GPUs, which is a special scenario that requires some tuning. With Docker plus NVIDIA, it’s basically the equivalent of bare metal performance.

换句话说，你不会因为基于 Docker 的虚拟化而损失性能——除非你使用了诸如 VMware 或单根输入/输出虚拟化（Single Root Input/Output Virtualization，SR-IOV）虚拟 GPU 之类的东西，那是一种需要一些调优的特殊场景。有了 Docker 加 NVIDIA，其性能基本等同于裸机。

> The NVIDIA Container Toolkit works with containerd and Podman as well, not only Docker. This is relevant for modern Kubernetes environments that use containerd as the default container runtime.

> NVIDIA Container Toolkit 不仅适用于 Docker，也适用于 containerd 和 Podman。这一点与使用 containerd 作为默认容器运行时的现代 Kubernetes 环境相关。

### Avoiding Container Overlay Filesystem Overhead

### 避免容器叠加文件系统的开销

The main difference when running in a Docker container versus running directly on the host might be in I/O. Containers often use a union filesystem that transparently overlays multiple underlying filesystems, like the host filesystem and the container filesystem, into a single, unified view.

在 Docker 容器中运行与直接在宿主机上运行的主要差异，可能在于 I/O。容器通常使用联合文件系统（union filesystem），它以透明的方式将多个底层文件系统（如宿主机文件系统和容器文件系统）叠加成单一的、统一的视图。

In a union filesystem such as OverlayFS, files and directories from multiple sources will appear as if they belong to one filesystem. This mechanism is especially useful for containers, where the read-only filesystem from the base image layer is combined with a writable container layer.

在诸如 OverlayFS 这样的联合文件系统中，来自多个来源的文件和目录看起来就像属于同一个文件系统。这一机制对容器尤其有用：来自基础镜像层的只读文件系统与一个可写的容器层结合在一起。

There is some overhead when using an overlay filesystem, however. This extra latency arises because the filesystem must check multiple underlying layers—both read-only and writable—to determine which version of a file should be returned. The additional metadata lookups and the logic for merging these layers can add a small amount of overhead compared to reading from a single, simple filesystem.

不过，使用叠加文件系统（overlay filesystem）会有一些开销。这种额外延迟产生的原因是：文件系统必须检查多个底层的层——既包括只读层也包括可写层——以确定应返回文件的哪个版本。与从单一、简单的文件系统读取相比，额外的元数据查找以及合并这些层的逻辑会带来少量开销。

Furthermore, there is overhead when writing to the copy-on-write (CoW) mechanism used by the overlay. CoW means that when you modify a file in the read-only layer (e.g., the base image), the file must first be copied to the writable layer. The write then happens to the copied writable file—instead of the original, read-only file. As mentioned earlier, reading a modified file requires looking at both the read-only and writable layers to determine which is the correct version to return.

此外，向叠加所使用的写时复制（copy-on-write，CoW）机制写入时也会有开销。CoW 意味着，当你修改只读层（例如基础镜像）中的某个文件时，必须先将该文件复制到可写层。写入随后发生在这个被复制到可写层的文件上——而不是原始的只读文件上。如前所述，读取一个被修改过的文件时，需要同时查看只读层和可写层，以确定应返回哪个才是正确的版本。

Model training often involves heavy I/O operations when reading datasets, loading a model, and writing model checkpoints. To work around this, you can mount a host directory—or network filesystem—into the container using bind mounts.

模型训练在读取数据集、加载模型和写入模型检查点时，往往涉及繁重的 I/O 操作。为了绕过这一问题，你可以使用绑定挂载（bind mount）将宿主机目录——或网络文件系统——挂载进容器。

Bind mounts bypass the overlay and therefore perform similarly to disk I/O directly on the host. If the host filesystem is something like an NVMe SSD or an NFS mount, you get the full performance of that underlying storage device. We purposely do not package a multi-terabyte dataset inside the image. Instead, we bring the data in through the mounts.

绑定挂载会绕过叠加层，因此其性能与直接在宿主机上进行磁盘 I/O 相近。如果宿主机文件系统是诸如 NVMe SSD 或 NFS 挂载之类的存储，你就能获得该底层存储设备的完整性能。我们特意不把多 TB 级的数据集打包进镜像。相反，我们通过挂载把数据引入进来。

For example, if your training data is on `/data/dataset` on the host, you’d run the

例如，如果你的训练数据位于宿主机的 `/data/dataset`，你会运行

> container with `-v /data/dataset:/mnt/dataset:ro`, where ro means read-only

> 带 `-v /data/dataset:/mnt/dataset:ro` 的容器，其中 ro 表示只读

mount. Then your training script reads from `/mnt/dataset`. This way, you’re reading directly from the host filesystem.

挂载。然后你的训练脚本从 `/mnt/dataset` 读取数据。这样，你就是直接从宿主机文件系统读取。

In fact, it’s a best practice to avoid heavy data reads/writes against the container’s writable layer. Instead, mount your data directory and output directory from the host into the container. You want to ensure that I/O is not bottlenecked by the overhead of the container’s CoW mechanism.

事实上，避免对容器的可写层进行繁重的数据读写是一项最佳实践。相反，应将你的数据目录和输出目录从宿主机挂载进容器。你要确保 I/O 不会被容器 CoW 机制的开销所拖累。

### Reduce Image Size for Faster Container Startup

### 缩减镜像大小以加快容器启动

Container startup times can be quite a bit slower if the image is huge and needs to be pulled over the network. But in a typical long-running training loop, a startup time of a few minutes is negligible compared to the hours, days, or months of training time. It’s still worth keeping images reasonably slim by not including unnecessary build tools or temporary build files. This saves disk space and improves container startup time.

如果镜像非常庞大且需要通过网络拉取，容器的启动时间可能会慢上不少。但在典型的长时间运行的训练循环中，与数小时、数天乃至数月的训练时间相比，几分钟的启动时间可以忽略不计。尽管如此，通过不包含不必要的构建工具或临时构建文件来让镜像保持合理精简，仍然是值得的。这既节省磁盘空间，又能改善容器启动时间。

Some HPC centers prefer Singularity (Apptainer) over Docker, because it can run images in user space without a root daemon. It also uses the host filesystem directly and tends to have virtually zero overhead beyond what the OS already has.

一些 HPC 中心更青睐 Singularity（Apptainer）而非 Docker，因为它可以在用户空间中运行镜像，而不需要 root 守护进程。它还直接使用宿主机文件系统，往往除了操作系统本身已有的开销之外，几乎没有额外开销。

In either case, Docker or Apptainer (formerly Singularity), studies and benchmarks have shown that once properly configured, these container solutions measure only a couple percent difference between running a container or directly on the host. Essentially, if someone gave you a log of GPU utilization and throughput, it would be difficult to tell from the log alone whether the job ran in a container or not.

无论是 Docker 还是 Apptainer（原 Singularity），研究和基准测试都表明，一旦配置得当，无论是在容器中还是直接在宿主机上运行，这些容器方案测得的差异都只有几个百分点。本质上讲，如果有人给你一份 GPU 利用率和吞吐量的日志，你单凭该日志几乎无法分辨作业是否运行在容器中。

## Kubernetes for Topology-Aware Container Orchestration and Networking

## 用 Kubernetes 实现拓扑感知的容器编排与网络

Kubernetes (also known as K8s) is a popular container orchestrator for AI training and inference. The NVIDIA device plugin for Kubernetes is a lightweight component that advertises GPU hardware (`/dev/nvidia0`, `/dev/nvidiactl`, etc.) to the scheduler. It mounts those device nodes into your pods when you request `nvidia.com/gpu` under `resources.limits` and optionally under `resources.requests`, if you want to set both explicitly. This way, when you deploy a container on Kubernetes with this device plugin, Kubernetes takes care of making the GPUs available to the container. The device plugin is topology aware, as well. This means it can prefer to allocate multiple GPUs from the same NVLink Switch or the same NUMA node for a given pod.

Kubernetes（也称为 K8s）是一款用于 AI 训练和推理的流行容器编排器。用于 Kubernetes 的 NVIDIA 设备插件是一个轻量级组件，它向调度器通告 GPU 硬件（`/dev/nvidia0`、`/dev/nvidiactl` 等）。当你在 `resources.limits` 下（如果你想同时显式设置，也可以在 `resources.requests` 下）请求 `nvidia.com/gpu` 时，它会把这些设备节点挂载进你的 Pod。这样，当你在 Kubernetes 上部署带有该设备插件的容器时，Kubernetes 就会负责把 GPU 提供给容器。该设备插件也是拓扑感知的。这意味着它可以倾向于为某个给定的 Pod 分配来自同一个 NVLink Switch 或同一个 NUMA 节点的多块 GPU。

The NVIDIA Kubernetes GPU Operator automates the installation and lifecycle of all NVIDIA software, including driver libraries, the NVIDIA Kubernetes device plugin mentioned previously, and the NVIDIA Container Toolkit. It’s also responsible for node labeling using NVIDIA’s GPU Feature Discovery to label each GPU with its NUMA node and NVLink/NVSwitch ID. The scheduler can then use these labels to intelligently allocate GPUs to jobs. The GPU Operator also implements GPU monitoring using DCGM.

NVIDIA Kubernetes GPU Operator 自动化了所有 NVIDIA 软件的安装和生命周期管理，包括驱动库、前文提到的 NVIDIA Kubernetes 设备插件以及 NVIDIA Container Toolkit。它还负责使用 NVIDIA 的 GPU Feature Discovery 进行节点标注，为每块 GPU 打上其 NUMA 节点和 NVLink/NVSwitch ID 的标签。随后，调度器便可以利用这些标签智能地把 GPU 分配给作业。GPU Operator 还使用 DCGM 实现 GPU 监控。

When using Kubernetes to orchestrate GPU-based containers, you want it to allocate resources to containers in a manner that is aware of the hardware topology, including the NUMA node and network bandwidth configurations. However, by default, Kubernetes is not topology-aware. It treats each GPU as a resource but doesn’t know if GPU 0 and GPU 1 are on the same NUMA node or if they use the same NVLink interconnect. This could make a big difference.

在使用 Kubernetes 编排基于 GPU 的容器时，你希望它以一种能感知硬件拓扑（包括 NUMA 节点和网络带宽配置）的方式为容器分配资源。然而，默认情况下 Kubernetes 并不具备拓扑感知能力。它把每块 GPU 视为一种资源，却不知道 GPU 0 和 GPU 1 是否位于同一个 NUMA 节点，或它们是否使用同一条 NVLink 互连。这可能会造成很大的差异。

Consider an 8-GPU server with two sets of 4 GPUs—each connected by NVLink. If you request 4 GPUs from Kubernetes for a job, it would be ideal if K8s gave you four GPUs that are all interconnected with NVLink, as they can share data faster. However, if Kubernetes picks four arbitrary GPUs spread anywhere in the system, your job might be allocated two GPUs from one NVLink domain and two GPUs from a different NVLink domain.

设想一台配有两组各 4 块 GPU 的 8-GPU 服务器——每组内部由 NVLink 连接。如果你向 Kubernetes 为某个作业请求 4 块 GPU，理想情况下 K8s 应该给你四块全部通过 NVLink 互连的 GPU，因为它们能更快地共享数据。然而，如果 Kubernetes 随意挑选了分散在系统各处的四块 GPU，你的作业可能会被分配到来自一个 NVLink 域的两块 GPU 和来自另一个 NVLink 域的两块 GPU。

Allocating GPUs without topology awareness will introduce slow interconnects (e.g., InfiniBand or Ethernet) into the GPU-to-GPU route. This can cut your inter-GPU bandwidth in half. In this case, Kubernetes would ideally allocate four GPUs that are all interconnected by NVLink and use the same NUMA node rather than four that span different racks and NUMA domains.

在不考虑拓扑的情况下分配 GPU，会把慢速互连（例如 InfiniBand 或 Ethernet）引入 GPU 到 GPU 的路径中。这可能会把你的 GPU 间带宽砍掉一半。在这种情况下，理想的做法是让 Kubernetes 分配四块全部由 NVLink 互连、且使用同一个 NUMA 节点的 GPU，而不是四块横跨不同机架和 NUMA 域的 GPU。

For instance, consider NVIDIA’s NVL72 rack built on NVLink 5 interconnects, which connects 72 GPUs into a single high-bandwidth domain with a combined ~130 TB/s throughput inside the rack (72 GPUs \* 1.8 TB/s per GPU). In this configuration, if a Kubernetes scheduler isn’t topology aware, it might place a multi-GPU job across different NVLink domains—or even outside the NVL72 group. Allocating GPUs to a job without respecting the system’s topology would negate the benefits of the NVL72’s massive intra-rack bandwidth.

举例来说，考虑 NVIDIA 基于 NVLink 5 互连构建的 NVL72 机架，它将 72 块 GPU 连接成一个单一的高带宽域，机架内部的合计吞吐量约为 130 TB/s（72 块 GPU × 每块 1.8 TB/s）。在这种配置中，如果 Kubernetes 调度器不具备拓扑感知能力，它可能会把一个多 GPU 作业放置在不同的 NVLink 域之间——甚至放到 NVL72 组之外。在不尊重系统拓扑的情况下把 GPU 分配给作业，会抵消 NVL72 巨大的机架内带宽所带来的好处。

To avoid resource contention, you should try to either reserve the resources that you need or request the entire node for your job. For the container/pod placements, you should align pods with CPU affinities and NUMA nodes using the Kubernetes Topology Manager component to bind the container’s CPUs to the same NUMA node as the GPUs that the container was allocated. Let’s discuss this next.

为了避免资源争用，你应该尽量要么预留你所需的资源，要么为你的作业请求整个节点。对于容器/Pod 的放置，你应使用 Kubernetes Topology Manager 组件，将 Pod 与 CPU 亲和性和 NUMA 节点对齐，把容器的 CPU 绑定到与该容器所分配 GPU 相同的 NUMA 节点上。接下来我们就讨论这一点。

### Orchestrating Containers with Kubernetes Topology Manager

### 使用 Kubernetes Topology Manager 编排容器

Kubernetes Topology Manager can provide detailed topology information. For example, it can detect that GPU 0 is connected to NUMA node 0, NVLink domain A, and PCIe bus Z. The Kubernetes scheduler can then use this information to allocate containers to GPUs in an optimal way for efficient processing and communication.

Kubernetes Topology Manager 可以提供详细的拓扑信息。例如，它可以检测到 GPU 0 连接到 NUMA 节点 0、NVLink 域 A 和 PCIe 总线 Z。随后 Kubernetes 调度器便可利用这些信息，以一种利于高效处理和通信的最优方式，把容器分配到 GPU 上。

Topology-aware GPU scheduling is still maturing. In many clusters, administrators explicitly label nodes using Kubernetes labels to capture the GPU and system topology. These labels ensure that multi-GPU pods land on servers whose GPUs share the same NVLink interconnect or reside within the same NUMA domain.

拓扑感知的 GPU 调度仍在不断成熟。在许多集群中，管理员会使用 Kubernetes 标签显式地标注节点，以刻画 GPU 和系统的拓扑。这些标签确保多 GPU 的 Pod 落到那些 GPU 共享同一条 NVLink 互连、或位于同一个 NUMA 域内的服务器上。

For our purposes, if you’re running multi-GPU jobs in Kubernetes, make sure to enable topology-aware scheduling. This typically involves configuring `--topology-`

> `manager-policy to best-effort`, `restricted`, or, in some cases, `single-numa-`

`node`. This policy configuration helps multi-GPU and CPU + GPU workloads achieve lower latency by avoiding remote memory access. This complements the OS-level NUMA tuning.

就我们的目的而言，如果你在 Kubernetes 中运行多 GPU 作业，请务必启用拓扑感知调度。这通常需要将 `--topology-manager-policy` 配置为 `best-effort`、`restricted`，或在某些情况下配置为 `single-numa-node`。这种策略配置有助于多 GPU 以及 CPU + GPU 的工作负载通过避免远程内存访问来降低延迟。它与操作系统层面的 NUMA 调优相辅相成。

Also, be sure to use the latest NVIDIA GPU device plugin and NVIDIA Kubernetes GPU Operator, mentioned in the previous section, as these are topology aware and support packing multi-GPU pods onto GPUs that are connected to the same NUMA node. These help optimize performance by minimizing cross-NUMA-node communication and reducing latency in multinode GPU workloads.

另外，务必使用上一节提到的最新 NVIDIA GPU 设备插件和 NVIDIA Kubernetes GPU Operator，因为它们是拓扑感知的，并支持把多 GPU 的 Pod 打包到连接同一个 NUMA 节点的 GPU 上。它们通过最小化跨 NUMA 节点通信、并降低多节点 GPU 工作负载中的延迟，来帮助优化性能。

On NVLink-5 NVL72 systems, a single rack-level NVLink domain provides up to 130 TB/s of aggregate bidirectional GPU-to-GPU bandwidth, equivalent to about 1.8 TB/s per GPU. When scheduling collective-heavy training, prefer placements that keep traffic inside the fast NVLink domain before crossing the slower network fabric.

在 NVLink-5 NVL72 系统上，单个机架级的 NVLink 域可提供高达 130 TB/s 的 GPU 到 GPU 双向聚合带宽，相当于每块 GPU 约 1.8 TB/s。在调度以集合通信为主的训练时，应优先选择能让流量在跨越较慢的网络 fabric 之前，先留在高速 NVLink 域内部的放置方案。

### Job Scheduling with Kubernetes and SLURM

### 使用 Kubernetes 与 SLURM 进行作业调度

In multinode deployments, job schedulers are essential for maximizing resource utilization across all nodes. Commonly, the Simple Linux Utility for Resource Management (SLURM) is used for training clusters, while Kubernetes is typically favored for inference clusters. However, hybrid solutions have emerged that integrate SLURM with Kubernetes. The open source Slinky project is an example solution to simplify cluster management across training and inference workloads.

在多节点部署中，作业调度器对于最大化所有节点上的资源利用率至关重要。通常，训练集群使用简单 Linux 资源管理工具（Simple Linux Utility for Resource Management，SLURM），而推理集群则通常偏爱 Kubernetes。不过，已经出现了将 SLURM 与 Kubernetes 集成的混合方案。开源的 Slinky 项目就是一个示例方案，用于简化跨训练与推理工作负载的集群管理。

These systems handle the allocation of GPUs to jobs and coordinate the launch of processes across nodes. If a training job requests 8 nodes with 8 GPUs per node, the scheduler will identify eligible nodes and start the job using tools like `mpirun` or container runtimes such as Docker. This way, each process is aware of all available GPUs in the job. Many clusters also rely on well-tested Docker repositories like NVIDIA’s NGC Docker repository to guarantee a consistent software environment—including GPU drivers, CUDA toolkits, PyTorch libraries, and other Python packages—across all nodes.

这些系统负责把 GPU 分配给作业，并协调跨节点的进程启动。如果一个训练作业请求 8 个节点、每节点 8 块 GPU，调度器会识别出符合条件的节点，并使用诸如 `mpirun` 之类的工具或诸如 Docker 之类的容器运行时来启动作业。这样，每个进程都能感知到该作业中所有可用的 GPU。许多集群还依赖经过充分测试的 Docker 仓库（如 NVIDIA 的 NGC Docker 仓库），以保证所有节点上有一致的软件环境——包括 GPU 驱动、CUDA 工具包、PyTorch 库和其他 Python 包。

With SLURM, similar issues exist. SLURM has the concept of “generic resources” for GPUs, and you can define that certain GPUs are attached to certain NUMA nodes or NVLinks/NVSwitches. Then in your job request, you can ask for GPUs that are, say, connected to the same NUMA node.

在 SLURM 上也存在类似的问题。SLURM 有针对 GPU 的“通用资源（generic resources）”概念，你可以定义某些 GPU 挂接到某些 NUMA 节点或 NVLink/NVSwitch 上。然后在你的作业请求中，你就可以申请那些（比如说）连接到同一个 NUMA 节点的 GPU。

If not properly set, a scheduler might treat all GPUs as identical and provide nonideal allocations for your multi-GPU container requests. Proper configuration can avoid unnecessary cross-NUMA-node and cross-NVLink GPU communication overhead.

如果没有正确设置，调度器可能会把所有 GPU 都当作完全相同，从而为你的多 GPU 容器请求给出并不理想的分配。恰当的配置可以避免不必要的跨 NUMA 节点、跨 NVLink 的 GPU 通信开销。

SLURM supports scheduling MIG partitions as distinct resources as well. This can be useful for packing multiple jobs onto one GPU. This is analogous to how Kubernetes can schedule GPU slices using the Kubernetes device plugin. Next, we’ll discuss how to use MIG slices with Kubernetes.

SLURM 同样支持把 MIG 分区作为独立资源来调度。这对于把多个作业打包到同一块 GPU 上很有用，其原理类似于 Kubernetes 通过 Kubernetes 设备插件调度 GPU 切片。接下来，我们将讨论如何在 Kubernetes 中使用 MIG 切片。

### Slicing a GPU with MIG

### 用 MIG 切分 GPU

When you enable NVIDIA’s MIG mode, introduced in an earlier section, a single physical GPU is sliced into smaller, fixed, and hardware-isolated partitions called MIG instances. Next is an example Kubernetes pod configuration for two of the `nvidia.com/mig-2g.45gb` MIG slices (this configuration assumes that the NVIDIA Kubernetes device plugin is configured to recognize MIG devices on each node):

当你启用前文介绍过的 NVIDIA MIG 模式时，单块物理 GPU 会被切分成更小、固定且硬件隔离的分区，称为 MIG 实例。下面是一个 Kubernetes pod 配置示例，申请两个 `nvidia.com/mig-2g.45gb` MIG 切片（该配置假定 NVIDIA Kubernetes 设备插件已配置为能够识别每个节点上的 MIG 设备）：

```
resources:
  limits:
    nvidia.com/mig-2g.45gb: "2"
```

```
resources:
  limits:
    nvidia.com/mig-2g.45gb: "2"
```

Here, the configuration specifies running a pod on a node with at least two free `2g` `.45gb` instances on one GPU; in other words, 2 slices in which each slice is 2/7 of the SMs (`2g`). If a GPU has a total of 132 SMs, each is 2/7 × 132 SMs = ~38 SMs. Multiply this by 2), and the pod is allocating a total of ~76 SMs. The total memory allocation is 45 GB of GPU RAM.

这里，该配置要求在某个节点上运行一个 pod，且该节点的一块 GPU 上至少有两个空闲的 `2g` `.45gb` 实例；换句话说，需要 2 个切片，每个切片占 2/7 的 SM（`2g`）。如果一块 GPU 总共有 132 个 SM，那么每个切片为 2/7 × 132 SM = ~38 SM。乘以 2，pod 总共分配到约 76 个 SM。总显存分配为 45 GB 的 GPU 显存。

Note that the scheduler cannot split these across GPUs or nodes. Kubernetes will schedule the pod only if a single node can provide both partitions. This is because pods cannot span multiple nodes. If no single node has two free `2g.45gb` slices available for a total of 76 SMs and 45 GB of GPU RAM (as calculated previously), the pod remains in a Kubernetes `Pending` (unscheduled) state and therefore won’t run—even if other nodes collectively have enough MIG capacity.

请注意，调度器无法把这些切片跨 GPU 或跨节点拆分。只有当某个单一节点能够同时提供这两个分区时，Kubernetes 才会调度该 pod。这是因为 pod 不能跨越多个节点。如果没有任何单一节点拥有两个空闲的 `2g.45gb` 切片（即前面算出的总共 76 个 SM 和 45 GB GPU 显存），该 pod 就会一直停留在 Kubernetes 的 `Pending`（未调度）状态，因而无法运行——即便其他节点合计起来拥有足够的 MIG 容量也是如此。

This constraint highlights the importance of planning your MIG sizes according to typical workload needs. For instance, if many jobs request `2g.45gb` slices, you might configure each GPU to host three `2g.45gb` instances—among its seven possible slices —so that two such instances can co-reside on one GPU for a single pod.

这一约束凸显了根据典型工作负载需求来规划 MIG 尺寸的重要性。例如，如果很多作业都申请 `2g.45gb` 切片，你或许应当把每块 GPU 配置为承载三个 `2g.45gb` 实例——在其可能的七个切片之中——这样一来，就能有两个这样的实例共同驻留在同一块 GPU 上，供单个 pod 使用。

> This single-node constraint can cause pods to never run—even if the combined MIG resources can be found across different nodes of the cluster. The request can be satisfied only if the requested MIG resources are available on a single node.

> 这一单节点约束可能导致 pod 永远无法运行——即使把集群中不同节点上的 MIG 资源合并起来能够满足需求也无济于事。只有当所请求的 MIG 资源在单一节点上可用时，请求才能被满足。

An administrative drawback with MIG is that switching a GPU between MIG mode and normal (non-MIG) mode requires resetting the GPUs—or rebooting the compute node. So it’s not something the scheduler can easily do dynamically per job. However, you usually create MIG partitions in advance and leave the configuration running for some period of time.

MIG 在运维上的一个缺点是：在 MIG 模式与普通（非 MIG）模式之间切换某块 GPU 需要重置该 GPU——或重启计算节点。因此，调度器难以按作业动态地完成这一切换。不过，通常你会提前创建好 MIG 分区，并让该配置持续运行一段时间。

In a Kubernetes environment, the NVIDIA Kubernetes GPU Operator’s MIG Manager can automatically configure and preserve MIG partitions on nodes. This way, the MIG slices remain active across reboots and driver reloads.

在 Kubernetes 环境中，NVIDIA Kubernetes GPU Operator 的 MIG Manager 可以自动配置并保留各节点上的 MIG 分区。这样，MIG 切片就能在重启和驱动重新加载后依然保持有效。

You can label one K8s node with “mig-enabled” and another as “mig-disabled” and let the scheduler place jobs/pods accordingly. This is more of an operational detail, but it’s good to know that MIG is a truly static partition—and not a product of a dynamic scheduler.

你可以给一个 K8s 节点打上“mig-enabled”标签，给另一个打上“mig-disabled”标签，让调度器据此放置作业/pod。这更多是一个运维细节，但值得了解的是：MIG 是真正的静态分区——而非动态调度器的产物。

> Persistence mode is recommended when using MIG so that the MIG configuration remains active on the GPU even if no jobs are running. This way, the GPU doesn’t have to keep rebuilding the slices before running each periodic job.

> 在使用 MIG 时，建议启用持久化模式（persistence mode），这样即使没有作业在运行，MIG 配置也能在 GPU 上保持有效。如此一来，GPU 就不必在每次运行周期性作业之前反复重建切片。

### Optimizing Network Communication for Kubernetes

### 优化 Kubernetes 的网络通信

When you run multinode GPU workloads using containers with Kubernetes, the pods need to talk to one another. In Kubernetes, by default, pods have their own IP, and there might be an overlay network or network-address translation (NAT) between pods on different nodes. This can introduce complications and additional overhead.

当你用 Kubernetes 上的容器运行多节点 GPU 工作负载时，这些 pod 之间需要相互通信。在 Kubernetes 中，默认情况下每个 pod 都有自己的 IP，不同节点上的 pod 之间可能存在叠加网络（overlay）或网络地址转换（network-address translation，NAT）。这会带来一些复杂性和额外开销。

Often, the simplest solution for GPU clusters is to use host networking for these performance-sensitive jobs. That means the container’s network is not isolated, as it uses the host’s network interface directly. To enable this in Kubernetes, you set `host` `Network: true` on the pod specification. In Docker, you could run with

对于 GPU 集群，往往最简单的方案是为这些性能敏感的作业使用主机网络（host networking）。这意味着容器的网络不再被隔离，而是直接使用主机的网络接口。要在 Kubernetes 中启用这一点，你需要在 pod 规格中设置 `host` `Network: true`。在 Docker 中，你可以这样运行：

> `--network=host`.

> `--network=host`。

Using host networking allows a container to access the InfiniBand interconnect exactly as the host does—without any additional translation or firewall layers. This is particularly useful for MPI jobs because it eliminates the need to configure port mappings for every MPI rank.

使用主机网络能让容器完全像主机一样访问 InfiniBand 互连——没有任何额外的转换层或防火墙层。这对 MPI 作业尤其有用，因为它省去了为每个 MPI rank 配置端口映射的麻烦。

However, if host networking is not an option due to security policies, you must ensure that your Kubernetes container network interface (CNI) and any overlay network can handle the required traffic. In such cases, you may need to open specific ports to support the handshake of NCCL and data exchange, using environment variables like `NCCL_PORT_RANGE` and `NCCL_SOCKET_IFNAME` to help establish connections.

然而，如果由于安全策略而无法使用主机网络，你就必须确保你的 Kubernetes 容器网络接口（container network interface，CNI）以及任何叠加网络都能处理所需的流量。在这种情况下，你可能需要开放特定端口以支持 NCCL 的握手与数据交换，并借助 `NCCL_PORT_RANGE` 和 `NCCL_SOCKET_IFNAME` 等环境变量来帮助建立连接。

When operating over an overlay network, it’s critical that latency remains low and operations run in kernel space. Also, make sure that no user-space proxies throttle traffic between nodes. These factors can significantly impact performance.

在叠加网络上运行时，至关重要的是保持低延迟并让操作在内核空间中运行。同时，要确保没有任何用户空间代理限制节点间的流量。这些因素都会显著影响性能。

When using a Kubernetes environment and you want to enable RDMA, consider installing the Kubernetes RDMA device plugin from Mellanox. This plugin exposes InfiniBand and GPUDirect RDMA endpoints on the pods interface to enable low-latency, zero-copy networking.

在 Kubernetes 环境中若想启用 RDMA，可以考虑安装 Mellanox 提供的 Kubernetes RDMA 设备插件。该插件会在 pod 接口上暴露 InfiniBand 和 GPUDirect RDMA 端点，从而实现低延迟、零拷贝的网络通信。

> If you have InfiniBand or RoCE networking, remember to enable GPUDirect RDMA in the NVIDIA driver if your NIC supports it. This allows GPUs to directly exchange data with the NIC—bypassing the CPU for internode communication. This is essential for maintaining high performance in a multinode environment.

> 如果你使用 InfiniBand 或 RoCE 网络，请记得在 NVIDIA 驱动中启用 GPUDirect RDMA（前提是你的 NIC 支持）。这样 GPU 就能直接与 NIC 交换数据——在节点间通信中绕过 CPU。这对于在多节点环境中维持高性能至关重要。

### Reducing Kubernetes Orchestration Jitter

### 降低 Kubernetes 编排抖动

Running an orchestrator like Kubernetes means there are some background processes running on every node (e.g., the Kubernetes “kubelet”), container runtime daemons, and (ideally) monitoring agents. While these services consume CPU and memory, the consumption is on the order of a few percent of a single core. So they won’t steal noticeable time from a GPU-based training job, which uses these cores for data loading and preprocessing.

运行像 Kubernetes 这样的编排器意味着每个节点上都会有一些后台进程在运行（例如 Kubernetes 的“kubelet”）、容器运行时守护进程，以及（最好还有）监控代理。虽然这些服务会消耗 CPU 和内存，但其消耗量约为单个核心的百分之几。因此，它们不会从基于 GPU 的训练作业中偷走明显的时间，而训练作业正是用这些核心来做数据加载和预处理的。

However, if the training job is running on a node that is also running an inference workload, you may experience some jitter, or unpredictable variation, in the execution timing and throughput. This is common in any multitenancy situation, though. If another container on the same machine unexpectedly uses a lot of CPU or I/O, it will affect your container—whether training or inference—by competing for the same resources.

然而，如果训练作业运行在一个同时也在运行推理工作负载的节点上，你可能会在执行时序和吞吐量上遇到一些抖动（jitter），即不可预测的波动。不过，这在任何多租户场景中都很常见。如果同一台机器上的另一个容器意外地占用了大量 CPU 或 I/O，它就会通过争抢相同资源来影响你的容器——无论是训练还是推理。

> Homogeneous workloads such as all training or all inference are much easier to debug and tune from a system’s perspective than a heterogeneous mix of both training and inference.

> 从系统的角度来看，同构工作负载（例如全部为训练或全部为推理）比训练与推理混杂的异构组合要容易调试和调优得多。

### Improving Resource Guarantees

### 改进资源保障

To safeguard against resource contention, Kubernetes lets you define resource requests and limits for pods. For example, you can specify that your training job requires 16 CPU cores and 64 GB of RAM. Kubernetes will then reserve those resources exclusively for your job and avoid scheduling other pods on the same CPUs.

为了防范资源争用，Kubernetes 允许你为 pod 定义资源请求（requests）和限制（limits）。例如，你可以指定训练作业需要 16 个 CPU 核心和 64 GB 内存。随后，Kubernetes 会把这些资源专门预留给你的作业，并避免在相同的 CPU 上调度其他 pod。

These limits are enforced using Linux cgroups, so if your container exceeds its allocation, it can be throttled or even terminated by the OOM killer. It’s common practice to use resource requests—and optionally the CPU Manager feature to pin cores—to ensure that performance-critical jobs get exclusive access to the necessary CPU resources so that other processes cannot steal CPU time from your reserved cores.

这些限制通过 Linux cgroups 强制执行，因此如果你的容器超出其分配额度，它可能会被限流，甚至被 OOM killer 终止。常见的做法是使用资源请求——并可选地使用 CPU Manager 功能来绑定核心——以确保性能关键的作业能够独占所需的 CPU 资源，从而使其他进程无法从你预留的核心中偷走 CPU 时间。

Another source of jitter is background kernel threads and interrupts, as we discussed in Chapter 2 in the context of using interrupt request (IRQ) affinity. Similar to Kubernetes, if other pods are using the same network or disks as your job, the other pods might cause a lot of interrupts and extra kernel work on the compute nodes that host your job. This will cause jitter and affect your job’s performance.

抖动的另一个来源是后台内核线程和中断，正如我们在第 2 章讨论中断请求（IRQ）亲和性时所提到的。与 Kubernetes 的情况类似，如果其他 pod 与你的作业使用相同的网络或磁盘，这些 pod 可能会在承载你作业的计算节点上引发大量中断和额外的内核工作。这会造成抖动，并影响你作业的性能。

Ideally, a GPU node is fully dedicated to your job. However, if it’s not, you should ensure that the node is carefully partitioned using Linux cgroup controllers for I/O and CPU so that other workloads don’t interfere.

在理想情况下，一个 GPU 节点应完全专用于你的作业。但如果做不到，你就应确保使用 Linux cgroup 控制器对该节点的 I/O 和 CPU 进行细致的分区，以免其他工作负载造成干扰。

Fortunately, Kubernetes supports CPU isolation, which ensures that pods get the dedicated CPU cores and memory they request—and prevents other pods from being scheduled on the same CPU core as yours. This avoids extra overhead from context switching and resource contention.

好在 Kubernetes 支持 CPU 隔离，它能确保 pod 获得其所请求的专用 CPU 核心和内存——并防止其他 pod 被调度到与你相同的 CPU 核心上。这样就避免了上下文切换和资源争用带来的额外开销。

> In practice, performance-sensitive Kubernetes jobs should request all of the CPUs and GPUs of a given node so that nothing else interferes or contends with the jobs’ resources. Easier said than done, but this is the ideal job configuration from a performance and consistency standpoint.

> 在实践中，性能敏感的 Kubernetes 作业应当请求某个节点的全部 CPU 和 GPU，以免任何其他东西干扰或争抢该作业的资源。这说起来容易做起来难，但从性能与一致性的角度看，这是最理想的作业配置。

### Memory Isolation and Avoiding the OOM Killer

### 内存隔离与避免 OOM Killer

Memory interference can also occur if not properly limited. Kubernetes provides first-class memory isolation support (using Linux cgroups). However, a greedy container, if unconstrained, could allocate too much memory on the host. This would cause the host to swap some of its memory to disk.

如果不加以恰当限制，内存干扰同样会发生。Kubernetes 提供一流的内存隔离支持（通过 Linux cgroups）。然而，一个贪婪的容器如果不受约束，就可能在主机上分配过多内存，进而导致主机把部分内存交换到磁盘。

If an unbounded container uses too much memory on the host, the infamous Linux “OOM killer” will start killing processes—and potentially your Kubernetes job—even if your job wasn’t the one using too much memory.

如果一个不受限制的容器在主机上使用了过多内存，臭名昭著的 Linux “OOM killer” 就会开始杀死进程——甚至可能杀掉你的 Kubernetes 作业——即便占用过多内存的并不是你的作业。

The OOM killer uses heuristics when deciding which pods to kill. Sometimes it decides to kill the largest running pod, which is likely your large training or inference job holding lots of data in CPU RAM to feed the GPUs. To avoid this, you can purposely not set strict memory limits on training or inference containers. This way, they can use all available memory, if needed.

OOM killer 在决定杀掉哪些 pod 时会使用启发式规则。有时它会决定杀掉正在运行的最大的 pod，而那很可能就是你那个在 CPU 内存中持有大量数据以喂给 GPU 的大型训练或推理作业。为避免这种情况，你可以有意不给训练或推理容器设置严格的内存限制。这样，它们在需要时就能使用所有可用内存。

With proper monitoring and alerting, you can ensure the job doesn’t try to over-allocate beyond what you expect. If you do set a memory limit, make sure it’s above what you actually expect to use. This provides a bit of headroom to avoid getting killed by the OOM killer three days into a long-running training job.

借助恰当的监控与告警，你可以确保作业不会试图超出你预期地过度分配。如果你确实要设置内存限制，请确保它高于你实际预期的用量。这样能留出一点余量，避免在长时间运行的训练作业进行到第三天时被 OOM killer 杀掉。

> In Kubernetes, a Pod with no requests/limits is treated as `BestEffort` and is the most likely to be evicted. To obtain `Guaranteed` QoS, every container must set `requests == limits` for both CPU and memory. Setting a high limit alone will result in a `Burstable` QoS, not `Guaranteed`.

> 在 Kubernetes 中，没有设置请求/限制的 Pod 会被视为 `BestEffort`，最容易被驱逐。要获得 `Guaranteed` QoS，每个容器都必须为 CPU 和内存都设置 `requests == limits`。仅仅设置一个较高的限制只会得到 `Burstable` QoS，而非 `Guaranteed`。

### Dealing with I/O Isolation

### 应对 I/O 隔离

As of this writing, Kubernetes does not offer native, first-class I/O isolation out of the box, unfortunately. While Linux does support I/O controls using cgroup controllers, Kubernetes itself does not automatically enforce I/O limits in the same way it does for CPU and memory.

遗憾的是，截至本文撰写时，Kubernetes 并未开箱即用地提供原生的一流 I/O 隔离。虽然 Linux 确实支持通过 cgroup 控制器进行 I/O 控制，但 Kubernetes 本身并不会像对待 CPU 和内存那样自动强制执行 I/O 限制。

If you need to ensure that heavy I/O workloads on a GPU node don’t interfere with one another, you might need to manually configure I/O controls at the node level. This can involve adjusting the cgroup v2 I/O controller or using other OS-level configurations to partition I/O resources. In short, while Kubernetes prevents CPU contention through scheduling and resource requests, I/O isolation usually requires additional, manual tuning of the underlying Linux system.

如果你需要确保 GPU 节点上的重 I/O 工作负载彼此不相互干扰，你可能需要在节点层面手动配置 I/O 控制。这可能涉及调整 cgroup v2 的 I/O 控制器，或使用其他 OS 层面的配置来分区 I/O 资源。简而言之，尽管 Kubernetes 通过调度和资源请求来防止 CPU 争用，但 I/O 隔离通常需要对底层 Linux 系统进行额外的手动调优。

It’s important to note that, inside a container, some system settings are inherited from the host. For instance, if the host has CPU frequency scaling set to performance mode, the container will inherit that setting. But if the container is running in a virtualized environment such as a cloud instance, you might not be able to change these settings.

需要重点指出的是，在容器内部，某些系统设置是从主机继承而来的。例如，如果主机把 CPU 频率调节设为性能模式（performance mode），容器就会继承该设置。但如果容器运行在诸如云实例这样的虚拟化环境中，你或许无法更改这些设置。

It’s a good idea to always ensure that the host machine is tuned since containers can’t change kernel parameters like hugepage settings or CPU governor limits. Usually, cluster admins set these parameters and settings through the base OS image. Or, in a Kubernetes environment, they might use something like the NVIDIA GPU Operator to set persistence mode and other `sysctl` knobs on each node.

始终确保对主机进行调优是个好主意，因为容器无法更改诸如大页设置或 CPU governor 限制之类的内核参数。通常，集群管理员会通过基础 OS 镜像来设置这些参数和配置。或者，在 Kubernetes 环境中，他们可能会使用类似 NVIDIA GPU Operator 的工具，在每个节点上设置持久化模式以及其他 `sysctl` 旋钮。

## Key Takeaways

## 关键要点

Here is a list of key takeaways from this chapter, including optimizations across the operating system, driver, GPU, CPU, and container layers:

以下是本章的一系列关键要点，涵盖了操作系统、驱动、GPU、CPU 与容器各层的优化：

- **Data and compute locality is critical.** Ensure that data is stored and processed as close to the computation units as possible. Use local, high-speed storage such as NVMe SSD caches to minimize latency and reduce reliance on remote filesystems or network I/O.

- **Implement NUMA-aware configuration and CPU affinity.** Optimize CPU-to-GPU data flow by aligning processes and memory allocations within the same NUMA node. Pin the CPU with tools like `numactl` and `taskset` prevents cross-node memory access. This will lead to lower latency and improved throughput.

- **Maximize GPU driver and runtime efficiency.** Fine-tune the GPU driver settings, such as enabling persistence mode to keep GPUs in a ready state. Consider features like Multi-Process Service (MPS) for overlapping work from multiple processes on a single GPU. For multitenant environments, explore MIG partitions to isolate workloads effectively.

- **Prefetch and batch data effectively.** Keep the GPUs fed by prefetching data ahead of time and batching small I/O operations into larger, more efficient reads. Leverage prefetching mechanisms like PyTorch’s DataLoader `prefetch_factor` (along with `num_workers`) to load multiple batches in advance.

- **Pin memory when data loading.** Combining data prefetching with memory pinning using PyTorch’s DataLoader `pin_memory=True` uses pinned CPU memory (page-locked, not swappable to disk) for faster, asynchronous data transfers to the GPU. As a result, data loading and model execution can overlap, idle times are reduced, and both CPU and GPU resources are continuously utilized.

- **Optimize memory transfers.** Leverage techniques such as pinned, page-locked memory and hugepages to accelerate data transfers between the host and GPU. This helps reduce copy overhead and allows asynchronous transfers to overlap with computations.

- **Overlap communication with computation.** Reduce the waiting time for data transfers by overlapping memory operations like gradient synchronization and data staging with ongoing GPU computations. This overlap helps maintain high GPU utilization and better overall system efficiency.

- **Tune and scale the networking stack.** In multinode environments, use RDMA-enabled networks (e.g., InfiniBand/Ethernet), and tune network settings such as TCP buffers, MTU, and interrupt affinities to maintain high throughput during distributed training and inference.

- **Use containerization and orchestration for consistency.** Use container runtimes like Docker with the NVIDIA Container Toolkit and orchestration platforms like Kubernetes with the NVIDIA GPU Operator and device plugin so that the entire software stack—including drivers, CUDA libraries, and application code—is consistent across nodes. These solutions help align CPU-GPU affinities and manage resource allocation based on hardware topology.

- **Eliminate container runtime overhead.** While containers increase reproducibility and ease of deployment, ensure that CPU and GPU affinities, host networking, and resource isolation are correctly configured to minimize any container overhead.

- **Use orchestration and scheduling best practices.** Robust container orchestrators like Kubernetes are essential components for ensuring efficient resource allocation. Advanced scheduling techniques—such as the Kubernetes Topology Manager—help ensure that GPUs with fast interconnects are clustered together.

- **Strive for flexibility through dynamic adaptability and scaling.** The orchestration layer distributes work and dynamically manages workload segmentation across nodes. This flexibility is crucial for both scaling up training tasks and ensuring efficient runtime in inference scenarios where data loads and request patterns vary widely.

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

Tune continuously and incrementally System-level optimizations are not one-and-done. Regularly monitor performance metrics; adjust CPU affinities, batch sizes, and prefetch settings as workloads evolve; and use these small improvements cumulatively to achieve significant performance gains.

持续且渐进地调优 系统级优化并非一劳永逸。要定期监控性能指标；随着工作负载的演变而调整 CPU 亲和性、批大小和预取设置；并把这些小的改进累积起来，以获得可观的性能提升。

Reduce bottlenecks across the stack The ultimate goal is to ensure that all components, from the OS and CPU to the GPU driver and runtime, work in harmony. Eliminating bottlenecks in one layer, such as CPU memory allocation or driver initialization, unlocks the full potential of the GPUs, which directly translates to faster training, lower costs, and more efficient resource usage.

在整个栈中减少瓶颈 最终目标是确保所有组件——从操作系统和 CPU，到 GPU 驱动和运行时——都能协调一致地工作。消除某一层中的瓶颈（例如 CPU 内存分配或驱动初始化），就能释放 GPU 的全部潜力，而这将直接转化为更快的训练、更低的成本和更高效的资源使用。

Together, these strategies work to minimize data transfer friction, reduce wait times, and ensure that your hardware is used to its fullest potential for efficient training and inference.

这些策略共同作用，最大限度地减少数据传输的摩擦、缩短等待时间，并确保你的硬件被充分利用，以实现高效的训练和推理。

## Conclusion

## 结论

This chapter has demonstrated that even the most advanced GPUs can be hindered by inefficiencies in their surrounding environment. A well-tuned operating system, container runtime, cluster orchestrator, and software stack form the backbone of high-performance AI systems. By aligning data with compute through NUMA-aware pinning and local storage solutions, overlapping communication with computation, and fine-tuning both the host system and GPU drivers, you can reduce latency and increase throughput.

本章表明，即便是最先进的 GPU，也可能因其周边环境中的低效而受到掣肘。一个调优良好的操作系统、容器运行时、集群编排器和软件栈，构成了高性能 AI 系统的骨架。通过 NUMA 感知的绑定与本地存储方案让数据与计算对齐、让通信与计算重叠，并对主机系统和 GPU 驱动都进行精细调优，你就能降低延迟、提升吞吐量。

Think of your entire system as a precision-engineered sports car where each component (CPU, memory, GPU, network, containers, orchestrators, and programming stack) must work seamlessly together to deliver maximum performance. Small tweaks, such as enabling persistence mode or optimizing CPU scheduling, may seem minor on their own, but when combined and scaled across a large GPU cluster, they can lead to substantial savings in time and cost. These optimizations ensure that GPUs are consistently operating near their peak efficiency when training massive transformer models and running complex inference pipelines.

不妨把你的整个系统想象成一辆精密工程打造的跑车，其中每个组件（CPU、内存、GPU、网络、容器、编排器和编程栈）都必须无缝协作，才能交付最大性能。诸如启用持久化模式或优化 CPU 调度之类的小调整，单独看似乎微不足道，但当它们被组合起来并在大型 GPU 集群上扩展开来时，就能在时间和成本上带来可观的节省。这些优化确保 GPU 在训练大规模 transformer 模型和运行复杂推理管线时，始终运行在接近峰值效率的状态。

As the field evolves and models continue to grow, the importance of system-level tuning will only increase. The techniques discussed in this chapter empower performance engineers and system architects to leverage every bit of hardware potential. This enables faster iteration cycles and more cost-effective AI deployments. Ultimately, a deeply optimized system accelerates research and makes cutting-edge AI applications more accessible to a broader audience.

随着这一领域的演进和模型的持续增长，系统级调优的重要性只会与日俱增。本章讨论的技术让性能工程师和系统架构师能够榨取硬件的每一分潜力。这带来了更快的迭代周期和更具成本效益的 AI 部署。归根结底，一个深度优化的系统能够加速研究，并让前沿 AI 应用惠及更广泛的受众。

Finally, remember that while the hardware and software stack may seem like an unmanageable amount of interconnected knobs and switches, small tweaks can translate into significant savings in time and cost. By continuously monitoring performance metrics and incrementally refining each layer of the stack, you can transform potential bottlenecks into opportunities for efficiency gains. Let the data guide you, and you will unlock the full potential of your AI system.

最后请记住，尽管硬件与软件栈看起来像是一大堆无从管理、相互关联的旋钮和开关，但小的调整却能转化为时间和成本上的显著节省。通过持续监控性能指标并对栈的每一层进行渐进式的精细调优，你就能把潜在的瓶颈转化为提升效率的机会。让数据来引导你，你就能释放 AI 系统的全部潜力。
