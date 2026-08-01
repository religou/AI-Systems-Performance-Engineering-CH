# 第 13 章　PyTorch 的性能剖析、调优与扩展

AI 训练与推理流水线可能在每一层都遭遇性能瓶颈（bottleneck），包括 Python 解释器开销（overhead）、CPU 主机侧的数据加载停顿、CUDA 核函数利用不足，以及 GPU 设备内存争用。要做到有效优化，你需要借助覆盖整个系统的多种工具，在技术栈的多个层次上进行剖析（profiling）。

本章聚焦于在现代 NVIDIA GPU 上运行的 PyTorch 工作负载的性能剖析、调试与系统级调优（tuning）。我们将探讨如何利用 PyTorch 内置的剖析器、NVIDIA 的 Nsight 工具，以及使用 Linux perf 的 CPU 剖析来定位并修复瓶颈——此外还包括 PyTorch 内存剖析与内存分配器调优。我们还会讨论 PyTorch 如何使用 CUDA 流（CUDA stream）实现并发（concurrency），以及如何用 CUDA Graphs 降低核函数启动开销。

接下来，我们将展示如何优化数据流水线，并借助 PyTorch 分布式数据并行（Distributed Data Parallel，DDP）、完全分片数据并行（Fully Sharded Data Parallel，FSDP）以及其他模型并行策略扩展（scaling）到多块 GPU。随后，我们会演示如何剖析多 GPU 与多节点环境，包括全局追踪分析（Holistic Trace Analysis，HTA）与 Perfetto。

在全章中，我们强调性能权衡，并给出量化示例，聚焦于核函数执行时间、硬件利用率指标、内存占用、数据加载效率，以及扩展的整体成本效益。读完本章，你应当理解如何在整个技术栈上，对 PyTorch 工作负载实施一套有效且全局性的剖析与调优方法。

## NVTX 标记与剖析工具

要全面把握性能全貌，就必须在多个层次上进行剖析，并使用覆盖整个系统的工具。从业者与性能工程师有一套通用工具和最佳实践，用于在系统技术栈的所有层次上执行全局性剖析。

在介绍这些工具之前，有必要先重点说明 NVIDIA Tools Extension（NVTX）以及 NVTX 标记（NVTX marker）。这些标记在剖析器的时间线视图（timeline view）中标示出时间区间，并让不同的剖析器能够在相同阶段之间关联事件。

例如，一个名为 "forward" 的 NVTX 区间（NVTX range）会同时出现在 PyTorch 剖析器的追踪（trace）和 Nsight Systems 的时间线中。这让技术栈不同层次上的跨工具分析变得容易得多。大多数现代 AI 框架和库都支持 NVTX 标记，包括 PyTorch 以及与 CUDA 生态相关的一切。

NVTX 标记可以通过 CUDA C++、PyTorch，或任何支持 NVIDIA GPU 的 C++ 或 Python 库（如 OpenAI Triton、PyCUDA、CuPy、cuTile、cuTe、CUTLASS 等）注入代码。大多数库已经替你在关键代码区域自动注入了 NVTX 标记，例如 "train_step"、"forward"、"backward"、"optimizer_step" 等。不过，你也可以自行注入，例如在 PyTorch 中使用 torch.profiler.record_function() 和 torch.cuda.nvtx.range_push()。

既然我们已经说明了如何用 NVTX 标记标注代码中值得关注的部分，接下来就讨论那些能够摄取、对齐并可视化这些标记的工具。表 13-1 汇总了常见的剖析工具，以及它们的适用范围、关键特性和典型用途。这张表可以帮助你在优化旅程的每个阶段选择合适的工具。

表 13-1. 剖析与可视化工具汇总

| 工具                     | 范围                                    | 特性                                                             | 典型用途                                                                                   |
| ------------------------ | --------------------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| PyTorch 剖析器（Kineto） | PyTorch 内部的算子级剖析（CPU/GPU）     | 支持 NVTX 标记、形状记录、内存统计、追踪导出，以及识别编译图中断 | 对模型代码做细粒度拆解；定位慢算子、GPU 核函数启动开销，或 forward/backward 时间的不均衡。 |
| Nsight Systems (nsys)    | 系统级时间线（CPU、GPU、操作系统、I/O） | 将 CPU 线程与 GPU 流统一到一条时间线、集成 NVTX、支持多进程      | 端到端查看训练/推理流水线；发现数据加载器停顿、CPU-GPU 重叠问题，或 GPU 间同步延迟。       |
| Nsight Compute (ncu)     | GPU 核函数分析（逐核函数）              | 逐核函数的硬件指标、源码关联、Roofline 分析、占用率与吞吐量报告  | 在定位到热点核函数后深入分析其效率；判断核函数是访存受限还是计算受限，以及原因。           |

下面是表 13-1 中每种剖析工具的详细说明：

_PyTorch 剖析器（PyTorch profiler，Kineto）_ 在 PyTorch 内部，torch.profiler 基于开源项目 Kineto，提供 CPU 与 CUDA/GPU 运行时（runtime）的算子级（op-level）拆解。此外，它还能通过简单的 Python 上下文管理器记录输入形状并抓取内存快照。PyTorch 剖析器可以在训练与推理工作负载中捕获详细的时间线追踪和硬件计数器，并利用 NVTX 区间对齐事件。它提供从 Python 代码一直到 CUDA 核函数的端到端可观测性——甚至还能针对数据加载停顿、低效 CUDA 代码等常见问题给出性能建议。

*Nsight Systems (*nsys*)* 为了实现系统级关联——涵盖 CPU 线程、GPU 核函数、操作系统事件、I/O 与互连流量——NVIDIA Nsight Systems 生成一条统一的时间线视图。它的 GUI 和 CLI 报告可以在多进程、多节点运行中合并 NVTX 区域、Python 调用栈与 CUDA 流。这让人很容易发现 I/O 和同步停顿在何处影响了计算性能。

*Nsight Compute (*ncu*)* 与 Nsight Systems 互补的是用于逐核函数分析的 NVIDIA Nsight Compute。Nsight Compute 收集详细的硬件指标，例如占用率（occupancy）、内存带宽和 SM（流式多处理器，streaming multiprocessor）利用率。它甚至能生成映射到源码的 Roofline 图。在其他更高层的工具识别出哪些核函数是热点之后，Nsight Compute 有助于回答某个核函数为何缓慢（例如访存受限（memory bound）、占用率偏低）。

_PyTorch 内存剖析器（PyTorch memory profiler）_ PyTorch 还内置了一个内存剖析器，你可以在 torch.profiler 中通过 profile_memory=True 启用它。PyTorch 内存剖析器会按操作拆解 GPU 内存的峰值分配和累计分配。这能揭示那些原本可能被忽视的内存使用热点。

_Linux_ perf 在主机侧，Linux 的 perf 工具可以采样 CPU 硬件计数器，包括周期数、指令数和缓存缺失——并展开完整的 C/C++ 与 Python 调用图。从 perf sched 入手，你能看到 CPU 线程何时因 I/O 或线程调度/同步而处于空闲。这能揭示数据预处理循环、Python 的 GIL 或同步中可能使 GPU 挨饿的瓶颈。

_全局追踪分析（Holistic Trace Analysis）_ Meta 开源的全局追踪分析（HTA）工具会摄取 PyTorch 剖析器的追踪，帮助诊断多 GPU 瓶颈。借助 HTA，你可以把带有 NVTX 区间的分布式训练时间线与 CUDA 核函数追踪并列可视化。通过深入分析内存分配模式随时间的变化，你能识别出 GPU 空闲的时段——包括 GPU 相互等待的时刻。

> TensorBoard 的 PyTorch 追踪可视化插件已被弃用。请改用 Perfetto 进行时间线查看，并使用 Meta 的 HTA 进行分布式追踪分析。

_Chrome 追踪与 Perfetto 查看器_ 要在网页端探索大型 PyTorch 剖析器追踪文件，你可以使用 Chrome tracing（例如在浏览器中打开 chrome://tracing）和 Perfetto 界面。它们会加载 JSON 追踪文件，让你交互式地探索时间线视图和火焰图。它们甚至支持对追踪数据做细粒度过滤和 SQL 查询——在事件追踪与关联层面可细至亚毫秒级。Chrome 追踪与 Perfetto 界面非常适合在组织成员之间共享剖析结果，以进行跨团队分析。（注：Chrome 的旧版追踪查看器已被弃用，因此查看和分析追踪时应优先使用 Perfetto 的网页界面和 SQL 引擎。）

_TorchEval（PyTorch 的指标库）_ 另一个项目 TorchEval 让你能够在统一的接口内，记录并监控模型的吞吐量（throughput）、延迟（latency）和质量指标，同时兼顾训练与评估指标。TorchEval 是 PyTorch 官方的指标库，为端到端的性能和质量指标提供了简洁的 API。这个库很容易接入训练循环，并可在分布式环境中集成。

_ExecuTorch_ 面向嵌入式、移动和边缘设备，ExecuTorch 项目支持在诸如 Meta 眼镜等轻量级运行时环境中对 PyTorch 模型进行剖析、可视化和调试。ExecuTorch 具有小而动态的内存占用，并支持 Linux、iOS、Android 和嵌入式系统。Hugging Face 通过其 Optimum ExecuTorch 项目支持 ExecuTorch；如果你已经在使用 Hugging Face 生态（如 Hugging Face Transformers），这个环境会很容易集成。

接下来，让我们深入一个示例，看看如何使用这些剖析器来定位性能瓶颈。随后，我们会施加有针对性的优化，并验证性能改进。

## 剖析 PyTorch 以定位瓶颈

让我们剖析一个示例——专家混合（mixture-of-experts，MoE）Transformer 模型，看看这些工具的实际效果。MoE 是带有多个专家层的 LLM——每个专家都是一个前馈网络。将 token 路由到各专家由一套专家门控系统管理。我们将运行一次训练迭代，捕获详细的性能追踪，并分析输出以指导优化。

### 使用 PyTorch 剖析器

首先，我们搭建模型和输入。我们用 Hugging Face Transformers 加载模型和分词器，将模型移动到 GPU，并准备一小批输入，如下所示：

```
# train.py
# Set up model and data
model_name = "..."
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
batch_size = 4
input_texts = ["MoEs are great."] * batch_size
enc = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
input_ids = enc.input_ids.to(device)
attention_mask = enc.attention_mask.to(device)
labels = input_ids.clone()  # For LM training, labels are the inputs
                            # (next-token prediction)
```

为避免把一次性的初始化开销计入测量，我们在剖析之前先运行几次预热迭代。这会通过编译 JIT 核函数、填充缓存等方式，为分析和基准测试准备好模型。这样一来，我们测量的那次迭代就能代表稳态性能。下面是准备模型的代码：

```
# Warm-up (not profiled)
for _ in range(5):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

```

现在，我们使用 PyTorch 的剖析器和 NVTX 剖析一次训练迭代。我们用 torch.profiler.profile() 包裹这次迭代，并用 record_function 和 NVTX 区间标记高层区域，包括 "forward"、"backward" 和 "optimizer_step"，如下所示：

```
from torch import profiler
with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA],
    record_shapes=True,  # record tensor shapes
    profile_memory=True, # track GPU memory usage per op
    with_stack=True,     # enable stack tracing
    with_flops=True      # capture FLOPs counters
) as prof:
    with profiler.record_function("train_step"):
        # Forward pass
        torch.cuda.nvtx.range_push("forward")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)
        loss = outputs.loss
        # end of forward
        torch.cuda.nvtx.range_pop()
        # Backward pass and optimization
        torch.cuda.nvtx.range_push("backward")
        loss.backward()
        torch.cuda.nvtx.range_push("optimizer_step")
        optimizer.step()

        # end of optimizer_step
        torch.cuda.nvtx.range_pop()
        optimizer.zero_grad()

        # end of backward
        torch.cuda.nvtx.range_pop()
```

在这段代码中，PyTorch 剖析器会记录 train_step 期间的所有 CPU 与 GPU 活动。我们用 record_function("train_step") 定义一个顶层区域，并为各子阶段（"forward"、"backward"、"optimizer_step"）插入 NVTX 标记。这些标记会出现在剖析器的时间线中，用来划分迭代的各个阶段。

剖析器还能突出显示模型中已编译与未编译的区域。我们会在本章后面以及下一章介绍 PyTorch 编译器（PyTorch Compiler）、图中断（graph break）以及缓解图中断的机制。

例如，使用 torch.compile 时，追踪中会显示诸如 Compiled Function 之类的事件，并标示出任何图中断（见图 13-1）。这有助于精确定位模型在何处回退到了 eager 执行，从而指导进一步的优化。

![图 13-1. 已编译区域（左、中，粉色）与未编译区域（右，绿色）（来源：https://oreil.ly/Z_fJG）](AI%20Systems%20Performance%20Engineering-ch13_images/figure-13-1.png)

执行完成后，我们可以调用 prof.key_averages().table() 来查看算子级结果，打印出一张按运行时间排序的头部算子简表。在下一个代码块中，我们请求按自身 CUDA 时间排序的前 10 个操作；自身 CUDA 时间指的是花在每个操作自有 CUDA 核函数上的时间，不包括该核函数派生出的子操作。按 CUDA 执行时间排名前 10 的操作汇总于表 13-2：

```
with profiler.profile(
    activities=[ProfilerActivity.CPU,
                ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    train_step(...)
...
print(
    prof.key_averages()
        .table(
            sort_by="self_cuda_time_total",
            row_limit=10,
            fields=["self_cuda_time_total",
                    "calls"]
        )
)
```

表 13-2. 剖析器给出的一次训练迭代中按 CUDA 执行时间排名前 10 的操作

| 操作             | 自身 CUDA 总计 | 调用次数 |
| ---------------- | -------------- | -------- |
| aten::matmul     | 43.00 ms       | 128      |
| aten::linear     | 35.50 ms       | 64       |
| dispatch         | 18.70 ms       | 2        |
| combine          | 12.10 ms       | 2        |
| aten::layer_norm | 10.20 ms       | 16       |
| aten::softmax    | 5.70 ms        | 4        |
| aten::scatter    | 4.10 ms        | 16       |
| aten::gather     | 3.60 ms        | 16       |
| aten::to         | 2.90 ms        | 8        |
| aten::add\_      | 2.20 ms        | 64       |

这里可以看到，矩阵乘法操作（aten::matmul，以及它在 aten::linear 中的使用）主导了 CUDA 时间，占据了这次迭代的绝大部分。这些操作对应于专家前馈网络（feed-forward network，FFN）的 GEMM（通用矩阵乘法，general matrix multiply）。具体来说，每次迭代对 matmul 有 128 次调用。这是合理的，因为我们有 64 个专家——而每个专家在 forward 和 backward 两个阶段各做一次 matmul。

在表 13-2 中，接下来开销最大的是 dispatch 和 combine 操作。它们是自定义的 C++/CUDA 核函数，负责把 token 重新分发给各专家——再收集专家的输出。dispatch 操作运行两次——forward 阶段一次、backward 阶段一次——总计 18.7 ms。combine 也运行两次，共 12.1 ms。这两个操作合计又占用了 30.8 ms 的 GPU 时间。其余时间则分散在层归一化、激活函数等其他较小的操作上。

这个剖析示例的关键要点是：专家 FFN 的 matmul 是最大的瓶颈，其次是 dispatch 和 combine 核函数。它们共同主导了一次训练迭代的运行时间。要进一步提升性能，我们应当针对这些操作——要么直接优化它们，要么减少它们被调用的次数。

### 使用 Nsight Systems 与 NVTX 时间线做系统剖析

我们插入的 NVTX 标记让用 Nsight Systems 分析时间线变得直截了当。要按阶段聚合指标，我们可以用 nsys 结合基于 NVTX 的汇总来剖析代码，如下面的 CLI 命令所示：

```
nsys profile \
   --output=profile \
   --stats=true \
   -t cuda,nvtx \
   python train.py
```

这里，-t cuda,nvtx 选项指示 Nsight Systems 同时追踪 CUDA API 调用和 NVTX 区间。剖析完成后，我们可以在 Nsight Systems 的 GUI 中打开 profile.nsys-rep 文件（--output=profile），或用 CLI 将 NVTX 汇总打印到终端。随后，我们可以用 CLI，对 profile.nsys-rep 文件执行下列命令之一来生成 NVTX GPU 投影汇总（GPU projection summary），如下所示，以便用 Nsight Systems 将区间与投影出的 GPU 工作量相互校验：

```
nsys stats --report=nvtx_gpu_proj_sum \
  profile.nsys-rep
# or
nsys recipe nvtx_gpu_proj_sum \
  profile.nsys-rep
```

你可以在持续构建与集成流水线中使用这些命令之一，来监控并检测任何性能回归（regression）。该 CLI 命令的结果汇总于表 13-3。

表 13-3. 使用 Nsight Systems 得到的一次 train_step 迭代的 NVTX GPU 投影汇总

| NVTX 区间      | GPU 时间（ms） | 自身 GPU 时间（ms） | 子 GPU 时间（ms） | 实例数（调用次数） |
| -------------- | -------------- | ------------------- | ----------------- | ------------------ |
| train_step     | 138.0          | 0.0                 | 138.0             | 1                  |
| forward        | 60.5           | 60.5                | 0.0               | 130                |
| backward       | 58.3           | 58.3                | 0.0               | 130                |
| optimizer_step | 19.2           | 19.2                | 0.0               | 1                  |

这里可以看到，train_step 区间包含 forward、backward 和 optimizer 三个子区间。这份 NVTX GPU 投影汇总确认，train_step 下的 GPU 总时间为 138 ms。这一时间与表 13-2 中 PyTorch 剖析器输出的 forward、backward 和 optimizer_step 时间之和相吻合，表明两种工具之间是一致的。

此外，尽管表 13-3 只显示了一次 optimizer*step 调用，但它的 NVTX 区间实际上把全部 64 次 aten::add* CUDA 核函数启动（如表 13-2 所示，每个专家一次 add\_）都归拢在这个 optimizer_step 标记之下。

请注意，Nsight Systems 之所以把这 64 次 aten::add* 调用（例如 64 路专家并行策略）归并到单个 optimizer_step 标记下，是因为它使用 CUDA Profiling Tools Interface（CUPTI）在主机端捕获 NVTX 的 push/pop 事件。随后，它把异步的 GPU 核函数执行时间"投影"到这些由 CPU 定义的区间上。因此，它会把每个 GPU 起止时间戳落在相应 push 与 pop 调用之间的核函数的持续时间累加起来。这就得到一个累计 GPU 时间，恰好等于各个 aten::add* 核函数时间的总和。

> 由于在没有附加剖析器时 NVTX 标记的开销极低，这种投影机制堪称理想：它在提供 GPU 工作与高层代码区域端到端关联的同时，几乎不增加任何开销。

forward 和 backward 区间各自的自身 GPU 时间（self GPU time）都等于其总时间，因为我们没有在它们内部再嵌套更深的区间。因此，子 GPU 时间（child GPU time）为 0 ms。而 train_step 则几乎全部时间都是子 GPU 时间，因为它只是包裹这些嵌套阶段的一层外壳。

NVTX GPU 投影汇总还显示，在每次迭代中，我们在 train_step 内部观察到 130 个 GPU 活动。它们包括核函数启动以及内存拷贝等其他设备操作，因此并不是严格地与核函数一一对应。

如表 13-3 所示，这 130 次 GPU 核函数调用同样发生在 backward 和 forward 两个阶段。操作与 NVTX 实例之间的这种一一对应，意味着我们的插桩捕获了每一个重要操作。

> 我们在表 13-3 中展示的 NVTX 汇总是一份便捷的文本概览。若要做可视化分析，时间线 GUI 可以显示重叠的核函数执行、CPU 线程状态，甚至 CUDA API 的开销事件。实践中，你需要在可视化时间线中确认主机侧的数据加载和预处理是否与 GPU 计算相互重叠。任何较大的空隙，即"气泡"（bubble），都表明存在问题。用于同步的小空隙则是预期且合理的。

在多 GPU 运行中，Nsight Systems 或 HTA 的时间线视图可以揭示 NVLink 或 InfiniBand/以太网是否被有效利用——或者某个节点是否在等待通信或网络延迟时无事可做。这会提示存在次优的同步或负载不均衡。

追踪 GPU 通信事件很重要，包括用 Nsight Systems 和 HTA 提供追踪，来记录 NCCL 的 all-reduce 调用以及 NVLink/NVSwitch 活动。这些有助于验证 GPU 在诸如基于 NVL72 的系统这类超大规模 GPU 域中是否保持繁忙。

细致的剖析可以确保系统在这些大型 NVLink 集群中使用了恰当的 GPU 间同步，并均衡了工作负载。现在，让我们聚焦系统中开销最大的核函数之一：矩阵乘法，即 matmul。

### 通用矩阵乘法（GEMM）的核函数 Roofline 分析

为了更深入地分析专家的 matmul，我们调用命令行剖析器 Nsight Compute（ncu），按名称定位特定的 GEMM 核函数。我们将收集与 roofline 相关的指标，以判断它是计算受限（compute bound）还是访存受限，如下所示：

```
ncu \
  --target-processes all \
  --kernel-name-regex "matmul" \
  --metrics \
    gpu__time_duration.avg, \
    gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed, \
    lts__throughput.avg.pct_of_peak_sustained_elapsed, \
    sm__sass_thread_inst_executed_op_fp32_pred_on.sum, \
    sm__warps_active.avg.pct_of_peak_sustained_active \
  --csv full \
  -o matmul_roofline_report \
  python train.py
```

这里，我们为任何名称匹配 "matmul" 的核函数收集硬件计数器。具体来说，我们收集几个关键指标，包括以峰值百分比表示的 GPU DRAM 带宽和 L2 带宽（gpu**dram_throughput.avg.pct_of_peak_sustained_elapsed、lts**throughput.avg.pct_of_peak_sustained_elapsed）、作为计算代理的 FP32 指令数（sm**sass_thread_inst_executed_op_fp32_pred_on.sum）、核函数时间（gpu**time_duration.avg），以及实测占用率（sm\_\_warps_active.avg.pct_of_peak_sustained_active）。

> 指标标识符在不同的 Nsight Compute 版本之间可能略有差异。如果找不到某个指标，请用 UI 定位当前的名称并相应替换。这里，我们剖析的是 SM 和 DRAM 占峰值持续值的百分比，以及实测占用率。

在应用了目前为止讨论过的优化——例如降低精度、融合核函数以及提升算术强度（arithmetic intensity）——之后，我们可以重新运行 ncu 命令，验证这些优化是否产生了正面效果。表 13-4 展示了应用这些提升算术强度的优化前后的对比。

表 13-4. 专家 matmul 核函数在算术强度优化前后的 Roofline 分析

| 核函数配置 | 峰值 FLOPS 占比（SM 计算吞吐量） | 峰值内存带宽占比（内存吞吐量） | 实测 SM 占用率 | 特征                          |
| ---------- | -------------------------------- | ------------------------------ | -------------- | ----------------------------- |
| Baseline   | 50%                              | 70%                            | 60%            | 访存受限（因内存传输而停顿）  |
| Optimized  | 85%                              | 40%                            | 80%            | 计算受限（接近硬件 roofline） |

这里可以看到，在基线运行中，主要的 GEMM 核函数只达到了约 50% 的峰值计算 FLOPS、70% 的峰值内存带宽，以及中等的 60% SM 占用率（每个 SM 上平均活跃的 warp 数）。这样的占用率不足以完全隐藏内存延迟。

> 并不存在一个通用的占用率目标。许多高性能核函数在 25–50% 的实测占用率下就能实现完全的延迟隐藏。请结合 Nsight Compute 的占用率指标、停顿原因（stall reason）拆解，以及每活跃周期可调度的 warp 数，来判断提高占用率是否能减少该核函数的停顿。如果核函数无法调度足够多的 warp 来覆盖停顿时段，就会因内存请求得不到足够快的响应而产生空闲周期。

基线指标表明这是一个访存受限的核函数，因为它的执行被内存传输所阻塞而停顿。其结果是有大量计算能力未被利用，这进一步印证了该工作负载目前并非计算受限。我们的目标是让这个核函数更偏向计算受限，从而充分利用这块 GPU 所提供的大量 FLOPS。

在优化后的版本中（例如融合核函数、提升算术强度并减少内存搬运），峰值 FLOPS 占比提升到 85%，峰值内存带宽降到 40%，占用率提升到 80%。我们实质上把这个核函数从访存受限转变为了计算受限——大大接近硬件的 roofline 上限，如图 13-2 所示。

![图 13-2. 提升该核函数算术强度前后的 Roofline 图](AI%20Systems%20Performance%20Engineering-ch13_images/figure-13-2.png)

到目前为止，我们的剖析都集中在 GPU 性能上。同样重要的是，不要把时间浪费在 CPU 上或 I/O 操作上。在下一节中，我们将在主机侧继续这段剖析之旅。

### 使用 Linux perf 做 CPU 与 GPU 剖析

为了更全面地了解时间在主机和设备两侧的花费情况，我们可以用 Linux perf 分析 CPU 周期数、缓存缺失、分支预测缺失等。随后，我们可以根据这些洞见推动一系列优化，逐一应用它们，并测量改进效果。

首先，让我们在一个搭载基于 ARM 的 Grace CPU 并配对 Blackwell GPU 的节点上，运行一次轻量级的 perf stat，以便在 MoE 训练运行期间收集 CPU 侧的统计数据。下面是 CLI 命令及示例输出：

```
perf stat -e \
  cycles,instructions,cache-misses,branch-misses \
  python train.py
  Performance counter stats for 'python train.py':
# 0.600 CPUs utilized
1,200.345 msec task-clock
# Approximately 2.0 GHz
2,400,567,890      cycles
# 1.58 insn per cycle
3,800,123,456      instructions
# 0.32% of all cache refs
12,345,678      cache-misses
# 0.12% of all branches
4,567,890      branch-misses
 1.234567890 seconds time elapsed
```

这份来自 perf stat 的报告显示了 CPU 利用率、周期数、每周期指令数（instructions per cycle，IPC）以及缓存/分支缺失。在我们的运行中，任务时钟约为 1.2 秒，在测量区间内只用到了单个 CPU 核心的约 60%（0.600）。这在意料之中，因为繁重的工作大多由 GPU 完成。较低的缓存缺失率和分支缺失率暗示，对于这个工作负载，CPU 侧的内存访问模式和分支预测都相对高效。

然而，每周期指令数（IPC）仅为 1.58，表明 CPU 的发射速率远低于单个 Grace CPU 核心（ARM Neoverse V2）每周期 8 条指令的理论最大 IPC。这说明在这个特定工作负载中可能存在低效之处，例如内存延迟、I/O 停顿或主机计算问题。

我们可以进一步使用 perf record 和 perf report，来精确定位训练期间是哪些 Python 和 C++ 函数主导了 CPU 执行时间。这些 CLI 命令如下所示：

```
perf record -F 2000 -g --call-graph dwarf -o perf.data \
    python train.py
```

这里，我们用 perf record 以 2000 Hz（-F 2000）采样，并通过指定 -g --call-graph dwarf 捕获完整的 C/C++ 与 Python 调用栈。DWARF 是 Debugging With Attributed Record Formats 的缩写，是一种嵌入在编译后二进制文件（如 ELF 文件）中的标准调试数据格式。DWARF 输出的追踪被保存到 perf.data（-o perf.data）。随后，我们用 perf report 生成一份关于最热调用栈及其采样占比的汇总报告：

```
perf report --stdio -n -g -i perf.data
# Samples  Command   Shared Object        Symbol
# ........ ........  ...................  .................................
     45.0%  python    python               py::forward <...> /src/train.py
     20.5%  python    python               aten::matmul
     10.2%  python    python               dataloader_iter_next
      8.7%  python    libnccl.so           ncclAllReduce
      5.3%  python    libc.so.6            read
      ...   ...       ...                  ...
```

这里可以看到，Python 解释器的 forward 函数——也就是我们训练循环的 Python 侧——占了 45.0% 的 CPU 采样。PyTorch 的 C++ aten::matmul 操作占 20.5%，DataLoader 的迭代器 next 函数占 10.2%，一次 NCCL all-reduce 调用占 8.7%，I/O 读取占 5.3%。

这些百分比告诉我们应该把优化精力投向何处。基于这份剖析结果，我们针对每个瓶颈制定了具体的缓解方案，以提升系统性能：

_Python 开销过大（占 45%，位于_ py::forward*）* 使用 PyTorch 的 JIT 编译器 torch.compile（下一节讨论），以消除解释器开销，并把 Python 侧的操作融合进优化后的 CUDA 代码。

_大型_ matmul _热点（占 20.5%，位于_ aten::matmul*）* 要么使用 PyTorch 编译器来优化这段代码，要么把这个关键的矩阵乘法迁移到自定义的 CUDA C++ 核函数（例如融合核函数）中，从而绕过 Python，直接使用优化后的 CUDA 代码。

_数据加载停顿（占 10.2%，位于_ dataloader_iter_next*）* 增大 PyTorch DataLoader 的 num_workers。常见的经验法则是每个 CPU 配一个 worker，但你可以尝试更多，以找到合适的 I/O 并行度。只要注意不要让 CPU 核心过载即可。你还应当启用 persistent_workers=True，让 worker 进程在多个 epoch 之间保持存活，从而避免每个 epoch 的启动开销。融合或并行化多个 torch.utils.data.DataPipe，这能减少复杂数据流水线中的 Python 开销。

_梯度同步（gradient synchronization）开销（占 8.7%，位于_ ncclAllReduce*）* 优化多 GPU 通信。例如，你可以增大 DistributedDataParallel 中的梯度桶大小。常见做法是把 bucket_cap_mb 从 25 MB 增大到 50 MB，好让 NCCL 能更早启动 all-reduce 操作，并把它们与 backward 计算重叠。你也可以考虑梯度压缩技术，或 NVIDIA 针对 8 位梯度的 NCCL 压缩，以降低带宽占用。这些做法可能会略微牺牲一点精度。

_主机 I/O 瓶颈（占 5.3%，位于_ read _系统调用）_ 在 DataLoader 中使用锁页内存（pin_memory=True）和非阻塞的 GPU 拷贝（.to(device, non_blocking=True)），以重叠 CPU 到 GPU 的数据传输。此外，你可以批量读取文件，或把大量小文件打包成有利于大块顺序读取的优化数据集格式，例如 Arrow、WebDataset（tar 分片）、TFRecord 或 Parquet 分块。最好优先选用连续的分片格式，而不是每样本一个文件。在使用较大的预取深度（例如 prefetch_factor=4 或 8）时，优先使用锁页的主机缓冲区。再配合 persistent_workers=True，由于计算与通信高效重叠，你的系统会让加载线程持续保持繁忙。这样在读取大型语料库中的大量小文件时，就能消除逐文件的开销。

这些方法，再结合较大的操作系统预读和 NVMe SSD，将提升 I/O 吞吐量。此外，较新的文件系统和存储库（如 NVIDIA Magnum IO）也有助于高效地把数据流水线式地送往 GPU。

制定好这一方案后，你应当系统地逐项应用每个优化并测量其效果。记得逐一实现并测试这些优化，以验证每一项确实提升了性能。这有助于避免多处改动以意想不到的方式相互作用的情况。通过隔离每一处改动，你就能知道哪些调整有正面效果，哪些没有。

在装有 NVIDIA 性能监控单元（Performance Monitoring Unit，PMU）驱动的系统上，你可以用 perf 在采样 CPU 计数器的同时，采样 NVIDIA 芯片互连与 fabric 计数器，例如在 /sys/bus/event_source/devices 下以 nvidia_nvlink_c2c\* 形式暴露的 NVLink-C2C 设备。可以用 perf list 并检查 sysfs 下的 nvidia_pmu 条目来确认其可用性。
