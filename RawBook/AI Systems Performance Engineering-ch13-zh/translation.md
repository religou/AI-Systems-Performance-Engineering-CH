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

| 工具 | 范围 | 特性 | 典型用途 |
| --- | --- | --- | --- |
| PyTorch 剖析器（Kineto） | PyTorch 内部的算子级剖析（CPU/GPU） | 支持 NVTX 标记、形状记录、内存统计、追踪导出，以及识别编译图中断 | 对模型代码做细粒度拆解；定位慢算子、GPU 核函数启动开销，或 forward/backward 时间的不均衡。 |
| Nsight Systems (nsys) | 系统级时间线（CPU、GPU、操作系统、I/O） | 将 CPU 线程与 GPU 流统一到一条时间线、集成 NVTX、支持多进程 | 端到端查看训练/推理流水线；发现数据加载器停顿、CPU-GPU 重叠问题，或 GPU 间同步延迟。 |
| Nsight Compute (ncu) | GPU 核函数分析（逐核函数） | 逐核函数的硬件指标、源码关联、Roofline 分析、占用率与吞吐量报告 | 在定位到热点核函数后深入分析其效率；判断核函数是访存受限还是计算受限，以及原因。 |

下面是表 13-1 中每种剖析工具的详细说明：

*PyTorch 剖析器（PyTorch profiler，Kineto）* 在 PyTorch 内部，torch.profiler 基于开源项目 Kineto，提供 CPU 与 CUDA/GPU 运行时（runtime）的算子级（op-level）拆解。此外，它还能通过简单的 Python 上下文管理器记录输入形状并抓取内存快照。PyTorch 剖析器可以在训练与推理工作负载中捕获详细的时间线追踪和硬件计数器，并利用 NVTX 区间对齐事件。它提供从 Python 代码一直到 CUDA 核函数的端到端可观测性——甚至还能针对数据加载停顿、低效 CUDA 代码等常见问题给出性能建议。

*Nsight Systems (*nsys*)* 为了实现系统级关联——涵盖 CPU 线程、GPU 核函数、操作系统事件、I/O 与互连流量——NVIDIA Nsight Systems 生成一条统一的时间线视图。它的 GUI 和 CLI 报告可以在多进程、多节点运行中合并 NVTX 区域、Python 调用栈与 CUDA 流。这让人很容易发现 I/O 和同步停顿在何处影响了计算性能。

*Nsight Compute (*ncu*)* 与 Nsight Systems 互补的是用于逐核函数分析的 NVIDIA Nsight Compute。Nsight Compute 收集详细的硬件指标，例如占用率（occupancy）、内存带宽和 SM（流式多处理器，streaming multiprocessor）利用率。它甚至能生成映射到源码的 Roofline 图。在其他更高层的工具识别出哪些核函数是热点之后，Nsight Compute 有助于回答某个核函数为何缓慢（例如访存受限（memory bound）、占用率偏低）。

*PyTorch 内存剖析器（PyTorch memory profiler）* PyTorch 还内置了一个内存剖析器，你可以在 torch.profiler 中通过 profile_memory=True 启用它。PyTorch 内存剖析器会按操作拆解 GPU 内存的峰值分配和累计分配。这能揭示那些原本可能被忽视的内存使用热点。

*Linux* perf 在主机侧，Linux 的 perf 工具可以采样 CPU 硬件计数器，包括周期数、指令数和缓存缺失——并展开完整的 C/C++ 与 Python 调用图。从 perf sched 入手，你能看到 CPU 线程何时因 I/O 或线程调度/同步而处于空闲。这能揭示数据预处理循环、Python 的 GIL 或同步中可能使 GPU 挨饿的瓶颈。

*全局追踪分析（Holistic Trace Analysis）* Meta 开源的全局追踪分析（HTA）工具会摄取 PyTorch 剖析器的追踪，帮助诊断多 GPU 瓶颈。借助 HTA，你可以把带有 NVTX 区间的分布式训练时间线与 CUDA 核函数追踪并列可视化。通过深入分析内存分配模式随时间的变化，你能识别出 GPU 空闲的时段——包括 GPU 相互等待的时刻。

> TensorBoard 的 PyTorch 追踪可视化插件已被弃用。请改用 Perfetto 进行时间线查看，并使用 Meta 的 HTA 进行分布式追踪分析。

*Chrome 追踪与 Perfetto 查看器* 要在网页端探索大型 PyTorch 剖析器追踪文件，你可以使用 Chrome tracing（例如在浏览器中打开 chrome://tracing）和 Perfetto 界面。它们会加载 JSON 追踪文件，让你交互式地探索时间线视图和火焰图。它们甚至支持对追踪数据做细粒度过滤和 SQL 查询——在事件追踪与关联层面可细至亚毫秒级。Chrome 追踪与 Perfetto 界面非常适合在组织成员之间共享剖析结果，以进行跨团队分析。（注：Chrome 的旧版追踪查看器已被弃用，因此查看和分析追踪时应优先使用 Perfetto 的网页界面和 SQL 引擎。）

*TorchEval（PyTorch 的指标库）* 另一个项目 TorchEval 让你能够在统一的接口内，记录并监控模型的吞吐量（throughput）、延迟（latency）和质量指标，同时兼顾训练与评估指标。TorchEval 是 PyTorch 官方的指标库，为端到端的性能和质量指标提供了简洁的 API。这个库很容易接入训练循环，并可在分布式环境中集成。

*ExecuTorch* 面向嵌入式、移动和边缘设备，ExecuTorch 项目支持在诸如 Meta 眼镜等轻量级运行时环境中对 PyTorch 模型进行剖析、可视化和调试。ExecuTorch 具有小而动态的内存占用，并支持 Linux、iOS、Android 和嵌入式系统。Hugging Face 通过其 Optimum ExecuTorch 项目支持 ExecuTorch；如果你已经在使用 Hugging Face 生态（如 Hugging Face Transformers），这个环境会很容易集成。

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

例如，使用 torch.compile 时，追踪中会显示诸如 Compiled Function 之类的事件，并标示出任何图中断（见图 13-1）。这有助于精确定位模型在何处回退到了即时执行，从而指导进一步的优化。

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

| 操作 | 自身 CUDA 总计 | 调用次数 |
| --- | --- | --- |
| aten::matmul | 43.00 ms | 128 |
| aten::linear | 35.50 ms | 64 |
| dispatch | 18.70 ms | 2 |
| combine | 12.10 ms | 2 |
| aten::layer_norm | 10.20 ms | 16 |
| aten::softmax | 5.70 ms | 4 |
| aten::scatter | 4.10 ms | 16 |
| aten::gather | 3.60 ms | 16 |
| aten::to | 2.90 ms | 8 |
| aten::add_ | 2.20 ms | 64 |

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

| NVTX 区间 | GPU 时间（ms） | 自身 GPU 时间（ms） | 子 GPU 时间（ms） | 实例数（调用次数） |
| --- | --- | --- | --- | --- |
| train_step | 138.0 | 0.0 | 138.0 | 1 |
| forward | 60.5 | 60.5 | 0.0 | 130 |
| backward | 58.3 | 58.3 | 0.0 | 130 |
| optimizer_step | 19.2 | 19.2 | 0.0 | 1 |

这里可以看到，train_step 区间包含 forward、backward 和 optimizer 三个子区间。这份 NVTX GPU 投影汇总确认，train_step 下的 GPU 总时间为 138 ms。这一时间与表 13-2 中 PyTorch 剖析器输出的 forward、backward 和 optimizer_step 时间之和相吻合，表明两种工具之间是一致的。

此外，尽管表 13-3 只显示了一次 optimizer_step 调用，但它的 NVTX 区间实际上把全部 64 次 aten::add_ CUDA 核函数启动（如表 13-2 所示，每个专家一次 add_）都归拢在这个 optimizer_step 标记之下。

请注意，Nsight Systems 之所以把这 64 次 aten::add_ 调用（例如 64 路专家并行策略）归并到单个 optimizer_step 标记下，是因为它使用 CUDA Profiling Tools Interface（CUPTI）在主机端捕获 NVTX 的 push/pop 事件。随后，它把异步的 GPU 核函数执行时间"投影"到这些由 CPU 定义的区间上。因此，它会把每个 GPU 起止时间戳落在相应 push 与 pop 调用之间的核函数的持续时间累加起来。这就得到一个累计 GPU 时间，恰好等于各个 aten::add_ 核函数时间的总和。

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

这里，我们为任何名称匹配 "matmul" 的核函数收集硬件计数器。具体来说，我们收集几个关键指标，包括以峰值百分比表示的 GPU DRAM 带宽和 L2 带宽（gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed、lts__throughput.avg.pct_of_peak_sustained_elapsed）、作为计算代理的 FP32 指令数（sm__sass_thread_inst_executed_op_fp32_pred_on.sum）、核函数时间（gpu__time_duration.avg），以及实测占用率（sm__warps_active.avg.pct_of_peak_sustained_active）。

> 指标标识符在不同的 Nsight Compute 版本之间可能略有差异。如果找不到某个指标，请用 UI 定位当前的名称并相应替换。这里，我们剖析的是 SM 和 DRAM 占峰值持续值的百分比，以及实测占用率。

在应用了目前为止讨论过的优化——例如降低精度、融合核函数以及提升算术强度（arithmetic intensity）——之后，我们可以重新运行 ncu 命令，验证这些优化是否产生了正面效果。表 13-4 展示了应用这些提升算术强度的优化前后的对比。

表 13-4. 专家 matmul 核函数在算术强度优化前后的 Roofline 分析

| 核函数配置 | 峰值 FLOPS 占比（SM 计算吞吐量） | 峰值内存带宽占比（内存吞吐量） | 实测 SM 占用率 | 特征 |
| --- | --- | --- | --- | --- |
| Baseline | 50% | 70% | 60% | 访存受限（因内存传输而停顿） |
| Optimized | 85% | 40% | 80% | 计算受限（接近硬件 roofline） |

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

*Python 开销过大（占 45%，位于* py::forward*）* 使用 PyTorch 的 JIT 编译器 torch.compile（下一节讨论），以消除解释器开销，并把 Python 侧的操作融合进优化后的 CUDA 代码。

*大型* matmul *热点（占 20.5%，位于* aten::matmul*）* 要么使用 PyTorch 编译器来优化这段代码，要么把这个关键的矩阵乘法迁移到自定义的 CUDA C++ 核函数（例如融合核函数）中，从而绕过 Python，直接使用优化后的 CUDA 代码。

*数据加载停顿（占 10.2%，位于* dataloader_iter_next*）* 增大 PyTorch DataLoader 的 num_workers。常见的经验法则是每个 CPU 配一个 worker，但你可以尝试更多，以找到合适的 I/O 并行度。只要注意不要让 CPU 核心过载即可。你还应当启用 persistent_workers=True，让 worker 进程在多个 epoch 之间保持存活，从而避免每个 epoch 的启动开销。融合或并行化多个 torch.utils.data.DataPipe，这能减少复杂数据流水线中的 Python 开销。

*梯度同步（gradient synchronization）开销（占 8.7%，位于* ncclAllReduce*）* 优化多 GPU 通信。例如，你可以增大 DistributedDataParallel 中的梯度桶大小。常见做法是把 bucket_cap_mb 从 25 MB 增大到 50 MB，好让 NCCL 能更早启动 all-reduce 操作，并把它们与 backward 计算重叠。你也可以考虑梯度压缩技术，或 NVIDIA 针对 8 位梯度的 NCCL 压缩，以降低带宽占用。这些做法可能会略微牺牲一点精度。

*主机 I/O 瓶颈（占 5.3%，位于* read *系统调用）* 在 DataLoader 中使用锁页内存（pin_memory=True）和非阻塞的 GPU 拷贝（.to(device, non_blocking=True)），以重叠 CPU 到 GPU 的数据传输。此外，你可以批量读取文件，或把大量小文件打包成有利于大块顺序读取的优化数据集格式，例如 Arrow、WebDataset（tar 分片）、TFRecord 或 Parquet 分块。最好优先选用连续的分片格式，而不是每样本一个文件。在使用较大的预取深度（例如 prefetch_factor=4 或 8）时，优先使用锁页的主机缓冲区。再配合 persistent_workers=True，由于计算与通信高效重叠，你的系统会让加载线程持续保持繁忙。这样在读取大型语料库中的大量小文件时，就能消除逐文件的开销。

这些方法，再结合较大的操作系统预读和 NVMe SSD，将提升 I/O 吞吐量。此外，较新的文件系统和存储库（如 NVIDIA Magnum IO）也有助于高效地把数据流水线式地送往 GPU。

制定好这一方案后，你应当系统地逐项应用每个优化并测量其效果。记得逐一实现并测试这些优化，以验证每一项确实提升了性能。这有助于避免多处改动以意想不到的方式相互作用的情况。通过隔离每一处改动，你就能知道哪些调整有正面效果，哪些没有。

在装有 NVIDIA 性能监控单元（Performance Monitoring Unit，PMU）驱动的系统上，你可以用 perf 在采样 CPU 计数器的同时，采样 NVIDIA 芯片互连与 fabric 计数器，例如在 /sys/bus/event_source/devices 下以 nvidia_nvlink_c2c* 形式暴露的 NVLink-C2C 设备。可以用 perf list 并检查 sysfs 下的 nvidia_pmu 条目来确认其可用性。

> 用于 NVIDIA PMU 的 Linux perf 仅限于设备级的链路与 fabric 事件，例如 Grace-Blackwell 上的 NVLink-C2C。SM 流水线、warp 停顿和内存吞吐量计数器仍然只能通过 CUPTI 和 Nsight 工具获取。这些 PMU 不暴露 SM 级的核函数指标。要获取 SM 利用率、占用率和内存吞吐量，请使用 Nsight Compute 或基于 CUPTI 的剖析器。务必设置 NVreg_RestrictProfilingToAdminUsers=0，以允许非 root 用户剖析 SM 级硬件计数器。

一旦 PMU 设备就绪，你就可以同时采集 CPU 和 NVIDIA 事件。使用 perf list 报告的符号化事件名：

```
perf list | grep -i nvidia
perf stat -a \
  -e nvidia_nvlink_c2c0_pmu_0/cycles/ \
  -e cycles,cache-misses \
  python train.py
```

这里，NVLink-C2C PMU 上的 cycles 事件让你能够把 GPU 互连活动与主机 CPU 行为关联起来。以下是前述 perf stat 命令的示例输出，它显示了运行期间 NVLink C2C PMU 记录到的活动，同时 CPU 产生了周期数和缓存未命中：

```
Performance counter stats for 'python train_deepseek_v3.py':
       3,567,890,123  nvidia_nvlink_c2c0_pmu_0/cycles
          45,678,901  cycles
           7,890,123  cache-misses
       2.345678901 seconds time elapsed
```

基于 PyTorch 剖析器和 Nsight 工具，我们最初的剖析揭示出：GPU 计算（例如专家矩阵乘法）和 GPU 通信（例如 dispatch 与 combine 操作）是主要瓶颈。不过，CPU、数据加载和 GPU 集合通信操作同样会影响性能——perf 通过显示在慢速区段期间哪些 CPU 线程和互连 PMU 处于活动状态，证明了这一点。

> 若要深入挖掘更多的链路与 fabric 请求计数器，可从 perf list 中挑选系统上出现的其他 NVIDIA PMU 事件，并像前面演示的那样把它们添加到 perf stat 命令中。

简而言之，把来自 perf 的高层 CPU 吞吐量指标与调用图热点，同来自 Nsight Systems 与 Nsight Compute 的设备指标与时间线结合起来，你就能在主机和设备两侧构建出一个整体的性能全貌。先着手解决最大的 CPU 侧瓶颈和数据停顿。接着，优化 GPU 通信并调优 GPU 核函数。

## PyTorch 编译器

在 PyTorch 中，最快见效的优化之一是使用 PyTorch 编译器配合 torch.compile()。该编译器栈包含 TorchDynamo、AOT Autograd 与 TorchInductor，它们负责捕获计算图、融合算子，并为目标后端（如 NVIDIA GPU）生成高性能代码。

PyTorch 编译器通过把许多小操作融合成更大的核函数，能够消除大量 Python 解释器开销与 GPU 核函数启动延迟。在完成基线性能剖析后，我们在模型上启用了 torch.compile，看看能否轻松获得提速。下面就来介绍这个过程及其结果。

### 使用 PyTorch 编译器

使用默认设置的 PyTorch 编译器非常简单，除了像这样包装模型外无需改动任何代码：model = torch.compile(model)。在底层，TorchDynamo 会追踪 Python 代码，AOT Autograd 会捕获反向传播，而 TorchInductor（它借助 OpenAI 的 Triton 生成 GPU 核函数代码，下一章会讨论）则自动产出高效的融合核函数。

编译器观察模型的前向传播，识别出许多可融合连续操作的机会，例如逐元素激活、层归一化等。它会为这些操作生成融合核函数——也会为部分反向传播生成。其结果是每次迭代的核函数启动次数显著减少、CPU 开销降低。

编译这一步确实会引入一些开销，量级在数秒——对超大模型甚至可达数分钟——但这一成本会在长时间训练任务或反复推理运行中被摊薄。所幸 TorchInductor 会缓存已编译的核函数，因此后续运行无需再次支付编译成本。PyTorch 社区也在持续改进编译/启动性能，允许你跨运行保存并复用已编译产物。使用 torch.compiler.save_cache_artifacts() 与 torch.compiler.load_cache_artifacts() 可将 TorchInductor 的输出在多次运行或多个节点间持久化。这能减少长时间训练或服务时的启动耗时。

一个例子是 PyTorch Mega-Cache 特性。它是一种端到端的编译缓存，可将已编译的核函数保存到磁盘，并在未来运行中重新加载。借助 PyTorch Mega-Cache，你可以只编译一次（例如离线编译），并在多次训练会话间复用优化后的核函数，从而缩短启动时间。你仍能享受 TorchInductor 的核函数优化（如 warp 专门化，warp specialization），同时避免每次都重新编译计算图。

> 你甚至可以在其他计算节点上使用这份编译缓存。若要这么做，请确保各节点间的 CUDA、PyTorch 与 Triton 版本相互兼容。

值得一提的是，PyTorch 编译器在内部运用了精巧的优化技术。例如，我们在第 10 章提到过 warp 专门化。TorchInductor 的自动调优器会跨不同的 tile 尺寸、内存访问模式等生成多种核函数变体，并在幕后运用诸如访存 warp 与计算 warp 专门化（memory-warp versus compute-warp specialization）之类的技术，然后自动为你的硬件挑选最快的变体。

TorchInductor 支持在 GEMM 核函数前后进行前导（prologue）与收尾（epilogue）融合。例如，bias-add 在 matmul 之前；而在 matmul 之后，收尾部分由激活、dropout、残差等逐元素操作构成。

通过把这些核函数前导与收尾操作合并到单个优化核函数中，TorchInductor 减少了内存流量、降低了核函数启动开销，并提升了占用率。你可以用剖析器验证这一点——它会显示更高的 SM 利用率。

这些优化的复杂性对开发者完全透明，因为 PyTorch 提供的是简洁、以张量为中心的接口，并不暴露 CUDA 层面的 warp 细节。因此，虽然你在 PyTorch API 中看不到 “memory warp” 或 “compute warp” 这样的标志，但要知道这些技术正在底层发挥作用。一旦代码被编译，你就会在剖析器指标中注意到 warp 专门化带来的收益，包括更高的占用率、更少的访存延迟停顿，以及更高的 SM 利用率。

为了说明编译模式（compilation mode）的好处，我们在一个 MoE 模型上对比 PyTorch 的即时执行模式（eager mode）与编译执行。我们先在常规即时模式下对模型的单次训练迭代计时，然后再用 "max-autotune" 编译模式计时。代码如下，随后是示例输出：

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# ---- Setup Model ----
device = 'cuda'
model_name = "..."
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
# ---- Create a dummy batch of token IDs ----
batch_size = 4
input_texts = ["MoE's are awesome!"] * batch_size
enc = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
input_ids = enc.input_ids.to(device)
attention_mask = enc.attention_mask.to(device)
labels = input_ids.clone()  # for causal LM, labels = input_ids
# ---- Make runs deterministic ----
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# --- Eager timing ---
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
for _ in range(iters_warmup):
    out = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = out.loss if hasattr(out, "loss") else out
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()
start.record()
for _ in range(iters_meas):
    out = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = out.loss if hasattr(out, "loss") else out
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
end.record()
torch.cuda.synchronize()
print(f"Eager mode step time: {start.elapsed_time(end)/iters_meas:.3f} ms")
# --- Compile the model (choose one mode)
# enables graph trees
compiled_model = torch.compile(model, mode="reduce-overhead")
# Alternatives:
# more tuning, longer compile
# compiled_model = torch.compile(model, mode="max-autotune")
# balanced
# compiled_model = torch.compile(model, mode="default")
# Warm-up compiled
for _ in range(iters_warmup):
    out = compiled_model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = out.loss if hasattr(out, "loss") else out
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()
start.record()
for _ in range(iters_meas):
    out = compiled_model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = out.loss if hasattr(out, "loss") else out
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
end.record()
torch.cuda.synchronize()
print(f"Compiled mode step time: {start.elapsed_time(end)/iters_meas:.3f} ms")
```

在本例中，PyTorch 即时模式每次迭代约需 248 ms。经过预热并让编译器完成优化后，编译模式约需 173 ms。我们使用 "max-autotune" 的编译版本比即时执行快约 30%。实际提速会随模型结构、批大小与动态形状（dynamic shapes）而变化。由单个大型 GEMM 主导的稠密模型可能只能看到 <10% 的提速。

这些节省主要来自把许多小 GPU 核函数合并起来。在 MoE 架构中，用于 token dispatch/combine 以及逐 token 激活模式的小 GPU 核函数很常见。通过把这些小操作融合成更少、更大的核函数，我们把中间数据保留在更快的片上内存中（如寄存器和共享内存），而不是反复在全局设备内存（HBM）之间搬运数据。

> 对于 MoE 架构所用的高度动态的 token 路由，优先选择 default 或 max-autotune-no-cudagraphs。待输入形状稳定后——或当你在有限的离散形状下使用 CUDA Graph Trees 时——再切换到 max-autotune。

在本例中，许多小操作（如 dispatch/combine、激活函数等）被 TorchInductor 融合掉了。如果你查看编译运行的追踪，会发现时间线上的 GPU 核函数少了很多。取而代之的是数量更少、但运行稍长的核函数，它们对应于一次性执行多个步骤的合并操作。

在多次迭代中，随着一次性编译开销被摊薄，收益会更加明显。对编译后模型的执行进行剖析可见 GPU 核函数启动次数大幅减少，因为许多在即时模式下相互独立的操作如今被融合到了一起。

值得注意的是，如果一个稠密模型因某个大型线性层而被单个巨型 GEMM 主导，那么它从 torch.compile 得到的收益可能很有限（例如 < 10%）。这是因为该模型很可能已经在高效使用 Tensor Core——而且几乎没有核函数融合和消除 Python 开销的空间。

然而，像 MoE 这样包含数百个中等规模 matmul 操作的稀疏架构则会大获裨益，因为编译能减少 Python 开销、降低核函数启动延迟，并把多个步骤融合进优化核函数。因此，相比稠密模型，PyTorch 编译器为 MoE 带来的性能提升要显著得多。

除了自动算子融合，你还可以把用户自定义的核函数直接集成进 torch.compile 工作流。这种做法兼得两者之长：它对通用模式采用编译器托管的图级优化，同时在需要时给你完全的控制权。

例如，你可以为某个性能关键操作编写专门的 Triton 或 CUDA 核函数，并将其注册为自定义算子。当模型被追踪并编译时，TorchInductor 会把它当作更大执行图中的单个融合操作来处理。其结果是：手工调优的自定义代码被嵌入到一个完全优化、由编译器托管的执行图中。

TorchInductor 的灵活性让你的自定义核函数也能受益于编译器对周边操作的优化（例如与相邻层融合等）。在实践中，这意味着你可以使用自己的高性能核函数，同时不失去 PyTorch 编译器优化模型其余部分的能力。

简而言之，在训练循环中使用 PyTorch 编译（包括其 "max-autotune" 模式），你就能以相对较小的投入在现代 GPU 上获得可观的提速。这可以借助 torch.profiler、Linux perf、Nsight Systems、Nsight Compute 以及其他有用的剖析器进行整体验证。

### 编译 vs 编写自定义核函数

借助像 TorchInductor 这样的编译器后端，许多操作会被自动融合成高效的核函数。正如我们所见，仅仅使用 torch.compile 就能以极小的投入获得相当可观的提升。然而，有时自动生成的代码不如精心手写的核函数那样优化——或者某个操作根本没有被编译器捕获。这就引出一个问题：什么时候该依赖编译器的融合，什么时候该自己编写自定义 CUDA 核函数？

在大多数情况下，优先使用带图捕获（graph capture）的高层 torch.compile——底层由 TorchInductor 支撑。这比编写自定义 CUDA 核函数省力得多——而且往往无需专门编码就能带来足够好的性能提升。

TorchInductor 在内部已经运用了许多高级优化，例如逐元素操作融合、层操作合并、布局优化等。手工编写融合核函数既耗时又脆弱，而编译器在大多数情况下能自动完成这些工作。

如果你的模型使用了编译器处理不佳的新颖操作或模式，你可能需要编写自定义核函数并把自定义算子集成进 PyTorch。下一章会更详细地介绍如何做到这一点。

简而言之，把 torch.compile 作为性能调优的首选，因为它简单、够用且相对 “免费”。创建自定义核函数是下一层级的优化，当内置的自动化不够用时才动用。只有在那之后，你才应考虑为剩余热点编写自定义核函数，以融合某些不受支持的优化器操作或专门的注意力模式。不过，即便是专门的注意力模式，PyTorch 也提供了 FlexAttention API（预填充，prefill）与 FlexDecoding API（解码，decode），它们是在 PyTorch 中为训练和推理实现自定义注意力核函数的首选方式，我们将在后续小节看到。

### 编译模式及其在速度、内存与编译时间上的权衡

PyTorch 为 torch.compile 提供了若干模式，用于针对不同场景调整编译器的激进程度与能力。你可以用 torch.compile(model, mode="...") 显式选择模式。可选项有 "default"、"reduce-overhead"、"max-autotune" 与 "max-autotune-no-cudagraphs"。每种模式都是关于 CUDA Graphs、自动调优（autotuning）与优化级别的一组选项组合。这些模式汇总于表 13-5。

表 13-5. 编译模式及其关键特性汇总

| 模式 | 说明 | 编译时间 | 额外内存 | 主要特性 |
| --- | --- | --- | --- | --- |
| default | 均衡优化（速度不错，且无需长编译时间或额外内存）；包含少量自动调优；对稳定段可能使用 CUDA Graphs | 低–中 | 无 | 通用融合、基础自动调优 |
| reduce-overhead | 降低每次迭代的开销（适合小批次）；非常适合推理或小批次；若检测到动态形状会自动跳过 CUDA Graphs 以保证正确性 | 中 | 是（工作区缓存） | 尽可能使用 CUDA Graphs 以消除启动开销 |
| max-autotune | 最大化运行时性能（最适合长时间运行）；编译时间较长；最适合面向大量 SM 与 GPU 内存做激进调优 | 高（编译慢） | 可能（若使用图） | 激进的 Triton 自动调优；在 GPU 上启用 CUDA Graphs |
| max-autotune-no-cudagraphs | 完成 max-autotune 的一切，但不做 CUDA Graph 捕获；最适合动态形状，或调试被 CUDA Graphs 掩盖的问题 | 高 | 无 | 与 max-autotune 相同，但禁用图以保持灵活性 |

以下是表 13-5 中各模式的详细说明：

default 如果没有显式指定模式，这就是默认模式。该模式在编译代码足够快与不过度消耗编译时间或内存之间取得平衡。它会执行标准优化，并使用默认的 TorchInductor 后端。当编译时间需要保持适中——或内存吃紧时——该模式通常最合适。它包含少量自动调优，并可能对稳定段使用 CUDA Graphs，但会尽力在速度与编译时间成本之间取得平衡。

reduce-overhead 该模式专注于最小化 Python 与运行时开销。它对小模型——或每次迭代只执行少量操作的模型——尤其有用。在这些情况下，哪怕一点点开销都会损害性能。该模式积极利用 CUDA Graphs 来消除每次迭代的启动开销。它还可能分配一些额外内存供持久使用，例如可复用的工作区内存，从而避免频繁的 CUDA malloc 与 free 调用。例如，它可能缓存一个大的工作张量，而不是每次迭代都重新分配。若检测到动态形状，该模式会自动跳过 CUDA Graphs 并回退到即时模式，以此保证正确性。

在低延迟场景下，该模式能加速推理和训练——代价是一些额外内存。注意，CUDA Graphs 要求图的内存地址保持不变，因此只有在输入尺寸不变、且不存在诸如动态形状操作之类的某些操作时，才能使用该模式。否则，图会中断或被重新编译。若编译器在某一段中无法应用 CUDA Graph，它会自动回退。

max-autotune 该模式生成尽可能最快的代码，而不顾及编译时间。它会对核函数进行广泛的自动调优，例如为矩阵乘法尝试多种 tiling 配置，并利用 TorchInductor 中所有已知的性能优化。在现代 GPU 上，max-autotune 还会默认启用 CUDA Graphs 以实现稳定执行。

该模式的编译过程可能明显更长——对大模型而言可达数分钟量级。它适用于编译一次、多次运行模型的场景，例如运行长时间训练任务，或部署一个将长期处理大量请求的模型。作为对前期长编译时间的回报，你通常能获得最佳的运行时性能。例如，经过自动调优后，你的 matmul 可能会以针对你特定 GPU 与张量形状的最优 block 尺寸运行。这让该模式相比默认启发式更有优势。

max-autotune-no-cudagraphs 顾名思义，该模式与 max-autotune 相同，但禁用了 CUDA Graph 捕获。当 CUDA Graphs 会干扰某些与之不兼容的期望运行时行为时，这一点很重要。例如，由于 CUDA Graphs 要求静态形状和内存地址，你就无法使用变化的输入形状，也不能依赖每次迭代分配新内存。

此外，在测量性能时，使用 CUDA Graphs 会掩盖核函数启动的开销，这在某些基准测试中可能并非所愿。因此该模式允许在不使用 CUDA Graphs 的情况下进行最大程度的核函数优化。这有助于保持灵活性，并让你能够调试 CUDA Graphs 可能引入（或掩盖）的任何问题，例如依赖形状的控制流，或图捕获期间偶发的分配器重新寻址。当每次迭代输入尺寸变化时——或为调试可能被 CUDA Graphs 掩盖的问题时——请使用该模式。

对大多数使用场景，default 模式是个不错的起点。它旨在以最小的麻烦带来可观的提速。如果你发现模型仍然不够快，且能容忍更长的编译时间，可以尝试 reduce-overhead 和 max-autotune，以期获得更好的融合核函数——尤其当你的模型由可自动调优的大型 matmul 操作主导时。max-autotune 有时会在某些模型上让延迟变差。务必针对你特定的工作负载和硬件对不同模式进行剖析。

另一方面，如果你要优化的是一个非常小的模型，或某段以 Python 开销为瓶颈的代码，例如包含大量小张量操作的紧凑训练循环，那么使用 reduce-overhead 能带来最佳收益——它通过 CUDA Graph 捕获几乎消除了所有核函数启动的运行时开销。只是要留意 reduce-overhead 的约束条件。当每次迭代的工作负载一致、且满足图捕获要求（包括没有动态形状变化、没有新的内存分配等）时，它效果最好。

max-autotune-no-cudagraphs 模式更像是一个专门化的选项。如果你想要最大程度的核函数优化，但因输入尺寸变化而无法使用图，或想在不做图摊薄的情况下测量核函数的原始性能，它就很有用。

无论哪种情况，在更改 PyTorch 编译器模式后进行剖析和测量都是明智之举。之所以存在不同模式，是因为在性能工程中没有一种方案能通吃一切。此外，选择模式时应监控内存使用。使用 CUDA Graphs 的模式会分配大块静态缓冲区，从而增加内存占用。

对于内存极度受限的情形，你可能更倾向于这种无图模式，以避免 CUDA Graphs 带来的额外内存开销。接下来，我们讨论如何检视编译器正在做什么，包括它创建了一个图还是多个图——或者它是否使用了 CUDA Graphs 等。

### 区域化编译

对于包含许多相同块的模型（例如 Transformer 和 MoE），你可以使用区域化编译（regional compilation）来缩短冷启动执行时间。此外，它有助于减少重新编译（recompilation）的反复折腾——而这一切都不会牺牲核函数融合的威力。

具体来说，区域化编译只把重复的块（例如一个 Transformer 块）编译一次，并在所有出现处复用该代码，从而减少编译时间。

PyTorch 通过 torch.compiler.nested_compile_region() 支持区域化编译。该 API 把一个块标记为嵌套编译区域。该区域在第一次时被编译，随后在后续运行中被复用。

此外，区域化编译能保持正确性。如果编译器检测到新的输入条件（形状、dtype、设备、步幅、全局变量），它会透明地重新编译该区域。

区域化编译对启动延迟很重要的推理引擎和短任务——或图失效频繁发生的场景——大有裨益。区域化编译代码的性能与完整编译代码的吞吐量相近。

### 剖析与调试编译器性能问题

使用 torch.compile 时，了解如何调试编译器无法优化模型某部分的情形很有用——例如，某些操作没有被融合，而你怀疑是某个 “图中断”导致回退到即时执行。PyTorch 提供了检视这些情形的工具。

> 现代 PyTorch 版本使用形状保护条件（shape guard）实现了对动态形状的部分支持。这些保护条件可以消除一些不必要的图中断。然而，真正动态的工作负载可能仍需回退到即时执行（或使用 max-autotune-no-cudagraphs）以确保正确性。

torch._dynamo.explain(model) 会打印一份报告，列出所有图中断（例如未被 TorchDynamo 捕获的模型部分）、图中断发生的原因，以及模型中哪些部分未被 TorchDynamo 捕获。它还会列出未被 TorchDynamo 捕获、需要在较慢的 Python 即时模式下执行的操作或依赖数据的控制流。

图中断的一个常见原因是模型中存在不受支持的操作。Dynamo 的 explain() 输出会就如何获取更多细节给出建议，帮助诊断问题。利用这些提示有助于精确定位导致中断的具体操作或控制流。

另一个有用的技巧是在运行脚本前设置环境变量 TORCH_LOGS="+dynamo" 或 TORCH_LOGS="+dynamo,+inductor"。前缀 + 会为 torch.compile 流水线中的 TorchDynamo、TorchInductor 等组件启用详细（DEBUG 级别）日志。这些详细日志包含关于图中断、回退到即时模式以及编译各阶段的细节。如果模型在使用 torch.compile 时意外地慢，这些日志有助于识别执行在何时、何处退出了编译图。

如果模型具有编译时无法确定的真正动态形状或动态控制流，你可能需要引导编译器。例如，你可以把模型拆分成可编译的若干部分，而让真正动态的部分在 Python 中运行。

要对 TorchInductor 生成的核函数进行剖析和基准测试，你可以指定环境变量 TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 和 TORCHINDUCTOR_BENCHMARK_KERNEL=1。设置这些变量后，Inductor 会为生成的核函数模块生成基准测试脚手架代码。该脚手架代码生成的日志有助于精确定位意外的图中断和性能问题。

你还可以用 torch._dynamo.mark_dynamic(tensor, dim) 标记部分代码，让编译器知道要预期动态形状。这可以消除因形状不匹配而产生的不必要图中断。我们将在下一章深入 PyTorch 编译器时更详细地介绍这些技术。

简而言之，当 torch.compile 未能产生预期的提速时，你可以使用 torch._dynamo.explain()——配合编译器日志——来识别是哪些操作或代码区域导致了回退。据此，你需要采取一些变通办法，例如替换某个操作、以不同方式重塑张量、接受较少的动态行为，或干脆对模型的那一特定部分禁用编译。其结果是：你为模型的大部分保留了性能收益，同时仍能处理边界情况。

## PyTorch 优化的注意力机制

Transformer 模型在其注意力（attention）机制上耗费大量时间。你可以运用若干 PyTorch 注意力优化技术，确保它不会成为瓶颈。下面简要总结其中几种技术及其适用场景：

*缩放点积注意力（Scaled Dot-Product Attention，SDPA）* PyTorch 的高层 API torch.nn.functional.scaled_dot_product_attention（即 SDPA）会针对给定硬件自动使用可用的最快注意力核函数（例如 FlashAttention）。当你模型的注意力模式和 dtype 被所选后端（Flash、memory-efficient 或 math）支持时，用它可以毫不费力地获得提速。如果不受支持，它会回退到标准注意力实现。

*FlexAttention* 一种基于编译器、用于注意力中自定义稀疏模式（sparsity pattern）的方法。通过为特定稀疏注意力模式（例如块稀疏或滑动窗口注意力）生成优化核函数，FlexAttention 可以显著更快，如图 13-3 所示。对于 scaled_dot_product_attention 不支持的特殊情形，请使用 FlexAttention。

![图 13-3. FlexAttention 为自定义注意力变体（attention variant）提供支持。](AI%20Systems%20Performance%20Engineering-ch13_images/figure-13-3.png)

*FlexDecoding* 它是 FlexAttention 的对应物，用于优化解码或文本生成阶段。FlexDecoding 与 torch.compile 及动态缓存布局集成。它对序列生成的解码器一侧运用编译时优化，包括跨时间步高效地进行 KV 缓存。FlexDecoding 通过减少解码期间的冗余计算，能够加速自回归生成。FlexDecoding 面向 LLM 推理工作负载，包括那些具有长生成序列的场景。它不改变训练时的注意力语义。

*上下文并行（context parallel）* 上下文并行沿序列长度维度，把注意力在参与的设备或 rank 之间分片，以扩展上下文长度。使用 context_parallel() API 可以限定用上下文并行感知的核函数替换 scaled_dot_product_attention 的范围。该机制按序列在各 rank 之间切分 query-key-value（QKV），并在注意力计算期间进行同步，而不是在单个 GPU 内的线程之间并行化注意力。

## PyTorch 架构优化（PyTorch Architecture Optimization，torchao）、量化、稀疏化与剪枝

PyTorch Architecture Optimization（torchao）把量化（quantization）、稀疏化（sparsity）、剪枝（pruning）以及相关的数值调试工具汇聚到单一命名空间中。它的量化子包（torchao.quantization）提供了常见的 FX-graph-mode 工作流，包括训练后量化（post-training quantization，PTQ）、量化感知训练（quantization-aware training，QAT），以及用于将模型转换并优化为 INT8、FP8 及新兴格式的 QConfigMapping API。

除量化外，torchao 还支持剪枝（torchao.pruning）以及诸如 2:4 稀疏和块稀疏（torchao.sparsity）之类的稀疏化技术。它们能在精度损失极小的情况下带来显著提速。

torch.compile() 与 torchao 量化框架集成。在底层，TorchDynamo 把每个子模块的计算捕获为优化图，随后 TorchInductor 发出利用 torchao 的硬件感知核函数。这为模型训练和推理都带来一致的端到端性能提升。与此同时，它保留了对数值格式与内存布局的精确控制。这使它成为一个适合量化等生产级性能优化的出色库。

## 用 CUDA 流实现并发

正如前面章节所述，CUDA 流可在 GPU 上实现操作的并发与重叠。默认情况下，PyTorch 会把所有操作顺序调度到设备的默认流（stream 0）上。然而，许多任务是相互独立的；在资源允许时，GPU 可以用多个流并行执行它们。例如，GPU 可以通过使用独立的非默认流，把数据传输与计算重叠——或并发运行不同的神经网络分支。

> 请记住，现代 GPU 拥有多个 DMA 拷贝引擎。为 H2D 拷贝使用独立的流，可以在不阻塞计算的情况下实现真正并行的数据传输。这种硬件支持让流并发更加有效。

在 PyTorch 中，你用 torch.cuda.Stream() 创建一个流。然后可以通过 Python 上下文管理器 with torch.cuda.stream(stream) 在该流上启动工作，或显式地把操作指派到该流。PyTorch 会以 FIFO 顺序把操作（如内存传输、CUDA 核函数等）下发到指定的流——就像它在默认流上所做的那样。

### 通信与计算重叠

CUDA 流的一个常见用法是把主机到设备（host-to-device，H2D）的数据加载与 GPU 计算重叠。这有助于掩盖使用 GPU 这类外部设备时——相对于在主机上运行的 CPU——所产生的数据传输延迟。

例如，当默认流正忙于在当前批次上训练时，另一个流可以把下一批输入数据从 CPU 拷贝到 GPU 内存。等到默认流准备好处理下一批时，数据传输早已完成，GPU 便可直接处理这下一批。这有效地隐藏了 I/O 延迟。下面是一个在 PyTorch 中使用两个流——compute_stream（默认）与 transfer_stream（非默认）——来重叠数据传输与计算的示例：

```
# Set up streams
device = 'cuda'
# for H2D data transfers
transfer_stream = torch.cuda.Stream(device=device)
# for compute
compute_stream  = torch.cuda.default_stream(device=device)
# Create an iterator so we can preload "next" batches
dataloader_iter = iter(dataloader)
# Preload the very first batch onto GPU
first_batch = next(dataloader_iter, None)
if first_batch:
    with torch.cuda.stream(transfer_stream):
        next_inputs, next_labels = (
            first_batch[0].to(device, non_blocking=True),
            first_batch[1].to(device, non_blocking=True),
        )
for _ in range(len(dataloader)):
    # 1) Wait for transfer of `next` batch to finish, then swap into compute var
    # Multiple copy engines allow H2D/peer copy concurrency.
    # Verify parallelism in Nsight Systems (Copy engines lanes)
    # And tracing/profiling tools (HTA, etc.)
    compute_stream.wait_stream(transfer_stream)
    inputs, labels = next_inputs, next_labels
    # 2) Kick off transfer of the *following* batch on the transfer_stream
    batch = next(dataloader_iter, None)
    if batch:
        with torch.cuda.stream(transfer_stream):
            next_inputs, next_labels = (
                batch[0].to(device, non_blocking=True),
                batch[1].to(device, non_blocking=True),
            )
    # 3) Run forward/backward on compute_stream
    with torch.cuda.stream(compute_stream):
        outputs = model(inputs)
        loss    = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

本示例使用了两条 CUDA 流：一条专用的传输流用于异步的主机到设备拷贝，另一条默认的计算流用于模型计算。这样可以隐藏 H2D 延迟，并契合 PyTorch 推荐的通信与计算重叠模式。

具体来说，当模型在默认流上处理某个批次时，下一个批次的数据传输已经在 transfer_stream 上进行了。在使用预加载的批次之前，通过 compute_stream.wait_stream(transfer_stream) 进行同步，可以在不设置全设备范围栅栏（barrier）的情况下保证正确的执行顺序。而 .to(device, non_blocking=True) 调用则确保拷贝采用基于异步 DMA 的方式，不会阻塞发起调用的 CPU 线程。

使用 next(dataloader_iter, None) 可以显式控制传输何时入队、以及核函数操作何时运行。这样就能保证：当一个批次的数据正在传输流上传输时，另一个批次正在计算流上执行，如图 13-4 所示。

![图 13-4. 使用专用的计算流与传输流实现计算与数据传输的重叠](AI%20Systems%20Performance%20Engineering-ch13_images/figure-13-4.png)

此外，通过提前从 dataloader_iter 取数据并存入 next_inputs、next_labels，这段代码将批次加载（在 transfer_stream 中于 CPU 上运行）与批次处理（在 compute_stream 中于 GPU 上运行）分离开来。

这种拆分意味着每条流上始终有一个批次在处理中。它将数据加载与计算解耦，并最大化重叠。

> 在添加流时，务必用 Nsight Systems 或 PyTorch 剖析器做性能剖析。观察 GPU 利用率——如果做法正确，你会看到接近 100% 的利用率，传输与计算相互重叠。如果利用率下降——或者你看到数据传输与计算是顺序进行的——请仔细检查是否存在任何隐式同步，例如对 CUDA 张量调用 print()，或者代码中额外的 CUDA 同步。

### 用事件进行流同步

在使用多条流时，有时需要在它们之间进行协调，例如确保一条流的工作完成之后，另一条流才使用它的结果。正如第 11 章所讨论的，使用 CUDA 事件（event）是一种在特定节点上进行同步的轻量级方式。

借助 CUDA 事件，你可以在一条流上记录一个事件，并让另一条流等待该事件。这样就避免了 torch.cuda.synchronize() 那种重量级的全局同步，转而只对处理某个具体事件所需的流进行同步。

事实上，即便在多 GPU 场景下，事件也可以用来跨设备同步工作。在这种情况下，一个 GPU 在其设备流上记录事件，而另一个 GPU 在其设备流上等待该事件。NCCL 在底层正是这样处理依赖关系的。

通过巧妙地使用事件，你可以让多个 GPU 尽可能地并行工作。下面是一个在 PyTorch 中使用 CUDA 事件同步两条流 stream1 与 stream2 的示例：

```
# Disable timing on the event since we’re using it purely for synchronization.
event = torch.cuda.Event(enable_timing=False)
# In first stream:
with torch.cuda.stream(stream1):
    kernel_launch(...)
    event.record()           # record event at end of work in stream1
# In another stream or on host:
stream2.wait_event(event)    # make stream2 wait until event is signaled
with torch.cuda.stream(stream2):
    other_kernel_launch(...)
```

在这段代码中，我们在 stream1 上某些工作结束时记录一个事件。随后，在 stream2 上启动工作之前，我们调用 stream2.wait_event(event)。这会插入一个依赖关系，使得 stream2 在 stream1 执行到那一点并触发事件之前，不会执行它的下一个核函数。事件适合用来在流之间调度轻量级的依赖关系，因为它们避免了那种会让所有流的执行都停顿的重量级全局同步。

让我们回顾上一节中 PyTorch 数据加载器 / 重叠的示例，并把它改写为用 CUDA 事件来同步。我们仍然使用之前的那对流（transfer_stream 与 compute_stream），但会额外加入一个 transfer_done CUDA 事件，以便在细粒度、针对具体事件的层面上进行同步：

```
import torch
device = 'cuda'
# for H2D copies
transfer_stream = torch.cuda.Stream(device=device)
# for compute
compute_stream  = torch.cuda.default_stream(device=device)
# sync-only event (low overhead)
transfer_done = torch.cuda.Event(enable_timing=False)
# Iterator so we can preload ahead
dataloader_iter = iter(dataloader)
# ---- Preload first batch ----
first_batch = next(dataloader_iter, None)
if first_batch:
    with torch.cuda.stream(transfer_stream):
        next_inputs, next_labels = (
            first_batch[0].to(device, non_blocking=True),
            first_batch[1].to(device, non_blocking=True),
        )
        # mark when H2D is done
        transfer_done.record(stream=transfer_stream)
for _ in range(len(dataloader)):
    # ---- Sync: wait for the transfer to complete ----
    compute_stream.wait_event(transfer_done)
    inputs, labels = next_inputs, next_labels
    # ---- Kick off next transfer ----
    batch = next(dataloader_iter, None)
    if batch:
        with torch.cuda.stream(transfer_stream):
            next_inputs, next_labels = (
                batch[0].to(device, non_blocking=True),
                batch[1].to(device, non_blocking=True),
            )
            transfer_done.record(stream=transfer_stream)
    # ---- Compute on the compute stream ----
    with torch.cuda.stream(compute_stream):
        outputs = model(inputs)
        loss    = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

这与上一节采用的重叠模式相同，但它使用 CUDA 事件来完成传输 → 计算之间的同步，这与另一个示例中使用的 wait_stream() 机制形成对比。

这段代码仍然使用 next(dataloader_iter) 提前预加载批次（与上一节的示例相同）。这样，数据传输与计算就始终处于重叠状态。

不过，在这个示例中，transfer_done 事件是在异步拷贝入队之后立即通过 transfer_done.record(stream=transfer_stream) 在 transfer_stream 上记录的。这会为该事件打上时间戳。

随后，compute_stream.wait_event(transfer_done) 会让 compute_stream 停顿，直到拷贝完成、transfer_done 事件被触发为止。之后它便使用预取的批次，并在 compute_stream 上执行其计算操作。

除了数据加载之外，CUDA 流在许多不同的场景中都很有用。下面我们来讨论它们在 MoE（专家混合，mixture of experts）模型中的用法。

### 在 MoE 模型中使用 CUDA 流

在实践中，Transformer 模型的各层之间存在顺序依赖，因此你无法随意地并行运行这些层。然而，在 MoE 架构中，由于不同专家（expert）的计算彼此独立，它们可以在不同的 CUDA 流上并发运行。

每个专家处理输入中一个独立的片段。在汇合点，各专家的输出会被聚合起来。至关重要的一点是，每个专家只能写入分配给它的那一片输出张量。如果两个专家不小心写入了相互重叠的内存区域，就会引入竞态条件（race condition），从而可能破坏结果——或者因为各专家写入顺序的不确定性而触发同步问题。

为避免这类问题，你应确保使用恰当的流级同步（例如流事件），并核实内存在各专家核函数之间被清晰地划分开来。通过在专家执行与输出聚合之间强制施加这种隔离，你就能在不牺牲并行性的前提下保持正确性。

> NVIDIA 的 Compute Sanitizer 能够检测 CUDA 代码中的并发与同步问题，包括竞态条件和死锁。此外，你还可以设置 CUDA_LAUNCH_BLOCKING=1 来强制核函数同步执行。这会通过让核函数执行变得确定，从而暴露出顺序和依赖方面的 bug。它可以揭示输出是否在尚未完全生成之前就被使用了。

在我们的示例中，理论上每个专家都可以运行在自己的流上——如果框架被扩展到多个 GPU，甚至可以运行在各自的 GPU 上。在这种情况下，收集结果时需要进行同步——最好使用流事件。

流水线并行（pipeline parallelism，PP），即让多个微批次（microbatch）流经位于不同设备上的不同模型阶段，以及同时服务多个推理请求，是两种天然能从多条流中获益的场景。例如，在流水线并行的工作流中，模型的每个阶段都有自己的流，用于并发处理不同的微批次。与此同时，它还在与相邻的阶段进行通信。

在多请求推理服务中，每个请求的模型执行都可以在各自的流上启动。在硬件资源充足的情况下，这可以通过重叠推理计算来提升吞吐量——代价是由于资源共享开销，单个请求的延迟会有所增加。

简而言之，CUDA 流通过在多个核函数、阶段或请求之间重叠工作，帮助你从硬件中榨取出额外的性能。它们需要谨慎的同步来避免竞态条件。但只要使用得当，它们就能隐藏延迟，让 GPU 得到更充分的利用。

建议在引入并发时持续地对代码进行性能剖析。同时要记住，在 100% 利用率下的顺序执行，其性能可能实际上优于会引入资源争用的并行执行。不过，流通常能让你利用到那些原本会闲置的 GPU 部分。找到恰当的平衡点很重要。

> 务必确保你是在预期的流上启动操作。例如，不小心用了默认流，就可能重新引入不必要的串行化。这一点很容易搞错，所以值得再强调一遍。

## 用 CUDA Graphs 降低核函数启动开销

在前面的章节中我们已经看到，CUDA Graphs 能够消除每次迭代的启动开销，降低 CPU 启动开销，并消除核函数之间微小的空闲间隙。而哪怕消除最微小的空闲间隙，也能带来更高的有效利用率——以及更一致的迭代耗时。现在让我们展示如何在 PyTorch 中使用它们。

### 捕获 CUDA Graph 并预分配内存

PyTorch 提供了 torch.cuda.CUDAGraph API 来捕获并重放（graph replay）CUDA Graphs。一般的使用模式是：首先通过正常运行几次迭代来对模型进行预热，以初始化所有必要的数据和内存分配。接着，你创建一个 CUDAGraph 对象，以及一条专用的、非默认的 CUDA 流来隔离此次捕获。

> 当使用 "reduce-overhead" 或 "max-autotune" 时，如果模型是稳定的，编译器会自动为你捕获 CUDA Graphs。在这种情况下，你甚至不需要编写这些样板代码，因为只要你在这些模式下使用 PyTorch 编译器，它就会自动完成。而如果你的模型每次迭代形状都在变化，可以考虑使用 "max-autotune-no-cudagraphs" 模式来避免图捕获，因为 CUDA Graphs 目前要求静态形状。

随后，你在 torch.cuda.graph() 上下文中指定的捕获流上，完整执行一遍模型，以记录下操作序列。一旦这些操作被捕获进 CUDA Graph，你就可以按需在新的输入上 replay() 该图（例如用于模型训练或推理）。

在捕获 CUDA Graph 之前，捕获期间使用的所有静态内存都必须预先分配——并且最好按其最大尺寸分配。这些缓冲区包括输入、输出以及中间张量。如果在捕获期间发生任何新的内存分配，图捕获就会失败，并报出诸如 “operation not permitted when stream is capturing” 之类的错误。

为了减少碎片、并为这些固定缓冲区最大化连续的内存空间，你可以在进入捕获代码块之前立即调用 torch.cuda.empty_cache()。这会清除未使用的缓存内存，让分配器有最佳机会不受打扰地布局你预留的缓冲区。

> 频繁使用 torch.cuda.empty_cache() 会扰乱分配器的效率，并带来更长期的性能代价。请把这个调用当作捕获图时的一次性安全手段——而不是常规的维护工具。

请记住，PyTorch 的缓存分配器支持 CUDA 的异步分配器（cudaMallocAsync），以复用固定的内存地址。然而，这并不能绕过 CUDA Graphs 的要求：即不能在图内创建新的内存分配。

你仍然需要预先分配固定大小的缓冲区，以免在图内尝试分配新内存时触发运行时错误。请确保在图捕获之前的预热阶段，所有张量都达到其所需的最大尺寸。我们会在后续的图重放一节中进一步讨论这一点。

你需要为图捕获使用一条专用的、非默认的流，以避免干扰那些不应被纳入 CUDA Graph 的操作。下面是一段代码，演示如何在 PyTorch 中用专用的 capture_stream 捕获一个 CUDA Graph：

```
g = torch.cuda.CUDAGraph()
capture_stream = torch.cuda.Stream()
# Prepare static inputs and outputs
static_input = torch.randn(batch_shape, device='cuda')
static_output = torch.empty(output_shape, device='cuda')
# Warm-up step on capture_stream to allocate buffers without recording
with torch.cuda.stream(capture_stream):
    tmp = model(static_input)
    static_output.copy_(tmp)
# ensure warm-up is complete
capture_stream.synchronize()
# Begin graph capture
with torch.cuda.graph(g, stream=capture_stream):
    tmp = model(static_input)
    static_output.copy_(tmp)
# ensure capture is complete before using the graph
capture_stream.synchronize()
```

在这段代码中，我们首先在 GPU 上以固定形状分配 static_input 和 static_output。我们在 capture_stream 上运行一次预热迭代，以确保 model(static_input) 内部所需的任何内存都已分配——同时完成任何一次性的初始化工作。

> 通过预分配输出缓冲区，并在预热和捕获两个阶段都使用 static_output.copy_(tmp)，这段代码会把结果写入一个固定的内存区域。这使得被捕获的 CUDA Graph 正确、可重放、可复现，且不会出现意料之外的张量分配。

接着，我们对 capture_stream 进行同步，以确保在开始实际的图捕获之前，预热步骤已经完全完成。随后，我们在同一条流上进入 torch.cuda.graph(...) 上下文，并重新运行一遍模型的前向传播。

在捕获阶段，实际上并不会启动任何核函数。相反，这些操作只是被记录进 CUDA Graph 对象 g 中。退出捕获代码块之后，我们需要再次同步，以确保记录已经最终完成。

在捕获 CUDA Graph 时，严格的隔离至关重要。捕获流上的操作绝不能受到任何其他流上活动的影响。即便是另一个线程上一次看似无关的核函数启动，也可能使此次捕获失效，从而导致诸如 “operation not permitted when stream is capturing” 之类的运行时错误。

如果你在捕获进行期间不小心在其他流上执行了 CUDA 操作，就会出现这个错误。在这种情况下，它会使图上下文失效并触发该错误。如果你在捕获流上启动了一个执行动态内存分配的核函数，例如在捕获期间创建张量或调用 torch.empty，也会出现同样的情况。

> 务必在开始图捕获之前以及退出图捕获之后，都调用 capture_stream.synchronize()。这能确保所有操作都被正确记录，并且图已准备好可以安全地重放。前面的代码示例就遵循了这一最佳实践。

图被捕获之后，可以在任意流上重放，包括默认流——而且与捕获时所用的是哪条流无关。如果你触发了任何依赖于图完全执行完毕的 CUDA 操作，就必须在运行这些操作之前进行同步，如下所示。否则，由于图在重放时是异步运行的，那些依赖图结果的 CUDA 操作可能会在图执行完成之前就运行：

```
# replay the graph which writes to static_output
g.replay()
# synchronize on the stream that is replaying the graph
torch.cuda.current_stream().synchronize()
# Now it's safe to read or post-process the output
print(static_output)
...
```

如果没有这种显式同步，你的程序就可能继续往下执行，并错误地假定图已经执行完毕、且已把结果写入了 static_output。如果没有任何同步措施，代码可能会读到过期或只写了一部分的数据，因为图可能还没有写完 static_output。这种情形会导致不确定的行为、损坏的读取、隐蔽的竞态条件以及死锁。

### 重放图

要用新数据重放图，你需要把新的输入拷贝进预分配的输入张量 static_input，然后调用 g.replay()。GPU 会以这些张量当前的内容作为输入，执行整个被捕获的操作序列。图会把结果放入预分配的输出张量（static_output），如下所示：

```
# load new data into pre-allocated input tensor
static_input.copy_(new_batch)
# execute the captured graph
g.replay()
# retrieve the output (clone if you plan to modify it)
result = static_output.clone()
```

这里，我们把 new_batch 的数据加载到 static_input 的内存空间中，然后调用 g.replay()。图会以当前 static_input 中的数据，运行完全一致的被捕获操作，并把输出写入 static_output。之后我们便可按需使用或克隆 static_output。

建议用几个测试输入来验证图的输出与正常执行的结果一致，以确认此次捕获是成功的。另外要记住，在不同步的情况下，你不能直接对 static_output 调用 print 或 .item()。如有需要，可在重放之后、且在任何异步区域之外，执行 result = static_output.cpu().numpy() 以便调试。

由于图每次都复用相同的输入、输出以及内部内存分配，它会在每次迭代时覆写同一个输出张量。因此，如果你需要把输出保留到单次迭代之后，就需要像前面的代码那样对缓冲区调用 clone()。这也是为什么我们需要确保预热 / 捕获步骤中的内存分配覆盖了所需的最大尺寸。请记住，图无法临时处理额外的内存分配。

值得注意的是，在使用 CUDA Graph 之后，某些 PyTorch 操作在你重新捕获图之前不会生效。例如，如果你在图之外临时改动模型权重，这些更新不会应用，因为图持有自己的一份操作副本。因此，图最适合用于稳态循环——即每次迭代都重复相同操作的场景。

此外，当使用 cudaMallocAsync 为 CUDA Graph 预分配内存时，它在重放期间会复用图捕获期间预分配的那些内存地址——或者，比如当图是从磁盘加载时，复用预热期间预分配的地址。这样，后续的图重放就不需要额外的内存了。

默认情况下，每个 CUDA Graph 实例都使用自己私有的内存池。这样就能保证每个图都预分配自己的内存缓冲区，不会与该图的其他实例竞争。换句话说，两个相同的图即便并发重放，也不会争抢内存，因为它们各自使用自己的内存池和缓冲区空间。

你可以通过传入 torch.cuda.graph(pool=...) 来选择在多个图实例之间共享一个内存池。不过，这只有在非常特殊的情况下才有用——即当你出于性能考虑，想要刻意编排相关的图，让它们复用预分配的内存缓冲区时。例如，考虑同时运行多个推理图变体——每个变体使用不同的批次大小（例如 1、2、4、8 等）。在这种情况下，你可以让这些针对特定形状特化的变体复用同一块大的内存分配，从而降低整体内存占用；这块内存供 PagedAttention 使用，而 PagedAttention 用到了一个名为 block_tables 的可变大小张量。

这种做法在 FireworksAI 的一篇博客文章中有所描述。在那里，他们为每个批次大小编译一个不同的 CUDA Graph 变体。而且，他们不是为每个图创建单独的内存池，而是让所有图变体共享同一个内存池。通过按批次大小递减的顺序编译各图，共享池中的内存会从最大的变体（例如本例中的 8）开始被复用。较小批次大小的图变体，则由上一次迭代中分配的较大缓冲区来提供服务。这样，在不占用过多 GPU 内存的前提下，就支持了多个图变体。

> 这是一种冷门而巧妙的实现选择，需要对内存段进行细致的协调。不过，它确实能在同时运行多个针对特定形状特化的图时，降低 GPU 内存的总占用。对于那些必须尽量减小整体内存占用的部署场景来说，这非常有用。

### CUDA Graphs 最佳实践

CUDA Graphs 是在训练和推理中都能达到峰值稳态吞吐量的一种强有力手段。对于那些 CPU 开销和核函数启动波动正在损害性能的大规模部署，它们尤其有用。在现代 GPU 上，成千上万个操作会被极快地执行，此时图对于让 GPU 设备保持繁忙、并把每个核函数的启动开销降到最低，就变得不可或缺。

值得注意的是，PyTorch 的 torch.compile 在许多情况下都会在底层使用 CUDA Graphs——除非被显式禁用。CUDA Graphs 用于为已编译的模型最小化核函数启动开销。以下是使用 CUDA Graphs 时需要记住的几点要事：

*在图捕获期间避免分配内存* 请记住，你不能在图捕获内部动态地分配 GPU 内存。任何需要用到的张量都应事先在预热步骤中分配，如前所示。如果你的图需要临时缓冲区，请使用 PyTorch 基于 cudaMallocAsync CUDA 后端的、图感知的缓存分配器。这样，每次重放都会复用相同的缓冲区地址。请确保这些临时张量在预热阶段（图捕获之前）就按其潜在的最大尺寸创建。这会预先分配好所有需要的内存。

*保持图结构固定* 被捕获的图无法改变操作序列、张量形状或内存大小。如果你的工作负载偶尔会有形状变化，一种策略是捕获多个图（例如，为你预期的每种输入尺寸各捕获一个），然后在运行时选择合适的图。或者，你也可以为那些迭代禁用图（PyTorch 编译器为此类情况提供了 max-autotune-no-cudagraphs 模式）。

> CUDA 提供了一套底层的图管理 API（Graph Management API），其中包括前面章节介绍过的 cudaGraphExecUpdate()，它允许对已捕获的图做少量修改。然而，截至本文撰写时，PyTorch 并未暴露这个能力。就目前的 PyTorch 而言，最好把图当作不可变的对象来对待。

*尽可能多地捕获* 尽量把训练循环中尽可能多的部分纳入图中——理想情况下是一整次迭代（前向传播、反向传播、优化器步进以及任何 all-reduce 通信）。你捕获得越多，消除的 CPU 开销和启动延迟就越多。

此外，如果内存允许，可以考虑在一个图中捕获多次迭代（包括循环展开等）。尽管这会让图变得更大，但它通常能通过在多次迭代之间启用更多优化来提升吞吐量。这样做的代价是牺牲灵活性，但值得去探索和剖析。

在多 GPU 环境中捕获大型图时，如果硬件支持，可使用 CUDA 流优先级。例如，如果你希望计算核函数以更高的优先级运行、不被拖延，就可以把 NCCL 调用设置到一条较低优先级的流上。图捕获也会记录下这些优先级。

> NVIDIA 的 MLPerf 提交结果（以及内部基准测试）常常把整个训练步骤捕获进每次迭代的单个图中。这包括前向传播、反向传播、优化器以及 all-reduce 通信步骤。这能消除几乎所有的运行时开销（启动抖动等），代价是内存和灵活性。

*为内存复用做好规划* 图被捕获之后，你无法释放或重新分配该图所使用的任何张量——也无法改变它们的任何尺寸。最好按你预期的最大批次大小来捕获，从而预留出比实际所需略多一些的容量。这样，稍大一些的输入在之后重放时就不会破坏图。

例如，如果你的最大批次大小是 64，但偶尔会用 96 来运行，那么为保险起见，可以考虑按批次大小 96 来捕获。通常，捕获最坏情况、运行较小批次、浪费一点内存——要好过冒图失败的风险。

> 记得把优化器状态的大小也算进去（例如 Adam 优化器的中间缓冲区），因为它们是模型训练图的一部分。这些是比较容易被忽略的！

在使用得当的情况下，CUDA Graphs 能为大模型的训练和推理工作负载带来显著的加速。既然计算方面的优化已经就绪，让我们转向 PyTorch 中的内存优化。

### CUDA Graph Trees（PyTorch 编译器内部机制）

我们在前面的小节中已经介绍过 PyTorch 编译器，但在 CUDA Graphs 的语境下，值得一提的是 PyTorch 的 CUDA Graph Trees。它们被 torch.compile（具体来说是 mode="reduce-overhead"）用来为每种输入形状编译并缓存各自独立的静态图。

一旦一个静态图被记录下来，它的张量维度就必须保持固定。任何新的输入形状都会触发一次全新的记录和缓存条目。为了最大化缓存命中、减少图捕获，建议在多次迭代之间尽量保持输入形状的一致。不同形状越少，就越能复用已记录的图，开销也越小。

值得注意的是，你通常不会直接调用 CUDA Graph Trees 的 API。当指定 mode="reduce-overhead" 时，torch.compile 会为你处理这一切。

由于 CUDA Graphs 要求静态地址和稳定的控制流，对于 LLM 推理而言，整图捕获是很困难的，因为 LLM 推理支持可变的输入尺寸、批次大小和步数（例如采样、KV 缓存增长、主机侧决策等）。

对于模型训练，CUDA Graph Trees 允许多个被捕获的子图在前向和反向捕获之间共享同一个内存池。CUDA Graph Trees 对推理也很有用，因为它们允许根据输入形状和批次大小动态地选择子图。这通常被称为“分段捕获”（piecewise capture）。PyTorch 借助 CUDA Graph Trees 来维护按形状划分的分段捕获，并管理共享内存池。

> CUDA Graph Trees 提供的分段捕获模式，正是 vLLM 推理引擎用来支持不同输入形状与批次大小组合下的不同图的机制。我们将从第 15 章开始更详细地介绍 vLLM 和推理引擎的优化。

## 在 PyTorch 中剖析与调优内存

大模型可能会受限于 GPU 的内存容量和内存带宽。此外，即便 HBM 容量足够，低效的内存使用（例如内存碎片（memory fragmentation））也会损害性能。你可以从多个方面来应对内存问题，包括内存分配器调优、激活检查点（activation checkpointing）、内存卸载（offloading）以及输入流水线优化。

另外，PyTorch 有一个内置于 torch.profiler 的内存剖析器，只需像前面展示的那样启用 profile_memory=True 即可。你可以用它找出哪些操作分配了大量内存——并优先设法处理这些操作，如图 13-5 中由 PyTorch 内存可视化工具生成的可视化结果所示。

![图 13-5. 三次前向与反向传播迭代的 PyTorch 内存剖析可视化](AI%20Systems%20Performance%20Engineering-ch13_images/figure-13-5.png)

此外，NVIDIA Nsight Systems 的 CUDA Memory Inspector 可以帮助你可视化内存碎片随时间是如何产生的。利用这些工具可以指导你的内存分配器调优工作，接下来我们就来探讨这一点。

### 调优 CUDA 内存分配器

PyTorch 为 CUDA 内存使用了一个缓存内存分配器。默认情况下，它会通过按需切分和回收 GPU 内存块来适应各种分配模式。然而，某些使用可变大小内存分配的工作负载模式可能会导致内存碎片。

内存碎片发生在 GPU 内存随时间被切分成许多不连续的空闲块时。这会使得分配一个大张量变得困难，即便总的空闲内存仍然足够。在 MoE 模型中，这尤其成问题，因为路由到每个专家的 token 数量会随每个批次而变化。因此，每个专家的输出激活张量在每次迭代时都可能是不同的大小。

可变大小的内存分配会留下参差不齐、碎片化的内存块。这些碎片内存块会在训练或推理运行的过程中不断累积。

为避免这种情况，你应当预先分配一个固定大小的专家输出缓冲区，并把它的尺寸设为你批次中任何专家可能处理的最大 token 数量。然后，你就可以在每次迭代中复用这个缓冲区。

通过保持缓冲区维度恒定，GPU 内存分配器就不会随时间产生内存碎片。每个专家写入的是预分配缓冲区中属于自己的那一片，而不是触发新的分配。这种方法能稳定内存使用、提升复用效率，并避免与碎片相关的失败或性能下降。

你可以通过一个环境变量来调整 PyTorch 分配器的配置，从而调优其行为。下面是一个例子：

```
export PYTORCH_ALLOC_CONF=\
max_split_size_mb:256,\
roundup_power2_divisions:[256:1,512:2,1024:4,>:8],\
backend:cudaMallocAsync
```

下面是对代码中每个指定配置参数的说明：

max_split_size_mb:256 这个参数指示分配器保持大的空闲块完整（最多 256 MB），而不是持续地把它们切分成细小的碎片。这有助于减少碎片。默认情况下，PyTorch 切分大块分配的力度较小。显式设置 max_split_size_mb 可以让大的连续空闲块能够供现代 LLM 中的大型神经网络层使用。

roundup_power2_divisions:[N:M,...] 该参数控制 PyTorch 的 CUDA 缓存分配器如何将张量尺寸的请求归入固定的分桶。它把每个“2 的幂”区间划分为 *N* 个大小相等的子桶——例如，当某个请求的大小落在 512 MB 与 1,024 MB 之间时。若在代码中指定 512:2，则 512 到 1,024 MB 的区间会被划分为 :2 个桶，并向上取整到 [512MB, 768 MB, 1 GB] 之一。举例来说，在使用 '512:2' 的 512 到 1,024 MB 区间内，一个 600 MB 的请求会向上取整为 768 MB。请检查分配器日志，以确认在你的环境中实际的分桶情况。这种策略能减少内存碎片，使分配尺寸标准化，并提高缓存复用率，因为相似的请求会命中同一个桶。

backend:cudaMallocAsync 指定该选项会启用 NVIDIA 的 CUDA 异步分配器作为底层的内存分配机制。这有助于避免在内存释放事件上的同步——并能在多线程场景（如多工作进程的数据加载）中提升性能。

通过自定义内存分配器配置，你可以保持更平稳、更可预测的内存使用模式。在长时间运行中，你可以用 torch.cuda.memory_stats() 监控内存碎片，确保内存占用保持稳定、不会急剧膨胀。

你也可以在运行时使用 torch.cuda.mem_get_info() 获取空闲内存与总内存。这能间接反映碎片情况：如果已分配张量的数量保持不变，而空闲内存却在下降，就说明碎片正在增加。

### 用于节省内存的激活检查点

对于超大模型来说，激活检查点——一些从业者也称之为梯度检查点（gradient checkpointing）——是管理内存的关键手段。在处理大型 LLM 时，有时无法在不耗尽内存的情况下，为反向传播（backpropagation）保存全部中间激活值。

使用激活检查点时，你不必在前向传播（forward pass）期间保存中间激活值（以备反向传播使用），而是仅在需要时于反向传播过程中即时重新计算它们。PyTorch 提供了 torch.utils.checkpoint 来自动完成这一过程。你只需包裹某个模型层——或一段连续的层——它们的前向激活值就不会被保存。

你可以在每个 transformer 块以及模型中每个专家的前馈网络层这样的粒度上应用检查点。这样一来，在计算完每个块的前向输出后，你就无需再把那些中间激活值保留在内存中。取而代之的是，在反向传播期间，PyTorch 会重新运行该块的前向传播，以重新生成用于梯度计算的激活值。

值得注意的是，你不必对所有内容都做检查点，只需聚焦于最大的那些层即可。一种常见策略是：只对持有海量激活值的 transformer 块做检查点，而不对层归一化（layer norm）和嵌入层等较小的层做检查点。这样能以最小的重算开销换来最大的内存节省。

> 使用 FSDP（完全分片数据并行，Fully Sharded Data Parallel）时，你还可以启用自动检查点，它会为你递归地将检查点应用到多个层上。

这种取舍是以增加计算量来换取更低的内存占用。所幸的是，相对于 HBM 内存的容量，现代 GPU 提供了充裕的 FLOPS，因此这项技术非常契合最新几代硬件。也正因如此，GPU 拥有额外的计算余量来承担这些额外的重新计算。

这种权衡往往是值得的。如果不使用激活检查点，你就需要减小输入批大小（batch size）——或减少专家数量——才能塞进有限的 GPU 内存。而使用检查点后，你就能从容地把模型放进内存，同时保留较大的批大小。

尽管激活检查点会让训练稍微变慢，但它让你能够训练更大的模型变体——并使用更大的输入批大小——这些原本是无法装入内存的。本质上，你是用 GPU 充裕的计算 FLOPS 去弥补其有限的 HBM 容量。

### 将参数卸载到 CPU 与 NVMe

除了检查点之外，你还可以卸载一部分无需常驻 GPU 的模型参数。例如，MoE 模型有一些访问频率较低的专家层，可以把它们卸载到 CPU 内存——仅在需要时才传输到 GPU。

关键在于让数据传输与计算重叠：当某一层在 GPU 上运行时，异步地从 CPU 或 SSD 预取下一层的权重。在实践中，DeepSpeed 的 ZeRO-Infinity（用于训练）和 ZeRO-Inference（用于推理）等框架可以自动完成这种预取。它们把模型权重逐层地从 CPU 或 NVMe 流式传输到 GPU，在使数据传输与计算重叠的同时，将 GPU 峰值内存占用降到最低。

你可以把这些组件锁页（pin）在主机上，并使用异步、非阻塞的 DMA 调用（如 .to(device) 和 cudaMemcpyAsync），在其他计算正在运行时将它们传输到 GPU。这样就能隐藏因从 CPU 拷贝而产生的额外传输延迟。

NVIDIA 的统一内存（Unified Memory）也是一种选择——尤其适用于像 Grace Blackwell GB200/GB300 这样、在 CPU 与 GPU 之间配备了 NVLink-C2C 等高速互连的 superchip 系统。在这类情形下，统一内存允许把很少使用的 GPU 内存页面驱逐到 CPU 的系统内存中。

如果为了容量需要，操作系统甚至可能把它们交换到 NVMe/SSD。NVMe 应作为通过操作系统正常交换机制的最后手段来使用，而不应作为统一内存的主要目标。

然而，由于内存分页等原因，统一内存可能带来不可预测性。因此，为了保持完全的掌控和可预测的性能，显式地管理卸载往往是更可取的做法。

GPUDirect Storage 等近期的进展让 GPU 能够直接从 NVMe 驱动器读取数据。这意味着在某些情况下，你的模型参数可以即时地直接从 NVMe 分页调入，而完全无需 CPU 参与。在使用超大模型时，这对训练和推理服务都很有用。

对于参数量达到万亿量级的更大模型，你可以把部分组件卸载到 NVMe 存储，并在恰好需要时（just-in-time）将它们换入 GPU 内存。关键在于让数据传输与计算重叠，从而不会拖慢训练循环。

### SuperOffload：面向 CPU-GPU superchip 的优化卸载

SuperOffload 是一套专门为发挥 CPU-GPU superchip 硬件效率而设计的卸载系统。superchip（例如 Grace Hopper、Grace Blackwell、Vera Rubin 等）在 CPU 与 GPU 之间提供了高带宽的 NVLink-C2C 互连。再结合共享、一致的内存地址空间，与传统卸载技术相比，针对 superchip 优化的卸载策略能带来巨大的加速与利用率提升。

SuperOffload 展示了几项关键创新，包括先推测后验证（speculation-then-validation，STV）、异构优化计算、感知 superchip 的数据类型转换以及内存拷贝。下面我们逐一讨论。

传统卸载会先等待梯度归约和全局检查，然后才更新参数；相比之下，STV 会在 GPU 执行反向传播的同时，在 CPU 上进行推测性的优化器更新，从而让这些步骤相互重叠，之后再对结果进行验证。这有效减少了同步停顿，并提升了整体 GPU 利用率。

SuperOffload 使用异构优化器计算，把优化器的工作在 CPU 与 GPU 之间进行划分。例如，它将计算量大的张量更新分配给 GPU，而由 CPU 核心处理较轻量的状态更新，比如 Adam 优化器所用的动量缓冲区。这让两种设备都保持忙碌，减少空闲周期，并提升整体芯片利用率。

由于 NVLink-C2C 在 CPU 与 GPU 之间提供了高带宽，SuperOffload 可以改变张量类型转换和数据传输的精度与放置策略。通过把类型转换和拷贝转移到 GPU 一侧，SuperOffload 充分利用了快速、一致的互连，从而将 CPU-GPU 传输延迟降到最低。

此外，SuperOffload 使用了一种针对 CPU 优化的 Adam 优化器变体，名为 GraceAdam，它是为 Grace 的 ARM 可伸缩向量扩展（Scalable Vector Extension，SVE）架构而设计的。SVE 是 ARM A64 指令集所采用的长向量架构。具体而言，它拥有 32 个向量寄存器和 16 个谓词寄存器，与 SuperOffload 结合后，有助于提升被卸载的参数更新的吞吐量与能效。

### FSDP 的自动检查点与卸载

PyTorch 的 FSDP 是一种分布式并行策略，在模型训练期间把模型参数、梯度和激活值分片（shard）到多个 GPU 上。这降低了训练期间的内存开销——并让你能够训练比不分片时更大的模型。从技术上讲，FSDP 实现了 ZeRO Stage-3 策略，将模型状态分片到多个 GPU 上，如图 13-6 所示。

![图 13-6. FSDP 将模型参数、梯度和优化器状态分片到多个 GPU 上（ZeRO Stage-3）](AI%20Systems%20Performance%20Engineering-ch13_images/figure-13-6.png)

FSDP 可以在底层自动应用激活检查点，并卸载参数/梯度。只需用 FSDP() 包裹模型，然后按如下方式指定 activation_checkpointing_policy 和 CPUOffload 参数：

```
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload, ShardingStrategy, BackwardPrefetch, MixedPrecision
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
# Initialize distributed
dist.init_process_group("nccl")
torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
# Build your model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
        )
    def forward(self, x):
        return self.layers(x)
model = MyModel().cuda()
# Auto-wrap transformer blocks if needed
auto_wrap_policy = transformer_auto_wrap_policy(
    model,
    min_num_params=1e8,
)
# Wrap with FSDP + checkpointing + CPU offload
fsdp_model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    use_orig_params=True,
    cpu_offload=CPUOffload(offload_params=True, pin_memory=True),
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    ),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    activation_checkpointing_policy={
        nn.TransformerEncoderLayer,
        nn.TransformerDecoderLayer,
        nn.MultiheadAttention
    }
)
# Setup optimizer
optimizer = torch.optim.AdamW(fsdp_model.parameters(), weight_decay=0.01)
...
```

这里我们使用 transformer_auto_wrap_policy，让 FSDP 根据你的 transformer 块结构自动分片参数、梯度和优化器状态。我们还通过 CPUOffload(offload_params=True) 启用了向 CPU 的卸载。这会在参数和梯度不需要驻留 GPU 时，透明地把它们移动到 CPU，从而降低 GPU 峰值内存占用。

> 通过设置 use_orig_params=True，每个 FSDP 单元在处理其参数时无需展平。这带来了更好的重叠以及更简单的状态字典（state-dictionary）处理——从而改善内存管理和优化器兼容性。

设置 activation_checkpointing_policy 会告诉 FSDP 在那些特定的核心 transformer 子模块中重新计算（或称重新物化，rematerialize）激活值。这会以额外的计算换取显著更低的峰值内存。它能达到与手动的 checkpoint() 包裹和自定义卸载脚本相同的巨大内存节省，却无需额外的样板代码。FSDP 甚至能处理各 GPU 之间不均衡的批大小，这对 MoE 工作负载很有用。

FSDP 还支持一种混合分片策略，通过 ShardingStrategy.HYBRID_SHARD 使用。该策略把每个节点的参数、梯度和优化器状态分片到该节点内的各个 GPU 上，同时把这些相同的分片复制到其他节点。本质上，混合分片提供了比 FULL_SHARD 更高的吞吐量，但代价是每个节点占用更多内存。

当你的互连不错但并非极快时，应使用 HYBRID_SHARD。它在每个节点内部分片参数、梯度和优化器状态——并将这些分片跨节点复制。这减少了跨节点流量，代价是每个节点的内存占用比完全分片略高一些。

混合分片让你能够管理内存与通信之间的权衡。例如，你可以比 FULL_SHARD（ZeRO 3）占用更多的每节点内存，因为你在每个节点上都持有一份完整的本地分片。这减少了节点间通信，通常能提供更高的端到端吞吐量。当你拥有非常快的多节点网络结构、并希望每个 GPU 的内存占用最小时，FULL_SHARD 是最佳选择。

当你的网络较慢——或只有少量几块 GPU——时，你可以使用 ShardingStrategy.SHARD_GRAD_OP（ZeRO Stage-2），只把梯度和优化器状态分片到所有 GPU 上。该策略会在每块 GPU 上都保留一份完整的参数副本。

### 将 FSDP 与张量并行、流水线并行结合

如果模型大到单个层都无法装入单块 GPU 的内存，你就需要把 FSDP 与其他并行策略（如张量并行（tensor parallelism，TP））结合起来，把庞大的模型层铺展到多块 GPU 上。

你还可以跨 GPU 和计算节点使用 FSDP：在节点内用 TP 把巨大的层拆分到多块 GPU 上，再用流水线并行把这些经过 TP 拆分的层跨节点串联起来。FSDP 支持这类灵活的组合。

简而言之，FSDP 能以少得多的编码工作量来降低内存占用。在应用检查点和卸载之后，你就能把模型装入内存，并启用更大的批大小。这有助于提升 GPU 利用率。

### 可插拔内存分配器（pluggable memory allocator）与跨 GPU 数据传输

你可以配置 PyTorch，为关键的 GPU 通信操作（如 NCCL 的梯度 all-reduce）使用专门的内存分配器。通过用 torch.cuda.MemPool 插入自定义分配器，你可以让 NCCL 以一种能利用专用硬件引擎（如 NVLink 拷贝引擎、InfiniBand 卸载或 NVIDIA SHARP）的方式来分配缓冲区。这些都有助于改善通信与计算之间的重叠。例如，PyTorch 支持在 NVSwitch 环境中使用 NCCL 的内存分配器来进行快速、零拷贝的归约，如下所示：

```
import torch
import torch.distributed as dist
from torch.cuda.memory import MemPool
from torch.distributed.distributed_c10d import _get_default_group
# Initialize NCCL distributed backend
dist.init_process_group(backend="nccl")
torch.cuda.set_device(
  dist.get_rank() % torch.cuda.device_count()
)
# Get the NCCL backend object for this device
default_pg = _get_default_group()
backend = default_pg._get_backend(
  torch.device(f"cuda:{torch.cuda.current_device()}")
)
# The backend exposes ncclMemAlloc using mem_allocator
nccl_allocator = backend.mem_allocator
# Create a dedicated memory pool using this allocator
nccl_pool = MemPool(nccl_allocator)
# Register the pool so NCCL uses it for collective gradient buffers
backend.register_mem_pool(nccl_pool)
# Use the pool explicitly for NCCL operations
with torch.cuda.use_mem_pool(nccl_pool):
    tensor = torch.randn(10_000_000, device="cuda")
    dist.all_reduce(tensor)
```

通过把 NCCL 的原生分配器（backend.mem_allocator）绑定到 torch.cuda.memory.MemPool 中，并使用 backend.register_mem_pool(...)，PyTorch 会把梯度归约缓冲区直接放置到经过优化的内存区域中，以便通过 NVLink、InfiniBand 和支持 NVIDIA SHARP 的硬件实现最优路由。这样一来，all-reduce 操作就能从硬件加速中受益，减少大规模数据归约期间的 SM 争用。其结果是大型多 GPU 工作负载的吞吐量得到提升。

> 使用第 4 章讨论过的 SHARP，需要兼容的网络结构（例如支持 SHARP 的 HDR InfiniBand）。如果条件具备，启用 NCCL 基于 SHARP 的网络内聚合能大幅降低大型集群的 all-reduce 延迟。当扩展到大量节点时，这绝对值得考虑。

通过用 backend.mem_allocator 使用 NCCL 的原生分配器，并将其包裹进 PyTorch 的 MemPool，梯度 all-reduce 缓冲区会被分配到为 GPUDirect RDMA 和网络内 SHARP 卸载而优化的内存区域中。这会把缓冲区对齐到大页边界，并有助于把数据传输放到专用的 DMA 引擎上——减少 SM 的介入，释放出计算能力去执行更有价值的计算工作。

其结果是，像张量并行归约这样的 NCCL 集合通信操作既能从硬件加速中受益，又能获得更低的 SM 争用。这显著提升了多 GPU 同步吞吐量，减少了 SM 和拷贝引擎上相互重叠的计算与通信核函数之间的争用——对张量并行工作负载而言尤其如此。

当你逼近带宽极限时，这类优化会变得愈发重要。例如，Blackwell 的 NVLink 5 在理论上为每块 GPU 提供高达 1.8 TB/s 的双向带宽（每个方向约 900 GB/s）。使用专用拷贝引擎、网络硬件（SHARP）和优化过的内存分配器，有助于逼近峰值网络吞吐量，同时释放 SM 去达到峰值 FLOPS。

借助 torch.cuda.MemPool，你还可以通过注册一个自定义库（共享对象，即 .so）来创建自己的内存分配器。此外，你还可以在同一个 PyTorch 应用中混用不同的 CUDA 内存分配器，如下所示：

```
import torch  # PyTorch main namespace
import os     # for path operations (used here for .so extensions)
from torch.cuda.memory import CUDAPluggableAllocator
# 1. Create a CUDAPluggableAllocator and MemPool
# Build a pluggable allocator that calls into your NCCL library:
#   - allocator_path: path to your .so
#   - "ncclMemAlloc": symbol for allocation
#   - "ncclMemFree":  symbol for deallocation
# .allocator() returns a callable that matches the CUB/CUDA allocator API
allocator = CUDAPluggableAllocator(
    "./nccl_allocator.so",
    "ncclMemAlloc",
    "ncclMemFree"
).allocator()
# Wrap that allocator in a MemPool for efficient sub-allocations
pool = torch.cuda.memory.MemPool(allocator)
# 2. Start recording events (set a high cap for long runs)
torch.cuda.memory._record_memory_history(max_entries=100000)
# 3. Allocate tensors with different allocators
# tensor0 uses the *default* cudaMalloc allocator
# - Shape: (1024, 1024), you can change to your desired size
tensor0 = torch.randn(1024, 1024, device="cuda")
# tensor1 uses *your* NCCL-backed allocator using MemPool
with torch.cuda.use_mem_pool(pool):
    # Inside this context, all cuda allocations go through `pool`
    tensor1 = torch.randn(1024, 1024, device="cuda")
# Exiting the context restores the default allocator
# tensor2 again uses the *default* cudaMalloc allocator
tensor2 = torch.randn(1024, 1024, device="cuda")
# 4. Inspect memory pool stats
# Pool-specific snapshot with list of segments/blocks in use
pool_state = pool.snapshot()
print(f"Pool segments count: {len(pool_state)}")
# 5. Dump the snapshot and optionally load in the PyTorch
# memory viewer tool (https://oreil.ly/tX6gA)
torch.cuda.memory._dump_snapshot('memory_snapshot.pkl')
# Global allocator stats (allocated/reserved, peak, counts)
global_stats = torch.cuda.memory_stats()
print("Peak allocated bytes:", global_stats["allocated_bytes.all.peak"])
# 6. Stop recording
torch.cuda.memory._record_memory_history(enabled=None)
# 7. Reset peak counters for a fresh measurement
torch.cuda.reset_peak_memory_stats()
```

这里，CUDAPluggableAllocator 会加载你的自定义 .so，并绑定到你指定的两个符号上。MemPool 把底层的 CUDA 内存分配器包裹进一个用于内存分配与内存释放的缓存中，以获得更好的性能。

torch.cuda.use_mem_pool(pool) 会为 Python 上下文管理器（例如 with）代码块内所有后续的内存分配换入你的内存池。退出该上下文管理器代码块时，会恢复之前的分配器。

### 启用点对点 DMA 与 UCX

在多 GPU 系统中使用流水线并行时，你通常希望用尽可能最快的点对点（peer-to-peer，P2P）连接，把激活值直接从一块 GPU 搬到下一块 GPU——避免往返以及占用主机 CPU 资源。当张量在一个进程内跨设备移动时，PyTorch 会自动探测并启用对等访问。不过，如果你想手动确认，可以使用 torch.cuda.can_device_access_peer(i, j) 来确认 GPU i 与 GPU j 之间的 P2P DMA。对于自定义的 C++ 操作，你可以用 CUDA 驱动 API 显式地启用对等访问。

启用 P2P DMA 可提供高效的直接传输，而无需动用 GPU 的 SM 或 CPU 内存。一旦启用，跨 GPU 的 copy_() 和 .to() 就会走更快的对等内存路径，无需额外的代码开销，如下所示：

```
dst.copy_(src, non_blocking=True)
# or
src.to(device="cuda:1", non_blocking=True)
```

当你的拓扑启用了 P2P 访问时，这些方法会使用 cudaMemcpyPeerAsync。要让 P2P 传输正常工作，你的硬件拓扑必须支持它。GB200/GB300 NVL72 机架在内部使用了具备 P2P 能力的 NVLink 和 NVSwitch，因此它开箱即用地已经为 P2P DMA 做好了准备。

到目前为止，我们只在 P2P 数据传输的语境下讨论了多 GPU 配置。然而，当使用多节点 GPU 拓扑时，你需要退回到通过网络通信的 NCCL。有时这会经由 GPUDirect RDMA 进行。但把 UCX（Unified Communication X）与 NCCL 搭配使用可以提升性能。在 GB200 和 GB300 NVL72 拓扑中，NVLink 和 NVSwitch 在机架内部提供了 P2P 路径。而跨机架时，你就需要通过启用了 UCX 的 NCCL 传输来穿越网络结构。

要让跨节点传输经由 UCX 路由，需安装 NCCL-UCX 插件并设置 NCCL_PLUGIN_P2P=ucx。许多环境还要求设置 NCCL_NET=UCX，并为你的网络结构配置合适的 UCX_TLS 传输选择。这会启用硬件卸载，并改善 NVL72 机架之间 InfiniBand 网络结构上的流水线化。

以下是 NCCL-UCX 插件的一个示例配置：

```
export NCCL_NET=UCX
export NCCL_PLUGIN_P2P=ucx
# UCX transports vary by fabric.
# These are safe defaults to start with:
export UCX_TLS=rc,self,gdr_copy,cuda_copy
```

如果你的应用需要跨多个节点或机架进行扩展——训练 TB 级模型、面向众多租户对超大 LLM 做推理，或管理数据密集型流水线——那么 UCX 就必不可少。将 NCCL 与 UCX 搭配使用，能提供高吞吐、低延迟的跨节点通信，既支持硬件卸载（RDMA），又具备智能的拓扑感知。UCX 是生产级 AI 基础设施的核心组成部分，在扩展到单个 NVLink 域之外时不可或缺。

简而言之，与使用 NCCL 的等效 send/recv 调用相比，直接的 P2P DMA 通信和 UCX 能实现与计算更好的重叠。这是一项相对底层的优化，但如果你拥有合适的拓扑以及 NVLink/NVSwitch 等专用高速互连，它就能大幅提升系统性能。

## PyTorch 对称内存

对称内存（symmetric memory）是一种编程模型，它在多块 GPU 之间暴露出一个分区全局地址空间（partitioned global address space），使核函数能够执行单边的 put 和 get 操作。这让它们可以在无需 CPU 握手或介入的情况下，调用超低延迟、跨 GPU 的直接访问集合通信。本质上，对称内存分配的缓冲区可以被组内任意 GPU 直接寻址——而无需显式的点对点拷贝。

使用对称内存时，你可以完全在设备上执行 all-to-all 操作，比如 MoE 的 token 混洗。由于没有 CPU 参与，整个 all-to-all 都可以被 CUDA Graph 捕获。若没有对称内存，一次 all-to-all 会触发一次主机同步，从而在时间线上造成设备到主机（device-to-host，D2H）的空隙。这会在 CUDA Graph 中制造出不想要的中断。

你可以分配对称张量，在一个进程组内执行一次汇合（rendezvous）以获取远程访问句柄，然后调用直接访问的集合通信（例如 all_to_all_v、one_shot_all_reduce）。此外，你还可以启动使用这些远程访问句柄执行单边读/写的核函数。与 OpenAI Triton 和 NVSHMEM（triton.nvshmem.put()/get()）结合后，对称内存成为一种强大的通信机制，可用于自定义的、核函数内的数据传输。

> 对于细粒度、延迟敏感、且需在设备上完成的交换（如无需主机 CPU 介入的 MoE all-to-all token 混洗），你应当在 PyTorch 中使用对称内存。这将消除设备到主机（D2H）的时间线空隙，并实现更好的 CUDA Graph 捕获。

## 优化数据输入流水线

在讲完模型的内存与计算之后，我们来看看数据输入流水线（data input pipeline）。造成低效的一个常见原因是 GPU 空闲地等待数据。PyTorch 的 DataLoader 支持在训练模型的同时，派生多个工作线程或进程来并行地加载和预处理数据（例如文本分词）。请务必指定 pin_memory=True，使主机 CPU 内存到 GPU 设备的传输使用页锁定（锁页）内存。

如果不锁定内存，你可能会在数据加载器（data loader）进程中看到很高的 CPU 利用率，而在训练进程中看到很低的 GPU 利用率。这是双重的坏事。这些不理想的利用率可以用 htop 和 Nsight Systems 等工具观察到。你会看到 CPU 线程忙于执行数据加载，而 GPU 却处于空闲。

这表明数据加载没能跟上每一次迭代的节奏。你可以通过增大 DataLoader 的 num_workers 参数值来解决，直到数据队列备好足够多的批次为止。对于 CPU 受限的数据流水线，一个常见的经验法则是每块 GPU 配 4 个工作进程，但最优数量会随你的工作负载而变化。

为了找到最优数量，你可以做一次快速扫描（例如 4、8、16、32），找出再增加工作进程也不再有帮助的临界点。要留意 CPU 是否饱和。如果所有核心都处于 100%，那么增加工作进程也无济于事。

请记住，像 Grace Hopper、Grace Blackwell 和 Vera Rubin 这样的 NVLink C2C superchip 系统提供了一个大容量、一致、高带宽的 CPU–GPU 内存地址空间。在这类环境中，页锁定的 pin_memory=True 往往不那么关键，因为传输本就走的是高带宽的一致路径。

在非一致路径上，为了最大化可分页的主机到设备拷贝的重叠，可能仍然需要锁页内存。虽然统一内存和一致映射在 NVLink-C2C 系统上可能降低对锁页的需求，但衡量你自身工作负载的性能仍然很重要。

即便使用 CPU-GPU superchip 架构，当加载器线程受限于 CPU 时，仍建议把 pin_memory=True（或大页）与 non_blocking=True 和 persistent_workers=True 结合使用。使用 persistent_workers=True 可以避免在各个 epoch 之间重新派生进程。这对分词密集型的工作负载（如 LLM 的训练与推理）很有帮助。请用 Nsight Systems 做剖析，验证实际的重叠情况。在移除锁页之前，务必在主机上启用大页，并确认 H2D 流量与计算相互重叠。

CPU 与 GPU 共享一个一致的内存空间，因此即便不显式锁页，传输也已走高速路径。相比之下，在没有此类硬件一致性的标准（非 superchip）CPU + GPU 系统上，启用 pin_memory=True 会把主机内存页锁定，并带来数据传输吞吐量的显著提升。

> 如果你不锁定内存，而是依赖 Grace Blackwell 和 Vera Rubin 等 superchip 系统的统一 CPU-GPU 内存，那么务必启用大页，并用 Nsight Systems 验证传输是否确实发生了重叠。这是因为 NVLink-C2C superchip 上的统一 CPU-GPU 内存改变了锁页方面的权衡。

另一种提升数据加载响应速度的技术，是增大 DataLoader 的 prefetch_factor。它控制每个工作进程提前预加载多少个批次，其计算方式为 prefetch_factor * num_workers。这意味着当你的模型忙于处理当前批次时，工作进程已经在加载并缓存后续的批次了。

预取能让数据流水线保持充盈，避免 GPU 空转。你应当根据数据集和硬件能力，把 prefetch_factor 与 num_workers 一起调整。请通过剖析来验证，以免过度预取而造成不必要的内存占用。

你还可以预先计算好分词后的数据集，并把分词后的数据集存到磁盘上，从而避免在数据加载循环中做繁重的处理。像 Hugging Face 的 Dataset.cache() 和 WebDataset 这样的工具，让你只需对文件预处理一次即可反复复用。此外，可以考虑对数据集使用混合精度和压缩，以降低对 I/O 带宽的需求。

还有 PyTorch 原生的 TorchData 及其 DataPipes。它提供了良好的可组合性，并与 PyTorch 调度器集成，以便与训练计算相互重叠。

另一个有用的工具是 NVIDIA Data Loading Library（DALI）。DALI 可以在训练的同时并行地执行 CPU 或 GPU 预处理。它对图像/视频数据（例如解码和增强）尤其有用，并且能通过 CUDA 流水线把数据直接喂给你的训练代码。对于那些适合卸载到 GPU 的即时数据变换，它是一个很有用的工具。

在所有情况下，目标都是尽可能让数据加载与 GPU 计算相互重叠。通过用 Nsight Systems 做剖析，你可以确认数据加载器是在与 GPU 并行工作的。Nsight Systems 的时间线应当显示：一个 CPU 线程在持续地加载/预处理下一个批次，同时多条 GPU 流在执行训练步骤。此外，还要验证不存在 GPU 等待数据的空隙。这能确保你的 GPU SM 几乎 100% 的时间都在忙于有价值的训练工作。

假设内存约束已经通过激活检查点和卸载得到解决，你就可以增大每块 GPU 的输入批大小。使用更大的批次能够提高算术强度、改善 GPU 利用率，并降低多 GPU 训练中通信的相对开销，因为每个 epoch 的步数变少了。

然而，批大小过大会把优化器推向尖锐极小值（sharp minima），从而降低泛化能力、影响收敛。在大幅增大批大小时，务必监控 GPU 内存占用。请记住，梯度累积会增大有效批大小（batch_size * accumulation_steps）。PyTorch 的 TorchEval 指标可以帮助判断更大的批大小是否在损害验证损失。

你或许可以通过相应地调整超参数，来把大批量带来的不稳定影响降到最低。例如，你可以改变学习率、采用带预热期的线性学习率缩放，或使用像 LAMB 这样的大批量优化器。

简而言之，如果内存允许——并且前几节介绍的其他内存优化已经落实并验证过——那么增大批大小可以提高算术强度，并更好地利用 GPU。

> 在优化系统的过程中，你应当定期重新审视并重新调优批大小、学习率等超参数。在你改动批大小等设置之后，最初选定的取值可能就不再是最优的了。

## 使用 PyTorch Distributed 进行扩展

在多块 GPU 和多个计算节点上扩展并剖析 PyTorch，通常会用到 PyTorch 的分布式库，如 PyTorch DDP（分布式数据并行，Distributed Data Parallel）和 FSDP。好消息是，PyTorch 的编译器可以与这些并行方法协同工作，但其中有一些细微之处，下面将加以说明。

### DDP 与 torch.compile

DistributedDataParallel 使用 all-reduce 集合通信在多块 GPU 之间同步梯度；当使用它时，PyTorch 会在同步点自动产生图中断。在实践中，DDP 会把梯度划分到若干桶中，并让通信与计算相互重叠。PyTorch 的设计是把每个桶的反向计算编译为一个单独的图，从而能在这些图之间执行 all-reduce。

在 torch._dynamo.explain(model, ...) 的输出中，你会看到与 all-reduce 和 torch.distributed 操作相关的图中断。这是符合预期的，因为每个桶（bucket）的计算都会被编译——而 all-reduce 发生在各子图之间。这样一来，通信重叠得以保留，而这对性能至关重要。

如果你将 DDP 与编译器一起使用，请确保 DDP 的桶大小合理。默认情况下，DDP 的桶为 25 MB。如果你选择覆盖这一默认值，用一个巨大的桶来容纳所有梯度，那么就会生成一个巨大的图，并在末尾进行一次庞大的 all-reduce。这会导致图段更少、更大，几乎没有机会将通信与计算交错进行。

使用单个桶很有诱惑力，例如让整个反向传播在单个图中完成，从而实现最大程度的核函数融合（kernel fusion）。然而，这样你会失去重叠的机会。建议对系统进行剖析，为你的工作负载和硬件环境找到合适的平衡点。

> 在将 DDP 与 torch.compile 一起使用时，你会在通信点看到刻意的图中断。这些是正常的，也是让网络通信得以进行所必需的。TorchDynamo 的 explain() 输出会显示关于 all-reduce 和 scatter 导致中断的消息——这是符合预期的。

### FSDP 与 torch.compile

PyTorch 的 FSDP 并行策略会将模型参数、梯度和优化器状态分片到多块 GPU 上。它在前向传播期间执行一次 all-gather 集合通信，在反向传播期间执行一次 reduce-scatter。相比使用 DDP，这使得更大的模型能够装入 GPU 内存。

配合 torch.compile，FSDP 可以更加高效，但这需要精心包装以最大化计算与通信的重叠。建议将每个 transformer 块——而不仅仅是单个底层层——用 use_orig_params=True 包装进各自的 FSDP 模块。这样，当需要通信时，TorchDynamo 会在每个分片边界插入一个图中断。

于是，每个块的前向和反向计算作为单个已编译的图执行，而 all-gather（前向）与 reduce-scatter（反向）通信发生在这些图之间。这与 DDP 的分桶方式如出一辙——后者对每个通信块单独编译。这样便可在计算步骤与通信步骤之间实现适当的重叠。

这与把整个模型包装进单个 FSDP 实例形成对比——后者很有诱惑力。如果你这么做，TorchDynamo 会产生更少、更大的图。其结果是通信点更少、通信与计算重叠的机会更少，以及前向和反向两个阶段跨图融合受限。

以 transformer 块的粒度用 FSDP 包装模型，可以让你的训练流水线实现最大程度的重叠。这是因为每个块的计算密集型逻辑都被编译和融合。而这部分计算与将各块连接起来所需的通信相重叠。

PyTorch 在 torch.distributed.fsdp.wrap 中提供了一个可调用对象 transformer_auto_wrap_policy，让你可以直截了当地对模型中的每个 TransformerBlock 应用 FSDP，而无需手动嵌套。采用块级分片时，任一时刻内存中只会实体化一个块的完整权重。all-gather 与 reduce-scatter 集合通信被交错进行，并隐藏在每个块的计算之后。下面是一个使用 PyTorch 的端到端示例，它定义了一个自动包装策略，包装了一个示例 transformer 块，并运行了前向和反向步骤的示例：

```
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP,
    CPUOffload, ShardingStrategy, BackwardPrefetch, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper
    import apply_activation_checkpointing, checkpoint_wrapper
# Prefer BF16 for Blackwell/Hopper-class GPUs; allow TF32 matmul for speed
# where numerically safe.
# enables TF32 use for FP32 matmuls when allowed
torch.set_float32_matmul_precision('high') # {'highest'|'high'|'medium'}
# Compile the model with a mode that reduces Python/launch overhead and
# enables CUDA Graphs
compiled_model = torch.compile(model, mode="reduce-overhead")
# Auto-wrap transformer blocks; adjust set below to match model’s block classes
auto_wrap_policy = torch.partial(
    transformer_auto_wrap_policy,
    transformer_layer_cls={
        nn.TransformerEncoderLayer,
        nn.TransformerDecoderLayer,
        nn.MultiheadAttention,
    },
    min_num_params=int(1e8),
)
# Optional activation checkpointing for target modules
def _ckpt_check_fn(m: nn.Module) -> bool:
    return isinstance(m, (nn.TransformerEncoderLayer,
                          nn.TransformerDecoderLayer,
                          nn.MultiheadAttention))
apply_activation_checkpointing(
    compiled_model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=_ckpt_check_fn,
)
# FSDP wrapper with correct argument types; MixedPrecision explicitly specified
fsdp_model = FSDP(
    compiled_model,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    cpu_offload=CPUOffload(offload_params=True),
    use_orig_params=True,
    mixed_precision=MixedPrecision(param_dtype=torch.bfloat16,
                                   reduce_dtype=torch.bfloat16,
                                   buffer_dtype=torch.bfloat16),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
)
# Example input
batch = torch.randn(8, 128, dtype=torch.float32, device='cuda')
labels = torch.empty(8, 128, dtype=torch.long, device='cuda').random_(0, 100)
optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4, fused=True)
# Forward + backward
outputs = fsdp_model(batch)
loss    = nn.CrossEntropyLoss()(outputs.view(-1, outputs.size(-1)),
                                labels.view(-1))
loss.backward()
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

这里，transformer_layer_cls 被设置为你的块类，从而让每个块独立分片。以块粒度包装意味着每次前向传播只会为当前块调用 all-gather，然后丢弃其分片。因此，任一时刻只有一个块的参数和优化器状态被完整实体化，从而将峰值内存占用最多减少到总参数中相当于一个块大小的比例。

此外，当每个块的计算运行时，下一个块的权重可以被异步预取或搬移。这就把通信延迟隐藏在了块计算之后。

注意，训练循环并不改变。auto_wrap_policy 抽象掉了手动嵌套，因此你只需指定一次你的块类，其余的——包括逐块通信——都交由 FSDP 在底层处理。

简而言之，在 transformer 块级别包装 FSDP 能带来更好的性能和内存效率。由于每个已编译块使用了融合、优化的核函数，因此占用的峰值内存更少。通信与计算被适当地重叠。而 FSDP 的分片以及按需、逐层的 all-gather 集合通信意味着任一时刻内存中只完整存在一个块的权重——而且只在每次前向和反向传播中需要它们时才存在。

> 请记住，TorchDynamo 的 explain() 输出会在每个 FSDP 边界显示图中断。这些中断是符合预期的，反映了正确的重叠行为。

### 张量并行与流水线并行结合 torch.compile

张量并行和流水线并行与 torch.compile 是正交的。只要跨 GPU 的通信操作被 TorchDynamo 识别为集合调用，Dynamo 就会相应地对它们进行追踪或中断。

PyTorch 编译器主要专注于优化每个片段内部的计算。它（目前）不会融合通信操作，也不会改变它们的调度——除了前面描述的那种自然重叠之外。具体而言，TorchInductor 为计算选择 cublasLt/cuDNN/Triton 核函数，但把 NCCL 集合通信（及其顺序）留给分布式策略处理。在使用任何分布式训练策略时，你都应当分别在启用和不启用 torch.compile 的情况下进行测试，以确保得到预期的结果。启用编译时，务必用剖析器追踪来验证重叠。

建议用 torch.compile 优化每层或每个桶内部的计算——并信任分布式策略（DDP、FSDP、TP、PP 等）像往常一样处理 GPU 之间的通信。留意 TorchDynamo 的 explain() 输出中任何与 all-reduce 相关的警告，但请记住 PyTorch 的设计力求确保重叠得以保留。因此，你不需要做太多改动，因为 torch.compile 对分布式训练的支持已经相对成熟。

### TorchTitan、AsyncTP、AutoParallel 与 SimpleFSDP

TorchTitan 是一套流行的、基于 PyTorch 的大规模模型训练参考实现。它提供了一组可扩展的方案，组合了多种分布式训练策略，包括 FSDP、张量并行和异步张量并行（asynchronous tensor parallel，AsyncTP）。

具体而言，AsyncTP 使用双流（dual streams）和一种感知 SM 波次（SM-wave aware）的调度，来错开 TP 集合通信，并将迟到的 TP all-gather 与下一波的矩阵乘法（matmul）相重叠。这种重叠有助于隐藏传统 TP 中出现的通信与调度间隙。团队在使用这类异步并行策略时，正在前向传播和端到端两方面都看到加速。请务必仅在能够验证数值正确性和扩展性的场景下使用异步并行。

你可以通过 torch.compile 启用 AsyncTP。这会使矩阵乘法有资格被降低（lower）为融合的 all-gather 和 reduce-scatter。AsyncTP 与 torch.compile 组合良好，使得融合的计算核函数与已调度的通信步调一致地运行。

AutoParallel 是 PyTorch 的另一项计划，它会自动为模型图规划并应用 FSDP、张量并行和流水线并行的不同组合。

AutoParallel 构建在 DTensor 的算子策略系统之上。DTensor 是 PyTorch 原生的分片原语，支撑着 PyTorch 可组合的张量并行与流水线并行方法。DTensor 在 TorchTitan 的参考实现中被大量使用。

借助一种基于启发式的选择机制，AutoParallel 采用不同的并行方案来应用分片和分区，这些方案会考虑给定工作负载的具体内存和通信成本。

对于大型模型和复杂集群，你应当使用 AutoParallel，以帮助减少手动并行化调优——而且它与 TorchTitan 组合良好。

SimpleFSDP 以一种对 torch.compile 友好的方式重新实现了 FSDP，使用了 DTensor 以及诸如选择性激活检查点（selective activation checkpointing）等技术。编译器借助 TorchInductor 对中间表示（intermediate representation，IR）节点进行分桶和重排序，从而帮助追踪并优化计算与通信的重叠。

在 TorchTitan 的实验中，SimpleFSDP 将内存使用最多降低了约 28%。而当它与其他分布式技术组合时，相比传统的 FSDP2 即时执行路径，训练吞吐量最多提升了约 69%。

> TorchTitan、AsyncTP、AutoParallel 和 SimpleFSDP 都是维护良好的项目，值得关注。它们代表了实用的 PyTorch 参考实现，并纳入了业界众多 PyTorch 专家的许多优化。

### 用 HTA 进行多 GPU 剖析

当你扩展到数百万块 GPU 时，处理数百万份独立的追踪和剖析结果会变得难以驾驭。Meta 的 HTA 有助于合并并可视化多工作节点的追踪。HTA 由 Meta AI 开源，它摄取 torch.profiler 从每块 GPU/rank 生成的 JSON 追踪，并呈现一条统一的时间线。

借助 HTA，你可以（例如）看到全部 8 块 GPU 的追踪按时间对齐。来自每个 rank 的 NVTX 标记都被对齐并可见。这样，你可能会注意到 rank 0 在时刻 T 进入“反向”传播，而 rank 1 直到 T+1 才进入——也许是因为 rank 1 在一次 all-reduce 中等待 rank 0。或者你可能看到 rank 0 有一段空隙，而其间 rank 1–7 正忙于计算——如果 rank 0 提前完成并空闲等待，这或许表明存在负载不均衡。

HTA 还会提供一份 GPU 空闲时间的报告——甚至通过重叠来给出提升效率的建议。对于分布式训练，HTA 在定位掉队者（straggler）和同步问题方面极为有用。

> 过去，人们尝试通过手动合并追踪来使用 TensorBoard 的追踪查看器。但截至 2025 年，用于追踪可视化的 PyTorch TensorBoard 剖析器插件已被弃用。取而代之，请使用 Perfetto 的 Trace Viewer 查看时间线，使用 Meta 的 Holistic Trace Analysis 进行多节点聚合。

使用 HTA 通常涉及借助 torch.profiler.schedule 之类的机制从每个 rank 生成追踪，在每块 GPU 上记录若干次迭代的追踪，然后将结果保存到共享位置。将这些追踪加载到 HTA 工具后，你可以看到每个线程的时间线、操作重叠，甚至每个 rank 的内存使用。

例如，你可以用 HTA 来确认你的 all-reduce 优化是否奏效。在这种情况下，优化前的追踪会显示出清晰的顺序模式：反向传播计算结束后，出现一段等待 all-reduce 完成的空隙，然后另一段计算才开始。在增大桶大小并实现重叠之后，HTA 应当显示出更小的空隙，因为现在大部分通信都与剩余的反向传播计算并发进行。

简而言之，HTA 是为 PyTorch 多 GPU 剖析而设计的。建议在跨多块 GPU 和多个计算节点的分布式 PyTorch 环境中，使用 HTA 对训练循环行为进行深入分析。

## 持续集成与性能基准测试

在应用了所有优化之后，持续维持这些优化并不断检查性能回归至关重要。随着代码和配置的演进（例如新的 PyTorch 版本、新的模型特性、代码重构等），如果不加以跟踪，性能很容易发生回退。

你应当使用 TorchBench 和 GitHub Actions 搭建一套简单的性能回归持续集成（continuous integration，CI），以自动捕捉性能下降。TorchBench 是一套开源的 PyTorch 模型基准测试。TorchBench 还包含以 torch.compile 运行的流行模型和基准，以便同时跟踪编译器性能。

你也可以用你自己的模型——或一个更小的代理模型——来扩展 TorchBench。例如，你可以在本地 fork TorchBench，加入你的模型，并将 TorchBench 作为构建和 CI 工作流的一部分来运行。这样，你就拥有了一个一等的性能回归任务，它会持续运行——要么按计划运行，要么在每次代码提交时运行。

性能回归任务会加载你的模型——或一个更小但有代表性的模型变体——运行若干次迭代，并测量吞吐量，例如以 tokens/sec 或 samples/sec 为单位。CI 任务将其与存储的基线数值进行比较，如果吞吐量跌破某个阈值，就让 CI 构建失败。下面是一段基于 TorchBench 定义性能回归任务的 GitHub Action 片段：

```
- name: Run MoE benchmark
  run: |
    torchbench run --model moe --iters 10 --batch-size 4 --json results.json
- name: Compare throughput
  run: |
    python scripts/compare_perf.py baseline.json results.json
```

第一步以 batch size 4 运行我们的 MoE 基准 10 次迭代。这样一次短暂的运行足以在 CI 中衡量性能。不过，为了得到最终数值，你会想测量更长时间的运行。我们把它保持简短，以适应 CI 的时间限制。

这里，输出是一个 JSON 文件，即 --json results.json。第二步运行一个小脚本，将新结果与（例如）存储在 main 分支上的基线 JSON 进行比较。如果新提交比如说慢了 ≥ 5%，我们就会将其标记为回归并让 CI 构建失败。

> 请确保 CI 运行器具有一致的硬件——或使用基于云的预留专用实例以获得可复现的结果。不同类型的 GPU 以及不同的云提供商之间，性能数值可能差异很大。

你还应当编写正确性单元测试，以确保优化后的自定义核函数产生与 PyTorch 等价操作相同的结果。尤其要测试边界情况和随机种子。

一个自定义核函数（custom kernel）可能通过基本测试，却在极端输入上失败——比如非常大的数值，若处理不当就会导致溢出。请使用 PyTorch 的 torch.allclose 以严格的容差对你的优化进行数值精度检查。这样你就能及早发现任何正确性问题。

将内存使用和数据加载时间作为 CI 工作流的一部分记录下来也很有用。例如，你应当在运行期间捕获 torch.cuda.max_memory_allocated()——以及对实际数据加载进行计时。请记住，性能优化是多维度的。一项让计算加速 20% 但让内存使用增加 200% 的改动，在实践中可能带来净负面的效果。

还要记住，软件并不完美。即便是 PyTorch 的更新也可能无意中改变你工作负载的性能画像。例如，较新版本的 PyTorch 可能会改变核函数融合模式或调度启发式。

如果这类更新导致你的模型慢了 5%，你会希望及早发现它、调整你的代码，并作为一个 PyTorch issue 向上游报告。虽然这种情况对于未被纳入 PyTorch 常规性能测试的自定义或较少使用的 LLM 更为常见，但仍是需要留意的一点。

PyTorch 团队通常对性能回归反应非常迅速。一旦收到提醒，他们常常会在 nightly 构建中推送修复。通过及早捕捉回归，你甚至可以自己贡献一个修复——或者至少提供一份翔实的报告。

简而言之，只优化一次是不够的。随着系统演进、新的性能优化被应用、新特性被加入，你需要保护你优化成果的有效性。通过将性能测试集成到 CI 中，任何损害性能的代码改动都会立即显现并得到处理。这会让你确信，你的性能收益不会在下一次代码重构中悄然流失。

> 建议通过在让构建失败之前运行多次迭代来引入一定的方差容忍度。性能可能因外部噪声而略有波动。例如，你可以要求 5% 的回归在 3 次运行中持续出现才让构建失败。PyTorch 自己的持续性能回归系统（将在下一节讨论）就使用统计平滑来避免误报。

除了你自己的 CI 之外，留意 PyTorch 的性能*抬头显示*（heads-up display，HUD）也很有用。这是一个公开的仪表板，它使用 PyTorch 的 CI 系统跟踪常见模型的性能，我们接下来会讨论。

### PyTorch HUD 性能仪表板

在优化过程中，能够洞察更广泛框架层面上性能随时间的变化很有帮助。PyTorch 的开源性能 HUD 提供来自 nightly 基准运行的实时反馈。这个 Web UI 展示了 PyTorch 仓库在多种硬件后端上的构建状态、测试结果和性能指标，这些后端包括 NVIDIA GPU、AMD GPU 和 CPU——以及许多常见模型。

PyTorch 工程师以及社区中的任何人都可以依靠 HUD 来检测回归。如果一个新提交导致某个模型的 tokens/sec 显著下降，HUD 会将其标红作为早期预警。这会促使人们对导致回归的改动进行更深入的调查。

通常 HUD 的阈值在 5% 左右的回归。轻微的噪声波动是正常的，但任何超出该阈值的情况都会触发调查。HUD 还包含趋势线。因此，如果你看到（例如）由于许多小提交累积而导致数周内性能逐渐衰减，这同样会被标记以供调查。

通过导航到 HUD 的 Benchmarks → vLLM 部分，你可以看到各种语言模型的基准结果。该仪表板会随时间跟踪各项指标，例如每个模型和硬件类型的编译时间、内存使用、吞吐量（每秒 token 数）和 FLOPS 利用率，如图 13-7 所示。

例如，你可能会看到在某个日期之后，吞吐量下降而内存使用上升。这表明内存碎片可能有所增加——或者选用了一个效率较低的核函数。HUD 有助于将这类变化直接与 GitHub 提交关联起来。

监控 HUD 以判断上游 PyTorch 的改动是否会影响你的模型很有用。如果你的模型没有直接包含在 HUD 中，一个架构相似的模型或许可以作为代理。此外，HUD 是开源的，因此你可以针对自己的模型、硬件和环境自行仿制它。

![图 13-7. PyTorch HUD 仪表板（来源：https://oreil.ly/JxLKJ）](AI%20Systems%20Performance%20Engineering-ch13_images/figure-13-7.png)

图表上的每个数据点都是通过比较基准提交与 PyTorch main 分支上最新提交之间的性能基准生成的。该仪表板允许选择时间范围（例如最近 7 天、30 天）和粒度（每小时、每天、每周），以便放大或缩小来展示随时间变化的趋势。

HUD 的实现开源于 pytorch/test-infra 仓库。它使用 Python 脚本与 Grafana 仪表板的组合。你可以通过随时间收集自己模型的性能数据并将其可视化，在内部运行一个迷你版 HUD。即便是一个简单的电子表格或 Grafana 实例也能仿制这类可视化。关键在于始终如一地跟踪相同的指标。

在 CI 中添加或修改仪表板、贡献新的 YAML 配置以及更新基准脚本都相当直接。原则上，如果你想在 HUD 中跟踪你的自定义模型，你可以为它添加一个基准并让它在 PyTorch 的 CI 中运行。这样，任何影响你模型性能的 PyTorch 改动都会显示在图表中。

> 强烈建议为你自己的模型搭建一套像 pytorch/test-infra 那样的持续基准测试和性能回归系统，以便针对你的特定工作负载及早捕捉回归。

### 性能基准与 MLPerf Logging

除了临时的基准测试和 CI 之外，像 MLPerf 这样的行业标准基准也能提供宝贵的反馈并鼓励良好实践。MLPerf 的训练和推理基准以严格的日志记录推动最先进的性能，以确保结果可信并经得起同类对比。即便你不参加竞赛，MLPerf 的日志记录方法论对你自己系统的性能分析也很有用。

MLPerf logging 将训练运行期间关键事件和指标的输出标准化。每条日志条目都是一个 JSON 对象，以类似 :::MLL 的前缀打印。条目包含键值对以及元数据。在 PyTorch 中，你可以使用 GitHub 上开源的 MLPerf Logging 库，将你的输出格式化为符合 MLPerf 标准的形式。

在一次 MLPerf 训练运行中，日志包含初始化开始时间、训练开始时间、每个 epoch 的时间、训练结束时间、最终精度等条目。所有内容都精确到毫秒打上时间戳并被正确标注。

MLPerf 合规脚本会解析这些日志，以验证运行是否遵守规则。例如，开始之后不应有超参数改动；你必须使用规定的 epoch 数量，等等。

日志还会包含计算得到的吞吐量——以及运行是否在允许的时间内达到了目标精度。这确保了公平性，因为你无法在不满足精度要求的情况下宣称某个吞吐量数值。

> 在你自己的训练中，你应当始终将性能测量与精度检查配对。这样，你就不会把模型优化到一种不稳定的状态。

一些 MLPerf 日志，尤其是研究类提交，会包含每个 epoch 和迭代计时的细分。这种细节程度对于竞赛并非必需，但在内部非常有用。例如，你可以记录每次迭代中有多少时间花在前向传播、反向传播以及诸如 all-reduce 之类的通信上。你还可以记录多节点集群中所有节点上这些计时的平均值，以定位分布式瓶颈。

下面是一个 MLPerf JSON 日志的小示例片段。紧随其后的是示例表 13-6，它由这些日志推导而来，并包含每个组件相对于整体步骤时间的百分比：

```
{
  "step_time_ms": 24.0,
  "forward_ms": 10.5,
  "backward_ms": 9.0,
  "allreduce_ms": 4.0,
  "other_ms": 0.5
}
```

表 13-6. 每次训练迭代耗时的 MLPerf Logging 细分示例

| 步骤组件 | 每次迭代耗时（ms） | 占步骤的百分比 |
| --- | --- | --- |
| 前向传播 | 10.5 ms | 43.8% |
| 反向传播 | 9.0 ms | 37.5% |
| All-reduce（梯度同步） | 4.0 ms | 16.7% |
| 其他开销 | 0.5 ms | 2.1% |
| 总步骤时间 | 24.0 ms | 100% |

在这个 24 ms 的训练步骤示例中，计算（前向和反向）耗时 19.5 ms（10.5 ms + 9.0 ms），即步骤时间的 81.3%（43.8% + 37.5%）。通信（梯度 all-reduce）耗时 4.0 ms（16.7%），其他开销（例如数据加载）占步骤时间的 0.5 ms（2.1%）。

这样的细分极其有用，因为它告诉我们大约六分之一的时间花在了梯度同步上。如果我们想进一步加速训练，就可以专注于重叠或减少 all-reduce 时间。

例如，我们可以尝试激活压缩，或者尝试用异步 all-reduce、流水线并行等技术更好地将通信与计算重叠，以减少花在 all-reduce 上的 4 ms。如果“其他开销”足够大，我们就会针对数据加载下手，而那需要用不同的方式来处理。

MLPerf 针对 all-reduce 的优化包括延迟 all-reduce（称为 *slack*）或将多个更小的 all-reduce 与计算相重叠等技术。这些是超出本次讨论范围的高级技巧，但重点在于，这类细分能把你精确地引导到需要优化的地方。

虽然 MLPerf Logging 是针对竞赛规则的，但结构化日志（structured logging）、性能指标和计时细分这一通用做法可以应用到你自己的训练与推理仿真中。例如，你可以对训练循环进行埋点，在每个 epoch 记录一行 JSON，附带诸如吞吐量、延迟、GPU 利用率（来自 nvidia-smi）等额外指标。

在一次长时间的训练运行中，这些日志会成为训练后分析的宝库。你可以绘制性能从训练第 1 天到第 7 天的变化，以判断作业是否因内存碎片而变慢。或者你可以看到不同阶段（包括数据加载、计算等）如何扩展。

通过记录各项指标而不仅仅是最终精度，你让自己的结果可复现、可调试。如果之后有人重新训练你的模型却发现更慢了，这些日志将有助于定位问题。也许是数据输入层变慢了。又或许他们使用了不同的硬件配置。这一做法与 CI 相辅相成，后者鼓励一种记录、监控、比较的方法论。

通过对流水线进行埋点以输出各种计时组件的 JSON 日志，你可以在实施优化的过程中跟踪改进。这也让你能以标准化的方式，更容易地与团队沟通瓶颈——以及性能调优的结果。

> 由于 MLPerf 包含跨多块 GPU 和多个计算节点的大规模模型基准，你可以研究 MLPerf 的提交，以洞悉流行 LLM 和集群配置的最佳实践。本书中讨论的许多优化都被获胜的 MLPerf 提交所采用。这是持续获取性能技巧与窍门——以及大规模下最优集群拓扑——的绝佳来源。

## 关键要点

PyTorch 相对的简洁性和高度的抽象有时会带来一种虚假的性能安全感。因此，在开发过程中出人意料地容易引入微妙的性能 bug。下面总结了常见的 PyTorch 性能陷阱——以及如何应对它们：

*坚持剖析优先的方法* 在超大规模下，瓶颈可能潜藏在任何一层——Python 开销、PyTorch 框架调度、CPU 数据加载停顿、GPU 核函数低效、内存问题等。仅凭直觉往往会错过真正的热点。使用结合多种工具的整体剖析策略（正如我们在本章所做的那样），在每个层面捕获性能。现代剖析器具有低开销模式，可在生产环境中用于捕捉回归。将这些与来自 nvidia-smi 的 GPU SM 利用率等硬件指标相结合，你就能有把握地识别瓶颈并正确地排定优化优先级——而不是在错误的地方做优化。

*优先使用编译模式而非即时执行模式* 在即时执行模式下，每个微小的操作都作为它自己的核函数启动。这每次都会带来 Python 分派和 GPU 启动开销。相反，应使用 PyTorch 通过 torch.compile 实现的 JIT 编译。基本上只需一行改动（model = torch.compile(model)），PyTorch 就能捕获模型图并生成融合、优化的代码。

*使用你的工作负载所允许的最高优化编译器模式* 对于长时间运行的作业，max-autotune 往往在稳态速度上取胜，但对于小批次或动态形状，reduce-overhead 可能更好。请在你的工作负载上验证各种模式。max-autotune 中的 CUDA Graphs 可以掩盖启动开销，但与频繁变化的形状不兼容。

*保存已编译的产物以便复用* 如果启动时间是个顾虑，最好缓存已编译的产物以供日后复用。为此，你可以使用 torch.compiler.save_cache_artifacts() 和 load_cache_artifacts()。对于在多节点机群上长时间运行的作业，建议将编译器产物作为“超级缓存”（mega-cache）持久化到一个共享路径（例如 TORCHINDUCTOR_CACHE_DIR 环境变量），并在各节点上挂载到相同的位置。这有助于在启动新节点时避免冷启动。

*避免同步陷阱* PyTorch 是为易用性而设计的，这意味着很容易在无意中写出强制 CPU 与 GPU 之间同步的代码。例如，对一个 CUDA 张量调用 tensor.item() 以取回一个 Python 值会同步 GPU。在协调多个流之间时，应使用带流事件的 torch.cuda.Stream.wait_stream()，而不是强制同步。类似地，在不使用 non_blocking=True 的情况下将数据从 GPU 传输到 CPU 会导致一次同步。请使用异步传输，并让剖析器引导你发现任何隐藏的同步。

*避免用* time.time() *做 Python 侧的剖析，因为这会隐式同步* 用 time.time() 为 GPU 代码块计时会包含一次同步。最好使用 torch.cuda.Event(enable_timing=True) 为 GPU 代码计时，避免多余的同步。

*充分利用 Tensor Core* 出人意料地容易——而且并不理想——的是在不知不觉中回退到完整的 FP32 而不使用 Tensor Core。为确保你在使用 Tensor Core，请将前向传播和损失计算包裹在 torch.autocast 中，并选择较低精度的 dtype，以便 GEMM 能够使用 Tensor Core。（注意，autocast 不会改变模型权重的存储 dtype，除非你显式地转换模型。相反，它为符合条件的操作选择计算用的 dtype，而把数值敏感的操作保留为 float32。）

*用 TF32 替代 FP32 以启用 Tensor Core* 对于能够容忍 TF32 精度的 FP32 工作负载，使用 torch.set_float32_matmul_precision() 并传入 "high" 或 "medium" 来启用 TF32。在 CUDA 设备上，该设置底层会映射到 torch.backends.cuda.matmul.fp32_precision。在现代 GPU 上，TF32 会使用 Tensor Core，这正是我们所期望的。务必用 Nsight Compute 的 SpeedOfLight 视图或 sm__inst_executed_pipe_tensor 指标来确认你的核函数确实使用了 Tensor Core。（注：使用 "highest" 会强制启用真正的 FP32 并禁用 TF32，因此要小心。这在调试时可能有用，但如果你的目标是提升性能，则不建议使用。）

*确认 Tensor Core 确实被启用* 要确认核函数使用了 Tensor Core 以及预期的数据类型，你应当借助剖析器。在 PyTorch 中，先用 torch.profiler 检查算子耗时，再用 Nsight Compute 确认 Tensor 流水线的活动情况。Nsight Compute 的 SpeedOfLight 与 Compute Workload Analysis 小节会通过性能监视器（performance monitor，PM）采样报告 Tensor 流水线利用率以及核函数执行的时间线。

*尽可能优先选择 BF16 而非 FP16* 在现代 GPU 上优先使用 dtype=torch.bfloat16，因为 BF16 保留了 FP32 的指数范围，通常不需要梯度缩放。如果你必须以 FP16 训练，请启用 torch.cuda.amp.GradScaler，并在日志中确认没有出现上溢/下溢。在现代 GPU 的 Tensor Core 上，BF16 一般无需缩放。

*融合小算子* 许多模型计算都包含由小型逐元素算子或矩阵算子构成的长序列。这些小算子各自都会带来额外的内存读写与启动开销。用 torch.profiler 找出连续出现的大量小算子（例如 < 1 ms 的算子）。如果你看到类似 linear *→* gelu *→* dropout 的模式，可以考虑用一个融合模块来替换它，或依赖 torch.compile 来完成融合。而对于编译器遗漏的算子，可以考虑自己编写自定义融合核函数。每一次自定义融合都能节省几个百分点。这些小的“最后一公里”收益累加起来将相当可观。

*减少内存碎片* 如果你的训练任务或推理流水线要连续运行数天/数周/数月，内存碎片可能成为问题——尤其是在大多数 LLM 应用中，张量大小会逐次迭代变化。可以考虑使用像 PyTorch 缓存分配器这样的内存池化库，它能在一开始就把所有大张量分配好。你拥有的不同分配尺寸越少越好。因此在可能的情况下要复用并坚持使用固定形状。通过环境变量（例如 max_split_size_mb）调优 CUDA 分配器，主动管理内存。同时避免把大块内存拆成许多小块，为常用缓冲区预先按固定的最大尺寸分配，并复用缓冲区以保持分配尺寸一致。这些做法能让内存占用随时间保持稳定，避免与碎片相关的性能下降和崩溃。

*使用激活检查点* 默认情况下，PyTorch 会保存反向传播所需的全部激活值，这对大模型而言会消耗巨量内存。可以考虑使用激活检查点，以计算换内存。torch.utils.checkpoint 让这一点很容易应用：把前向传播的若干片段包裹起来，其激活值就不会被保存，而是在反向传播时重新计算。这会以额外计算为代价降低内存占用。不过，这通常是值得的，因为对现代 GPU 来说，显存比算力更为紧俏。在现代 GPU 硬件上，对于参数量达到数百亿以上的模型，这几乎是必需的。另外要记住，你可以混合精度使用。例如，你可以把少数关键激活保留在较高精度以维持准确度，而以较低精度重算其余激活以节省内存和时间。

*将内存卸载到 CPU 或 NVMe 存储* 模型参数、梯度、优化器状态和激活值都在争抢有限的 GPU 显存。即便拥有大容量 HBM，一个数千亿参数的模型也会将其耗尽——哪怕在降低精度的情况下也是如此。要善用系统的内存层级，把不常用的数据卸载到 CPU 内存或 NVMe 磁盘——只在需要时再取回。这些数据传输可以通过异步拷贝进行流式传输，并与计算重叠。执行此操作时要监控互连吞吐量，确保没有把链路打满。

*减少输入流水线停顿* 即使是完美优化过的训练循环，如果输入流水线无法足够快地喂入数据，也会成为瓶颈。这通常表现为 GPU 在等待下一个批次时出现的空闲气泡。这在剖析中往往很明显，因为大多数工具会显示迭代前的空隙——或显示 CPU 线程忙于数据加载代码而 GPU 却处于闲置状态。使用足够数量的 DataLoader worker 以及合适的 prefetch_factor 大小来最大化数据吞吐量。

充分利用所有 CPU 核心来进行数据加载和准备十分重要。使用 pin_memory=True 和非阻塞传输，以加快 H2D 拷贝。离线预处理你的数据集（例如分词），让数据加载器只做最少的工作。此外，你还可以增大 DataLoader 的 prefetch_factor。建议持续地略微提前预取更多批次，赶在它们被使用之前准备好。

*剖析并在可能时卸载 CPU 端的数据变换* 由于 Python 数据变换（例如分词）可能出人意料地慢，务必让这些主机端变换实现向量化。在可用时考虑使用经过优化的 C++ 库。并且，可以考虑用像 NVIDIA DALI 这样的库把复杂变换卸载到 GPU 上。

*优化多 GPU 与多节点通信* 如果管理不当，分布式训练往往会因通信开销而触及天花板。使用经过优化的 PyTorch 分布式实现，比如 DDP，它默认会将通信与计算重叠。调优梯度的 bucket_cap_mb，以找到最利于重叠的桶大小。如果你的网络能够承受，更大的桶（例如用 50 MB 替代默认的 25 MB）可以降低每条消息的开销并实现更好的重叠。

*监控网络带宽* 如果带宽被打满，你可以探索激活压缩技术来降低带宽占用。务必把 all-reduce 集合通信与反向计算重叠起来，从而隐藏通信时间。对于多节点场景，要考虑拓扑结构，把频繁通信的进程放置在同一节点或同一交换机上，以降低延迟。

*避免随时间推移的“劣化（bit rot）”* 性能很容易因持续的代码改动、框架升级和新增应用功能而回退。例如，一次小的重构可能引入意料之外的同步，或者一次 PyTorch 升级可能改变某个算子的实现。要把性能当作一等指标，建立持续集成测试、仪表盘和告警来监控性能。每一次代码提交（或每日/每周运行）都应包含一次快速的性能基准测试，以捕捉性能回归。

*每当硬件变更时更新你的基线* 举例来说，如果你从 B200 迁移到 GB300，务必重新建立你的性能基线和阈值。新硬件对某些模式的处理会更好或更差。据此重新校准并调整你的告警。

*在自动化流程中使用 TorchBench 之类的工具或自定义计时脚本* 关注来自 PyTorch 每夜性能基准测试的外部信号（发布在公开的 HUD 回归仪表盘上）——尤其是与你相似的模型。如果 PyTorch 中发生了普遍性回归，你会在升级之前就得知。一旦检测到变慢，立即排查，定位根因（例如是你的代码还是上游改动），加以缓解（例如回退或改写代码），并将其上报给上游（例如 PyTorch、Triton、NVIDIA 等），以便相应社区处理该问题，大家都能从修复中受益。同时，为你的优化维护正确性测试，以免悄无声息地破坏准确度。把性能回归测试纳入你的工作流程，你所获得的加速就能经得起时间的考验。

## 结语

将微观层面的剖析（例如逐算子、逐核函数）与宏观层面的基准测试（例如端到端的吞吐量与延迟）结合起来非常重要。这样你才能对系统性能有全面的认识。优化单个核函数固然重要，调优你完整的端到端训练与推理流水线——并让它长期保持在调优状态——同样重要。随着硬件不断演进，这种整体性的方法至关重要。

本章的这些基本功将帮助你适应新平台。始终要先剖析、定位瓶颈，然后在技术栈的各个层级施加恰当的优化。通过系统性地应用编译器优化和内存优化，你可以在仍然保持可接受准确度的同时，将工作负载的性能显著提升到更接近硬件极限的水平。

正如我们所展示的，细致的剖析能够定位真正的瓶颈。而有针对性的优化可以在训练和推理性能上带来巨大的加速。同样重要的是，要建立自动化机制，以保护这些收益不被随时间发生的性能回归所侵蚀。

这段端到端的优化之旅需要在技术栈的每一个层级投入努力。最终得到的是一个经过优化、高效且抗回归的系统：它能从现代 GPU 硬件中榨取最大性能，能扩展到多节点集群和机架级拓扑，并能适应硬件、软件与算法未来的进步。