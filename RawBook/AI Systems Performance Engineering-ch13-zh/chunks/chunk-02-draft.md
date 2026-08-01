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

## PyTorch 编译器（PyTorch Compiler）

在 PyTorch 中，最快见效的优化之一是使用 PyTorch 编译器（PyTorch Compiler）配合 torch.compile()。该编译器栈包含 TorchDynamo、AOT Autograd 与 TorchInductor，它们负责捕获计算图、融合算子，并为目标后端（如 NVIDIA GPU）生成高性能代码。

PyTorch 编译器通过把许多小操作融合成更大的核函数，能够消除大量 Python 解释器开销与 GPU 核函数启动延迟。在完成基线性能剖析后，我们在模型上启用了 torch.compile，看看能否轻松获得提速。下面就来介绍这个过程及其结果。

### 使用 PyTorch 编译器

使用默认设置的 PyTorch 编译器非常简单，除了像这样包装模型外无需改动任何代码：model = torch.compile(model)。在底层，TorchDynamo 会追踪 Python 代码，AOT Autograd 会捕获反向传播，而 TorchInductor（它借助 OpenAI 的 Triton 生成 GPU 核函数代码，下一章会讨论）则自动产出高效的融合核函数。

编译器观察模型的前向传播，识别出许多可融合连续操作的机会，例如逐元素激活、层归一化等。它会为这些操作生成融合核函数——也会为部分反向传播生成。其结果是每次迭代的核函数启动次数显著减少、CPU 开销降低。

编译这一步确实会引入一些开销，量级在数秒——对超大模型甚至可达数分钟——但这一成本会在长时间训练任务或反复推理运行中被摊薄。所幸 TorchInductor 会缓存已编译的核函数，因此后续运行无需再次支付编译成本。PyTorch 社区也在持续改进编译/启动性能，允许你跨运行保存并复用已编译产物。使用 torch.compiler.save_cache_artifacts() 与 torch.compiler.load_cache_artifacts() 可将 TorchInductor 的输出在多次运行或多个节点间持久化。这能减少长时间训练或服务时的启动耗时。

一个例子是 PyTorch Mega-Cache 特性。它是一种端到端的编译缓存，可将已编译的核函数保存到磁盘，并在未来运行中重新加载。借助 PyTorch Mega-Cache，你可以只编译一次（例如离线编译），并在多次训练会话间复用优化后的核函数，从而缩短启动时间。你仍能享受 TorchInductor 的核函数优化（如 warp 专门化，warp specialization），同时避免每次都重新编译计算图。

> 你甚至可以在其他计算节点上使用这份编译缓存。若要这么做，请确保各节点间的 CUDA、PyTorch 与 Triton 版本相互兼容。

值得一提的是，PyTorch 编译器在内部运用了精巧的优化技术。例如，我们在第 10 章提到过 warp 专门化。TorchInductor 的自动调优器会跨不同的 tile 尺寸、内存访问模式等生成多种核函数变体，并在幕后运用诸如访存 warp 与计算 warp 专门化（memory-warp versus compute-warp specialization）之类的技术，然后自动为你的硬件挑选最快的变体。

TorchInductor 支持在 GEMM（通用矩阵乘法）核函数前后进行前导（prologue）与收尾（epilogue）融合。例如，bias-add 在 matmul 之前；而在 matmul 之后，收尾部分由激活、dropout、残差等逐元素操作构成。

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

| 模式                       | 说明                                                                                                        | 编译时间     | 额外内存         | 主要特性                                          |
| -------------------------- | ----------------------------------------------------------------------------------------------------------- | ------------ | ---------------- | ------------------------------------------------- |
| default                    | 均衡优化（速度不错，且无需长编译时间或额外内存）；包含少量自动调优；对稳定段可能使用 CUDA Graphs            | 低–中        | 无               | 通用融合、基础自动调优                            |
| reduce-overhead            | 降低每次迭代的开销（适合小批次）；非常适合推理或小批次；若检测到动态形状会自动跳过 CUDA Graphs 以保证正确性 | 中           | 是（工作区缓存） | 尽可能使用 CUDA Graphs 以消除启动开销             |
| max-autotune               | 最大化运行时性能（最适合长时间运行）；编译时间较长；最适合面向大量 SM 与 GPU 内存做激进调优                 | 高（编译慢） | 可能（若使用图） | 激进的 Triton 自动调优；在 GPU 上启用 CUDA Graphs |
| max-autotune-no-cudagraphs | 完成 max-autotune 的一切，但不做 CUDA Graph 捕获；最适合动态形状，或调试被 CUDA Graphs 掩盖的问题           | 高           | 无               | 与 max-autotune 相同，但禁用图以保持灵活性        |

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

使用 torch.compile 时，了解如何调试编译器无法优化模型某部分的情形很有用——例如，某些操作没有被融合，而你怀疑是某个 “图中断”（graph break）导致回退到即时执行。PyTorch 提供了检视这些情形的工具。

> 现代 PyTorch 版本使用形状保护条件（shape guard）实现了对动态形状的部分支持。这些保护条件可以消除一些不必要的图中断。然而，真正动态的工作负载可能仍需回退到即时执行（或使用 max-autotune-no-cudagraphs）以确保正确性。

torch.\_dynamo.explain(model) 会打印一份报告，列出所有图中断（例如未被 TorchDynamo 捕获的模型部分）、图中断发生的原因，以及模型中哪些部分未被 TorchDynamo 捕获。它还会列出未被 TorchDynamo 捕获、需要在较慢的 Python 即时模式下执行的操作或依赖数据的控制流。

图中断的一个常见原因是模型中存在不受支持的操作。Dynamo 的 explain() 输出会就如何获取更多细节给出建议，帮助诊断问题。利用这些提示有助于精确定位导致中断的具体操作或控制流。

另一个有用的技巧是在运行脚本前设置环境变量 TORCH_LOGS="+dynamo" 或 TORCH_LOGS="+dynamo,+inductor"。前缀 + 会为 torch.compile 流水线中的 TorchDynamo、TorchInductor 等组件启用详细（DEBUG 级别）日志。这些详细日志包含关于图中断、回退到即时模式以及编译各阶段的细节。如果模型在使用 torch.compile 时意外地慢，这些日志有助于识别执行在何时、何处退出了编译图。

如果模型具有编译时无法确定的真正动态形状或动态控制流，你可能需要引导编译器。例如，你可以把模型拆分成可编译的若干部分，而让真正动态的部分在 Python 中运行。

要对 TorchInductor 生成的核函数进行剖析和基准测试，你可以指定环境变量 TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 和 TORCHINDUCTOR_BENCHMARK_KERNEL=1。设置这些变量后，Inductor 会为生成的核函数模块生成基准测试脚手架代码。该脚手架代码生成的日志有助于精确定位意外的图中断和性能问题。

你还可以用 torch.\_dynamo.mark_dynamic(tensor, dim) 标记部分代码，让编译器知道要预期动态形状。这可以消除因形状不匹配而产生的不必要图中断。我们将在下一章深入 PyTorch 编译器时更详细地介绍这些技术。

简而言之，当 torch.compile 未能产生预期的提速时，你可以使用 torch.\_dynamo.explain()——配合编译器日志——来识别是哪些操作或代码区域导致了回退。据此，你需要采取一些变通办法，例如替换某个操作、以不同方式重塑张量、接受较少的动态行为，或干脆对模型的那一特定部分禁用编译。其结果是：你为模型的大部分保留了性能收益，同时仍能处理边界情况。

## PyTorch 优化的注意力机制

Transformer 模型在其注意力（attention）机制上耗费大量时间。你可以运用若干 PyTorch 注意力优化技术，确保它不会成为瓶颈。下面简要总结其中几种技术及其适用场景：

_缩放点积注意力（Scaled Dot-Product Attention，SDPA）_ PyTorch 的高层 API torch.nn.functional.scaled_dot_product_attention（即 SDPA）会针对给定硬件自动使用可用的最快注意力核函数（例如 FlashAttention）。当你模型的注意力模式和 dtype 被所选后端（Flash、memory-efficient 或 math）支持时，用它可以毫不费力地获得提速。如果不受支持，它会回退到标准注意力实现。

_FlexAttention_ 一种基于编译器、用于注意力中自定义稀疏模式（sparsity pattern）的方法。通过为特定稀疏注意力模式（例如块稀疏或滑动窗口注意力）生成优化核函数，FlexAttention 可以显著更快，如图 13-3 所示。对于 scaled_dot_product_attention 不支持的特殊情形，请使用 FlexAttention。

![图 13-3. FlexAttention 为自定义注意力变体（attention variant）提供支持。](AI%20Systems%20Performance%20Engineering-ch13_images/figure-13-3.png)

_FlexDecoding_ 它是 FlexAttention 的对应物，用于优化解码或文本生成阶段。FlexDecoding 与 torch.compile 及动态缓存布局集成。它对序列生成的解码器一侧运用编译时优化，包括跨时间步高效地进行 KV 缓存。FlexDecoding 通过减少解码期间的冗余计算，能够加速自回归生成。FlexDecoding 面向 LLM 推理工作负载，包括那些具有长生成序列的场景。它不改变训练时的注意力语义。

_上下文并行（context parallel）_ 上下文并行沿序列长度维度，把注意力在参与的设备或 rank 之间分片，以扩展上下文长度。使用 context_parallel() API 可以限定用上下文并行感知的核函数替换 scaled_dot_product_attention 的范围。该机制按序列在各 rank 之间切分 query-key-value（QKV），并在注意力计算期间进行同步，而不是在单个 GPU 内的线程之间并行化注意力。

## PyTorch 架构优化（PyTorch Architecture Optimization，torchao）、量化、稀疏化与剪枝

PyTorch Architecture Optimization（torchao）把量化（quantization）、稀疏化（sparsity）、剪枝（pruning）以及相关的数值调试工具汇聚到单一命名空间中。它的量化子包（torchao.quantization）提供了常见的 FX-graph-mode 工作流，包括训练后量化（post-training quantization，PTQ）、量化感知训练（quantization-aware training，QAT），以及用于将模型转换并优化为 INT8、FP8 及新兴格式的 QConfigMapping API。

除量化外，torchao 还支持剪枝（torchao.pruning）以及诸如 2:4 稀疏和块稀疏（torchao.sparsity）之类的稀疏化技术。它们能在精度损失极小的情况下带来显著提速。

torch.compile() 与 torchao 量化框架集成。在底层，TorchDynamo 把每个子模块的计算捕获为优化图，随后 TorchInductor 发出利用 torchao 的硬件感知核函数。这为模型训练和推理都带来一致的端到端性能提升。与此同时，它保留了对数值格式与内存布局的精确控制。这使它成为一个适合量化等生产级性能优化的出色库。

## 用 CUDA 流实现并发

正如前面章节所述，CUDA 流（CUDA stream）可在 GPU 上实现操作的并发（concurrency）与重叠。默认情况下，PyTorch 会把所有操作顺序调度到设备的默认流（stream 0）上。然而，许多任务是相互独立的；在资源允许时，GPU 可以用多个流并行执行它们。例如，GPU 可以通过使用独立的非默认流，把数据传输与计算重叠——或并发运行不同的神经网络分支。

> 请记住，现代 GPU 拥有多个 DMA 拷贝引擎。为 H2D 拷贝使用独立的流，可以在不阻塞计算的情况下实现真正并行的数据传输。这种硬件支持让流并发更加有效。

在 PyTorch 中，你用 torch.cuda.Stream() 创建一个流。然后可以通过 Python 上下文管理器 with torch.cuda.stream(stream) 在该流上启动工作，或显式地把操作指派到该流。PyTorch 会以 FIFO 顺序把操作（如内存传输、CUDA 核函数等）下发到指定的流——就像它在默认流上所做的那样。

### 通信与计算重叠

CUDA 流的一个常见用法是把主机到设备（host-to-device，H2D）的数据加载与 GPU 计算重叠。这有助于掩盖使用 GPU 这类外部设备时——相对于在主机上运行的 CPU——所产生的数据传输延迟。

例如，当默认流正忙于在当前批次上训练时，另一个流可以把下一批输入数据从 CPU 拷贝到 GPU 内存。等到默认流准备好处理下一批时，数据传输早已完成，GPU 便可直接处理这下一批。这有效地隐藏了 I/O 延迟。下面是一个在 PyTorch 中使用两个流——compute_stream（默认）与 transfer_stream（非默认）——来重叠数据传输与计算的示例：
