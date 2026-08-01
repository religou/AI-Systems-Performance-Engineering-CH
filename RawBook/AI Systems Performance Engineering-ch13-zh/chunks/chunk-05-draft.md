在 torch.\_dynamo.explain(model, ...) 的输出中，你会看到与 all-reduce 和 torch.distributed 操作相关的图中断（graph break）。这是符合预期的，因为每个桶（bucket）的计算都会被编译——而 all-reduce 发生在各子图之间。这样一来，通信重叠得以保留，而这对性能至关重要。

如果你将 DDP（分布式数据并行，Distributed Data Parallel）与编译器一起使用，请确保 DDP 的桶大小合理。默认情况下，DDP 的桶为 25 MB。如果你选择覆盖这一默认值，用一个巨大的桶来容纳所有梯度，那么就会生成一个巨大的图，并在末尾进行一次庞大的 all-reduce。这会导致图段更少、更大，几乎没有机会将通信与计算交错进行。

使用单个桶很有诱惑力，例如让整个反向传播在单个图中完成，从而实现最大程度的核函数融合（kernel fusion）。然而，这样你会失去重叠的机会。建议对系统进行剖析，为你的工作负载和硬件环境找到合适的平衡点。

> 在将 DDP 与 torch.compile 一起使用时，你会在通信点看到刻意的图中断。这些是正常的，也是让网络通信得以进行所必需的。TorchDynamo 的 explain() 输出会显示关于 all-reduce 和 scatter 导致中断的消息——这是符合预期的。

### FSDP 与 torch.compile

PyTorch 的 FSDP（完全分片数据并行，Fully Sharded Data Parallel）并行策略会将模型参数、梯度和优化器状态分片（shard）到多块 GPU 上。它在前向传播期间执行一次 all-gather 集合通信，在反向传播期间执行一次 reduce-scatter。相比使用 DDP，这使得更大的模型能够装入 GPU 内存。

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

张量并行（tensor parallelism，TP）和流水线并行（pipeline parallelism，PP）与 torch.compile 是正交的。只要跨 GPU 的通信操作被 TorchDynamo 识别为集合调用，Dynamo 就会相应地对它们进行追踪或中断。

PyTorch 编译器（PyTorch Compiler）主要专注于优化每个片段内部的计算。它（目前）不会融合通信操作，也不会改变它们的调度——除了前面描述的那种自然重叠之外。具体而言，TorchInductor 为计算选择 cublasLt/cuDNN/Triton 核函数，但把 NCCL 集合通信（及其顺序）留给分布式策略处理。在使用任何分布式训练策略时，你都应当分别在启用和不启用 torch.compile 的情况下进行测试，以确保得到预期的结果。启用编译时，务必用剖析器追踪来验证重叠。

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

在 TorchTitan 的实验中，SimpleFSDP 将内存使用最多降低了约 28%。而当它与其他分布式技术组合时，相比传统的 FSDP2 即时执行（eager）路径，训练吞吐量最多提升了约 69%。

> TorchTitan、AsyncTP、AutoParallel 和 SimpleFSDP 都是维护良好的项目，值得关注。它们代表了实用的 PyTorch 参考实现，并纳入了业界众多 PyTorch 专家的许多优化。

### 用 HTA 进行多 GPU 剖析

当你扩展到数百万块 GPU 时，处理数百万份独立的追踪和剖析结果会变得难以驾驭。Meta 的 HTA 有助于合并并可视化多工作节点的追踪。HTA 由 Meta AI 开源，它摄取 torch.profiler 从每块 GPU/rank 生成的 JSON 追踪，并呈现一条统一的时间线。

借助 HTA，你可以（例如）看到全部 8 块 GPU 的追踪按时间对齐。来自每个 rank 的 NVTX 标记（NVTX marker）都被对齐并可见。这样，你可能会注意到 rank 0 在时刻 T 进入“反向”传播，而 rank 1 直到 T+1 才进入——也许是因为 rank 1 在一次 all-reduce 中等待 rank 0。或者你可能看到 rank 0 有一段空隙，而其间 rank 1–7 正忙于计算——如果 rank 0 提前完成并空闲等待，这或许表明存在负载不均衡。

HTA 还会提供一份 GPU 空闲时间的报告——甚至通过重叠来给出提升效率的建议。对于分布式训练，HTA 在定位掉队者（straggler）和同步问题方面极为有用。

> 过去，人们尝试通过手动合并追踪来使用 TensorBoard 的追踪查看器。但截至 2025 年，用于追踪可视化的 PyTorch TensorBoard 剖析器插件已被弃用。取而代之，请使用 Perfetto 的 Trace Viewer 查看时间线，使用 Meta 的 Holistic Trace Analysis（HTA）进行多节点聚合。

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

例如，你可能会看到在某个日期之后，吞吐量下降而内存使用上升。这表明内存碎片（memory fragmentation）可能有所增加——或者选用了一个效率较低的核函数。HUD 有助于将这类变化直接与 GitHub 提交关联起来。

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

| 步骤组件               | 每次迭代耗时（ms） | 占步骤的百分比 |
| ---------------------- | ------------------ | -------------- |
| 前向传播               | 10.5 ms            | 43.8%          |
| 反向传播               | 9.0 ms             | 37.5%          |
| All-reduce（梯度同步） | 4.0 ms             | 16.7%          |
| 其他开销               | 0.5 ms             | 2.1%           |
| 总步骤时间             | 24.0 ms            | 100%           |

在这个 24 ms 的训练步骤示例中，计算（前向和反向）耗时 19.5 ms（10.5 ms + 9.0 ms），即步骤时间的 81.3%（43.8% + 37.5%）。通信（梯度 all-reduce）耗时 4.0 ms（16.7%），其他开销（例如数据加载）占步骤时间的 0.5 ms（2.1%）。

这样的细分极其有用，因为它告诉我们大约六分之一的时间花在了梯度同步（gradient synchronization）上。如果我们想进一步加速训练，就可以专注于重叠或减少 all-reduce 时间。

例如，我们可以尝试激活压缩，或者尝试用异步 all-reduce、流水线并行等技术更好地将通信与计算重叠，以减少花在 all-reduce 上的 4 ms。如果“其他开销”足够大，我们就会针对数据加载下手，而那需要用不同的方式来处理。

MLPerf 针对 all-reduce 的优化包括延迟 all-reduce（称为 _slack_）或将多个更小的 all-reduce 与计算相重叠等技术。这些是超出本次讨论范围的高级技巧，但重点在于，这类细分能把你精确地引导到需要优化的地方。

虽然 MLPerf Logging 是针对竞赛规则的，但结构化日志（structured logging）、性能指标和计时细分这一通用做法可以应用到你自己的训练与推理仿真中。例如，你可以对训练循环进行埋点，在每个 epoch 记录一行 JSON，附带诸如吞吐量、延迟、GPU 利用率（来自 nvidia-smi）等额外指标。

在一次长时间的训练运行中，这些日志会成为训练后分析的宝库。你可以绘制性能从训练第 1 天到第 7 天的变化，以判断作业是否因内存碎片而变慢。或者你可以看到不同阶段（包括数据加载、计算等）如何扩展。

通过记录各项指标而不仅仅是最终精度，你让自己的结果可复现、可调试。如果之后有人重新训练你的模型却发现更慢了，这些日志将有助于定位问题。也许是数据输入层变慢了。又或许他们使用了不同的硬件配置。这一做法与 CI 相辅相成，后者鼓励一种记录、监控、比较的方法论。

通过对流水线进行埋点以输出各种计时组件的 JSON 日志，你可以在实施优化的过程中跟踪改进。这也让你能以标准化的方式，更容易地与团队沟通瓶颈——以及性能调优的结果。

> 由于 MLPerf 包含跨多块 GPU 和多个计算节点的大规模模型基准，你可以研究 MLPerf 的提交，以洞悉流行 LLM 和集群配置的最佳实践。本书中讨论的许多优化都被获胜的 MLPerf 提交所采用。这是持续获取性能技巧与窍门——以及大规模下最优集群拓扑——的绝佳来源。

## 关键要点

PyTorch 相对的简洁性和高度的抽象有时会带来一种虚假的性能安全感。因此，在开发过程中出人意料地容易引入微妙的性能 bug。下面总结了常见的 PyTorch 性能陷阱——以及如何应对它们：

_坚持剖析优先的方法_ 在超大规模下，瓶颈可能潜藏在任何一层——Python 开销、PyTorch 框架调度、CPU 数据加载停顿、GPU 核函数低效、内存问题等。仅凭直觉往往会错过真正的热点。使用结合多种工具的整体剖析策略（正如我们在本章所做的那样），在每个层面捕获性能。现代剖析器具有低开销模式，可在生产环境中用于捕捉回归。将这些与来自 nvidia-smi 的 GPU SM 利用率等硬件指标相结合，你就能有把握地识别瓶颈并正确地排定优化优先级——而不是在错误的地方做优化。

_优先使用编译模式而非即时执行模式_ 在即时执行（eager）模式下，每个微小的操作都作为它自己的核函数启动。这每次都会带来 Python 分派和 GPU 启动开销。相反，应使用 PyTorch 通过 torch.compile 实现的 JIT 编译。基本上只需一行改动（model = torch.compile(model)），PyTorch 就能捕获模型图并生成融合、优化的代码。

_使用你的工作负载所允许的最高优化编译器模式_ 对于长时间运行的作业，max-autotune 往往在稳态速度上取胜，但对于小批次或动态形状，reduce-overhead 可能更好。请在你的工作负载上验证各种模式。max-autotune 中的 CUDA Graphs 可以掩盖启动开销，但与频繁变化的形状不兼容。

_保存已编译的产物以便复用_ 如果启动时间是个顾虑，最好缓存已编译的产物以供日后复用。为此，你可以使用 torch.compiler.save_cache_artifacts() 和 load_cache_artifacts()。对于在多节点机群上长时间运行的作业，建议将编译器产物作为“超级缓存”（mega-cache）持久化到一个共享路径（例如 TORCHINDUCTOR_CACHE_DIR 环境变量），并在各节点上挂载到相同的位置。这有助于在启动新节点时避免冷启动。

_避免同步陷阱_ PyTorch 是为易用性而设计的，这意味着很容易在无意中写出强制 CPU 与 GPU 之间同步的代码。例如，对一个 CUDA 张量调用 tensor.item() 以取回一个 Python 值会同步 GPU。在协调多个流之间时，应使用带流事件的 torch.cuda.Stream.wait_stream()，而不是强制同步。类似地，在不使用 non_blocking=True 的情况下将数据从 GPU 传输到 CPU 会导致一次同步。请使用异步传输，并让剖析器引导你发现任何隐藏的同步。

_避免用_ time.time() _做 Python 侧的剖析，因为这会隐式同步_ 用 time.time() 为 GPU 代码块计时会包含一次同步。最好使用 torch.cuda.Event(enable_timing=True) 为 GPU 代码计时，避免多余的同步。

_充分利用 Tensor Core_ 出人意料地容易——而且并不理想——的是在不知不觉中回退到完整的 FP32 而不使用 Tensor Core。为确保你在使用 Tensor Core，请将前向传播和损失计算包裹在 torch.autocast 中，并选择较低精度的 dtype，以便 GEMM 能够使用 Tensor Core。（注意，autocast 不会改变模型权重的存储 dtype，除非你显式地转换模型。相反，它为符合条件的操作选择计算用的 dtype，而把数值敏感的操作保留为 float32。）
