> 现代 LLM 推理引擎会为每个生成的 token 执行这类缓冲区复用。这与每次都释放并重新分配内存的做法形成对比，后者会带来大量开销——尤其是在 token 粒度上。

分配器切换可能会为系统的不同部分使用完全不同的分配器。例如，你可以对模型权重使用带大块静态分配的默认缓存分配器（caching allocator），因为在推理期间这些内存很少被释放。然后，你可以对每个 token 的临时分配使用自定义分配器。

> 建议将你的代码架构为把长生命周期分配（如模型权重）与短生命周期分配（如 token 缓冲区）分离开。这样更容易将短生命周期与长生命周期的分配分别导向不同的分配器和内存池（memory pool）。

你应当始终为 OOM 错误实现一套回退策略。许多生产系统采用多层内存，使得当 GPU 出现 OOM 错误时，它会卸载到 CPU 并重试，而不是让请求失败。例如，你可以动态释放缓存、把部分层卸载到 CPU，或压缩缓存。

简而言之，像动态分配器管理与多层内存这样的技术，能确保你的系统在不同类型的负载下都能应对长时间的运行——而且不会引发内存碎片化（fragmentation）或分配延迟尖峰。这是一种用户不会直接看到的幕后优化，但它对于长期运行的推理服务器实现超大规模鲁棒性至关重要。

## 运行时核函数性能改进与可热插拔实现

在飞速演进的 GPU 硬件与算法创新世界里，更新、更快的核函数（kernel）实现不断涌现。这包括 FlashAttention 的更新变体、megakernel，以及特定于硬件的软件优化。

运行时核函数打补丁（runtime kernel patching）是指能够将这些新实现集成进正在运行的系统，而无需完整重新部署或重新加载模型。本质上，我们希望把一个较慢的核函数即时热插拔（hot swap）成一个更快的。

设想你的推理服务器使用默认的 PyTorch scaled dot product attention（SDPA）核函数来做多头注意力。随后你发现了一个新的核函数实现，比如 FlashAttention-3，它在某些情况下对长序列能带来 20% 的加速。

传统上，你必须安装更新后的库并重启服务器才能使用新实现。但借助运行时打补丁，你可以在运行时动态加载并把调用重定向到新核函数，而不中断服务器的运行。

这种零停机（zero-downtime）升级方式对于 24/7 服务至关重要，在这类服务中，一次重启或模型重新加载会带来过高的延迟或造成宕机。在 Python 中，这可以是一次简单的猴子补丁（monkey patch），如下所示：

```
import new_flash_attn_lib
# 对模型的注意力 forward 打猴子补丁，改用新库
old_attn_forward = model.transformer.self_attn.forward
def new_attn_forward(self, *args, **kwargs):
    return new_flash_attn_lib.forward(*args, **kwargs)
model.transformer.self_attn.forward =
new_attn_forward.__get__(
model.transformer.self_attn,
type(model.transformer.self_attn))
```

这段代码用一个会调用我们的 new_flash_attn_lib.forward 的方法替换了注意力模块的 forward 方法。我们把它绑定到实例（**get**）以模拟一个正规的方法。打完这个补丁后，对该注意力层的后续调用都会走新实现。这样，我们就有效地热插拔了这个核函数。

> 务必确保 new_flash_attn_lib.forward() 是可直接替换的（drop-in），并且已经经过充分测试，能在可接受的数值容差内产生完全一致的输出。如此，你才能避免任何模型质量的退化。

另一种技术是 JIT 打补丁，它使用像 PyTorch Inductor 或 OpenAI Triton 这样的即时编译器在运行时生成更快的核函数——然后将其插入推理流水线。如下所示，可以用 PyTorch 的 torch.compile 来优化一个函数并返回其编译后的版本：

```
compiled_forward =
    torch.compile(model.transformer.self_attn.forward,
                  backend="inductor")
model.transformer.self_attn.forward = compiled_forward
```

现在，每当 forward 被调用时，它都会执行优化后的代码，这会融合多个算子等。如果我们在模型初始化之后做这件事，这就是一种热补丁（hot patching），因为它不需要重新加载模型。它只是替换掉了函数指针。

在 CUDA 一侧，运行时模块加载可以通过 CUDA 驱动 API 实现。例如，你可以为一个新核函数编译自定义 PTX——并在不重置设备的情况下把它加载进上下文。如果函数签名匹配，系统只需更新函数指针即可。像 NVIDIA Runtime Compilation（NVRTC）这样的工具可以在运行时从字符串编译 PTX，从而启用这一工作流。

用 CUDA C++ 来做这件事略显复杂。像 Python 的猴子补丁机制——或 PyTorch 的可扩展后端——这类更高层的方法，从长远看很可能更实用、也更易维护。不过，你的高性能推理引擎很可能并不使用 Python/PyTorch 运行时。

> 在做这类热插拔时，保证线程安全很重要。例如，你应当排空进入的请求队列，或使用一个屏障（barrier）来确保没有其他线程正处在你要打补丁的函数中途。这样可以避免与正在运行的线程发生竞态条件。这类似于操作系统内核加载模块的方式——需要小心的同步，但它避免了完整的重启。

使用界定良好的模块边界——比如把注意力核函数和 MLP 核函数分开——可以让替换内部实现更容易。这也主张以高度模块化的方式编写你的模型代码，以便拥有这些可替换点。单体式（monolithic）模型实现要打补丁困难得多，但它们可能更高性能（例如 megakernel）。

当然，必须确保新核函数产生的结果与旧实现一致——或足够接近。通常这些优化过的核函数被设计为在相同精度（例如 FP16）下数值等价。

热补丁也可用于快速修复。也许某个特定的算子序列正在当前核函数中触发一个已知 bug。与其等待一次完整的更新，你可以快速打入一个修复。

设想你部署了一个 FP8 核函数，它对某些你未曾预料到的边缘输入开始出现异常行为。如果你有一个更安全但稍慢的版本，你可以检测到这种异常状况，并在这些情况下热插拔进那个安全核函数。务必记录这些事件以供日后分析。

> 可以考虑分阶段上线，把一小部分流量路由到新核函数（影子测试或金丝雀发布）。然后你可以用遥测数据把延迟和吞吐量指标与旧核函数做对比。如果新核函数表现符合预期，且其输出与预期结果匹配，你就可以把它发布给所有用户。

为了管理多个可能的实现，引擎可以维护一个注册表，例如 attention_impl = "fast" 或 "safe"——并据此分支。这可以实现为一个功能开关（feature flag），使你无需改动代码就能切换实现。这对于仅通过改动开关来快速回滚很有用（只要确保你的代码正在主动从功能开关系统读取这些值——否则开关值会保持静态，那就失去了功能开关的意义）。

设计良好的推理引擎还应当测量新核函数的运行时性能。如果新核函数在真实生产数据下并不更快，推理引擎可以将其回滚。这可能是由于硬件略有差异、未测试过的 batch size，或未测试过的负载状况。

这正是前一节所述自动调优（autotuning）的一种完美形式。如果自动调优器发现了一个超过某个阈值、优于当前默认实现的更快实现，系统就可以触发一次“补丁”，把更快的实现提升为往后的默认实现。

这实际上与自动调优器闭环。系统不仅能即时发现更好的核函数，还能把它们替换进来。这就带来了一种自优化的核函数选择机制。

你可以把这一策略与我们前面讨论过的 RL 策略结合起来。这样，你可以同时加载多个实现——并让 RL 智能体为每一种不同类型的请求选择使用哪一个。

> 要留意在内存中维护多个实现所需的开销。新增的代码体积不应把其他关键代码从指令缓存中挤出——也不应耗尽 GPU 内存。

例如，一个高度优化的核函数对长序列可能最优，但一个更简单的核函数实现对短序列可能就够了。运行时系统可以按请求在它们之间做选择。这实际上是基于输入序列长度按请求逐个打补丁。

简而言之，运行时核函数打补丁关乎灵活性。它承认今天最优的东西明天可能会变，并提供了快速适应的手段。这种敏捷性能确保你的服务基础设施跟得上模型加速技术的飞速进步。像 vLLM、SGLang、NVIDIA TensorRT-LLM 和 NVIDIA Dynamo 这样流行的推理引擎架构良好，提供了相当多的动态能力。例如，TensorRT-LLM 允许你使用运行时插件加载优化过的核函数。请善用这些能力为你所用。

## 使用时间序列预测对 CUDA Graphs 和缓存进行持续预热

在高吞吐推理场景中，冷启动开销可能是请求-响应延迟的一大来源。CUDA Graphs（如第 12 章所述）允许你捕获一段 GPU 操作序列，并以极小的启动开销重放该序列。

预热（prewarming）一个 CUDA Graph 意味着在真正需要它之前就把这张图搭建好。如果我们能预测某种请求类型或某个 batch size 何时会出现，预热好的 CUDA Graph 就已准备好执行。

这项技术依赖预测的准确性。如果你的预报有偏差，你可能会预热一张用不上的图。你应当监控预测命中率，以确保这项优化确有回报。

通过预热，当请求真正到来时，图可以跳过代价高昂的启动与初始化步骤。系统使用像 ARIMA 和 Prophet 这样的时间序列预测（time-series prediction）算法来预判未来的负载模式，包括流量激增和 batch size 变化，从而主动为快速执行准备好图。

设想一个推理服务观察到一个每日周期：每天早上 9 点，由于与时区相关的流量模式，用户流量会出现一次尖峰。了解这一点后，系统可以在 9 点之前预先运行几个请求，把模型加载进 GPU 缓存、即时编译（JIT）所需的核函数，并为预期的各个 batch size 捕获 CUDA Graphs。到了 9 点，当真实流量负载激增时，进入的请求就会复用为预期 batch size 和使用模式预热好的状态。

在实践中，这可以由一个类 cron 的作业或一个调度服务来编排，在预计流量尖峰即将到来之前触发预热例程。只需记得为资源的置备留出足够的时间。

在 PyTorch 中，把模型的计算包裹在 torch.cuda.graph() 上下文里，可以让你捕获一张包含 GPU 核函数启动的静态计算图。重放这张图时，PyTorch 会绕过 Python 到 CUDA 的分派开销，用单次 cudaGraphLaunch 提交整个工作负载。这会带来显著更快的执行——尤其是对大而重复的输入批次，因为 CPU 的参与极少。

其缺点在于这些图是静态的，因为它们不容易允许可变的形状或长度。但推理服务器往往处理的是各不相同的 batch size（1、2、4、8、16 等）——由于 GPU 内存限制和算法优化——这与图模型很契合。

为应对可变性，你可以为每个常见的 batch size 或序列长度维护一个预捕获图的池。你可以使用持续预热（continuous prewarming）策略来为这些不同的 batch size 准备好图池。例如，在高峰时段，16 个请求组成的批次很常见。服务器可以预先用 16 的 batch size 捕获模型前向传播的一张图——然后把它存起来供后续使用。

下一次同时有 16 个请求进来时，推理系统就把这批输入喂进预捕获好的图。图里的核函数已经为图执行预热并优化好了，因此无需入队并启动许多单独的操作，只需一次图启动就够了。

请记住，池中存储的每一张已捕获的图都会为其工作区消耗额外的 GPU 内存。存储许多图时你应当监控内存使用。如果内存吃紧，可能有必要把使用较少的图从池中逐出（evict）。

为避免一个池带来的额外内存开销，你或许可以使用第 12 章讨论过的图打补丁（graph patching），来针对细微的尺寸差异调整图节点。不过在实践中，就性能而言，用池是更好的选择。

CUDA Graphs 还能降低 CPU 开销，因为它只需协调单次图启动，而不是许多单独的核函数启动。这让 CPU 可以去做数据预处理等其他类型的“真正”工作——而不是去协调许多核函数启动。

同样地，你可以使用时间序列预测算法（例如 ARIMA 和 Prophet）来预报诸如 RPS 和平均提示长度这类指标。如果模型预测到 batch size 会跳升，或出现某种特定的请求模式（例如长输入序列），系统就可以开始准备并预热相应的 CUDA Graphs、缓存和其他资源。例如，系统可以主动增大连续批处理算法的 batch size、把模型权重从磁盘预取到 GPU 内存，并分配额外的 GPU 实例。

> 重要的是要经常用近期的流量数据来重新训练并更新这些时间序列模型。这是因为使用模式会随时间变化，例如有新的用户群上线等。此外，未曾预料的“节假日”，比如加州著名的滑雪周（Ski Week），会出人意料地抬高负载曲线——当我刚搬到加州时，在 Netflix 就遇到过这种情况！

与缓存和预热相关的一点，是预判 prefill 和 decode worker 的横向扩容。如果我们知道由于一个计划中的批处理作业或每日报表场景，一批带长提示的请求很可能在某个时间到来，系统就可以横向扩容 prefill worker，并用有代表性的样本数据执行一次有代表性的前向传播，以预热 CUDA Graphs 和 KV 缓存。

横向扩容和预热事件也应当包含 decode worker 和 CUDA Graphs。以 decode 为例，推测式解码（speculative decoding）的草稿模型就可以被预热并加载进 GPU 内存——草稿模型的 KV 缓存等也是如此。其他 decode 优化包括针对将要生成的固定 token 数做循环展开——无论是 1 个 token，还是推测式解码情况下的多个 token。

你还应当考虑缓存 CUDA 核函数本身。CUDA 通常会缓存编译好的核函数（C++ 代码 => PTX 指令 => SASS 汇编）以加速执行。核函数首次运行时，很可能会有一些即时（JIT）编译开销。

你应当把这类预热与你的集群自动扩缩器（autoscaler）协调起来，使得当新的 GPU 实例启动时，自动扩缩器用几次预热 API 调用在它们上面跑一小组推理。这样做会验证推理引擎、缓存 JIT 编译输出、分配内存池、准备 CUDA Graphs 等。如此一来，这些引擎在被加入实时流量池之前就已经处于生产就绪状态。

> 利用你对系统的了解，在这个受控的预热阶段尽可能识别并触发尽量多的不同路径。这包括每一个 batch size、每一种 CUDA Graph 变体等。

你还应当通过对比预测尖峰期间头几个请求的延迟——分别在有预热和无预热两种情况下——来监控预热是否真的有帮助。按需调整时机和阈值。并且要意识到时间序列预测可能出错。务必确保预热任务在错误的时机运行时不会影响整体推理性能。

例如，你应当尽量在 GPU 利用率不高时执行预热运行（用示例数据）。CUDA 允许按 stream 优先级调度，因此你可以把预热 stream 的优先级设得低于你的推理 stream。这样，预热就不会与实时的大流量请求争抢资源。

此外，你应当为这些预热任务使用较低优先级的 CUDA stream，这样在预热期间若有更高优先级的实时流量到来，它们就会让位。好的一面是，空闲的 GPU 是一种被浪费的机会，因此在空闲 GPU 上做预热计算基本上是免费的——只要它不与真正的工作发生冲突。

Grace Blackwell 系统允许一些额外的技巧，因为 CPU 和 GPU 以低延迟共享统一、一致的内存。例如，你可以让 CPU 开始把数据预填进 GPU 计划要用的统一内存里。这可以避免之后显式的 GPU 拷贝调用。

由时间序列预测引导的持续预热，能让延迟变得可预测得多。它把推理引擎变成一个自适应系统，能学习常见的流量模式、自动准备数据、备好硬件，并始终领先于需求一步。这会通过在可以摊销的时机把编译和内存传输这类昂贵操作平滑掉，从而减少抖动并降低延迟尖峰。

这对于有着沉重一次性初始化成本的大型模型尤其有价值。随着模型和上下文的增长，这类自适应预加载将从“锦上添花”变为生产 LLM 系统中的“必需品”。以预测性的方式支付这些成本，远好于因糟糕体验而让终端用户流失。

## 自适应批处理与分块 prefill 调度

在第 16 章中，我们讨论过现代推理服务器如何使用不同类型的请求批处理（例如连续批处理，continuous batching）来在所有请求间最大化吞吐、最小化延迟。然而，批处理会增加单个请求的延迟。可以用一种叫做 _自适应批处理（adaptive batching）_ 的技术，随着一天中条件的变化动态地处理这些权衡。

自适应批处理会根据负载——以及请求进展得如何——动态调整请求被分组成批的方式。这类动态策略可以在环境变化时实时调整 batch size 和阈值参数。

例如，在高峰负载时，系统可以使用较大的 batch size（例如 8 或 16），因为此时吞吐量至关重要。在低负载时段，系统可以减小 batch size 以更快地服务请求。这会把延迟置于吞吐量之上优先考虑。

要决定 batch size，你可以使用一个简单的启发式规则，例如 _“如果 GPU 利用率 > 80%，允许更大的批次；如果 < 20%，使用 batch size 1 以最小化延迟。”_ 或者你可以使用前几节讨论过的更复杂的 RL 智能体或预测式策略。

prefill 和 decode 两个阶段在算术强度上的这种差异，会导致执行时长的不匹配。因此，最好把这两个阶段拆解开，将它们视为可以独立调优的、彼此分离的工作负载——对单节点用分离的线程/进程，对多节点集群用 worker 池。

通过拆解 prefill 和 decode，我们把这两个阶段当作彼此分离的操作。这使得它们可以针对各自独特的计算与内存带宽需求独立优化。其中一项优化就是 prefill 和 decode 两个阶段所用的 batch size。

在实践中，vLLM 和其他现代推理引擎正是这么做的。它们组成分离的批次分别发给 prefill worker 和 decode worker。这样，一批 prefill 请求就可以独立于一批 decode 请求执行。例如，decode 阶段可以从更大的批次中获益以提高算术强度，因为它是一个内存受限（memory-bound）的工作负载。

像 vLLM 这样的现代推理服务框架使用自适应调度循环，在处理一个 prefill 批次还是一个 decode 批次之间动态选择。具体来说，vLLM 支持分块 prefill（chunked prefill）和 decode 最大化（decode-maximal）调度，以交错 prefill 与 decode 来获得更好的利用率。这些技术在不显著增加延迟的前提下提升了整体利用率和吞吐量。

prefill 和 decode 在系统资源特性上的不匹配也会影响 PP。设想一个微批在一条长序列上做 prefill，而另一个微批正在一次一个 token 地做 decode。在这种情况下，它们的时长不匹配，流水线气泡（pipeline bubble）就会出现。

你可以把大的 prefill 请求切成小块，并在它们之间夹带（piggyback）decode，从而将大 prefill 请求与延迟敏感的 decode 任务交错起来。这能让所有流水线阶段都保持忙碌，并把 GPU 调度中空闲的“气泡”降到最低。

_分块 prefill（chunked prefill）_ 是所有现代 LLM 推理引擎都很好支持的一种模式，用于减少流水线气泡。它有效地对一个大任务（prefill）做时间切片，为小任务（decode）腾出空间，让它们在由分块所制造的流水线间隙中执行，如图 19-7 所示。

![图 19-7. 分块 prefill 在四个请求间实现 decode 最大化批处理的收益](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-7.png)

SARATHI 论文证明，这类分块 prefill 与夹带的做法能帮助你找到合适的 _decode 最大化批处理（decode-maximal batching）_ 水平、减少气泡，并相比朴素调度把吞吐量提升约 1.3–1.9×。SARATHI 这个名字取自一位驭者（charioteer），他能把 prefill 和 decode 两类任务一起智慧地驾驭。有趣吧！

举例来说，设想一个不使用分块的 10,000-token prefill 请求。在这种情况下，单次 prefill 遍历会阻塞整条流水线，导致 decode 任务排队等待，直到 prefill 完成。

然而，如果你使用分块 prefill，把这个 10,000-token 的 prefill 请求切成五个 2,000-token 的块，你就可以在各 prefill 块之间交错 decode 批次，让 GPU 忙于同时处理两个阶段并推动进度。这会挤掉流水线气泡、提高吞吐量，并平滑 GPU 利用率。

> 一条经验法则是选择这样一个块大小：让一个 prefill 块耗时 ~50–100 ms。这样，你就有频繁的机会在其间调度 decode 批次。根据模型架构/规模和 GPU 硬件的不同，这可能对应几千个 token。

像 vLLM 这样的现代推理引擎使用自适应调度循环，基于 GPU 利用率和队列状态来决定是处理下一个 prefill 块——还是执行一个 decode 批次。具体来说，vLLM 持续监控 token 队列来做这些决策。vLLM 的调度器显式支持分块 prefill 和 decode 最大化批处理。它的执行器和分块 prefill 特性被设计为把大的 prefill 与较小的交互式 decode 重叠起来。

自适应调度器在选择分块 prefill 大小时，需要考虑 GPU 共享内存限制和占用率（occupancy）。下面展示了一个简单的自适应分块 prefill 实现。这段代码动态地把块大小调到合适，以让 GPU 上的 SM 和占用率保持在高位：

```
# Example adaptive scheduler for chunked prefill/decode
import cupy as cp
import torch
# Hardware constraints
SHMEM_LIMIT   = 256 * 1024
BLOCK_THREADS = 256
TARGET_UTIL   = 0.85
OCC_THRESHOLD = 0.5
# Cache for occupancy results and tile lookup
_occ_cache = {}
_tile_table = {}  # e.g., {L: optimal_T}
# 1) Precompute tile_table offline; here we lazy-initialize on first use
def get_optimal_tile(L):
    if L in _tile_table:
        return _tile_table[L]
    # compute block size by querying for an occupancy-based suggestion
    min_grid, max_grid, block_size = ... # left out for brevity
    T = min(block_size, L)
    T = max(32, (T // 32) * 32)
    _tile_table[L] = T
    return T
# 2) Cached occupancy query
def get_occupancy(threads, shared_bytes):
    key = (threads, shared_bytes)
    if key in _occ_cache:
        return _occ_cache[key]
    max_blocks = cp.cuda.runtime.\
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            attention_kernel_ptr, threads, shared_bytes
        )
    props = torch.cuda.get_device_properties(0)
    warps_per_block = threads // props.warp_size
    max_warps = props.max_threads_per_multi_processor // props.warp_size
    occ = (max_blocks * warps_per_block) / max_warps
    _occ_cache[key] = occ
    return occ
def scheduler_loop():
    stream = cp.cuda.Stream(non_blocking=True)
    while True:
        pending = get_pending_requests()
        util = gpu_utilization()
        if util < TARGET_UTIL and any(r.phase=='prefill' for r in pending):
            req = select_heaviest_prefill(pending)
            L   = req.remaining_length()
            T   = get_optimal_tile(L)
            shared_bytes = 3 * T * T * 4
            occ = get_occupancy(BLOCK_THREADS,
              shared_bytes)
            if occ < OCC_THRESHOLD:
                T = max(32, T // 2)
                shared_bytes = 3 * T * T * 4
            chunk = req.next_prefill_chunk(T)
            # Launch with CuPy RawKernel on our stream
            attention_kernel((...grid...), (BLOCK_THREADS,),
               (chunk, ...), shared_mem=shared_bytes,
               stream=stream)
            # Record an event to know when done
            event = cp.cuda.Event()
            event.record(stream)
        elif any(r.phase=='decode' for r in pending):
            batch = form_decode_batch(pending, max_batch=16)
            # Trace the adaptive logic
            logger.info(f"T={T}, occ={occ:.2f},
                util={util:.2f}")
            launch_decode_kernel(batch, stream=stream)
            event = cp.cuda.Event()
            event.record(stream)
        else:
            # Poll the last event rather than sleep
            if not event.query():
                cp.cuda.get_current_stream().synchronize()  # or pass
            continue
```

这里，调度器在调整块大小，以尽可能多地使用共享内存。而且重要的是，它这样做时并不牺牲并行度。

具体来说，调度器首先计算一个 tile 宽度 T，使得用于 queries、keys 和 values 的三个共享内存缓冲区——每个都需要 T x T 个 float——能装进 GPU 每个 SM 的动态共享内存限制之内。然后它调用 CuPy API（Python）来测量在该 T 值下每个 SM 上能并发运行多少个线程块。

如果占用率跌破给定的 50% 阈值，T 就减半。这会释放共享内存，从而让更多的块可以共同驻留，在数据复用（更少的 DRAM 加载）与并行度之间取得最优平衡。

当整体 GPU 利用率降到 85% 以下且有 prefill 工作待处理时，调度器会选出剩余最大的 prefill 请求，并把它切成大小相等、每块 T 个 token 的块，使得每个块都能流经流水线而不独占每一个阶段。

而且，这个辅助函数 next_prefill_chunk 并不固定块大小，而是基于实时指标即时调整 T。如果占用率低，它会缩小块大小——如果 DRAM 流量过大，它会增大块大小。这确保了每个切片都能在不停顿的情况下最大化 GPU 利用率。

> 务必给调度器加上埋点，记录所选的 T 以及由此产生的占用率/利用率。这样，你就能分析并验证这种自适应方法是否始终如一地维持了高 GPU 利用率。

在各 prefill 块之间，调度器可以使用 _decode 最大化批次（decode-maximal batches）_，用 form_decode_batch 把所有就绪的 decode 请求打包进一次启动。这让短小、延迟敏感的 token 生成得以夹带在本来空闲的流水线间隙上。如此一来，即便是提示很短的用户也能看到低延迟，因为他们的 decode 不必等待一个庞大的 prefill 完成。这些 decode 会被调度进流水线的间隙里。

通过持续监控 gpu*utilization()，调度器选择是处理下一个 prefill 块还是排空 decode 队列。无论哪种方式，它总是挑选那个能填满 SM 槽位并把空转时间降到最低的动作。这被称为 *利用率最大化策略（utilization-maximization policy）\_，类似于一个以 100% CPU 利用率为目标的操作系统调度器。

这些机制合在一起，确保大上下文的 prefill 作业永远不会让交互式解码挨饿。与此同时，小而延迟关键的请求会被立即服务。这在不降低终端用户体验的前提下，于现代 GPU 上产生了最优吞吐量。

> 分块 prefill 使工作与 GPU 的强项对齐：大的 prefill 矩阵运算被成批处理，而小的 decode 运算被交错穿插。这同时最大化了整体吞吐量和延迟表现。

如你所见，调度器监控实时指标并即时适应。分块调度确保在多变的条件下，没有任何单个请求能阻塞整个阶段。这让所有 GPU 都保持活跃，并减少了令人头疼的流水线气泡。

把 prefill 和 decode 当作各有其 SLA 的分离队列很重要。通常最优的做法是先用专用的时间片或 CUDA stream 清空延迟敏感的 decode 任务，再用剩下的周期去处理大的 prefill 作业。例如，你可以给 decode 任务分配一定的时间预算（例如每个 decode 1–5 ms），以确保它们获得更即时的关注。

通过优先处理面向用户、实时的 decode，你就把感知到的推理滞后降到最低。与此同时，在系统有余力时，你仍在全力推进批量的上下文构建。

许多高性能推理引擎使用 _生产者-消费者（producer-consumer）_ 模型，为不同阶段设置分离的线程。例如，vLLM 使用多线程，使得一个线程准备 decode 输入，而另一个线程准备新的 prefill 等，并让它们以优化过的顺序汇入单一的执行 stream。这是一种经过验证的、能高效重叠工作的模式。

如果你有多个节点，你可以把 prefill 请求发给一组节点，把 decode 请求发给另一组节点。在这种情况下，prefill 节点和 decode 节点可以使用为各自任务（prefill 或 decode）专门定制的、不同的异构硬件。

例如，prefill 计算节点可以使用高 FLOPS、内存带宽较低的 GPU，因为 prefill 阶段是计算受限（compute bound）的。而 decode 节点则可能使用内存带宽更高但 FLOPS 容量较低的 GPU。

要注意，虽然异构的 prefill 和 decode worker 配置可以节省成本，但它可能使动态负载均衡变得复杂——并可能对其形成限制。如果 prefill/decode 比例意外偏移，需要在 decode 优化型 worker（FLOPS 较低）上做更多的 prefill 工作，这些 decode worker 可能会成为瓶颈。
