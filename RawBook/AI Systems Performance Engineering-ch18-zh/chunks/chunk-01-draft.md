# 第 18 章 高级 Prefill-Decode 与 KV 缓存调优

本章在第 17 章的基础上，深入探讨推理 prefill 与 decode 阶段的高级优化。我们将在高层扩展策略之上，介绍一系列底层技术，包括单次 decode 的“超级核函数（mega kernel）”、跨 GPU 的智能 KV 缓存调优与共享、提示状态的快速 GPU 到 GPU 传输、自适应资源调度，以及 prefill 与 decode worker 之间的动态路由。

我们还会重点介绍那些带来全新性能与效率水平的软硬件创新。运用这些技术，你可以大幅降低 decode 延迟、提升单 GPU 吞吐量，并在大规模场景下满足严格的延迟 SLO。

## 优化的 Decode 核函数

到目前为止，我们一直聚焦于高层的系统与集群优化策略。当推动超大规模推理时，另一类值得考虑的技术是底层核函数与内存管理调优——尤其是针对 decode 阶段。

decode 阶段是分布式的，且往往访存受限（memory bound）。这促使研究者与从业者尽可能地加快 decode 阶段——并针对特定硬件进行调优。该领域中有三项值得关注的创新：FlashMLA（DeepSeek）、ThunderMLA（Stanford）以及 FlexDecoding（PyTorch）。它们专门针对 transformer 在 decode 期间的多头注意力效率，应对 LLM 负载中常见的变长序列场景。下面我们逐一介绍。

### FlashMLA（DeepSeek）

Flash Multi-Latent Attention，即 FlashMLA，是 DeepSeek 推出的一种优化的 decode 核函数。它专门聚焦于单 token 的 decode 步骤——本质上就是用于生成下一个 token 的 transformer 层前向传播。FlashMLA 通过融合运算并更好地利用 GPU 内存层次结构，使 decode 更快。

FlashMLA（decode）之于推理，正如 FlashAttention（prefill）之于训练。它降低了内存访问开销与延迟。使用 FlashMLA，与标准核函数相比，你可以在 decode 阶段实现大幅的延迟下降。

FlashMLA 通过把多个注意力运算融合成一个，从而提升算术强度（arithmetic intensity）。这样，它就能在一次融合核函数启动中处理多个头和多个时间步。即便批量很小，也能让数学计算单元保持繁忙，从而提高 decode 期间的 GPU 利用率。图 18-1 展示了在 Hopper H100 GPU 上，MLA（多头潜在注意力，Multi-head Latent Attention）相较于其他注意力实现（如分组查询注意力（grouped-query attention，GQA）和多查询注意力（multi-query attention，MQA））在算术强度上的提升。（注：Blackwell 凭借更高的 TFLOPs 与 HBM 带宽，将两条 roofline 都向上抬升。）

![图 18-1. MLA 逼近计算受限区间（在 NVIDIA Hopper H100 架构上测得）](AI%20Systems%20Performance%20Engineering-ch18_images/figure-18-1.png)

FlashMLA 的推出意义重大，因为它表明——即便在并非最优的 GPU 硬件上——decode 阶段的瓶颈、内存带宽以及核函数启动开销也可以被降低。它减少了独立 GPU 核函数启动的数量，并优化了内存访问模式——为 decode 任务尽可能压榨受限硬件的每一分性能。

DeepSeek 开源的 FlashMLA 实现已经可用，并逐渐被采纳。SGLang 和 vLLM 都为 DeepSeek 模型提供了一流的支持。因此，你应当评估 FlashMLA，以在不改变更高层架构的前提下提升单 token 的 decode 吞吐量。

由于 DeepSeek 开源的 FlashMLA 已集成到现代推理服务系统中，你应把它作为一种手段来探索——在不做任何更高层架构改动的前提下，提升每个 decode worker 的吞吐量，或降低每 token 的延迟。

### ThunderMLA（Stanford）

在 FlashMLA 的基础上，Stanford 的研究者推出了 ThunderMLA，这是一个完全融合的注意力 decode “超级核函数（megakernel）”，聚焦于 decode 与调度（而非融合整个前馈块）。这个“超级核函数”通过把多次核函数启动合并为一次——并整合中间内存写入——来降低启动开销与尾部效应。ThunderMLA 报告称，在不同负载下，其 decode 吞吐量比 FlashMLA 快 20–35%

ThunderMLA 的核心思想是：当 decode 不同长度的序列时，采用细粒度调度与融合运算，可以避免这样一种尾部效应——某些序列较早完成，而其他序列让 GPU 部分空闲。即便某些 decode 流较早完成，ThunderMLA 也能让 GPU 保持繁忙。它通过用融合方式动态打包并处理剩余的流来实现这一点。

在拥有更大 L2 缓存和更快注意力原语（attention primitive）的现代 GPU 上，这些收益会被进一步放大。值得注意的是，现代 NVIDIA GPU 还提供 Transformer Engine 对 FP8 和 FP4 的支持（以及 FP6，不过本书主要聚焦 FP8/FP4 格式，因为 FP6 在现有 AI 框架与工具中尚未广泛使用）。结合更高的内存带宽，Tensor Core 让像 ThunderMLA 这样的核函数能够运行得远远更接近硬件极限。在现代 GPU 上，得益于这些架构进步，ThunderMLA 实现了更低的每 token 延迟。

### FlexDecoding（PyTorch）

在第 14 章中，我们讨论了 PyTorch 的 FlexAttention，它让你能够为任意注意力稀疏模式（包括局部窗口、块稀疏模式等）即时编译（JIT）融合核函数——而无需编写自定义 CUDA。在底层，TorchInductor + OpenAI 的 Triton 会生成一个融合核函数，只计算该模式下允许的 query-key 对。当在给定硬件上有利时，Triton 会自动应用 warp specialization 和异步拷贝等性能优化技术。不过，你也可以调优 triton.Config，例如通过配置 num_consumer_groups 来进一步定制。

FlexDecoding 是 torch.nn.attention.flex_attention 的 decode 后端。FlexDecoding 同样让你能够就地管理 KV，并像 FlexAttention 一样支持掩码与偏置。具体来说，FlexDecoding 为 decode 阶段（Q_len=1）编译一个专用核函数，在不断增长的 KV 缓存上进行注意力计算。

在运行时，FlexDecoding 实现会选择这个专用的 decode 核函数，并在多个 decode 步骤间复用它。当形状与 dtype 保持兼容时，这有助于最小化开销——从而大幅加速长序列 LLM 推理。

> 一旦重编译处于可控状态，对于稳定、延迟敏感的 decode，优先使用 torch.compile(mode="max-autotune")。保持捕获边界较窄（按层或按注意力块），以减少不规则批处理带来的图失效。prefill 与 decode 都优先使用 Transformer Engine FP8（MXFP8）。当精度允许且性能有提升时，可考虑 FP4（NVFP4）。截至本文撰写时，FP4 支持仍在成熟中，短期内可能不及 8 位和 16 位格式。继续设置 torch.set_float32_matmul_precision("high")，以便在剩余的 FP32 运算上启用 TF32 回退。FlexAttention 的 decode 后端支持常见的性能增强，包括分组查询注意力（GQA）和 PagedAttention。

FlexAttention 与 FlexDecoding 的一个关键特性是支持嵌套锯齿布局张量（nested jagged-layout tensors，NJT）。这些张量允许在 decode 期间对变长序列进行不规则批处理（在 LLM 负载中很常见）。图 18-2 展示了多条序列的锯齿张量表示。

![图 18-2. 以嵌套锯齿张量（偏移量）表示的不规则批；三条序列（上方）被表示为带偏移量的单个嵌套锯齿张量表示（下方）；decode 时的批处理优先使用 PyTorch NJT](AI%20Systems%20Performance%20Engineering-ch18_images/figure-18-2.png)

此外，FlexDecoding 支持偏置项，并通过一个块掩码转换接口与 PagedAttention 集成，该接口将逻辑块映射到物理缓存布局。如图 18-3 所示，这会把逻辑 KV 块散布（scatter）到物理缓存布局中——且不产生额外拷贝。

FlexDecoding 利用被捕获的张量，在每次迭代中改变某些掩码或偏置值——而无需重编译。并且它与 PagedAttention 集成。要使用像 vLLM LMCache 这样的全局 KV 缓存，可将缓存的页表映射到 FlexAttention 的 BlockMask。这会即时地把逻辑 KV 页转换为物理内存地址。

借助 FlexDecoding，开发者对自定义注意力稀疏模式拥有完整的 Python 层级灵活性。这对 MoE 模型推理尤其有用。FlexDecoding 让你无需编写任何自定义 CUDA 核函数即可实现接近最优的性能。本质上，它让任意注意力模式都能像稠密注意力模式一样得到优化。随着新推理技术不断涌现，这一点会变得愈发有价值。

![图 18-3. PagedAttention 将逻辑 KV 块散布到物理 KV 块中，以在序列间实现最优缓存复用；将块大小与 LMCache 页大小对齐——更大的页（如 64–128 tokens）可降低分离式配置中的 RDMA 开销](AI%20Systems%20Performance%20Engineering-ch18_images/figure-18-3.png)

其中许多能力——例如用于 decode 的融合注意力，以及对 PyTorch 嵌套锯齿张量（NJT）批处理的支持——都已在核心 PyTorch 库中提供。这使得对于典型模式而言，自定义融合变得不那么必要。

> 在对 LLM 负载中常见的不规则序列进行批处理时，优先使用 NJT 布局。

这些核函数层级的进步高度技术化，充分发挥了 GPU、网络与内存的全部能力。这些软件优化可以显著提升 decode 性能——即便在同一硬件上也是如此。在设计超大规模系统时，如有可能，你应当纳入这些优化核函数。务必使用 Nsight Systems 配合基于硬件的 CUDA 追踪来验证重叠与核函数效率。此外，使用 Nsight Compute 查看特定的内存与链路指标。

> 启用其中某些高级核函数，可能需要安装自定义库或启用专用的 CUDA 核函数——对于较新的技术尤其如此。不过，这些技术通常在发布后不久就会得到 PyTorch 和主流推理引擎的支持。即便你确实需要安装自定义资源，这份付出也是值得的，因为它会直接转化为更低的延迟和更少的 decode worker 池 GPU 数量。

## 调优 KV 缓存利用与管理

分离（disaggregation）要求你把 KV 缓存视为跨集群的一等共享资源。由于 KV 缓存现在可以存活更久并在节点间移动，高性能推理系统改进了 KV 缓存的存储与共享方式。

具体而言，分布式 KV 缓存池和跨请求的前缀复用成为强大的技术。此外，密切关注新一代 GPU 与 HBM 带来的内存带宽提升也很重要。下面我们围绕提升 KV 缓存性能，逐一讨论这些方面。

### 分离式 KV 缓存池

与其让每个 GPU 只为其当前正在服务的请求存储 KV，分离式 KV 缓存池（disaggregated KV cache pool）将 KV 存储与单个 GPU 解耦。取而代之的是，它把数据分散到整个集群的 GPU 内存中。

该池还可以卸载到 CPU 内存，包括 Grace Blackwell 和 Vera Rubin 平台上统一的 CPU 与 GPU 内存。它也可以卸载到 NVMe SSD 这样的持久化存储。

使用分离式 KV 缓存池，当一次 prefill 为某个提示计算 KV 张量时——或当一次 decode 扩展 KV 张量时——KV 块会以分布式方式存储在众多计算节点上。如图 18-4 所示，该图改编自关于分离式 KV 池的工作。

![图 18-4. 分离式（分布式）KV 缓存池（来源：https://oreil.ly/2xtK-）](AI%20Systems%20Performance%20Engineering-ch18_images/figure-18-4.png)

设想一个非常长的 250,000 token 上下文（例如一个包含很多轮次的聊天会话），使用一个具有 80 层、32 头、每头维度为 128 的 700 亿参数 transformer 模型。这会为每个 token 产生巨大的 KV 缓存占用。

每个 token 会生成一个键向量和一个值向量，其长度等于模型的隐藏维度（num_heads × head_dim）。我们假设本模型该值为 4,096。于是每层产生 8,192 个浮点数。跨 80 层求和，每个 token 会产生 655,360 个浮点数的 KV 数据。假设采用 16 位精度，即每个浮点数 2 字节。这样每个 token 大约需要 1.31 MB。扩展到 250,000 个 token，大约会产生 328 GB——而这仅仅是 KV 数据！

> 这些计算基于大模型的每 token KV 大小。这说明了为何 FP8 KV 缓存在 vLLM 等引擎中被广泛采用，以减少占用并增加批处理机会。

假设我们把 KV 缓存量化到 FP8，并使用诸如选择性分层缓存之类的技术，对于这个 250,000 token 的提示，我们大概能把占用降到 100–150 GB 的范围。单个 GPU 很可能没有容量同时容纳所有 token 的 KV 以及它需要保存的其他一切（如模型权重等）——尤其是随着多轮对话（multiturn conversation）持续进行。因此，系统将不得不截断上下文——或对较早的 token 触发代价高昂的 KV 重算。

然而，有了分离式 KV 池，上下文中较旧部分的 KV 可以从 GPU 逐出（eviction），推送到分散在集群中的 KV 缓存池——存放于 CPU DRAM 或 NVMe 存储中。之后当需要时，数据再被取回 GPU 内存。

分离式 KV 缓存池实现了一个多层内存层次结构，其中 GPU 设备内存保存活跃的 KV 缓存，而 CPU 主机 RAM（或 NVMe 存储）充当溢出的后备存储。现代推理引擎可以选择把 KV 缓存卸载到 CPU 内存或 NVMe。这有效地虚拟化了 GPU 内存，很像操作系统的虚拟内存子系统。

这种设计通过在 GPU 内存与 KV 缓存池之间异步分页 KV 块——且不停顿计算流水线——来支持超长上下文，前提是良好的通信-计算重叠，正如本书通篇所讨论的那样。

此外，通过把请求状态与单个 GPU 解耦，系统可以利用全局 KV 缓存池，在集群的多个节点间动态分片 KV 数据，以自适应地均衡负载。这简化了扩展，并改善了大型推理集群中的故障隔离（fault isolation）。

而且，既然任何 decode 节点都能访问全局 KV 池，那么在因故障转移或负载均衡而需要时，任何 decode 节点都可以参与任何请求的 decode。这为调度器增加了灵活性，因为它可以选择离相关 KV 缓存块所在位置最近的 decode 节点。

如果某个前缀的一些 KV 块被缓存在服务器 A 的 DRAM 中，那么把 decode 调度到服务器 A 可能更快，因为它可以迅速把这些块拉入自己的 GPU。这与服务器 B 形成对比——后者不得不通过网络去获取这些 KV 块。

> 这描述的是分布式系统的一条经典最佳实践：选择离待计算数据最近的计算节点。这样，系统就能最小化代价高昂的数据移动。

一个高效的 KV 缓存调度器可以查看池中 KV 块的分布——再结合网络拓扑——据此分配 prefill 与 decode 任务。因此，一个 prefill 节点可以把 KV 数据放入一个实现为跨集群可访问的分布式内存空间的池中。

当 KV 缓存位于集群侧的共享内存空间中时，任何 decode 节点都能取回数据。这就避免了每次都必须为 prefill 到 decode 的直接传输而调度。

由于要多一跳从池中取回 KV 数据，这会增加一点额外开销，但它带来了更多灵活性，因为所有 decode 节点都能访问所有已缓存的 KV 数据。这也意味着，一个并未直接从某个特定 prefill 接收数据的 decode 节点，在需要时仍可从池中访问该 KV 数据。

如果某个 decode 节点崩溃——或某个请求由于任何原因需要在生成过程中迁移——KV 数据不会丢失。数据存放在池中，另一个节点可以接手，并使用保存的 KV 从中断处继续。这提升了容错（fault tolerance）能力。

全局 KV 缓存池还提供跨请求的缓存持久化。这样，如果两个请求共享某个前缀，那么该前缀的 KV 可以只计算一次，并在整个集群中复用——即便这些请求最终落在不同的 decode 服务器上。

简而言之，分离式 KV 缓存池以内存（或更冷的存储）换取计算。通过存储更大的 KV 缓存，系统在许多场景下可以避免重算 KV 数据。这种方法利用了这样一个事实：复用数据——即便来自 DRAM 或 SSD——通常比反复重算具有平方时间复杂度 O(N2) 的大型注意力矩阵乘法更便宜。

### KV 缓存复用与前缀共享

如前所述，对于共享同一公共前缀的提示，跨请求复用已缓存的 KV 数据是有益的。这种场景相当常见，表现为多轮对话、共享系统提示以及附带文档。

系统无需为每个请求都重算该前缀的 transformer 注意力输出，而是可以存储该前缀的 KV 输出并直接复用。

本质上，这跳过了输入中那一部分的 prefill 计算，从而节省了大量时间与 GPU 周期。

一个恰当的以 KV 缓存为中心的调度器在分配工作时，会通过查看“前缀缓存命中长度（prefix cache hit length）”——也就是该提示中已有多少 token 存在于缓存池中——来考虑前缀缓存命中。实践中，如果一个新请求到来，且其前 _N_ 个 token 与 KV 池中某个已缓存的前缀匹配，系统就可以决定复用那部分 KV 数据。

vLLM 利用其 PagedAttention 机制，使用一个 KV“页（page）”的全局哈希表来实现自动前缀缓存（prefix caching）。这里，每个唯一的 16 token 上下文块都有一个哈希值。如果一个新请求需要的前缀与某个已存储的块（按哈希）匹配，它就可以直接拷贝那些 KV 张量，而不必重算。

如果同一个上下文再次出现，系统就从内存中提供它。本质上，它把一个上下文的 KV 视为可复用的数据，可以通过哈希按内容查找。实现通常会维护一棵全局“提示树（prompt tree）”来管理这些已缓存的上下文，并在必要时将其逐出。这会针对最频繁复用的前缀进行优化。

有效 KV 复用的关键在于识别相同或重叠的前缀。为简单起见，系统通常聚焦于精确匹配——即如果前 _N_ 个 token 完全匹配，就复用那一块。合并部分前缀重叠更为复杂，因为你需要以某种方式合并缓存，而这并不总是直截了当。所以典型的缓存采用精确前缀缓存。

不过，这里存在一个权衡。无限期地存储许多用户的 KV 缓存会消耗大量内存。系统必须为 KV 块实现诸如 LRU 之类的逐出策略，以丢弃不太可能被复用的缓存。这为新缓存腾出空间。调度器也可能基于复用可能性来决定保留哪些缓存。其思路是在内存约束下最大化缓存命中。

如果某个 prefill 节点已在其本地 GPU 内存或本地 DRAM 缓存中持有所需 KV 的一部分，那么把请求路由到该节点以最小化数据传输可能是有益的。这是数据感知调度的一个例子——它把计算送到数据所在之处，而不是总把数据拉到有计算可用之处。

这类似于分布式系统中的局部性感知调度。在前面关于路由的讨论中，我们曾提及这一点。如有可能，你应把请求路由到生成其前缀的那台服务器。这最大化了缓存命中的可能性。

在分离的更广泛语境中，前缀缓存的支持依赖于对跨众多请求的 KV 拥有统一视图，并可能将其存储在像全局池这样的可共享位置。这与孤立的按请求或按节点的方式形成对比。

这也有助于降低分离本可能带来的重算开销——若同一个提示在不同时间被送到不同节点时。有了全局 KV 存储或协调式缓存，即便一个用户的请求命中了不同的 decode 服务器，它们也能受益于彼此已缓存的工作成果。

### 优化的 KV 缓存内存布局

另一个底层创新领域是优化 KV 缓存内存布局。KV 缓存存储每条序列中所有过往 token 的键与值，对于许多并发的 decode 流而言，它可能变得非常庞大，因为每条流所用内存大致正比于 num_layers × 2 × sequence_length × d_head。

诸如分层缓存之类的技术很有用，因为并非所有 KV 对都需要始终保留在 GPU 内存中。KV 缓存中较旧的部分可以换出到 CPU——甚至被压缩。

由于我们强调保持 decode 延迟低，大多数设计会把活跃的 KV 缓存保留在 GPU 内存中以便快速访问。在这种情况下，你可以调优内存的布局与访问方式。

DeepSeek 的 FlashMLA 对 KV 缓存分页，并以固定大小的块（页）分配缓存，使活跃序列能够进行连续内存访问。这减少了缓存未命中与 DRAM 流量。

此外，一些系统实现前缀压缩——例如当某个提示的前缀因上下文窗口已移动而不再被关注时。在这种情况下，KV 缓存管理器可能丢弃或压缩这些 KV 条目。这在长对话中更为相关，因为上下文窗口会滑动。但对于极长的序列，它可以节省内存与带宽。

> 当模型使用滑动窗口或其他受限注意力模式时，这种逐出/压缩技术是安全的。然而，对于那些在整个内容窗口上保留完整注意力的层（或检索钩子），未经仔细评估不应对其应用该技术。

另一项称为 _POD-Attention_ 的技术同样重组注意力计算以减少 HBM 流量。具体而言，它采用 SM 感知的线程块（或协作线程阵列 [cooperative thread array，CTA]）调度。这实现了*运行时操作绑定*，动态地把在某个 SM 上运行的每个 CTA 分配去执行 prefill 或 decode 任务。如图 18-5 所示。

![图 18-5. SM 感知的线程块（CTA）调度，在 SM 上将 prefill 任务与 decode 任务匹配起来，以最小化内存移动](AI%20Systems%20Performance%20Engineering-ch18_images/figure-18-5.png)

于是，与其为每个阶段静态地启动独立核函数，不如由单个核函数启动足够数量的 CTA 来覆盖两种负载。在运行时，每个 CTA 检查自己位于哪个 SM 上，并使用每 SM 的计数器，根据该 SM 上还有什么在运行，来决定应当执行哪种操作（prefill 或 decode）。

这套 SM 感知的调度逻辑试图在运行时把 prefill 与 decode 操作匹配起来。这避免了孤立的内存流量突发，并平滑了资源需求。

具体而言，POD-Attention 把 prefill 与 decode 工作共置在同一个 SM 上，使融合核函数能够改善局部性并减少冗余的 HBM 事务。这最小化了内存移动、最大化了带宽利用率，并在每个 SM 上平衡了计算受限与访存受限的负载。通过把 prefill 与 decode 工作共置在同一批 SM 上，并配以恰当的 SM 感知 CTA 调度以释放完全重叠，POD-Attention 可以将注意力性能提升多达约 29%。

POD-Attention 的动态绑定把硬件的 CTA-SM 分配与软件的 CTA 角色分配（prefill 或 decode）解耦。这类创新表明，人们越来越关注软硬件协同设计，以最小化内存移动并最大限度地发挥系统性能。

### GPU 与 CPU-GPU 超级芯片改进

你还应考虑新硬件中的内存带宽提升。更高的内存带宽和更大的 L2 缓存，直接惠及访存受限的 decode 阶段的性能。

NVIDIA 的 Grace Blackwell GB200 NVL72 系统是一个机架级平台，配有 36 个 Grace CPU 和 72 个 Blackwell GPU，它让单个逻辑 decode 单元拥有数十 TB 的内存用于 KV 缓存。这套硬件拥有约 30 TB 的统一内存，非常适合把超大上下文保留在内存中。这些上下文的规模可以达到数百万 token。

有了这样的平台，统一内存占用会很大。然而，对于延迟敏感的 decode，你仍然希望活跃的键与值驻留在 GPU HBM 中。因此，你应把 Grace CPU 内存（LPDDR5X，而非 HBM）用作较低层级的缓存或用于存放非常旧的 token。当上下文超出可用 HBM 时——即便是在像 NVL72 这样的系统上——prefill 与键值卸载仍然重要。

简而言之，宏观层面的分离应当与微观层面的优化相配合，才能充分实现最高的推理性能。像 FlashMLA/ThunderMLA 这样的高级 decode 核函数、高效的内存布局（分页缓存等），以及最新的 GPU 架构，将带来高效且可扩展的 decode。

## Prefill 与 Decode 之间的快速 KV 缓存传输

分离式推理的一个关键要求是快速高效地把 KV 缓存从 prefill worker 传输到 decode worker。如果这一传输不够快，那么并行化 prefill 与 decode 所节省的任何时间，都可能因等待数据移动而损失掉。

在本节中，我们讨论用于最小化传输开销的技术。随后我们描述系统如何利用高速互连并避免额外的 KV 拷贝来实现这一交接。

### KV 缓存大小

prefill 的输出主要由所有提示 token 的 KV 缓存构成。这可能是大量数据。设想一个具有 _L_ 层的模型，每层有 _h_ 个维度为 _d_ 的注意力头，以及一个含 _N_ 个 token 的提示。KV 缓存大小大致为 2 × L × N × (h × d)，其中因子 2 对应键和值两者。

实际大小取决于精度（FP16 对比 INT8 等）与模型细节，但它很大。例如，一个 40 层、16 个大小为 64 的头、提示为 1,000 token 的模型，会产生约 40,000 个 KV 向量。这可能是数百 MB 的数据。如果 token 数为 5,000，则大 5×。

若以朴素方式在网络上传输这么大量的数据，可能引入显著延迟。例如，一种朴素做法可能是把 KV 拷贝到 prefill worker 的 CPU 内存，然后通过 TCP 发送——甚至写入磁盘供 decode 进程加载。对于大型提示，这可能极其缓慢，达到数百毫秒量级。目标是把传输时间降到仅几毫秒。这让 prefill 与 decode 真正并行重叠。

> 要实现低延迟的 KV 数据传输时间，通常需要把小的 PagedAttention 块归并（collation）成更大的缓冲区，并用基于 GPUDirect RDMA 的通路而非 CPU 套接字来移动它们。

### 零拷贝 GPU 到 GPU 传输

现代分离式系统采用零拷贝 GPU 到 GPU 传输（zero-copy GPU-to-GPU transfer）技术。实践中，这涉及在高速网络架构上使用远程直接内存访问（remote direct memory access，RDMA）。例如，你可以用 InfiniBand 在机架/节点之间传输——或在单节点（多 GPU）平台内用 NVLink/NVSwitch 进行直接 GPU 内存写入。这些方法在 GPU 之间直接发送数据，而无需通过 CPU 内存拷贝。

NVIDIA 用于推理的高性能 GPU 到 GPU 传输库称为 _NVIDIA Inference Xfer Library_（NIXL）。NIXL 提供一种插件式架构（如 NVLink、UCX 网络架构、GPUDirect Storage），用于零拷贝的 GPU↔GPU 与 GPU↔存储数据移动。

NIXL 简化了 RDMA 风格的传输，允许一个 GPU 在可用的高速网络架构（例如基于 InfiniBand 或 NVLink 的连接）上直接写入另一个 GPU 的内存。换句话说，prefill GPU 可以直接把 KV 张量注入 decode worker GPU 的内存。

基于 RDMA 的协议绕过 CPU，充分利用 GPU 互连带宽。像 NVIDIA Dynamo 以及开源的 vLLM 加 LMCache 集成这样的系统都依赖 NIXL。具体来说，它们使用 NIXL 通过 NVLink 或 RDMA 把 KV 张量直接写入远端 GPU 内存。现代 GPU 互连提供非常高的带宽，1 GB 的传输可以在几毫秒到数十毫秒内完成，具体取决于链路类型与竞争情况。

实践中，实现通过让数据传输与计算重叠来获得低传输时间。例如，借助 RDMA，一个 decode GPU 可以继续为其他序列生成 token，与此同时一个 prefill worker 正异步地把 KV 数据写入它的内存缓冲区。prefill 可以推送数据（RDMA 写推送模型），或者 decode 可以拉取数据（RDMA 读），取决于设计。无论哪种方式，数据通路中都不需要 CPU 参与。

快速 KV 传输的常见策略包括：prefill 侧推送、decode 侧拉取、共享内存（CUDA IPC）缓冲区、连接器/队列抽象，以及非阻塞重叠。下面逐一讨论：

_prefill 侧推送_ prefill worker 在完成提示后，发起一次 RDMA 写，把 KV 数据直接写入 decode worker GPU 上一块预留的缓冲区。这可以非阻塞地完成；prefill 可以启动传输，然后在 DMA 于后台进行的同时转去处理其他工作。

_decode 侧拉取_ 或者，decode worker 在准备开始 decode 时，可以直接从 prefill GPU 的内存进行 RDMA 读。推送或拉取都能达到相同的最终结果（无 CPU 拷贝）。一些实现可能偏好推送，以把协调工作卸给发送方；另一些则偏好拉取，让接收方控制时机。

_共享内存（IPC）缓冲区_ 如果 prefill 与 decode 恰好在同一台机器上（同一服务器中的不同 GPU），它们可能使用 CUDA 进程间通信来共享一个内存句柄，甚至一个 PCIe bar，实际上就是在同一主机上用 NVLink 或 NVSwitch 进行拷贝。这是零拷贝传输的本地变体，无需经过网络。

_连接器/队列抽象_ vLLM 的实现把传输机制抽象在一个逻辑接口（一个 Pipe 或 LookupBuffer）之后。prefill 进程把 KV 放入这个缓冲区或发出其可用信号，decode 侧则取回它。在底层，这可以使用 RDMA，甚至一条高性能的发布-订阅消息（在 Dynamo 的情形中用 NATS 传递控制信号）。关键在于把逻辑交接与传输解耦，以便可以插入不同的传输方式（RDMA、共享内存等）。

_非阻塞重叠_ 如前所述，优化过的系统会把 KV 传输与正在进行的 decode 计算重叠。例如，Dynamo 的 decode worker 继续为其他请求生成 token，与此同时一个 prefill worker 正为一个新请求把 KV 数据写入它的 GPU 内存。这隐藏了大部分传输延迟。因此，你可以把一次约 5 ms 的 KV 传输与 decode 计算重叠，从而给请求首个生成 token 增加的净延迟几乎为零。
