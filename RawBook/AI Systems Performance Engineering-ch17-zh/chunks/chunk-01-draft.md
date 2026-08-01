# 第 17 章 面向推理的分离式 Prefill 与 Decode 扩展

正如前面章节所述，LLM 推理可以分为两个截然不同的阶段：prefill 阶段与 decode 阶段。prefill 阶段处理输入提示词，为该提示词生成模型内部的键值（key-value，KV）缓存，而 decode 阶段则利用这些缓存值逐个生成输出 token——在推测解码（speculative decoding）的情形下则是一次生成若干个。

这两个阶段的性能特征存在根本差异。prefill 阶段是计算受限（compute bound）的，需要对可能多达数千个 token 并行执行大量矩阵乘法，消耗可观的 FLOPS。相比之下，decode 阶段是访存受限（memory bound）的，每生成一个 token 都要读取庞大的 KV 缓存、写入新值，对内存带宽形成压力。简而言之，prefill 是高吞吐量、可并行的工作负载，而 decode 是顺序执行、对延迟敏感的工作负载。

早期的 LLM 服务（serving）系统把这两个阶段当作同一硬件上的单个单体（monolithic）流水线来处理。因此，它们通常通过请求批处理（request batching）优先保证吞吐量，从而偏向 prefill 阶段。然而，随着交互式应用的兴起，诸如首 token 时延（time to first token，TTFT，即所有 token 的 prefill 延迟）与每输出 token 时延（time per output token，TPOT，即每个 token 的 decode 延迟）之类的延迟指标，变得与原始吞吐量同等重要。当单个基于 GPU 的推理引擎同时服务这两个阶段时，很难同时优化 TTFT 与 TPOT。

批量处理大量请求会提升吞吐量，但会恶化 TTFT，因为每个请求都要等待最慢的那次 prefill。它同样会影响 TPOT，因为 decode 步骤会积压在新提示词的 prefill 之后。

单体推理系统必须在两者之间取舍：要么改善（降低）首 token 时延，代价是后续 token 生成变慢；要么改善（提高）每 token 吞吐量，却让新请求承受很高的初始延迟。在极端情况下，一个长提示词可能完全占满 GPU，从而阻塞其他用户所有的提示词 prefill 工作。而一旦开始解码，逐个 token 的处理方式又会让 GPU 的核心在每次 token 生成之间处于空闲状态。

为解决这些问题，研究者与工程师们开始寻找将两个阶段解耦的方法。其关键洞见在于：prefill 与 decode 实际上并不需要运行在同一硬件上——甚至不需要运行在同一类型的硬件上。

分离（disaggregation）prefill 与 decode 阶段，意味着把它们分派到各自专门针对该特定阶段需求而优化的不同资源上。这一思路由 DistServe 论文中的系统率先提出，该工作证明：通过消除两个阶段之间的干扰（interference），可以同时满足 TTFT 与 TPOT 的严格延迟要求。

DistServe 的评估表明，与不做 prefill/decode 分离的最先进基线相比，它有望在严格的延迟服务级目标（service-level objective，SLO）内多服务 7.4× 的请求。因此，业界框架开始尝试分设独立的 prefill 与 decode 服务器。

开源库 vLLM 结合 LMCache 及其他组件引入了分离式运行。NVIDIA 的 Dynamo 实现了带动态路由与自动伸缩（autoscaling）的分离式 prefill 与 decode，并公开记录了运维细节。众多服务商与开源框架都在实现或评估分离方案。例如，为满足严格的延迟 SLO，据报道来自 OpenAI、Meta 与 xAI 的工业级服务系统均已采用这种分离式方法。因此，分离式 prefill 与 decode 已成为大规模 LLM 推理的标准做法。

在超大规模（ultrascale）下，大型推理部署可能涉及数十万乃至数百万块 GPU，服务数十亿次请求。在这类环境中，分离带来的成本与性能收益是巨大的。

通过拆分工作负载，你可以孤立地优化每个阶段，避免其中一个成为另一个的瓶颈。本章余下部分将探讨如何在极端规模下设计并运行分离式 prefill/decode 推理系统。

在本章中，我们将探讨用于在 prefill 与 decode 工作节点之间路由请求的调度算法、在重负载下维持服务质量（quality of service，QoS）的技术，以及让这种分离得以高效运作的机制。我们将从高速互连（interconnect）一直讲到专用的解码核函数。我们还将讨论为每个阶段使用不同 GPU 类型的异构硬件策略。

## 为什么要做 prefill-decode 分离？

现代交互式 LLM 服务通常将 p99（99% 的请求）的 TTFT 延迟目标设定为 < 200–300 ms。如果不把 prefill 工作分离出来，这几乎无法保证，因为对 LLM 服务采用“一刀切”的方式会白白浪费大量性能。

作为参考，MLPerf v5.0（2025 年）针对 Llama2 70B（700 亿参数）的推理基准，把 p99（第 99 百分位）SLO 设定为 ~450 ms TTFT 与 40 ms TPOT 延迟。对于 Llama 3.1 405B（4050 亿参数），该基准的目标为 ~6 秒 TTFT 与 175 ms TPOT。具体而言，这些 SLO 反映的是 Llama 2 Chat 70B 与 Llama 3.1 405B Instruct 的 p99 TTFT 与 p99 TPOT 目标。

设想这样一个场景：一个用户的请求带有极长的提示词，长度达数千个 token 量级——而另一个用户的请求提示词非常短。在没有分离式 prefill 与 decode 的情况下，如果这些请求大约在同一时间到达，长提示词的 prefill 计算就会长时间阻塞 GPU。

没有分离，带短提示词的第二个请求甚至在开始其 decode 之前，就得不必要地等待很长时间。这被称为*干扰*，因为一个请求的 prefill 工作延误了另一个请求的 decode 工作。图 17-1 在连续批处理（continuous batching）的背景下展示了 prefill 与 decode 之间的干扰。

在简单的 FIFO 调度策略下，长提示词会放大所有人的尾延迟（tail latency）。一般而言，位于队列前端的长 prefill 或计算量大的 prefill，会阻塞排在其后的更短、更轻量的请求。这被称为*队头阻塞（head-of-line blocking）*，会导致利用率低下、延迟离群值以及终端用户的不满。

在灵活的分离式架构中，可以把大型提示词的 prefill 发送到专用的、面向计算优化的 prefill 工作节点池，而轻量提示词的 prefill 则可以直接发送给 decode 工作节点——绕过 prefill 工作节点。这种灵活性让较短的 token 不必遭受队头阻塞。这样既最大化了总体吞吐量，又最小化了延迟尾部效应。

![图 17-1. 由同置（colocated）在同一 GPU 上运行的 prefill 与 decode 所引起的干扰（来源：https://oreil.ly/GRkHs）](AI%20Systems%20Performance%20Engineering-ch17_images/figure-17-1.png)

### 分离的优势

分离有两大主要优势：减少干扰，以及针对各阶段的专门优化。下面我们逐一讨论。

有了分离，prefill 任务不再与 decode 任务在同一设备上争抢资源。一个正忙于生成大量 token 的 decode 工作节点，不会妨碍另一个用户的提示词被处理，反之亦然。

为每个阶段配备专用资源，意味着一个长提示词的计算不会阻塞另一个用户的 token 生成。在实践中，这带来了更可预测的延迟。图 17-2 展示了同置与分离式 prefill 和 decode 之间的对比。该实验在 DistServe 论文以及作者随后发布的博客文章中有更详细的描述。

![图 17-2. 同置与分离式 prefill 和 decode 之间的对比（来源：https://oreil.ly/GRkHs）](AI%20Systems%20Performance%20Engineering-ch17_images/figure-17-2.png)

此处，SLO 被设定为 P90 TTFT 0.4 秒、P90 TPOT 0.04 秒（即图 17-2 中的水平线）。同置系统在给定的 TTFT 延迟界限内只能支撑约 ~3 每秒请求数（requests per second，RPS）的有效吞吐量（goodput）。而在给定的 TPOT 延迟界限内只能维持 1.6 RPS。因此，由于必须同时满足 TTFT 与 TPOT 两项延迟 SLO，同置配置的有效吞吐量仅为 1.6 RPS。

将两个阶段分离，并分配两个 prefill 工作节点（两块 GPU）与一个 decode 工作节点（一块 GPU）——即所谓的*2P1D 配置*——之后，prefill 与 decode 工作节点的总体 RPS 都优于使用单块 GPU 的同置配置。具体而言，prefill 工作节点达到约 ~5.6 RPS，decode 工作节点在三块 GPU 上合计达到约 ~10 RPS。因此，2P1D 配置的有效吞吐量为每块 GPU 3.3 RPS。

每块 GPU 3.3 RPS 的计算方法是：取 prefill 工作节点的 RPS（5.6 RPS × 2 = 11.2 RPS）与 decode 工作节点的 RPS（10 RPS）中的最小值。这样在全部三块 GPU 上总计为 10 RPS。因此，我们必须用该 RPS 除以 GPU 数量，此处为 3。在所配置的 SLO 下，该系统的有效吞吐量结果为 10 RPS ÷ 3 GPUs = 每块 GPU 3.3 RPS。

> 在此对比中，decode 侧的改进主要影响每 token 延迟。与此同时，prefill 隔离主要改善首 token 时延。只有同时满足两项 SLO 才计入有效吞吐量。

这种隔离同样能改善尾延迟。从经验来看，采用分离的系统表现出更紧凑的延迟分布——并避免了单体系统中常见的长尾。通过消除跨阶段干扰，每个阶段都能更可靠地满足其 SLO——且一致性更可预测。

此刻，你也应该会问：“3× 的成本值得换取 2× 的提升吗？”这样问是对的。要让该方案更具成本效益还需要额外的调优，但它指明了收紧延迟分布、提升有效吞吐量的正确方向。你需要根据自己的工作负载来决定。分离是满足有效吞吐量 RPS 需求的一种热门选择。

针对各阶段的专门优化，让每个阶段都能使用最适合自己的硬件与并行方式。例如，prefill 阶段是计算受限的。因此，你通常会提高张量并行（tensor parallelism，TP）度，以在高 FLOPS 的 GPU 上逼近峰值 FLOPS。此外，现代 GPU 提供了更低精度的模式（FP8 与 FP4），可为计算量大的 prefill 阶段提升吞吐量。

> 对于权重，你应优先选择 FP4；在经过验证的情况下，激活值也应如此。许多技术栈对权重使用 FP4、对激活值使用 FP8。这些降低的精度有助于最大化吞吐量、最小化 HBM 占用——且精度损失极小。这些精度受到现代软硬件栈的支持，包括 NVIDIA Tensor Core 与 Transformer Engine。

相比之下，decode 阶段是内存带宽受限的，并且会受到跨 GPU 同步开销的困扰。因此，它在很少甚至不使用张量并行（通常 TP=1）时效率最高，因为它更依赖融合核函数（fused kernels）来提高算术强度（arithmetic intensity）——以及依赖高内存吞吐量的 GPU。

在单体系统中，你不得不为两个阶段选定同一种 GPU 与同一种并行策略，这对其中至少一个阶段而言是次优的。而分离则让你能够独立地调优每个阶段，以获得最高效率。

拆分阶段还为异构集群（heterogeneous cluster）打开了大门——在异构集群中，不同的 GPU 类型被分派给 prefill 与 decode 角色，以获得最优的性价比。例如，为提示词 prefill 使用面向计算优化的 GPU、为 token 生成使用面向内存优化的 GPU，其每美元吞吐量可以优于同构部署。

> 在实践中，最新的 GPU 通常同时具备更高的 FLOPS 与更大的 GPU 显存。因此，直接为 prefill 与 decode 都使用最新一代 GPU 既诱人又更为常见。但要知道，异构是降低成本的一个可行选项。

本章稍后将探讨异构集群的思路。我们会展示：为提示词使用高端 GPU、为生成使用更便宜的 GPU，如何在大规模下带来可观的成本节省。

总而言之，分离消除了交叉干扰，并使每个阶段都能得到专门的处理。预期的效果包括：更紧凑的延迟分布——因为不再有由提示词大小不匹配所导致的长尾、在延迟约束下更高的吞吐量（有效吞吐量），以及更好的整体资源利用率。

> 你应使用剖析工具（如 NVIDIA Nsight Systems）来识别 prefill 与 decode 阶段中的瓶颈。这些工具可以跨不同的工作节点追踪 GPU 核函数与 RDMA 传输。这有助于验证 decode 核函数是否完全与通信重叠等。

接下来，我们讨论如何真正实现一个分离式服务系统，包括系统架构、通信，以及决定如何充分利用分离式集群的调度策略。

### 分离式 prefill 与 decode 集群池

在分离式部署中，我们维护两个（或更多）工作节点池，使得一组 GPU 专用于 prefill 提示词处理，另一组专用于 token 生成。这些工作节点可以位于数据中心内不同的节点或机架上——如果互连足够快，甚至可以位于不同的数据中心。（注：把 prefill 与 decode 保持在同一数据中心内，是实现现实可行 SLO 的务实设计选择。）

工作节点池之间通过网络通信，把 prefill 产生的模型 KV 缓存移交给将要执行 decode 的那块 GPU。一个调度器（或路由器）负责协调这一通信。

设想这样一种配置：模型权重被加载到两组 GPU 服务器上。一组是 prefill 工作节点，负责处理提示词并计算 KV 缓存。另一组是 decode 工作节点，利用 prefill 工作节点生成的 KV 缓存来处理 token 生成。

这两组工作节点通常使用高速互连（如 NVLink/NVSwitch 与 InfiniBand）以及基于 RDMA 的零拷贝 GPU 到 GPU 传输进行通信。在实践中，这些传输使用 GPUDirect RDMA 或 UCX，并且可以在 Nsight Systems 中与 CUDA 核函数、NVLink 活动、存储指标以及 InfiniBand 交换机指标相关联，以进行端到端验证。

> 对于基于超级芯片的 NVL 互连（如 Grace-Blackwell、Vera-Rubin 等），应使用 NVIDIA Multi-Node NVLink（MNNVL），为 TP decode 保持启用 NVLink 优先的集合通信，并在可用时为 AllGather 与 ReduceScatter 集合通信启用 SHARP。

当系统收到一个新请求时，通常由 decode 工作节点来接收。这被称为*以 decode 为中心的设计（decode-centric design）*。之所以更倾向于这种做法，是因为 prefill 工作节点已经被 KV 计算压得计算受限了。

通过让 decode 工作节点处理客户端 I/O、路由与会话状态管理，推理系统避免了让 prefill 工作节点过载。此外，把请求入口（ingress）集中在 decode 节点上，简化了网络管理、自动伸缩与策略执行。

> 这只是 prefill-decode 的一种系统架构风格，其中 decode 工作节点是所有请求的集中式入口。NVIDIA Dynamo 推理系统采用了这种架构。另一种常见架构是使用专用的、集中式的 API 路由器，把请求路由到 prefill 或 decode 工作节点。然而，这需要在系统中增加一个额外的活动部件，以及路由器与 prefill/decode 工作节点之间的额外协调——还有额外的扩展与延迟方面的考量。

decode 工作节点收到请求后，会决定是自己执行 prefill，还是把它“卸载（offload）”给 prefill 工作节点池。如果它决定卸载给 prefill 工作节点池，decode 工作节点稍后会收到返回的 KV 结果，然后继续解码并生成下一个 token。

下面是一段简化的 NVIDIA Dynamo 集群配置片段，它定义了两个角色，二者使用 NVIDIA 的 Inference Xfer Library（NIXL）进行基于 GPUDirect RDMA 的 KV 缓存传输，从而让一块 GPU 能够通过网络直接写入另一块 GPU 的显存：

```
roles:
  - name: prefill_worker     # prefill 工作节点角色
    model_path: models/llm-70b
    instance_count: 4        # 4 个 prefill 工作节点
    gpu_type: B200           # B200 Blackwell，计算受限的 prefill
  - name: decode_worker      # decode 工作节点角色
    model_path: models/llm-70b
    instance_count: 8        # 8 个 decode 工作节点
    gpu_type: B300           # B300 Blackwell Ultra，高显存的 decode
```

> NVIDIA 的 Rubin CPX 加速器是 prefill 工作节点的另一个选择。Rubin CPX（CP 代表 “context processing”，即上下文处理）专为诸如 prefill 之类的计算受限工作负载而设计。Rubin CPX 标志着 NVIDIA 从“通用加速计算”GPU 转向专用芯片——这些芯片针对诸如推理之类更广泛 AI 工作负载中的某个特定阶段（如 prefill）进行优化。

在此配置中，我们有四个使用 B200 GPU 的 prefill 工作节点（其算力足以应对计算量大的 prefill）以及八个使用 B300 GPU 的 decode 工作节点（其高 HBM 容量可应对访存受限严重的 decode）。混用 B200 与 B300 有助于匹配它们各自的 FLOPS 与 HBM 容量特性，同时最小化成本。两个角色都将使用 NIXL 与 GPUDirect RDMA 来传输 KV 缓存块。NIXL 为通过 NVLink 与 RDMA 网卡进行的 GPU 到 GPU 数据搬运抽象了传输层。它还为 GPUDirect Storage 提供了连接器，使得 KV 缓存页可以从不同的存储层级读取（或写入）。

在底层，当该系统运行时，每个 decode 工作节点都会注册其 GPU 显存的一块区域，以便 prefill 工作节点可以使用 RDMA 直接写入其中。通常，诸如 NIXL 描述符之类的内存注册元数据会在启动时或首次接触时交换。这样，对于每个远程 prefill 任务，只需发送一个小标识符，而不是完整的内存地址结构。

例如，Dynamo 使用 etcd 进行工作节点发现与租约管理。工作节点会把必要的内存句柄注册到路由器或控制平面，以便对端在需要时能够获取描述符。prefill 工作节点会在首次使用时检索它们。这样，一个 prefill 请求只需包含目标 KV 缓冲区的一个 ID，从而使控制消息保持轻量。

此外，NVIDIA Dynamo 的 NIXL 实现为推理数据搬运提供了高吞吐量的 RDMA 与存储抽象，并包含用于 NVLink、基于 UCX 的互连以及 GPUDirect Storage 的插件。因此，prefill 工作节点可以把 KV 块直接写入 decode GPU 的显存。

> 在 prefill 与 decode 使用不同 TP 布局的混合并行部署中，你需要在 NIXL 读取之后立即在 decode 侧执行一次布局变换。这样，KV 页就能匹配 decode 核函数所期望的布局。与网络传输相比，这一变换的延迟微不足道，而且它避免了重新 prefill。

这种架构使每个阶段的扩展相互解耦。举例来说，如果你发现由于大量并发的长提示词导致 prefill 成为吞吐量瓶颈，你可以增加更多 prefill 工作节点，以提升提示词处理能力。

又比如，如果由于许多用户生成很长的输出导致 decode 成为瓶颈，你就会横向扩展 decode 工作节点。由于 decode 与 prefill 是分离的，扩展其中一个不会直接干扰另一个。

诸如 NVIDIA Dynamo 之类的系统支持动态的、可在运行时配置的分离，使得你能够即时添加或移除 prefill 工作节点——而无需停止集群。新的 prefill 工作节点只需注册，就能开始从队列拉取任务。如果某个 prefill 工作节点因任何原因离开集群（崩溃、重启、自动伸缩事件、网络分区等），decode 工作节点会临时执行更多本地 prefill 来加以弥补。

NVIDIA Dynamo 的分布式运行时使用 etcd 进行工作节点发现与租约管理。其 Planner 组件可以通过撤销租约或启动会被自动发现的新工作节点来扩缩工作节点。当负载经常波动时，这种动态灵活性在超大规模下至关重要。此时，你会希望按需在角色之间调换工作节点。

prefill 工作节点（即*提示词服务器*）是专门用于执行请求初始提示词 prefill 处理阶段的计算节点。本节讨论 prefill 节点是如何架构以高效处理繁重计算的——以及它们在负载下如何为 KV 缓存的填充在延迟与吞吐量之间取得平衡。

由于 prefill 工作负载是计算密集型的，prefill 节点应使用高 FLOPS 的 GPU，并针对大型矩阵乘法进行优化。每个 prefill 任务都会把 _n_ 个输入 token 送入所有模型层。

prefill 工作节点会并行使用数千个 GPU 线程——如果可用，还会跨越许多 GPU 节点。它们使用我们熟悉的并行技术（包括张量并行与流水线并行（pipeline parallelism））来降低 TTFT。

并且还会为该提示词分配 KV 缓存。随后，这份 KV 缓存会被传输给 decode 工作节点——我们稍后就会看到。

prefill 会用模型参数以及模型对提示词输入执行前向传播时的工作激活值来填充 GPU 显存。一旦 KV 缓存被创建，它就会立即被发送给 decode 工作节点。KV 缓存不会在 prefill 节点的显存中长久驻留。

如果模型极大或提示词极长，由于内存限制，prefill 可能需要在多块 GPU 之间进行张量或并行切分。prefill 服务器应在其并行策略（数据、张量、流水线、专家（MoE）以及上下文）上保持灵活，以满足延迟目标。

一些推理框架会预先分配一大块 GPU 显存供 prefill 用作工作空间。这减少了整体的内存碎片以及缓冲区分配时间。

在重负载下，你面临一个根本性的权衡：是最小化每个单独提示词的 TTFT，还是最大化整体的每秒请求数（RPS）或降低 TPOT。

分离式系统通过为延迟优先方法与吞吐量优先方法支持不同的调度策略，来应对这一权衡。下面我们分别描述这两种方法：

_延迟优先方法_ 为降低 TTFT，prefill 节点应在提示词一到达时就立即处理——且几乎不做批处理。在这种模式下，你避免了等待其他请求来凑满一个批次。因此，每个提示词都会立即开始执行并尽快完成——前提是集群中有可用的 GPU。

这种延迟优先方法的缺点是 GPU 利用率较低，因为你使用的是小批次或不批处理。因此，GPU 常常处于空闲状态，而在给定的集群规模下，你的系统所能服务的并发请求也更少。在这种情况下，你可以要么为 prefill 集群超额配置容量，要么使用大小为 1 的极小批次，以保证请求的严格延迟 SLO。

_吞吐量优先方法_ 如果峰值吞吐量（或 RPS）与最小的 TPOT 是你的优先事项，你应把提示词批量组成更大的分组，以充分加载每块 GPU。通过把 8–32 个提示词累积进单个批次，你提高了算术强度，并让 GPU 计算单元保持繁忙。这会提升总体吞吐量。

吞吐量优先方法的缺点是，每个请求都会产生一段批处理延迟，其长度等于收集该批次所花的时间。批次越大，延迟越长。

对于追求极致吞吐量的推理系统配置，你可以选择使用数据并行（data parallelism）或流水线并行，为每个请求分配多块 GPU。

采用数据并行时，整个模型会在每块 GPU 上复制一份。批次被切分成小批次，分散到各块 GPU 上。每块 GPU 用其完整的模型副本对自己那部分数据执行一次前向传播。随后，从所有 GPU 汇聚输出，得到最终结果。

数据并行汇聚了所有 GPU 的内存带宽与算力，以提升每批次的性能。然而，它把最大并行度降低到 GPU 总数 ÷ 每请求 GPU 数。这降低了你整体的并发请求容量。如果系统为单个请求使用了过多的 GPU，就可能让资源闲置。这会在吞吐量与并发之间造成失衡。

流水线并行把模型的各层划分为分布在不同 GPU（如 GPU 0 与 GPU 1）上的顺序阶段。一旦 GPU 0 完成 microbatch 0 的阶段，它就把激活值转发给 GPU 1，并开始 microbatch 1 的阶段 1。这种流水线（assembly-line）模式让所有 GPU 都忙于处理不同的工作块。

流水线并行提升了每批次的吞吐量，但如果 microbatch 大小或阶段切分没有仔细平衡，它会增加 GPU 间的通信开销以及流水线“气泡（bubble）”。

归根结底，在集群规模固定的情况下，你每多投入一块 GPU，都会提升吞吐量，但会减少你能同时处理的请求数。你当然可以随时横向扩展 GPU 集群，但在假定集群规模固定的前提下，你应根据延迟 SLO 还是吞吐量 SLO 对你的用例最重要来选择配置。

感知型调度策略来平衡这些因素。例如，它们可能保证单请求执行——而不做批处理——除非负载高到足以让合并少量请求也不会违反 TTFT 目标。

许多集群设计在调度器中纳入了 SLO 约束。例如，如果 p90 TTFT 必须 ≤ _X_ ms，系统就会为典型的提示词大小选择仍能满足该 SLO 的最大批次大小或并行策略。

另一种策略是自适应批处理窗口。例如，在低负载时，它可以使用大小为 1 的批次立即运行请求。而在较高负载时，系统可以允许把在一个很小时间窗口（如 2–10 ms）内到达的请求组成 microbatch。这样，轻微的延迟就能换来 GPU 利用率的大幅提升——但仅在需要且可容忍时才这么做。

许多推理引擎的 prefill 工作节点偏向延迟。系统往往会尽快执行提示词任务，甚至容忍一定程度的 GPU 利用不足，因为快速产出首个 token 能显著改善用户体验。

为平均负载配置超出所需的 prefill 容量是很常见的做法。这样，prefill 集群就能吸收提示词的突发而不产生延迟尖峰。在下一章中，我们将讨论即时重新平衡资源的自适应机制，使 prefill 与 decode 工作节点都不会随时间推移而成为瓶颈。

像 Kubernetes 这样的现代编排器可以自动扩缩每一层。例如，如果 prefill 的 GPU 利用率持续偏高而 decode 偏低，编排器就可以触发一次自动伸缩事件来添加 prefill pod（或节点）——甚至可能移除一些 decode pod/节点。

> 这类自适应扩缩通常借助诸如 prefill 队列长度之类的指标来实现，以辅助驱动决策。

另一个选择是实现优先级队列，使短提示词被调度到一条单独的、批处理更少的快车道上。长的、可批处理的提示词则进入面向吞吐量优化的队列。NVIDIA Dynamo 在调度中支持延迟类别。你可以通过给请求打标签、并为每个类别设置不同的批处理窗口来模拟这一点。

关键要点在于，prefill 工作节点优先追求快速周转。分离让我们能够做到这一点而不损害 decode 性能，因为 decode 运行在另一组工作节点上。我们或许会在低流量时段“浪费”一些 prefill GPU 周期，但在高峰流量时段维持了较低的 TTFT。对交互式服务而言，这是一个值得的权衡。

decode 工作节点（即*生成服务器*）专门负责自回归的 decode 阶段。一旦某个提示词的 KV 缓存就绪，它就会被发送到一个 decode 工作节点，该节点利用 KV 缓存尽可能快地产出剩余的输出 token，以维持较低的 TPOT 延迟。

如图 17-3 所示，如果一个请求最初被路由到 decode 工作节点，它必须首先借助分离式路由器（disaggregated router）来决定 prefill 应在本地还是远程执行。如果它决定远程执行 prefill，就会把该 prefill 请求推入一个 prefill 队列，等待 prefill 工作节点来领取。

prefill 工作节点持续从 prefill 队列拉取任务，读取前缀缓存（prefix caching）中缓存的任何 KV 块，并计算 prefill 运算。随后它把 KV 块写回给 decode 工作节点，由后者完成解码。

![图 17-3. prefill 工作节点设计：读取前缀缓存 → 计算 prefill → 为 decode 写入 KV 缓存](AI%20Systems%20Performance%20Engineering-ch17_images/figure-17-3.png)

decode 工作节点的设计聚焦于高效处理大量并发的序列生成——以及管理 KV 缓存的内存占用。在本节中，我们描述 decode 服务器如何借助连续批处理与巧妙的内存管理技巧等技术来实现高吞吐量。这些有助于降低 TPOT 延迟并提升可扩展性——尤其是对于长序列。让我们先从详述两个阶段之间的 KV 缓存传输开始。

在 prefill 与 decode 工作节点之间尽可能高效地搬运 KV 缓存数据。通过使用诸如 NIXL（在第 4 章中介绍）之类的库进行直接的 GPU 到 GPU 传输，我们可以避免 CPU 介入并利用非阻塞操作。这样，当一块 GPU 正在传输 KV 数据时，它也能服务其他前向传播请求，而无需等待传输完成。

设想一个到达 decode 工作节点的用户请求。在这种情况下，decode 工作节点的调度器会分配必要的 KV 块，并向 prefill 队列添加一个远程 prefill 请求。该 prefill 请求包含那些 KV 块的标识符。图 17-4 展示了这一交互。

![图 17-4. 使用 NIXL 在 prefill 与 decode 工作节点之间传输 KV 缓存数据；在 RDMA 之前把多个 PagedAttention 块合并为 ~128-token 的负载（注：vLLM 在 CUDA 上默认每块 16 个 token）](AI%20Systems%20Performance%20Engineering-ch17_images/figure-17-4.png)

prefill 工作节点使用 NIXL，通过所选的传输方式执行直接的远程 GPU 显存读写。这避免了 CPU 拷贝，并实现了非阻塞的推进。一旦 prefill 工作节点完成 prefill 请求，decode 工作节点的调度器就会向其自身的 decode 流水线添加一个相应的 decode 请求。这让计算与数据搬运得以无缝重叠。务必使用预先注册的、具有较大固定（pinned）窗口大小的对端内存，以最小化重复注册的抖动。你可以通过 Nsight Systems 时间线来验证零拷贝传输的重叠情况。

在验证端到端数据搬运与重叠时，建议使用带追踪标志的 Nsight Systems 进行剖析。对于 InfiniBand 链路遥测，添加 --nic-metrics=true 以获取 HCA/NIC 计数器，添加 --ib-switch-metrics-device=<GUIDs> 以获取交换机计数器。这些会捕获交换机指标并采样主机/设备活动。综合起来，这将产生相互关联的 CUDA 核函数、UCX 活动、存储指标以及网络行为。下面是一条统一的命令，它启用 CUDA/UCX 追踪并收集 CPU 活动、GPU 指标、存储指标以及 InfiniBand 交换机遥测：
