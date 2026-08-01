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

```
nsys profile --trace=cuda-hw,osrt,nvtx,ucx,gds \
  --trace-fork-before-exec=true \
  --cuda-event-trace=true \
  --cuda-graph-trace=node \
  --cuda-memory-usage=true \
  --sample=cpu \
  --gpu-metrics-device=all \
  --nic-metrics=true \
  --ib-switch-metrics-device=<GUIDs> \
  --storage-metrics --storage-devices=all \
  --gds-metrics=driver \
  -o nsys_reports/prefill_decode \
  <your_launch_here>
```

> prefill 与 decode 引擎有可能采用不同的并行（例如张量并行）布局。在这种情况下，系统可以在接收端插入一个布局转换核函数——在 NIXL 读取之后（且在数据被使用之前）——将每个 KV 块重新对齐为 decode 工作节点所期望的布局。

为 _迭代级批处理_（iteration-level batching）。与执行大型矩阵-矩阵计算的 prefill 阶段不同，decode 阶段执行的是许多小型计算，因为每次生成新 token 都是一次相对较小的向量-矩阵计算——单个 token 被表示为一个向量。

为避免小规模、逐 token 工作负载导致 GPU 利用率低下，decode 工作节点可以将多个输入序列批处理在一起，从而在每次迭代中构成更大的矩阵-矩阵计算（例如同时生成多个 token）。这提高了每个 decode 任务的算术强度（arithmetic intensity）。

例如，假设有 32 个不同的文本生成请求正处于中途，都准备好生成各自的下一个 token。连续批处理（continuous batching）调度器不会分别计算 32 次单 token 的前向传播，而是将这些请求合并，执行一次前向传播，并行生成 32 个 token（每个序列一个）。

如此一来，矩阵乘法看到的有效批大小为 32，从而让 GPU 计算单元保持繁忙。挑战在于，并非所有序列都在完全相同的时刻请求 token。有些可能提早结束，有些则可能较晚开始。

请记住，你可以跨不同请求与用户进行批处理。如果多个序列在某一时刻具有相同的上下文长度，大多数现代推理服务器会自动将来自不同用户请求的 decode 步骤归为一组。这实际上是在动态执行批量解码。为最大化 decode 吞吐量，可以考虑此类方案。

> 在 vLLM 中，CUDA Graph 捕获的覆盖范围由 --max-seq-len-to-capture 这个参数控制。捕获尺寸通常会对齐到最大序列数。当某个序列超过此长度时，vLLM 会回退到 eager 模式。需要注意的是，此参数并不控制运行时批处理多少个序列。并发 decode 槽位的数量由 --max-num-seqs（一次迭代中序列数的上限）控制。请显式设置该参数，并配合 --max-num-batched-tokens，以获得可预测的内存使用。

连续批处理通过在每一步动态更新批大小，解决了序列在不同时间请求 token 的问题。具体来说，在每一步，连续批处理策略都会收集当下所有已准备好生成下一个 token 的序列，并将它们批处理为单个 decode 步骤。

如果在 decode 进行过程中有新请求完成 prefill 并准备就绪，它们会加入下一批，而无需等待一段任意长的空闲期。这与静态批处理（static batching）形成对比——后者会等到凑满一整批后才开始解码。

如果某个序列因命中文本结束 token 或达到序列长度上限而完成，它会立即从后续批次中移除。如果某个序列尚未就绪——也许它是一个仍在 prefill 中处理的长提示词——那么在它就绪之前不会被纳入。

实际上，采用连续批处理后，每次迭代的批大小可能会波动。不过，服务器始终试图最大化批大小，以纳入当时可用的所有序列——直到达到某个上限。这在最小化每 token 等待时间的同时实现了高利用率。

连续批处理确保 decode GPU 工作节点永不闲置。它们始终在处理可用的请求。通过让 GPU 保持繁忙——即便个别序列正在等待新 token——这在延迟约束下最大化了它们的吞吐量。

类似地，Microsoft 的 DeepSpeed 与 NVIDIA 的 TensorRT-LLM 推理引擎都实现了配合分页 KV 缓存的连续批处理（或称 in-flight 批处理），以在解码期间保持较高的 GPU 利用率。具体而言，DeepSpeed 会合并多个生成请求，而 TensorRT-LLM 则使用调度器跨流将解码任务归组。

在分离式 decode 集群中，连续批处理变得更加强大。由于 decode GPU 只负责生成，它们可以将 100% 的算力周期投入到这一连续循环中，而不会被大型的、定制化的提示词任务所打断。这带来了更平滑的吞吐量指标——尤其是在负载之下。

在高负载下，一个 decode 节点可能同时有数十甚至数百个序列处于活跃状态。它可以在每次迭代中批处理其中的大量序列。这将最大化硬件利用率。

而在低负载下，即便只有一个序列处于活跃状态，decode 工作节点也能立即生成 token，无需等待凑满一批。此时，GPU 在那一刻会利用不足，但该单一序列的延迟仍然很低。因此，连续批处理能够应对两种极端情况：在高并发时高效，在低并发时响应迅速。这在高吞吐量与低延迟之间取得了良好的平衡。

需要精心的调度与批处理，以避免浪费计算与内存。将短提示词与长提示词混在一批中会带来填充（padding）开销——相对于 token 数量，有时填充多达 50%。这会浪费稀缺的 GPU 与网络资源。

当你把不同长度的提示词批处理在一起时，每个较短的序列都必须填充到与最长序列一致的长度。这种填充引入了“空操作”（no-op）token，它们仍会消耗 GPU 算力周期、内存带宽以及 GPU 间或网络传输。在某些情况下，在常见的生成式 AI 工作负载中，填充可占到全部 token 的一半之多。这会显著降低推理效率。

一个直接的解决方案是根据序列长度将请求分到不同的桶（bucket）中。这样，每一批都包含大小相近的序列。使用诸如 0–512 token、513–1,024 token 等静态长度桶，可以固定批次边界并最小化填充开销。

vLLM 的 decode 调度器维护着一个由 SequenceGroup 实例组成的轮转池（每个提示词就是一个 SequenceGroup）。调度器按每次 decode 迭代固定的 token 预算推进每个组。一旦某个 SequenceGroup 处理完自己的分块，它就离开池，随后一个新的 SequenceGroup 加入池。这使得流水线持续保持满负荷——既不依赖静态填充桶，也不会让 GPU 利用不足。

这些批处理与调度技术，与采用独立 prefill 集群和 decode 集群的分离式 prefill-decode 部署高度契合。采用这种部署配置，独立的 decode 节点可以使用连续批处理等技术，在严格的 SLO 下最小化 TPOT 方差。与此同时，专用的 prefill 节点可以独立调优，以获得最大的输入处理吞吐量和最小的 TPOT。

NVIDIA 的程序化依赖启动（Programmatic Dependent Launch，PDL）与设备端发起的 CUDA Graph Launch（在第 12 章讨论过）用于降低每 token 的启动开销、重叠工作，并消除 decode 迭代之间的气泡。这些特性通常通过框架启用，而非在应用代码中手动开启。

在使用设备端启动的图时，请使用 cudaGraphInstantiateFlagDeviceLaunch 进行实例化，并将节点保持在单个设备上。使用 PDL 在一步（例如一次 decode 迭代）结束时重叠有依赖关系的核函数。这可以进一步削减每 token 的启动气泡。

通过结合长度分桶、连续批处理、分离、PDL 以及设备端发起的 CUDA Graph，vLLM、SGLang 和 NVIDIA Dynamo 等现代推理系统能够同时实现高吞吐量与低延迟——即便面对差异极大的提示词长度也是如此。而且它们做到这一点时，并不会影响资源效率或可扩展性。

> 在 vLLM 中，--max-seq-len-to-capture 控制 CUDA Graph 覆盖的最大序列长度。该值默认设置为 8192。在连续批处理中，vLLM 可能会填充到最接近的已捕获尺寸，因此应对齐 --max-num-seqs 与 --max-num-batched-tokens 以最小化填充浪费。CUDA Graph 有助于最小化针对常见序列长度反复重建 CUDA 图的开销。它并不直接决定运行时的批处理行为。如前所述，vLLM 中的运行时批处理由其 decode 调度器的动态 SequenceGroup 池管理。在生产环境中，建议将 --max-num-seqs 与 --max-num-batched-tokens 连同 --max-seq-len-to-capture 一起调优，以约束 HBM（KV）使用并减少连续批处理下的填充。

迄今为止所见的整个序列——包括先前已解码的 token——意味着 KV 缓存内存是 decode 工作节点的一项关键资源。每个序列都为每个 transformer 层以及每个过往 token 存储键（key）与值（value）张量。图 17-5 展示了一个在不同请求间共享 KV 缓存的示例。

![图 17-5. 在请求之间管理并复用 KV 缓存数据](AI%20Systems%20Performance%20Engineering-ch17_images/figure-17-5.png)

对于大模型和长序列，KV 内存随 token 线性增长，并取决于注意力布局和 dtype。一个实用的估算是 bytes_per_token = 2 x n_layers x n_kv_heads x head_dim x bytes_per_element。（注：其中的 2 x 对应每个 token 每层的键与值。）

以一个 Llama 类的 13B 模型为例，它有 40 层、40 个注意力头（头维度为 128），采用 FP16。对于使用标准多头注意力（multi-headed attention，MHA）的 4,096-token 上下文，KV 大小约为 0.819 MB/token，即总计约 3.36 GB。若使用 FP8 KV，则变为约 1.68 GB。

若使用带 8 个查询组（n_kv_heads = 8）的分组查询注意力（grouped-query attention，GQA），则 4,096-token 的 KV 在 FP16 下约为 0.671 GB，在 FP8 下约为 0.336 GB。而对于只有 1 个 kv 头的多查询注意力（multi-query attention，MQA），在 FP16 下约为 0.084 GB。

> 请务必始终使用模型实际的 n_layers、n_kv_heads、head_dim 和 KV 精度来计算，因为 FP8 和 FP4 会改变 bytes_per_element。

一个服务于大量并发序列的 GPU，除了模型权重之外，仅仅因为 KV 缓存存储就可能迅速逼近其内存上限。因此，请确保你的 decode 工作节点使用这样一种优化过的内存分配器，以防在大量长序列处理过程中耗尽 GPU 内存。以下是 decode 工作节点用来高效管理 KV 内存的策略：

_分页 GPU 内存分配器_ vLLM 的 PagedAttention 机制就是一个典型例子——它将 KV 缓存划分为固定大小的页，并能将不活跃的页换出到 CPU 内存。除 vLLM 外，SGLang 和 NVIDIA TensorRT-LLM 也实现了分页 KV 内存管理器。NVIDIA Dynamo 在这些技术之上构建。这些系统还使用 DRAM 和 NVMe 分层构建外部 KV 层级。并且它们可以在重计算与 I/O 之间进行调度，以平衡带宽。这在 LMCache 等项目以及其他类似的库和运行时中很常见。

_高内存 GPU 与自定义分配器_ decode 服务器常使用具有大 HBM 容量的 GPU（例如 Blackwell B200 的 180 GB HBM 和 Blackwell B300 的 288 GB HBM）来存储 KV 缓存。此外，vLLM 和 NVIDIA TensorRT-LLM 等系统使用优化过的内存管理器，以固定大小的页来分配 KV 内存，从而减少碎片、支持跨请求高效复用前缀，并管理数百个不同长度的序列。这样可以高效共享内存，而不会造成过多的碎片和浪费。

_KV 缓存卸载（例如换出）_ 当 GPU 内存被占满时，decode 工作节点可以将较旧的 KV 块卸载到 CPU RAM 或 NVMe 等更冷的存储。例如，如果某个序列生成了 1,000 个 token，但并非当下立刻都需要用到，那么一些较早 token 的 KV 就可以被移到主机 CPU 内存中。

当需要它们时，可以按需将它们重新调回 GPU 内存。当这些 token 用于注意力计算时，卸载会带来一点延迟代价，因此使用卸载（offload）时应当谨慎。decode 服务器通常会尝试预取或重叠数据传输，使得换入 KV 数据不至于对生成造成太大影响。

_上下文限制与压缩_ 一些部署会对解码 token 的数量（即输出序列长度）设置硬性上限。这是一种应用层的权衡，用于给 KV 缓存大小设定上限并避免无界增长。KV 压缩同样可以降低每 token 所需的 KV 内存。例如，以更低精度（FP16、FP8、INT8）存储 KV 可以大幅降低内存占用。

另一个例子是使用多查询注意力（MQA），其中各个头对每个 token 共享同一个 KV 向量。这会按头数成比例地减小 KV 大小。这是一种直接降低 KV 占用的模型架构改动。分组查询注意力（GQA）和 DeepSeek 的多潜在注意力（Multi-Latent Attention，MLA）同样有助于减小 KV 缓存的大小。

_分离中的内存层级_ 分离式设计的另一个优势是，decode 集群的 GPU 内存完全专用于存储模型权重 + KV 缓存。它也不必去处理大型提示词的 prefill 计算——而在单体（monolithic）服务系统中，这类计算会在 prefill 期间临时消耗大量额外内存。

每个 decode GPU 通常会加载完整的模型权重——除非使用了模型并行——然后将剩余内存用于 KV 存储。例如，如果模型权重占用了 GPU 的 70 GB 内存，而该 GPU 共有 180 GB，那么大约还剩 122 GB 可用于 KV。

这直接影响了该 GPU 上大约能同时处理多少 token × 序列。分离并不能消除 KV 内存问题，但通过分离角色，你可以选择在内存容量和内存带宽方面进行优化的 decode 节点类型。

在集群如此配置后，你需要决定何时将 prefill 卸载到 prefill 工作节点池——或何时在 decode 工作节点本地执行。卸载有开销，包括排队延迟、网络传输等。因此，只有在确实有助于降低延迟时才应使用它。这一决策由路由策略（routing policy）做出，下文将展开介绍。

### 分离式路由与调度策略

并非每个请求都需要卸载到 prefill 工作节点。事实上，在没有必要时这样做只会增加开销，却带不来多少收益。因此，分离式推理系统会使用路由策略来有条件地进行分离，仅在可能有帮助时才使用远程 prefill 路径。表 17-1 给出了路由策略的高层概览，包括 KV 感知路由（KV-aware routing）和前缀感知路由（prefix-aware routing）。

表 17-1. 路由策略的高层概览，包括前缀感知路由和 KV 感知路由

| 路由策略                   | 说明                                 |
| --- | --- |
| 轮询（round robin）        | 逐个依次路由到每个节点               |
| 最少请求（least requests） | 路由到活跃请求最少的工作节点         |
| 前缀感知（prefix aware）   | 使用请求的前缀来选择工作节点         |
| KV 感知（KV aware）        | 路由到 KV 缓存与请求最匹配的工作节点 |

分离式路由器（disaggregated router）会在最初接收请求的 decode 工作节点上，为每个新请求运行。它会迅速做出决策：要么在本地 prefill，要么在 prefill 工作节点池上远程 prefill。

将 prefill 卸载到 prefill 工作节点池的决策，可以基于与请求和系统状态相关的若干因素。常见的路由因子（routing factor）包括当前队列长度、GPU 内存可用量，甚至还包括专门化程度——因为某些 GPU 更适合特定的模型或提示词类型。

像 vLLM 的 KV 缓存感知路由器这样的高级路由器还会考虑缓存局部性（cache locality）。它们会把请求路由到缓存中已持有其部分前缀的 decode 工作节点。图 17-6 展示了一个 KV 缓存感知路由器如何依据从工作节点发出的 KV 缓存事件所接收到的数据，在系统中转发请求。

其目标是以最大化缓存命中（cache hit）并均衡负载的方式进行路由。表 17-2 总结了在典型分离式设计中影响路由决策的一些关键因素。

表 17-2. 影响路由器是否卸载 prefill 决策的因素

| 因素                | 说明                                                                 | 对路由决策的影响                                                                                                                                                     |
| --- | --- | --- |
| 提示词长度          | 输入提示词中的 token 数量（在任何前缀缓存之后）。                    | 长提示词 ⇒ 计算量更大 → 若长度超过阈值则卸载 prefill。短提示词 ⇒ 在本地处理。                                                                                        |
| 前缀缓存命中        | 提示词已存在于 decode 工作节点 KV 缓存中的程度（来自先前的请求）。   | 大量前缀缓存命中（提示词大部分已缓存）⇒ prefill 实际上更短且更偏访存受限 → 在本地处理。若无缓存命中（全部为新 token）⇒ 计算量大 → 卸载很可能有益。                   |
| prefill 队列长度    | 全局 prefill 队列中待处理任务的数量（即 prefill 工作节点有多繁忙）。 | 若队列很长（prefill 工作节点滞后）⇒ 避免卸载新请求（在本地处理）。若队列为空或很轻 ⇒ prefill 工作节点有余量 → 若满足其他条件则卸载。                                 |
| decode 工作节点负载 | 本地 decode 工作节点的当前负载（正在进行的 decode 任务等）。         | 若 decode GPU 正忙于处理许多 decode 流，卸载有助于并行（把 GPU 从繁重计算中解放出来）。若 decode 大多空闲而 prefill 队列积压，则在本地执行 prefill 以利用可用容量。  |
| 延迟 SLO 紧迫度     | 请求延迟要求的优先级或紧迫程度。                                     | 紧迫的低延迟要求 ⇒ 可能会卸载，以确保提示词尽快被计算（尤其是当本地 decode 繁忙时）。宽松的要求则可能只在本地运行以节省资源（参见第 792 页的“QoS 与早期拒绝策略”）。 |

![图 17-6. 基于从工作节点发出的 KV 缓存事件所接收数据的 KV 缓存感知请求路由](AI%20Systems%20Performance%20Engineering-ch17_images/figure-17-6.png)

这些因素倾向于仅卸载那些能从远程执行中获益的请求，包括长且计算密集的提示词。与此同时，短且命中缓存的提示词则在本地处理，以最小化开销。这些阈值是可调的。表 17-2 中的每个因素都对应一个特定的权衡，如下所述：

_提示词长度_ 我们不希望把时间浪费在卸载微不足道的提示词上。远程执行的开销即便很小，对于比如一个五 token 的提示词也不值得，因为 decode GPU 自己就能很快处理它。卸载只保留给那些会让 decode GPU 长时间被占用的繁重提示词。

_前缀缓存_ 现代推理系统通常会实现一个 KV 缓存，用于存储先前所见上下文的 KV 对，例如对话中较早的轮次或反复出现的样板系统提示词。如果一个新请求的提示词具有与已处理过的提示词重叠的长前缀，那么 decode 工作节点可能已经在内存中持有该前缀的 KV 数据。在这种情况下，它只需为缓存未命中的部分计算提示词的剩余部分。

如果剩余部分很短，卸载的收益就会降低。此外，如果因为完全的前缀命中而整个提示词都已缓存，那么根本不需要任何 prefill 计算。decode 可以立即使用缓存状态继续进行。路由器会通过实际考虑有效提示词长度 = (prompt_length – prefix_cached_length) 来纳入这一点。

大量的前缀命中不仅减少了所需的计算，还意味着会有大量 KV 数据不得不来回传输到 prefill 工作节点，而这毫无意义。因此，这类请求会保留在本地并利用缓存。

_prefill 队列长度_ 这本质上衡量的是 prefill 工作节点集群的负载。如果 prefill 工作节点被大量等待中的任务所压垮，再往它们那里发送一个任务对 TTFT 的损害可能大于帮助，因为请求只会在队列中干等。在这些情况下，decode 工作节点会被指示暂时自己多做一些工作。

这为 prefill 集群创造了一种天然的减载机制，因为当专用 prefill 层达到容量上限时，系统会优雅地退回到本地计算。当 prefill 队列再次变短、工作节点赶上进度时，针对长提示词的卸载便会恢复。这种动态平衡正是分离能够在各种条件下都表现良好的原因之一。

_decode 工作节点负载_ 尽管并不总是被显式地这样编码，路由器本质上有助于分配工作负载。如果一个 decode 工作节点已经忙于解码许多流，它仍然可以卸载新到来的提示词。这是好事，否则那个 GPU 就不得不同时兼顾繁重的计算与解码——可能会拖慢两者。

反过来，如果因为系统负载很轻而 decode GPU 空闲，它们可以在本地处理更长的 prefill。实际上，一个空闲的 decode GPU 很可能也拥有一个空的 prefill 队列，因为整体负载很低。这些条件已经间接覆盖了这种情况。但某些实现可能会包含对本地 GPU 利用率的直接检查，以决定是否需要 prefill 卸载。

_延迟 SLO 与优先级_ 在具有混合 SLA 或优先级类别的系统中，可以修改路由以改善 QoS（服务质量，quality of service）。对于必须获得最快 TTFT 的高优先级请求，可以绕过队列检查并立即卸载，使其立刻开始计算——即使 decode GPU 空闲也是如此。在这种情况下，系统可以将那个 decode 工作节点保留给另一个高优先级的 decode 任务。

或者，如果某个请求是低优先级的，系统可能会选择完全不使用 prefill 集群资源。相反，它只会让其在 decode 工作节点本地进行——甚至将其延后。我们将在第 792 页的“QoS 与早期拒绝策略”中再次讨论优先级请求的处理，但只需知道，基本的路由器可以扩展以纳入这类考量。

路由策略中使用的具体阈值和权重应通过经验来确定。例如，你可能会发现，在某一类型的 GPU（例如 Blackwell B200）上，低于 50 token 的提示词在本地计算更快，而更大的提示词则能从卸载中获益。

随着新硬件的发布，这些阈值可能会改变，因为更新的 GPU 拥有更高的 FLOPS 和更大的内存带宽。因此，卸载的盈亏平衡提示词长度可能会略微提高——例如，因为单个 B200 可以在 decode 工作节点上完整处理更多 token。

> 在升级硬件时，由于计算 FLOPS 和内存带宽的提升，你应当根据经验调整像 PREFILL_LENGTH_THRESHOLD 这样的阈值。

有效路由的结果是系统兼得两者之长。短提示词由于在本地运行而拥有更快的 TTFT——并且不会产生额外开销。较长的提示词则获得并行的好处，因为 prefill 与 decode 在不同的 GPU 上并行发生。此外，prefill 工作节点池仅在需要时才被使用——并且能避免在跟不上时被淹没。

这类条件式策略在 DistServe 原型以及 vLLM 和 Dynamo 等现代推理服务器中都至关重要。它使得它们能够提升延迟约束下的有用吞吐量（即有效吞吐量，goodput），而不仅仅是原始吞吐量。

在实践中，路由策略可以实现为一个简单的条件检查。下面的代码展示了这一路由逻辑的简化版本：

```
# 在 decode 工作节点上运行的 prefill 卸载决策
# （针对 B200/B300 调优）
def should_offload_prefill(prompt_length: int,
                           prefix_cached_length: int,
                           prefill_queue_size: int,
                           decode_active_reqs: int,
                           ttft_slo_ms: int = 500) -> bool:
    # 前缀 KV 命中后的有效 prefill 长度
    eff_len = max(0, prompt_length - prefix_cached_length)
    # 可调参数（保存在配置中；此处仅为清晰展示）
    PREFILL_LENGTH_THRESHOLD = 256  # see split_policy.prompt_length_threshold
    PREFILL_QUEUE_MAX        = 10   # see autoscale/prefill queue-length guidance
    DECODE_LOAD_THRESHOLD    = 8    # active decode streams
    long_prefill = (eff_len >= PREFILL_LENGTH_THRESHOLD)
    prefill_available = (prefill_queue_size < PREFILL_QUEUE_MAX)
    # 当 prefill 计算量大且资源池有余量时，
    # 优先选择远程执行。
    if long_prefill and prefill_available:
        return True
    # 若本地 decode 繁忙且 prefill 长度中等，
    # 则通过卸载来释放 decode。
    if decode_active_reqs >= DECODE_LOAD_THRESHOLD and eff_len >= 64:
        return True
    # 否则将 prefill 保留在本地
    # （更低开销 / 更好的缓存局部性）。
    return False
```

在这段伪代码中，PREFILL_LENGTH_THRESHOLD 是一个经系统调优的参数，例如 50 或 100 个 token，用于定义何为“长”提示词。PREFILL_QUEUE_MAX 是一个阈值，超过它则认为 prefill 工作节点池已饱和，具体来说就是有过多未完成的任务。

decode 工作节点一收到新请求就会调用 should_offload_prefill()。如果该函数返回 True，decode 工作节点会将提示词打包成一条消息，并将其推入全局 prefill 任务队列。随后它会在等待 KV 缓存结果返回的同时执行其他工作。

如果 should_offload_prefill() 返回 False，decode 工作节点会立即自行执行 prefill 计算。这样一来，如果 prefill 工作节点开始滞后，新请求就会回退到本地计算，以避免排队延迟。这是一种在 decode 池与 prefill 池之间均衡负载的自适应路由。

在生产部署中，路由策略应通过文件或 UI 来配置，而非硬编码。例如，NVIDIA 的 Dynamo 允许在 JSON 或 YAML 配置中指定复杂的路由和自动伸缩规则。下面是一个封装了部分策略逻辑的 Dynamo Planner JSON 的简化示例：

```
model: ...
split_policy:
  prompt_length_threshold: 256
  prefix_cache_weight: 10.0
  queue_length_weight: 1.5
  decode_load_weight: 0.5
  enable_hotspot_prevention: true
cache:
  reuse_prefix: true
  min_cache_hit_ratio: 0.75
autoscale:
  prefill:
    min_replicas: 4
    max_replicas: 12
    scale_up:   { queue_length: 8,  gpu_utilization: 80 }
    scale_down: { queue_length: 2,  gpu_utilization: 40 }
  decode:
    min_replicas: 8
    max_replicas: 24
    scale_up:   { queue_length: 16, kv_cache_usage: 75 }
    scale_down: { queue_length: 4,  kv_cache_usage: 30 }
qos:
  enable_early_rejection: true
  low_priority_threshold_ms: 500
  reject_on_slo_violation: true
```

在这里，配置定义了一个 prompt_length_threshold 为 256 个 token 的 split_policy。它还为缓存命中、队列长度、decode 负载等因素指定了权重。它同时为 prefill 和 decode 两种角色配置了 autoscale 行为，包括如何根据队列长度、GPU 利用率和 KV 缓存使用情况来扩容或缩容。

此外，它还可以应用一些 QoS 规则，例如早期拒绝（early rejection），并调整将请求视为“慢”的阈值。实际上，Dynamo 的路由器会在启动时读取此 JSON——或动态获取它——以决定分布在整个集群中的请求的每一次路由决策。

如前一节所述，NVIDIA Dynamo 支持动态路由策略。这一动态路由能力的一个实现是 Dynamo GPU Planner。该 Planner 使用 TTFT、TPOT 以及 KV 缓存传输的估计代价等指标，来决定是否修改路由——甚至重新分配/伸缩特定阶段的 GPU——以减少瓶颈并适应工作负载的变化。如图 17-7 所示，这使得系统能够在需求剧烈激增期间保持高性能。

![图 17-7. NVIDIA 的 Dynamo GPU Planner 根据 GPU 利用率指标决定如何处理传入的请求，并将 GPU 工作节点分配给 prefill 和 decode](AI%20Systems%20Performance%20Engineering-ch17_images/figure-17-7.png)

在这里，Dynamo 的 Planner 选择将更多 GPU 转移到 prefill（上下文）阶段，因为涌入了大量的摘要类提示词，这些提示词相对于 decode（较小的摘要输出）需要大量的 prefill（大输入）。

相比之下，如果涌入的是推理类请求，Planner 可以选择将 GPU 重新分配到 decode 阶段，因为推理类请求相对于输入 token 数量会生成大量的输出 token。在其他情况下，Planner 可能会选择以传统的单体方式处理请求，即 prefill 和 decode 在同一个 GPU 工作节点上进行。

简而言之，像 NVIDIA 的 Dynamo Planner 这样的组件可以通过持续监控实时指标，并将其与 TTFT、TPOT 延迟等应用层 SLO 进行比较，来自动化这一决策过程。利用这些信息，Dynamo Planner 可以动态地决定是以完全分离、不分离，还是介于两者之间（为每个 prefill/decode 阶段分配或多或少的 GPU 资源）的方式来服务一个请求。由此产生的自适应系统会在 prefill 与 decode 工作负载之间优化资源利用率——并达成激进的性能目标。

推理路由器可以超越简单的阈值规则，使用更复杂的延迟评分模型来为每个请求挑选最佳工作节点。例如，它可能会基于实时指标——如其繁忙程度、已用内存量、是否持有相关缓存等——为每个候选工作节点持续计算一个延迟代价（latency cost）。然后，它会把请求发送到延迟代价最低的工作节点。

假设延迟代价最低的工作节点就是最空闲、最优先的工作节点。一个简单的延迟代价函数可能如下所示：

```
# Lower cost is preferable
latency_cost = 0.7 * (occupancy_percent)
  + 0.3 * (active_req_count)
```

这个特定的延迟代价函数更侧重于 GPU 的占用率。它与 GPU 当前正在做多少工作相关。其次，它与该引擎上当前有多少请求正在处理相关。新请求会被发送到延迟代价最低的引擎。

本例中的权重 0.7 和 0.3 可以根据经验数据进行调优。如果发现 KV 内存使用因为需要换出大量数据或更高的内存带宽占用而成为放缓的重要预测指标，那么你就会想要给它更高的权重。

更高级的路由策略可能包含额外的因素。例如，你可以将以下部分内容纳入你的延迟代价函数：

_精确缓存匹配可用性_ 如果某个工作节点的缓存中已经有所需的前缀 KV，即前缀命中，那么它可以更快地服务该请求。在这种情况下，系统可以赋予一个较大的负权重来降低延迟代价，从而优先选择这个工作节点——因为在此场景下代价越低越好。

_KV 占用率（KV occupancy）_ 内存占用越高，说明 GPU 越繁忙。这种情况下，系统会赋予一个正权重来抬高延迟代价（latency cost），促使路由器避开该工作节点——因为在此场景中数值越低越好。

_活跃请求（active requests）_ 并行请求越多，就可能带来上下文切换开销，因此这会增加延迟代价，从而避开该工作节点。

_内存带宽利用率（memory bandwidth utilization）_ 如果某个 GPU 当前正因处理许多长序列等原因占用了大量内存带宽，那么再给它加任务只会拖慢速度。这会抬高延迟代价，促使系统不要选择该工作节点。

_近期 KV 使用（recent KV use）_ 如果某个工作节点引擎近期正好使用过完全相同的前缀——缓存处于“热”状态——那么性能很可能会提升，因为 KV 仍在 L2 缓存中，或正从某个 prefill 工作节点传来。这种情况下，你可以加入一个较小的负权重来降低延迟代价，优先选择该工作节点，因为它近期见过该前缀。

表 17-3 总结了这类高级路由策略（routing policy）配置，列出了一些因子及示例代价。

表 17-3. 路由因子及相对代价

| 因子                              | 含义                    | 对延迟代价的影响 |
| --- | --- | --- |
| KV 占用率 (%) (occupancy_percent) | 越高 = 内存压力越大     | +3               |
| 活跃请求数 (active_req_count)     | 在途请求越多 = 可能排队 | +1               |
| KV 缓存命中 (cache_match_flag)    | 引擎已持有所需的前缀 KV | –10（大负值）    |
| 内存带宽 % (mem_bw_percent)       | 高 = 内存总线繁忙       | +0.5             |
| 近期 KV 使用 (recent_prefix_flag) | 该引擎近期使用过此前缀  | –1               |

我们用这些额外指标来修改延迟代价的计算。现在路由器会计算一个综合延迟代价，如下所示：

```
# 代价越低越优先
def latency_cost(occupancy_percent: float,
                 active_reqs: int,
                 cache_match_flag: bool,
                 mem_bw_percent: float,
                 recent_prefix_flag: bool) -> float:
    return (
        3.0 * occupancy_percent
      + 1.0 * active_reqs
      - 10.0 * int(cache_match_flag)
      + 0.5 * mem_bw_percent
      - 1.0 * int(recent_prefix_flag)
    )
```

prefill 与 decode 集群甚至可能使用略有不同的公式。对于 prefill，缓存命中（cache hit，例如前缀已计算完毕）在计入时可能更有价值；而对于 decode，内存带宽或可用 KV 空间则可能主导决策。

路由器需要来自每个工作节点的遥测数据，包括等待队列长度、KV 缓存使用量、内存利用率等指标，以便持续更新延迟代价。

其结果是，系统可以动态地把流量路由到对延迟影响最小的地方。在低负载下，所有工作节点得分都很低，选哪个可能都无所谓，因为任何工作节点都能快速处理请求。

然而在高负载下，路由器会把新任务发给最空闲的工作节点，从而有效地实现负载均衡。这还会利用数据局部性（缓存命中）来减少工作量。这样可同时降低平均延迟与尾延迟（tail latency），因为当其他引擎空闲时，任务不太可能堆积在繁忙引擎后面。

前面我们看到，把缓存命中纳入考量后，如果某台服务器缓存了相关数据，路由器可能会选择这台稍微繁忙一些的服务器，从而让该请求得到更快的服务。打分函数定量地刻画了这一权衡。

NVIDIA Dynamo 执行的是一种类似但相反的计算，其中数值越高越好。具体来说，它会计算一个远程 prefill 分数，如果该分数超过某个阈值，就会卸载（offload）到 prefill 工作节点池。下面是一段示例 YAML 片段，它将系统配置为：当计算出的分数高于可配置的 remote_prefill_min_score 值时，采用条件式分离：

```
disaggregated_router:
  enable: true
  policies:
    - metric: prefill_length   # 前缀缓存命中后的提示词长度
      threshold: 256
      action: prefer_remote    # 若提示词 > 256 个 token，则卸载 prefill
      weight: 0.7
    - metric: prefill_queue_depth  # 排队等待 prefill 工作节点的请求
      threshold: 10
      action: prefer_local     # 若队列 > 10，则倾向于本地 prefill
      weight: 0.3
  remote_prefill_min_score: 0.5  # 决定是否远程 prefill 的总体阈值
```

这里，路由器基于 prefill_length 和 prefill_queue_depth 计算一个分数。在此例中，如果提示词长于 256 个 token，它就会投票给 prefill_remote 并卸载 prefill。如此处配置，这部分分数计算的权重为 0.7。

但如果 prefill 队列很深、有超过 10 个等待任务（如此处配置），它就会投票给 prefill_local，权重为 0.3（如此处配置）。如果综合分数大于 remote_prefill_min_score（在此例中为 0.5），Dynamo 就会卸载 prefill；否则会把 prefill 保留在本地，不做卸载。

多路径推理（multipath inference），即把同一请求发送给两种不同规模的模型或两条不同的路由，用于实现高可靠性。它本质上是在竞速取最快结果，常被称为*竞速（racing）*。已知 Google 与 Meta 的生产系统会让多个模型竞速以降低尾延迟。这是一种成本高但有效的技术。

> 如果你自行实现多路径推理，务必确保请求是幂等的，即使两条路径都执行也不会引发问题。并且务必及时取消较慢的那条路径以节省 GPU 周期。

正如第 15 章所讨论的，一些高级推理服务器支持推测解码（speculative decoding）。请记住，推测解码会并行生成多个 token 分支，并丢弃那些并非最理想的分支。

虽然推测解码并非分离的直接组成部分，但它可以叠加在路由器的决策过程之上。例如，系统可以检测到某个不确定的推测解码分支，并把多个推测请求并行发送给不同的 decode 工作节点。这会生成多个推测性的下一 token 分支，掩盖 token 分支生成中的不可预测性。对于分支不可预测的生成，这是用额外算力换取更低延迟。

若实现该机制，路由器将协调这些推测性工作并合并结果。资源使用需要设定上限以免压垮集群，但多路径推理值得作为一种针对超低延迟敏感推理的潜在优化来关注——代价是额外的、可能被浪费的算力与内存带宽集群资源。

总之，延迟感知的路由器会综合考虑当前负载与诸如缓存复用之类的潜在加速，确保每个请求都被发送到最优位置。它充当分布式服务系统的“大脑”，与每个 GPU 上的底层调度协同工作。再加上全局伸缩与 KV 缓存策略，它构成了一套最大化有效吞吐量（goodput）、最小化延迟的综合方法。

QoS（服务质量，quality of service）策略会对某些请求进行优先处理或限流，以满足延迟目标。早期拒绝（early rejection），也称*准入控制（admission control）*，可以在系统饱和时拒绝低优先级查询。这样可以为其他请求保留资源并守住 SLO，例如延迟。

现代系统会基于*队列时间阈值（queue time threshold）*来做这件事。例如，如果某个请求在队列中停留超过 _X_ ms，那么与其让它严重违反 SLO，不如拒绝它或把它卸载到较低层级的服务。这类 QoS 防护在关键任务型推理系统中越来越常见。

在超大规模（ultrascale）推理系统中，需要 QoS 机制来在重负载下维持尾延迟保证。即便采用了优化过的分离式集群配置，一旦负载超过容量，请求就会开始排队，延迟随之上升。

与其让所有请求的延迟 SLO 都被违反，设计良好的系统会更倾向于优雅地卸载负载。当明确知道服务某些请求会破坏其他请求的延迟保证时，它可以拒绝或延后这些请求，尤其是较低优先级的请求。

这类似于 Web 服务器在极端负载下返回 HTTP 503 “过载”错误。对于 LLM 服务，如果我们无法保证及时服务，就可能主动拒绝请求或对请求做降采样。以下是 prefill/decode 分离场景下 QoS 的几个组成部分：

_延迟 SLO 跟踪（latency SLO tracking）_ 系统应了解目标 TTFT 与 TPOT。例如，首 token 必须在 99% 的情况下于 200 ms 内返回。借助内部遥测，每个 decode 工作节点可以基于当前队列长度等，估算一个新请求若现在被接受时的当前 TTFT。类似地，它也可以估算再增加一路 decode 流时的 TPOT。

_准入控制（早期拒绝）_ 在请求被完全接受并分配之前，系统可以做一次检查，看看接纳这个请求是否会让系统过载或违反 SLO。若会，它可以立即拒绝该请求，返回类似“服务器繁忙，请稍后重试”的响应。这被称为*早期拒绝*。实践中，你可以使用系统负载的全局视图，或仅用单个节点的启发式规则来触发早期拒绝。

例如，OpenAI 的公开 API 在积压过高时会返回错误，而不是违反延迟承诺。一些服务提供方会在高峰负载期间动态降低最大生成长度，这实际上是用回答质量换取延迟。

_优先级排序（prioritization）_ 并非所有请求都同等重要。例如，来自付费客户或关键服务的请求可能是高优先级（priority）；来自免费用户或后台作业的请求则可能较低。系统可以把优先级纳入调度决策。例如，decode 工作节点的调度器可以先服务高优先级的 decode 任务；或者 prefill 任务队列可以按优先级排序。一旦繁忙起来，低优先级任务可能等待更久，或被快速失败以让位给高优先级任务。

_优雅降级（graceful degradation）_ 如果系统开始过载，它可以不直接拒绝请求，而是对不太重要的请求降低服务质量。例如，它可以为不太重要的请求使用更小的模型、截断它们的提示词，或限制输出 token 数。一种简单的做法是：在负载高时，临时降低较低层级请求所允许的最大提示词长度或输出长度。

在分离场景下，一种有趣的优雅降级形式是：当系统承压时，临时对较低优先级的请求禁用远程 prefill。由于远程 prefill 会占用额外的集群资源来优化延迟，可以决定让低优先级查询只做本地 prefill，仅使用 decode 工作节点的资源。这会让这些请求的响应变慢，但能腾出 prefill 工作节点服务高优先级查询。

例如，推理服务器可以给每个请求打上优先级标签，当 prefill 工作节点正忙于高优先级任务时，让路由器对低优先级请求忽略 should_offload_prefill 逻辑。

下面是一个早期拒绝策略的示例。它可以在每个 decode 工作节点上于处理请求之前运行，甚至可以在上游的 LLM 网关上运行：

```
# 基于预估延迟与优先级的早期拒绝
from dataclasses import dataclass
class QoSController:
    def __init__(self, ttft_slo_ms: int = 500):
        self.ttft_slo_ms = ttft_slo_ms
    def admit_request(self, priority: str) -> bool:
        # 队列长度与每请求平均值由指标系统提供
        est_ttft = (get_current_prefill_queue_length()
                    * get_avg_prefill_time_per_req()
                    + get_current_decode_queue_length()
                    * get_avg_decode_time_per_req())
        if est_ttft > self.ttft_slo_ms and priority.lower() == "low":
            # 拒绝低优先级请求以保护 SLO
            return False
        return True
```

这里，我们通过查看排队的 prefill 任务数量（每个都会增加一些延迟）以及前面有多少 decode 任务来估算 TTFT。请注意，decode 任务通常会重叠，所以这只是一个粗略估计。如果估算出的 TTFT 超过了允许的上限（即 SLO），那么对于低优先级请求，我们就拒绝它，并从 should_offload_prefill 函数返回 False。

对于高优先级请求，我们仍然接受它，必要时甚至牺牲一些其他已排队的工作。更复杂的做法是抢占已经排队的低优先级请求，为新的高优先级请求腾出空间。这通常通过为每个优先级维护独立队列来实现。

再深入一点，在上例中，avg_prefill_time_per_req() 与 avg_decode_time_per_req() 函数会即时计算数值，方法是对观测到的每请求 prefill 与 decode 时长取指数移动平均（exponential moving average）。它们分别按提示词 token 数（prefill）与生成 token 数（decode）做归一化。对于长度不一的输入，引擎会把这些每 token 平均值乘以每个请求的实际 token 数来进行外推。

get_current_prefill_queue_length() 与 get_current_decode_queue_length() 函数会获取调度器内部队列所跟踪的待处理 prefill 与 decode 任务数量。这些数值由调度器维护。

这些每 token 的耗时估算会通过调度器循环实时刷新，默认情况下每秒通过 /metrics 端点更新一次，从而捕捉动态的工作负载变化。

早期拒绝与优先级排序确保系统在接近饱和时以受控方式失败，而不是崩溃。拥有最重要请求的用户仍会在既定 SLO 内得到服务；与此同时，不太重要的流量则被卸载掉。

在许多生产部署中，实现这一点需要与 LLM 网关层协调。具体来说，网关可能会向客户端返回特定的错误码，或返回表示服务器过载的代码。关键在于，系统要让自己保持在这样一种状态：能够为它确实接受的工作兑现延迟承诺，而不是超额认购、进而让所有人都错失 SLO。

另一个 QoS 考量是自适应生成上限。如果 decode 阶段有跑得过久、导致违反 TPOT SLO 的风险，系统可能会提前中断生成。例如，如果用户请求了 1,000 个 token，但系统正承压，它可能只允许生成 200 个 token 就停止，从而为其他请求留出资源。QoS 策略会对某些请求进行优先处理或限流以满足延迟目标；早期拒绝可在系统饱和时拒绝低优先级查询，以为其他请求保留 SLA。

简而言之，分离减少了固有干扰（interference），但负载尖峰仍可能压垮任何固定容量。QoS 机制通过确保分离架构兑现其延迟承诺来对其形成补充。早期拒绝会拒绝某些工作或降低其优先级，以保护其余请求的延迟。

在构建超大规模系统时，这些策略与核心性能优化同等重要。没有它们，请求洪流会制造巨大的队列与拖慢，毁掉所有提升性能的努力。

### 分离式 Prefill 与 Decode 的可扩展性

可扩展性的另一个方面是，随着节点增多，性能能否保持。分离式方法在设计上于许多方面呈相对线性的扩展。你可以增加更多 prefill 节点来处理更多的提示词吞吐量，也可以增加更多 decode 节点来处理更多的 token 生成吞吐量。

主要挑战在于平衡两者。随着规模增大，自适应调度器变得愈发重要，因为在大型系统中出现失衡（例如流量模式变化）的概率高得多。动态比例配置允许集群即时重新平衡 prefill 与 decode 的比例。

另一个考量是多模型服务。如果不同模型的解码特性相似，分离允许你在模型之间共享 decode 容量。例如，如果模型 A 和模型 B 托管在同一集群上，你可以专门指定一些 decode 工作节点来同时处理这两类模型。当托管复用同一基础模型架构的不同 LoRA 适配器时，这尤其有用。

这会带来一套超级灵活的、多租户、多模型、prefill-decode 分离的推理服务器系统。这超出了本书范围，但只需知道分离的模块化带来了这类灵活性即可。你可以让多个模型共用一个 decode 池，每个模型各有自己的 prefill 前端。

最后，我们来分析超大型系统中的尾延迟。随着超大规模推理系统横向扩展到数千乃至数百万个 GPU 节点，控制尾延迟变得越来越困难。这是因为服务器越多，其中一台发生故障的概率就越高。

分离在这里同样有帮助。把 prefill 任务隔离在一个节点、把 decode 任务隔离在另一个节点，意味着一侧的慢节点不会剧烈影响另一侧的任务。

本章讨论的早期拒绝、两级调度、KV 缓存复用等技术，有助于缓解掉队者（straggler）。例如，缓存请求上下文的大部分内容，意味着一次慢操作不会那么严重地拖慢整个生成——因为大量计算已经完成了。

总之，经过恰当配置与调优后，分离式系统在大规模下更为稳健。即便并发请求增长到数百万乃至数十亿，这些系统仍能维持稳定的延迟。通过消除一个干扰因素并增加自适应能力，尾延迟分布得以改善——至少不会随负载而快速恶化。

## 关键要点

本章涵盖的技术——从 KV 缓存的高速 RDMA 传输，到动态路由算法与 QoS 策略——已被证明是大规模生产推理工作负载的关键组成部分。以下是关键要点：

_消除 prefill-decode 干扰以改善延迟_ 分离 prefill 与 decode 两个阶段可消除干扰与队头阻塞（head-of-line blocking）。由于长提示词不会延误较短提示词，这会带来更紧凑的延迟分布，也让每个阶段都能可靠地满足其延迟 SLO。

_独立优化每个阶段_ prefill（提示词处理）是计算受限（compute bound）的，得益于最大化并行度与高 FLOPS。它偏好高算力 GPU 以及 FP8/FP4 降精度等技术来最小化 TTFT。decode（token 生成）是访存受限（memory bound）的，得益于高内存带宽 GPU 以及连续批处理（continuous batching）等技术来最大化每 token 吞吐量。分离允许为每个阶段设置不同的硬件与并行配置，而固定的、一刀切的配置做不到这一点。

_善用 KV 缓存_ 常见做法是使用 KV 缓存来避免重复计算提示词的 prefill。智能路由器可以决定一个新请求应卸载给 prefill 工作节点，还是直接在 decode 工作节点上处理。路由策略应考虑提示词长度、缓存命中与集群负载。

_智能路由_ 像 vLLM、SGLang、NVIDIA Dynamo 这样的现代推理服务器使用多因子打分（multifactor scoring）算法来路由 prefill 与 decode 请求，以最大化吞吐量并维持足够的延迟水平。许多框架在调度器与仪表盘中也把 TPOT 称为*token 间延迟（inter-token latency，ITL）*。建议为 prefill 节点与 decode 节点分别跟踪 TTFT（p50/p95/p99）与 ITL/TPOT（p50/p95/p99）。这样，比如在升级期间更容易调试性能回退。举例来说，如果 prefill 服务器饱和或提示词很短，它们就会跳过远程 prefill。

_用 QoS 在大规模下维持 SLA_ 在重负载下施加准入控制与优先级排序。快速失败多余或低优先级请求，好过让系统的延迟对所有人爆炸。现实系统通过返回“繁忙”错误或降级的响应长度来实现早期拒绝——以保证第 99 百分位延迟保持在给定阈值之下。分离与 QoS 共同防止过载级联——即便在每天十亿请求的场景中也是如此。

## 结论

分离式 prefill 与 decode 如今在现代高规模 LLM 推理中已很常见。几乎所有主流 AI 提供方，如 Meta、Amazon、NVIDIA、Google 和 OpenAI（即“MANGO”），都在其大规模 LLM 部署中采用了某种形式的分离架构。

尽管这些公司大多不会公开其完整的生产架构，但已有大量开源实现（如 vLLM、SGLang、NVIDIA 的 Dynamo）展示了这种方法及其优势。相较于单体（monolithic）设计，分离式 PD 能带来更高的有效吞吐量（即延迟约束下的有效吞吐量）以及更好的成本效率。

在下一章中，我们将继续把现代 GPU 与网络硬件同更先进的调度与缓存优化结合起来，在不牺牲延迟、不使成本爆炸的前提下，为数十亿用户服务数万亿参数规模的 LLM。
