# Chapter 15. Multinode Inference, Parallelism, Decoding, and Routing Optimizations

# 第 15 章 多节点推理、并行、解码与路由优化

LLMs continue to scale up to a massive number of parameters. In particular, the emergence of mixture-of-experts (MoE) LLMs, models that combine many specialist subnetworks (“experts”) with a built-in, expert-gating mechanism, has pushed model parameter sizes into the hundreds of billions or multiple trillions. And while only a fraction of those parameters are active for a given input, running inference on these enormous model sizes requires distributing the workload across multiple GPUs.

大模型（LLM）的参数规模仍在持续攀升。尤其是 MoE（专家混合，mixture of experts）大模型的出现，将许多专精子网络（“专家”）与内置的专家门控机制结合在一起，使模型参数量迈入了千亿甚至数万亿级别。虽然对于给定输入只有其中一小部分参数处于激活状态，但要在如此庞大的模型上运行推理，仍然需要把工作负载分布到多块 GPU 上。

This chapter focuses on advanced optimization techniques used to perform efficient, high-performance multinode inference for these massive LLMs using modern NVIDIA GPUs. We will discuss how to architect distributed inference systems that minimize latency and maximize throughput—leveraging both hardware and algorithmic innovations.

本章聚焦于使用现代 NVIDIA GPU 为这些超大 LLM 执行高效、高性能多节点推理（multinode）所用的高级优化技术。我们将讨论如何架构分布式推理系统，在最小化延迟（latency）的同时最大化吞吐量（throughput）——同时借力硬件与算法两方面的创新。

We start by discussing disaggregated prefill and decode (PD, disagg PD) architectures that split the inference workload into distinct stages, which can be tuned independently. Next, we explore core inference-focused parallelism strategies like data, tensor, pipeline, expert, and context—and how they can be used in combination to serve large models across many GPUs.

我们首先讨论分离式 prefill 与 decode（disaggregated prefill and decode，PD、disagg PD）架构，它把推理工作负载拆分为可以各自独立调优的不同阶段。接着，我们探讨以推理为核心的几种并行策略——数据、张量、流水线、专家与上下文——以及如何组合运用它们，把大模型服务部署到众多 GPU 上。

We then cover speculative decoding methods, including techniques like Medusa, EAGLE, and draft-and-verify schemes. These allow multiple tokens to be generated and evaluated during inference instead of the standard single-token generation from traditional autoregressive LLMs. This helps overcome the sequential decoding bottleneck. We also discuss constrained decoding for enforcing output formats (e.g., custom JSON schemas) and dynamic routing strategies for MoE models to improve the system’s expert gating and load-balancing efficiency.

随后我们介绍投机解码（speculative decoding）方法，包括 Medusa、EAGLE 以及“起草-验证”（draft-and-verify）等技术。这些技术允许在推理过程中一次性生成并评估多个 token，而不是像传统自回归（autoregressive）LLM 那样一次只生成单个 token，从而帮助克服顺序解码的瓶颈。我们还会讨论用于强制输出格式（例如自定义 JSON 模式）的受约束解码（constrained decoding），以及用于提升 MoE 模型专家门控与负载均衡效率的动态路由（dynamic routing）策略。

## Disaggregated Prefill and Decode Architecture

## 分离式 Prefill 与 Decode 架构

As mentioned previously, the inference workflow for modern LLMs consists of two different phases: prefill and decode. We can implement *disaggregated prefill and* *decode* to separate the stages. This lets us scale the prefill and decode clusters independently—even on different hardware platforms—and significantly improve performance for large-scale LLM serving, as detailed later in this chapter.

如前所述，现代 LLM 的推理流程包含两个不同的阶段：prefill（预填充）与 decode（解码）。我们可以实现*分离式 prefill 与* *decode* 来把这两个阶段拆分开。这样一来，我们就能独立地扩展 prefill 与 decode 集群——甚至可以放在不同的硬件平台上——并显著提升大规模 LLM 服务的性能，本章后文将详述这一点。

> Cross-vendor or cross-architecture deployments require that the KV cache layout and dtypes match across both sides. In practice, production systems should keep prefill and decode on compatible GPU families. This way, they use the same numeric formats to easily enable KV cache transfer and data reuse.

> 跨厂商或跨架构部署要求两侧的 KV 缓存（KV cache）布局与数据类型（dtype）保持一致。在实践中，生产系统应当把 prefill 与 decode 部署在兼容的 GPU 家族上。这样它们使用相同的数值格式，便于实现 KV 缓存传输与数据复用。

In the prefill stage, the model processes the entire input prompt—often thousands, tens of thousands, or even millions of tokens—in a single forward pass to produce initial hidden states as calculated by the LLM. It then populates the attention key-value (KV) cache for all tokens in the input prompt. Figure 15-1 shows how disaggregated prefill and decode share the KV cache and overlap KV transfers with computations.

在 prefill 阶段，模型在一次前向传播中处理整个输入提示词（prompt）——通常是数千、数万甚至数百万个 token——以计算出 LLM 的初始隐藏状态（hidden state）。随后它会为输入提示词中的所有 token 填充注意力键值（KV）缓存。图 15-1 展示了分离式 prefill 与 decode 如何共享 KV 缓存，并将 KV 传输与计算相互重叠。

![Figure 15-1. Disaggregated prefill and decode sharing the KV cache and overlapping KV transfers with computations](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-1.png)

![图 15-1. 分离式 prefill 与 decode 共享 KV 缓存，并将 KV 传输与计算相互重叠](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-1.png)

In the decode stage, the model performs an autoregressive generation to predict each new token in the sequence. It does this by consuming the cached attention KV representations of all previously generated tokens.

在 decode 阶段，模型执行自回归生成，逐个预测序列中的每一个新 token。它通过消费此前已生成的所有 token 的缓存注意力 KV 表示来完成这一过程。

> Speculative decoding accelerates the decode process by pregenerating multiple tokens in a single batch. In parallel, it then verifies that the tokens are correct. This reduces the sequential nature of standard token-by-token autoregressive decoding. We’ll cover speculative decoding in a bit.

> 投机解码通过在单个批次中预生成多个 token 来加速 decode 过程。与此同时，它并行地验证这些 token 是否正确。这降低了标准逐 token 自回归解码的顺序性。稍后我们会讲到投机解码。

### Prefill-Decode Interference

### Prefill-Decode 干扰

Traditionally, LLM inference systems colocate these two stages on the same nodes and simply batch all computations together. However, this naive approach leads to what’s commonly called *prefill-decode interference*. For instance, a long prompt prefill can occupy the GPU and delay time-sensitive decoding work for other requests—and vice versa.

传统上，LLM 推理系统会把这两个阶段共置（colocate）在同一批节点上，并简单地把所有计算批处理（batch）在一起。然而，这种朴素做法会导致通常所说的 *prefill-decode 干扰*。例如，一个长提示词的 prefill 可能会占用 GPU，从而延误其他请求对时延敏感的 decode 工作——反之亦然。

Colocating prefill and decode on the same nodes forces a single scheduling and resource allocation strategy for these two phases, which have very different characteristics. Prefill consists of large, parallel computations. In contrast, decode requires many small, sequential computations. As a result, systems have to either prioritize one phase’s performance over the other or over-provision hardware to meet both demands.

把 prefill 与 decode 共置在同一批节点上，会迫使这两个特性迥异的阶段共用单一的调度与资源分配策略。prefill 由大规模并行计算构成；相比之下，decode 需要大量小而顺序的计算。结果是，系统要么必须让某一阶段的性能优先于另一阶段，要么就得过度配置硬件以同时满足两者的需求。

With a *disaggregated prefill and decode* architecture, the prefill and decode phases are assigned to different GPU pools. This eliminates direct interference between the two workloads. The DistServe system, by disaggregating prefill and decode, reported up to 7.4× more goodput requests served within both TTFT and TPOT constraints (up to 12.6× tighter latency SLOs).

采用*分离式 prefill 与 decode* 架构后，prefill 与 decode 阶段被分配到不同的 GPU 池，从而消除了两种工作负载之间的直接干扰。DistServe 系统通过分离 prefill 与 decode，据报告在同时满足 TTFT 与 TPOT 约束的前提下，有效吞吐量（goodput）请求量最高提升到 7.4×（延迟 SLO 最高收紧至 12.6×）。

### Scaling Prefill and Worker Nodes Independently

### 独立扩展 Prefill 与工作节点

If we can eliminate cross-phase interference, we can reduce resource “dead time” in which decode tasks are stalled behind long prefill computations—and vice versa. This way, GPUs spend more time doing more useful work and less time idling. This increases utilization and useful throughput at a given latency target (aka *goodput*).

如果能够消除跨阶段干扰，我们就能减少资源的“死时间”——即 decode 任务因排在长 prefill 计算之后而停滞的时间，反之亦然。这样，GPU 就能把更多时间用于更有用的工作，而更少地处于空闲状态。这在给定延迟目标下提升了利用率与有效吞吐量。

We can scale prefill and decode separately by dedicating one set of nodes to handle the prefill and another set of nodes to handle the decode. The two clusters communicate only when transferring the encoded prompt state, or attention KV cache, from the prefill workers to the decode workers, as shown in Figure 15-2.

我们可以通过让一组节点专门处理 prefill、另一组节点专门处理 decode，来分别扩展 prefill 与 decode。两个集群仅在把编码后的提示词状态（即注意力 KV 缓存）从 prefill 工作节点（worker node）传输给 decode 工作节点时才进行通信，如图 15-2 所示。

Here, you see separate GPU workers handling the prefill stage to process the input prompt—along with the decode stage to generate output tokens iteratively. The output of the prefill stage includes the KV cache for the prompt. It is transferred to the decode workers to generate the next tokens.

在图中，你可以看到独立的 GPU 工作节点分别处理 prefill 阶段（处理输入提示词）与 decode 阶段（迭代地生成输出 token）。prefill 阶段的输出包含该提示词的 KV 缓存，它被传输给 decode 工作节点以生成后续 token。

![Figure 15-2. Disaggregated inference: Prefill pool (hidden state + KV) → KV handoff using NVLink/NVSwitch (intranode) or GPUDirect RDMA (internode) → Decode pool](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-2.png)

![图 15-2. 分离式推理：prefill 池（隐藏状态 + KV）→ 使用 NVLink/NVSwitch（节点内，intranode）或 GPUDirect RDMA（节点间，internode）进行 KV 交接（handoff）→ decode 池](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-2.png)

By dedicating separate GPU pools, the system keeps both prefill and decode pipelines busy in parallel. In practice, disaggregation has been shown to significantly improve throughput under strict latency constraints. Some studies show that large gains are possible, but results range from moderate improvements to several times higher goodput once the prefill and decode stages are separated. The results greatly depend on the workload and network fabric.

通过专用的独立 GPU 池，系统让 prefill 与 decode 两条流水线并行地保持繁忙。在实践中，分离已被证明能够在严格的延迟约束下显著提升吞吐量。一些研究表明有望获得可观的收益，但结果范围不一：一旦 prefill 与 decode 阶段被分开，收益从适度改善到有效吞吐量数倍提升均有出现。结果在很大程度上取决于工作负载与网络架构（fabric）。

This separation allows each stage to be optimized and scaled independently for throughput or latency. Prefills can be batched aggressively on the prefill GPUs to maximize throughput without burdening the decode performance (e.g., increased decode latency). Also, with separate clusters we can tune parallelism settings, instance counts, and scheduling policies specific to each phase.

这种分离使得每个阶段都能针对吞吐量或延迟被独立优化和扩展。prefill 可以在 prefill GPU 上被激进地批处理，以在不拖累 decode 性能（例如增加 decode 延迟）的前提下最大化吞吐量。此外，有了独立的集群，我们就能为每个阶段单独调优并行设置、实例数量与调度策略。

### Impact on Latency (TTFT) and Throughput (TPOT)

### 对延迟（TTFT）与吞吐量（TPOT）的影响

Decode GPUs can run at lower batch sizes—or with specialized scheduling—to minimize *time per output token* (TPOT) for streaming generation. For example, you can use a scheduler that prioritizes urgent decode tasks to avoid queuing delays.

decode GPU 可以以较低的批大小运行——或采用专门的调度——以最小化流式生成的*每输出 token 时延*（time per output token，TPOT）。例如，你可以使用一个优先处理紧急 decode 任务的调度器，以避免排队延迟。

This separation works well because each phase has different performance expectations. *Time to first token* (TTFT) for the prefill stage is optimized for low latency, while the decode stage prioritizes low time per output token (TPOT) and stable streaming latency. End-to-end throughput is largely determined by concurrency and scheduling. In traditional setups, one had to compromise between TTFT and TPOT per-token latency. Disaggregation allows both SLO targets to be met simultaneously.

这种分离之所以奏效，是因为每个阶段有着不同的性能预期。prefill 阶段的*首 token 时延*（time to first token，TTFT）以低延迟为优化目标，而 decode 阶段则优先追求低每输出 token 时延与稳定的流式延迟。端到端吞吐量在很大程度上由并发度与调度决定。在传统方案中，人们不得不在 TTFT 与逐 token 的 TPOT 延迟之间做折中。分离则允许两个 SLO 目标同时得到满足。

> Monitor NVLink and NIC utilization during KV transfer and all-to-all phases. The goal is to overlap communication and compute using separate streams and events.

> 在 KV 传输与 all-to-all 阶段，监控 NVLink 与 NIC 的利用率。目标是使用独立的流（stream）与事件（event）来重叠通信与计算。

The KV handoff incurs minimal additional latency because the communication uses high-bandwidth interconnects for multi-GPU and multinode transfers. These interconnects include NVLink, NVSwitch, InfiniBand, and Ethernet (RoCE on Ethernet) using GPUDirect RDMA. For instance, multinode clusters with ConnectX-8 SuperNICs (800 GbE-class) provides up to 800 Gb/s per port with GPUDirect RDMA. This greatly reduces KV transfer time compared to host-mediated communication paths. Additionally, it’s recommended to deploy 1 NIC per GPU to optimize prefill-decode disaggregation and improve MoE all-to-all performance.

KV 交接带来的额外延迟极小，因为其通信使用高带宽互连（interconnect）来进行多 GPU 与多节点传输。这些互连包括 NVLink、NVSwitch、InfiniBand，以及通过 GPUDirect RDMA 的以太网（Ethernet 上的 RoCE）。例如，配备 ConnectX-8 SuperNIC（800 GbE 级）的多节点集群，借助 GPUDirect RDMA 每端口可提供高达 800 Gb/s 的带宽。相比经由主机中转的通信路径，这大大缩短了 KV 传输时间。此外，建议为每块 GPU 部署 1 张 NIC，以优化 prefill-decode 分离并提升 MoE 的 all-to-all 性能。

### KV Cache Data Transfer and NIXL

### KV 缓存数据传输与 NIXL

Disaggregated systems use a connector or scheduler to transfer the prompt’s intermediate results (the final hidden state and the KV cache) from the prefill workers to the decode workers once the prompt processing is done. This handoff incurs some communication overhead, but if the cluster’s interconnect is high-bandwidth (e.g., NVLink or InfiniBand), this overhead is small compared to the gains from eliminating resource contention.

分离式系统使用一个连接器（connector）或调度器，在提示词处理完成后，把提示词的中间结果（最终隐藏状态与 KV 缓存）从 prefill 工作节点传输给 decode 工作节点。这一交接会带来一些通信开销，但如果集群的互连是高带宽的（例如 NVLink 或 InfiniBand），相比消除资源争用所带来的收益，这一开销是很小的。

In practice, NVIDIA’s NIXL library minimizes transfer overhead by selecting NVLink/NVSwitch, RDMA, or host-staged paths automatically based on topology and policy. For example, NVIDIA’s NIXL library, introduced in Chapter 4, will automatically select the fastest available path to transmit the KV cache. NIXL integrates with frameworks and inference engines, including Dynamo and vLLM. The NIXL-vLLM integration is shown in Figure 15-3.

在实践中，NVIDIA 的 NIXL 库会根据拓扑与策略自动选择 NVLink/NVSwitch、RDMA 或主机中转路径，从而最小化传输开销。例如，第 4 章介绍过的 NVIDIA NIXL 库会自动选择当前可用的最快路径来传输 KV 缓存。NIXL 与多种框架和推理引擎集成，包括 Dynamo 和 vLLM。NIXL 与 vLLM 的集成如图 15-3 所示。

Specifically, LMCache and NIXL are integrated in vLLM’s disaggregated prefilling as the supported path. NIXL is also used by NVIDIA Dynamo and TensorRT-LLM to transport KV cache data using peer-to-peer GPU interconnects and RDMA.

具体而言，LMCache 与 NIXL 作为受支持的路径，被集成进了 vLLM 的分离式 prefilling。NVIDIA Dynamo 与 TensorRT-LLM 也使用 NIXL，通过点对点 GPU 互连与 RDMA 来传输 KV 缓存数据。

Within a node, NIXL performs device-to-device transfers over NVLink and NVSwitch without host staging. Across nodes, NIXL uses GPUDirect RDMA over InfiniBand or RoCEv2 to avoid host copies. These paths keep KV cache handoff latency low even for multigigabyte payloads.

在节点内，NIXL 通过 NVLink 与 NVSwitch 执行设备到设备（device-to-device）的传输，无需主机中转。跨节点时，NIXL 通过 InfiniBand 或 RoCEv2 使用 GPUDirect RDMA，以避免主机拷贝。即便对于数 GB 级的载荷，这些路径也能让 KV 缓存交接保持低延迟。

![Figure 15-3. KV cache data transfers with NIXL in the vLLM inference engine; Intranode: NVLink/NVSwitch (device-to-device); Internode: GPUDirect RDMA (InfiniBand/RoCEv2) using ConnectX-class NICs](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-3.png)

![图 15-3. 在 vLLM 推理引擎中使用 NIXL 进行 KV 缓存数据传输；节点内（Intranode）：NVLink/NVSwitch（设备到设备）；节点间（Internode）：使用 ConnectX 级 NIC 的 GPUDirect RDMA（InfiniBand/RoCEv2）](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-3.png)

Placement of prefill and decode workers should follow the fabric. Same node placement keeps KV transfers on NVLink and NVSwitch via CUDA peer access, while cross-node placement should use GPUDirect RDMA over InfiniBand or RoCEv2. NVIDIA Dynamo integrates with NIXL to move KV cache between GPUs, CPU memory, and storage across nodes, and vLLM integrates through LMCache and NIXL for disaggregated prefilling.

prefill 与 decode 工作节点的放置应当顺应网络架构。同节点放置可让 KV 传输经由 CUDA peer access 走 NVLink 与 NVSwitch，而跨节点放置则应使用经 InfiniBand 或 RoCEv2 的 GPUDirect RDMA。NVIDIA Dynamo 与 NIXL 集成，在节点间的 GPU、CPU 内存与存储之间搬运 KV 缓存；vLLM 则通过 LMCache 与 NIXL 集成，实现分离式 prefilling。

> When the fabric or virtualization layer prevents direct peer access, NIXL can fall back to host-staged paths, which are nonoptimal. Always validate end-to-end KV transfer time on your deployment.

> 当网络架构或虚拟化层阻碍了直接的 peer access 时，NIXL 可以回退到主机中转路径，但这并非最优。请务必在你的部署环境中验证端到端的 KV 传输时间。

### Deploying Disaggregated Prefill and Decode with Kubernetes

### 使用 Kubernetes 部署分离式 Prefill 与 Decode

In an advanced deployment, a cluster orchestration system like Kubernetes can dynamically shift the GPU pool allocations—or scale the pools out separately—based on load and input characteristics. For instance, if many users arrive with superlong prompts and relatively small outputs (e.g., large-document summarization use cases), Kubernetes can temporarily shift the allocation to use more GPUs in the prefill pool. This will decrease the number of GPUs allocated to the decode phase.

在高级部署中，像 Kubernetes 这样的集群编排系统可以根据负载与输入特征，动态地调整 GPU 池的分配——或分别对各池进行扩缩容。例如，如果大量用户带着超长提示词与相对较短的输出到来（例如长文档摘要类用例），Kubernetes 可以临时调整分配，让 prefill 池使用更多 GPU，这会减少分配给 decode 阶段的 GPU 数量。

In contrast, if many users arrive requesting superlong outputs (e.g., long reasoning chains, “think step-by-step,” etc.), more GPUs can be shifted to the decode pool. In both cases, new instances can be scaled out for each worker type.

反之，如果大量用户请求超长输出（例如长推理链、“逐步思考”等），则可以把更多 GPU 转移到 decode 池。在这两种情况下，都可以为每种工作节点类型扩容出新的实例。

Figure 15-4 shows a distributed, Kubernetes-based vLLM cluster of separate prefill and decode workers using the open source llm-d project. vLLM implements disaggregated prefilling by running two instances and handing off KV using LMCache and NIXL, but llm-d extends this with Kubernetes native orchestration for disaggregated serving and KV-aware routing. This diagram shows a component called the *variant* *autoscaler*, which is responsible for updating the number of replicas for the prefill and decode workers in the pool.

图 15-4 展示了一个基于 Kubernetes 的分布式 vLLM 集群，它使用开源项目 llm-d，将 prefill 与 decode 工作节点分开。vLLM 通过运行两个实例并借助 LMCache 与 NIXL 交接 KV 来实现分离式 prefilling，而 llm-d 在此基础上以 Kubernetes 原生编排进行了扩展，实现了分离式服务与 KV 感知路由。该图展示了一个名为 *变体自动扩缩器*（variant autoscaler）的组件，它负责更新池中 prefill 与 decode 工作节点的副本数量。

![Figure 15-4. Kubernetes-based vLLM cluster of separate prefill and decode workers using the open source llm-d project; variant autoscaler tunes prefill/decode replica counts based on the prompt-response mix; KV moved using LMCache and NIXL](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-4.png)

![图 15-4. 基于 Kubernetes 的 vLLM 集群，使用开源项目 llm-d 将 prefill 与 decode 工作节点分开；变体自动扩缩器根据提示词-响应组合来调节 prefill/decode 的副本数量；KV 使用 LMCache 与 NIXL 搬运](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-4.png)

> In modern inference deployments, all nodes can perform both prefill and decode functionality since they all share the same runtime and code base (e.g., vLLM, SGLang, NVIDIA Dynamo, etc.). It’s up to the cluster orchestrator to assign them a specific role, either prefill or decode, statically upon startup—and dynamically throughout the worker’s lifecycle.

> 在现代推理部署中，所有节点都可以同时具备 prefill 与 decode 两种功能，因为它们共享相同的运行时与代码库（例如 vLLM、SGLang、NVIDIA Dynamo 等）。至于给它们分配具体角色——prefill 还是 decode——则由集群编排器决定：既可以在启动时静态指定，也可以在工作节点的整个生命周期中动态调整。

Overall, a disaggregated prefill/decode architecture provides a foundation for high-throughput, low-latency LLM serving. It does introduce complexity, however, as intermediate data must be transferred and managed—on the order of a few gigabytes for the KV cache of a long prompt. And scheduling is more involved, but the benefits of utilizing hardware efficiently are significant at ultra scale.

总体而言，分离式 prefill/decode 架构为高吞吐、低延迟的 LLM 服务奠定了基础。不过它确实引入了复杂性，因为中间数据必须被传输和管理——对于一个长提示词的 KV 缓存来说，其量级可达数 GB。而且调度也更为复杂，但在超大规模下，高效利用硬件所带来的收益是十分显著的。

> Chapters 17 and 18 dive deeper into additional techniques for disaggregated PD, including advanced scheduling, routing, and deployment optimizations.

> 第 17 章与第 18 章将更深入地探讨分离式 PD 的更多技术，包括高级调度、路由与部署优化。

## Parallelism Strategies for Serving Massive MoE Models

## 服务超大 MoE 模型的并行策略

Serving massive MoE models efficiently requires multiple forms of parallelism due to limited GPU memory. We break down the key parallelism strategies, including tensor, pipeline, expert, data, and context parallel. We’ll also discuss how they can be combined to distribute an LLM across many GPUs. Table 15-1 provides a high-level summary of these strategies, their typical use, and a description of each strategy in more detail as they relate to inference.

由于 GPU 内存有限，要高效地服务超大 MoE 模型，需要多种形式的并行。我们将拆解几种关键的并行策略，包括张量、流水线、专家、数据与上下文并行。我们还会讨论如何组合它们，把一个 LLM 分布到众多 GPU 上。表 15-1 从宏观上概括了这些策略、它们的典型用途，并就每种策略与推理的关系给出更详细的说明。

Table 15-1. Parallelism strategies for LLM inference

表 15-1. LLM 推理的并行策略

| Parallelism strategy | Partition basis | Use case | Pros | Cons |
| --- | --- | --- | --- | --- |
| Tensor parallelism | Within each layer (split neural network weight matrices across GPUs) | Single model is too large or you need to speed up heavy compute within layers across multiple GPUs | Near-linear speedup on compute-bound layers; Reduced overhead due to overlapping all-reduce communication with computation | Frequent communication at each layer (all-reduce); Requires high-bandwidth interconnect (NVLink/NVSwitch); Less efficient across nodes and slow networks due to high latencies (Recommended to keep tensor parallel groups within a single node) |
| Pipeline parallelism | Different layers on different GPUs (the model is sliced by layer sequence) | Extremely deep models that don't fit on one GPU; Memory scaling across layers | Allows distribution of model state; Uses microbatching to process multiple tokens from different users/requests concurrently; Multiple layers can be processed in parallel for long sequences (improves throughput for large batches or long inputs) | Adds pipeline fill/flush latency (bubbles)—not helpful for one-token-at-a-time decoding; More complex to implement and higher activation memory footprint (must store intermediate activations between pipeline stages) |
| Expert parallelism | Different MoE on different GPUs (sparse activation per token) | Massive MoE models with many experts; Needed to shard model parameters across GPUs | Enables virtually unlimited model size—total parameters scale with number of GPUs; Each GPU computes only a fraction of tokens (sparse compute); High parameter count boosts model capacity/quality | High runtime communication overhead (all-to-all) at each MoE layer; Potential load imbalance if gating is uneven; Each GPU must have enough work (tokens) per expert to amortize communication |
| Data parallelism | Replicate the entire model on multiple GPUs, serving different requests on each | Scaling out throughput (more concurrent requests) once model deployment is fixed; Multi-instance serving for many users | Nearly linear throughput scaling; Simple to implement (no model partitioning needed) | No latency improvement for individual queries (aka per-query latency); Multiplies memory usage (each replica uses full memory); Must handle consistency if stateful (e.g., caches) or use stateless model calls |
| Context parallelism | Partition the input sequence tokens across GPUs at each layer | Ultralong sequences (e.g., 100k+ tokens) to reduce prompt latency and memory per GPU | Achieves near-linear speedup for long context prefill; Enables processing contexts that exceed one GPU's memory by splitting KV cache | Requires custom attention algorithms to handle attention across partitions; Adds communication per layer for boundary tokens; Beneficial for very long contexts (100k+ tokens) due to additional communication overhead |

| 并行策略 | 划分依据 | 适用场景 | 优点 | 缺点 |
| --- | --- | --- | --- | --- |
| 张量并行 | 层内划分（将神经网络权重矩阵跨 GPU 拆分） | 单个模型过大，或需要跨多块 GPU 加速层内的繁重计算 | 在计算受限的层上接近线性加速；通过将 all-reduce 通信与计算重叠来降低开销 | 每一层都有频繁通信（all-reduce）；需要高带宽互连（NVLink/NVSwitch）；因高延迟而在跨节点及慢速网络上效率较低（建议将张量并行组保持在单个节点内） |
| 流水线并行 | 不同层放在不同 GPU 上（模型按层序切分） | 单块 GPU 装不下的极深模型；跨层的内存扩展 | 允许分布模型状态；使用微批处理（microbatching）并发处理来自不同用户/请求的多个 token；对于长序列可并行处理多层（提升大批量或长输入的吞吐量） | 增加流水线填充/清空延迟（气泡）——对逐个 token 的 decode 无益；实现更复杂，且激活内存占用更高（必须在流水线各阶段之间存储中间激活） |
| 专家并行 | 不同 MoE 放在不同 GPU 上（每 token 稀疏激活） | 具有大量专家的超大 MoE 模型；需要将模型参数跨 GPU 分片 | 支持几乎无限的模型规模——总参数量随 GPU 数量而扩展；每块 GPU 只计算一部分 token（稀疏计算）；高参数量提升模型容量/质量 | 每个 MoE 层都有很高的运行时通信开销（all-to-all）；若门控不均则可能出现负载不均衡；每块 GPU 上每个专家必须有足够的工作量（token）才能摊薄通信成本 |
| 数据并行 | 在多块 GPU 上复制整个模型，各自服务不同请求 | 在模型部署固定后横向扩展吞吐量（更多并发请求）；面向大量用户的多实例服务 | 吞吐量接近线性扩展；实现简单（无需模型划分） | 对单个查询的延迟（即每查询延迟）无改善；成倍增加内存占用（每个副本使用完整内存）；若有状态（例如缓存）则必须处理一致性，或采用无状态的模型调用 |
| 上下文并行 | 在每一层将输入序列的 token 跨 GPU 划分 | 超长序列（例如 100k+ tokens），以降低提示词延迟与每块 GPU 的内存 | 对长上下文 prefill 实现接近线性的加速；通过拆分 KV 缓存，支持处理超出单块 GPU 内存的上下文 | 需要自定义注意力算法来处理跨分区的注意力；每一层都为边界 token 增加通信；由于额外的通信开销，仅对极长上下文（100k+ tokens）有益 |

> For intranode TP on Blackwell NVL72, prefer keeping TP groups within a single NVSwitch domain; extend inter-rack only when topology permits to avoid extra hops.

> 对于 Blackwell NVL72 上的节点内 TP，优先将 TP 组保持在单个 NVSwitch 域内；仅在拓扑允许时才跨机架扩展，以避免额外的跳数（hop）。

These parallelism strategies define how the model weights and data are split over the GPUs. Figure 15-5 shows how they are split up for the different parallelism strategies—as well as common combinations of strategies.

这些并行策略定义了模型权重与数据如何在 GPU 之间拆分。图 15-5 展示了不同并行策略下它们是如何被拆分的——以及若干常见的策略组合。

![Figure 15-5. Model weights and input data split over the GPUs](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-5.png)

![图 15-5. 模型权重与输入数据在 GPU 之间的拆分](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-5.png)

### Tensor Parallelism

### 张量并行

*Tensor parallelism* (TP) splits the computations within each layer of the neural network across multiple GPUs. For instance, a large matrix multiply in a transformer layer can be partitioned by columns or rows—and computed in parallel on two or more GPUs. These GPUs then exchange their results by performing an all-reduce to aggregate their partial outputs.

*张量并行*（tensor parallelism，TP）将神经网络每一层内部的计算拆分到多块 GPU 上。例如，transformer 层中的一个大型矩阵乘法可以按列或按行划分——并在两块或更多 GPU 上并行计算。随后这些 GPU 通过执行一次 all-reduce 来交换结果，从而汇聚各自的部分输出。

TP is commonly used when a model’s layers, called *hidden layers*, are too large to fit into a single GPU’s memory—or when we want to accelerate a single instance of a model by utilizing multiple GPUs for the same layer in parallel. It keeps all GPUs in lockstep for each layer’s computation, which requires extremely high inter-GPU bandwidth to be efficient.

当模型的层（称为*隐藏层*，hidden layer）太大以致无法装入单块 GPU 的内存时，或者当我们希望通过让多块 GPU 并行处理同一层来加速某个模型实例时，通常会使用 TP。它让所有 GPU 在每一层的计算上保持步调一致（lockstep），因此需要极高的 GPU 间带宽才能高效运行。

Ideally, TP runs on GPUs that are connected with high-bandwidth NVLink and NVSwitch interconnects. On multi-GPU Blackwell systems that use fifth-generation NVLink (up to 1.8 TB/s aggregate bidirectional GPU-to-GPU bandwidth and advanced network topologies), TP can scale efficiently across a node of 8–16 GPUs—or even up to 72 GPUs in the case of an NVL72 rack. Remember, within the GB200/GB300 NVL72 racks, NVLink Switch provides about 130 TB/s of aggregate GPU bandwidth within the 72 GPU domain. This is a large amount of intra-rack bandwidth.

理想情况下，TP 应运行在通过高带宽 NVLink 与 NVSwitch 互连的 GPU 上。在使用第五代 NVLink（GPU 到 GPU 双向聚合带宽高达 1.8 TB/s，并具备先进网络拓扑）的多 GPU Blackwell 系统上，TP 能够在 8–16 块 GPU 的节点内高效扩展——在 NVL72 机架的情况下甚至可达 72 块 GPU。请记住，在 GB200/GB300 NVL72 机架内，NVLink Switch 在 72 GPU 域内提供约 130 TB/s 的聚合 GPU 带宽。这是相当可观的机架内带宽。

TP provides near-linear speedups for compute-bound portions of the model as long as collective communications, such as the all-reduce of activations, are fast relative to computation. In inference, tensor parallelism is mainly applied to the large matrix multiplications of the attention projections and feed-forward multilayer perceptron (MLP) networks.

只要诸如激活值 all-reduce 之类的集合通信相对于计算足够快，TP 就能为模型中计算受限的部分带来接近线性的加速。在推理中，张量并行主要应用于注意力投影与前馈多层感知机（MLP）网络的大型矩阵乘法。

Since the activations for one token are relatively small, those all-reduce communications are not too costly on NVLink. As such, TP is a common strategy for splitting giant dense models across GPUs without approximating model weights.

由于单个 token 的激活值相对较小，这些 all-reduce 通信在 NVLink 上的代价并不太高。因此，TP 是在不近似模型权重的前提下、把巨型稠密模型拆分到多块 GPU 上的常用策略。

We mainly use TP to serve models in which single MoE experts or transformer layers are too large for a single GPU. However, we also use it when we want to reduce latency by parallelizing each layer’s computations across multiple GPUs.

我们主要在单个 MoE 专家或 transformer 层太大以致无法装入单块 GPU 时使用 TP。不过，当我们希望通过在多块 GPU 上并行化每一层的计算来降低延迟时，也会使用它。

One must be mindful that beyond a certain scale, however, TP efficiency drops—especially when using it across nodes or racks running over InfiniBand or Ethernet. In practice, it’s best to use TP for intranode communication—or between nodes in an NVLink domain.

然而，必须注意的是，超过某个规模后 TP 的效率会下降——尤其是在跨节点或跨机架、运行于 InfiniBand 或 Ethernet 之上时。在实践中，最好将 TP 用于节点内通信——或用于同一 NVLink 域内的节点之间。

Modern multi-GPU racks like NVIDIA’s GB200/GB300 NVL72 allow TP to be used at a much larger scale without saturating interconnect bandwidth. NVLink Switch extends the NVLink domain across the rack to 72 GPUs per NVL72. NVLink Switch Systems can scale an all-to-all, fully connected NVLink fabric across up to 8 racks, or 576 GPUs (576 GPUs = 8 racks * 72 GPUs per rack). This enables large-scale parallelism strategies, not just inside a single NVL72 but also across multiple racks. For instance, in an 8-rack topology, you can increase to 576-way tensor parallelism using the 576 GPUs across 8 racks (576 GPUs = 8 racks × 72 GPUs per rack.)

像 NVIDIA GB200/GB300 NVL72 这样的现代多 GPU 机架，允许在更大规模上使用 TP 而不至于耗尽互连带宽。NVLink Switch 将 NVLink 域跨机架扩展到每个 NVL72 的 72 块 GPU。NVLink Switch System 能够把一个 all-to-all、全连接的 NVLink 架构扩展到最多 8 个机架、即 576 块 GPU（576 GPU = 8 机架 * 72 GPU/机架）。这使得大规模并行策略不仅能在单个 NVL72 内部使用，还能跨多个机架使用。例如，在 8 机架拓扑中，你可以利用跨 8 机架的 576 块 GPU 扩展到 576 路张量并行（576 GPU = 8 机架 × 72 GPU/机架）。

> While it’s possible to extend TP across racks, it’s recommended to choose your TP group sizes with topology-awareness in mind and within a single NVLink/NVSwitch island whenever possible. This will avoid unnecessary inter-rack switch latency and help to increase overall system efficiency.

> 尽管可以将 TP 跨机架扩展，但建议在选择 TP 组规模时考虑拓扑感知，并尽可能保持在单个 NVLink/NVSwitch 岛屿之内。这将避免不必要的机架间交换机延迟，并有助于提升整体系统效率。

### Pipeline Parallelism

### 流水线并行

*Pipeline parallelism* (PP) partitions the model *layer-wise* across different GPUs. For example, in a 60-layer transformer-based model, GPU 0 might hold layers 1–20, GPU 1 holds layers 21–40, and GPU 2 holds layers 41–60.

*流水线并行*（pipeline parallelism，PP）按*层*（layer-wise）将模型划分到不同的 GPU 上。例如，在一个基于 transformer 的 60 层模型中，GPU 0 可能持有第 1–20 层，GPU 1 持有第 21–40 层，GPU 2 持有第 41–60 层。

When processing a sequence of tokens, the data flows through the GPUs in sequence such that GPU 0 computes layers 1–20, then passes its intermediate activations to GPU 1 for layers 21–40, and so on. This allows models that are too deep to fit on one accelerator to be distributed.

在处理一段 token 序列时，数据依次流经各 GPU：GPU 0 计算第 1–20 层，然后把它的中间激活传给 GPU 1 计算第 21–40 层，以此类推。这使得深到无法装入单个加速器的模型得以被分布。

In inference, PP can improve throughput by partitioning the model across multiple batches—similar to an assembly line. During the *prefill* phase, which processes a long sequence of input tokens, PP achieves high GPU utilization by streaming different portions of the sequence into the layers in staggered fashion.

在推理中，PP 可以通过将模型划分到多个批次之间来提升吞吐量——类似于流水装配线。在处理长输入 token 序列的 *prefill* 阶段，PP 通过以错峰方式把序列的不同部分流式送入各层，从而实现高 GPU 利用率。

In contrast, during the *decode* phase, generating one token at a time, pure PP offers less benefit since each new token must still pass through the pipeline stages sequentially. This creates pipeline bubbles, or idle periods, in which earlier stages wait for later ones to finish for each and every token.

相比之下，在一次生成一个 token 的 *decode* 阶段，纯 PP 带来的收益较小，因为每个新 token 仍必须顺序地穿过各个流水线阶段。这会产生流水线气泡（pipeline bubble），即空闲期：对每一个 token，靠前的阶段都要等待靠后的阶段完成。

To reduce pipeline bubbles, implementations use microbatching, which allows the pipeline to process multiple tokens from different requests concurrently. Still, pipeline parallelism primarily helps with memory scaling by enabling very large models to split layers among GPUs. It also helps throughput—especially when handling large batch sizes or long inputs.

为减少流水线气泡，各种实现会使用微批处理，它允许流水线并发处理来自不同请求的多个 token。尽管如此，流水线并行主要还是通过让超大模型把层拆分到多块 GPU 上来帮助内存扩展。它同样有助于吞吐量——尤其是在处理大批量或长输入时。

PP tends to increase end-to-end latency for a single item due to transfer overhead between pipeline stages. As such, PP is usually chosen for model capacity reasons—rather than latency reasons. This is because it can fit the model into memory. The slight latency hit is acceptable when no single GPU can hold the whole model.

由于流水线各阶段之间的传输开销，PP 往往会增加单个条目的端到端延迟。因此，PP 通常出于模型容量而非延迟方面的考虑而被选用。这是因为它能把模型装入内存。当没有任何单块 GPU 能容纳整个模型时，轻微的延迟代价是可以接受的。

Additionally, PP is often used in combination with other parallelism strategies like TP to balance memory and speed. When serving large MoE models, one might use pipeline parallelism across 2–4 GPUs to split the deep layer stack—while relying on TP to handle the wide, intralayer compute.

此外，PP 常与 TP 等其他并行策略组合使用，以在内存与速度之间取得平衡。在服务大型 MoE 模型时，可以跨 2–4 块 GPU 使用流水线并行来拆分很深的层堆栈——同时依靠 TP 来处理宽的层内计算。

> PP splits the model across layers, and TP splits the model within layers. TP is mainly used for models with layers or experts that are too wide for a single GPU, while PP is primarily used for models that are too deep to fit into a single GPU. Note that TP and PP can be combined, which we’ll see in a bit.

> PP 跨层拆分模型，而 TP 在层内拆分模型。TP 主要用于层或专家宽到超出单块 GPU 的模型，而 PP 主要用于深到无法装入单块 GPU 的模型。请注意，TP 与 PP 可以组合使用，稍后我们会看到这一点。

### Expert Parallelism

### 专家并行

*Expert parallelism* (EP) is specific to MoE architectures. In an MoE layer, there are many expert networks, or feed-forward sublayers. For each input token, only one or a few experts are activated. This naturally lends itself to distributing different experts on different GPUs.

*专家并行*（expert parallelism，EP）专用于 MoE 架构。在一个 MoE 层中，存在许多专家网络（即前馈子层）。对于每个输入 token，只有一个或少数几个专家被激活。这天然地适合把不同的专家分布到不同的 GPU 上。

For instance, if an MoE layer has 16 experts and we have 4 GPUs, each GPU could host 4 experts. During inference, when a token arrives at that MoE layer, its internal gating network will choose the top two experts, for instance, for that token. The token’s data is then sent to whichever GPUs own those two experts for processing. Then the results are combined back to generate the next token, as shown in Figure 15-6.

例如，如果一个 MoE 层有 16 个专家，而我们有 4 块 GPU，则每块 GPU 可以承载 4 个专家。在推理时，当一个 token 到达该 MoE 层，其内部的门控网络会为该 token 选出（比如）排名前二的两个专家。该 token 的数据随后被发送到拥有这两个专家的相应 GPU 进行处理。接着，结果被合并回来以生成下一个 token，如图 15-6 所示。

![Figure 15-6. Mixture-of-experts (MoE) communication (source: https://oreil.ly/pzn5t)](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-6.png)

![图 15-6. 专家混合（MoE）通信（来源：https://oreil.ly/pzn5t）](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-6.png)

Implementing this requires an all-to-all communication pattern such that tokens (technically, their activation vectors) are dynamically shuffled between GPUs so that each token lands on the GPU of its assigned expert. After each expert computes its output for its assigned tokens, another all-to-all is performed to return the token outputs back to their original order.

实现这一点需要一种 all-to-all 通信模式，使得 token（严格来说是它们的激活向量）在 GPU 之间被动态打乱，从而让每个 token 落到其被分配专家所在的 GPU 上。在每个专家为其分配到的 token 计算出输出之后，会再执行一次 all-to-all，把各 token 的输出按原始顺序返回。

This dynamic routing introduces a communication-heavy step at each MoE layer. This can become a performance bottleneck if not carefully optimized—especially as the number of experts/GPUs increases. The upside is that expert parallelism allows the total model capacity to scale almost linearly with the number of GPUs.

这种动态路由在每个 MoE 层都引入了一个通信密集的步骤。如果不加以精心优化，它可能成为性能瓶颈——尤其是随着专家/GPU 数量的增加。好处在于，专家并行使得模型总容量能够随 GPU 数量近乎线性地扩展。

Consider an MoE with 100 experts distributed across 100 GPUs. If each token activates only two experts, the compute per token is similar to a smaller dense model. As such, expert parallelism is what allows some massive MoE models to be served at all since the model weights are sharded across many GPUs—and each GPU handles only a fraction of the tokens at any given layer.

设想一个拥有 100 个专家、分布在 100 块 GPU 上的 MoE。如果每个 token 只激活两个专家，那么每 token 的计算量就与一个较小的稠密模型相当。因此，正是专家并行使得某些超大 MoE 模型得以被服务，因为模型权重被分片到众多 GPU 上——每块 GPU 在任一层只处理一小部分 token。

> At scale, and with a properly balanced set of experts in an MoE model, all of the experts will likely be active simultaneously. So, while an individual token activates only a small number of experts, in aggregate across all tokens across all end users, all experts will be active, consuming and contending for all GPU resources concurrently.

> 在大规模下，且当 MoE 模型中的专家集合被恰当地均衡时，所有专家很可能会同时处于激活状态。因此，尽管单个 token 只激活少数几个专家，但把所有终端用户的所有 token 汇总起来看，所有专家都会处于激活状态，同时消费并争用全部 GPU 资源。

Efficient expert parallel inference requires careful load balancing—as well as high-bandwidth interconnects, such as NVLink/NVSwitch for intra-rack communication and InfiniBand/RDMA for internode communication. Modern MoE inference frameworks use fast collective communication libraries like NCCL. It often helps to group multiple experts per GPU to reduce communication steps.

高效的专家并行推理需要精心的负载均衡（load balancing）——以及高带宽互连，例如用于机架内通信的 NVLink/NVSwitch 和用于节点间通信的 InfiniBand/RDMA。现代 MoE 推理框架使用像 NCCL 这样的快速集合通信库。把每块 GPU 上的多个专家分组，通常有助于减少通信步骤。

Modern MoE inference often uses top-2 gating, in which each token is assigned to two experts. To reduce communication, you can group commonly paired experts onto the same GPU or compute node. For instance, if tokens use two experts, placing those two frequently paired experts on the same GPU or node means that many token assignments will stay on a single GPU or node, which will localize communication traffic and reduce overhead.

现代 MoE 推理常使用 top-2 门控，即每个 token 被分配给两个专家。为减少通信，你可以把经常配对的专家分到同一块 GPU 或同一计算节点上。例如，如果 token 使用两个专家，把这两个经常配对的专家放在同一块 GPU 或同一节点上，就意味着许多 token 的分配会停留在单块 GPU 或单个节点内，从而把通信流量本地化并减少开销。

Another technique is to use top-1 gating, in which each token goes to only one expert. This will reduce the communication volume relative to top-2 gating described previously, which doubles the number of expert outputs. While top-1 gating is faster, it can lead to a lower model quality and uneven load.

另一种技术是使用 top-1 门控，即每个 token 只去往一个专家。相比前面描述的会使专家输出数量翻倍的 top-2 门控，这会降低通信量。虽然 top-1 门控更快，但它可能导致模型质量下降与负载不均。

> Google’s GLaM introduced load-balancing losses (and gating noise) to achieve balanced expert usage during MoE model training. Building on this, researchers are exploring truly adaptive, inference-time gating that uses real-time load metrics to reroute tokens when an expert is overloaded. These improve utilization without degrading quality. However, in most production serving environments, top-2 gating with a modest capacity factor—and occasional load-based expert swapping or replication—remains the most common compromise technique to balance both quality and performance.

> Google 的 GLaM 引入了负载均衡损失（以及门控噪声），以在 MoE 模型训练期间实现均衡的专家使用。在此基础上，研究者正在探索真正自适应的、推理时的门控：当某个专家过载时，利用实时负载指标重新路由 token。这些做法在不降低质量的前提下提升了利用率。然而，在大多数生产服务环境中，配以适度容量因子的 top-2 门控——外加偶尔基于负载的专家交换或复制——仍是同时兼顾质量与性能的最常见折中技术。

Many MoE LLMs use at least top-2 gating for better quality, good speed, and balanced load. Additionally, it’s important to place experts strategically—or even use backup expert replicas to avoid overloading experts.

许多 MoE LLM 至少使用 top-2 门控，以获得更好的质量、良好的速度与均衡的负载。此外，战略性地放置专家——甚至使用备份专家副本——以避免某些专家过载，是很重要的。

When experts become overloaded, or hot, they can become throughput bottlenecks if the gating network is disproportionately routing tokens to them. The *straggler effect*, as it’s called, requires that all expert computations be complete before they can progress. As such, an overloaded expert that receives more work (tokens) than other experts will stall the inference pipeline. This will leave some experts, and their GPUs, idle while the other experts catch up.

当专家变得过载、或者说变“热”时，如果门控网络不成比例地把 token 路由给它们，它们就会成为吞吐量瓶颈。所谓的*掉队者效应*（straggler effect）要求所有专家计算全部完成后才能继续推进。因此，一个接收到比其他专家更多工作的过载专家，会使推理流水线停滞。这会让一些专家及其 GPU 空闲，等待其他专家追赶上来。

To prevent this, high-performance MoE serving systems expose a capacity factor parameter, typically set at 1.2–1.5× the average token load, which caps how many tokens each expert can process per batch. Tokens beyond that are routed to a second-choice overflow expert—or queued for a second pass. This complements any load-balancing loss or gating noise used during training to encourage the MoE to assign tokens to experts uniformly.

为防止这种情况，高性能 MoE 服务系统会暴露一个容量因子（capacity factor）参数，通常设为平均 token 负载的 1.2–1.5×，用于限制每个专家每批次最多能处理多少 token。超出该上限的 token 会被路由到次优的溢出（overflow）专家——或排队等待第二轮处理。这与训练期间用来促使 MoE 均匀地把 token 分配给各专家的任何负载均衡损失或门控噪声相互补充。

Some full-featured inference servers can also replicate hot experts onto multiple GPUs to split the load if necessary. This comes at a cost of additional GPU memory. We will see an example of load imbalance a bit later.

一些功能完备的推理服务器还能在必要时把热点专家复制到多块 GPU 上以分摊负载。这以额外的 GPU 内存为代价。稍后我们会看到一个负载不均衡的例子。

The combination of inference-time spillover (capacity factor), training-time penalties (load-balancing loss and gating noise), and hot-expert replicas should smooth out the token-expert load distribution when capacity is reached—maximizing overall MoE inference throughput.

推理时溢出（容量因子）、训练时惩罚（负载均衡损失与门控噪声）以及热点专家副本三者的组合，应能在达到容量上限时平滑 token-专家的负载分布——从而最大化整体 MoE 推理吞吐量。

### Data Parallelism

### 数据并行

In inference, *data parallelism* (DP) refers to replicating the entire model on multiple GPUs and assigning different incoming requests—or shards of a request batch—to each GPU. Unlike data parallelism in training, which keeps models in sync with gradient averaging, data parallelism in inference runs independent forward passes on each replica. This is the simplest way to scale out throughput.

在推理中，*数据并行*（data parallelism，DP）是指把整个模型复制到多块 GPU 上，并将不同的传入请求——或一个请求批次的分片——分配给每块 GPU。与训练中通过梯度平均保持模型同步的数据并行不同，推理中的数据并行在每个副本上运行独立的前向传播。这是横向扩展吞吐量最简单的方式。

For instance, with DP, if one GPU can handle 10 requests per second, using 8 GPUs with 8 replicas ideally gives 80 requests per second of throughput—assuming the requests are independent. In practice, using data parallelism for inference requires spinning up multiple model-serving engines and load-balancing requests among them.

例如，采用 DP 时，如果一块 GPU 每秒能处理 10 个请求，那么使用 8 块 GPU、8 个副本，理想情况下可提供每秒 80 个请求的吞吐量——前提是这些请求彼此独立。在实践中，将数据并行用于推理需要启动多个模型服务引擎，并在它们之间对请求做负载均衡。

When using DP, each GPU, or group of GPUs, handles a subset of queries from start to finish. The advantage is linear scaling of throughput and no inter-GPU communication during inference since each replica runs in isolation. The disadvantages are significant, however.

使用 DP 时，每块 GPU（或每组 GPU）从头到尾处理一部分查询。其优点是吞吐量线性扩展，且推理期间没有 GPU 间通信，因为每个副本都是隔离运行的。然而，其缺点也十分显著。

DP multiplies the memory requirement since each replica needs a full copy of model weights and cache. As such, serving a model with eight replicas uses 8× the GPU memory—and roughly 8× the hardware cost—compared to hosting the model on a single GPU.

DP 会成倍增加内存需求，因为每个副本都需要一份完整的模型权重与缓存副本。因此，用 8 个副本服务一个模型，相比把该模型托管在单块 GPU 上，会使用 8× 的 GPU 内存——以及大约 8× 的硬件成本。

In practice, data parallelism is often combined with request batching and multistream execution on each GPU to maximize utilization. Each replica should ideally be stateless—or use synchronized caches to avoid consistency issues.

在实践中，数据并行常与请求批处理以及每块 GPU 上的多流（multistream）执行相结合，以最大化利用率。理想情况下，每个副本应当是无状态的——或者使用同步缓存以避免一致性问题。

It’s important to note that DP does not reduce latency for any single request. It does, however, improve throughput since there are more replicas available to handle requests—assuming the memory and compute resources are available and relatively free of contention.

需要注意的是，DP 并不会降低任何单个请求的延迟。不过，由于有更多可用副本来处理请求，它确实能提升吞吐量——前提是内存与计算资源可用且相对没有争用。

> Because GPU memory is relatively scarce and expensive, DP is usually worthwhile for inference only when your throughput needs are very high—or if you can combine DP with other parallelism approaches, such as TP and PP.

> 由于 GPU 内存相对稀缺且昂贵，通常只有当你的吞吐量需求非常高时——或者当你能把 DP 与 TP、PP 等其他并行方法组合起来时——DP 才值得用于推理。

For inference, DP is often used in combination with other parallelism approaches, such as TP and PP, due to the additional memory and cost requirements of DP. For example, if a model must be spread across eight GPUs due to memory limitations, it should use DP with TP, PP, or EP—or even all of them together—to create four-dimensional (4D) parallelism (add in context parallelism (CP), and you’re using 5D parallelism!).

对于推理，由于 DP 有额外的内存与成本需求，它常与 TP、PP 等其他并行方法组合使用。例如，如果一个模型因内存限制必须铺开到 8 块 GPU 上，就应当把 DP 与 TP、PP 或 EP 组合起来——甚至把它们全部一起使用——以构成四维（4D）并行（再加上上下文并行（CP），你就用上了 5D 并行！）。

Specifically, you can use DP with TP to deploy two large, 8-GPU model replicas using two DP groups of eight GPUs. This will double the throughput and shard each of the large model replicas across eight GPUs within its layers using TP.

具体而言，你可以用 DP 结合 TP，通过两个各含 8 块 GPU 的 DP 组来部署两个大型的 8-GPU 模型副本。这将使吞吐量翻倍，并用 TP 在层内把每个大型模型副本分片到 8 块 GPU 上。

In a massive inference cluster, you can dedicate a large number of GPUs and nodes to serve multiple copies of the large model to handle high request volumes. DP is particularly useful when throughput needs to scale beyond what a single model instance can provide. In practice, production systems often run many DP replicas of the large model to meet traffic demands.

在一个大规模推理集群中，你可以专门划出大量 GPU 与节点来服务大模型的多个副本，以应对高请求量。当吞吐量需求需要扩展到超出单个模型实例所能提供的水平时，DP 尤为有用。在实践中，生产系统常运行大模型的许多 DP 副本以满足流量需求。

> Modern inference servers treat each replica as a separate model instance behind a load balancer. This requires careful request routing, but it’s relatively straightforward compared to other complex model-sharding methods.

> 现代推理服务器把每个副本都视为负载均衡器背后的一个独立模型实例。这要求对请求做精细的路由，但相比其他复杂的模型分片方法，它相对简单直接。

### Context (Sequence) Parallelism

### 上下文（序列）并行

*Context parallelism* (CP) is more of a specialized strategy that partitions a single sequence of tokens across multiple GPUs. This technique benefits extremely long context inputs on the order of tens of thousands and millions of tokens. These would otherwise be too slow or memory-heavy or simply not fit on a single GPU.

*上下文并行*（context parallelism，CP）更像是一种专门化的策略，它把单条 token 序列切分到多块 GPU 上。这一技术对数万乃至数百万 token 量级的超长上下文输入很有帮助——否则这些输入要么太慢、太耗内存，要么根本无法装进单块 GPU。

The idea behind CP is to split the sequence into chunks and have different GPUs handle different parts of the sequence in parallel at each layer. The GPUs exchange only the necessary information at the boundary between chunks of the sequence.

CP 的思路是把序列切成若干块，让不同的 GPU 在每一层并行处理序列的不同部分。GPU 之间只在序列块的边界处交换必要的信息。

CP handles contexts larger than a single GPU’s memory by splitting the KV cache across GPUs. It also reduces prompt-processing latency roughly in proportion to the number of GPUs used. As such, CP can achieve near-linear speedup for very long context prefill runs by using multiple GPUs to perform the prefill for a long context in parallel.

CP 通过把 KV 缓存切分到多块 GPU 上，来处理超出单块 GPU 内存的上下文。它还能让提示处理延迟大致随所用 GPU 数量成比例下降。因此，CP 可以对超长上下文的 prefill 运行实现接近线性的加速——用多块 GPU 并行完成长上下文的 prefill。

The challenge with CP is that transformers have global self-attention such that every token attends to every earlier token. Naively splitting the sequence would require lots of information exchange between GPUs to compute attention across the partition boundary.

CP 的挑战在于，transformer 具有全局自注意力：每个 token 都要关注它之前的所有 token。天真地切分序列，会需要在 GPU 之间进行大量信息交换，以便跨切分边界计算注意力。

CP methods use clever schemes like ring parallelism and blocked attention to reduce the quadratic self-attention communication time complexity—and limit each GPU to attending mostly within its partition of the context as well as a small amount of chunk-boundary data.

CP 方法采用环形并行（ring parallelism）和分块注意力（blocked attention）等巧妙方案，来降低自注意力二次方级的通信时间复杂度——并把每块 GPU 限制为主要关注它自己那部分上下文，外加少量块边界数据。

In effect, each GPU handles a subset of positions in the input sequence for each layer. They pass intermediate results around in a pipelined fashion—often arranged in a ring. CP is analogous to pipeline parallelism but along the sequence-length dimension instead of the layer dimension.

实际上，每块 GPU 在每一层都处理输入序列中的一部分位置。它们以流水线方式——通常排成一个环——来回传递中间结果。CP 类似于流水线并行，只不过是沿序列长度维度而非层维度进行。

Context parallelism excels at prefill for very long inputs by slicing a document into segments, allocating segments across multiple GPUs, and processing each segment concurrently. This reduces prompt-processing time roughly in half for every doubling of GPUs—with only a bit of overhead for cross-segment attention at the boundaries.

上下文并行在处理超长输入的 prefill 时表现出色：它把一份文档切成若干段，把这些段分配到多块 GPU 上，并发处理每一段。每当 GPU 数量翻倍，提示处理时间大约减半——只在边界处的跨段注意力上带来一点点开销。

In short, while CP doesn’t speed up the sequential, token-by-token decode phase, it can shorten TTFT for extremely long prompts. It does this by distributing the attention KV cache across devices such that each GPU stores only its segment’s cache. CP also lets you handle contexts larger than what a single GPU can fit into memory.

简而言之，虽然 CP 无法加速逐 token 串行的 decode 阶段，但它能缩短超长提示的 TTFT。做法是把注意力 KV 缓存分布到多个设备上，让每块 GPU 只存储自己那一段的缓存。CP 还能让你处理超出单块 GPU 内存容量的上下文。

> Context parallelism requires an attention implementation that communicates across partitions. This adds extra per-layer communication. As such, for short prompts, CP often adds outsized overhead. But for very long inputs and strict TTFT goals, CP can reduce prefill latency and memory per GPU. Measure both prefill speed and accuracy for your maximum context.

> 上下文并行需要一种能跨切分通信的注意力实现。这会增加每层额外的通信。因此，对于短提示，CP 往往带来过大的开销。但对于超长输入和严格的 TTFT 目标，CP 可以降低 prefill 延迟和每块 GPU 的内存占用。请针对你的最大上下文同时测量 prefill 速度与准确性。

### Hybrid Parallelism

### 混合并行

In practice, serving massive MoE LLMs uses a combination of the previous parallelism strategies. Today’s LLM models are so large and complex that no single parallelization method is sufficient. Figure 15-7 shows a hybrid parallel configuration using four GPUs.

在实践中，服务超大 MoE 大模型要综合使用前面介绍的各种并行策略。如今的大模型如此庞大而复杂，任何单一的并行方法都不够用。图 15-7 展示了一种使用四块 GPU 的混合并行配置。

![Figure 15-7. High-level diagram of a 4 × 2 × EP hybrid parallel combination (source: https://oreil.ly/q1AEf)](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-7.png)

![图 15-7. 一种 4 × 2 × EP 混合并行组合的高层示意图（来源：https://oreil.ly/q1AEf）](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-7.png)

Here, we are using four pipeline stages (one per GPU) and two-way tensor parallelism. The tokens are routed across two experts using expert parallelism. This is called a *4 × 2 EP hybrid parallel strategy*.

这里，我们使用了四个流水线阶段（每块 GPU 一个）和两路张量并行。token 通过专家并行被路由到两个专家上。这被称为 *4 × 2 EP 混合并行策略*。

Let’s go even larger and create logic groups of GPUs. For example, consider a cluster of 64 GPUs. We can group the GPUs into 16 groups of 4 GPUs each. Using an MoE with 60 layers and 64 experts, we can use 4-way pipeline parallelism with 15 layers per stage. This splits the depth of the model. Then, within each stage (15 layers), we can use 2-way TP to split the heavy matrix multiplies within each layer. This splits the width of the model.

我们再把规模做得更大，构建 GPU 的逻辑分组。例如，考虑一个 64 块 GPU 的集群。我们可以把这些 GPU 分成 16 组，每组 4 块 GPU。假设 MoE 有 60 层、64 个专家，我们可以用 4 路流水线并行，每个阶段 15 层。这切分的是模型的深度。然后，在每个阶段（15 层）内部，我们可以用 2 路 TP 来切分每层内部沉重的矩阵乘法。这切分的是模型的宽度。

For the MoE layers, you can use EP to spread 16 experts per group using top-2 gating. This will send tokens to, at most, two GPUs in the group. Finally, data parallelism can deploy two such 64-GPU replicas to double your system’s throughput.

对于 MoE 层，你可以用 EP，配合 top-2 门控，把每组 16 个专家铺开。这样最多会把 token 发送到组内两块 GPU。最后，数据并行可以部署两份这样的 64 块 GPU 副本，让系统吞吐量翻倍。

This is just one configuration. In practice, you’ll need to experiment with different combinations of parallelism strategies. Your profiling tools will help you find the right balance. Specifically, you can use Nsight Systems for end-to-end traces and Nsight Compute for kernel-specific GPU metrics.

这只是一种配置。在实践中，你需要对不同的并行策略组合进行实验。你的性能剖析工具会帮你找到合适的平衡点。具体来说，你可以用 Nsight Systems 做端到端追踪，用 Nsight Compute 获取特定内核的 GPU 指标。

> Remember to always verify interconnect traffic, Tensor Core utilization, and other performance-enhancing mechanisms before and after making each change.

> 记住，在每次改动前后，都要核对互连流量、Tensor Core 利用率以及其他性能增强机制。

The guiding principle is to use TP up to the point of diminishing returns—usually within a node or in a tightly coupled unit like an NVL72 chassis. You would then use pipeline parallelism minimally—just enough to fit the model in memory. Then, you want to maximize EP to distribute MoE parameters across experts and GPUs. Finally, you add data parallel replicas to improve throughput as you scale to more and more end users and concurrent inference requests (CP can be optionally layered on top if extremely long inputs are expected).

指导原则是：把 TP 用到收益递减的临界点为止——通常在一个节点内，或在像 NVL72 机箱这样紧耦合的单元内。然后，尽量少用流水线并行——只用到刚好能把模型装进内存的程度。接着，你要最大化 EP，把 MoE 参数分散到各个专家和各块 GPU 上。最后，随着你扩展到越来越多的终端用户和并发推理请求，再增加数据并行副本来提升吞吐量（如果预期会有超长输入，可以选择性地在其上叠加 CP）。

We also want to align parallelization with our hardware topology. For instance, if using an NVL72 system, all 72 GPUs are fully connected using NVLink and NVSwitch. In this case, you can form TP and EP groups within the 72 GPU NVLink domain. However, TP groups that approach the full domain will often see diminishing returns from the all-reduce latency. As such, many production systems keep TP groups smaller and topology aware—even inside NVL72.

我们还希望让并行方式与硬件拓扑相匹配。例如，如果使用 NVL72 系统，全部 72 块 GPU 都通过 NVLink 和 NVSwitch 全互连。这种情况下，你可以在这 72 块 GPU 的 NVLink 域内组建 TP 和 EP 分组。不过，接近整个域规模的 TP 分组，往往会因 all-reduce 延迟而出现收益递减。因此，许多生产系统会让 TP 分组保持较小并具备拓扑感知——即便是在 NVL72 内部也是如此。

In contrast, a smaller cluster of two 8-GPU nodes connected with InfiniBand has full NVLink connectivity only within each node—with cross-node traffic traveling along the InfiniBand fabric. In such environments, it’s best to keep TP local to each 8-GPU node—and avoid internode parallelism if possible to avoid the higher latency and lower bandwidth of internode communications.

相比之下，一个由两个 8 卡节点通过 InfiniBand 连接的较小集群，只有在每个节点内部才有完整的 NVLink 连接——跨节点流量要走 InfiniBand 网络。在这类环境中，最好把 TP 保持在每个 8 卡节点本地——并尽量避免节点间并行，以规避节点间通信更高的延迟和更低的带宽。

In the next sections, we discuss higher-level optimizations that can be combined with these parallelism strategies to achieve even faster inference on multinode clusters. As we shall see, a well-designed serving system will combine many of these high-level and low-level techniques.

在接下来的几节里，我们会讨论一些更高层的优化，它们可以与这些并行策略结合，在多节点集群上实现更快的推理。正如我们将看到的，一个设计良好的服务系统会把许多这类高层与底层技术结合起来。

## Speculative Decoding and Parallel Token Generation Techniques

## 投机解码与并行 token 生成技术

One of the fundamental performance challenges in LLM inference is the sequential nature of decoding. Remember that after the initial prompt is processed during prefill, the model typically generates one token at a time. Each token’s computation depends on the result of the previous token, so this is difficult to parallelize as it’s inherently a serial process, which introduces a latency bottleneck—even with the latest, fastest GPUs. Generating hundreds of tokens sequentially can take on the order of seconds for massive models on modern GPU hardware.

大模型推理的根本性能挑战之一，是 decode 的串行本质。请记住，在 prefill 阶段处理完初始提示之后，模型通常一次只生成一个 token。每个 token 的计算都依赖于前一个 token 的结果，因此很难并行化——它本质上是一个串行过程，会引入延迟瓶颈——即便用上最新、最快的 GPU 也是如此。在现代 GPU 硬件上，对超大模型而言，串行生成数百个 token 可能耗时数秒量级。

In this section, we discuss several techniques to accelerate the decode stage. Some of these techniques involve generating multiple tokens per inference step, while others reduce the number of sequential inference steps altogether.

在本节中，我们讨论若干加速 decode 阶段的技术。其中一些技术是在每个推理步生成多个 token，另一些则从根本上减少串行推理步的数量。

Speculative decoding traditionally pairs a small, fast “draft” model that proposes several tokens in one batch with a larger “target” model that then validates each candidate. This trades a second inference pass for parallel generation to achieve higher, multitoken throughput. However, running two separate models adds deployment complexity and can still stall inference when the verification pass becomes a bottleneck.

投机解码传统上是把一个小而快的“草稿”（draft）模型与一个更大的“目标”（target）模型配对：前者一次成批地提出若干 token，后者随后验证每个候选。这以第二次推理为代价换取并行生成，从而实现更高的多 token 吞吐。然而，运行两个独立的模型会增加部署复杂度，而且当验证过程成为瓶颈时，仍可能拖慢推理。

Medusa simplifies this by attaching multiple lightweight decoding heads directly to a single LLM. At each step, these heads use a tree-based attention mechanism to concurrently generate and verify several token candidates within a single forward pass.

Medusa 对此做了简化：它把多个轻量级的解码头直接附加到单个大模型上。在每一步，这些头使用一种基于树的注意力机制，在一次前向传播中并发生成并验证若干 token 候选。

This unified design of Medusa avoids cross-model token transfers and achieves large speedups without sacrificing the quality of the token generation. By consolidating draft-and-verify into a single-model pass, Medusa is an improvement over conventional two-model speculative decoding techniques. Let’s take a closer look at some of these techniques.

Medusa 的这种统一设计避免了跨模型的 token 传输，在不牺牲 token 生成质量的前提下取得了大幅加速。通过把“起草与验证”合并进单模型的一次传播，Medusa 相较传统的双模型投机解码技术是一大改进。下面我们来更仔细地看看其中一些技术。

### Two-Model, Draft-Based Speculative Decoding and EAGLE

### 双模型、基于草稿的投机解码与 EAGLE

Speculative decoding is a technique that trades extra work on a small model to save time on the expensive large model. The idea is to run a lightweight “draft” LLM alongside the main LLM. The draft model is faster and generates a batch of *k* tokens speculatively beyond the current context.

投机解码是一种技术：它在小模型上多做一些工作，以节省昂贵大模型上的时间。其思路是让一个轻量级的“草稿”大模型与主大模型并肩运行。草稿模型更快，会在当前上下文之外投机性地生成一批 *k* 个 token。

The big model, called the *target* model, then validates the draft tokens by predicting next-token probabilities for the entire *k*-token sequence in a single batch sent to the GPU. By handling all *k* tokens at once, the target model increases arithmetic intensity by performing more computations per byte of data transferred. This results in efficient and fast verification of the draft sequence tokens.

这个大模型被称为*目标*模型，随后它会通过在一个发往 GPU 的批次中，对整段 *k* 个 token 的序列预测下一个 token 的概率，来验证草稿 token。通过一次性处理全部 *k* 个 token，目标模型提高了算术强度——每传输一字节数据就执行更多的计算。这样就能高效而快速地验证草稿序列的 token。

If the target model’s output agrees with the draft model’s *k* proposed tokens, then we have effectively generated *k* tokens in the same time that a large model would have generated a single token. If the large model’s verification diverges from the draft model’s prediction at any token in the sequence of *k* tokens generated by the draft model, the speculative tokens beyond that point are discarded. At least one generated token will be kept in each step because the verification procedure always accepts the first token from either the draft or the target model.

如果目标模型的输出与草稿模型提出的 *k* 个 token 一致，那么我们就在大模型生成单个 token 的同样时间内，有效地生成了 *k* 个 token。如果在草稿模型生成的这 *k* 个 token 序列中，大模型的验证在任一 token 处与草稿模型的预测发生分歧，那么该点之后的投机 token 就会被丢弃。每一步至少会保留一个已生成的 token，因为验证过程总会接受来自草稿模型或目标模型的第一个 token。

Once the target model’s corrected token is used and decoding continues, a new speculative decoding cycle can start from that point. Figure 15-8 shows a small draft model predicting multiple tokens ahead. Then the target (big) model verifies these tokens one by one.

一旦用上目标模型修正后的 token 并继续解码，就可以从该点开始一轮新的投机解码循环。图 15-8 展示了一个小的草稿模型向前预测多个 token，随后目标（大）模型逐个验证这些 token。

Over time, speculative decoding reduces the overall number of one-by-one, sequential invocations needed by the large model. In theory, it provides a theoretical k× speedup, where *k* is the number of tokens generated by the draft model. In practice, with overhead and occasional speculative-token rejections, the gain is more like a 2× speedup.

随着时间推移，投机解码减少了大模型所需的逐个串行调用的总次数。理论上，它提供 k× 的理论加速，其中 *k* 是草稿模型生成的 token 数。而在实践中，由于存在开销和偶尔的投机 token 拒绝，收益更接近 2× 的加速。

![Figure 15-8. Speculative decoding with a draft (small) for decoding and a target (large) model for multitoken verification](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-8.png)

![图 15-8. 用草稿（小）模型做解码、目标（大）模型做多 token 验证的投机解码](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-8.png)

The draft model must be chosen to have reasonably high fidelity to the large model’s distribution. This means that the draft model’s predictions should have high overlap with the large model’s likely outputs. If the draft frequently guesses wrong, speculative decoding provides little benefit and wastes compute. If it often predicts tokens that the larger target model would not, many speculative tokens will be rejected. This will waste compute time.

草稿模型的选择必须对大模型的分布有相当高的保真度。这意味着草稿模型的预测应当与大模型的可能输出有高度重叠。如果草稿模型频繁猜错，投机解码收益甚微，还会浪费算力。如果它经常预测出更大的目标模型不会给出的 token，许多投机 token 就会被拒绝。这会浪费计算时间。

The draft model must also be much faster than the large model—typically by a factor of 4× or more—for speculative decoding to provide a practical performance improvement. A common strategy is to use a distilled version of the main model—or a smaller model fine-tuned on the same data so that its outputs correlate well.

草稿模型还必须比大模型快得多——通常要快 4× 或更多——投机解码才能带来实际的性能提升。一种常见策略是使用主模型的蒸馏版本——或一个在相同数据上微调过、使其输出关联度良好的更小模型。

> The draft model must use the same tokenizer and vocabulary as the large model. This is sometimes overlooked, and it will lead to poor results.

> 草稿模型必须使用与大模型相同的分词器和词表。这一点有时会被忽视，而它会导致糟糕的结果。

The draft model generates tokens using the same prompt and perhaps a higher temperature to increase the chance of matching one of the large model’s probable continuations. Meanwhile, the target model skips ahead and processes all of the draft tokens in one single batch.

草稿模型使用相同的提示——也许再配上更高的温度——来生成 token，以提高命中大模型某个概率延续的机会。与此同时，目标模型向前跳跃，在单个批次中处理全部草稿 token。

Under the standard speculative decoding acceptance procedure, the target model’s output distribution is preserved. As such, the final samples match the large model’s distribution when sampling settings are aligned. If the draft diverges, the target model’s verification rejects those tokens and correctness is maintained. The only difference is that up to *k* number of target-model calls are avoided when the small model guesses correctly for these speculative tokens.

在标准的投机解码接受过程下，目标模型的输出分布得以保留。因此，当采样设置对齐时，最终的样本与大模型的分布一致。如果草稿发生分歧，目标模型的验证会拒绝那些 token，从而维持正确性。唯一的区别在于：当小模型对这些投机 token 猜对时，最多可以省去 *k* 次目标模型调用。

Speculative decoding can be implemented in a variety of ways. Most modern LLM inference engines like vLLM, SGLang, and TensorRT-LLM have built-in support to coordinate draft and target model generation. Empirically, speedups around 1.5–2.5× are common, and higher gains are possible with careful batching and draft model design.

投机解码可以用多种方式实现。大多数现代大模型推理引擎，如 vLLM、SGLang 和 TensorRT-LLM，都内置了协调草稿模型与目标模型生成的支持。经验上，1.5–2.5× 左右的加速很常见，而通过精细的批处理和草稿模型设计，还可能获得更高的收益。

> In PyTorch, the operation torch.nn.functional.scaled_dot_product_attention auto-selects the optimal backend (FlashAttention or a cuDNN kernel) based on device generation, shapes, mask, and dtype. You can pin a backend using torch.nn.attention.sdpa_kernel() with an explicit SDPABackend type. It’s important to verify that the fused backend is active when benchmarking LLM decoding, for instance.

> 在 PyTorch 中，torch.nn.functional.scaled_dot_product_attention 这个操作会根据设备代次、张量形状、掩码和数据类型，自动选择最优后端（FlashAttention 或某个 cuDNN 内核）。你可以用 torch.nn.attention.sdpa_kernel() 并指定明确的 SDPABackend 类型来固定某个后端。例如在对大模型解码做基准测试时，确认所用的融合后端确实处于激活状态就很重要。

For instance, the Extrapolation Algorithm for Greater LLM Efficiency (EAGLE) algorithm rethinks speculative decoding by operating at the feature level rather than the token level. EAGLE uses a one-step extrapolation of the large model’s own intermediate representation to predict the next token’s features. This resolves uncertainty and achieves higher acceptance rates.

举例来说，“面向更高大模型效率的外推算法”（Extrapolation Algorithm for Greater LLM Efficiency，EAGLE）重新思考了投机解码：它在特征层面而非 token 层面运作。EAGLE 用一步外推大模型自身的中间表示，来预测下一个 token 的特征。这消解了不确定性，并取得更高的接受率。

EAGLE reported up to about 3.5× speedup over vanilla decoding for a 4-token draft while preserving the large model’s output distribution. EAGLE approaches the theoretical limit of its draft depth in favorable environments. And it shows that, with innovative techniques, speculative decoding can reduce decoding time even further—and preserve output quality at the same time.

EAGLE 报告称，在有利环境下，对 4-token 草稿相较原始解码可获得约 3.5× 的加速，同时保留大模型的输出分布。EAGLE 在有利环境下逼近了其草稿深度的理论极限。它表明，凭借创新技术，投机解码可以进一步缩短解码时间——同时保持输出质量。

EAGLE-2 extends EAGLE by introducing a context-aware dynamic *draft tree* of possibilities. While EAGLE achieved up to 3.5× speedup versus vanilla decoding in some evaluations, EAGLE-2’s approach reported speedups of about 20%–40% faster than EAGLE depending on the task and model. With EAGLE-2, the draft model generates a branching set of token sequences, which further increases parallelism in the verification step. This is shown in Figure 15-9.

EAGLE-2 通过引入一棵上下文感知的动态*草稿树*（draft tree）扩展了 EAGLE。在某些评测中，EAGLE 相较原始解码达到了最高 3.5× 的加速；而 EAGLE-2 的方案报告称，视任务和模型不同，比 EAGLE 快约 20%–40%。有了 EAGLE-2，草稿模型会生成一组分支的 token 序列，从而在验证步进一步提升并行度。如图 15-9 所示。

EAGLE-3 continues to improve on earlier versions, EAGLE-1 and EAGLE-2, by preferring direct token prediction and fusing multi-layer features. EAGLE-3 reports up to 1.4× improvements over EAGLE-2 in certain tasks—and up to 6.5× speedups over non-optimized baseline variants. In EAGLE-1 and EAGLE-2, the draft model predicts internal feature vectors which are then decoded to tokens. These earlier EAGLE methods worked by guessing internal features, essentially—and then mapping them to tokens.

EAGLE-3 在早先版本 EAGLE-1 和 EAGLE-2 的基础上继续改进：它倾向于直接预测 token，并融合多层特征。EAGLE-3 报告称在某些任务上相较 EAGLE-2 有最高 1.4× 的提升——相较未优化的基线变体则有最高 6.5× 的加速。在 EAGLE-1 和 EAGLE-2 中，草稿模型预测的是内部特征向量，然后再解码为 token。这些较早的 EAGLE 方法，本质上是靠猜测内部特征——再把它们映射到 token。

EAGLE-3 skips the feature-level prediction step and predicts the draft tokens more directly. However, it still uses internal features but fused into multiple layer representations (lower, middle, upper) to guide the draft predictions. This is in contrast to using just the top layer. This makes EAGLE-3 more streamlined and less constrained—allowing better scaling.

EAGLE-3 跳过了特征层面的预测步骤，更直接地预测草稿 token。不过，它仍然使用内部特征，但把这些特征融合进多层表示（低层、中层、高层）来引导草稿预测。这与仅使用顶层形成对比。这让 EAGLE-3 更精简、约束更少——从而具备更好的可扩展性。

![Figure 15-9. Speculative decoding with EAGLE-2 (source: https://oreil.ly/uG07b)](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-9.png)

![图 15-9. 使用 EAGLE-2 的投机解码（来源：https://oreil.ly/uG07b）](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-9.png)

Another technique is *dynamic depth decoding*, which can adaptively skip layers that minimize the impact on output quality. Other techniques that reduce computations include skipping every *N*th transformer layer, using lower precision (e.g., FP8 and NVFP4) for the draft model, and using a smaller hidden size temporarily for the draft stage.

另一种技术是*动态深度解码*（dynamic depth decoding），它可以自适应地跳过那些对输出质量影响最小的层。其他能减少计算量的技术包括：每隔 *N* 层跳过一个 transformer 层、对草稿模型使用更低精度（如 FP8 和 NVFP4），以及在草稿阶段临时使用更小的隐藏维度。

> Some research prototypes offer modes that skip portions of the network or reduce precision during decoding to trade accuracy for speed. As of this writing, these are not universally available in production models, so always validate quality and speed on your target tasks before enabling any such mode.

> 一些研究原型提供了在解码期间跳过网络部分或降低精度的模式，以准确性换速度。截至撰写本文时，这些模式在生产模型中尚未普遍可用，因此在启用任何此类模式之前，务必在你的目标任务上验证质量与速度。

Future LLMs might include an optimized “fast-generation” mode. For example, a model might have alternate lightweight layers or a configuration for reduced-precision decoding. This built-in optimization would allow the model to skip computations.

未来的大模型可能会内置一种优化过的“快速生成”模式。例如，一个模型可能有备选的轻量层，或有一套降低精度解码的配置。这种内置的优化将允许模型跳过部分计算。

### Single-Model Self-Speculative Decoding

### 单模型自投机解码

Another approach to speculative decoding is to avoid using an external draft model altogether and use the larger target model to both draft and verify its own outputs by selectively skipping some computations. One such method is self-speculative decoding, also called the draft-and-verify scheme.

投机解码的另一种思路是彻底不使用外部草稿模型，而是让更大的目标模型通过选择性地跳过一些计算，来同时起草并验证它自己的输出。其中一种方法就是自投机解码（self-speculative decoding），也称为“起草-验证”方案。

With self-speculative decoding, the large target model generates *k* tokens using a fast, approximate pass. For instance, it can choose to run only half of its layers—and possibly in reduced precision—for each new token. It would skip every other layer. This produces draft output much like the smaller draft model would. And it can approach similar speedups when acceptance rates and draft depth are favorable. This produces draft output much like the smaller draft model would—and achieves a similar 2× speedup as well.

在自投机解码中，大目标模型用一次快速的近似传播来生成 *k* 个 token。例如，它可以对每个新 token 只运行一半的层——并且可能以降低的精度运行。它会每隔一层跳过一层。这样产生的草稿输出，很像那个更小的草稿模型会产生的输出。而且当接受率和草稿深度有利时，它能逼近类似的加速。这样产生的草稿输出很像那个更小的草稿模型会产生的输出——并且同样能取得约 2× 的加速。

Then, in a second stage of self-speculative decoding, the target model performs a full forward pass to verify those *k* tokens in one go. If they all match, then we save executing half the layers for those tokens. If not, we simply fall back to the traditional speculative approach by accepting tokens up to the mismatch and proceeding normally.

然后，在自投机解码的第二阶段，目标模型执行一次完整的前向传播，一次性验证这 *k* 个 token。如果它们全部匹配，我们就为这些 token 省去了执行一半层的开销。如果不匹配，我们就直接退回到传统的投机方式：接受到不匹配处为止的 token，然后正常继续。

Because it’s the same model doing both draft and verification in self-speculative decoding, no separate model needs to be trained, maintained, or loaded into memory at runtime. The challenge is finding a good way to reduce the amount of compute needed in the draft stage (dropping layers, reducing precision, etc.) without hurting accuracy too much.

由于在自投机解码中是同一个模型既起草又验证，就无需在运行时训练、维护或加载一个独立的模型到内存中。挑战在于，找到一种好办法来减少草稿阶段所需的计算量（丢层、降精度等），同时又不过多损害准确性。

A related technique is *consistent decoding*, in which you train one LLM to both generate and validate multiple tokens. This single-model approach produces ~3× speedups without a separate draft model. This shows a trend of baking speculative decoding into the model’s own weights.

一种相关技术是*一致性解码*（consistent decoding），你要训练一个大模型来同时生成并验证多个 token。这种单模型方法在不需要独立草稿模型的情况下产生约 3× 的加速。这体现出一种趋势：把投机解码烘焙进模型自身的权重里。

These methods represent very active areas of research. And they are particularly exciting and promising because they let the model accelerate itself using its inherent internal redundancy. And since the optimizations are local to the model, the inference engine’s implementation can be simplified.

这些方法代表着非常活跃的研究领域。它们尤其令人兴奋、前景广阔，因为它们让模型利用自身固有的内部冗余来自我加速。而且由于这些优化局限在模型本地，推理引擎的实现可以得到简化。

### Multitoken Decoding with Medusa’s Multiple Heads

### 用 Medusa 的多头做多 token 解码

Speculative decoding still ultimately generates tokens one by one using the draft model—it’s just faster because the draft model is smaller. However, the Medusa framework takes a more radical approach. It modifies the model architecture itself to predict multiple new tokens in parallel for each decoding step.

投机解码归根结底仍是用草稿模型逐个生成 token——只是因为草稿模型更小才更快。然而，Medusa 框架采取了更激进的做法。它直接修改模型架构本身，在每个解码步并行预测多个新 token。

Unlike two-model speculative decoding, which still generates tokens one by one (albeit faster), Medusa’s architecture truly generates multiple tokens per iteration from a single model. As such, Medusa’s multiheaded approach has reported about 2.2–3.6× speedups in published experiments across both Medusa-1 and Medusa-2.

不同于仍逐个生成 token（尽管更快）的双模型投机解码，Medusa 的架构真正做到了从单个模型每次迭代生成多个 token。因此，在已发表的实验中，横跨 Medusa-1 和 Medusa-2，Medusa 的多头方案报告了约 2.2–3.6× 的加速。

However, Medusa requires custom model training since it modifies the transformer-based LLM with additional decoder heads that branch off at certain layers—hence, the name *Medusa*. This lets the model propose several next tokens simultaneously.

不过，Medusa 需要定制的模型训练，因为它对基于 transformer 的大模型做了修改，添加了若干在特定层分叉出去的额外解码头——名字 *Medusa*（美杜莎）正由此而来。这让模型可以同时提出若干个下一个 token。

The multiple token candidates generated by the different Medusa heads are structured like a tree. For instance, Medusa can generate a binary tree of depth 2 in one pass to produce up to 4 tokens. It can then verify the sequence of multiple tokens in parallel, as shown in Figure 15-10.

由不同 Medusa 头生成的多个 token 候选被组织成树状结构。例如，Medusa 可以在一次传播中生成一棵深度为 2 的二叉树，产出多达 4 个 token。随后，它可以并行验证这一串多个 token，如图 15-10 所示。

![Figure 15-10. Multitoken decoding with Medusa (source: https://oreil.ly/MJMOQ)](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-10.png)

![图 15-10. 用 Medusa 做多 token 解码（来源：https://oreil.ly/MJMOQ）](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-10.png)

Internally, Medusa uses a specialized, tree-based attention pattern to improve consistency among the parallel token predictions. The model learns to extend and verify the multiple-token sequences concurrently. With Medusa, the LLM essentially learns to “think ahead” by a few tokens—and output them all at once.

在内部，Medusa 使用一种专门的、基于树的注意力模式，来提升并行 token 预测之间的一致性。模型学会同时扩展并验证多 token 序列。有了 Medusa，大模型本质上学会了向前“预想”几个 token——并一次性把它们全部输出。

During inference, Medusa can reduce the number of sequential decoding iterations by the factor of the parallel predictions. For instance, if Medusa predicts 4 tokens in one pass, it can reduce the required number of forward passes by up to 4× for depth-2 binary tree outputs.

在推理时，Medusa 可以把串行解码迭代的次数按并行预测的倍数缩减。例如，如果 Medusa 一次传播预测 4 个 token，对于深度为 2 的二叉树输出，它可以把所需的前向传播次数减少最多 4×。

In practice, however, Medusa typically achieves about a 2–3× speedup due to the overhead of occasionally having to backtrack when a prediction branch fails validation and needs a partial redo. This is similar to the overhead of speculative decoding verification and rejection. For example, if a Medusa LLM generates 4 tokens per iteration, a reply of 100 tokens would take only 25 iterations of the model instead of 100.

然而在实践中，Medusa 通常只能取得约 2–3× 的加速，因为偶尔会出现某个预测分支验证失败、需要部分重做时的回溯开销。这与投机解码验证与拒绝的开销类似。举例来说，如果一个 Medusa 大模型每次迭代生成 4 个 token，那么 100 个 token 的回复只需模型迭代 25 次，而不是 100 次。

Medusa requires modifying and retraining the model to add these extra heads. In addition, Medusa models are slightly larger in size given the additional head parameters. This increases development complexity and training cost a bit. However, once trained, Medusa models can significantly reduce inference times.

Medusa 需要修改并重新训练模型来添加这些额外的头。此外，由于多了头的参数，Medusa 模型的体积会略大一些。这会略微增加开发复杂度和训练成本。不过，一旦训练完成，Medusa 模型可以显著缩短推理时间。

If trained for your use case, a Medusa-enabled LLM can provide excellent low-latency performance. However, this technique requires modifying and fine-tuning the model architecture to add the additional heads.

如果针对你的用例进行了训练，一个启用 Medusa 的大模型可以提供出色的低延迟性能。然而，这一技术要求修改并微调模型架构以添加额外的头。

### Interleaving Decode Steps from Multiple Requests

### 交错来自多个请求的解码步

Another parallel decoding technique is to interleave decode steps from multiple concurrent requests across multiple end users. This parallelism is more of an inference-engine capability than an algorithmic trick or model architecture modification. But it helps keep the GPUs busy by batching at the token level—and across end-user requests.

另一种并行解码技术，是把跨多个终端用户的多个并发请求的解码步交错起来。这种并行更像是推理引擎的一种能力，而非某种算法技巧或模型架构改动。但它有助于让 GPU 保持忙碌——办法是在 token 层面、跨终端用户请求进行批处理。

Frameworks like vLLM implement this in the form of request routing, continuous batching, and token scheduling. The idea is to keep the GPUs busy by filling in the gaps between sequential steps. Specifically, if a GPU is stalled due to a sequence waiting on I/O or a data dependency, the scheduler can run another sequence’s next token prediction on that same GPU in the meantime.

像 vLLM 这样的框架，以请求路由、连续批处理（continuous batching）和 token 调度的形式实现了这一点。其思路是通过填补串行步之间的空隙来让 GPU 保持忙碌。具体来说，如果某块 GPU 因为某个序列在等待 I/O 或数据依赖而停顿，调度器可以在此期间在同一块 GPU 上运行另一个序列的下一个 token 预测。

> Combining an inference engine’s advanced routing, batching, and scheduling capabilities with CUDA streams. Then verify with Nsight Systems that token-step kernels overlap with NIC/NVLink transfers rather than serializing on a single stream. Then you can achieve massive parallelism at the application, system, and hardware levels.

> 把推理引擎先进的路由、批处理和调度能力与 CUDA 流结合起来。然后用 Nsight Systems 验证 token 步的内核确实与 NIC/NVLink 传输重叠，而不是在单个流上串行化。这样你就能在应用、系统和硬件层面实现大规模并行。

It’s important to note that interleaving decode steps doesn’t speed up a single sequence’s latency. In fact, it can even add a tiny bit of overhead per token due to the additional context switching. However, it can greatly improve overall throughput and GPU utilization when serving many users concurrently. This will reduce the average end-user request latency under heavy load.

需要指出的是，交错解码步并不会加速单个序列的延迟。事实上，由于额外的上下文切换，它甚至可能给每个 token 增加一丁点开销。然而，当并发服务众多用户时，它能大幅提升整体吞吐量和 GPU 利用率。在重负载下，这会降低终端用户请求的平均延迟。

### Combining Decoding Techniques and Evaluating Complexity

### 组合解码技术并评估复杂度

It’s worth noting that these decoding optimizations can be combined. For instance, one could use speculative decoding with a Medusa-enabled target model in which the big target model verifies multiple tokens at a time predicted by a small model.

值得注意的是，这些解码优化可以组合使用。例如，你可以把投机解码与一个启用 Medusa 的目标模型结合起来，让大的目标模型一次性验证由小模型预测出的多个 token。

These techniques also come at the cost of additional complexity. Either you have to maintain an extra model, alter the main model, or manage more elaborate control logic. In production, you should evaluate this complexity against the performance gains. For interactive applications, reducing response time by even a few milliseconds is worth the complexity—especially at scale.

这些技术也伴随着额外复杂度的代价。你要么得维护一个额外的模型，要么得改动主模型，要么得管理更繁复的控制逻辑。在生产中，你应当权衡这种复杂度与性能收益。对于交互式应用，哪怕只把响应时间缩短几毫秒也值得为之付出复杂度——尤其是在大规模场景下。

In short, advanced decoding techniques like speculative decoding—with or without a draft model—and Medusa-style multitoken prediction can reduce overall inference times of modern autoregressive LLMs that traditionally generate one token at a time. By generating more tokens at a time, increasing overall arithmetic intensity, and pushing the decoding step toward the compute-bound regime, you can take advantage of the GPU’s extreme compute capabilities.

简而言之，投机解码（无论用不用草稿模型）以及 Medusa 式多 token 预测这类先进解码技术，可以缩短传统上一次只生成一个 token 的现代自回归大模型的整体推理时间。通过一次生成更多 token、提高整体算术强度，并把解码步推向计算受限区间，你就能充分利用 GPU 极强的计算能力。

> The decoding-optimization landscape is continuously evolving and growing in complexity; the trend is clear: make LLM decoding more parallel and efficient.

> 解码优化的格局在持续演进，复杂度也在不断增长；但趋势很明确：让大模型解码更并行、更高效。

## Constrained Decoding Performance Implications

## 受约束解码的性能影响

A separate but important aspect of the LLM decoding process is generating text under certain constraints. For instance, the model can force the output to match a predefined format (JSON), use a particular grammar, or disallow certain sequences for safety. As such, constrained decoding, often called *structured outputs*, is often required for production use cases.

大模型解码过程中一个独立但重要的方面，是在特定约束下生成文本。例如，模型可以强制输出匹配某种预定义格式（JSON）、使用特定语法，或出于安全考虑禁止某些序列。因此，受约束解码——常被称为*结构化输出*（structured outputs）——在生产用例中往往是必需的。

OpenAI’s function-calling API, for example, relies on well-structured output formats to determine which function to call—as well as the functions’ inputs and outputs. These constraints are designed to match the function’s API signature and are simply nonnegotiable at the application level.

以 OpenAI 的函数调用 API 为例，它依赖良好结构化的输出格式来决定调用哪个函数——以及这些函数的输入与输出。这些约束被设计为匹配函数的 API 签名，在应用层面根本没有商量的余地。

Constrained decoding alters the model’s token-selection process so that, at each step, only valid tokens are allowed. This might be as simple as supplying a list of allowed tokens—or as sophisticated as embedding a formal grammar and using a state machine to filter out invalid tokens.

受约束解码改变了模型的 token 选择过程，使得在每一步只允许有效的 token。这可能简单到只是提供一份允许的 token 列表——也可能复杂到嵌入一套形式语法，并用一个状态机来过滤掉无效的 token。

The drawback of constrained decoding is that extra latency is needed to enforce these constraints—often a few milliseconds per token. This is because the model may need to backtrack repeatedly, navigate the pruned vocabulary, and ultimately generate a valid token. Backtracking happens inside the LLM’s generation loop. As such, debugging, profiling, and tuning constrained decoding can be difficult without fine-grained telemetry in the model itself.

受约束解码的缺点在于，强制执行这些约束需要额外的延迟——通常每个 token 几毫秒。这是因为模型可能需要反复回溯、在被剪枝的词表中穿行，最终才生成一个有效的 token。回溯发生在大模型的生成循环内部。因此，如果模型自身缺乏细粒度的遥测，受约束解码的调试、剖析和调优会很困难。

Another performance consideration is cache efficiency. Pruning the full probability distribution at every decode step can blow caches and reduce throughput even further. This is particularly noticeable when the allowed token set is heavily limited.

另一个性能考量是缓存效率。在每个解码步对完整的概率分布做剪枝，可能冲垮缓存，进一步降低吞吐量。当允许的 token 集被严重限制时，这一点尤其明显。

To mitigate these costs, many frameworks compile a JSON grammar and precompute valid tokens for each state. This allows the inference engine to mask out invalid softmax outputs at runtime. This approach reduces backtracking, improves caching, and raises acceptance rates.

为缓解这些开销，许多框架会编译一套 JSON 语法，并为每个状态预先计算好有效的 token。这让推理引擎可以在运行时掩蔽掉无效的 softmax 输出。这种做法减少了回溯、改善了缓存，并提高了接受率。

For moderate constraints such as simple JSON schemas, modern engines that use compiled grammars and overlap mask computation with GPU execution can push overhead into the low single-digit percent range at scale, but complex grammars or small batches can still incur double-digit overhead. Always measure TTFT and tokens per second under your exact schema.

对于适中的约束（如简单的 JSON 模式），使用已编译语法、并把掩码计算与 GPU 执行重叠起来的现代引擎，可以在大规模下把开销压到个位数低百分比区间；但复杂的语法或小批次仍可能带来两位数的开销。请始终在你确切的模式下测量 TTFT 和每秒 token 数。

These techniques have been implemented in popular libraries and inference engines, including Hugging Face Transformers, NVIDIA NeMo, and vLLM. For instance, NVIDIA’s TensorRT-LLM and Hugging Face Transformers both allow user-defined vocabulary masks to enforce constraints with negligible speed penalty. Specifically, TensorRT-LLM exposes guided decoding with JSON schema and context-free grammar (CFG) support through the XGrammar backend. And vLLM and Transformers provide structured output APIs.

这些技术已在流行的库和推理引擎中实现，包括 Hugging Face Transformers、NVIDIA NeMo 和 vLLM。例如，NVIDIA 的 TensorRT-LLM 和 Hugging Face Transformers 都允许用户自定义词表掩码来强制约束，且速度损失可忽略不计。具体来说，TensorRT-LLM 通过 XGrammar 后端暴露了带 JSON 模式和上下文无关文法（context-free grammar，CFG）支持的引导式解码。而 vLLM 和 Transformers 则提供了结构化输出 API。

> When using TensorRT-LLM’s guided decoding, XGrammar supplies JSON-schema and CFG support on-GPU. XGrammar compiles constraints and avoids large Python-side token-mask overhead. However, be aware that certain configurations may fall back to slower backends and cause excessive first-request stalls. As such, it’s important to keep grammars compact to preserve cache locality and token-mask bandwidth.

> 使用 TensorRT-LLM 的引导式解码时，XGrammar 在 GPU 上提供 JSON 模式和 CFG 支持。XGrammar 会编译约束，避免了大量的 Python 侧 token 掩码开销。不过要注意，某些配置可能回退到较慢的后端，导致首次请求出现过度停顿。因此，让语法保持紧凑很重要，以保住缓存局部性和 token 掩码带宽。

Another popular strategy is to push constrained decoding into the kernel itself. In this case, it would inject token masks during the final softmax step to achieve similar performance gains.

另一种流行策略是把受约束解码下推到内核本身。这种情况下，它会在最后的 softmax 步注入 token 掩码，以取得类似的性能收益。

When implemented efficiently, constrained decoding can often run as fast as unconstrained decoding—especially if the constraint ruleset is not too large or too restrictive. If the decoding constraints are too restrictive, decoding effectively becomes a search through a small token space. This leads to more backtracking and slower token generation.

在实现得当时，受约束解码往往可以跑得和无约束解码一样快——尤其是当约束规则集不太大也不太严苛时。如果解码约束过于严苛，解码实际上就变成了在一个很小的 token 空间里做搜索。这会导致更多回溯和更慢的 token 生成。

> When possible, avoid large or overly restrictive constraints, such as grammars, formats, and vocabularies. One option is to let the LLM decode as normal and simply postprocess or filter the outputs. However, this is not always a performant or viable option. Profile the different options and choose what’s best for your use case.

> 在可能的情况下，避免使用过大或过于严苛的约束，如语法、格式和词表。一个选项是让大模型正常解码，之后再对输出做后处理或过滤。然而，这并不总是高性能或可行的选项。请对不同选项做剖析，为你的用例选出最优方案。

## Dynamic Routing Strategies for MoE Inference

## 面向 MoE 推理的动态路由策略

Serving MoE models efficiently requires careful partitioning of experts across GPUs using expert parallelism. In addition, you need an intelligent and dynamic mechanism, or gating network, to dynamically route tokens to these experts at runtime.

要高效地服务 MoE 模型，需要用专家并行把专家精细地划分到各块 GPU 上。此外，你还需要一种智能且动态的机制，即门控网络（gating network），在运行时把 token 动态路由到这些专家。

During MoE inference, each token’s forward pass must be routed by the gating network to one or more expert GPUs. The system needs to handle this communication in a balanced way to keep the GPUs busy—and avoid overloaded GPUs, or hotspots. Let’s examine how to address these in a high-scale, multinode inference environment.

在 MoE 推理期间，每个 token 的前向传播都必须由门控网络路由到一个或多个专家 GPU。系统需要以均衡的方式处理这种通信，以让 GPU 保持忙碌——并避免过载的 GPU，即热点。下面我们来看看如何在高规模的多节点推理环境中解决这些问题。

### Expert Communication Optimization

### 专家通信优化

During the all-to-all exchange of token activations across experts, an MoE shuffles a batch of tokens between the GPUs. Each GPU receives only the tokens it needs for the experts that it hosts, as shown in Figure 15-11. This happens for every layer in the MoE and is a costly operation, which can potentially dominate inference time if not handled efficiently.

在跨专家进行 token 激活的 all-to-all 交换期间，MoE 会在各块 GPU 之间打乱并重排一批 token。每块 GPU 只接收它所托管的那些专家需要的 token，如图 15-11 所示。这一过程会在 MoE 的每一层发生，是一项代价高昂的操作——如果处理不当，它可能会主导整个推理时间。

![Figure 15-11. Each GPU receives only the tokens it needs for the expert(s) that it hosts](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-11.png)

![图 15-11. 每块 GPU 只接收它所托管的专家所需要的 token](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-11.png)

One strategy to reduce communication overhead is to use a hierarchical routing strategy for our GPU cluster by first routing tokens between GPUs within the same node using NVSwitch/NVLink (fast) and routing across nodes only for any remaining tokens that need nonlocal experts. This two-stage all-to-all can reduce the volume of internode traffic.

一种降低通信开销的策略，是为我们的 GPU 集群采用分层路由策略：先用 NVSwitch/NVLink（快）在同一节点内的各块 GPU 之间路由 token，只对剩下那些需要非本地专家的 token 才跨节点路由。这种两阶段的 all-to-all 可以减少节点间流量的体量。

Additionally, you can use asynchronous communication by overlapping the communication with computation. High-performance MoE inference servers use double-buffer communications so that while one batch of tokens is being sent around, the previous batch’s expert computations occur in parallel. This pipelining hides much of the communication latency.

此外，你可以通过把通信与计算相互重叠来使用异步通信。高性能的 MoE 推理服务器采用双缓冲通信：在一批 token 被来回发送的同时，上一批 token 的专家计算并行地进行。这种流水线化隐藏了大部分通信延迟。

> It’s possible to achieve near-optimal all-to-all completion times when the internode fabric is kept fully utilized while intranode shuffles run in the background. Be sure to measure link utilization on both the internode NIC and intranode NVLink paths.

> 当节点间网络保持充分利用、而节点内的打乱在后台运行时，是有可能实现接近最优的 all-to-all 完成时间的。请务必同时测量节点间 NIC 与节点内 NVLink 两条路径上的链路利用率。

A naive implementation of expert routing that uses a single global all-to-all barrier can leave GPUs waiting on synchronization overhead. This wastes bandwidth if not all links are utilized optimally.

一种天真的专家路由实现，如果使用单个全局 all-to-all 屏障，会让 GPU 卡在同步开销上等待。如果并非所有链路都被最优地利用，这会浪费带宽。

Techniques such as a *butterfly schedule* (aka *shifted all-to-all schedule*) can break the communication into phased rounds. This way, every NVLink/NIC is busy with partial exchanges—as opposed to one big synchronization. This staggered approach improves link utilization and reduces idle time.

诸如*蝶形调度*（butterfly schedule，又称*移位 all-to-all 调度*，shifted all-to-all schedule）之类的技术，可以把通信拆分成分阶段的多轮。这样一来，每条 NVLink/NIC 都在忙于部分交换——而不是挤在一次大同步里。这种错开的方式改善了链路利用率，减少了空闲时间。

All-to-all exchanges may use the built-in ncclAllToAll collective or grouped send and receive calls. Throughput often improves when the exchange is chunked and pipelined or when a hierarchical schedule such as butterfly is used across nodes. Validate and choose the algorithm that matches your topology.

all-to-all 交换可以使用内置的 ncclAllToAll 集合通信，或使用成组的 send/receive 调用。当交换被分块并流水线化，或跨节点采用蝶形之类的分层调度时，吞吐量往往会提升。请验证并选择与你的拓扑相匹配的算法。

> Keep first-stage all-to-all intra-rack (NVLink) and spill inter-rack only for residual tokens. Profile link utilization to confirm NICs are saturated while NVLink shuffles run in the background. Also, when internode links are the bottleneck, it’s recommended to double-buffer MoE all-to-all communication with expert computation. You can use chunked/pipelined exchanges and butterfly/shifted schedules to avoid global barrier slowdowns and outperform flat global collectives.

> 让第一阶段的 all-to-all 保持在机架内（NVLink），只把残余的 token 溢出到机架间。剖析链路利用率，以确认 NIC 处于饱和状态、同时 NVLink 打乱在后台运行。此外，当节点间链路成为瓶颈时，建议把 MoE 的 all-to-all 通信与专家计算做双缓冲。你可以使用分块/流水线化的交换以及蝶形/移位调度，来避免全局屏障造成的减速，并胜过扁平（非分层）全局集合通信。

Another solution to reduce communication traffic is called *expert collocation.* The idea is to collocate certain experts together on the same GPU or node to avoid unnecessary communication. Consider two experts, experts 5 and 7, that are often activated for the same token by the token router. Placing experts 5 and 7 on the same GPU can eliminate an extra all-to-all hop. Profiling tools and gating-frequency analysis can help identify such pairings for your workload.

另一种降低通信流量的方案称为*专家共置*（expert collocation）。其思路是把某些专家共置在同一块 GPU 或同一节点上，从而避免不必要的通信。设想有两个专家——专家 5 和专家 7——它们经常被 token 路由器为同一个 token 同时激活。把专家 5 和专家 7 放在同一块 GPU 上，就能省去一次额外的 all-to-all 跳转。剖析工具与门控频率分析有助于为你的工作负载识别出这类配对。

And yet another solution is to compress the communication between experts, including expert-exchange activations. For instance, you can cast to FP8 or NVFP4 on Tensor Cores with the NVIDIA Transformer Engine before performing an all-to-all communication. This will reduce NIC load which amortizes the compression computation. This trades a tiny bit of numerical precision for faster activation transfers between GPUs. The overhead to cast and pack/unpack is usually small relative to network and memory transfer costs.

还有一种方案是压缩专家之间的通信，包括专家交换的激活值。例如，你可以在执行 all-to-all 通信之前，先用 NVIDIA Transformer Engine 在 Tensor Core 上把数据转换为 FP8 或 NVFP4。这会降低 NIC 负载，从而把压缩计算的开销摊薄掉。这样做以极小的数值精度损失换取 GPU 之间更快的激活值传输。相对于网络与内存传输成本，转换以及打包/解包的开销通常很小。

In short, optimizing MoE communication involves analyzing the hardware and network topology to optimize the placement of experts onto GPUs. For your deployments, it’s important to configure the cluster’s interconnects for efficient all-to-all communication. For instance, use the NVLink Switch mesh in an NVL72 rack to get the full bandwidth communication between up to 72 GPUs in a single domain. Naive all-to-all choices can dominate layer time and achieve very low SM efficiency. Prioritize expert traffic and overlap where possible, and make sure to profile and verify when making different interconnect and communication-algorithm choices.

简而言之，优化 MoE 通信需要分析硬件与网络拓扑，以优化专家在 GPU 上的放置。对于你的部署而言，为集群互连配置高效的 all-to-all 通信非常重要。例如，在 NVL72 机架中使用 NVLink Switch 网状拓扑，可在单个域内的最多 72 块 GPU 之间获得满带宽通信。粗糙的 all-to-all 选择可能主导整层耗时，并导致极低的 SM 效率。应尽量优先处理专家流量并做重叠，并且在做出不同的互连与通信算法选择时，务必进行剖析与验证。

> For MoEs, configure your cluster to optimize for all-to-all communication. This means selecting the appropriate NCCL all-to-all algorithm or grouped send and receive implementation for your topology, then confirming GPUDirect RDMA is enabled for the internode paths. Also, make sure your InfiniBand links are properly bonded such that multiple physical links (ports) are configured as a single logical channel. Their bandwidth should be combined—and failover should be seamless. In other words, make sure your network topology is tuned for MoEs. This includes both the hardware and software layers in the stack.

> 对于 MoE，应把集群配置为针对 all-to-all 通信做优化。这意味着为你的拓扑选择合适的 NCCL all-to-all 算法或分组收发（grouped send and receive）实现，然后确认节点间路径已启用 GPUDirect RDMA。此外，确保你的 InfiniBand 链路已正确绑定，使多条物理链路（端口）配置为单个逻辑通道。它们的带宽应被合并——并且故障切换应无缝进行。换句话说，确保你的网络拓扑针对 MoE 做过调优。这既包括栈中的硬件层，也包括软件层。

### Load Balancing, Capacity Factor, and Expert Replication

### 负载均衡、容量因子与专家复制

It’s recommended that each expert and GPU get an equal share of the work to avoid “hotspots.” Otherwise, if the MoE’s gating network directs a disproportionate number of tokens to a particular expert, those GPUs will become overloaded. This will increase latency and reduce overall system throughput.

建议让每个专家和每块 GPU 获得均等份额的工作，以避免“热点”。否则，如果 MoE 的门控网络把不成比例的 token 数量导向某个特定专家，那些 GPU 就会过载。这会增加延迟，并降低系统整体吞吐量。

Consider a single expert GPU that becomes a hotspot with utilization hitting 99% while the other expert GPUs are averaging around 60% utilization. This can bottleneck an entire training or inference cluster if not properly handled.

设想某块专家 GPU 成为热点，利用率飙升到 99%，而其他专家 GPU 平均利用率约为 60%。如果处理不当，这会成为整个训练或推理集群的瓶颈。

During model training, hotspots can be addressed by adding a load-balancing loss term that penalizes the gate if it overuses some experts and underuses others. The result is that the trained MoE model tends to distribute tokens fairly evenly across the experts.

在模型训练期间，可以通过添加一个负载均衡损失项来应对热点，该损失项会在门控过度使用某些专家、而对另一些专家使用不足时对其施加惩罚。其结果是，训练出的 MoE 模型倾向于把 token 相当均匀地分布到各个专家上。

At inference time, however, specific input prompts or topics might still cause imbalance by concentrating on a subset of “hot” experts. One strategy to avoid inference hot spots is to use a *capacity factor* that triggers an overflow mechanism.

然而在推理时，特定的输入提示词或主题仍可能因集中于一小部分“热点”专家而造成不均衡。避免推理热点的一种策略是使用会触发溢出机制的*容量因子*。

By specifying a capacity factor, the model can be configured such that each expert can process only a maximum number of tokens (e.g., 32 tokens) at a given time. If an expert receives more than this capacity of tokens, the extra tokens can either be forwarded to a fallback expert with the next highest routing score or the tokens will be serialized and processed in a second pass. Figure 15-12 compares a capacity factor of 1.0 versus 1.5.

通过指定容量因子，可以把模型配置为每个专家在给定时刻最多只能处理一定数量的 token（例如 32 个 token）。如果某个专家收到的 token 超过该容量，多出的 token 既可以被转发到路由分数次高的后备专家，也可以被序列化并在第二遍中处理。图 15-12 对比了容量因子为 1.0 与 1.5 的情形。

![Figure 15-12. Comparing expert capacity factors of 1.0 versus 1.5](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-12.png)

![图 15-12. 对比专家容量因子 1.0 与 1.5](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-12.png)

In practice, a capacity factor of 1.2 (20% overflow allowance) with top-2 gating is common. This means that each expert will take up to 120% of its average load. After that, it will send excess tokens to the next expert. This will effectively smooth out the load across experts in the system.

在实践中，容量因子取 1.2（20% 的溢出余量）并配合 top-2 门控较为常见。这意味着每个专家最多承担其平均负载的 120%。超出之后，它会把多余的 token 发送给下一个专家。这能有效地把系统中各专家之间的负载抹平。

Another strategy to avoid hotspots is expert replication. If one expert is consistently a hotspot, the system can clone that expert onto another GPU. This way, the gating function can send some fraction of tokens to the expert clone running on the other GPU.

避免热点的另一种策略是专家复制（expert replication）。如果某个专家持续成为热点，系统可以把该专家克隆到另一块 GPU 上。这样，门控函数就可以把一部分 token 发送给运行在另一块 GPU 上的专家副本。

The replicas are a pure application-level optimization implemented by the inference engine. The model itself is not aware of the replicas. Because replicas are registered as separate experts with their own indices—and often on different GPUs—the engine can route tokens across both the original experts and their clones according to their relative routing scores.

这些副本是一种纯应用层优化，由推理引擎实现。模型本身并不知道副本的存在。由于副本被注册为拥有各自索引的独立专家——而且往往位于不同的 GPU 上——引擎便可以根据原始专家及其克隆体的相对路由分数，在两者之间路由 token。

> Replicating experts will increase the memory—and cost—of each replicated expert. But since only a few experts tend to be overloaded, replicating a small number of hot experts is a targeted fix—and won’t double the cost by replicating a full model and all of its experts.

> 复制专家会增加每个被复制专家的内存占用——以及成本。但由于往往只有少数几个专家会过载，只复制少量热点专家是一种有针对性的修复手段——它不会因为复制整个模型及其全部专家而使成本翻倍。

Also, replication requires careful handling to keep the replicas synchronized if the model is updated. It’s important that all replica experts remain identical (e.g., same weights) since the gating router knows about only a single expert and does not actually know about the replicas.

此外，如果模型有更新，复制需要小心处理以保持各副本同步。让所有副本专家保持完全一致（例如权重相同）非常重要，因为门控路由器只知道单个专家，而实际上并不知道副本的存在。

> Typically, replicas are loaded from the same checkpoint as the original model—and not updated independently. This prevents divergence between the original expert and its replica.

> 通常，副本从与原始模型相同的检查点（checkpoint）加载——而不会被独立更新。这可以防止原始专家与其副本之间产生分歧。

### Adaptive Expert Routing and Real-Time Monitoring

### 自适应专家路由与实时监控

Unlike traditional MoE expert gating, which is fixed after training, adaptive routing can adjust the gate’s decisions in real time during inference to react to current conditions and expert load. For instance, if the system detects that one expert GPU is lagging behind, it could instruct the gating function to divert some tokens to another expert. The other expert might have a slightly lower routing score, but it receives the request because it has available capacity.

传统的 MoE 专家门控在训练之后就固定不变，而自适应专家路由（adaptive expert routing）则不同：它可以在推理期间实时调整门控的决策，以响应当前状况和专家负载。例如，如果系统检测到某块专家 GPU 落后了，它可以指示门控函数把一部分 token 转移到另一个专家上。另一个专家的路由分数可能略低，但因为它有可用容量，所以会接到这些请求。

You should implement continuous monitoring of per-expert utilization and response latency metrics. Modern MoE systems integrate with telemetry frameworks such that each expert emits utilization metrics to Prometheus/Grafana. This way, the system can dynamically adjust the capacity factor or gating algorithm on the fly.

你应当对每个专家的利用率与响应延迟指标实施持续监控。现代 MoE 系统会与遥测框架集成，使每个专家把利用率指标上报到 Prometheus/Grafana。这样，系统就能动态地即时调整容量因子或门控算法。

Most LLM expert gating functions only consider routing scores determined at training time. However, a truly adaptive system needs to be handled by the inference engine and performed dynamically at inference time.

大多数 LLM 专家门控函数只考虑在训练时确定的路由分数。然而，真正自适应的系统需要由推理引擎来处理，并在推理时动态执行。

To implement adaptive routing, the inference engine needs to wrap the model’s forward pass in custom logic. For example, it can intercept the gating softmax and reallocate some tokens to different experts based on current load metrics.

要实现自适应路由，推理引擎需要用自定义逻辑包裹模型的前向传播。例如，它可以拦截门控的 softmax，并根据当前的负载指标把一部分 token 重新分配给不同的专家。

Inference engines rely on real-time metrics like per-GPU utilization and per-expert token counts to continuously measure expert load. If the system sees one expert’s GPU at 99% utilization while other experts’ GPUs are at 60%, the system could temporarily lower its load by routing some tokens to its expert replica—or to a different expert with a slightly lower expert-preference score.

推理引擎依赖诸如每 GPU 利用率和每专家 token 计数之类的实时指标来持续测量专家负载。如果系统看到某个专家的 GPU 处于 99% 利用率、而其他专家的 GPU 处于 60%，系统就可以通过把一部分 token 路由到它的专家副本——或路由到另一个专家偏好分数略低的专家——来暂时降低其负载。

Figure 15-13 shows an adaptive MoE routing strategy that uses a biased gating score approach. While this approach was originally used in a training context, a simpler approach applies to inference. In this case, it would use a modified expert-bias algorithm to divert tokens to alternate experts when the primary experts are heavily loaded.

图 15-13 展示了一种采用偏置门控分数（biased gating score）方法的自适应 MoE 路由策略。虽然这一方法最初用于训练场景，但推理场景可以采用一种更简单的方式。在这种情况下，它会使用一种改良的专家偏置算法，在主专家负载沉重时把 token 转移到备用专家上。

![Figure 15-13. Adaptive MoE routing in action](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-13.png)

![图 15-13. 运行中的自适应 MoE 路由](AI%20Systems%20Performance%20Engineering-ch15_images/figure-15-13.png)

This approach can reduce unnecessary communication and balance the load more uniformly. However, it does incur some additional cost in the form of extra monitoring, decision making, complexity, configuration management, and logging. The benefits may or may not outweigh the cost. Every scenario and workload is unique, but it’s definitely something worth exploring.

这种方法可以减少不必要的通信，并把负载更均匀地平衡开来。不过，它确实会带来一些额外成本，表现为额外的监控、决策、复杂度、配置管理和日志记录。这些收益未必总能盖过成本。每种场景和工作负载都各不相同，但这绝对值得探索。

When profiling with tools like Nsight Systems, you want to monitor the timeline traces of the expert GPUs’ all-to-all communications. If one GPU’s segment in the timeline is much longer, for instance, it is likely processing more tokens.

当使用 Nsight Systems 等工具进行剖析时，你会希望监控各专家 GPU 的 all-to-all 通信的时间线轨迹。例如，如果某块 GPU 在时间线中的片段明显更长，那么它很可能正在处理更多的 token。

Your inference system can use these insights to adjust the expert gating probabilities and dynamically reassign experts to different GPUs. It can also spawn additional expert replica instances, etc. This helps to rebalance the load by adjusting the expert gating algorithm, modifying expert placement, or creating/removing expert replicas.

你的推理系统可以利用这些洞见来调整专家门控概率，并动态地把专家重新分配到不同的 GPU 上。它还可以派生额外的专家副本实例，等等。这有助于通过调整专家门控算法、修改专家放置或创建/移除专家副本来重新平衡负载。

> Dynamically spawning new expert replicas at inference time is nontrivial. This approach requires pre-provisioned capacity or rapid model loading for those experts. This is an advanced optimization technique.”

> 在推理时动态派生新的专家副本并不简单。这种方法需要预先预留的容量，或为这些专家提供快速的模型加载。这是一种高级优化技术。”

Grouping certain experts differently across GPUs can lead to more uniform token routing and increased overall throughput due to better parallelism. This is because the GPUs can finish each layer’s work more synchronously. If persistent imbalance is detected, modern MoE schedulers can dispatch additional expert replicas on the fly by adjusting the capacity factor. This can mitigate a hotspot caused by uneven gating.

把某些专家在各 GPU 上以不同方式分组，可以带来更均匀的 token 路由，并因更好的并行性而提升整体吞吐量。这是因为 GPU 可以更同步地完成每一层的工作。如果检测到持续的不均衡，现代 MoE 调度器可以通过调整容量因子即时派发额外的专家副本。这能缓解由门控不均衡引起的热点。

> Remember to log any dynamic changes that the system makes. It’s also recommended to set up alerts for when a particular expert’s utilization goes above a threshold, such as 80% utilization.

> 记得记录系统所做的任何动态更改。此外，建议为某个专家利用率超过阈值（例如 80% 利用率）的情形设置告警。

Dynamic routing strategies target two core objectives: reduce routing overhead and evenly distribute work across expert GPUs. Achieving low overhead depends on utilizing high-bandwidth interconnects, overlapping data transfers with computation, and minimizing redundant data movement with co-location and intelligent scheduling.

动态路由策略瞄准两个核心目标：降低路由开销，以及在各专家 GPU 之间均匀分配工作。实现低开销取决于利用高带宽互连、把数据传输与计算重叠，以及通过共置和智能调度来最小化冗余的数据搬移。

Load balancing is achieved using simple top-1 or top-2 gating or more advanced capacity-aware gates. It can also be achieved using dynamic replication and expert reassignment. Using a combination of these techniques is common to keep the GPUs busy with maximum computations and minimal communication delays.

负载均衡可以通过简单的 top-1 或 top-2 门控、或更高级的容量感知门控来实现。也可以通过动态复制和专家重分配来实现。组合使用这些技术很常见，以最大限度的计算和最小的通信延迟让 GPU 保持繁忙。

As long as routing overhead is minimized and the load is balanced, adaptive MoE inference systems can achieve near-linear throughput scaling as you increase the number of experts. For example, doubling GPUs can nearly double your inference throughput.

只要路由开销被最小化且负载得到均衡，自适应 MoE 推理系统就能在你增加专家数量时实现近乎线性的吞吐量扩展。例如，把 GPU 数量翻倍几乎就能让你的推理吞吐量翻倍。

In short, these adaptive expert routing and load-balancing optimizations can be integrated with the parallelism and decoding techniques covered earlier. This way, you can tune your MoE inference system for high performance at ultrascale. Continuous profiling and adaptive algorithms can keep your GPUs busy with computations and avoid idling on communication delays.

简而言之，这些自适应专家路由与负载均衡优化，可以与前文介绍的并行和解码技术集成在一起。这样，你就能为超大规模下的高性能调优你的 MoE 推理系统。持续的剖析与自适应算法能让你的 GPU 忙于计算，避免因通信延迟而空转。

> Advanced inference engines can dynamically bypass certain expert computations, turn off underutilized experts, or use caching to skip experts for certain tokens. This further reduces latency. These techniques build on the concepts described in this chapter.

> 先进的推理引擎可以动态绕过某些专家计算、关闭利用不足的专家，或利用缓存对某些 token 跳过专家。这能进一步降低延迟。这些技术建立在本章所述概念之上。

## Key Takeaways

## 关键要点

Serving massive LLMs to billions of end users requires optimizations at every stage of the inference pipeline. The following are some key takeaways from this chapter:

向数十亿终端用户提供超大 LLM 服务，需要在推理流水线的每个阶段进行优化。以下是本章的一些关键要点：

*Disaggregate to optimize both latency and throughput* Splitting the prompt prefill and decode stages onto separate GPU pools eliminates interference. This lets you achieve low time-to-first-token and high tokens-per-second simultaneously, instead of trading one for the other. It’s a foundational technique for large-scale LLM serving.

*分离以同时优化延迟与吞吐* 把提示词 prefill 与 decode 两个阶段拆分到独立的 GPU 池上，可以消除相互干扰。这让你能够同时实现低首 token 时延和高每秒 token 数，而不必以牺牲一者来换取另一者。这是大规模 LLM 服务的一项基础性技术。

*Use hybrid parallelism for massive models* No single parallelism strategy is sufficient for multi-trillion-parameter models. Combine tensor, pipeline, expert, and data parallelism as needed. For example, shard layers across GPUs (TP/PP) to fit the model in memory, use expert parallelism for MoE layers to scale model capacity, and add data-parallel replicas to meet throughput demands. The optimal mix is hardware-dependent. Always profile and tune your configuration for your workload.

*为超大模型使用混合并行* 对于数万亿参数的模型，任何单一的并行策略都不够用。应按需组合张量、流水线、专家和数据并行。例如，跨 GPU 分片各层（TP/PP）以把模型装入内存，对 MoE 层使用专家并行以扩展模型容量，并添加数据并行副本以满足吞吐需求。最优组合取决于硬件。务必针对你的工作负载对配置进行剖析和调优。

*Mitigate sequential decoding bottlenecks* Advanced decoding methods can greatly accelerate generation. Two-model speculative decoding with a fast draft model often delivers about a 2–3× speedup when acceptance rates are tuned for the task. EAGLE-2 reports up to 3.5× speedup (20%–40% more than EAGLE-1) on some tasks while preserving the target distribution. Medusa implementations report up to 3.6× speedup over nonspeculative decoding when trained and validated for the target workload. These techniques increase token-level parallelism and arithmetic intensity while preserving the large model’s output distribution under standard verification. Overall, the result is faster responses without retraining the main model’s output. This shows that speculative decoding is a quality-preserving technique.

*缓解顺序解码瓶颈* 先进的解码方法可以极大地加速生成。当接受率针对任务调优后，配合快速草稿模型的双模型投机解码通常能带来约 2–3× 的加速。EAGLE-2 报告在某些任务上可达最高 3.5× 加速（比 EAGLE-1 多 20%–40%），同时保持目标分布。Medusa 的实现在针对目标工作负载训练和验证后，报告相较非投机解码可达最高 3.6× 加速。这些技术在提升 token 级并行性和算术强度的同时，在标准验证下保持大模型的输出分布。总的来说，其结果是在不重新训练主模型输出的前提下获得更快的响应。这表明投机解码是一种保质技术。

*Maintain output quality and format with constraints* In production, you often need the LLM to follow strict formats or avoid certain tokens. Constrained-decoding techniques let you enforce rules—like JSON schemas and banned words—during generation. They add some overhead per token, but with compiled grammars and optimized mask paths, constrained decoding can often run within a low single-digit percent of normal decoding at scale, though complex grammars or small batches may incur higher overhead. Always test the performance impact of your constraints. Avoid extremely strict rules, if possible. They can slow down generation by causing excessive backtracking.

*用约束维持输出质量与格式* 在生产环境中，你常常需要让 LLM 遵循严格的格式或避免某些 token。受约束解码技术让你能在生成过程中强制执行规则——比如 JSON 模式和禁用词。它们会给每个 token 带来一些开销，但借助编译后的语法和优化的掩码路径，受约束解码在大规模下往往能运行在正常解码的低个位数百分比开销以内，尽管复杂语法或小批量可能带来更高的开销。务必测试你的约束所带来的性能影响。如有可能，避免使用极其严格的规则。它们会因导致过多回溯而拖慢生成。

*Balance MoE workloads to scale effectively* Mixture-of-experts models offer almost linear scaling of model size versus GPU/expert count—but only if you handle routing efficiently. Use high-bandwidth interconnects and hierarchical all-to-all communication to reduce network bottlenecks. Ensure each expert gets a similar amount of work by applying capacity limits and top-2 gating to avoid straggler experts. Replicate any consistently hot experts to split their load. A well-tuned MoE inference system can approach near-linear throughput scaling as you add GPUs.

*均衡 MoE 工作负载以实现有效扩展* 专家混合模型在模型规模相对于 GPU/专家数量上提供了几乎线性的扩展——但前提是你要高效地处理路由。使用高带宽互连和分层的 all-to-all 通信来减少网络瓶颈。通过施加容量上限和 top-2 门控来确保每个专家获得相近的工作量，以避免掉队专家。对任何持续过热的专家进行复制，以分摊其负载。一个调优良好的 MoE 推理系统，可以在你增加 GPU 时接近线性的吞吐量扩展。

*Leverage hardware-software codesign* Modern GPU hardware is built to support these parallel and distributed inference methods. Use software that takes full advantage of the hardware and topology, including inference engines like vLLM, SGLang, and NVIDIA Dynamo. These can orchestrate multi-GPU and multinode inference with minimal overhead. Align your strategy with your hardware’s strengths by keeping intranode communication on NVSwitch, using InfiniBand only when necessary, and overlapping communications with computations. This alignment is key to achieving the best latency and cost-efficiency.

*利用软硬件协同设计* 现代 GPU 硬件正是为支持这些并行与分布式推理方法而构建的。使用能充分发挥硬件与拓扑优势的软件，包括 vLLM、SGLang 和 NVIDIA Dynamo 等推理引擎。它们能以最小的开销编排多 GPU 和多节点推理。让节点内通信保持在 NVSwitch 上、仅在必要时才使用 InfiniBand，并把通信与计算重叠，从而使你的策略与硬件的强项对齐。这种对齐是实现最佳延迟和成本效率的关键。

*Understand complexity versus return on investment (ROI)* Each optimization adds system complexity. Techniques like speculative decoding and adaptive MoE routing can significantly improve performance, but they require extra models or intricate logic. Always weigh the cost. For interactive applications, a 2–3× latency improvement is usually worth it. For simpler use cases, a straightforward approach might suffice. Start with the biggest bottlenecks, such as eliminating prefill/decode interference. Then incrementally add complexity if needed. Monitor and profile to determine which optimizations give the best return on investment.

*理解复杂度与投资回报率（ROI）之间的权衡* 每一项优化都会增加系统复杂度。投机解码和自适应 MoE 路由等技术可以显著提升性能，但它们需要额外的模型或错综复杂的逻辑。务必权衡代价。对于交互式应用，2–3× 的延迟提升通常是值得的。对于更简单的用例，一种直截了当的方法可能就够用了。从最大的瓶颈入手，比如消除 prefill/decode 干扰。然后在需要时逐步增加复杂度。通过监控和剖析来确定哪些优化能带来最佳的投资回报率。

## Conclusion

## 结论

By bringing together disaggregated prefill/decode pipelines, multitoken speculative decoding, dynamic expert routing, and adaptive orchestration, it’s possible to serve LLMs in real time with minimal resource contention and ultra-low latency.

通过把分离式 prefill/decode 流水线、多 token 投机解码、动态专家路由和自适应编排结合在一起，就有可能以最小的资源争用和超低延迟实时地为 LLM 提供服务。

Modern inference-serving platforms like vLLM, SGLang, and NVIDIA Dynamo embrace many of these optimizations. They efficiently allocate cluster resources, coordinate the KV cache across nodes, implement speculative and constrained decoding, schedule prefill/decode tasks, and much more.

vLLM、SGLang 和 NVIDIA Dynamo 等现代推理服务平台采纳了其中许多优化。它们能高效地分配集群资源、跨节点协调 KV 缓存、实现投机解码和受约束解码、调度 prefill/decode 任务，等等。

The key is an end-to-end, adaptable architecture that matches algorithmic innovations with high-performance hardware capabilities. Over the next few chapters, we’ll dive deeper into model inference performance optimizations, including dynamic, adaptive, and multinode serving strategies.

关键在于一个端到端、可适配的架构，它把算法创新与高性能硬件能力相匹配。在接下来的几章中，我们将更深入地探讨模型推理性能优化，包括动态、自适应和多节点的服务策略。

We’ll cover everything from application-level prefix caching, latency-aware request routing, and reinforcement-learning (RL)-based cluster tuning to systems-level adaptive memory allocation, precision switching, and congestion-aware resource scheduling.

我们将涵盖从应用层前缀缓存、延迟感知的请求路由、基于强化学习（reinforcement learning，RL）的集群调优，到系统层的自适应内存分配、精度切换和拥塞感知的资源调度等全部内容。
