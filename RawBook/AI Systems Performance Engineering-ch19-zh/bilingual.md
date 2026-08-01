# Chapter 19. Dynamic and Adaptive Inference Engine Optimizations

# 第 19 章 动态与自适应推理引擎优化

Ultralarge language model (LLM) inference on modern hardware requires dynamic runtime adaptation to achieve both high throughput and low latency under varying conditions. A static “one-size-fits-all” approach to model-serving optimizations is no longer sufficient.

在现代硬件上进行超大规模语言模型（large language model，LLM）推理，需要在运行时进行动态适配，才能在多变的条件下同时实现高吞吐量与低延迟。对模型服务优化采用静态的“一刀切”（one-size-fits-all）方案已不再够用。

Instead, state-of-the-art model serving systems use *adaptive strategies* that adjust parallelism, numerical precision, CUDA-kernel scheduling, and memory usage on the fly. This chapter explores these advanced techniques, including dynamic parallelism switching, precision scaling, real-time cache management, and reinforcement learning (RL)-based tuning.

取而代之，最先进的模型服务系统采用*自适应策略*，在运行中动态调整并行、数值精度、CUDA 核函数调度与内存使用。本章将探讨这些高级技术，包括动态并行切换、精度伸缩、实时缓存管理，以及基于强化学习（reinforcement learning，RL）的调优。

This chapter provides best practices for ultrascale LLM inference, teaching you how to orchestrate an engine that monitors its own performance and adapts in real time to maximize efficiency.

本章给出超大规模 LLM 推理的最佳实践，教你如何编排一个能够自我监测性能、并实时自适应以最大化效率的引擎。

## Adaptive Parallelism Strategies (TP Versus PP Versus Hybrid)

## 自适应并行策略（TP、PP 与混合）

Massive LLMs require model parallelism, such as tensor and pipeline—or a hybrid approach—to spread computation across multiple GPUs. Each approach has benefits and drawbacks. Table 19-1 summarizes the recommended parallelism strategies for specific inference traffic patterns.

超大规模 LLM 需要模型并行——例如张量并行与流水线并行，或二者的混合方案——才能将计算分散到多块 GPU 上。每种方案各有利弊。表 19-1 总结了针对特定推理流量模式所推荐的并行策略。

Table 19-1. Summary of common inference traffic patterns mapped to the recommended parallelism strategy

表 19-1. 常见推理流量模式与推荐并行策略的对应汇总

| Inference traffic pattern | Recommended parallelism | Rationale |
| --- | --- | --- |
| Many short requests (< 256 tokens, high RPS) | Data parallel/replica scaling | Minimizes inter-GPU communications; each GPU runs replicas handling independent requests (assuming the model fits into a single GPU's memory) |
| Few long requests (≥ 8k tokens, low concurrency) | Pipeline parallelism (with microbatches) | Reduces per-request latency by splitting layers across GPUs |
| Mixed load (short + some long) | Hybrid dynamic (autoswitching) | Runs small chats on single GPUs and pipelines long ones to meet latency SLAs |
| Extremely large model (> 1 GPU memory) | Tensor + pipeline hybrid | Required to fit the model; balances compute and memory across both dimensions |
| MoE model inference (sparse expert selection) | Expert parallelism | Distributes individual experts across GPUs; each request invokes only a subset of experts, reducing per-device memory and compute load |

| 推理流量模式 | 推荐并行 | 依据 |
| --- | --- | --- |
| 大量短请求（< 256 tokens，高 RPS） | 数据并行/副本扩展 | 最小化 GPU 间通信；每块 GPU 运行副本、各自处理相互独立的请求（前提是模型能装入单块 GPU 的内存） |
| 少量长请求（≥ 8k tokens，低并发） | 流水线并行（配合微批） | 通过将层拆分到多块 GPU 上以降低单请求延迟 |
| 混合负载（短请求 + 部分长请求） | 混合动态（自动切换） | 在单块 GPU 上运行小型对话，对长请求走流水线，以满足延迟 SLA |
| 超大模型（> 1 块 GPU 内存） | 张量 + 流水线混合 | 为装下模型所必需；在两个维度上平衡计算与内存 |
| MoE 模型推理（稀疏专家选择） | 专家并行 | 将各个专家分布到多块 GPU 上；每个请求只调用一部分专家，从而降低单设备的内存与计算负载 |

The data parallel and replica scaling strategy will replicate the full model on each GPU and load-balance incoming requests across these replicas. This requires no inter-GPU synchronization for individual inferences since each GPU handles separate requests independently.

数据并行与副本扩展策略会在每块 GPU 上复制完整模型，并将进入的请求在这些副本之间做负载均衡。由于每块 GPU 各自独立处理不同的请求，单次推理无需 GPU 间同步。

This maximizes throughput for many small- or medium-sized inputs with minimal communication overhead. However, data parallelism is not an option if the model does not fit into a single GPU’s memory.

对于大量中小规模输入，这能以极小的通信开销最大化吞吐量。然而，如果模型无法装入单块 GPU 的内存，数据并行便不可行。

Tensor parallelism (TP) is a form of model parallelism (as opposed to data parallelism) that splits model matrices (e.g., weights, layers, etc.) across GPUs to speed up matrix multiplies. However, it introduces extra all-reduce communications to keep the GPUs in sync.

张量并行（tensor parallelism，TP）是一种模型并行形式（与数据并行相对），它将模型矩阵（例如权重、层等）拆分到多块 GPU 上以加速矩阵乘法。不过，它会引入额外的 all-reduce 通信来保持各 GPU 的同步。

Pipeline parallelism (PP) is another form of model parallelism that splits the model as well. But instead of splitting individual model layers and matrices, it assigns whole layers to different GPUs to overcome memory limits—assuming the layers fit into a single GPU. PP incurs additional overhead in the form of sequential stage delays. These are called *pipeline bubbles*, as shown in Figure 19-1.

流水线并行（pipeline parallelism，PP）是另一种同样对模型进行拆分的模型并行形式。但它不拆分单个模型层与矩阵，而是把整层分配给不同的 GPU，以突破内存限制——前提是单层能装入一块 GPU。PP 会带来以顺序阶段延迟为形式的额外开销。这些延迟被称为*流水线气泡*（pipeline bubble），如图 19-1 所示。

Expert parallelism, used in mixture-of-experts (MoE) model architectures, assigns each expert subnetwork its own GPU. A lightweight gating network then directs each input request or token to only the top-k active experts identified by the router. In this case, each GPU processes just the subset of experts that it hosts.

专家并行（expert parallelism）用于 MoE（专家混合，mixture-of-experts）模型架构，为每个专家子网络分配各自的 GPU。随后，一个轻量的门控网络（gating network）将每个输入请求或 token 只导向由路由器（router）识别出的 top-k 个激活专家。此时，每块 GPU 只处理它所承载的那一部分专家。

![Figure 19-1. Pipeline bubbles caused by PP](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-1.png)

![图 19-1. PP 造成的流水线气泡](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-1.png)

By activating only a few experts per input, expert parallelism reduces per-device memory, inference time, and compute costs for models with a large number of experts, often called *wide* expert models. The conditional, router-based expert compute pattern scales efficiently as you add more experts. For instance, DeepSeek-R1 has 256 total experts, but only the top 9 experts (including 1 shared expert) are chosen by the router during inference.

由于每个输入只激活少数几个专家，对于拥有大量专家、常被称为*宽*（wide）专家模型的模型，专家并行可降低单设备内存、推理时间与计算成本。这种基于路由器的条件式专家计算模式，会随着专家数量的增加而高效扩展。例如，DeepSeek-R1 共有 256 个专家，但在推理时路由器只选择 top 9 个专家（其中包含 1 个共享专家）。

Traditionally, the parallelization strategy—including a hybrid strategy of multiple parallelism techniques combined—is chosen and fixed upfront when the model is loaded. However, to maximize performance under dynamic workloads, modern inference engines can choose different parallelism strategies at runtime based on the characteristics of the input.

传统上，并行化策略——包括将多种并行技术组合而成的混合策略——是在加载模型时预先选定并固定下来的。然而，为在动态负载下最大化性能，现代推理引擎可以在运行时根据输入特征选择不同的并行策略。

High-performance, adaptive inference systems use runtime metrics to choose TP, PP, or a hybrid approach on the fly. Key factors include batch size, sequence length, and memory utilization—as well as response latency and throughput requirements. For instance, very long prompts may be routed to a TP + PP instance since this spreads layers across GPUs to avoid out-of-memory (OOM) errors.

高性能的自适应推理系统利用运行时指标来动态选择 TP、PP 或混合方案。关键因素包括 batch size、序列长度与内存利用率——以及响应延迟与吞吐量要求。例如，超长提示可能被路由到 TP + PP 实例，因为这会把层分散到多块 GPU 上，以避免内存不足（out-of-memory，OOM）错误。

Meanwhile, short latency-sensitive requests would route to a TP-only model instance to avoid pipeline-stage overhead. To support this, your serving engine maintains multiple presharded model instances, each optimized for different workload profiles, and dynamically dispatches incoming queries to the model instance whose parallelism strategy best satisfies the job’s SLOs.

与此同时，对延迟敏感的短请求会被路由到仅使用 TP 的模型实例，以避免流水线阶段开销。为支持这一点，你的服务引擎会维护多个预分片（presharded）的模型实例，每个都针对不同的负载画像做了优化，并将进入的查询动态分派给其并行策略最能满足该作业 SLO 的模型实例。

You can also use a different number of shards. This is shown in Figure 19-2, which uses two different numbers of TP shards in two different hybrid TP + PP parallelism configurations across eight GPUs.

你也可以使用不同的分片数量。如图 19-2 所示，它在跨八块 GPU 的两种不同的混合 TP + PP 并行配置中使用了两种不同数量的 TP 分片。

![Figure 19-2. Preprovisioning two different hybrid-sharding pools (TP = 4, PP = 1 and TP = 2, PP = 1) for a given model across eight GPUs](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-2.png)

![图 19-2. 为某个模型在八块 GPU 上预置两种不同的混合分片池（TP = 4, PP = 1 与 TP = 2, PP = 1）](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-2.png)

Using PP on the fly for long sequence inputs helps to avoid OOM errors caused by the large input sequence. Conversely, for short prompts and latency-sensitive queries, the system can instead route to a tensor-parallel model instance optimized for low latency. In this case, the request avoids the overhead of PP.

对长序列输入动态使用 PP，有助于避免由大型输入序列引发的 OOM 错误。反之，对于短提示与延迟敏感的查询，系统可以改为路由到为低延迟优化的张量并行模型实例。此时，请求便可避开 PP 的开销。

Since each request can use a different parallelism strategy, the system needs to maintain multiple instances of the model for the inference scheduler/router to choose. One instance of the model would be optimized for low latency using TP, while another instance is optimized for high throughput and large input sequences using both TP and PP.

由于每个请求可以使用不同的并行策略，系统需要维护该模型的多个实例，供推理调度器/路由器选择。其中一个模型实例使用 TP 针对低延迟做优化，另一个实例则同时使用 TP 与 PP 针对高吞吐量与大型输入序列做优化。

Maintaining multiple instances of the model is required because resharding on the fly would wreak havoc on GPU caches. This would also put too much pressure on the memory and network subsystems—especially when resharding massive models.

之所以必须维护模型的多个实例，是因为在运行中动态重新分片（resharding）会严重扰乱 GPU 缓存。它还会给内存与网络子系统带来过大压力——尤其是在对超大模型重新分片时。

At runtime, each query is dispatched to the best-fitting model instance (sharding strategy) based on its length and the specified service-level agreements (SLAs). DeepSeek-R1, for instance, is a ~680 billion-parameter sparse mixture-of-experts model that activates only 37 billion parameters per token across the experts.

在运行时，每个查询会根据其长度与指定的服务级别协议（service-level agreement，SLA）被分派到最契合的模型实例（分片策略）。以 DeepSeek-R1 为例，它是一个约 6800 亿参数的稀疏专家混合模型，在各专家之间每个 token 只激活 370 亿参数。

To support different workload profiles, GPUs can be organized into logical worker pools, each presharded with a specific parallelism strategy—either tensor-parallel or hybrid tensor + pipeline parallel, for instance.

为支持不同的负载画像，可以将 GPU 组织成逻辑 worker 池，每个池按特定并行策略预先分片——例如张量并行，或张量 + 流水线混合并行。

Let’s consider an example. If we have an 8× GPU Blackwell B200 server totaling 1,440 GB of HBM memory (1,440 GB = 180 GB per GPU × 8 GPUs), we can serve DeepSeek-R1 with four-way TP across four GPUs—leaving the other four GPUs idle.

来看一个例子。假设我们有一台配备 8× GPU 的 Blackwell B200 服务器，HBM 内存共计 1,440 GB（1,440 GB = 每块 GPU 180 GB × 8 块 GPU），我们可以用跨四块 GPU 的四路 TP 来服务 DeepSeek-R1——让另外四块 GPU 处于空闲。

If a single query arrives with an extremely long context (e.g., > 1 million tokens), the scheduler can spawn a two-stage pipeline such that stage 1 spans GPUs 0–3 and stage 2 spans GPUs 4–7. This effectively doubles the available GPU memory per stage to ~720 GB (180 GB per GPU × 4 GPUs) of total HBM (720 GB usable HBM). This helps avoid OOM errors when processing large inputs.

如果有单个查询携带极长的上下文（例如 > 100 万 tokens）到来，调度器可以生成一个两阶段流水线，使阶段 1 覆盖 GPU 0–3、阶段 2 覆盖 GPU 4–7。这实际上把每个阶段可用的 GPU 内存翻倍到约 720 GB 的总 HBM（180 GB 每块 GPU × 4 块 GPU）（720 GB 可用 HBM）。这有助于在处理大型输入时避免 OOM 错误。

Conversely, when dozens of short, latency-sensitive prompts arrive concurrently, the system routes them to the tensor-parallel instance only. By avoiding pipeline bubbles, or idle periods that occur while filling and draining pipeline stages, this configuration delivers the lowest possible per-request latency across all available GPUs.

反之，当数十个短的、延迟敏感的提示并发到来时，系统只将它们路由到张量并行实例。通过避免流水线气泡（即在填充与排空流水线阶段时出现的空闲期），这种配置能在所有可用 GPU 上提供尽可能低的单请求延迟。

To implement dynamic parallelism switching, you can implement a decision function that inspects runtime metrics like input sequence length, GPU memory usage, and current load. You would use these metrics to select the best-sharded model instance for each request, as shown here:

要实现动态并行切换，你可以实现一个决策函数，检查诸如输入序列长度、GPU 内存使用与当前负载等运行时指标。你会用这些指标为每个请求选择分片最优的模型实例，如下所示：

```
def choose_worker_pool(seq_len, gpu_mem_util, concurrent_reqs):
    # For long contexts or high memory pressure,
    # use hybrid pipeline + tensor parallelism
    # (example thresholds shown here)
    if seq_len > 4096 or gpu_mem_util > 0.8:
        return "tp_pp_hybrid"
    # For many simultaneous small requests, stick with tensor parallelism
    if concurrent_reqs > 4:
        return "tensor_parallel"
    # Fallback to tensor-parallel for typical workloads
    return "tensor_parallel"
```

```
def choose_worker_pool(seq_len, gpu_mem_util, concurrent_reqs):
    # For long contexts or high memory pressure,
    # use hybrid pipeline + tensor parallelism
    # (example thresholds shown here)
    if seq_len > 4096 or gpu_mem_util > 0.8:
        return "tp_pp_hybrid"
    # For many simultaneous small requests, stick with tensor parallelism
    if concurrent_reqs > 4:
        return "tensor_parallel"
    # Fallback to tensor-parallel for typical workloads
    return "tensor_parallel"
```

You’d prelaunch multiple model replicas on your GPU cluster—some sharded for TP-only and others for TP + PP—and have a router send each query to the appropriate replica based on the inputs and the decision strategy. This approach ensures that large, memory-intensive jobs get full pipeline support, while short, latency-sensitive calls run on TP-only instances to avoid unnecessary pipeline overhead.

你会在 GPU 集群上预先启动多个模型副本——一些仅分片为 TP，另一些分片为 TP + PP——并让路由器根据输入与决策策略把每个查询发送到合适的副本。这种做法可确保大型、内存密集的作业获得完整的流水线支持，而短的、延迟敏感的调用则在仅 TP 的实例上运行，以避免不必要的流水线开销。

It’s recommended to use telemetry from the model and hardware to inform parallelism switching. You can monitor GPU memory utilization, compute utilization, and interconnect (e.g., NVLink/NVSwitch) traffic in real time to make the decision. If you notice idle GPUs because of long pipeline bubbles—and you have extra memory headroom—you can collapse your pipeline into fewer stages so each GPU does more work and stays busy. Conversely, if some stages are hitting memory limits or compute bottlenecks, you can expand into more pipeline stages—or raise your tensor-parallel degree. This will spread the computations and memory footprint across additional GPUs.

建议利用来自模型与硬件的遥测数据来指导并行切换。你可以实时监控 GPU 内存利用率、计算利用率以及互连（例如 NVLink/NVSwitch）流量来做出决策。如果你发现由于较长的流水线气泡导致 GPU 空闲——且你还有额外的内存余量——你可以把流水线收缩为更少的阶段，使每块 GPU 承担更多工作并保持繁忙。反之，如果某些阶段触及内存上限或计算瓶颈，你可以扩展为更多的流水线阶段——或提高张量并行度。这会把计算与内存占用分散到更多的 GPU 上。

The key is to adjust the balance of tensor and pipeline splits dynamically to keep every GPU well utilized. At the same time, you need to stay within memory constraints and hit latency targets. This is something a static, one-size-fits-all configuration cannot achieve.

关键在于动态调整张量拆分与流水线拆分之间的平衡，让每块 GPU 都保持良好利用。与此同时，你需要保持在内存约束之内并达成延迟目标。这是静态的一刀切配置所无法做到的。

## Dynamic Precision Changes

## 动态精度切换

Modern GPUs like Blackwell introduce support for 8-bit and 4-bit floating point (FP8/FP4) Tensor Core math units. These lower precisions offer large speedups, memory savings, and minimal quality loss.

像 Blackwell 这样的现代 GPU 引入了对 8 位与 4 位浮点（FP8/FP4）Tensor Core 计算单元的支持。这些更低的精度带来了巨大的加速、内存节省与极小的质量损失。

Dynamic precision switching is an advanced technique in which the inference engine adjusts the numerical precision at runtime based on model confidence or resource pressure. The goal is to increase throughput without significant quality loss. In practice, this means the system might execute certain parts of the model in FP8 or FP4 for efficiency but fall back to higher precision (FP16/BF16) when needed for stability.

动态精度切换是一种高级技术，推理引擎会在运行时根据模型置信度（confidence）或资源压力调整数值精度。其目标是在不明显损失质量的前提下提高吞吐量。在实践中，这意味着系统可能为了效率以 FP8 或 FP4 执行模型的某些部分，但在为稳定性所需时回退到更高精度（FP16/BF16）。

One trigger for precision adaptation is *logit sharpness*, or the model’s output confidence. For example, if the model’s probability distribution for the next token shows extreme peaks due to high confidence in a specific token, small numerical errors from low precision are unlikely to change the outcome.

触发精度自适应的一个因素是*logit 锐度*（logit sharpness），即模型输出的置信度。例如，如果模型对下一个 token 的概率分布因为对某个特定 token 高度自信而呈现出极端的尖峰，那么低精度带来的微小数值误差不太可能改变结果。

If low precision can be tolerated for the next token generation, the engine will safely use FP4 for the next few steps to gain speed. Conversely, if the distribution is flatter due to high uncertainty, the engine should stick to FP8 or FP16 to preserve fidelity.

如果下一个 token 的生成能够容忍低精度，引擎便会在接下来的几步中安全地使用 FP4 以获得速度提升。反之，如果由于高度不确定性使分布较为平坦，引擎则应坚持使用 FP8 或 FP16 以保持保真度。

The inference engine quantifies uncertainty by computing the Shannon entropy of the softmax distribution over the vocabulary. Lower entropy indicates a sharper (more confident) prediction. A fixed entropy threshold, tuned on a held-out validation set, determines when to drop to FP4 and when to remain in FP8/FP16 for numerical stability. The goal is to balance latency gains versus accuracy loss.

推理引擎通过计算词表上 softmax 分布的香农熵（Shannon entropy）来量化不确定性。熵越低，表示预测越尖锐（越自信）。一个在留出验证集上调好的固定熵阈值，决定何时降到 FP4、何时为数值稳定性而保持在 FP8/FP16。其目标是在延迟收益与精度损失之间取得平衡。

> Use the lowest precision that maintains accuracy, and revert to higher precision when the model’s confidence drops as measured by the maximum softmax probability.

> 使用能维持精度的最低精度，并在模型置信度（以最大 softmax 概率衡量）下降时回退到更高精度。

This leverages the fact that large LLMs often become more certain as they generate deterministic continuations, such as closing quotes or finishing a list. In these cases, lower precision is usually sufficient.

这利用了这样一个事实：大型 LLM 在生成确定性的续写（例如闭合引号或完成一个列表）时，往往会变得更加确定。在这些情况下，较低的精度通常就足够了。

Another factor is memory pressure. If GPU memory usage is approaching its limit due to a very long context—or many parallel requests, the system can dynamically compress activations to a lower precision.

另一个因素是内存压力。如果由于非常长的上下文——或大量并行请求，GPU 内存使用逼近上限，系统可以动态地把激活压缩到更低精度。

One could store the attention key/value tensors in INT4 instead of INT8 when memory is scarce. This would reduce the memory footprint by 50%. However, make sure the quantization error from using INT4 does not compound across many decoding steps. It’s recommended to periodically reevaluate output quality.

在内存紧张时，可以把注意力的 key/value 张量以 INT4 而非 INT8 存储。这会把内存占用减少 50%。不过，要确保使用 INT4 带来的量化误差不会在许多解码步骤中累积放大。建议定期重新评估输出质量。

For instance, if an inference reaches a point where the KV cache is using 90% of memory, the engine might decide to quantize new cache entries from INT8 down to INT4—or even retroactively compress older entries—to free space. This can be done without stopping the model. In this case, the next attention layers simply read the INT4 cached values—with minor quantization error.

例如，如果某次推理达到 KV 缓存占用 90% 内存的临界点，引擎可能决定把新的缓存条目从 INT8 量化到 INT4——甚至回溯性地压缩较旧的条目——以释放空间。这可以在不停止模型的情况下完成。此时，后续的注意力层只需读取 INT4 缓存值——仅带有轻微的量化误差。

Combining 4-bit weight quantization with 8-bit activations can reduce memory significantly. For instance, pure compute-limited kernels FP8 activations can achieve up to 2× throughput—especially on high-bandwidth modern GPUs. For mixed or memory-bound workloads, 1.5× is achievable. Using FP4 for activations can push memory savings even further. However, it may introduce slightly higher cumulative error that requires careful layer-wise tuning.

将 4 位权重量化与 8 位激活相结合可以显著减少内存。例如，对于纯计算受限的核函数，FP8 激活可实现最高 2× 的吞吐量——尤其是在高带宽的现代 GPU 上。对于混合型或内存受限的负载，1.5× 是可以实现的。对激活使用 FP4 可以把内存节省推进得更远。不过，它可能引入略高的累积误差，需要仔细的逐层调优。

Modern GPUs provide native FP8 and FP4 Tensor Cores. However, PyTorch’s AMP support (torch.autocast) still only targets FP16 and BF16 as of this writing. It does not target FP8 or FP4. While FP8 dtypes exist in PyTorch (e.g., torch.float8_e4m3 and torch.float8_e5m2) alongside scaled math paths, AMP does not manage them. For inference and training, it’s recommended to use NVIDIA’s Transformer Engine (TE) and adopt its MXFP8 and NVFP4 when appropriate.

现代 GPU 提供原生的 FP8 与 FP4 Tensor Core。然而，截至本文撰写时，PyTorch 的 AMP（自动混合精度，automatic mixed precision）支持（torch.autocast）仍只面向 FP16 与 BF16。它并不面向 FP8 或 FP4。尽管 PyTorch 中存在 FP8 dtype（例如 torch.float8_e4m3 与 torch.float8_e5m2）以及带缩放的数学通路，但 AMP 并不管理它们。对于推理与训练，建议使用 NVIDIA 的 Transformer Engine（TE），并在适当时采用其 MXFP8 与 NVFP4。

> For latency-critical decode, prefer BF16 over FP16 on Blackwell when using AMP. For FP8 paths, the Transformer Engine’s MXFP8 format is the recommended default on Blackwell. Use NVFP4 selectively for KV cache and light layers with careful regression testing. Remember to validate numerics per layer on your specific workload.

> 对于延迟关键的 decode，在 Blackwell 上使用 AMP 时优先选择 BF16 而非 FP16。对于 FP8 通路，Transformer Engine 的 MXFP8 格式是 Blackwell 上推荐的默认选项。在 KV 缓存与轻量层上有选择地使用 NVFP4，并配合仔细的回归测试。记得在你的具体负载上逐层验证数值。

Table 19-2 summarizes some example precision configurations and their trade-offs. Here, you see that lower precision reduces memory and increases throughput. However, there is slight quality degradation.

表 19-2 汇总了一些示例精度配置及其权衡。可以看到，更低的精度会减少内存并提高吞吐量。不过，会带来轻微的质量下降。

Table 19-2. Approximate trade-offs of precision modes in LLM inference

表 19-2. LLM 推理中各精度模式的近似权衡

| Precision mode | Memory usage (relative) | Compute throughput | Quality impact (accuracy delta) |
| --- | --- | --- | --- |
| FP16 (baseline) | 1.0× (100%) | 1.0× (baseline) | No impact (full fidelity) |
| FP16 weights + FP8 activations | ~0.5× (50%) | ~1.5× | Negligible (< 0.1%) |
| INT4 weights + FP8 activations | ~0.25× (25%) | ~1.8× | ~0.5% drop in quality (mixed compute and memory bound) |
| INT4 weights + FP4 activations | ~0.2× (20%) | ~3.5× | ~1% drop (requires careful tuning) |

| 精度模式 | 内存使用（相对） | 计算吞吐量 | 质量影响（精度变化） |
| --- | --- | --- | --- |
| FP16（基线） | 1.0× (100%) | 1.0×（基线） | 无影响（完全保真） |
| FP16 权重 + FP8 激活 | ~0.5× (50%) | ~1.5× | 可忽略（< 0.1%） |
| INT4 权重 + FP8 激活 | ~0.25× (25%) | ~1.8× | 质量下降 ~0.5%（计算与内存混合受限） |
| INT4 权重 + FP4 激活 | ~0.2× (20%) | ~3.5× | 下降 ~1%（需要仔细调优） |

Here, we see that with FP8 activations, we get a memory reduction of ~50% from the baseline FP16 as expected by reducing the activation bit-width by 50%. Additionally, the quality loss measured here is negligible (< 0.1%) for FP8 activations. (Note: quality impact with reduced precision is model-dependent and kernel-dependent. You should validate with your own data and workloads.)

这里可以看到，使用 FP8 激活时，我们相对基线 FP16 获得了约 50% 的内存减少，这与把激活位宽降低 50% 的预期一致。此外，这里测得的 FP8 激活的质量损失可忽略不计（< 0.1%）。（注：降低精度带来的质量影响与模型相关、也与核函数相关。你应当用自己的数据与负载进行验证。）

INT4 weight + FP8 activation workflows can produce about ~1.8× of the baseline throughput when memory is the main bottleneck. INT4 weights + FP4 activations can reduce memory down to 20% of the baseline. 4-bit targets. The speedup is around 3.5×, which is consistent with the theoretical peak 4× improvement over FP16.

当内存是主要瓶颈时，INT4 权重 + FP8 激活的工作流可产生约 ~1.8× 的基线吞吐量。INT4 权重 + FP4 激活可将内存减少到基线的 20%。4 位目标。加速比约为 3.5×，这与相对 FP16 的理论峰值 4× 提升一致。

The goal of dynamic precision switching is to maximize performance while keeping output quality within acceptable bounds. Ideally, the kernel runs in the fastest possible precision (e.g., FP8 or FP4) and falls back to higher precision (e.g., FP16) only when necessary. In practice, libraries like NVIDIA’s Transformer Engine for PyTorch allow layer-wise control over precision at runtime.

动态精度切换的目标是在把输出质量保持在可接受范围内的同时最大化性能。理想情况下，核函数以尽可能最快的精度（例如 FP8 或 FP4）运行，只在必要时才回退到更高精度（例如 FP16）。在实践中，像 NVIDIA 面向 PyTorch 的 Transformer Engine 这样的库，允许在运行时对精度进行逐层控制。

Linear layers might default to FP8, but a runtime hook could increase a layer’s precision to FP16 or reduce it to FP4 depending on the layer’s role. For instance, FP4 could be applied to lightweight layers like output projections in which minor accuracy degradation is tolerable, while FP8 or FP16 might be used for early layers that process raw user inputs and benefit from higher precision.

线性层可能默认使用 FP8，但一个运行时钩子可以根据层的作用，把某层的精度提高到 FP16 或降低到 FP4。例如，FP4 可以应用于输出投影这类轻量层——在这些层上轻微的精度下降是可以容忍的——而 FP8 或 FP16 则可用于处理原始用户输入、并从更高精度中受益的早期层。

Beyond per-layer mixed-precision control, you can use a more fine-grained optimization strategy, which adjusts the precision per token. This approach lets the inference system run in the fastest mode possible, FP8 for instance, when predictions are confident. It would then fall back to higher-precision modes like FP16 when it’s more uncertain.

除了逐层的混合精度控制之外，你还可以使用一种更细粒度的优化策略——按 token 调整精度。这种方法让推理系统在预测有把握时以尽可能最快的模式（例如 FP8）运行。随后，当它更不确定时，便回退到 FP16 这类更高精度的模式。

In practice, the model generates the current token using a default precision (e.g., FP16), then evaluates confidence based on runtime metrics, such as output entropy, maximum softmax probability, or logit variance.

在实践中，模型先以默认精度（例如 FP16）生成当前 token，然后基于诸如输出熵、最大 softmax 概率或 logit 方差等运行时指标来评估置信度。

If the model is highly confident in its prediction, the next token can be processed at a lower precision. If uncertainty is high, the system reverts to a more stable format to maintain output quality. Here is example code that demonstrates the concept:

如果模型对其预测高度自信，下一个 token 就可以用更低的精度处理。如果不确定性很高，系统则回退到更稳定的格式以维持输出质量。下面是演示该概念的示例代码：

```
import contextlib
import torch
# ----------------------------
# Safe Transformer Engine (TE) FP8 autocast import
# ----------------------------
try:
    # TE is only effective if your model actually uses TE-enabled layers
    # (e.g., Linear, LayerNorm wrappers).
    from transformer_engine.pytorch import fp8_autocast as _te_fp8_autocast
    # type: ignore
    _TE_AVAILABLE = True
except Exception:
    _TE_AVAILABLE = False
    # No-op stand-in so the code runs without TE installed. It never changes
    # numerical behavior.
    class _NullCtx(contextlib.ContextDecorator):
        def __init__(self, **_): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return False
    def _te_fp8_autocast(**_):
        return _NullCtx()
# ----------------------------
# Helper: choose the precision context *for this step* safely
# ----------------------------
def _precision_context_cuda(use_fp8: bool,
                            prefer_bfloat16: bool,
                            enable_fp8: bool):
    """
    Enter exactly one precision context. If FP8 isn't enabled or TE is
    missing/unused, fall back to AMP (BF16/FP16).
    """
    if use_fp8 and enable_fp8 and _TE_AVAILABLE:
        # Note: fp8_autocast affects only TE-enabled modules. Non-TE modules
        # run at their native dtypes.
        return _te_fp8_autocast(enabled=True)
    amp_dtype = torch.bfloat16 if prefer_bfloat16 else torch.float16
    return torch.autocast(device_type="cuda", dtype=amp_dtype)
def _precision_context(device: torch.device, use_fp8: bool,
                       prefer_bfloat16: bool, enable_fp8: bool):
    return _precision_context_cuda(use_fp8, prefer_bfloat16,
                                   enable_fp8) if device.type == "cuda"
                                   else contextlib.nullcontext()
# ----------------------------
# Main decode loop with smoothed, hysteretic precision switching
# ----------------------------
@torch.no_grad()
def decode_with_dynamic_precision(
    model,
    tokens: torch.Tensor,
    max_steps: int,
    *,
    device: torch.device = torch.device("cuda"),
    prefer_bfloat16: bool = True,      # B200: prefer BF16 over FP16 for AMP
    enable_fp8: bool = True,           # Allow FP8 when TE present
    enter_fp8_threshold: float = 6.0,  # hysteresis upper bound
                                       # (logit margin average)
    exit_fp8_threshold: float = 3.0,   # hysteresis lower bound (avoid flapping)
    reeval_interval: int = 8,          # compute/inspect confidence every N steps
                                       # to avoid per-step sync
    topk_dim: int = -1,                # last dimension holds vocabulary logits
    eos_id: int | None = None,
):
    """
    Autoregressive decode loop that *smoothly* switches between AMP (BF16/FP16)
    and FP8 (TE) without per-step host sync. Works even when TE is not
    installed; in that case, runs AMP only.
    - Confidence signal: mean(top1 - top2) logits margin across the batch.
    - Smoothing: EMA + interval re-evaluation to minimize CPU-GPU sync pressure.
    - Hysteresis: separate enter/exit thresholds to avoid precision flapping.
    """
    assert exit_fp8_threshold <= enter_fp8_threshold,
        "Hysteresis requires exit <= enter threshold"
    model.eval()
    tokens = tokens.to(device, non_blocking=True)
    # Internal state
    use_fp8: bool = False  # start in AMP.
                           # Upgrade to FP8 when sustained confidence permits
    ema_conf: torch.Tensor | None = None  # stays on device;
                                          # host consults only at intervals
    alpha = 0.2  # EMA smoothing factor for confidence
  # A tiny helper to update on-device EMA without host sync
    def _update_confidence_ema(logits: torch.Tensor) -> torch.Tensor:
        # logits: [B, vocab] or [B, T, vocab]. Use the last time-step if 3D.
        last = logits if logits.dim() == 2 else logits[:, -1, :]
        # Compute top-2 margin on-device
        top2 = torch.topk(last, k=2, dim=topk_dim).values  # [B, 2]
        margin = (top2[:, 0] - top2[:, 1]).mean()      # scalar tensor on device
        nonlocal ema_conf
        ema_conf = (1 - alpha)
                   * (ema_conf if ema_conf is not None else margin)+alpha*margin
        return ema_conf  # device scalar
    # Decode
    for step in range(max_steps):
        # 1) Precision context (exactly one).
        # No nested contexts, no leakage across iterations.
        with _precision_context(device, use_fp8, prefer_bfloat16, enable_fp8):
            # Forward pass (HF-style or plain)
            try:
                logits = model(input_ids=tokens)
                if hasattr(logits, "logits"):
                    logits = logits.logits
            except TypeError:
                logits = model(tokens)
            # 2) Pick next token from the *last* position
            last_step_logits = logits if logits.dim() == 2 else logits[:, -1, :]
            next_token = torch.argmax(last_step_logits, dim=-1,
                                      keepdim=True)  # [B, 1]
            tokens = torch.cat([tokens, next_token], dim=1)
        # 3) Update on-device EMA signal every step (no host sync yet)
        conf_dev = _update_confidence_ema(logits)
        # 4) Periodically re-evaluate precision choice on host
        # to avoid per-step sync
        if (step + 1) % reeval_interval == 0:
            conf_value = float(conf_dev)  # exactly one tiny sync every N steps
            if not use_fp8 and enable_fp8 and _TE_AVAILABLE
               and (conf_value > enter_fp8_threshold):
                use_fp8 = True
            elif use_fp8 and (conf_value < exit_fp8_threshold):
                use_fp8 = False
        # 5) EOS handling
        if eos_id is not None:
            if (tokens[:, -1] == eos_id).all():
                break
    return tokens
# ----------------------------
# Example (commented):
# ----------------------------
# model = ...  # your TE-enabled model (or any torch.nn.Module)
# input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
# out = decode_with_dynamic_precision(model, input_ids, max_steps=128,
# eos_id=tokenizer.eos_token_id)
# print(out.shape)
```

```
import contextlib
import torch
# ----------------------------
# Safe Transformer Engine (TE) FP8 autocast import
# ----------------------------
try:
    # TE is only effective if your model actually uses TE-enabled layers
    # (e.g., Linear, LayerNorm wrappers).
    from transformer_engine.pytorch import fp8_autocast as _te_fp8_autocast
    # type: ignore
    _TE_AVAILABLE = True
except Exception:
    _TE_AVAILABLE = False
    # No-op stand-in so the code runs without TE installed. It never changes
    # numerical behavior.
    class _NullCtx(contextlib.ContextDecorator):
        def __init__(self, **_): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return False
    def _te_fp8_autocast(**_):
        return _NullCtx()
# ----------------------------
# Helper: choose the precision context *for this step* safely
# ----------------------------
def _precision_context_cuda(use_fp8: bool,
                            prefer_bfloat16: bool,
                            enable_fp8: bool):
    """
    Enter exactly one precision context. If FP8 isn't enabled or TE is
    missing/unused, fall back to AMP (BF16/FP16).
    """
    if use_fp8 and enable_fp8 and _TE_AVAILABLE:
        # Note: fp8_autocast affects only TE-enabled modules. Non-TE modules
        # run at their native dtypes.
        return _te_fp8_autocast(enabled=True)
    amp_dtype = torch.bfloat16 if prefer_bfloat16 else torch.float16
    return torch.autocast(device_type="cuda", dtype=amp_dtype)
def _precision_context(device: torch.device, use_fp8: bool,
                       prefer_bfloat16: bool, enable_fp8: bool):
    return _precision_context_cuda(use_fp8, prefer_bfloat16,
                                   enable_fp8) if device.type == "cuda"
                                   else contextlib.nullcontext()
# ----------------------------
# Main decode loop with smoothed, hysteretic precision switching
# ----------------------------
@torch.no_grad()
def decode_with_dynamic_precision(
    model,
    tokens: torch.Tensor,
    max_steps: int,
    *,
    device: torch.device = torch.device("cuda"),
    prefer_bfloat16: bool = True,      # B200: prefer BF16 over FP16 for AMP
    enable_fp8: bool = True,           # Allow FP8 when TE present
    enter_fp8_threshold: float = 6.0,  # hysteresis upper bound
                                       # (logit margin average)
    exit_fp8_threshold: float = 3.0,   # hysteresis lower bound (avoid flapping)
    reeval_interval: int = 8,          # compute/inspect confidence every N steps
                                       # to avoid per-step sync
    topk_dim: int = -1,                # last dimension holds vocabulary logits
    eos_id: int | None = None,
):
    """
    Autoregressive decode loop that *smoothly* switches between AMP (BF16/FP16)
    and FP8 (TE) without per-step host sync. Works even when TE is not
    installed; in that case, runs AMP only.
    - Confidence signal: mean(top1 - top2) logits margin across the batch.
    - Smoothing: EMA + interval re-evaluation to minimize CPU-GPU sync pressure.
    - Hysteresis: separate enter/exit thresholds to avoid precision flapping.
    """
    assert exit_fp8_threshold <= enter_fp8_threshold,
        "Hysteresis requires exit <= enter threshold"
    model.eval()
    tokens = tokens.to(device, non_blocking=True)
    # Internal state
    use_fp8: bool = False  # start in AMP.
                           # Upgrade to FP8 when sustained confidence permits
    ema_conf: torch.Tensor | None = None  # stays on device;
                                          # host consults only at intervals
    alpha = 0.2  # EMA smoothing factor for confidence
  # A tiny helper to update on-device EMA without host sync
    def _update_confidence_ema(logits: torch.Tensor) -> torch.Tensor:
        # logits: [B, vocab] or [B, T, vocab]. Use the last time-step if 3D.
        last = logits if logits.dim() == 2 else logits[:, -1, :]
        # Compute top-2 margin on-device
        top2 = torch.topk(last, k=2, dim=topk_dim).values  # [B, 2]
        margin = (top2[:, 0] - top2[:, 1]).mean()      # scalar tensor on device
        nonlocal ema_conf
        ema_conf = (1 - alpha)
                   * (ema_conf if ema_conf is not None else margin)+alpha*margin
        return ema_conf  # device scalar
    # Decode
    for step in range(max_steps):
        # 1) Precision context (exactly one).
        # No nested contexts, no leakage across iterations.
        with _precision_context(device, use_fp8, prefer_bfloat16, enable_fp8):
            # Forward pass (HF-style or plain)
            try:
                logits = model(input_ids=tokens)
                if hasattr(logits, "logits"):
                    logits = logits.logits
            except TypeError:
                logits = model(tokens)
            # 2) Pick next token from the *last* position
            last_step_logits = logits if logits.dim() == 2 else logits[:, -1, :]
            next_token = torch.argmax(last_step_logits, dim=-1,
                                      keepdim=True)  # [B, 1]
            tokens = torch.cat([tokens, next_token], dim=1)
        # 3) Update on-device EMA signal every step (no host sync yet)
        conf_dev = _update_confidence_ema(logits)
        # 4) Periodically re-evaluate precision choice on host
        # to avoid per-step sync
        if (step + 1) % reeval_interval == 0:
            conf_value = float(conf_dev)  # exactly one tiny sync every N steps
            if not use_fp8 and enable_fp8 and _TE_AVAILABLE
               and (conf_value > enter_fp8_threshold):
                use_fp8 = True
            elif use_fp8 and (conf_value < exit_fp8_threshold):
                use_fp8 = False
        # 5) EOS handling
        if eos_id is not None:
            if (tokens[:, -1] == eos_id).all():
                break
    return tokens
# ----------------------------
# Example (commented):
# ----------------------------
# model = ...  # your TE-enabled model (or any torch.nn.Module)
# input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
# out = decode_with_dynamic_precision(model, input_ids, max_steps=128,
# eos_id=tokenizer.eos_token_id)
# print(out.shape)
```

Here we see that PyTorch autocast supports only reduced-precision FP16 and BF16 as of this writing. In this case, you need to use the Transformer Engine library to route supported modules to FP8 kernels.

这里我们看到，截至本文撰写时，PyTorch autocast 只支持 FP16 与 BF16 这两种降精度格式。此时，你需要使用 Transformer Engine 库把受支持的模块路由到 FP8 核函数。

> The threshold used in this example (enter = 6.0, exit = 3.0) should be calibrated on a validation set using representative prompts to prevent latency gains from impacting accuracy.

> 本例中使用的阈值（enter = 6.0, exit = 3.0）应当在验证集上用有代表性的提示进行校准，以防延迟收益影响精度。

This pattern creates an elastic precision regime and maximizes throughput. When the model operates in predictable (e.g., low-entropy) regions, such as generating punctuation or boilerplate completions, it continues in FP8 to maximize performance. When it enters higher-entropy segments, such as ambiguous prompts or decision points, it returns to FP16 to preserve numerical accuracy.

这种模式创建了一种弹性的精度机制，并最大化吞吐量。当模型运行在可预测（例如低熵）区域时，例如生成标点或样板式的补全，它会继续使用 FP8 以最大化性能。当它进入更高熵的片段时，例如含糊的提示或决策点，它会回到 FP16 以保持数值精度。

When paired with a modern GPU’s support for low-precision operations, token-level dynamic precision switching offers an adaptive strategy for high-throughput, latency-sensitive inference. It applies low precision only when needed, reduces compute overhead, and maintains response quality across many different prompt conditions.

当与现代 GPU 对低精度运算的支持相结合时，token 级的动态精度切换为高吞吐、延迟敏感的推理提供了一种自适应策略。它只在需要时才应用低精度，减少计算开销，并在许多不同的提示条件下维持响应质量。

## Kernel Autotuning for Transformer Self-Attention and MLP Paths

## 面向 Transformer 自注意力与 MLP 通路的核函数自动调优

The performance of neural network layers on GPUs can vary drastically depending on low-level parameters like thread block size, tile dimensions, loop unrolling, and memory access patterns. For fixed-size models, libraries typically choose these parameters only once—often using general heuristics or offline tuning.

神经网络层在 GPU 上的性能，会因线程块（thread block）大小、分块维度、循环展开与内存访问模式等底层参数而剧烈变化。对于固定尺寸的模型，库通常只选择这些参数一次——往往使用通用的启发式方法或离线调优。

However, in an online inference service scenario, input sizes, including sequence lengths and batch sizes, can vary from request to request. Kernel autotuning refers to a runtime mechanism that selects—or even JIT-compiles—the optimal kernel variant for the current workload.

然而，在在线推理服务场景中，输入尺寸（包括序列长度与 batch size）会因请求而异。核函数自动调优（kernel autotuning）指的是一种运行时机制，它为当前负载选择——甚至即时（JIT）编译——最优的核函数变体。

In the context of large transformer models, the two major compute phases of inference are self-attention and feed-forward MLP layers. Both can benefit from autotuning of their GPU kernels. Let’s cover each of these in the context of kernel autotuning.

在大型 Transformer 模型的语境下，推理的两大计算阶段是自注意力（self-attention）与前馈 MLP 层。两者都能从各自 GPU 核函数的自动调优中受益。我们逐一在核函数自动调优的语境下讨论它们。

Consider an attention layer that processes a sequence of length *L* with *H* attention heads. There are many implementations of attention, including standard attention and optimized FlashAttention—and its multiple variants.

考虑一个用 *H* 个注意力头处理长度为 *L* 的序列的注意力层。注意力有许多实现，包括标准注意力与经过优化的 FlashAttention——及其多个变体。

FlashAttention and its variants are significantly faster for long sequences due to tiling, parallelism, and memory-access improvements. However, for very short sequences, its overhead might outweigh its benefit. A dynamic engine can switch between a FlashAttention kernel and a simpler kernel depending on the sequence length, *L*.

由于分块（tiling）、并行以及内存访问方面的改进，FlashAttention 及其变体在长序列上显著更快。不过，对于非常短的序列，它的开销可能超过收益。动态引擎可以根据序列长度 *L* 在 FlashAttention 核函数与更简单的核函数之间切换。

For instance, if a request has *L* = 256 tokens, the engine might use a straightforward kernel launch that computes attention in one go using global memory reads, which are sufficient for small *L*. If another request comes in with *L* = 2,048, it could switch to FlashAttention’s specialized tiling kernel known to scale better for large *L* by reusing data in shared memory and avoiding unnecessary HBM data fetches. This is demonstrated as a condition statement based on the input sequence length, as shown here:

例如，如果某个请求的 *L* = 256 tokens，引擎可能使用一次性完成计算的直接核函数启动，通过全局内存读取来计算注意力，这对小 *L* 已经足够。如果另一个请求带着 *L* = 2,048 到来，它可以切换到 FlashAttention 专门的分块核函数——已知它通过在共享内存中复用数据、避免不必要的 HBM 数据取用，从而对大 *L* 有更好的扩展性。这可以表示为一个基于输入序列长度的条件语句，如下所示：

```
// Note: example threshold shown here
if (seq_len < 256) {
    // global-memory version, best for small L
    attn_kernel = standard_attention_kernel;
} else {
    // tiled loads, best for large L
    attn_kernel = tiled_attention_kernel;
}
output = attn_kernel(Q, K, V, mask);
```

```
// Note: example threshold shown here
if (seq_len < 256) {
    // global-memory version, best for small L
    attn_kernel = standard_attention_kernel;
} else {
    // tiled loads, best for large L
    attn_kernel = tiled_attention_kernel;
}
output = attn_kernel(Q, K, V, mask);
```

Behind the scenes, attn_kernel picks between completely different CUDA implementations. One implementation is optimized for small inputs using the default attention kernel, and another is optimized for large contexts using the tiled kernel.

在幕后，attn_kernel 在完全不同的 CUDA 实现之间做选择。一种实现使用默认注意力核函数针对小输入做优化，另一种则使用分块核函数针对大上下文做优化。

The ideal tile dimensions depend on your GPU’s shared-memory capacity and compute resources. Frameworks like CUTLASS and OpenAI’s Triton include autotuners that benchmark a range of (TILE_Q, TILE_K) combinations at initialization—or even adaptively at runtime—to select the fastest variant. Table 19-3 shows examples of how different tile sizes perform on a Blackwell-class GPU.

理想的分块维度取决于你 GPU 的共享内存容量与计算资源。像 CUTLASS 与 OpenAI 的 Triton 这样的框架内置了自动调优器，会在初始化时——甚至在运行时自适应地——对一系列 (TILE_Q, TILE_K) 组合进行基准测试，以选出最快的变体。表 19-3 展示了不同分块大小在 Blackwell 级 GPU 上表现的示例。

Table 19-3. Example impact of tile-size choice on shared-memory footprint, SM occupancy, and achieved throughput (actual values will depend on thread-block dimensions, clock rates, and other microarchitectural factors)

表 19-3. 分块大小选择对共享内存占用、SM 占用率与实际吞吐量影响的示例（实际数值取决于线程块维度、时钟频率与其他微架构因素）

| Tile size | Shared memory (KB) | Occupancy (%) | Throughput (GOPS) |
| --- | --- | --- | --- |
| 64 × 64 | 48 | 85 | 8.2 |
| 128 × 64 | 64 | 78 | 10.5 |
| 128 × 128 | 96 | 72 | 9.8 |
| 256 × 128 | 128 | 60 | 11.3 |

| 分块大小 | 共享内存 (KB) | 占用率 (%) | 吞吐量 (GOPS) |
| --- | --- | --- | --- |
| 64 × 64 | 48 | 85 | 8.2 |
| 128 × 64 | 64 | 78 | 10.5 |
| 128 × 128 | 96 | 72 | 9.8 |
| 256 × 128 | 128 | 60 | 11.3 |

By choosing the right variant at runtime based on the input, you avoid the huge performance cliff of a one-size-fits-all approach. In practice you might benchmark on your target hardware to find that around *L* = 128 is the breakeven point.

通过在运行时根据输入选择正确的变体，你可以避免一刀切方案带来的巨大性能悬崖。在实践中，你可能会在目标硬件上做基准测试，发现大约 *L* = 128 是盈亏平衡点。

Next, let’s analyze the feed-forward MLP kernels in the context of autotuning. The feed-forward layers are essentially large matrix multiplications—specifically, two linear projections with a nonlinear activation in between.

接下来，我们在自动调优的语境下分析前馈 MLP 核函数。前馈层本质上是大型矩阵乘法——具体来说，是中间夹着一个非线性激活的两个线性投影。

Modern AI frameworks like PyTorch use highly optimized GEMM kernels using optimized CUDA libraries like cuBLAS and CUTLASS. There are often multiple algorithmic variants in these libraries for a given matrix size that use different tiling strategies, different Tensor Cores, and separate fallback paths.

像 PyTorch 这样的现代 AI 框架使用高度优化的 GEMM 核函数，借助 cuBLAS 与 CUTLASS 这类优化过的 CUDA 库。对于给定的矩阵尺寸，这些库中往往有多个算法变体，它们使用不同的分块策略、不同的 Tensor Core 以及各自的回退通路。

For instance, NVIDIA’s cuBLAS and cuBLASLt libraries can autotune GEMM kernels by first trying a few algorithms, then picking the fastest algorithm for the given dimensions. However, this typically happens the first time a GEMM of that shape is encountered—and not revisited.

例如，NVIDIA 的 cuBLAS 与 cuBLASLt 库可以对 GEMM 核函数进行自动调优——先尝试几种算法，再为给定维度挑选最快的算法。不过，这通常只发生在第一次遇到该形状的 GEMM 时——之后不再重新评估。

> Where available in cuBLAS/cuBLASLt or custom kernels, programmatic dependent launch (PDL) can reduce launch gaps and improve steady-state throughput. Make sure to profile to confirm overlap.

> 在 cuBLAS/cuBLASLt 或自定义核函数中可用之处，PDL（编程式依赖启动，programmatic dependent launch）可以减少启动间隙并改善稳态吞吐量。务必通过性能分析确认重叠效果。

In an inference server that sees many different batch sizes, one can explicitly invoke such autotuning mechanisms—or maintain a cache of best algorithms. For instance, for the MLP’s GEMM of shape [batch_size, hidden_dim] x [hidden_dim, 4*hidden_dim], the optimal kernel might differ for batch_size = 1 versus batch_size = 16.

在一个会遇到许多不同 batch size 的推理服务器中，可以显式地调用这类自动调优机制——或维护一个最优算法的缓存。例如，对于 MLP 中形状为 [batch_size, hidden_dim] x [hidden_dim, 4*hidden_dim] 的 GEMM，其最优核函数在 batch_size = 1 与 batch_size = 16 时可能不同。

The engine can detect a new batch size and run a quick microbenchmark of candidate kernels using cuBLASLt or a custom implementation to select the fastest kernel. Subsequent calls with that batch size can then directly use the chosen kernel.

引擎可以检测到新的 batch size，并用 cuBLASLt 或自定义实现对候选核函数运行一次快速的微基准测试（microbenchmark），以选出最快的核函数。之后使用该 batch size 的调用便可直接使用所选的核函数。

In addition, some inference frameworks and runtimes use OpenAI’s Triton GPU kernel domain-specific library (DSL) to compile attention and MLP kernels on the fly with autotuned tile sizes. In this case, the runtime would generate a few variants of a kernel with different tile sizes (e.g., 128 × 128, 64 × 256, etc.) and measure which performs better given the actual hardware and input shape.

此外，一些推理框架与运行时使用 OpenAI 的 Triton GPU 核函数领域特定库（domain-specific library，DSL），以自动调优的分块大小即时编译注意力与 MLP 核函数。此时，运行时会生成核函数的若干个不同分块大小的变体（例如 128 × 128、64 × 256 等），并测量在实际硬件与输入形状下哪个表现更好。

You can use tools like Nsight Systems to empirically profile different kernel variants side by side.

你可以使用 Nsight Systems 这类工具，把不同的核函数变体并排进行经验性能分析。

Specifically, Nsight Systems provides detailed CUDA timelines, including memcpy and NVLink activity, and Nsight Compute provides memory workload analysis that helps attribute cache and memory behavior to kernel sites. This is particularly useful when evaluating tile-size and shared-memory trade-offs. In addition, it can often reveal nonobvious bottlenecks like L2 cache misses that will further guide your tuning decisions.

具体而言，Nsight Systems 提供详细的 CUDA 时间线，包括 memcpy 与 NVLink 活动；Nsight Compute 则提供内存工作负载分析，帮助把缓存与内存行为归因到核函数位置。这在评估分块大小与共享内存权衡时尤其有用。此外，它常常能揭示诸如 L2 缓存未命中这类不明显的瓶颈，从而进一步指导你的调优决策。

> Because hardware can sometimes be somewhat unpredictable under load given L2 cache effects, memory bank conflicts, etc., empirical tuning will always beat theoretical guesses. But it’s good to start the tuning process with reasonable theoretical values.

> 由于 L2 缓存效应、内存 bank 冲突等原因，硬件在负载下有时会有些难以预测，因此经验调优总会胜过理论猜测。但用合理的理论值来开启调优过程是个不错的做法。

Dynamic tile switching affects GPU occupancy and should be considered when choosing a tile size. Using a larger tile can increase reuse and reduce kernel-launch overhead, but it can also use more registers and shared memory. This will potentially reduce the number of thread blocks that can run concurrently—reducing occupancy. A proper autotuner will consider this trade-off.

动态分块切换会影响 GPU 占用率（occupancy），在选择分块大小时应加以考虑。使用更大的分块可以增加复用并减少核函数启动开销，但也会使用更多寄存器（register）与共享内存。这可能会减少能够并发运行的线程块数量——从而降低占用率。一个合适的自动调优器会权衡这一取舍。

In attention kernels, a larger tile (e.g., 128 × 128) maximizes data reuse in shared memory. This is ideal for long sequences since you issue fewer global-memory loads, amortize loop overhead, and produce higher sustained throughput.

在注意力核函数中，更大的分块（例如 128 × 128）能最大化共享内存中的数据复用。这对长序列很理想，因为你发出的全局内存加载更少、摊薄了循环开销，并产生更高的持续吞吐量。

For shorter sequences, however, that same large tile can consume too much shared memory, which limits the occupancy, or number of concurrent thread blocks running on each SM. By reducing the tile size (e.g., 64 × 64) for shorter sequences, you free up shared memory so you can schedule more blocks in parallel. This boosts SM occupancy and reduces per-kernel latency.

然而，对于较短的序列，同样的大分块会消耗过多共享内存，从而限制占用率，即每个 SM 上并发运行的线程块数量。通过为较短序列减小分块大小（例如 64 × 64），你可以释放共享内存，从而并行调度更多的块。这提升了 SM 占用率并降低了单核函数延迟。

By adapting the tile size based on the input sequence length, the kernel can achieve near-optimal occupancy in most cases. Some systems even query the CUDA Occupancy API at runtime to choose kernel-launch parameters dynamically, such as thread block size. An example of the Occupancy API in C++ is shown next, but a Python API is also available:

通过根据输入序列长度自适应地调整分块大小，核函数在大多数情况下都能达到接近最优的占用率。有些系统甚至在运行时查询 CUDA Occupancy API，以动态选择诸如线程块大小之类的核函数启动参数。下面展示一个 C++ 中 Occupancy API 的示例，不过也有可用的 Python API：

```
// Pseudocode for occupancy-based launch configuration
int maxBlocks, bestThreads;
for (int threads = 64; threads <= 256; threads *= 2) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocks, MyKernel, threads,
        sharedMemPerBlock(threads));
    // choose "threads" to maximize occupancy
    // (remember to not exceed the max threads per SM limit (e.g., 2,048)
    float occupancy = (float) maxBlocks * threads /
                              hardwareMaxThreadsPerSM;
}
```

```
// Pseudocode for occupancy-based launch configuration
int maxBlocks, bestThreads;
for (int threads = 64; threads <= 256; threads *= 2) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocks, MyKernel, threads,
        sharedMemPerBlock(threads));
    // choose "threads" to maximize occupancy
    // (remember to not exceed the max threads per SM limit (e.g., 2,048)
    float occupancy = (float) maxBlocks * threads /
                              hardwareMaxThreadsPerSM;
}
```

This pseudo-C++ illustrates evaluating different thread block sizes for a kernel. It checks how many blocks per SM can run given their shared-memory usage. The kernel launch then adjusts the number of threads—or shared-memory use—accordingly. High-performance frameworks and inference engines automate this type of logic internally using the following set of steps:

这段伪 C++ 展示了如何为一个核函数评估不同的线程块大小。它检查在给定共享内存使用量的情况下，每个 SM 能运行多少个块。随后核函数启动会相应地调整线程数——或共享内存使用量。高性能框架与推理引擎会在内部用以下一组步骤将这类逻辑自动化：

*1. Measure workload* Inspect the current input dimensions (batch size B, sequence length L, etc.) for the next model forward.

*1. 测量负载* 检查下一次模型前向的当前输入维度（batch size B、序列长度 L 等）。

*2. Select candidate kernels* Determine available kernel implementations for each component, such as standard attention or flash attention for the attention phase and an appropriate GEMM algorithm for the MLP phase.

*2. 选择候选核函数* 为每个组件确定可用的核函数实现，例如注意力阶段的标准注意力或 flash attention，以及 MLP 阶段合适的 GEMM 算法。

*3. Estimate or benchmark* Run a quick run of each candidate by executing each algorithm for a few iterations on sample data.

*3. 估算或基准测试* 对每个候选项做一次快速试跑，即在样本数据上将每种算法执行若干次迭代。

*4. Choose best variant* Select the kernel with minimal execution time—or sufficient throughput—based on the measured dimensions.

*4. 选择最佳变体* 根据实测维度，选出执行时间最短——或吞吐量足够——的核函数。

*5. Cache result* Store the choice in a lookup table keyed by the input dimension or workload signature. This way, if a similar request appears, the best kernel is known without having to rerun these steps.

*5. 缓存结果* 将该选择存入以输入维度或负载特征为键的查找表中。这样，若出现类似请求，无需重跑这些步骤即可知道最佳核函数。

*6. Execute* Run the model layer using the chosen kernel implementation.

*6. 执行* 用选定的核函数实现运行该模型层。

> This is analogous to a database query optimizer picking a query plan. Here, the “plan” is the chosen kernel implementation.

> 这类似于数据库查询优化器挑选查询计划。这里的“计划”就是选定的核函数实现。

By following such a process, the inference runtime continuously tunes itself. Over time, the system builds a library of optimized paths for various scenarios, such as short versus long prompts, small versus large batches, etc. The overhead of on-the-fly tuning is kept low by either doing it asynchronously—testing new kernels in a separate stream while the current inference uses a default kernel—or during low-traffic periods so as not to impact latency.

按照这样的流程，推理运行时会持续自我调优。随着时间推移，系统会针对各种场景构建一个优化路径库，例如短提示与长提示、小批量与大批量等。即时调优的开销被控制在很低水平：要么异步进行——在单独的流中测试新核函数，同时当前推理仍使用默认核函数——要么在低流量时段进行，以免影响延迟。

It’s recommended to incorporate an initial warm-up phase when a model is loaded by running a variety of sample inputs through to trigger autotuning. This can include extremes—like max sequence length, max batch, etc.—so that the engine preoptimizes kernels for those cases.

建议在加载模型时纳入一个初始预热阶段，通过运行各种样本输入来触发自动调优。这可以包含一些极端情形——如最大序列长度、最大批量等——以便引擎为这些情形预先优化核函数。

Also, it’s best to monitor execution time at each layer during runtime. If a layer suddenly becomes a bottleneck due to a change in input characteristics, then it’s time to revisit the kernel selection.

此外，最好在运行时监控每一层的执行时间。如果某一层因输入特征变化而突然成为瓶颈，那么就该重新审视核函数选择了。

> Some advanced frameworks even use multiarmed bandit algorithms that continuously explore alternative kernels and update the choice of a different kernel as conditions change.

> 一些先进框架甚至使用多臂老虎机（multiarmed bandit）算法，持续探索备选核函数，并随条件变化更新为不同核函数的选择。

In short, autotuning transforms static kernels into adaptive ones. This squeezes the highest performance out of your GPU cluster for each set of inputs regardless of the workload. You can be confident that the system is constantly adapting.

简而言之，自动调优把静态核函数转变为自适应核函数。这能针对每一组输入压榨出 GPU 集群的最高性能，而无论负载如何。你可以确信系统一直在自适应调整。

## Dynamic Shared-Memory Allocation and Occupancy-Aware Kernel Selection

## 动态共享内存分配与占用率感知的核函数选择

Closely related to kernel tuning is the management of GPU shared memory and overall streaming multiprocessor (SM) occupancy. Modern GPUs feature a large shared memory per SM. By dynamically allocating threads at runtime based on the problem size—as well as current shared-memory utilization and occupancy—you can significantly improve overall AI system performance.

与核函数调优密切相关的是对 GPU 共享内存以及整体流式多处理器（streaming multiprocessor，SM）占用率的管理。现代 GPU 的每个 SM 都配有较大的共享内存。通过在运行时根据问题规模——以及当前的共享内存利用率和占用率——动态分配线程，你可以显著提升整个 AI 系统的性能。

With dynamic shared-memory allocation, the system adjusts the amount of shared memory that each thread block uses based on the problem size. With occupancy-aware kernel selection, the system is choosing kernel-launch parameters that make best use of the SM’s resources—including registers, shared memory, and warps—to keep the GPU busy.

借助动态共享内存分配（shared-memory allocation），系统会根据问题规模调整每个线程块所用的共享内存量。借助占用率感知的核函数选择（occupancy-aware kernel selection），系统会选择能最充分利用 SM 资源——包括寄存器、共享内存和 warp——的核函数启动参数，以让 GPU 保持繁忙。

Choosing a tiled attention algorithm should balance data reuse against SM occupancy. For instance, consider T as your tile width, in tokens, per thread block. Each thread block reserves on the order of *O*(*T*2) floats in shared memory to hold the query, key, and value chunks since self-attention is quadratic in nature.

选择分块（tiling）的注意力算法时，应在数据复用与 SM 占用率之间取得平衡。例如，设 T 为每个线程块的分块宽度（以 token 计）。由于自注意力本质上是二次复杂度，每个线程块需在共享内存中预留约 *O*(*T*2) 个浮点数，以存放 query、key 和 value 分块。

A large tile size (e.g., T=256) loads each key/value block once from DRAM and reuses it for many queries. This reduces global-memory traffic closer to O(T) floats per thread block. But because each thread block now uses a lot of shared memory, only a few thread blocks can run on an SM at once given hardware limits. This reduces occupancy. For example, if only 1 block per SM can run at T=256 versus 4 blocks at T=128, you might see only 30% SM occupancy using T=256.

大分块大小（tile size）（例如 T=256）会从 DRAM 一次性加载每个 key/value 块，并将其复用于多个 query。这将全局内存流量降低到接近每个线程块 O(T) 个浮点数。但由于每个线程块现在会使用大量共享内存，受硬件限制，一个 SM 上一次只能运行少数几个线程块。这会降低占用率。例如，若在 T=256 时每个 SM 只能运行 1 个块，而在 T=128 时能运行 4 个块，那么使用 T=256 时你可能只看到 30% 的 SM 占用率。

A small tile size (e.g., T=64) uses far less shared memory, which allows more thread blocks to fit into each SM. This better hides latency and boosts utilization. However, you end up reloading the same key/value data more often, which increases DRAM accesses.

小分块大小（例如 T=64）使用的共享内存少得多，这允许每个 SM 容纳更多线程块。这能更好地隐藏延迟并提升利用率。然而，你最终会更频繁地重新加载相同的 key/value 数据，从而增加 DRAM 访问。

The optimal tile size, T, depends on a few factors, including your sequence length L, the GPU’s shared-memory capacity, and the SM count. You want a tile that’s large enough to amortize DRAM reads but small enough to keep occupancy high enough that many thread blocks are active on the SM concurrently.

最优分块大小 T 取决于若干因素，包括序列长度 L、GPU 的共享内存容量以及 SM 数量。你希望分块足够大，以摊薄 DRAM 读取；又要足够小，以保持足够高的占用率，让许多线程块在 SM 上并发活跃。

In practice, you could manually pick a handful of candidate T values, such as 64, 128, and 256—and benchmark each value on your specific hardware using a sequence length, L, that represents your dataset. You would then choose the value of T that produces the best overall throughput. However, instead of hard-coding T ahead of time, you can compute it right before launching your kernel, as shown here:

实践中，你可以手动挑选少数几个候选 T 值，如 64、128 和 256——并在你的特定硬件上，用能代表你数据集的序列长度 L 对每个值做基准测试。然后选出总体吞吐量最佳的 T 值。不过，与其提前硬编码 T，你也可以在启动核函数前即时计算它，如下所示：

```
int T = choose_tile(L, gpu_shared_mem_per_block, num_sms);
// calculate shared memory in bytes based on the tile size
// (multiplying by 3 for Q, K, and V)
size_t shared_mem_bytes = 3 * T * T * sizeof(float);
numBlocks = ...
MyAttentionKernel<<<numBlocks, threadsPerBlock, shared_mem_bytes>>>(...);
```

```
int T = choose_tile(L, gpu_shared_mem_per_block, num_sms);
// calculate shared memory in bytes based on the tile size
// (multiplying by 3 for Q, K, and V)
size_t shared_mem_bytes = 3 * T * T * sizeof(float);
numBlocks = ...
MyAttentionKernel<<<numBlocks, threadsPerBlock, shared_mem_bytes>>>(...);
```

Here, T is computed from the sequence length L; the GPUs’ shared-memory limits per thread block, gpu_shared_mem_per_block; and the number of SMs, num_sms. Then, shared memory per thread block, shared_mem_bytes, is computed at runtime based on the computed tile size, T.

这里，T 由序列长度 L、GPU 每个线程块的共享内存上限 gpu_shared_mem_per_block 以及 SM 数量 num_sms 计算得出。然后，每个线程块的共享内存 shared_mem_bytes 会在运行时根据计算出的分块大小 T 计算得到。

You can then launch the CUDA kernel with the shared-memory argument, shared_mem_bytes. The kernel itself would contain the following to define an extern __shared__ array to allocate the shared-memory buffer of size shared_mem_bytes for each thread block:

随后你就可以用共享内存参数 shared_mem_bytes 启动该 CUDA 核函数。核函数本身会包含如下代码，定义一个 extern __shared__ 数组，为每个线程块分配大小为 shared_mem_bytes 的共享内存缓冲区：

```
// holds 3 tiles of T×T floats for Q, K, and V
extern __shared__ float smem[];
// Q tile: smem[0 ... T*T-1]
float* tile_q = smem;
// K tile: smem[T*T ... 2*T*T-1]
float* tile_k = smem + T*T;
// V tile: smem[2*T*T ... 3*T*T-1]
float* tile_v = smem + 2*T*T;
```

```
// holds 3 tiles of T×T floats for Q, K, and V
extern __shared__ float smem[];
// Q tile: smem[0 ... T*T-1]
float* tile_q = smem;
// K tile: smem[T*T ... 2*T*T-1]
float* tile_k = smem + T*T;
// V tile: smem[2*T*T ... 3*T*T-1]
float* tile_v = smem + 2*T*T;
```

By varying shared_mem_bytes per launch, the same kernel binary can run with different tile sizes. After selecting T, you can query occupancy using the CUDA Occupancy API to see how many blocks fit per SM.

通过在每次启动时改变 shared_mem_bytes，同一个核函数二进制就能以不同的分块大小运行。选定 T 后，你可以用 CUDA Occupancy API 查询占用率，看每个 SM 能容纳多少个块。

If occupancy is too low and only one block is allocated per SM, you can reduce T. If you’re thrashing DRAM, you can increase T. This can be implemented as an automatic feedback loop in which the kernel programmatically measures its own achieved occupancy using the CUDA Occupancy API or NVIDIA’s Data Center GPU Manager (DCGM)—and adjusts T on subsequent iterations. This way each attention layer uses the optimal configuration based on the current sequence length, L, and the hardware limits.

如果占用率过低、每个 SM 只分配到一个块，你可以减小 T。如果你在频繁抖动 DRAM，可以增大 T。这可以实现为一个自动反馈循环：核函数使用 CUDA Occupancy API 或 NVIDIA 的 Data Center GPU Manager（DCGM）以编程方式测量自身达成的占用率——并在后续迭代中调整 T。这样每个注意力层都能基于当前序列长度 L 和硬件限制使用最优配置。

As we saw in Chapter 6, you also need to consider register usage per thread when optimizing SM occupancy. Using more registers (e.g., unrolling loops) can speed up single-thread performance, but it can reduce overall SM occupancy since each SM has a limited register file.

正如我们在第 6 章所见，优化 SM 占用率时你还需考虑每个线程的寄存器用量。使用更多寄存器（例如展开循环）能提升单线程性能，但由于每个 SM 的寄存器文件有限，这会降低整体 SM 占用率。

Fewer warps can be scheduled if each warp uses many registers. A dynamic runtime can detect if a kernel is hitting occupancy limits due to registers and switch to a version that uses fewer registers—at the expense of extra instructions. These low-level considerations are critical for adaptive, high-performance inference servers.

如果每个 warp 使用很多寄存器，能调度的 warp 数量就会减少。动态运行时可以检测某个核函数是否因寄存器而触及占用率上限，并切换到一个使用更少寄存器的版本——代价是额外的指令。这些底层考量对自适应、高性能的推理服务器至关重要。

Dynamic shared-memory tuning requires profiling occupancy versus throughput. Tools such as NVIDIA Nsight Systems/Compute and the CUDA Occupancy API can show the achieved occupancy and execution efficiency of each kernel. Meanwhile, DCGM provides real-time GPU utilization and SM occupancy metrics at the system level. An adaptive system can use this information to notice that an attention kernel with sequence length 2,048 achieves only 30% occupancy, for example, because each thread block uses a large amount of shared memory.

动态共享内存调优需要对占用率与吞吐量进行剖析。诸如 NVIDIA Nsight Systems/Compute 和 CUDA Occupancy API 等工具可以展示每个核函数达成的占用率与执行效率。与此同时，DCGM 在系统层面提供实时的 GPU 利用率和 SM 占用率指标。自适应系统可以利用这些信息注意到，例如序列长度为 2,048 的注意力核函数只达成了 30% 的占用率，原因是每个线程块使用了大量共享内存。

In this case, the system could dynamically switch to a kernel configuration that reduces shared memory per thread block by splitting the attention computation across two passes, for instance. This would increase occupancy—and potentially increase throughput—if memory latency was the bottleneck.

这种情况下，系统可以动态切换到某种核函数配置，例如把注意力计算拆分为两趟，从而减少每个线程块的共享内存。如果内存延迟是瓶颈，这将提升占用率——并有可能提升吞吐量。

Conversely, if a kernel is memory bound and not saturating ALUs, using more shared memory—even if occupancy drops—can improve effective throughput by reducing memory stalls. It’s important to understand these trade-offs—especially with occupancy since it’s less intuitive, in some cases, than other metrics.

反过来，如果某个核函数是内存受限且未能充分利用 ALU，那么使用更多共享内存——即便占用率下降——也能通过减少内存停顿来提升有效吞吐量。理解这些权衡很重要——尤其是占用率，因为在某些情况下它不如其他指标那样直观。

It’s recommended to design kernels that allow tunable shared memory and thread block sizes at runtime. The system can then adapt the tuning configuration to runtime conditions based on input and hardware feedback. For example, it can provide runtime parameters and template parameters for tile sizes used by libraries like CUTLASS, which provide runtime-tunable kernel variants for exactly this reason.

建议将核函数设计为在运行时允许可调的共享内存和线程块大小。这样系统就能根据输入和硬件反馈，使调优配置适应运行时条件。例如，它可以为 CUTLASS 之类的库所用的分块大小提供运行时参数和模板参数——这些库正是出于这个原因提供了运行时可调的核函数变体。

You should also continuously monitor SM utilization metrics. Consider many idle warps (e.g., < 50% active warps) or memory stall cycles (> 70% stalled). This indicates an imbalance, as either occupancy is too low (idle warps)—or your tile size is too small and causes excessive memory traffic. As such, your system should adjust accordingly to restore the balance.

你还应持续监控 SM 利用率指标。设想出现大量空闲 warp（例如活跃 warp < 50%）或内存停顿周期（> 70% 停顿）。这表明存在不均衡：要么占用率过低（空闲 warp），要么你的分块大小过小、导致过多的内存流量。因此，你的系统应相应调整以恢复平衡。

For inference serving, it’s common to maintain a small table of optimal thread block configurations for different problem sizes. This mapping can be implemented as a JSON or config file that maps sequence length ranges to launch parameters. This allows easy updates as models and hardware evolve.

对于推理服务，常见做法是为不同问题规模维护一张最优线程块配置的小表。这个映射可以实现为一个 JSON 或配置文件，将序列长度区间映射到启动参数。这使得随着模型和硬件演进能够方便地更新。

For instance, whenever your system performs attention with sequence length 512, it will use 128 threads/block and 16 KB shared memory. Or, for sequence length 4,096, it will use 256 threads/block and 64 KB shared memory, etc. This extends the concept of autotuning to resource allocation.

例如，每当系统在序列长度 512 下执行注意力时，它会使用 128 threads/block 和 16 KB 共享内存。而对于序列长度 4,096，它会使用 256 threads/block 和 64 KB 共享内存，等等。这把自动调优的概念延伸到了资源分配。

Remember that modern NVIDIA GPUs provide a unified on-chip pool for L1 data cache and shared memory. And the carveout controls how much of that pool is reserved for shared memory versus L1. Adjust the carveout with cudaFuncSetAttribute to increase the fraction available to shared memory when kernels demand larger tiles.

请记住，现代 NVIDIA GPU 为 L1 数据缓存和共享内存提供了一个统一的片上池。而 carveout 控制该池中有多少预留给共享内存、多少给 L1。当核函数需要更大的分块时，用 cudaFuncSetAttribute 调整 carveout，以增大可供共享内存使用的比例。

Modern NVIDIA GPUs provide a unified on-chip pool for L1 data cache and shared memory. NVIDIA’s device driver allows you to set the L1 cache versus shared-memory split percentage, or “carveout” percentage. As such, you can configure an SM to prefer more shared memory or more L1 cache depending on the use case. For instance, you can increase the fraction available to shared memory when kernels demand larger tiles.

现代 NVIDIA GPU 为 L1 数据缓存和共享内存提供了一个统一的片上池。NVIDIA 的设备驱动允许你设置 L1 缓存与共享内存的划分百分比，即“carveout”百分比。因此，你可以根据使用场景，将某个 SM 配置为偏向更多共享内存或更多 L1 缓存。例如，当核函数需要更大的分块时，你可以增大可供共享内存使用的比例。

> The carveout is a per-kernel attribute and only a hint rather than a guarantee. It’s another knob you can tune to balance occupancy and caching behavior.

> carveout 是逐核函数（per-kernel）的属性，且仅是一个提示而非保证。它是你可以调节的又一个旋钮，用于平衡占用率与缓存行为。

A sophisticated runtime can toggle this carve-out percentage at launch time using cudaFuncSetAttribute() with cudaFuncAttributePreferredSharedMemoryCarveout or specific kernels. For instance, if an attention kernel uses very large tiles and needs more shared memory, you might want to reduce the L1 to 25% and increase shared memory to 75% (assuming the carveout value starts at 50%).

一个成熟的运行时可以在启动时，使用 cudaFuncSetAttribute() 配合 cudaFuncAttributePreferredSharedMemoryCarveout，为特定核函数切换这个 carve-out 百分比。例如，如果某个注意力核函数使用了非常大的分块、需要更多共享内存，你可能希望把 L1 降到 25%、把共享内存提到 75%（假设 carveout 值从 50% 起）。

> The shared-memory versus L1 carveout attribute is a hint rather than a guarantee. Always treat the setting as a hint and verify the effect with profiling. Check that the requested setting actually impacted occupancy and cache behavior.

> 共享内存与 L1 的 carveout 属性只是提示而非保证。始终把该设置当作提示，并用剖析验证其效果。检查所请求的设置是否真的影响了占用率与缓存行为。

In short, dynamic shared memory and occupancy-aware techniques ensure that every SM is kept as busy as possible for the given task. These techniques adapt the kernel’s resource usage to the specific use case. This is essential for large models in which some layers or batch sizes could otherwise underutilize the SMs.

简而言之，动态共享内存与占用率感知技术确保每个 SM 都为给定任务尽可能保持繁忙。这些技术让核函数的资源使用适配于具体使用场景。对于大模型而言这至关重要，否则某些层或批量大小可能会使 SM 利用不足。

## Speculative KV Prefetching for Faster TTFT

## 用推测式 KV 预取加速 TTFT

When serving LLMs in a real-time setting, time-to-first-token (TTFT) is a critical metric, as it measures how quickly the system can produce the first token of the model’s response. This directly affects the end user’s experience.

在实时场景中服务 LLM 时，首 token 时延（time-to-first-token，TTFT）是一个关键指标，因为它衡量系统产出模型响应第一个 token 的速度。这直接影响最终用户的体验。

One major contributor to TTFT in large models is the time spent setting up the model’s internal states, such as the key-value (KV) cache, before token generation can begin. Remember from earlier chapters that the attention KV cache stores the past tokens’ key and value projections for each layer.

在大模型中，TTFT 的一个主要来源是在 token 生成开始之前用于建立模型内部状态（如键值（key-value，KV）缓存）所花的时间。回想前几章的内容，注意力 KV 缓存为每一层存储过去 token 的 key 与 value 投影。

Speculative KV prefetching is an optimization in which the system anticipates the data needed for the first token—and loads the necessary data into the GPU in advance. This effectively overlaps KV cache preparation with other steps, such as compute. This way, the token generation can start more quickly. An example of speculative KV caching is SpeCache, as shown in Figure 19-3.

推测式 KV 预取（speculative KV prefetching）是一种优化：系统预判第一个 token 所需的数据——并提前将必要数据加载进 GPU。这实际上让 KV 缓存的准备与其他步骤（如计算）重叠。这样 token 生成就能更快开始。推测式 KV 缓存的一个例子是 SpeCache，如图 19-3 所示。

With SpeCache, the KV cache is compressed (16-bit, in this case) and moved off-GPU one layer at a time. This reduces the memory footprint. After generating the first output token, a speculative “next” token is computed. At the same time, the model prefetches the corresponding reduced-precision KV pairs needed for that first decoding step.

在 SpeCache 中，KV 缓存被压缩（此处为 16-bit）并逐层移出 GPU。这减少了内存占用。生成第一个输出 token 后，会计算一个推测的“下一个”token。与此同时，模型会预取该首个解码步骤所需的相应降精度 KV 对。

On each subsequent step, the model decodes two tokens in parallel, including the actual output token and the speculative token. Both results are fed into the next step, and, before each step, the top-k most relevant 16-bit KV pairs for the speculative path are prefetched. This way, both paths have their required KV cache data ready. In short, SpeCache reports TTFT improvements by prefetching reduced-precision KV and overlapping with compute.

在随后的每一步，模型并行解码两个 token，包括实际输出 token 和推测 token。两者的结果都会送入下一步；并且在每一步之前，都会预取推测路径最相关的 top-k 个 16-bit KV 对。这样，两条路径都备好了各自所需的 KV 缓存数据。简而言之，SpeCache 报告称，通过预取降精度 KV 并与计算重叠，可改善 TTFT。

> Integrate speculative prefetch techniques only after validating your access patterns and storage tiers.

> 只有在验证了你的访问模式与存储层级之后，才集成推测式预取技术。

![Figure 19-3. Speculative decoding with SpeCache (source: https://oreil.ly/b21E5)](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-3.png)

![图 19-3. 使用 SpeCache 的推测解码（来源：https://oreil.ly/b21E5）](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-3.png)

The KV cache can be extremely large due to the number of layers in modern LLMs, the increasing size of the LLM context window (effectively limitless at this point), and the large amount of reasoning chains generated by modern “thinking” models. Modern inference systems will often swap the KV cache between GPU, CPU memory, and SSD to better manage capacity—especially for extremely long contexts, which don’t fit in GPU memory.

由于现代 LLM 层数众多、LLM 上下文窗口不断增大（如今几乎无上限），以及现代“思考型”模型生成的大量推理链，KV 缓存可能极其庞大。现代推理系统常常在 GPU、CPU 内存和 SSD 之间换入换出 KV 缓存，以更好地管理容量——尤其是对于放不进 GPU 内存的超长上下文。

When a new token is being generated, a naive approach would be to fetch the KV data from wherever it resides (some on GPU, some on CPU) in synchronous fashion—and then perform the computation to decode the next token. This can add significant latency for the first token—especially if the cache has been paged out to CPU memory or NVMe storage.

生成新 token 时，一种朴素做法是以同步方式从 KV 数据所在之处（一部分在 GPU、一部分在 CPU）取回数据——然后执行计算以解码下一个 token。这会为第一个 token 增加显著延迟——尤其是当缓存已被换出到 CPU 内存或 NVMe 存储时。

KV cache prefetching helps by starting the KV data transfers ahead of time. As soon as the user’s prompt is received, the server can start copying the necessary KV pages—as well as the model weights—directly into GPU memory. By the time the model finishes computing the prompt in the prefill phase, the necessary data is in place to generate the first output token.

KV 缓存预取（prefetch）通过提前启动 KV 数据传输来提供帮助。一收到用户的提示，服务器就可以开始把必要的 KV 页——以及模型权重——直接拷贝进 GPU 内存。等到模型在 prefill 阶段计算完提示时，生成第一个输出 token 所需的数据已经就位。

Specifically, this mechanism keeps only the current layer’s KV in GPU memory—and offloads the other layers’ KV to the CPU. It asynchronously prefetches the next layer’s cache into the GPU while the current layer is being computed. Additionally, it simultaneously writes the previous layer’s cache back to the CPU.

具体来说，这种机制只在 GPU 内存中保留当前层的 KV——并把其他层的 KV 卸载到 CPU。它在计算当前层的同时，异步地将下一层的缓存预取进 GPU。此外，它还同时把上一层的缓存写回 CPU。

This overlap of communication and computation means the GPU rarely waits for data. The result is that using an offloaded KV cache in CPU memory has minimal impact on latency. For example, you might see ~5%–10% lower tokens/s throughput when offloading due to the extra data-transfer overhead.

这种通信与计算的重叠意味着 GPU 很少需要等待数据。其结果是，在 CPU 内存中使用卸载的 KV 缓存对延迟的影响极小。例如，由于额外的数据传输开销，卸载时你可能会看到 tokens/s 吞吐量降低约 5%–10%。

> Overlap can mask much of the latency of CPU-resident KV, but a throughput penalty typically remains due to CPU DRAM bandwidth and PCIe overhead. Profile with your batch and sequence lengths repeatedly.

> 重叠可以掩盖 CPU 驻留 KV 的大部分延迟，但由于 CPU DRAM 带宽和 PCIe 开销，通常仍会留有吞吐量代价。请用你自己的批量和序列长度反复剖析。

An example of KV cache offloading is in Hugging Face’s Transformers library in the form of the OffloadedCache mechanism. This can be enabled when calling generate(cache_implementation="offloaded") or generate(cache_implementation="offloaded_static"). This will generate tokens with the Transformer library, as shown here. This makes it a low-effort, high-impact optimization:

KV 缓存卸载的一个例子是 Hugging Face 的 Transformers 库中的 OffloadedCache 机制。它可以在调用 generate(cache_implementation="offloaded") 或 generate(cache_implementation="offloaded_static") 时启用。这会用 Transformer 库生成 token，如下所示。这使它成为一项低投入、高收益的优化：

```
# Dynamic, variable-length serving and sliding layers
# (recommended default)
out = model.generate(..., cache_implementation="offloaded")
# Static shapes + torch.compile and CUDA Graphs
# (highest throughput with fixed shapes, use with torch.compile)
# out = model.generate(..., cache_implementation="offloaded_static")
```

```
# Dynamic, variable-length serving and sliding layers
# (recommended default)
out = model.generate(..., cache_implementation="offloaded")
# Static shapes + torch.compile and CUDA Graphs
# (highest throughput with fixed shapes, use with torch.compile)
# out = model.generate(..., cache_implementation="offloaded_static")
```

Under the hood, when generation begins, the OffloadedCache will ensure that layer 1’s KV is moved to the GPU. While layer 1 computes, OffloadedCache issues an asynchronous DMA for layer 2’s KV from the CPU to the GPU, etc. It’s always prefetching one layer ahead.

在底层，当生成开始时，OffloadedCache 会确保第 1 层的 KV 被移动到 GPU。在第 1 层计算的同时，OffloadedCache 会为第 2 层的 KV 从 CPU 到 GPU 发起一次异步 DMA，依此类推。它始终提前预取一层。

By the time the forward pass reaches layer 2, its KV is already local. This reduces the stall that would occur if we used a synchronous copy for each layer. Now that we have described KV prefetching, let’s move to speculative KV prefetching.

等到前向传播到达第 2 层时，其 KV 已经在本地。这减少了如果我们对每一层都使用同步拷贝时会发生的停顿。既然我们已经描述了 KV 预取，接下来就转向推测式 KV 预取。

Speculative KV prefetching extends beyond just the one-layer lookahead of regular KV prefetching. Imagine an inference server configuration with multiple model replicas—or multiple possible paths like MoE models in which a token can be routed to one of several expert networks.

推测式 KV 预取的范围超出了常规 KV 预取仅提前一层的预判。设想这样一种推理服务器配置：拥有多个模型副本——或者存在多条可能路径，如 MoE 模型中一个 token 可被路由到若干专家网络之一。

KV prefetching helps at the boundary between the phases. By the end of the prefill phase, ideally, the caches for all layers are either already in GPU memory or queued to come into GPU memory. This directly minimizes TTFT since, once generation starts, the model isn’t waiting on memory transfers.

KV 预取在各阶段之间的边界处提供帮助。理想情况下，到 prefill 阶段结束时，所有层的缓存要么已在 GPU 内存中，要么已排队进入 GPU 内存。这直接把 TTFT 降到最低，因为一旦生成开始，模型就不必等待内存传输。

> It’s recommended to continuously monitor your TTFT using tracing tools like NVTX markers to measure the first token’s decode time. This will measure TTFT precisely. If you see excessive spikes of idle time immediately after the decode phase begins, this indicates a missed prefetch opportunity.

> 建议使用 NVTX markers 之类的追踪工具持续监控你的 TTFT，以测量第一个 token 的解码时间。这将精确测量 TTFT。如果你在解码阶段刚开始时就看到过多的空闲时间尖峰，这说明错失了一次预取机会。

To implement KV prefetching in your own stack without stalling inference, you can use CUDA streams for overlap (as described in Chapter 11). This way, it runs concurrently with your main computation stream. You would then use CUDA events to synchronize the streams only when the prefetched data is needed, as shown here:

要在你自己的技术栈中实现 KV 预取而不拖慢推理，你可以使用 CUDA 流来实现重叠（如第 11 章所述）。这样它就能与你的主计算流并发运行。随后你只在需要预取数据时才使用 CUDA 事件来同步这些流，如下所示：

```
// kv_prefetch_overlap.cu
#include <cstdio>
#include <cuda_runtime.h>
// Example sizes
static constexpr size_t KV_BYTES =
  /* set to your chunk size */ 8ull<<20; // 8 MiB
__global__ void forward_kernel(/* ... */) {
  // compute logits for current token ...
}
__global__ void consume_prefetched_kv(/* use prefetch_dest */) {
  // consumes KV in prefetch_dest ...
}
int main() {
  // Allocate destination buffer on this GPU
  void* prefetch_dest = nullptr;
  cudaMalloc(&prefetch_dest, KV_BYTES);
  // Example: staging source on host. MUST be pinned for real overlap.
  void* kv_src_host = nullptr;
  cudaMallocHost(&kv_src_host, KV_BYTES);  // pinned (page-locked)
  // Fill kv_src_host with data for the first iteration...
  cudaStream_t compute_stream, prefetch_stream;
  cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&prefetch_stream, cudaStreamNonBlocking);
  cudaEvent_t kv_ready;
  cudaEventCreateWithFlags(&kv_ready, cudaEventDisableTiming);
  bool done = false;
  while (!done) {
    // 1) Launch compute for current token
    forward_kernel<<< /*grid*/1, /*block*/1, 0, compute_stream>>>();
    // 2) Asynchronously prefetch next KV chunk
    // If your source is another GPU, use cudaMemcpyPeerAsync
    // and enable peer access.
    cudaMemcpyAsync(prefetch_dest, kv_src_host, KV_BYTES,
                               cudaMemcpyHostToDevice, prefetch_stream);
    cudaEventRecord(kv_ready, prefetch_stream);
    // 3) Ensure consumer on compute_stream waits just-in-time
    cudaStreamWaitEvent(compute_stream, kv_ready, /*flags*/0);
    // 4) Launch work that consumes the prefetched KV
    consume_prefetched_kv<<< /*grid*/1, /*block*/1, 0, compute_stream>>>();
    // 5) ...advance state, update kv_src_host for next iteration, set `done`
    done = true; // demo
  }
  cudaEventDestroy(kv_ready);
  cudaStreamDestroy(prefetch_stream);
  cudaStreamDestroy(compute_stream);
  cudaFree(prefetch_dest);
  cudaFreeHost(kv_src_host);
  return 0;
}
```

```
// kv_prefetch_overlap.cu
#include <cstdio>
#include <cuda_runtime.h>
// Example sizes
static constexpr size_t KV_BYTES =
  /* set to your chunk size */ 8ull<<20; // 8 MiB
__global__ void forward_kernel(/* ... */) {
  // compute logits for current token ...
}
__global__ void consume_prefetched_kv(/* use prefetch_dest */) {
  // consumes KV in prefetch_dest ...
}
int main() {
  // Allocate destination buffer on this GPU
  void* prefetch_dest = nullptr;
  cudaMalloc(&prefetch_dest, KV_BYTES);
  // Example: staging source on host. MUST be pinned for real overlap.
  void* kv_src_host = nullptr;
  cudaMallocHost(&kv_src_host, KV_BYTES);  // pinned (page-locked)
  // Fill kv_src_host with data for the first iteration...
  cudaStream_t compute_stream, prefetch_stream;
  cudaStreamCreateWithFlags(&compute_stream, cudaStreamNonBlocking);
  cudaStreamCreateWithFlags(&prefetch_stream, cudaStreamNonBlocking);
  cudaEvent_t kv_ready;
  cudaEventCreateWithFlags(&kv_ready, cudaEventDisableTiming);
  bool done = false;
  while (!done) {
    // 1) Launch compute for current token
    forward_kernel<<< /*grid*/1, /*block*/1, 0, compute_stream>>>();
    // 2) Asynchronously prefetch next KV chunk
    // If your source is another GPU, use cudaMemcpyPeerAsync
    // and enable peer access.
    cudaMemcpyAsync(prefetch_dest, kv_src_host, KV_BYTES,
                               cudaMemcpyHostToDevice, prefetch_stream);
    cudaEventRecord(kv_ready, prefetch_stream);
    // 3) Ensure consumer on compute_stream waits just-in-time
    cudaStreamWaitEvent(compute_stream, kv_ready, /*flags*/0);
    // 4) Launch work that consumes the prefetched KV
    consume_prefetched_kv<<< /*grid*/1, /*block*/1, 0, compute_stream>>>();
    // 5) ...advance state, update kv_src_host for next iteration, set `done`
    done = true; // demo
  }
  cudaEventDestroy(kv_ready);
  cudaStreamDestroy(prefetch_stream);
  cudaStreamDestroy(compute_stream);
  cudaFree(prefetch_dest);
  cudaFreeHost(kv_src_host);
  return 0;
}
```

In this setup, cudaMemcpyAsync runs on prefetch_stream while model.forward() uses the compute_stream. This allows the CUDA driver to overlap data transfer with compute. You synchronize only when the prefetched KV data is actually needed by waiting on kv_ready event before continuing with the computation that consumes it. The event enforces just-in-time synchronization at the handoff point.

在这种设置中，cudaMemcpyAsync 在 prefetch_stream 上运行，而 model.forward() 使用 compute_stream。这让 CUDA 驱动能够将数据传输与计算重叠。你只在实际需要预取的 KV 数据时才同步——即在继续执行消费它的计算之前，等待 kv_ready 事件。该事件在交接点强制实施即时（just-in-time）同步。

> Make sure the host buffers are pinned (page-locked). Otherwise, cudaMemcpyAsync may serialize and you won’t get the desired copy/compute overlap. If the KV source is on another GPU, use cudaMemcpyPeerAsync and enable peer access. And if you are using Unified Memory (e.g., Grace Blackwell, Vera Rubin superchips), consider using cudaMemPrefetchAsync to stage pages ahead of time. You can also use CUDA Graphs to capture this sequence if the pattern is repeatable. This can further reduce kernel-launch overhead when prefetching happens frequently.

> 确保主机缓冲区是固定的（页锁定，page-locked）。否则 cudaMemcpyAsync 可能会串行化，你将得不到期望的拷贝/计算重叠。如果 KV 源在另一块 GPU 上，请使用 cudaMemcpyPeerAsync 并启用对等访问（peer access）。而如果你使用统一内存（Unified Memory，例如 Grace Blackwell、Vera Rubin 超级芯片），可考虑使用 cudaMemPrefetchAsync 提前暂存页面。如果这一模式是可重复的，你也可以用 CUDA Graphs 来捕获这个序列。当预取频繁发生时，这能进一步降低核函数启动开销。

Using a separate stream ensures efficient pipelining. As one token is being generated, the next token’s KV cache is being prefetched without interrupting the compute stream. This maximizes GPU utilization by masking transfer latency and keeping the compute units continuously fed with data. Modern LLM inference engines use this automatically in the form of paged KV caching.

使用单独的流可确保高效的流水线化。在生成一个 token 的同时，下一个 token 的 KV 缓存正在被预取，而不打断计算流。这通过掩盖传输延迟、并让计算单元持续获得数据供给，从而最大化 GPU 利用率。现代 LLM 推理引擎会以分页 KV 缓存的形式自动使用这一点。

It’s best to consider weight and KV cache data movement as part of the overall inference pipeline. Just as you should pipeline compute operations, you should also pipeline data movement. Always have the next needed data in flight while the current computation is ongoing. KV cache compression is yet another option to improve performance at the KV cache layer. Let’s cover this next.

最好把权重和 KV 缓存的数据搬运视为整个推理流水线的一部分。正如你应当对计算操作做流水线化，你也应当对数据搬运做流水线化。在当前计算进行的同时，始终让下一份所需数据处于传输途中。KV 缓存压缩是又一个在 KV 缓存层提升性能的选项。接下来我们就讲这个。

## Real-Time KV Cache Compression and Policy Switching

## 实时 KV 缓存压缩与策略切换

As an LLM generates more and more tokens in a session, the KV cache grows linearly. For long conversations, documents, and reasoning chains, the KV cache consumes a huge amount of GPU memory and often utilizes the most GPU memory.

随着 LLM 在一次会话中生成越来越多的 token，KV 缓存会线性增长。对于长对话、长文档和长推理链，KV 缓存会消耗大量 GPU 内存，而且往往是占用 GPU 内存最多的部分。

KV cache is a good candidate for compression/quantization. Like any form of compression, KV cache compression reduces its memory footprint. Doing this in real time means performing compression on the fly during inference.

KV 缓存是压缩/量化的良好候选对象。与任何形式的压缩一样，KV 缓存压缩会减少其内存占用。实时地做这件事意味着在推理过程中即时执行压缩。

Policy switching means that the compression strategy can change based on the current context. The goal is to free up memory and network bandwidth when needed—without impacting model accuracy or slowing down computations that involve the KV cache data. Figure 19-4 shows a few different types of KV cache compression algorithms.

策略切换（policy switching）意味着压缩策略可以根据当前上下文而改变。目标是在需要时释放内存和网络带宽——同时不影响模型精度，也不拖慢涉及 KV 缓存数据的计算。图 19-4 展示了几种不同类型的 KV 缓存压缩算法。

![Figure 19-4. Different KV cache algorithms, including no caching (e.g., dense)](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-4.png)

![图 19-4. 不同的 KV 缓存算法，包括不缓存（例如稠密）](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-4.png)

A straightforward and simple approach to KV compression is to just reduce its precision. Many frameworks default to FP16 or BF16 for KV cache since 16-bit is typically what the model uses for activations. However, one can often compress keys and values to 8-bit or even 4-bit with minimal impact on output quality—especially for tokens at the end of the LLM’s context.

KV 压缩最直接、最简单的方法就是降低其精度。许多框架的 KV 缓存默认使用 FP16 或 BF16，因为 16-bit 通常正是模型用于激活的精度。然而，人们往往可以把 key 和 value 压缩到 8-bit 甚至 4-bit，而对输出质量影响极小——尤其是对于位于 LLM 上下文末尾的 token。

Hugging Face’s Transformers library supports a QuantizedCache, including INT8 and INT4 for KV memory. This feature can be enabled in one line by specifying cache_implementation="quantized" with a specific bit-width. The result is a massive memory savings at the cost of a tiny amount of extra compute for the quantization/dequantization operations. And the overall model quality does not suffer in most cases.

Hugging Face 的 Transformers 库支持一种 QuantizedCache，为 KV 内存提供包括 INT8 和 INT4 在内的支持。这个特性只需一行即可启用——指定 cache_implementation="quantized" 并给出具体位宽。其结果是内存大幅节省，代价仅是量化/反量化操作的少量额外计算。而且在大多数情况下整体模型质量不受损害。

> When quantization plus CPU offload is used concurrently, ensure host buffers are pinned (page-locked) to prevent serialized transfers. This will help to sustain copy bandwidth (e.g., PCIe/NVLink).

> 当量化与 CPU 卸载同时使用时，确保主机缓冲区是固定的（页锁定）以防止串行化传输。这有助于维持拷贝带宽（例如 PCIe/NVLink）。

Next, let’s discuss dynamic policy switching. An example of a policy is keeping the last 128 tokens in full precision but compressing the rest of the tokens in 4-bit. This way, the most recent context—which likely has the most impact on predicting the next token—is preserved with higher precision, whereas the older history is stored in lower precision to save storage.

接下来，我们讨论动态策略切换。一个策略的例子是：把最后 128 个 token 保持为全精度，而把其余 token 压缩为 4-bit。这样，最近的上下文——很可能对预测下一个 token 影响最大——以更高精度保留，而较早的历史则以更低精度存储以节省空间。

If the model suddenly needs to attend to older tokens, it’s usually not disastrous since many LLMs have recency bias anyway. This means that they prioritize recent context. This way, earlier parts of the input sequence may not affect the final output as much.

如果模型突然需要关注较早的 token，通常也不会是灾难性的，因为许多 LLM 本就带有近因偏好（recency bias）。这意味着它们优先考虑最近的上下文。这样，输入序列较早的部分对最终输出的影响可能没那么大。

You might further adapt this window based on user prompt length. For example, you can use a larger full-precision window for very long prompts—or compress more aggressively if GPU memory usage is above a threshold.

你还可以根据用户提示长度进一步调整这个窗口。例如，对于非常长的提示，你可以使用更大的全精度窗口——或者在 GPU 内存使用超过某个阈值时更激进地压缩。

Alternatively, a policy could be based on memory usage. For instance, the policy could dictate that if GPU memory usage exceeds 80%, it should compress the entire KV cache into 8-bit. This helps avoid OOM errors during long generations. The policy might include multitier compression in which the system compresses the KV cache to 8-bit under mild pressure, then changes to 4-bit compression under extreme pressure.

或者，策略也可以基于内存使用量。例如，策略可以规定：如果 GPU 内存使用超过 80%，就应把整个 KV 缓存压缩为 8-bit。这有助于在长时间生成期间避免 OOM 错误。该策略可以包含多级压缩：系统在轻度压力下把 KV 缓存压缩到 8-bit，然后在极端压力下改为 4-bit 压缩。

With true dynamic, real-time policy switching, the engine can change to a different compression during token generation. In this case, the implementation would need to maintain multiple representations of the cache simultaneously. For instance, it would initially store KV in FP16 but concurrently maintain an INT8 version of the same data.

借助真正的动态、实时策略切换，引擎可以在 token 生成过程中改用不同的压缩。这种情况下，实现需要同时维护缓存的多种表示。例如，它一开始以 FP16 存储 KV，但同时维护同一数据的 INT8 版本。

The system would use FP16 by default, but if memory utilization crosses a certain threshold, it could start using the INT8 version—with appropriate scaling factors—and free the FP16 memory to relieve memory pressure. Future attention reads would then retrieve dequantized values from INT8 storage.

系统默认使用 FP16，但如果内存利用率越过某个阈值，它可以开始使用 INT8 版本——配以适当的缩放因子——并释放 FP16 内存以缓解内存压力。之后的注意力读取便会从 INT8 存储中取回反量化后的值。

This requires careful synchronization, however, to ensure the compressed version is kept up-to-date and ready by the time it’s needed. Techniques like double buffering and background compression threads are useful in this case.

不过，这需要仔细的同步，以确保压缩版本保持最新、并在需要时已就绪。双缓冲（double buffering）和后台压缩线程之类的技术在这种情况下很有用。

Often a CPU thread can handle the compression asynchronously using vectorized INT8 quantization operations. It can then copy the compressed block to GPU memory when ready.

通常可以由一个 CPU 线程使用向量化的 INT8 量化操作异步处理压缩。然后在就绪时把压缩后的块拷贝到 GPU 内存。

> Implement real-time policy swaps at safe points, such as the end of an iteration. This way, you can avoid mid-calculation switches and hide the requantization latency by doing it in a background stream.

> 在安全点（例如一次迭代的结尾）实施实时策略切换。这样，你就能避免计算中途切换，并通过在后台流中执行来隐藏重新量化的延迟。

There are other techniques, such as lossless compression, that use entropy coding and clustering to compress activations without losing information bit-for-bit. However, these implementations are complex and may be too slow to do in real time—even on a GPU.

还有一些其他技术，例如无损压缩，使用熵编码和聚类来压缩激活，且不逐比特地丢失信息。然而，这些实现很复杂，且可能太慢而无法实时完成——即便在 GPU 上也是如此。

Simpler mechanisms like chunk-wise ZFP, a type of floating-point compression, or even generic CPU-based compression should be considered. However, the simplest, most-effective, and well-supported method so far has been quantization.

也应考虑一些更简单的机制，如分块（chunk-wise）ZFP（一种浮点压缩），甚至通用的基于 CPU 的压缩。然而，迄今为止最简单、最有效且支持最完善的方法一直是量化。

> As of this writing, lossless methods like ZFP are evaluated in offline and research contexts but remain uncommon in production LLM KV cache paths due to throughput constraints relative to quantized cache. As such, quantization remains the go-to approach for its balance of speed and 2–4× memory reduction.

> 在撰写本文时，像 ZFP 这样的无损方法在离线和研究场景中已被评估，但由于相对于量化缓存存在吞吐量约束，在生产级 LLM KV 缓存路径中仍不常见。因此，量化因其在速度与 2–4× 内存缩减之间的平衡，仍是首选方案。

For minimal quality impact, you can experiment with per-head and per-token scaling. Quantizing the KV cache is most effective with per-head, group-wise scaling rather than per-token scaling. The Hugging Face QuantizedCache Transformer implementation calibrates the range of values per attention head.

为把质量影响降到最低，你可以尝试逐头（per-head）和逐 token（per-token）的缩放。量化 KV 缓存时，采用逐头、分组（group-wise）缩放比逐 token 缩放更有效。Hugging Face 的 QuantizedCache Transformer 实现会按每个注意力头校准数值范围。

Specifically, QuantizedCache implements per-channel, group-wise quantization with a configurable group size and a residual window that keeps the most recent tokens in the original precision. You enable it by setting cache_implementation="quantized" and passing cache_config as a dictionary. You can compute the max absolute value in a tensor and scale the 4-bit or 8-bit quantization accordingly. This is essentially a form of min-max, or magnitude-based quantization.

具体来说，QuantizedCache 实现的是逐通道（per-channel）、分组量化，具有可配置的组大小，以及一个把最近 token 保持在原始精度的残差窗口。你通过设置 cache_implementation="quantized" 并把 cache_config 作为字典传入来启用它。你可以计算张量中的最大绝对值，并据此缩放 4-bit 或 8-bit 量化。这本质上是一种 min-max（即基于幅度）的量化。

A useful implementation of QuantizedCache is the Half-Quadratic Quantization (HQQ) backend. HQQ provides a calibration-free, on-the-fly quantizer that supports a wide range of low-bit formats, including 2-bit, 3-bit, 4-bit, and 8-bit. It uses a robust optimization to model outliers and heavy-tailed error distributions. And HQQ integrates well into the Hugging Face Transformers’ KV cache implementation. It provides both a PyTorch and custom CUDA kernel implementation for fast inference.

QuantizedCache 的一个有用实现是 Half-Quadratic Quantization（HQQ）后端。HQQ 提供了一个免校准、即时的量化器，支持广泛的低位格式，包括 2-bit、3-bit、4-bit 和 8-bit。它使用一种鲁棒优化来对离群值和重尾误差分布建模。而且 HQQ 能很好地集成进 Hugging Face Transformers 的 KV 缓存实现。它同时提供 PyTorch 实现和自定义 CUDA 核函数实现，以实现快速推理。

We can implement a dynamic policy that can switch between an 8-bit quantization and 4-bit quantization depending on memory pressure. The sharpness or distribution of values might also guide the decision. If the cached values are mostly small and have low variance, they can usually be quantized more aggressively. Switching the compression policy in real time can be integrated with the Hugging Face Quantized Cache mechanism.

我们可以实现一个动态策略，根据内存压力在 8-bit 量化和 4-bit 量化之间切换。数值的尖锐程度或分布也可以指导这一决策。如果缓存的数值大多较小且方差较低，通常就可以更激进地量化。实时切换压缩策略可以与 Hugging Face 的 QuantizedCache 机制集成。

Unfortunately, Transformers does not support changing the bit-width of an already-initialized cache object in place. However, to implement a dynamic policy, our code can generate tokens in small chunks and, on memory pressure, start the next chunk with a new quantized cache configuration on the fly. This implementation is similar to falling back to an offloaded cache upon hitting an out-of-memory (OOM) error. Here is the code:

遗憾的是，Transformers 不支持就地更改已初始化缓存对象的位宽。然而，为实现动态策略，我们的代码可以以小块的方式生成 token，并在内存压力下即时地用新的量化缓存配置开始下一个块。这个实现类似于在遇到内存不足错误时回退到卸载缓存。代码如下：

```
# dynamic_quantized_cache.py
# Dynamic KV cache policy using Hugging Face Transformers' QuantizedCache (2025).
# Starts with int8 HQQ and drops to int4 when device memory is tight.
# Requires: transformers >= 4.55, hqq (for HQQ backend).
#
# This uses only public APIs:
#   - cache_implementation="quantized"
#   - cache_config as a dict
#
# References:
#   - KV cache strategies docs (QuantizedCache, HQQ/Quanto backends)
from __future__ import annotations
from typing import Dict, Optional
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
def make_cache_config(
    *,
    backend: str,
    nbits: int,
    device: torch.device,
    compute_dtype: torch.dtype = torch.float16,
    q_group_size: int = 64,
    residual_length: int = 128,
    axis_key: int = 1,
    axis_value: int = 1,
) -> Dict:
    """
    Build a cache_config dictionary accepted by Transformers' quantized cache.
    HQQ supports nbits in {2, 4, 8}; Quanto supports {2, 4}.
    axis_key/axis_value=1 are recommended for HQQ.
    """
    return {
        "backend": backend,                 # "HQQ" or "quanto"
        "nbits": int(nbits),
        "axis_key": axis_key,
        "axis_value": axis_value,
        "q_group_size": int(q_group_size),  # group size along head_dim
        "residual_length": int(residual_length), # recent tokens (orig precision)
        "compute_dtype": compute_dtype,     # dequantization compute dtype
    }
def _gpu_used_ratio() -> float:
    """
    Return fraction of device memory used as 1 - free/total.
    Uses CUDA driver info, which reflects true device state,
    not just the PyTorch allocator's reserved bytes.
    """
    free, total = torch.cuda.mem_get_info()
    return 1.0 - (free / total)
@torch.no_grad()
def generate_with_dynamic_quantized_cache(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    chunk_tokens: int = 32,
    memory_threshold: float = 0.90,   # switch policy if used_ratio >= threshold
    backend: str = "hqq",             # "hqq" or "quanto"
    start_bits: int = 8,              # initial cache bit-width
    fallback_bits: int = 4,           # lower bit-width on pressure
    residual_length: int = 128,
) -> str:
    """
    Generate text in chunks while allowing mid-run policy changes.
    The policy applies to each chunk by choosing cache_config for that chunk.
    If memory is tight, we switch from int8 to int4 in subsequent chunks.
    """
    backend = backend.lower()
    assert backend in {"hqq", "quanto"}, "backend must be 'hqq' or 'quanto'"
    if backend == "quanto":
        assert start_bits in {2, 4} and fallback_bits in {2, 4},
               "Quanto supports only 2 or 4 bits"
    if backend == "hqq":
        assert start_bits in {2, 4, 8} and fallback_bits in {2, 4, 8},
               "HQQ supports 2, 4, or 8 bits"
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated_ids = inputs["input_ids"]   # [batch=1, seq_len]
    tokens_remaining = int(max_new_tokens)
    current_bits = int(start_bits)
    # Use EOS if available to terminate early.
    eos_id: Optional[int] = tokenizer.eos_token_id
    while tokens_remaining > 0:
        # Decide policy for this chunk based on current memory pressure.
        if torch.cuda.is_available():
            # Smooth the signal to avoid oscillation
            # when multiple processes are active.
            if 'used_ratio' in locals():
                used_ratio = 0.8 * used_ratio + 0.2 * _gpu_used_ratio()
            else:
                used_ratio = _gpu_used_ratio()
            if used_ratio >= memory_threshold:
                current_bits = min(current_bits, fallback_bits) # drop bits
                logging.info(f"Current bits {current_bits}")
        cache_cfg = make_cache_config(
            backend=backend,
            nbits=current_bits,
            device=device,
            compute_dtype=torch.bfloat16,
            q_group_size=64,
            residual_length=residual_length,
            axis_key=1,
            axis_value=1,
        )
        # Generate a small chunk with the chosen cache policy.
        this_chunk = min(chunk_tokens, tokens_remaining)
        out = model.generate(
            input_ids=generated_ids,
            max_new_tokens=this_chunk,
            do_sample=False,       # deterministic for clarity; adjust as needed
            use_cache=True,
            cache_implementation="quantized",          # select QuantizedCache
            cache_config=cache_cfg,                    # pass backend + settings
            pad_token_id=eos_id,
            return_dict_in_generate=False,        # we only need the tokens here
        )
        # 'out' is [1, old_len + this_chunk]; slice out newly generated suffix
        new_tokens = out[:, generated_ids.shape[1]:]
        generated_ids = out
        tokens_remaining -= new_tokens.shape[1]
        # Early termination if the model emitted EOS.
        if eos_id is not None and int(new_tokens[0, -1].item()) == eos_id:
            break
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
if __name__ == "__main__":
    # Example usage. Replace with a model that supports your hardware.
    ckpt = "meta-llama/Llama-3.1-8B-Instruct"
    tok = AutoTokenizer.from_pretrained(ckpt)
    mdl = AutoModelForCausalLM.from_pretrained(ckpt,
                                            torch_dtype=torch.float16).to("cuda")
    text = generate_with_dynamic_quantized_cache(
        mdl,
        tok,
        "Explain attention key-value caches in one paragraph.",
        max_new_tokens=120,
        chunk_tokens=32,
        memory_threshold=0.90,
        backend="hqq",         # or "quanto" if you installed Quanto
        start_bits=8,
        fallback_bits=4,
        residual_length=128,
    )
    print(text)
```

```
# dynamic_quantized_cache.py
# Dynamic KV cache policy using Hugging Face Transformers' QuantizedCache (2025).
# Starts with int8 HQQ and drops to int4 when device memory is tight.
# Requires: transformers >= 4.55, hqq (for HQQ backend).
#
# This uses only public APIs:
#   - cache_implementation="quantized"
#   - cache_config as a dict
#
# References:
#   - KV cache strategies docs (QuantizedCache, HQQ/Quanto backends)
from __future__ import annotations
from typing import Dict, Optional
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
def make_cache_config(
    *,
    backend: str,
    nbits: int,
    device: torch.device,
    compute_dtype: torch.dtype = torch.float16,
    q_group_size: int = 64,
    residual_length: int = 128,
    axis_key: int = 1,
    axis_value: int = 1,
) -> Dict:
    """
    Build a cache_config dictionary accepted by Transformers' quantized cache.
    HQQ supports nbits in {2, 4, 8}; Quanto supports {2, 4}.
    axis_key/axis_value=1 are recommended for HQQ.
    """
    return {
        "backend": backend,                 # "HQQ" or "quanto"
        "nbits": int(nbits),
        "axis_key": axis_key,
        "axis_value": axis_value,
        "q_group_size": int(q_group_size),  # group size along head_dim
        "residual_length": int(residual_length), # recent tokens (orig precision)
        "compute_dtype": compute_dtype,     # dequantization compute dtype
    }
def _gpu_used_ratio() -> float:
    """
    Return fraction of device memory used as 1 - free/total.
    Uses CUDA driver info, which reflects true device state,
    not just the PyTorch allocator's reserved bytes.
    """
    free, total = torch.cuda.mem_get_info()
    return 1.0 - (free / total)
@torch.no_grad()
def generate_with_dynamic_quantized_cache(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    chunk_tokens: int = 32,
    memory_threshold: float = 0.90,   # switch policy if used_ratio >= threshold
    backend: str = "hqq",             # "hqq" or "quanto"
    start_bits: int = 8,              # initial cache bit-width
    fallback_bits: int = 4,           # lower bit-width on pressure
    residual_length: int = 128,
) -> str:
    """
    Generate text in chunks while allowing mid-run policy changes.
    The policy applies to each chunk by choosing cache_config for that chunk.
    If memory is tight, we switch from int8 to int4 in subsequent chunks.
    """
    backend = backend.lower()
    assert backend in {"hqq", "quanto"}, "backend must be 'hqq' or 'quanto'"
    if backend == "quanto":
        assert start_bits in {2, 4} and fallback_bits in {2, 4},
               "Quanto supports only 2 or 4 bits"
    if backend == "hqq":
        assert start_bits in {2, 4, 8} and fallback_bits in {2, 4, 8},
               "HQQ supports 2, 4, or 8 bits"
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated_ids = inputs["input_ids"]   # [batch=1, seq_len]
    tokens_remaining = int(max_new_tokens)
    current_bits = int(start_bits)
    # Use EOS if available to terminate early.
    eos_id: Optional[int] = tokenizer.eos_token_id
    while tokens_remaining > 0:
        # Decide policy for this chunk based on current memory pressure.
        if torch.cuda.is_available():
            # Smooth the signal to avoid oscillation
            # when multiple processes are active.
            if 'used_ratio' in locals():
                used_ratio = 0.8 * used_ratio + 0.2 * _gpu_used_ratio()
            else:
                used_ratio = _gpu_used_ratio()
            if used_ratio >= memory_threshold:
                current_bits = min(current_bits, fallback_bits) # drop bits
                logging.info(f"Current bits {current_bits}")
        cache_cfg = make_cache_config(
            backend=backend,
            nbits=current_bits,
            device=device,
            compute_dtype=torch.bfloat16,
            q_group_size=64,
            residual_length=residual_length,
            axis_key=1,
            axis_value=1,
        )
        # Generate a small chunk with the chosen cache policy.
        this_chunk = min(chunk_tokens, tokens_remaining)
        out = model.generate(
            input_ids=generated_ids,
            max_new_tokens=this_chunk,
            do_sample=False,       # deterministic for clarity; adjust as needed
            use_cache=True,
            cache_implementation="quantized",          # select QuantizedCache
            cache_config=cache_cfg,                    # pass backend + settings
            pad_token_id=eos_id,
            return_dict_in_generate=False,        # we only need the tokens here
        )
        # 'out' is [1, old_len + this_chunk]; slice out newly generated suffix
        new_tokens = out[:, generated_ids.shape[1]:]
        generated_ids = out
        tokens_remaining -= new_tokens.shape[1]
        # Early termination if the model emitted EOS.
        if eos_id is not None and int(new_tokens[0, -1].item()) == eos_id:
            break
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
if __name__ == "__main__":
    # Example usage. Replace with a model that supports your hardware.
    ckpt = "meta-llama/Llama-3.1-8B-Instruct"
    tok = AutoTokenizer.from_pretrained(ckpt)
    mdl = AutoModelForCausalLM.from_pretrained(ckpt,
                                            torch_dtype=torch.float16).to("cuda")
    text = generate_with_dynamic_quantized_cache(
        mdl,
        tok,
        "Explain attention key-value caches in one paragraph.",
        max_new_tokens=120,
        chunk_tokens=32,
        memory_threshold=0.90,
        backend="hqq",         # or "quanto" if you installed Quanto
        start_bits=8,
        fallback_bits=4,
        residual_length=128,
    )
    print(text)
```

Here, we start with an INT8 HQQ cache for modest compression and switch to INT4 when actual GPU free memory drops below a threshold. This is measured with torch.cuda.mem_get_info(), which reflects true free versus total device memory. This provides the right signal for the policy choice.

这里，我们以一个用于适度压缩的 INT8 HQQ 缓存起步，当 GPU 实际空闲内存降至阈值以下时切换到 INT4。空闲内存通过 torch.cuda.mem_get_info() 测量，它反映的是设备真实的空闲与总量对比，从而为策略选择提供了正确的信号。

We then generate tokens in small chunks so that we can safely switch the policy between chunks without trying to mutate an existing cache instance. This avoids reaching into private attributes or quantizing tensors manually. The cache backend does the work inside the model’s forward pass.

随后，我们以小分块（chunk）的方式生成 token，这样就能在分块之间安全地切换策略，而无需尝试改写某个已存在的缓存实例。这避免了访问私有属性或手动量化张量。缓存后端会在模型的前向传播（forward pass）内部完成这些工作。

> As shown in this example, it’s recommended to log an event or increment a counter when the policy switches. This way, you can correlate compression events with any accuracy or output anomalies.

> 如本例所示，建议在策略切换时记录一条事件或递增一个计数器。这样，你就能将压缩事件与任何精度或输出异常关联起来。

Similarly, you can dynamically turn off compression if conditions improve. Suppose a long conversation just ended and the next question is short. The system could decide to stop compressing or even restore some caches to higher precision if it will produce better quality responses. The difference is likely small, so it might not be worth it.

同样地，如果条件改善，你也可以动态关闭压缩。假设一段长对话刚刚结束，而下一个问题很短。系统可以决定停止压缩，甚至把某些缓存恢复到更高的精度——只要这样能带来更高质量的回复。不过差异很可能很小，因此未必值得这样做。

It’s important to avoid rapid fluctuations in compression since toggling compression on/off too often could thrash performance. To do this, you can introduce intentional delays (aka *hysteresis* and *cooldown*) between changes. For example, if a higher-compression strategy is changed, keep it until GPU memory drops well below a given threshold. This way, you avoid oscillations and thrashing.

务必避免压缩状态的快速波动，因为过于频繁地开关压缩会导致性能抖动。为此，你可以在两次变更之间引入有意的延迟（即*迟滞（hysteresis）*与*冷却（cooldown）*）。例如，当切换到某种更高压缩率的策略后，就保持这一策略，直到 GPU 内存降到某给定阈值以下相当多为止。这样就能避免震荡与抖动。

Having this flexibility is useful if, for instance, your service sometimes prioritizes maximum quality (no compression) for premium users versus maximum throughput (heavy compression) for free-tier users. The policy can switch based on request metadata as well, including the user’s subscription type.

具备这种灵活性是有用的，例如，你的服务有时会为付费用户优先保证最高质量（不压缩），而为免费层用户优先保证最高吞吐量（重度压缩）。策略也可以基于请求元数据来切换，包括用户的订阅类型。

No discussion on caching is complete without considering eviction strategies, such as Least Recently Used (LRU) eviction. If context length becomes too long, some model architectures—like those with recency bias or sliding-window attention—might choose to discard or downsample very old tokens entirely. Sliding-window attention is shown in Figure 19-5.

任何关于缓存的讨论都离不开逐出（eviction）策略，例如最近最少使用（Least Recently Used，LRU）逐出。如果上下文长度变得过长，某些模型架构——例如带近因偏好或滑动窗口注意力（sliding-window attention）的架构——可能会选择完全丢弃或降采样非常旧的 token。滑动窗口注意力如图 19-5 所示。

![Figure 19-5. Sliding-window attention uses the intuition that the most useful tokens are the most recent](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-5.png)

![图 19-5. 滑动窗口注意力基于这样的直觉：最有用的 token 就是最近的那些](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-5.png)

While LRU eviction of earlier tokens from the context is not exactly compression, it’s yet another type of policy that can be dynamically chosen at runtime. For instance, the system can decide that, beyond 2,048 tokens, the model likely won’t need the earlier tokens—based on some heuristics or a smaller LLM.

虽然把上下文中较早的 token 做 LRU 逐出并不完全等同于压缩，但它是另一类可以在运行时动态选择的策略。例如，系统可以判定：在超过 2,048 个 token 之后，模型很可能不再需要更早的那些 token——这一判断可以基于某些启发式规则或一个更小的 LLM。

In this case, the system could start dropping those older tokens—or periodically compress them into a smaller summary. This starts getting into model and algorithmic territory—and requires more support, maintenance, and model training—but it is a form of dynamic context management that should be considered in advanced serving engines.

在这种情况下，系统可以开始丢弃那些更旧的 token——或者定期把它们压缩成一份更小的摘要。这已经开始涉足模型与算法领域了——需要更多的支持、维护与模型训练——但它是一种动态上下文管理的形式，在高级服务引擎中值得考虑。

In short, you should consider quantized cache mechanisms provided by your inference engine, as they can handle the details of maintaining quantization-scale factors, interfacing with attention kernels, and monitoring GPU memory allocators at runtime. At a minimum, when the system sees memory utilization approaching a certain limit, log that and see if enabling a compression policy at that point would avoid OOM without hurting latency.

简而言之，你应当考虑使用推理引擎所提供的量化缓存机制，因为它们可以处理诸多细节：维护量化尺度因子、与注意力核函数对接、以及在运行时监控 GPU 内存分配器。至少，当系统看到内存利用率接近某个上限时，把这一情况记录下来，并观察此时启用一项压缩策略是否能在不损害延迟的前提下避免 OOM。

> In practice, setting a high-watermark threshold on GPU memory (e.g., 80%) to trigger 8-bit compression has proven effective in preventing OOM crashes in production.

> 在实践中，对 GPU 内存设置一个高水位阈值（例如 80%）来触发 8 位压缩，已被证明能有效防止生产环境中的 OOM 崩溃。

If it makes sense to use dynamic compression policies, you can implement the trigger. As with any quantization and compression strategy, be sure to test their impact on your model’s output specific to your domain. Many generative tasks tolerate aggressive compression, but it’s always good to verify that using 4-bit versus 8-bit doesn’t introduce errors or unexpected outputs.

如果使用动态压缩策略是合理的，你就可以实现这个触发器。与任何量化与压缩策略一样，务必针对你所在领域测试它们对模型输出的影响。许多生成式任务能容忍激进的压缩，但始终有必要验证：使用 4 位相较于 8 位不会引入错误或意料之外的输出。

## Reinforcement Learning Agents for Tuning AI Systems at Runtime

## 在运行时用强化学习智能体调优 AI 系统

Many of the techniques we’ve discussed so far involve decisions based on the system’s current state. These decisions often require trade-offs, such as speed versus accuracy and throughput versus latency. Rather than collecting more and more heuristics to make decisions, we can tune our inference system with reinforcement learning with an RL agent, environment, policy, and reward.

到目前为止我们讨论的许多技术都涉及基于系统当前状态的决策。这些决策往往需要权衡取舍，例如速度与精度、吞吐量与延迟。与其不断收集越来越多的启发式规则来做决策，我们可以用强化学习来调优推理系统，配备一个 RL 智能体（agent）、环境（environment）、策略（policy）与奖励（reward）。

> This is a cutting-edge approach. You should start with simpler heuristics as your baseline. Then you can use RL as an incremental improvement once the basics are stable.

> 这是一种前沿方法。你应当先以更简单的启发式规则作为基线。等基础打稳后，再把 RL 作为一种增量式改进引入。

Specifically, our inference engine watches server metrics (the environment), chooses actions (the policy) to maximize throughput while keeping latency under a target, and receives feedback (the reward) that guides continual improvement. In this way, the system becomes an online optimizer—continually refining its decision making as conditions change.

具体来说，我们的推理引擎监视服务器指标（环境），选择动作（策略）以在把延迟保持在目标之内的同时最大化吞吐量，并接收反馈（奖励）来引导持续改进。如此一来，系统便成为一个在线优化器——随着条件变化不断精炼其决策。

For instance, one could set up an environment in which each RL “step” is an inference request. And the RL actions include things like the following:

例如，可以搭建这样一个环境：每个 RL“步”（step）就是一次推理请求。而 RL 动作包括下列这些：

*Action 1* Choose parallelism mode: single, TP, PP, and hybrid.

*动作 1* 选择并行模式：单卡、TP、PP 与混合。

*Action 2* Choose precision: full FP8 versus mixing FP8 and FP4.

*动作 2* 选择精度：全 FP8 还是混用 FP8 与 FP4。

*Action 3* Adjust batch size or batch-waiting time.

*动作 3* 调整批大小或批等待时间。

*Action 4* Enable or disable cache compression.

*动作 4* 启用或禁用缓存压缩。

*Action 5* Enable or disable speculative decoding.

*动作 5* 启用或禁用推测解码（speculative decoding）。

*Action 6* Select a smaller draft model for speculative decoding.

*动作 6* 为推测解码选择一个更小的草稿模型（draft model）。

*Action 7* Select a larger draft model for speculative decoding.

*动作 7* 为推测解码选择一个更大的草稿模型。

*Action 8* Enable or disable speculative KV prefetching.

*动作 8* 启用或禁用推测式 KV 预取。

…more actions…

……更多动作……

The current RL state observed by the agent might include GPU utilization, memory utilization, average latency, queue length of requests, etc. The RL reward is then defined by capturing business objectives, such as reward = throughput - λ * max(0, latency - SLA). λ is a tunable penalty weight in this reward function that scales how harshly you punish latency violations.

智能体所观测到的当前 RL 状态可能包括 GPU 利用率、内存利用率、平均延迟、请求队列长度等。RL 奖励随后通过刻画业务目标来定义，例如 reward = throughput - λ * max(0, latency - SLA)。在这个奖励函数中，λ 是一个可调的惩罚权重，用于缩放你对延迟违约的惩罚力度。

> It’s recommended that you normalize the state features so that the RL agent doesn’t have to learn the scale on its own. This can speed up training convergence. For example, you can scale queue length by a max value, etc.

> 建议对状态特征做归一化，这样 RL 智能体就不必自己去学习各特征的量纲尺度。这可以加快训练收敛。例如，你可以用一个最大值来缩放队列长度，等等。

Here, a larger λ makes the agent prioritize staying within the SLA over squeezing out extra throughput. A smaller λ lets it risk occasional latency overshoots to achieve higher token rates overall. Essentially, this reward function penalizes latency that exceeds the SLA but otherwise tries to increase throughput.

这里，较大的 λ 会让智能体更看重不违反 SLA，而不是压榨出额外的吞吐量。较小的 λ 则允许它冒偶尔延迟超标的风险，以换取整体上更高的 token 速率。本质上，这个奖励函数会惩罚超出 SLA 的延迟，但在其他情况下则尽力提升吞吐量。

> In practice, start with λ such that λ × (typical latency overshoot) is about equal to the throughput gain that you’d trade for it. For example, if a 10 ms delay is tolerable to gain 100 tokens/sec, set λ so that 10 ms × λ ≈ 100.

> 在实践中，起步时应让 λ 满足：λ ×（典型的延迟超标量）大致等于你愿意为之付出的吞吐量收益。例如，如果为换取 100 tokens/sec 而容忍 10 ms 的延迟，那就把 λ 设为使 10 ms × λ ≈ 100。

Over many iterations, the RL agent can learn when it’s beneficial to compress caches—for instance, when memory is high and latency isn’t immediately impacted. Or it could learn to switch to pipeline parallel mode when GPU utilization is low but one GPU is overloaded, etc.

经过多次迭代，RL 智能体可以学会何时压缩缓存是有益的——例如当内存占用很高而延迟不会立即受影响时。或者它可以学会：当 GPU 利用率偏低但某一块 GPU 过载时，切换到流水线并行模式，等等。

This helps because PP breaks the model into sequential stages across multiple devices and redistributes the heavy work away from the bottlenecked device—smoothing out utilization and avoiding single-GPU hotspots.

这之所以有帮助，是因为 PP 把模型拆分为跨多个设备的顺序阶段，并将繁重的工作从瓶颈设备上重新分配出去——从而平滑利用率、避免单 GPU 热点（hotspot）。

The agent can find nonintuitive configurations that produce better performance. For instance, it might learn that for prompts above a certain length, it should enable both PP and FP4 compression to produce the best token throughput, whereas, for shorter prompts, it learns to use just pure tensor parallel in FP8. If we tried to encode this logic as a set of static rules, we might miss complex interaction.

智能体能够找到产生更好性能的反直觉配置。例如，它可能学到：对于超过某一长度的提示词，应同时启用 PP 与 FP4 压缩以获得最佳的 token 吞吐量；而对于较短的提示词，则学会仅用 FP8 下的纯张量并行。如果我们试图把这套逻辑编码为一组静态规则，就可能漏掉复杂的交互作用。

An RL agent can more easily discover optimal combinations by exploring the action space—and ultimately exploiting the optimal configuration until the environmental conditions change. At this point, the RL would adjust the inference system accordingly since it’s always exploring the action space and trying new configurations.

RL 智能体可以通过探索动作空间更容易地发现最优组合——并最终利用（exploit）这一最优配置，直到环境条件发生变化。此时，RL 会相应地调整推理系统，因为它始终在探索动作空间、尝试新的配置。

Training such an agent can be done offline using simulators and libraries such as Hugging Face’s Transformer RL (TRL) libraries. For instance, we could log a bunch of data from a running system under various conditions—and then train an RL policy in simulation to predict outcomes. At a very high level, the RL reward and update loop would look something like the following pseudocode:

训练这样一个智能体可以离线完成，使用诸如 Hugging Face 的 Transformer RL（TRL）之类的模拟器与库。例如，我们可以在各种条件下从运行中的系统采集大量数据——然后在模拟中训练一个 RL 策略来预测结果。在非常高的层面上，RL 的奖励与更新循环大致如下面的伪代码所示：

```
# Pseudo structure for an RL-driven tuner
# This loop runs in separate thread/process alongside main inference service.
# e.g., {gpu_util:0.7, mem_util:0.9, avg_latency:120ms, req_queue:5}
state = get_system_state()
# e.g., 0 -> high precision, 1 -> low precision
action = rl_agent.select_action(state)
# Map action to actual parameter changes
if action == 0:
    precision_policy = "FP8"
else:
    precision_policy = "FP4"
# (We could have multiple actions, but single action for illustration)
apply_precision_policy(precision_policy)
# ... After the next token or set of tokens ...
new_state = get_system_state()
reward = compute_reward(old_state, new_state)
rl_agent.update(state, action, reward, new_state)
```

```
# Pseudo structure for an RL-driven tuner
# This loop runs in separate thread/process alongside main inference service.
# e.g., {gpu_util:0.7, mem_util:0.9, avg_latency:120ms, req_queue:5}
state = get_system_state()
# e.g., 0 -> high precision, 1 -> low precision
action = rl_agent.select_action(state)
# Map action to actual parameter changes
if action == 0:
    precision_policy = "FP8"
else:
    precision_policy = "FP4"
# (We could have multiple actions, but single action for illustration)
apply_precision_policy(precision_policy)
# ... After the next token or set of tokens ...
new_state = get_system_state()
reward = compute_reward(old_state, new_state)
rl_agent.update(state, action, reward, new_state)
```

Here, the loop continuously runs in the background of the inference server. The com pute_reward function incorporates throughput (e.g., tokens per second since last step) and latency metrics. Since we are trying to balance throughput with latency, this is a multi-objective optimization problem in which we are optimizing multiple goals at once. A common approach is to use a weighted sum to combine the multiple objectives into a single objective.

这里，该循环在推理服务器的后台持续运行。compute_reward 函数纳入了吞吐量（例如自上一步以来每秒的 token 数）与延迟指标。由于我们试图在吞吐量与延迟之间取得平衡，这是一个多目标优化（multi-objective optimization）问题，即同时优化多个目标。一种常见做法是用加权和把多个目标合并为单一目标。

> For more flexibility, especially under uncertainty, you can instead model the multi-objective optimization problem—or Pareto front analysis—as a partially observable decision process. This allows the agent to learn its own trade-off strategy between objectives like throughput versus latency, etc. This is helpful if a single-weighted reward is not sufficient.

> 为获得更高的灵活性，尤其是在存在不确定性时，你可以转而把多目标优化问题——或帕累托前沿（Pareto front）分析——建模为一个部分可观测决策过程。这让智能体能够自行学习在诸如吞吐量与延迟等目标之间的权衡策略。当单一的加权奖励不够用时，这会很有帮助。

These kinds of multiparameter interactions are hard to tune with basic grid search methods. As such, RL and optimization techniques like proximal policy optimization (PPO) are best used for tuning inference workloads. PPO is known for stabler learning in continuous action spaces. It’s well-suited for continuous updating in real-time environments as it adjusts the policy gradually. This avoids extreme oscillations, which is important for inference stability. We don’t want the agent thrashing between decisions on every request.

这类多参数交互很难用基本的网格搜索方法来调优。因此，RL 以及诸如近端策略优化（proximal policy optimization，PPO）之类的优化技术最适合用来调优推理工作负载。PPO 以在连续动作空间中更稳定的学习而著称。它很适合在实时环境中做连续更新，因为它会逐步调整策略。这避免了剧烈震荡，而这对推理稳定性很重要。我们不希望智能体在每个请求上都在各决策之间反复横跳。

Another technique to reduce oscillations is called *damping*. This requires that an action stay in effect for a minimum amount of time—or minimum number of requests. You can override damping for critical SLO violations, if needed, but this should be done sparingly.

另一种减少震荡的技术叫做*阻尼（damping）*。它要求某个动作至少保持一段最短的时间——或至少作用于最少数量的请求。必要时你可以为关键的 SLO 违约情况覆盖阻尼，但这应当谨慎为之。

It’s important to know that RL agents might make unsafe or suboptimal moves while learning. To mitigate that, you can constrain the action space to a reasonable set of ranges. It’s also recommended to start with a good default policy using the heuristics that we have already identified. The agent can then fine-tune around that initial default policy.

需要知道的是，RL 智能体在学习过程中可能做出不安全或次优的动作。为缓解这一点，你可以把动作空间约束到一组合理的取值范围内。同样建议：先用我们已经识别出的启发式规则设定一个良好的默认策略。智能体随后可以围绕这个初始默认策略进行微调。

Alternatively, the agent can be trained online in *shadow mode* using a live system that incorporates an exploration phase. During exploration, the system occasionally tries a random or slightly modified strategy to gather new data. Otherwise, it exploits the current best policy.

或者，也可以在*影子模式（shadow mode）*下用一个包含探索阶段的在线系统来训练智能体。在探索阶段，系统偶尔会尝试一种随机或略作修改的策略以采集新数据。其余时候，它则利用当前的最优策略。

Another technique is to apply reward shaping, which keeps the agent from violating critical constraints. For instance, the RL system would generate a high negative reward if latency is greater than a hard limit—or if an OOM error occurs due to a bad action.

另一种技术是应用奖励塑形（reward shaping），它能防止智能体违反关键约束。例如，当延迟超过某个硬性上限时——或当某个糟糕的动作导致 OOM 错误时——RL 系统会给出一个很高的负奖励。

Additionally, you can hard-code the system to avoid unsafe actions—even if the reward suggests the system do so. This puts in place extra safeguards so that the agent’s natural exploration won’t cause a catastrophic failure. This is a practical approach that combines RL with rule-based guardrails.

此外，你还可以把系统硬编码为规避不安全的动作——即便奖励暗示系统应当这样做。这设置了额外的安全防线，使得智能体自然的探索不会导致灾难性故障。这是一种把 RL 与基于规则的护栏相结合的实用做法。

Designing a proper reward function is important. For instance, if we care about throughput under a latency limit, a reward would look like the code here:

设计一个恰当的奖励函数很重要。例如，如果我们关心的是在延迟上限之下的吞吐量，奖励会像这里的代码那样：

```
reward = tokens_per_second - 1000 * (1 if latency > SLA else 0)
```

```
reward = tokens_per_second - 1000 * (1 if latency > SLA else 0)
```

Here, a large penalty is applied if the latency SLA is exceeded. Otherwise, no penalty is applied. Another option is to apply a continuous penalty that is proportional to how far the latency overshoots the target SLA. A simple continuous-penalty reward can be written, as shown here:

这里，一旦超出延迟 SLA，就施加一个很大的惩罚。否则不施加任何惩罚。另一种选择是施加一个连续惩罚，其大小与延迟超出目标 SLA 的程度成正比。一个简单的连续惩罚奖励可以写成如下形式：

```
reward = tokens_per_second – λ * max(0.0, latency – SLA)
```

```
reward = tokens_per_second – λ * max(0.0, latency – SLA)
```

Here, λ is your penalty weight, and max(0.0, latency – SLA) grows linearly with how far you exceed the SLA. This way, the agent receives a smoothly increasing penalty the longer its latency overshoots the target. This will produce smoother gradients and more gradual trade-off decisions. In practice, a continuous (soft) penalty often produces a more stable policy than a binary (hard) penalty.

这里，λ 是你的惩罚权重，而 max(0.0, latency – SLA) 随你超出 SLA 的程度线性增长。这样，延迟超出目标越久，智能体收到的惩罚就越平滑地增大。这会产生更平滑的梯度与更渐进的权衡决策。在实践中，连续（软）惩罚往往比二值（硬）惩罚产生更稳定的策略。

> It’s recommended to start with a simple, static set of heuristics for tuning. Once the system is stable, you can start to introduce an RL agent to handle the more complex tuning that the heuristics can’t capture.

> 建议先用一组简单、静态的启发式规则来做调优。一旦系统稳定，你就可以开始引入 RL 智能体，来处理那些启发式规则无法捕捉的更复杂的调优。

Logging and observability are important. You should continuously log the decisions that the RL agent makes—as well as the decision outcomes. For example, you should use structured logging—or even counters and telemetry dashboards—to track state → action → reward sequences in real time. This will help debug the agent’s behavior if it starts behaving erratically.

日志记录与可观测性很重要。你应当持续记录 RL 智能体所做的决策——以及决策的结果。例如，你应使用结构化日志——甚至计数器与遥测仪表盘——来实时追踪 state → action → reward 的序列。当智能体开始出现异常行为时，这将有助于调试。

It’s also recommended to provide an escape hatch, or *kill switch*. This way, if the agent starts doing something obviously bad, like consistently making latency worse, you can have the system fall back to a safe, static configuration while you diagnose the issue and retrain a new policy offline. For example, if p95 latency increases by more than 50% after enabling the agent, the system will automatically disable the agent’s actions and send an alert to the system on call.

同样建议提供一个逃生舱口，或称*断路开关（kill switch）*。这样，如果智能体开始做出明显糟糕的事情，比如持续使延迟变差，你就可以让系统回退到一个安全、静态的配置，同时你去诊断问题并在离线重新训练一个新策略。例如，如果启用智能体后 p95 延迟增加超过 50%，系统会自动禁用智能体的动作，并向值班人员发送告警。

While not yet mainstream in modern inference serving engines as of this writing, RL-based, online inference tuning is just beginning to appear. Expect more inference platforms to include self-tuning capabilities as these techniques mature. This is important since these models and systems are becoming more complex. Manually managing all of these tuning knobs is difficult under rapidly changing conditions—it’s difficult for humans, anyway!

尽管截至本文撰写时，基于 RL 的在线推理调优在现代推理服务引擎中尚未成为主流，但它才刚刚开始出现。随着这些技术的成熟，可以预期会有更多推理平台纳入自调优能力。这一点很重要，因为这些模型与系统正变得越来越复杂。在快速变化的条件下手动管理所有这些调优旋钮很困难——反正对人类来说是很困难的！

An intelligent agent that adapts in real time is a natural evolution of system optimization. We are starting to see self-optimizing AI inference servers that achieve expert-level performance tuning automatically. And they’re doing this just by learning from their own real-time telemetry metrics.

一个能实时自适应的智能体，是系统优化的自然演进。我们已经开始看到能自动达到专家级性能调优水平的自优化 AI 推理服务器。而它们做到这一点，仅仅是通过学习自身的实时遥测指标。

## Dynamic Memory-Allocation Switching (Slab Versus Caching Versus Stream-Ordered)

## 动态内存分配切换（slab、缓存与流序三者之间）

GPU memory fragmentation and nonoptimal memory allocation can be silent performance killers. Inference servers allocate and free thousands of tensors per second for many objects, including tokens, intermediate activations, etc. The strategy used by the memory allocator can influence fragmentation and allocation latency.

GPU 内存碎片化（fragmentation）与非最优的内存分配可能是无声的性能杀手。推理服务器每秒为众多对象——包括 token、中间激活等——分配并释放数千个张量。内存分配器所采用的策略会影响碎片化与分配延迟。

Switching the memory allocator dynamically means that the system can change how it allocates memory on the fly. For instance, the system can use a slab allocator for certain allocation sizes—or switch to use CUDA’s stream-ordered (cudaMallocAsync) allocator. The decision depends on the observed pattern of allocations and expected memory fragmentation.

动态切换内存分配器意味着系统可以即时改变它分配内存的方式。例如，系统可以对某些分配大小使用 slab 分配器——或者切换为使用 CUDA 的流序（cudaMallocAsync）分配器。这一决策取决于观测到的分配模式与预期的内存碎片化。

By default, PyTorch uses a variant of the buddy/best-fit memory allocator called *best-fit with coalescing*, or BFC. It grabs big chunks of GPU memory and subdivides the chunks to satisfy allocation requests. This reuses free space and avoids frequent calls to the relatively slow and synchronous cudaMalloc and cudaFree.

默认情况下，PyTorch 使用一种被称为*带合并的最佳适配（best-fit with coalescing）*（即 BFC）的伙伴/最佳适配（buddy/best-fit）分配器变体。它抓取大块的 GPU 内存并将其细分以满足分配请求。这重用了空闲空间，并避免了频繁调用相对缓慢且同步的 cudaMalloc 与 cudaFree。

A buddy allocator splits memory into blocks whose sizes are powers of two. A slab allocator works on top of the buddy system to efficiently manage small, fixed-size objects. It preallocates slabs, or collections of objects of a given type, and maintains a free list within each slab, as shown in Figure 19-6.

伙伴分配器（buddy allocator）把内存拆分成大小为 2 的幂次的块。slab 分配器则工作在伙伴系统之上，用以高效管理小的、固定大小的对象。它预分配 slab（即某一给定类型对象的集合），并在每个 slab 内维护一个空闲列表（free list），如图 19-6 所示。

![Figure 19-6. Slab allocator maintains a free list of memory objects within each preallocated slab](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-6.png)

![图 19-6. slab 分配器在每个预分配的 slab 内维护一个内存对象的空闲列表](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-6.png)

A slab allocator allows fast reuse without fragmentation. A buddy allocator handles coarse-grained page allocation, while a slab allocator optimizes fine-grained object reuse.

slab 分配器允许在不产生碎片化的情况下快速重用。伙伴分配器处理粗粒度的页分配，而 slab 分配器则优化细粒度的对象重用。

The default PyTorch caching allocator works well for many workloads. However, it can suffer fragmentation if the pattern of allocations varies widely due to alternating between large and small allocations. In a long-running server that handles different types of queries, fragmentation can build up.

默认的 PyTorch 缓存分配器（caching allocator）对许多工作负载都工作良好。然而，如果分配模式因在大分配与小分配之间来回交替而变化剧烈，它就可能出现碎片化。在一个处理不同类型查询的长期运行的服务器中，碎片会逐渐累积。

In this case, plenty of memory is free, but the memory is not contiguous enough for large tensor allocations. This leads to OOM errors—even though memory is technically available.

在这种情况下，会有大量内存空闲，但这些内存不够连续，无法满足大张量分配。这会导致 OOM 错误——尽管从技术上说内存是可用的。

Remember that PyTorch provides torch.cuda.memory_summary() to evaluate memory fragmentation, as well as a memory profiler built into torch.profiler (pro file_memory=True). You can use these to determine which operations allocate a lot of memory. Also, NVIDIA Nsight Systems provides CUDA memory event and Unified Memory page-fault tracking on the timeline, and Nsight Compute provides memory workload analysis.

请记住，PyTorch 提供了 torch.cuda.memory_summary() 来评估内存碎片化，以及一个内置于 torch.profiler 的内存分析器（profile_memory=True）。你可以用它们来判定哪些操作分配了大量内存。此外，NVIDIA Nsight Systems 在时间线上提供 CUDA 内存事件与统一内存（Unified Memory）缺页追踪，而 Nsight Compute 则提供内存工作负载分析。

Together, these tools let you observe allocation behavior and fragmentation effects over time. And you can use these tools during development and testing to initiate your memory-allocator tuning strategy—including a dynamic tuning strategy, as discussed next.

这些工具合在一起，让你能够随时间观察分配行为与碎片化效应。你可以在开发与测试阶段使用它们来启动你的内存分配器调优策略——包括接下来讨论的动态调优策略。

A brute-force way to reduce memory fragmentation is to periodically reset the allocator’s state. However, a more clever way is to use CUDA’s stream-ordered caching allocator, cudaMallocAsync, which uses a similar concept internally by binning per allocation size up to a certain limit. But slab allocation takes it even further by never mixing sizes.

一种减少内存碎片化的暴力方法是定期重置分配器的状态。不过，一种更巧妙的方法是使用 CUDA 的流序缓存分配器 cudaMallocAsync，它在内部通过按分配大小分箱（直到某个上限）采用了类似的思路。但 slab 分配更进一步——它绝不混用不同大小。

cudaMallocAsync behaves somewhat like a slab allocator combined with a buddy system—and it’s managed by the CUDA driver for you. This gives you most of the benefit of custom allocators with little effort—and makes it a great default memory allocator to leave on all the time, if you prefer.

cudaMallocAsync 的行为有点像 slab 分配器与伙伴系统的结合——而且它由 CUDA 驱动为你托管。这让你几乎不费力就能获得自定义分配器的大部分收益——如果你愿意，它也是一个很棒的、可以始终开启的默认内存分配器。

Specifically, cudaMallocAsync uses stream-ordered pools that automatically recycle memory when references to the memory are released. It then coalesces the freed blocks behind the scenes since it knows the dependency order of the memory-frees—unlike standard allocators.

具体来说，cudaMallocAsync 使用流序内存池，当对内存的引用被释放时会自动回收内存。随后它会在幕后合并已释放的块，因为它知道内存释放的依赖顺序——这一点与标准分配器不同。

When using cudaMallocAsync with a PyTorch runtime, you can dynamically adjust max_split_size_mb by setting the PYTORCH_ALLOC_CONF=max_split_size_mb: <value> environment variable. This can adjust the split size under different conditions.

在 PyTorch 运行时中使用 cudaMallocAsync 时，你可以通过设置环境变量 PYTORCH_ALLOC_CONF=max_split_size_mb: <value> 来动态调整 max_split_size_mb。这可以在不同条件下调整切分大小。

For instance, a dynamic system could increase max_split_size_mb when large allocations are expected. This way they don’t get broken into small pieces. Conversely, the system can decrease max_split_size_mb when running many small requests to allow more reuse of large blocks.

例如，一个动态系统可以在预期出现大分配时增大 max_split_size_mb。这样它们就不会被切成小碎片。反之，当运行许多小请求时，系统可以减小 max_split_size_mb，以允许更多地重用大块。

> Too small a split size can flood the allocator with many tiny blocks, which will increase metadata overhead and potential fragmentation. Too large a split size reduces the block count (and metadata) but may leave bigger “holes” in memory that go unused when you free only part of a block.

> 切分大小过小会让分配器充斥大量微小的块，这会增加元数据开销与潜在的碎片化。切分大小过大会减少块数量（以及元数据），但当你只释放某个块的一部分时，可能会在内存中留下更大的、无人使用的“空洞”。

Consider a scenario in which your service detects fragmentation—perhaps using PyTorch’s memory snapshot functionality that shows holes caused by fragmentation. In this case, the system could dynamically switch to use cudaMallocAsync, which can consolidate memory usage.

考虑这样一个场景：你的服务检测到了碎片化——也许是借助 PyTorch 的内存快照功能，它会展示由碎片化造成的空洞。在这种情况下，系统可以动态切换为使用 cudaMallocAsync，它能整合内存使用。

You should use memory monitoring tools to track—and log—memory fragmentation. For instance, in PyTorch, you can use torch.cuda.memory_reserved() and torch.cuda.memory_allocated(). Here, the reserved memory is the total GPU memory held by the allocator. And allocated is how much of it is actually in use by tensors.

你应当使用内存监控工具来追踪——并记录——内存碎片化。例如，在 PyTorch 中，你可以使用 torch.cuda.memory_reserved() 与 torch.cuda.memory_allocated()。这里，reserved 内存是分配器所持有的 GPU 内存总量，而 allocated 则是其中实际被张量使用的量。

A large gap between reserved and allocated means fragmentation since a lot of memory is reserved but not used. If that gap grows over time, a dynamic policy could be to periodically purge the cache to free all the unused memory back to the GPU or even restart the worker process to fully reset the allocator. These are intrusive but effective methods that are sometimes used in production for long-running processes with heavy fragmentation.

reserved 与 allocated 之间的巨大差距意味着碎片化，因为有大量内存被保留但未被使用。如果这一差距随时间增长，一种动态策略可以是：定期清空缓存以把所有未使用的内存归还给 GPU，甚至重启 worker 进程以彻底重置分配器。这些方法具有侵入性但很有效，有时会在生产环境中用于碎片化严重的长期运行进程。

> You should use intrusive defragmentation methods like purging and restarting only during maintenance windows—or in a rolling restart manner across a fleet to avoid downtime. If you need to resort to these disruptive mechanisms, you likely have a deeper issue that needs to be addressed and optimized.

> 只应在维护窗口期间使用清空、重启这类侵入式碎片整理方法——或者以跨整个集群滚动重启的方式来避免停机。如果你不得不诉诸这些破坏性机制，那你很可能面临一个更深层、需要被解决与优化的问题。

To implement dynamic allocation switching in PyTorch, for example, you can start with the PyTorch native allocator. Then, if you catch an OOM error, you can retry using the cudaMallocAsync-based allocator.

举例来说，要在 PyTorch 中实现动态分配切换，你可以先从 PyTorch 原生分配器起步。然后，如果你捕获到一个 OOM 错误，就可以改用基于 cudaMallocAsync 的分配器重试。

Unfortunately, the CUDA caching allocator is created the first time torch is imported or when the first CUDA context is touched. And Python’s importlib.reload does not unload C++ extensions or tear down the allocator. As such, changing PYTORCH_ALLOC_CONF (formerly PYTORCH_CUDA_ALLOC_CONF) on the fly and reloading the Python module will not reconfigure the allocator in-process.

遗憾的是，CUDA 缓存分配器在 torch 首次被导入时——或在首次触及 CUDA 上下文时——就已创建。而 Python 的 importlib.reload 并不会卸载 C++ 扩展或拆除分配器。因此，在进程内即时更改 PYTORCH_ALLOC_CONF（旧称 PYTORCH_CUDA_ALLOC_CONF）并重新加载 Python 模块，并不会重新配置分配器。

However, you can spawn a fresh process in which the environment variable is set before torch is imported. Below is a snippet of code that catches OOM, frees memory in the parent, and then spins up a clean child process with PYTORCH_ALLOC_CONF. This is a bit hacky but shows how you can dynamically set the backend:cudaMallocAsync and rerun the same call with the different allocator backend. Next is a PyTorch example that implements this dynamic strategy when the code catches a torch.cuda.OutOfMemoryError:

不过，你可以派生（spawn）一个全新的进程，在其中于导入 torch 之前就设置好该环境变量。下面是一段代码，它捕获 OOM，在父进程中释放内存，然后带着 PYTORCH_ALLOC_CONF 启动一个干净的子进程。这有点取巧，但展示了你如何动态设置 backend:cudaMallocAsync 并用不同的分配器后端重跑同一次调用。接下来是一个 PyTorch 示例，它在代码捕获到 torch.cuda.OutOfMemoryError 时实现这一动态策略：

```
# dynamic_memory_allocator.py
# Retry generation in fresh process with cudaMallocAsync if first attempt OOMs.
# This is the only reliable way to change the CUDA allocator at runtime.
import os
import sys
import gc
import pickle
import tempfile
import subprocess
import importlib
from typing import Callable, Any
def _resolve_factory(factory_path: str) -> Callable[[], Any]:
    """
    Resolves a factory string like "my_package.my_mod:build_model" to a callable.
    The callable must return ready-to-use model with .generate(request) method.
    """
    module_name, func_name = factory_path.split(":", 1)
    module = importlib.import_module(module_name)  # safe to import, no torch yet
    return getattr(module, func_name)
def generate_with_allocator_retry(
    model_factory_path: str,
    request_object: Any,
    *,
    allocator_conf: str = "backend:cudaMallocAsync"
) -> Any:
    """
    Attempts model.generate(request_object) in the current process.
    On torch.cuda.OutOfMemoryError, retries in a fresh subprocess with
    PYTORCH_ALLOC_CONF set to allocator_conf. request_object and the
    returned value must be picklable.
    """
    # Import torch only inside function; avoid importing at module import time.
    import torch
    model_factory = _resolve_factory(model_factory_path)
    model = model_factory()  # user-supplied function builds model, moves to GPU
    try:
        # First attempt uses whatever allocator current process started with.
        return model.generate(request_object)
    except torch.cuda.OutOfMemoryError:
        # Free as much as possible in the parent before spawning the child.
        # Avoids compounding pressure when two processes momentarily overlap.
        try:
            del model
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        # Serialize request to temp file and ask fresh interpreter to do work.
        with tempfile.TemporaryDirectory() as td:
            req_path = os.path.join(td, "request.pkl")
            out_path = os.path.join(td, "output.pkl")
            with open(req_path, "wb") as f:
                pickle.dump(request_object, f)
            # In the child, we want torch to see allocator config at import.
            env = os.environ.copy()
            env["PYTORCH_ALLOC_CONF"] = allocator_conf
            # Re-run this module as a helper child. The child will import torch
            # only after PYTORCH_ALLOC_CONF is set in its environment.
            cmd = [
                sys.executable,
                __file__,
                "--child",
                "--factory", model_factory_path,
                "--request", req_path,
                "--output", out_path,
            ]
            completed = subprocess.run(cmd, env=env, capture_output=True,
                                       text=True)
            if completed.returncode != 0:
                # Bubble up child stderr to aid debugging
                raise RuntimeError(
                  f"Retry failed with exit code {completed.returncode}\n"
                  f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
                )
            with open(out_path, "rb") as f:
                return pickle.load(f)
def _child_main(factory_path: str, request_path: str, output_path: str) -> None:
    """
    Child entrypoint: assumes PYTORCH_ALLOC_CONF is already present in env.
    Imports torch only now, builds the model, runs generate, and pickles result.
    """
    # Import torch after env var set by the parent’s subprocess.run(env=...).
    import torch
    model_factory = _resolve_factory(factory_path)
    model = model_factory()  # build the model inside the child process
    with open(request_path, "rb") as f:
        request_object = pickle.load(f)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--factory", type=str, default="")
    parser.add_argument("--request", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    if args.child:
        _child_main(args.factory, args.request, args.output)
    else:
        print("This module is intended to be imported.")
```

```
# dynamic_memory_allocator.py
# Retry generation in fresh process with cudaMallocAsync if first attempt OOMs.
# This is the only reliable way to change the CUDA allocator at runtime.
import os
import sys
import gc
import pickle
import tempfile
import subprocess
import importlib
from typing import Callable, Any
def _resolve_factory(factory_path: str) -> Callable[[], Any]:
    """
    Resolves a factory string like "my_package.my_mod:build_model" to a callable.
    The callable must return ready-to-use model with .generate(request) method.
    """
    module_name, func_name = factory_path.split(":", 1)
    module = importlib.import_module(module_name)  # safe to import, no torch yet
    return getattr(module, func_name)
def generate_with_allocator_retry(
    model_factory_path: str,
    request_object: Any,
    *,
    allocator_conf: str = "backend:cudaMallocAsync"
) -> Any:
    """
    Attempts model.generate(request_object) in the current process.
    On torch.cuda.OutOfMemoryError, retries in a fresh subprocess with
    PYTORCH_ALLOC_CONF set to allocator_conf. request_object and the
    returned value must be picklable.
    """
    # Import torch only inside function; avoid importing at module import time.
    import torch
    model_factory = _resolve_factory(model_factory_path)
    model = model_factory()  # user-supplied function builds model, moves to GPU
    try:
        # First attempt uses whatever allocator current process started with.
        return model.generate(request_object)
    except torch.cuda.OutOfMemoryError:
        # Free as much as possible in the parent before spawning the child.
        # Avoids compounding pressure when two processes momentarily overlap.
        try:
            del model
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        # Serialize request to temp file and ask fresh interpreter to do work.
        with tempfile.TemporaryDirectory() as td:
            req_path = os.path.join(td, "request.pkl")
            out_path = os.path.join(td, "output.pkl")
            with open(req_path, "wb") as f:
                pickle.dump(request_object, f)
            # In the child, we want torch to see allocator config at import.
            env = os.environ.copy()
            env["PYTORCH_ALLOC_CONF"] = allocator_conf
            # Re-run this module as a helper child. The child will import torch
            # only after PYTORCH_ALLOC_CONF is set in its environment.
            cmd = [
                sys.executable,
                __file__,
                "--child",
                "--factory", model_factory_path,
                "--request", req_path,
                "--output", out_path,
            ]
            completed = subprocess.run(cmd, env=env, capture_output=True,
                                       text=True)
            if completed.returncode != 0:
                # Bubble up child stderr to aid debugging
                raise RuntimeError(
                  f"Retry failed with exit code {completed.returncode}\n"
                  f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
                )
            with open(out_path, "rb") as f:
                return pickle.load(f)
def _child_main(factory_path: str, request_path: str, output_path: str) -> None:
    """
    Child entrypoint: assumes PYTORCH_ALLOC_CONF is already present in env.
    Imports torch only now, builds the model, runs generate, and pickles result.
    """
    # Import torch after env var set by the parent’s subprocess.run(env=...).
    import torch
    model_factory = _resolve_factory(factory_path)
    model = model_factory()  # build the model inside the child process
    with open(request_path, "rb") as f:
        request_object = pickle.load(f)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--factory", type=str, default="")
    parser.add_argument("--request", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    if args.child:
        _child_main(args.factory, args.request, args.output)
    else:
        print("This module is intended to be imported.")
```

Here, the model is constructed in the child using a factory so that nothing CUDA-related is imported before the PYTORCH_ALLOC_CONF env variable takes effect. The code empties the cache and releases unused memory to the GPU using torch.cuda.empty_cache(). This pattern guarantees that the allocator configuration is applied before torch is imported in the child. It also avoids trying to unload a native extension at runtime, which CPython does not support.

这里，模型是在子进程中使用一个工厂函数构建的，从而确保在 PYTORCH_ALLOC_CONF 环境变量生效之前不会导入任何与 CUDA 相关的内容。代码使用 torch.cuda.empty_cache() 清空缓存并把未使用的内存归还给 GPU。这一模式保证了分配器配置会在子进程中导入 torch 之前被应用。它还避免了在运行时尝试卸载原生扩展——而这是 CPython 所不支持的。

In a non-PyTorch environment, such as a C++-based LLM inference engine, you can implement a pure slab allocator that allows configuration for specific allocation sizes. This type of slab allocator prepartitions memory into fixed-size “slabs.” It’s very efficient for repeated allocations of the same size and leads to virtually zero fragmentation for that specific size allocation.

在非 PyTorch 环境中，例如一个基于 C++ 的 LLM 推理引擎，你可以实现一个纯粹的 slab 分配器，允许针对特定分配大小进行配置。这类 slab 分配器会把内存预先划分为固定大小的“slab”。它对同一大小的重复分配非常高效，并且对该特定大小的分配几乎不产生碎片化。

In an LLM server, one very common technique is to allocate per-token output tensors such that each time you generate a token, you allocate a [layers, hidden_dim] tensor for that token’s activations, for instance. If those allocations are the same size every time, such as 64 KB, a slab for that exact size is ideal.

在 LLM 服务器中，一种非常常见的技术是分配每个 token 的输出张量，例如每生成一个 token，就为该 token 的激活分配一个 [layers, hidden_dim] 张量。如果这些分配每次都是相同大小，比如 64 KB，那么为这个确切大小设置一个 slab 就是理想之选。

The system detects that it’s allocating a lot of 64 KB tensors repeatedly—and creates a “slab” of dedicated 64 KB blocks. A slab allocator often does not return memory to the general pool until the entire slab is freed.

系统检测到自己在反复分配大量的 64 KB 张量——于是创建一个由专用 64 KB 块组成的“slab”。slab 分配器通常不会把内存归还给通用内存池，直到整个 slab 被释放为止。

> Modern LLM inference engines perform this type of buffer reuse for each generated token. This is in contrast to freeing and reallocating the memory each time, which would incur a high amount of overhead—especially at token granularity.

> 现代 LLM 推理引擎会为每个生成的 token 执行这类缓冲区复用。这与每次都释放并重新分配内存的做法形成对比，后者会带来大量开销——尤其是在 token 粒度上。

Allocator switching might involve completely different allocators for different parts of the system. For instance, you can use the default caching allocator with large static allocations for model weights since those are freed less often during inference. You can then use a custom allocator for ephemeral per-token allocations.

分配器切换可能会为系统的不同部分使用完全不同的分配器。例如，你可以对模型权重使用带大块静态分配的默认缓存分配器，因为在推理期间这些内存很少被释放。然后，你可以对每个 token 的临时分配使用自定义分配器。

> It’s recommended to architect your code to separate long-lived allocations, such as model weights, from short-lived allocations, such as token buffers. This makes it easier to direct the short-lived and long-lived allocations to different allocators and memory pools.

> 建议将你的代码架构为把长生命周期分配（如模型权重）与短生命周期分配（如 token 缓冲区）分离开。这样更容易将短生命周期与长生命周期的分配分别导向不同的分配器和内存池（memory pool）。

You should always implement a fallback strategy for OOM errors. Many production systems use multitier memory such that if the GPU experiences an OOM error, it will offload to the CPU and try again rather than failing the request. For example, you can dynamically free the cache, offload some layers to the CPU, or compress the cache.

你应当始终为 OOM 错误实现一套回退策略。许多生产系统采用多层内存，使得当 GPU 出现 OOM 错误时，它会卸载到 CPU 并重试，而不是让请求失败。例如，你可以动态释放缓存、把部分层卸载到 CPU，或压缩缓存。

In short, techniques like dynamic allocator management and multitiered memory make sure your system can handle long uptimes under different types of load—all without incurring memory fragmentation or allocation latency spikes. This is a behind-the-scenes optimization that users won’t directly see, but it’s essential for ultrascale robustness for long-running inference servers.

简而言之，像动态分配器管理与多层内存这样的技术，能确保你的系统在不同类型的负载下都能应对长时间的运行——而且不会引发内存碎片化或分配延迟尖峰。这是一种用户不会直接看到的幕后优化，但它对于长期运行的推理服务器实现超大规模鲁棒性至关重要。

## Runtime Kernel Performance Improvements and Hot-Swappable Implementations

## 运行时核函数性能改进与可热插拔实现

In the fast-evolving world of GPU hardware and algorithmic innovations, new and faster kernel implementations are constantly emerging. This includes newer variants of FlashAttention, megakernels, and hardware-specific software optimizations.

在飞速演进的 GPU 硬件与算法创新世界里，更新、更快的核函数（kernel）实现不断涌现。这包括 FlashAttention 的更新变体、megakernel，以及特定于硬件的软件优化。

Runtime kernel patching is the ability to integrate these new implementations into a running system without requiring a full redeployment or reload of the model. Essentially, we want to hot-swap a slower kernel function for a faster one on the fly.

运行时核函数打补丁（runtime kernel patching）是指能够将这些新实现集成进正在运行的系统，而无需完整重新部署或重新加载模型。本质上，我们希望把一个较慢的核函数即时热插拔（hot swap）成一个更快的。

Consider your inference server that uses the default PyTorch scaled dot product attention (SDPA) kernel for multiheaded attention. You then discover a new kernel implementation like FlashAttention-3, which gives a 20% speed boost for long sequences in some cases.

设想你的推理服务器使用默认的 PyTorch scaled dot product attention（SDPA）核函数来做多头注意力。随后你发现了一个新的核函数实现，比如 FlashAttention-3，它在某些情况下对长序列能带来 20% 的加速。

Traditionally, you’d have to install the updated library and restart the server to use the new implementation. But with runtime patching, you can dynamically load and redirect calls to the new kernel during runtime without interrupting the server’s uptime.

传统上，你必须安装更新后的库并重启服务器才能使用新实现。但借助运行时打补丁，你可以在运行时动态加载并把调用重定向到新核函数，而不中断服务器的运行。

This zero-downtime upgrade approach is crucial in 24/7 services in which a restart or model reload would incur too much latency or cause an outage. In Python, this can be an easy monkey patch, as shown here:

这种零停机（zero-downtime）升级方式对于 24/7 服务至关重要，在这类服务中，一次重启或模型重新加载会带来过高的延迟或造成宕机。在 Python 中，这可以是一次简单的猴子补丁（monkey patch），如下所示：

```
import new_flash_attn_lib
# Monkey-patch the model's attention forward to use the new library
old_attn_forward = model.transformer.self_attn.forward
def new_attn_forward(self, *args, **kwargs):
    return new_flash_attn_lib.forward(*args, **kwargs)
model.transformer.self_attn.forward =
new_attn_forward.__get__(
model.transformer.self_attn,
type(model.transformer.self_attn))
```

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

This code replaces the forward method of the attention module with one that calls our new_flash_attn_lib.forward. We bind it to the instance (__get__) to simulate a proper method. After this patch, subsequent calls to that attention layer will go through the new implementation. As such, we have effectively hot-swapped the kernel.

这段代码用一个会调用我们的 new_flash_attn_lib.forward 的方法替换了注意力模块的 forward 方法。我们把它绑定到实例（__get__）以模拟一个正规的方法。打完这个补丁后，对该注意力层的后续调用都会走新实现。这样，我们就有效地热插拔了这个核函数。

> Make sure that new_flash_attn_lib.forward() is drop-in compatible—and that it has been thoroughly tested to produce identical outputs within acceptable numerical tolerances. This way, you avoid any model-quality regressions.

> 务必确保 new_flash_attn_lib.forward() 是可直接替换的（drop-in），并且已经经过充分测试，能在可接受的数值容差内产生完全一致的输出。如此，你才能避免任何模型质量的退化。

Another technique is JIT patching, which uses a just-in-time compiler like PyTorch Inductor or OpenAI Triton to generate a faster kernel at runtime—and then plugging it into the inference pipeline. As shown next, one can use PyTorch’s torch.compile to optimize a function and return its compiled version:

另一种技术是 JIT 打补丁，它使用像 PyTorch Inductor 或 OpenAI Triton 这样的即时编译器在运行时生成更快的核函数——然后将其插入推理流水线。如下所示，可以用 PyTorch 的 torch.compile 来优化一个函数并返回其编译后的版本：

```
compiled_forward =
    torch.compile(model.transformer.self_attn.forward,
                  backend="inductor")
model.transformer.self_attn.forward = compiled_forward
```

```
compiled_forward =
    torch.compile(model.transformer.self_attn.forward,
                  backend="inductor")
model.transformer.self_attn.forward = compiled_forward
```

Now, whenever forward is called, it will execute the optimized code, which will fuse multiple operations, etc. If we do this after model initialization, it’s a form of hot patching since it does not require a model reload. It simply swaps out function pointers.

现在，每当 forward 被调用时，它都会执行优化后的代码，这会融合多个算子等。如果我们在模型初始化之后做这件事，这就是一种热补丁（hot patching），因为它不需要重新加载模型。它只是替换掉了函数指针。

On the CUDA side, runtime module loading is possible using CUDA driver APIs. For instance, you can compile a custom PTX for a new kernel—and load it into the context without resetting the device. If the function signatures match, the system would just update the function pointers. Tools like NVIDIA Runtime Compilation (NVRTC) can compile PTX from strings at runtime, enabling this workflow.

在 CUDA 一侧，运行时模块加载可以通过 CUDA 驱动 API 实现。例如，你可以为一个新核函数编译自定义 PTX——并在不重置设备的情况下把它加载进上下文。如果函数签名匹配，系统只需更新函数指针即可。像 NVIDIA Runtime Compilation（NVRTC）这样的工具可以在运行时从字符串编译 PTX，从而启用这一工作流。

Doing this with CUDA C++ is a bit complex. Higher-level approaches like Python’s monkey-patch mechanism—or PyTorch’s extensible backends—are likely more practical and maintainable in the long term. However, your high-performance inference engine is likely not using a Python/PyTorch runtime.

用 CUDA C++ 来做这件事略显复杂。像 Python 的猴子补丁机制——或 PyTorch 的可扩展后端——这类更高层的方法，从长远看很可能更实用、也更易维护。不过，你的高性能推理引擎很可能并不使用 Python/PyTorch 运行时。

> When doing this type of hot swapping, it’s important to guarantee thread safety. For example, you should drain the incoming-request queue or use a barrier to make sure that no other thread is in the middle of the function you’re patching. This will avoid race conditions with running threads. This is similar to how OS kernels load modules—careful synchronization is needed, but it avoids full restarts.

> 在做这类热插拔时，保证线程安全很重要。例如，你应当排空进入的请求队列，或使用一个屏障（barrier）来确保没有其他线程正处在你要打补丁的函数中途。这样可以避免与正在运行的线程发生竞态条件。这类似于操作系统内核加载模块的方式——需要小心的同步，但它避免了完整的重启。

Using well-defined module boundaries—like separating attention and MLP kernels—makes it easier to swap internals. This argues for writing your model code in a highly modular way so that you have these swap points. Monolithic model implementations are much harder to patch, but they can be more performant (e.g., megakernels).

使用界定良好的模块边界——比如把注意力核函数和 MLP 核函数分开——可以让替换内部实现更容易。这也主张以高度模块化的方式编写你的模型代码，以便拥有这些可替换点。单体式（monolithic）模型实现要打补丁困难得多，但它们可能更高性能（例如 megakernel）。

Of course, one must ensure that the new kernel produces identical—or acceptably close—results to the old implementation. Typically these optimized kernels are designed to be numerically equivalent at the same precision (e.g., FP16).

当然，必须确保新核函数产生的结果与旧实现一致——或足够接近。通常这些优化过的核函数被设计为在相同精度（例如 FP16）下数值等价。

Hot patching can also be used for quick fixes. Perhaps a particular sequence of operations is causing a known bug in the current kernel. Instead of waiting for a full update, you could quickly patch in a fix.

热补丁也可用于快速修复。也许某个特定的算子序列正在当前核函数中触发一个已知 bug。与其等待一次完整的更新，你可以快速打入一个修复。

Consider deploying an FP8 kernel that starts to misbehave for certain edge inputs that you didn’t anticipate. If you have a safer but slightly slower version, you can detect the misbehaving condition and hot swap in the safe kernel in those cases. Make sure you log these events for future analysis.

设想你部署了一个 FP8 核函数，它对某些你未曾预料到的边缘输入开始出现异常行为。如果你有一个更安全但稍慢的版本，你可以检测到这种异常状况，并在这些情况下热插拔进那个安全核函数。务必记录这些事件以供日后分析。

> Consider a phased rollout by routing a small percentage of traffic to the new kernel (shadow testing or canary). You can then use telemetry to compare the latency and throughput metrics against the old kernel. If the new kernel is performing as expected and its output matches the expected results, you can release it to all users.

> 可以考虑分阶段上线，把一小部分流量路由到新核函数（影子测试或金丝雀发布）。然后你可以用遥测数据把延迟和吞吐量指标与旧核函数做对比。如果新核函数表现符合预期，且其输出与预期结果匹配，你就可以把它发布给所有用户。

To manage multiple possible implementations, an engine might maintain a registry, such as attention_impl = "fast" or "safe"—and branch accordingly. This can be implemented as a feature flag so that you can toggle implementations without code changes. This is useful for quick rollbacks simply by changing the flag (just make sure your code is actively reading the values from the feature-flag system—otherwise the flag value remains static, which defeats the purpose of a feature flag).

为了管理多个可能的实现，引擎可以维护一个注册表，例如 attention_impl = "fast" 或 "safe"——并据此分支。这可以实现为一个功能开关（feature flag），使你无需改动代码就能切换实现。这对于仅通过改动开关来快速回滚很有用（只要确保你的代码正在主动从功能开关系统读取这些值——否则开关值会保持静态，那就失去了功能开关的意义）。

A well-designed inference engine should also measure runtime performance of the new kernel. If the new kernel is not faster with live production data, the inference engine can roll it back. This can be due to slightly different hardware, untested batch sizes, or untested load conditions.

设计良好的推理引擎还应当测量新核函数的运行时性能。如果新核函数在真实生产数据下并不更快，推理引擎可以将其回滚。这可能是由于硬件略有差异、未测试过的 batch size，或未测试过的负载状况。

This is a perfect form of autotuning, described in the previous section. If the autotuner finds a faster implementation that outperforms the current default implementation beyond some threshold, the system could trigger a “patch” to promote the faster implementation to the default going forward.

这正是前一节所述自动调优（autotuning）的一种完美形式。如果自动调优器发现了一个超过某个阈值、优于当前默认实现的更快实现，系统就可以触发一次“补丁”，把更快的实现提升为往后的默认实现。

This effectively closes the loop with the autotuner. The system not only finds better kernels on the fly, but it also swaps them in. This leads to a self-optimizing kernel-selection mechanism.

这实际上与自动调优器闭环。系统不仅能即时发现更好的核函数，还能把它们替换进来。这就带来了一种自优化的核函数选择机制。

You can combine this strategy with the previous RL strategy we discussed. This way, you can have multiple implementations loaded—and let the RL agent choose which one to use for each different type of request.

你可以把这一策略与我们前面讨论过的 RL 策略结合起来。这样，你可以同时加载多个实现——并让 RL 智能体为每一种不同类型的请求选择使用哪一个。

> Be mindful of the overhead needed to maintain multiple implementations in memory. The added code size should not evict other critical code from the instruction cache—or exhaust GPU memory.

> 要留意在内存中维护多个实现所需的开销。新增的代码体积不应把其他关键代码从指令缓存中挤出——也不应耗尽 GPU 内存。

For instance, a highly optimized kernel might be best for long sequences, but a simpler kernel implementation might be fine for short sequences. A runtime system could choose between them per request. This effectively patches on a per-request basis based on the length of the input sequence.

例如，一个高度优化的核函数对长序列可能最优，但一个更简单的核函数实现对短序列可能就够了。运行时系统可以按请求在它们之间做选择。这实际上是基于输入序列长度按请求逐个打补丁。

In short, runtime kernel patching is about flexibility. It acknowledges that what is optimal today might change tomorrow and provides the means to adapt quickly. This type of agility makes sure that your serving infrastructure keeps up with the rapid advancements in model acceleration techniques. Popular inference engines like vLLM, SGLang, NVIDIA TensorRT-LLM, and NVIDIA Dynamo are well architected and provide a good amount of dynamic capabilities. For example, TensorRT-LLM allows you to load optimized kernels using runtime plugins. Use these capabilities to your advantage.

简而言之，运行时核函数打补丁关乎灵活性。它承认今天最优的东西明天可能会变，并提供了快速适应的手段。这种敏捷性能确保你的服务基础设施跟得上模型加速技术的飞速进步。像 vLLM、SGLang、NVIDIA TensorRT-LLM 和 NVIDIA Dynamo 这样流行的推理引擎架构良好，提供了相当多的动态能力。例如，TensorRT-LLM 允许你使用运行时插件加载优化过的核函数。请善用这些能力为你所用。

## Continuous Prewarming of CUDA Graphs and Caches Using Time-Series Prediction

## 使用时间序列预测对 CUDA Graphs 和缓存进行持续预热

In high-throughput inference situations, cold-start overheads can be a large contributor to request-response latency. CUDA Graphs, as described in Chapter 12, allow you to capture a sequence of GPU operations and replay the sequence with minimal launch overhead.

在高吞吐推理场景中，冷启动开销可能是请求-响应延迟的一大来源。CUDA Graphs（如第 12 章所述）允许你捕获一段 GPU 操作序列，并以极小的启动开销重放该序列。

Prewarming a CUDA Graph means setting up the graph before it’s actually needed. If we can predict when a certain request type or batch size will occur, the prewarmed CUDA Graph will be ready to execute.

预热（prewarming）一个 CUDA Graph 意味着在真正需要它之前就把这张图搭建好。如果我们能预测某种请求类型或某个 batch size 何时会出现，预热好的 CUDA Graph 就已准备好执行。

This technique relies on prediction accuracy. If your forecast is off, you might prewarm a graph that isn’t used. You should monitor prediction hit rates to make sure the optimization is paying off.

这项技术依赖预测的准确性。如果你的预报有偏差，你可能会预热一张用不上的图。你应当监控预测命中率，以确保这项优化确有回报。

By prewarming, the graph can skip costly launch and initialization steps when the request actually arrives. Using time-series prediction algorithms like ARIMA and Prophet, the system anticipates future workload patterns, including traffic surges and batch size changes, to proactively prepare the graph for fast execution.

通过预热，当请求真正到来时，图可以跳过代价高昂的启动与初始化步骤。系统使用像 ARIMA 和 Prophet 这样的时间序列预测（time-series prediction）算法来预判未来的负载模式，包括流量激增和 batch size 变化，从而主动为快速执行准备好图。

Consider an inference service that observes a daily cycle when, at 9 A.M. every day, there’s a spike in user traffic due to time zone–related traffic patterns. Knowing this, the system could prerun a few requests just before 9 A.M. to load the model into the GPU caches, JIT-compile the necessary kernels, and capture the CUDA Graphs for the expected batch sizes. At 9 A.M., when the real traffic load spikes, the incoming requests will reuse the warmed state for the expected batch size and usage pattern.

设想一个推理服务观察到一个每日周期：每天早上 9 点，由于与时区相关的流量模式，用户流量会出现一次尖峰。了解这一点后，系统可以在 9 点之前预先运行几个请求，把模型加载进 GPU 缓存、即时编译（JIT）所需的核函数，并为预期的各个 batch size 捕获 CUDA Graphs。到了 9 点，当真实流量负载激增时，进入的请求就会复用为预期 batch size 和使用模式预热好的状态。

In practice, this can be orchestrated by a cron-like job or a scheduling service that triggers the prewarm routine right before the traffic spike is expected to occur. Just remember to leave enough time for the resources to provision.

在实践中，这可以由一个类 cron 的作业或一个调度服务来编排，在预计流量尖峰即将到来之前触发预热例程。只需记得为资源的置备留出足够的时间。

In PyTorch, wrapping your model’s computation within a torch.cuda.graph() context allows you to capture a static computation graph that includes GPU kernel launches. When replaying the graph, PyTorch bypasses the Python-to-CUDA dispatch overhead and submits the entire workload with a single cudaGraphLaunch. This leads to significantly faster execution—especially for large, repetitive input batches due to the minimal CPU involvement.

在 PyTorch 中，把模型的计算包裹在 torch.cuda.graph() 上下文里，可以让你捕获一张包含 GPU 核函数启动的静态计算图。重放这张图时，PyTorch 会绕过 Python 到 CUDA 的分派开销，用单次 cudaGraphLaunch 提交整个工作负载。这会带来显著更快的执行——尤其是对大而重复的输入批次，因为 CPU 的参与极少。

The downside is that these graphs are static since they don’t easily allow variable shapes or lengths. But an inference server often deals with distinct batch sizes (1, 2, 4, 8, 16, etc.)—due to GPU memory limitations and algorithmic optimizations—which fits the graph model well.

其缺点在于这些图是静态的，因为它们不容易允许可变的形状或长度。但推理服务器往往处理的是各不相同的 batch size（1、2、4、8、16 等）——由于 GPU 内存限制和算法优化——这与图模型很契合。

To handle variability, you can maintain a pool of precaptured graphs for each common batch size or sequence length. You could use a continuous prewarming strategy to prepare the pool of graphs for these distinct batch sizes. For instance, a batch of 16 requests is common during peak hours. The server can capture a graph of the model’s forward pass upfront using a batch size of 16—and then store it for subsequent use.

为应对可变性，你可以为每个常见的 batch size 或序列长度维护一个预捕获图的池。你可以使用持续预热（continuous prewarming）策略来为这些不同的 batch size 准备好图池。例如，在高峰时段，16 个请求组成的批次很常见。服务器可以预先用 16 的 batch size 捕获模型前向传播的一张图——然后把它存起来供后续使用。

The next time a batch of 16 requests comes in simultaneously, the inference system feeds the batched inputs into the precaptured graph. The kernels inside are already prewarmed and optimized for graph execution, so there’s no need to enqueue and launch many individual operations, as just one graph launch is all that’s needed.

下一次同时有 16 个请求进来时，推理系统就把这批输入喂进预捕获好的图。图里的核函数已经为图执行预热并优化好了，因此无需入队并启动许多单独的操作，只需一次图启动就够了。

Keep in mind that each captured graph stored in the pool will consume additional GPU memory for its workspace. You should monitor memory usage when storing many graphs. It might be necessary to evict less-used graphs from the pool if memory gets tight.

请记住，池中存储的每一张已捕获的图都会为其工作区消耗额外的 GPU 内存。存储许多图时你应当监控内存使用。如果内存吃紧，可能有必要把使用较少的图从池中逐出（evict）。

To avoid the extra memory of a pool, you can potentially use graph patching, discussed in Chapter 12, to adjust graph nodes for minor size differences. However, in practice, a pool is a better option for performance.

为避免一个池带来的额外内存开销，你或许可以使用第 12 章讨论过的图打补丁（graph patching），来针对细微的尺寸差异调整图节点。不过在实践中，就性能而言，用池是更好的选择。

CUDA Graphs also reduce CPU overhead since it coordinates just a single graph launch versus many individual kernel launches. This lets the CPU perform other operations like data preprocessing and other types of “real” work—instead of coordinating many kernel launches.

CUDA Graphs 还能降低 CPU 开销，因为它只需协调单次图启动，而不是许多单独的核函数启动。这让 CPU 可以去做数据预处理等其他类型的“真正”工作——而不是去协调许多核函数启动。

Again, you can use time-series prediction algorithms (e.g., ARIMA and Prophet) to forecast metrics like RPS and average prompt length. If the model predicts a jump in batch size or a particular pattern of requests, such as long input sequences, the system can start preparing and prewarming the appropriate CUDA Graphs, caches, and other resources. For instance, the system can proactively increase the batch size of the continuous batching algorithm, prefetch model weights from disk to GPU memory, and allocate additional GPU instances.

同样地，你可以使用时间序列预测算法（例如 ARIMA 和 Prophet）来预报诸如 RPS 和平均提示长度这类指标。如果模型预测到 batch size 会跳升，或出现某种特定的请求模式（例如长输入序列），系统就可以开始准备并预热相应的 CUDA Graphs、缓存和其他资源。例如，系统可以主动增大连续批处理算法的 batch size、把模型权重从磁盘预取到 GPU 内存，并分配额外的 GPU 实例。

> It’s important to retrain and update these time-series models frequently with recent traffic data. This is because usage patterns can change over time due to new user segments coming online, etc. In addition, unanticipated “holidays” like California’s famous Ski Week can unexpectedly increase the load curve—as it did for me at Netflix when I first moved to California!

> 重要的是要经常用近期的流量数据来重新训练并更新这些时间序列模型。这是因为使用模式会随时间变化，例如有新的用户群上线等。此外，未曾预料的“节假日”，比如加州著名的滑雪周（Ski Week），会出人意料地抬高负载曲线——当我刚搬到加州时，在 Netflix 就遇到过这种情况！

Related to caching and prewarming is anticipating the scale-out of prefill and decode workers. If we know a bunch of requests with long prompts are likely to arrive at a certain time due to a scheduled batch job or daily report scenario, the system can scale out the prefill workers and execute a representative forward pass with representative sample data to prewarm the CUDA Graphs and KV cache.

与缓存和预热相关的一点，是预判 prefill 和 decode worker 的横向扩容。如果我们知道由于一个计划中的批处理作业或每日报表场景，一批带长提示的请求很可能在某个时间到来，系统就可以横向扩容 prefill worker，并用有代表性的样本数据执行一次有代表性的前向传播，以预热 CUDA Graphs 和 KV 缓存。

The scale-out and prewarming events should also include the decode workers and CUDA Graphs. In the decode case, the speculative decoding draft model, for instance, can be prewarmed and loaded into GPU memory—as well as the draft model’s KV cache, etc. Other decode optimizations include loop unrolling for a fixed number of tokens that will be generated—either 1 token or multiple in the case of speculative decoding.

横向扩容和预热事件也应当包含 decode worker 和 CUDA Graphs。以 decode 为例，推测解码的草稿模型就可以被预热并加载进 GPU 内存——草稿模型的 KV 缓存等也是如此。其他 decode 优化包括针对将要生成的固定 token 数做循环展开——无论是 1 个 token，还是推测解码情况下的多个 token。

You should also consider caching the CUDA kernels themselves. CUDA will often cache the compiled kernels (C++ code => PTX instructions => SASS assembly) to speed up execution. The first time a kernel runs, there is likely some on-the-fly JIT compilation overhead.

你还应当考虑缓存 CUDA 核函数本身。CUDA 通常会缓存编译好的核函数（C++ 代码 => PTX 指令 => SASS 汇编）以加速执行。核函数首次运行时，很可能会有一些即时（JIT）编译开销。

You should coordinate this type of warm-up with your cluster autoscaler such that when new GPU instances spin up, the autoscaler runs a quick set of inferences on them using a few warm-up API calls. Doing this will validate the inference engines, cache JIT-compilation outputs, allocate memory pools, prepare CUDA Graphs, etc. This way, the engines are production-ready before adding them to the live traffic pool.

你应当把这类预热与你的集群自动扩缩器（autoscaler）协调起来，使得当新的 GPU 实例启动时，自动扩缩器用几次预热 API 调用在它们上面跑一小组推理。这样做会验证推理引擎、缓存 JIT 编译输出、分配内存池、准备 CUDA Graphs 等。如此一来，这些引擎在被加入实时流量池之前就已经处于生产就绪状态。

> Use your knowledge of the system to identify and invoke as many distinct paths as possible during this controlled warm-up phase. This includes every batch size, CUDA Graph variant, etc.

> 利用你对系统的了解，在这个受控的预热阶段尽可能识别并触发尽量多的不同路径。这包括每一个 batch size、每一种 CUDA Graph 变体等。

You should also monitor that the warm-up actually helps by comparing the latency of the first few requests during the predicted surge—both with and without warm-up. Adjust the timing and threshold as needed. And be aware that the time-series prediction can be wrong. Make sure that warm-up tasks do not impact overall inference performance if they run at the wrong time.

你还应当通过对比预测尖峰期间头几个请求的延迟——分别在有预热和无预热两种情况下——来监控预热是否真的有帮助。按需调整时机和阈值。并且要意识到时间序列预测可能出错。务必确保预热任务在错误的时机运行时不会影响整体推理性能。

For instance, you should try to perform prewarming runs (with example data) when the GPUs are underutilized. CUDA allows stream priority scheduling so you can assign prewarm streams to a lower priority than your inference stream. This way, the prewarming does not compete for resources with the live, high-volume traffic requests.

例如，你应当尽量在 GPU 利用率不高时执行预热运行（用示例数据）。CUDA 允许按 stream 优先级调度，因此你可以把预热 stream 的优先级设得低于你的推理 stream。这样，预热就不会与实时的大流量请求争抢资源。

Also, you should use lower-priority CUDA streams for these prewarming tasks, so they will yield to higher-priority live traffic that may arrive during prewarming. On the plus side, idle GPUs are a wasted opportunity, so doing warm-up computations on idle GPUs is essentially free—as long as it doesn’t collide with real work.

此外，你应当为这些预热任务使用较低优先级的 CUDA stream，这样在预热期间若有更高优先级的实时流量到来，它们就会让位。好的一面是，空闲的 GPU 是一种被浪费的机会，因此在空闲 GPU 上做预热计算基本上是免费的——只要它不与真正的工作发生冲突。

Grace Blackwell systems allow some additional tricks since the CPU and GPU share unified, coherent memory at low latency. For instance, you can have the CPU start prefilling data into unified memory that the GPU plans to use. This can avoid explicit GPU copy calls later.

Grace Blackwell 系统允许一些额外的技巧，因为 CPU 和 GPU 以低延迟共享统一、一致的内存。例如，你可以让 CPU 开始把数据预填进 GPU 计划要用的统一内存里。这可以避免之后显式的 GPU 拷贝调用。

Continuous prewarming—guided by time series predictions—can make latency far more predictable. It turns the inference engine into an adaptive system that learns common traffic patterns, automatically prepares the data, readies the hardware, and stays a step ahead of demand. This will reduce jitter and decrease latency spikes by smoothing out expensive operations like compilation and memory transfers at times when they can be amortized.

由时间序列预测引导的持续预热，能让延迟变得可预测得多。它把推理引擎变成一个自适应系统，能学习常见的流量模式、自动准备数据、备好硬件，并始终领先于需求一步。这会通过在可以摊销的时机把编译和内存传输这类昂贵操作平滑掉，从而减少抖动并降低延迟尖峰。

This is especially valuable for large models that have heavy one-time initialization costs. As models and contexts grow, such adaptive preloading will move from a nice-to-have to a necessity in production LLM systems. Paying these costs predictively is much better than having end users churn because of a poor experience.

这对于有着沉重一次性初始化成本的大型模型尤其有价值。随着模型和上下文的增长，这类自适应预加载将从“锦上添花”变为生产 LLM 系统中的“必需品”。以预测性的方式支付这些成本，远好于因糟糕体验而让终端用户流失。

## Adaptive Batching and Chunked Prefill Scheduling

## 自适应批处理与分块 prefill 调度

In Chapter 16, we discussed how modern inference servers use different types of request batching (e.g., continuous batching) to maximize throughput and minimize latency across all requests. However, batching can increase latency for individual requests. The trade-offs can be dynamically addressed as conditions change throughout the day using a technique called *adaptive batching*.

在第 16 章中，我们讨论过现代推理服务器如何使用不同类型的请求批处理（例如连续批处理，continuous batching）来在所有请求间最大化吞吐、最小化延迟。然而，批处理会增加单个请求的延迟。可以用一种叫做 *自适应批处理（adaptive batching）* 的技术，随着一天中条件的变化动态地处理这些权衡。

Adaptive batching dynamically adjusts how requests are grouped into batches depending on the load—and how well the requests are progressing. This type of dynamic strategy can adjust the batch size and threshold parameters in real time as the environment changes.

自适应批处理会根据负载——以及请求进展得如何——动态调整请求被分组成批的方式。这类动态策略可以在环境变化时实时调整 batch size 和阈值参数。

For example, during peak load, the system can use a large batch size (e.g., 8 or 16) because throughput is critical. During periods of low load, the system can reduce the batch size to serve the requests sooner. This will prioritize latency over throughput.

例如，在高峰负载时，系统可以使用较大的 batch size（例如 8 或 16），因为此时吞吐量至关重要。在低负载时段，系统可以减小 batch size 以更快地服务请求。这会把延迟置于吞吐量之上优先考虑。

To decide on the batch size, you can use a simple heuristic, such as, *“If GPU utilization is > 80%, allow larger batches; if < 20%, use batch size 1 to minimize latency.”* Or you can use a more sophisticated RL agent or predictive strategy, as discussed in the previous sections.

要决定 batch size，你可以使用一个简单的启发式规则，例如 *“如果 GPU 利用率 > 80%，允许更大的批次；如果 < 20%，使用 batch size 1 以最小化延迟。”* 或者你可以使用前几节讨论过的更复杂的 RL 智能体或预测式策略。

This difference in arithmetic intensity between the prefill and decode phases leads to mismatched durations of execution. As such, it’s best to disaggregate the stages and treat them as separate workloads that can be tuned independently using separate threads/processes for a single node or worker pools for a multinode cluster.

prefill 和 decode 两个阶段在算术强度上的这种差异，会导致执行时长的不匹配。因此，最好把这两个阶段拆解开，将它们视为可以独立调优的、彼此分离的工作负载——对单节点用分离的线程/进程，对多节点集群用 worker 池。

By disaggregating prefill and decode, we treat the two stages as separate operations. This allows them to be independently optimized for their unique compute and memory bandwidth needs. One of these optimizations is the batch sizes used for the prefill and decode phases.

通过拆解 prefill 和 decode，我们把这两个阶段当作彼此分离的操作。这使得它们可以针对各自独特的计算与内存带宽需求独立优化。其中一项优化就是 prefill 和 decode 两个阶段所用的 batch size。

In practice, vLLM and other modern inference engines do exactly this. They form separate batches to send to the prefill and decode workers. As such, a batch of prefill requests can execute independently of the batch of decode requests. For example, the decode phase can benefit from larger batches to increase arithmetic intensity since it’s a memory-bound workload.

在实践中，vLLM 和其他现代推理引擎正是这么做的。它们组成分离的批次分别发给 prefill worker 和 decode worker。这样，一批 prefill 请求就可以独立于一批 decode 请求执行。例如，decode 阶段可以从更大的批次中获益以提高算术强度，因为它是一个内存受限（memory-bound）的工作负载。

Modern inference-serving frameworks like vLLM use adaptive scheduling loops to dynamically choose between processing a prefill or a decode batch. Specifically, vLLM supports chunked prefill and decode-maximal scheduling to interleave prefill and decode for better utilization. These techniques boost overall utilization and throughput without adding significant latency.

像 vLLM 这样的现代推理服务框架使用自适应调度循环，在处理一个 prefill 批次还是一个 decode 批次之间动态选择。具体来说，vLLM 支持分块 prefill（chunked prefill）和 decode 最大化（decode-maximal）调度，以交错 prefill 与 decode 来获得更好的利用率。这些技术在不显著增加延迟的前提下提升了整体利用率和吞吐量。

The mismatched system-resource characteristics of prefill and decode can impact PP as well. Consider one microbatch doing prefill on a long sequence while another microbatch is performing decode one token at a time. In this case, their durations are mismatched, and pipeline bubbles emerge.

prefill 和 decode 在系统资源特性上的不匹配也会影响 PP。设想一个微批在一条长序列上做 prefill，而另一个微批正在一次一个 token 地做 decode。在这种情况下，它们的时长不匹配，流水线气泡就会出现。

You can interleave large prefill requests with latency-sensitive decode tasks by slicing the prefill into small chunks and piggybacking decodes between them. This keeps all pipeline stages busy and minimizes idle “bubbles” in your GPU schedule.

你可以把大的 prefill 请求切成小块，并在它们之间夹带（piggyback）decode，从而将大 prefill 请求与延迟敏感的 decode 任务交错起来。这能让所有流水线阶段都保持忙碌，并把 GPU 调度中空闲的“气泡”降到最低。

*Chunked prefill* is a well-supported pattern used by all modern LLM inference engines to reduce pipeline bubbles. It effectively time-slices a big task (prefill) to create room for small tasks (decode) to execute in the pipeline gaps created by the chunks, as shown in Figure 19-7.

*分块 prefill（chunked prefill）* 是所有现代 LLM 推理引擎都很好支持的一种模式，用于减少流水线气泡。它有效地对一个大任务（prefill）做时间切片，为小任务（decode）腾出空间，让它们在由分块所制造的流水线间隙中执行，如图 19-7 所示。

![Figure 19-7. Benefits of chunked prefills for decode-maximal batching across four requests](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-7.png)

![图 19-7. 分块 prefill 在四个请求间实现 decode 最大化批处理的收益](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-7.png)

The SARATHI paper demonstrated that this type of chunked prefill and piggybacking can help you find the right level of *decode-maximal batching*, reduce bubbles, and improve throughput by ~1.3–1.9× compared to naive scheduling. The name SARATHI is a reference to a charioteer that intelligently steers both prefill and decode tasks together. Fun!

SARATHI 论文证明，这类分块 prefill 与夹带的做法能帮助你找到合适的 *decode 最大化批处理（decode-maximal batching）* 水平、减少气泡，并相比朴素调度把吞吐量提升约 1.3–1.9×。SARATHI 这个名字取自一位驭者（charioteer），他能把 prefill 和 decode 两类任务一起智慧地驾驭。有趣吧！

For example, consider a 10,000-token prefill request that does not use chunking. In this case, the single prefill pass will block the entire pipeline and cause decode tasks to queue up until the prefill completes.

举例来说，设想一个不使用分块的 10,000-token prefill 请求。在这种情况下，单次 prefill 遍历会阻塞整条流水线，导致 decode 任务排队等待，直到 prefill 完成。

However, if you use chunked prefill and divide the 10,000-token prefill request into five 2,000-token chunks, you can interleave decode batches in between prefill chunks to keep the GPU busy processing both phases and moving things forward. This will squeeze out pipeline bubbles, improve throughput, and smooth out GPU utilization.

然而，如果你使用分块 prefill，把这个 10,000-token 的 prefill 请求切成五个 2,000-token 的块，你就可以在各 prefill 块之间交错 decode 批次，让 GPU 忙于同时处理两个阶段并推动进度。这会挤掉流水线气泡、提高吞吐量，并平滑 GPU 利用率。

> A rule of thumb is to choose a chunk size such that a prefill chunk takes ~50–100 ms. This way, you have frequent opportunities to schedule decode batches in between. This may correspond to a few thousand tokens depending on the model architecture/size and GPU hardware.

> 一条经验法则是选择这样一个块大小：让一个 prefill 块耗时 ~50–100 ms。这样，你就有频繁的机会在其间调度 decode 批次。根据模型架构/规模和 GPU 硬件的不同，这可能对应几千个 token。

Modern inference engines like vLLM use adaptive scheduling loops to decide whether to process another prefill chunk—or perform a decode batch—based on GPU utilization and queue status. Specifically, vLLM continuously monitors token queues to make these decisions. vLLM’s scheduler explicitly supports chunked prefill and decode-maximal batching. Its executor and chunked-prefill features are designed to overlap large prefills with smaller interactive decodes.

像 vLLM 这样的现代推理引擎使用自适应调度循环，基于 GPU 利用率和队列状态来决定是处理下一个 prefill 块——还是执行一个 decode 批次。具体来说，vLLM 持续监控 token 队列来做这些决策。vLLM 的调度器显式支持分块 prefill 和 decode 最大化批处理。它的执行器和分块 prefill 特性被设计为把大的 prefill 与较小的交互式 decode 重叠起来。

An adaptive scheduler needs to consider GPU shared-memory limits and occupancy when choosing a chunked prefill size. A simple adaptive chunked prefill implementation is shown next. This code dynamically right-sizes the chunk size to keep SM and occupancy high on the GPU:

自适应调度器在选择分块 prefill 大小时，需要考虑 GPU 共享内存限制和占用率。下面展示了一个简单的自适应分块 prefill 实现。这段代码动态地把块大小调到合适，以让 GPU 上的 SM 和占用率保持在高位：

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

Here, the scheduler is adjusting chunk size to use as much shared memory as possible. And, importantly, it does this without sacrificing parallelism.

这里，调度器在调整块大小，以尽可能多地使用共享内存。而且重要的是，它这样做时并不牺牲并行度。

Specifically, the scheduler first computes a tile width T so that the three shared-memory buffers for queries, keys, and values—each requiring T x T floats—fit within the GPUs’ per-SM dynamic shared-memory limit. It then calls the CuPy API (Python) to measure how many thread blocks can run concurrently on each SM with that value of T.

具体来说，调度器首先计算一个 tile 宽度 T，使得用于 queries、keys 和 values 的三个共享内存缓冲区——每个都需要 T x T 个 float——能装进 GPU 每个 SM 的动态共享内存限制之内。然后它调用 CuPy API（Python）来测量在该 T 值下每个 SM 上能并发运行多少个线程块。

If occupancy falls below a given threshold of 50%, T is reduced by half. This will free up shared memory so that more blocks can co-reside, striking an optimal balance between data reuse (fewer DRAM loads) and parallelism.

如果占用率跌破给定的 50% 阈值，T 就减半。这会释放共享内存，从而让更多的块可以共同驻留，在数据复用（更少的 DRAM 加载）与并行度之间取得最优平衡。

When overall GPU utilization drops below 85% and prefill work is pending, the scheduler selects the largest remaining prefill request and breaks it into equal-sized chunks of T tokens so that each chunk can flow through the pipeline without monopolizing every stage.

当整体 GPU 利用率降到 85% 以下且有 prefill 工作待处理时，调度器会选出剩余最大的 prefill 请求，并把它切成大小相等、每块 T 个 token 的块，使得每个块都能流经流水线而不独占每一个阶段。

And rather than fixing the chunk size, the helper function, next_prefill_chunk, adjusts T on the fly based on live metrics. It will shrink the chunk size if occupancy is low—and grow it if DRAM traffic is excessive. This makes sure that each slice maximizes GPU utilization without stalls.

而且，这个辅助函数 next_prefill_chunk 并不固定块大小，而是基于实时指标即时调整 T。如果占用率低，它会缩小块大小——如果 DRAM 流量过大，它会增大块大小。这确保了每个切片都能在不停顿的情况下最大化 GPU 利用率。

> Be sure to instrument the scheduler to log the chosen T and resulting occupancy/utilization. This way, you can analyze and verify that this adaptive approach is consistently maintaining high GPU utilization.

> 务必给调度器加上埋点，记录所选的 T 以及由此产生的占用率/利用率。这样，你就能分析并验证这种自适应方法是否始终如一地维持了高 GPU 利用率。

Between prefill chunks, the scheduler can use *decode-maximal batches* to bundle all of the ready decode requests into a single launch using form_decode_batch. This lets short, latency-sensitive token generations piggyback on otherwise idle pipeline gaps. This way, even users with short prompts see low latency because their decodes don’t wait for a huge prefill to finish. These decodes get scheduled in the pipeline gaps.

在各 prefill 块之间，调度器可以使用 *decode 最大化批次（decode-maximal batches）*，用 form_decode_batch 把所有就绪的 decode 请求打包进一次启动。这让短小、延迟敏感的 token 生成得以夹带在本来空闲的流水线间隙上。如此一来，即便是提示很短的用户也能看到低延迟，因为他们的 decode 不必等待一个庞大的 prefill 完成。这些 decode 会被调度进流水线的间隙里。

By continuously monitoring gpu_utilization(), the scheduler chooses whether to process another prefill chunk or drain the decode queue. Either way, it is always picking the action that fills SM slots and minimizes dead time. This is called a *utilization-maximization policy* and is similar to an OS scheduler aiming for 100% CPU utilization.

通过持续监控 gpu_utilization()，调度器选择是处理下一个 prefill 块还是排空 decode 队列。无论哪种方式，它总是挑选那个能填满 SM 槽位并把空转时间降到最低的动作。这被称为 *利用率最大化策略（utilization-maximization policy）*，类似于一个以 100% CPU 利用率为目标的操作系统调度器。

Together, these mechanisms ensure that large-context prefill jobs never starve interactive decoding. At the same time, small, latency-critical requests are served immediately. This produces optimal throughput on modern GPUs without degrading the end-user experience.

这些机制合在一起，确保大上下文的 prefill 作业永远不会让交互式解码挨饿。与此同时，小而延迟关键的请求会被立即服务。这在不降低终端用户体验的前提下，于现代 GPU 上产生了最优吞吐量。

> Chunked prefill aligns work with the strengths of the GPU: big prefill matrix operations are batched, while small decode operations are interleaved. This maximizes overall throughput and latency together.

> 分块 prefill 使工作与 GPU 的强项对齐：大的 prefill 矩阵运算被成批处理，而小的 decode 运算被交错穿插。这同时最大化了整体吞吐量和延迟表现。

As you can see, the scheduler monitors real-time metrics and adapts on the fly. The chunked scheduling makes sure no single request can block an entire stage under varying conditions. This keeps all GPUs active and reduces the dreaded pipeline bubbles.

如你所见，调度器监控实时指标并即时适应。分块调度确保在多变的条件下，没有任何单个请求能阻塞整个阶段。这让所有 GPU 都保持活跃，并减少了令人头疼的流水线气泡。

It’s important to treat prefill and decode as separate queues with their own SLAs. It’s usually optimal to clear out latency-sensitive decode tasks first using dedicated time slices or CUDA streams, then use the leftover cycles to process large prefill jobs. For example, you can allocate a certain time budget (e.g., 1–5 ms per decode) to decode tasks to make sure they get more immediate attention.

把 prefill 和 decode 当作各有其 SLA 的分离队列很重要。通常最优的做法是先用专用的时间片或 CUDA stream 清空延迟敏感的 decode 任务，再用剩下的周期去处理大的 prefill 作业。例如，你可以给 decode 任务分配一定的时间预算（例如每个 decode 1–5 ms），以确保它们获得更即时的关注。

By prioritizing the user-facing, real-time decodes, you minimize perceived inference lag. At the same time, you are still powering through bulk context builds when the system has spare capacity.

通过优先处理面向用户、实时的 decode，你就把感知到的推理滞后降到最低。与此同时，在系统有余力时，你仍在全力推进批量的上下文构建。

Many high-performance inference engines use a *producer-consumer* model with separate threads for the separate phases. For instance, vLLM uses multithreading such that one thread prepares decode inputs while another prepares new prefills, etc., and they feed into a single execution stream in an optimized order. This is a proven pattern to overlap work efficiently.

许多高性能推理引擎使用 *生产者-消费者（producer-consumer）* 模型，为不同阶段设置分离的线程。例如，vLLM 使用多线程，使得一个线程准备 decode 输入，而另一个线程准备新的 prefill 等，并让它们以优化过的顺序汇入单一的执行 stream。这是一种经过验证的、能高效重叠工作的模式。

If you have multiple nodes, you can send prefill requests to one set of nodes and decode requests to another set. In this case, the prefill and decode nodes can use different, heterogenous hardware specialized for their specific task of either prefill or decode.

如果你有多个节点，你可以把 prefill 请求发给一组节点，把 decode 请求发给另一组节点。在这种情况下，prefill 节点和 decode 节点可以使用为各自任务（prefill 或 decode）专门定制的、不同的异构硬件。

For instance, the prefill compute nodes can use GPUs with high FLOPS and less memory bandwidth since the prefill phase is compute bound. And the decode nodes can potentially use GPUs with higher memory bandwidth but less FLOPS capacity.

例如，prefill 计算节点可以使用高 FLOPS、内存带宽较低的 GPU，因为 prefill 阶段是计算受限（compute bound）的。而 decode 节点则可能使用内存带宽更高但 FLOPS 容量较低的 GPU。

Be aware that while a heterogeneous prefill and decode worker configuration can save cost, it can complicate—and potentially limit—dynamic load balancing. If the prefill/decode ratio shifts unexpectedly and more prefill work needs to be done on decode-optimized workers (less FLOPS), these decode workers may become a bottleneck.

要注意，虽然异构的 prefill 和 decode worker 配置可以节省成本，但它可能使动态负载均衡变得复杂——并可能对其形成限制。如果 prefill/decode 比例意外偏移，需要在 decode 优化型 worker（FLOPS 较低）上做更多的 prefill 工作，这些 decode worker 可能会成为瓶颈。

> Using homogeneous nodes will simplify scheduling. Hardware specialization should be used only if the workload ratio is predictable and can handle the shifts in load. However, this isn’t always possible due to capital investments, rapidly evolving GPU architectures, and cost budgets.

> 使用同构节点会简化调度。只有当负载比例可预测、且能承受负载变化时，才应采用硬件专用化。然而，由于资本投入、GPU 架构快速演进以及成本预算的限制，这并不总是可行。

Regarding prefill and decode batching, it’s recommended that you batch your decode calls together whenever possible to maximize GPU throughput and minimize launch overhead. For prefill, you should avoid mixing very short and very long prompts in the same batch.

关于 prefill 与 decode 批处理，建议你尽可能将 decode 调用批处理在一起，以最大化 GPU 吞吐量并最小化启动开销。对于 prefill，则应避免在同一个批次中混入极短与极长的提示。

You should use length-based bucketing so each batch has similar sequence lengths. This might mean grouping incoming requests to the nearest 512-token bucket before completing the batch. This way, you don’t waste compute on excessive padding.

你应使用基于长度的分桶（bucketing），使每个批次的序列长度相近。这可能意味着在凑齐批次之前，将传入请求归组到最接近的 512-token 桶中。这样一来，你就不会在过度填充（padding）上浪费算力。

In practice, most modern inference engines implement a token-level scheduler that dynamically forms batches at each generation step. It will wait a few milliseconds to gather ready tokens, cap batch sizes to stay within memory and occupancy limits, and employ round-robin/maximum-delay rules to improve fairness between long and short prompts. (Make sure this feature is enabled in your inference engine’s configuration.)

实践中，大多数现代推理引擎都实现了 token 级调度器，在每个生成步动态组建批次。它会等待几毫秒以收集就绪的 token，将批次大小上限设定在内存与占用率限制之内，并采用轮询/最大延迟规则来改善长短提示之间的公平性。（请确保在你的推理引擎配置中启用了该特性。）

In short, adaptive batching and prefill/decode disaggregation can maximize GPU utilization and increase throughput without sacrificing too much latency. In fact, these techniques can often improve latency in aggregate, because they keep the GPU busy—and less GPU idle time means faster task completion overall.

简而言之，自适应批处理（adaptive batching）与 prefill/decode 分离能够在不过多牺牲延迟的前提下最大化 GPU 利用率并提升吞吐量。事实上，这些技术往往能在总体上改善延迟，因为它们让 GPU 保持忙碌——而 GPU 空闲时间越少，整体任务完成就越快。

## Congestion-Aware and Topology-Aware Scheduling with Multiple GPUs

## 面向多 GPU 的拥塞感知与拓扑感知调度

Modern multi-GPU and multirack systems like Grace Blackwell GB200 NVL72 systems (72 Blackwell B200 GPUs with 180 GB HBM each) and the newer Grace Blackwell Ultra GB300 NVL72 (72 B300 GPUs with 288 GB each) connect 72 GPUs in a single high-bandwidth NVLink/NVSwitch fabric. These architectures create a unified 72-GPU domain and give each GPU up to ~1.8 TB/s of aggregate bidirectional NVLink throughput. This provides over 130 TB/s of aggregate cross-sectional bandwidth across the NVSwitch network.

现代多 GPU 与多机架（multirack）系统，如 Grace Blackwell GB200 NVL72 系统（72 个 Blackwell B200 GPU，每个配 180 GB HBM）以及更新的 Grace Blackwell Ultra GB300 NVL72（72 个 B300 GPU，每个配 288 GB），将 72 个 GPU 连接在单一的高带宽 NVLink/NVSwitch 结构中。这些架构构建出统一的 72-GPU 域，并为每个 GPU 提供高达约 1.8 TB/s 的双向 NVLink 聚合吞吐量。这在 NVSwitch 网络上提供了超过 130 TB/s 的聚合横截面带宽。

However, achieving peak performance for large-scale inference requires more than raw bandwidth. It needs intelligent and adaptive communication scheduling. Congestion-aware and topology-aware strategies make sure that data transfers avoid bottlenecks in real time, as shown in Figure 19-8.

然而，要为大规模推理实现峰值性能，仅有原始带宽是不够的。它还需要智能且自适应的通信调度。拥塞感知（congestion-aware）与拓扑感知（topology-aware）策略可确保数据传输实时避开瓶颈，如图 19-8 所示。

![Figure 19-8. Topology-aware routing to avoid bottlenecks across GPUs and multinode clusters](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-8.png)

![图 19-8. 拓扑感知路由，避免跨 GPU 与多节点集群的瓶颈](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-8.png)

To address these bottlenecks, let’s consider link utilization telemetry, dynamic message routing, and scheduling waves of collectives. Next are some key principles and techniques that enable efficient scheduling of inter-GPU communication while maintaining low latency and high throughput. To keep things concrete, we’ll do this in the context of an NVL72 rack environment.

为应对这些瓶颈，我们来考虑链路利用率遥测、动态消息路由以及集合通信的分波调度。接下来是一些关键原则与技术，它们能在保持低延迟、高吞吐量的同时实现 GPU 间通信的高效调度。为了具体起见，我们将在 NVL72 机架环境的语境下展开讨论。

### NVLink/NVSwitch Topology and Bandwidth Constraints

### NVLink/NVSwitch 拓扑与带宽约束

NVLink provides high-speed point-to-point links between GPUs—and between GPUs and the Grace CPU in Grace Blackwell Superchip modules. NVSwitch acts as an on-rack network switch connecting all GPUs into an all-to-all topology.

NVLink 在 GPU 之间——以及在 Grace Blackwell Superchip 模块中的 GPU 与 Grace CPU 之间——提供高速点对点链路。NVSwitch 则充当机架内的网络交换机，将所有 GPU 连接成全互连（all-to-all）拓扑。

In an NVL72 rack system, each GPU features multiple NVLink ports (e.g., 18 NVLink links per GPU) that connect into a set of NVSwitch chips. This design gives every GPU peer-to-peer connectivity such that any GPU can reach any other in a single hop, or one traversal, through the NVSwitch fabric. This topology allows the 72 GPUs to behave like a single giant board with uniform connectivity.

在 NVL72 机架系统中，每个 GPU 都配有多个 NVLink 端口（例如每个 GPU 18 条 NVLink 链路），连接到一组 NVSwitch 芯片。这种设计为每个 GPU 提供点对点连通性，使任一 GPU 都能通过 NVSwitch 结构以单跳、即一次穿越就到达任意其他 GPU。该拓扑让这 72 个 GPU 表现得像一块具有均匀连通性的巨型单板。

> Aggregate capacity still obeys per-port and per-switch limits. As such, many-to-one traffic patterns can oversubscribe ingress.

> 聚合容量仍受每端口与每交换机的限制约束。因此，多对一的流量模式可能使入口（ingress）超额订阅。

Despite their high bandwidth, the interconnects have finite capacity. Each NVLink 5 port provides 100 GB/s per direction. And each GB200/GB300 superchip exposes 18 NVLink links per GPU for up to 1.8 TB/s bidirectional throughput per GPU. And while NVSwitch provides nonblocking all-to-all connectivity under balanced loads, certain patterns like all GPUs sending to one GPU can oversubscribe links since each NVSwitch chip has a limit on total switching throughput.

尽管带宽很高，这些互连的容量仍是有限的。每个 NVLink 5 端口在每个方向提供 100 GB/s。每个 GB200/GB300 超级芯片为每个 GPU 暴露 18 条 NVLink 链路，可达每 GPU 1.8 TB/s 的双向吞吐量。虽然 NVSwitch 在均衡负载下提供无阻塞的全互连连通性，但某些模式——例如所有 GPU 都向同一个 GPU 发送数据——会使链路超额订阅，因为每个 NVSwitch 芯片对总交换吞吐量都有上限。

Congestion will occur if too many GPUs send data simultaneously through the same switch—or into the same destination. This causes queues to build up and a drop in effective transfer throughput. For instance, a single GPU can theoretically communicate bidirectionally at 1.8 TB/s, but if multiple peers all target that same GPU simultaneously, they must share the same NVLink ingress bandwidth. Similarly, NVSwitch can be oversubscribed by certain communication patterns, like all GPUs exchanging data at the same time—even though it’s designed for full nonblocking bandwidth (under balanced loads.)

如果太多 GPU 同时通过同一个交换机——或发往同一个目的地——发送数据，就会发生拥塞。这会导致队列堆积，并使有效传输吞吐量下降。例如，单个 GPU 理论上能以 1.8 TB/s 双向通信，但如果多个对端同时以该 GPU 为目标，它们就必须共享同一 NVLink 入口带宽。类似地，NVSwitch 也可能被某些通信模式超额订阅，比如所有 GPU 同时交换数据——尽管它被设计为提供完全无阻塞的带宽（在均衡负载下）。

Understanding the topology—which GPUs share NVSwitch components or NVLink paths—is critical to inference performance. On a smaller scale, GPUs on the same board or tray likely have faster and slightly more direct paths compared to GPUs in different trays or across nodes.

理解拓扑——即哪些 GPU 共享 NVSwitch 组件或 NVLink 路径——对推理性能至关重要。在较小的尺度上，与不同托盘（tray）或跨节点的 GPU 相比，位于同一单板或托盘上的 GPU 很可能具有更快、更直接一些的路径。

> Avoid making assumptions about the topology. Use CUDA’s topology APIs and NCCL’s topology hints to programmatically retrieve this info. You can query NVML and DCGM for per-NVLink port counters and remote endpoints and combine that with Fabric Manager or NVSwitch tooling for switch-level mapping when needed.

> 不要对拓扑做任何假设。使用 CUDA 的拓扑 API 与 NCCL 的拓扑提示以编程方式获取这些信息。你可以查询 NVML 与 DCGM 获取每 NVLink 端口的计数器与远端端点，并在需要时结合 Fabric Manager 或 NVSwitch 工具进行交换机级映射。

On a larger scale, communications that cross node/rack boundaries and leave the NVLink domain using InfiniBand/Ethernet will incur even higher latency and lower bandwidth than intra-NVL72 transfers. For example, InfiniBand NDR might add 5–10 µs of latency per hop versus less than 1 µs of latency for NVSwitch hops.

在更大的尺度上，跨越节点/机架边界、离开 NVLink 域并使用 InfiniBand/Ethernet 的通信，会比 NVL72 内部传输承受更高的延迟和更低的带宽。例如，InfiniBand NDR 每一跳可能增加 5–10 µs 的延迟，而 NVSwitch 每跳的延迟不到 1 µs。

Because of the physical topology, some communication paths are cheaper than others. A congestion-aware scheduler uses this knowledge to prefer higher-bandwidth, lower-latency, less congested links to maximize performance.

由于物理拓扑的存在，某些通信路径比其他路径更“便宜”。拥塞感知调度器利用这一知识，优先选择带宽更高、延迟更低、拥塞更少的链路，以最大化性能。

### Real-Time Link Telemetry and Monitoring

### 实时链路遥测与监控

To manage congestion, the system must first *observe* it. NVIDIA provides telemetry interfaces to monitor link utilization in real time. The NVIDIA Management Library (NVML) and, specifically, the nvmlDeviceGetNvLinkUtilizationCounter, expose per-link throughput counters and utilization statistics for NVLink.

要管理拥塞，系统必须首先*观测*到它。NVIDIA 提供了遥测接口来实时监控链路利用率。NVIDIA Management Library（NVML），特别是 nvmlDeviceGetNvLinkUtilizationCounter，暴露了 NVLink 的每链路吞吐量计数器与利用率统计。

> Enabling NVLink counters will introduce overhead. Sample them at a reasonable interval to avoid impacting performance while you’re monitoring performance!

> 启用 NVLink 计数器会引入开销。请以合理的间隔采样，以免在监控性能时反而影响了性能！

An adaptive inference serving system can query metrics, such as bytes transferred per NVLink port, error rates, and traffic load between specific GPU pairs. For instance, you can query NVML or DCGM for throughput counters, bandwidth statistics, and other telemetry (e.g., errors, etc.). Note that nvidia-smi nvlink --status provides link health and configuration. NVML and DCGM are the preferred mechanisms for performance counters such as throughput.

自适应推理服务系统可以查询各类指标，例如每个 NVLink 端口传输的字节数、错误率以及特定 GPU 对之间的流量负载。例如，你可以查询 NVML 或 DCGM 获取吞吐量计数器、带宽统计以及其他遥测数据（如错误等）。注意，nvidia-smi nvlink --status 提供链路健康状态与配置。而对于吞吐量等性能计数器，NVML 与 DCGM 是首选机制。

This is useful for identifying hotspot links that are saturated at close to 100% utilization while others are underused. These low-level hardware counters allow a scheduler to find exactly where bottlenecks are occurring. This includes specific NVSwitch uplinks—or the links between two specific GPUs.

这有助于识别那些利用率接近 100% 而处于饱和的热点（hotspot）链路，同时其他链路却未被充分利用。这些底层硬件计数器让调度器能够精确地找到瓶颈发生的位置。这包括特定的 NVSwitch 上行链路——或两个特定 GPU 之间的链路。

In addition to NVML, higher-level profiling tools like Nsight Systems provide timeline views of GPU activity, including communication events. Nsight can display when data transfers occur on NVLink/NVSwitch—and how long they take.

除 NVML 外，Nsight Systems 等更高层的性能剖析工具还提供了 GPU 活动的时间线视图，其中包括通信事件。Nsight 可以显示数据传输何时在 NVLink/NVSwitch 上发生——以及它们持续多久。

By instrumenting inference runs with Nsight Systems, one can visualize if multiple transfers overlap and cause delays—or if certain stages are waiting on communication. For instance, the timeline might reveal that all pipeline stages attempt to send activations at the same moment over the same link, which will overwhelm the interconnect.

通过用 Nsight Systems 对推理运行进行插桩，人们可以直观地看到是否有多个传输相互重叠并造成延迟——或某些阶段是否在等待通信。例如，时间线可能会揭示所有流水线阶段试图在同一时刻通过同一链路发送激活值，这将使互连不堪重负。

It’s recommended to integrate these metrics into your monitoring Grafana dashboards using Prometheus and DCGM exporter. This way, you can see link utilization in real time—as well as historically and over time. And when you identify hotspots, such as synchronization points, your system can insert a slight delay, reschedule tasks, or reassign GPU roles to alleviate the hotspot and smooth out traffic.

建议你使用 Prometheus 与 DCGM exporter 将这些指标集成到你的 Grafana 监控仪表盘中。这样，你就能实时——以及在历史和长期时间维度上——看到链路利用率。当你识别出诸如同步点之类的热点时，你的系统可以插入轻微的延迟、重新调度任务，或重新分配 GPU 角色，以缓解热点并平滑流量。

For example, the system can adjust the scheduling by inserting slight delays or overlapping differently to reduce contention. This real-time telemetry enables dynamic, adaptive, feedback-driven decisions. The scheduler can react on the fly to spikes in link utilization by rerouting traffic or rescheduling tasks to different GPUs, as described next.

例如，系统可以通过插入轻微延迟或以不同方式重叠来调整调度，从而减少争用。这种实时遥测使动态、自适应、反馈驱动的决策成为可能。调度器可以对链路利用率的骤升即时做出反应，将流量重新路由或把任务重新调度到不同的 GPU 上，如下文所述。

### Adaptive Process-GPU Mapping

### 自适应进程-GPU 映射

One powerful strategy is topology-aware placement of computation processes to minimize heavy communication across slow or congested links. For example, consider a multi-GPU inference pipeline in which different LLM layers (“processes”)

一种强有力的策略是对计算进程进行拓扑感知的放置，以最小化跨慢速或拥塞链路的繁重通信。例如，考虑一个多 GPU 推理流水线，其中不同的 LLM 层（“进程”）

reside on different GPUs. In this case, large intermediate tensors must be passed along the inference pipeline.

驻留在不同的 GPU 上。在这种情况下，大型中间张量必须沿推理流水线传递。

This is essentially a process-GPU placement-optimization problem, which requires mapping the graph of neural-network model layers onto GPU hardware that incurs the minimum amount of communication cost. If the original assignment of layers/processes to GPUs is naive, these tensors might travel over long, expensive, congested paths. This could include multiple NVSwitch hops—or even off the NVLink fabric entirely onto another rack or data center. This will definitely reduce throughput and overall performance.

这本质上是一个进程-GPU 放置优化问题，需要将神经网络模型各层构成的图映射到 GPU 硬件上，且使产生的通信成本最小。如果层/进程到 GPU 的原始分配很朴素，这些张量可能会经由漫长、昂贵、拥塞的路径传输。这可能包括多次 NVSwitch 跳转——甚至完全离开 NVLink 结构而进入另一个机架或数据中心。这必将降低吞吐量与整体性能。

With adaptive process-GPU mapping, the system dynamically assigns processes to GPUs such that communication is kept as local (and balanced) as possible. For instance, consider our LLM layers (processes) partitioned across many GPUs in an NVL72 rack. If layer/process 0 on GPU 0 feeds layer/process 2 on GPU 2, but their GPUs are on opposite ends of the NVSwitch network, the data has to traverse more links. In this case, moving layer/process 2 to GPU 1 is the preferred process-GPU mapping, as shown in Figure 19-9, in the context of NVIDIA’s Topology-Aware GPU Selection (NVTAGS) system.

借助自适应进程-GPU 映射（process-GPU mapping），系统动态地将进程分配到 GPU，使通信尽可能保持本地化（且均衡）。例如，考虑我们的 LLM 各层（进程）被划分到 NVL72 机架中的多个 GPU 上。如果 GPU 0 上的层/进程 0 向 GPU 2 上的层/进程 2 供数，但它们的 GPU 位于 NVSwitch 网络的两端，数据就必须穿越更多链路。在这种情况下，将层/进程 2 移动到 GPU 1 是更优的进程-GPU 映射，如图 19-9 所示，其背景为 NVIDIA 的拓扑感知 GPU 选择（Topology-Aware GPU Selection，NVTAGS）系统。

![Figure 19-9. NVIDIA’s Topology-Aware GPU Selection (NVTAGS) process-to-GPU mapping](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-9.png)

![图 19-9. NVIDIA 的拓扑感知 GPU 选择（NVTAGS）进程到 GPU 的映射](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-9.png)

Here, NVTAGS automatically assigns GPU affinity to processes based on the communication patterns between the GPUs. NVTAGS is a topology-aware GPU selection framework from NVIDIA that automates process-to-GPU mapping using fabric distances and link metrics. It actively profiles the topology and reassigns processes to GPUs with the fastest mutual links.

在这里，NVTAGS 会根据 GPU 之间的通信模式，自动为进程分配 GPU 亲和性。NVTAGS 是 NVIDIA 提供的拓扑感知 GPU 选择框架，利用结构距离与链路指标自动完成进程到 GPU 的映射。它主动剖析拓扑，并将进程重新分配给彼此之间链路最快的 GPU。

If telemetry indicates this link is becoming saturated because the activation tensor is very large, for instance, the scheduler can remap process 2 onto another GPU that is “closer” to GPU 0—ideally one that shares a high-bandwidth connection or is in the same NVSwitch module. The adaptive process-GPU remapping will dynamically reassign which GPU holds which model layer in the LLM inference example.

举例来说，如果遥测显示某条链路由于激活张量非常大而趋于饱和，调度器就可以把进程 2 重映射到另一个“更靠近” GPU 0 的 GPU 上——理想情况下是一个共享高带宽连接、或处于同一 NVSwitch 模块中的 GPU。在 LLM 推理这个例子中，自适应进程-GPU 重映射将动态地重新分配哪个 GPU 承载哪个模型层。

> As a starting point, and if you’re not using NVTAGS, you can use your system’s topology map to help identify which GPU groupings are “closer” in the context of the network topology.

> 作为起点，如果你没有使用 NVTAGS，你可以利用系统的拓扑图来帮助识别在网络拓扑语境下哪些 GPU 分组彼此“更近”。

This remapping is done at initialization as well as between inferences. Some systems use upfront or periodic profiling runs to decide an optimal placement. If two processes exchange tens of GB per second, they should reside on the same node, if possible. Conversely, processes that are more compute bound or have minimal data transfer between them can tolerate more distance from one another.

这种重映射在初始化时进行，也在多次推理之间进行。有些系统使用预先或周期性的剖析运行来决定最优放置。如果两个进程之间每秒交换数十 GB 数据，它们应尽可能驻留在同一节点上。反之，那些更受算力约束、或彼此之间数据传输极少的进程，则能容忍更大的相互距离。

Remapping is *adaptive* when the system monitors performance and iteratively improves the mapping as conditions change. For instance, if after one pass the highest-traffic connection is between process 3 and process 4 on different nodes, the scheduler might swap one of those processes with another process on the same node to bring 3 and 4 together.

当系统监控性能，并随着条件变化迭代地改进映射时，重映射就是*自适应*的。例如，如果一趟之后流量最高的连接是位于不同节点上的进程 3 与进程 4 之间，调度器可能会将其中一个进程与同一节点上的另一个进程互换，从而把 3 和 4 拉到一起。

The impact of adaptive remapping is an evolving GPU assignment schedule that responds to the observed traffic pattern. This approach directly reduces cross-node traffic by keeping data exchanges confined to local domains.

自适应重映射的效果是形成一份不断演进的 GPU 分配方案，它会响应观测到的流量模式。这种方法通过将数据交换限制在本地域内，直接减少了跨节点流量。

For example, after remapping, what was a 50 GB/s cross-node transfer might become two 25 GB/s within-node transfers. This eliminates a network bottleneck and reduces network latency by 50% for that communication.

例如，重映射之后，原本 50 GB/s 的跨节点传输可能变成两个各 25 GB/s 的节点内传输。这消除了一处网络瓶颈，并使该通信的网络延迟降低 50%。

Remapping can be formulated as an optimization problem that uses a graph-partitioning algorithm. In this case, the graph’s edge weights are the volume of data traveling over the links. You would solve for the minimal cut.

重映射可以表述为一个使用图划分算法的优化问题。在这种情况下，图中的边权是流经链路的数据量。你要求解的是最小割（minimal cut）。

> Be aware that moving a layer/process means moving model weights. If a model layer has many GB of data, you won’t want to do this too frequently. It’s best to apply this strategy between large batches—or when the mapping will remain relatively static for a minimum period of time.

> 请注意，移动一个层/进程意味着移动模型权重。如果某个模型层有数 GB 的数据，你就不会想太频繁地这样做。最好在大批次之间——或当映射将在某个最短时段内保持相对静止时——再应用此策略。

In deep learning inference, we can apply the idea of adaptive mapping to inter-GPU communications using pipeline, tensor, and expert parallel techniques. The GPUs that talk to one another the most should be assigned the strongest, least congested connections between them.

在深度学习推理中，我们可以把自适应映射的思想应用于使用流水线、张量与专家并行技术的 GPU 间通信。相互通信最频繁的 GPU 应被分配彼此之间最强、最不拥塞的连接。

### Optimizing Collective Communication with NCCL

### 用 NCCL 优化集合通信

NVIDIA’s Collective Communications Library (NCCL) is the standard library managing these GPU collectives. It offers multiple algorithms and optimizations for multiple GPU environments.

NVIDIA 的集合通信库（Collective Communications Library，NCCL）是管理这些 GPU 集合通信的标准库。它为多 GPU 环境提供了多种算法与优化。

Many inference workloads involve collective communication patterns, such as gathering outputs from multiple experts, broadcasting parameters, or performing all-reduce operations, as shown in Figure 19-10. Here, NCCL communication (stream 1) overlaps with GEMM computations (stream 0).

许多推理工作负载都涉及集合通信模式，例如从多个专家收集输出、广播参数，或执行 all-reduce 操作，如图 19-10 所示。在这里，NCCL 通信（stream 1）与 GEMM 计算（stream 0）相重叠。

![Figure 19-10. Distributed GEMM using multiple GPUs and all-reduce across NVLink](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-10.png)

![图 19-10. 使用多个 GPU 并跨 NVLink 执行 all-reduce 的分布式 GEMM](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-10.png)

These NCCL optimizations can be applied dynamically by the scheduler in congestion-aware environments. By choosing the right collective algorithm and tuning it using the topology and congestion information, we can reduce communication overhead of our inference system. Next are some key considerations that the scheduler can use when tuning NCCL on the fly.

在拥塞感知环境中，调度器可以动态地应用这些 NCCL 优化。通过选择正确的集合通信算法，并利用拓扑与拥塞信息对其调优，我们可以降低推理系统的通信开销。接下来是调度器在动态调优 NCCL 时可以使用的一些关键考量。

NCCL can perform reductions (along with other collectives) using a ring algorithm or a tree algorithm. In a ring all-reduce, each GPU passes data along a closed loop/ring so that every piece of data traverses all GPUs in sequence.

NCCL 可以使用环（ring）算法或树（tree）算法执行归约（以及其他集合通信）。在环 all-reduce 中，每个 GPU 沿一个闭环/环传递数据，使得每一份数据都依次穿越所有 GPU。

The ring approach maximizes bandwidth utilization on NVLink/NVSwitch by keeping all links busy, but it means the latency scales linearly with the number of GPUs. For instance, on a 72-GPU ring, the data makes 71 hops to complete one reduction.

环方法通过让所有链路保持忙碌来最大化 NVLink/NVSwitch 上的带宽利用率，但这意味着延迟随 GPU 数量线性增长。例如，在 72-GPU 的环上，数据需要 71 跳才能完成一次归约。

A tree algorithm, in contrast, reduces or broadcasts data in a logarithmic fashion since GPUs are organized into a logical binary tree where each step halves the number of participants. However, the GPUs are physically connected linearly, link-by-link, into what can be logically considered a *tree-chain*. Figure 19-11 compares tree-based and ring-based communication among GPUs.

相比之下，树算法以对数方式进行归约或广播，因为 GPU 被组织成一棵逻辑二叉树，每一步都将参与者数量减半。然而，这些 GPU 在物理上是逐链路线性相连的，可在逻辑上视为一条*树链*（tree-chain）。图 19-11 对比了 GPU 之间基于树与基于环的通信。

![Figure 19-11. Tree-based (chain) versus ring-based communication](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-11.png)

![图 19-11. 基于树（链）与基于环的通信对比](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-11.png)

> In practice, in a tree all-reduce, the GPUs first connect in a simple NVLink/NVSwitch “chain” within each node. Across nodes, they connect in a pipelined, dual-binary-tree topology. This is what allows the O(log N) latency of a tree-based algorithm.

> 实践中，在树 all-reduce 中，GPU 首先在每个节点内以简单的 NVLink/NVSwitch “链”相连。跨节点时，它们以流水线化的双二叉树拓扑相连。这正是树算法能实现 O(log N) 延迟的原因。

Tree algorithms complete in fewer steps (e.g., log₂(72) in the case of a 72-GPU ring), which directly reduces latency—especially for “long haul” transfers across a large number of GPUs. Trees utilize the parallelism of NVSwitch since multiple independent reduction flows can occur simultaneously along different branches of the tree. The trade-off is that a tree may not fully saturate every link’s bandwidth at every moment, but it avoids any single link or GPU becoming a chokepoint through the long ring path.

树算法以更少的步数完成（例如，在 72-GPU 环的情形下为 log₂(72)），这直接降低了延迟——尤其是对于跨大量 GPU 的“长途”传输。树利用了 NVSwitch 的并行性，因为多个独立的归约流可以沿树的不同分支同时进行。其代价是，树可能无法在每一时刻都完全占满每条链路的带宽，但它避免了任何单条链路或 GPU 通过漫长的环路径成为咽喉点。

> For latency-sensitive reductions with small messages, a tree algorithm is usually superior. For extremely large messages in which bandwidth dominates, a ring algorithm is usually better—assuming the network is not congested and bandwidth dominates. Choose per message size and topology, and consider hierarchical schemes.

> 对于消息较小、延迟敏感的归约，树算法通常更优。对于带宽占主导的极大消息，环算法通常更好——前提是网络未拥塞且带宽占主导。请按消息大小与拓扑逐一选择，并考虑分层方案。

By default, NCCL selects ring, tree, or hierarchical variants heuristically based on message size and topology. On NVSwitch-based intranode paths, rings are often favored for bandwidth, while across nodes, hierarchical and tree variants are common. On a fully connected, single-node NVSwitch system like the NVL72, it’s usually better to force a tree for large reductions.

默认情况下，NCCL 会基于消息大小与拓扑，用启发式方法在环、树或分层变体之间做选择。在基于 NVSwitch 的节点内路径上，为追求带宽通常偏好环；而跨节点时，分层与树变体则较为常见。在像 NVL72 这样完全互连的单节点 NVSwitch 系统上，对大规模归约通常最好强制使用树。

Forcing a tree algorithm with the NCCL_ALGO environment variable, for instance, can alleviate congestion in a single NVL72 rack by not sending all data through one long ring path across all 72 GPUs. For example, a 72-GPU tree-based all-reduce will complete much faster than a ring-based algorithm. This is because the tree algorithm performs 6 (log₂(72)) sequential steps versus the ring algorithm’s 71 (71 = 72 – 1) sequential steps.

例如，通过 NCCL_ALGO 环境变量强制使用树算法，可以缓解单个 NVL72 机架内的拥塞，因为不必让所有数据都经由横跨全部 72 个 GPU 的一条长环路径传输。例如，72-GPU 的基于树的 all-reduce 会比基于环的算法完成得快得多。这是因为树算法执行 6（log₂(72)）个顺序步骤，而环算法执行 71（71 = 72 – 1）个顺序步骤。

The scheduler can explicitly choose the algorithm that best fits the current topology and message size using dynamic NCCL tuning parameters. For instance, it can favor tree all-reduce for very large GPU counts to avoid looping across the entire cluster for each update.

调度器可以使用动态 NCCL 调优参数，显式选择最适合当前拓扑与消息大小的算法。例如，它可以对极大的 GPU 数量偏好树 all-reduce，以避免每次更新都在整个集群中绕环。

When a ring algorithm is chosen due to its simplicity and bandwidth efficiency, for instance, one concern is that the same links could consistently carry the heaviest load—particularly the links between certain GPU pairs where the ring wraps around.

例如，当出于简单性与带宽效率而选择环算法时，一个顾虑是相同的链路可能会持续承载最重的负载——尤其是环绕回卷处特定 GPU 对之间的链路。

A congestion-aware approach is to rotate the ring ordering across iterations or collective operations. By periodically shifting which GPU is the start of the ring—and therefore which pairs communicate first—the communication load is distributed more evenly over all NVLink connections.

一种拥塞感知的做法是在各次迭代或集合通信操作之间轮转环的排序。通过周期性地改变哪个 GPU 作为环的起点——从而改变哪些对先通信——通信负载就能在所有 NVLink 连接上更均匀地分布。

And while NCCL’s “inside/outside” ring mechanism already alternates ring direction on successive calls, the additional shuffling of ranks between steps will help if your workload is persistently imbalanced. This makes sure that no single NVLink becomes a perpetual bottleneck.

虽然 NCCL 的“内/外”环机制已经在连续调用中交替环方向，但如果你的工作负载持续不均衡，在各步之间额外打乱 rank 顺序会有所帮助。这能确保没有任何单条 NVLink 成为永久的瓶颈。

In practice, NCCL has an alternating rings enhancement that implements this kind of rotation under the hood, but it can also be managed by the scheduler using GPU re-indexing in the communicator for different collectives. To do this, you can periodically call ncclCommInitRank with a permuted rank order. The effect is that, over time, no single link or GPU is always on the critical path for every collective. This smooths out utilization.

实践中，NCCL 有一项交替环（alternating rings）增强，在底层实现了这类轮转，但它也可以由调度器通过在通信器（communicator）中对不同集合通信重新索引 GPU 来管理。为此，你可以周期性地以置换后的 rank 顺序调用 ncclCommInitRank。其效果是，随着时间推移，没有任何单条链路或 GPU 会始终处于每一次集合通信的关键路径上。这会平滑利用率。

Rather than launching one giant collective operation that uses all GPUs at once, wave scheduling breaks communications into phased waves to reduce instantaneous load. For instance, suppose an inference workload needs to perform an all-to-all exchange of embeddings among 72 GPUs a pattern common in mixtures-of-experts or certain ensemble methods.

与其发起一个一次性使用所有 GPU 的巨型集合通信操作，分波调度将通信拆分为分阶段的波次，以降低瞬时负载。例如，假设某推理工作负载需要在 72 个 GPU 之间执行嵌入（embedding）的全互连交换，这是专家混合或某些集成方法中常见的模式。

Doing this exchange as one monolithic step would mean that each GPU sends data to 71 others simultaneously. This is 72 GPUs × 71 messages that are saturating every link and NVSwitch port at once.

把这次交换作为单一的整块步骤来做，将意味着每个 GPU 同时向其他 71 个 GPU 发送数据。这就是 72 个 GPU × 71 条消息，会一次性使每条链路和 NVSwitch 端口都饱和。

With all 72 GPUs exchanging data simultaneously, this will cause a spike. Instead, you can split the exchange into 4 groups, or waves, of 18 GPUs to smooth out the traffic.

当全部 72 个 GPU 同时交换数据时，这会造成骤升。你可以改为把交换拆分成 4 组、即 4 波，每波 18 个 GPU，以平滑流量。

This is called *wave scheduling*, and it structures the exchange as a series of smaller all-to-all exchanges that use only a subset of GPUs during each wave. It can also pipeline smaller chunks such that only a fraction of the traffic is in flight at any given moment.

这被称为*分波调度*（wave scheduling），它把交换组织成一系列更小的全互连交换，每一波只使用 GPU 的一个子集。它还可以对更小的分块进行流水线化，使得在任一给定时刻只有一小部分流量在途。

In NCCL terms, this might correspond to splitting a large all-reduce into multiple slices internally. NCCL actually does this automatically to pipeline data through a ring. The scheduler can also use NCCL to orchestrate a sequence of smaller collectives.

用 NCCL 的术语来说，这可能对应于在内部把一个大的 all-reduce 拆分为多个切片。NCCL 实际上会自动这样做，以将数据流水线化地穿过环。调度器也可以用 NCCL 来编排一系列更小的集合通信。

By staggering the start times of these communication waves, the network fabric has some headroom since one wave’s data is partly through the system before the next wave adds more traffic. This is called *temporal multiplexing*, and it avoids overwhelming the NVSwitch fabric. This technique is conceptually similar to pacing network traffic in order to avoid burstiness.

通过错开这些通信波次的起始时间，网络结构就有了一些余量，因为在下一波加入更多流量之前，上一波的数据已经部分穿过系统。这被称为*时间复用*（temporal multiplexing），它避免了让 NVSwitch 结构不堪重负。该技术在概念上类似于对网络流量进行节奏控制以避免突发性。

Another example is overlapping computation and communication—a pattern we have seen repeatedly throughout this book. If layer outputs are reduced in waves, the system can schedule the next layer’s computation to overlap with later waves of the reduction.

另一个例子是计算与通信的重叠——一种我们在本书中反复见到的模式。如果层输出以波次进行归约，系统就可以调度下一层的计算，使其与归约的后续波次重叠。

This creates a pipeline between compute and communication such that while some GPUs are finalizing a reduction, other GPUs have moved on to the next layer’s compute. And this overlap essentially time-shifts some of the communication to a time when the compute units would otherwise be idle. The result is improved utilization of NVLink bandwidth without a massive one-time spike that causes congestion.

这在计算与通信之间形成流水线，使得当一些 GPU 正在完成一次归约时，另一些 GPU 已经推进到下一层的计算。而这种重叠本质上把一部分通信在时间上平移到了计算单元本来会空闲的时刻。其结果是在不产生引发拥塞的巨大一次性骤升的前提下，改善了 NVLink 带宽的利用率。

It’s important to carefully optimize collectives, pick the right algorithm, and structure communication in balanced waves. This is essential to topology-aware scheduling—and it leads to more efficient, fair, and balanced usage of the NVLink/NVSwitch network among all GPUs.

精心优化集合通信、选对算法、并将通信组织成均衡的波次，这一点很重要。它对拓扑感知调度至关重要——并能让所有 GPU 之间对 NVLink/NVSwitch 网络的使用更高效、更公平、更均衡。

### Multinode and Multirack Communication with GPUDirect RDMA

### 用 GPUDirect RDMA 进行多节点与多机架通信

When scaling beyond a single node (e.g., NVL72 rack), additional challenges will start to surface since communication is traveling over relatively slow network interfaces, such as InfiniBand and Ethernet. In this case, NVLink and NVSwitch no longer directly connect all of the GPUs in the system. Instead, GPUs in different nodes exchange data using NICs and network switches.

当扩展到单个节点（如 NVL72 机架）之外时，额外的挑战将开始浮现，因为通信要经由相对较慢的网络接口传输，例如 InfiniBand 与 Ethernet。在这种情况下，NVLink 与 NVSwitch 不再直接连接系统中的所有 GPU。取而代之的是，不同节点中的 GPU 使用 NIC 与网络交换机来交换数据。

To maintain high performance in a multinode and multirack environment, modern AI systems use GPUDirect RDMA. As covered in Chapter 4, GPUDirect RDMA allows GPUs to directly send/receive data with remote GPUs’ memory—and without host CPU involvement, as shown in Figure 19-12.

为在多节点与多机架环境中保持高性能，现代 AI 系统会使用 GPUDirect RDMA。正如第 4 章所述，GPUDirect RDMA 允许 GPU 直接与远端 GPU 的内存收发数据——且无需主机 CPU 参与，如图 19-12 所示。

![Figure 19-12. Direct GPU-to-GPU memory transfers with GPUDirect RDMA—and without involving the host CPU memory (source: https://oreil.ly/445a9)](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-12.png)

![图 19-12. 使用 GPUDirect RDMA 进行 GPU 到 GPU 的直接内存传输——且不涉及主机 CPU 内存（来源：https://oreil.ly/445a9）](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-12.png)

Even with RDMA efficiency, however, network bandwidth is still lower and latency is still higher than intranode NVLink. As such, network congestion in the cluster fabric will become the limiting factor without a dynamic and adaptive routing schedule. A congestion-aware scheduler can intelligently route and balance internode traffic in addition to intranode traffic, as we discussed earlier.

然而，即便有 RDMA 的高效性，网络带宽仍然低于、延迟仍然高于节点内的 NVLink。因此，在没有动态自适应路由调度的情况下，集群结构中的网络拥塞将成为限制因素。拥塞感知调度器除了能处理我们前面讨论的节点内流量之外，还能智能地路由并平衡节点间流量。

One key technique is leveraging multiple network interfaces, called *multirail*. High-end GPU servers often have several NICs, including dual InfiniBand ports per node—or even one per GPU in some designs. For instance, using two NICs per node can produce nearly 2× the throughput versus using one NIC. There is a bit of overhead when using multiple NICs, but it still provides a large gain.

一项关键技术是利用多个网络接口，称为*多轨*（multirail）。高端 GPU 服务器通常配有多个 NIC，包括每节点双 InfiniBand 端口——在某些设计中甚至每个 GPU 一个。例如，每节点使用两个 NIC 相比使用一个 NIC，可产生近 2× 的吞吐量。使用多个 NIC 会有一点开销，但它仍能带来很大的增益。

NCCL automatically supports using multiple NICs in parallel to increase bandwidth. In addition, it will split rings and trees across these NICs. Topology awareness is critical here. Consider if each NIC connects to a different network switch to form separate “rails” in the cluster network. In this case, using more than one NIC per collective can reduce the load on any single network path.

NCCL 自动支持并行使用多个 NIC 以提升带宽。此外，它会将环与树拆分到这些 NIC 上。这里拓扑感知至关重要。考虑每个 NIC 连接到不同网络交换机，从而在集群网络中形成各自独立的“轨道”（rail）的情形。在这种情况下，每次集合通信使用不止一个 NIC 可以减轻任何单一网络路径上的负载。

The NCCL environment variable NCCL_CROSS_NIC controls whether a collective operation is allowed to use different NICs on different nodes for the same ring/tree. By enabling this with a well-designed network topology, NCCL might send half the GPUs’ data out of NIC1 and the other half out of NIC2. This effectively doubles throughput and avoids bottlenecks on a single link.

NCCL 环境变量 NCCL_CROSS_NIC 控制是否允许一次集合通信操作在不同节点上为同一个环/树使用不同的 NIC。在精心设计的网络拓扑下启用它，NCCL 可能会把一半 GPU 的数据从 NIC1 发出，另一半从 NIC2 发出。这实际上使吞吐量翻倍，并避免了单条链路上的瓶颈。

> If your GPU nodes have multiple NICs and your NCCL version supports NCCL_CROSS_NIC, enable it for large collectives to stripe traffic across rails on topologies designed for multirail.

> 如果你的 GPU 节点有多个 NIC，且你的 NCCL 版本支持 NCCL_CROSS_NIC，请为大规模集合通信启用它，以在为多轨设计的拓扑上跨轨道条带化（stripe）流量。

The scheduler can detect if one NIC or path is reaching capacity. If it is reaching capacity, it can redistribute the traffic by moving some GPUs’ communications to an alternate interface or alternate network route if one is available. This can be improved using network-level adaptive routing, but your application can also pin certain GPU traffic to less-used NICs. To do this, you just need to set different NCCL channels to different NICs.

调度器可以检测某个 NIC 或路径是否正达到容量上限。如果达到容量上限，它就可以通过把某些 GPU 的通信迁移到备用接口或备用网络路由（如果有的话）来重新分配流量。这可以借助网络级的自适应路由来改善，但你的应用也可以把特定的 GPU 流量固定（pin）到使用较少的 NIC 上。为此，你只需将不同的 NCCL 通道设置到不同的 NIC 上。

In a multinode context, rerouting can also mean cooperating with the network’s adaptive routing features. Modern InfiniBand networks have adaptive routing in which congested flows are automatically moved to less-congested paths in the fabric. While this is handled at the network level, a higher-level scheduler can influence it by changing which destination IP/route is used for a given GPU transfer, for instance. Or the scheduler can split transfers into smaller chunks so the network can balance them.

在多节点语境中，重新路由也可能意味着与网络的自适应路由特性协同。现代 InfiniBand 网络具有自适应路由，其中拥塞的流会被自动迁移到结构中拥塞较轻的路径上。虽然这是在网络层处理的，但更高层的调度器可以通过改变某次 GPU 传输所使用的目的 IP/路由等方式来影响它。或者，调度器可以把传输拆分成更小的分块，以便网络能对它们进行均衡。

Additionally, it’s important to enforce NIC affinity by binding each GPU’s communication to the NIC closest (e.g., the NIC on the same PCIe root complex or the same NVSwitch/CPU complex). This will reduce local contention.

此外，通过将每个 GPU 的通信绑定到最近的 NIC（例如，位于同一 PCIe 根复合体、或同一 NVSwitch/CPU 复合体上的 NIC），来强制 NIC 亲和性，这一点很重要。这将减少本地争用。

To enforce GPU-NIC affinity, you can use NVML or NCCL to map GPU ↔ NIC locality. You need to configure NCCL to respect this mapping by supplying it with a static topology description that associates each GPU with its specific NIC (e.g., GPUs 0–3 ↔ NIC 0, GPUs 4–7 ↔ NIC 1).

要强制 GPU-NIC 亲和性，你可以使用 NVML 或 NCCL 来映射 GPU ↔ NIC 的局部性。你需要为 NCCL 提供一份将每个 GPU 与其特定 NIC 关联起来的静态拓扑描述（例如，GPU 0–3 ↔ NIC 0，GPU 4–7 ↔ NIC 1），从而配置 NCCL 遵守该映射。

A well-designed system will align GPU-to-NIC pairings so that each GPU’s data takes the shortest path out of the node. If a particular NIC is congested due to multiple heavy GPU flows out of the same port, the scheduler can reassign one GPU’s traffic to another port on the node for the next round of communication. And while NCCL’s autotuning will do most of this for you, manual overrides may be needed to tackle specific and persistent issues.

精心设计的系统会对齐 GPU 到 NIC 的配对，使每个 GPU 的数据都走出节点的最短路径。如果某个特定 NIC 因同一端口上有多条繁重的 GPU 流而拥塞，调度器可以在下一轮通信中把某个 GPU 的流量重新分配到该节点上的另一个端口。虽然 NCCL 的自动调优会为你完成大部分工作，但对于特定且持续存在的问题，可能仍需要手动覆盖。

Consider an extreme case, such as a large, heavily loaded inference cluster hosting a massive MoE LLM model. Here, the network will be a major bottleneck if the system is not tuned properly. This is because of the heavy communication between the nodes in the cluster.

考虑一种极端情形，例如一个承载着庞大 MoE LLM 模型、负载沉重的大型推理集群。在这里，如果系统未被正确调优，网络将成为主要瓶颈。这是因为集群中节点之间存在繁重的通信。

In such extreme cases, the scheduler may decide to replicate certain data (e.g., experts) on multiple nodes to reduce cross-node queries. Or it can perform operations hierarchically by first aggregating results within each node, then exchanging a summary between nodes. This is in contrast to exchanging full data between all GPUs across nodes.

在这类极端情形下，调度器可能会决定在多个节点上复制某些数据（如专家），以减少跨节点查询。或者它可以分层地执行操作，先在每个节点内聚合结果，再在节点之间交换一份摘要。这与在跨节点的所有 GPU 之间交换完整数据形成对比。

> NVIDIA SHARP can offload certain aggregation operations to the switch hardware. In inference clusters, using SHARP and adaptive routing together helps minimize communication bottlenecks.

> NVIDIA SHARP 可以把某些聚合操作卸载到交换机硬件上。在推理集群中，将 SHARP 与自适应路由结合使用有助于最小化通信瓶颈。

For multinode environments, congestion-aware scheduling is needed to avoid saturating any single network link. This type of scheduling requires careful routing/binding decisions, GPUDirect RDMA to bypass needless memory copies, and multirail NIC utilization to maximize bandwidth.

对于多节点环境，需要拥塞感知调度来避免任何单条网络链路饱和。这类调度需要审慎的路由/绑定决策、用于绕过不必要内存拷贝的 GPUDirect RDMA，以及用于最大化带宽的多轨 NIC 利用。

The goal is to extend topology awareness beyond NVSwitch and understand the cluster network’s full topology (e.g., fat-tree, dragonfly, etc.) and adapt communication patterns to that topology. The result is that even at scale with thousands and millions of GPU nodes, internode transfers are orchestrated in a balanced way. This will prevent one slow link from throttling the entire distributed inference system.

其目标是把拓扑感知扩展到 NVSwitch 之外，理解集群网络的完整拓扑（如 fat-tree、dragonfly 等），并让通信模式适配该拓扑。其结果是，即便在拥有数千乃至数百万 GPU 节点的规模上，节点间传输也能以均衡的方式被编排。这将防止一条慢速链路拖垮整个分布式推理系统。

> Treat the network as a schedulable resource just like GPUs, memory, etc. It should be planned and adaptively managed like other dynamically allocated resources.

> 把网络当作一种可调度资源来对待，就像 GPU、内存等一样。它应当像其他动态分配的资源那样被规划并自适应地管理。

### MoE Expert Rebalancing and Regrouping

### MoE 专家再平衡与重新分组

Large-scale language models increasingly use MoE layers, which introduce unique communication patterns. In an MoE model, different subsets of the neural network, or experts, reside on different GPUs. And each input token is routed to a small number of expert networks for processing.

大规模语言模型越来越多地使用 MoE 层，这引入了独特的通信模式。在 MoE 模型中，神经网络的不同子集（即专家）驻留在不同的 GPU 上。每个输入 token 被路由到少量专家网络进行处理。

During inference, for instance, this produces an all-to-all traffic pattern such that tokens are sent to whichever GPU hosts the selected experts. The results are then gathered back. In a naive static assignment of experts to GPUs, certain GPUs may become communication hotspots if many tokens frequently route to experts on those specific GPUs. Additionally, if experts that often work together to process similar tokens are placed far apart in the topology (e.g., one on GPU 0 and another on GPU 71 across the fabric), the tokens will need to continuously travel long paths.

例如，在推理期间，这会产生一种全互连的流量模式，使得 token 被发送到承载所选专家的那个 GPU 上。随后结果再被收集回来。在朴素的专家到 GPU 的静态分配中，如果许多 token 频繁路由到某些特定 GPU 上的专家，这些 GPU 就可能成为通信热点。此外，如果经常协同处理相似 token 的专家在拓扑上被放置得相距很远（例如一个在 GPU 0、另一个在结构另一端的 GPU 71），token 就需要持续走很长的路径。

Expert rebalancing is a strategy to localize communication by periodically rearranging which GPU each expert lives on. The key idea is to take advantage of any skew or patterns in the workload. If, for example, expert 5 and expert 19 both receive segments of the same queries, it makes sense to place them on the same GPU (or same node), if possible, so that the communication doesn’t travel too far for those operations.

专家再平衡（rebalancing）是一种通过周期性地重排每个专家所在的 GPU 来使通信本地化的策略。其核心思想是利用工作负载中的任何偏斜（skew）或模式。举例来说，如果专家 5 和专家 19 都接收同一批查询的片段，那么在可能的情况下把它们放在同一个 GPU（或同一节点）上是合理的，这样这些操作的通信就不会走太远。

Likewise, if expert 7 is very popular and receives many tokens, it may incur heavy inbound traffic to its GPU. The scheduler might move that expert to a less communication-heavy GPU—or even duplicate it if the system allows—to split the load. This rebalancing can happen between inference runs or during periodic maintenance windows in a long-running service. The system collects statistics on communication frequency between experts—or between experts and the expert-gating nodes—and then remaps the experts to GPUs that minimize the highest-traffic links.

同样地，如果专家 7 非常热门、接收大量 token，它可能给其所在 GPU 带来沉重的入站流量。调度器可能会把该专家迁移到一个通信较轻的 GPU 上——或者在系统允许的情况下甚至复制它——以分摊负载。这种再平衡可以在多次推理运行之间进行，或在长期运行的服务的周期性维护窗口中进行。系统收集专家之间——或专家与专家门控节点之间——通信频率的统计数据，然后把专家重映射到能最小化最高流量链路的 GPU 上。

In practice, implementing MoE expert rebalancing involves a coordinated redistribution of model parameters since experts are essentially subsets of model weights. This should be done infrequently since it’s a heavy operation. But occasional rebalancing can have a big impact in reducing congested transfers.

实践中，实现 MoE 专家再平衡涉及模型参数的协调重分布，因为专家本质上是模型权重的子集。由于这是一项繁重的操作，应当不频繁地进行。但偶尔的再平衡在减少拥塞传输方面能产生很大影响。

Expert rebalancing should be done during scheduled maintenance windows since moving an expert means transferring potentially GBs of weights. The key is to use logged routing metrics to choose a better placement strategy at runtime.

专家再平衡应在计划内的维护窗口中进行，因为移动一个专家意味着传输可能达数 GB 的权重。关键在于利用记录下来的路由指标，在运行时选择更优的放置策略。

After rebalancing, each GPU will ideally host a combination of experts such that most tokens’ routing stays on a single GPU or at least within the local NVLink group. Any other nonlocal communication will be spread more evenly across the NVLink/NVSwitch network—rather than repeatedly hitting the same GPU-to-GPU links.

再平衡之后，理想情况下每个 GPU 都将承载一种专家组合，使得大多数 token 的路由都停留在单个 GPU 上，或至少停留在本地 NVLink 组内。任何其他非本地通信都会更均匀地分散到 NVLink/NVSwitch 网络上——而不是反复击中相同的 GPU 到 GPU 链路。

> In short, spread out the popular experts and colocate experts that frequently communicate with one another.

> 简而言之，把热门的专家分散开，把彼此频繁通信的专家放在一起。

Another related optimization is expert bucketing or grouping. This technique arranges experts that are commonly used together and are assigned to the same group of GPUs (for example, on the same NVSwitch or same server), which reduces cross-group traffic.

另一项相关的优化是专家分桶（bucketing）或分组。该技术把常被一起使用的专家安排到同一组 GPU 上（例如在同一 NVSwitch 或同一服务器上），从而减少跨组流量。

The scheduler can treat expert placement as a graph partitioning problem. The experts (GPUs) are the nodes in the graph. The edge weights represent the token traffic between experts—as well as from the router to the experts. The graph is partitioned using a minimal cut through the fewest heavy edges. By doing this, MoE communication becomes topology-aware, respects the NVLink/NVSwitch boundaries, and keeps the data exchange confined within those boundaries.

调度器可以把专家放置当作一个图划分问题来处理。专家（GPU）是图中的节点。边权表示专家之间——以及从路由器到各专家——的 token 流量。图通过一条穿过尽量少的重边的最小割来划分。这样做之后，MoE 通信就变得拓扑感知、尊重 NVLink/NVSwitch 边界，并将数据交换限制在这些边界之内。

MoE expert regrouping is an example of congestion-aware scheduling at the model architecture level. It rearranges the workload itself to fit the network—rather than rearranging the network to fit the workload.

MoE 专家重新分组是模型架构层面拥塞感知调度的一个例子。它重排工作负载本身以适配网络——而不是重排网络以适配工作负载。

### Dynamic Congestion-Aware Scheduling

### 动态拥塞感知调度

While all the techniques we’ve mentioned can be configured at system startup or design time, the most robust and advanced systems use dynamic scheduling to respond to congestion as it happens. Dynamic congestion-aware scheduling means the system continuously monitors network conditions using the telemetry discussed earlier—and adjusts the scheduling of tasks or communications in real time.

尽管我们提到的所有技术都可以在系统启动时或设计阶段配置，但最健壮、最先进的系统会使用动态调度来实时应对拥塞。动态的拥塞感知调度意味着系统利用前文讨论的遥测数据持续监控网络状况，并实时调整任务或通信的调度。

Congestion-aware scheduling and routing helps reduce bottlenecks and maintains high performance under dynamic conditions. This is analogous to network-level dynamic packet routing, as shown in Figure 19-13.

拥塞感知的调度与路由有助于减少瓶颈，并在动态条件下保持高性能。这类似于网络层的动态分组路由，如图 19-13 所示。

![Figure 19-13. Network-level packet routing to avoid congestion](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-13.png)

![图 19-13. 网络层的分组路由以规避拥塞](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-13.png)

In a multi-GPU inference context, dynamic strategies include throttling, rerouting, or reordering operations based on congestion feedback. For instance, suppose the scheduler detects that NVLink link 0, connecting two particular GPUs, is currently maxed out because it’s transferring data for a massive tensor during a large pipeline-parallel activation transfer, for instance.

在多 GPU 推理场景中，动态策略包括基于拥塞反馈对操作进行限流、重新路由或重新排序。举例来说，假设调度器检测到连接两个特定 GPU 的 NVLink link 0 当前已被打满，因为它正在一次大型流水线并行激活传输中搬运某个巨大张量的数据。

If another high-priority transfer is scheduled to use the same link, the scheduler might delay that second transfer by a few milliseconds to let the first one finish and clear out. This is called *temporal load balancing*, and it essentially inserts a tiny gap to prevent queue buildup. This is analogous to network-switch queue management and NIC-level backpressure. This is better to enqueue briefly than to overload and drop packets.

如果另一次高优先级传输被安排使用同一条链路，调度器可能会将第二次传输延迟几毫秒，让第一次传输完成并清空链路。这称为*时间维度的负载均衡*，它本质上是插入一个微小的间隙以防止队列堆积。这类似于网络交换机的队列管理和 NIC 层的反压。短暂排队要好于过载并丢弃分组。

Conversely, if a normally large transfer is detected to be idle because its source GPU is waiting on computation, for instance, the scheduler could use that time slice to send lower-priority data over the link and fill the idle time. This utilizes available bandwidth and doesn’t interfere with a critical data transfer.

反之，如果检测到某次通常很大的传输因其源 GPU 正在等待计算而处于空闲状态，调度器就可以利用这段时间片，在该链路上发送较低优先级的数据以填满空闲时间。这样既利用了可用带宽，又不会干扰关键的数据传输。

Another dynamic tactic is adaptive routing at the software level such that if one path is congested, an alternate path is chosen if available. In a network with multiple NVSwitch planes or multiple NIC rails, the adaptive runtime will choose a less busy plane for the next communication.

另一种动态策略是软件层的自适应路由，即如果一条路径发生拥塞，就在可用的情况下选择另一条路径。在拥有多个 NVSwitch 平面或多条 NIC rail 的网络中，自适应运行时会为下一次通信选择一个较不繁忙的平面。

NCCL does some of this internally when multiple paths exist, but an advanced scheduler could maintain multiple NCCL communicators that are mapped to different path configurations. It could then select among the different paths based on congestion.

当存在多条路径时，NCCL 会在内部完成一部分这类工作，但更先进的调度器可以维护多个映射到不同路径配置的 NCCL communicator，然后基于拥塞情况在不同路径之间进行选择。

Choosing alternate paths dynamically requires the scheduler to evaluate the best path to take. This might be achieved by using different virtual channels—or adjusting which NVLink port is used for a transfer.

动态选择备用路径需要调度器评估应采用的最佳路径。这可以通过使用不同的虚拟通道，或调整某次传输使用哪个 NVLink 端口来实现。

Modern NVSwitch systems support multiple virtual channels and hardware quality-of-service (HQOS) settings. The scheduler can use these features to direct nonurgent traffic to a lower-priority channel. This avoids contention with urgent transfers.

现代 NVSwitch 系统支持多个虚拟通道和硬件服务质量（HQOS）设置。调度器可以利用这些特性把非紧急流量引导到较低优先级的通道，从而避免与紧急传输发生争用。

Load-dependent task scheduling is another feature of dynamic congestion management. If an inference server is handling many simultaneous queries that share resources, the scheduler can temporarily queue or reorder some of the queries to avoid peak overlap. This is similar to the earlier discussion on staggering large collectives so that they don’t run concurrently.

负载相关的任务调度是动态拥塞管理的另一项特性。如果一个推理服务器正在处理许多共享资源的并发查询，调度器可以临时将部分查询排队或重新排序，以避免峰值重叠。这与前文讨论的错开大型集合通信、使其不并发运行的做法类似。

For instance, consider a situation in which the scheduler knows that query A’s next step will involve a massive all-gather across GPUs. And it sees that query B is just starting and would add another large all-gather at the same time. In this case, the scheduler might postpone launching query B’s step by a brief moment so that query A’s communication can complete without the burden of query B’s resource contention.

举例来说，考虑这样一种情况：调度器知道查询 A 的下一步将涉及一次跨 GPU 的大规模 all-gather，同时它看到查询 B 刚刚开始，会在同一时刻再叠加一次大规模 all-gather。在这种情况下，调度器可能会将查询 B 那一步的启动推迟片刻，以便查询 A 的通信能够在不受查询 B 资源争用拖累的情况下完成。

This kind of fine-grained scheduling optimizes the pattern of communications over time. Heavy flows are serialized or staggered rather than launched concurrently. The decision is guided by recent telemetry. If the system sees a big spike in NVSwitch utilization when it runs 8 queries in parallel, it might try running only 4 in the first wave, and then 4 immediately after. This makes the system self-tuning because it monitors real-time telemetry data and continuously searches for execution plans that avoid congestion.

这种细粒度调度会随时间推移优化通信模式。重流量被串行化或错开，而不是并发启动。该决策由最近的遥测数据引导。如果系统发现并行运行 8 个查询时 NVSwitch 利用率出现大幅飙升，它可能会尝试在第一波只运行 4 个，紧接着再运行 4 个。这使系统具备自调优能力，因为它监控实时遥测数据，并持续搜索能规避拥塞的执行计划。

Dynamic scheduling is typically implemented as a centralized scheduler that monitors all GPUs and network links. This can be combined with a distributed protocol in which GPUs signal congestion, or backpressure, to one another if the destination GPU’s NVLink buffers are full.

动态调度通常实现为一个集中式调度器，监控所有 GPU 和网络链路。这可以与一种分布式协议相结合：当目标 GPU 的 NVLink 缓冲区已满时，各 GPU 相互发出拥塞信号或反压信号。

In this backpressure scenario, the destination GPU notifies the source GPU to pause incoming transfers. The smart scheduler can then reschedule tasks on the source GPU while it’s paused so it can perform compute tasks while it waits for the destination GPU to unpause the transfers.

在这种反压场景下，目标 GPU 通知源 GPU 暂停发送传入的传输。智能调度器随后可以在源 GPU 暂停期间为其重新调度任务，使其在等待目标 GPU 恢复传输的同时执行计算任务。

> NCCL will apply backpressure when receivers can’t keep up. A custom scheduler can piggyback on this functionality by noticing that send operations are blocked. It can then use that time to perform other useful work.

> 当接收方跟不上时，NCCL 会施加反压。自定义调度器可以借助这一功能，通过注意到发送操作被阻塞来搭便车，然后利用这段时间执行其他有用的工作。

Over time, dynamic adjustments like these will keep communication efficient—even with varying batch sizes, input data distributions, and other dynamic workload changes. The system learns the congestion patterns and adapts quickly by modifying the scheduling based on live feedback. The system can use an RL agent (discussed in a previous section) or a set of heuristic rules. This is essential for environments with bursty and unpredictable inference requests.

随着时间推移，诸如此类的动态调整会持续保持通信高效——即使批大小、输入数据分布以及其他动态工作负载不断变化。系统会学习拥塞模式，并基于实时反馈修改调度，从而快速自适应。系统可以使用 RL 智能体（前一节讨论过）或一组启发式规则。这对于推理请求突发且不可预测的环境至关重要。

### Coordinating NVSwitch Transfers with Fine-Tuned Scheduling

### 用精细化调度协调 NVSwitch 传输

The core of an NVLink/NVSwitch system is the NVSwitch fabric itself. This is a centralized crossbar that handles many simultaneous GPU-to-GPU transfers. NVSwitch is extremely high-bandwidth and has its own internal scheduling algorithms, including adaptive routing across multiple switch chips and planes.

NVLink/NVSwitch 系统的核心是 NVSwitch 交换结构本身。这是一个集中式交叉开关（crossbar），负责处理许多并发的 GPU 到 GPU 传输。NVSwitch 具有极高带宽，并拥有自己的内部调度算法，包括跨多个交换芯片和平面的自适应路由。

However, software can multiply its effectiveness by scheduling data transfers with application-level knowledge, such as pipeline, tensor, and expert-level parallelism strategies. The idea is to orchestrate which GPU pairs communicate—and which times they communicate—in order to maximize parallelism without oversubscribing the cluster fabric.

然而，软件可以通过利用应用层知识（例如流水线、张量和专家级并行策略）来调度数据传输，从而成倍提升其效能。其思路是编排哪些 GPU 对进行通信——以及它们在什么时间通信——以在不超订集群交换结构的前提下最大化并行度。

A proven technique is staggering communication waves. This is related to the wave scheduling strategy mentioned earlier for collectives, but it applies more broadly to any overlapping transfers.

一种经过验证的技术是错开通信波次。这与前文针对集合通信提到的波次调度策略相关，但它更广泛地适用于任何相互重叠的传输。

Consider all 72 GPUs in a NVL72 rack needing to send data to a specific peer, such as a central parameter server in which GPU 0 collects all the results from all 72 GPUs. If all 71 other GPUs send their data at the exact same time, GPU 0’s 18 NVLink links—and the NVSwitch that connects them—will experience a huge burst of 71 inputs, as shown in Figure 19-14. This will exceed the amount of bandwidth that can be delivered at that moment.

考虑一个 NVL72 机架中的全部 72 个 GPU 都需要向某个特定对端发送数据，例如向一个中央参数服务器发送，其中 GPU 0 收集来自全部 72 个 GPU 的所有结果。如果其余 71 个 GPU 在完全相同的时刻同时发送数据，GPU 0 的 18 条 NVLink 链路——以及连接它们的 NVSwitch——将经历一次 71 路输入的巨大突发，如图 19-14 所示。这将超出当时能够交付的带宽量。

![Figure 19-14. All 72 GPUs sending data to a centralized parameter server](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-14.png)

![图 19-14. 全部 72 个 GPU 向一个集中式参数服务器发送数据](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-14.png)

In this case, NVSwitch will need to buffer and serialize many of those transfers. This leads to latency spikes. Instead, a coordinated and optimized approach is to partition the senders into four groups: group 1 (GPUs 1–18) sends first, then a few microseconds later group 2 (GPUs 19–36) sends, and so on.

在这种情况下，NVSwitch 将需要缓冲并串行化其中许多传输，这会导致延迟尖峰。相反，一种协调且优化的做法是把发送方划分为四组：第 1 组（GPU 1–18）先发送，几微秒后第 2 组（GPU 19–36）发送，依此类推。

From GPU 0’s perspective, it receives four smaller waves of traffic in sequence. At any given instant, roughly only 18 GPUs are actively sending to GPU 0. This perfectly fits within the GPU’s 18-port capacity. NVSwitch routes the traffic without needing to queue. By the time group 4 finishes, GPU 0 has received all of the data—and none of the NVLink links were saturated since the traffic was smoothed and balanced over time.

从 GPU 0 的视角看，它按顺序接收到四个较小的流量波次。在任一时刻，大约只有 18 个 GPU 在向 GPU 0 主动发送。这恰好落在该 GPU 的 18 端口容量之内。NVSwitch 无需排队即可路由流量。当第 4 组完成时，GPU 0 已收到全部数据——并且由于流量在时间上被平滑和均衡，没有任何一条 NVLink 链路被饱和。

This wave-staggering approach generalizes to many patterns. All-to-all exchanges can be broken into pairwise exchanges that rotate in rounds. This is often called the *butterfly* or *shuffle pattern*. Shuffling schedules which GPUs talk to one another at each timestep such that each NVSwitch port stays busy, but not excessively busy.

这种波次错开方法可以推广到许多模式。all-to-all 交换可以拆分为分轮轮换的成对交换。这通常称为*蝶式（butterfly）*或*洗牌模式（shuffle pattern）*。洗牌会调度每个时间步哪些 GPU 相互通信，使每个 NVSwitch 端口保持繁忙，但又不过度繁忙。

The scheduler for NVSwitch transfers can use a time-sliced algorithm, which allocates communication slots to specific GPU pairs or GPU groups. So instead of launching one large, free-for-all bulk transfer, the scheduler can perform many small, synchronized communication steps—each allotted a specific time slot. This is similar to time-division multiplexing, described earlier, and it creates a predictable, conflict-free use of the NVSwitch crossbar.

NVSwitch 传输的调度器可以使用一种时间分片算法，为特定的 GPU 对或 GPU 组分配通信时隙。因此，调度器不再启动一次大型的、无序竞争的批量传输，而是可以执行许多小型的、同步的通信步骤——每一步都分配到一个特定的时隙。这类似于前文描述的时分复用，它使 NVSwitch 交叉开关的使用变得可预测且无冲突。

It’s worth noting that NVSwitch hardware itself will attempt to reduce contention on the network. For instance, if multiple flows are contending for the same link, NVSwitch will interleave packets from each flow to ensure fair scheduling.

值得注意的是，NVSwitch 硬件本身也会尝试减少网络上的争用。例如，如果多个流正在争用同一条链路，NVSwitch 会将各个流的分组交错排布，以确保公平调度。

It may also adaptively choose different internal crossbar paths, if available. However, from a software perspective, we can avoid hitting these limits in the first place by applying these adaptive techniques into our network design.

它还可能在可用时自适应地选择不同的内部交叉开关路径。然而，从软件的角度看，我们可以通过把这些自适应技术应用到网络设计中，从一开始就避免触及这些上限。

Fine-tuned scheduling also includes concurrency control by limiting how many heavy transfers run in parallel during inference. For example, during a multi-GPU inference pipeline, you might avoid launching all expert-gather or broadcast operations across GPUs at the same time. By design, this trades a bit of parallelism for less contention.

精细化调度还包括并发控制，即限制推理期间并行运行多少次重传输。例如，在一条多 GPU 推理流水线中，你可能会避免在同一时刻跨 GPU 启动所有的专家 gather 或广播操作。这种设计以牺牲一点并行度来换取更少的争用。

> Often, 2–4 simultaneous large expert-gather or broadcast transfers across GPUs during inference is the sweet spot. More than that can produce diminishing returns or congestion.

> 通常，推理期间跨 GPU 同时进行 2–4 次大型专家 gather 或广播传输是最佳平衡点。超过这个数量可能会产生收益递减或拥塞。

For instance, instead of triggering 12 collective transfers simultaneously—which risks saturating NVSwitch and NVLink—a scheduler can stagger them by running up to 4 high-volume transfers at a time, waiting for those to complete, and then launching the next set. Because NVSwitch is extremely fast, this serialized approach likely finishes sooner because it avoids the congestion caused by too many overlapping transfers.

例如，与其同时触发 12 次集合通信传输——这有饱和 NVSwitch 和 NVLink 的风险——调度器可以将它们错开，每次最多运行 4 次高容量传输，等这些完成后再启动下一批。由于 NVSwitch 极快，这种串行化做法很可能反而更早完成，因为它避免了过多重叠传输所导致的拥塞。

Coordinating NVSwitch transfers is about treating the communication fabric as a shared resource that can be scheduled—similar to how one would schedule GPU kernels and CPU threads. By scheduling network resources, the system makes sure that high-priority traffic avoids interference. It fills idle gaps with lower-priority traffic to keep utilization high.

协调 NVSwitch 传输的核心，是把通信结构当作一种可被调度的共享资源来对待——就如同调度 GPU 核函数和 CPU 线程那样。通过调度网络资源，系统确保高优先级流量不受干扰，并用低优先级流量填满空闲间隙以保持高利用率。

Techniques like staggering and grouping communications will increase effective throughput of the NVSwitch by avoiding severe contention patterns. This leads to more predictable and lower-latency communication, which is vital for inference serving where tail latency, or slow outlier responses caused by a congested network, is a concern.

错开与分组通信之类的技术会通过避免严重的争用模式来提高 NVSwitch 的有效吞吐量。这带来更可预测、更低延迟的通信，而这对推理服务至关重要，因为尾延迟——即由拥塞网络造成的缓慢离群响应——是一个需要关注的问题。

In short, congestion-aware, topology-aware scheduling in multi-GPU inference systems is all about how to intelligently match the communication pattern to the given hardware layout. High-performance inference systems will monitor link usage in real time and adapt to the NVLink/NVSwitch topology. It does this through careful placement of tasks, optimized collective algorithm configurations, multinode routing tweaks, MoE expert reallocation, dynamic runtime adjustments, and fine-grained coordination of data transfers.

简而言之，多 GPU 推理系统中的拥塞感知、拓扑感知调度，关键在于如何智能地把通信模式与给定的硬件布局相匹配。高性能推理系统会实时监控链路使用情况并适配 NVLink/NVSwitch 拓扑。它通过精心放置任务、优化集合通信算法配置、多节点路由微调、MoE 专家重新分配、动态运行时调整以及数据传输的细粒度协调来做到这一点。

## Additional Adaptive and Dynamic Optimization Techniques

## 更多自适应与动态优化技术

Next are some additional dynamic inference and runtime-adaptation techniques that complement the core tuning strategies we presented. As of this writing, these ideas are experimental and not widely available. However, they are promising and worth covering. Each technique here includes a brief description and links to further reading.

接下来是一些额外的动态推理与运行时自适应技术，它们与我们已介绍的核心调优策略互为补充。截至撰写本文时，这些想法仍属实验性质，尚未广泛可用。不过它们很有前景，值得一并介绍。这里的每项技术都包含简要说明和进一步阅读的链接。

### Dynamic Early-Exit Networks

### 动态提前退出网络

Early-exit models allow an LLM to self-truncate its generation when sufficient confidence is reached. This reduces unnecessary compute for easy inputs, for instance. Dynamic early-exit methods monitor intermediate representations and logit entropies to decide, at each layer or token, if it should stop computation and emit a final output.

提前退出（early exit）模型允许 LLM 在达到足够置信度时自行截断其生成过程。举例来说，这会为简单输入减少不必要的计算。动态提前退出方法会监控中间表示和 logit 熵，以在每一层或每个 token 上决定是否应停止计算并输出最终结果。

These networks require special model architecture or training since they add auxiliary classifiers at intermediate layers. However, they can produce up to 30%–50% inference speedup on reasoning tasks without accuracy loss (see *https://oreil.ly/dn3vl* and *https://oreil.ly/73AeE*).

这类网络需要特殊的模型架构或训练，因为它们在中间层添加了辅助分类器。不过，它们在推理任务上可带来高达 30%–50% 的推理加速而不损失准确率（参见 *https://oreil.ly/dn3vl* 和 *https://oreil.ly/73AeE*）。

### Input-Aware Layer Skipping (DASH)

### 输入感知的层跳过（DASH）

Frameworks like DASH present inference as a Markov Decision Process, which dynamically decides, per-token, whether to execute or skip each transformer layer based on input characteristics. By learning a small scoring network, DASH can skip 20%–40% of layers for many tokens.

DASH 之类的框架把推理表述为一个马尔可夫决策过程，基于输入特征，逐 token 动态决定是执行还是跳过每一个 transformer 层。通过学习一个小型评分网络，DASH 可以为许多 token 跳过 20%–40% 的层。

DASH typically requires a modified model with gating at each layer. However, it can reduce inference cost significantly while maintaining performance on NLP benchmarks (*https://oreil.ly/mz59T* and *https://oreil.ly/oL_Lr*).

DASH 通常需要一个在每层带有门控的改进模型。不过，它可以在保持 NLP 基准性能的同时显著降低推理成本（*https://oreil.ly/mz59T* 和 *https://oreil.ly/oL_Lr*）。

### Speculative MoE Expert Routing and Communication Reduction

### 推测式 MoE 专家路由与通信削减

For MoE models, speculative expert routing anticipates which experts will be activated for upcoming tokens and co-shuffles tokens and expert assignments ahead of time.

对于 MoE 模型，推测式专家路由会预判即将到来的 token 将激活哪些专家，并提前对 token 和专家分配进行协同洗牌。

This technique involves sending tokens to predicted experts early. If prediction is wrong, some work is wasted. However, overall communication is reduced when predictions are good. This helps to reduce cross-node bandwidth use by up to 30% compared to static expert parallel (EP) + tensor parallel (TP) deployments.

该技术会提前把 token 发送给预测的专家。如果预测错误，一部分工作会被浪费。然而，当预测准确时，总体通信量会减少。与静态的专家并行（EP）+ 张量并行（TP）部署相比，这有助于将跨节点带宽使用最多降低 30%。

### Dynamic Token Pruning with LazyLLM

### 用 LazyLLM 进行动态 token 剪枝

LazyLLM selectively computes KV cache only for tokens deemed as important defined using a lightweight scoring function. It prunes the low-impact tokens (e.g., stopwords and filler tokens) out of both the prefill and decode. By focusing expensive attention computations on only relevant tokens, LazyLLM reports 20%–30% end-to-end latency reduction on long-context workloads.

LazyLLM 仅为那些被一个轻量评分函数判定为重要的 token 计算 KV 缓存。它把低影响的 token（例如停用词和填充 token）从 prefill 和 decode 中都剪除掉。通过只在相关 token 上进行昂贵的注意力计算，LazyLLM 报告在长上下文工作负载上实现了 20%–30% 的端到端延迟降低。

### Edge-Oriented MoE Memory Budgeting

### 面向边缘的 MoE 内存预算

Dual routing and dynamic scheduling techniques introduce a potential memory issue for expert weights in constrained environments, such as edge deployments. In practice, this means maybe keeping a subset of experts active in GPU memory and swapping others from flash storage as needed. It does this based on usage frequency.

双路由和动态调度技术在受限环境（如边缘部署）中给专家权重带来了潜在的内存问题。实际上，这意味着可能只在 GPU 内存中保持一部分专家处于激活状态，并按需从闪存中换入其他专家。它依据使用频率来做出这种决策。

By dynamically adjusting which low-bit experts reside in memory (versus being offloaded), inference systems can maintain high expert-activation rates while maintaining lower memory usage.

通过动态调整哪些低比特专家驻留在内存中（而非被卸载），推理系统可以在保持较低内存占用的同时维持较高的专家激活率。

### Dynamic Quantization and Activation Range Adjustment

### 动态量化与激活范围调整

While static PTQ and QAT are well-known techniques, an on-the-fly quantization strategy can adjust activation-quantization parameters during inference in real time. They can use sliding-window statistics to modify the observer ranges (for activation quantizers) every *N* tokens, for example.

尽管静态 PTQ 和 QAT 是众所周知的技术，但一种即时（on-the-fly）量化策略可以在推理期间实时调整激活量化参数。例如，它们可以使用滑动窗口统计量，每隔 *N* 个 token 修改一次（激活量化器的）观测器范围。

This type of dynamic activation quantization will monitor activation statistics in real time and recompute observer ranges every *N* tokens. This way, FP16 can be allocated to “hot” layers with high variance and FP8 to “cool” layers with low-variance. Low-variance layers are constrained within a narrow dynamic range. This minimizes quantization error and maximizes throughput.

这类动态激活量化会实时监控激活统计量，并每隔 *N* 个 token 重新计算观测器范围。这样一来，FP16 可以分配给方差较高的"热"层，而 FP8 分配给低方差的"冷"层。低方差层被约束在一个狭窄的动态范围内。这既最小化了量化误差，又最大化了吞吐量。

Because low-variance layers produce activations that are clustered tightly around a mean, FP8’s limited exponent and mantissa bits (e.g., E4M3 format) are sufficient enough to represent the activations’ values accurately. This produces significant compute and memory savings—and without noticeable accuracy degradation.

由于低方差层产生的激活值紧密聚集在均值附近，FP8 有限的指数位和尾数位（例如 E4M3 格式）就足以准确表示这些激活值。这带来了可观的计算和内存节省——且没有明显的准确率下降。

Consider using a hybrid FP8 strategy such as E4M3 for the forward pass (e.g., activations and weights) and E5M2 for the backward pass (e.g., gradients). It’s recommended to use a delayed scaling mechanism as shown in the following code using the Transformer Engine:

考虑采用一种混合 FP8 策略，例如前向传播（如激活和权重）使用 E4M3，反向传播（如梯度）使用 E5M2。推荐使用一种延迟缩放机制，如下方使用 Transformer Engine 的代码所示：

```
from transformer_engine.common.recipe import Format, DelayedScaling
recipe = DelayedScaling(
    # E4M3 fwd, E5M2 bwd
    fp8_format=Format.HYBRID,
    amax_history_len=1024,
    # delayed scaling window
    amax_compute_algo="max",
)
```

```
from transformer_engine.common.recipe import Format, DelayedScaling
recipe = DelayedScaling(
    # E4M3 fwd, E5M2 bwd
    fp8_format=Format.HYBRID,
    amax_history_len=1024,
    # delayed scaling window
    amax_compute_algo="max",
)
```

Here, we set the delayed scaling window using amax_history_len=1024, which is a common default. It’s recommended to keep amax_compute_algo='max' unless convergence analysis suggests otherwise.

这里，我们用 amax_history_len=1024 设置延迟缩放窗口，这是一个常见的默认值。除非收敛分析另有提示，否则推荐保持 amax_compute_algo='max'。

Meanwhile, FP16 remains reserved for layers with larger activation swings, or “hot” layers. These layers need to use a broader dynamic range to capture the numerical fidelity required for critical computations.

与此同时，FP16 仍保留给激活波动较大的层，即"热"层。这些层需要使用更宽的动态范围，以捕捉关键计算所要求的数值保真度。

This hybrid precision strategy works well for inference pipelines that don’t require offline calibration. They can dynamically modify numerical fidelity at each layer. This balances overall performance and model accuracy.

这种混合精度策略非常适合无需离线校准的推理流水线。它们可以在每一层动态修改数值保真度，从而在整体性能与模型准确率之间取得平衡。

## Key Takeaways

## 关键要点

On modern GPUs, the approaches discussed in this chapter are practical ways to squeeze every bit of performance from these multi-trillion-parameter models. In real-world deployments, these dynamic runtime adaptation techniques can make or break a high-performance inference service offering. The following are some key takeaways from this chapter:

在现代 GPU 上，本章讨论的方法是从这些多万亿参数模型中榨取每一分性能的实用手段。在真实世界的部署中，这些动态运行时自适应技术足以决定一个高性能推理服务能否成功。以下是本章的一些关键要点：

*Steady-state inference with torch.compile* Prefer mode="reduce-overhead" or autotune modes if you can afford the warmup time. This will help minimize runtime overhead for low-latency inference workloads.

*用 torch.compile 实现稳态推理* 如果你能承受预热时间，优先选择 mode="reduce-overhead" 或 autotune 模式。这将有助于为低延迟推理工作负载最小化运行时开销。

*Kernel-level autotuning* Dynamically optimize GPU kernels and tile sizes. Leverage the Tensor Memory Accelerator (TMA) for asynchronous memory prefetch when possible. Use libraries and compilers that provide autotuning like CUTLASS and Triton—rather than hand-tuning them, unless absolutely necessary for performance.

*核函数级自动调优* 动态优化 GPU 核函数和分块大小。在可能时利用张量内存加速器（TMA）进行异步内存预取。使用像 CUTLASS 和 Triton 这样提供自动调优的库和编译器——而不是手动调优它们，除非为了性能而绝对必要。

*Adaptive precision* Switch between 8-bit and 4-bit floating point (FP8/FP4) during inference to balance speed and accuracy. You can also mix with 16-bit precision as needed. Use the Transformer Engine for FP8 in PyTorch since torch.autocast() does not support FP8 directly.

*自适应精度* 在推理期间在 8 位和 4 位浮点（FP8/FP4）之间切换，以平衡速度和准确率。你也可以按需与 16 位精度混用。在 PyTorch 中，使用 Transformer Engine 来实现 FP8，因为 torch.autocast() 并不直接支持 FP8。

*Disaggregated inference pipeline* Separate the prefill (prompt processing) and decode (generation) phases across resources, and context-aware request routing based on real-time factors like KV cache hits, queue depth, and load. This maintains high throughput with long prompts—without slowing down short-prompt responses.

*解耦式推理流水线* 将 prefill（提示处理）和 decode（生成）两个阶段分散到不同资源上，并基于实时因素（如 KV 缓存命中、队列深度和负载）进行上下文感知的请求路由。这样可以在处理长提示时保持高吞吐——而不拖慢短提示响应。

*Dynamic parallelism strategies* Perform on-the-fly decisions between data-parallel, tensor-parallel, pipeline-parallel, and hybrid execution combinations depending on input/output sequence lengths and model structure—including MoE routing. These decisions include replicating or sharding the model.

*动态并行策略* 根据输入/输出序列长度和模型结构（包括 MoE 路由），在数据并行、张量并行、流水线并行以及混合执行组合之间即时做出决策。这些决策包括复制或分片模型。

*Adaptive decoding and scheduling* Use techniques, such as speculative decoding, in-flight batch reshaping, and token-level scheduling, to improve throughput and latency. These techniques are implemented in engines like vLLM, SGLang, and NVIDIA TensorRT-LLM. This validates their effectiveness in production settings.

*自适应解码与调度* 使用诸如推测解码、飞行中批次重塑和 token 级调度等技术来改善吞吐和延迟。这些技术已在 vLLM、SGLang 和 NVIDIA TensorRT-LLM 等引擎中实现，验证了它们在生产环境中的有效性。

*Memory management and Unified Memory tuning* Utilize Grace CPU’s memory with unified addressing to offload infrequently used KV cache pages to CPU or NVMe. Using APIs like cudaMemAdvise and cudaMemPrefetchAsync for optimal placement. Make sure to use GPUDirect Storage, if available. This will directly page data from NVMe to GPU memory when needed—bypassing the CPU.

*内存管理与统一内存调优* 利用 Grace CPU 的内存与统一寻址，把不常用的 KV 缓存页卸载到 CPU 或 NVMe。使用像 cudaMemAdvise 和 cudaMemPrefetchAsync 这样的 API 来实现最优放置。如果可用，务必使用 GPUDirect Storage。这将在需要时把数据从 NVMe 直接分页到 GPU 内存——绕过 CPU。

*Profiling-driven optimization* Use tools like NVML, Nsight Systems/Compute, NVTX instrumentation, and Prometheus metrics to identify bottlenecks at runtime and automatically apply graph and kernel optimizations. Analyze telemetry using AI to detect anomalies and optimization opportunities.

*性能剖析驱动的优化* 使用像 NVML、Nsight Systems/Compute、NVTX 插桩和 Prometheus 指标这样的工具，在运行时识别瓶颈并自动应用计算图和核函数优化。用 AI 分析遥测数据，以检测异常和优化机会。

## Conclusion

## 结论

The techniques covered in this chapter transform a static inference deployment into a self-optimizing, adaptive engine. By monitoring runtime signals (latency, utilization, memory, network throughput) and applying strategies like dynamic parallelism, precision scaling, autotuned kernels, proactive caching, RL-based control, and smart scheduling, one can push ultralarge model inference to its limits.

本章介绍的技术把一个静态的推理部署转变为一个自优化、自适应的引擎。通过监控运行时信号（延迟、利用率、内存、网络吞吐量）并应用诸如动态并行、精度缩放、自动调优核函数、主动缓存、基于 RL 的控制以及智能调度等策略，人们可以把超大模型推理推向极限。

A successful inference service can handle massive amounts of users, large input contexts, lots of uploaded documents, extensive reasoning, and strict latency SLAs all while supporting massive model sizes. And it will do this cost-effectively since it won’t need as much hardware to achieve the same throughput.

一个成功的推理服务能够处理海量用户、大型输入上下文、大量上传文档、繁重推理以及严格的延迟 SLA，同时还支持巨大的模型规模。而且它会以低成本做到这一点，因为达到相同吞吐所需的硬件更少。

Remember that the network fabric is part of the system codesign—and not an afterthought. The idea is to keep the NVLink and NVSwitch cluster fabric fully utilized. This can improve throughput-scaling that approaches near-linear in ideal conditions—while maintaining low latency—as models and GPU clusters continue to grow. With proper scheduling, the GPUs in an NVLink/NVSwitch fabric behave like a tightly coupled accelerator—acting almost as a single large GPU from a software perspective.

请记住，网络交换结构是系统协同设计的一部分——而非事后补充。其思路是让 NVLink 和 NVSwitch 集群交换结构保持完全利用。随着模型和 GPU 集群持续增长，这可以在保持低延迟的同时改善吞吐扩展性——在理想条件下接近线性。有了恰当的调度，NVLink/NVSwitch 交换结构中的各个 GPU 表现得就像一个紧耦合的加速器——从软件角度看几乎就像一个单一的大型 GPU。

Every strategy is a tool in the AI system performance engineer’s toolkit. The most effective solutions usually combine these tools. For example, you might train an RL policy to decide when to switch parallelism modes or adjust precision—or use prewarming to keep a continuous batching scheduler primed and ready.

每一种策略都是 AI 系统性能工程师工具箱中的一件工具。最有效的解决方案通常会组合使用这些工具。例如，你可能训练一个 RL 策略来决定何时切换并行模式或调整精度——或者使用预热来让连续批处理调度器保持就绪待命。

And remember that you don’t have to implement these all at once. Even just one or two can produce noticeable improvements. Start with what’s easiest (e.g., caching and batching improvements), then layer in the others.

还请记住，你不必一次性实现所有这些技术。哪怕只做一两项，也能带来明显的改善。从最容易的入手（例如缓存和批处理方面的改进），然后再逐层叠加其他技术。

The theme with these techniques is flexibility and adaptability. The inference runtime should be able to reconfigure itself in response to the current workload. This is how you can turn a massive LLM into a well-tuned, scalable production service—efficiently and cost-effectively.

这些技术的主题是灵活性与适应性。推理运行时应当能够根据当前工作负载重新配置自身。这就是你把一个庞大的 LLM 变成一个精细调优、可扩展的生产服务的方法——既高效又经济。
