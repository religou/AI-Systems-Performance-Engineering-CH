# Chapter 16. Profiling, Debugging, and Tuning Inference at Scale

# 第 16 章. 大规模推理的剖析、调试与调优

Operating a large LLM inference cluster requires monitoring and debugging tools that make sure everything is running as expected. They also help you quickly identify bottlenecks when performance strays from its target.

运营大型 LLM 推理集群，需要监控与调试工具来确保一切按预期运行。当性能偏离目标时，它们还能帮你快速定位瓶颈。

In this chapter, we demonstrate how to monitor and debug these complex systems using tools such as NVIDIA Nsight Systems for profiling and Prometheus/Grafana for cluster-wide telemetry. We also show how to collect and interpret key metrics like GPU utilization, memory pressure, tail latency percentiles, cache hit rates, per-token timing, and more. These help guide our inference engine performance optimizations.

在本章中，我们将演示如何使用各类工具来监控和调试这些复杂系统，例如用 NVIDIA Nsight Systems 做剖析，用 Prometheus/Grafana 做集群级遥测。我们还会展示如何采集并解读关键指标，如 GPU 利用率、内存压力、尾延迟（tail latency）分位数、缓存命中率、逐 token 计时等等。这些指标可以指导我们对推理引擎做性能优化。

Next, we discuss operational performance tuning, including production-proven methods to optimize GPU utilization, reduce inference latency, and increase throughput in large clusters. This includes techniques for overlapping computation and communication, scheduling and batching requests, and using high-speed interconnects like NVLink, NVSwitch, and InfiniBand effectively.

接着，我们讨论运营层面的性能调优，包括在大型集群中优化 GPU 利用率、降低推理延迟、提升吞吐量的经生产验证的方法。这涵盖了计算与通信重叠、请求调度与批处理，以及高效使用 NVLink、NVSwitch、InfiniBand 等高速互连的技术。

We’ll also compare techniques for real-time quantization for inference, including methods to compress models to 8-bit and 4-bit precision using implementations like Generalized Post-Training Quantization (GPTQ) and Activation-Aware Weight Quantization (AWQ). Along the way, we’ll discuss the trade-offs between weight-only quantization versus quantizing both weights and activations. We provide practical guidance on applying quantization in serving pipelines to reduce memory usage and increase throughput—all while preserving model accuracy.

我们还会对比面向推理的实时量化（quantization）技术，包括用 Generalized Post-Training Quantization（GPTQ）和 Activation-Aware Weight Quantization（AWQ）等实现将模型压缩到 8 位与 4 位精度的方法。在此过程中，我们会讨论仅权重量化（weight-only quantization）与同时量化权重和激活之间的权衡。我们提供在服务流水线中应用量化以降低内存占用、提升吞吐量的实用指南——同时保持模型精度不下降。

Finally, we consider application-level optimizations that complement low-level performance tuning. These include strategies like prompt compression, prefix caching, deduplication, query routing (e.g., fallback models), and partial-output streaming.

最后，我们探讨与底层性能调优互补的应用级优化（application-level optimization），包括提示压缩（prompt compression）、前缀缓存（prefix caching）、去重、查询路由（例如回退模型），以及部分输出流式返回等策略。

## Profiling, Debugging, and Tuning Inference Performance

## 推理性能的剖析、调试与调优

There are a lot of moving parts in modern LLM inference engines—especially with disaggregated prefill and decode. The lifecycle of a typical request involves many components, as shown in Figure 16-1.

现代 LLM 推理引擎有大量相互关联的环节——尤其是在 prefill 与 decode 分离（disaggregated）的情况下。一个典型请求的生命周期涉及许多组件，如图 16-1 所示。

![Figure 16-1. Lifecycle of a typical request in a disaggregated prefill and decode LLM inference system](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-1.png)

![图 16-1. prefill 与 decode 分离的 LLM 推理系统中典型请求的生命周期](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-1.png)

Given such complexity, the workflow for tuning inference performance is very iterative. It requires careful tuning and continuous verification.

面对如此复杂度，推理性能调优的工作流高度迭代。它需要细致的调优与持续的验证。

First, you should observe metrics and identify the current bottleneck, including a GPU not fully utilized or higher-than-expected latency. Next, form a hypothesis for improvement, such as “increase the batch size” or “increase communication-computation overlap for operation X.” Then, implement a fix and test the hypothesis.

首先，你应当观察指标，识别当前的瓶颈，例如某块 GPU 未被充分利用，或延迟高于预期。接着，提出改进假设，例如“增大批大小”或“为操作 X 增加通信与计算的重叠”。然后，实现修复方案并验证该假设。

Then, you should ideally test the fix in a staging environment with a representative workload using profiling tools to validate that the change behaves as expected. For instance, you can verify that an operation is demonstrating proper memory and compute overlap.

之后，理想情况下你应当在预发布（staging）环境中，用具有代表性的负载配合剖析工具测试该修复，验证改动的行为符合预期。例如，你可以确认某个操作确实呈现出恰当的内存与计算重叠。

Last, you should deploy the fix into production and monitor Grafana and the logs to validate that the fix improved throughput and latency on a real workload. Repeat this workflow as new bottlenecks appear.

最后，你应将修复部署到生产环境，并监控 Grafana 与日志，验证该修复确实在真实负载上提升了吞吐量与延迟。当新的瓶颈出现时，重复这个工作流。

This observe-hypothesize-tune loop should be continuous. Modern deployments often automate these steps. For instance, you can use scheduled load tests—and subsequent anomaly detection on key metrics—to trigger a tuning workflow.

这套“观察—假设—调优”的循环应当持续进行。现代部署往往会将这些步骤自动化。例如，你可以用定期的负载测试——以及随后对关键指标的异常检测——来触发一次调优工作流。

> It’s recommended to perform canary rollouts when deploying optimizations to production, including updated inference runtimes and model variants. By deploying the optimization to a small subset of traffic running on a small number of servers, you can validate the optimizations before full production deployment to all end users. This incremental approach helps reduce the “blast radius” of unexpected side effects by catching them early without impacting all users.

> 在向生产环境部署优化（包括更新后的推理运行时和模型变体）时，建议执行金丝雀（canary）发布。通过先把优化部署到运行在少量服务器上的一小部分流量，你可以在向所有终端用户全量部署之前先验证优化效果。这种增量式方法通过尽早发现意外副作用，帮助缩小其“爆炸半径”，避免影响所有用户。

Consider a scenario in which host-side CPU utilization is spiking to 100% due to excessive tokenization or inference data preprocessing. This will limit how many concurrent streams the inference engine can handle. One fix might be to move the preprocessing to the GPU using a GPU-accelerated tokenizer library or a custom GPU kernel written in CUDA or OpenAI’s Triton language.

设想这样一个场景：由于过多的分词或推理数据预处理，主机侧 CPU 利用率飙升到 100%。这会限制推理引擎能处理的并发流数量。一种修复办法是用 GPU 加速的分词库，或用 CUDA、OpenAI 的 Triton 语言编写的自定义 GPU 核函数，把预处理迁移到 GPU 上。

After deploying the new library or kernel, you should monitor CPU utilization before and after. If you see the CPU utilization decrease and overall throughput increase, then the system is no longer bottlenecked by the CPU-based input preprocessing.

在部署新的库或核函数后，你应当对比部署前后的 CPU 利用率。如果看到 CPU 利用率下降、整体吞吐量上升，那么系统就不再受基于 CPU 的输入预处理所瓶颈约束了。

You should also pay attention to cache hit rates for any type of cache, including prefix cache, prompt-embedding cache, and KV cache. You should have metrics for “cache hits” versus “cache misses.” A high cache-hit rate means that the system is reusing data effectively. In contrast, if you see a high cache-miss rate, then you likely want to tune the cache size, eviction policy, or caching strategy to maximize cache hits.

你还应关注各类缓存的命中率，包括前缀缓存、提示嵌入缓存和 KV 缓存（KV cache）。你应当有“缓存命中”对比“缓存未命中”的指标。高缓存命中率意味着系统在有效复用数据。反之，如果你看到很高的缓存未命中率，那多半需要调整缓存大小、逐出（eviction）策略或缓存策略，以最大化缓存命中。

vLLM’s LMCache component allows adjusting the GPU versus CPU cache ratio. If misses are high due to GPU memory limits, you can enable its paged cache offload capability so that the CPU can help out. Always make sure your cache eviction policy (Least Recently Used [LRU], Least Frequently Used [LFU], etc.) aligns with the access patterns.

vLLM 的 LMCache 组件允许调整 GPU 与 CPU 的缓存比例。如果因 GPU 内存受限导致未命中率偏高，你可以启用它的分页缓存卸载能力，让 CPU 来分担。务必确保你的缓存逐出策略（Least Recently Used [LRU]、Least Frequently Used [LFU] 等）与访问模式相匹配。

Another scenario is using a KV cache to reuse data for identical input-sequence prefixes among batched requests to avoid recomputing the KV for the prefix. In this case, you want to measure how often requests share a prefix. This leads to a *prefix* *merge* event and increments the prefix cache hit metrics in vLLM including vllm:gpu_prefix_cache_queries and vllm:gpu_prefix_cache_hits. These let you compute the hit rate as hits per queries, for example.

另一个场景是使用 KV 缓存，在批处理请求之间复用相同输入序列前缀的数据，从而避免为该前缀重新计算 KV。这时，你需要度量请求共享前缀的频率。这会触发一次*前缀**合并*（prefix merge）事件，并递增 vLLM 中的前缀缓存命中指标，包括 vllm:gpu_prefix_cache_queries 和 vllm:gpu_prefix_cache_hits。例如，借助它们你就能用“命中数 ÷ 查询数”算出命中率。

Measuring prefix-merge rates helps you correlate with actual cache-hit rates to gauge the real benefit of your caching layer. This way, you can adjust batching and scheduling policies to maximize shared prefixes—and predict end-to-end throughput and latency improvements under different workloads.

度量前缀合并率有助于你将其与实际缓存命中率相关联，从而衡量缓存层的真实收益。这样，你就能调整批处理与调度策略以最大化共享前缀——并预测不同负载下端到端的吞吐量与延迟改善。

You can run synthetically generated data on the inference engine to test prompts with many repeated prefixes. Hopefully, you will see a reduction in prefill compute due to the prefix merging.

你可以在推理引擎上运行合成生成的数据，测试包含大量重复前缀的提示。理想情况下，你会看到因前缀合并而带来的 prefill 计算量下降。

Modern LLM inference engines like vLLM and SGLang expose prefix-merge metrics natively. But if prefix-merging is not a first-class metric exported by your inference engine, you should instrument a custom counter for “prefix deduplicated tokens” to monitor its effectiveness.

vLLM 和 SGLang 等现代 LLM 推理引擎原生暴露前缀合并指标。但如果前缀合并不是你的推理引擎导出的一等指标，你应当自行埋点一个“前缀去重 token 数”的自定义计数器（counter）来监控其有效性。

> If you see that prefix merging is not performing as expected, you should check if the prefix-matching logic is failing. Start the debugging process by checking if there are tokenizer differences. This is the likely cause of most prefix-matching issues.

> 如果你发现前缀合并的表现不如预期，应当检查前缀匹配逻辑是否失效。调试时先从检查分词器差异入手。这是绝大多数前缀匹配问题的可能根因。

In addition to performance, monitoring helps with capacity planning. By tracking how utilization and latency behave as load increases, you can project at what point the system will hit a particular limit, such as p95 (95th percentile) latency starting to rise exponentially. In this case, the dynamic batch size might be increasing to a point of diminishing return.

除了性能之外，监控还有助于容量规划。通过跟踪利用率和延迟随负载上升的表现，你可以推断系统在何时会触及某个特定上限，例如 p95（第 95 百分位）延迟开始呈指数上升。这种情况下，动态批大小可能正在增大到收益递减的临界点。

> If you are using a tiered caching strategy, including an NVMe-based KV cache extension, make sure to monitor the I/O latency of the device. High I/O latency will significantly decrease cache performance.

> 如果你采用分层缓存策略，包括基于 NVMe 的 KV 缓存扩展，务必监控该设备的 I/O 延迟。高 I/O 延迟会显著降低缓存性能。

When per-GPU concurrency reaches its limit and further batch-size increases no longer increase throughput, you may want to scale out by adding more GPUs, deploying additional model replicas, or increasing the number of experts to distribute work across more compute units.

当单 GPU 的并发达到上限、进一步增大批大小不再提升吞吐量时，你可以考虑横向扩展（scale out）：增加更多 GPU、部署额外的模型副本，或增加专家数量，把工作分摊到更多计算单元上。

You should also consider model compression—or switching to lower precision (FP8/FP4)—to get more effective throughput per GPU before scaling out. However, once hardware is saturated (e.g., SMs at 100% and memory bandwidth near peak utilization), adding more GPUs or using tensor/pipeline parallelism is likely the only path to higher throughput.

在横向扩展之前，你还应考虑模型压缩——或切换到更低精度（FP8/FP4）——以获得每块 GPU 更高的有效吞吐量。不过，一旦硬件饱和（例如 SM 达到 100%、内存带宽接近峰值利用），增加更多 GPU 或使用张量/流水线并行很可能就是提升吞吐量的唯一路径了。

And remember to always weigh the cost of new hardware against the efficiency gains. There are times when upgrading to newer GPUs with more memory and FLOPS will be more cost-effective than scaling out a fleet of older GPUs.

而且要记住，始终权衡新硬件的成本与效率收益。有时候，升级到内存更大、FLOPS 更高的更新款 GPU，会比横向扩展一批老旧 GPU 更具成本效益。

Increasing the expert count can raise your throughput ceiling—but only if you also improve expert routing and scheduling to manage the extra all-to-all communication. Otherwise, naive scaling may simply shift the bottleneck to the network. Next, let’s discuss monitoring and how to verify that our optimization efforts are actually paying off.

增加专家数量可以抬高你的吞吐量上限——但前提是你同时改进专家路由与调度，以应对额外的全对全（all-to-all）通信。否则，粗放的扩展可能只是把瓶颈转移到网络上。接下来，我们讨论监控，以及如何验证优化努力是否真正见效。

### Monitoring System Metrics and Counters

### 监控系统指标与计数器

Unlike traditional microservice invocations, which are relatively uniform and predictable in their execution time, LLM requests are nonuniform and can vary wildly in terms of latency. This difference is shown in Figure 16-2.

传统微服务（microservice）调用的执行时间相对统一且可预测，而 LLM 请求则不然：它们不统一，延迟可能天差地别。这一差异如图 16-2 所示。

![Figure 16-2. Difference between traditional microservice invocations and LLM invocations](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-2.png)

![图 16-2. 传统微服务调用与 LLM 调用之间的差异](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-2.png)

For ongoing monitoring in production, it’s common to use Prometheus to collect metrics from each GPU compute node—as well as Grafana dashboards to visualize them. Key GPU metrics to track include GPU utilization (percent of time the SMs are busy), GPU memory usage, copy engine utilization, PCIe and NVLink throughput, and GPU temperature and power (e.g., throttling). Note: Low-level counters such as L1 and L2 activity, occupancy, and instruction throughput can be collected with Nsight Compute or CUPTI rather than DCGM and Prometheus.

对于生产中的持续监控，常见做法是用 Prometheus 从每个 GPU 计算节点采集指标——并用 Grafana 仪表盘将其可视化。需要跟踪的关键 GPU 指标包括 GPU 利用率（SM 处于繁忙状态的时间占比）、GPU 内存用量、拷贝引擎利用率、PCIe 与 NVLink 吞吐量，以及 GPU 温度和功耗（例如降频）。注：L1、L2 活动、占用率（occupancy）和指令吞吐等底层计数器，可用 Nsight Compute 或 CUPTI 采集，而非 DCGM 与 Prometheus。

> The cudaMemPool metrics and asynchronous allocator statistics are helpful when monitoring memory fragmentation. These should be integrated into your monitoring, as this will greatly facilitate debugging system performance issues in production.

> 在监控内存碎片时，cudaMemPool 指标和异步分配器统计信息很有帮助。应将它们整合进你的监控体系，因为这会极大方便在生产中调试系统性能问题。

It’s also important to monitor interconnect utilization, including NVLink, NVSwitch bandwidth, and NIC throughput. This way, you will catch communication bottlenecks in multi-GPU and multinode cluster configurations.

监控互连利用率同样重要，包括 NVLink、NVSwitch 带宽和网卡（NIC）吞吐。这样，你就能在多 GPU 和多节点集群配置中捕捉到通信瓶颈。

NVIDIA’s Data Center GPU Manager (DCGM) exposes many GPU metrics, which Prometheus can scrape and collect. For instance, DCGM provides DCGM_FI_DEV_GPU_UTIL for SM utilization %, DCGM_FI_DEV_MEM_COPY_UTIL for memory copy engine utilization, and DCGM_FI_DEV_FB_USED for framebuffer memory used, among others.

NVIDIA 的 Data Center GPU Manager（DCGM）暴露了许多 GPU 指标，Prometheus 可以抓取并采集它们。例如，DCGM 提供 DCGM_FI_DEV_GPU_UTIL 表示 SM 利用率百分比，DCGM_FI_DEV_MEM_COPY_UTIL 表示内存拷贝引擎利用率，DCGM_FI_DEV_FB_USED 表示已用的帧缓冲内存等等。

DCGM exposes NVLink error counters and can expose throughput counters on some platforms and driver versions. For sustained link utilization, also use nvidia-smi nvlink and Nsight tools. You should integrate these metrics into your dashboards and set up alerts to help identify when the network is saturated with cross-GPU and cross-node communication traffic. DCGM tracks Xid counters, as well as critical GPU errors.

DCGM 暴露 NVLink 错误计数器，并可在部分平台和驱动版本上暴露吞吐计数器。对于持续的链路利用率，还应使用 nvidia-smi nvlink 和 Nsight 工具。你应当把这些指标整合进仪表盘，并设置告警，以便识别网络何时被跨 GPU 和跨节点的通信流量占满。DCGM 会跟踪 Xid 计数器以及严重的 GPU 错误。

> While DCGM exposes NVLink counters, as of this writing, dcgm-exporter does not expose per-link bandwidth on all platforms by default. So if you need link-level throughput, you may need to query DCGM directly or extend the exporter.

> 尽管 DCGM 暴露了 NVLink 计数器，但截至本文撰写时，dcgm-exporter 默认并不在所有平台上暴露每条链路的带宽。因此，如果你需要链路级吞吐，可能需要直接查询 DCGM 或扩展该 exporter。

It’s also recommended to collect high-level application metrics like queries/requests per second, average latency and p95/p99 latency, number of active contexts, and throughput in tokens/sec. Metrics on KV cache utilization and size (overall and per node) are extremely important to monitor as well.

同样建议采集高层应用指标，如每秒查询/请求数、平均延迟以及 p95/p99 延迟、活跃上下文数量，以及以 token/秒计的吞吐量。KV 缓存利用率与大小（整体的以及每个节点的）相关指标也极其重要，需要监控。

You can set up Prometheus node exporters to gather all of these metrics from each node, collect the data in one place, and even set up alerts for critical thresholds. Grafana can then plot these metrics for real-time dashboards to share with your team. Figure 16-3 shows how the metrics are collected from each GPU in a Kubernetes cluster and exported to Prometheus to be visualized with Grafana, for example.

你可以设置 Prometheus node exporter 从每个节点采集所有这些指标，把数据汇集到一处，甚至为关键阈值设置告警。随后，Grafana 就能将这些指标绘制成实时仪表盘，与团队共享。例如，图 16-3 展示了如何从 Kubernetes 集群中的每块 GPU 采集指标，导出到 Prometheus，再用 Grafana 可视化。

This way, when you deploy a new optimization to increase batching, for instance, Grafana will immediately show if GPU utilization on each GPU increases. You can also monitor to make sure p95/p99 latencies stay within the target.

这样，当你部署一项新的优化（比如增大批处理）时，Grafana 会立即显示每块 GPU 的利用率是否上升。你还可以监控以确保 p95/p99 延迟保持在目标范围内。

Counters are extremely useful to measure as well—especially with dynamic and adaptive systems. For instance, if your inference engine dynamically adapts the batch size to current conditions, you may want to increment a “batch size change” counter.

计数器同样极其有用——尤其对于动态自适应系统。例如，如果你的推理引擎会根据当前状况动态调整批大小，你可能会想递增一个“批大小变更”计数器。

![Figure 16-3. DCGM collects metrics from the Kubernetes GPU nodes and sends them to Prometheus](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-3.png)

![图 16-3. DCGM 从 Kubernetes GPU 节点采集指标并发送到 Prometheus](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-3.png)

The other option is to log the change in a logfile, but this would require a slow, text-based search/aggregation to analyze the logfile offline using Apache Spark, for instance. You would then need to manually correlate the result of the logfile analysis with the Prometheus metrics.

另一种选择是把变更记录到日志文件里，但这样一来，就需要用 Apache Spark 之类的工具对日志文件做缓慢的、基于文本的搜索/聚合来离线分析。之后，你还得手动把日志分析结果与 Prometheus 指标相关联。

By incrementing a simple counter for interesting application-level events (including errors), the data is pushed to Prometheus and instantly viewable in the Grafana dashboard alongside all of the other metrics in real time. In addition, consider using structured logging and distributed tracing for critical events.

只要为感兴趣的应用级事件（包括错误）递增一个简单的计数器，数据就会被推送到 Prometheus，并可在 Grafana 仪表盘中与所有其他指标一起实时查看。此外，考虑对关键事件使用结构化日志与分布式追踪。

Modern application performance management (APM) tools—as well as OpenTelemetry—can ingest these logs/traces and correlate them with metrics. This provides a consistent timeline view of events across the entire system. Having insight into this timeline will help speed up the time it takes to debug performance issues.

现代应用性能管理（application performance management，APM）工具——以及 OpenTelemetry——可以摄取这些日志/追踪，并将它们与指标相关联。这样就能提供跨整个系统的一致的事件时间线视图。洞察这条时间线，有助于加快调试性能问题的速度。

If you continuously monitor these metrics, you gain insight into where to tune next. For instance, if GPU utilization is below expected, you can check if GPU memory is maxed out or not. If it’s not fully utilized, you can try to increase the batch size or maximum number of concurrent requests. Make sure to keep an eye on latency service-level objectives (SLOs), however. You don’t want to exceed these.

如果你持续监控这些指标，就能获知下一步该调优什么。例如，如果 GPU 利用率低于预期，你可以检查 GPU 内存是否已被占满。如果尚未充分利用，你可以尝试增大批大小或最大并发请求数。不过，务必留意延迟服务级目标（service-level objective，SLO）。你可不希望突破这些目标。

> Modern inference servers expose a “maximum latency” setting for dynamic batching. Tune this to meet your SLOs. Increasing it raises the batch size (throughput). Increasing it too much will hurt p99 latency. Continuously adjust this in light of your latency targets.

> 现代推理服务器会为动态批处理暴露一个“最大延迟”设置。调整它以满足你的 SLO。调大它会增大批大小（提升吞吐量）。但调得过大又会损害 p99 延迟。请结合你的延迟目标持续调整。

In contrast, if GPU memory is near maximum, the inference engine might start swapping out inactive KV cache data to host CPU memory or NVMe storage. This will reduce GPU utilization, as the GPUs need to wait for the additional data transfers from slow CPU memory or disk.

反之，如果 GPU 内存接近上限，推理引擎可能开始把不活跃的 KV 缓存数据换出到主机 CPU 内存或 NVMe 存储。这会降低 GPU 利用率，因为 GPU 需要等待来自慢速 CPU 内存或磁盘的额外数据传输。

> If you see a spike in GPU memory copy-engine utilization—or abnormal NVLink utilization that aligns with low SM utilization—your inference engine is likely swapping KV cache data in and out of GPU memory. This will bottleneck your system due to excessive data transfer latency.

> 如果你看到 GPU 内存拷贝引擎利用率出现尖峰——或者出现与低 SM 利用率同步的异常 NVLink 利用率——那么你的推理引擎很可能正在把 KV 缓存数据换入换出 GPU 内存。这会因过多的数据传输延迟而使系统受阻。

If you are swapping, you can adjust the inference engines’ paging parameters to reduce thrashing, apply FP8 or FP4 quantization, increase GPU memory allocation for cache, and potentially change the swapping strategy. This should bring copy utilization down and compute utilization up—exactly what you want to see.

如果确实在发生换页，你可以调整推理引擎的分页参数以减少抖动（thrashing），应用 FP8 或 FP4 量化，为缓存增大 GPU 内存分配，并可能改变换页策略。这应当能把拷贝利用率降下来、把计算利用率提上去——正是你想看到的结果。

Grafana is also used for latency tracking. You can plot the distribution of end-to-end request latency—often measuring both prefill latency and per-token latency as well. If the p99 latency spikes at certain times, you should correlate it with GPU metrics and other logs.

Grafana 同样可用于延迟跟踪。你可以绘制端到端请求延迟的分布——通常还会同时度量 prefill 延迟和逐 token 延迟。如果 p99 延迟在某些时刻出现尖峰，你应当把它与 GPU 指标和其他日志相关联。

For instance, a p99 latency spike might correlate with a period when GPU utilization drops. Perhaps the latency spike correlates with a traffic surge that triggered a larger dynamic batch size. This could lead to higher latency for that period of time. To verify, you can overlay RPS (requests per second) on the latency graph in the Grafana dashboard to see if the two charts correlate.

例如，p99 延迟尖峰可能与 GPU 利用率下降的时段相关。也许这个延迟尖峰与一次触发了更大动态批大小的流量激增相关。这会导致该时段内延迟升高。为了验证，你可以在 Grafana 仪表盘的延迟图上叠加 RPS（每秒请求数），看看这两张图是否相关。

If the spike was expected due to a dynamic increase in batch size per our hypothesis, make sure it isn’t exceeding your service-level agreement (SLA). If it is, you can try decreasing the maximum request-batch queue delay or reducing the maximum batch size to put a limit on the latency.

如果这个尖峰正如我们假设的那样，是由批大小的动态增大所致，那么务必确认它没有突破你的服务级协议（service-level agreement，SLA）。如果突破了，你可以尝试减小最大请求批队列延迟，或降低最大批大小，以给延迟设一个上限。

Logs are invaluable when diagnosing issues as well. You should instrument the code to log key events such as when a batch is formed, when a communication starts/ends, etc. It’s best to use the DEBUG level so that you can enable/disable it as needed—and not impact request-response latency.

诊断问题时，日志同样极其宝贵。你应当在代码中埋点，记录关键事件，例如批何时形成、通信何时开始/结束等。最好使用 DEBUG 级别，以便按需启用/禁用——从而不影响请求-响应延迟。

When you enable debug logging, you’ll see a step-by-step timeline in text format. In one debugging session, it’s likely that you’ll use both the logging timeline and Prometheus/Grafana metrics together. For instance, you can see how often an all-to-all communication takes longer than 5 ms.

当你启用调试日志时，会看到一条文本格式的逐步时间线。在一次调试会话中，你很可能会把日志时间线与 Prometheus/Grafana 指标结合起来一起用。例如，你可以看到全对全通信耗时超过 5 ms 的频率有多高。

With the combination of log-based timeline and metrics, you can see outliers such as network issues that may have slowed down one iteration in the all-to-all communication exchange. If this continues to happen, you can raise the expert capacity factor so that any excess tokens automatically spill over to a secondary expert replica—ideally hosted on a GPU with a more stable network path. This will balance the load and minimize the latency.

把基于日志的时间线与指标结合起来，你就能发现离群点，比如某些网络问题可能拖慢了全对全通信交换中的某一次迭代。如果这种情况持续发生，你可以调高专家容量因子（capacity factor），让任何超额的 token 自动溢出到备用专家副本上——理想情况下，该副本托管在网络路径更稳定的 GPU 上。这会均衡负载并把延迟降到最低。

> In practice, setting the capacity factor to 1.2–1.5 is common, as this allows 20%–50% extra tokens to be reassigned when a primary expert is overloaded. This can significantly smooth out tail latency in MoE inference. Spilling over to a second expert is better than queuing requests behind a slightly stalled expert on a GPU with a degraded interconnect. This will reduce sensitivity to outliers if your network continues to experience issues for whatever reason.

> 实践中，把容量因子设为 1.2–1.5 是常见做法，因为这允许在主专家过载时，把额外 20%–50% 的 token 重新分配出去。这能显著平滑 MoE 推理中的尾延迟。当某个专家所在 GPU 的互连状况变差、略有停顿时，与其把请求排队堵在它后面，不如溢出到第二个专家。如果你的网络出于某种原因持续出现问题，这样做会降低系统对离群点的敏感度。

### Profiling with Nsight Systems and Nsight Compute

### 用 Nsight Systems 与 Nsight Compute 做剖析

When developing and tuning your inference code, you can use Nsight Systems to capture traces of the workload across both the CPU and GPU. Nsight Systems provides a timeline view that shows CPU threads, GPU kernels, CUDA events, NCCL communications, and more using microsecond resolution.

在开发和调优推理代码时，你可以用 Nsight Systems 捕获跨 CPU 与 GPU 的工作负载追踪。Nsight Systems 提供时间线视图，以微秒级分辨率展示 CPU 线程、GPU 核函数、CUDA 事件、NCCL 通信等等。

By instrumenting your code with NVTX annotations, we can label regions like “Prefill stage,” “Decode step,” or “All-to-all communication” on the timeline for clarity. The following code shows NVTX range markers around example prefill and decode steps using the NVTX v3 C API with explicit push and pop ranges:

通过用 NVTX 注解为代码埋点，我们可以在时间线上清晰地标注出“Prefill 阶段”“Decode 步骤”或“全对全通信”等区域。下面的代码展示了如何使用 NVTX v3 C API 的显式 push 与 pop 区间，在示例 prefill 与 decode 步骤周围加上 NVTX 区间标记：

```
// Example C++ snippet with NVTX annotations using the C API.
#include <nvtx3/nvToolsExt.h>   // or <nvToolsExt.h>
#include "my_model.hpp"         // Your model’s C++ interface
#include <vector>
// Small helpers to keep callsites tidy.
#define NVTX_PUSH(name, argb)                                           \
  do {                                                                  \
    nvtxEventAttributes_t a{};                                          \
    a.version = NVTX_VERSION;                                           \
    a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                             \
    a.colorType = NVTX_COLOR_ARGB;                                      \
    a.color = (unsigned int)(argb);                                     \
    a.messageType = NVTX_MESSAGE_TYPE_ASCII;                            \
    a.message.ascii = (name);                                           \
    nvtxRangePushEx(&a);                                                \
  } while (0)
#define NVTX_POP() do { nvtxRangePop(); } while (0)
struct Token { int id; };
void run_inference(
    const std::vector<Token>& prompt_tokens,
    Model& model,
    int num_generate_steps) {
  // Prefill
  NVTX_PUSH("Prefill", 0xFF4F86F7);
  model.encode(prompt_tokens);
  NVTX_POP();
  // Decode one token at a time
  for (int t = 0; t < num_generate_steps; ++t) {
    NVTX_PUSH("Decode", 0xFFFF8C00);
    Token next_token = model.decode_next();
    // ... (sampling / streaming to client)
    NVTX_POP();
  }
}
```

```
// Example C++ snippet with NVTX annotations using the C API.
#include <nvtx3/nvToolsExt.h>   // or <nvToolsExt.h>
#include "my_model.hpp"         // Your model’s C++ interface
#include <vector>
// Small helpers to keep callsites tidy.
#define NVTX_PUSH(name, argb)                                           \
  do {                                                                  \
    nvtxEventAttributes_t a{};                                          \
    a.version = NVTX_VERSION;                                           \
    a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;                             \
    a.colorType = NVTX_COLOR_ARGB;                                      \
    a.color = (unsigned int)(argb);                                     \
    a.messageType = NVTX_MESSAGE_TYPE_ASCII;                            \
    a.message.ascii = (name);                                           \
    nvtxRangePushEx(&a);                                                \
  } while (0)
#define NVTX_POP() do { nvtxRangePop(); } while (0)
struct Token { int id; };
void run_inference(
    const std::vector<Token>& prompt_tokens,
    Model& model,
    int num_generate_steps) {
  // Prefill
  NVTX_PUSH("Prefill", 0xFF4F86F7);
  model.encode(prompt_tokens);
  NVTX_POP();
  // Decode one token at a time
  for (int t = 0; t < num_generate_steps; ++t) {
    NVTX_PUSH("Decode", 0xFFFF8C00);
    Token next_token = model.decode_next();
    // ... (sampling / streaming to client)
    NVTX_POP();
  }
}
```

Here, we mark regions explicitly with nvtxRangePushEx/nvtxRangePop. We push a "Prefill" range immediately before model.encode(...) and pop it right after. Inside the decode loop, we push "Decode" at the top of each iteration and pop it after model.decode_next(). The small NVTX_PUSH/NVTX_POP helpers also attach color (hex values) and text. This helps to reduce mismatch in timeline visualizations—while keeping call sites concise. The explicit push/pop pairings are clearly visible in the code, which makes them easy to audit.

在这里，我们用 nvtxRangePushEx/nvtxRangePop 显式标注区域。我们在 model.encode(...) 之前紧接着 push 一个 "Prefill" 区间，并在其后立即 pop。在 decode 循环内，我们在每次迭代开头 push "Decode"，并在 model.decode_next() 之后 pop。小巧的 NVTX_PUSH/NVTX_POP 辅助宏还会附上颜色（十六进制值）和文本。这有助于减少时间线可视化中的错配——同时保持调用点简洁。显式的 push/pop 配对在代码中清晰可见，便于审查。

The colored block annotations will appear in the Nsight Systems GPU activity timeline labeled "Prefill" and "Decode". This makes it easy to see how long each phase takes—and how the phases overlap with communication operations. This helps to identify issues such as GPU idle gaps and unexpected synchronizations.

带颜色的区块注解会出现在 Nsight Systems 的 GPU 活动时间线上，标注为 "Prefill" 和 "Decode"。这让你很容易看出每个阶段耗时多久——以及这些阶段如何与通信操作重叠。这有助于识别诸如 GPU 空闲间隙和意外同步之类的问题。

> Note that we use the NVTX C API (nvToolsExt) directly rather than PyTorch’s record_function(). This lets us annotate hot paths in a pure C++ runtime and keeps the markers consistent when work is launched from Python or other languages.

> 注：请注意，我们直接使用 NVTX C API（nvToolsExt），而非 PyTorch 的 record_function()。这让我们能在纯 C++ 运行时中标注热点路径，并在工作从 Python 或其他语言发起时保持标记一致。

By tightening the scope to the smallest region necessary around model.encode(prompt_tokens), the profiling marker covers exactly the prefill work and no other code. This improves trace clarity and performance diagnostics.

通过把范围收紧到 model.encode(prompt_tokens) 周围所需的最小区域，剖析标记就恰好覆盖 prefill 工作，而不涉及其他代码。这提升了追踪的清晰度和性能诊断效果。

You should use per-stream ranges when enqueuing work on multiple CUDA streams (e.g., dedicated “transfer” stream for H2D/D2H copies and “compute” stream for kernels). To do this, you can wrap the host code for each stream with distinct NVTX ranges.

在多个 CUDA 流上入队工作时（例如用专门的“传输”流做 H2D/D2H 拷贝、用“计算”流跑核函数），你应当使用按流划分的区间。为此，你可以用各不相同的 NVTX 区间把每条流的主机端代码包裹起来。

For instance, you can name streams using nvtxNameCudaStreamA(transfer_stream, "transfer_stream") and nvtxNameCudaStreamA(compute_stream, "compute_stream"), for instance. You would then use nvtxRangePushA("transfer_stream")

例如，你可以用 nvtxNameCudaStreamA(transfer_stream, "transfer_stream") 和 nvtxNameCudaStreamA(compute_stream, "compute_stream") 为流命名。然后，你会在内存拷贝/传输周围使用 nvtxRangePushA("transfer_stream")

and nvtxRangePop() around memory copies/transfers and nvtxRangePushA("compute_stream") and nvtxRangePop() around kernel launches.

和 nvtxRangePop()，在核函数启动周围使用 nvtxRangePushA("compute_stream") 和 nvtxRangePop()。

Using NVTX-named streams makes overlap (or lack thereof) obvious in the Nsight Systems timeline. Here is some code that demonstrates how these all fit together:

使用 NVTX 命名的流，会让重叠（或缺乏重叠）在 Nsight Systems 时间线上一目了然。下面这段代码演示了这些如何组合到一起：

```
// One-time after creating the streams
nvtxNameCudaStreamA(transfer_stream, "transfer_stream");
nvtxNameCudaStreamA(compute_stream,  "compute_stream");
// Around H2D/D2H copies (transfer stream)
nvtxRangePushA("transfer_stream");
cudaMemcpyAsync(h_logits, d_logits, bytes, cudaMemcpyDeviceToHost,
                transfer_stream);
nvtxRangePop();
// Around kernel enqueues (compute stream)
nvtxRangePushA("compute_stream");
my_kernel<<<grid, block, 0, compute_stream>>>(...);
nvtxRangePop();
```

```
// One-time after creating the streams
nvtxNameCudaStreamA(transfer_stream, "transfer_stream");
nvtxNameCudaStreamA(compute_stream,  "compute_stream");
// Around H2D/D2H copies (transfer stream)
nvtxRangePushA("transfer_stream");
cudaMemcpyAsync(h_logits, d_logits, bytes, cudaMemcpyDeviceToHost,
                transfer_stream);
nvtxRangePop();
// Around kernel enqueues (compute stream)
nvtxRangePushA("compute_stream");
my_kernel<<<grid, block, 0, compute_stream>>>(...);
nvtxRangePop();
```

Here, we name the streams and wrap the enqueue sites in per-stream ranges so the Nsight timeline stays readable. It’s important to note that the NVTX ranges annotate the host-thread timeline. The GPU lanes show kernels/memcpys by stream. Naming the streams helps tie the host ranges to the right GPU lanes during analysis.

在这里，我们为流命名，并用按流划分的区间把入队点包裹起来，让 Nsight 时间线保持可读。需要注意的一点是，NVTX 区间标注的是主机线程时间线。GPU 通道则按流展示核函数/内存拷贝。为流命名有助于在分析时把主机区间与正确的 GPU 通道对应起来。

Nsight Compute lets us profile individual kernels to pinpoint inefficiencies. We can use the Nsight Compute’s section-based profiling feature to focus on specific parts of the kernel, such as memory transactions.

Nsight Compute 让我们能剖析单个核函数，以精确定位低效之处。我们可以使用 Nsight Compute 基于区段（section）的剖析功能，聚焦于核函数的特定部分，例如内存事务。

Another super useful tool that isn’t well known is Nsight Compute’s CUDA Program Counter (PC) Sampling feature. This samples program counters and identifies hotspots without requiring full, heavyweight instrumentation, as shown in Figure 16-4.

另一个鲜为人知却极其有用的工具，是 Nsight Compute 的 CUDA Program Counter（PC）Sampling 功能。它对程序计数器进行采样并识别热点，而无需完整、重量级的埋点，如图 16-4 所示。

![Figure 16-4. Nsight Compute’s CUDA Program Counter (PC) sampling feature helps identify hotspots in a low-overhead manner (source: https://oreil.ly/DyKWR)](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-4.png)

![图 16-4. Nsight Compute 的 CUDA 程序计数器采样（Program Counter sampling，PC sampling）功能以低开销的方式帮助识别热点（来源：https://oreil.ly/DyKWR）](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-4.png)

Specifically, we can use this to profile live inference servers and pinpoint exactly which kernel instructions are taking the most time. And we can do this in a low-overhead manner. Now that we’ve covered profiling with Nsight Systems and Nsight Compute, let’s discuss some common troubleshooting recipes for inference.

具体来说，我们可以用它来剖析在线运行的推理服务器，精确定位到底是哪些核函数指令耗时最多。而且我们能以低开销的方式做到这一点。既然我们已经讲完了用 Nsight Systems 与 Nsight Compute 做剖析，接下来就讨论一些常见的推理排障方案（troubleshooting recipe）。

> In production investigations on live services, prefer Program Counter Sampling first to localize hotspots with minimal overhead. Only switch to full tracing if the sample points to a specific kernel or phase.

> 在对在线服务做生产环境排查时，优先使用程序计数器采样，以最小开销定位热点。只有当采样结果指向某个具体核函数或阶段时，才切换到完整追踪。

### Inference Troubleshooting Recipes

### 推理排障方案

In production environments, it’s impractical to run heavy profilers continuously. As such, you need to rely on lightweight, metric-based monitoring such as GPU SM utilization, KV cache warnings, tail-latency percentiles, cache-hit rates, and OOM alerts to detect anomalies and guide targeted fixes. When a metric crosses a specific threshold, you can form a hypothesis about its root cause, such as a small batch size, insufficient KV cache, routing hotspot, unbalanced sharding, memory overcommitment, etc.

在生产环境中，持续运行重量级剖析器并不现实。因此，你需要依赖轻量的、基于指标的监控，如 GPU SM 利用率、KV 缓存告警、尾延迟分位数、缓存命中率和 OOM 告警，来检测异常并指导有针对性的修复。当某个指标越过特定阈值时，你可以针对其根因提出假设，例如批大小过小、KV 缓存不足、路由热点、分片不均衡、内存超额分配等。

Then you apply a fix such as tuning batch sizes, raising memory-utilization limits (if possible), adjusting router thresholds, or enabling CPU offload. Once the fix is pushed, you should verify the impact and confirm that the metrics have settled below their thresholds. Table 16-1 shows some key metrics, symptoms, probable causes, and recommended fixes for common production issues.

然后你施加修复，例如调整批大小、提高内存利用率上限（如果可行）、调整路由器阈值，或启用 CPU 卸载。修复推送后，你应当验证其影响，并确认指标已回落到阈值以下。表 16-1 列出了一些常见生产问题的关键指标、症状、可能原因和推荐处置。

Table 16-1. Common troubleshooting symptoms, causes, and recommended actions

表 16-1. 常见排障症状、原因与推荐处置

| Metric/symptom | Probable cause | Recommended action |
| --- | --- | --- |
| SM utilization < 50% | Small batches or lack of fused kernels | Increase batch size, enable fused kernels (FlashAttention or the cuDNN-fused scaled_dot_product_attention (SDPA) backend in PyTorch), or add custom fused kernels (e.g., using Triton); then profile with nsys --trace=cuda. |
| KV cache preemption warnings | Insufficient KV cache space (vLLM) | Increase GPU memory utilization threshold, reduce max number of batched tokens, consider PagedAttention for dynamic KV allocation. |
| High tail latency (p95 > 200 ms) | Decode-node hotspot or head-of-line blocking | Inspect router logs for routing patterns. Tune prefetch threshold. Enable speculative decoding paths. |
| Cache-hit rate < 60% under load | Unbalanced shard placement or missing prefix cache | Validate the prefix caching connector configuration (e.g., LMCache's NIXL for vLLM and NVIDIA Dynamo's NIXL connector), and increase prefix-cache TTL or replica count if needed. |
| Unexpected OOM on multitenant GPU | Overcommitted GPU memory | Lower per-instance GPU memory utilization, enable CPU/NVMe offload, pin processes to CPU sockets to reduce cross-socket traffic. |
| Irregular performance outliers | Mismatched clocks or thermal throttling | Make sure all clocks are synchronized, and monitor for thermal and power throttling. |

| 指标/症状 | 可能原因 | 推荐处置 |
| --- | --- | --- |
| SM utilization < 50% | 批过小或缺少融合核函数 | 增大批大小，启用融合核函数（FlashAttention 或 PyTorch 中经 cuDNN 融合的 scaled_dot_product_attention (SDPA) 后端），或添加自定义融合核函数（例如用 Triton）；然后用 nsys --trace=cuda 剖析。 |
| KV 缓存抢占（preemption）告警 | KV 缓存空间不足（vLLM） | 提高 GPU 内存利用率阈值，减小批处理 token 的最大数量，考虑用 PagedAttention 做动态 KV 分配。 |
| 高尾延迟（p95 > 200 ms） | decode 节点热点或队头阻塞（head-of-line blocking） | 检查路由器日志中的路由模式。调整预取阈值。启用推测解码（speculative decoding）路径。 |
| 负载下缓存命中率 < 60% | 分片放置不均衡或缺少前缀缓存 | 校验前缀缓存连接器配置（例如用于 vLLM 的 LMCache 的 NIXL，以及 NVIDIA Dynamo 的 NIXL 连接器），并按需增大前缀缓存 TTL 或副本数。 |
| 多租户 GPU 上意外 OOM | GPU 内存超额分配 | 降低每实例 GPU 内存利用率，启用 CPU/NVMe 卸载，把进程绑定到 CPU 插槽以减少跨插槽流量。 |
| 不规则的性能离群点 | 时钟不一致或热降频 | 确保所有时钟同步，并监控热与功耗降频。 |

> Note: The numeric values in all metrics tables are illustrative to explain the concepts. For actual benchmark results on different GPU architectures, see the GitHub repository.

> 注：所有指标表格中的数值均为示意，用于阐释概念。不同 GPU 架构上的实际基准结果，请参见 GitHub 仓库。

You may also find some of this information buried in logfiles. Cloud providers like AWS support regular expression (RegEx) filters on logfiles to extract numeric values from loglines and export them directly as metrics. AWS CloudWatch, for instance, supports this useful feature. There are example log lines in the next code block that are useful to monitor.

你也可能会在日志文件里发现一些此类信息。AWS 等云厂商支持对日志文件使用正则表达式（regular expression，RegEx）过滤器，从日志行中提取数值并直接导出为指标。例如，AWS CloudWatch 就支持这一实用功能。下一个代码块中给出了一些便于监控的示例日志行。

Here is a sample vLLM log snippet that indicates KV cache preemption due to not enough KV cache space. As such, a KV recomputation was triggered, which uses more GPU compute resources and increases latency:

下面是一段 vLLM 日志示例，表明由于 KV 缓存空间不足而发生了 KV 缓存抢占。因此触发了一次 KV 重算，这会占用更多 GPU 计算资源并增加延迟：

```
WARNING 2025-05-03 14:22:07 scheduler.py:1057 Sequence group 0 is preempted by
PreemptionMode.RECOMPUTE because not enough KV cache space.
total_cumulative_preemption_cnt=1
```

```
WARNING 2025-05-03 14:22:07 scheduler.py:1057 Sequence group 0 is preempted by
PreemptionMode.RECOMPUTE because not enough KV cache space.
total_cumulative_preemption_cnt=1
```

Next is a sample NVIDIA Dynamo routing log. Here, the first line shows a 90% prefix-cache hit on the local decode worker, which kept the prefill running locally. The next line shows a local cache miss. The router then dispatches the prefill to the remote GPU-node-03 worker node:

接下来是一段 NVIDIA Dynamo 路由日志示例。这里，第一行显示本地 decode 工作节点上 90% 的前缀缓存命中，因而 prefill 得以在本地继续运行。下一行显示了一次本地缓存未命中。路由器随后把 prefill 分派到远端的 GPU-node-03 工作节点：

```
[Router] 2025-05-03T14:23:11Z INFO KVRouter: prefix-cache hit (90%) for
model=DeepSeek-R1; routing to local vLLM worker
[Router] 2025-05-03T14:23:12Z INFO KVRouter: cache miss; dispatching remote
prefill to GPU-node-03
```

```
[Router] 2025-05-03T14:23:11Z INFO KVRouter: prefix-cache hit (90%) for
model=DeepSeek-R1; routing to local vLLM worker
[Router] 2025-05-03T14:23:12Z INFO KVRouter: cache miss; dispatching remote
prefill to GPU-node-03
```

### Full-Stack Inference Optimizations

### 全栈推理优化

High-performance LLM inference demands coordinated optimizations across every layer of the stack. This includes everything from model architecture and kernel implementations to runtime engines, system orchestration, and deployment infrastructure.

高性能 LLM 推理要求在技术栈的每一层做协调一致的优化。这涵盖从模型架构、核函数实现，到运行时引擎、系统编排和部署基础设施的方方面面。

Model-level techniques like pruning, distillation, sparsity, MoE routing, efficient attention (e.g., FlashAttention), and quantization-aware training can reduce compute and memory requirements. At the kernel level, fused operations, custom attention engines (e.g., FlashInfer), Tensor Core utilization, block tiling, and asynchronous memory transfers will help maximize GPU throughput.

模型级技术，如剪枝（pruning）、蒸馏（distillation）、稀疏性、MoE 路由、高效注意力（例如 FlashAttention）和量化感知训练（quantization-aware training，QAT），可以降低计算与内存需求。在核函数级，融合操作、自定义注意力引擎（例如 FlashInfer）、Tensor Core 利用、分块平铺（block tiling）和异步内存传输，都有助于最大化 GPU 吞吐量。

Runtime strategies like dynamic batching, paged KV caches, CUDA Graphs, and overlap of computation and communication will keep GPUs saturated under variable loads. System orchestration layers can use prefill/decode disaggregation, intelligent routing, multitenancy isolation, and autoscaling (with warm spares) to balance latency and improve cost efficiency.

运行时策略，如动态批处理（dynamic batching）、分页 KV 缓存、CUDA Graphs，以及计算与通信重叠，能在变化的负载下让 GPU 保持饱和。系统编排层可以使用 prefill/decode 分离、智能路由、多租户隔离和自动扩缩（配合热备），来平衡延迟并提升成本效率。

> Many production systems use Kubernetes-based orchestration to run separate prefill versus decode deployments. They use ingress controllers to route requests based on load or user priority. And they keep warm standby GPU pods ready to spin up when traffic spikes.

> 许多生产系统使用基于 Kubernetes 的编排，分别运行 prefill 与 decode 两套部署。它们使用入口控制器（ingress controller）根据负载或用户优先级路由请求。并且它们会准备好热备 GPU pod，以便在流量激增时随时拉起。

Finally, you should explore deployment patterns like geo-distributed edge serving, smart API gateway batching, CI/CD for model variants, and real-time profiling. This will provide peak reliability and adaptability in production. Table 16-2 describes some common optimization approaches for each layer in the stack.

最后，你应当探索各类部署模式，如地理分布式边缘服务、智能 API 网关批处理、面向模型变体的 CI/CD，以及实时剖析。这将在生产中提供顶级的可靠性与适应性。表 16-2 描述了技术栈各层的一些常见优化方法。

Table 16-2. Common optimization approaches for each layer in the stack

表 16-2. 技术栈各层的常见优化方法

| Stack layer | Key techniques |
| --- | --- |
| Model | Pruning and knowledge distillation to shrink model size with minimal accuracy loss; Sparsity (MoE) to skip computations; Efficient attention (FlashAttention) to reduce memory footprint and intermediate buffers; Quantization-aware training in FP16/BF16 or INT4/FP8 for robustness at low precision |
| Kernel | Fused operator kernels (e.g., Linear + GELU + LayerNorm) to reduce launch overhead and memory traffic; Custom attention kernels (FlashInfer) for block-sparse KV- and JIT-compiled kernels; Tensor Core and specialized instruction utilization (cp.async, TMA) for matrix ops |
| Runtime | Dynamic batching with latency controls as implemented in vLLM, SGLang, and NVIDIA Dynamo (e.g., continuous batching) to consolidate requests; Paged KV cache management to flexibly allocate memory and merge batches (vLLM's PagedAttention); CUDA Graphs and buffer pooling to reduce per-inference overhead; Use multiple CUDA streams (one stream for data transfer and another stream for compute) to overlap computation and communication. Use event-based synchronization—and only when needed. |
| System orchestration | Prefill–decode disaggregation for head-of-line blocking elimination; Intelligent routing and cache affinity to balance load and cache hits; Multitenancy isolation and per-user quotas to prevent noisy neighbors; Autoscaling with warm spare instances to hide model load times and accept a slight cost increase for significantly better latency during traffic spikes |
| Deployment and infrastructure | Geo-distributed and edge deployment to reduce network RTT; Smart API gateways with request-level batching across server pools; CI/CD pipelines for rolling out new quantized or kernel-optimized model variants in canary mode; High-bandwidth interconnects (NVLink/NVSwitch and InfiniBand) and NUMA affinity between GPUs and CPUs for optimal memory access |
| QoS and scaling | SLA-aware dynamic batching and tail-latency controls; GPU isolation with MIG or stream priorities to enforce QoS; Real-time profiling dashboards for TTFT, TPOT, utilization, and memory bandwidth utilization; Dynamic parallelism switching (TP, PP, DP) based on workload characteristics |

| 技术栈层 | 关键技术 |
| --- | --- |
| 模型 | 剪枝与知识蒸馏（knowledge distillation），在精度损失最小的前提下缩小模型体积；稀疏性（MoE）以跳过部分计算；高效注意力（FlashAttention）以降低内存占用和中间缓冲；在 FP16/BF16 或 INT4/FP8 下做量化感知训练以增强低精度下的鲁棒性 |
| 核函数 | 融合算子核函数（例如 Linear + GELU + LayerNorm）以减少启动开销和内存流量；自定义注意力核函数（FlashInfer）用于块稀疏 KV 和 JIT 编译的核函数；Tensor Core 及专用指令（cp.async、TMA）用于矩阵运算 |
| 运行时 | 带延迟控制的动态批处理，如 vLLM、SGLang 和 NVIDIA Dynamo 中实现的那样（例如连续批处理，continuous batching），以合并请求；分页 KV 缓存管理，用于灵活分配内存并合并批（vLLM 的 PagedAttention）；CUDA Graphs 和缓冲池化（buffer pooling）以减少每次推理的开销；使用多条 CUDA 流（一条流做数据传输、另一条做计算）来重叠计算与通信。使用基于事件的同步——且仅在需要时使用。 |
| 系统编排 | prefill–decode 分离以消除队头阻塞；智能路由与缓存亲和性以均衡负载和缓存命中；多租户隔离与按用户配额以防止“吵闹邻居”；带热备实例的自动扩缩以掩盖模型加载时间，并接受略微增加的成本，以换取流量激增期间显著更优的延迟 |
| 部署与基础设施 | 地理分布式与边缘部署以降低网络往返时延（RTT）；带请求级批处理、可跨服务器池合批的智能 API 网关；用于以金丝雀模式滚动发布新的量化或核函数优化模型变体的 CI/CD 流水线；高带宽互连（NVLink/NVSwitch 和 InfiniBand），以及 GPU 与 CPU 之间的 NUMA 亲和性以实现最优内存访问 |
| QoS 与扩缩 | 感知 SLA 的动态批处理与尾延迟控制；用 MIG 或流优先级做 GPU 隔离以强制保障 QoS；面向 TTFT、TPOT、利用率和内存带宽利用率的实时剖析仪表盘；根据负载特征动态切换并行方式（TP、PP、DP） |

When optimizing, it’s important to consider cross-layer synergies. For instance, quantization (model) reduces memory footprint, enabling larger batch sizes (runtime) without OOM errors, which in turn allows orchestration components to merge more requests per GPU cycle.

在做优化时，考虑跨层协同效应很重要。例如，量化（模型层）降低了内存占用，从而无需触发 OOM 错误就能使用更大的批大小（运行时层），这反过来又让编排组件能在每个 GPU 周期合并更多请求。

You should also have a profiling-driven focus. Continuous profiling should guide which layer to optimize next. For instance, after fusion and quantization, if the CPU becomes the bottleneck for preprocessing and postprocessing, invest in faster tokenizers or offload some tasks to the GPU.

你还应当以剖析为驱动。持续的剖析应当指导下一步优化哪一层。例如，在完成融合与量化之后，如果 CPU 成了预处理和后处理的瓶颈，就投入更快的分词器，或把部分任务卸载到 GPU。

There are always trade-offs to consider when applying optimizations. Techniques like layer-level CPU offload and advanced decoding methods add complexity. For example, speculative decoding adds a draft model, and Medusa adds multihead parallel decoding. These are typically reserved for extreme cases such as ultralong contexts or erratic latency variances. Lighter-weight methods, including sparsity, batching, and disaggregation, deliver the bulk of benefits in production.

应用优化时，总有权衡需要考量。逐层 CPU 卸载和高级解码方法等技术会增加复杂度。例如，推测解码会额外引入一个草稿模型，而 Medusa 会额外引入多头并行解码。这些通常保留给极端情形，如超长上下文或延迟波动无常的场景。更轻量的方法，包括稀疏性、批处理和分离，才在生产中贡献了绝大部分收益。

It’s recommended to adopt a full-stack optimization approach by aligning model architecture, kernel design, runtime behavior, system orchestration, and deployment strategies. This means keeping your software stack up to date, including CUDA, cuDNN, NCCL, etc. Newer versions often include the latest optimizations and bug fixes.

建议采用全栈优化（full-stack optimization）方法，把模型架构、核函数设计、运行时行为、系统编排和部署策略对齐协调。这意味着让你的软件栈保持更新，包括 CUDA、cuDNN、NCCL 等。较新的版本往往包含最新的优化和缺陷修复。

A full-stack approach reduces the likelihood that each stack layer becomes a bottleneck. This way, teams can systematically eliminate bottlenecks, achieve consistent low latency, and maximize hardware utilization for large-scale LLM inference.

全栈方法降低了技术栈每一层成为瓶颈的可能性。这样，团队就能系统性地消除瓶颈、实现持续的低延迟，并为大规模 LLM 推理最大化硬件利用率。

### Debugging Correctness Issues

### 调试正确性问题

Monitoring can also help to catch anomalies caused by bugs. For instance, if memory usage keeps climbing over time, it may be caused by a memory leak in your CUDA kernel. It’s recommended to use Compute Sanitizer (compute-sanitizer) during testing to catch device memory errors, race conditions, and out-of-bounds accesses. An example is shown here:

监控还有助于捕捉由 bug 引发的异常。例如，如果内存使用量随时间持续攀升，可能是 CUDA 核函数中存在内存泄漏。建议在测试期间使用 Compute Sanitizer（compute-sanitizer）来捕捉设备内存错误、竞态条件和越界访问。示例如下：

```
compute-sanitizer --tool memcheck your_binary
```

```
compute-sanitizer --tool memcheck your_binary
```

If one GPU shows much lower utilization than others, it might have dropped out of the NCCL communication group due to a silent NCCL failure or uncaught error. You can check for NCCL error codes in logs by looking for WARN NCCL_COMM_FAILURE. It provides very verbose error logs.

如果某个 GPU 的利用率明显低于其他 GPU，它可能因为静默的 NCCL 故障或未捕获的错误而退出了 NCCL 通信组。你可以在日志中查找 WARN NCCL_COMM_FAILURE 来检查 NCCL 错误码。它会提供非常详尽的错误日志。

> Enable NCCL debugging by setting the environment variable, NCCL_DEBUG=WARN. This will help surface errors that would otherwise be silent. Be warned, however, that NCCL logs are very verbose!

> 通过设置环境变量 NCCL_DEBUG=WARN 来启用 NCCL 调试。这有助于暴露那些原本会被静默处理的错误。但要注意，NCCL 日志非常冗长！

Use the NCCL test suite to debug all-reduce and all-to-all performance and correctness issues. You can also use ncclCommGetAsyncError with ncclCommAbort to detect and handle asynchronous communication errors. Consider enabling NCCL’s IB GID tracing and using NVSwitch system telemetry to detect issues at the interconnect level as well.

使用 NCCL 测试套件来调试 all-reduce 和 all-to-all 的性能与正确性问题。你还可以用 ncclCommGetAsyncError 配合 ncclCommAbort 来检测和处理异步通信错误。此外，可考虑启用 NCCL 的 IB GID 追踪，并利用 NVSwitch 系统遥测在互连层面检测问题。

You should set up alerts in Prometheus’s alert manager to detect unusual patterns like “GPU utilization < 10% for at least 60 seconds,” “memory usage above W% threshold,” “NVLink error rate > X,” “PCIe replays above Y threshold,” “temperature above Z degrees,” etc.

你应该在 Prometheus 的告警管理器（alert manager）中设置告警，以检测异常模式，例如“GPU 利用率 < 10% 且持续至少 60 秒”“内存使用量超过 W% 阈值”“NVLink 错误率 > X”“PCIe 重放超过 Y 阈值”“温度超过 Z 度”等。

In practice, you might configure Prometheus Alertmanager rules, as shown in Table 16-3. This way, you can proactively investigate issues.

在实践中，你可以配置 Prometheus Alertmanager 规则，如表 16-3 所示。这样你就能主动排查问题。

Table 16-3. Example set of common Prometheus alerts for GPU-based systems

表 16-3. 面向 GPU 系统的一组常见 Prometheus 告警示例

| Metric | Condition | Severity | Notes |
| --- | --- | --- | --- |
| GPU utilization | < 10% for > 60 s | Idle | Underutilization |
| GPU utilization | > 90% | Bottleneck | Possible saturation |
| Memory usage | > 80% | Warning | Approaching OOM |
| Memory usage | > 95% | Critical | High risk of out-of-memory errors |
| Temperature | > 85 °C | Warning | Approaching thermal throttling |
| Temperature | > 95 °C | Critical | Risk of shutdown or hardware damage |
| NVLink replay/recovery errors | ≥ 1 | Critical | Indicates link retry or recovery |
| NVLink CRC errors | > 100 errors/sec | Critical | High CRC failure rate on link |
| PCIe replay errors | ≥ 1 | Critical | Packet retries on PCIe bus |
| Uncorrectable ECC errors | ≥ 1 | Critical | Data corruption requiring reset |

| 指标 | 条件 | 严重级别 | 备注 |
| --- | --- | --- | --- |
| GPU 利用率 | < 10% 且持续 > 60 s | 空闲 | 利用不足 |
| GPU 利用率 | > 90% | 瓶颈 | 可能已饱和 |
| 内存使用率 | > 80% | 警告 | 接近 OOM |
| 内存使用率 | > 95% | 严重 | 内存溢出错误的高风险 |
| 温度 | > 85 °C | 警告 | 接近热降频 |
| 温度 | > 95 °C | 严重 | 有关机或硬件损坏风险 |
| NVLink 重放/恢复错误 | ≥ 1 | 严重 | 表明链路重试或恢复 |
| NVLink CRC 错误 | > 100 errors/sec | 严重 | 链路上的 CRC 失败率高 |
| PCIe 重放错误 | ≥ 1 | 严重 | PCIe 总线上的数据包重试 |
| 不可纠正的 ECC 错误 | ≥ 1 | 严重 | 需要重置的数据损坏 |

You also want to set up hardware error counters and alerts as well. For example, if ECC errors or NVLink retries are reported, alert immediately, as these can quickly degrade performance or cause drop-outs. Dropouts happen when a GPU—or its interconnect—silently disconnects. For instance, an NVLink might drop—or the GPU might “drop off the bus” after a fatal error.

你还应该设置硬件错误计数器和告警。例如，如果报告了 ECC 错误或 NVLink 重试，应立即告警，因为它们会迅速拖垮性能或导致掉线（drop-out）。当某个 GPU 或其互连静默断开时就会发生掉线。例如，某条 NVLink 可能掉线，或者 GPU 在发生致命错误后“从总线上掉落（drop off the bus）”。

> Use DCGM for per-link NVLink throughput and totals including DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L*, DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L*, and *_TOTAL. Fall back to nvidia-smi nvlink, Nsight Systems/Compute, or NVSwitch counters if needed.

> 使用 DCGM 获取每条链路的 NVLink 吞吐量及总量，包括 DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L*、DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L* 和 *_TOTAL。必要时可退回使用 nvidia-smi nvlink、Nsight Systems/Compute 或 NVSwitch 计数器。

Consider a NCCL failure case. You might get an alert that shows one node’s GPUs are near 0% utilization, while others are at 90%. You can start debugging the issue by checking the node logs and finding which node is generating the NCCL errors.

设想一个 NCCL 故障场景。你可能会收到告警，显示某个节点的 GPU 利用率接近 0%，而其他节点为 90%。你可以先检查节点日志、找出是哪个节点在产生 NCCL 错误，从而开始排查问题。

This type of active monitoring and alerting lets you catch these issues more quickly, find the failed node, and start restoring it back to normal. In this case, you might want to reinitialize the NCCL communicators or perform a full node reboot (just make sure the node rejoins the NCCL group after restart/reboot).

这种主动的监控与告警能让你更快地发现这类问题、定位故障节点，并着手将其恢复正常。在这种情况下，你可能需要重新初始化 NCCL 通信器，或对节点执行一次完整重启（只要确保节点在重启后重新加入 NCCL 组即可）。

You could catch these issues even quicker by incrementing a “NCCL Error” counter for NCCL errors. In addition, your inference server can log the NCCL errors, which will automatically be scraped by Fluentd or AWS CloudWatch and convert them into counters.

通过为 NCCL 错误递增一个“NCCL Error”计数器，你可以更快地发现这些问题。此外，你的推理服务器可以记录 NCCL 错误日志，这些日志会被 Fluentd 或 AWS CloudWatch 自动抓取并转换为计数器。

Then you can overlay the error counter chart on top of the per-node GPU utilization chart in Grafana. This will correlate the NCCL errors with a drop in GPU utilization so that you can identify and remediate the failed node much quicker.

然后你可以在 Grafana 中把错误计数器图表叠加到各节点的 GPU 利用率图表之上。这样就能把 NCCL 错误与 GPU 利用率的下降关联起来，从而更快地识别并修复故障节点。

> Application-level counters are extremely useful in production—especially when combined with system metrics.

> 应用级计数器在生产环境中极为有用——尤其是与系统指标结合使用时。

And optimizations should not be considered successful until they are verified with actual metrics that demonstrate increased throughput, reduced latency, improved utilization, etc. A rigorous measurement-driven approach to system performance tuning is essential given the complexity of modern AI inference systems.

而且，在用能证明吞吐量提升、延迟降低、利用率改善等的实际指标验证之前，不应认为优化已经成功。鉴于现代 AI 推理系统的复杂性，采用严谨的、以测量为驱动的系统性能调优方法至关重要。

In short, you should combine application-level counters, automatically scraped log counters from Fluentd or AWS CloudWatch, and low-level system metrics. This type of full-stack telemetry provides the visibility needed to operate—and optimize—a multinode LLM inference system running at peak performance on production workloads. You should treat metrics, counters, and logs as the ground truth of your system’s behavior.

简而言之，你应该把应用级计数器、由 Fluentd 或 AWS CloudWatch 自动抓取的日志计数器，以及底层系统指标结合起来。这种全栈遥测提供了运维——以及优化——一个在生产负载下以峰值性能运行的多节点 LLM 推理系统所需的可见性。你应该把指标、计数器和日志当作系统行为的真实依据（ground truth）。

> Our intuitions and gut feelings can often lead us down the wrong debugging path. But metrics don’t lie. Instrument your code properly upfront—and trust the metrics when things go wrong.

> 我们的直觉和第六感常常会把我们引向错误的调试方向。但指标不会说谎。请提前为代码做好埋点——并在出问题时相信指标。

## Dynamic Batching, Scheduling, and Routing

## 动态批处理、调度与路由

Even after the model has been partitioned and parallelized optimally across the cluster, there are still more opportunities for application-level optimizations in a multinode inference cluster deployment. In this section, we focus on techniques to maximize GPU utilization, minimize latency, and boost throughput by dynamically batching requests and using optimized scheduling and routing strategies.

即使模型已经在集群中被最优地切分与并行化，在多节点推理集群部署中仍有更多应用级优化的机会。在本节中，我们聚焦于通过动态批处理请求以及使用优化的调度与路由策略，来最大化 GPU 利用率、最小化延迟并提升吞吐量的技术。

### Dynamic Batching

### 动态批处理

One of the most powerful performance techniques in an inference serving system is batching. By combining multiple incoming inputs into a single batch for the model to process together, batching improves throughput by amortizing fixed costs—like kernel launches and memory loads—over multiple inputs. It does this at the expense of individual-request latency, however. Some individual requests may need to wait for a period of time (e.g., 2 ms) to join a batch and be processed.

推理服务系统中最强大的性能技术之一就是批处理（batching）。通过把多个到来的输入合并为一个批（batch）交给模型一起处理，批处理把核函数启动、内存加载等固定开销摊薄到多个输入上，从而提升吞吐量。不过，这是以单个请求的延迟为代价的。某些单个请求可能需要等待一段时间（例如 2 ms）才能加入一个批并被处理。

Dynamic batching is a specialization of request batching that assembles batches of incoming inference requests of dynamic sizes on the fly. It can be configured to buffer requests for a period of time or until a given batch size is reached. All modern LLM inference engines support dynamic batching, including vLLM, SGLang, and NVIDIA Dynamo.

动态批处理是请求批处理的一种特化形式，它在运行时即时地把到来的推理请求组装成大小可变的批。可以将它配置为把请求缓冲一段时间，或缓冲到达到给定批大小为止。所有现代 LLM 推理引擎都支持动态批处理，包括 vLLM、SGLang 和 NVIDIA Dynamo。

Dynamic batching is in contrast to *static batching*, which locks in a fixed batch size (or pads all sequences to the longest one) and then waits for every request in that batch to finish before returning results. This can incur unbounded queuing delay for early arrivals—and leave GPU cycles wasted on padding.

动态批处理与*静态批处理（static batching）*形成对比：静态批处理锁定一个固定的批大小（或把所有序列填充到最长的那一条），然后等待该批中的每个请求都完成后才返回结果。这会给早到的请求带来无上界的排队延迟，并让 GPU 周期浪费在填充上。

With dynamic batching, the system accumulates incoming requests and dispatches “whatever has arrived” once either a target batch size is met or a short timeout (e.g., 2 ms) elapses. This bounds maximum latency to the timeout value that you specify.

使用动态批处理时，系统会累积到来的请求，一旦达到目标批大小或经过一个短暂的超时（例如 2 ms），就把“已经到达的请求”派发出去。这把最大延迟限定在你所指定的超时值以内。

With its on-the-fly sizing, dynamic batching lets you amortize kernel-launch overheads across multiple sequences—while avoiding the worst-case delays of static batching. This improves both GPU utilization and predictable latency under variable load. Figure 16-5 shows the difference between static batching and dynamic batching.

凭借即时确定批大小的能力，动态批处理让你把核函数启动开销摊薄到多个序列上，同时避免静态批处理的最坏情况延迟。这在负载可变的情况下既改善了 GPU 利用率，又带来了可预测的延迟。图 16-5 展示了静态批处理与动态批处理之间的区别。

![Figure 16-5. Difference between static and dynamic batching](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-5.png)

![图 16-5. 静态批处理与动态批处理的区别](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-5.png)

Dynamic batching lets the system automatically grow or shrink batch sizes based on actual request-arrival patterns and latency targets (e.g., max delay). The key is to batch intelligently in a way that increases overall throughput while maintaining latency within acceptable bounds. Modern inference engines implement microbatching, which accumulates requests for only a few milliseconds before dispatching the batch to the GPU. Typically, a delay of 2–10 ms is used, but this should be tuned to meet latency SLOs.

动态批处理让系统根据实际的请求到达模式和延迟目标（例如最大延迟）自动增大或缩小批大小。关键在于以一种既能提升整体吞吐量、又能把延迟维持在可接受范围内的方式来智能地批处理。现代推理引擎实现了微批处理（microbatching），它只累积请求几毫秒，就把批派发给 GPU。通常使用 2–10 ms 的延迟，但应针对延迟 SLO 进行调优。

For example, you can configure a batch delay of 2 ms to determine how long the server waits for additional requests before dispatching a batch. If three requests arrive within that interval, the batcher immediately groups and sends them to the GPU. Any further requests that arrive after the 2 ms timer expires will be collected into the next batch. This timeout-driven trigger bounds per-request queuing latency (never exceeding 2 ms) while improving throughput by grouping multiple sequences together.

例如，你可以配置 2 ms 的批延迟，用来决定服务器在派发一个批之前等待更多请求的时长。如果在该时间间隔内到达了三个请求，批处理器会立即把它们分组并发送给 GPU。在 2 ms 定时器到期之后再到达的请求，会被收进下一个批。这种由超时驱动的触发机制把每个请求的排队延迟限定住（绝不超过 2 ms），同时通过把多个序列分到一组来提升吞吐量。

These microbatches reduce the delay added and allow the GPU to work on multiple requests at once—instead of one at a time. Large LLM models are memory-bandwidth-bound at small batch sizes, so increasing the batch size will improve arithmetic intensity—and overall hardware utilization.

这些微批减少了额外增加的延迟，并让 GPU 一次处理多个请求，而不是一次一个。大型 LLM 模型在小批大小时受内存带宽约束（memory-bandwidth-bound），因此增大批大小会提升算术强度（arithmetic intensity）以及整体硬件利用率。

In practice, LLM serving systems pick a balanced batch size that gives good throughput without incurring much latency. Under high load, dynamic batching can improve both throughput and latency because it prevents the GPU from sitting idle between requests. In fact, if the arrival rate of requests is high, batching can reduce overall queue wait times since more work is processed in the same amount of time. This benefits end-to-end latency and tail latency for high-load traffic patterns.

在实践中，LLM 服务系统会选择一个折中的批大小，既能获得良好的吞吐量，又不会引入太多延迟。在高负载下，动态批处理可以同时改善吞吐量和延迟，因为它避免了 GPU 在请求之间闲置。事实上，如果请求的到达率很高，批处理可以缩短整体的队列等待时间，因为在相同的时间内处理了更多的工作。对于高负载流量模式，这有利于端到端延迟和尾延迟。

Dynamic batching will, at low RPS (requests per second), add a small delay as it waits for other requests to join the batch. This will slightly increase latency at low RPS. However, at moderate to high RPS, the delay is amortized and becomes negligible compared to the queuing delay that would occur if we ran everything one by one. As such, batching lowers overall latency, including tail latency.

在低 RPS 时，动态批处理会因为等待其他请求加入批而增加一点小延迟。这会在低 RPS 时略微增加延迟。然而，在中到高 RPS 时，这点延迟会被摊薄，相比逐个处理所有请求时会出现的排队延迟，它变得可以忽略不计。因此，批处理会降低整体延迟，包括尾延迟。

> You should validate the improvement by plotting latency percentiles against load using tools like Grafana. With batching, you’ll often see overall p50 latency stay flat—or even drop—as throughput increases. This will happen up until an inflection point. Make note of this inflection point and stay under this value.

> 你应该使用 Grafana 等工具，将延迟百分位相对负载作图来验证这一改进。使用批处理时，你通常会看到整体 p50 延迟随着吞吐量的提升而保持平稳——甚至下降。这种情况会持续到某个拐点为止。请记下这个拐点，并保持在该值以下。

It is important to configure batching with respect to latency SLOs. For instance, if you promise a p99 latency of 2 seconds for a request of a certain length, you can’t afford to delay one request by 500 ms waiting in a batch queue. By default, your dynamic batch delay should be initially set well below the p99 latency requirement (on the order of 1–2 ms) to avoid excessive batch delays.

针对延迟 SLO 来配置批处理很重要。例如，如果你承诺对某个特定长度的请求提供 2 秒的 p99 延迟，那你就不能让某个请求在批队列中等待而被延迟 500 ms。默认情况下，你的动态批延迟一开始应设置得远低于 p99 延迟要求（在 1–2 ms 量级），以避免过度的批延迟。

Using an adaptive batching delay, the batch delay value can dynamically drop to near 0 ms at low RPS and increase higher to 5–10 ms at peak load when needed. This adaptive approach is used by vLLM and others to maintain SLO compliance across different traffic patterns.

使用自适应批延迟时，批延迟值可以在低 RPS 时动态降到接近 0 ms，并在峰值负载时按需升高到 5–10 ms。vLLM 等系统采用这种自适应方法，以在不同的流量模式下都保持对 SLO 的遵守。

Batching primarily benefits high-traffic scenarios. If traffic is low, the system will just run single requests, and the latency will be very low. At high load, however, the system can apply aggressive batching to achieve high throughput and will provide better overall latency since it avoids queue buildup and amortizes the latency across many requests.

批处理主要在高流量场景中获益。如果流量很低，系统只会逐个处理单个请求，延迟会非常低。然而在高负载下，系统可以采取激进的批处理来实现高吞吐量，并且由于避免了队列堆积、把延迟摊薄到许多请求上，会提供更好的整体延迟。

### Continuous Batching

### 连续批处理

Continuous batching, also known as *in-flight batching* or *iteration-level scheduling*, maintains high GPU utilization by refilling batches on every token-generation iteration rather than waiting for complete sequences to finish. It evicts completed requests and immediately pulls in new ones based entirely on GPU readiness. This technique is particularly important for low-latency use cases such as chat assistants.

连续批处理，也称为*运行中批处理（in-flight batching）*或*迭代级调度（iteration-level scheduling）*，通过在每次 token 生成迭代时回填批来维持高 GPU 利用率，而不是等待完整的序列全部完成。它逐出已完成的请求，并完全根据 GPU 的就绪情况立即拉入新请求。对于聊天助手等低延迟用例，这项技术尤为重要。

In contrast to timeout-driven approaches like dynamic batching, the event-driven continuous batching strategy eliminates idle compute slots and the padding overhead of these other approaches. By never relying on a fixed “max-delay” timer, continuous batching allows new requests to join an ongoing batch mid-generation—and without blocking on the longest sequence, as shown in Figure 16-6.

与动态批处理这类由超时驱动的方法相比，由事件驱动的连续批处理策略消除了空闲的计算槽位以及这些其他方法的填充开销。由于从不依赖固定的“最大延迟”定时器，连续批处理允许新请求在生成过程中加入一个正在进行的批——并且不会因最长序列而阻塞，如图 16-6 所示。

![Figure 16-6. Continuous batching allows new sequences (requests) to join a batch mid-generation](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-6.png)

![图 16-6. 连续批处理允许新序列（请求）在生成过程中加入批处理](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-6.png)

Here, continuous batching minimizes wasted slots in the inference compute pipeline. It eliminates the idle time caused by waiting for the longest response to finish for each batch. Instead of waiting for all sequences in a batch to finish (which is inefficient due to variable output lengths and leads to GPU underutilization), continuous batching replaces completed sequences with new ones at each iteration. This approach allows new requests to fill GPU slots immediately—resulting in higher throughput, reduced latency, and more efficient GPU utilization.

在这里，连续批处理把推理计算流水线中被浪费的槽位降到最少。它消除了因每个批都要等待最长响应完成而造成的空闲时间。连续批处理不会等待一个批中的所有序列都完成（由于输出长度可变，这样做效率低下并导致 GPU 利用不足），而是在每次迭代时用新序列替换已完成的序列。这种方法让新请求可以立即填满 GPU 槽位，从而带来更高的吞吐量、更低的延迟和更高效的 GPU 利用率。

Batching 4–8 requests can often double or triple the throughput over 1–2 request batches on large models. This is because the GPU’s math units and memory pipelines are better utilized. And the additional latency per request is on the order of only a few milliseconds—making this strategy ideal for low-latency use cases. Table 16-4 summarizes the different batching strategies discussed so far, including a naive static batching strategy.

在大模型上，把 4–8 个请求组成一个批，其吞吐量往往能达到 1–2 个请求批的两到三倍。这是因为 GPU 的计算单元和内存流水线得到了更充分的利用。而每个请求增加的延迟只在几毫秒量级——这使得该策略非常适合低延迟用例。表 16-4 总结了到目前为止讨论过的不同批处理策略，包括一种朴素的静态批处理策略。

Table 16-4. Comparing static, dynamic, and continuous batching

表 16-4. 静态、动态与连续批处理的对比

| Aspect | Static batching | Dynamic batching | Continuous batching |
| --- | --- | --- | --- |
| Trigger | Fixed batch size or sequence length | Batch-size target or timeout | Token-generation completion event |
| Latency bound | Unbounded; wait for full batch | Bounded by max_batch_delay_ms | Minimal; evicts and refills mid-batch |
| Padding overhead | High overhead: pad to longest sequence | Moderate overhead: pad within each batch | Low: slots refilled without waiting for padding |
| GPU idle mitigation | Poor: especially under variable-sized loads | Better: but can cause idle GPU if the timer fires on a small batch | Excellent: keeps GPU saturated at every iteration |
| Implementation | Simple | Moderate: requires timers and queue management | Complex: requires per-token coordination and state tracking |

| 方面 | 静态批处理 | 动态批处理 | 连续批处理 |
| --- | --- | --- | --- |
| 触发条件 | 固定批大小或序列长度 | 批大小目标或超时 | token 生成完成事件 |
| 延迟上界 | 无上界；需等待整个批完成 | 由 max_batch_delay_ms 限定 | 极小；在批处理过程中逐出并回填 |
| 填充开销 | 开销高：填充到最长序列 | 开销中等：在每个批内填充 | 低：回填空位而无需等待填充 |
| GPU 空闲缓解 | 差：尤其在负载大小可变时 | 较好：但若定时器在小批上触发，仍可能导致 GPU 空闲 | 极佳：在每次迭代都让 GPU 保持饱和 |
| 实现 | 简单 | 中等：需要定时器与队列管理 | 复杂：需要逐 token 协调与状态跟踪 |

### Continuous Scheduling

### 连续调度

Another dimension of request scheduling involves concurrent model execution. The vLLM community calls this *continuous scheduling*. The idea is that if the model is small—or if the batch size is limited by latency targets—a single model instance might not fully saturate the GPUs.

请求调度的另一个维度涉及并发的模型执行。vLLM 社区称之为*连续调度（continuous scheduling）*。其思路是：如果模型很小，或者批大小受限于延迟目标，那么单个模型实例可能无法完全占满 GPU。

Continuous scheduling treats the GPU like an OS scheduler treats a CPU. It launches independent small kernels on separate CUDA streams. This lets the hardware warp scheduler interleave warps across tasks without explicit context switching.

连续调度对待 GPU 的方式，就像操作系统调度器对待 CPU 一样。它在各自独立的 CUDA 流上启动互不相关的小核函数。这让硬件 warp 调度器无需显式上下文切换，就能在多个任务之间交错执行 warp。

For example, if our inference engine is not fully saturating the GPUs, it can run multiple model inference streams concurrently on the same GPU. This way, while one instance is waiting on a data transfer or a non-GPU operation, another instance can execute.

例如，如果我们的推理引擎没有完全占满 GPU，它可以在同一个 GPU 上并发运行多个模型推理流。这样，当一个实例在等待数据传输或某个非 GPU 操作时，另一个实例就可以执行。

Specifically, during the decode phase, generating one token at a time can often leave the GPU underutilized—especially if performed on a single sequence of text. This is because the workload is relatively small and requires a large amount of memory movement relative to the amount of compute.

具体来说，在 decode 阶段，一次生成一个 token 往往会让 GPU 利用不足——尤其是在单条文本序列上执行时。这是因为该工作负载相对较小，而且相对于计算量而言需要大量的内存搬运。

To address this inefficiency, the inference engine can interleave multiple decode tasks from across different user requests. For example, if 10 users are decoding different sequences at the same time, you can’t simply batch them into one large matrix multiply. This is because each sequence may be a different length. This would prevent the GPU from performing efficient matrix operations without requiring a large amount of padding.

为了解决这种低效，推理引擎可以把来自不同用户请求的多个 decode 任务交错执行。例如，如果有 10 个用户同时在解码不同的序列，你不能简单地把它们批成一个大的矩阵乘法。这是因为每条序列的长度可能各不相同。若不引入大量填充，这会妨碍 GPU 执行高效的矩阵运算。

Instead, the continuous scheduler can launch 10 small decode kernels for each sequence’s next token in rapid succession on separate CUDA streams. This allows true concurrency across sequences by utilizing the GPU’s fine-grained warp scheduler. As such, the kernels interleave execution across SMs. This prevents idle cycles by approximating a round-robin processing pattern.

相反，连续调度器可以在各自独立的 CUDA 流上，快速接连地为每条序列的下一个 token 启动 10 个小的 decode 核函数。通过利用 GPU 细粒度的 warp 调度器，这实现了跨序列的真正并发。于是，这些核函数在多个 SM 上交错执行。这以近似轮询（round-robin）的处理模式避免了空闲周期。

By enqueuing each decode task on its own stream, the GPU can overlap memory transfers and computation across sequences. This reduces per-token latency and improves overall throughput.

通过把每个 decode 任务排入各自的流，GPU 可以在多条序列之间重叠内存传输与计算。这降低了每个 token 的延迟，并提升了整体吞吐量。

The GPU switches between the kernels at the warp level. And when one decode kernel stalls due to inefficient control flow on the CPU or a network I/O wait, other active kernels can immediately proceed. This keeps the GPUs fully utilized.

GPU 在 warp 层面在这些核函数之间切换。当某个 decode 核函数因 CPU 上低效的控制流或网络 I/O 等待而停顿时，其他活跃的核函数可以立即继续。这让 GPU 保持充分利用。

Continuous scheduling achieves high utilization even at token-level granularity by maintaining a queue of ready-to-run token tasks. The end result is similar to batching. The GPU is always working on token generation, but it doesn’t require combining them into one giant matrix multiply. It just means that we don’t let the GPU sit idle between token computations. This type of token-level granularity scheduling is essential to maximizing throughput across millions of users.

连续调度通过维护一个就绪可运行的 token 任务队列，即使在 token 级粒度上也能实现高利用率。最终效果类似于批处理。GPU 始终在进行 token 生成，但并不需要把它们合并成一个巨大的矩阵乘法。它只是意味着我们不让 GPU 在各次 token 计算之间闲置。这种 token 级粒度的调度对于在数百万用户之间最大化吞吐量至关重要。

At fixed intervals or whenever the GPU frees up, a continuous scheduler gathers all pending next-token requests into a configurable-sized batch and then dispatches the batch as a single GPU call. This helps balance throughput and latency.

在固定的时间间隔，或每当 GPU 空出来时，连续调度器会把所有待处理的下一个 token 请求汇集成一个大小可配置的批，然后作为单次 GPU 调用把该批派发出去。这有助于在吞吐量与延迟之间取得平衡。

> If you are designing a custom scheduler, consider the hybrid approach: use a short timer and a maximum batch size for token scheduling.

> 如果你要设计一个自定义调度器，可以考虑这种混合方法：为 token 调度使用一个短定时器和一个最大批大小。

State-of-the-art systems like vLLM and SGLang merge dynamic batching with continuous scheduling. Their custom continuous scheduler is built around the idea of squeezing out “bubbles” of GPU time.

像 vLLM 和 SGLang 这样的前沿系统，把动态批处理与连续调度融合在一起。它们的自定义连续调度器围绕“挤出”GPU 时间中的“气泡（bubble）”这一思路来构建。

vLLM’s PagedAttention is a specific example of this hybrid approach. PagedAttention breaks the attention key-value (KV) cache into slices, or pages, and dynamically groups the page-specific compute across active sequences. This sustains near-100% GPU utilization under heavy load by interleaving large prefill GEMMs with rapid, small decode kernels.

vLLM 的 PagedAttention 就是这种混合方法的一个具体例子。PagedAttention 把注意力的键值（KV）缓存拆分成切片（即页），并在活跃序列之间动态地把针对各页的计算分组。它通过把大的 prefill GEMM 与快速的小 decode 核函数交错执行，在重负载下维持接近 100% 的 GPU 利用率。

vLLM efficiently uses block-level KV cache structures to track each request’s state separately. This fine-grained bookkeeping provides fast streaming responses and optimal hardware utilization by continuously packing and multiplexing workloads in real time.

vLLM 高效地使用块级的 KV 缓存结构来分别跟踪每个请求的状态。这种细粒度的记账通过实时地持续打包并复用工作负载，提供了快速的流式响应和最优的硬件利用率。

Another example is SGLang’s RadixAttention, which uses tree-based KV cache grouping. This achieves similar near-100% GPU utilization and introduces lazy eviction for unused cache pages. We’ll cover vLLM’s PagedAttention and SGLang’s RadixAttention in more detail in a bit. Both approaches are open source, so you can review their scheduling implementations directly in code.

另一个例子是 SGLang 的 RadixAttention，它使用基于树的 KV 缓存分组。它实现了类似的接近 100% 的 GPU 利用率，并为未使用的缓存页引入了惰性逐出（lazy eviction）。稍后我们会更详细地介绍 vLLM 的 PagedAttention 和 SGLang 的 RadixAttention。这两种方法都是开源的，因此你可以直接在代码中查看它们的调度实现。

### Stall-Free Scheduling (Chunked Prefill)

### 无停顿调度（分块 prefill）

When prompts are extremely long, you can split them into chunks and interleave the prefill and decode work. This is called *stall-free scheduling* or *chunked prefill*.

当提示极长时，你可以把它们切分成块，并把 prefill 与 decode 工作交错进行。这被称为*无停顿调度（stall-free scheduling）*或*分块 prefill（chunked prefill）*。

Consider a 20K token prompt. By splitting the prompt into 5K token chunks, you reduce the maximum stall per iteration and bound the per-iteration work to a fixed number of tokens. This provides predictable per-iteration latency. The cost of chunked prefill for different chunk sizes is shown in Table 16-5.

设想一个 20K token 的提示。通过把该提示切分成 5K token 的块，你减少了每次迭代的最大停顿，并把每次迭代的工作量限定为固定数量的 token。这提供了可预测的每次迭代延迟。不同分块大小下分块 prefill 的开销如表 16-5 所示。

Table 16-5. Cost for a 20 K token prompt using chunked prefill of different chunk sizes

表 16-5. 对一个 20 K token 提示使用不同分块大小的分块 prefill 的开销

| Chunk size | Number of chunks | Attention ops per chunk | Total MLP tokens |
| --- | --- | --- | --- |
| 20K | 1 | 20K × (20K + 1) ÷ 2 = 200M | 1 × 20K = 20K |
| 10K | 2 | 50M + 150M = 200M | 2 × 10K = 20K |
| 5K | 4 | 12.5M + 37.5M + 62.5M + 87.5M = 200M | 4 × 5K = 20K |

| 分块大小 | 分块数量 | 每块注意力运算量 | MLP token 总数 |
| --- | --- | --- | --- |
| 20K | 1 | 20K × (20K + 1) ÷ 2 = 200M | 1 × 20K = 20K |
| 10K | 2 | 50M + 150M = 200M | 2 × 10K = 20K |
| 5K | 4 | 12.5M + 37.5M + 62.5M + 87.5M = 200M | 4 × 5K = 20K |

Here, the per-chunk attention cost grows with the full context due to KV caching. Each new chunk computes attention for its own tokens against all prior tokens from previous chunks.

在这里，由于 KV 缓存，每个块的注意力开销会随着完整上下文的增长而增长。每个新块都要针对来自先前各块的所有先前 token，为它自己的 token 计算注意力。

In self-attention, the number of QK dot products for a sequence of length *N* is triangular, approximately N(N + 1) ÷ 2. This is the number of Q × K dot product operations per layer per head. This cost is quadratic in *N* (O(N2). So for N = 20,000, this is about 200 million operations. Chunking does not reduce this total. Chunking only changes when work becomes available to the decoder. The benefit of chunked prefill is latency smoothing and improved overlap, not fewer total attention dot-product operations.

在自注意力（self-attention）中，长度为 *N* 的序列的 QK 点积数量是三角形的，约为 N(N + 1) ÷ 2。这是每层每个头的 Q × K 点积运算数量。该开销关于 *N* 呈二次增长（O(N2)）。因此当 N = 20,000 时，这约为 2 亿次运算。分块并不会减少这个总量。分块只改变工作何时对解码器可用。分块 prefill 的好处在于平滑延迟和改善重叠，而不是减少注意力点积运算的总数。

> Prefill self-attention performs N(N + 1) ÷ 2 Q × K dot products per layer per head. This cost is quadratic in *N*, O(N2). Note that chunking and pipelined prefill change only overlap and latency; they do not change the total amount of attention work. The only ways to reduce the total attention cost are to reduce the effective context or to use local or sparse attention. For example, you can reduce the cost to O(NW) with a fixed attention window *W*.

> Prefill 自注意力每层每个头执行 N(N + 1) ÷ 2 次 Q × K 点积。该开销关于 *N* 是二次的，即 O(N2)。注意，分块和流水线化的 prefill 只改变重叠和延迟；它们并不改变注意力工作的总量。减少注意力总开销的唯一途径是减少有效上下文，或使用局部或稀疏注意力。例如，使用固定的注意力窗口 *W*，你可以把开销降到 O(NW)。

In short, the chunked prefill technique schedules prompt prefill processing and token generation in a tightly interleaved way. It unifies the high efficiency of large-batch prefill compute with the responsiveness of streaming decode. This can lower tail latency while maintaining high throughput on many workloads.

简而言之，分块 prefill 技术以紧密交错的方式来调度提示的 prefill 处理与 token 生成。它把大批 prefill 计算的高效率与流式 decode 的响应性统一起来。在许多工作负载上，这可以在维持高吞吐量的同时降低尾延迟。

### Latency-Aware Scheduling and Dynamic Routing

### 延迟感知调度与动态路由

Latency-aware scheduling can analyze incoming requests, create dynamic batches, and route the batches to minimize latency. Consider six prompts arriving into a two-GPU inference system at the same time. The prompt lengths are 6K, 2K, 6K, 2K, 2K, and 2K in that order. Let’s compare a naive first-in-first-out (FIFO) scheduler with a latency-aware scheduler.

延迟感知调度（latency-aware scheduling）可以分析到来的请求、创建动态批，并对这些批进行路由以最小化延迟。设想有六个提示同时到达一个双 GPU 的推理系统。这些提示的长度依次为 6K、2K、6K、2K、2K 和 2K。让我们把一个朴素的先进先出（first-in-first-out，FIFO）调度器与一个延迟感知调度器进行对比。

First, the FIFO scheduler creates two batches based on arrival order. Batch 1 is [6K, 2K, 6K] and is sent to GPU 1. Batch 2 is [2K, 2K, 2K] and is sent to GPU 2. The results are summarized in Table 16-6.

首先，FIFO 调度器根据到达顺序创建两个批。批 1 是 [6K, 2K, 6K]，发送给 GPU 1。批 2 是 [2K, 2K, 2K]，发送给 GPU 2。结果汇总在表 16-6 中。

Table 16-6. FIFO versus latency-aware scheduling for a sequence of incoming prompts of length [6K, 2K, 6K, 2K, 2K, 2K]

表 16-6. 对长度依次为 [6K, 2K, 6K, 2K, 2K, 2K] 的入站提示序列，FIFO 与延迟感知调度的对比

| GPU | Prompt batch | Self-attention QK ops T(N) = N(N + 1) ÷ 2 | Tokens N |
| --- | --- | --- | --- |
| GPU 1 | 6K, 2K, 6K | T(6K) + T(2K) + T(6K) = 18,003,000 + 2,001,000 + 18,003,000 = 38,007,000 | 14K |
| GPU 2 | 2K, 2K, 2K | 3 × T(2K) = 3 × 2,001,000 = 6,003,000 | 6K |

| GPU | 提示批 | 自注意力 QK 运算 T(N) = N(N + 1) ÷ 2 | token 数 N |
| --- | --- | --- | --- |
| GPU 1 | 6K, 2K, 6K | T(6K) + T(2K) + T(6K) = 18,003,000 + 2,001,000 + 18,003,000 = 38,007,000 | 14K |
| GPU 2 | 2K, 2K, 2K | 3 × T(2K) = 3 × 2,001,000 = 6,003,000 | 6K |

Here you see that the FIFO strategy sends the first three prompts [6K, 2K, 6K] to GPU 1 for approximately 38,007,000 self-attention Q × K dot products and 14,000 tokens of MLP, while GPU 2 sees approximately 6,003,000 self-attention Q × K computations and 6,000 tokens of MLP. The critical path is determined by GPU 1 at approximately 38,007,000 self-attention Q × K dot products.

在这里可以看到，FIFO 策略把前三个提示 [6K, 2K, 6K] 发送给 GPU 1，进行约 38,007,000 次自注意力 Q × K 点积和 14,000 个 token 的 MLP，而 GPU 2 则进行约 6,003,000 次自注意力 Q × K 计算和 6,000 个 token 的 MLP。关键路径（critical path）由 GPU 1 决定，约为 38,007,000 次自注意力 Q × K 点积。

In contrast, the latency-aware scheduler analyzes the expected latency of the six requests and rearranges them into two batches of [2K, 2K, 6K]. In this strategy, the self-attention cost per GPU is 22,005,000 dot products with 10,000 tokens of MLP, as shown in Table 16-7.

相比之下，延迟感知调度器会分析这六个请求的预期延迟，并把它们重新编排成两个 [2K, 2K, 6K] 的批。在这种策略下，每个 GPU 的自注意力开销为 22,005,000 次点积，外加 10,000 个 token 的 MLP，如表 16-7 所示。

Table 16-7. Latency-aware scheduling for prompts of length [6K, 2K, 6K, 2K, 2K, 2K] in that order

表 16-7. 对依次为 [6K, 2K, 6K, 2K, 2K, 2K] 长度的提示进行延迟感知调度

| GPU | Prompt batch | Self-attention QK ops T(N) = N(N + 1) ÷ 2 | Tokens N |
| --- | --- | --- | --- |
| GPU 1 | 2K, 2K, 6K | T(2K) + T(2K) + T(6K) = 2,001,000 + 2,001,000 + 18,003,000 = 22,005,000 | 10K |
| GPU 2 | 2K, 2K, 6K | T(2K) + T(2K) + T(6K) = 2,001,000 + 2,001,000 + 18,003,000 = 22,005,000 | 10K |

| GPU | 提示批 | 自注意力 QK 运算 T(N) = N(N + 1) ÷ 2 | token 数 N |
| --- | --- | --- | --- |
| GPU 1 | 2K, 2K, 6K | T(2K) + T(2K) + T(6K) = 2,001,000 + 2,001,000 + 18,003,000 = 22,005,000 | 10K |
| GPU 2 | 2K, 2K, 6K | T(2K) + T(2K) + T(6K) = 2,001,000 + 2,001,000 + 18,003,000 = 22,005,000 | 10K |

This reduces the critical path self-attention by about 42% from 38,007,000 to 22,005,000. As such, the latency-aware scheduler can significantly reduce TTFT. Because the latency-aware scheduler is aware of prompt lengths, the triangular N(N + 1) ÷ 2 cost in self-attention, and the O(N) cost in MLP, it can balance compute weight and minimize overall latency. For instance, in this example, the latency-aware scheduler moves the 2K token requests earlier. This allows these lighter requests to finish prefill sooner and begin decode without waiting for the heavier 6K token prefill to complete.

这把关键路径上的自注意力从 38,007,000 减少到 22,005,000，降低了约 42%。因此，延迟感知调度器可以显著降低 TTFT。由于延迟感知调度器了解提示长度、自注意力中三角形的 N(N + 1) ÷ 2 开销以及 MLP 中的 O(N) 开销，它可以平衡计算权重并最小化整体延迟。例如，在这个例子中，延迟感知调度器把 2K token 的请求提前。这让这些较轻的请求更早完成 prefill 并开始 decode，而无需等待较重的 6K token prefill 完成。

> These counts assume a variable-length fused attention kernel that computes only over valid tokens—for example, FlashAttention or PyTorch scaled dot product attention (SDPA) with the cuDNN backend. If your kernel pads to the maximum sequence length in a batch, the attention arithmetic cost approaches the batch size multiplied by T tokens of the longest sequence in that batch. In this case, the difference between FIFO and latency-aware strategies will shrink.

> 这些计数假设使用了一个变长的融合注意力核函数，它只在有效 token 上计算——例如 FlashAttention，或使用 cuDNN 后端的 PyTorch scaled dot product attention（SDPA）。如果你的核函数会填充到一个批中的最大序列长度，那么注意力算术开销将接近批大小乘以该批中最长序列的 T 个 token。在这种情况下，FIFO 与延迟感知策略之间的差异会缩小。

In short, a naive FIFO scheduler can overload one of the GPUs if the arrival order is not globally ideal within the batch. A latency-aware scheduler can analyze incoming requests and rearrange them into more balanced batches to minimize latency.

简而言之，如果批内的到达顺序在全局上并不理想，朴素的 FIFO 调度器可能会让其中一个 GPU 过载。延迟感知调度器可以分析到来的请求，并把它们重新编排成更均衡的批，以最小化延迟。

> Some inference serving frameworks have incorporated reinforcement learning to adjust scheduling policies online. This builds on the latency-aware strategies described here by automatically tuning for changing traffic patterns.

> 一些推理服务框架已经引入强化学习来在线调整调度策略。它在这里描述的延迟感知策略基础之上，通过针对不断变化的流量模式自动调优来加以扩展。

## Systems-Level Optimizations

## 系统级优化

Systems-level optimization techniques include overlapping communication with computation, maximizing GPU utilization, managing power and thermal concerns, handling errors, and optimizing memory access. Overall, we’re making sure that our GPU hardware is doing useful work, or goodput, nearly 100% of the time. We also discuss the trade-offs between throughput and latency and how to find a balance appropriate for production SLOs.

系统级优化（systems-level optimization）技术包括：让通信与计算重叠、最大化 GPU 利用率、管理功耗与热问题、处理错误以及优化内存访问。总体而言，我们要确保 GPU 硬件在几乎 100% 的时间里都在做有用功，即有效吞吐量（goodput）。我们还会讨论吞吐量与延迟之间的权衡，以及如何找到适合生产 SLO 的平衡点。

### Overlapping Communication and Computation

### 通信与计算重叠

As we’ve seen repeatedly throughout this book, overlapping communication with computation is critical for inference performance with large models distributed across many GPUs. The massively high bandwidth of systems like the GB200/GB300 NVL72 (up to ~130 TB/s of aggregate GPU-to-GPU NVLink bandwidth within a rack) means that overlap is even more effective because data transfers are fast enough that computation is less likely to be starved. However, even with the NVL72, it’s critical to use multiple CUDA streams and nonblocking collectives. This will allow you to perform communication-compute overlap to hide latency and keep all 72 GPUs busy—even at these speeds.

正如我们在本书中反复看到的，对于分布在众多 GPU 上的大模型，让通信与计算重叠对推理性能至关重要。像 GB200/GB300 NVL72 这样的系统具有极高的带宽（机架内 GPU 到 GPU 的 NVLink 聚合带宽高达约 130 TB/s），这意味着重叠的效果更加明显，因为数据传输足够快，计算不太可能被“饿着”。然而，即便有了 NVL72，使用多个 CUDA 流和非阻塞集合通信（nonblocking collectives）仍然至关重要。这将让你能够进行通信与计算的重叠，以隐藏延迟，并让全部 72 个 GPU 都保持忙碌——即使在如此高的速度下也是如此。

> In practice, you want to enable NCCL’s GPUDirect RDMA support—and use NCCL’s group calls to overlap multiple small all-reduce operations. Also, consider using SHARP (available on InfiniBand with suitable switches) to offload reduction operations to the network fabric. This optimization can improve throughput on some fabrics and topologies. In some tests, it has shown roughly 10%–20% throughput improvements for large all-to-all communications.

> 在实践中，你会想要启用 NCCL 的 GPUDirect RDMA 支持，并使用 NCCL 的分组调用来重叠多个小的 all-reduce 操作。此外，可以考虑使用 SHARP（在配备合适交换机的 InfiniBand 上可用）把归约操作卸载到网络交换结构上。这项优化可以在某些交换结构和拓扑上提升吞吐量。在一些测试中，它对大型 all-to-all 通信展现出大约 10%–20% 的吞吐量提升。

Consider an inference step that requires transferring data between GPUs (e.g., all-to-all for MoE)—or even just sending the prompt data from CPU to GPU and the response from the GPU to the CPU to return to the end user. In all of these scenarios, we want to perform these transfers asynchronously while the GPU is doing other work, whenever possible.

设想一个推理步骤需要在 GPU 之间传输数据（例如 MoE 的 all-to-all），甚至只是把提示数据从 CPU 发送到 GPU、再把响应从 GPU 发送回 CPU 以返回给最终用户。在所有这些场景中，我们都希望尽可能在 GPU 做其他工作的同时异步地执行这些传输。

One basic example is using separate CUDA streams to overlap compute and data transfer. For example, launch cudaMemcpyAsync on a dedicated transfer stream (with pinned host memory) while compute kernels run on the default stream.

一个基本的例子是使用各自独立的 CUDA 流来重叠计算与数据传输。例如，在一个专用的传输流上（配合固定的主机内存）启动 cudaMemcpyAsync，同时让计算核函数在默认流上运行。

> Remember to use page-locked host buffers for transfers over PCIe or NICs. This way, the CUDA driver can DMA directly and overlap with compute.

> 记住，对于通过 PCIe 或网卡进行的传输，要使用页锁定的主机缓冲区。这样，CUDA 驱动就可以直接进行 DMA 并与计算重叠。

You would then synchronize with a CUDA event only when the data is needed. This prevents the GPU from idling on I/O. This way, data transfers will use nonblocking CUDA calls (e.g., pinned memory) and CUDA streams so that the GPU doesn’t stall waiting on these operations. On modern GPUs with full duplex NVLink bandwidth, such overlap can largely hide communication latency. However, you should verify this with profiling on your specific workload.

然后，只有在需要数据时你才用一个 CUDA 事件进行同步。这可以避免 GPU 因 I/O 而闲置。这样，数据传输就会使用非阻塞的 CUDA 调用（例如固定内存）和 CUDA 流，从而使 GPU 不会因等待这些操作而停顿。在具备全双工 NVLink 带宽的现代 GPU 上，这种重叠可以在很大程度上隐藏通信延迟。不过，你应该在你的具体工作负载上通过剖析来验证这一点。

Data transfers in this context can include sending new prompt embeddings from the CPU to the GPU—or moving the last token’s logits from the GPU to the CPU. For instance, right after a GPU computes the logits for a batch of new tokens, we can kick off an asynchronous copy of those logits to the CPU for postprocessing (e.g., sampling)—or for sending back to the client as a streaming response—and immediately launch the next compute kernel for another batch of tokens that are still on the GPU.

在这种情况下，数据传输可以包括把新的提示嵌入从 CPU 发送到 GPU，或把最后一个 token 的 logits 从 GPU 搬到 CPU。例如，就在 GPU 为一批新 token 计算完 logits 之后，我们可以启动一次把这些 logits 异步拷贝到 CPU 的操作，用于后处理（例如采样），或作为流式响应发送回客户端，然后立即为另一批仍在 GPU 上的 token 启动下一个计算核函数。

In multinode scenarios, a communication collective like all-to-all can overlap by dividing work into chunks. A technique used in some MoE runtimes is to split the batch of tokens into two halves, and, while the first half’s tokens are being processed by experts, it can start sending the second half’s tokens.

在多节点场景中，像 all-to-all 这样的集合通信可以通过把工作划分成块来实现重叠。一些 MoE 运行时使用的一种技术是：把这一批 token 分成两半，在第一半的 token 正被专家处理时，就可以开始发送第二半的 token。

By the time the experts finish with the first half, the second half’s data has arrived at the destination GPUs—and they can proceed without waiting. This requires careful scheduling of the NCCL calls relative to compute kernels.

等到专家处理完第一半时，第二半的数据已经到达目标 GPU，于是它们无需等待就能继续。这需要相对于计算核函数对 NCCL 调用进行精心调度。

For example, you can implement this overlap using CUDA events to signal when half the batch is done. At this point, you can launch the NCCL all-to-all for that portion of the data—while the next portion finishes computing. You can use CUDA Graphs to capture these asynchronous patterns and reduce launch overhead. The result is improved GPU utilization because the GPUs spend less time sitting idle waiting for communications.

例如，你可以使用 CUDA 事件来实现这种重叠，用它来标示半个批何时完成。此时，你可以为那部分数据启动 NCCL all-to-all，同时让下一部分完成计算。你可以使用 CUDA Graphs 来捕获这些异步模式并减少启动开销。其结果是 GPU 利用率得到提升，因为 GPU 花在等待通信而闲置上的时间更少了。

Another area of overlap is between the CPU and GPU. For instance, while the GPU is busy generating the next token, the CPU can concurrently prepare the next input or perform result handling for previously generated tokens. A highly optimized inference engine will overlap any CPU-side preprocessing (e.g., input-text tokenization) in parallel with GPU computations from other requests. For example, the engine can prepare the next batch while overlapping with GPU computations of the current batch.

另一个可以重叠的领域是 CPU 与 GPU 之间。例如，当 GPU 忙于生成下一个 token 时，CPU 可以并发地准备下一个输入，或对先前已生成的 token 进行结果处理。一个高度优化的推理引擎会把任何 CPU 端的预处理（例如输入文本的分词）与来自其他请求的 GPU 计算并行重叠。例如，引擎可以在与当前批的 GPU 计算重叠的同时，准备下一个批。

This sounds obvious, but it requires a multithreaded architecture such that one thread handles networking and queuing, while another thread launches GPU operations. And yet another thread might handle response postprocessing, etc. The Python or C++ inference loop should never block the GPU from starting new work due to waiting on an operation that could otherwise be done concurrently.

这听起来理所当然，但它需要一种多线程架构：一个线程处理网络和排队，另一个线程启动 GPU 操作，再有一个线程可能处理响应的后处理，等等。Python 或 C++ 的推理循环绝不应该因为等待一个本可以并发完成的操作，而阻止 GPU 开始新的工作。

In pipeline parallel scenarios, the system should overlap communications (between pipeline stages) with computations inside of each stage. For instance, GPU 0, after finishing its part for token t, can start sending activations to GPU 1 at the same time as it begins processing token t+1 for another sequence.

在流水线并行场景中，系统应该把（流水线各阶段之间的）通信与每个阶段内部的计算重叠。例如，GPU 0 在完成它负责的 token t 的那部分之后，可以在开始为另一条序列处理 token t+1 的同时，开始把激活发送给 GPU 1。

Modern interconnects and frameworks allow compute kernels and data transfers to overlap as long as they use different resources. You should use multiple CUDA streams in an inference pipeline such that each pipeline stage has an independent stream. This way, the sending/receiving of activations can happen in parallel with other streams that are performing computations for different microbatches. The effect is that pipeline bubbles are reduced.

现代互连和框架允许计算核函数与数据传输重叠，只要它们使用不同的资源即可。你应该在推理流水线中使用多个 CUDA 流，使每个流水线阶段都有一个独立的流。这样，激活的发送/接收就可以与正在为不同微批（microbatch）执行计算的其他流并行进行。其效果是流水线气泡减少了。

On the networking side, technologies like NVIDIA GPUDirect RDMA allow network adapters like InfiniBand NICs to read and write GPU memory directly without involving the CPU—and without staging through host memory. By utilizing GPUDirect for cross-node transfers of KV cache or expert activations, the inference engine reduces latency and CPU overhead.

在网络方面，像 NVIDIA GPUDirect RDMA 这样的技术，允许 InfiniBand 网卡等网络适配器直接读写 GPU 内存，而无需涉及 CPU，也无需经过主机内存中转。通过利用 GPUDirect 进行 KV 缓存或专家激活的跨节点传输，推理引擎降低了延迟和 CPU 开销。

GPUDirect RDMA removes staging through host memory and allows the NIC DMA engines to access GPU memory directly. The CPU posts work requests, but the data path bypasses the CPU. This frees CPU cores for other tasks.

GPUDirect RDMA 消除了经过主机内存的中转，并允许网卡的 DMA 引擎直接访问 GPU 内存。CPU 负责提交工作请求，但数据路径绕过了 CPU。这把 CPU 核心解放出来用于其他任务。

Consider a practical example using InfiniBand for all-to-all communication in an MoE layer across two NVL72 racks. If you do this synchronously by first performing the all-to-all and then computing, the GPU utilization can remain low without overlap. By overlapping batches of the all-to-all communication with compute, however, you can significantly increase GPU utilization.

来看一个实用示例：在跨两个 NVL72 机架的 MoE 层中，使用 InfiniBand 进行全对全通信。如果同步执行——先完成全对全通信、再进行计算——由于没有重叠，GPU 利用率会保持在较低水平。但如果把一批批全对全通信与计算重叠起来，就能显著提升 GPU 利用率。

Overlapping all-to-all communication in an MoE layer involves splitting the token batch in half and beginning the second half’s exchange while the first half’s experts are still computing. This hides communication latency by interleaving it with compute.

在 MoE 层中重叠全对全通信，做法是把 token 批一分为二，在前一半的专家仍在计算时就开始后一半的交换。这样通过让通信与计算交错进行，把通信延迟隐藏了起来。

Essentially, each GPU begins computing its local experts’ outputs for the tokens it already has—while simultaneously receiving the remaining tokens from the other node. By the time it finishes the first batch, the second batch has arrived and can be computed immediately.

本质上，每块 GPU 会先用它已经拿到的 token 计算本地专家的输出，同时从另一个节点接收剩余的 token。等它算完第一批时，第二批已经到达，可以立即开始计算。

This kind of optimization is fairly low level and involves CUDA event synchronization between NCCL groups and CUDA streams, but it produces worthwhile throughput improvements and smoother latencies. In practice, you first launch the NCCL all-to-all as part of a NCCL group without waiting for completion. Then you immediately launch the next compute kernel. Make sure to check for completion using CUDA events.

这类优化相当底层，需要在 NCCL 组与 CUDA 流之间做 CUDA 事件同步，但它带来的吞吐提升和更平滑的延迟是值得的。实践中，你先把 NCCL 全对全通信作为一个 NCCL 组的一部分发起，但不等待其完成；然后立即启动下一个计算核函数；并务必用 CUDA 事件检查通信是否完成。

We can also overlap I/O with compute for streaming outputs. For instance, as soon as a few tokens are generated, we send them over the socket to the end user—while the model is already working on the next tokens. This hides the network latency (e.g., sending the response back to the end user) behind computation (e.g., subsequent tokens). As such, the end user sees a steady stream without pauses.

对于流式输出，我们也可以把 I/O 与计算重叠。例如，一旦生成了几个 token，就通过 socket 发送给终端用户，而此时模型已经在生成下一批 token。这样就把网络延迟（例如把响应发回终端用户）隐藏在计算（例如生成后续 token）之后。于是终端用户看到的是稳定、无停顿的输出流。

This pattern of nonblocking streaming to clients is adopted in all modern LLM inference engines. They use separate threads for sending token updates to clients using SSE/WebSockets.

这种向客户端进行非阻塞流式传输的模式，被所有现代 LLM 推理引擎所采用。它们用单独的线程，通过 SSE/WebSockets 向客户端发送 token 更新。

> If you are implementing streaming outputs yourself, make sure to use thread-safe queues or locks to hand off generated tokens to the networking thread. You don’t want to introduce synchronization issues in this handoff. These can be very difficult to identify and debug.

> 如果你自己实现流式输出，务必使用线程安全的队列或锁，把生成的 token 交接给网络线程。你不会希望在这个交接过程中引入同步问题——这类问题往往极难定位和调试。

If we didn’t overlap, we would generate a small number of tokens and then have the GPUs sit idle while we send the tokens back to the end user. Instead, our network send is handled by an async thread that takes the output from the response buffer and streams it back to the user. This allows the main inference thread (e.g., CUDA stream) to immediately continue generating more tokens. This effectively pipelines token generation (compute) with output transmission (I/O).

如果不做重叠，我们就会生成少量 token，然后在把这些 token 发回终端用户时让 GPU 闲置。相反，我们的网络发送由一个异步线程处理，它从响应缓冲区取出输出并流式回传给用户。这样主推理线程（例如 CUDA 流）就能立即继续生成更多 token。这实际上把 token 生成（计算）与输出传输（I/O）流水线化了。

In short, overlapping communication and computation requires thinking in terms of pipelines and using asynchronous operations whenever possible. Modern GPUs and networking hardware provide the primitives to implement this, including CUDA streams, nonblocking collectives, and RDMA. CUDA device-side primitives such as cuda::memcpy_async used with cuda::barrier (from the CUDA pipeline header) help overlap global to shared memory movement with compute inside a kernel. (Note: Host-to-device and device-to-host transfers still require explicit CUDA streams and pinned host memory to overlap with compute.) Efficient LLM inference systems take full advantage of these features.

简而言之，重叠通信与计算需要以流水线的方式思考，并尽可能使用异步操作。现代 GPU 与网络硬件提供了实现这一点的原语，包括 CUDA 流、非阻塞集合通信和 RDMA。诸如 cuda::memcpy_async（配合来自 CUDA pipeline 头文件的 cuda::barrier 使用）这样的 CUDA 设备侧原语，有助于在核函数内部把从全局内存到共享内存的搬运与计算重叠起来。（注：主机到设备、设备到主机的传输仍然需要显式的 CUDA 流和固定（pinned）主机内存，才能与计算重叠。）高效的 LLM 推理系统会充分利用这些特性。

In short, always overlap communication with computation whenever possible. Otherwise, scaling to many GPUs—even with NVLink/NVSwitch—will fall short of peak hardware performance. These optimizations are mostly invisible to the end user since there’s no change in model outputs. However, these are essential to squeeze out every bit of performance from the inference cluster and keep users coming back to your application.

简而言之，只要有可能，就始终让通信与计算重叠。否则，扩展到大量 GPU——即便用上 NVLink/NVSwitch——也达不到硬件的峰值性能。这些优化对终端用户来说基本是不可见的，因为模型输出没有任何变化。然而，要从推理集群中榨出每一分性能、并让用户不断回到你的应用，它们至关重要。

### Maximizing GPU Utilization and Throughput Versus Latency Trade-Offs

### 最大化 GPU 利用率，以及吞吐与延迟的权衡

The ultimate goal of performance tuning is to keep GPUs as busy as possible doing useful work, such as matrix multiplies—while minimizing any idle and underutilized GPU resources. Many of the techniques described previously, including batching, parallelism, and overlap, are designed to achieve near-100% GPU utilization.

性能调优的终极目标，是让 GPU 尽可能忙于做有用的工作（例如矩阵乘法），同时把空闲和未充分利用的 GPU 资源降到最低。前面介绍的许多技术——包括批处理、并行和重叠——都是为了实现接近 100% 的 GPU 利用率。

GPU utilization percentage—and specifically useful utilization, or goodput—is an important performance metric to monitor continuously. And while you want to push for 100% useful utilization, you need to make sure you are not violating your SLOs, such as latency. At some point, you may reach a point of diminishing returns.

GPU 利用率百分比——具体来说是有用利用率，即有效吞吐量——是需要持续监控的重要性能指标。虽然你希望把有用利用率推到 100%，但必须确保没有违反 SLO（例如延迟）。到了某个点之后，你可能会遇到收益递减。

Consider an inference server that is currently showing 60% GPU utilization with a naive implementation. Investigating, you find that the decode stage is a bottleneck because it’s waiting for a single thread to handle all sequences sequentially. Let’s introduce concurrent decoding using multiple streams. By interleaving decodes, as described earlier, we raise GPU utilization to 95% and double the throughput. This confirms that concurrent decoding is working.

设想一台推理服务器，采用朴素实现时 GPU 利用率为 60%。深入排查后你发现，decode 阶段是瓶颈，因为它在等待单个线程顺序处理所有序列。我们引入使用多个流的并发 decode。如前所述，通过让各次 decode 交错进行，我们把 GPU 利用率提升到 95%，吞吐翻倍。这证实并发 decode 确实起了作用。

We can try to reach 100% by putting more requests into a massive batch, but this will slow down individual queries. It’s often helpful to plot a throughput-versus-latency curve by testing various batch sizes. In this chart, there’s usually a sharp “knee” where throughput gains start costing too much latency.

我们可以通过把更多请求塞进一个超大批来逼近 100%，但这会拖慢单个查询。测试不同批大小、绘制一条吞吐-延迟曲线通常很有帮助。在这张图上，往往有一个明显的“拐点”，越过它之后，吞吐的提升就要以过高的延迟为代价。

To find the sweet spot in the throughput-latency curve, first turn on full concurrency and overlap as much as possible so you know the hardware can stay busy. Then gradually increase the batch size until you hit maximum resource utilization. At this point, measure single-query latency and scale back just enough (e.g., from 16 to 8) to meet your response-time targets.

要在吞吐-延迟曲线上找到最佳平衡点，先尽可能开启完全并发和重叠，以确认硬件能够保持忙碌。然后逐步增大批大小，直到达到最大资源利用率。此时测量单查询延迟，并适度回调（例如从 16 降到 8），刚好满足你的响应时间目标即可。

It’s recommended to monitor your p95 and p99 tail latencies along with p50 median latency. This is because small jitters and uneven batch fills can produce long-tail outliers noticeable at p95/p99. In a large cluster serving many requests, even a 0.1% outlier can occur frequently. Thus, p99—or even p99.9—might be more important than p50 for measuring user experience. In addition, long-tail latency often forces overprovisioning to meet aggressive SLOs. As such, reducing tail latency has direct cost benefits.

建议同时监控 p95、p99 尾延迟以及 p50 中位延迟。这是因为，微小的抖动和不均匀的批填充会产生长尾离群值，在 p95/p99 上尤为明显。在服务大量请求的大型集群中，即便是 0.1% 的离群值也会频繁出现。因此，就衡量用户体验而言，p99——甚至 p99.9——可能比 p50 更重要。此外，长尾延迟往往迫使你为满足激进的 SLO 而过度预置资源。因此，降低尾延迟能带来直接的成本收益。

> A good rule of thumb is to cap your batch size slightly below the one that gives peak throughput. Teams typically target 90% of maximum peak throughput, called the *headroom buffer*. This is because running at full speed can cause unpredictable latency spikes. For instance, if increasing the batch size to 16 requests starts to add a small amount of latency to a few requests, you should reduce the batch limit to 8. This will provide more consistent latency and reduce throughput only slightly.

> 一条不错的经验法则是：把批大小上限设置得略低于能带来峰值吞吐的那个值。团队通常以最大峰值吞吐的 90% 为目标，称为*余量缓冲*（headroom buffer）。这是因为满速运行会导致不可预测的延迟尖峰。例如，如果把批大小增加到 16 个请求后，开始给少数请求带来少量额外延迟，你就应该把批上限降到 8。这样能提供更一致的延迟，而吞吐只会略微下降。

Some inference systems can dynamically adjust the batch size on the fly based on these metrics. We’ll cover this technique—and many more adaptive inference strategies—in Chapters 17–19.

有些推理系统能够基于这些指标动态实时调整批大小。我们会在第 17–19 章介绍这项技术以及更多自适应推理策略。

It’s important to remember that running GPUs at a constant 100% can cause throttling by hitting power and thermal limits. Sometimes running at 90% with efficient kernels can outperform 100% with throttling. Next, let’s discuss GPU power and thermal constraints—characteristics that CPU-based application developers and systems engineers often don’t consider.

要记住一点：让 GPU 持续跑在 100%，可能因触及功耗和热限制而引发降频（throttling）。有时，用高效核函数跑在 90%，反而胜过在降频状态下跑 100%。接下来，我们讨论 GPU 的功耗与热约束——这是基于 CPU 的应用开发者和系统工程师常常不会考虑的特性。

### Power and Thermal Constraints

### 功耗与热约束

Another dimension to consider is power and thermal constraints. Continuously running GPUs at 100% will cause thermal throttling if cooling is insufficient. Modern GPU systems are liquid-cooled to reduce thermal throttling and sustain performance. If you’re running older, air-cooled systems, watch out for downclocks.

另一个需要考虑的维度是功耗与热约束。如果散热不足，让 GPU 持续跑在 100% 会引发热降频。现代 GPU 系统采用液冷，以减少热降频并维持性能。如果你运行的是较老的风冷系统，就要当心降频（downclock）。

You can monitor downclocks with nvidia-smi or DCGM when GPU utilization is high. Specifically, DCGM exposes XID throttling reasons through DCGM_FI_DEV_CLOCK_THROTTLE_REASONS. And nvidia-smi will show a “Pwr Throttle” flag when a GPU is being power-throttled.

当 GPU 利用率很高时，你可以用 nvidia-smi 或 DCGM 监控降频。具体来说，DCGM 通过 DCGM_FI_DEV_CLOCK_THROTTLE_REASONS 暴露 XID 降频原因。而当某块 GPU 被功耗降频时，nvidia-smi 会显示“Pwr Throttle”标志。

You may also hit power limits on the board—especially with boost clocks. Modern GPUs draw significantly more power under boost, so you should monitor if GPUs downclock due to hitting a power limit. If this happens, you will experience less-than-ideal GPU performance. Always monitor the DCGM_FI_DEV_POWER_USAGE metric from DCGM and alert anytime this exceeds the normal range.

你还可能触及板卡的功耗上限——尤其是在使用加速时钟（boost clock）时。现代 GPU 在 boost 状态下功耗显著更高，因此你应该监控 GPU 是否因触及功耗上限而降频。一旦发生，你会得到不理想的 GPU 性能。请始终监控 DCGM 的 DCGM_FI_DEV_POWER_USAGE 指标，并在其超出正常范围时告警。

> If your GPUs consistently hit power limits, consider enabling Dynamic Boost mode—or underclocking slightly—to avoid thermal throttling that can spike latency.

> 如果你的 GPU 持续触及功耗上限，可以考虑启用 Dynamic Boost 模式——或稍微降频——以避免会导致延迟尖峰的热降频。

To work around these constraints in software, you can slightly ease off utilization by adding a tiny interbatch delay. You can also limit concurrency. Modern GPUs also support clock capping. So rather than adding delay, you could cap the GPU clocks marginally using nvidia-smi -lgc or similar to prevent hitting thermal limits. This will only slightly reduce throughput—and provide more consistent performance.

要在软件层面绕开这些约束，你可以通过增加极小的批间延迟来略微降低利用率；也可以限制并发度。现代 GPU 还支持时钟封顶。因此，与其增加延迟，你也可以用 nvidia-smi -lgc 之类的命令把 GPU 时钟略微封顶，以防触及热限制。这只会略微降低吞吐，却能带来更一致的性能。

From a hardware perspective, you can try to improve cooling or raise the power limit if cooling is already adequate. To increase the power cap, use nvidia-smi -pl or tune your GPU boost settings.

从硬件角度看，你可以尝试改善散热；如果散热已经足够，则可以提高功耗上限。要提高功耗上限，使用 nvidia-smi -pl 或调整你的 GPU boost 设置。

These are all reminders that pushing real-world systems to their limits can cause unexpected side effects that greatly impact performance. It’s recommended to include a “boost-off” flag in your inference engine to apply the software workarounds on the fly when the system detects performance degradation due to throttling caused by power or thermal constraints. This will run your system in a slightly underutilized and cooler state until full stability and performance are restored.

这些都在提醒我们：把真实系统推到极限，可能引发意料之外的副作用，严重影响性能。建议在推理引擎中加入一个“boost-off”开关：当系统检测到因功耗或热约束导致降频、进而造成性能下降时，就实时应用上述软件绕行方案。这会让系统在略微欠载、温度更低的状态下运行，直到完全恢复稳定性和性能。

### Error Handling

### 错误处理（error handling）

While not typically associated with performance, it’s important to handle errors efficiently. A fully utilized inference system has less headroom to handle excessive error spikes—especially since error handling is likely not the most optimal code path in your system.

虽然错误处理通常不被视为性能话题，但高效地处理错误很重要。一个已被充分利用的推理系统，应对过量错误尖峰的余量更少——尤其因为错误处理很可能并不是你系统中最优化的代码路径。

In failure scenarios, failing fast is the key since, if a request will error, it’s best to return the error immediately rather than waste GPU time with a slow failure. Be sure to implement proper exceptions and timeouts around inference calls to catch hangs or crashes.

在失败场景中，快速失败（fail fast）是关键：如果一个请求注定要出错，最好立即返回错误，而不是以缓慢的失败浪费 GPU 时间。务必在推理调用周围实现恰当的异常处理和超时，以捕获挂起或崩溃。

Additionally, it’s recommended to implement backpressure. This means that if errors spike, you can start rejecting new requests—or reduce batch sizes—to give the system some headroom to recover.

此外，建议实现背压（backpressure）。也就是说，如果错误激增，你可以开始拒绝新请求——或减小批大小——给系统一些恢复的余量。

At scale, it’s recommended to build in some headroom, adding some extra replicas that sit mostly idle. These can act as a buffer for spikes in errors or other unexpected situations. While autoscaling is likely the first option you think of to reduce cost, just remember that autoscaling takes time to provision the new resources.

在大规模场景下，建议预留一些余量，增加若干大多处于空闲的额外副本。它们可以充当缓冲，应对错误尖峰或其他意外情况。虽然自动扩缩（autoscaling）很可能是你为降低成本首先想到的方案，但要记住，自动扩缩需要时间来预置新资源。

While idle capacity costs money, the cost of lost customers or SLA violations is often higher. At least 10%–15% of buffer capacity is recommended during steady state. For more critical services that cannot incur downtime, it’s recommended to provision 100% buffer capacity—or a full cluster replica.

空闲容量虽然要花钱，但流失客户或违反 SLA 的代价往往更高。建议在稳态期间至少预留 10%–15% 的缓冲容量。对于不能承受宕机的更关键服务，建议预置 100% 的缓冲容量——即一整套集群副本。

There is simply no substitute for prewarmed, idle nodes that can handle the extra load immediately on demand. The cost of losing end users is likely higher than the cost of keeping a few replicas idle and ready to handle any additional, unexpected load.

能够按需立即承接额外负载的预热空闲节点，是无可替代的。流失终端用户的代价，很可能高于让少数副本保持空闲待命、随时应对额外意外负载的代价。

### Memory

### 内存

Optimizing resource utilization extends to memory as well. While GPU memory and memory bandwidth are growing somewhat incrementally, model sizes and context lengths are growing exponentially. As such, memory remains a precious resource to optimize—and will remain precious in the near future.

优化资源利用同样延伸到内存。GPU 内存和内存带宽只是渐进式增长，而模型规模和上下文长度却在指数级增长。因此，内存仍是需要优化的宝贵资源——在可预见的未来也将一直如此宝贵。

As such, you want to utilize GPU memory and memory bandwidth effectively. Memory is typically filled by the model weights and KV cache. During inference, it’s best to keep the model weights in GPU HBM at all times. If you page memory in and out of CPU DRAM or NVMe storage, you will incur extra page faults and transfer latency.

因此，你要高效利用 GPU 内存和内存带宽。内存通常被模型权重和 KV 缓存占满。推理期间，最好始终把模型权重保留在 GPU HBM 中。如果你在 CPU DRAM 或 NVMe 存储之间换入换出内存，就会引入额外的缺页（page fault）和传输延迟。

This additional transfer latency is true even for modern CPU-GPU superchips like Grace Blackwell and Vera Rubin. As such, high-performance LLM inference engines explicitly manage memory rather than relying on on-demand paging.

即便是 Grace Blackwell、Vera Rubin 这样的现代 CPU-GPU 超级芯片，这种额外的传输延迟依然存在。因此，高性能 LLM 推理引擎会显式管理内存，而不是依赖按需分页。

Compression is an effective technique to reduce memory usage in your inference system. Specifically for the KV cache, which is generated by the prefill stage for every query that comes into the system.

压缩是减少推理系统内存占用的一种有效技术，尤其针对 KV 缓存——它由 prefill 阶段为进入系统的每个查询生成。

Since the size of the KV cache scales with the number of queries and the size of the inputs, you should consider KV cache compression and quantization. KV cache compression/quantization means storing reduced-precision keys and values if the model can tolerate the precision loss. And, while not ideal, KV cache offloading is an option for rarely used KV cache data.

由于 KV 缓存的大小随查询数量和输入规模而增长，你应当考虑 KV 缓存压缩与量化。KV 缓存压缩/量化意味着：在模型能够容忍精度损失的前提下，以降低的精度存储键和值。此外，虽然不算理想，但对于很少使用的 KV 缓存数据，KV 缓存卸载也是一种选择。

### KV Cache Offloading and Memory Pool Allocation

### KV 缓存卸载与内存池分配（memory pool allocation）

By offloading (paging out) rarely used KV cache entries to CPU memory or disk, inference engines make room for more active data. This is similar to handling virtual memory.

通过把很少使用的 KV 缓存条目卸载（换出，paging out）到 CPU 内存或磁盘，推理引擎为更活跃的数据腾出空间。这与处理虚拟内存类似。

For instance, vLLM’s PagedAttention offloads to CPU memory and NVMe storage using a managed memory pool for the KV cache. Similarly, SGLang’s RadixAttention uses a tree-structured cache that can lazily evict least-used prefixes. NVIDIA Dynamo has a similar mechanism for KV cache offloading and memory management.

例如，vLLM 的 PagedAttention 使用一个受管理的内存池来管理 KV 缓存，并把数据卸载到 CPU 内存和 NVMe 存储。类似地，SGLang 的 RadixAttention 使用树状结构的缓存，可以惰性逐出最少使用的前缀。NVIDIA Dynamo 也有类似的 KV 缓存卸载与内存管理机制。

And without a good KV cache allocator, you can have excessive memory fragmentation (e.g., ~20%–30%) when requests of varying lengths flow through the system. This will limit how many requests you can handle before experiencing an OOM error. Remember to use allocators with large pool sizes, as discussed in an earlier chapter.

而如果没有一个好的 KV 缓存分配器，当长度不一的请求流经系统时，你可能出现过度的内存碎片（例如 ~20%–30%）。这会限制你在遇到 OOM 错误之前能处理的请求数量。记住要使用具有较大池容量的分配器，正如前面某章讨论过的那样。

Adopting a proper memory-paging strategy will reduce fragmentation down to a tolerable few percent. This means you can pack more contexts into the GPU and keep utilization high without crashing the system.

采用恰当的内存分页策略，可以把碎片率降到可容忍的百分之几。这意味着你能把更多上下文塞进 GPU，在不让系统崩溃的前提下保持高利用率。

With proper KV cache memory management, modern inference engines serve many more concurrent users without running out of GPU memory. They do this by managing their own KV cache memory pools and offloading data to CPU and NVMe storage. As such, they achieve near-full memory utilization with minimal fragmentation.

有了恰当的 KV 缓存内存管理，现代推理引擎能够在不耗尽 GPU 内存的情况下服务多得多的并发用户。它们的做法是自行管理 KV 缓存内存池，并把数据卸载到 CPU 和 NVMe 存储。因此，它们以极少的碎片实现接近满额的内存利用率。

The trade-off is a bit of extra data transfer latency if the contexts become active again and need to be paged in. This overhead is typically small, however, compared to the total request latency.

其代价是：如果这些上下文再次变为活跃、需要被换入，会产生少量额外的数据传输延迟。不过，与请求的总延迟相比，这点开销通常很小。

Memory utilization and compute utilization go hand in hand. If memory is poorly managed, you will waste memory, and you can’t fully utilize the GPU with more concurrent tasks. By keeping as much relevant data on the GPU as possible, the system can service a massive number of concurrent requests.

内存利用率与计算利用率密切相关。如果内存管理不善，你就会浪费内存，也无法用更多并发任务充分利用 GPU。通过尽可能把相关数据保留在 GPU 上，系统能够服务海量的并发请求。

Poor memory management can cause repeated OOM crashes under peak load. This will take GPUs out of the pool and cause cascading latency issues. Avoid OOMs using proper memory management. This will maximize utilization and maintain cluster stability.

糟糕的内存管理会在峰值负载下导致反复的 OOM 崩溃。这会把 GPU 从资源池中移除，并引发级联的延迟问题。用恰当的内存管理来避免 OOM，这样才能最大化利用率并维持集群稳定。

In short, efficient memory management can improve effective throughput, reduce memory fragmentation, avoid unexpected OOM errors, and allow more concurrent in-flight requests—especially for high-throughput scenarios.

简而言之，高效的内存管理能够提升有效吞吐、减少内存碎片、避免意外的 OOM 错误，并允许更多并发的在途（in-flight）请求——在高吞吐场景中尤为如此。

## Quantization Approaches for Real-Time Inference

## 面向实时推理的量化方法

One of the most effective ways to increase inference performance is to reduce precision. This will instantly decrease memory usage and memory bandwidth utilization—and increase compute speed.

提升推理性能最有效的方法之一，就是降低精度。这会立即减少内存占用和内存带宽占用，并提高计算速度。

Quantization represents the model’s weights—and sometimes the activations—with fewer bits. Modern NVIDIA GPUs support low-precision arithmetic natively using reduced-precision Tensor Cores for FP16, FP8, and FP4 formats, among others.

量化用更少的比特来表示模型的权重——有时还包括激活值。现代 NVIDIA GPU 原生支持低精度运算，其降精度 Tensor Core 可处理 FP16、FP8、FP4 等格式。

In this section, we discuss quantization techniques specifically for inference. This includes weight-only quantization methods like GPTQ, AWQ, SpQR, and other structured-sparsity-aware methods—as well as full precision reduction for weight and activation quantization. Specifically, GPTQ and AWQ have proven very effective in practice.

在本节中，我们讨论专门面向推理的量化技术。这既包括 GPTQ、AWQ、SpQR 等仅权重量化方法以及其他结构化稀疏感知方法，也包括针对权重和激活量化（activation quantization）的全面降精度。具体而言，GPTQ 和 AWQ 在实践中已被证明非常有效。

For many large models, 4-bit weight-only quantization with GPTQ can retain 99%+ of the accuracy of the FP16 model. And it provides ~2× inference speedups and a ~4× smaller model footprint. AWQ further improves accuracy on 3–4-bit quantization. These techniques are integrated into many AI frameworks, including Hugging Face Transformers, PyTorch, vLLM, and many others. They support loading GPTQ and AWQ quantized models directly. Next, let’s cover the trade-offs in accuracy and performance using reduced-precision formats—and how to integrate quantization effectively and safely into a serving workflow.

对许多大模型而言，用 GPTQ 做 4-bit 仅权重量化可以保留 FP16 模型 99%+ 的准确率，同时带来 ~2× 的推理加速和 ~4× 更小的模型体积。AWQ 在 3–4-bit 量化上进一步提升了准确率。这些技术已集成到许多 AI 框架中，包括 Hugging Face Transformers、PyTorch、vLLM 等，它们支持直接加载 GPTQ 和 AWQ 量化后的模型。接下来，我们讨论使用降精度格式在准确率与性能上的权衡，以及如何把量化有效且安全地集成进服务工作流。

### Reducing Precision from FP16 to FP8 and FP4

### 从 FP16 降到 FP8 与 FP4

Initially, LLM inference saw major gains by moving from FP32 to reduced-precision formats like TF32, FP16, or BF16. NVIDIA Tensor Cores, for instance, perform 2× the throughput when using FP16 versus FP32. It does this by fusing half-precision multiply-add operations into the specialized Tensor Core hardware pipelines that double the math performance—and without noticeable accuracy loss.

最初，LLM 推理通过从 FP32 转向 TF32、FP16 或 BF16 等降精度格式获得了重大收益。例如，NVIDIA Tensor Core 使用 FP16 时的吞吐是 FP32 的 2×。其做法是把半精度乘加运算融合进专用的 Tensor Core 硬件流水线，从而在没有明显准确率损失的情况下把数学性能翻倍。

FP8 reduces precision even lower to 8-bit floating point. This reduces the memory footprint by half compared to FP16/BF16. And it doubles Tensor-Core math throughput again because the GPUs execute twice as many 8-bit multiply-adds per cycle versus 16-bit operations.

FP8 把精度进一步降到 8 位浮点。与 FP16/BF16 相比，这把内存占用减少了一半。而且它再次把 Tensor Core 的数学吞吐翻倍，因为相比 16 位运算，GPU 每个周期能执行两倍数量的 8 位乘加。

And while you can gain a moderate speedup in PyTorch by simply enabling TF32 math with torch.set_float32_matmul_precision("high"), you want to fully utilize the 8-bit and 4-bit precision support provided by NVIDIA’s Transformer Engine (TE). The TE provides FP8 and FP4 kernels as a library, which allows existing code to use these reduced precisions with minimal changes.

虽然在 PyTorch 中只需用 torch.set_float32_matmul_precision("high") 启用 TF32 运算就能获得中等程度的加速，但你会想充分利用 NVIDIA 的 Transformer Engine（TE）所提供的 8 位和 4 位精度支持。TE 以库的形式提供 FP8 和 FP4 核函数，让现有代码只需极少改动即可使用这些降精度格式。

NVIDIA’s TE automatically manages per-tensor scaling factors at these reduced precisions. At inference time, your inference server can load a model in FP16 but use FP8 matrix multiplies.

NVIDIA 的 TE 会在这些降精度下自动管理逐张量（per-tensor）的缩放因子（scaling factor）。在推理时，你的推理服务器可以用 FP16 加载模型，但使用 FP8 矩阵乘法。

The TE applies scaling to each tensor to maintain numerical stability using a scaling factor that is typically chosen one of two ways: a fixed, ahead-of-time calibration step using representative data during training, called *static calibration*—or a dynamically computed value that tracks the tensor’s maximum absolute value, called *amax-based* *dynamic scaling*. Figure 16-7 shows the TE’s using range analysis, scaling factor, and target format for the precision conversion.

TE 对每个张量施加缩放以维持数值稳定性，其缩放因子通常有两种选择方式：一种是在训练期间用代表性数据进行的固定、提前的校准（calibration）步骤，称为*静态校准*（static calibration）；另一种是动态计算、跟踪张量最大绝对值的值，称为*基于 amax 的动态缩放*（amax-based dynamic scaling）。图 16-7 展示了 TE 在精度转换中使用范围分析（range analysis）、缩放因子和目标格式的过程。

![Figure 16-7. NVIDIA Transformer Engine (TE) using range analysis, scaling factor, and target format for the precision conversion on a transformer layer](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-7.png)

![图 16-7. NVIDIA Transformer Engine 在一个 transformer 层上，使用范围分析、缩放因子和目标格式进行精度转换](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-7.png)

For more compression, the FP4 format reduces model weight storage and traffic better than FP8 does. Accounting for scaling metadata and packing, the effective reduction is commonly around 1.8× compared with FP8 and about 3.5× compared with FP16. However, because FP4’s dynamic range is very limited, reliable inference at FP4 requires per-channel scaling—or other calibration such as NVIDIA’s per-block *microscaling* supported in the GPU’s TE. These techniques are needed to make FP4 usable for large networks by minimizing accuracy loss.

若要更高的压缩率，FP4 格式在减少模型权重的存储和传输方面比 FP8 更胜一筹。把缩放元数据和打包计入后，其有效压缩率相比 FP8 通常约为 1.8×，相比 FP16 约为 3.5×。然而，由于 FP4 的动态范围非常有限，要在 FP4 下实现可靠推理，需要逐通道（per-channel）缩放——或其他校准方式，例如 GPU 的 TE 所支持的 NVIDIA 逐块 *microscaling*。这些技术通过把准确率损失降到最低，使 FP4 得以用于大型网络。

### Weight-Only Quantization (GPTQ, AWQ)

### 仅权重量化（GPTQ、AWQ）

Weight-only quantization in modern LLM serving stacks typically compresses weights to 4-bit integers using methods like GPTQ or AWQ while keeping activations in higher precision such as FP8, FP16, or INT8. This reduces the weight memory footprint by roughly four times versus FP16 and halves or better the weight bandwidth, usually with minimal accuracy loss when properly calibrated.

在现代 LLM 服务栈中，仅权重量化通常用 GPTQ 或 AWQ 之类的方法把权重压缩为 4 位整数，同时把激活值保留在更高精度（如 FP8、FP16 或 INT8）。相比 FP16，这把权重的内存占用减少约四倍，并把权重带宽减半甚至更多；在经过恰当校准时，通常只有极小的准确率损失。

NVIDIA’s FP4 implementation (officially called *NVFP4* but referred to as just *FP4* in this text) uses block-wise microscaling in hardware. The NVIDIA TE provides the hardware support for NVFP4 microscaling at block granularity.

NVIDIA 的 FP4 实现（官方称为 *NVFP4*，但本文简称 *FP4*）在硬件中采用逐块 microscaling。NVIDIA TE 在块粒度上为 NVFP4 microscaling 提供硬件支持。

> Use per-tensor or per-channel scaling depending on the kernel. It’s recommended to explore per-tensor scaling for activations and per-channel scaling for weights. For instance, when using FP8 E4M3 for KV cache quantization, it’s common to use per-tensor scaling.

> 根据核函数的不同，选用逐张量或逐通道缩放。建议为激活值探索逐张量缩放、为权重探索逐通道缩放。例如，使用 FP8 E4M3 做 KV 缓存量化时，常用逐张量缩放。

Per-block microscaling means that instead of using a single scaling factor for an entire tensor, it maintains a separate scale for each fixed-size block, typically 32 elements, within the tensor. These separate scales adapt to local value distributions to preserve range and reduce quantization errors compared to single-scale quantization.

逐块 microscaling 意味着：它不是对整个张量使用单一缩放因子，而是为张量内每个固定大小的块（通常为 32 个元素）维护一个独立的缩放因子。相比单一缩放的量化，这些独立缩放因子会自适应局部数值分布，从而保留数值范围并减少量化误差。

> Always quantize with the calibration data that reflects your inference workload. A one-time calibration on a subset of your training data may not capture runtime usage patterns.

> 始终使用能反映你推理工作负载的校准数据来做量化。仅在训练数据子集上做的一次性校准，可能无法捕捉运行时的使用模式。

In practice, GPTQ (post-training quantization) and AWQ are commonly used for 4-bit weights on LLMs—often with negligible accuracy loss. Open source tools from Hugging Face and others can apply these techniques automatically.

实践中，GPTQ（训练后量化，post-training quantization，PTQ）和 AWQ 常用于 LLM 的 4 位权重——通常准确率损失可以忽略。来自 Hugging Face 等的开源工具可以自动应用这些技术。

The MoE expert structure makes weight quantization even more appealing since we can fit even more experts in memory using reduced-precision weights. As such, we can swap fewer experts to/from CPU memory. Additionally, we can use larger models with more active experts—and more expert replicas—if needed.

MoE 的专家结构让权重量化更具吸引力，因为用降精度权重可以在内存中容纳更多专家。这样，我们在 CPU 内存与显存之间换入换出的专家就更少。此外，如有需要，我们还能使用更大的模型、更多活跃专家以及更多专家副本。

Post-training quantization (PTQ) tools like GPTQ apply an approximate second-order algorithm to quantize weights, layer-by-layer, down to 3–4 bits in a few GPU hours with almost no accuracy degradation. Newer GPTQ variants further refine this algorithm with asymmetric calibration and parallel computation. This reduces quantization error and extends efficient low-bit support to even larger models.

像 GPTQ 这样的训练后量化（PTQ）工具，运用一种近似的二阶算法，逐层把权重量化到 3–4 位，只需几个 GPU 小时，几乎没有准确率下降。更新的 GPTQ 变体通过非对称校准和并行计算进一步改进了该算法。这既减少了量化误差，又把高效的低比特支持扩展到了更大的模型。

AWQ identifies a small fraction of “salient” weight channels. Salient channels produce large activation magnitudes that have a disproportionately large effect on model output.

AWQ 会识别出一小部分“显著”（salient）权重通道。显著通道会产生较大的激活幅度，对模型输出的影响格外大。

AWQ preserves these channels using channel-specific scaling before casting all weights into 4-bit precision (e.g., INT4). NVIDIA’s TE knows how to use the channel-specific scaling factors on the preserved channels to maintain model fidelity.

在把所有权重转换为 4 位精度（例如 INT4）之前，AWQ 会用逐通道专属的缩放来保留这些通道。NVIDIA 的 TE 知道如何在这些被保留的通道上使用逐通道专属的缩放因子，以维持模型保真度。

> Most AI frameworks like PyTorch and inference engines like vLLM and TensorRT-LLM natively support loading GPTQ-quantized and AWQ-quantized model checkpoints.

> 大多数 AI 框架（如 PyTorch）和推理引擎（如 vLLM 和 TensorRT-LLM）都原生支持加载经 GPTQ 量化和 AWQ 量化的模型检查点。

### Activation Quantization

### 激活量化

Quantizing activations along with weights can improve performance by using reduced precision for GEMM inputs—and potentially accumulators—using lower-precision values. This will reduce memory traffic for both attention-based KV cache and intermediate activations in MLP layers.

在量化权重的同时量化激活值，可以通过对 GEMM 的输入——乃至累加器——使用更低精度的值来提升性能。这会减少基于注意力的 KV 缓存以及 MLP 层中间激活值的内存流量。

However, activation distributions can vary greatly with varying inputs. As such, activation quantization can sometimes be challenging without proper fine-tuning or calibration. A middle ground is INT8 activation with calibration. This comes from NVIDIA’s INT8 mode, which uses per-tensor calibration and is used in TensorRT to choose scaling factors from activation histograms generated from a representative set of calibration data.

然而，激活值的分布会随输入的不同而大幅变化。因此，在没有恰当微调或校准的情况下，激活量化有时颇具挑战。一种折中方案是带校准的 INT8 激活。它来自 NVIDIA 的 INT8 模式：该模式使用逐张量校准，并在 TensorRT 中根据由代表性校准数据生成的激活直方图来选择缩放因子。

SmoothQuant, a training/calibration-free PTQ method for 8-bit activation quantization, can be used to shift some of the quantization error from activations to weights using a simple row/column scaling algorithm. This lets us use INT8 for both weights and activations with minimal fine-tuning—leading to full INT8 inference with low (e.g., < 1%) accuracy loss.

SmoothQuant 是一种面向 8 位激活量化、无需训练/校准的 PTQ 方法，它用一个简单的行/列缩放算法，把一部分量化误差从激活值转移到权重上。这让我们只需极少的微调就能对权重和激活都使用 INT8——从而实现完整的 INT8 推理，且准确率损失很低（例如 < 1%）。

> Using SmoothQuant activation quantization before applying GPTQ/AWQ on weights has been shown to preserve accuracy better at low precision.

> 已有研究表明，在对权重应用 GPTQ/AWQ 之前先用 SmoothQuant 做激活量化，能在低精度下更好地保持准确率。

### Post-Training Quantization Workflow

### 训练后量化流程

The quantization techniques described previously are applied post-training—as opposed to quantization-aware training, which is done during model training (aka *model pretraining*). The typical workflow is to train or fine-tune the model in FP16/FP32 and then run a post-training calibration script such as GPTQ or AWQ on a representative dataset to determine quantization parameters. Then you load the quantized model weights into the inference engine for serving.

前面介绍的量化技术都是在训练后应用的——与之相对的是量化感知训练（QAT），后者在模型训练（也称*模型预训练*）期间进行。典型流程是：先用 FP16/FP32 训练或微调模型，然后在代表性数据集上运行 GPTQ 或 AWQ 等训练后校准脚本来确定量化参数；接着把量化后的模型权重加载进推理引擎进行服务。

If needed, you can also run a small fine-tuning job to recover any lost accuracy; however, this is typically not needed with the GPTQ and AWQ techniques. If you need full QAT, you can just run a few epochs of training with “fake quant” operations in the model graph to mimic low-precision math during evaluation. This will help you gauge expected accuracy at this precision.

如有需要，你也可以运行一个小的微调任务来恢复损失的准确率；不过使用 GPTQ 和 AWQ 技术时通常并不需要。如果你需要完整的 QAT，只需在模型计算图中加入“伪量化”（fake quant）操作跑几个训练轮次，在评估时模拟低精度运算即可。这有助于你估计该精度下预期的准确率。

> In practice, quantization-aware fine-tuning for LLMs is computationally expensive and not always feasible. However, smaller calibration datasets of 1,000 prompts, for instance, can be used with techniques like percentile clipping or LMS (loss-aware quantization) to fine-tune the quantization scales without full retraining.

> 实践中，针对 LLM 的量化感知微调计算代价高昂，并不总是可行。不过，可以使用较小的校准数据集（例如 1,000 条提示），配合百分位裁剪（percentile clipping）或 LMS（损失感知量化，loss-aware quantization）等技术，在不完全重训练的情况下微调量化缩放因子。

It’s worth noting that you should be extra careful to balance the trade-off between compression and model robustness. Quantization can amplify certain errors. For instance, if a model was barely at the threshold of some factual knowledge, quantization might push it to make an error.

值得注意的是，你应格外小心地在压缩与模型鲁棒性之间做权衡。量化可能放大某些误差。例如，如果一个模型对某项事实知识本就勉强达到阈值，量化可能会把它推向出错。

Always validate on downstream tasks since PTQ makes assumptions that might miss subtle distribution shifts. As such, you should do extensive A/B testing of responses when using reduced-precision quantized models to make sure your evaluations show no regressions in quality or safety. If you see regressions, you should leave the model in higher precision—or try performing a light fine-tune in quantized form to restore performance.

务必在下游任务上验证，因为 PTQ 所做的假设可能会忽略细微的分布偏移。因此，在使用降精度量化模型时，你应对响应进行充分的 A/B 测试，确保评估中在质量或安全性上没有回退。如果发现回退，你应把模型保留在更高精度——或尝试以量化形式做一次轻量微调来恢复性能。

### Combining Weight and Activation Quantization

### 结合权重量化与激活量化

Activation quantization to 4-bit remains challenging. Combining low-precision weight quantization with higher-precision activation quantization often produces the best trade-off between memory savings, compute efficiency, and accuracy. As such, many production systems use weight-only 4-bit (e.g., GPTQ/AWQ) quantization combined with 8-bit activation quantization.

把激活量化到 4 位仍然颇具挑战。把低精度的权重量化与更高精度的激活量化结合起来，往往能在节省内存、计算效率与准确率之间取得最佳权衡。因此，许多生产系统采用 4 位仅权重量化（例如 GPTQ/AWQ）结合 8 位激活量化。

Specifically, in one W4A8 (8-bit activation) variant, the runtime unpacks INT4 weights, dequantizes to FP8 using learning or calibration scales, and executes the matrix multiply on FP8 Tensor Cores. This hybrid path is provided by inference engines like TensorRT-LLM and achieves near-lossless accuracy when properly calibrated. This preserves the full dynamic range of activations while reducing weight storage by 4× compared to FP16.

具体来说，在一种 W4A8（8 位激活）变体中，运行时先解包 INT4 权重，用学习到的或校准得到的缩放因子反量化为 FP8，再在 FP8 Tensor Core 上执行矩阵乘法。这条混合路径由 TensorRT-LLM 等推理引擎提供，在恰当校准时可达到近乎无损的准确率。它既保留了激活值的完整动态范围，又相比 FP16 把权重存储减少了 4×。

In contrast, the traditional INT4 and INT8 W4A8 scheme pairs 4-bit integer (INT4) weights with 8-bit integer (INT8) activations and runs the computations on INT8 Tensor Cores. This approach relies on histogram-based calibration to map activation ranges into INT8 without quality loss.

相比之下，传统的 INT4 加 INT8 的 W4A8 方案，把 4 位整数（INT4）权重与 8 位整数（INT8）激活配对，并在 INT8 Tensor Core 上运行计算。该方法依赖基于直方图的校准，把激活范围无损地映射到 INT8。

Although the INT4/INT8 kernels can deliver slightly higher raw throughput with modern integer Tensor Core pipelines, they require careful activation calibration and don’t have as much dynamic range as FP8.

尽管借助现代整数 Tensor Core 流水线，INT4/INT8 核函数能提供略高的原始吞吐，但它们需要仔细的激活校准，且动态范围不如 FP8。

A hybrid INT4 and FP8 approach combines the best of both worlds. In the hybrid approach, 4-bit integer (INT4) weights are unpacked and reinterpreted as FP8 inputs. The computation is then executed on FP8 Tensor Cores. This INT4 and FP8 hybrid W4A8 variant delivers near-lossless accuracy, massive memory-bandwidth reductions, and excellent throughput on modern GPUs.

INT4 加 FP8 的混合方案兼具两者之长。在这种混合方案中，4 位整数权重被解包并重新解释为 FP8 输入，然后在 FP8 Tensor Core 上执行计算。这种 INT4 加 FP8 的混合 W4A8 变体，在现代 GPU 上带来近乎无损的准确率、巨大的内存带宽削减以及出色的吞吐。

For modern GPUs, two distinct low-precision paths are common in production. First, NVFP4 workflows use FP4 blocks with microscaling managed by the Transformer Engine or TensorRT. Second, W4A8 workflows use INT4 weights with FP8 or INT8 activations executed using fused dequantization in TensorRT-LLM. Choose the path based on your model calibration and accuracy targets.

对于现代 GPU，生产中常见两条截然不同的低精度路径。第一，NVFP4 工作流使用带 microscaling 的 FP4 块，由 Transformer Engine 或 TensorRT 管理。第二，W4A8 工作流使用 INT4 权重搭配 FP8 或 INT8 激活，并在 TensorRT-LLM 中用融合反量化来执行。请根据你的模型校准和准确率目标来选择路径。

In summary, quantization is one of the best ways to reduce inference cost. By cutting model size 2× or more, you effectively double the throughput per GPU. The techniques mentioned previously, including GPTQ, AWQ, and SmoothQuant, help you achieve these gains with minimal accuracy loss (e.g., often < 1% decrease).

总而言之，量化是降低推理成本的最佳手段之一。把模型体积削减 2× 或更多，就相当于把每块 GPU 的吞吐翻倍。前面提到的技术——包括 GPTQ、AWQ 和 SmoothQuant——能帮你在极小的准确率损失（例如通常 < 1% 的下降）下实现这些收益。

> It’s recommended that you start with 8-bit weights and then evaluate 4-bit weight-only quantization for additional gains. Then you can move to W4A8 only if you need maximum optimization and can spend time on calibration.

> 建议你从 8 位权重开始，然后评估 4 位仅权重量化以获得额外收益。只有当你需要极致优化、并且能投入时间做校准时，才进一步转向 W4A8。

The next step is to eliminate any conversion overhead. By fusing quantize-dequantize operations directly into the compute kernels, you preserve the gains from using quantization.

下一步是消除任何转换开销。通过把量化-反量化（quantize/dequantize）操作直接融合进计算核函数，你就能保住量化带来的收益。

### Fusing Quantization-Dequantization Steps into the Execution Graph

### 把量化-反量化步骤融合进执行图

Modern inference engines use the TE to perform weight-packing and provide high-efficiency mixed-precision math computations without the explicit calibration overhead of using INT8. These inference engines typically implement CUDA/Triton kernels to manually fuse “quant-dequant” steps into the execution graph when needed.

现代推理引擎使用 TE 来执行权重打包，并提供高效的混合精度数学运算，而无需使用 INT8 那样的显式校准开销。这些推理引擎通常实现 CUDA/Triton 核函数，在需要时手动把“量化-反量化”步骤融合进执行图。

These quant-dequant steps should be fused into the graph whenever separate Quantize/Dequantize kernels would introduce too many extra launches and negate the latency and bandwidth benefits of reduced-precision math. This is especially useful for inference backends that lack native fused INT8 support.

每当独立的量化/反量化核函数会引入过多额外的启动、从而抵消降精度数学在延迟和带宽上的收益时，就应把这些量化-反量化步骤融合进图中。这对于缺乏原生融合 INT8 支持的推理后端尤其有用。

Fusing quant-dequant into the main compute kernels is especially valuable for high-throughput inference pipelines to restore performance by removing bottlenecks caused by the conversion. Next, let’s explore various application-level optimizations supported by modern AI inference serving engines.

把量化-反量化融合进主计算核函数，对高吞吐推理流水线尤其有价值：它能移除由转换造成的瓶颈，从而恢复性能。接下来，我们探讨现代 AI 推理服务引擎所支持的各种应用级优化。

## Application-Level Optimizations

## 应用级优化

Beyond the core model and system optimizations, there are several higher-level techniques that can significantly improve the performance and user experience of an LLM service. These methods operate at the application or inference-serving layer and improve how prompts are constructed and cached, how conversation history is preserved, how requests are routed to different models, etc.

在核心的模型和系统优化之外，还有若干更高层的技术能够显著提升 LLM 服务的性能和用户体验。这些方法作用于应用层或推理服务层，改进提示的构造与缓存方式、对话历史的保存方式、请求路由到不同模型的方式等等。

These optimizations don’t involve modifying model weights or deploying new hardware. They are algorithmic and system-level improvements at the application layer. They can produce significant gains in efficiency and usability for “free” essentially, as they incur very little cost.

这些优化不涉及修改模型权重或部署新硬件。它们是应用层上算法与系统层面的改进。由于代价极低，它们几乎可以“免费”带来效率和可用性上的显著提升。

In this section, we discuss a few such optimizations, including prompt compression, prefix caching and deduplication, fallback model routing, and streaming outputs. These strategies improve performance by reducing input size, avoiding redundant computations, providing graceful handling of different request types, and improving perceived latency for end users.

在本节中，我们讨论其中几种优化，包括提示压缩、前缀缓存与去重、回退模型路由以及流式输出。这些策略通过减小输入规模、避免冗余计算、优雅地处理不同类型的请求，以及改善终端用户的感知延迟来提升性能。

### Prompt Compression

### 提示压缩

Users often send very long prompts or conversation histories to an LLM. However, not all of this context may be necessary for producing an adequate response. Prompt compression refers to a set of techniques that shorten or simplify the input prompt without losing relevant information.

用户经常向 LLM 发送很长的提示或对话历史。然而，并非所有这些上下文都是产出合格回答所必需的。提示压缩指的是一组在不丢失相关信息的前提下缩短或简化输入提示的技术。

Some system prompts or system-injected instructions are quite verbose (“You are ChatGPT, a friendly assistant designed to help users…”). Large system prompts will occupy a lot of space in the input context—and for every request.

有些系统提示或系统注入的指令相当冗长（“你是 ChatGPT，一个旨在帮助用户的友好助手……”）。庞大的系统提示会在输入上下文中占用大量空间——而且是每个请求都如此。

Prompt compression reduces the amount of work that the model needs to do. It directly translates to cost savings because a shorter input means fewer GPU computations.

提示压缩减少了模型需要完成的工作量。它直接转化为成本节省，因为更短的输入意味着更少的 GPU 计算。

> Remember that the attention mechanism within a transformer-based LLM model is O(n²) time complexity, where *n* is the size of the input measured in number of tokens.

> 请记住，基于 transformer 的 LLM 模型中的注意力机制的时间复杂度为 O(n²)，其中 *n* 是以 token 数量衡量的输入大小。

One simple form of prompt compression is removing redundant or irrelevant text. Consider a user prompt that contains a large chunk of text that is not relevant to answering their query. This can include a copy-pasted article in which the question relates to only one paragraph of the text. In this case, an upstream component could summarize or extract the relevant parts before feeding it to the LLM.

提示压缩的一种简单形式是移除冗余或无关的文本。设想一个用户提示，其中包含一大段与回答其查询无关的文本。这可能是一篇复制粘贴进来的文章，而问题只与其中一段相关。在这种情况下，上游组件可以先总结或抽取相关部分，再把它喂给 LLM。

Another form of prompt compression is dialogue summarization or truncation. For long chat histories, for instance, early parts of the conversation might no longer be relevant. The system can intelligently trim the conversation by summarizing older parts into a condensed form.

提示压缩的另一种形式是对话摘要或截断。例如，对于很长的聊天历史，对话早期的部分可能不再相关。系统可以智能地裁剪对话，把较早的部分总结成精简的形式。

> Many production chatbots like ChatGPT do prompt compression automatically to stay within their context-length limits—and to speed up overall processing. This is a must-have for long-running conversations.

> 像 ChatGPT 这样的许多生产级聊天机器人会自动进行提示压缩，以保持在其上下文长度限制之内，并加快整体处理速度。对于长时间运行的对话，这是必备能力。

To do this, you can run a small LLM—or a traditional rule-based system—to summarize the oldest parts of the conversation into a brief summary. Often the policy is something like: when the conversation exceeds 75% of the maximum context length, summarize the oldest 25% of messages. This summary would then be prepended to the more recent parts of the conversation. And this becomes the new prompt going forward.

要做到这一点，你可以运行一个小型 LLM——或一个传统的基于规则的系统——把对话中最早的部分总结成一段简短的摘要。常见的策略大致是这样：当对话超过最大上下文长度的 75% 时，就把最早的 25% 消息进行总结。随后，这段摘要会被前置到对话中较新的部分之前，成为后续使用的新提示。

When implementing prompt compression, make sure that no critical facts are lost. One approach is to have the model compress the prompt (e.g., generate a summary) and then verify the compressed prompt by asking the model questions about it. This way, your algorithm checks if key information was retained in the compressed version of the prompt.

在实现提示压缩时，务必确保没有丢失任何关键事实。一种方法是让模型压缩提示（例如生成一段摘要），然后针对压缩后的提示向模型提问，以此验证压缩结果。这样，你的算法就能检查关键信息是否在压缩版本的提示中得以保留。

This prevents excessive latency on very long chats and keeps the model focused on the most recent parts of the conversation. By truncating earlier turns, you reduce the effective prompt length *N*, which reduces prefill self-attention work from N(N + 1) ÷ 2 (roughly O(N2)) to a smaller triangular cost for the shorter window. So while the cost approaches quadratic in the window size, the smaller *N* makes a significant difference for very long conversations.

这可以避免在超长对话中产生过高的延迟，并让模型专注于对话中最近的部分。通过截断较早的轮次，你减小了有效提示长度 *N*，从而把 prefill 自注意力的工作量从 N(N + 1) ÷ 2（大致为 O(N2)）降低为更短窗口下更小的三角形代价。因此，尽管代价随窗口大小接近二次增长，但更小的 *N* 对于超长对话而言会带来显著差异。

A good summary can improve the response by filtering out irrelevant details. However, a bad summary can omit important parts of the input that the user really cares about. Summarizing is best when the system is confident—or if the conversation is clearly digressing.

好的摘要能通过滤除无关细节来改善回答。然而，糟糕的摘要可能会遗漏用户真正关心的重要输入内容。当系统有把握时——或当对话明显跑题时——进行摘要最为合适。

### Prompt Cleansing

### 提示清洗

Another technique is prompt cleansing. This is used to improve input formatting and tokenization. It helps reduce the amount of unnecessary whitespace or markup sent to the model. Tokenizers process every character, including spaces and newlines; the less unnecessary tokens we send, the better.

另一种技术是提示清洗（prompt cleansing）。它用于改善输入的格式与分词（tokenization）。它有助于减少发送给模型的不必要空白或标记。分词器会处理每一个字符，包括空格和换行符；我们发送的无用 token 越少越好。

While tokenizers like OpenAI’s tiktoken are very efficient with whitespace, large prompts with lots of markdown and HTML can bloat the token count. Simple preprocessing like removing HTML tags and converting fancy quotes to plain text can avoid odd tokenizations and reduce the number of tokens.

虽然像 OpenAI 的 tiktoken 这样的分词器处理空白非常高效，但包含大量 markdown 和 HTML 的大型提示仍可能使 token 数量膨胀。一些简单的预处理——例如移除 HTML 标签、把花式引号转换为纯文本——可以避免奇怪的分词结果，并减少 token 数量。

For instance, we can potentially reduce the amount of inference engine computations by not sending blank lines and repetitive punctuations that don’t impact the meaning of the input. This might save only a few tokens here and there, but it adds up across thousands of requests—especially for long input prompts.

例如，我们可以不发送那些不影响输入含义的空行和重复标点，从而有望减少推理引擎的计算量。这一次可能只省下寥寥几个 token，但在成千上万次请求中累积起来就相当可观——尤其是对于很长的输入提示。

In some cases, we can compress prompts by using references instead of the full content. For instance, if a user’s prompt includes a long piece of text that our system has seen before—perhaps because they are referring to a document that we have previously stored—we could replace it with a reference like “file0”.

在某些情况下，我们可以用引用来代替完整内容，从而压缩提示。例如，如果用户的提示中包含一段我们系统此前见过的长文本——也许是因为他们引用了一份我们之前存储过的文档——我们就可以用类似"file0"这样的引用来替换它。

The model could then be fine-tuned to retrieve the actual content for that reference—or we can handle it using a retrieval system. This crosses into retrieval-augmented generation (RAG) territory, which we won’t cover any further here, but the key point is that we don’t always need to feed the raw content through the model if there are alternative ways.

随后，可以对模型进行微调，让它去检索该引用对应的实际内容——或者我们也可以用一套检索系统来处理。这已经进入检索增强生成（retrieval-augmented generation，RAG）的范畴，本文不再深入展开，但关键在于：如果存在替代方案，我们并不总是需要把原始内容喂给模型。

> Some inference engines support setting a system prompt once per session rather than sending it each time. This is a better solution than recompressing the system prompt for each request.

> 一些推理引擎支持在每个会话中只设置一次系统提示（system prompt），而不必每次都发送。相比为每个请求重新压缩系统提示，这是更好的方案。

We’ll discuss how to improve the efficiency of a large system prompt with prefix caching in the next section. But, for now, let’s use prompt compression to create a shorter and functionally equivalent set of instructions. For instance, the model can be trained to use special tokens or metadata to represent a long system prompt. This way, it doesn’t have to always process the long natural language version.

我们将在下一节讨论如何用前缀缓存来提升大型系统提示的效率。但就目前而言，先用提示压缩来创建一套更短、功能上等价的指令。例如，可以训练模型使用特殊 token 或元数据来表示一个很长的系统提示。这样，它就不必总是处理冗长的自然语言版本。

Consider a 200-token system prompt full of text-based rules that can be replaced with 10 special tokens that trigger the same text-based rules expressed in 200 tokens of natural language. This requires that the model be trained to parse the metadata, derive the rules, and follow them.

设想一个 200-token 的系统提示，里面满是基于文本的规则，而它可以被 10 个特殊 token 所替换——这些特殊 token 会触发用 200 token 自然语言表达的同一批基于文本的规则。这要求模型经过训练，能够解析元数据、推导出规则并遵循它们。

There’s ongoing research into “config” tokens that tell the model to load a certain preconfigured set of instructions for a given config token. Think of this like assigning a unique ID for a given prompt prefix (e.g., system prompt). It’s common to fine-tune a model to recognize tokens like <POLICY_A> as a stand-in replacement for a long, 500-word policy, for example.

目前有关于"config" token 的持续研究：这类 token 告诉模型为某个给定的 config token 加载一套预先配置好的指令。可以把它理解为给某个提示前缀（例如系统提示）分配一个唯一 ID。举例来说，常见做法是对模型进行微调，让它识别像 <POLICY_A> 这样的 token，作为一段长达 500 词的策略的替身。

> The Hugging Face Transformers library implements the popular CTRL approach. This is a good place to start working with config tokens for prompt trimming.

> Hugging Face Transformers 库实现了流行的 CTRL 方法。如果你想开始用 config token 做提示裁剪，这是一个很好的起点。

Using special tokens and metadata to replace long system prompts is more of a training consideration. But it can provide a massive inference speedup if it reduces the size of the prompt and decreases prefill overhead. At scale, this directly translates to less compute needed per query—and more cost savings.

用特殊 token 和元数据来替换长系统提示，更多是训练层面的考量。但如果它能减小提示规模、降低 prefill 开销，就能带来巨大的推理提速。在大规模场景下，这直接意味着每次查询所需的计算更少——以及更多的成本节省。

### Prefix Caching

### 前缀缓存

Often multiple requests to an LLM share a common *prefix* in their input, as many queries might start with the same system prompt. Instead of recomputing the model’s output for the same prefix each time, you can compute it once and reuse the same KV cache for subsequent requests.

对 LLM 的多个请求，其输入中往往共享一个共同的*前缀*，因为许多查询可能以相同的系统提示开头。与其每次都为相同的前缀重新计算模型输出，你可以只计算一次，然后在后续请求中复用同一份 KV 缓存。

This technique is known as *prefix caching*, sometimes called prefix memoization. With prefix caching, the transformer’s state, including the keys and values in the attention layers, is stored and reused when a request with the same prefix appears again.

这项技术被称为*前缀缓存*，有时也叫前缀记忆化（prefix memoization）。在前缀缓存中，Transformer 的状态（包括注意力层中的键和值）会被存储下来，并在再次出现具有相同前缀的请求时复用。

> Prefix caching can turn what would normally be O(N × L) total work (N requests of length L) into O(N + L) work by reusing the cached prefix computations for repeated parts.

> 通过复用缓存的前缀计算来处理重复部分，前缀缓存可以把原本 O(N × L) 的总工作量（N 个长度为 L 的请求）降为 O(N + L)。

vLLM implements prefix caching (enable_prefix_caching=True) to avoid recomputation. It first identifies if an incoming prompt’s first *N* number of tokens match a prefix that’s already in the cache from a previous request—or earlier in the same session. If so, vLLM avoids an expensive attention recomputation for those *N* tokens—and just copies data from the cached KV into the new context. Make sure you have prefix caching enabled—and with a sufficient memory allocation.

vLLM 实现了前缀缓存（enable_prefix_caching=True）以避免重复计算。它首先判断传入提示的前 *N* 个 token 是否与缓存中已有的某个前缀匹配——该前缀可能来自之前的某个请求，或来自同一会话中较早的部分。如果匹配，vLLM 就会避免对这 *N* 个 token 做昂贵的注意力重算——而只是把数据从缓存的 KV 复制到新的上下文中。请确保你启用了前缀缓存——并分配了足够的内存。

An inference engine like vLLM can automatically group incoming queries into microbatches to achieve high GPU utilization, amortize overheads across many requests, and keep latency low using continuous batching, for instance. Make sure that if you’re not using an inference engine like vLLM directly, you look for ways to use prefix-sharing in your stack.

像 vLLM 这样的推理引擎可以自动把传入查询归入微批，以实现高 GPU 利用率、在众多请求间摊薄开销，并借助连续批处理等手段把延迟维持在较低水平。请注意：如果你没有直接使用 vLLM 这类推理引擎，也要在自己的技术栈中设法利用前缀共享。

Prefix caching can speed up workloads with repeated prefixes. Consider a scenario in which we want to ask 10 separate questions from a long document. Each would include the document in the prompt followed by one question, such as, “[Long document text] Question 1: …”, “[Long document text] Question 2: …”, etc.

前缀缓存可以加速带有重复前缀的工作负载。设想这样一个场景：我们想针对一份长文档提出 10 个各自独立的问题。每个提示都会先包含该文档，然后跟上一个问题，例如"[Long document text] Question 1: …"、"[Long document text] Question 2: …"等等。

The document text is the same across all 10 prompts. Normally, the model would need to reprocess the KV entries for the whole document for each question. This would lead to 10× redundant work as the model will re-encode the long document 10 times. With prefix caching, it encodes it once, and each subsequent question incurs the compute only for the smaller question suffix.

这份文档文本在全部 10 个提示中都是相同的。通常，模型需要为每个问题都重新处理整份文档的 KV 条目。这会导致 10× 的冗余工作，因为模型要把这份长文档重新编码 10 次。有了前缀缓存，它只编码一次，而后续每个问题只需为更小的问题后缀付出计算代价。

Specifically, with prefix caching, the first query would compute the transformer states for the document portion, and for subsequent queries, the model can jump straight to processing the “Question 1: …”, “Question 2: …” parts since the document’s content matches a prefix found in the KV cache.

具体来说，有了前缀缓存，第一次查询会为文档部分计算 Transformer 状态；而对于后续查询，由于文档内容匹配 KV 缓存中已有的一个前缀，模型可以直接跳到处理"Question 1: …"、"Question 2: …"这些部分。

With prefix caching, your inference engine can produce near-linear speedups. For instance, if you ask 10 separate questions on one document, these will be processed roughly 10× faster with prefix caching than without because the long document’s attention is calculated only once instead of 10 separate times.

有了前缀缓存，你的推理引擎可以获得接近线性的加速。例如，如果你针对同一份文档提出 10 个各自独立的问题，那么在使用前缀缓存时，它们的处理速度大约会比不使用时快 10×，因为长文档的注意力只计算一次，而不是分开计算 10 次。

Another application of prefix caching is chat sessions since the conversation history is a common prefix for each new “turn” in a “multiturn” chat session. When the end user sends a new message, the earlier messages form a prefix that keeps growing.

前缀缓存的另一个应用是聊天会话，因为在"多轮"聊天会话中，对话历史对每一个新的"轮次"来说都是一个共同前缀。当终端用户发送一条新消息时，先前的消息就构成了一个不断增长的前缀。

Optimized inference systems keep the KV cache from the last turn and compute KV only for the new user message. This is what most chat models like ChatGPT are doing if you don’t reset the conversation in between turns.

经过优化的推理系统会保留上一轮的 KV 缓存，只为新的用户消息计算 KV。如果你在轮次之间不重置对话，像 ChatGPT 这样的大多数聊天模型正是这么做的。

> The benefits of prefix caching are most noticeable in interactive settings—especially if users follow up quickly in the same context/session. In this case, the second answer will come much faster than the first since the prefix (conversation history) is already cached.

> 前缀缓存的收益在交互式场景中最为明显——尤其是当用户在同一上下文/会话中快速追问时。在这种情况下，由于前缀（对话历史）已被缓存，第二个回答会比第一个来得快得多。

In addition to supporting state in their ChatGPT consumer product, OpenAI supports stateful conversations in their API as well, which uses a form of prefix caching—among many other optimizations—behind the scenes. For stateless API invocations, prefix caching can achieve a similar effect if the client passes the conversation history. In this case, the server can retrieve the precomputed hidden state from the prefix cache—up to the last turn. It will then continue from there with the new user message.

除了在其 ChatGPT 消费者产品中支持状态外，OpenAI 在其 API 中同样支持有状态的对话——这在幕后使用了某种形式的前缀缓存，以及许多其他优化。对于无状态的 API 调用，如果客户端传入对话历史，前缀缓存也能达到类似效果。在这种情况下，服务器可以从前缀缓存中取回预先计算好的隐藏状态——直到上一轮为止。然后它会从那里开始，接着处理新的用户消息。

To implement this yourself, you can hash the conversation histories. This will give you a consistent key to look up in the cache. However, you will have to manage memory carefully, as storing entire KV caches for many conversations can use GPU memory very quickly. vLLM’s paging helps by paging inactive caches to CPU memory—or disk—and paging them back into GPU memory on a cache hit.

要自己实现这一点，你可以对对话历史做哈希。这会为你提供一个一致的键，用于在缓存中查找。不过，你必须谨慎地管理内存，因为为许多对话存储完整的 KV 缓存会非常快地耗尽 GPU 内存。vLLM 的分页机制会有所帮助：它把不活跃的缓存分页到 CPU 内存——或磁盘——并在缓存命中时把它们分页回 GPU 内存。

You can also deduplicate portions of a prompt across multiple requests—or even within a single request. Here, deduplication refers to combining identical subprompts to compute their transformer states just once. For instance, if two users send the exact same prompt prefix at nearly the same time, the system could merge the two prompts and process them only once.

你还可以在多个请求之间——甚至在单个请求内部——对提示的部分内容做去重。这里的去重指的是把相同的子提示合并起来，只计算一次它们的 Transformer 状态。例如，如果两个用户在几乎同一时刻发送了完全相同的提示前缀，系统就可以把这两个提示合并，只处理一次。

This can also happen within a single request if the input has repeated sequences. This is more common in training, which uses techniques like memoized decoding to deduplicate repetitive text. In inference, it’s rare to get long, repeated input sequences in the same request, but it is possible.

如果输入中含有重复序列，这种情况也可能发生在单个请求内部。这在训练中更常见，训练会使用记忆化解码（memoized decoding）之类的技术来对重复文本去重。在推理中，同一个请求里出现很长的重复输入序列的情况很少见，但也并非不可能。

> If your workload has many repeat queries on the same prefixes, you should allocate more memory to the cache to maximize hits. Conversely, if prefixes are rarely reused, you should use a smaller cache —or even disable prefix caching entirely. Always measure prefix hit rates in production and tune accordingly.

> 如果你的工作负载中有许多针对相同前缀的重复查询，你应该为缓存分配更多内存以最大化命中率。反之，如果前缀很少被复用，你应该使用更小的缓存——甚至完全禁用前缀缓存。请始终在生产环境中测量前缀命中率，并据此调优。

The prefix cache is typically implemented as a token-sequence *trie* (pronounced “try”). A trie, often called a *prefix tree*, is a tree-based data structure in which each edge represents a single token and each node encodes the sequence of tokens from the root to that point.

前缀缓存通常实现为一棵 token 序列的*字典树（trie）*（读作"try"）。字典树也常被称为*前缀树（prefix tree）*，它是一种基于树的数据结构，其中每条边代表单个 token，每个节点则编码了从根到该处的 token 序列。

In a token-sequence trie, every observed prompt prefix is stored as its own path of tokens. This enables fast lookups of shared prefixes. When a new request arrives, the inference engine traverses the trie—token by token—from the root until it can no longer match the next token. The sequence of matching tokens lands at the node that completes the longest-cached prefix, as shown in Figure 16-8 for an example system prompt, “You are ChatGPT, a friendly assistant designed to help users…”

在 token 序列字典树中，每一个观察到的提示前缀都被存储为它自己的一条 token 路径。这使得共享前缀的查找非常快速。当新请求到来时，推理引擎从根开始——逐个 token 地——遍历字典树，直到无法再匹配下一个 token 为止。匹配到的 token 序列会停在那个完成了最长缓存前缀的节点上，如图 16-8 所示，其示例系统提示为"You are ChatGPT, a friendly assistant designed to help users…"。

![Figure 16-8. Prefix cache implemented as a trie data structure](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-8.png)

![图 16-8. 以字典树数据结构实现的前缀缓存](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-8.png)

At this point, the system reuses the shared KV cache data (clones the pointers) and resumes decoding but only for the remaining tokens in the sequence. This avoids redundant self-attention computations for the already-cached prefix since the attention scores for this prefix have already been computed.

此时，系统复用共享的 KV 缓存数据（克隆指针），并恢复解码，但只针对序列中剩余的 token。这避免了对已缓存前缀的冗余自注意力计算，因为该前缀的注意力分数此前已经算过。

SGLang’s RadixAttention uses a compressed trie, or radix tree, over entire token sequences. This type of prefix cache collapses token sequences into single edges in the tree to save space. Each node in the tree points to a KV-cache tensor stored as a contiguous GPU page that holds the prefix’s KV cache.

SGLang 的 RadixAttention 在整段 token 序列上使用一棵压缩字典树，即基数树（radix tree）。这类前缀缓存把 token 序列折叠为树中的单条边，以节省空间。树中的每个节点都指向一个 KV 缓存张量，该张量以一块连续的 GPU 页存储，保存着该前缀的 KV 缓存。

The RadixTree data structure is space efficient and provides rapid prefix searches, efficient insertions, and LRU-style eviction. Here is pseudocode that shows a KV cache lookup using a radix tree during token generation:

RadixTree 数据结构空间高效，能提供快速的前缀搜索、高效的插入，以及 LRU（最近最少使用，least recently used）风格的逐出。下面这段伪代码展示了在 token 生成过程中，如何使用基数树进行 KV 缓存查找：

```
# Simplified RadixAttention KV cache example
# radix_attention_example.py
radix: RadixTree = RadixTree()  # holds edge labels + node.cache pointers
def generate_with_radix(prompt_tokens: List[int]):
    # 1) Find longest cached prefix
    node, prefix_len = radix.longest_prefix(prompt_tokens)
    # shallow-clone the KV cache for that prefix
    model_state = ModelState.from_cache(node.cache)  # refcount bump
    # 2) Process remaining prompt suffix
    for token in prompt_tokens[prefix_len:]:
        model_state = model.forward(token, state=model_state)
    # 3) As we go, insert or split edges in the radix tree
        matched = prompt_tokens[:prefix_len + 1]
        # insert returns the node for this full prefix
        node = radix.insert(matched, cache=model_state.kv_cache)
        prefix_len += 1
    # 4) Now generate new tokens autoregressively
    output_tokens = []
    while not model_state.is_finished():
        token, model_state = model.generate_next(model_state)
        output_tokens.append(token)
        # cache each generated prefix as well
        matched = prompt_tokens + output_tokens
        node = radix.insert(matched, cache=model_state.kv_cache)
    return output_tokens
```

```
# Simplified RadixAttention KV cache example
# radix_attention_example.py
radix: RadixTree = RadixTree()  # holds edge labels + node.cache pointers
def generate_with_radix(prompt_tokens: List[int]):
    # 1) Find longest cached prefix
    node, prefix_len = radix.longest_prefix(prompt_tokens)
    # shallow-clone the KV cache for that prefix
    model_state = ModelState.from_cache(node.cache)  # refcount bump
    # 2) Process remaining prompt suffix
    for token in prompt_tokens[prefix_len:]:
        model_state = model.forward(token, state=model_state)
    # 3) As we go, insert or split edges in the radix tree
        matched = prompt_tokens[:prefix_len + 1]
        # insert returns the node for this full prefix
        node = radix.insert(matched, cache=model_state.kv_cache)
        prefix_len += 1
    # 4) Now generate new tokens autoregressively
    output_tokens = []
    while not model_state.is_finished():
        token, model_state = model.generate_next(model_state)
        output_tokens.append(token)
        # cache each generated prefix as well
        matched = prompt_tokens + output_tokens
        node = radix.insert(matched, cache=model_state.kv_cache)
    return output_tokens
```

When generation begins, the engine calls radix.longest_prefix(prompt_tokens) to walk multitoken edges down the radix tree until it reaches the deepest node matching the prompt’s longest cached prefix. It then does a lightweight clone of that node’s KV cache page using ModelState.from_cache(node.cache). This seeds model_state without recomputing any self-attention for the cached prefix.

当生成开始时，引擎调用 radix.longest_prefix(prompt_tokens)，沿基数树向下走过多 token 的边，直到到达匹配该提示最长缓存前缀的最深节点。随后，它使用 ModelState.from_cache(node.cache) 对该节点的 KV 缓存页做一次轻量克隆。这为 model_state 提供了初值，而无需为已缓存前缀重新计算任何自注意力。

Next, it processes only the unseen suffix of the prompt—token by token—and updates the radix tree on the fly. For each new token, it calls radix.insert(...), splits edges, or creates new ones as needed. It then stores the intermediate model_state.kv_cache at each new node.

接下来，它只处理提示中未见过的后缀——逐个 token——并即时更新基数树。对每个新 token，它调用 radix.insert(...)，按需拆分边或创建新边。然后，它在每个新节点处存储中间的 model_state.kv_cache。

Once the entire prompt has been consumed, the loop switches to the autoregressive decoding phase, which generates new tokens with model.generate_next(model_state). Similarly, this inserts each generated prefix into the radix tree. This approach minimizes redundant computations, uses space-efficient storage of KV pages, and performs fast prefix lookups—all while supporting incremental cache updates.

一旦整个提示被消费完毕，循环便切换到自回归解码（autoregressive decoding）阶段，用 model.generate_next(model_state) 生成新的 token。类似地，这会把每个生成的前缀插入基数树。这种方法把冗余计算降到最低，以空间高效的方式存储 KV 页，并执行快速的前缀查找——同时全程支持增量式的缓存更新。

SGLang’s KV cache design automatically captures all common reuse patterns, including multiturn chats, few-shot examples, and branching logic. At the same time, it makes sure that shared prefixes are fetched as large, coalesced memory chunks for efficient GPU access.

SGLang 的 KV 缓存设计能自动捕捉所有常见的复用模式，包括多轮聊天、少样本（few-shot）示例和分支逻辑。与此同时，它确保共享前缀会以大块、合并后的内存块被取出，以实现高效的 GPU 访问。

Knowing when to invalidate the cache is always a challenge. If the cache memory is needed for other things, you may need to evict some prefixes. Caching systems support different policies, such as “least recently used” (LRU), in which the least recently used prefix gets evicted first. SGLang’s RadixAttention, for instance, will lazily evict the least recently used radix-tree leaf when GPU memory is scarce, as shown in Figure 16-9.

知道何时让缓存失效始终是一个难题。如果缓存内存需要挪作他用，你可能需要逐出一些前缀。缓存系统支持不同的策略，例如"最近最少使用"（LRU），即最久未被使用的前缀最先被逐出。举例来说，当 GPU 内存紧张时，SGLang 的 RadixAttention 会惰性地逐出最久未使用的基数树叶子节点，如图 16-9 所示。

This is a prefix cache tree structure—and LRU eviction policy—used by SGLang for multiple incoming requests. This example is based on an awesome SGLang blog post from LMSys.

这是一棵前缀缓存树结构——以及 LRU 逐出策略——由 SGLang 用于处理多个传入请求。本示例基于 LMSys 发布的一篇很棒的 SGLang 博客文章。

![Figure 16-9. Prefix cache evolution for multiple requests (source: https://oreil.ly/7LBoC)](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-9.png)

![图 16-9. 多个请求下前缀缓存的演化（来源：https://oreil.ly/7LBoC）](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-9.png)

Here, there are two chat sessions and multiple queries across those chat sessions. The label of each is a sequence of tokens (e.g., substring). Green nodes represent new nodes in the tree. Blue nodes are cached nodes that are currently being accessed. Red nodes have been evicted. Here is the breakdown of each step:

这里有两个聊天会话，以及跨这些会话的多个查询。每个节点的标签是一段 token 序列（例如子串）。绿色节点表示树中的新节点。蓝色节点是当前正在被访问的已缓存节点。红色节点则已被逐出。下面是各步骤的拆解：

1. The initial empty radix tree is empty.

2. The server processes the incoming prompt “Hello!” and responds with the LLM-generated “Hi!” With this simple response, many tokens are added to the tree as a single edge. This edge is linked to a new node in green. Specifically, the system prompt, “You are a helpful assistant”; the user message, “Hello!”; and the LLM response, “Hi!” are consolidated.

3. The server receives a new prompt. This is the first turn of the multiturn conversation. The server successfully looks up the prompt prefix in the tree and reuses its KV cache data. A new turn is added to the tree as a new green node.

4. A new chat session begins, and node *b* from step 3 is split into two separate nodes. This lets the two chat sessions share the system prompt.

5. The second chat session from step 4 continues. Memory is limited, however, so node *c* from step 4 must be evicted, and it’s shown in red. A new turn is appended after node *d* in step 4.

6. The server receives a new prompt (query). After processing, the server inserts the prompt into the tree. This requires the root node to split because this new prompt does not share any prefix with existing prompts/nodes.

7. The server receives a batch with more prompts (queries). These prompts share prefixes with the prompt from step 6. As such, the system splits node *e* from step 6 and shares the prefix.

8. The server receives a new message from the conversation in step 3 (the first chat session). In this case, it evicts all nodes from the second chat session in step 5 (e.g., nodes *g* and *h*). This is because they are the least recently used (LRU) at that moment.

9. The server receives a message requesting more answers for the query in node *j* from step 8. Due to memory limitations, the system is required to evict nodes *i*, *k*, and *l* from step 8.

1. 最初的空基数树是空的。

2. 服务器处理传入的提示"Hello!"，并用 LLM 生成的"Hi!"作出回应。在这个简单的回应中，许多 token 作为单条边被加入树中。这条边连接到一个绿色的新节点。具体来说，系统提示"You are a helpful assistant"、用户消息"Hello!"、以及 LLM 回应"Hi!"被合并到了一起。

3. 服务器收到一个新提示。这是多轮对话的第一轮。服务器成功地在树中查到了该提示前缀，并复用其 KV 缓存数据。一个新的轮次作为一个绿色新节点被加入树中。

4. 一个新的聊天会话开始，第 3 步中的节点 *b* 被拆分为两个独立的节点。这让这两个聊天会话得以共享系统提示。

5. 第 4 步中的第二个聊天会话继续进行。然而，由于内存有限，第 4 步中的节点 *c* 必须被逐出，它以红色显示。一个新的轮次被追加在第 4 步中的节点 *d* 之后。

6. 服务器收到一个新提示（查询）。处理完成后，服务器把该提示插入树中。由于这个新提示与已有的提示/节点没有任何共享前缀，这需要对根节点进行拆分。

7. 服务器收到一个包含更多提示（查询）的批。这些提示与第 6 步的提示共享前缀。因此，系统拆分了第 6 步中的节点 *e*，并共享该前缀。

8. 服务器收到来自第 3 步对话（第一个聊天会话）的一条新消息。在这种情况下，它逐出了第 5 步中第二个聊天会话的所有节点（例如节点 *g* 和 *h*）。这是因为它们在那一刻是最久未被使用的。

9. 服务器收到一条消息，要求为第 8 步中节点 *j* 的查询提供更多答案。由于内存限制，系统需要逐出第 8 步中的节点 *i*、*k* 和 *l*。

This example demonstrates how prefixes are shared between multiple requests. In addition, it shows how an LRU cache-eviction policy works in the context of prefix caching. Next, let’s turn to using a cascading model deployment pattern to better utilize GPU resources.

这个示例演示了前缀如何在多个请求之间共享。此外，它还展示了 LRU 缓存逐出策略在前缀缓存场景下是如何工作的。接下来，让我们转向使用级联式的模型部署模式，以更好地利用 GPU 资源。

### Model Cascading and Tiered Model Deployment

### 模型级联与分层模型部署

Not all queries require the power (and cost) of the largest, state-of-the-art models. Some user requests are simple enough to be answered by a much smaller, faster, and cheaper model. Choosing the right model is called *model cascading* or *fallback model* *routing*.

并非所有查询都需要动用最大、最先进模型的能力（和成本）。有些用户请求足够简单，用一个小得多、更快、也更便宜的模型就能回答。选择合适的模型这一做法，被称为*模型级联（model cascading）*或*回退模型路由（fallback model routing）*。

To implement model cascading, you can use tiered model deployment. This is an approach that maintains multiple models of different sizes and abilities. For each incoming request, we dynamically choose which model to use based on query complexity, required precision, or current system load.

要实现模型级联，你可以采用分层模型部署（tiered model deployment）。这是一种维护多个不同规模、不同能力模型的方法。对每个传入请求，我们会根据查询复杂度、所需精度或当前系统负载，动态地选择使用哪个模型。

Consider a question-answer (QA) inference deployment with both a large 700-billion-parameter state-of-the-art model and a smaller 70-billion-parameter model that is faster and cheaper to host since it requires fewer GPU resources. If a user asks a very straightforward and factual question (as determined by a classifier or heuristic), the 70-billion-parameter model might handle it well.

设想一个问答（question-answer，QA）推理部署，其中既有一个 7000 亿参数的大型最先进模型，又有一个 700 亿参数的较小模型——后者因所需 GPU 资源更少，托管起来更快、也更便宜。如果用户问了一个非常直接、事实性的问题（由分类器或启发式规则判定），那么这个 700 亿参数的模型也许就能处理得很好。

For straightforward questions, you can route the query to the smaller model, which will respond in 50 ms instead of 500 ms with the 700-billion-parameter model. If the small model’s answer is deemed unsatisfactory due to low confidence (determined by the logits returned in the response), the system can route the question to the bigger model.

对于直接的问题，你可以把查询路由到较小的模型，它会在 50 ms 内响应，而 7000 亿参数的模型则需要 500 ms。如果小模型的回答因置信度过低（由响应中返回的 logits 判定）而被认为不令人满意，系统可以把该问题路由到更大的模型。

This two-stage approach can reduce overall average latency and compute usage—although individual requests may experience longer latencies if they’re routed to both models due to lack of confidence in the small model’s initial response. It’s widely believed that many commercial AI services, such as ChatGPT, use this approach of routing simpler queries to smaller, faster models to reduce cost and latency.

这种两阶段方法可以降低整体的平均延迟和计算用量——尽管个别请求如果因为对小模型初始回答缺乏信心而被同时路由到两个模型，可能会经历更长的延迟。人们普遍认为，许多商业 AI 服务（例如 ChatGPT）都采用这种做法：把较简单的查询路由到更小、更快的模型，以降低成本和延迟。

In practice, routing, say, 60% of queries to a 10× smaller model can cut your inference compute costs significantly—potentially a 5× overall cost reduction if designed and tuned properly. This is because the expensive model is used only when needed.

在实践中，比如说把 60% 的查询路由到一个小 10× 的模型，就能显著削减你的推理计算成本——如果设计和调优得当，整体成本有望降低 5×。这是因为昂贵的模型只在需要时才会被启用。

> Maintaining multiple models and a routing system is an engineering overhead, as you need to monitor not just one model’s performance but also the second model, the interplay between the two models, and the routing system. Pursue this only if your traffic volume and cost structure make it worthwhile.

> 维护多个模型和一套路由系统是一种工程开销，因为你不仅要监控一个模型的性能，还要监控第二个模型、两个模型之间的相互作用，以及路由系统本身。只有当你的流量规模和成本结构让这样做物有所值时，才去追求它。

Implementing model cascading requires a query classifier or heuristic mechanism to assist the model-routing decision. And this is a relatively difficult problem to solve correctly, as the mechanism often requires continuous adaptation, comprehensive heuristics, and constant router tuning. As such, many organizations still rely on heuristic and rule-based routers using offline analysis to update the decision criteria.

实现模型级联需要一个查询分类器或启发式机制来辅助模型路由决策。而这是一个相对难以正确解决的问题，因为这套机制往往需要持续适配、全面的启发式规则以及不断的路由器调优。因此，许多组织仍然依赖基于启发式和规则的路由器，并利用离线分析来更新决策标准。

One approach is to consider this a speculative problem, similar to speculative decoding described earlier. You can first run the smaller model in a draft mode and have it generate the answer with its confidence score. If confidence is high, just return the response. If not, call the bigger model.

一种思路是把它看作一个推测性问题，类似于前面描述过的推测解码。你可以先让较小的模型以草稿模式运行，让它连同置信度分数一起生成答案。如果置信度高，就直接返回该响应。如果不高，就调用更大的模型。

Make sure that if the big model is needed, the additional latency of the small model doesn’t make the user wait too long. In practice, you might run the small and large models in parallel when you suspect that the question is hard. This overlaps the small model’s latency with the large model. This seems wasteful and counterintuitive, but it can be beneficial for some latency-sensitive use cases to keep latency low—at the expense of additional compute.

务必确保：当需要大模型时，小模型带来的额外延迟不会让用户等太久。在实践中，当你怀疑问题较难时，可以让小模型和大模型并行运行。这样就把小模型的延迟与大模型重叠了起来。这看似浪费且有违直觉，但对某些延迟敏感的用例而言，以额外计算为代价来把延迟维持在较低水平，是有益的。

Another approach is to train a separate classifier for well-known user queries ahead of time. For instance, you can classify complexity and ask, “Does this question require the big model’s extensive knowledge or reasoning?” If not, use the small model.

另一种思路是提前为常见的用户查询训练一个专门的分类器。例如，你可以对复杂度进行分类，并追问："这个问题是否需要大模型广博的知识或推理能力？"如果不需要，就使用小模型。

It’s important to note that this classifier will itself need periodic retraining as user queries evolve over time. Also, it introduces another component that can degrade and fail—and therefore needs to be monitored. Make sure this model is lightweight, fast, and robust—otherwise it becomes a new bottleneck in your inference system.

需要注意的是，随着用户查询随时间演变，这个分类器本身也需要定期重新训练。此外，它引入了又一个可能退化和失效的组件——因此需要被监控。请确保这个模型轻量、快速且健壮——否则它会成为你推理系统中一个新的瓶颈。

It’s common to use a simple heuristic initially. For instance, if the user’s query is short and factual, such as “What’s the weather today?” or “Who is president of X?”, you can just use the smaller model. For longer prompts (e.g., > 50 tokens) or more creative queries that contain words like *explain*, *analyze*, or *elaborate*, you can just send it to the big LLM directly and bypass the smaller model.

起初常用一个简单的启发式规则。例如，如果用户的查询既短又偏事实性，比如"What's the weather today?"或"Who is president of X?"，你就可以直接使用较小的模型。而对于更长的提示（例如 > 50 token），或包含 *explain*、*analyze*、*elaborate* 之类词语的更具创造性的查询，你可以直接把它发送给大 LLM，绕过较小的模型。

You should log and tag every request with which model handled it (e.g., “small” versus “large”)—and whether it fell back. This lets you break down latency, accuracy, and fallback counts by model.

你应该记录并标注每个请求，标明它由哪个模型处理（例如"small"对"large"）——以及它是否发生了回退。这让你能够按模型来分解延迟、准确率和回退次数。

This tagging data also lets you monitor the dispatchers’ decisions. By counting the number of times when the small model’s answer was rejected, you can identify patterns that may lead to retraining the smaller model or improving the routers’ heuristics.

这些标注数据还让你能够监控调度器的决策。通过统计小模型回答被拒绝的次数，你可以识别出某些模式，从而促成对小模型的重新训练，或对路由器启发式规则的改进。

Conversely, if the small model handles many queries well, you can use this information to raise its confidence threshold. This will increase the small model’s usage, reduce response latency, and improve the end user experience.

反过来，如果小模型很好地处理了许多查询，你可以利用这一信息来提高它的置信度阈值。这会增加小模型的使用比例、降低响应延迟，并改善终端用户体验。

You can also route based on capacity. If the big model cluster is at maximum load, rather than queue requests and increase latency, we might temporarily route less critical requests to a smaller model, which might not be as good but still gives some answer quickly. This is a form of graceful degradation under heavy load. The user might notice a quality dip, but it’s better than timing out or waiting excessively.

你也可以基于容量来路由。如果大模型集群处于满负荷，与其让请求排队从而增加延迟，我们不如把不那么关键的请求临时路由到较小的模型——它也许没那么好，但仍能快速给出某种答案。这是一种重负载下的优雅降级（graceful degradation）。用户或许会察觉到质量下降，但这总比超时或漫长等待要好。

Many AI services do this intentionally. For example, during peak loads, free-tier users might silently get routed to a smaller model to free up the larger model for premium-tier (paid) users. This is a realistic trade-off when resources are constrained.

许多 AI 服务会有意这么做。例如，在高峰负载期间，免费层用户可能会被悄悄路由到较小的模型，以便把较大的模型腾出来给付费的高级层用户。当资源受限时，这是一个现实的权衡。

> We will cover more advanced dynamic and adaptive inference server capabilities in Chapters 17–19. These features depend heavily on the current system load, including available GPU memory, memory bandwidth utilization, KV cache utilization, etc.

> 我们将在第 17–19 章介绍更高级的动态与自适应推理服务器能力。这些特性在很大程度上取决于当前的系统负载，包括可用 GPU 内存、内存带宽利用率、KV 缓存利用率等。

Another form of model cascading is based on the content itself. For instance, if a user asks something that the LLM refuses to answer due to safety, some inference systems will fall back to a special model fine-tuned specifically to handle unsafe prompts—or even to a retrieval system.

模型级联的另一种形式基于内容本身。例如，如果用户问了某些 LLM 出于安全考虑而拒绝回答的问题，一些推理系统会回退到一个专门为处理不安全提示而微调的特殊模型——甚至回退到一套检索系统。

This is more about content policy than performance, but it’s still an important consideration for your model-routing logic. And often the fallback model is much smaller, so this actually ends up being a performance win since it handles the unsafe request quickly—and frees up the main model to handle the safe queries.

这更多关乎内容策略而非性能，但对你的模型路由逻辑而言，它仍是一个重要考量。而且回退模型往往小得多，因此这实际上最终成了一次性能上的胜利——它能快速处理不安全的请求，并把主模型腾出来处理安全的查询。

From a deployment perspective, multimodel setups require careful scaling since each model needs enough instances to handle its portion of traffic. Sometimes one model might become a bottleneck if many simple queries come in, for instance, and the small LLM is saturated while the big model sits idle.

从部署角度看，多模型的搭建需要谨慎地做扩缩容，因为每个模型都需要足够的实例来处理属于它的那部分流量。有时某个模型可能成为瓶颈——例如，当大量简单查询涌入、小 LLM 已被打满而大模型却处于闲置状态时。

You can autoscale and run more or fewer instances of the small model during certain times of day—or even dynamically shift GPUs from the large model pool to the small model pool as needed using container orchestration.

你可以进行自动扩缩容，在一天中的某些时段运行更多或更少的小模型实例——甚至可以借助容器编排，按需把 GPU 从大模型池动态调配到小模型池。

### Streaming Responses

### 流式响应

To improve the “perceived” performance of your inference system, you should stream the output tokens as they are generated. This is much better than forcing an end user to wait for the full completion to finish before they see a response. This way, users can begin reading and processing information as it arrives. This can make the effective interaction much faster—even if total time is the same.

为提升推理系统的"感知"性能，你应该在输出 token 生成的同时就以流式响应（streaming responses）的方式把它们发送出去。这远比强迫终端用户等到整段补全全部完成后才看到响应要好得多。这样，用户就能在信息到达时便开始阅读和处理。即使总时间相同，这也能让有效的交互体验快得多。

Humans can read anywhere from 200 to 300 words per minute, which translates to approximately 4–7 tokens per second—or up to 13 tokens per second for faster readers. Maintaining this pace of streaming response is ideal. Given that every performance decision comes down to trade-offs, this is an important metric to consider when making optimization choices, planning capacity, etc.

人类的阅读速度大约在每分钟 200 到 300 词之间，换算下来约为每秒 4–7 个 token——阅读较快的人则可达每秒 13 个 token。把流式响应维持在这个节奏是最理想的。鉴于每一项性能决策归根结底都是权衡，这是一个在做优化选择、规划容量等工作时需要考量的重要指标。

> It’s important to monitor your system’s token throughput against this human reading rate. For instance, if you find your system is streaming at only 2 tokens/sec due to model latency, that’s a sign to optimize further.

> 重要的是，要对照这一人类阅读速率来监控你系统的 token 吞吐量。例如，如果你发现系统由于模型延迟只能以每秒 2 个 token 的速度进行流式输出，那就是需要进一步优化的信号。

Streaming is supported by most modern inference engines, including vLLM, SGLang, and NVIDIA Dynamo. When you enable streaming, the server flushes tokens to the client using WebSockets, Server-Sent Events (SSEs), the HTTP Streaming protocol, etc. And it should do this immediately when the tokens are generated to meet the 4–13 tokens-per-second human-reading metric mentioned earlier.

大多数现代推理引擎都支持流式输出，包括 vLLM、SGLang 和 NVIDIA Dynamo。当你启用流式输出时，服务器会使用 WebSockets、服务器发送事件（Server-Sent Events，SSE）、HTTP 流式协议等把 token 刷新给客户端。而且它应该在 token 一生成时就立即这么做，以满足前面提到的每秒 4–13 个 token 的人类阅读指标。

The model effectively generates one token at a time—except when using speculative or multitoken decoding techniques like EAGLE or Medusa, respectively. As such, the system needs to group those generated tokens into batches of 2–5 tokens, flush the stream, and send the batched tokens back to the end user.

模型实际上是一次生成一个 token——除非使用推测解码或多 token 解码技术，例如分别对应的 EAGLE 或 Medusa。因此，系统需要把这些生成的 token 归为每批 2–5 个 token，刷新数据流，并把成批的 token 发回给终端用户。

It’s important not to accumulate too many tokens in a batch before flushing, or you defeat the purpose of streaming. On the other hand, sending one token per packet may be inefficient due to overhead. Often frameworks flush on every newline or end-of-sentence token.

重要的是，在刷新之前不要在一批里累积过多的 token，否则你就违背了流式输出的初衷。另一方面，每个数据包只发一个 token 又可能因开销而低效。框架通常会在每个换行符或句末 token 处进行刷新。

For example, if an answer is 100 tokens and takes 5 seconds to fully generate, with streaming, the first batch of tokens would arrive after 0.25 seconds if the batches are 5 tokens each (5 tokens ÷ 20 tokens per second = 0.25 seconds). This would be followed by a steady stream of token batches until the response is complete. This way, the user can start reading after just 0.25 seconds.

举例来说，如果一个答案有 100 个 token，完整生成需要 5 秒，那么在使用流式输出、每批 5 个 token 的情况下，第一批 token 会在 0.25 秒后到达（5 token ÷ 每秒 20 token = 0.25 秒）。随后是稳定的一批批 token 流，直到响应完成。这样，用户只需 0.25 秒就能开始阅读。

Without streaming, the user would stare at a blank screen for 5 seconds, then suddenly see the whole answer shown all at once, which is not ideal. This could lead to rage clicking and other forms of user frustration.

如果不使用流式输出，用户就得盯着空白屏幕 5 秒，然后突然一下子看到整个答案，这并不理想。这可能引发暴躁点击（rage clicking）以及其他形式的用户挫败感。

From a performance standpoint, streaming imposes a slight overhead due to additional, small network packets being sent instead of one big message. But this overhead is usually negligible compared to the model’s computation time—and compared to the improved end-user experience. Using HTTP/2 and persistent connections helps reduce overhead by sending tokens as a continuous stream without needing to reestablish connections.

从性能角度看，流式输出会带来轻微的开销，因为发送的是若干额外的小网络数据包，而不是一条大消息。但相比模型的计算时间——以及相比它所带来的更好的终端用户体验——这点开销通常可以忽略不计。使用 HTTP/2 和持久连接有助于降低开销，它把 token 作为一条连续的流发送，无需反复重新建立连接。

> Be mindful of Nagle’s algorithm and delayed acknowledgments. The interaction can add tens of milliseconds of delay and, on some stacks, up to roughly 200 ms in worst-case settings. Use TCP_NODELAY and, where available, quick-ack or reduced delayed-ack timers to minimize token flush latency. These help reduce token-send latency, which is critical for real-time streaming. The downside is more small packets fill the pipe—and bandwidth efficiency is reduced. But keep this in mind when tuning for ultralow latency.

> 要留意 Nagle 算法和延迟确认（delayed acknowledgment）。二者的相互作用可能增加数十毫秒的延迟，在某些技术栈上，最坏情况下甚至可达约 200 ms。请使用 TCP_NODELAY，并在可用时使用快速确认（quick-ack）或调低延迟确认定时器，以最小化 token 刷新延迟。这些做法有助于降低 token 发送延迟，而这对实时流式输出至关重要。其代价是更多的小数据包填满了管道——带宽效率有所下降。但在为超低延迟做调优时，请把这一点牢记于心。

It’s important to properly manage flow control such that if the client is slow to consume the stream due to network issues, for instance, you don’t want to block the model’s generation. Ideally, the inference engine will continue generating the whole response—even if the client falls behind consuming the stream.

妥善管理流控（flow control）很重要：这样，如果客户端因为网络问题等原因消费数据流较慢，你也不希望阻塞模型的生成。理想情况下，即便客户端在消费数据流上落后了，推理引擎也会继续生成完整的响应。

The system should use separate threads and CUDA streams to send the data back to the end user. This way, the main token generation loop isn’t interrupted by issues while sending the response.

系统应该使用单独的线程和 CUDA 流把数据发回给终端用户。这样，主 token 生成循环就不会因发送响应过程中的问题而被打断。

The inference engine maintains a bounded buffer to manage flow control and prevent unbounded memory growth. This can happen if a client stalls or disconnects. It’s important to handle these types of edge case scenarios as they are fairly common in production scenarios. In practice, you might allow up to 50–100 tokens to accumulate.

推理引擎维护一个有界缓冲区来管理流控，并防止内存无界增长——后者在客户端停滞或断开连接时可能发生。妥善处理这类边界情形很重要，因为它们在生产场景中相当常见。在实践中，你可以允许最多累积 50–100 个 token。

Beyond a certain limit, the inference engine can either pause generation or close the connection completely. This way, if a client drops mid-generation—or simply can’t keep up due to a slow connection—the engine can stop producing tokens and free those resources to handle other requests.

超过某个限度后，推理引擎可以暂停生成，或者干脆完全关闭连接。这样，如果某个客户端在生成过程中掉线——或仅仅因为连接缓慢而跟不上——引擎就能停止产出 token，并释放这些资源去处理其他请求。

Choosing the buffer limit involves balancing memory use and user experience. This is rarely an issue for short responses, but it is somewhat common for very large responses—especially for slower consumer connections.

选择缓冲区上限需要在内存使用和用户体验之间取得平衡。对于较短的响应，这很少成为问题；但对于很大的响应——尤其是消费者端连接较慢时——这就相当常见了。

Another trick to improve flow control is to use token pooling. If the model generates tokens faster than needed for streaming, the system can intentionally add a delay to smooth out the rate of generation. For instance, if the model bursts out 20 tokens in 0.5 seconds because it was a simple part of the response, for instance, sending them all at once can display a big chunk and then pause for the next part.

改善流控的另一个技巧是使用 token 池化（token pooling）。如果模型生成 token 的速度快于流式传输所需的速度，系统可以有意加入延迟，让生成速率变得平滑。举例来说，如果模型因为某段回答很简单，在 0.5 秒内一下子迸出 20 个 token，把它们一次性全部发送会先显示一大块内容，然后又停顿等待下一段。

You may prefer your application’s UI to use a steadier typewriter effect. In this case, you can introduce an artificial stream delay like 50 ms between token sends. This doesn’t affect actual latency much, and it improves the UX by avoiding janky bursts of tokens.

你可能更希望应用的 UI 呈现更平稳的打字机效果。这种情况下，可以在每次发送 token 之间引入一个人为的流延迟，比如 50 ms。这对实际延迟影响不大，却能避免 token 突发抖动，从而改善用户体验。

You can make the token-pooling delay configurable to adapt to different UXs for different types of users (e.g., paid users, free users, etc.). Paid users can stream faster, while free users will be throttled a bit. This can help manage load with limited resources.

你可以把 token 池化的延迟做成可配置的，以适配不同类型用户（例如付费用户、免费用户等）的不同体验。付费用户可以更快地流式输出，免费用户则会被稍加限流。这有助于在资源有限的情况下管理负载。

Streaming responses also allow end users to start evaluating the intermediate results and take action before the response completes. For instance, if the user sees the first part of the answer and realizes it’s not going in the direction they want, they can stop the generation using a Stop or Interrupt button.

流式响应还让终端用户能够开始评估中间结果，并在响应完成之前就采取行动。举例来说，如果用户看到答案的开头部分，意识到它的走向不是自己想要的，就可以用 Stop 或 Interrupt 按钮停止生成。

This saves unnecessary compute and improves the UX as the end user doesn’t have to wait for a bad completion to finish before they provide their feedback. You should monitor—or at least measure—early stops.

这既节省了不必要的算力，也改善了用户体验，因为终端用户无需等一个糟糕的回答生成完毕才能给出反馈。你应当监控——或至少度量——提前停止（early stop）的情况。

Too many explicit stops might indicate an issue with the model’s relevance. If you see many users stopping at a certain point, it might indicate the model often goes off-track or is too verbose beyond that length. This could inform you that the model needs additional fine-tuning to make it more concise. Or you can make adjustments to the maximum tokens generated and sent to the end user.

过多的显式停止可能意味着模型的相关性出了问题。如果你发现很多用户都在某个位置停止，这可能说明模型经常在该长度之后跑偏，或过于啰嗦。这可以提示你：模型需要额外的微调来让它更简洁。或者，你也可以调整生成并发送给终端用户的最大 token 数。

Or the early stops could be caused by a frustrated “rage click” type of user who changes their mind frequently. Either way, it’s best to allow the user to interrupt the token-generation process. This lets the inference cluster reclaim the resources to handle other requests.

也有可能这些提前停止是由一个沮丧的“暴躁点击”型用户造成的，他们频繁改变主意。无论哪种情况，最好都允许用户中断 token 生成过程。这能让推理集群回收资源去处理其他请求。

In short, streaming is a must-have for a responsive LLM service. It doesn’t increase raw throughput—if anything, it adds a slight overhead, but it improves the user-perceived speed of the system. Streaming responses should be implemented carefully to not interfere with generation. Different threads and CUDA streams should handle sending responses back to the end user so that the main token generation loop isn’t stalled on flakey end-user network connections.

简而言之，流式传输是响应式 LLM 服务的必备能力。它不会提升原始吞吐量——如果有影响，反而会增加一点开销——但它能提升系统在用户感知层面的速度。流式响应的实现必须谨慎，以免干扰生成。应当用不同的线程和 CUDA 流来负责把响应发回终端用户，这样主 token 生成循环就不会因终端用户不稳定的网络连接而停顿。

> It’s recommended that you continuously profile your end-to-end latency with streaming enabled versus disabled in a test environment. Make sure that token emission is evenly spaced—and that no significant bottlenecks like lock contention or I/O waits are introduced. Tools like Locust are Python friendly and can simulate clients. This lets you test your low-latency streaming workloads at scale.

> 建议你在测试环境中持续剖析开启与关闭流式传输时的端到端延迟。确保 token 的发送在时间上均匀分布，并且没有引入锁竞争或 I/O 等待之类的显著瓶颈。像 Locust 这样的工具对 Python 很友好，可以模拟客户端。这让你能够大规模地测试低延迟的流式工作负载。

### Debouncing and Request Coalescing

### 去抖与请求合并

Many production systems also implement UX features called *debouncing* and *request* *coalescing*. By debouncing, or pausing, before responding, the system can recognize if a user sends multiple requests in quick succession—either by accident or from rage clicking, as shown in Figure 16-10.

许多生产系统还实现了称为 *去抖*（debouncing）和 *请求合并*（request coalescing）的用户体验特性。通过在响应前去抖，也就是先暂停一下，系统可以识别出用户是否在短时间内接连发送了多个请求——要么是不小心，要么是暴躁点击所致，如图 16-10 所示。

![Figure 16-10. Debouncing pauses a bit before performing an action](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-10.png)

![图 16-10. 去抖会在执行某个动作之前稍作暂停](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-10.png)

In this case, the system can either coalesce the multiple queries into one query or drop all but the latest query. These types of application-level guardrails help reduce excessive, repeated, wasteful load on the backend.

这种情况下，系统既可以把多个查询合并成一个查询，也可以只保留最新的查询而丢弃其余的。这类应用级护栏有助于减少后端上过度、重复、无谓的负载。

For example, if a user double-clicks Submit or sends two very similar queries within a second, the system can drop the duplicate. Rage clickers impatiently resubmit multiple times when they are frustrated—often because of the application’s poor response time.

举例来说，如果用户双击了 Submit，或在一秒内发送了两个非常相似的查询，系统就可以丢弃重复的那个。暴躁点击者在沮丧时会不耐烦地反复重新提交——往往正是因为应用的响应时间太差。

Ironically, debouncing and request coalescing can create more latency, frustrate the “rage” user segment even more, and cause them to click even more! In this case, it’s helpful if the UI disables the input while a request is in flight. This can prevent double-clicking at the UI level. But it’s also good to have server-side protection in place, as well.

具有讽刺意味的是，去抖和请求合并反而可能制造更多延迟，让“暴躁”用户群更加恼火，导致他们点击得更频繁！这种情况下，如果在请求处理途中让 UI 禁用输入，会很有帮助。这可以在 UI 层面防止双击。但在服务端也设置相应的防护同样很有必要。

> Modern inference load balancers support a debouncing interval. Use a modest interval (e.g., 2–5 ms), and find a good trade-off between batching efficiency and added latency. And next time you submit something in ChatGPT, notice the debouncing delay. It’s a bit annoying now that you know about it, isn’t it?!

> 现代推理负载均衡器支持设置去抖间隔。使用一个适度的间隔（例如 2–5 ms），在批处理效率与额外延迟之间找到一个好的权衡点。下次你在 ChatGPT 里提交内容时，留意一下那个去抖延迟。既然你现在知道了它的存在，是不是觉得有点烦人？！

### Token Output Limits and Timeouts

### token 输出上限与超时

Because response token counts directly influence latency, you can implement output length limits or timeouts. This kind of application-layer constraint can prevent runaway generations in which the model keeps generating beyond a reasonable number of tokens.

由于响应的 token 数量直接影响延迟，你可以实现输出长度上限或超时。这类应用层约束可以防止失控生成——即模型不断生成，远超合理的 token 数量。

Token output limits and timeouts help maintain consistent latencies. They also prevent malicious and accidental prompts from causing extremely long generations that could tie up GPU resources. Many public APIs set strict output limits for exactly this reason. It’s both an abuse prevention mechanism as well as a performance safeguard.

token 输出上限与超时有助于维持一致的延迟。它们还能防止恶意或意外的提示引发极长的生成，从而占用 GPU 资源。许多公开 API 正是出于这个原因设置了严格的输出上限。这既是一种滥用防护机制，也是一道性能防线。

Always set server-side timeouts, as well. For example, if a generation exceeds 30 seconds, return a partial result or an apology. Users expect quick failures rather than a hanging response.

同样，务必设置服务端超时。举例来说，如果一次生成超过 30 秒，就返回部分结果或一句致歉。用户宁愿快速失败，也不愿面对一个挂起的响应。

> Also monitor if your model tends to ramble. Applying a moderate token limit, perhaps 4,096 tokens for a chat answer, can actually improve quality by keeping the model on track and avoiding long answers.

> 另外也要监控模型是否倾向于啰嗦。施加一个适度的 token 上限，比如聊天回答设为 4,096 个 token，实际上可以通过让模型保持在正轨、避免冗长回答来提升质量。

It’s important to choose these limits and timeouts based on user needs. These limits keep latency predictable and also cap the worst-case compute scenario. This helps with capacity planning, etc.

根据用户需求来选择这些上限与超时很重要。这些上限让延迟保持可预测，同时也为最坏情况下的算力开销设了上限。这有助于容量规划等工作。

In summary, the combination of input optimizations (e.g., compression and cleansing), caching, smart routing, and UX optimizations (e.g., streaming data) can reduce the workload, reduce the size of the inference cluster, save cost, and improve user satisfaction. These application-level strategies complement the low-level optimizations from earlier sections—rounding out the holistic approach to improving LLM serving performance.

总而言之，输入优化（例如压缩与清洗）、缓存、智能路由与用户体验优化（例如流式数据）相结合，可以减少工作负载、缩小推理集群规模、节省成本并提升用户满意度。这些应用级策略与前几节的底层优化互为补充——共同构成了改善 LLM 服务性能的整体性方法。

## Key Takeaways

## 关键要点

The techniques in this chapter show that efficient LLM serving is a holistic engineering effort that combines modern GPU hardware, novel software optimizations, and comprehensive monitoring. Profiling tools identify the bottlenecks. Systematic debugging techniques fix the issues. Here are some key takeaways from this chapter:

本章的这些技术表明，高效的 LLM 服务是一项整体性的工程工作，它把现代 GPU 硬件、新颖的软件优化与全面的监控结合在一起。剖析工具找出瓶颈，系统化的调试技术修复问题。以下是本章的一些关键要点：

*Comprehensive profiling* Perform end-to-end profiling across the inference stack. By measuring latency and resource usage at each stage using profilers, engineers can pinpoint slowdowns and inefficiencies. This data-driven approach guides targeted optimizations to eliminate bottlenecks.

*全面剖析* 在整个推理栈上进行端到端剖析。通过用剖析器测量每个阶段的延迟与资源占用，工程师可以精确定位变慢与低效之处。这种数据驱动的方法可以指导有针对性的优化，从而消除瓶颈。

*Monitoring and observability* Implement robust monitoring for deployed inference services. Track key metrics such as latency percentiles, throughput, GPU utilization, and memory usage in real time to detect regressions or resource saturation early. Use logging and tracing to get visibility into per-request processing and identify hotspots or anomalies in a large-scale workload.

*监控与可观测性* 为已部署的推理服务实现健壮的监控。实时跟踪延迟分位数、吞吐量、GPU 利用率与内存占用等关键指标，以便尽早发现性能回退或资源饱和。使用日志与追踪来洞察每个请求的处理过程，并在大规模工作负载中识别热点或异常。

*Debugging and iterative tuning* By adopting a systematic debugging workflow, you can resolve performance and correctness issues quickly—even across tens of thousands of nodes. This way, when an unexpected spike occurs (e.g., decrease in throughput or increase in latency), you can easily drill down from the high-level symptom to the low-level issue.

*调试与迭代调优* 通过采用系统化的调试工作流，你可以快速解决性能与正确性问题——即便跨越数万个节点也不例外。这样一来，当出现意外的尖峰（例如吞吐量下降或延迟上升）时，你就能轻松地从高层症状一路下钻到底层问题。

*Validating optimizations with metrics* Tools such as GPU debuggers, memory leak detectors, and performance-related unit tests help verify that optimizations like quantization and kernel fusion do not introduce silent performance and correctness errors. This iterative tune-and-test cycle is essential for maintaining high performance and reliability with each new optimization released into production.

*用指标验证优化* GPU 调试器、内存泄漏检测器与性能相关的单元测试等工具，有助于验证量化、核函数融合等优化不会引入静默的性能与正确性错误。这种“调优—测试”的迭代循环，对于在每次将新优化投入生产时都保持高性能与高可靠性至关重要。

*Efficiency and cost optimization* Focus on improvements that make inference more cost-effective. Every optimization that increases throughput and utilization will directly improve the cost-per-query. By profiling and refining the system, teams can serve more requests with fewer GPUs. This leads to significant savings in infrastructure costs and power efficiency.

*效率与成本优化* 聚焦于让推理更具成本效益的改进。每一项提升吞吐量与利用率的优化，都会直接改善每次查询的成本。通过剖析并精细打磨系统，团队可以用更少的 GPU 服务更多的请求。这带来基础设施成本与能效上的显著节省。

## Conclusion

## 结语

Profiling, debugging, and full-stack system tuning are critical for maintaining efficient, reliable, and cost-effective LLM inference at scale. As model sizes grow toward multi-trillion parameters, production clusters are scaling toward hundreds of thousands and even millions of GPUs per deployment. Continued codesign of software and hardware remains essential at this scale. It’s no longer enough to rely on just hardware to increase performance—especially as power is becoming a limiting factor. Both the software and hardware need to be codesigned and continuously tuned together—and at every layer in the stack.

剖析、调试与全栈系统调优，对于大规模维持高效、可靠且经济的 LLM 推理至关重要。随着模型规模朝着数万亿参数增长，生产集群的规模也正朝着每次部署数十万乃至上百万个 GPU 扩展。在这样的规模下，软硬件的持续协同设计仍然不可或缺。仅靠硬件来提升性能已经不够了——尤其是在功耗正在成为限制因素的当下。软件与硬件都需要协同设计并持续一起调优——而且要贯穿栈的每一层。

An optimized inference infrastructure provides fast response times for end users, predictable and stable system behavior, and low operational costs. Only then can you deliver inference systems that serve the largest and most powerful models in the world.

经过优化的推理基础设施能为终端用户提供快速的响应时间、可预测且稳定的系统行为，以及低廉的运营成本。唯有如此，你才能交付出能够服务全球最大、最强模型的推理系统。

In the next chapters, we will dive deep to understand and optimize the most compute, memory, and network intensive parts of modern LLM inference systems: calculating (prefilling) the KV cache, sharing it with all workers in cluster, and using it to generate (decode) new tokens.

在接下来的几章中，我们将深入理解并优化现代 LLM 推理系统中计算、内存与网络最密集的环节：计算（prefill）KV 缓存、把它共享给集群中的所有 worker，以及用它来生成（decode）新的 token。
