# 第 16 章. 大规模推理的剖析、调试与调优

运营大型 LLM 推理集群，需要监控与调试工具来确保一切按预期运行。当性能偏离目标时，它们还能帮你快速定位瓶颈。

在本章中，我们将演示如何使用各类工具来监控和调试这些复杂系统，例如用 NVIDIA Nsight Systems 做剖析，用 Prometheus/Grafana 做集群级遥测。我们还会展示如何采集并解读关键指标，如 GPU 利用率、内存压力、尾延迟（tail latency）分位数、缓存命中率、逐 token 计时等等。这些指标可以指导我们对推理引擎做性能优化。

接着，我们讨论运营层面的性能调优，包括在大型集群中优化 GPU 利用率、降低推理延迟、提升吞吐量的经生产验证的方法。这涵盖了计算与通信重叠、请求调度与批处理，以及高效使用 NVLink、NVSwitch、InfiniBand 等高速互连的技术。

我们还会对比面向推理的实时量化（quantization）技术，包括用 Generalized Post-Training Quantization（GPTQ）和 Activation-Aware Weight Quantization（AWQ）等实现将模型压缩到 8 位与 4 位精度的方法。在此过程中，我们会讨论仅权重量化（weight-only quantization）与同时量化权重和激活之间的权衡。我们提供在服务流水线中应用量化以降低内存占用、提升吞吐量的实用指南——同时保持模型精度不下降。

最后，我们探讨与底层性能调优互补的应用级优化（application-level optimization），包括提示压缩（prompt compression）、前缀缓存（prefix caching）、去重、查询路由（例如回退模型），以及部分输出流式返回等策略。

## 推理性能的剖析、调试与调优

现代 LLM 推理引擎有大量相互关联的环节——尤其是在 prefill 与 decode 分离（disaggregated）的情况下。一个典型请求的生命周期涉及许多组件，如图 16-1 所示。

![图 16-1. prefill 与 decode 分离的 LLM 推理系统中典型请求的生命周期](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-1.png)

面对如此复杂度，推理性能调优的工作流高度迭代。它需要细致的调优与持续的验证。

首先，你应当观察指标，识别当前的瓶颈，例如某块 GPU 未被充分利用，或延迟高于预期。接着，提出改进假设，例如“增大批大小”或“为操作 X 增加通信与计算的重叠”。然后，实现修复方案并验证该假设。

之后，理想情况下你应当在预发布（staging）环境中，用具有代表性的负载配合剖析工具测试该修复，验证改动的行为符合预期。例如，你可以确认某个操作确实呈现出恰当的内存与计算重叠。

最后，你应将修复部署到生产环境，并监控 Grafana 与日志，验证该修复确实在真实负载上提升了吞吐量与延迟。当新的瓶颈出现时，重复这个工作流。

这套“观察—假设—调优”的循环应当持续进行。现代部署往往会将这些步骤自动化。例如，你可以用定期的负载测试——以及随后对关键指标的异常检测——来触发一次调优工作流。

> 在向生产环境部署优化（包括更新后的推理运行时和模型变体）时，建议执行金丝雀（canary）发布。通过先把优化部署到运行在少量服务器上的一小部分流量，你可以在向所有终端用户全量部署之前先验证优化效果。这种增量式方法通过尽早发现意外副作用，帮助缩小其“爆炸半径”，避免影响所有用户。

设想这样一个场景：由于过多的分词或推理数据预处理，主机侧 CPU 利用率飙升到 100%。这会限制推理引擎能处理的并发流数量。一种修复办法是用 GPU 加速的分词库，或用 CUDA、OpenAI 的 Triton 语言编写的自定义 GPU 核函数，把预处理迁移到 GPU 上。

在部署新的库或核函数后，你应当对比部署前后的 CPU 利用率。如果看到 CPU 利用率下降、整体吞吐量上升，那么系统就不再受基于 CPU 的输入预处理所瓶颈约束了。

你还应关注各类缓存的命中率，包括前缀缓存、提示嵌入缓存和 KV 缓存（KV cache）。你应当有“缓存命中”对比“缓存未命中”的指标。高缓存命中率意味着系统在有效复用数据。反之，如果你看到很高的缓存未命中率，那多半需要调整缓存大小、逐出（eviction）策略或缓存策略，以最大化缓存命中。

vLLM 的 LMCache 组件允许调整 GPU 与 CPU 的缓存比例。如果因 GPU 内存受限导致未命中率偏高，你可以启用它的分页缓存卸载能力，让 CPU 来分担。务必确保你的缓存逐出策略（Least Recently Used [LRU]、Least Frequently Used [LFU] 等）与访问模式相匹配。

另一个场景是使用 KV 缓存，在批处理请求之间复用相同输入序列前缀的数据，从而避免为该前缀重新计算 KV。这时，你需要度量请求共享前缀的频率。这会触发一次*前缀\*\*合并*（prefix merge）事件，并递增 vLLM 中的前缀缓存命中指标，包括 vllm:gpu_prefix_cache_queries 和 vllm:gpu_prefix_cache_hits。例如，借助它们你就能用“命中数 ÷ 查询数”算出命中率。

度量前缀合并率有助于你将其与实际缓存命中率相关联，从而衡量缓存层的真实收益。这样，你就能调整批处理与调度策略以最大化共享前缀——并预测不同负载下端到端的吞吐量与延迟改善。

你可以在推理引擎上运行合成生成的数据，测试包含大量重复前缀的提示。理想情况下，你会看到因前缀合并而带来的 prefill 计算量下降。

vLLM 和 SGLang 等现代 LLM 推理引擎原生暴露前缀合并指标。但如果前缀合并不是你的推理引擎导出的一等指标，你应当自行埋点一个“前缀去重 token 数”的自定义计数器（counter）来监控其有效性。

> 如果你发现前缀合并的表现不如预期，应当检查前缀匹配逻辑是否失效。调试时先从检查分词器差异入手。这是绝大多数前缀匹配问题的可能根因。

除了性能之外，监控还有助于容量规划。通过跟踪利用率和延迟随负载上升的表现，你可以推断系统在何时会触及某个特定上限，例如 p95（第 95 百分位）延迟开始呈指数上升。这种情况下，动态批大小可能正在增大到收益递减的临界点。

> 如果你采用分层缓存策略，包括基于 NVMe 的 KV 缓存扩展，务必监控该设备的 I/O 延迟。高 I/O 延迟会显著降低缓存性能。

当单 GPU 的并发达到上限、进一步增大批大小不再提升吞吐量时，你可以考虑横向扩展（scale out）：增加更多 GPU、部署额外的模型副本，或增加专家数量，把工作分摊到更多计算单元上。

在横向扩展之前，你还应考虑模型压缩——或切换到更低精度（FP8/FP4）——以获得每块 GPU 更高的有效吞吐量。不过，一旦硬件饱和（例如 SM 达到 100%、内存带宽接近峰值利用），增加更多 GPU 或使用张量/流水线并行很可能就是提升吞吐量的唯一路径了。

而且要记住，始终权衡新硬件的成本与效率收益。有时候，升级到内存更大、FLOPS 更高的更新款 GPU，会比横向扩展一批老旧 GPU 更具成本效益。

增加专家数量可以抬高你的吞吐量上限——但前提是你同时改进专家路由与调度，以应对额外的全对全（all-to-all）通信。否则，粗放的扩展可能只是把瓶颈转移到网络上。接下来，我们讨论监控，以及如何验证优化努力是否真正见效。

### 监控系统指标与计数器

传统微服务（microservice）调用的执行时间相对统一且可预测，而 LLM 请求则不然：它们不统一，延迟可能天差地别。这一差异如图 16-2 所示。

![图 16-2. 传统微服务调用与 LLM 调用之间的差异](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-2.png)

对于生产中的持续监控，常见做法是用 Prometheus 从每个 GPU 计算节点采集指标——并用 Grafana 仪表盘将其可视化。需要跟踪的关键 GPU 指标包括 GPU 利用率（SM 处于繁忙状态的时间占比）、GPU 内存用量、拷贝引擎利用率、PCIe 与 NVLink 吞吐量，以及 GPU 温度和功耗（例如降频）。注：L1、L2 活动、占用率（occupancy）和指令吞吐等底层计数器，可用 Nsight Compute 或 CUPTI 采集，而非 DCGM 与 Prometheus。

> 在监控内存碎片时，cudaMemPool 指标和异步分配器统计信息很有帮助。应将它们整合进你的监控体系，因为这会极大方便在生产中调试系统性能问题。

监控互连利用率同样重要，包括 NVLink、NVSwitch 带宽和网卡（NIC）吞吐。这样，你就能在多 GPU 和多节点集群配置中捕捉到通信瓶颈。

NVIDIA 的 Data Center GPU Manager（DCGM）暴露了许多 GPU 指标，Prometheus 可以抓取并采集它们。例如，DCGM 提供 DCGM_FI_DEV_GPU_UTIL 表示 SM 利用率百分比，DCGM_FI_DEV_MEM_COPY_UTIL 表示内存拷贝引擎利用率，DCGM_FI_DEV_FB_USED 表示已用的帧缓冲内存等等。

DCGM 暴露 NVLink 错误计数器，并可在部分平台和驱动版本上暴露吞吐计数器。对于持续的链路利用率，还应使用 nvidia-smi nvlink 和 Nsight 工具。你应当把这些指标整合进仪表盘，并设置告警，以便识别网络何时被跨 GPU 和跨节点的通信流量占满。DCGM 会跟踪 Xid 计数器以及严重的 GPU 错误。

> 尽管 DCGM 暴露了 NVLink 计数器，但截至本文撰写时，dcgm-exporter 默认并不在所有平台上暴露每条链路的带宽。因此，如果你需要链路级吞吐，可能需要直接查询 DCGM 或扩展该 exporter。

同样建议采集高层应用指标，如每秒查询/请求数、平均延迟以及 p95/p99 延迟、活跃上下文数量，以及以 token/秒计的吞吐量。KV 缓存利用率与大小（整体的以及每个节点的）相关指标也极其重要，需要监控。

你可以设置 Prometheus node exporter 从每个节点采集所有这些指标，把数据汇集到一处，甚至为关键阈值设置告警。随后，Grafana 就能将这些指标绘制成实时仪表盘，与团队共享。例如，图 16-3 展示了如何从 Kubernetes 集群中的每块 GPU 采集指标，导出到 Prometheus，再用 Grafana 可视化。

这样，当你部署一项新的优化（比如增大批处理）时，Grafana 会立即显示每块 GPU 的利用率是否上升。你还可以监控以确保 p95/p99 延迟保持在目标范围内。

计数器同样极其有用——尤其对于动态自适应系统。例如，如果你的推理引擎会根据当前状况动态调整批大小，你可能会想递增一个“批大小变更”计数器。

![图 16-3. DCGM 从 Kubernetes GPU 节点采集指标并发送到 Prometheus](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-3.png)

另一种选择是把变更记录到日志文件里，但这样一来，就需要用 Apache Spark 之类的工具对日志文件做缓慢的、基于文本的搜索/聚合来离线分析。之后，你还得手动把日志分析结果与 Prometheus 指标相关联。

只要为感兴趣的应用级事件（包括错误）递增一个简单的计数器，数据就会被推送到 Prometheus，并可在 Grafana 仪表盘中与所有其他指标一起实时查看。此外，考虑对关键事件使用结构化日志与分布式追踪。

现代应用性能管理（application performance management，APM）工具——以及 OpenTelemetry——可以摄取这些日志/追踪，并将它们与指标相关联。这样就能提供跨整个系统的一致的事件时间线视图。洞察这条时间线，有助于加快调试性能问题的速度。

如果你持续监控这些指标，就能获知下一步该调优什么。例如，如果 GPU 利用率低于预期，你可以检查 GPU 内存是否已被占满。如果尚未充分利用，你可以尝试增大批大小或最大并发请求数。不过，务必留意延迟服务级目标（service-level objective，SLO）。你可不希望突破这些目标。

> 现代推理服务器会为动态批处理暴露一个“最大延迟”设置。调整它以满足你的 SLO。调大它会增大批大小（提升吞吐量）。但调得过大又会损害 p99 延迟。请结合你的延迟目标持续调整。

反之，如果 GPU 内存接近上限，推理引擎可能开始把不活跃的 KV 缓存数据换出到主机 CPU 内存或 NVMe 存储。这会降低 GPU 利用率，因为 GPU 需要等待来自慢速 CPU 内存或磁盘的额外数据传输。

> 如果你看到 GPU 内存拷贝引擎利用率出现尖峰——或者出现与低 SM 利用率同步的异常 NVLink 利用率——那么你的推理引擎很可能正在把 KV 缓存数据换入换出 GPU 内存。这会因过多的数据传输延迟而使系统受阻。

如果确实在发生换页，你可以调整推理引擎的分页参数以减少抖动（thrashing），应用 FP8 或 FP4 量化，为缓存增大 GPU 内存分配，并可能改变换页策略。这应当能把拷贝利用率降下来、把计算利用率提上去——正是你想看到的结果。

Grafana 同样可用于延迟跟踪。你可以绘制端到端请求延迟的分布——通常还会同时度量 prefill 延迟和逐 token 延迟。如果 p99 延迟在某些时刻出现尖峰，你应当把它与 GPU 指标和其他日志相关联。

例如，p99 延迟尖峰可能与 GPU 利用率下降的时段相关。也许这个延迟尖峰与一次触发了更大动态批大小的流量激增相关。这会导致该时段内延迟升高。为了验证，你可以在 Grafana 仪表盘的延迟图上叠加 RPS（每秒请求数），看看这两张图是否相关。

如果这个尖峰正如我们假设的那样，是由批大小的动态增大所致，那么务必确认它没有突破你的服务级协议（service-level agreement，SLA）。如果突破了，你可以尝试减小最大请求批队列延迟，或降低最大批大小，以给延迟设一个上限。

诊断问题时，日志同样极其宝贵。你应当在代码中埋点，记录关键事件，例如批何时形成、通信何时开始/结束等。最好使用 DEBUG 级别，以便按需启用/禁用——从而不影响请求-响应延迟。

当你启用调试日志时，会看到一条文本格式的逐步时间线。在一次调试会话中，你很可能会把日志时间线与 Prometheus/Grafana 指标结合起来一起用。例如，你可以看到全对全通信耗时超过 5 ms 的频率有多高。

把基于日志的时间线与指标结合起来，你就能发现离群点，比如某些网络问题可能拖慢了全对全通信交换中的某一次迭代。如果这种情况持续发生，你可以调高专家容量因子（capacity factor），让任何超额的 token 自动溢出到备用专家副本上——理想情况下，该副本托管在网络路径更稳定的 GPU 上。这会均衡负载并把延迟降到最低。

> 实践中，把容量因子设为 1.2–1.5 是常见做法，因为这允许在主专家过载时，把额外 20%–50% 的 token 重新分配出去。这能显著平滑 MoE 推理中的尾延迟。当某个专家所在 GPU 的互连状况变差、略有停顿时，与其把请求排队堵在它后面，不如溢出到第二个专家。如果你的网络出于某种原因持续出现问题，这样做会降低系统对离群点的敏感度。

### 用 Nsight Systems 与 Nsight Compute 做剖析

在开发和调优推理代码时，你可以用 Nsight Systems 捕获跨 CPU 与 GPU 的工作负载追踪。Nsight Systems 提供时间线视图，以微秒级分辨率展示 CPU 线程、GPU 核函数、CUDA 事件、NCCL 通信等等。

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

在这里，我们用 nvtxRangePushEx/nvtxRangePop 显式标注区域。我们在 model.encode(...) 之前紧接着 push 一个 "Prefill" 区间，并在其后立即 pop。在 decode 循环内，我们在每次迭代开头 push "Decode"，并在 model.decode_next() 之后 pop。小巧的 NVTX_PUSH/NVTX_POP 辅助宏还会附上颜色（十六进制值）和文本。这有助于减少时间线可视化中的错配——同时保持调用点简洁。显式的 push/pop 配对在代码中清晰可见，便于审查。

带颜色的区块注解会出现在 Nsight Systems 的 GPU 活动时间线上，标注为 "Prefill" 和 "Decode"。这让你很容易看出每个阶段耗时多久——以及这些阶段如何与通信操作重叠。这有助于识别诸如 GPU 空闲间隙和意外同步之类的问题。

> 注：请注意，我们直接使用 NVTX C API（nvToolsExt），而非 PyTorch 的 record_function()。这让我们能在纯 C++ 运行时中标注热点路径，并在工作从 Python 或其他语言发起时保持标记一致。

通过把范围收紧到 model.encode(prompt_tokens) 周围所需的最小区域，剖析标记就恰好覆盖 prefill 工作，而不涉及其他代码。这提升了追踪的清晰度和性能诊断效果。

在多个 CUDA 流上入队工作时（例如用专门的“传输”流做 H2D/D2H 拷贝、用“计算”流跑核函数），你应当使用按流划分的区间。为此，你可以用各不相同的 NVTX 区间把每条流的主机端代码包裹起来。

例如，你可以用 nvtxNameCudaStreamA(transfer_stream, "transfer_stream") 和 nvtxNameCudaStreamA(compute_stream, "compute_stream") 为流命名。然后，你会在内存拷贝/传输周围使用 nvtxRangePushA("transfer_stream")

和 nvtxRangePop()，在核函数启动周围使用 nvtxRangePushA("compute_stream") 和 nvtxRangePop()。

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

在这里，我们为流命名，并用按流划分的区间把入队点包裹起来，让 Nsight 时间线保持可读。需要注意的一点是，NVTX 区间标注的是主机线程时间线。GPU 通道则按流展示核函数/内存拷贝。为流命名有助于在分析时把主机区间与正确的 GPU 通道对应起来。

Nsight Compute 让我们能剖析单个核函数，以精确定位低效之处。我们可以使用 Nsight Compute 基于区段（section）的剖析功能，聚焦于核函数的特定部分，例如内存事务。

另一个鲜为人知却极其有用的工具，是 Nsight Compute 的 CUDA Program Counter（PC）Sampling 功能。它对程序计数器进行采样并识别热点，而无需完整、重量级的埋点，如图 16-4 所示。

![图 16-4. Nsight Compute 的 CUDA 程序计数器采样（Program Counter sampling，PC sampling）功能以低开销的方式帮助识别热点（来源：https://oreil.ly/DyKWR）](AI%20Systems%20Performance%20Engineering-ch16_images/figure-16-4.png)

具体来说，我们可以用它来剖析在线运行的推理服务器，精确定位到底是哪些核函数指令耗时最多。而且我们能以低开销的方式做到这一点。既然我们已经讲完了用 Nsight Systems 与 Nsight Compute 做剖析，接下来就讨论一些常见的推理排障方案（troubleshooting recipe）。

> 在对在线服务做生产环境排查时，优先使用程序计数器采样，以最小开销定位热点。只有当采样结果指向某个具体核函数或阶段时，才切换到完整追踪。

### 推理排障方案

在生产环境中，持续运行重量级剖析器并不现实。因此，你需要依赖轻量的、基于指标的监控，如 GPU SM 利用率、KV 缓存告警、尾延迟分位数、缓存命中率和 OOM 告警，来检测异常并指导有针对性的修复。当某个指标越过特定阈值时，你可以针对其根因提出假设，例如批大小过小、KV 缓存不足、路由热点、分片不均衡、内存超额分配等。

然后你施加修复，例如调整批大小、提高内存利用率上限（如果可行）、调整路由器阈值，或启用 CPU 卸载。修复推送后，你应当验证其影响，并确认指标已回落到阈值以下。表 16-1 列出了一些常见生产问题的关键指标、症状、可能原因和推荐处置。

表 16-1. 常见排障症状、原因与推荐处置

| 指标/症状                     | 可能原因                                           | 推荐处置                                                                                                                                                                                     |
| ----------------------------- | -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SM utilization < 50%          | 批过小或缺少融合核函数                             | 增大批大小，启用融合核函数（FlashAttention 或 PyTorch 中经 cuDNN 融合的 scaled_dot_product_attention (SDPA) 后端），或添加自定义融合核函数（例如用 Triton）；然后用 nsys --trace=cuda 剖析。 |
| KV 缓存抢占（preemption）告警 | KV 缓存空间不足（vLLM）                            | 提高 GPU 内存利用率阈值，减小批处理 token 的最大数量，考虑用 PagedAttention 做动态 KV 分配。                                                                                                 |
| 高尾延迟（p95 > 200 ms）      | decode 节点热点或队头阻塞（head-of-line blocking） | 检查路由器日志中的路由模式。调整预取阈值。启用推测解码（speculative decoding）路径。                                                                                                         |
| 负载下缓存命中率 < 60%        | 分片放置不均衡或缺少前缀缓存                       | 校验前缀缓存连接器配置（例如用于 vLLM 的 LMCache 的 NIXL，以及 NVIDIA Dynamo 的 NIXL 连接器），并按需增大前缀缓存 TTL 或副本数。                                                             |
| 多租户 GPU 上意外 OOM         | GPU 内存超额分配                                   | 降低每实例 GPU 内存利用率，启用 CPU/NVMe 卸载，把进程绑定到 CPU 插槽以减少跨插槽流量。                                                                                                       |
| 不规则的性能离群点            | 时钟不一致或热降频                                 | 确保所有时钟同步，并监控热与功耗降频。                                                                                                                                                       |

> 注：所有指标表格中的数值均为示意，用于阐释概念。不同 GPU 架构上的实际基准结果，请参见 GitHub 仓库。

你也可能会在日志文件里发现一些此类信息。AWS 等云厂商支持对日志文件使用正则表达式（regular expression，RegEx）过滤器，从日志行中提取数值并直接导出为指标。例如，AWS CloudWatch 就支持这一实用功能。下一个代码块中给出了一些便于监控的示例日志行。

下面是一段 vLLM 日志示例，表明由于 KV 缓存空间不足而发生了 KV 缓存抢占。因此触发了一次 KV 重算，这会占用更多 GPU 计算资源并增加延迟：

```
WARNING 2025-05-03 14:22:07 scheduler.py:1057 Sequence group 0 is preempted by
PreemptionMode.RECOMPUTE because not enough KV cache space.
total_cumulative_preemption_cnt=1
```

接下来是一段 NVIDIA Dynamo 路由日志示例。这里，第一行显示本地 decode 工作节点上 90% 的前缀缓存命中，因而 prefill 得以在本地继续运行。下一行显示了一次本地缓存未命中。路由器随后把 prefill 分派到远端的 GPU-node-03 工作节点：

```
[Router] 2025-05-03T14:23:11Z INFO KVRouter: prefix-cache hit (90%) for
model=DeepSeek-R1; routing to local vLLM worker
[Router] 2025-05-03T14:23:12Z INFO KVRouter: cache miss; dispatching remote
prefill to GPU-node-03
```

### 全栈推理优化

高性能 LLM 推理要求在技术栈的每一层做协调一致的优化。这涵盖从模型架构、核函数实现，到运行时引擎、系统编排和部署基础设施的方方面面。

模型级技术，如剪枝（pruning）、蒸馏（distillation）、稀疏性、MoE 路由、高效注意力（例如 FlashAttention）和量化感知训练（quantization-aware training，QAT），可以降低计算与内存需求。在核函数级，融合操作、自定义注意力引擎（例如 FlashInfer）、Tensor Core 利用、分块平铺（block tiling）和异步内存传输，都有助于最大化 GPU 吞吐量。

运行时策略，如动态批处理（dynamic batching）、分页 KV 缓存、CUDA Graphs，以及计算与通信重叠，能在变化的负载下让 GPU 保持饱和。系统编排层可以使用 prefill/decode 分离、智能路由、多租户隔离和自动扩缩（配合热备），来平衡延迟并提升成本效率。

> 许多生产系统使用基于 Kubernetes 的编排，分别运行 prefill 与 decode 两套部署。它们使用入口控制器（ingress controller）根据负载或用户优先级路由请求。并且它们会准备好热备 GPU pod，以便在流量激增时随时拉起。

最后，你应当探索各类部署模式，如地理分布式边缘服务、智能 API 网关批处理、面向模型变体的 CI/CD，以及实时剖析。这将在生产中提供顶级的可靠性与适应性。表 16-2 描述了技术栈各层的一些常见优化方法。

表 16-2. 技术栈各层的常见优化方法

| 技术栈层       | 关键技术                                                                                                                                                                                                                                                                                                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 模型           | 剪枝与知识蒸馏（knowledge distillation），在精度损失最小的前提下缩小模型体积；稀疏性（MoE）以跳过部分计算；高效注意力（FlashAttention）以降低内存占用和中间缓冲；在 FP16/BF16 或 INT4/FP8 下做量化感知训练以增强低精度下的鲁棒性                                                                                                                                         |
| 核函数         | 融合算子核函数（例如 Linear + GELU + LayerNorm）以减少启动开销和内存流量；自定义注意力核函数（FlashInfer）用于块稀疏 KV 和 JIT 编译的核函数；Tensor Core 及专用指令（cp.async、TMA）用于矩阵运算                                                                                                                                                                         |
| 运行时         | 带延迟控制的动态批处理，如 vLLM、SGLang 和 NVIDIA Dynamo 中实现的那样（例如连续批处理，continuous batching），以合并请求；分页 KV 缓存管理，用于灵活分配内存并合并批（vLLM 的 PagedAttention）；CUDA Graphs 和缓冲池化（buffer pooling）以减少每次推理的开销；使用多条 CUDA 流（一条流做数据传输、另一条做计算）来重叠计算与通信。使用基于事件的同步——且仅在需要时使用。 |
| 系统编排       | prefill–decode 分离以消除队头阻塞；智能路由与缓存亲和性以均衡负载和缓存命中；多租户隔离与按用户配额以防止“吵闹邻居”；带热备实例的自动扩缩以掩盖模型加载时间，并接受略微增加的成本，以换取流量激增期间显著更优的延迟                                                                                                                                                      |
| 部署与基础设施 | 地理分布式与边缘部署以降低网络往返时延（RTT）；带请求级批处理、可跨服务器池合批的智能 API 网关；用于以金丝雀模式滚动发布新的量化或核函数优化模型变体的 CI/CD 流水线；高带宽互连（NVLink/NVSwitch 和 InfiniBand），以及 GPU 与 CPU 之间的 NUMA 亲和性以实现最优内存访问                                                                                                   |
| QoS 与扩缩     | 感知 SLA 的动态批处理与尾延迟控制；用 MIG 或流优先级做 GPU 隔离以强制保障 QoS；面向 TTFT、TPOT、利用率和内存带宽利用率的实时剖析仪表盘；根据负载特征动态切换并行方式（TP、PP、DP）                                                                                                                                                                                       |

在做优化时，考虑跨层协同效应很重要。例如，量化（模型层）降低了内存占用，从而无需触发 OOM 错误就能使用更大的批大小（运行时层），这反过来又让编排组件能在每个 GPU 周期合并更多请求。

你还应当以剖析为驱动。持续的剖析应当指导下一步优化哪一层。例如，在完成融合与量化之后，如果 CPU 成了预处理和后处理的瓶颈，就投入更快的分词器，或把部分任务卸载到 GPU。

应用优化时，总有权衡需要考量。逐层 CPU 卸载和高级解码方法等技术会增加复杂度。例如，推测解码会额外引入一个草稿模型，而 Medusa 会额外引入多头并行解码。这些通常保留给极端情形，如超长上下文或延迟波动无常的场景。更轻量的方法，包括稀疏性、批处理和分离，才在生产中贡献了绝大部分收益。

建议采用全栈优化（full-stack optimization）方法，把模型架构、核函数设计、运行时行为、系统编排和部署策略对齐协调。这意味着让你的软件栈保持更新，包括 CUDA、cuDNN、NCCL 等。较新的版本往往包含最新的优化和缺陷修复。

全栈方法降低了技术栈每一层成为瓶颈的可能性。这样，团队就能系统性地消除瓶颈、实现持续的低延迟，并为大规模 LLM 推理最大化硬件利用率。

### 调试正确性问题
