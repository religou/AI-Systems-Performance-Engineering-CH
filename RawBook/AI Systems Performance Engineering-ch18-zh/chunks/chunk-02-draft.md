借助这些方法，KV 传输可以缩短到几毫秒量级。这远小于在 prefill worker 上实际计算出该 KV 所需的数百毫秒。因此，prefill → 传输 → decode 这条流水线获得了良好的并行度，因为 decode 几乎可以在 prefill 完成后立即开始——不会有长时间的停顿。

在发送 KV 缓存数据时，务必注意避免碎片化与开销。例如，vLLM 的 PagedAttention 以固定大小的 token 块存储 KV 缓存，通常每块 16 tokens。这些 KV 块相对较小（尽管每块的字节数会随着头数、头维度、层数和 dtype 的增加而变大）。若不加处理地通过 RDMA 发送数千个小 KV 页，将带来过高的开销，因为每次传输都有固定的延迟与协议开销。这会导致带宽利用率低下。

> 现代 LLM 引擎支持多种页大小，例如每块 8、16、32、64 或 128 tokens。较大的页大小可以在通过 RDMA 搬运 KV 时降低传输开销，因为更大的归并缓冲区和更少的工作队列元素（work queue elements，WQEs）会提升持续链路吞吐量。在可能的情况下，为每次 RDMA 写入归并 ≥ 128-token 的页。务必将传输重叠到一条专用的 CUDA 流上。优先使用非阻塞流并使用事件栅栏（event fences）。始终用 Nsight Systems 等工具进行性能分析以确认重叠效果。LMCache 报告称，一个 7.5k-token 的 KV 在 RDMA 上经过归并后从 ~20 ms → ~8 ms。

LMCache 扩展通过在传输前将 KV 页归并为大的连续缓冲区来解决这种低效问题。本质上，它把小块聚集到 GPU 内存中的一个大块里，然后用一次传输发送那个大缓冲区。

例如，如果将一个 7,500-token 的 KV 缓存作为 470 次小传输发送需要 20 ms，那么把它们归并成更大的块（例如 128-token 的页）可将传输时间降到 8 ms。这个简单的批处理优化让网络管道保持充盈，并降低了每包开销。

下面展示如何为快速的 GPU 到 GPU KV 传输配置一个系统。这是 LMCache 的 prefill-decode 模式使用 NIXL 传输通道的一个示例配置：

```
# Prefill server config (lmcache-prefiller-config.yaml)
enable_pd: true
transfer_channel: "nixl"
pd_role: "sender"              # this instance sends KV data
pd_proxy_host: "decode-host"   # PD proxy / decode coordinator
pd_proxy_port: 7500            # control-plane port on the proxy/decoder
# size the buffer to the KV you plan to transfer
# FP8/FP4 KV should shrink it significantly
pd_buffer_size: 1073741824     # 1 GiB transfer buffer size
pd_buffer_device: "cuda"       # buffer stays in GPU memory
```

这里，prefill 服务器被配置为 RDMA 发送方。它以 decode 主机的 7500 端口为目标，并为 KV 传输分配了一个 1 GB 的 GPU 缓冲区。decode 服务器则被配置为该端口上的接收方，并配有一个匹配的 1 GB GPU 缓冲区，如下所示：

```
# Decode server config (lmcache-decoder-config.yaml)
enable_pd: true
transfer_channel: "nixl"
pd_role: "receiver"            # this instance receives KV
pd_peer_host: "0.0.0.0"        # bind address for NIXL peer
pd_peer_init_port: 7300        # NIXL handshake/control port
pd_peer_alloc_port: 7400       # NIXL allocation/data port
pd_buffer_size: 1073741824     # 1 GiB (match sender unless you plan to shard)
pd_buffer_device: "cuda"       # keep buffer in GPU memory
nixl_backends: [UCX]           # UCX backend is sufficient for disagg
```

这一配置使得 prefill 能够将 KV 缓存直接写入 decode GPU 的内存——每次传输最多 1 GB——且无需 CPU 介入。两端都将传输缓冲区保留在 GPU 内存中，以实现零拷贝操作。

在为传输缓冲区确定大小时，从 pd_buffer_size = 1 GB 开始。对于一个具有 700 亿参数、80 层、32 头、128 维头的模型，这大致相当于一个估计为 ~4–8k tokens 的 FP16 KV 缓存。如果提示词超过 ~7.5k tokens，则使用 2 GB。你可以随 dtype 和头数进行缩放：bytes ≈ 2 × L × N × (H × Dh) × bytes_per_val。务必在传输前归并页。这将避免小 IO 的低效。

> 如果你把 KV 缓存量化为 FP8 或 FP4，那么在固定 token 数下所需的传输缓冲区会减小，因为每个 token 的字节数相应减少。因此，你既可以每个缓冲区传输更多 token，也可以相应地减小缓冲区大小。1–2 GiB 的缓冲区适用于许多部署，但应根据上面的 KV 公式来确定大小，并向上取整到 256 MB 边界。如果使用 FP8 或 FP4 KV，你可以按比例缩小缓冲区。始终针对你将要传输的最大归并页组进行验证。为获得最佳链路利用率，优先使用 GPUDirect RDMA 并归并到 ≥ 128-token 的页。

在实践中，可以用下面的 shell 脚本通过 CLI 启动 decode 服务器。这将减少急切模式（eager）碎片化，并促使在更大的缓冲区上进行会合（rendezvous）：

```
# Example decode worker
# (select device by index or UUID)
UCX_RNDV_THRESH=16384
UCX_MAX_EAGER_RAILS=1
UCX_TLS=cuda_ipc,rc,rdmacm,cuda_copy,cuda_ipc,tcp \
CUDA_VISIBLE_DEVICES=1 \
LMCACHE_CONFIG_FILE=lmcache-decoder-config.yaml \
python run_vllm_decoder.py --port 8200
```

你会用类似的方式在另一块 GPU 上以其配置文件启动 prefill 服务器。这些设置确保系统在进行 KV 传输时，跨节点使用 InfiniBand RDMA、节点内使用 NVLink 点对点，而不是标准的 TCP 套接字。

> 对于单节点、多 GPU 的运行，你应启用 CUDA IPC。跨节点运行时，优先使用 RDMA。LMCache/vLLM worker 的一个典型 UCX 配置是设置 UCX_TLS=rc,rdmacm,cuda_copy,cuda_ipc,tcp，并确保在网络结构上应用 RoCE/IB 无损设置（ECN/PFC）。对于节点间 RDMA，可考虑设置 UCX_RNDV_THRESH=16384，以便大的 KV 缓冲区使用会合（rendezvous）、小的 KV 缓冲区使用急切模式（eager）。始终用 ucx_info -f 进行验证。

在 RDMA 和适当的缓冲到位后，交接延迟可以在个位数到数十毫秒之间，具体取决于互连和页大小。例如，对于一个 7,500-token 的上下文，LMCache 测得使用大量小传输时约为 20 毫秒，归并成更大的块后约为 8 毫秒。具体而言，建议在 RDMA 前将 16-token 的页归并成 ≥ 128-token 的大块（slabs）。这将有助于降低每包开销。

简而言之，分离式系统应使用快速互连与智能数据归并，让 prefill → decode 的过渡无缝而快速。最小化交接时间至关重要，因为如果交接很慢，就会抵消一开始将各阶段并行化所带来的收益。

> 在多进程运行中，通过设置 export PYTHONHASHSEED=0 来为 KV 块路由使用确定性哈希。

## 连接器与数据通路设计

在零拷贝优化的基础上，让我们看看 prefill 与 decode 节点如何端到端地协调传输——而不仅仅是搬运比特。prefill 与 decode worker 通常使用调度器或路由器进行通信。在实践中，这个调度器常被实现为一个集中式组件（如 NVIDIA Dynamo 所采用），或一种去中心化的协调方式（如 SGLang 所采用）。

例如，NVIDIA Dynamo 实现了一个全局调度队列，decode worker 将新的提示词任务推入队列，由 prefill worker 消费。在这种设计中，一个 decode 节点将一个请求入队以进行提示词处理，如图 18-6 中的“Put RemovePrefillRequest”（步骤 6）所示。

![图 18-6. NVIDIA Dynamo 中的请求生命周期；decode 拉取提示词，prefill 使用 NIXL 推送 KV](AI%20Systems%20Performance%20Engineering-ch18_images/figure-18-6.png)

一个 prefill 节点接手这个请求，完成后能准确知道应把结果发送给哪个 decode 节点，因为该请求携带了 decode 节点的来源或回复目标 ID。随后，KV 便使用 NIXL RDMA 直接传输到那个 decode worker 的 GPU。

在 vLLM + LMCache 实现中，采用了一种更去中心化的方式。decode 与 prefill 进程为每个请求的 KV 使用一个管道或缓冲区建立一条直接通道。在底层，这可能使用在请求开始时协商的一对一 TCP 或 RDMA 连接。它不使用全局队列，而是每个请求各自建立自己的传输通道。两种方式各有利弊。全局队列在负载均衡和故障处理方面更简单。直接通道则可以最小化排队。

在决定采用哪种模式时，要考虑你的工作负载与基础设施约束。如果你需要健壮的多租户负载均衡与便捷的故障转移，并且能够接受少量排队延迟，那么全局队列模型通常更合适。反之，如果你有严格的尾延迟要求、一组相对稳定的 decode-prefill 配对以及高速互连，那么按请求建立直接通道的方式可以最小化跳数与抖动。

> 在实践中，应针对你预期的请求组合对两种设计进行基准测试。改变提示词长度、并发级别和故障场景，看看哪种设计能为你的 SLO 提供最佳的延迟-吞吐量权衡。

关键的设计目标是让流水线既非阻塞又高吞吐，从而在一个请求正在被 decode 时，另一个提示词可以开始 prefill——与此同时，还有一个请求的 KV 可以在传输途中。这样一来，只要另一个阶段还有工作要做，就没有任何阶段处于空闲。这正是分离化在大规模下提升整体吞吐量的原因——所有阶段都被并行地保持忙碌。

> 通常，prefill 生成的第一个 token 的 logits 往往不会被显式传输，因为 decode worker 可以直接从 KV 重新计算第一个 token 的概率。有些系统确实会传输第一个 token 的输出，以在 decode worker 中省下几百微秒的额外计算。但另一些系统则保持简单，只传输 KV，让 decode worker 重新计算最后一层。

确保这条流水线对故障具有健壮性很重要。如果一个 decode 节点在生成过程中失败，前面讨论过的全局 KV 缓存池可以让另一个节点使用已保存的 KV 从中断处接手。类似地，如果一个 prefill 节点在处理提示词的中途失败，该提示词可以在别处重试。连接器设计应优雅地处理这些故障，从而使一个节点的失败不会让整个请求报错。

> 这些路由器通常使用心跳检查与超时，这样如果一次 prefill 到 decode 的传输停滞，请求就可以被重新分配或安全中止。

## 面向 Prefill 与 Decode 的异构硬件与并行策略

分离化的一个强大优势是：可以自由地为 prefill 和 decode 集群分别选择最适合各自需求的不同硬件——甚至不同的模型并行配置。在统一的单体式部署中，你通常只能为两个阶段使用一种硬件类型和配置。有了分离化，你可以按阶段混搭硬件与策略，如下文所述。

### 计算优化型 vs 内存优化型硬件

prefill 阶段受益于具有高计算吞吐量、大量 TFLOPS、专用 Tensor Core 和高时钟频率的 GPU。它也受益于可观的内存带宽，但除提示词的 KV 缓存所需之外，它并不一定需要庞大的 HBM 容量。

另一方面，decode 阶段既受益于大的内存容量也受益于内存带宽，因为它要处理相当于许多 token 的 KV。它不需要极强的计算能力，但越强越好。

这就带来了为每个阶段使用不同代 GPU 的可能性。例如，一种设计可以为 prefill 集群使用最新的高算力 GPU，而为 decode 集群沿用具有足够内存带宽的旧一代或高性价比 GPU。

这样，我们就避免了把最新的 GPU（例如最新的 Tensor Core）浪费在像 decode 这样并不能充分发挥其潜力的任务上。prefill 任务往往因把 GPU 的数学单元拉满而消耗更多功率，而 decode 任务在同一块 GPU 上消耗的功率则少得多。

在异构硬件之间拆分各阶段可以提升每成本吞吐量和每瓦吞吐量。在 Splitwise 研究中，使用阶段专用硬件让一种配置相较于同构基线实现了 1.4× 的更高吞吐量，同时成本降低 20%。

在另一种旨在固定成本/功耗预算下实现最高性能的配置中，他们在相同成本与功耗下实现了 2.35× 的更多吞吐量。具体来说，该研究为 prefill 使用 4× H100（高算力），为 decode 使用 4× A100（高内存）。这种混合配置在相同成本/功耗下达到了一个 8-GPU 同构系统（全 H100 或全 A100）约 2.35× 的 RPS。

或者，他们发现，为了达到与基线相同的吞吐量，异构系统实际上可以总体使用更少的 GPU（例如用五块或六块代替八块），方法是把 decode 卸载到更便宜的 GPU 上。这凸显了节省成本的机会，并展示了在每种 GPU 最有效之处使用它的价值。具体而言，你可以在每美元算力最高的 GPU（例如 Blackwell 或 Rubin 代）上执行计算受限的工作，并把访存受限的工作分配给带宽合适、更具性价比的旧一代 GPU（例如 Hopper 或 Ampere）。

Splitwise 评估考虑了异构 GPU 之间状态传输的开销。该测试通过 NVSwitch 结构传输 KV 数据，即使在不同代的 GPU 之间也仅产生极小的开销。这表明像 NVSwitch 和 NVLink 这样的高带宽互连能够以对性能几乎可忽略的影响实现 prefill/decode 分离化——即便在混合 GPU 的配置中也是如此。

另一个系统是 HexGen-2，这是一个分布式推理框架，它把异构 GPU 上分离式推理的分配当作一个优化问题来处理。它的调度器会一并优化资源分配、每阶段的并行策略与通信效率。

在像 Llama 2 70B 这样的模型上的实验中，HexGen-2 相较于同价位的最先进系统，在服务吞吐量上展现出最高 2× 的提升（平均约 1.3×）。此外，它在使用约 30% 更少成本的情况下达到了与高端基线相近的吞吐量。这些改进来自于混合 GPU 类型并优化工作拆分。这基本上是一种自动化的方式，去做 Splitwise 在概念上所做的事。

这些结果证实，分离化不仅仅关乎速度。它也关乎效率和以更少资源做更多事。当在云环境中部署推理时，这可以为一个支撑数百万或数十亿终端用户的大型推理服务在 GPU 时间上带来数以百万美元计的显著成本节省。

例如，你可以用 6 块 GPU（prefill + decode 混合）而非 8 块顶级 GPU 来承载相同的流量；你就能为该服务节省大约 25% 的硬件成本。因此，分离化让你在相同硬件上服务更多用户。这很关键，因为 GPU 供应往往有限——尤其是最新的 GPU。

考虑到世界某些地区（包括美国）的功率约束，能效同样重要。Splitwise 通过在低功率 GPU 上运行 decode 任务——以略微降低的速度——展示了更好的能效。

通过把 prefill 与 decode 任务分派到不同的硬件类型上，你可以选择在何处、以何种方式运行每个阶段，从而提升性能并降低成本。分离化提供了这种灵活性，因为各阶段是相互独立的。

简而言之，来自 Splitwise、HexGen-2 及相关异构部署研究的评估表明，分离化除了纯粹的速度之外，还可用于成本优化。通过将硬件与工作负载相匹配，你可以显著降低每次查询的成本，同时在固定预算内提升性能。

对于大规模服务而言，这对于维持其经济可行性至关重要。其权衡包括略微增加的系统复杂度，因为你必须管理多种 GPU 类型。而且由于 GPU 在能力上并不匹配，你在集群配置的灵活性以及在 prefill 与 decode 任务之间动态重新分配 GPU 方面会受到限制。但在许多情况下，为每个阶段使用不同硬件所带来的效率收益可能是值得的。

异构与按阶段专用化的另一种形式，是为每个阶段在 GPU 之间选择不同的模型并行方式（例如张量并行、流水线并行等）。这对于因内存约束而在多块 GPU 上分片的超大模型很有意义。

在传统配置中，你可能以固定的并行策略运行模型，并为 prefill 和 decode 两个阶段都使用张量并行或流水线并行将模型拆分到多块 GPU 上。但对 prefill 最优的并行策略未必对 decode 同样最优。

例如，prefill 阶段是对 _N_ 个提示词 token 的一次大规模前向传播，它受益于高度的并行化。你可以跨许多 GPU 使用张量并行（tensor parallelism，TP）来更快地完成计算并降低 TTFT。

同步各 GPU 的开销被摊薄到可以一次性处理的大量 token 上。这缩短了该阶段的墙钟时间，而这对 TTFT 至关重要。

你甚至可以使用流水线并行（pipeline parallelism，PP）来进一步加速 prefill 并提升吞吐量。这会把模型层拆分到多块 GPU 上，并让提示词流经多个流水线阶段。

另一方面，decode 阶段是顺序的，且每一步对延迟敏感。为单次 decode 使用过多 GPU 实际上会损害每输出 token 时间（time-per-output-token，TPOT）延迟，也称为 _token 间延迟_（inter-token latency，ITL）。这是因为每个 token 步都需要额外的多 GPU 通信开销。因此，加速的潜力有限，因为每次只有相当于一个 token 的计算可供拆分（若使用推测解码则为几个 token）。

分离化使得混合这些方法成为可能：为一个阶段使用 TP、为另一个阶段使用 PP——或对每种技术使用不同的程度。例如，你可以用 TP=8 运行 prefill 以横跨 8 块 GPU 并最小化提示词延迟。然后你可以用 TP=1（即单块 GPU）运行 decode，以最大化每 token 吞吐量并最小化每步延迟。通过这种方式，每个阶段的吞吐量与延迟都可以分别调优，如图 18-7 所示。

![图 18-7. 为 prefill 和 decode 使用不同的并行策略（来源：https://oreil.ly/1-Ti0）](AI%20Systems%20Performance%20Engineering-ch18_images/figure-18-7.png)

在这里，张量并行额外的 all-reduce 通信开销在 prefill 阶段更为突出，因为有大量 token 正在被并行处理。因此，我们选择流水线并行，因为它对我们的 prefill 工作负载更高效。

> 张量并行和流水线并行对 prefill 都可能有效。接下来的示例为 prefill 使用流水线并行。然而，较大的张量并行度在某些集群上可以降低 TTFT。最佳选择取决于网络带宽、集合通信延迟和模型形状。

然而，对于 decode 阶段，流水线并行会导致更多但更小的前向传播，因为 token 在 GPU 之间传递。这需要为仅仅生成单个 token 而在 GPU 内外进行大量数据搬运。因此，我们选择张量并行，因为它更适合我们的解码工作负载。

不过，这引入了一个复杂问题。由于我们模型的并行化方案在 prefill 和 decode 之间不同，KV 缓存的格式也必须不同。例如，如果 prefill 阶段使用 TP = 1（因为它用的是 PP）搭配四块 GPU，那么这四块 GPU 中的每一块都持有一个全尺寸的 KV 张量。

再假设 decode 阶段使用 TP = 4。在这种情况下，每块 GPU 只期望 1/4 的 KV 张量，因为数据应沿模型的隐藏维度拆分。为处理这种情况，像 NVIDIA Dynamo 这样的系统可以在传输期间执行 KV 转置（transpose）或转换。本质上，它把 KV 缓存从 [TP_p parts] 转换并重新排列为 [TP_d parts] 所需的格式，其中 TP_p 是 prefill 的并行度、TP_d 是 decode 的并行度。

Dynamo 包含一个高性能核函数，用于在从 NIXL 读取之后——写入 decode worker 的内存之前——即时完成这个转置。这样，接收端的 decode 就得到了它所期望布局的 KV 缓存数据。

不过，这个转置的开销相较于网络传输可以很小——尤其是考虑到 NVLink 的吞吐量能够快速处理这些数据重组。在这种情况下，它很容易被为每个阶段选用不同优化并行策略所节省的计算所证明是值得的。

让我们探讨一个按阶段并行的示例。设想一个大模型，我们可以对其应用各种并行方案：张量（TP）、流水线（PP）、数据（DP）、序列（SP，即跨 GPU 拆分序列）等。我们可以选择彼此独立的并行配置。表 18-1 展示了 prefill 阶段的一个示例并行配置。

表 18-1. Prefill 并行示例

| 并行策略   | 符号 | 取值     | 说明                                                                                  |
| ---------- | ---- | -------- | ------------------------------------------------------------------------------------- |
| 张量并行   | TP_p | 2        | 将模型的权重张量拆分到 2 块 GPU 上，以在可控的通信开销下将 prefill 延迟减半。         |
| 流水线并行 | PP_p | 2        | 将模型层划分为两个流水线阶段，为深层模型让微批（microbatch）流经每个阶段。            |
| 序列并行   | SP_p | 1        | 不跨 GPU 拆分输入序列（不做序列分片），除非处理极大的上下文。                         |
| 上下文并行 | CP   | 1        | 在优化后能装入内存时，将整个上下文保留在单块 GPU 上（不做上下文级划分）。             |
| 数据并行   | DP_p | 1 (or 2) | 每块 GPU 使用一个模型副本（或用两个副本，通过权重复制在批处理提示词上使吞吐量翻倍）。 |

这些数值在仍然使用多块 GPU 加速大提示词的同时，最小化了 GPU 间开销。接下来，让我们看看 decode 的一个示例并行策略，如表 18-2 所示。

表 18-2. Decode 并行策略示例

| 并行策略   | 符号 | 取值                               | 说明                                                                                                                                     |
| ---------- | ---- | ---------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 张量并行   | TP_d | 1 (default) ... N (number of GPUs) | 默认使用 TP_d = 1 以求简单并将同步开销降到最低。当在小批次上处理微小 GEMM，或单块 GPU 无法容纳模型时，TP_d = N（GPU 数量）可以提升效率。 |
| 流水线并行 | PP_d | 1                                  | 对于单 token 解码，流水线并行会增加气泡，因此 PP_d = 1 可避免空闲阶段。                                                                  |
| 序列并行   | SP_d | 1                                  | 跨 GPU 拆分输出序列并不常见。SP_d = 1 让每条 decode 流保持本地，除非要处理极长的输出。                                                   |
| 数据并行   | DP_d | 1                                  | 每条 decode 流在每块 GPU 上使用一个模型副本。用彼此独立的副本来处理并行请求，而不是为单条流做复制。                                      |

> 例如，如果模型无法装入单块 Blackwell B200，对于 decode 应优先使用 TP_d = 2 或 4，而不是使用 PP。这将有助于避免流水线气泡。

理想情况下，每条 decode 流运行在单块 GPU 上，并避免跨 GPU 开销。只有当模型能装入 GPU 内存时才有可能做到这一点。在这种情况下，TP*d = 1，意味着 decode 期间不使用任何张量并行。如果模型无法装入内存，你可以提高张量并行度（例如 TP_d = \_N*，其中 _N_ 是 GPU 数量）。

如果系统正在发出微小的 GEMM 来处理小批次，提高张量并行也很有用。这是因为把小矩阵乘法分布到多个设备上可以将通信延迟隐藏在计算之后，并有可能产生更高的整体吞吐量。

这些只是示意性的取值，但要点在于：分离化允许你在提示词一侧单独配置资源以达到某个 TTFT 目标。与此同时，你可以独立地调整 decode 一侧的资源，以在向终端用户流式返回 token 时达到吞吐量和延迟目标。

这样，两种并行策略就不会相互干扰。在统一系统中，如果你试图这样做，就不得不选一个对两个阶段都不理想的折中策略。

一些推理引擎允许你在 prefill 和 decode 之间使用不同的精度。例如，你可以以较低精度（如 FP8、INT8 或 FP4）执行 prefill 以加速它。与此同时，如果为了更好的生成准确度而有需要，你可以以较高精度进行 decode。

一般来说，你会让两者以相同精度运行，以便 prefill 阶段计算出的 KV 缓存在 decode 阶段可用。然而，你可以应用一种类似于上一节所述并行转换的转换。你会选择在发送前量化 KV 并将其从 FP16 压缩到 INT8/FP8/FP4。然后如有需要，在接收端再将其转换回来。

例如，你可以选择通过网络发送较低精度以加速传输。或者，你可以根据可用的 FLOPS 等，选择在发送端或接收端执行转换。

这些都是高级思路。但它们凸显出：在分离式配置中，几乎每个方面——硬件类型、GPU 数量和精度——都可以为每个阶段独立调优。

## GPU-CPU 协同的混合 Prefill

到目前为止，我们一直假设 prefill 和 decode 两个阶段都运行在 GPU 上——而且可能是不同类型的 GPU。然而，在极端规模下——或面对极大的模型与提示词时——值得评估 CPU 是否能为 GPU 分担压力。

现代 CPU 在神经网络计算上远比 GPU 慢，但它们带来了其他优势，例如充裕的 RAM、不争用 GPU 内存带宽，以及灵活处理 GPU 可能不太擅长的任务的能力，比如极长的序列以及分词、填充等非 transformer 操作。

采用混合 prefill 策略时，一部分 prefill 计算在 CPU 上完成。一种场景是为超长提示词进行 CPU 卸载。设想一个来自大型文档附件、包含数万个 token 的提示词。即便是强大的 GPU，也可能因内存约束而难以处理如此大的提示词。

在极长提示词的情况下，系统可以选择在一个拥有大量 RAM 的 CPU worker 上执行模型的初始若干层，以容纳这条长序列。随后它会把中间结果流式传输到 GPU——或者，如果延迟不成问题，甚至在 CPU 上执行整个 prefill。之后 decode GPU 便从该 CPU worker 接收巨大的 KV 缓存。

> 虽然在交互式推理中并不常见，但一些批处理或离线流水线，比如长时间运行的“Deep Research”作业，可以对非常长的文本使用 CPU 预处理。

CPU 卸载的一个实际用途是处理后台或低优先级的 prefill 任务。例如，一个 LLM 服务可能允许提交非常大的提示词，用于对非交互式请求进行离线处理。这些可以被分配给仅使用 CPU 的 worker，它们最终馈入一块 decode GPU 以进行快速 token 生成。由于 CPU 较慢，延迟会很高，但因为这是离线作业，这或许是可以接受的。而且这种配置能为更具交互性的工作负载释放出 GPU 资源。

混合 prefill 在像 Grace Blackwell 这样的 CPU-GPU 超级芯片（superchip）上更为常见，其上的芯片到芯片互连极快。它还利用了 CPU 内存相较于 GPU 内存极为庞大这一事实。

设想在 CPU 内存中存储一个巨大的 KV 缓存，并从 GPU 快速访问。混合 prefill 会利用 CPU 的内存来缓冲或预处理输入 token，而 GPU 则专注于繁重的 transformer 层。

一枚 Grace Blackwell 超级芯片可以通过让 CPU 管理内存和初始层、让 GPU 处理序列各分块上的密集注意力，来应对庞大的上下文。Grace CPU 还可用于将装不进 HBM 的 KV 缓存数据溢出到 CPU DDR 内存中。这实际上扩展了 GPU 所能支持的上下文长度。

你可以跨设备切分 transformer：在 GPU 上运行前 _N_ 层，在其中处理大部分 token 并使序列基本被压缩。然后你会把接下来的 _M_ 层卸载到拥有大量内存的 CPU 上，最后再把剩余的层拿回 GPU 以生成最终输出。

这种分层技术增加了可观的数据搬运与编排复杂度——只应在极少数情况下才被证明是值得的，比如超长上下文或严重的 GPU 内存限制。无论如何，它展示了在极端推理场景中你可以如何突破硬件边界。

在我们的分离式架构中，引入 CPU 意味着引入第三类 worker：CPU prefill worker。调度逻辑于是可以在三个选项中做选择：GPU prefill worker、CPU prefill worker，或本地 decode GPU 的 prefill。这个决策会取决于提示词长度或优先级等因素。

例如，策略可能是：如果 prompt_length > 5,000，就路由到 CPU prefill worker，明知它会慢，但至少不会占用 GPU 且能使用大内存。decode 阶段随后会为 KV 等待更久。或者在极端情况下，如果确实是离线的，decode 也可以在 CPU 上完成。

一般来说，CPU 卸载会增加 TTFT，因此不用于普通的延迟敏感请求。它更像是一种可扩展性与安全网特性。若被采用，系统应监控这条路径被选用的频率，因为频繁的 CPU 卸载可能反而表明需要更多的 GPU 容量或模型优化。

CPU 卸载还让系统能够处理诸如超长输入或在 GPU 全忙时的突发流量等边缘情况。它通过回退到较慢的 CPU 而非彻底失败来做到这一点。然而，请记住：如果处理过慢并超出 SLO 要求，最好还是快速失败。
