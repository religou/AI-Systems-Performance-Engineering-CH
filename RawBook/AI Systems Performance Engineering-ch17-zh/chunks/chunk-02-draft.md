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
| -------------------------- | ------------------------------------ |
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
| ------------------- | -------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 提示词长度          | 输入提示词中的 token 数量（在任何前缀缓存之后）。                    | 长提示词 ⇒ 计算量更大 → 若长度超过阈值则卸载 prefill。短提示词 ⇒ 在本地处理。                                                                                        |
| 前缀缓存命中        | 提示词已存在于 decode 工作节点 KV 缓存中的程度（来自先前的请求）。   | 大量前缀缓存命中（提示词大部分已缓存）⇒ prefill 实际上更短且更偏访存受限 → 在本地处理。若无缓存命中（全部为新 token）⇒ 计算量大 → 卸载很可能有益。                   |
| prefill 队列长度    | 全局 prefill 队列中待处理任务的数量（即 prefill 工作节点有多繁忙）。 | 若队列很长（prefill 工作节点滞后）⇒ 避免卸载新请求（在本地处理）。若队列为空或很轻 ⇒ prefill 工作节点有余量 → 若满足其他条件则卸载。                                 |
| decode 工作节点负载 | 本地 decode 工作节点的当前负载（正在进行的 decode 任务等）。         | 若 decode GPU 正忙于处理许多 decode 流，卸载有助于并行（把 GPU 从繁重计算中解放出来）。若 decode 大多空闲而 prefill 队列积压，则在本地执行 prefill 以利用可用容量。  |
| 延迟 SLO 紧迫度     | 请求延迟要求的优先级或紧迫程度。                                     | 紧迫的低延迟要求 ⇒ 可能会卸载，以确保提示词尽快被计算（尤其是当本地 decode 繁忙时）。宽松的要求则可能只在本地运行以节省资源（参见第 792 页的“QoS 与早期拒绝策略”）。 |

![图 17-6. 基于从工作节点发出的 KV 缓存事件所接收数据的 KV 缓存感知请求路由](AI%20Systems%20Performance%20Engineering-ch17_images/figure-17-6.png)

这些因素倾向于仅卸载那些能从远程执行中获益的请求，包括长且计算密集的提示词。与此同时，短且命中缓存的提示词则在本地处理，以最小化开销。这些阈值是可调的。表 17-2 中的每个因素都对应一个特定的权衡，如下所述：

_提示词长度_ 我们不希望把时间浪费在卸载微不足道的提示词上。远程执行的开销即便很小，对于比如一个五 token 的提示词也不值得，因为 decode GPU 自己就能很快处理它。卸载只保留给那些会让 decode GPU 长时间被占用的繁重提示词。

_前缀缓存_ 现代推理系统通常会实现一个 KV 缓存，用于存储先前所见上下文的 KV 对，例如对话中较早的轮次或反复出现的样板系统提示词。如果一个新请求的提示词具有与已处理过的提示词重叠的长前缀，那么 decode 工作节点可能已经在内存中持有该前缀的 KV 数据。在这种情况下，它只需为缓存未命中的部分计算提示词的剩余部分。

如果剩余部分很短，卸载的收益就会降低。此外，如果因为完全的前缀命中而整个提示词都已缓存，那么根本不需要任何 prefill 计算。decode 可以立即使用缓存状态继续进行。路由器会通过实际考虑 有效提示词长度 = (prompt_length – prefix_cached_length) 来纳入这一点。

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
