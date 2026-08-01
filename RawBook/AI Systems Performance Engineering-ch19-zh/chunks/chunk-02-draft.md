_2. 选择候选核函数_ 为每个组件确定可用的核函数实现，例如注意力阶段的标准注意力或 flash attention，以及 MLP 阶段合适的 GEMM 算法。

_3. 估算或基准测试_ 对每个候选项做一次快速试跑，即在样本数据上将每种算法执行若干次迭代。

_4. 选择最佳变体_ 根据实测维度，选出执行时间最短——或吞吐量足够——的核函数。

_5. 缓存结果_ 将该选择存入以输入维度或负载特征为键的查找表中。这样，若出现类似请求，无需重跑这些步骤即可知道最佳核函数。

_6. 执行_ 用选定的核函数实现运行该模型层。

> 这类似于数据库查询优化器挑选查询计划。这里的“计划”就是选定的核函数实现。

按照这样的流程，推理运行时会持续自我调优。随着时间推移，系统会针对各种场景构建一个优化路径库，例如短提示与长提示、小批量与大批量等。即时调优的开销被控制在很低水平：要么异步进行——在单独的流中测试新核函数，同时当前推理仍使用默认核函数——要么在低流量时段进行，以免影响延迟。

建议在加载模型时纳入一个初始预热阶段，通过运行各种样本输入来触发自动调优。这可以包含一些极端情形——如最大序列长度、最大批量等——以便引擎为这些情形预先优化核函数。

此外，最好在运行时监控每一层的执行时间。如果某一层因输入特征变化而突然成为瓶颈，那么就该重新审视核函数选择了。

> 一些先进框架甚至使用多臂老虎机（multiarmed bandit）算法，持续探索备选核函数，并随条件变化更新为不同核函数的选择。

简而言之，自动调优把静态核函数转变为自适应核函数。这能针对每一组输入压榨出 GPU 集群的最高性能，而无论负载如何。你可以确信系统一直在自适应调整。

## 动态共享内存分配与占用率感知的核函数选择

与核函数调优密切相关的是对 GPU 共享内存以及整体流式多处理器（streaming multiprocessor，SM）占用率（occupancy）的管理。现代 GPU 的每个 SM 都配有较大的共享内存。通过在运行时根据问题规模——以及当前的共享内存利用率和占用率——动态分配线程，你可以显著提升整个 AI 系统的性能。

借助动态共享内存分配（shared-memory allocation），系统会根据问题规模调整每个线程块（thread block）所用的共享内存量。借助占用率感知的核函数选择（occupancy-aware kernel selection），系统会选择能最充分利用 SM 资源——包括寄存器（register）、共享内存和 warp——的核函数启动参数，以让 GPU 保持繁忙。

选择分块（tiling）的注意力算法时，应在数据复用与 SM 占用率之间取得平衡。例如，设 T 为每个线程块的分块宽度（以 token 计）。由于自注意力（self-attention）本质上是二次复杂度，每个线程块需在共享内存中预留约 _O_(*T*2) 个浮点数，以存放 query、key 和 value 分块。

大分块大小（tile size）（例如 T=256）会从 DRAM 一次性加载每个 key/value 块，并将其复用于多个 query。这将全局内存流量降低到接近每个线程块 O(T) 个浮点数。但由于每个线程块现在会使用大量共享内存，受硬件限制，一个 SM 上一次只能运行少数几个线程块。这会降低占用率。例如，若在 T=256 时每个 SM 只能运行 1 个块，而在 T=128 时能运行 4 个块，那么使用 T=256 时你可能只看到 30% 的 SM 占用率。

小分块大小（例如 T=64）使用的共享内存少得多，这允许每个 SM 容纳更多线程块。这能更好地隐藏延迟并提升利用率。然而，你最终会更频繁地重新加载相同的 key/value 数据，从而增加 DRAM 访问。

最优分块大小 T 取决于若干因素，包括序列长度 L、GPU 的共享内存容量以及 SM 数量。你希望分块足够大，以摊薄 DRAM 读取；又要足够小，以保持足够高的占用率，让许多线程块在 SM 上并发活跃。

实践中，你可以手动挑选少数几个候选 T 值，如 64、128 和 256——并在你的特定硬件上，用能代表你数据集的序列长度 L 对每个值做基准测试。然后选出总体吞吐量最佳的 T 值。不过，与其提前硬编码 T，你也可以在启动核函数前即时计算它，如下所示：

```
int T = choose_tile(L, gpu_shared_mem_per_block, num_sms);
// calculate shared memory in bytes based on the tile size
// (multiplying by 3 for Q, K, and V)
size_t shared_mem_bytes = 3 * T * T * sizeof(float);
numBlocks = ...
MyAttentionKernel<<<numBlocks, threadsPerBlock, shared_mem_bytes>>>(...);
```

这里，T 由序列长度 L、GPU 每个线程块的共享内存上限 gpu_shared_mem_per_block 以及 SM 数量 num_sms 计算得出。然后，每个线程块的共享内存 shared_mem_bytes 会在运行时根据计算出的分块大小 T 计算得到。

随后你就可以用共享内存参数 shared_mem_bytes 启动该 CUDA 核函数。核函数本身会包含如下代码，定义一个 extern **shared** 数组，为每个线程块分配大小为 shared_mem_bytes 的共享内存缓冲区：

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

通过在每次启动时改变 shared_mem_bytes，同一个核函数二进制就能以不同的分块大小运行。选定 T 后，你可以用 CUDA Occupancy API 查询占用率，看每个 SM 能容纳多少个块。

如果占用率过低、每个 SM 只分配到一个块，你可以减小 T。如果你在频繁抖动 DRAM，可以增大 T。这可以实现为一个自动反馈循环：核函数使用 CUDA Occupancy API 或 NVIDIA 的 Data Center GPU Manager（DCGM）以编程方式测量自身达成的占用率——并在后续迭代中调整 T。这样每个注意力层都能基于当前序列长度 L 和硬件限制使用最优配置。

正如我们在第 6 章所见，优化 SM 占用率时你还需考虑每个线程的寄存器用量。使用更多寄存器（例如展开循环）能提升单线程性能，但由于每个 SM 的寄存器文件有限，这会降低整体 SM 占用率。

如果每个 warp 使用很多寄存器，能调度的 warp 数量就会减少。动态运行时可以检测某个核函数是否因寄存器而触及占用率上限，并切换到一个使用更少寄存器的版本——代价是额外的指令。这些底层考量对自适应、高性能的推理服务器至关重要。

动态共享内存调优需要对占用率与吞吐量进行剖析。诸如 NVIDIA Nsight Systems/Compute 和 CUDA Occupancy API 等工具可以展示每个核函数达成的占用率与执行效率。与此同时，DCGM 在系统层面提供实时的 GPU 利用率和 SM 占用率指标。自适应系统可以利用这些信息注意到，例如序列长度为 2,048 的注意力核函数只达成了 30% 的占用率，原因是每个线程块使用了大量共享内存。

这种情况下，系统可以动态切换到某种核函数配置，例如把注意力计算拆分为两趟，从而减少每个线程块的共享内存。如果内存延迟是瓶颈，这将提升占用率——并有可能提升吞吐量。

反过来，如果某个核函数是内存受限且未能充分利用 ALU，那么使用更多共享内存——即便占用率下降——也能通过减少内存停顿来提升有效吞吐量。理解这些权衡很重要——尤其是占用率，因为在某些情况下它不如其他指标那样直观。

建议将核函数设计为在运行时允许可调的共享内存和线程块大小。这样系统就能根据输入和硬件反馈，使调优配置适应运行时条件。例如，它可以为 CUTLASS 之类的库所用的分块大小提供运行时参数和模板参数——这些库正是出于这个原因提供了运行时可调的核函数变体。

你还应持续监控 SM 利用率指标。设想出现大量空闲 warp（例如活跃 warp < 50%）或内存停顿周期（> 70% 停顿）。这表明存在不均衡：要么占用率过低（空闲 warp），要么你的分块大小过小、导致过多的内存流量。因此，你的系统应相应调整以恢复平衡。

对于推理服务，常见做法是为不同问题规模维护一张最优线程块配置的小表。这个映射可以实现为一个 JSON 或配置文件，将序列长度区间映射到启动参数。这使得随着模型和硬件演进能够方便地更新。

例如，每当系统在序列长度 512 下执行注意力时，它会使用 128 threads/block 和 16 KB 共享内存。而对于序列长度 4,096，它会使用 256 threads/block 和 64 KB 共享内存，等等。这把自动调优的概念延伸到了资源分配。

请记住，现代 NVIDIA GPU 为 L1 数据缓存和共享内存提供了一个统一的片上池。而 carveout 控制该池中有多少预留给共享内存、多少给 L1。当核函数需要更大的分块时，用 cudaFuncSetAttribute 调整 carveout，以增大可供共享内存使用的比例。

现代 NVIDIA GPU 为 L1 数据缓存和共享内存提供了一个统一的片上池。NVIDIA 的设备驱动允许你设置 L1 缓存与共享内存的划分百分比，即“carveout”百分比。因此，你可以根据使用场景，将某个 SM 配置为偏向更多共享内存或更多 L1 缓存。例如，当核函数需要更大的分块时，你可以增大可供共享内存使用的比例。

> carveout 是逐核函数（per-kernel）的属性，且仅是一个提示而非保证。它是你可以调节的又一个旋钮，用于平衡占用率与缓存行为。

一个成熟的运行时可以在启动时，使用 cudaFuncSetAttribute() 配合 cudaFuncAttributePreferredSharedMemoryCarveout，为特定核函数切换这个 carve-out 百分比。例如，如果某个注意力核函数使用了非常大的分块、需要更多共享内存，你可能希望把 L1 降到 25%、把共享内存提到 75%（假设 carveout 值从 50% 起）。

> 共享内存与 L1 的 carveout 属性只是提示而非保证。始终把该设置当作提示，并用剖析验证其效果。检查所请求的设置是否真的影响了占用率与缓存行为。

简而言之，动态共享内存与占用率感知技术确保每个 SM 都为给定任务尽可能保持繁忙。这些技术让核函数的资源使用适配于具体使用场景。对于大模型而言这至关重要，否则某些层或批量大小可能会使 SM 利用不足。

## 用推测式 KV 预取加速 TTFT

在实时场景中服务 LLM 时，首 token 时延（time-to-first-token，TTFT）是一个关键指标，因为它衡量系统产出模型响应第一个 token 的速度。这直接影响最终用户的体验。

在大模型中，TTFT 的一个主要来源是在 token 生成开始之前用于建立模型内部状态（如键值（key-value，KV）缓存）所花的时间。回想前几章的内容，注意力 KV 缓存为每一层存储过去 token 的 key 与 value 投影。

推测式 KV 预取（speculative KV prefetching）是一种优化：系统预判第一个 token 所需的数据——并提前将必要数据加载进 GPU。这实际上让 KV 缓存的准备与其他步骤（如计算）重叠。这样 token 生成就能更快开始。推测式 KV 缓存的一个例子是 SpeCache，如图 19-3 所示。

在 SpeCache 中，KV 缓存被压缩（此处为 16-bit）并逐层移出 GPU。这减少了内存占用。生成第一个输出 token 后，会计算一个推测的“下一个”token。与此同时，模型会预取该首个解码步骤所需的相应降精度 KV 对。

在随后的每一步，模型并行解码两个 token，包括实际输出 token 和推测 token。两者的结果都会送入下一步；并且在每一步之前，都会预取推测路径最相关的 top-k 个 16-bit KV 对。这样，两条路径都备好了各自所需的 KV 缓存数据。简而言之，SpeCache 报告称，通过预取降精度 KV 并与计算重叠，可改善 TTFT。

> 只有在验证了你的访问模式与存储层级之后，才集成推测式预取技术。

![图 19-3. 使用 SpeCache 的推测式解码（来源：https://oreil.ly/b21E5）](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-3.png)

由于现代 LLM 层数众多、LLM 上下文窗口不断增大（如今几乎无上限），以及现代“思考型”模型生成的大量推理链，KV 缓存可能极其庞大。现代推理系统常常在 GPU、CPU 内存和 SSD 之间换入换出 KV 缓存，以更好地管理容量——尤其是对于放不进 GPU 内存的超长上下文。

生成新 token 时，一种朴素做法是以同步方式从 KV 数据所在之处（一部分在 GPU、一部分在 CPU）取回数据——然后执行计算以解码下一个 token。这会为第一个 token 增加显著延迟——尤其是当缓存已被换出到 CPU 内存或 NVMe 存储时。

KV 缓存预取（prefetch）通过提前启动 KV 数据传输来提供帮助。一收到用户的提示，服务器就可以开始把必要的 KV 页——以及模型权重——直接拷贝进 GPU 内存。等到模型在 prefill 阶段计算完提示时，生成第一个输出 token 所需的数据已经就位。

具体来说，这种机制只在 GPU 内存中保留当前层的 KV——并把其他层的 KV 卸载到 CPU。它在计算当前层的同时，异步地将下一层的缓存预取进 GPU。此外，它还同时把上一层的缓存写回 CPU。

这种通信与计算的重叠意味着 GPU 很少需要等待数据。其结果是，在 CPU 内存中使用卸载的 KV 缓存对延迟的影响极小。例如，由于额外的数据传输开销，卸载时你可能会看到 tokens/s 吞吐量降低约 5%–10%。

> 重叠可以掩盖 CPU 驻留 KV 的大部分延迟，但由于 CPU DRAM 带宽和 PCIe 开销，通常仍会留有吞吐量代价。请用你自己的批量和序列长度反复剖析。

KV 缓存卸载的一个例子是 Hugging Face 的 Transformers 库中的 OffloadedCache 机制。它可以在调用 generate(cache_implementation="offloaded") 或 generate(cache_implementation="offloaded_static") 时启用。这会用 Transformer 库生成 token，如下所示。这使它成为一项低投入、高收益的优化：

```
# Dynamic, variable-length serving and sliding layers
# (recommended default)
out = model.generate(..., cache_implementation="offloaded")
# Static shapes + torch.compile and CUDA Graphs
# (highest throughput with fixed shapes, use with torch.compile)
# out = model.generate(..., cache_implementation="offloaded_static")
```

在底层，当生成开始时，OffloadedCache 会确保第 1 层的 KV 被移动到 GPU。在第 1 层计算的同时，OffloadedCache 会为第 2 层的 KV 从 CPU 到 GPU 发起一次异步 DMA，依此类推。它始终提前预取一层。

等到前向传播到达第 2 层时，其 KV 已经在本地。这减少了如果我们对每一层都使用同步拷贝时会发生的停顿。既然我们已经描述了 KV 预取，接下来就转向推测式 KV 预取。

推测式 KV 预取的范围超出了常规 KV 预取仅提前一层的预判。设想这样一种推理服务器配置：拥有多个模型副本——或者存在多条可能路径，如 MoE 模型中一个 token 可被路由到若干专家网络之一。

KV 预取在各阶段之间的边界处提供帮助。理想情况下，到 prefill 阶段结束时，所有层的缓存要么已在 GPU 内存中，要么已排队进入 GPU 内存。这直接把 TTFT 降到最低，因为一旦生成开始，模型就不必等待内存传输。

> 建议使用 NVTX markers 之类的追踪工具持续监控你的 TTFT，以测量第一个 token 的解码时间。这将精确测量 TTFT。如果你在解码阶段刚开始时就看到过多的空闲时间尖峰，这说明错失了一次预取机会。

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

在这种设置中，cudaMemcpyAsync 在 prefetch_stream 上运行，而 model.forward() 使用 compute_stream。这让 CUDA 驱动能够将数据传输与计算重叠。你只在实际需要预取的 KV 数据时才同步——即在继续执行消费它的计算之前，等待 kv_ready 事件。该事件在交接点强制实施即时（just-in-time）同步。

> 确保主机缓冲区是固定的（页锁定，page-locked）。否则 cudaMemcpyAsync 可能会串行化，你将得不到期望的拷贝/计算重叠。如果 KV 源在另一块 GPU 上，请使用 cudaMemcpyPeerAsync 并启用对等访问（peer access）。而如果你使用统一内存（Unified Memory，例如 Grace Blackwell、Vera Rubin 超级芯片），可考虑使用 cudaMemPrefetchAsync 提前暂存页面。如果这一模式是可重复的，你也可以用 CUDA Graphs 来捕获这个序列。当预取频繁发生时，这能进一步降低核函数启动开销。

使用单独的流可确保高效的流水线化。在生成一个 token 的同时，下一个 token 的 KV 缓存正在被预取，而不打断计算流。这通过掩盖传输延迟、并让计算单元持续获得数据供给，从而最大化 GPU 利用率。现代 LLM 推理引擎会以分页 KV 缓存的形式自动使用这一点。

最好把权重和 KV 缓存的数据搬运视为整个推理流水线的一部分。正如你应当对计算操作做流水线化，你也应当对数据搬运做流水线化。在当前计算进行的同时，始终让下一份所需数据处于传输途中。KV 缓存压缩是又一个在 KV 缓存层提升性能的选项。接下来我们就讲这个。

## 实时 KV 缓存压缩与策略切换

随着 LLM 在一次会话中生成越来越多的 token，KV 缓存会线性增长。对于长对话、长文档和长推理链，KV 缓存会消耗大量 GPU 内存，而且往往是占用 GPU 内存最多的部分。

KV 缓存是压缩/量化的良好候选对象。与任何形式的压缩一样，KV 缓存压缩会减少其内存占用。实时地做这件事意味着在推理过程中即时执行压缩。

策略切换（policy switching）意味着压缩策略可以根据当前上下文而改变。目标是在需要时释放内存和网络带宽——同时不影响模型精度，也不拖慢涉及 KV 缓存数据的计算。图 19-4 展示了几种不同类型的 KV 缓存压缩算法。

![图 19-4. 不同的 KV 缓存算法，包括不缓存（例如稠密）](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-4.png)

KV 压缩最直接、最简单的方法就是降低其精度。许多框架的 KV 缓存默认使用 FP16 或 BF16，因为 16-bit 通常正是模型用于激活的精度。然而，人们往往可以把 key 和 value 压缩到 8-bit 甚至 4-bit，而对输出质量影响极小——尤其是对于位于 LLM 上下文末尾的 token。

Hugging Face 的 Transformers 库支持一种 QuantizedCache，为 KV 内存提供包括 INT8 和 INT4 在内的支持。这个特性只需一行即可启用——指定 cache_implementation="quantized" 并给出具体位宽。其结果是内存大幅节省，代价仅是量化/反量化操作的少量额外计算。而且在大多数情况下整体模型质量不受损害。

> 当量化与 CPU 卸载同时使用时，确保主机缓冲区是固定的（页锁定）以防止串行化传输。这有助于维持拷贝带宽（例如 PCIe/NVLink）。

接下来，我们讨论动态策略切换。一个策略的例子是：把最后 128 个 token 保持为全精度，而把其余 token 压缩为 4-bit。这样，最近的上下文——很可能对预测下一个 token 影响最大——以更高精度保留，而较早的历史则以更低精度存储以节省空间。

如果模型突然需要关注较早的 token，通常也不会是灾难性的，因为许多 LLM 本就带有近因偏好（recency bias）。这意味着它们优先考虑最近的上下文。这样，输入序列较早的部分对最终输出的影响可能没那么大。

你还可以根据用户提示长度进一步调整这个窗口。例如，对于非常长的提示，你可以使用更大的全精度窗口——或者在 GPU 内存使用超过某个阈值时更激进地压缩。

或者，策略也可以基于内存使用量。例如，策略可以规定：如果 GPU 内存使用超过 80%，就应把整个 KV 缓存压缩为 8-bit。这有助于在长时间生成期间避免 OOM 错误。该策略可以包含多级压缩：系统在轻度压力下把 KV 缓存压缩到 8-bit，然后在极端压力下改为 4-bit 压缩。

借助真正的动态、实时策略切换，引擎可以在 token 生成过程中改用不同的压缩。这种情况下，实现需要同时维护缓存的多种表示。例如，它一开始以 FP16 存储 KV，但同时维护同一数据的 INT8 版本。

系统默认使用 FP16，但如果内存利用率越过某个阈值，它可以开始使用 INT8 版本——配以适当的缩放因子——并释放 FP16 内存以缓解内存压力。之后的注意力读取便会从 INT8 存储中取回反量化后的值。

不过，这需要仔细的同步，以确保压缩版本保持最新、并在需要时已就绪。双缓冲（double buffering）和后台压缩线程之类的技术在这种情况下很有用。

通常可以由一个 CPU 线程使用向量化的 INT8 量化操作异步处理压缩。然后在就绪时把压缩后的块拷贝到 GPU 内存。

> 在安全点（例如一次迭代的结尾）实施实时策略切换。这样，你就能避免计算中途切换，并通过在后台流中执行来隐藏重新量化的延迟。

还有一些其他技术，例如无损压缩，使用熵编码和聚类来压缩激活，且不逐比特地丢失信息。然而，这些实现很复杂，且可能太慢而无法实时完成——即便在 GPU 上也是如此。

也应考虑一些更简单的机制，如分块（chunk-wise）ZFP（一种浮点压缩），甚至通用的基于 CPU 的压缩。然而，迄今为止最简单、最有效且支持最完善的方法一直是量化。

> 在撰写本文时，像 ZFP 这样的无损方法在离线和研究场景中已被评估，但由于相对于量化缓存存在吞吐量约束，在生产级 LLM KV 缓存路径中仍不常见。因此，量化因其在速度与 2–4× 内存缩减之间的平衡，仍是首选方案。

为把质量影响降到最低，你可以尝试逐头（per-head）和逐 token（per-token）的缩放。量化 KV 缓存时，采用逐头、分组（group-wise）缩放比逐 token 缩放更有效。Hugging Face 的 QuantizedCache Transformer 实现会按每个注意力头校准数值范围。

具体来说，QuantizedCache 实现的是逐通道（per-channel）、分组量化，具有可配置的组大小，以及一个把最近 token 保持在原始精度的残差窗口。你通过设置 cache_implementation="quantized" 并把 cache_config 作为字典传入来启用它。你可以计算张量中的最大绝对值，并据此缩放 4-bit 或 8-bit 量化。这本质上是一种 min-max（即基于幅度）的量化。

QuantizedCache 的一个有用实现是 Half-Quadratic Quantization（HQQ）后端。HQQ 提供了一个免校准、即时的量化器，支持广泛的低位格式，包括 2-bit、3-bit、4-bit 和 8-bit。它使用一种鲁棒优化来对离群值和重尾误差分布建模。而且 HQQ 能很好地集成进 Hugging Face Transformers 的 KV 缓存实现。它同时提供 PyTorch 实现和自定义 CUDA 核函数实现，以实现快速推理。

我们可以实现一个动态策略，根据内存压力在 8-bit 量化和 4-bit 量化之间切换。数值的尖锐程度或分布也可以指导这一决策。如果缓存的数值大多较小且方差较低，通常就可以更激进地量化。实时切换压缩策略可以与 Hugging Face 的 QuantizedCache 机制集成。

遗憾的是，Transformers 不支持就地更改已初始化缓存对象的位宽。然而，为实现动态策略，我们的代码可以以小块的方式生成 token，并在内存压力下即时地用新的量化缓存配置开始下一个块。这个实现类似于在遇到内存不足（out-of-memory，OOM）错误时回退到卸载缓存。代码如下：
