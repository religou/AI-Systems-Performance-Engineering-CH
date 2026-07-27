此外，升级时应始终查阅 NCCL 的发布说明。新版本往往带来各种优化——尤其是在有新网络硬件问世时。升级 NCCL 之后也务必进行测试，因为默认设置和性能可能发生变化。通常，NCCL 的性能会随每个新版本而提升。但默认值有时也会变化，如果你没有显式固定 NCCL 的环境变量，就需要重新调优你的系统。

### 剖析与调试 NCCL

NCCL 支持异步错误处理，并在网络错误等情形下支持故障转移（failover）。要启用异步错误处理，可设置环境变量 NCCL_ASYNC_ERROR_HANDLING=1。而在调试 NCCL 时，务必同时启用 NCCL_DEBUG=WARN 或 INFO。这样你就能排查诸如 rank 不匹配或 socket 配置错误之类的常见问题。

调试 NCCL 时还可以使用 NCCL profiler 插件 API。该 API 让你能够监控 GPU 通信的内部时间线，并精准定位系统中任何滞后的设备或瓶颈。NCCL profiler 插件 API 的设计目标，正是解决那些随着 GPU 集群规模扩大而越来越难以诊断的性能问题。

NCCL profiler 插件可以通过 NCCL_PROFILER_PLUGIN 接口动态加载，并由支持它的工具集成。NCCL_PROFILER_PLUGIN 环境变量以类似于其他 NCCL 插件的方式，控制该插件的加载和初始化。

NVIDIA 创建这套灵活的 API，是为了简化第三方剖析工具（如 PyTorch Kineto）与 NCCL 的集成，并确保在运行时以清晰、分层、低开销的方式监控和捕获复杂的通信活动。如果未启用 NCCL 插件，PyTorch 的 Kineto 也可以借助 CUPTI 和 NVTX 来采集 NCCL 活动。

> NCCL profiler 插件与 NVIDIA 的工具以及 PyTorch/Kineto 剖析器等第三方剖析器捆绑在一起。可用它来给出 all-reduce 操作的时间线视图。

一旦加载，NCCL profiler 插件会配置一个事件激活掩码（event activation mask）——这是一个 32 位整数，其中每一位对应一种不同的 NCCL 事件，例如分组事件、集合事件、点对点事件以及各种与 proxy 相关的操作。这种结构天然地形成了事件的层级，有助于以有意义的方式呈现详细的性能信息，并快速定位问题。

NCCL profiler 插件 API 定义了五个函数回调（callback）。init 回调通过提供一个不透明的上下文并确定应剖析哪些事件来完成插件的初始化。startEvent 回调从 NCCL 接收一个事件描述符并分配一个新的事件对象，返回一个供 NCCL 用于后续操作的不透明句柄。

stopEvent 回调标记一个事件的完成，以便回收其资源。recordEventState 回调允许插件在事件经历不同状态转换时对其进行更新。finalize 回调在剖析完成后释放与该剖析器上下文关联的所有资源。

### 在网 SHARP 聚合

当使用支持在网计算（in-network computing）的高级网络硬件时，例如 NVIDIA 的可扩展分层聚合归约协议（Scalable Hierarchical Aggregation and Reduction Protocol，SHARP），可以通过将集合操作的一部分卸载到网络本身来获得额外的性能提升。

SHARP 是一种 InfiniBand 在网归约技术，与 Quantum 系列 InfiniBand 交换机配合使用，需借助 NCCL-SHARP 插件。在 NVLink 域中，与之类似的能力是 NVLink SHARP（NVLS），它在 NVSwitch 结构（fabric）内部卸载集合操作。在现代 NVLink Switch 域（如 NVL72）中，NVLS 加速集合操作，并支持在整个域内高效地执行 all-to-all 和 broadcast（例如 72-GPU 的 NVL72 域）。

具体而言，SHARP 使得 all-reduce 之类的集合操作能够由网络结构部分地计算。当来自多个 GPU 的数据流入交换机时，它会对数据进行归约/聚合（例如求和），并分发部分归约后的结果。这样一来，每个 GPU 便无需在彼此之间冗余地传输大量中间结果。这减少了每个 GPU 必须处理的数据总量，从而降低了大型 MPI 和 NCCL 集合操作的延迟。

具体来说，对于环形 reduce-scatter 操作，每个 GPU 通常要在 (n−1) 跳中接收 B (n−1)/n 字节。而借助在网归约，交换机只向每个 GPU 聚合并返回 B/n。这使得每个端点的接收量为完整环形方案的 1/(n−1)。

对于 all-gather 操作，NVLS 的硬件多播（multicast）让每个 GPU 只需发送一次它的 B/n 分段，再由网络进行复制。这将发送方的数据量相比完整环形方案减少了 1/(n−1)。而当你把多播 all-gather 与在网 reduce-scatter 重叠起来时，对于带宽受限的阶段，端到端的阶段时间可以下降约 1/2，因为聚合和复制工作由网络而非端点来完成。在这种情况下，一次分片交换的有效墙钟时间是两个操作的最大值，而不是二者之和。

简而言之，NVLS 意味着每个端点的数据更少、串行跳数更少。这带来了更高的有效带宽，并缩短了分片式训练/推理阶段中的停顿时间。

> all-gather 不涉及算术归约，因此 NVLS 主要通过执行多播复制来提供帮助。加速比取决于拓扑和消息大小，但它小于 NVLS 用于 all-reduce 和 reduce-scatter 时的性能提升。

NCCL 可以通过使用 NCCL RDMA SHARP 插件，配合交换机上的 SHARP 固件，以及在管理服务器上与 Subnet Manager 一同运行的 SHARP Aggregation Manager，将 all-reduce 等集合操作卸载到支持 SHARP 的 InfiniBand 结构上。此外，每台主机都必须加载 GPUDirect RDMA 内核模块。一旦结构和主机完成配置并选定了 NCCL RDMA SHARP 插件，NCCL 就能把符合条件的集合操作卸载给 SHARP。SHARP 对性能的影响可能非常显著。在某些情况下，NVIDIA 报告称，在大规模 AI 系统上使用 SHARP 可使 all-reduce 获得 2× 到 5× 的加速。

SHARP 带来的收益在拥有大量 GPU 和计算节点、网络通常成为瓶颈的大规模场景下更为明显。在较小的集群上，比如两到四个 GPU 计算节点，你可能不会注意到那么大的性能提升。但在 32 个节点上，SHARP 可以通过削减整体通信步数，显著降低集合操作的延迟。

> SHARP 默认并未启用。它必须通过插件选择或策略来配置。你可以用 NCCL_SHARP_DISABLE=1 将其禁用，以便进行 A/B 测试来验证其性能影响。不过，建议保持其启用状态，以在大规模场景下改善 all-reduce 延迟。

使用 SHARP 通常不需要改动代码。可以通过查看 NCCL 日志（NCCL_DEBUG=INFO）来验证 SHARP 是否在使用。如果 SHARP 正在被使用，日志中会提及它。此外还有一些诊断工具（如 ibv_devinfo 等）可用于检查某设备是否支持 SHARP。

总之，SHARP 把归约计算搬进了网络。这进一步印证了现代系统设计如何模糊了计算与通信之间的边界。对性能工程师而言，如果你的集群运行在高端 InfiniBand 网络上，就值得检查一下 SHARP 是否已启用并被利用。它可以充当一个“涡轮加速按钮”，为海量 all-reduce 操作提供更快的扩展和更高的效率。SHARP 以网络为中心的优化，补足了 NCCL 以 GPU 为中心的优化。

值得一提的是，截至撰写本文时，SHARP 主要是一项 InfiniBand 技术。虽然 NVIDIA 的 Spectrum-X 以太网平台通过拥塞控制和自适应路由改善了 all-reduce 性能，但截至撰写本文时，它仍未像 InfiniBand 上的 SHARP 那样暴露驻留于交换机的归约引擎。公开资料强调的是端到端遥测和拥塞控制，以在大型域中改善 NCCL 性能；它并未暴露与 SHARP 类似的、驻留于交换机的归约引擎。

SHARP 有时会在交换机上带来额外的内存开销。而且由于交换机上用于执行归约的缓冲区数量有限，包含许多 MB 或 GB 消息的超大集合操作如果超出硬件限制，可能会回退到常规方法。

> 建议持续监控 NCCL 日志，并设置告警，以便在它因内存压力随时间累积而开始回退到非 SHARP 聚合时及时获知。

### 持久化 NCCL 用户缓冲区与零拷贝注册

NCCL 支持用户缓冲区注册（user buffer registration），它让集合操作可以直接在你的张量缓冲区上运行，而无需内部暂存（staging）。这减少了拷贝和内部通道压力。

> 持久化 NCCL 用户缓冲区（user buffers）对于在节点内（NVLS）和节点外（InfiniBand）两种场景下都走出使用 SHARP 的最佳路径至关重要。零拷贝（zero-copy）注册可以加速集合操作，并降低 SM/通道的占用。

你可以使用显式的 ncclCommRegister() 和 ncclCommDeregister() 来注册和注销持久化 NCCL 用户缓冲区。如果一次通信中的任何 rank 使用了已注册的缓冲区，那么所有 rank 都必须使用。此外，对于某些算法，相对于缓冲区头部的偏移量必须在各 rank 之间保持一致。

## NVIDIA 的 NIXL 与分离式推理

NCCL 擅长模型训练中常用的多对多分组通信模式，而现代大规模 AI 推理则带来了新的通信需求。NVIDIA 的 NIXL（NVIDIA Inference Xfer Library）是一个于 2025 年初发布的开源、高吞吐、低延迟的点对点（point-to-point，P2P）通信库。

NIXL 的设计初衷正是加速大规模 LLM 的分布式与分离式推理（disaggregated inference）。我们将介绍分离式推理如何把推理的不同阶段拆分到各自独立的 worker 上。分离式推理利用 NIXL 在这些跨 GPU 的阶段之间进行快速数据交换，同时将延迟和开销降至最低。

> 分离式推理和 NIXL 是在一个由多节点组成的集群上高效服务巨型模型的最佳实践。

NIXL 是 NVIDIA 开源 Dynamo 推理引擎的核心组件之一。NIXL 精简了一对一和一对少的数据传输，例如以最小的延迟和开销搬运由各分离阶段共享的键值（key-value，KV）缓存。它补足了主要用于多对多集合操作的 NCCL。

NIXL 提供了一套一致的异步 API，用于在 GPU、CPU、SSD 和共享网络存储之间搬运数据。它总是为每一块被搬运的缓存数据选择最快的路径进行放置。这一层级关系在图 4-5 中以 NVIDIA Dynamo 的 KV Cache Manager 为背景加以展示，该组件利用 NIXL 为每一次 KV 缓存传输选择当前可用的最快路径。

![图 4-5. NVIDIA Dynamo 分布式 KV Cache Manager 将访问频率较低的 KV 缓存卸载到更经济的内存层级（source: https://oreil.ly/nsxdl)](AI%20Systems%20Performance%20Engineering-ch4_images/figure-4-5.png)

在扩展 LLM 推理时，在由 GPU、CPU、计算节点和机架组成的集群中，高效地在各对等端（peer）之间传输大型数据缓存（如 Transformer 注意力机制的 KV 缓存）非常重要。例如，借助 NIXL，推理引擎可以以极小的开销将一个大型（如 100 GB）KV 缓存从某个 GPU 卸载到某个对等端，其间使用 NVLink/InfiniBand。这样便释放出该 GPU 去处理新的请求。在服务具有大上下文窗口的 LLM 时，这一点至关重要。

NIXL 利用 GPUDirect RDMA 在跨节点的 GPU 之间直接搬运数据，完全绕过主机内存。实际上，是支持 RDMA 的网卡（或 DPU）在 GPU 显存之间直接完成传输。这正是延迟如此之低的原因。CPU 并不参与数据路径。

NCCL 仍用于同步的集合操作，但 NIXL 聚焦于 LLM 推理系统中常见的高效一对一或一对多数据传输场景。NVIDIA Dynamo 在其预填充（prefill）和解码（decode）这两个分离式推理阶段中广泛使用 NIXL，接下来我们会详细介绍。

NCCL 仍然是大规模训练中常见的多对多集合操作（如 all-reduce）的标准方案。而 NIXL 针对的是大规模推理中常见的一对一或一对少数据传输，例如搬运 KV 缓存数据。

NIXL 在 NVIDIA Dynamo 推理框架中的应用，展示了它在多节点 LLM 服务场景下带来的吞吐提升。NIXL 是对 NCCL 的补充——而非替代——用于高性能推理流水线。

### 分离的预填充与解码推理阶段

我们会在后面的章节更深入地探讨一个高度调优的推理系统的性能细节，但在进一步讨论 NIXL 之前，理解一点背景很重要。基于 Transformer 的模型的推理路径实际上被拆分为两个不同的阶段：预填充与解码。

第一个阶段——预填充——通常是计算受限（compute bound）的，因为它使用大量矩阵乘法，从传入的请求数据（即 prompt）构建 KV 缓存。第二个阶段——解码——通常是内存吞吐受限（memory-throughput bound）的，因为它需要从 GPU HBM 内存中收集模型权重，以计算下一组 token（即 completion 或 response）。

这种预填充/解码的拆分已在常见推理引擎 vLLM、SGLang 以及 NVIDIA 的 Dynamo 和 TensorRT-LLM 中实现。预填充（prompt 摄入）创建 KV 缓存，解码（生成）使用这份缓存。在这一工作流中，NIXL 专门加速 KV 缓存在节点之间的传输。图 4-6 对比了传统的“单体式（monolithic）”服务模型与“分离式（disaggregated）”服务模型，后者将两个阶段运行在不同的基于 GPU 的计算节点上，以提升规模并最大化吞吐。

![图 4-6. 分离式服务将预填充与解码阶段分置于不同的 GPU 集群上（source: https://oreil.ly/nsxdl)](AI%20Systems%20Performance%20Engineering-ch4_images/figure-4-6.png)

图中上方展示的是传统配置，其中每个 GPU 节点同时处理预填充（计算受限）和解码（内存受限、I/O 受限）两个阶段。下方是分离式服务配置，它将预填充 worker 放在一个 GPU 集群中，将解码 worker 放在另一个 GPU 集群中。

预填充集群中的 GPU 为输入序列生成 KV 缓存，并使用 NIXL 将其传输到解码集群中的某个 GPU。这种专业化分工带来了更高的整体吞吐，以及更高级的扩展配置。

在这类场景中，KV 缓存（在长 prompt 下可达数十 GB）必须近乎实时地、无缝地从一个处理单元流转到另一个处理单元。这样，文本生成才能以终端用户察觉不到的速度进行。

那些让数据途经 CPU 内存、乃至途经存储的传统方法，无法满足所需的速度和低延迟体验。NVIDIA 正是为了解决这一具体场景而打造了 NIXL。NIXL 让多节点推理得以扩展，而不会被互连延迟所拖累。

我们真正想要的，是各组件之间高带宽的 GPU-to-GPU 直接传输。而且我们希望这种通信能与计算重叠。这样，目标 GPU 就可以在从源 GPU 接收下一组输入 token 的 KV 缓存的同时，开始计算下一个 token。

NIXL 提供了一个直接通道，可跨计算节点、乃至跨机架，将数据从一个 GPU 传输到另一个 GPU 或一小组 GPU。系统会审视可用的路径，并始终选择能最快把数据送达的那一条。

这种智能路由类似于 NCCL 的路径选择，但针对推理模式做了优化，包括一对一的大消息传输。例如，在一个 GB200/GB300 NVL72 机架内，NIXL 会优先利用 NVSwitch 网络；而在 NVL72 机架之间，它会根据所支持的情况自动切换到 InfiniBand 或以太网 RDMA。

简而言之，NIXL 会自动选择最快的通道，无论是同一板卡上的 NVLink/NVLink-C2C、跨机架域的 NVSwitch、机架之间的 InfiniBand/RoCE，还是直接的 NVMe 存储访问。

### 面向 KV 缓存传输的智能互连路由

传统上，人们可能会通过 CPU 把这些数据从一个 GPU 传到另一个 GPU，但正如我们已经讨论过的，这样太慢了。另一种提升性能的选择是要求源 GPU 和目标 GPU 位于同一个计算节点上，但这会限制我们的扩展灵活性。NIXL 正是为解决这一问题而生。它的设计目标，是在必要时跨 GPU、跨计算节点、跨机架，对 KV 缓存这类大负载进行直接的 GPU-to-GPU 传输。

NIXL 以高带宽运行，并尽可能地将通信与计算重叠。这让目标 GPU 可以在接收来自源 GPU 的 KV 缓存的同时，开始生成下一组 token。

此外，NIXL 与互连无关（interconnect-agnostic）。如果 GPU 位于同一计算节点，它会使用 NVLink；同一计算节点内也会使用 NVSwitch；跨节点则使用带 RDMA 的 InfiniBand 或以太网；必要时甚至会使用 PCIe 或 NVMe。与 NCCL 类似，NIXL 总是会选择最快的互连来路由数据传输。它还支持以统一的方式在不同内存层级之间传输，横跨 GPU HBM、CPU DRAM，甚至直达 NVMe SSD！

### 带回调的 NIXL 异步 API

从开发者的角度看，NIXL 提供了一套直截了当的 API。你提交一个传输请求，附上指向数据的指针和一个目的地——可以是 GPU、CPU，或像 Amazon S3 这样的存储目标。NIXL 会尽可能快地把该数据传输过去。

例如，一个 NIXL 传输请求可以把一段 KV 缓存发送到另一个 GPU、一块 CPU 主机内存缓冲区，甚至是一个对象存储服务。而且它可以在同一套 API 内完成所有这些操作，如图 4-7 所示。

![图 4-7. NIXL 架构（source: https://oreil.ly/nsxdl)](AI%20Systems%20Performance%20Engineering-ch4_images/figure-4-7.png)

这种模块化设计意味着 NIXL 也可以采纳未来的传输方式。例如，它可以纳入即将出现的协议或更快的存储级内存，而无需改动面向用户的 API。在底层，NIXL 会使用任何合适的后端来协调数据搬运。

NIXL 在不同层级之间高效搬运数据，例如 GPU HBM、CPU 内存（DRAM）、文件存储（NVMe SSD）和对象存储。它提供单一、统一的 API，自动选择最快的传输方式（例如经 NVLink 的 GPU → GPU、借助 GDS 的 GPU → NVMe SSD，或 NVSwitch 结构）。这样，在卸载 KV 缓存分段时，你总能获得接近线速（line-rate）的性能。

在底层，NIXL 隐藏了所有这些复杂性。你用 registerMem 注册内存，用 trim 获取传输描述符，用 prepXfer 准备一个非阻塞请求，再用 postXfer 提交它。NIXL 会决定是执行直接的 PCIe 或 NVLink 拷贝、一次 RDMA 传输，还是像 GPUDirect Storage 这样的存储路径。

NIXL 库是非阻塞的，它返回一个请求句柄，你通过 checkXfer 轮询该句柄来检测完成。这种模式以极小的 CPU 开销将通信与计算重叠。NIXL 的非阻塞 API 让下游 kernel 可以消费传入的数据，而不必阻塞在传输本身上。例如，目标 GPU 可以在收到的 KV 缓存块一到达就开始消费它们——即使其余的块仍在传输途中。

nixlAgent 是 NIXL 的核心传输对象。它封装了端点配置、内存注册和后端选择，还管理元数据、连接信息，以及与其他 agent 之间往来的异步传输请求。

一次传输需要两个 agent，因为每个 nixlAgent 实例代表传输中的一个端点。源 agent（agentSrc）为数据的来源方封装了上下文、内存注册和后端。目标 agent（agentDst）则为接收方做同样的事。

每个端点各有一个 agent——agentSrc 和 agentDst——NIXL 会在这两个端点之间协商出最优路径，并管理它们各自独立的资源和请求生命周期。下面的代码展示了这些 NIXL agent 如何在源 GPU 和目标 GPU 之间传输数据：

```
// NIXL 0.5.x style example: nonblocking VRAM->VRAM transfer between two agents
#include <nixl.h>
#include <nixl_types.h>

#include <cuda_runtime.h>
#include <iostream>
#include <thread>
#include <vector>
#include <cassert>
#include <cstdint>

int main() {
    // 1) Configure agents. Prefer UCX for GPU<->GPU,
    // allow GDS if you later target storage.
    nixl_agent_config cfg{};
    cfg.backends = {"UCX"};  // {"UCX","GDS"} if you also plan storage transfers
    cfg.thread_safe = true;  // thread-safe mode added in early 0.2.x

    // 2) Create source and destination agents
    nixlAgent agentSrc("srcAgent", cfg);
    nixlAgent agentDst("dstAgent", cfg);

    // 3) Allocate simple test buffers on the same GPU for illustration
    int deviceId = 0;
    cudaSetDevice(deviceId);
    const size_t bytes = 1 << 20; // 1 MiB

    void* d_src = nullptr;
    void* d_dst = nullptr;
    cudaMalloc(&d_src, bytes);
    cudaMalloc(&d_dst, bytes);

    // 4) Build registration descriptors for VRAM
    //    Each descriptor uses an address, length, and associated device
    nixl_desc_t srcDesc{};
    srcDesc.addr      = reinterpret_cast<uintptr_t>(d_src);
    srcDesc.len       = bytes;
    srcDesc.devId     = deviceId;
    srcDesc.seg       = VRAM_SEG;

    nixl_desc_t dstDesc{};
    dstDesc.addr      = reinterpret_cast<uintptr_t>(d_dst);
    dstDesc.len       = bytes;
    dstDesc.devId     = deviceId;
    dstDesc.seg       = VRAM_SEG;

    std::vector<nixl_desc_t> srcList{srcDesc};
    std::vector<nixl_desc_t> dstList{dstDesc};

    // 5) Register memory with each agent and trim to xfer descriptors
    auto srcRegs = agentSrc.registerMem(srcList);
    auto dstRegs = agentDst.registerMem(dstList);

    auto srcXfer = srcRegs.trim();  // metadata-free descriptors used for xfer
    auto dstXfer = dstRegs.trim();

    // 6) Prepare a WRITE from srcAgent->dstAgent, then post it (nonblocking)
    nixlReqH reqHandle = nullptr;

    // prepare + post
    if (agentSrc.prepXfer(NIXL_WRITE, srcXfer, dstXfer, "dstAgent", reqHandle)
      != NIXL_SUCCESS) {
        std::cerr << "prepXfer failed\n";
        return 1;
    }
    if (agentSrc.postXfer(NIXL_WRITE, srcXfer, dstXfer, "dstAgent", reqHandle)
      != NIXL_SUCCESS) {
        std::cerr << "postXfer failed\n";
        return 1;
    }

    std::cout << "Transfer posted — doing other work...\n";

    // 7) Poll for completion (replaces deprecated getNotifs/poll map)
    nixl_status_t st;
    do {
        st = agentSrc.checkXfer(reqHandle);
        if (st == NIXL_INPROGRESS) std::this_thread::yield();
    } while (st == NIXL_INPROGRESS);

    if (st != NIXL_SUCCESS) {
        std::cerr << "Transfer completed with error: " << st << "\n";
        agentSrc.releaseReqH(reqHandle);
        agentSrc.deregisterMem(srcRegs);
        agentDst.deregisterMem(dstRegs);
        cudaFree(d_src);
        cudaFree(d_dst);
        return 1;
    }

    std::cout << "Transfer completed!\n";

    // 8) Cleanup
    agentSrc.releaseReqH(reqHandle);
    agentSrc.deregisterMem(srcRegs);
    agentDst.deregisterMem(dstRegs);

    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
```

在这里，NIXL agent 通过名称和一份配置进行初始化。内存用 registerMem 注册，并用 trim 裁剪为传输描述符。一次从 srcAgent 到 dstAgent 的非阻塞写用 prepXfer 准备、用 postXfer 提交。在传输进行的同时，程序继续做其他工作。通过用 checkXfer 轮询请求句柄来检测完成，若请求仍在进行中则让出（yield）线程。成功之后，用 releaseReqH 释放句柄，并注销这些注册。

在内部，NIXL 使用 Unified Communication X（UCX）——一个 HPC 库，它在各种互连之上提供统一的 API，NIXL 借助它进行底层传输，包括 InfiniBand、TCP、共享内存等。NIXL 还使用 GPUDirect RDMA 和 InfiniBand GPUDirect Async（IBGDA），让 GPU 无需 CPU 参与即可发起传输。这是一项重要的优化，因为在较老的系统中，即便数据路径是纯粹的 RDMA，也可能需要 CPU 来启动传输。IBGDA 把这一发起动作卸载给了 GPU/网卡，从而进一步降低延迟。

NIXL 另一个有趣的特性是，它避免了诸如暂存缓冲区之类的不必要拷贝。例如，如果数据位于可分页（pageable）的 CPU 内存中，它会选择将数据固定（pin），以防其被换出。但如果数据位于 GPU 内存中，它会直接把数据发送出去。换句话说，NIXL 会尽量避免那种在把数据传输到目的地之前先拷贝到中间主机缓冲区的暂存缓冲区。

### 使用 NIXL 进行 KV 缓存卸载

NIXL 的设计动机与处理 LLM 推理中大内存的最佳实践密切相关。如果 GPU 没有足够的内存来容纳长序列或多轮对话的整个 KV 缓存，NIXL 就允许推理服务器（如 NVIDIA Dynamo）把 KV 缓存卸载（offloading）到 CPU 内存——甚至 NVMe SSD——并在需要时再取回来。

以 Dynamo 为例，NIXL 与其中的 KV Cache Manager 相配合，就能高效地管理这一传输层级。设想 NVIDIA 的 Grace Hopper 和 Grace Blackwell Superchip，它们拥有海量的、通过高速 NVLink 互连共享的统一 CPU 与 GPU 内存（见图 4-8）。

![图 4-8. 基于 ARM 的 Grace Hopper Superchip 架构利用 900 GB/s 的 NVLink-C2C，克服了传统的 PCIe 瓶颈（source: https://oreil.ly/zf6rF)](AI%20Systems%20Performance%20Engineering-ch4_images/figure-4-8.png)

推理服务器可以迅速把一个大型 KV 缓存卸载到大容量 CPU 内存，从而释放有限的 GPU HBM。这带来了推理性能的巨大提升。具体而言，对于长输入序列，与从头重算缓存相比，基于 PCIe 的 x86 + H100 系统可将首 token 时间（time to first token，TTFT）延迟改善多达 14×。这一加速如图 4-9 所示。

![图 4-9. 在基于 x86 的 NVIDIA H100 GPU 系统上，对大输入序列长度，测得使用 KV 缓存卸载相比从头重算可获得 14× 的 TTFT 加速（source: https://oreil.ly/zf6rF)](AI%20Systems%20Performance%20Engineering-ch4_images/figure-4-9.png)

此外，凭借 900 GB/s 的 NVLink-C2C 互连，基于 ARM 的 Grace Hopper Superchip 相比前面所述的非 superchip、基于 x86 的 H100 版本，能提供 2× 更快的 TTFT 延迟。这一加速如图 4-10 所示。

这些令人瞩目的收益，源自 NIXL 软件与 NVIDIA superchip 硬件的协同设计。而 NIXL 正是针对这些数字而设计的，它通过将传输成本保持在低位，使卸载 KV 缓存成为一个可行的选项。正如我们将在后续章节看到的，KV 缓存卸载是大规模推理部署的关键一环——对于内存容量成为限制因素的超大 LLM 尤其如此。

![图 4-10. 由于借助 900 GB/s 的 NVLink-C2C 互连进行 KV 缓存卸载，基于 ARM 的 Grace Hopper Superchip 相比基于 x86 的 H100 GPU 系统在 TTFT 上获得 2× 加速（source: https://oreil.ly/zf6rF)](AI%20Systems%20Performance%20Engineering-ch4_images/figure-4-10.png)

随着模型越来越大、工作负载越来越复杂，拥有 NIXL 这样一个能高效异步搬运大块数据（blob）的库至关重要。对于性能工程师而言，如果你的用例涉及在系统中的各阶段之间（如流水线并行）以及其他组件之间（如 GPU 或存储）搬运大量数据，就应考虑 NCCL 是否已经够用，或者像 NIXL 这样的专用方案是否可作为优化该数据流的一个选项。

### NIXL 与 NVIDIA Dynamo 等高性能推理系统

对于像 NVIDIA Dynamo（同样于 2025 年初发布）这样的分布式推理系统，NIXL 对性能的影响是巨大的。根据 NVIDIA 的内部测试，开源的 NVIDIA Dynamo 框架在使用一个 72-GPU 的 Blackwell NVL72 机架时，借助 NIXL 在一个约 680B 参数的 LLM 上实现了高达 30× 的推理吞吐提升。

曾经的一大延迟障碍——在节点之间搬移数 GB 的上下文数据——如今在 NIXL 之下已成为一项相对迅速的异步操作。我们将在后面的章节深入介绍 NVIDIA Dynamo、TensorRT、vLLM 以及各自的模型推理优化。

### NCCL 与 NIXL 对比

需要重点指出的是，NIXL 并非 NCCL 的替代品，而是补充。NCCL 仍负责处理那些在单个任务/阶段上并行工作的 GPU 之间的同步集合操作，例如拆分到多个 GPU 上的 all-reduce。而 NIXL 则在任务/阶段之间——或在分布式系统中不同的组件之间（如 GPU、CPU、存储）——执行异步数据传输。表 4-5 展示了 NCCL 与 NIXL 的对比。

_表 4-5. NCCL 与 NIXL 的对比_

| 方面         | NCCL（集合通信）                                                                  | NIXL（点对点通信）                                                                                                                 |
| ------------ | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| 主要用例     | 训练中面向紧耦合 GPU 组的多对多集合操作（如 all-reduce、all-gather）。            | 面向分布式推理或流水线化的一对一或一对少传输（如发送大型张量或缓存）。                                                             |
| 通信模式     | 同步的集合操作——所有参与者都必须到达该调用点（屏障语义，barrier）。               | 异步的 send/receive——一个发起方、一个或多个目标（支持单向数据搬运）。                                                              |
| 与计算的重叠 | 使用独立的 CUDA 流可在一定程度上重叠（如在 DDP 中将反向计算与 all-reduce 重叠）。 | 为最大化重叠而设计——传输与计算完全并行运行，并以轮询通知来检测完成。                                                               |
| 拓扑感知     | 是——自动检测拓扑，为集合操作最优地使用 ring/tree 以及 NVLink/NVSwitch。           | 是——与互连无关；根据源与目的地的位置，自动使用 NVLink、NVSwitch、PCIe、InfiniBand/RDMA 或 GDS。                                    |
| 数据范围     | 通常是需要在所有 GPU 之间聚合的中小型张量（如梯度）。                             | 针对需要快速点对点传输的大数据块（如数百 MB 或更大，例如 LLM KV 缓存或模型分片）进行优化。                                         |
| 集成         | 集成于训练框架中（PyTorch DDP、Horovod 等在底层调用 NCCL）。                      | 作为一个开源库提供，被 NVIDIA Dynamo 使用。它在 Dynamo 项目中开发。开发者在推理服务器或自定义代码中按需调用 NIXL API 来发送/接收。 |
| 示例         | 在 8 个 GPU 上并行地对 100 MB 梯度执行 all-reduce。                               | 在推理流水线中，把 1 GB 的 KV 缓存从 GPU 0 发送到 GPU 1（或发送到 CPU 内存或 NVMe SSD）。                                          |

虽然 NCCL 确实支持点对点的 send()/recv() 操作，但它最适合同步训练环境中的集合操作。而 NIXL 则满足了大规模推理和流水线并行中常见的异步点对点数据传输需求。例如，将在后面章节讨论的 NVIDIA Dynamo 推理服务器，就使用 NIXL 来编排 KV 缓存在各推理组件（包括 GPU、CPU 和 SSD）之间的搬运。

总之，NCCL 旨在最大化跨多个 GPU（无论是在单台主机内还是跨节点）的集合吞吐。它自动选择硬件拓扑感知的 ring 和 tree 通信算法，充分饱和给定的 PCIe、NVLink、InfiniBand 和以太网链路。NCCL 通常与超大规模模型训练工作负载相关联。NIXL 则在 NCCL 的高性能原则之上更进一步，编排跨 GPU、CPU 和存储设备的异步、硬件无关的点对点传输。NIXL 是为大规模分布式推理工作负载而设计的，包括超快的 KV 缓存数据传输。

## 关键要点
