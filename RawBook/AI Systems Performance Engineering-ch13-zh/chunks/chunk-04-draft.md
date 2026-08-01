roundup*power2_divisions:[N:M,...] 该参数控制 PyTorch 的 CUDA 缓存分配器如何将张量尺寸的请求归入固定的分桶。它把每个“2 的幂”区间划分为 \_N* 个大小相等的子桶——例如，当某个请求的大小落在 512 MB 与 1,024 MB 之间时。若在代码中指定 512:2，则 512 到 1,024 MB 的区间会被划分为 :2 个桶，并向上取整到 [512MB, 768 MB, 1 GB] 之一。举例来说，在使用 '512:2' 的 512 到 1,024 MB 区间内，一个 600 MB 的请求会向上取整为 768 MB。请检查分配器日志，以确认在你的环境中实际的分桶情况。这种策略能减少内存碎片（memory fragmentation），使分配尺寸标准化，并提高缓存复用率，因为相似的请求会命中同一个桶。

backend:cudaMallocAsync 指定该选项会启用 NVIDIA 的 CUDA 异步分配器作为底层的内存分配机制。这有助于避免在内存释放事件上的同步——并能在多线程场景（如多工作进程的数据加载）中提升性能。

通过自定义内存分配器配置，你可以保持更平稳、更可预测的内存使用模式。在长时间运行中，你可以用 torch.cuda.memory_stats() 监控内存碎片，确保内存占用保持稳定、不会急剧膨胀。

你也可以在运行时使用 torch.cuda.mem_get_info() 获取空闲内存与总内存。这能间接反映碎片情况：如果已分配张量的数量保持不变，而空闲内存却在下降，就说明碎片正在增加。

### 用于节省内存的激活检查点

对于超大模型来说，激活检查点（activation checkpointing）——一些从业者也称之为梯度检查点（gradient checkpointing）——是管理内存的关键手段。在处理大型 LLM 时，有时无法在不耗尽内存的情况下，为反向传播（backpropagation）保存全部中间激活值。

使用激活检查点时，你不必在前向传播（forward pass）期间保存中间激活值（以备反向传播使用），而是仅在需要时于反向传播过程中即时重新计算它们。PyTorch 提供了 torch.utils.checkpoint 来自动完成这一过程。你只需包裹某个模型层——或一段连续的层——它们的前向激活值就不会被保存。

你可以在每个 transformer 块以及模型中每个专家的前馈神经网络（Feedforward Neural Network，FFN）层这样的粒度上应用检查点。这样一来，在计算完每个块的前向输出后，你就无需再把那些中间激活值保留在内存中。取而代之的是，在反向传播期间，PyTorch 会重新运行该块的前向传播，以重新生成用于梯度计算的激活值。

值得注意的是，你不必对所有内容都做检查点，只需聚焦于最大的那些层即可。一种常见策略是：只对持有海量激活值的 transformer 块做检查点，而不对层归一化（layer norm）和嵌入层等较小的层做检查点。这样能以最小的重算开销换来最大的内存节省。

> 使用 FSDP（完全分片数据并行，Fully Sharded Data Parallel）时，你还可以启用自动检查点，它会为你递归地将检查点应用到多个层上。

这种取舍是以增加计算量来换取更低的内存占用。所幸的是，相对于 HBM 内存的容量，现代 GPU 提供了充裕的 FLOPS，因此这项技术非常契合最新几代硬件。也正因如此，GPU 拥有额外的计算余量来承担这些额外的重新计算。

这种权衡往往是值得的。如果不使用激活检查点，你就需要减小输入批大小（batch size）——或减少专家数量——才能塞进有限的 GPU 内存。而使用检查点后，你就能从容地把模型放进内存，同时保留较大的批大小。

尽管激活检查点会让训练稍微变慢，但它让你能够训练更大的模型变体——并使用更大的输入批大小——这些原本是无法装入内存的。本质上，你是用 GPU 充裕的计算 FLOPS 去弥补其有限的 HBM 容量。

### 将参数卸载到 CPU 与 NVMe

除了检查点之外，你还可以卸载（offloading）一部分无需常驻 GPU 的模型参数。例如，MoE（专家混合，mixture of experts）模型有一些访问频率较低的专家层，可以把它们卸载到 CPU 内存——仅在需要时才传输到 GPU。

关键在于让数据传输与计算重叠：当某一层在 GPU 上运行时，异步地从 CPU 或 SSD 预取下一层的权重。在实践中，DeepSpeed 的 ZeRO-Infinity（用于训练）和 ZeRO-Inference（用于推理）等框架可以自动完成这种预取。它们把模型权重逐层地从 CPU 或 NVMe 流式传输到 GPU，在使数据传输与计算重叠的同时，将 GPU 峰值内存占用降到最低。

你可以把这些组件锁页（pin）在主机上，并使用异步、非阻塞的 DMA 调用（如 .to(device) 和 cudaMemcpyAsync），在其他计算正在运行时将它们传输到 GPU。这样就能隐藏因从 CPU 拷贝而产生的额外传输延迟。

NVIDIA 的统一内存（Unified Memory）也是一种选择——尤其适用于像 Grace Blackwell GB200/GB300 这样、在 CPU 与 GPU 之间配备了 NVLink-C2C 等高速互连的 superchip 系统。在这类情形下，统一内存允许把很少使用的 GPU 内存页面驱逐到 CPU 的系统内存中。

如果为了容量需要，操作系统甚至可能把它们交换到 NVMe/SSD。NVMe 应作为通过操作系统正常交换机制的最后手段来使用，而不应作为统一内存的主要目标。

然而，由于内存分页等原因，统一内存可能带来不可预测性。因此，为了保持完全的掌控和可预测的性能，显式地管理卸载往往是更可取的做法。

GPUDirect Storage 等近期的进展让 GPU 能够直接从 NVMe 驱动器读取数据。这意味着在某些情况下，你的模型参数可以即时地直接从 NVMe 分页调入，而完全无需 CPU 参与。在使用超大模型时，这对训练和推理服务都很有用。

对于参数量达到万亿量级的更大模型，你可以把部分组件卸载到 NVMe 存储，并在恰好需要时（just-in-time）将它们换入 GPU 内存。关键在于让数据传输与计算重叠，从而不会拖慢训练循环。

### SuperOffload：面向 CPU-GPU superchip 的优化卸载

SuperOffload 是一套专门为发挥 CPU-GPU superchip 硬件效率而设计的卸载系统。superchip（例如 Grace Hopper、Grace Blackwell、Vera Rubin 等）在 CPU 与 GPU 之间提供了高带宽的 NVLink-C2C 互连。再结合共享、一致的内存地址空间，与传统卸载技术相比，针对 superchip 优化的卸载策略能带来巨大的加速与利用率提升。

SuperOffload 展示了几项关键创新，包括先推测后验证（speculation-then-validation，STV）、异构优化计算、感知 superchip 的数据类型转换以及内存拷贝。下面我们逐一讨论。

传统卸载会先等待梯度归约和全局检查，然后才更新参数；相比之下，STV 会在 GPU 执行反向传播的同时，在 CPU 上进行推测性的优化器更新，从而让这些步骤相互重叠，之后再对结果进行验证。这有效减少了同步停顿，并提升了整体 GPU 利用率。

SuperOffload 使用异构优化器计算，把优化器的工作在 CPU 与 GPU 之间进行划分。例如，它将计算量大的张量更新分配给 GPU，而由 CPU 核心处理较轻量的状态更新，比如 Adam 优化器所用的动量缓冲区。这让两种设备都保持忙碌，减少空闲周期，并提升整体芯片利用率。

由于 NVLink-C2C 在 CPU 与 GPU 之间提供了高带宽，SuperOffload 可以改变张量类型转换和数据传输的精度与放置策略。通过把类型转换和拷贝转移到 GPU 一侧，SuperOffload 充分利用了快速、一致的互连，从而将 CPU-GPU 传输延迟降到最低。

此外，SuperOffload 使用了一种针对 CPU 优化的 Adam 优化器变体，名为 GraceAdam，它是为 Grace 的 ARM 可伸缩向量扩展（Scalable Vector Extension，SVE）架构而设计的。SVE 是 ARM A64 指令集所采用的长向量架构。具体而言，它拥有 32 个向量寄存器和 16 个谓词寄存器，与 SuperOffload 结合后，有助于提升被卸载的参数更新的吞吐量与能效。

### FSDP 的自动检查点与卸载

PyTorch 的 FSDP 是一种分布式并行策略，在模型训练期间把模型参数、梯度和激活值分片（shard）到多个 GPU 上。这降低了训练期间的内存开销——并让你能够训练比不分片时更大的模型。从技术上讲，FSDP 实现了 ZeRO Stage-3 策略，将模型状态分片到多个 GPU 上，如图 13-6 所示。

![图 13-6. FSDP 将模型参数、梯度和优化器状态分片到多个 GPU 上（ZeRO Stage-3）](AI%20Systems%20Performance%20Engineering-ch13_images/figure-13-6.png)

FSDP 可以在底层自动应用激活检查点，并卸载参数/梯度。只需用 FSDP() 包裹模型，然后按如下方式指定 activation_checkpointing_policy 和 CPUOffload 参数：

```
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload, ShardingStrategy, BackwardPrefetch, MixedPrecision
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
# Initialize distributed
dist.init_process_group("nccl")
torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
# Build your model
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
        )
    def forward(self, x):
        return self.layers(x)
model = MyModel().cuda()
# Auto-wrap transformer blocks if needed
auto_wrap_policy = transformer_auto_wrap_policy(
    model,
    min_num_params=1e8,
)
# Wrap with FSDP + checkpointing + CPU offload
fsdp_model = FSDP(
    model,
    auto_wrap_policy=auto_wrap_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    use_orig_params=True,
    cpu_offload=CPUOffload(offload_params=True, pin_memory=True),
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    ),
    backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
    activation_checkpointing_policy={
        nn.TransformerEncoderLayer,
        nn.TransformerDecoderLayer,
        nn.MultiheadAttention
    }
)
# Setup optimizer
optimizer = torch.optim.AdamW(fsdp_model.parameters(), weight_decay=0.01)
...
```

这里我们使用 transformer_auto_wrap_policy，让 FSDP 根据你的 transformer 块结构自动分片参数、梯度和优化器状态。我们还通过 CPUOffload(offload_params=True) 启用了向 CPU 的卸载。这会在参数和梯度不需要驻留 GPU 时，透明地把它们移动到 CPU，从而降低 GPU 峰值内存占用。

> 通过设置 use_orig_params=True，每个 FSDP 单元在处理其参数时无需展平。这带来了更好的重叠以及更简单的状态字典（state-dictionary）处理——从而改善内存管理和优化器兼容性。

设置 activation_checkpointing_policy 会告诉 FSDP 在那些特定的核心 transformer 子模块中重新计算（或称重新物化，rematerialize）激活值。这会以额外的计算换取显著更低的峰值内存。它能达到与手动的 checkpoint() 包裹和自定义卸载脚本相同的巨大内存节省，却无需额外的样板代码。FSDP 甚至能处理各 GPU 之间不均衡的批大小，这对 MoE 工作负载很有用。

FSDP 还支持一种混合分片策略，通过 ShardingStrategy.HYBRID_SHARD 使用。该策略把每个节点的参数、梯度和优化器状态分片到该节点内的各个 GPU 上，同时把这些相同的分片复制到其他节点。本质上，混合分片提供了比 FULL_SHARD 更高的吞吐量，但代价是每个节点占用更多内存。

当你的互连不错但并非极快时，应使用 HYBRID_SHARD。它在每个节点内部分片参数、梯度和优化器状态——并将这些分片跨节点复制。这减少了跨节点流量，代价是每个节点的内存占用比完全分片略高一些。

混合分片让你能够管理内存与通信之间的权衡。例如，你可以比 FULL_SHARD（ZeRO 3）占用更多的每节点内存，因为你在每个节点上都持有一份完整的本地分片。这减少了节点间通信，通常能提供更高的端到端吞吐量。当你拥有非常快的多节点网络结构、并希望每个 GPU 的内存占用最小时，FULL_SHARD 是最佳选择。

当你的网络较慢——或只有少量几块 GPU——时，你可以使用 ShardingStrategy.SHARD_GRAD_OP（ZeRO Stage-2），只把梯度和优化器状态分片到所有 GPU 上。该策略会在每块 GPU 上都保留一份完整的参数副本。

### 将 FSDP 与张量并行、流水线并行结合

如果模型大到单个层都无法装入单块 GPU 的内存，你就需要把 FSDP 与其他并行策略（如张量并行（tensor parallelism，TP））结合起来，把庞大的模型层铺展到多块 GPU 上。

你还可以跨 GPU 和计算节点使用 FSDP：在节点内用 TP 把巨大的层拆分到多块 GPU 上，再用流水线并行（pipeline parallelism，PP）把这些经过 TP 拆分的层跨节点串联起来。FSDP 支持这类灵活的组合。

简而言之，FSDP 能以少得多的编码工作量来降低内存占用。在应用检查点和卸载之后，你就能把模型装入内存，并启用更大的批大小。这有助于提升 GPU 利用率。

### 可插拔内存分配器（pluggable memory allocator）与跨 GPU 数据传输

你可以配置 PyTorch，为关键的 GPU 通信操作（如 NCCL 的梯度 all-reduce）使用专门的内存分配器。通过用 torch.cuda.MemPool 插入自定义分配器，你可以让 NCCL 以一种能利用专用硬件引擎（如 NVLink 拷贝引擎、InfiniBand 卸载或 NVIDIA SHARP）的方式来分配缓冲区。这些都有助于改善通信与计算之间的重叠。例如，PyTorch 支持在 NVSwitch 环境中使用 NCCL 的内存分配器来进行快速、零拷贝的归约，如下所示：

```
import torch
import torch.distributed as dist
from torch.cuda.memory import MemPool
from torch.distributed.distributed_c10d import _get_default_group
# Initialize NCCL distributed backend
dist.init_process_group(backend="nccl")
torch.cuda.set_device(
  dist.get_rank() % torch.cuda.device_count()
)
# Get the NCCL backend object for this device
default_pg = _get_default_group()
backend = default_pg._get_backend(
  torch.device(f"cuda:{torch.cuda.current_device()}")
)
# The backend exposes ncclMemAlloc using mem_allocator
nccl_allocator = backend.mem_allocator
# Create a dedicated memory pool using this allocator
nccl_pool = MemPool(nccl_allocator)
# Register the pool so NCCL uses it for collective gradient buffers
backend.register_mem_pool(nccl_pool)
# Use the pool explicitly for NCCL operations
with torch.cuda.use_mem_pool(nccl_pool):
    tensor = torch.randn(10_000_000, device="cuda")
    dist.all_reduce(tensor)
```

通过把 NCCL 的原生分配器（backend.mem_allocator）绑定到 torch.cuda.memory.MemPool 中，并使用 backend.register_mem_pool(...)，PyTorch 会把梯度归约缓冲区直接放置到经过优化的内存区域中，以便通过 NVLink、InfiniBand 和支持 NVIDIA SHARP 的硬件实现最优路由。这样一来，all-reduce 操作就能从硬件加速中受益，减少大规模数据归约期间的 SM 争用。其结果是大型多 GPU 工作负载的吞吐量得到提升。

> 使用第 4 章讨论过的 SHARP，需要兼容的网络结构（例如支持 SHARP 的 HDR InfiniBand）。如果条件具备，启用 NCCL 基于 SHARP 的网络内聚合能大幅降低大型集群的 all-reduce 延迟。当扩展到大量节点时，这绝对值得考虑。

通过用 backend.mem_allocator 使用 NCCL 的原生分配器，并将其包裹进 PyTorch 的 MemPool，梯度 all-reduce 缓冲区会被分配到为 GPUDirect RDMA 和网络内 SHARP 卸载而优化的内存区域中。这会把缓冲区对齐到大页边界，并有助于把数据传输放到专用的 DMA 引擎上——减少 SM 的介入，释放出计算能力去执行更有价值的计算工作。

其结果是，像张量并行归约这样的 NCCL 集合通信操作既能从硬件加速中受益，又能获得更低的 SM 争用。这显著提升了多 GPU 同步吞吐量，减少了 SM 和拷贝引擎上相互重叠的计算与通信核函数之间的争用——对张量并行工作负载而言尤其如此。

当你逼近带宽极限时，这类优化会变得愈发重要。例如，Blackwell 的 NVLink 5 在理论上为每块 GPU 提供高达 1.8 TB/s 的双向带宽（每个方向约 900 GB/s）。使用专用拷贝引擎、网络硬件（SHARP）和优化过的内存分配器，有助于逼近峰值网络吞吐量，同时释放 SM 去达到峰值 FLOPS。

借助 torch.cuda.MemPool，你还可以通过注册一个自定义库（共享对象，即 .so）来创建自己的内存分配器。此外，你还可以在同一个 PyTorch 应用中混用不同的 CUDA 内存分配器，如下所示：

```
import torch  # PyTorch main namespace
import os     # for path operations (used here for .so extensions)
from torch.cuda.memory import CUDAPluggableAllocator
# 1. Create a CUDAPluggableAllocator and MemPool
# Build a pluggable allocator that calls into your NCCL library:
#   - allocator_path: path to your .so
#   - "ncclMemAlloc": symbol for allocation
#   - "ncclMemFree":  symbol for deallocation
# .allocator() returns a callable that matches the CUB/CUDA allocator API
allocator = CUDAPluggableAllocator(
    "./nccl_allocator.so",
    "ncclMemAlloc",
    "ncclMemFree"
).allocator()
# Wrap that allocator in a MemPool for efficient sub-allocations
pool = torch.cuda.memory.MemPool(allocator)
# 2. Start recording events (set a high cap for long runs)
torch.cuda.memory._record_memory_history(max_entries=100000)
# 3. Allocate tensors with different allocators
# tensor0 uses the *default* cudaMalloc allocator
# - Shape: (1024, 1024), you can change to your desired size
tensor0 = torch.randn(1024, 1024, device="cuda")
# tensor1 uses *your* NCCL-backed allocator using MemPool
with torch.cuda.use_mem_pool(pool):
    # Inside this context, all cuda allocations go through `pool`
    tensor1 = torch.randn(1024, 1024, device="cuda")
# Exiting the context restores the default allocator
# tensor2 again uses the *default* cudaMalloc allocator
tensor2 = torch.randn(1024, 1024, device="cuda")
# 4. Inspect memory pool stats
# Pool-specific snapshot with list of segments/blocks in use
pool_state = pool.snapshot()
print(f"Pool segments count: {len(pool_state)}")
# 5. Dump the snapshot and optionally load in the PyTorch
# memory viewer tool (https://oreil.ly/tX6gA)
torch.cuda.memory._dump_snapshot('memory_snapshot.pkl')
# Global allocator stats (allocated/reserved, peak, counts)
global_stats = torch.cuda.memory_stats()
print("Peak allocated bytes:", global_stats["allocated_bytes.all.peak"])
# 6. Stop recording
torch.cuda.memory._record_memory_history(enabled=None)
# 7. Reset peak counters for a fresh measurement
torch.cuda.reset_peak_memory_stats()
```

这里，CUDAPluggableAllocator 会加载你的自定义 .so，并绑定到你指定的两个符号上。MemPool 把底层的 CUDA 内存分配器包裹进一个用于内存分配与内存释放的缓存中，以获得更好的性能。

torch.cuda.use_mem_pool(pool) 会为 Python 上下文管理器（例如 with）代码块内所有后续的内存分配换入你的内存池。退出该上下文管理器代码块时，会恢复之前的分配器。

### 启用点对点 DMA 与 UCX

在多 GPU 系统中使用流水线并行时，你通常希望用尽可能最快的点对点（peer-to-peer，P2P）连接，把激活值直接从一块 GPU 搬到下一块 GPU——避免往返以及占用主机 CPU 资源。当张量在一个进程内跨设备移动时，PyTorch 会自动探测并启用对等访问。不过，如果你想手动确认，可以使用 torch.cuda.can_device_access_peer(i, j) 来确认 GPU i 与 GPU j 之间的 P2P DMA。对于自定义的 C++ 操作，你可以用 CUDA 驱动 API 显式地启用对等访问。

启用 P2P DMA 可提供高效的直接传输，而无需动用 GPU 的 SM 或 CPU 内存。一旦启用，跨 GPU 的 copy\_() 和 .to() 就会走更快的对等内存路径，无需额外的代码开销，如下所示：

```
dst.copy_(src, non_blocking=True)
# or
src.to(device="cuda:1", non_blocking=True)
```

当你的拓扑启用了 P2P 访问时，这些方法会使用 cudaMemcpyPeerAsync。要让 P2P 传输正常工作，你的硬件拓扑必须支持它。GB200/GB300 NVL72 机架在内部使用了具备 P2P 能力的 NVLink 和 NVSwitch，因此它开箱即用地已经为 P2P DMA 做好了准备。

到目前为止，我们只在 P2P 数据传输的语境下讨论了多 GPU 配置。然而，当使用多节点 GPU 拓扑时，你需要退回到通过网络通信的 NCCL。有时这会经由 GPUDirect RDMA 进行。但把 UCX（Unified Communication X）与 NCCL 搭配使用可以提升性能。在 GB200 和 GB300 NVL72 拓扑中，NVLink 和 NVSwitch 在机架内部提供了 P2P 路径。而跨机架时，你就需要通过启用了 UCX 的 NCCL 传输来穿越网络结构。

要让跨节点传输经由 UCX 路由，需安装 NCCL-UCX 插件并设置 NCCL_PLUGIN_P2P=ucx。许多环境还要求设置 NCCL_NET=UCX，并为你的网络结构配置合适的 UCX_TLS 传输选择。这会启用硬件卸载，并改善 NVL72 机架之间 InfiniBand 网络结构上的流水线化。

以下是 NCCL-UCX 插件的一个示例配置：

```
export NCCL_NET=UCX
export NCCL_PLUGIN_P2P=ucx
# UCX transports vary by fabric.
# These are safe defaults to start with:
export UCX_TLS=rc,self,gdr_copy,cuda_copy
```

如果你的应用需要跨多个节点或机架进行扩展——训练 TB 级模型、面向众多租户对超大 LLM 做推理，或管理数据密集型流水线——那么 UCX 就必不可少。将 NCCL 与 UCX 搭配使用，能提供高吞吐、低延迟的跨节点通信，既支持硬件卸载（RDMA），又具备智能的拓扑感知。UCX 是生产级 AI 基础设施的核心组成部分，在扩展到单个 NVLink 域之外时不可或缺。

简而言之，与使用 NCCL 的等效 send/recv 调用相比，直接的 P2P DMA 通信和 UCX 能实现与计算更好的重叠。这是一项相对底层的优化，但如果你拥有合适的拓扑以及 NVLink/NVSwitch 等专用高速互连，它就能大幅提升系统性能。

## PyTorch 对称内存

对称内存（symmetric memory）是一种编程模型，它在多块 GPU 之间暴露出一个分区全局地址空间（partitioned global address space），使核函数能够执行单边的 put 和 get 操作。这让它们可以在无需 CPU 握手或介入的情况下，调用超低延迟、跨 GPU 的直接访问集合通信。本质上，对称内存分配的缓冲区可以被组内任意 GPU 直接寻址——而无需显式的点对点拷贝。

使用对称内存时，你可以完全在设备上执行 all-to-all 操作，比如 MoE 的 token 混洗。由于没有 CPU 参与，整个 all-to-all 都可以被 CUDA Graph 捕获。若没有对称内存，一次 all-to-all 会触发一次主机同步，从而在时间线上造成设备到主机（device-to-host，D2H）的空隙。这会在 CUDA Graph 中制造出不想要的中断。

你可以分配对称张量，在一个进程组内执行一次汇合（rendezvous）以获取远程访问句柄，然后调用直接访问的集合通信（例如 all_to_all_v、one_shot_all_reduce）。此外，你还可以启动使用这些远程访问句柄执行单边读/写的核函数。与 OpenAI Triton 和 NVSHMEM（triton.nvshmem.put()/get()）结合后，对称内存成为一种强大的通信机制，可用于自定义的、核函数内的数据传输。

> 对于细粒度、延迟敏感、且需在设备上完成的交换（如无需主机 CPU 介入的 MoE all-to-all token 混洗），你应当在 PyTorch 中使用对称内存。这将消除设备到主机（D2H）的时间线空隙，并实现更好的 CUDA Graph 捕获。

## 优化数据输入流水线

在讲完模型的内存与计算之后，我们来看看数据输入流水线（data input pipeline）。造成低效的一个常见原因是 GPU 空闲地等待数据。PyTorch 的 DataLoader 支持在训练模型的同时，派生多个工作线程或进程来并行地加载和预处理数据（例如文本分词）。请务必指定 pin_memory=True，使主机 CPU 内存到 GPU 设备的传输使用页锁定（锁页）内存。

如果不锁定内存，你可能会在数据加载器（data loader）进程中看到很高的 CPU 利用率，而在训练进程中看到很低的 GPU 利用率。这是双重的坏事。这些不理想的利用率可以用 htop 和 Nsight Systems 等工具观察到。你会看到 CPU 线程忙于执行数据加载，而 GPU 却处于空闲。

这表明数据加载没能跟上每一次迭代的节奏。你可以通过增大 DataLoader 的 num_workers 参数值来解决，直到数据队列备好足够多的批次为止。对于 CPU 受限的数据流水线，一个常见的经验法则是每块 GPU 配 4 个工作进程，但最优数量会随你的工作负载而变化。

为了找到最优数量，你可以做一次快速扫描（例如 4、8、16、32），找出再增加工作进程也不再有帮助的临界点。要留意 CPU 是否饱和。如果所有核心都处于 100%，那么增加工作进程也无济于事。

请记住，像 Grace Hopper、Grace Blackwell 和 Vera Rubin 这样的 NVLink C2C superchip 系统提供了一个大容量、一致、高带宽的 CPU–GPU 内存地址空间。在这类环境中，页锁定的 pin_memory=True 往往不那么关键，因为传输本就走的是高带宽的一致路径。

在非一致路径上，为了最大化可分页的主机到设备拷贝的重叠，可能仍然需要锁页内存。虽然统一内存和一致映射在 NVLink-C2C 系统上可能降低对锁页的需求，但衡量你自身工作负载的性能仍然很重要。

即便使用 CPU-GPU superchip 架构，当加载器线程受限于 CPU 时，仍建议把 pin_memory=True（或大页）与 non_blocking=True 和 persistent_workers=True 结合使用。使用 persistent_workers=True 可以避免在各个 epoch 之间重新派生进程。这对分词密集型的工作负载（如 LLM 的训练与推理）很有帮助。请用 Nsight Systems 做剖析，验证实际的重叠情况。在移除锁页之前，务必在主机上启用大页，并确认 H2D 流量与计算相互重叠。

CPU 与 GPU 共享一个一致的内存空间，因此即便不显式锁页，传输也已走高速路径。相比之下，在没有此类硬件一致性的标准（非 superchip）CPU + GPU 系统上，启用 pin_memory=True 会把主机内存页锁定，并带来数据传输吞吐量的显著提升。

> 如果你不锁定内存，而是依赖 Grace Blackwell 和 Vera Rubin 等 superchip 系统的统一 CPU-GPU 内存，那么务必启用大页，并用 Nsight Systems 验证传输是否确实发生了重叠。这是因为 NVLink-C2C superchip 上的统一 CPU-GPU 内存改变了锁页方面的权衡。

另一种提升数据加载响应速度的技术，是增大 DataLoader 的 prefetch_factor。它控制每个工作进程提前预加载多少个批次，其计算方式为 prefetch_factor \* num_workers。这意味着当你的模型忙于处理当前批次时，工作进程已经在加载并缓存后续的批次了。

预取能让数据流水线保持充盈，避免 GPU 空转。你应当根据数据集和硬件能力，把 prefetch_factor 与 num_workers 一起调整。请通过剖析来验证，以免过度预取而造成不必要的内存占用。

你还可以预先计算好分词后的数据集，并把分词后的数据集存到磁盘上，从而避免在数据加载循环中做繁重的处理。像 Hugging Face 的 Dataset.cache() 和 WebDataset 这样的工具，让你只需对文件预处理一次即可反复复用。此外，可以考虑对数据集使用混合精度和压缩，以降低对 I/O 带宽的需求。

还有 PyTorch 原生的 TorchData 及其 DataPipes。它提供了良好的可组合性，并与 PyTorch 调度器集成，以便与训练计算相互重叠。

另一个有用的工具是 NVIDIA Data Loading Library（DALI）。DALI 可以在训练的同时并行地执行 CPU 或 GPU 预处理。它对图像/视频数据（例如解码和增强）尤其有用，并且能通过 CUDA 流水线把数据直接喂给你的训练代码。对于那些适合卸载到 GPU 的即时数据变换，它是一个很有用的工具。

在所有情况下，目标都是尽可能让数据加载与 GPU 计算相互重叠。通过用 Nsight Systems 做剖析，你可以确认数据加载器是在与 GPU 并行工作的。Nsight Systems 的时间线应当显示：一个 CPU 线程在持续地加载/预处理下一个批次，同时多条 GPU 流在执行训练步骤。此外，还要验证不存在 GPU 等待数据的空隙。这能确保你的 GPU SM 几乎 100% 的时间都在忙于有价值的训练工作。

假设内存约束已经通过激活检查点和卸载得到解决，你就可以增大每块 GPU 的输入批大小。使用更大的批次能够提高算术强度（arithmetic intensity）、改善 GPU 利用率，并降低多 GPU 训练中通信的相对开销，因为每个 epoch 的步数变少了。

然而，批大小过大会把优化器推向尖锐极小值（sharp minima），从而降低泛化能力、影响收敛。在大幅增大批大小时，务必监控 GPU 内存占用。请记住，梯度累积会增大有效批大小（batch_size \* accumulation_steps）。PyTorch 的 TorchEval 指标可以帮助判断更大的批大小是否在损害验证损失。

你或许可以通过相应地调整超参数，来把大批量带来的不稳定影响降到最低。例如，你可以改变学习率、采用带预热期的线性学习率缩放，或使用像 LAMB 这样的大批量优化器。

简而言之，如果内存允许——并且前几节介绍的其他内存优化已经落实并验证过——那么增大批大小可以提高算术强度，并更好地利用 GPU。

> 在优化系统的过程中，你应当定期重新审视并重新调优批大小、学习率等超参数。在你改动批大小等设置之后，最初选定的取值可能就不再是最优的了。

## 使用 PyTorch Distributed 进行扩展

在多块 GPU 和多个计算节点上扩展并剖析 PyTorch，通常会用到 PyTorch 的分布式库，如 PyTorch DDP（分布式数据并行，Distributed Data Parallel）和 FSDP。好消息是，PyTorch 的编译器可以与这些并行方法协同工作，但其中有一些细微之处，下面将加以说明。

### DDP 与 torch.compile

DistributedDataParallel 使用 all-reduce 集合通信在多块 GPU 之间同步梯度；当使用它时，PyTorch 会在同步点自动产生图中断（graph break）。在实践中，DDP 会把梯度划分到若干桶中，并让通信与计算相互重叠。PyTorch 的设计是把每个桶的反向计算编译为一个单独的图，从而能在这些图之间执行 all-reduce。
