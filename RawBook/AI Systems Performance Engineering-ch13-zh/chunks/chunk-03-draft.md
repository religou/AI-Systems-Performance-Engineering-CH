```
# Set up streams
device = 'cuda'
# for H2D data transfers
transfer_stream = torch.cuda.Stream(device=device)
# for compute
compute_stream  = torch.cuda.default_stream(device=device)
# Create an iterator so we can preload "next" batches
dataloader_iter = iter(dataloader)
# Preload the very first batch onto GPU
first_batch = next(dataloader_iter, None)
if first_batch:
    with torch.cuda.stream(transfer_stream):
        next_inputs, next_labels = (
            first_batch[0].to(device, non_blocking=True),
            first_batch[1].to(device, non_blocking=True),
        )
for _ in range(len(dataloader)):
    # 1) Wait for transfer of `next` batch to finish, then swap into compute var
    # Multiple copy engines allow H2D/peer copy concurrency.
    # Verify parallelism in Nsight Systems (Copy engines lanes)
    # And tracing/profiling tools (HTA, etc.)
    compute_stream.wait_stream(transfer_stream)
    inputs, labels = next_inputs, next_labels
    # 2) Kick off transfer of the *following* batch on the transfer_stream
    batch = next(dataloader_iter, None)
    if batch:
        with torch.cuda.stream(transfer_stream):
            next_inputs, next_labels = (
                batch[0].to(device, non_blocking=True),
                batch[1].to(device, non_blocking=True),
            )
    # 3) Run forward/backward on compute_stream
    with torch.cuda.stream(compute_stream):
        outputs = model(inputs)
        loss    = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

本示例使用了两条 CUDA 流（CUDA stream）：一条专用的传输流用于异步的主机到设备（host-to-device，H2D）拷贝，另一条默认的计算流用于模型计算。这样可以隐藏 H2D 延迟（latency），并契合 PyTorch 推荐的通信与计算重叠模式。

具体来说，当模型在默认流上处理某个批次时，下一个批次的数据传输已经在 transfer_stream 上进行了。在使用预加载的批次之前，通过 compute_stream.wait_stream(transfer_stream) 进行同步，可以在不设置全设备范围栅栏（barrier）的情况下保证正确的执行顺序。而 .to(device, non_blocking=True) 调用则确保拷贝采用基于异步 DMA 的方式，不会阻塞发起调用的 CPU 线程。

使用 next(dataloader_iter, None) 可以显式控制传输何时入队、以及核函数操作何时运行。这样就能保证：当一个批次的数据正在传输流上传输时，另一个批次正在计算流上执行，如图 13-4 所示。

![图 13-4. 使用专用的计算流与传输流实现计算与数据传输的重叠](AI%20Systems%20Performance%20Engineering-ch13_images/figure-13-4.png)

此外，通过提前从 dataloader_iter 取数据并存入 next_inputs、next_labels，这段代码将批次加载（在 transfer_stream 中于 CPU 上运行）与批次处理（在 compute_stream 中于 GPU 上运行）分离开来。

这种拆分意味着每条流上始终有一个批次在处理中。它将数据加载与计算解耦，并最大化重叠。

> 在添加流时，务必用 Nsight Systems 或 PyTorch 剖析器（PyTorch profiler）做性能剖析（profiling）。观察 GPU 利用率——如果做法正确，你会看到接近 100% 的利用率，传输与计算相互重叠。如果利用率下降——或者你看到数据传输与计算是顺序进行的——请仔细检查是否存在任何隐式同步，例如对 CUDA 张量调用 print()，或者代码中额外的 CUDA 同步。

### 用事件进行流同步

在使用多条流时，有时需要在它们之间进行协调，例如确保一条流的工作完成之后，另一条流才使用它的结果。正如第 11 章所讨论的，使用 CUDA 事件（event）是一种在特定节点上进行同步的轻量级方式。

借助 CUDA 事件，你可以在一条流上记录一个事件，并让另一条流等待该事件。这样就避免了 torch.cuda.synchronize() 那种重量级的全局同步，转而只对处理某个具体事件所需的流进行同步。

事实上，即便在多 GPU 场景下，事件也可以用来跨设备同步工作。在这种情况下，一个 GPU 在其设备流上记录事件，而另一个 GPU 在其设备流上等待该事件。NCCL 在底层正是这样处理依赖关系的。

通过巧妙地使用事件，你可以让多个 GPU 尽可能地并行工作。下面是一个在 PyTorch 中使用 CUDA 事件同步两条流 stream1 与 stream2 的示例：

```
# Disable timing on the event since we’re using it purely for synchronization.
event = torch.cuda.Event(enable_timing=False)
# In first stream:
with torch.cuda.stream(stream1):
    kernel_launch(...)
    event.record()           # record event at end of work in stream1
# In another stream or on host:
stream2.wait_event(event)    # make stream2 wait until event is signaled
with torch.cuda.stream(stream2):
    other_kernel_launch(...)
```

在这段代码中，我们在 stream1 上某些工作结束时记录一个事件。随后，在 stream2 上启动工作之前，我们调用 stream2.wait_event(event)。这会插入一个依赖关系，使得 stream2 在 stream1 执行到那一点并触发事件之前，不会执行它的下一个核函数。事件适合用来在流之间调度轻量级的依赖关系，因为它们避免了那种会让所有流的执行都停顿的重量级全局同步。

让我们回顾上一节中 PyTorch 数据加载器 / 重叠的示例，并把它改写为用 CUDA 事件来同步。我们仍然使用之前的那对流（transfer_stream 与 compute_stream），但会额外加入一个 transfer_done CUDA 事件，以便在细粒度、针对具体事件的层面上进行同步：

```
import torch
device = 'cuda'
# for H2D copies
transfer_stream = torch.cuda.Stream(device=device)
# for compute
compute_stream  = torch.cuda.default_stream(device=device)
# sync-only event (low overhead)
transfer_done = torch.cuda.Event(enable_timing=False)
# Iterator so we can preload ahead
dataloader_iter = iter(dataloader)
# ---- Preload first batch ----
first_batch = next(dataloader_iter, None)
if first_batch:
    with torch.cuda.stream(transfer_stream):
        next_inputs, next_labels = (
            first_batch[0].to(device, non_blocking=True),
            first_batch[1].to(device, non_blocking=True),
        )
        # mark when H2D is done
        transfer_done.record(stream=transfer_stream)
for _ in range(len(dataloader)):
    # ---- Sync: wait for the transfer to complete ----
    compute_stream.wait_event(transfer_done)
    inputs, labels = next_inputs, next_labels
    # ---- Kick off next transfer ----
    batch = next(dataloader_iter, None)
    if batch:
        with torch.cuda.stream(transfer_stream):
            next_inputs, next_labels = (
                batch[0].to(device, non_blocking=True),
                batch[1].to(device, non_blocking=True),
            )
            transfer_done.record(stream=transfer_stream)
    # ---- Compute on the compute stream ----
    with torch.cuda.stream(compute_stream):
        outputs = model(inputs)
        loss    = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

这与上一节采用的重叠模式相同，但它使用 CUDA 事件来完成传输 → 计算之间的同步，这与另一个示例中使用的 wait_stream() 机制形成对比。

这段代码仍然使用 next(dataloader_iter) 提前预加载批次（与上一节的示例相同）。这样，数据传输与计算就始终处于重叠状态。

不过，在这个示例中，transfer_done 事件是在异步拷贝入队之后立即通过 transfer_done.record(stream=transfer_stream) 在 transfer_stream 上记录的。这会为该事件打上时间戳。

随后，compute_stream.wait_event(transfer_done) 会让 compute_stream 停顿，直到拷贝完成、transfer_done 事件被触发为止。之后它便使用预取的批次，并在 compute_stream 上执行其计算操作。

除了数据加载之外，CUDA 流在许多不同的场景中都很有用。下面我们来讨论它们在 MoE（专家混合，mixture of experts）模型中的用法。

### 在 MoE 模型中使用 CUDA 流

在实践中，Transformer 模型的各层之间存在顺序依赖，因此你无法随意地并行运行这些层。然而，在 MoE 架构中，由于不同专家（expert）的计算彼此独立，它们可以在不同的 CUDA 流上并发（concurrency）运行。

每个专家处理输入中一个独立的片段。在汇合点，各专家的输出会被聚合起来。至关重要的一点是，每个专家只能写入分配给它的那一片输出张量。如果两个专家不小心写入了相互重叠的内存区域，就会引入竞态条件（race condition），从而可能破坏结果——或者因为各专家写入顺序的不确定性而触发同步问题。

为避免这类问题，你应确保使用恰当的流级同步（例如流事件），并核实内存在各专家核函数之间被清晰地划分开来。通过在专家执行与输出聚合之间强制施加这种隔离，你就能在不牺牲并行性的前提下保持正确性。

> NVIDIA 的 Compute Sanitizer 能够检测 CUDA 代码中的并发与同步问题，包括竞态条件和死锁。此外，你还可以设置 CUDA_LAUNCH_BLOCKING=1 来强制核函数同步执行。这会通过让核函数执行变得确定，从而暴露出顺序和依赖方面的 bug。它可以揭示输出是否在尚未完全生成之前就被使用了。

在我们的示例中，理论上每个专家都可以运行在自己的流上——如果框架被扩展到多个 GPU，甚至可以运行在各自的 GPU 上。在这种情况下，收集结果时需要进行同步——最好使用流事件。

流水线并行（pipeline parallelism，PP），即让多个微批次（microbatch）流经位于不同设备上的不同模型阶段，以及同时服务多个推理请求，是两种天然能从多条流中获益的场景。例如，在流水线并行的工作流中，模型的每个阶段都有自己的流，用于并发处理不同的微批次。与此同时，它还在与相邻的阶段进行通信。

在多请求推理服务中，每个请求的模型执行都可以在各自的流上启动。在硬件资源充足的情况下，这可以通过重叠推理计算来提升吞吐量（throughput）——代价是由于资源共享开销，单个请求的延迟会有所增加。

简而言之，CUDA 流通过在多个核函数、阶段或请求之间重叠工作，帮助你从硬件中榨取出额外的性能。它们需要谨慎的同步来避免竞态条件。但只要使用得当，它们就能隐藏延迟，让 GPU 得到更充分的利用。

建议在引入并发时持续地对代码进行性能剖析。同时要记住，在 100% 利用率下的顺序执行，其性能可能实际上优于会引入资源争用的并行执行。不过，流通常能让你利用到那些原本会闲置的 GPU 部分。找到恰当的平衡点很重要。

> 务必确保你是在预期的流上启动操作。例如，不小心用了默认流，就可能重新引入不必要的串行化。这一点很容易搞错，所以值得再强调一遍。

## 用 CUDA Graphs 降低核函数启动开销

在前面的章节中我们已经看到，CUDA Graphs 能够消除每次迭代的启动开销，降低 CPU 启动开销，并消除核函数之间微小的空闲间隙。而哪怕消除最微小的空闲间隙，也能带来更高的有效利用率——以及更一致的迭代耗时。现在让我们展示如何在 PyTorch 中使用它们。

### 捕获 CUDA Graph 并预分配内存

PyTorch 提供了 torch.cuda.CUDAGraph API 来捕获（graph capture）并重放（graph replay）CUDA Graphs。一般的使用模式是：首先通过正常运行几次迭代来对模型进行预热，以初始化所有必要的数据和内存分配。接着，你创建一个 CUDAGraph 对象，以及一条专用的、非默认的 CUDA 流来隔离此次捕获。

> 当使用 "reduce-overhead" 或 "max-autotune" 时，如果模型是稳定的，编译器会自动为你捕获 CUDA Graphs。在这种情况下，你甚至不需要编写这些样板代码，因为只要你在这些模式下使用 PyTorch 编译器，它就会自动完成。而如果你的模型每次迭代形状都在变化，可以考虑使用 "max-autotune-no-cudagraphs" 模式来避免图捕获，因为 CUDA Graphs 目前要求静态形状。

随后，你在 torch.cuda.graph() 上下文中指定的捕获流上，完整执行一遍模型，以记录下操作序列。一旦这些操作被捕获进 CUDA Graph，你就可以按需在新的输入上 replay() 该图（例如用于模型训练或推理）。

在捕获 CUDA Graph 之前，捕获期间使用的所有静态内存都必须预先分配——并且最好按其最大尺寸分配。这些缓冲区包括输入、输出以及中间张量。如果在捕获期间发生任何新的内存分配，图捕获就会失败，并报出诸如 “operation not permitted when stream is capturing” 之类的错误。

为了减少碎片、并为这些固定缓冲区最大化连续的内存空间，你可以在进入捕获代码块之前立即调用 torch.cuda.empty_cache()。这会清除未使用的缓存内存，让分配器有最佳机会不受打扰地布局你预留的缓冲区。

> 频繁使用 torch.cuda.empty_cache() 会扰乱分配器的效率，并带来更长期的性能代价。请把这个调用当作捕获图时的一次性安全手段——而不是常规的维护工具。

请记住，PyTorch 的缓存分配器支持 CUDA 的异步分配器（cudaMallocAsync），以复用固定的内存地址。然而，这并不能绕过 CUDA Graphs 的要求：即不能在图内创建新的内存分配。

你仍然需要预先分配固定大小的缓冲区，以免在图内尝试分配新内存时触发运行时错误。请确保在图捕获之前的预热阶段，所有张量都达到其所需的最大尺寸。我们会在后续的图重放一节中进一步讨论这一点。

你需要为图捕获使用一条专用的、非默认的流，以避免干扰那些不应被纳入 CUDA Graph 的操作。下面是一段代码，演示如何在 PyTorch 中用专用的 capture_stream 捕获一个 CUDA Graph：

```
g = torch.cuda.CUDAGraph()
capture_stream = torch.cuda.Stream()
# Prepare static inputs and outputs
static_input = torch.randn(batch_shape, device='cuda')
static_output = torch.empty(output_shape, device='cuda')
# Warm-up step on capture_stream to allocate buffers without recording
with torch.cuda.stream(capture_stream):
    tmp = model(static_input)
    static_output.copy_(tmp)
# ensure warm-up is complete
capture_stream.synchronize()
# Begin graph capture
with torch.cuda.graph(g, stream=capture_stream):
    tmp = model(static_input)
    static_output.copy_(tmp)
# ensure capture is complete before using the graph
capture_stream.synchronize()
```

在这段代码中，我们首先在 GPU 上以固定形状分配 static_input 和 static_output。我们在 capture_stream 上运行一次预热迭代，以确保 model(static_input) 内部所需的任何内存都已分配——同时完成任何一次性的初始化工作。

> 通过预分配输出缓冲区，并在预热和捕获两个阶段都使用 static*output.copy*(tmp)，这段代码会把结果写入一个固定的内存区域。这使得被捕获的 CUDA Graph 正确、可重放、可复现，且不会出现意料之外的张量分配。

接着，我们对 capture_stream 进行同步，以确保在开始实际的图捕获之前，预热步骤已经完全完成。随后，我们在同一条流上进入 torch.cuda.graph(...) 上下文，并重新运行一遍模型的前向传播。

在捕获阶段，实际上并不会启动任何核函数。相反，这些操作只是被记录进 CUDA Graph 对象 g 中。退出捕获代码块之后，我们需要再次同步，以确保记录已经最终完成。

在捕获 CUDA Graph 时，严格的隔离至关重要。捕获流上的操作绝不能受到任何其他流上活动的影响。即便是另一个线程上一次看似无关的核函数启动，也可能使此次捕获失效，从而导致诸如 “operation not permitted when stream is capturing” 之类的运行时错误。

如果你在捕获进行期间不小心在其他流上执行了 CUDA 操作，就会出现这个错误。在这种情况下，它会使图上下文失效并触发该错误。如果你在捕获流上启动了一个执行动态内存分配的核函数，例如在捕获期间创建张量或调用 torch.empty，也会出现同样的情况。

> 务必在开始图捕获之前以及退出图捕获之后，都调用 capture_stream.synchronize()。这能确保所有操作都被正确记录，并且图已准备好可以安全地重放。前面的代码示例就遵循了这一最佳实践。

图被捕获之后，可以在任意流上重放，包括默认流——而且与捕获时所用的是哪条流无关。如果你触发了任何依赖于图完全执行完毕的 CUDA 操作，就必须在运行这些操作之前进行同步，如下所示。否则，由于图在重放时是异步运行的，那些依赖图结果的 CUDA 操作可能会在图执行完成之前就运行：

```
# replay the graph which writes to static_output
g.replay()
# synchronize on the stream that is replaying the graph
torch.cuda.current_stream().synchronize()
# Now it's safe to read or post-process the output
print(static_output)
...
```

如果没有这种显式同步，你的程序就可能继续往下执行，并错误地假定图已经执行完毕、且已把结果写入了 static_output。如果没有任何同步措施，代码可能会读到过期或只写了一部分的数据，因为图可能还没有写完 static_output。这种情形会导致不确定的行为、损坏的读取、隐蔽的竞态条件以及死锁。

### 重放图

要用新数据重放图，你需要把新的输入拷贝进预分配的输入张量 static_input，然后调用 g.replay()。GPU 会以这些张量当前的内容作为输入，执行整个被捕获的操作序列。图会把结果放入预分配的输出张量（static_output），如下所示：

```
# load new data into pre-allocated input tensor
static_input.copy_(new_batch)
# execute the captured graph
g.replay()
# retrieve the output (clone if you plan to modify it)
result = static_output.clone()
```

这里，我们把 new_batch 的数据加载到 static_input 的内存空间中，然后调用 g.replay()。图会以当前 static_input 中的数据，运行完全一致的被捕获操作，并把输出写入 static_output。之后我们便可按需使用或克隆 static_output。

建议用几个测试输入来验证图的输出与正常执行的结果一致，以确认此次捕获是成功的。另外要记住，在不同步的情况下，你不能直接对 static_output 调用 print 或 .item()。如有需要，可在重放之后、且在任何异步区域之外，执行 result = static_output.cpu().numpy() 以便调试。

由于图每次都复用相同的输入、输出以及内部内存分配，它会在每次迭代时覆写同一个输出张量。因此，如果你需要把输出保留到单次迭代之后，就需要像前面的代码那样对缓冲区调用 clone()。这也是为什么我们需要确保预热 / 捕获步骤中的内存分配覆盖了所需的最大尺寸。请记住，图无法临时处理额外的内存分配。

值得注意的是，在使用 CUDA Graph 之后，某些 PyTorch 操作在你重新捕获图之前不会生效。例如，如果你在图之外临时改动模型权重，这些更新不会应用，因为图持有自己的一份操作副本。因此，图最适合用于稳态循环——即每次迭代都重复相同操作的场景。

此外，当使用 cudaMallocAsync 为 CUDA Graph 预分配内存时，它在重放期间会复用图捕获期间预分配的那些内存地址——或者，比如当图是从磁盘加载时，复用预热期间预分配的地址。这样，后续的图重放就不需要额外的内存了。

默认情况下，每个 CUDA Graph 实例都使用自己私有的内存池。这样就能保证每个图都预分配自己的内存缓冲区，不会与该图的其他实例竞争。换句话说，两个相同的图即便并发重放，也不会争抢内存，因为它们各自使用自己的内存池和缓冲区空间。

你可以通过传入 torch.cuda.graph(pool=...) 来选择在多个图实例之间共享一个内存池。不过，这只有在非常特殊的情况下才有用——即当你出于性能考虑，想要刻意编排相关的图，让它们复用预分配的内存缓冲区时。例如，考虑同时运行多个推理图变体——每个变体使用不同的批次大小（例如 1、2、4、8 等）。在这种情况下，你可以让这些针对特定形状特化的变体复用同一块大的内存分配，从而降低整体内存占用；这块内存供 PagedAttention 使用，而 PagedAttention 用到了一个名为 block_tables 的可变大小张量。

这种做法在 FireworksAI 的一篇博客文章中有所描述。在那里，他们为每个批次大小编译一个不同的 CUDA Graph 变体。而且，他们不是为每个图创建单独的内存池，而是让所有图变体共享同一个内存池。通过按批次大小递减的顺序编译各图，共享池中的内存会从最大的变体（例如本例中的 8）开始被复用。较小批次大小的图变体，则由上一次迭代中分配的较大缓冲区来提供服务。这样，在不占用过多 GPU 内存的前提下，就支持了多个图变体。

> 这是一种冷门而巧妙的实现选择，需要对内存段进行细致的协调。不过，它确实能在同时运行多个针对特定形状特化的图时，降低 GPU 内存的总占用。对于那些必须尽量减小整体内存占用的部署场景来说，这非常有用。

### CUDA Graphs 最佳实践

CUDA Graphs 是在训练和推理中都能达到峰值稳态吞吐量的一种强有力手段。对于那些 CPU 开销和核函数启动波动正在损害性能的大规模部署，它们尤其有用。在现代 GPU 上，成千上万个操作会被极快地执行，此时图对于让 GPU 设备保持繁忙、并把每个核函数的启动开销降到最低，就变得不可或缺。

值得注意的是，PyTorch 的 torch.compile 在许多情况下都会在底层使用 CUDA Graphs——除非被显式禁用。CUDA Graphs 用于为已编译的模型最小化核函数启动开销。以下是使用 CUDA Graphs 时需要记住的几点要事：

_在图捕获期间避免分配内存_ 请记住，你不能在图捕获内部动态地分配 GPU 内存。任何需要用到的张量都应事先在预热步骤中分配，如前所示。如果你的图需要临时缓冲区，请使用 PyTorch 基于 cudaMallocAsync CUDA 后端的、图感知的缓存分配器。这样，每次重放都会复用相同的缓冲区地址。请确保这些临时张量在预热阶段（图捕获之前）就按其潜在的最大尺寸创建。这会预先分配好所有需要的内存。

_保持图结构固定_ 被捕获的图无法改变操作序列、张量形状或内存大小。如果你的工作负载偶尔会有形状变化，一种策略是捕获多个图（例如，为你预期的每种输入尺寸各捕获一个），然后在运行时选择合适的图。或者，你也可以为那些迭代禁用图（PyTorch 编译器为此类情况提供了 max-autotune-no-cudagraphs 模式）。

> CUDA 提供了一套底层的图管理 API（Graph Management API），其中包括前面章节介绍过的 cudaGraphExecUpdate()，它允许对已捕获的图做少量修改。然而，截至本文撰写时，PyTorch 并未暴露这个能力。就目前的 PyTorch 而言，最好把图当作不可变的对象来对待。

_尽可能多地捕获_ 尽量把训练循环中尽可能多的部分纳入图中——理想情况下是一整次迭代（前向传播、反向传播、优化器步进以及任何 all-reduce 通信）。你捕获得越多，消除的 CPU 开销和启动延迟就越多。

此外，如果内存允许，可以考虑在一个图中捕获多次迭代（包括循环展开等）。尽管这会让图变得更大，但它通常能通过在多次迭代之间启用更多优化来提升吞吐量。这样做的代价是牺牲灵活性，但值得去探索和剖析。

在多 GPU 环境中捕获大型图时，如果硬件支持，可使用 CUDA 流优先级。例如，如果你希望计算核函数以更高的优先级运行、不被拖延，就可以把 NCCL 调用设置到一条较低优先级的流上。图捕获也会记录下这些优先级。

> NVIDIA 的 MLPerf 提交结果（以及内部基准测试）常常把整个训练步骤捕获进每次迭代的单个图中。这包括前向传播、反向传播、优化器以及 all-reduce 通信步骤。这能消除几乎所有的运行时开销（启动抖动等），代价是内存和灵活性。

_为内存复用做好规划_ 图被捕获之后，你无法释放或重新分配该图所使用的任何张量——也无法改变它们的任何尺寸。最好按你预期的最大批次大小来捕获，从而预留出比实际所需略多一些的容量。这样，稍大一些的输入在之后重放时就不会破坏图。

例如，如果你的最大批次大小是 64，但偶尔会用 96 来运行，那么为保险起见，可以考虑按批次大小 96 来捕获。通常，捕获最坏情况、运行较小批次、浪费一点内存——要好过冒图失败的风险。

> 记得把优化器状态的大小也算进去（例如 Adam 优化器的中间缓冲区），因为它们是模型训练图的一部分。这些是比较容易被忽略的！

在使用得当的情况下，CUDA Graphs 能为大模型的训练和推理工作负载带来显著的加速。既然计算方面的优化已经就绪，让我们转向 PyTorch 中的内存优化。

### CUDA Graph Trees（PyTorch 编译器内部机制）

我们在前面的小节中已经介绍过 PyTorch 编译器，但在 CUDA Graphs 的语境下，值得一提的是 PyTorch 的 CUDA Graph Trees。它们被 torch.compile（具体来说是 mode="reduce-overhead"）用来为每种输入形状编译并缓存各自独立的静态图。

一旦一个静态图被记录下来，它的张量维度就必须保持固定。任何新的输入形状都会触发一次全新的记录和缓存条目。为了最大化缓存命中、减少图捕获，建议在多次迭代之间尽量保持输入形状的一致。不同形状越少，就越能复用已记录的图，开销也越小。

值得注意的是，你通常不会直接调用 CUDA Graph Trees 的 API。当指定 mode="reduce-overhead" 时，torch.compile 会为你处理这一切。

由于 CUDA Graphs 要求静态地址和稳定的控制流，对于 LLM 推理而言，整图捕获是很困难的，因为 LLM 推理支持可变的输入尺寸、批次大小和步数（例如采样、KV 缓存增长、主机侧决策等）。

对于模型训练，CUDA Graph Trees 允许多个被捕获的子图在前向和反向捕获之间共享同一个内存池。CUDA Graph Trees 对推理也很有用，因为它们允许根据输入形状和批次大小动态地选择子图。这通常被称为“分段捕获”（piecewise capture）。PyTorch 借助 CUDA Graph Trees 来维护按形状划分的分段捕获，并管理共享内存池。

> CUDA Graph Trees 提供的分段捕获模式，正是 vLLM 推理引擎用来支持不同输入形状与批次大小组合下的不同图的机制。我们将从第 15 章开始更详细地介绍 vLLM 和推理引擎的优化。

## 在 PyTorch 中剖析与调优内存

大模型可能会受限于 GPU 的内存容量和内存带宽。此外，即便 HBM 容量足够，低效的内存使用（例如内存碎片（memory fragmentation））也会损害性能。你可以从多个方面来应对内存问题，包括内存分配器调优、激活检查点（activation checkpointing）、内存卸载（offloading）以及输入流水线优化。

另外，PyTorch 有一个内置于 torch.profiler 的内存剖析器，只需像前面展示的那样启用 profile_memory=True 即可。你可以用它找出哪些操作分配了大量内存——并优先设法处理这些操作，如图 13-5 中由 PyTorch 内存可视化工具生成的可视化结果所示。

![图 13-5. 三次前向与反向传播迭代的 PyTorch 内存剖析可视化](AI%20Systems%20Performance%20Engineering-ch13_images/figure-13-5.png)

此外，NVIDIA Nsight Systems 的 CUDA Memory Inspector 可以帮助你可视化内存碎片随时间是如何产生的。利用这些工具可以指导你的内存分配器调优工作，接下来我们就来探讨这一点。

### 调优 CUDA 内存分配器

PyTorch 为 CUDA 内存使用了一个缓存内存分配器。默认情况下，它会通过按需切分和回收 GPU 内存块来适应各种分配模式。然而，某些使用可变大小内存分配的工作负载模式可能会导致内存碎片。

内存碎片发生在 GPU 内存随时间被切分成许多不连续的空闲块时。这会使得分配一个大张量变得困难，即便总的空闲内存仍然足够。在 MoE 模型中，这尤其成问题，因为路由到每个专家的 token 数量会随每个批次而变化。因此，每个专家的输出激活张量在每次迭代时都可能是不同的大小。

可变大小的内存分配会留下参差不齐、碎片化的内存块。这些碎片内存块会在训练或推理运行的过程中不断累积。

为避免这种情况，你应当预先分配一个固定大小的专家输出缓冲区，并把它的尺寸设为你批次中任何专家可能处理的最大 token 数量。然后，你就可以在每次迭代中复用这个缓冲区。

通过保持缓冲区维度恒定，GPU 内存分配器就不会随时间产生内存碎片。每个专家写入的是预分配缓冲区中属于自己的那一片，而不是触发新的分配。这种方法能稳定内存使用、提升复用效率，并避免与碎片相关的失败或性能下降。

你可以通过一个环境变量来调整 PyTorch 分配器的配置，从而调优其行为。下面是一个例子：

```
export PYTORCH_ALLOC_CONF=\
max_split_size_mb:256,\
roundup_power2_divisions:[256:1,512:2,1024:4,>:8],\
backend:cudaMallocAsync
```

下面是对代码中每个指定配置参数的说明：

max_split_size_mb:256 这个参数指示分配器保持大的空闲块完整（最多 256 MB），而不是持续地把它们切分成细小的碎片。这有助于减少碎片。默认情况下，PyTorch 切分大块分配的力度较小。显式设置 max_split_size_mb 可以让大的连续空闲块能够供现代 LLM 中的大型神经网络层使用。
