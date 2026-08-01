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

This example uses two CUDA streams: a dedicated transfer stream for asynchronous host-to-device copies and the default compute stream for model work. This hides H2D latency and matches PyTorch’s recommended pattern for overlapping communication and computation.

Specifically, while the model is processing a batch on the default stream, the next batch’s data transfer is already underway on the transfer_stream. Synchronizing with compute_stream.wait_stream(transfer_stream) before consuming the preloaded batch enforces correct ordering without a full device-wide barrier. And the .to(device, non_blocking=True) calls make sure that the copy uses asynchronous DMA-based copies that don’t block the calling CPU thread.

Using next(dataloader_iter, None) gives explicit control over when transfers are enqueued versus when the kernel operations run. This makes sure one batch of data is moving on the transfer stream while another batch is executing on the compute stream, as shown in Figure 13-4.

![Figure 13-4. Overlapping compute and data transfer with dedicated compute and transfer streams](AI%20Systems%20Performance%20Engineering-ch13_images/figure-13-4.png)

Additionally, by pulling from dataloader_iter ahead of time and storing in next_inputs, next_labels, this code separates batch loading (running on the CPU in transfer_stream) from batch processing (running on the GPU in compute_stream).

This split means you always have one batch in flight for each stream. This decouples data loading from compute and maximizes overlap.

> Always profile with Nsight Systems or the PyTorch profiler when adding streams. Look at GPU utilization, and, if done correctly, you’ll see near 100% utilization with transfer and compute overlapping. If utilization drops—or you see data transfers and compute happening sequentially—double-check for any implicit synchronizations such as CUDA tensor print()—or extra CUDA synchronizations in your code.

### Stream Synchronization with Events

When using multiple streams, it’s sometimes necessary to coordinate between them to make sure that one stream’s work is done before another stream uses its results, for instance. A lightweight way to synchronize specific points is using CUDA events, as discussed in Chapter 11.

With CUDA events, you can record an event on one stream and make another stream wait for that event. This avoids the heavyweight full synchronization of torch.cuda.synchronize() and, instead, synchronizes only the necessary streams needed to process an individual event.

In fact, even in a multi-GPU context, events can be used to synchronize work across devices. In this case, one GPU records an event on its device stream while another GPU waits for the event on its device stream. This is how NCCL handles dependencies under the hood.

By using events smartly, you keep the multiple GPUs working in parallel as much as possible. Next is an example of using CUDA events to synchronize two streams, stream1 and stream2 in PyTorch:

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

In this code, we record an event at the end of some work on stream1. Later, before launching work on stream2, we call stream2.wait_event(event). This inserts a dependency such that stream2 will not execute its next kernel until the event is signaled by stream1 reaching that point. Events are useful for scheduling lightweight dependencies between streams, as they avoid heavy, global synchronizations that will stall all stream execution.

Let’s revisit the PyTorch data loader/overlap example in the previous section and rewrite it to synchronize with CUDA events. We are using the same pair of streams from earlier (transfer_stream and compute_stream), but we’re adding a transfer_done CUDA event to synchronize at a fine-grained, event-specific level:

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

This is the same overlap pattern as the previous section, but it uses a CUDA event for the transfer → compute synchronization. This is in contrast to the wait_stream() mechanism used in the other example.

This code still uses next(dataloader_iter) to preload batches ahead of time (same as the example in the previous section). This way, data transfer and compute are always overlapping.

However, in this example, the transfer_done event is recorded with transfer_done.record(stream=transfer_stream) on the transfer_stream right after the asynchronous copy is enqueued. This timestamps the event.

The compute_stream.wait_event(transfer_done) then stalls the compute_stream until the copy is complete and the transfer_done event is triggered. It then consumes the prefetched batch and performs its compute operations on the compute_stream.

Besides data loading, CUDA streams are useful in a lot of different contexts. Let’s discuss how they’re used in MoE models.

### Using CUDA Streams with MoE Models

In practice, transformer-model layers are sequentially dependent, so you can’t arbitrarily run layers in parallel. However, in an MoE architecture, different experts can run concurrently on separate CUDA streams since their computations are independent.

Each expert processes a distinct segment of the input. At the join point, the expert outputs are aggregated. It’s essential that each expert writes only to its assigned slice of the output tensor. If two experts accidentally write to overlapping memory regions, this will introduce a race condition, which can corrupt the results—or trigger synchronization issues caused by the nondeterministic order of writes from the experts.

To avoid such issues, you should make sure to use proper stream-level synchronization (e.g., stream events) and verify that memory is cleanly partitioned across expert kernels. By enforcing this separation between expert execution and output aggregation, you will maintain correctness without sacrificing parallelism.

> NVIDIA’s Compute Sanitizer can detect concurrency and synchronization issues in CUDA code, including race conditions and deadlocks. Also, you can set CUDA_LAUNCH_BLOCKING=1 to force synchronous kernel execution. This will surface ordering and dependency bugs by making kernel execution deterministic. This will reveal if outputs are being consumed before they’re fully produced.

In our example, each expert could, in theory, run on its own stream—or even its own GPU if the framework is extended to multiple GPUs. In this case, synchronization—ideally using stream events—is needed when gathering the results.

Pipeline parallelism, or pipelining microbatches through different model stages on different devices, and serving multiple inference requests are two scenarios that naturally benefit from multiple streams. In a pipeline parallel workflow, for instance, each stage of the model has its own stream that processes a different microbatch concurrently. Meanwhile, it’s also communicating with neighboring stages.

In multirequest inference serving, each request’s model execution can be launched in its own stream. With sufficient hardware resources, this can increase throughput by overlapping inference computations—at the cost of some per-request latency due to resource-sharing overhead.

In short, CUDA streams help to squeeze extra performance out of your hardware by overlapping work across multiple kernels, stages, or requests. They require careful synchronization to avoid race conditions. But, when used correctly, they can hide latency and keep a GPU more fully utilized.

It’s recommended to continuously profile your code when introducing concurrency. And keep in mind that sequential execution at 100% utilization may actually perform better than parallel execution that introduces resource contention. Often, though, streams let you utilize parts of the GPU that would otherwise sit idle. Finding the right balance is important.

> Always make sure you are launching on the intended stream. Accidentally using the default stream, for example, can reintroduce unnecessary serialization. This is easy to mess up, so it’s worth repeating again.

## Reducing Kernel Launch Overhead with CUDA Graphs

We’ve seen in earlier chapters that CUDA Graphs eliminate per-iteration launch overhead, reduce CPU launch overhead, and eliminate tiny idle gaps between kernels. And removing even the smallest idle gaps leads to higher effective utilization—and more consistent iteration times. Now let’s show how to use them in PyTorch.

### Capturing a CUDA Graph and Preallocating Memory

PyTorch provides a torch.cuda.CUDAGraph API to capture and replay CUDA Graphs. The general usage pattern is to first warm up the model by running a few iterations normally to initialize all necessary data and allocations. Next, you create a CUDAGraph object and a dedicated, nondefault CUDA stream to isolate the capture.

> When using "reduce-overhead" or "max-autotune", the compiler will automatically capture CUDA Graphs for you if the model is stable. In this case, you don’t even need to write this boilerplate since it’s done for you automatically if you’re using the PyTorch compiler with these modes. And if your model has varying shapes each iteration, consider the "max-autotune-no-cudagraphs" mode to avoid graph capture, as CUDA Graphs currently require static shapes.

You then perform a full pass of the model’s execution to record the sequence of operations using the capture stream specified in the torch.cuda.graph() context. Once the operations are captured in the CUDA Graph, you can replay() the graph on new inputs as needed (e.g., model training or inference.)

Before capturing a CUDA Graph, all static memory used during capture must be preallocated—and preferably at its maximum size. These buffers include inputs, outputs, and intermediate tensors. If any new memory allocation happens during capture, the graph will fail with an error such as “operation not permitted when stream is capturing.”

To reduce fragmentation and maximize contiguous memory space for these fixed buffers, you can invoke torch.cuda.empty_cache() immediately before entering the capture block. This will clear unused cached memory and give the allocator the best chance to lay out your prereserved buffers without interruption.

> Frequent use of torch.cuda.empty_cache() can disrupt allocator efficiency and incur longer-term performance costs. Treat this call as a one-time safety mechanism when capturing a graph—and not a regular maintenance tool.

Remember that PyTorch’s caching allocator supports CUDA’s asynchronous allocator (cudaMallocAsync) to reuse fixed memory addresses. However, this does not bypass CUDA Graphs’ requirement to not create new allocations within the graph.

You still need to allocate fixed-size buffers upfront to avoid a runtime error if you try to allocate new memory inside the graph. Make sure all tensors reach their maximum required size during the warm-up prior to graph capture. We’ll cover more about this in the upcoming graph replay section.

You need to use a dedicated, nondefault stream for graph capture to avoid interference with any operations that should not be included in the CUDA Graph. Here is a code snippet demonstrating how to capture a CUDA Graph in PyTorch with a dedicated capture_stream:

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

In this code, we first allocate static_input and static_output on the GPU with fixed shapes. We run one warm-up iteration on capture_stream to ensure that any memory needed inside model(static_input) is allocated—and to perform any one-time setup.

> By preallocating the output buffer and using static_output.copy_(tmp) in both warm-up and capture phases, the code writes its results into a fixed memory region. This makes the captured CUDA Graph correct, replayable, and reproducible without unexpected tensor allocations.

We then synchronize the capture_stream to make sure that the warm-up step is fully complete before beginning the actual graph capture. Next, we enter the torch.cuda.graph(...) context on the same stream and rerun the model forward pass.

During the capture phase, no kernels are actually launched. Instead, the operations are just recorded into the CUDA Graph object g. After exiting the capture block, we need to synchronize once more to make sure the recording is finalized.

When capturing a CUDA Graph, strict isolation is essential. Operations on the capture stream must not be affected by activity on any other streams. Even a seemingly unrelated kernel launch on another thread can invalidate the capture. This could lead to runtime errors like “operation not permitted when stream is capturing.”

This error happens if you accidentally perform CUDA operations on other streams while the capture is in progress. In this case, it invalidates the graph context and triggers this error. This can also happen if you launch a kernel on the capture stream that performs dynamic memory allocation, like tensor creation or calling torch.empty during capture.

> Always call capture_stream.synchronize() both before starting the graph capture and after exiting it. This ensures that all operations are correctly recorded and that the graph is ready for safe replay. The previous code example follows this best practice.

After capturing the graph, it can be replayed on any stream, including the default stream—and regardless of which stream was used for capture. If you trigger any CUDA operations that depend on the graph completing fully, you must synchronize before running the operations, as shown next. Otherwise, because the graph runs asynchronously when replayed, the CUDA operations that depend on the graph results may run before the graph completes execution:

```
# replay the graph which writes to static_output
g.replay()
# synchronize on the stream that is replaying the graph
torch.cuda.current_stream().synchronize()
# Now it's safe to read or post-process the output
print(static_output)
...
```

Without this explicit synchronization, your program could proceed and incorrectly assume that the graph has finished executing and written its results to static_output. If no synchronization is in place, the code may read stale or partially written data because the graph may not have finished writing to static_output. This scenario will cause nondeterministic behaviors, corrupted reads, subtle race conditions, and deadlocks.

### Replaying the Graph

To replay the graph with new data, you copy the new inputs into the preallocated input tensors, static_input, and call g.replay(). The GPU will execute the entire captured sequence of operations using the current contents of those tensors as input. The graph will place the results in the preallocated output tensor (static_output), as shown here:

```
# load new data into pre-allocated input tensor
static_input.copy_(new_batch)
# execute the captured graph
g.replay()
# retrieve the output (clone if you plan to modify it)
result = static_output.clone()
```

Here, we load new_batch data into the memory space of static_input and then call g.replay(). The graph runs the exact captured operations using the current static_input data and writes the outputs into static_output. We can then use or clone static_output as needed.

It’s recommended to validate that the graph’s output matches a normal execution for a few test inputs to make sure the capture was successful. Also, remember that you cannot directly print or call .item() on static_output without syncing. If needed, do result = static_output.cpu().numpy() after replay and outside of any asynchronous regions for debugging.

Since the graph reuses the same input, output, and internal memory allocations each time, it will overwrite the same output tensor on each iteration. So if you need to preserve the output beyond a single iteration, you need to clone() the buffer, as shown in the previous code. This is why we need to make sure that the allocations in the warm-up/capture step cover the maximum sizes needed. Remember that the graph cannot handle additional memory allocations on the fly.

It’s worth noting that after using a CUDA Graph, certain PyTorch operations will not be reflected until you recapture the graph. For instance, if you change model weights on the fly and outside of the graph, these updates won’t apply because the graph has its own copy of operations. As such, graphs work best in steady-state loops in which operations are repeated on each iteration.

Also, when using cudaMallocAsync to preallocate memory for a CUDA Graph, it will reuse the same memory addresses during replay that it preallocated during graph capture—or during warm-up if the graph is loaded from disk, for instance. This way, subsequent graph replays do not require additional memory.

By default, every instance of a CUDA Graph uses its own private memory pool. This way, you’re guaranteed that each graph preallocates its own memory buffers and does not compete with other instances of the graph. In other words, two identical graphs will not compete for memory if they’re replayed concurrently since they each use their own memory pool and buffer space.

You can choose to share a memory pool between graph instances by passing torch.cuda.graph(pool=...). However, this is useful only in very special cases when you want to purposely orchestrate related graphs to reuse the preallocated memory buffers for performance reasons. For example, consider simultaneously running multiple inference graph variants—each using a different batch size (e.g., 1, 2, 4, 8, etc.). In this case, you can reduce overall memory usage by having the shape-specialized variants reuse a single, large memory allocation for PagedAttention, which uses a varying-size tensor called block_tables.

This approach is described in a FireworksAI blog post. Here, they compile a different CUDA Graph variant for each batch size. And, instead of creating a separate memory pool for each graph, they share a single shared memory pool across all graph variants. By compiling graphs in decreasing order of batch size, memory from the shared pool is reused from the largest variant (e.g., 8, in this case). Smaller batch-size graph variants are serviced by the larger allocated buffers from a previous iteration. This way, multiple graph variants are supported without using excessive GPU memory.

> This is an obscure and clever implementation choice that requires careful coordination of the memory segments. However, it does reduce the total GPU memory usage when running multiple shape-specialized graphs simultaneously. This is great for deployment scenarios in which minimizing overall memory usage is critical.

### Best Practices for CUDA Graphs

CUDA Graphs are a powerful way to reach peak steady-state throughput in both training and inference. They are especially useful for large deployments in which CPU overhead and kernel-launch variability are hurting performance. On modern GPUs, which execute thousands of operations extremely quickly, graphs become essential to keep the GPU devices busy and minimize per-kernel launch overhead.

It’s worth noting that PyTorch’s torch.compile uses CUDA Graphs under the hood in many cases—unless explicitly disabled. CUDA Graphs are used to minimize kernel launch overhead for compiled models. Here are a few important things to remember when using CUDA Graphs:

*Avoid allocating memory during graph capture* Remember that you can’t dynamically allocate GPU memory inside the graph capture. Any tensor that’s needed should be allocated beforehand during the warm-up step, as shown previously. If your graph needs temporary buffers, use PyTorch’s graph-aware caching allocator using the cudaMallocAsync CUDA backend. This way, each replay reuses the same buffer addresses. Make sure these temporary tensors are created during the warm-up phase (before graph capture) using their maximum potential size. This will preallocate all the needed memory upfront.

*Keep the graph structure fixed* A captured graph cannot change the sequence of operations, tensor shapes, or memory sizes. If your workload has occasional shape variations, one strategy is to capture multiple graphs (e.g., one for each input size you expect) and then select the appropriate graph at runtime. Alternatively, you can disable graphs for those iterations (PyTorch’s compiler has a mode max-autotune-no-cudagraphs for such cases).

> CUDA supports a low-level Graph Management API, including cudaGraphExecUpdate(), described in an earlier chapter, which allows minor modifications to a captured graph. However, as of this writing, PyTorch does not expose this. Within PyTorch currently, it’s best to treat graphs as immutable.

*Capture as much as possible* Include as much of the training loop as you can in the graph—ideally an entire iteration (forward pass, backward pass, optimizer step, and any all-reduce communications). The more you capture, the more CPU overhead and launch latency you eliminate.

Also, consider capturing multiple iterations in one graph if memory allows (including loop unrolling, etc.). Although this makes the graph bigger, it can often improve throughput by enabling even more optimizations across iterations. This comes at the cost of flexibility, but it’s worth exploring and profiling.

When capturing large graphs in multi-GPU setups, use CUDA stream priorities if supported. For instance, you can set NCCL calls to a lower-priority stream if you want the compute kernels to run at a higher priority and not be delayed. Graph capture will record these priorities as well.

> NVIDIA’s MLPerf submissions (and internal benchmarks) often capture the entire training step into one graph per iteration. This includes the forward pass, backward pass, optimizer, and all-reduce communication steps. This eliminates nearly all runtime overhead (launch jitter, etc.), at the expense of memory and flexibility.

*Plan for memory reuse* After a graph is captured, you can’t free or reallocate any tensors used by that graph—or change any of their sizes. It’s best to reserve a bit more capacity than needed by capturing with the maximum batch size that you expect. This way, a slightly larger input won’t break the graph later during replay.

For example, if your maximum batch size is 64 but you occasionally run with 96, consider capturing with batch size 96 just to be safe. It’s usually better to capture the worst-case scenario, run smaller batches, and waste a bit of memory—rather than risking a graph failure.

> Remember to account for the sizes of the optimizer states (e.g., the Adam optimizer intermediate buffers) since they’re part of model-training graphs. These are somewhat easy to forget!

When used appropriately, CUDA Graphs can provide significant speedups for large model training and inference workloads. With compute optimizations in place, let’s turn to memory optimizations in PyTorch.

### CUDA Graph Trees (PyTorch Compiler Internal)

We’ve covered the PyTorch compiler in a previous section, but in the context of CUDA Graphs, it’s worth mentioning PyTorch’s CUDA Graph Trees. These are used by torch.compile, and specifically mode="reduce-overhead", to compile and cache separate static graphs for each input shape.

Once a static graph is recorded, its tensor dimensions must remain fixed. Any new input shape will trigger a fresh recording and cache entry. To maximize cache hits and reduce graph captures, it’s recommended to keep your input shapes as consistent as possible across iterations. Fewer distinct shapes mean more reuse of the recorded graphs and less overhead.

It’s worth noting that you don’t typically invoke the CUDA Graph Trees API directly. This is handled for you by torch.compile when specifying mode="reduce-overhead".

Since CUDA Graphs require static addresses and steady control flow, full-graph capture is difficult for LLM inference, which supports variable input sizes, batch sizes and number of steps (e.g., sampling, KV cache growth, host-side decisions, etc.)

For model training, CUDA Graph Trees allow multiple captured subgraphs to share a single memory pool across forward and backward captures. CUDA Graph Trees are also useful for inference since they allow dynamic subgraph selection depending on the input shape and batch size. This is commonly called “piecewise capture.” PyTorch leverages CUDA Graph Trees to maintain per-shape piecewise captures and manage shared pools.

> The piecewise capture pattern provided by CUDA Graph Trees is the mechanism used by the vLLM inference engine to support different graphs for different input shape and batch-size combinations. We cover vLLM and inference-engine optimizations in more detail starting in Chapter 15.

## Profiling and Tuning Memory in PyTorch

Large models can be limited by GPU memory capacity and memory bandwidth. Additionally, inefficient memory use such as memory fragmentation can hurt performance even if HBM capacity is sufficient. You can address memory issues on several fronts, including memory-allocator tuning, activation checkpointing, memory offloading, and input-pipeline optimization.

Also, PyTorch has a memory profiler that’s built into torch.profiler by enabling profile_memory=True, as shown earlier. You can use this to find out which operations allocate a lot of memory—and try to address those operations first, as shown in the visualization in Figure 13-5 generated by the PyTorch memory visualizer tool.

![Figure 13-5. PyTorch memory profile visualization for three iterations of a forward and backward pass](AI%20Systems%20Performance%20Engineering-ch13_images/figure-13-5.png)

Also, NVIDIA’s Nsight System’s CUDA Memory Inspector can help visualize how memory fragmentation happens over time. Utilizing these can guide your memory-allocator tuning efforts, as we’ll explore next.

### Tuning the CUDA Memory Allocator

PyTorch uses a caching memory allocator for CUDA memory. By default, it adapts to allocation patterns by splitting and recycling GPU memory blocks on demand. However, certain workload patterns that use variable-sized memory allocations can lead to memory fragmentation.

Memory fragmentation happens when GPU memory gets split into many noncontiguous free chunks over time. This makes it difficult to allocate a large tensor even if enough total memory remains free. In MoE models, this is especially problematic because the number of tokens routed to each expert can change with every batch. As such, each expert’s output activation tensor may be a different size on every iteration.

Variable-sized memory allocations leave behind uneven, fragmented memory blocks. These fragment memory blocks accumulate across training or inference runs.

To avoid this, you should allocate a fixed-size expert output buffer upfront and size it to the maximum possible number of tokens any expert may process in your batch. Then you can reuse this buffer on every iteration.

By keeping the buffer dimensions constant, the GPU memory allocator won’t fragment memory over time. Each expert writes into its slice of the preallocated buffer rather than triggering new allocations. This method stabilizes memory usage, improves reuse efficiency, and avoids fragmentation-related failures or slowdowns.

You can adjust PyTorch’s allocator configuration using an environment variable to tune its behavior. Here is an example:

```
export PYTORCH_ALLOC_CONF=\
max_split_size_mb:256,\
roundup_power2_divisions:[256:1,512:2,1024:4,>:8],\
backend:cudaMallocAsync
```

Next is a description of each specified configuration parameter in the code:

max_split_size_mb:256 This parameter instructs the allocator to keep large free blocks intact (up to 256 MB) rather than continually splitting them into tiny pieces. This helps reduce fragmentation. By default, PyTorch splits large allocations less aggressively. Setting max_split_size_mb explicitly allows large contiguous free blocks to be available for large neural network layers in modern LLMs.