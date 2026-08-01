> Linux perf for NVIDIA PMUs is limited to device-level link and fabric events such as NVLink-C2C on Grace-Blackwell. SM pipeline, warp stall, and memory throughput counters remain CUPTI and Nsight tools only. These PMUs do not expose SM-level kernel metrics. For SM utilization, occupancy, and memory throughput, use Nsight Compute or a CUPTI-based profiler. Make sure to set NVreg_RestrictProfilingToAdminUsers=0 to allow non-root profiling of SM-level hardware counters.

Once the PMU devices are present, you can collect CPU and NVIDIA events together. Use symbolic event names reported by perf list:

```
perf list | grep -i nvidia
perf stat -a \
  -e nvidia_nvlink_c2c0_pmu_0/cycles/ \
  -e cycles,cache-misses \
  python train.py
```

Here, the cycles event on the NVLink-C2C PMU lets you correlate GPU interconnect activity with host CPU behavior. The following is example output from the preceding perf stat command, which shows the NVLink C2C PMU recorded activity during the run while the CPU incurred cycles and cache misses:

```
Performance counter stats for 'python train_deepseek_v3.py':
       3,567,890,123  nvidia_nvlink_c2c0_pmu_0/cycles
          45,678,901  cycles
           7,890,123  cache-misses
       2.345678901 seconds time elapsed
```

Our initial profiling revealed that GPU compute (e.g., expert matrix multiplies) and GPU communication (e.g., dispatch and combine operations) are the primary bottlenecks based on the PyTorch profiler and Nsight tools. However, CPU, data loading, and GPU collective communication operations also impact performance as perf demonstrates by showing which CPU threads and interconnect PMUs are active during the slow regions.

> To dig into additional link and fabric request counters, pick additional NVIDIA PMU events that appear on your system from perf list and add them to the perf stat command, as shown previously.

In short, by combining high-level CPU throughput metrics and call graph hotspots from perf with device metrics and timelines from Nsight Systems and Nsight Compute, you can build a holistic performance story across host and device. Start by addressing the largest CPU-side bottlenecks and data stalls. Next, optimize GPU communication and tune the GPU kernels.

## PyTorch Compiler (torch.compile)

One of the quickest wins in PyTorch is to use the PyTorch compiler with torch.compile(). The compiler stack includes TorchDynamo, AOT Autograd, and TorchInductor, which capture graphs, fuse ops, and generate high-performance code for the target backend (e.g., NVIDIA GPUs).

The PyTorch Compiler can eliminate a lot of Python interpreter overhead and GPU kernel launch latency by fusing together many small operations into larger kernels. After doing our baseline profiling, we enabled torch.compile on the model to see if we could get an easy speedup. Next, let’s describe this process—and the results.

### Using the PyTorch Compiler

Using the PyTorch compiler with the default settings is straightforward and requires no code changes besides wrapping the model as shown here: model = torch.compile(model). Under the hood, TorchDynamo traces the Python code, AOT Autograd captures the backward pass, and TorchInductor, which leverages OpenAI’s Triton for GPU kernel code generation (as discussed in the next chapter), produces efficient fused kernels automatically.

The compiler observes our model’s forward pass and identifies many opportunities to fuse consecutive operations, such as elementwise activations, layer norms, etc. It generates fused kernels for those operations—and also for parts of the backward pass. The result is significantly fewer kernel launches and less CPU overhead per iteration.

The compile step does introduce some overhead on the order of seconds—or even minutes for very large models—but this cost is amortized over long training jobs or repeated inference runs. Fortunately, TorchInductor caches compiled kernels so that subsequent runs don’t pay the compile cost again. The PyTorch community is also continuously working to improve compile/startup performance by allowing you to save and reuse compiled artifacts across runs. Use torch.compiler.save_cache_artifacts() and torch.compiler.load_cache_artifacts() to persist TorchInductor outputs across runs or nodes. This reduces startup on long-running training or serving.

One example is the PyTorch Mega-Cache feature. This is an end-to-end compile cache that lets you save compiled kernels to disk and reload them in future runs. With PyTorch Mega-Cache, you can compile once (e.g., offline) and reuse the optimized kernels across multiple training sessions. This helps to reduce startup time. You’ll still benefit from TorchInductor’s kernel optimizations like warp specialization, but you’ll avoid recompiling the graph each time.

> You can even use this compile cache on other compute nodes. If you do this, make sure CUDA, PyTorch, and Triton versions are compatible across the nodes.

It’s worth noting that PyTorch’s Compiler applies sophisticated optimization techniques internally. For example, we mentioned warp specialization in Chapter 10. TorchInductor’s autotuner generates multiple kernel variants across tile sizes, memory access patterns, etc. It will apply techniques like memory-warp versus compute-warp specialization behind the scenes. It will then choose the fastest variant for your hardware automatically.

TorchInductor supports prologue and epilogue fusion around GEMM kernels. For example, bias-add comes before the matmul. And, after the matmul, the epilogue consists of elementwise operations such as activation, dropout, and residual.

By merging these kernel prologue and epilogue operations into a single optimized kernel, TorchInductor reduces memory traffic, minimizes kernel-launch overhead, and increases occupancy. You can verify this with the profiler, which will show higher SM utilization.

This optimization complexity remains entirely transparent to the developer since PyTorch presents a clean, tensor-centric interface without exposing CUDA-level warp details. So while you won’t see “memory warp” or “compute warp” flags in the PyTorch API, just know that these techniques are being used under the hood. Once the code is compiled, you will notice the benefits of warp specialization in profiler metrics, including higher occupancy, fewer memory latency stalls, and increased SM utilization.

To illustrate the benefit of a compiled mode, let’s compare PyTorch’s eager mode versus compiled execution on an MoE model. We’ll time a single training iteration of the model in regular eager mode and then again with the "max-autotune" compiled mode. The code is shown here, followed by the example output:

```
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
# ---- Setup Model ----
device = 'cuda'
model_name = "..."
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
# ---- Create a dummy batch of token IDs ----
batch_size = 4
input_texts = ["MoE's are awesome!"] * batch_size
enc = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
input_ids = enc.input_ids.to(device)
attention_mask = enc.attention_mask.to(device)
labels = input_ids.clone()  # for causal LM, labels = input_ids
# ---- Make runs deterministic ----
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# --- Eager timing ---
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
for _ in range(iters_warmup):
    out = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = out.loss if hasattr(out, "loss") else out
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()
start.record()
for _ in range(iters_meas):
    out = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = out.loss if hasattr(out, "loss") else out
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
end.record()
torch.cuda.synchronize()
print(f"Eager mode step time: {start.elapsed_time(end)/iters_meas:.3f} ms")
# --- Compile the model (choose one mode)
# enables graph trees
compiled_model = torch.compile(model, mode="reduce-overhead")
# Alternatives:
# more tuning, longer compile
# compiled_model = torch.compile(model, mode="max-autotune")
# balanced
# compiled_model = torch.compile(model, mode="default")
# Warm-up compiled
for _ in range(iters_warmup):
    out = compiled_model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = out.loss if hasattr(out, "loss") else out
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
torch.cuda.synchronize()
start.record()
for _ in range(iters_meas):
    out = compiled_model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = out.loss if hasattr(out, "loss") else out
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
end.record()
torch.cuda.synchronize()
print(f"Compiled mode step time: {start.elapsed_time(end)/iters_meas:.3f} ms")
```

In this case, PyTorch eager mode takes roughly 248 ms per iteration. After warming up and letting the compiler perform its optimizations, the compiled mode runs in about 173 ms. Our compiled version, using "max-autotune", runs ~30% faster than eager execution. Actual speedup varies with model structure, batch size, and dynamic shapes. Dense models dominated by a single large GEMM may only see <10% speedup.

These savings come primarily from combining many small GPU kernels. Small GPU kernels are common in an MoE architecture for token dispatch/combine and per-token activation patterns. By fusing these small operations into fewer, larger kernels, we keep intermediate data in faster on-chip memory—such as registers and shared memory—rather than repeatedly moving data to and from global device memory (HBM).

> For highly dynamic token routing used by MoE architectures, prefer default or max-autotune-no-cudagraphs. Then switch to max-autotune once input shapes stabilize—or when you use CUDA Graph Trees with limited discrete shapes.

In this case, many of the small operations, like dispatch/combine, activation functions, etc., are fused away by TorchInductor. If you examine the trace of the compiled run, you’ll see far fewer GPU kernels on the timeline. Instead, there will be fewer, but slightly longer-running, kernels that correspond to merged operations performing multiple steps at once.

Over many iterations, the benefit is even more pronounced as the one-time compilation overhead is amortized. Profiling the compiled model’s execution shows far fewer GPU kernel launches, as many operations that were separate in eager mode are now fused together.

It’s worth noting that if a dense model is dominated by one massive GEMM due to a large linear layer, for example, it may see only modest gains (e.g., < 10%) from torch.compile. This is because the model is likely using Tensor Cores efficiently—and because there is little opportunity for kernel fusion and Python overhead removal.

However, sparse architectures like MoE, with hundreds of medium-sized matmul operations, will see big wins since compilation reduces Python overhead, lowers kernel launch latency, and fuses multiple steps into optimized kernels. As such, the PyTorch compiler leads to significantly larger performance gains for MoEs compared to dense models.

In addition to automatic operator fusion, you can integrate user-defined custom kernels directly into the torch.compile workflow. This approach combines the best of both worlds since it uses compiler-managed graph-level optimizations for general patterns while giving you full control when needed.

For instance, you can write a specialized Triton or CUDA kernel for a performance-critical operation and register it as a custom operator. When the model is traced and compiled, TorchInductor will treat it as a single fused operation within the larger execution graph. The result is a combination of custom hand-tuned code embedded in a fully optimized, compiler-managed execution graph.

TorchInductor’s flexibility lets your custom kernel benefit from the compiler optimizing the surrounding operations (e.g., fusion with adjacent layers, etc.). In practice, this means you can use your own high-performance kernel without losing PyTorch compiler’s ability to optimize the rest of the model.

In short, by using PyTorch compile, including its "max-autotune" mode within our training loop, you can get decent speedups on modern GPUs with relatively low effort. This can be verified holistically using torch.profiler, Linux perf, Nsight Systems, Nsight Compute, and other helpful profilers.

### Compiling Versus Writing Custom Kernels

With a compiler backend like TorchInductor, many operations will be fused into efficient kernels automatically. As we saw, simply using torch.compile gives a decent-sized boost with minimal effort. However, there may be times when the automatically generated code is not as optimal as a custom-crafted kernel—or when an operation isn’t captured by the compiler at all. This raises the question: when should you rely on the compiler’s fusion versus writing a custom CUDA kernel yourself?

For most cases, using high-level torch.compile with graph capture—and TorchInductor under the hood—is preferred. This is much less effort than writing custom CUDA kernels—and often produces good enough performance improvement without specialized coding.

TorchInductor already applies many advanced optimizations internally, such as fusion of elementwise operations, merging of layer operations, layout optimization, etc. Writing fused kernels by hand would be time-consuming and brittle, whereas the compiler can do these automatically in most cases.

If your model uses a novel operation or pattern that the compiler doesn’t handle well, you may need to write a custom kernel and integrate a custom operator with PyTorch. In the next chapter, you’ll see how to do this in more detail.

In short, use torch.compile as the first resort for performance tuning since it’s easy, sufficient, and relatively “free.” Creating custom kernels is the next level of optimization and is used when built-in automation isn’t enough. Only after that should you consider writing custom kernels for the remaining hotspots to fuse certain unsupported optimizer operations or specialized attention patterns. However, even for specialized attention patterns, PyTorch provides the FlexAttention API (prefill) and FlexDecoding API (decode), which are the preferred ways to implement custom attention kernels in PyTorch for training and inference, as we’ll see in an upcoming section.

### Compilation Modes and Trade-Offs in Speed, Memory, and Compile Time

PyTorch provides several modes for torch.compile that tune the compiler’s aggressiveness and capabilities for different scenarios. You can explicitly select a mode using torch.compile(model, mode="..."). The choices are "default", "reduce-overhead", "max-autotune", and "max-autotune-no-cudagraphs". Each mode provides a combination of options regarding CUDA Graphs, autotuning, and optimization level. The modes are summarized in Table 13-5.

Table 13-5. Summary of compilation modes and their key characteristics

| Mode                       | Description                                                                                                                                                                         | Compile time        | Extra memory               | Notable features                                            |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------- | -------------------------- | ----------------------------------------------------------- |
| default                    | Balanced optimizations (good speed without long compile or extra mem); includes minor autotuning; may use CUDA Graphs for stable segments                                           | Low–Medium          | No                         | General fusion, basic autotuning                            |
| reduce-overhead            | Reduces per-iteration overhead (good for small batches); ideal for inference or small batches; automatically skips CUDA Graphs if it detects dynamic shapes to preserve correctness | Medium              | Yes (workspace caching)    | Uses CUDA Graphs (if possible) to eliminate launch overhead |
| max-autotune               | Maximizes runtime performance (best for long runs); longer compile time; best for aggressive tuning for a large amount of SMs and GPU memory                                        | High (slow compile) | Maybe (if graphs are used) | Aggressive Triton autotuning; enables CUDA Graphs on GPU    |
| max-autotune-no-cudagraphs | Does everything max-autotune does but without CUDA Graph capture; best for dynamic shapes or for debugging issues masked by CUDA Graphs                                             | High                | No                         | Same as max-autotune but disables graphs for flexibility    |

Here’s a detailed description of the modes in Table 13-5:

default This is the default mode if no mode is explicitly specified. This mode provides a balance of reasonably fast compiled code without excessive compile time or memory usage. It will do standard optimizations and use the default TorchInductor backend. This mode is often best for large models where compile time needs to be moderate—or when memory is tight. This mode includes minor autotuning and may use CUDA Graphs for stable segments. But it tries to balance speed and compile-time cost.

reduce-overhead This mode focuses on minimizing Python and runtime overhead. This is especially useful for small models—or models that perform a short number of executions per iteration. In these cases, even a little bit of overhead hurts performance. This mode leverages CUDA Graphs aggressively to eliminate per-iteration launch overhead. It may also allocate some extra memory for persistent use, such as workspace memory that is reused. This avoids frequent CUDA malloc and free calls. For example, it might cache a large working tensor instead of allocating it each iteration. This mode will automatically skip CUDA Graphs and fall back to eager if it detects dynamic shapes. It does this to preserve correctness.

This mode can speed up inference and training in low-latency scenarios—at the cost of some additional memory. Note that CUDA Graphs require that the graph’s memory addresses stay constant, so this mode can be used only when input sizes do not change and certain operations, such as dynamic-shape operations, are not present. Otherwise, graphs would break or be recompiled. The compiler will automatically fall back if it can’t apply a CUDA Graph in a given segment.

max-autotune This mode generates the fastest possible code without regard for compile time. It will run extensive autotuning of kernels by trying many tiling configurations for matrix multiplications, for instance, and utilize any known performance optimizations in TorchInductor. On modern GPUs, max-autotune also enables CUDA Graphs by default for stable execution.

The compilation process in this mode can be significantly longer—on the order of minutes for a large model. It’s intended for scenarios in which you compile once and run the model many times, such as running long training jobs or deploying a model that will handle a lot of requests over time. In exchange for long upfront compile times, you often get the best runtime performance. For instance, after automatic tuning, your matmuls might run with an optimal block size for your specific GPU and tensor shapes. This gives this mode an edge over default heuristics.

max-autotune-no-cudagraphs As the name suggests, this mode is the same as max-autotune but with CUDA Graph capture disabled. This is important for cases in which CUDA Graphs interfere with desired runtime behaviors that are not compatible with CUDA Graphs. For instance, since CUDA Graphs require static shapes and memory addresses, you can’t use varying input shapes or rely on allocating new memory for each iteration.

Also, when measuring performance, using CUDA Graphs can mask the overhead of launching kernels, which might not be desired in some benchmarks. So this mode allows maximal kernel optimization without CUDA Graphs. This will help maintain flexibility and allow you to debug any issues that CUDA Graphs might introduce (or mask), such as shape-dependent control flow or occasional allocator readdressing during graph capture. Use this mode when input sizes vary each iteration—or for debugging issues that might be masked by the use of CUDA Graphs.

For most use cases, the default mode is a good start. It’s meant to give sizable speedups with minimal hassle. If you find that your model still isn’t fast enough and you can tolerate longer compilation, try reduce-overhead and max-autotune for potentially better fused kernels—especially if your model is dominated by large matmul operations that can be autotuned. max-autotune can sometimes regress latency on some models. Be sure to profile the different modes for your specific workload and hardware.

On the other hand, if you are optimizing a very small model or a portion of code where Python overhead is the bottleneck, such as a tight training loop with lots of small tensor operations, using reduce-overhead can produce the best gains by removing virtually all kernel-launch runtime overhead using CUDA Graph capture. Just be mindful of the constraints of reduce-overhead. It works best when the workload per iteration is consistent and fits the graph-capture requirements, including no dynamic shape changes, no new memory allocations, etc.

The max-autotune-no-cudagraphs mode is more of a specialized option. It’s useful if you want maximum kernel optimization but either cannot use graphs due to varying input sizes or want to measure raw kernel performance without graph amortization.

In all cases, it’s wise to profile and measure after changing the PyTorch compiler mode. The different modes exist because one size doesn’t fit all in performance engineering. Furthermore, you should monitor memory usage when choosing a mode. Modes that use CUDA Graphs will allocate large, static buffers that increase memory footprint.

For extremely memory-constrained cases, you might prefer this no-graphs mode to avoid the extra memory overhead of CUDA Graphs. Next, let’s discuss how to inspect what the compiler is doing, including whether it created one graph or many—or whether it used CUDA Graphs, etc.

### Regional Compilation

For models with many identical blocks, such as Transformers and MoEs, you can use regional compilation to decrease cold-start execution time. Additionally, it’s useful to reduce recompilation churn—all without sacrificing the power of kernel fusion.

Specifically, regional compilation reduces compile time by compiling the repeated block (e.g., a Transformer block) once and reusing that code across all occurrences.

PyTorch supports regional compilation with torch.compiler.nested_compile_region(). This API marks a block as a nested compile region. This region is compiled the first time and then reused for subsequent runs.

In addition, regional compilation preserves correctness. If the compiler detects new input conditions (shape, dtype, device, stride, globals), it will transparently recompile the region.

Regional compilation benefits inference engines and short jobs in which startup latency matters—or when graph invalidations occur frequently. The performance of code compiled regionally is similar to the throughput of code compiled in full.

### Profiling and Debugging Compiler Performance Issues

When using torch.compile, it’s useful to know how to debug cases in which the compiler is unable to optimize part of your model—for instance, if certain operations are not being fused and you suspect a “graph break” is causing fallback to eager execution. PyTorch provides tools to inspect these situations.

> Modern PyTorch versions implement partial support for dynamic shapes using shape guards. These can eliminate some unnecessary graph breaks. However, truly dynamic workloads may still require falling back to eager execution (or use max-autotune-no-cudagraphs) to ensure correctness.

torch.\_dynamo.explain(model) prints a report of any graph breaks (e.g., parts of the model not captured by TorchDynamo), the reason the graph break occurred, and which parts of the model were not captured by TorchDynamo. It will also list the operations or data-dependent control flows that were not captured by TorchDynamo and needed to be executed in the slower Python eager mode.

A common cause of graph breaks is unsupported operations in the model. The Dynamo explain() output will make suggestions on how to get more details and help diagnose the issue. Taking advantage of these hints can help pinpoint the exact operation or control flow that caused the break.

Another useful technique is to set the environment variable TORCH_LOGS="+dynamo" or TORCH_LOGS="+dynamo,+inductor" before running your script. The + prefix enables verbose (DEBUG-level) logging for components like TorchDynamo and TorchInductor in the torch.compile pipeline. The verbose logs include details about graph breaks, fallbacks to eager mode, and compilation phases. If a model is unexpectedly slow with torch.compile, these logs can help identify when and where the execution is exiting the compiled graph.

If the model has truly dynamic shapes or dynamic control flow that can’t be resolved at compile time, you might need to guide the compiler. For example, you can break the model into sections that are compilable and leave the truly dynamic parts to run in Python.

To profile and benchmark the kernels generated by TorchInductor, you can specify the TORCHINDUCTOR_UNIQUE_KERNEL_NAMES=1 and TORCHINDUCTOR_BENCHMARK_KERNEL=1 environment variables. When these are set, Inductor will generate benchmark harness code for the generated kernel modules. The logs generated by this harness code can help pinpoint unexpected graph breaks and performance issues.

You can also mark part of the code as torch.\_dynamo.mark_dynamic(tensor, dim) to let the compiler know to expect dynamic shapes. This can eliminate unnecessary graph breaks due to shape mismatches. We’ll cover these techniques in more detail in the next chapter’s deep dive into the PyTorch compiler.

In short, when torch.compile doesn’t produce the expected speedup, you can use torch.\_dynamo.explain()—along with compiler logging—to identify which operations or code regions caused the fallback. From there, you will need to apply a workaround such as replacing an operation, reshaping a tensor differently, accepting less dynamic behavior, or simply disabling compilation for that specific part of the model. The result is that you keep the performance benefits for the majority of the model while still handling edge cases.

## PyTorch Optimized Attention Mechanisms

Transformer models spend significant time in their attention mechanisms. You can apply several PyTorch attention-optimization techniques to make sure it’s not a bottleneck. Here is a quick summary of a few of these techniques and when to use them:

_Scaled dot-product attention (SDPA)_ PyTorch’s high-level API torch.nn.functional.scaled_dot_product_attention, or SDPA, automatically uses the fastest available attention kernel for the given hardware (e.g., FlashAttention). Use this for a no-hassle speedup when your model’s attention pattern and dtype are supported by the selected backend (Flash, memory-efficient, or math). If it’s not supported, it will fall back to the standard attention implementation.

_FlexAttention_ A compiler-based approach for custom sparsity patterns in attention. FlexAttention can be substantially faster for specific sparse attention patterns (e.g., block-sparse or sliding-window attention) by generating optimized kernels for these patterns, as shown in Figure 13-3. Use FlexAttention for special cases that scaled_dot_product_attention does not support.

![Figure 13-3. FlexAttention provides support for custom attention variants.](AI%20Systems%20Performance%20Engineering-ch13_images/figure-13-3.png)

_FlexDecoding_ This is a counterpart to FlexAttention that optimizes the decoding or text generation phases. FlexDecoding integrates with torch.compile and dynamic cache layouts. It uses compile-time optimizations for the decoder side of sequence generation, including KV caching efficiently across time steps. FlexDecoding can speed up autoregressive generation by reducing redundant compute during decoding. FlexDecoding is intended for LLM inference workloads, including those with long-generation sequences. It does not change training-time attention semantics.

_Context parallel_ Context Parallel shards attention along the sequence-length dimension across participating devices or ranks to scale context length. Use the context_parallel() API to scope replacement of scaled_dot_product_attention with context-parallel-aware kernels. The mechanism splits query-key-value (QKV) by sequence across ranks and synchronizes during attention, rather than parallelizing attention across threads within a single GPU.

## PyTorch Architecture Optimization (torchao), Quantization, Sparsity, and Pruning

PyTorch Architecture Optimization (torchao) brings together quantization, sparsity, pruning, and related numerical-debugging tools into a single namespace. Its quantization subpackage (torchao.quantization) provides common FX-graph-mode workflows, including post-training quantization (PTQ), quantization-aware training (QAT), and QConfigMapping APIs to convert and optimize models for INT8, FP8, and emerging formats.

Beyond quantization, torchao supports pruning (torchao.pruning) and sparsity techniques like 2:4 and block sparsity (torchao.sparsity). These provide significant speedups with minimal loss in accuracy.

torch.compile() integrates with the torchao framework for quantization. Under the hood, TorchDynamo captures each submodule’s computation as an optimized graph, then TorchInductor emits hardware-aware kernels that leverage torchao. This produces consistent, end-to-end performance improvements for both model training and inference. Meanwhile, it preserves precise control over numerical formats and memory layouts. This makes it a great library for production-level performance optimizations such as quantization.

## Concurrency with CUDA Streams

As covered in an earlier chapter, CUDA streams enable concurrency and overlap of operations on the GPU. By default, PyTorch schedules all operations on the device’s default stream, stream 0, sequentially. However, many tasks are independent, and, if resources permit, a GPU can perform them in parallel using multiple streams. For example, GPUs can overlap data transfers with compute—or run separate neural network branches concurrently—by using separate, nondefault streams.

> Remember that modern GPUs have multiple DMA copy engines. Using separate streams for H2D copies can achieve truly parallel data transfers without blocking compute. This hardware support makes stream concurrency even more effective.

In PyTorch, you create a stream with torch.cuda.Stream(). You can then launch work on this stream using the Python context manager, with torch.cuda.stream(stream), or by explicitly assigning operations to that stream. PyTorch will issue operations (e.g., memory transfers, CUDA kernels, etc.) into the specified stream in a FIFO order—just as it does on the default stream.

### Overlapping Communication and Computation

A common use of CUDA streams is to overlap host-to-device (H2D) data loading with GPU computation. This helps mask the data transfer latency incurred when using an external device such as a GPU—relative to the CPU running on the host.

For instance, one stream can copy the next batch of input data from CPU to GPU memory while the default stream is busy training on the current batch. By the time the default stream is ready for the next batch, the data transfer is already done, and the GPU can process this next batch. This effectively hides the I/O latency. Here is an example of using two streams, the compute_stream (default) and the transfer_stream (nondefault), to overlap data transfer and compute in PyTorch:
