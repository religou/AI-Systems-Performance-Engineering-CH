# Chapter 13. Profiling, Tuning, and Scaling PyTorch

AI training and inference pipelines can suffer from performance bottlenecks at every layer, including Python interpreter overhead, CPU host-side data-loading stalls, CUDA kernel underutilization, and GPU device-memory contention. To optimize effectively, you need to profile at multiple levels of the stack using multiple tools that cover the entire system.

This chapter focuses on profiling, debugging, and system-level tuning of PyTorch workloads running on modern NVIDIA GPUs. We will explore how to identify and fix bottlenecks using PyTorch’s built-in profiler, NVIDIA’s Nsight tools, and CPU profiling with Linux perf—as well as PyTorch memory profiling and memory allocator tuning. We’ll also discuss how PyTorch uses CUDA streams for concurrency and CUDA Graphs to reduce kernel launch overhead.

Next, we’ll show how to optimize data pipelines and scale out to multiple GPUs with PyTorch Distributed Data Parallel (DDP), Fully Sharded Data Parallel (FSDP), and other model parallelism strategies. We’ll then demonstrate how to profile multi-GPU and multinode environments, including Holistic Trace Analysis (HTA) and Perfetto.

Throughout the chapter, we emphasize performance trade-offs and quantitative examples that focus on kernel execution times, hardware utilization metrics, memory footprint, data loading efficiency, and overall cost-efficiency of scaling. By the end of this chapter, you should have an understanding of how to implement an effective, holistic approach to profiling and tuning PyTorch workloads across the entire stack.

## NVTX Markers and Profiling Tools

To capture a holistic view of performance, it’s important to profile at multiple levels and use tools that cover the entire system. There exists a set of common tools and best practices used by practitioners and performance engineers to perform holistic profiling across all the layers in the system stack.

Before we get to the tools, it’s important to highlight the NVIDIA Tools Extension (NVTX) and NVTX markers. These markers denote time ranges in a profiler’s timeline view and allow different profilers to correlate events across the same phases.

For example, an NVTX range for "forward" will appear in both PyTorch profiler traces and Nsight Systems’ timelines. This makes cross-tool analysis much easier at different layers of the stack. NVTX markers are supported by most modern AI frameworks and libraries, including PyTorch and anything related to the CUDA ecosystem.

NVTX markers are injected into code using either CUDA C++, PyTorch, or any C++ or Python library that supports NVIDIA GPUs (e.g., OpenAI Triton, PyCUDA, CuPy, cuTile, cuTe, CUTLASS, etc.). Most libraries already inject NVTX markers on your behalf for critical regions of code, such as "train_step", "forward", "backward", "optimizer_step", etc. But you can also inject them yourself using torch.profiler.record_function() and torch.cuda.nvtx.range_push() in PyTorch, for instance.

Now that we’ve described how to annotate interesting sections of your code using NVTX markers, let’s discuss the tools that can ingest, align, and visualize these markers. Common profiling tools are summarized in Table 13-1 along with their scope, key features, and typical use cases. This table can help you choose the right tool for each stage of your optimization journey.

Table 13-1. Summary of profiling and visualization tools

| Tool | Scope | Features | Typical use case |
| --- | --- | --- | --- |
| PyTorch profiler (Kineto) | In-PyTorch op-level profiling (CPU/GPU) | NVTX marker support, shape recording, memory stats, trace export, identification of compile graph breaks | Fine-grained breakdown of model code; identify slow ops, GPU kernel launch overhead, or imbalance between forward/backward times. |
| Nsight Systems (nsys) | System-wide timeline (CPU, GPU, OS, I/O) | Unified timeline of CPU threads and GPU streams, NVTX integration, multiprocess support | End-to-end view of training/inference pipeline; detect data loader stalls, CPU-GPU overlap issues, or inter-GPU synchronization delays. |
| Nsight Compute (ncu) | GPU kernel analysis (per kernel) | Per-kernel hardware metrics, source correlation, roofline analysis, occupancy, and throughput reports | Deep dive into kernel efficiency after identifying hot kernels; determine if a kernel is memory bound or compute bound, and why. |

Here is a detailed description of each profiling tool in Table 13-1:

*PyTorch profiler (Kineto)* Within PyTorch, the torch.profiler, based on the Kineto open source project, provides operator-level breakdowns of CPU and CUDA/GPU runtimes. In addition, it can record input shapes and take memory snapshots using simple Python context managers. The PyTorch profiler can capture detailed timeline traces and hardware counters across training and inference workloads using NVTX ranges to align the events. It provides end-to-end observability from Python code down to the CUDA kernels—and even provides performance tips for common issues like data-loading stalls and inefficient CUDA code.

*Nsight Systems (*nsys*)* For system-wide correlation, including CPU threads, GPU kernels, OS events, I/O, and interconnect traffic, NVIDIA Nsight Systems produces a unified timeline view. Its GUI and CLI reports can merge NVTX zones, Python call stacks, and CUDA streams across multiprocess and multinode runs. This makes it easy to spot where I/O and synchronization stalls might be impacting compute performance.

*Nsight Compute (*ncu*)* Complementing Nsight Systems is NVIDIA Nsight Compute for per-kernel analysis. Nsight Compute collects detailed hardware metrics such as occupancy, memory bandwidth, and SM utilization. It can even generate roofline charts mapped to source code. Nsight Compute helps answer why a particular kernel is slow (e.g., memory bound, low occupancy) after other higher-level tools identify which kernels are the hotspots.

*PyTorch memory profiler* PyTorch also includes a memory profiler, which you can enable with profile_memory=True in torch.profiler. The PyTorch memory profiler breaks down peak and cumulative GPU memory allocations per operation. This reveals memory usage hotspots that might otherwise go unnoticed.

*Linux* perf On the host side, Linux’s perf tool can sample CPU hardware counters, including cycles, instructions, and cache misses—and unwind full C/C++ and Python call graphs. Starting with perf sched, you can see when CPU threads sit idle due to I/O or thread scheduling/synchronizing. This uncovers bottlenecks in data preprocessing loops, Python’s GIL, or synchronization that can starve the GPU.

*Holistic Trace Analysis* Meta’s open source Holistic Trace Analysis (HTA) tool ingests PyTorch profiler traces to help diagnose multi-GPU bottlenecks. With HTA, one can visualize distributed training timelines with NVTX ranges alongside CUDA kernel traces. By drilling into memory allocation patterns over time, you can identify periods of idle GPU—including when GPUs are waiting on each other.

> TensorBoard’s PyTorch trace visualization plugin is deprecated. Instead, use Perfetto for timeline viewing and Meta’s HTA for distributed trace analysis.

*Chrome trace and Perfetto viewer* For web-based exploration of large PyTorch profiler trace files, you can use Chrome tracing (e.g., chrome://tracing in the browser) and Perfetto UIs. These will load the JSON traces and let you interactively explore timeline views and flame charts. They even let you perform fine-grained filtering and SQL queries on the trace data—down to the submillisecond level for event tracing and correlation. Chrome traces and the Perfetto UI are ideal for sharing profile results between members of your organization for cross-team analysis. (Note: Chrome’s legacy trace viewer is deprecated, so you should prefer the Perfetto web UI and SQL engine for viewing and analyzing traces.)

*TorchEval (PyTorch’s metrics library)* Another project, TorchEval, lets you log and monitor model throughput, latency, and quality metrics alongside training and evaluation metrics—all within a unified interface. TorchEval is PyTorch’s official metrics library and provides a simple API for end-to-end performance and quality metrics. This library makes it easy to plug into training loops and integrate across distributed environments.

*ExecuTorch* For embedded, mobile, and edge devices, the ExecuTorch project allows profiling, visualizing, and debugging PyTorch models in lightweight runtime environments like Meta glasses. ExecuTorch has a small, dynamic memory footprint and supports Linux, iOS, Android, and embedded systems. Hugging Face supports ExecuTorch through its Optimum ExecuTorch project, which makes this environment easy to integrate if you’re already using the Hugging Face ecosystem, like Hugging Face Transformers.

Next, let’s dive into an example of using these profilers to identify performance bottlenecks. We’ll then apply targeted optimizations and verify the performance improvements.

## Profiling PyTorch to Identify Bottlenecks

Let’s profile an example mixture-of-experts (MoE) transformer model to see these tools in action. MoEs are LLMs with multiple expert layers—each expert is a feed-forward network. Routing tokens to experts is managed by an expert gating system. We will run a single training iteration, capture detailed performance traces, and analyze the output to guide our optimizations.

### Using PyTorch Profiler

First, we set up the model and input. We use Hugging Face Transformers to load the model and tokenizer, move the model to GPU, and prepare a small batch of inputs, as shown here:

```
# train.py
# Set up model and data
model_name = "..."
tokenizer = AutoTokenizer.from_pretrained(model_name)
device = torch.device("cuda")
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
batch_size = 4
input_texts = ["MoEs are great."] * batch_size
enc = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)
input_ids = enc.input_ids.to(device)
attention_mask = enc.attention_mask.to(device)
labels = input_ids.clone()  # For LM training, labels are the inputs
                            # (next-token prediction)
```

To avoid capturing one-time setup costs, we run a few warm-up iterations before profiling. This will prepare the model for analyzing and benchmarking by compiling JIT kernels, filling caches, etc. This way, our measured iteration is representative of steady-state performance. Here is the code to prepare the model:

```
# Warm-up (not profiled)
for _ in range(5):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

```

Now we profile one training iteration using PyTorch’s profiler and NVTX. We wrap the iteration in torch.profiler.profile() and mark high-level regions with record_function and NVTX ranges, including "forward", "backward", and "optimizer_step", as shown here:

```
from torch import profiler
with profiler.profile(
    activities=[profiler.ProfilerActivity.CPU,
                profiler.ProfilerActivity.CUDA],
    record_shapes=True,  # record tensor shapes
    profile_memory=True, # track GPU memory usage per op
    with_stack=True,     # enable stack tracing
    with_flops=True      # capture FLOPs counters
) as prof:
    with profiler.record_function("train_step"):
        # Forward pass
        torch.cuda.nvtx.range_push("forward")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask=attention_mask,
                            labels=labels)
        loss = outputs.loss
        # end of forward
        torch.cuda.nvtx.range_pop()
        # Backward pass and optimization
        torch.cuda.nvtx.range_push("backward")
        loss.backward()
        torch.cuda.nvtx.range_push("optimizer_step")
        optimizer.step()

        # end of optimizer_step
        torch.cuda.nvtx.range_pop()
        optimizer.zero_grad()

        # end of backward
        torch.cuda.nvtx.range_pop()
```

In this code, the PyTorch profiler is recording all CPU and GPU activities during the train_step. We use record_function("train_step") to define a top-level region. We also insert NVTX markers for the subphases ("forward", "backward", "optimizer_step"). These markers will appear in profiler timelines to delineate phases of the iteration.

The profiler can also highlight compiled versus noncompiled regions of the model. We’ll cover the PyTorch Compiler, graph breaks, and mechanisms to mitigate graph breaks later in this chapter—as well as the next chapter.

For instance, when using torch.compile, the trace will show events like Compiled Function and indicate any graph breaks (see Figure 13-1). This helps pinpoint where the model fell back to eager execution, which will guide further optimizations.

![Figure 13-1. Compiled (left and middle, pink) versus noncompiled (right, green) regions (source: https://oreil.ly/Z_fJG)](AI%20Systems%20Performance%20Engineering-ch13_images/figure-13-1.png)

After execution, we can examine the operation-level results by calling prof.key_averages().table() to print a concise table of the top operators by runtime. In the next code block, we request the top 10 operations sorted by their self CUDA time, which is the time spent in each operation’s own CUDA kernels excluding child operations spawned by the kernel. The top 10 operations by CUDA execution time are summarized in Table 13-2:

```
with profiler.profile(
    activities=[ProfilerActivity.CPU,
                ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
) as prof:
    train_step(...)
...
print(
    prof.key_averages()
        .table(
            sort_by="self_cuda_time_total",
            row_limit=10,
            fields=["self_cuda_time_total",
                    "calls"]
        )
)
```

Table 13-2. Profiler’s top 10 operations by CUDA execution time over one training iteration

| Operation | Self CUDA total | Calls |
| --- | --- | --- |
| aten::matmul | 43.00 ms | 128 |
| aten::linear | 35.50 ms | 64 |
| dispatch | 18.70 ms | 2 |
| combine | 12.10 ms | 2 |
| aten::layer_norm | 10.20 ms | 16 |
| aten::softmax | 5.70 ms | 4 |
| aten::scatter | 4.10 ms | 16 |
| aten::gather | 3.60 ms | 16 |
| aten::to | 2.90 ms | 8 |
| aten::add_ | 2.20 ms | 64 |

Here, we see that the matrix multiplication operations (aten::matmul, and its use in aten::linear) dominate the CUDA time and consume the majority of the iteration. These operations correspond to the expert feed-forward network (FFN) GEMMs. Specifically, there are 128 calls to matmul per iteration. This makes sense since we have 64 experts—and each expert does a matmul in both the forward and backward passes.

In Table 13-2, we see the next largest costs are from the dispatch and combine operations. These are custom C++/CUDA kernels that redistribute tokens to experts—and then gather the outputs of the expert. The dispatch operation runs twice—once in the forward pass and once in the backward pass—for a total of 18.7 ms. The combine ran twice for 12.1 ms total. Together, these two operations account for another 30.8 ms of GPU time. The remaining time is spread across other smaller ops like layer norms, activations, etc.

The key takeaway from this profiling example is that the expert FFN matmul is the top bottleneck, followed by the dispatch and combine kernels. Together, these dominate a training iteration’s runtime. To improve performance further, we should target those operations either by optimizing them directly or by reducing the number of times they’re called.

### System Profiling with Nsight Systems and NVTX Timelines

The NVTX markers that we inserted make it straightforward to analyze the timeline with Nsight Systems. To aggregate metrics per phase, we can profile the code using nsys with an NVTX-based summary, as shown in the CLI command here:

```
nsys profile \
   --output=profile \
   --stats=true \
   -t cuda,nvtx \
   python train.py
```

Here, the -t cuda,nvtx option instructs Nsight Systems to trace both CUDA API calls and NVTX ranges. After profiling, we can open the profile.nsys-rep file (--output=profile) in the Nsight Systems GUI or use the CLI to print the NVTX summary to the terminal. We can then use the CLI to generate the NVTX GPU Projection Summary using one of the following commands on the profile.nsys-rep file, as shown here to validate ranges against projected GPU work with Nsight Systems:

```
nsys stats --report=nvtx_gpu_proj_sum \
  profile.nsys-rep
# or
nsys recipe nvtx_gpu_proj_sum \
  profile.nsys-rep
```

You can use one of these commands in your continuous build and integration pipelines to monitor and detect any performance regressions. The results of this CLI command are summarized in Table 13-3.

Table 13-3. NVTX GPU projection summary for one train_step iteration using Nsight Systems

| NVTX range | GPU time (ms) | Self GPU time (ms) | Child GPU time (ms) | Instances (calls) |
| --- | --- | --- | --- | --- |
| train_step | 138.0 | 0.0 | 138.0 | 1 |
| forward | 60.5 | 60.5 | 0.0 | 130 |
| backward | 58.3 | 58.3 | 0.0 | 130 |
| optimizer_step | 19.2 | 19.2 | 0.0 | 1 |

Here, we see the train_step range includes the forward, backward, and optimizer subranges. This NVTX GPU Projection Summary confirms that the total GPU time under train_step is 138 ms. This time matches the sum of the forward, backward, and optimizer_step times from the PyTorch profiler output in Table 13-2. This shows consistency between tools.

And although Table 13-3 shows a single optimizer_step call, its NVTX range actually brackets all 64 aten::add_ CUDA kernel launches (one add_ per expert as shown in Table 13-2) under the single optimizer_step marker.

Note that Nsight Systems groups the 64 aten::add_ calls (e.g., 64-way expert parallelism strategy) into a single optimizer_step marker because it uses the CUDA Profiling Tools Interface (CUPTI) to capture NVTX push/pop events on the host. It then “projects” asynchronous GPU kernel execution times onto these CPU-defined intervals. As such, it sums the durations of every kernel with GPU start/end timestamps that fall between the corresponding push and pop calls. This produces one cumulative GPU time that exactly matches the total of the individual aten::add_ kernels.

> Because NVTX markers have very low overhead when no profiler is attached, this projection mechanism is ideal because it adds negligible overhead while still providing end-to-end correlation of GPU work with high-level code regions.

The forward and backward ranges each have self GPU time equal to their total time since we didn’t choose to nest deeper ranges inside of them. As such, child GPU time is 0 ms. However, train_step has nearly all of its time as child GPU time since it’s just a wrapper around the nested phases.

The NVTX GPU projection summary also shows that, in each iteration, we observed 130 GPU activities inside train_step. These include kernel launches and other device operations such as memory copies, so they are not strictly one-to-one with kernels.

As you can see in Table 13-3, the 130 GPU kernel calls happen for both the backward and forward passes as well. This one-to-one correspondence between operations and NVTX instances means that our instrumentation captures every important operation.

> The NVTX summary we show in Table 13-3 is a convenient text overview. For visual analysis, the timeline GUI can show overlapping kernel execution, CPU thread states, and even CUDA API overhead events. In practice, you want to verify that host-side data loading and preprocessing are overlapping with GPU compute in the visual timeline. Any large gaps, or “bubbles,” indicate a problem. Small gaps for synchronization are expected and appropriate.

On a multi-GPU run, the Nsight Systems or HTA timeline view can reveal if NVLink or InfiniBand/Ethernet is being utilized effectively—or if a node is starved for work while waiting for communication or network delays. This would hint at suboptimal synchronization or load imbalance.

It’s important to trace GPU communication events, including NCCL all-reduce calls and NVLink/NVSwitch activity using Nsight Systems and HTA to provide traces. These help verify that the GPUs stay busy in massive GPU domains such as an NVL72-based system.

Careful profiling makes sure the system is using proper inter-GPU synchronization and balancing the workload in these large NVLink clusters. Let’s now zoom in on one of the most expensive kernels in the system: the matrix multiply, or matmul.

### Kernel Roofline Analysis for General Matrix Multiply (GEMM)

To dive deeper and analyze the expert matmul, we invoke the CLI profiler Nsight Compute (ncu) to target specific GEMM kernels by name. We’ll collect roofline-related metrics to determine if it’s compute bound or memory bound, as shown here:

```
ncu \
  --target-processes all \
  --kernel-name-regex "matmul" \
  --metrics \
    gpu__time_duration.avg, \
    gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed, \
    lts__throughput.avg.pct_of_peak_sustained_elapsed, \
    sm__sass_thread_inst_executed_op_fp32_pred_on.sum, \
    sm__warps_active.avg.pct_of_peak_sustained_active \
  --csv full \
  -o matmul_roofline_report \
  python train.py
```

Here, we are collecting hardware counters for any kernel whose name matches “matmul”. Specifically, we collect a few key metrics, including GPU DRAM bandwidth and L2 bandwidth as a percent of peak (gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed, lts__throughput.avg.pct_of_peak_sustained_elapsed), FP32 instruction count as a compute proxy (sm__sass_thread_inst_executed_op_fp32_pred_on.sum), kernel time (gpu__time_duration.avg), and achieved occupancy (sm__warps_active.avg.pct_of_peak_sustained_active).

> Metric identifiers can vary slightly across Nsight Compute releases. If a metric is not found, use the UI to locate the current name and substitute accordingly. Here, we are profiling SM and DRAM % of peak sustained and achieved occupancy.

After applying the optimizations discussed until now, like reducing precision, fusing kernels, and increasing arithmetic intensity, we can rerun the ncu command to verify that our optimizations made a positive impact. Table 13-4 shows the comparison before and after applying these optimizations to improve arithmetic intensity.

Table 13-4. Roofline analysis for the expert matmul kernel before and after arithmetic intensity optimizations

| Kernel configuration | % of peak FLOPS (SM compute throughput) | % of peak memory BW (memory throughput) | Achieved SM occupancy | Characteristic |
| --- | --- | --- | --- | --- |
| Baseline | 50% | 70% | 60% | Memory bound (stalling on memory transfers) |
| Optimized | 85% | 40% | 80% | Compute bound (near hardware roofline) |

Here we see, in the baseline run, the primary GEMM kernel achieves only about 50% of peak compute FLOPS, 70% of peak memory bandwidth, and a moderate 60% SM occupancy (average active warps per SM). This occupancy is not enough to fully hide memory latency.

> There is no universal occupancy target. Many high-performance kernels achieve full latency hiding at 25–50% achieved occupancy. Use Nsight Compute’s occupancy metrics together with stall-reason breakdowns and eligible warps per active cycle to judge whether more occupancy would reduce stalls for the kernel. If the kernel can’t schedule enough warps to cover the stall periods, this will lead to idle cycles since memory requests aren’t being serviced quickly enough.

The baseline metrics indicate a kernel that is memory bound since its execution is stalled by memory transfers. The result is a substantial amount of unused compute capacity, which further reinforces that this workload is not currently compute bound. The goal is to make this kernel more compute bound to take advantage of the large number of FLOPS available with this GPU.

In the optimized version (e.g., fusing kernels, increasing arithmetic intensity, and reducing memory movement), the peak FLOPS increases to 85%, peak memory bandwidth drops to 40%, and occupancy increases to 80%. We effectively shifted the kernel from memory bound to compute bound—much closer to the hardware’s roofline limits, as shown in Figure 13-2.

![Figure 13-2. Roofline chart before and after increasing arithmetic intensity of this kernel](AI%20Systems%20Performance%20Engineering-ch13_images/figure-13-2.png)

Up to this point, our profiling has focused on GPU performance. It’s also important to not waste time on the CPU or performing I/O. In the next section, we continue our profiling journey on the host side.

### CPU and GPU Profiling with Linux perf

To get a more holistic view of where time is being spent across both the host and device, we can use Linux perf to analyze CPU cycles, cache misses, branch misses, etc. We can then use these insights to drive a series of optimizations, apply them one by one, and measure the improvements.

First, let’s run a lightweight perf stat to gather CPU-side statistics during the MoE training run on a node with an ARM-based Grace CPU paired with a Blackwell GPU. Here is the CLI command followed by example output:

```
perf stat -e \
  cycles,instructions,cache-misses,branch-misses \
  python train.py
  Performance counter stats for 'python train.py':
# 0.600 CPUs utilized
1,200.345 msec task-clock
# Approximately 2.0 GHz
2,400,567,890      cycles
# 1.58 insn per cycle
3,800,123,456      instructions
# 0.32% of all cache refs
12,345,678      cache-misses
# 0.12% of all branches
4,567,890      branch-misses
 1.234567890 seconds time elapsed
```

This report from perf stat shows the CPU utilization, cycles, instructions per cycle (IPC), and cache/branch misses. In our run, the task clock shows ~1.2 seconds with only ~60% (0.600) of a single CPU core used over the measured interval. This is expected, as the GPU is doing most of the heavy lifting. The low cache-miss and branch-miss rates hint that memory access patterns and branch prediction were relatively efficient on the CPU side for this workload.

However, the instructions per cycle (IPC) measurement of just 1.58 shows that the CPU is issuing well below the eight instructions per cycle theoretical maximum IPC for a single Grace CPU core (ARM Neoverse V2). This indicates potential inefficiencies such as memory latency, I/O stalls, or host-compute issues in this specific workload.

We can explore further using perf record and perf report to pinpoint which Python and C++ functions dominate the CPU execution time during training. These CLI commands are shown here:

```
perf record -F 2000 -g --call-graph dwarf -o perf.data \
    python train.py
```

Here, we use perf record to collect samples at 2000 Hz (-F 2000) and capture full C/C++ and Python call stacks by specifying -g --call-graph dwarf. DWARF stands for Debugging With Attributed Record Formats, and it’s a standard debugging data format embedded in compiled binary files (e.g., ELF files). The DWARF output trace is saved to perf.data (-o perf.data). We then use perf report to generate a summary report of the hottest call stacks and their sample percentages:

```
perf report --stdio -n -g -i perf.data
# Samples  Command   Shared Object        Symbol
# ........ ........  ...................  .................................
     45.0%  python    python               py::forward <...> /src/train.py
     20.5%  python    python               aten::matmul
     10.2%  python    python               dataloader_iter_next
      8.7%  python    libnccl.so           ncclAllReduce
      5.3%  python    libc.so.6            read
      ...   ...       ...                  ...
```

Here, we see that the Python interpreter’s forward function, the Python side of our training loop, accounts for 45.0% of CPU samples. PyTorch’s C++ aten::matmul operation accounts for 20.5%, the DataLoader’s iterator next function for 10.2%, an NCCL all-reduce call for 8.7%, and I/O reads for 5.3%.

These percentages tell us where to invest our optimization effort. Based on this profile, we address each bottleneck with a specific mitigation plan to improve performance of our system:

*Excess Python overhead (45% in* py::forward*)* Use PyTorch’s JIT compiler using torch.compile (discussed in the next section) to eliminate interpreter overhead and fuse Python-side operations into optimized CUDA code.

*Large* matmul *hotspot (20.5% in* aten::matmul*)* Either use the PyTorch Compiler to optimize this code or move this critical matrix multiply into a custom CUDA C++ kernel (e.g., fused kernel) to bypass Python and use the optimized CUDA code directly.

*Data loading stalls (10.2% in* dataloader_iter_next*)* Increase PyTorch’s DataLoader num_workers. A common guideline is one worker per CPU, but you can experiment with more to find the right level of I/O parallelization. Just make sure you don’t oversubscribe the CPU cores. You should also enable persistent_workers=True so that worker processes stay alive across epochs and avoid startup overhead for each epoch. Fuse or parallelize multiple torch.utils.data.DataPipe. This can reduce Python overhead in complex data pipelines.

*Gradient synchronization overhead (8.7% in* ncclAllReduce*)* Optimize multi-GPU communication. For instance, you can increase the gradient bucket size in DistributedDataParallel. It’s common to increase the bucket_cap_mb from 25 MB to 50 MB so that NCCL can launch all-reduce operations sooner and overlap them with backward computations. You can also consider gradient compression techniques or NVIDIA’s NCCL compression for 8-bit gradients to reduce bandwidth usage. These may incur a slight cost to accuracy.

*Host I/O bottleneck (5.3% in* read syscalls*)* Use pinned memory (pin_memory=True) and nonblocking GPU copies (.to(device, non_blocking=True)) in the DataLoader to overlap CPU-to-GPU data transfers. Also, you can batch file reads or bundle many small files into an optimized dataset format like Arrow, WebDataset (tar shards), TFRecord, or Parquet chunks that facilitate large sequential reads. It’s best to prefer contiguous shard formats over per-sample files. Prefer pinned host buffers when using a larger prefetch depth (e.g., prefetch_factor=4 or 8). Combined with persistent_workers=True, your system will keep loader threads busy since compute-communication are overlapping efficiently. This will eliminate per-file overhead when reading many small files in large corpora.

These approaches, combined with large OS read-ahead and NVMe SSDs, will boost I/O throughput. Also, newer filesystems and storage libraries like NVIDIA Magnum IO can help pipeline data efficiently to the GPU.

After formulating this plan, you should systematically apply each optimization and measure the effect. Remember to implement and test these optimizations one by one to verify that each actually improves performance. This helps avoid situations in which multiple changes interact in unexpected ways. By isolating each change, you know which adjustments have a positive effect and which do not.

On systems with the NVIDIA Performance Monitoring Unit (PMU) drivers present, you can use perf to sample NVIDIA chip interconnect and fabric counters alongside CPU counters, including NVLink-C2C devices exposed under /sys/bus/event_source/devices as nvidia_nvlink_c2c*, for instance. Verify availability with perf list and by checking the nvidia_pmu entries under sysfs.