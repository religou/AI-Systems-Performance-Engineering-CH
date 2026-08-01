*Inter-kernel pipelining* Uses PDL, which lets the next GEMM kernel’s processing begin (using the cudaTriggerProgrammaticLaunchCompletion() → cudaGridDependencySynchronize() mechanism) before the primary kernel is completely torn down. This masks kernel-launch overhead.

*Interblock cooperation* By annotating kernels with __cluster_dims__, this code creates thread block pairs that are coscheduled on nearby SMs. Along with calling cluster.sync() (the mbarrier-based cluster barrier), these thread blocks share TMA-multicast data and dynamically load balance work across the grid.

In short, by layering warp-level pipelining with TMA, cluster-level synchronization and multicast, and inter-kernel overlap using PDL, you form a highly overlapping pipeline that hides DRAM latency, masks kernel-launch overheads, and maximizes Tensor Core utilization. The result is a high-performance GEMM in every stage: within warp copies, cross-thread-block barriers, and during kernel handoffs. These operate together to keep the hardware busy and avoid stalls.

## Key Takeaways

This chapter covered some advanced topics related to CUDA streams, stream-ordered memory allocators, event-based synchronization, inter-kernel pipelining, thread block clusters, and PDLs. These help to create highly efficient pipelines for inference and training workloads. The following are some key takeaways:

*Explicit versus default streams* Avoid the legacy default stream (stream 0), which serializes all work and acts as a global barrier. Instead, create explicit non-blocking streams (cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)) so kernels and copies can run concurrently without hidden synchronizations.

*Stream-ordered memory allocator* Use cudaMallocAsync and cudaFreeAsync to allocate/free device memory within a specific stream. This nonblocking allocator records requests in the stream’s queue, avoids global device synchronizations, and enables allocation overlap with in-flight kernels and copies.

*Overlapping H2D, compute, and D2H* By enqueuing asynchronous host-to-device (cudaMemcpyAsync), kernel launches, and device-to-host copies on different streams, you can achieve three-way overlap. While one stream runs a kernel, another can copy the next batch H2D, and a third can copy results D2H. This hides latency and reduces idle periods.

*CUDA events for fine-grained synchronization* Use cudaEventRecord and cudaStreamWaitEvent to coordinate producer-consumer dependencies across streams without stalling the entire GPU or CPU. Events enable a consumer stream to wait precisely until a producer stream finishes a copy or kernel, preserving maximum concurrency.

*Inter-kernel pipelining* Combine warp-specialized (multirole), two-stage (double-buffered) pipeline within a kernel with multistream launches. Launching multiple instances of a warp-specialized kernel (loader → compute → storer) in separate streams feeds successive mini-batches into the GPU. This combines intra-kernel memory/compute overlap with inter-kernel concurrency.

*Thread block clusters with streams* Extending intra-kernel warp-specialized pipelines to a grid-wide thread block cluster (cooperative launch) allows loader/compute/storer warps across blocks. Launching these cooperative kernels in multiple streams lets host-side allocations and copies for subsequent batches occur while a cooperative kernel is executing.

*In-kernel signaling and overlap with PDL* Kernel A calls cudaTriggerProgrammaticLaunchCompletion() once its data writes are flushed, and Kernel B uses cudaGridDependencySynchronize() to wait on that signal. This allows Kernel B’s prologue to begin and overlap with A’s epilogue—without CPU intervention.

*Host-side PDL setup* The host configures Kernel B’s launch via cudaLaunchKernelExC() with a cudaLaunchConfig_t that sets cudaLaunchAttributeProgrammaticStreamSerialization. This allows the driver to enqueue B early and maximize inter-kernel overlap. PDL uses cudaTriggerProgrammaticLaunchCompletion in the primary kernel, cudaGridDependencySynchronize in the dependent kernel, and the cudaLaunchAttributeProgrammaticStreamSerialization launch attribute on the host.

## Conclusion

In conclusion, inter-kernel concurrency with CUDA streams has evolved from a manual optimization to an automatic feature used by modern AI frameworks and GPU runtimes. By understanding the core principles such as streams, resource occupancy, and synchronization points, developers can maximize GPU utilization.

Inter-kernel concurrency is critical for maximizing GPU utilization in modern workloads by overlapping kernel execution and data transfers both within a single GPU and across multiple GPUs. As hardware continues to add more parallelism—and software abstracts more of the scheduling complexity—understanding how to maximize concurrency is a critical part of getting the most from your high-performance AI system hardware.

This chapter demonstrated how to orchestrate kernels, memory operations, and allocations across multiple CUDA streams to keep all GPU hardware units actively running. These hardware units include compute pipelines, DMA engines, and interconnects. CUDA streams serve as the foundational mechanism to enqueue kernels, memory operations, and allocations in independent queues, allowing the GPU’s compute engines and DMA engines to run simultaneously.

By avoiding the default stream’s hidden barriers, leveraging the stream-ordered allocator, employing events for precise synchronization, and combining intra-kernel warp specialization with inter-kernel multistream pipelines (extending to thread block clusters from Chapter 10), you can achieve near-peak utilization even for complex LLM workloads.

In multi-GPU contexts, overlapping peer-to-peer transfers, collective communications, and computations across distinct streams further minimize idle time. We looked back at Chapter 10 by showing how to capture these stream workflows using CUDA Graphs to reduce CPU overhead for repeated iterations.

In the next chapter, we will dive even deeper and build upon these principles by introducing dynamic kernel orchestration and meta-scheduling with dynamic parallelism and CUDA Graphs. We’ll coordinate entire pipelines of kernels and data movements—at runtime—to adapt to changing workloads. This dynamic resource balancing and device-side orchestration will push us to the next level of performance optimizations for large-scale AI systems.