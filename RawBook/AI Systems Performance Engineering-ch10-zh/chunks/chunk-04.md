This “copy once, share through DSMEM” pattern shares the leader’s tiles using DSMEM and map_shared_rank following a cluster barrier. It does not perform a TMA multicast as we covered previously with tensor-map descriptors, cp.async.bulk.tensor, and shared::cluster. This is an important distinction to understand as, without the cluster.sync() in the DSMEM pattern, followers can read stale leader SMEM data. As such, cluster.sync() is required before map_shared_rank().

> The following should help you choose TMA multicast versus DSMEM sharing: If tile reuse per block is high (e.g., each block touches the same tile many times) or the cluster size (C) is large (8-16 CTAs), TMA multicast usually wins since this pattern writes once, then reads locally many times. If SMEM is tight (e.g., large tiles) or cluster size (C) is small (2-4 CTAs) and each follower touches the tile once, DSMEM sharing is usually the better option since it uses a smaller footprint.

Every block computes a distinct band of rows directly from the leader’s shared memory by using map_shared_rank and writes its results to global memory. This removes duplicate global loads across the cluster and keeps the overlap advantages of a pipeline at the points where it matters.

The leader block’s loader warp calls producer_commit() to publish its local stage to that block. Other blocks do not wait on the leader’s pipeline. Instead, they observe the leader’s tiles after the cluster wide barrier and then read from distributed shared memory.

This allows tile loading, computing, and storing to interleave across multiple thread blocks, which drives the SMs to even higher utilization. Table 10-5 compares the single-block warp-specialized kernel with the thread-block-clustered (multiblock) warp-specialized kernel presented in this section.

Table 10-5. Comparison of naive tiling, two-stage double buffering, warp-specialized, and thread block cluster pipeline kernels.

| Metric | Naive tiling | Two-stage, double-buffered (double_buffer_pipeline) | Warp-specialized (warp_specialized_pipeline) | Thread block cluster pipeline (warp_specialized_cluster_pipeline) |
| --- | --- | --- | --- | --- |
| Kernel execution time | 41.3 ms | 20.5 ms (+2.01× faster versus naive) | 18.4 ms (+10.2% speedup versus two-stage) | 17.2 ms (+6.5% speedup versus warp-specialized) |
| Warp execution efficiency | 68% | 92% (+24% versus naive) | 96% (+4% versus two-stage) | 97% (+1% versus warp-specialized) |
| Warp state stall % (for shared memory and barrier waits) | High | Low | Minimal | Minimal (further reduced) |
| L2 throughput | 80 GB/s | 155 GB/s (+94% versus naive) | 165 GB/s (+6.45% versus two-stage) | 170 GB/s (+3% versus warp-specialized) |
| Throughput scalability | Scales up to 2–3 warps/SM | Scales to ~6 warps/SM | Scales nearly linearly to SM warp limit (64 resident warps per SM on Blackwell) | Scales fully across thread blocks until they are constrained by SM warp limit of 64 resident warps per SM on Blackwell |
| DRAM read throughput versus kernel duration | Poor overlap | Great overlap | Excellent overlap | Excellent overlap (even under thread block handoff) |
| Instruction count | 1.7 B | 1.05 B (–38% versus naive) | ~1.00 B (–4.76% versus two-stage) | ~0.98 B (–2% versus warp-specialized) |

Here, we see that the thread block cluster implementation further improves on warp-specialized by distributing loader, compute, and storer roles across multiple thread blocks in one cooperative launch. This increases overall SM utilization and reduces both execution time and redundant memory traffic.

Specifically, the thread block cluster pipeline kernel implementation (warp_specialized_cluster_pipeline) squeezes out another ~1.2 ms (6.5%) over the single-thread-block, warp-specialized kernel. This is because the thread block cluster version interleaves tile loads, computes, and stores across all thread blocks.

SM utilization reaches ~97%, since any idle SM in one thread block can be kept busy by another thread block performing a different pipeline stage. L2 load throughput peaks around 170 GB/s, thanks to better shared-memory reuse and fewer redundant loads. As such, the cluster can avoid duplicate global reads. With DSMEM, the leader block loads a tile once and other blocks read it using map_shared_rank() after cluster.sync() is called. With TMA multicast, a single tensor copy from global can be broadcast directly into each block’s shared memory.

Instruction count drops to ~0.98 B because thread-block-wide pipelining further reduces wasted cycles on redundant global reads. Throughput scalability is also maintained since, as long as you have enough thread blocks and warps, the pipeline can keep them all saturated up to the warps-per-SM hardware limit of your GPU, or 64 warps per SM on Blackwell.

Note that both the single thread block and thread block cluster versions of warp specialization will eventually hit that same “warps-per-SM” ceiling, but they differ in how those warps are fed data and synchronized. This difference translates into a measurable performance advantage for the thread block cluster implementation, as we saw in Table 10-5.

Specifically, in the single thread block, warp-specialized kernel, each thread block allocates exactly three warps: one to load a tile-sized chunk of A and B into shared memory, one to compute on that chunk, and one to store the results back to global memory. Once those three warps complete their roles, the thread block moves on to its next tile, and the cycle repeats.

Even if every SM has three active warps for load, compute, and store, you are not saturating the warp limit of the SM. Modern GPUs allow up to 64 concurrent warps per SM, and actual residency is limited by registers, shared memory, and blocks per SM.

At similar total live warp counts, a single-block warp specialized design and a cooperative grid can show comparable occupancy, but the memory behavior differs. In a single-block design, each block fetches its own copy of any input tile that it touches. When multiple blocks reuse the same tile at about the same time, the device performs duplicate reads that consume L2 capacity and HBM bandwidth.

A thread block cluster changes that pattern. Blocks in the cluster can share data through distributed shared memory and can receive one multicast of a global tile into the shared memory of all blocks in the cluster. Use a block-scoped pipeline in the leader block together with cluster synchronization and map_shared_rank, or configure for multicast when broadcast from global memory is preferred. This removes duplicate loads within the cluster and reduces pressure on L2 and HBM.

A cuda::pipeline is scoped to the creating block. A producer_commit() in one block does not release consumer_wait() in other blocks. To coordinate across blocks in a cluster, first publish data (e.g., into the leader’s shared memory), then use cluster.sync(). Followers then access the leader’s shared memory with map_shared_rank(). Optional TMA multicast can broadcast from global memory directly into each block’s shared memory.

At this point, every block observes the leader’s tiles through DSMEM (or TMA multicast) and proceeds. The block-scoped pipeline does not explicitly synchronize other blocks. cluster.sync() and DSMEM semantics provide the cluster-wide visibility.

When designed to use TMA multicast and DSMEM efficiently, the thread block cluster pipeline will fetch each tile exactly once and multicast it into every thread block’s shared memory. This is in contrast to redundantly reloading the data from each thread block. As such, although every SM still cannot exceed its 64-warp limit, the global thread block cluster coordination ensures that those warps spend far less time on duplicate loads and far more time on actual computation.

Moreover, in the single thread block version, each block loads its next tile as soon as it finishes computing the current one. If one thread block completes its computation slightly earlier than another, it immediately issues its own load for the next tile—even if that tile is already being loaded by another block. This results in redundant memory traffic.

However, with the thread block cluster pipeline version, if a compute warp on SM 0 finishes early, it does not fetch its next tile immediately. Instead, it stalls at consumer_wait() until the global loader—possibly running on SM 7—finishes bringing that tile into shared memory for all thread blocks. In other words, the compute warp waits for the cluster’s single, shared load rather than issuing its own redundant copy.

By pausing until the tile is guaranteed available, each SM participating in a thread block cluster avoids spinning idle or performing duplicate loads. This alignment across thread block clusters smooths out variations in load times and keeps every SM’s compute warps busy with data loaded only once per cluster. This improves overall throughput.

Consider a problem so large and a grid so full of work that every SM already has loader, compute, and storer warps active under the single-block approach. In that case the raw warp counts and per-SM occupancy can look similar between the two kernels. The cluster version helps when multiple blocks reuse the same input tiles by eliminating duplicate global reads within a cluster and by using cluster-wide synchronization to align stages.

This often produces a modest throughput improvement on reuse-heavy workloads, but the gain depends on tile reuse, alignment, and resource limits. Both designs remain bounded by the same architectural limits, such as resident warps, registers, and shared memory.

All of the kernels here use the CUDA pipeline for producer and consumer handoffs without calling a block-wide __syncthreads. The naive tiled kernel does not overlap memory and compute. The two-stage double-buffered pipeline overlaps copies with compute and can significantly reduce time to solution on memory-bound cases.

The warp-specialized pipeline dedicates separate warps to load, compute, and store to avoid implicit full-block waits. The thread block cluster variant shares tiles across blocks in a cluster through distributed shared memory or multicast so that a tile is fetched once per cluster rather than once per block. These patterns improve utilization and reduce redundant memory traffic.

## Key Takeaways

The following are key takeaways from this chapter, which is focused on extracting peak performance on modern GPUs. These will keep kernels from idling on DRAM, warp schedulers, and global barriers:

*Hide latency with pipeline depth* Use two-stage (cuda::pipeline_shared_state<cuda::thread_scope_block, 2>) tiling to overlap asynchronous loads with compute and add another stage (cuda::pipeline_shared_state<cuda::thread_scope_block, 3>) when compute outweighs memory (e.g., modern GPUs like Blackwell.) This will help to eliminate idle warps.

*Balance workloads with warp specialization* Assign separate warps to loading, computing, and storing when compute phases dominate, ensuring near-peak warp efficiency on modern GPU hardware.

*Remove launch overhead with persistent kernels* Run a single long-lived kernel over a device-side work queue and use grid.sync() for multiphase algorithms. This will reduce host-device round trips and overall launch costs.

*Enable on-chip sharing with thread block clusters and DSMEM* Group thread blocks (CTAs) into clusters so they share a contiguous on-chip buffer. For a cluster-wide broadcast, use cp.async.bulk.tensor with a TMA multicast descriptor. Use TMA multicast to broadcast tiles once to every thread block. This will boost L2 hit rates and trim DRAM bandwidth.

*Pay special attention to barrier semantics* Both cluster.sync() and grid.sync() require every participating thread block to reach the same synchronization point. Mismatched control flow—or an excessive cluster size—may lead to deadlock or launch failure.

*Profile before tuning* Use Nsight Compute to identify whether your kernel is memory bound or compute bound. If memory bound, start with a two-stage pipeline. If compute bound, consider warp specialization or thread block clustering.

*Verify compiler-generated pipelines before hand-tuning* Profile and inspect the framework’s generated code from compilers like PyTorch’s compiler and NVCC. If the compiled code uses an asynchronous pipeline using cuda::memcpy_async, producer_commit(), and consumer_wait(), manual tuning likely won’t produce much of a speedup.

## Conclusion

The techniques discussed in this chapter help to systematically hide latency and remove redundant loads. This keeps the GPU at near-peak utilization for the duration of the kernel’s execution.

Warp-specialized pipelines overlap load, compute, and store operations. Cooperative-group barriers (grid.sync() and cluster.sync()) help coordinate multiphase work without host round trips. And persistent kernels loop over a device-side queue to eliminate launch overhead.

As always, you should start by profiling. If global-memory stalls dominate, a two-stage asynchronous-copy pipeline like double buffer usually suffices. If compute warps still stall, switch to a multistage warp-specialized pipeline such that loader, compute, and storer warps can operate with minimal contention. The only contention would be at the memory subsystem.

> Remember that concurrency is beneficial up to the point of hardware (e.g., memory bandwidth) saturation. Beyond that, parallel tasks contend for throughput.

For multiphase reductions or irregular tasks, replace multiple launches with a single persistent kernel plus grid.sync() to preserve occupancy. And when thread blocks need the same data (e.g., multiheaded attention), you can form a thread block cluster so that DSMEM and TMA load each tile only once—and multicast the data to the other thread blocks—without repeatedly accessing global memory. These techniques will move performance closer to the GPU’s peak theoretical limits.

As you continue through the next chapters, it’s important to remember these principles because they apply to many more optimizations. Specifically, you should overlap work at the warp and thread block levels, synchronize directly on-device (versus on the host), and share data on chip. These are essential mechanisms for tuning ultra-high-performance GPU workloads.

In the next chapter, we keep these intra-kernel building blocks—cuda::pipeline double-buffering, warp-specialized roles, and thread-block clusters—and show how to drive them through CUDA streams. The goal is to hide latency between kernels and between host ↔ device communication and not just inside a single kernel. Concretely, we will reuse the kernels from this chapter and run them in multistream pipelines with cudaMemcpyAsync, cudaMallocAsync/cudaFreeAsync, and event-based synchronization. This will help push the entire system toward achieving peak performance across many GPUs in your AI system.