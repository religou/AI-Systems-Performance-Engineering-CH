As in Chapter 10, we launch the kernel with a cluster dimension so that blocks join a cluster_group and can access each other’s distributed shared memory. The leader block stages its copies through a block-scoped cuda::thread_scope_block, while all blocks read the leader’s tiles using cluster.map_shared_rank.

Using cluster scope and distributed shared memory, the load stage runs once per cluster in the leader, while compute and store run concurrently across the cluster’s blocks. As before, each warp’s warp_id determines its role, and warps loop persistently over all tiles, fetching, computing, and storing in a synchronized rotation.

By adding CUDA streams and launching independent copies of this kernel in separate CUDA streams (NUM_STREAMS = 2), we keep the GPU busy on multiple batches of input data. In each stream, we perform the following steps:

1. Allocate per-batch device buffers for each thread block to use with cudaMallocAsync.

2. Stage inputs from host to device with cudaMemcpyAsync.

3. Launch the clustered warp-specialized kernel using cudaLaunchKernelExC.

4. Copy the outputs back to the host with cudaMemcpyAsync.

5. Free device buffers with cudaFreeAsync.

Because each stream enqueues its own cooperative launch along with its asynchronous copies and frees, the host can keep several mini-batches in flight even though only one cooperative kernel runs at a time. Remember that the GPU serializes cooperative kernel launches because each cooperative kernel pins every thread block simultaneously (this limitation will be reemphasized after the code example).

Using multiple streams (NUM_STREAMS = 2) still lets us overlap host-side allocations and copies with the previous kernel’s execution. For instance, while stream 0’s thread block cluster works on tile *n*, stream 1 can asynchronously allocate buffers (cudaMallocAsync) and copy tile n+1 into device memory, and stream 2 could be writing tile n–1’s results back to the host.

> The cooperative kernel must occupy every thread-block slot (CTA) that it needs on the GPU. This prevents any other cooperative launch from running simultaneously because all those CTA resources are already in use.

In practice, stream 0 issues its cooperative launch and begins executing, while stream 1 can immediately enqueue its own launch. But the second launch remains pending until stream 0’s kernel finishes.

However, as soon as stream 0 completes, stream 1’s launch starts instantly. This is because its inputs were already staged by cudaMemcpyAsync—and its buffers were already allocated by cudaMallocAsync.

Combining thread block clusters, intra-kernel warp specialization (the three-stage cuda::thread_scope_cluster), and inter-kernel CUDA streams, we create two layers of pipelining that span multiple thread blocks. These two-layer pipelines push the hardware to its peak.

In the first layer, the thread block cluster pipeline lets loader, compute, and storer warps cooperate across all thread blocks. Thread block clusters can use TMA multicast to replicate a global memory tile transfer into the shared memory of each block within the cluster. Multicast is local to the thread block cluster. In essence, the thread block cluster fetches each tile exactly once. In the second layer, multiple streams hide host-side allocations, copies, and kernel launches behind one another.

In short, these combined techniques make sure that memory latency is hidden at both the grid level and the host-to-device level. This drives hardware utilization close to peak on modern GPUs—and maximizes throughput for LLM workloads.

> In many scenarios, memory-hierarchy bottlenecks like DRAM bandwidth and model-parallel communication like NCCL all-reduce will dominate the workload. So once your kernel is close to saturating the HBM bandwidth and your streams already hide CPU → GPU latency, the additional few percent of SM utilization that you’d get by adding warp specialization on top of that—and then adding thread block clusters on top of that—rarely justifies the steep engineering cost. In practice, most real-world LLM training and inference workloads benefit sufficiently from simpler designs presented earlier, such as double-buffered kernels or two-stage pipelines with CUDA streams.

## Multi-GPU Compute and Data Transfer Overlap with CUDA Streams

When training or serving LLMs across multiple GPUs, CUDA streams let you overlap local compute, peer-to-peer transfers, collective communication, data preparation, and memory management so that no device idles. For instance, suppose you split a transformer model across two GPUs such that GPU 0 handles layers 0–3 and GPU B handles layers 4–7. On GPU A, you might write:

```
// Stream 0 on GPU A: compute layers 0–3
myTransformerLayers0to3<<<gridA, blockA, 0, stream0_A>>>(
    inputActivationsA, outputActivationsA);
cudaEvent_t eventA, eventFromA;
cudaEventCreateWithFlags(eventA,
    cudaEventDisableTiming); // lowers overhead for sync-only events
cudaEventCreateWithFlags(eventFromA,
    cudaEventDisableTiming);  // lowers overhead for sync-only events
cudaEventRecord(eventA, stream0_A);
```

Meanwhile, on the CPU, you have already decoded or prepared the next microbatch (N+1) and issued a cudaMemcpyAsync into a pinned host buffer. This way, by the time GPU A’s stream 0 finishes the forward pass for batch *N*, the CPU-to-GPU copy for batch N+1 can begin immediately without waiting.

At the same time, you call cudaStreamWaitEvent(stream1_A, eventA, 0) to ensure that stream1_A waits until outputActivationsA is ready before launching the direct NVLink/PCIe copy into GPU B’s memory, as shown here:

```
// Stream 1 on GPU A: wait for layer computation, then copy to GPU B
cudaStreamWaitEvent(stream1_A, eventA, 0);
cudaMemcpyPeerAsync(
    destActivationsB, /*dstGPU=*/1,
    outputActivationsA, /*srcGPU=*/0,
    bytes, stream1_A);
```

As soon as you record eventA, GPU A’s stream0 can immediately launch the next forward kernel for batch N+1, using cudaMemcpyAsync to copy its inputs from the pinned host buffer into device memory without blocking. On GPU B, you record a matching event when the copy begins or ends:

```
// Stream 1 on GPU A: after peer copy starts,
// record eventFromA
cudaEventRecord(eventFromA, stream1_A);
```

Then on GPU B, you wait for data to arrive and run layers 4–7, as shown here:

```
// Stream 1 on GPU B: wait for data arrival, then run layers 4–7
cudaStreamWaitEvent(stream1_B, eventFromA, 0);
myTransformerLayers4to7<<<gridB, blockB, 0, stream1_B>>>(
    destActivationsB, nextActivationsB);
```

This explicit event-and-wait pattern guarantees that GPU B’s layers 4–7 compute only begins after the peer copy has completed. Meanwhile, GPU B’s stream0_B can simultaneously prefetch weights or perform other preparatory work.

The P2P transfer used here runs on GPU A’s DMA copy engine and doesn’t use any of GPU A’s SMs. This way, GPU A’s compute stream can immediately proceed to batch N+1 without having to manage the data transfer.

The result is a three-way overlap: GPU A’s SMs can immediately start batch N+1 in stream0_A, GPU A’s peer DMA engine can shuttle batch *N* activations to GPU B in stream1_A, and GPU B’s SMs can run batch *N* in stream0_B or begin batch *N* in its stream1_B. By partitioning work into separate streams and using pinned memory on the host, we hide P2P and H2D latency behind ongoing computation and data preparation.

When it’s time to synchronize gradients or broadcast parameters across many GPUs, NCCL handles the communication at scale using low-occupancy device kernels that drive GPUDirect P2P or RDMA over NVLink, PCIe, or InfiniBand. Under the hood, NCCL breaks the tensors into multiple contiguous chunks.

This design shows that each GPU can have multiple streams, including a compute stream for its primary work, a receive stream for incoming data, and a communication stream for reductions. Using dedicated streams per role—and giving communication streams higher priority—allows for optimal overlap. In fact, frameworks like PyTorch typically run NCCL collectives on these dedicated streams with higher priority for network transfers.

> PyTorch and NCCL both use dedicated, high-priority streams to interleave communication with compute-intensive operations. This way, they don’t get delayed behind large compute kernels.

NCCL will choose a ring or tree algorithm based on NVLink or PCIe topology. Consider a four-GPU ring, as shown in Figure 11-8, with four chunks (1–4).

![Figure 11-8. Chunked all-reduce with four GPUs in a ring](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-8.png)

In this ring all-reduce, each GPU sends chunk *i* to its neighbor (k → k+1) while receiving chunk i−1 from its other neighbor (k−1 → k), using device kernels to move and reduce chunks over NVLink or PCIe.

By pipelining these chunked sends and receives, NCCL keeps NVLink fully saturated. While chunk *i* traverses from GPU 0 → GPU 1, chunk i−1 moves GPU 1 → GPU 2, and so on. This minimizes idle gaps. In code, you would see something like the following:

```
// In a high-priority communication stream on each GPU:
cudaEventRecord(eventK, computeStream);
cudaStreamWaitEvent(commStream, eventK, 0);
ncclAllReduce( // this is asynchronous
    gradBuffer, gradBuffer, numElements,
    ncclFloat, ncclSum, comm, commStream);
```

Because NCCL uses only a few SM thread blocks, it orchestrates the chunked send/recv + reduce across NVLink or NVSwitch in a low-occupancy manner on the SMs. Meanwhile, your main compute stream running a backward pass for layer k+1, for instance, can continue running. When the all-reduce completes, the event is recorded as shown here:

```
cudaEventRecord(eventAllReduceDone, commStream); // signals collective completion
cudaStreamWaitEvent(computeStream, eventAllReduceDone, 0);
// Now apply optimizer updates on computeStream...
```

Remember that NCCL collectives like all-reduce are asynchronous. They return control immediately. By recording an event after the call and waiting on it in the compute stream, you ensure that the compute stream doesn’t resume until the reduction is actually completed.

> NCCL’s kernels have been purposely optimized to achieve maximum bandwidth at low occupancy using a small amount of SM resources per GPU. In practice, NCCL can saturate NVLink or PCIe bandwidth with a small number of thread blocks per GPU, thanks to low-occupancy kernels tuned for the interconnect. In addition, you can offload some of these collective aggregation operations to NVIDIA SHARP if your network hardware supports it. This will free up even more SM resources to perform more useful computational work.

NIXL provides a unified, high-throughput API for point-to-point and disaggregated transfers across NVLink, RDMA, and storage backends, so you typically use NIXL APIs instead of calling cudaMemcpyPeerAsync yourself. NIXL moves data quickly and asynchronously across different tiers of memory and interconnects. When you invoke a NIXL operation with nixlCommStream, the data is chunked and pipelined over the network using the fastest transport mechanism available, such as GPUDirect RDMA or NVLink.

And like NCCL, NIXL can use dedicated high-priority streams for chunked peer-to-peer and collective data transfers. This can reduce queuing delays so their copy commands hit the GPU’s DMA engines soon after they are issued—likely preempting lower-priority work.

Marking these transfer streams as high priority doesn’t mean they consume SM resources upfront. It simply guarantees that when the copy commands are ready, they’ll be issued ahead of other low-priority operations.

In practice, the copy engines then take advantage of whatever DRAM-to-DRAM bandwidth isn’t already being used by your compute kernels. As such, they effectively use only idle memory-fabric bandwidth to move data across the interconnects.

This design maximizes overlap such that while one stream drives cudaMemcpyPeerAsync transfers or NCCL reduction kernels on the copy engine, another stream’s SM kernels can continue processing forward/backward work. And a third stream can perform asynchronous allocations or event waits. This keeps all of the hardware units busy and without contention.

Peer-to-peer transfers launched using cudaMemcpyPeerAsync run entirely on the GPU’s copy engines, so they consume only DRAM-to-DRAM bandwidth and no SM cycles. Collective operations such as all-reduce, on the other hand, are implemented as device kernels that use a small number of SMs while driving interconnect bandwidth toward the peak.

At the same time, temporary buffers for activations or mixed-precision gradients are allocated and freed asynchronously to avoid global stalls. For example, inside each GPU’s compute stream, you would call the following code:

```
cudaMallocAsync(&tempBuffer, tempBytes, computeStream);
// ...use tempBuffer in kernels...
cudaFreeAsync(tempBuffer, computeStream);
```

Because the stream-ordered memory allocator records these operations without forcing a device-wide synchronization, CUDA does not pause the other streams such as P2P, NCCL, and NIXL streams when allocating or freeing memory. This makes sure that buffer management never interrupts the overlapped compute and communication pipeline.

> This is exactly the stream-ordered memory allocator that we discussed earlier—but now applied to multiple GPUs. Using this will help prevent memory management from bottlenecking your distributed workloads.

After you enqueue the forward pass, backward pass, P2P copies, NCCL all-reduce, memory free, and event-wait operations for a single iteration, you can capture an execution chain (e.g., training or inference iteration) into a single CUDA Graph.

Specifically, when you call cudaStreamBeginCapture, every operation you enqueue into the stream (e.g., kernel launches, computations, communications, data transfers, events, etc.) are inserted as nodes in the graph. This can include ncclAllReduce(), cudaMemcpyAsync, cudaMemcpyPeerAsync, cudaMallocAsync, cudaFreeAsync, cudaEventRecord, cudaStreamWaitEvent—as well as their dependencies. You end the capture with cudaStreamEndCapture(stream, &graph). The code looks something like this:

```
cudaStreamBeginCapture(streamA, cudaStreamCaptureModeGlobal);
computeGradientsLayer1<<<... , 0, streamA>>>(...);
ncclAllReduce(..., comm, streamB);
computeGradientsLayer2<<<... , 0, streamA>>>(...);
ncclAllReduce(..., comm, streamB);
cudaStreamEndCapture(streamA, &graph);
```

When capturing work submitted to multiple streams, use cudaStreamCaptureMode Global so operations enqueued on any participating stream in the same thread are recorded into the same capture session—and cross-stream dependencies are preserved. Otherwise, only the operations enqueued on the stream passed to cudaStream BeginCapture are captured.

You can then instantiate and launch the graph with cudaGraphLaunch(). This will replay the entire DAG with near-zero CPU overhead. By capturing the whole multi-GPU iteration once, you eliminate per-operation enqueue and launch overhead.

Moreover, CUDA Graphs support conditional nodes (e.g., gradient clipping), so infrequent branches stay inside the graph’s logic. We’ll cover CUDA Graphs in more detail in the next chapter, but it’s important to understand their relationship to CUDA streams, including cudaStreamBeginCapture and cudaStreamEndCapture.

> Modern frameworks like PyTorch hide much of this complexity. For example, PyTorch’s DistributedDataParallel automatically schedules data transfers, compute, and communications in separate streams. It also uses CUDA Graphs to reduce per-iteration overhead. While it can use CUDA Graphs to reduce overhead, you must explicitly enable this by using capture-safe code and APIs because it’s not automatically enabled.

While you don’t need to explicitly use these CUDA stream begin and end APIs to build a CUDA Graph, they are convenient when you want to capture the graph at the same time you are enqueuing operations into your stream (we’ll cover CUDA Graphs in more detail in the next chapter).

CUDA streams enable finely tuned pipelines in multi-GPU systems by overlapping work across CPUs, DMA engines, and SMs. On the CPU side, cudaMemcpyAsync into pinned host buffers stages data for the next batch while the GPU executes the current workload, ensuring that inputs for batch N+1 are ready before batch *N* completes. At the same time, peer-to-peer transfers between GPUs (using cudaMemcpyPeerAsync) can be scheduled on dedicated streams, synchronized with cudaStreamWaitEvent to hand off activations or gradients without stalling ongoing compute.

As mentioned, collective communication libraries like NCCL or NIXL are low-occupancy communication kernels that use separate streams. This way, while one GPU is reducing gradients or broadcasting parameters, other streams on that same GPU can continue executing local compute kernels (e.g., forward or backward passes for subsequent layers). In addition, using cudaMallocAsync and cudaFreeAsync in stream order prevents global synchronization for temporary buffers, letting allocation and deallocation proceed concurrently with compute and communication.

Capturing the entire iteration (e.g., CPU staging, P2P transfers, compute kernels, NCCL calls, allocations, frees, and events/waits) into a CUDA Graph can further reduce CPU overhead. Once the graph is instantiated, invoking cudaGraphLaunch() replays the recorded DAG in one go. This eliminates per-call enqueue overhead and preserves all dependencies automatically.

Together, these techniques ensure that each GPU’s SM pipelines, copy engines, and interconnect links remain busy. While one stream runs matrix-multiply kernels, another performs peer-to-peer copies, a third executes collective communication, and the CPU stages data for the next microbatch.

In short, you can use CUDA streams to orchestrate work across multiple layers and overlap computation, memory operations, and data transfers. This approach drives every hardware unit to its limits by hiding peer-to-peer and collective communication latencies behind active compute. Overall, CUDA streams maximize throughput, utilization, and efficiency for many AI workloads, including LLM training and inference.

## Programmatic Dependent Launch

Another type of inter-kernel pipelining and communication is called *Programmatic* *Dependent Launch* (PDL). PDL lets a kernel trigger another kernel’s execution directly on the device using the same CUDA stream—and without involving the CPU. For instance, Kernel A, the primary kernel, can trigger Kernel B, the secondary kernel, which is waiting on a signal from Kernel A.

This trigger can happen even before Kernel A finishes execution. To do this, it uses cudaTriggerProgrammaticLaunchCompletion(), as shown in Figure 11-9. Here, Kernel B is waiting on Kernel A with a call to cudaGridDependencySynchronize().

![Figure 11-9. Using PDL to launch Kernel B from Kernel A—partially overlapping Kernel B’s execution with Kernel A’s epilogue (in the same CUDA stream)](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-9.png)

Using the constructs provided by PDL, Kernel A can signal Kernel B to execute during Kernel A’s *epilogue* (e.g., wrap-up) phase. This way, Kernel A can execute alongside Kernel B for a bit of time. It’s important to note that Kernel A should not trigger Kernel B to execute until Kernel A has produced and synchronized all data (e.g., L2/shared memory/global memory) needed by Kernel B.

> The data dependencies of the dependent kernel should be visible in L2, shared memory, global memory, etc., before it continues processing.

The code here shows Kernel A using cudaTriggerProgrammaticLaunchCompletion() to signal to Kernel B that its main work has completed. This also notifies Kernel B that all necessary global-memory flushes have occurred—and that it’s safe to continue:

```
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
// Kernel A must trigger the PDL flag when it's
//    safe to launch Kernel B
__global__ void kernel_A(int *d_ptr) {
    // Perform work that produces data used by
    //   Kernel B

    // Signals that Kernel A's global-memory
    //   flushes are complete
    // This enables dependent Kernel B's launch
    cudaTriggerProgrammaticLaunchCompletion();
    // ... any further work that can overlap with  ...
    ...
}
// Kernel B must wait for Kernel A to write its memory
//    to global memory and become visible to Kernel B
__global__ void kernel_B(int *d_ptr) {
    // Wait on Kernel A to complete.
    // This ensures the Kernel B waits for the
    //   memory flush before accessing shared data
    cudaGridDependencySynchronize();
    // ... dependent work on d_ptr ...
    ...
}
int main() {
    // 1) Allocate device buffer
    int *d_ptr = nullptr;
    // Allocate an int (example)
    cudaMalloc((void**)&d_ptr, sizeof(int));
    // 2) Create a nonblocking stream for maximum overlap
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream,
        cudaStreamNonBlocking);  // Nonblocking
    // 3) Define grid/block sizes
    dim3 gridDim(128), blockDim(256);
    // 4) Launch Kernel A asynchronously
    kernel_A<<<gridDim, blockDim, 0,
        stream>>>(d_ptr);    // Async launch
    // 5) Configure PDL for Kernel B
    cudaLaunchConfig_t launch_cfg{};
    launch_cfg.gridDim           = gridDim;
    launch_cfg.blockDim          = blockDim;
    launch_cfg.dynamicSmemBytes  = 0;
    launch_cfg.stream            = stream;
    // Sets the PDL flag so cudaLaunchKernelExC overlaps
    //   with Kernel A's epilogue
    static cudaLaunchAttribute attrs[1];
    attrs[0].id  = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed =
        1;
    launch_cfg.attrs    = attrs;
    launch_cfg.numAttrs = 1;
    // 6) Pack the pointer argument
    void* kernelArgs[] = { &d_ptr };
    // 7) Launch Kernel B kernel early using PDL
    // Lookup device pointer for secondary_kernel
    cudaKernel_t kB;
    cudaGetKernel(&kB, kernel_B);
    void* funcPtr_kernel_B = reinterpret_cast<void*>(kB);
    cudaLaunchKernelExC(&launch_cfg, funcPtr_kernel_B, kernelArgs);
    // 8) Wait until all work in the stream completes
    cudaStreamSynchronize(stream);
    // 9) Cleanup
    cudaStreamDestroy(stream);
    cudaFree(d_ptr);
    return 0;
}
```

Here, Kernel B is waiting on cudaGridDependencySynchronize() until it receives this programmatic-launch completion signal from Kernel A. Once the handoff occurs, the two kernels can overlap their execution. By pairing the trigger in Kernel A, the synchronize in Kernel B, and the launch attribute on the host, this code achieves as much overlap as possible between kernels A and B.

As seen in this example, you need to create a cudaLaunchConfig_t and use special attributes with the cudaLaunchKernelExC() CUDA call when using PDL. Specifically, on the host side, PDL is enabled by configuring a cudaLaunchConfig_t for Kernel B’s launch with the cudaLaunchAttributeProgrammaticStreamSerialization attribute enabled to allow early, overlapped dispatch.

Calling cudaLaunchKernelExC() with cudaLaunchAttributeProgrammaticStream Serialization tells the CUDA runtime that it’s safe to enqueue Kernel B—even if Kernel A isn’t fully complete. cudaLaunchKernelExC() is then used to perform the actual launch with these extended attributes, which relies on the trigger mechanism to perform the handoff.

## Combining PDL and Thread Block Clusters with Warp Specialization

Let’s bring together three orthogonal techniques—PDL, warp-specialized pipelining, and thread block clusters—into one execution model.

PDL provides the mechanism for one kernel to signal completion of its prologue and trigger a dependent kernel to execute. It will then ramp down while the dependent kernel ramps up and executes.

Warp specialization subdivides each thread block into producer and consumer warps. Specifically, the producer warp asynchronously transfers tiles using TMA, while the compute warp executes matrix-multiply-accumulate (MMA) operations.

And thread block clusters guarantee that multiple thread blocks run on nearby groups of SMs. This facilitates multi-SM cooperation and shared-memory multicast for large-scale workloads.

Together, this combination of inter-kernel and intra-kernel techniques hides DRAM latency, reduces kernel-boundary overheads, and maximizes Tensor Core utilization. They create a pipeline with the following characteristics:

*Intra-kernel pipelining* Warp specialization makes sure that within each thread block, data transfers (producer warps) and compute (consumer warps) are fully overlapped using a multistage pipeline.

*Inter-kernel overlap* PDL allows the prologue of a dependent GEMM kernel, potentially operating on the next layer in a neural network, to begin as soon as the primary kernel finishes prefetching data (weights)—without waiting for the full thread block to tear down.

*Interblock cooperation* Thread block clusters enable groups of thread blocks to coordinate prefetch (e.g., multicast) and perform dynamic load balancing across SMs. This way, both producer and consumer tasks are evenly distributed cluster-wide.

For example, you can overlap the movement of model weights (data transfer) with GEMMs (compute) inside the same kernel so that one tile is being computed while the next tile is in flight. A warp-specialized, multistage software pipeline (stages 0… N) can coordinate these roles using PDL and *mbarrier* primitives, as shown in Figure 11-10.

![Figure 11-10. Warp-specialized, multistage pipeline with PDL and thread block clusters and TMA multicast for a high-performance, inter-kernel and intra-kernel GEMM implementation](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-10.png)

Here is a CUDA C++ example that shows how to combine PDL, thread block clusters, and warp specialization with a simple TMA-style async copy + compute pipeline. Specifically, the primary kernel, primary_gemm, uses cudaTriggerProgrammatic LaunchCompletion() to signal to the secondary kernel that all memory flushes have completed and the data is ready to be consumed. As such, it’s now safe for the secondary kernel, secondary_gemm, to continue from cudaGridDependencySynchronize(), as shown here:

```
#include <cstdio>
#include <cuda_runtime.h>
// Cooperative Groups for clusters/barriers
#include <cooperative_groups.h>
// C++ barrier for TMA-like sync
#include <cuda/barrier>
// Async copy API
#include <cuda/pipeline>
namespace cg = cooperative_groups;
// Tile size for our toy GEMM
constexpr int TILE_M = 128;
constexpr int TILE_K = 128;
constexpr int TILE_N = 128;
// A very simple “producer/consumer” pipeline within each CTA
__global__ __cluster_dims__(2,1,1)    // Compile-time cluster of 2 CTAs
void primary_gemm(const float* __restrict__ A,
                  const float* __restrict__ B,
                        float* __restrict__ C,
                  int M, int N, int K)
{
    // Identify thread-block cluster & within-block group
    cg::thread_block_cluster cluster = cg::this_thread_block_cluster();
    cg::thread_block       cta     = cg::this_thread_block();
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int warpId = tid / warpSize;
    const int numProducerWarps = 1;
    // Shared-memory tile buffers
    __shared__ float tileA[TILE_M * TILE_K];
    __shared__ float tileB[TILE_K * TILE_N];
    __shared__
    cuda::pipeline_shared_state<cuda::thread_scope_block, 1> pipe_state;
    auto pipe = cuda::make_pipeline(cta, &pipe_state);
    if (warpId < numProducerWarps) {
    pipe.producer_acquire();
    cuda::memcpy_async(cta, tileA, A, cuda::aligned_size_t<32>{TILE_M
                       * TILE_K * sizeof(float)}, pipe);
    cuda::memcpy_async(cta, tileB, B, cuda::aligned_size_t<32>{TILE_K
                       * TILE_N * sizeof(float)}, pipe);
    pipe.producer_commit();
    }
    cta.sync();
    pipe.consumer_wait();
    pipe.consumer_release();
// ... perform “compute” on the tile ...
// (e.g., a few fused multiply-adds)
do_compute();
    // Inter-CTA cluster-scope sync for load balancing
    cluster.sync();
    // Signal to dependent kernel that prologue is done
    cudaTriggerProgrammaticLaunchCompletion();
    // ... perform remaining epilogue work ...
    // ...
}
__global__ __cluster_dims__(2,1,1)
void secondary_gemm(const float* __restrict__ A,
                    const float* __restrict__ B,
                          float* __restrict__ C,
                    int M, int N, int K)
{
    // Wait for primary’s PDL signal before starting
    cudaGridDependencySynchronize();
    // Similar warp-specialized pipeline as above...
    //  (Omitted for brevity. Duplicate of primary logic,
    //  but reading from different offsets to compute
    //  next GEMM tile.)
}
// cudaLaunchKernelExC, cudaGetKernel
#include <cuda_runtime_api.h>
// cudaLaunchConfig_t
#include <cuda/launch_config.h>
int main()
{
    // Problem dimensions (must be multiples of TILE_)
    int M = TILE_M, N = TILE_N, K = TILE_K;
    // Allocate and initialize matrices A, B, C on device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M*K*sizeof(float));
    cudaMalloc(&d_B, K*N*sizeof(float));
    cudaMalloc(&d_C, M*N*sizeof(float));
    // (Initialize d_A, d_B via cudaMemcpy or kernels...)
    // Create a nonblocking stream for overlap
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream,
        cudaStreamNonBlocking);
    // Launch primary GEMM
    dim3 gridDim(M/TILE_M, N/TILE_N), blockDim(256);
    primary_gemm<<<gridDim, blockDim, 0, stream>>>(d_A,
        d_B, d_C, M, N, K);
    // Configure PDL attributes for secondary launch
    cudaLaunchConfig_t launch_cfg = {};
    launch_cfg.gridDim          = gridDim;
    launch_cfg.blockDim         = blockDim;
    launch_cfg.dynamicSmemBytes = 0;
    launch_cfg.stream           = stream;
    static cudaLaunchAttribute attrs[1];
    attrs[0].id  = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    launch_cfg.attrs    = attrs;
    launch_cfg.numAttrs = 1;
    // Prepare arguments and get function pointer
    //    for secondary_gemm
    void* kernelArgs[] = {&d_A, &d_B, &d_C, &M, &N, &K};
    cudaKernel_t k;
    cudaGetKernel(&k, secondary_gemm);
    void* funcPtr = reinterpret_cast<void*>(k);
    // Early enqueue of secondary GEMM via PDL
    cudaLaunchKernelExC(&launch_cfg, funcPtr, kernelArgs);
    // Wait for everything to finish
    cudaStreamSynchronize(stream);
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaStreamDestroy(stream);
    return 0;
}
```

On the host, you launch the primary kernel (primary_gemm) into a nonblocking stream and create a cudaLaunchConfig_t with the ProgrammaticStreamSerialization attribute. You use this to call cudaLaunchKernelExC for the secondary kernel (secondary_gemm).

Inside primary_gemm, a call to cudaTriggerProgrammaticLaunchCompletion() marks the completion of the memory flush and allows the dependent kernel’s processing to begin—even before the primary has fully torn down.

The dependent kernel then calls cudaGridDependencySynchronize(). This waits for the signal and the necessary memory flush from the primary kernel so that it can start executing in parallel with the primary kernel’s epilogue.

Within each thread block, we use warp specialization to overlap data movement and computation. By splitting each block into a single “producer” warp and multiple “consumer” warps, the producer issues cuda::memcpy_async calls into shared memory. This uses TMA multicast to broadcast a single DMA transfer to all thread blocks in the cluster, as shown in Figure 11-11.

![Figure 11-11. TMA multicast: a leader CTA issues cp.async.bulk.tensor into DSMEM (cluster shared memory); the follower CTAs consume tiles using cluster.map_shared_rank; cuda::memcpy_async drives TMA](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-11.png)

While the producer warp is loading data with TMA multicast, the consumers spin on a C++ block-scope barrier (cuda::barrier<cuda::thread_scope_block>) before executing their matrix-multiply steps (do_compute()). This lets each tile’s load and its fused-multiply-add (FMA) operations interleave in a fine-grained, multistage software pipeline that hides global-memory latency inside the kernel.

To coordinate work across SMs, we annotate both kernels with __cluster_dims__(2,1,1). This groups pairs of thread blocks on nearby SMs. A call to cluster.sync() (the cooperative-group’s wrapper over PTX’s mbarrier instructions) serves as a cluster-wide barrier and shared-memory fence. This way, all of the thread blocks in the cluster see the same TMA-loaded data and can dynamically rebalance remaining tiles. This interblock cooperation prevents idle SMs and increases the benefit of the warp-specialized pipeline.

> Production kernels typically use the Tensor Memory Accelerator (TMA) cp.async.bulk.tensor for 2D tiles and multicast across a thread block cluster when multiple thread blocks need the same tile. Consider using descriptor-based TMA reduce operations (e.g., cp.reduce.async.bulk(.tensor)) for on-the-fly reductions during tiled copies on Blackwell. Prefer descriptor-based loads/stores plus TMA reduce operations when fusing small reductions into the data-movement pipeline. This will reduce register pressure and improver overlap.

Let’s wrap up by revisiting the three characteristics of this high-performance GEMM pipeline:

*Intra-kernel pipelining* Achieved within each thread block using warp specialization to subdivide work so that producer warps issue cuda::memcpy_async tiles while consumer warps spin on a block-scope barrier. This lets data transfer and compute operations fully overlap.