# Chapter 11. Inter-Kernel Pipelining, Synchronization, and CUDA Stream-Ordered Memory Allocations

So far, we have focused on the intra-kernel tools—cuda::pipeline double-buffering, warp specialization (loader/compute/storer warps), persistent kernels, and thread-block clusters with DSMEM/TMA—to keep the SMs busy for a single kernel. In this chapter we keep those kernels and show how to pipeline across kernels and batches with CUDA streams, events, and the stream-ordered memory allocator. In short, Chapter 10 focused on hiding latency within a kernel. This chapter shows how to hide latency between kernels and between the GPU and the host.

This kind of inter-kernel concurrency is essential for keeping all of the GPU’s engines busy in real-world workloads. To achieve peak GPU utilization with modern GPUs, we need to keep the GPU’s compute engines and direct memory access (DMA) engines busy and running in parallel.

CUDA streams provide the foundation for this inter-kernel concurrency. By combining asynchronous memory operations, fine-grained synchronization, and CUDA Graphs (briefly introduced in this chapter and covered in more detail in the next chapter), you can construct highly efficient pipelines that avoid host-side stalls.

## Overlapping Kernel Execution with CUDA Streams

A CUDA stream is a sequence of operations—kernel launches, memory copies, and memory allocations—that execute in the order they are issued. Consider launching 5 kernels from the CPU onto the GPU using 2 streams, as shown in Figure 11-1.

![Figure 11-1. Launching five kernels from the CPU onto the two streams running on the GPU](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-1.png)

Here, we see ker_A and ker_B are running on stream 2, while ker_1, ker_2, and ker_3 are running on stream 1. All kernels may overlap with one another—and across CUDA streams—as long as hardware resources permit.

The CPU is able to continue performing work (cpu_code_1 and cpu_code_2) while the streams perform the kernel operations asynchronously. The code to launch these five kernels on the two CUDA streams is shown here:

```
#include <cstdio>
#include <cuda_runtime.h>
__global__ void ker_A()  { /* ... do some work ... */ }
__global__ void ker_B()  { /* ... do some work ... */ }
__global__ void ker_1()  { /* ... do some work ... */ }
__global__ void ker_2()  { /* ... do some work ... */ }
__global__ void ker_3()  { /* ... do some work ... */ }
int main() {
    // 1) Create two CUDA streams
    cudaStream_t stream1, stream2;
    cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
    // 2) Define your grid/block sizes
    dim3 grid(128);
    dim3 block(256);
    // 3) Launch ker_1 on stream1
    ker_1<<<grid, block, 0, stream1>>>();
    // 4) CPU code 1 runs immediately (asynchronously wrt GPU)
    printf("CPU code 1 executing\n");
    // ... do some host-side work here ...
    cpu_code_1();
    // 5) Launch ker_A on stream2
    ker_A<<<grid, block, 0, stream2>>>();
    // 6) Launch ker_B on stream1
    ker_B<<<grid, block, 0, stream1>>>();
    // 7) Launch ker_2 on stream2
    ker_2<<<grid, block, 0, stream2>>>();
    // 8) CPU code 2 runs immediately
    printf("CPU code 2 executing\n");
    // ... do some other host-side work here ...
    cpu_code_2();
    // 9) Launch ker_3 on stream1
    ker_3<<<grid, block, 0, stream1>>>();
    // 10) Wait for work on each stream to finish
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    // 11) Clean up
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    return 0;
}
```

ker_1 is enqueued on stream1, then control returns immediately to the CPU. cpu_code_1() runs on the host while ker_1 executes on the GPU. Meanwhile, we enqueue ker_A on stream2 and ker_B on stream1. We then enqueue ker_2 on stream2, interleave cpu_code_2, and enqueue ker_3 on stream1. Finally, we synchronize on each stream to wait for all of the work to complete and then destroy the streams to clean up the resources.

This example highlights five different kernel executions overlapping across two different streams. Increasing the complexity a bit, and building upon Chapter 10, following is the same warp specialization pipeline example but using CUDA streams:

```
// Run the warp-specialized kernel in multiple CUDA streams.
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
#define TILE_SIZE 128
#define TILE_ELEMS (TILE_SIZE * TILE_SIZE)
// re-using from Chapter 10
__global__ void warp_specialized_pipeline(const float* __restrict__ A_global,
                                          const float* __restrict__ B_global,
                                          float*       __restrict__ C_global,
                                          int numTiles);
int main() {
    const int NUM_STREAMS = 2;                   // keep it small; tune as needed
    const int batches     = 8;                   // in-flight batches
    const size_t elems    = TILE_ELEMS;          // elements per batch
    const size_t bytes    = elems * sizeof(float);
    // Create streams that do NOT synchronize with the legacy default stream
    cudaStream_t s[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);
    // Allocate pinned host buffers so H2D/D2H can truly overlap
    float *hA = nullptr, *hB = nullptr, *hC = nullptr;
    cudaMallocHost(&hA, batches * bytes);
    cudaMallocHost(&hB, batches * bytes);
    cudaMallocHost(&hC, batches * bytes);
    // ... initialize hA/hB ...
    for (int b = 0; b < batches; ++b) {
        const int sid = b % NUM_STREAMS;
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cudaMallocAsync(&dA, bytes, s[sid]);
        cudaMallocAsync(&dB, bytes, s[sid]);
        cudaMallocAsync(&dC, bytes, s[sid]);
        cudaMemcpyAsync(dA, hA + b * elems, bytes,
                        cudaMemcpyHostToDevice, s[sid]);
        cudaMemcpyAsync(dB, hB + b * elems, bytes,
                        cudaMemcpyHostToDevice, s[sid]);
        const dim3 block(96);    // three warps: loader(0), compute(1), storer(2)
        const dim3 grid(1);
        const size_t shmem = 3 * elems * sizeof(float); // [A|B|C] per tile
        // Reuse the Chapter 10 kernel exactly as-is
        warp_specialized_pipeline<<<grid, block, shmem, s[sid]>>>(dA, dB, dC,
                                                                /*numTiles=*/1);
        cudaMemcpyAsync(hC+b*elems, dC, bytes, cudaMemcpyDeviceToHost, s[sid]);
        cudaFreeAsync(dA, s[sid]);
        cudaFreeAsync(dB, s[sid]);
        cudaFreeAsync(dC, s[sid]);
    }
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(s[i]);
        cudaStreamDestroy(s[i]);
    }
    cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC);
    return 0;
}
```

Here, we are re-using the warp-specialized pipeline and showing how streams add a second layer of overlap such that while stream 1 computes on batch n, stream 2 is performing DMA-loads on batch b+1. At the same time, it can copy back batch b−1. The kernel’s internal cuda::pipeline overlap remains unchanged.

We’ll continue to build out the complexity later in the chapter by layering in thread block clusters, but let’s first dive into how streams help to overlap compute with data transfers. This will help solidify the fundamentals of CUDA streams and their role in GPU-based performance engineering.

## Using Streams to Overlap Compute with Data Transfers

For instance, you can enqueue each kernel launch and memory copy into its own stream. This allows the SMs to execute kernels while the two dedicated DMA engines (one for Host → Device transfers and one for Device → Host transfers) move data concurrently.

Since the SM compute pipeline runs independently of the two DMA engines, you can fully overlap the kernel’s computation with the two data transfers using CUDA streams. However, if the compute is fully maxed out by a kernel using all SM throughput—or a copy pipeline is saturating the memory bandwidth with excessive, additional overlapping—it will not improve performance.

When compute and memory throughput are saturated, you’ll start seeing two concurrent operations each running at 50%, for instance, since they are both contending for the same resource. You can profile GPU utilization to identify these saturation thresholds.

For example, consider an AI model training or inference workload that breaks work into batches. Here, you would launch a kernel on batch 0 in stream 0 at the same time that stream 1 invokes cudaMemcpyAsync() to copy batch 1 from the host to device, as shown in Figure 11-2.

![Figure 11-2. Timeline of three-way overlap](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-2.png)

On modern GPUs with at least two copy engines (deviceProp.asyncEngine Count()), you can extend this to a three-way overlap such that stream 0 runs the kernel for batch 0, stream 1 copies batch 1 host to device, and stream 2 writes the results of the previous batch back to the host. This extends to additional streams. This pattern hides data-transfer latency behind computation, and vice versa, keeping all of the GPU’s engines busy and minimizing idle time.

In practice, your kernel must meet several requirements to achieve this concurrent behavior. First, any host pointer used in an asynchronous transfer must be page-locked, or pinned. If you call cudaMemcpyAsync() on pageable memory, the runtime performs a host-side staging copy into pinned memory that blocks the calling host thread and the enqueuing stream until staging completes.

This prevents that transfer from being asynchronous. While this blocks the calling host thread, the GPU can still overlap compute and copies in other streams. But that specific transfer will not overlap correctly. To achieve fully asynchronous transfers in your stream, you must use pinned host memory.

> By setting pin_memory=True with PyTorch’s DataLoader, you are page locking your host buffers so that data can be transferred using DMA directly into GPU memory. This allows the copy to overlap with computation and return control to the host immediately. DMA engines can overlap transfers with compute. Pageable memory forces a hidden staging copy and defeats overlap for that transfer.

Second, you should use the asynchronous allocation and deallocation routines, cudaMallocAsync() and cudaFreeAsync(), instead of the synchronous and blocking cudaMalloc() or cudaFree() calls. Frameworks like PyTorch provide the option to use CUDA’s asynchronous, stream-ordered memory allocator. The idea is to not stall all active streams when you allocate memory. This would be very bad for performance.

The asynchronous stream-ordered allocator allows each stream to request or return device memory without waiting on other streams. Using the asynchronous allocator ensures memory operations in one stream don’t stall operations in other streams. This will avoid unnecessary global synchronization. Let’s explore the stream-ordered memory allocator in the next section.

PyTorch’s default CUDA caching allocator is stream-aware and, in normal operation (e.g., servicing allocations from its cache), it avoids device-wide synchronization. Only when it has to request more memory from the OS using cudaMalloc would a synchronization occur. In practice this means most tensor allocations and frees don’t block other streams. Enabling the cudaMallocAsync backend can further reduce fragmentation and improve reuse in many workloads, as you’ll see next.

## Stream-Ordered Memory Allocator

In PyTorch, you can enable CUDA’s stream-ordered allocator by setting the environment variable PYTORCH_ALLOC_CONF=backend:cudaMallocAsync before launching your PyTorch script. If this variable is set, PyTorch tensor memory allocation (cudaMallocAsync()) and free (cudaFreeAsync()) operations are enqueued in separate CUDA streams in the order in which they are invoked. When this environment variant is not set, PyTorch uses its own caching allocator.

If you use the legacy cudaMalloc(...), remember that it’s a blocking, device-wide operation that synchronizes the device before returning. This can stall work in other streams since every allocation forces the entire GPU to stall until the memory is reserved. This pauses all streams, limits parallelism, and destroys your workload’s performance.

In contrast, using the stream-ordered allocator with cudaMallocAsync(...) simply records the allocation request in the same CUDA stream that will use it—whether the stream is performing a kernel or memory operation. It will not block the other streams. This way, memory management never serializes streams that are feeding those kernels.

> CUDA’s stream-ordered allocator, used in PyTorch, avoids global device locks and reduces allocation overhead.

In practice, stream 0 might be executing an attention kernel on batch N, stream 1 copies batch N+1 from host to device, and stream 2 enqueues a cudaMallocAsync(...) for batch N+2. Because cudaMallocAsync(...) simply appends its work into stream 2’s queue, streams 0 and 1 continue without interruption.

With the stream-aware allocator, GPU memory for each mini-batch is allocated without blocking other streams. This is important for LLM pipelines that allocate per-batch scratch space. The asynchronous allocator prevents stalls—even under heavy memory churn.

> Using the stream-ordered memory allocator is particularly important if your pipeline allocates a scratch buffer for each mini-batch—common in LLM training and inference. For instance, each minibatch in an LLM pipeline needs its own temporary workspace to hold attention keys/values or intermediate activation buffers. In this case, you often call an allocator to reserve that “scratch buffer” on the GPU.

Allocations are satisfied from a per-device memory pool. You can tune the pool’s release threshold using cudaMemPoolSetAttribute() to trade off returning memory to the OS versus reusing it for performance. A higher threshold means the pool will keep memory allocated longer. This will reduce the number of times the memory is returned back to the OS. This leads to fewer OS calls and better performance by avoiding repetitive memory allocations and de-allocations.

The following example shows how to implement stream-based overlap using the stream-ordered memory allocator with cudaMallocAsync and cudaFreeAsync and demonstrates the use of cudaMemPoolSetAttribute(). This highlights how memory allocation, data transfer, and kernel execution can be fully pipelined using CUDA streams:

```
// initialize the async memory allocator
cudaMemPool_t pool;
int device = -1;
cudaGetDevice(&device); // Current device
cudaDeviceGetDefaultMemPool(&pool, device);
// Desired number of bytes to keep in pool before
// releasing back to the OS (tune as needed)
uint64_t threshold =/* e.g., prop.totalGlobalMem / 2 */; // bytes
cudaMemPoolSetAttribute(pool,
  cudaMemPoolAttrReleaseThreshold, &threshold);
cudaStream_t stream1, stream2;
cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&stream2, cudaStreamNonBlocking);
// Allocate memory using stream-ordered async allocation
void *d_data1, *d_result1;
void *d_data2, *d_result2;
size_t dataSizeBytes = N * sizeof(float);
// Use cudaMallocAsync as a best practice in modern multi-stream apps
cudaMallocAsync(&d_data1, dataSizeBytes, stream1);
cudaMallocAsync(&d_result1, dataSizeBytes, stream1);
cudaMallocAsync(&d_data2, dataSizeBytes, stream2);
cudaMallocAsync(&d_result2, dataSizeBytes, stream2);
// Asynchronously copy first chunk and launch its kernel in stream1
cudaMemcpyAsync(d_data1, h_data1, dataSizeBytes,
  cudaMemcpyHostToDevice, stream1);
computeKernel<<<gridDim, blockDim, 0,
  stream1>>>((float*)d_data1, (float*)d_result1);
cudaMemcpyAsync(h_result1, d_result1, dataSizeBytes,
  cudaMemcpyDeviceToHost, stream1);
// In parallel, do the same on stream2
cudaMemcpyAsync(d_data2, h_data2, dataSizeBytes,
                cudaMemcpyHostToDevice, stream2);
computeKernel<<<gridDim, blockDim, 0,
  stream2>>>((float*)d_data2, (float*)d_result2);
cudaMemcpyAsync(h_result2, d_result2, dataSizeBytes,
   cudaMemcpyDeviceToHost, stream2);
// Wait for both streams to finish
cudaStreamSynchronize(stream1);
cudaStreamSynchronize(stream2);
// Cleanup
cudaFreeAsync(d_data1, stream1);
cudaFreeAsync(d_result1, stream1);
cudaFreeAsync(d_data2, stream2);
cudaFreeAsync(d_result2, stream2);
cudaStreamDestroy(stream1);
cudaStreamDestroy(stream2);
```

Here, we create two CUDA streams (stream1 and stream2) and allocate device memory using cudaMallocAsync, ensuring that each stream has its own stream-ordered memory buffers. We then issue work for two independent data chunks.

On stream1, we perform an asynchronous copy from host to device (H2D), launch a compute kernel, and then asynchronously copy the results back from device to host (D2H). Simultaneously, we do the same for a second chunk of data on stream2.

Because these operations are issued on separate streams, the GPU device overlaps work between them. stream1 executes the kernel concurrently with the H2D copy on stream2. Once the stream1 kernel completes, it can copy the data back to the host (D2H) overlapping with stream2’s kernel execution.

Here, the memory allocations overlap with kernel computations thanks to CUDA streams and stream-ordered memory allocations. The staggered scheduling shown in this example reduces idle time and maximizes throughput. Without stream-ordered allocation, you’d either have to allocate all the memory upfront—increasing memory footprint—or incur heavy synchronization penalties.

With cudaMallocAsync, memory management is seamlessly integrated into CUDA streams. This allows per-stream allocations and deallocations without triggering a global device synchronization.

In addition, the stream-ordered allocator lets you issue fine-grained memory requests for variable-length buffers—such as token caches or intermediate activations. You can then immediately launch kernels that depend on those buffers. This happens all within the same stream.

> In practice, achieving peak throughput requires carefully tuning data-chunk sizes and staying within your GPU’s concurrency limits. Modern GPU devices provide multiple copy engines and can overlap host-to-device (H2D) and device-to-host (D2H) transfers. Query deviceProp.asyncEngineCount to determine how many copy engines your device supports to plan overlap accordingly.

Modern GPUs have a hard limit for the number of concurrent kernels that can run across all SMs on a device (up to the 128 resident-grid limit.) As discussed in Chapter 5, the modern GPU limit is 128 concurrently executing kernels per device. Once you exceed the limit of active kernels, additional kernel launches will queue until a slot frees up on one of the SMs.

And remember that kernels that share an SM will only execute together if their combined registers, shared memory, and thread block requirements fit within the SM’s resource limits. Balancing chunk (tile) sizes, launch order, and per-kernel resource usage is essential.

If chunks are too small, you will underutilize the copy engines and SM resources. If chunks are too large or too many kernels are enqueued simultaneously, you will exceed kernel slots or exhaust per-SM resources. This will lead to stalls.

In short, when tuned correctly, however, CUDA streams, combined with the stream-ordered memory allocator (cudaMallocAsync), will ensure that data transfers, kernel execution, and memory management will overlap seamlessly. This keeps the multiple DMA engines and SMs busy without unnecessary queuing.

## Using CUDA Streams and Stream-Ordered Memory Allocator with LLMs

The nonblocking behavior of CUDA streams combined with the stream-ordered memory allocator is crucial for LLM training and inference workloads. These workloads overlap computation and data movement across multiple streams to increase GPU utilization and reduce end-to-end latency.

In addition, LLMs utilize on-the-fly “scratch memory” allocations, which are facilitated by the stream-ordered memory allocator discussed in the previous section. For instance, when running a transformer layer, you often need extra shared memory or device memory, called *scratch memory*, to store the results of a matrix multiply before feeding it into the softmax operation.

Because different mini-batches in LLM workloads can vary in length (token count), you will want to use the stream-ordered memory allocator to provide a fresh scratch buffer on the GPU specifically for each input batch. This way, you have exactly enough space allocated for that batch’s intermediate computations—and not a single byte more.

If you use the old, blocking allocation API (cudaMalloc(...) and cudaFree(...)) to allocate these scratch buffers on the fly, every single allocation or deallocation would synchronize with the entire GPU since calling cudaMalloc(...) forces a global device synchronization. As such, no overlap is possible until all pending kernels and copies finish.

> Global device synchronizations are absolutely disastrous for performance. Avoid using blocking calls like cudaMalloc() and cuda Free() in your CUDA streams. Prefer events and stream waits. And definitely avoid synchronizing on the default stream 0 with cudaStreamSynchronize(0)!

In a pipeline where one stream is busy running an attention kernel for batch *N* and another stream is preparing batch N+1 for the attention kernel, calling a blocking cudaMalloc(...) on the second stream will stall all streams. Until the allocator finishes, every SM is effectively paused. This can wipe out any overlap you hoped to achieve between data transfers, computation, and memory management.

The solution is to use the stream-ordered allocator with cudaMallocAsync() and cudaFreeAsync(). These APIs enqueue the work of allocating and freeing regions of device memory at the stream level. As such, they synchronize only at the stream level—and not the device level.

For instance, consider a stream that needs a scratch buffer of 16 MB for attention on a batch of input data. It would invoke cudaMallocAsync(&scratchPtr, scratch Bytes, stream1), which records this allocation request in its operation queue but does not force any other stream to wait, as shown in Figure 11-3.

![Figure 11-3. Stream-ordered memory allocation](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-3.png)

The other streams continue launching kernels, copying data, or doing whatever they were doing—even while stream 1’s allocation is in flight. And once the CUDA runtime has reserved the memory behind the scenes, stream 1 can make progress again and launch the attention kernel into that newly allocated region—all without halting any other streams.

> Unlike legacy cudaMalloc, cudaMallocAsync does not stall other streams. Each allocation is synchronized only within its own stream.

In the context of LLM training and inference, this is particularly valuable because variable-length sequences often produce scratch-buffer size fluctuations. If batch *N* has 512 tokens per sequence and batch N+1 has 1,024 tokens, your attention module will need more space for batch N+1 than batch *N*, so reusing batch *N*’s allocation is not sufficient. With cudaMallocAsync(), you can enqueue a single, nonblocking allocation for the larger buffer without dragging all other streams to a stop.

Additionally, a typical LLM’s autoregressive token-by-token generation (aka *decoding*) phase uses a growing key/value cache. Each generated token requires appending new KV pairs to a per-sequence buffer. As the buffer grows, you need to reallocate or extend the scratch region. cudaMallocAsync(...) lets you do this in the same stream that runs the attention kernel. Meanwhile, upstream data-loading and downstream result-copying operations continue making progress in parallel, running in their own streams.

Another use of CUDA streams in an LLM context is layerwise pipelining in large LLMs. Suppose that you divide a large, transformer-based LLM model into two halves that run on different CUDA streams—stream 0 runs layers 0–5 and stream 1 runs layers 6–11. Between these halves, you need intermediate buffers for activations.

Each time stream 0 finishes its work on a mini-batch, it might call cudaMallocAsync(...) to grab a new buffer for the next batch’s activations. Because that call does not synchronize the device, stream 1 can continue computing layers 6–11 on the previous batch’s results while stream 0 allocates memory for the next batch’s inputs.

By contrast, if you had used the legacy cudaMalloc(...) inside that pipeline, every time you allocated a new scratch region for the next mini-batch or an expanded KV cache, the entire GPU would pause until the allocation completed. That would break any chance of overlapping computation and data movement across streams.

To summarize, in an LLM context, you frequently need temporary buffers for attention, layer normalization, softmax, KV cache, or intermediate activations. These are collectively referred to as a *scratch buffer*.

Using cudaMallocAsync(...) and cudaFreeAsync(...) to manage these scratch buffers within separate streams ensures that memory management never forces a global, cross-stream stall. Instead, the allocation enqueues into the same stream as your kernel or copy operation.

This allows all other streams to continue running and keeps your attention kernels, data transfers, and any host-side work overlapping as much as possible. This maximizes GPU utilization in large-scale, real-time LLM workloads.

## Legacy Default Stream

When you do not explicitly create or specify a stream, the operations go into the legacy default stream, often called *stream 0*. By default, stream 0 has two important behaviors that are worth highlighting:

*Implicit synchronization with itself* Any two operations enqueued into stream 0 execute strictly one after the other. You cannot overlap two kernels or a copy and a kernel in stream 0, because stream 0 serializes all of its own commands.

*Implicit synchronization with other streams* In the legacy default stream model, any operation launched into stream 0 will wait for all previously enqueued work in every other stream to finish before it begins. Conversely, any operation launched into a nondefault stream will also block until all prior work in stream 0 has completed. In effect, stream 0 acts as a global “barrier” across the entire GPU. Even if you issue commands into different streams, once you submit something into stream 0, it forces every other stream to stall until stream 0 is caught up, and vice versa. This is very bad for performance and should be avoided when possible.

Because of these implicit dependencies, putting all work into stream 0 prevents any form of concurrency. For instance, kernels and copy engines cannot overlap. As such, your GPU spends cycles idle waiting for the default-stream barrier to clear.

To unlock true parallelism, you should avoid using stream 0 for anything but operations that truly need to serialize with every other stream, which is relatively rare.

## Modern Per-Thread Default Stream

To mitigate the “global barrier” behavior of the legacy default stream, CUDA introduced per-thread default streams, sometimes abbreviated PTDS (as opposed to the posttraumatic stress disorder (PTSD) that the legacy stream has given us throughout the years).

Under per-thread default stream semantics, each CPU thread’s default stream is independent. In other words, when per-thread default streams are enabled, each host thread has its own implicit “stream 0.”

Operations enqueued into thread A’s default stream do not wait for work in thread B’s default stream. They run concurrently whenever hardware resources allow. Likewise, operations in thread B’s default stream do not wait for thread A’s default stream, and so on.

PTDS is widely used in multithreaded CUDA applications to avoid the “host-wide barrier” issue. To enable PTDS, you can compile your code using nvcc --default-stream per-thread or set the CUDA_API_PER_THREAD_DEFAULT_STREAM=1 environment variable (before including any CUDA headers).

> Once PTDS is active, each host CPU thread’s default stream behaves like a user-created stream that does not implicitly synchronize with other threads’ default streams. If you mix PTDS with the legacy default stream in the same process, PTDS streams still synchronize with the legacy default stream.

With PTDS, any kernel launch, copy, or allocation without an explicit stream parameter goes into a thread-local queue. Only commands within the same host thread’s default stream serialize, and they never impose an implicit global barrier on streams belonging to other threads.

In short, by enabling per-thread default streams, the legacy default-stream synchronization barrier is removed. Each host thread’s default stream never waits on other threads’ streams. This allows full overlap of multiple kernel launches across threads.

And if you issue kernel launches (or memory copies) from different CPU threads without specifying an explicit stream, the operations will overlap on the GPU whenever resources permit. This is shown in Figure 11-4.

![Figure 11-4. Timeline showing multiple GPU kernels running concurrently across separate CUDA streams issued from different threads on their respective default streams with PTDS enabled](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-4.png)

## Default Versus Explicit (Nondefault) Streams

Relying on default-stream behavior will eventually cause problems. It always does. Any work enqueued into the legacy default stream (stream 0) implicitly waits for—and blocks—every other stream, and vice versa.

In performance-critical code, it’s best to create and use your own nondefault, explicit, and named streams so that nothing accidentally goes into stream 0. If you accidentally use one kernel on stream 0—or copy data into it—you can stall every other active stream. Many libraries, such as cuBLAS, Thrust, etc., accept an explicit stream parameter. It’s recommended that you always create explicit streams and use those.

> In PyTorch, operations are scheduled on nondefault streams under the hood to avoid unintended synchronization. For example, PyTorch’s internal calls to cuDNN, cuBLAS, etc., use their own streams to avoid blocking the default stream 0. Also, PyTorch’s distributed backend launches NCCL communication operations on separate CUDA streams rather than the default stream. This lets it overlap gradient communication with compute, for instance. In addition, NCCL’s communication operations often run in a high-priority stream, as we’ll cover in a bit.

By managing your own streams—or using per-thread defaults—you retain control over concurrency. Here is an example of creating explicit, nondefault streams in CUDA C++ (we will show how to use streams in PyTorch in Chapters 13 and 14):

```
cudaStream_t streamA, streamB;
cudaStreamCreateWithFlags(&streamA, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&streamB, cudaStreamNonBlocking);
myKernel<<<grid, block, 0, streamA>>>(...);                 // streamA
cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice, streamB); // streamB
```

Here, streamA and streamB can overlap freely. Under the legacy default-stream model (PTDS disabled), however, any later call into stream 0 forces both streamA and streamB to wait until stream 0 is empty.

Similarly, any work enqueued in streamA or streamB will block if stream 0 still has pending tasks. To avoid these hidden global barriers, keep stream 0 idle and use it only for one-time operations for initial setup, final cleanup, etc.

In short, enable per-thread default streams so that each CPU thread’s default stream no longer synchronizes with any other thread’s default stream. Then create and use explicit streams (like streamA and streamB) for all performance-critical kernels and copies.

By doing both, nothing you enqueue into your explicit streams can accidentally collide with work in another explicit stream, another thread’s default stream, or the legacy default stream 0. This ensures safe, predictable overlap without implicit synchronization. Creating streams with cudaStreamNonBlocking ensures that they do not synchronize with the legacy default stream. This is required to avoid hidden barriers.

## Best Practices for Default Stream Usage

Because default streams can be problematic for performance, let’s highlight the synchronization characteristics of each type of stream—legacy default, per-thread default, and explicit (nondefault):

*Legacy default stream (*cudaStreamLegacy*)* This blocks and is blocked by every other stream. Do not issue work here if you need any form of concurrency.

*Per-thread default stream (*cudaStreamPerThread*)* Each host thread’s default stream is private. It still serializes its own commands but does not wait on or block any other thread’s default stream or explicit streams.