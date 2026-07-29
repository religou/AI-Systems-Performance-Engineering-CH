# Chapter 10. Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusters

In the previous chapters, we covered fundamental optimizations such as tuning memory access, maximizing parallelism, overlapping computation and data transfer, boosting occupancy, and minimizing warp stalls. These helped hide latency and eliminate bottlenecks. Modern GPUs, however, offer advanced hardware features and execution models that let us take the fundamental optimization techniques even further.

In this chapter, we introduce some more advanced CUDA techniques such as warp-specialized pipelines, cooperative groups with grid-level and cluster-level synchronization, persistent kernels that loop over dynamic work queues, and thread block clusters (aka *cooperative thread array cluster* [CTA]) that use distributed shared memory (DSMEM or DSM) and Tensor Memory Accelerator (TMA) multicast. At a high level, a thread block cluster is a group of thread blocks that are guaranteed to run concurrently. They can read, write, and perform atomics to each other’s shared memory using DSMEM.

These methods let us overlap memory accesses and compute operations without host intervention. We can also share data on-chip across thread blocks—and keep every SM fully utilized.

By understanding these modern GPU execution models, you’ll be ready to progress to the next chapter where we extend these optimizations even further by exploring inter-kernel pipelines with CUDA streams. The next chapter builds inter-kernel pipelines on the foundation of intra-kernel optimizations discussed throughout this chapter.

## Intra-Kernel Pipelining Techniques

*Intra-kernel pipelining* refers to a set of techniques that overlap memory operations and computations within a single kernel execution. (In the next chapter, we’ll explore inter-kernel pipelining, which overlaps work across multiple kernels running in different streams.)

The core idea is to structure a kernel into concurrent stages such that while one piece of data is being loaded or stored, previously loaded data is being processed. These stages operate in parallel over different tiles or data chunks. This improves throughput and efficiently hides latency.

Traditionally, GPUs rely on warp-level multithreading to hide latency. While one warp stalls on a memory load, other warps proceed with computation. This is the foundation of single instruction, multiple threads (SIMT) latency hiding in the execution model.

Intra-kernel pipelining pushes this further by overlapping memory and compute within the same warp or kernel. It uses fine-grained coordination to stagger memory loads and compute—sometimes within a single warp.

Intra-kernel pipelining with the CUDA Pipeline API overlaps asynchronous memory transfers and computation without any __syncthreads(). The two common approaches to intra-kernel pipelining are double buffering and warp specialization.

In the double-buffered (two stages) pipeline approach, all threads cooperate uniformly. In the warp-specialized pipeline approach, warps are specialized into distinct roles like memory loader, compute, and memory storer. The choice depends on your workload and performance requirements. Table 10-1 summarizes these two <cuda/pipeline> variants.

Table 10-1. Two approaches for intra-kernel pipelining on modern GPUs using the CUDA Pipeline API

| API variant | Best for | Main use |
| --- | --- | --- |
| Double-buffered pipeline | Loop-based tiling and double buffering | Overlapping loads and compute in the same warp or block |
| Warp-specialized pipeline (e.g., three-stage memory loader, compute, memory storer) | Persistent kernels with multiple distinct warp roles (3 in our case) | Assigning warps to separate roles/stages such as memory load, compute, and memory store |

### Cooperative Tiling and Double-Buffering with the CUDA Pipeline API

You can implement the traditional double-buffered tiling pattern using the C++ Pipeline API by instantiating a two-stage pipeline to overlap memory loads and computations. Specifically, you can declare a two-stage cuda::pipeline_shared_state <cuda::thread_scope_block, 2> object, which is scoped to a specific thread block using cooperative groups (discussed in a bit). This is essentially a producer-consumer pattern, as shown in Figure 10-1.

![Figure 10-1. Two-stage producer-consumer pattern with the CUDA Pipeline API](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-1.png)

The key CUDA Pipeline API calls are shown next. They are followed by an implementation of a double-buffered, cooperative tiling kernel using this API to demonstrate modern CUDA techniques that align with hardware features:

pipe.producer_acquire() Reserves the next pipeline stage for writing

pipe.producer_commit() Signals that the previously issued asynchronous operations for this stage are ready for consumption

pipe.consumer_wait() Waits until the previously committed operations for this stage complete to avoid race conditions in loops

pipe.consumer_release() Releases the current stage so it can be reused

The two stages in the pipeline overlap global-memory loads with computations. In the first, Stage 0, one warp in the thread block issues an asynchronous prefetch for the next tile into shared memory. The prefetch issues cooperative cuda::memcpy_async copies that lower to per thread cp.async into shared memory.

While Stage 0 is producing (loading) the data in one warp, the remaining warps in the thread block are consuming (computing) the loaded data in the second stage, 1. This simple producer-consumer implementation hides DRAM latency with ongoing computations, as shown in Figure 10-2.

![Figure 10-2. Hiding global DRAM load (L) latency with (C) compute using a producer–consumer pipeline](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-2.png)

This pattern can raise SM utilization and reduce kernel time, but whether you are compute bound or memory bound depends on the operation, tile sizes, and overlap efficiency. Even highly optimized attention kernels like FlashAttention-3 report around 75% percent of peak FP16 FLOPs due to practical limits in overlap and data movement.

This is a two-stage, double-buffering example using the CUDA C++ Pipeline API. This API enables the fine-grained producer-consumer synchronization code used here:

```
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <algorithm>
namespace cg = cooperative_groups;
#ifndef TILE_SIZE
#define TILE_SIZE 32
#endif
#ifndef STAGES
#define STAGES 2          // 2 = double buffer, 3 = triple buffer, etc.
#endif
__device__  float computeTile(const float* __restrict__ A_sub,
                              const float* __restrict__ B_sub,
                              int tx, int ty) {
    float s = 0.0f;
    #pragma unroll
    for (int k = 0; k < TILE_SIZE; ++k) {
        s += A_sub[ty * TILE_SIZE + k] * B_sub[k * TILE_SIZE + tx];
    }
    return s;
}
extern "C" __global__
void gemm_tiled_pipeline(const float* __restrict__ A_global, // [M x K]
                         const float* __restrict__ B_global, // [K x N]
                         float* __restrict__ C_global,       // [M x N]
                         int M, int N, int K) {
    cg::thread_block cta = cg::this_thread_block();
    // Shared memory layout: A[STAGES] then B[STAGES]
    extern __shared__ float shared_mem[];
    float* A_buf[STAGES];
    float* B_buf[STAGES];
    {
        float* p = shared_mem;
        for (int s = 0; s < STAGES; ++s) {
            A_buf[s] = p;
            p += TILE_SIZE * TILE_SIZE;
       }
        for (int s = 0; s < STAGES; ++s) {
            B_buf[s] = p;
            p += TILE_SIZE * TILE_SIZE;
       }
    }
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, STAGES> state;
    auto pipe = cuda::make_pipeline(cta, &state);
    int tx = threadIdx.x, ty = threadIdx.y;
    int block_row = blockIdx.y * TILE_SIZE;
    int block_col = blockIdx.x * TILE_SIZE;
    float accum = 0.0f;
    int numTiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    // Prologue: load first STAGES tiles (or fewer if short)
    for (int s = 0; s < std::min(STAGES, numTiles); ++s) {
        int aRow = block_row + ty;
        int aCol = s * TILE_SIZE + tx;
        int bRow = s * TILE_SIZE + ty;
        int bCol = block_col + tx;
        pipe.producer_acquire();
        if (aRow < M && aCol < K) {
            cuda::memcpy_async(cta, A_buf[s] + ty*TILE_SIZE + tx,
                               &A_global[aRow*K + aCol],
                               cuda::aligned_size_t<32>(sizeof(float)), pipe);
        } else {
            A_buf[s][ty*TILE_SIZE + tx] = 0.0f;
        }
        if (bRow < K && bCol < N) {
            cuda::memcpy_async(cta, B_buf[s] + ty*TILE_SIZE + tx,
                               &B_global[bRow*N + bCol],
                               cuda::aligned_size_t<32>(sizeof(float)), pipe);
        } else {
            B_buf[s][ty*TILE_SIZE + tx] = 0.0f;
        }
        pipe.producer_commit();
    }
    // Steady state
    for (int tile = 0; tile < numTiles; ++tile) {
        int s = tile % STAGES;
        // Block-scope wait
        pipe.consumer_wait();
        accum += computeTile(A_buf[s], B_buf[s], tx, ty);
        pipe.consumer_release();
        // Prefetch next tile into the same slot s (ring buffer)
        int nextTile = tile + STAGES;
        if (nextTile < numTiles) {
            int aRow = block_row + ty;
            int aCol = nextTile * TILE_SIZE + tx;
            int bRow = nextTile * TILE_SIZE + ty;
            int bCol = block_col + tx;
            pipe.producer_acquire();
            if (aRow < M && aCol < K) {
                cuda::memcpy_async(cta, A_buf[s] + ty*TILE_SIZE + tx,
                                   &A_global[aRow*K + aCol],
                                   cuda::aligned_size_t<32>{sizeof(float)}, pipe);
            } else {
                A_buf[s][ty*TILE_SIZE + tx] = 0.0f;
            }
            if (bRow < K && bCol < N) {
                cuda::memcpy_async(cta, B_buf[s] + ty*TILE_SIZE + tx,
                                   &B_global[bRow*N + bCol],
                                   cuda::aligned_size_t<32>{sizeof(float)}, pipe);
            } else {
                B_buf[s][ty*TILE_SIZE + tx] = 0.0f;
            }
            pipe.producer_commit();
        }
    }
    // Epilogue: final store (guard tails)
    int cRow = block_row + ty;
    int cCol = block_col + tx;
    if (cRow < M && cCol < N) {
        C_global[cRow * N + cCol] = accum;
    }
}
```

The code first retrieves a handle to the current thread block using cooperative groups (CG), discussed in a bit. It then immediately instantiates a two-stage cuda::pipeline object bound to that block. By creating the pipeline before any asynchronous operations, the pipelines’ internal barriers and internal synchronization mechanisms are in place prior to performing data movement.

The kernel then allocates one contiguous shared-memory region for both A and B tiles by defining an extern __shared__ float shared_mem[]. It divides this buffer into four subregions, two for A and two for B, using pointer arithmetic (float* A_buf[2] and float* B_buf[2]). This allows true double-buffering without extra dynamic allocations.

Before entering the main loop, the kernel asynchronously prefetches the first STAGES tiles using asynchronous copies bound to the following pipeline: producer_acquire() → memcpy_async() → producer_commit(). Consumer warps use consumer_wait() and consumer_release() so the compute starts exactly when the prefetched tile is ready.

This initial barrier replaces a future need for __syncthreads() and ensures the pipeline’s stage 0 and stage 1 buffers are correctly populated for the first iteration of the producer-consumer sequence.

Within each iteration, the kernel reserves the next buffer to load subsequent tiles by calling pipe.producer_acquire(). It then launches two cuda::memcpy_async operations—one for the A tile and one for the B tile. Each load is bound to the pipeline object with cuda::memcpy_async(..., pipe), which issues asynchronous copies from global memory into shared memory.

These asynchronous copies can overlap well with compute when accesses are coalesced and tiled correctly. This way, the pipeline stays fed with data to perform maximum useful work. Immediately after queuing the memcpy_async calls, the kernel signals completion with pipe.producer_commit(). This commit records the copy’s arrival and allows consumer warps to wait on that specific stage without blocking the entire thread block.

Concurrently, other warps in the block invoke pipe.consumer_wait(). This efficiently stalls only those threads dependent on the data in the curr buffer until the producer has committed it. Once the wait completes, each thread calls the device function computeTile(...) to perform its TILE_SIZE x TILE_SIZE dot-product computation. The results are incrementally accumulated in registers.

After finishing the computation on the current buffer, the warps invoke pipe.consumer_release() to free that stage for reuse in subsequent iterations. This fine-grained release prevents block-wide stalls and maximizes overlap between compute and memory transfer phases.

At the end of each iteration, swap the curr and next, which cycles the two double-buffer stages. This way, the memory buffer that was just computed now becomes the target for the next asynchronous memory.

These producer and consumer stages are repeated for all K-dimension tiles. After processing the final tile, each thread’s accumulator contains the full TILE_SIZE x TILE_SIZE dot-product result. This result is then written back to the appropriate element of C_global.

When using double buffering, it’s recommended to make sure that each asynchronous copy (memcpy_async) is followed by the appropriate producer_commit() and consumer_wait() calls for synchronization. This guarantees that the compute kernels will use the data only after it’s properly loaded. Modern CUDA compilers will often perform these optimizations automatically for simple loops.

> It’s recommended to validate that the pipeline is executing as expected using Nsight Compute’s asynchronous copy metrics. In particular, pay attention to warp occupancy and shared-memory bank conflicts. The goal is to increase SM Active percent and reduce stall cycles. Full hiding of DRAM latency depends on occupancy, access patterns, and shared-memory bank behavior.

This simple double-buffered scheme is roughly 2× faster than naive tiling, as described in Table 10-2’s performance comparison between the naive tiling implementation and the optimized, double-buffered, pipeline implementation.

Table 10-2. Performance comparison between the naive tiling and double-buffered kernel using the CUDA Pipeline API

| Metric | Naive tiling | Two-stage, double-buffered pipeline (double_buffered_pipeline) |
| --- | --- | --- |
| Kernel execution time | 41.3 ms | 20.5 ms (~2× faster versus naive) |
| SM Active % | 68% | 92% (+24% versus naive) |

> Note: The numeric values in all metrics tables are illustrative to explain the concepts. For actual benchmark results on different GPU architectures, see the GitHub repository.

In this experiment, the gemm_tiled_pipeline kernel using the CUDA C++ Pipeline API achieves a 2× speedup over the naive tiling version. By using the fine-grained pipe.producer_commit() and pipe.consumer_wait() primitives, the pipeline remains filled, and SM Active % jumps 24% from 68% to 92%.

### Warp Specialization and the Producer-Consumer Model

Warp specialization extends double buffering by assigning operations to warps that use different hardware, such as data movement (e.g., TMA) and compute (e.g., Tensor Cores). This is in contrast to reusing the same warps for both loading data and computing, as shown in Figure 10-3.

![Figure 10-3. Nonwarp specialized kernel with each warp performing a mix of both data loading and compute (source: https://oreil.ly/WZDbM)](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-3.png)

This type of specialization allows each set of warps to have their own instruction sequences. As such, instructions are issued and executed continuously without being interrupted by other types of operations. Specifically, warp specialization lets you assign one set of “producer” or “memory” warps to prefetch tiles asynchronously using cuda::memcpy_async. Then all other “consumer” or “compute” warps perform the computations, as shown in Figure 10-4.

Here, four warps are assigned the producer role, while the remaining eight warps are assigned the consumer role. Like most producer-consumer patterns, you can assign a different number of warps for the producer and consumer.

Because each warp has its own scheduler, the GPU can issue a load instruction, a math instruction, and a write instruction—all in the same cycle from different warps using different warp schedulers. So an SM with multiple schedulers can issue a memory instruction from Warp 0, a math instruction from Warp 1, and so on, in one cycle.

![Figure 10-4. Warp-specialized kernel with one set of warps for loading data and all other warps for computations (source: https://oreil.ly/WZDbM)](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-4.png)

This effectively creates a thread-block-level multi-issue scenario across warps. This is not possible with single-warp double buffering because a single warp’s scheduler can issue only one instruction per cycle.

An interesting pattern for warp specialization is using three different types of warps, such as “loader,” “compute,” and “storer” warps. The loader warp pushes tiles into the pipeline’s queue. The compute warp runs the compute kernel on each tile. And the storer warp writes out the results, as shown in Figure 10-5.

This warp-specialized pipeline squeezes out the idle cycles that a single-warp, double-buffered, sequential load-and-compute loop cannot address. Warp specialization’s efficient overlap of data transfer and computation increases GPU utilization—especially for long-running loops and persistent kernels. In these cases, the overhead of role coordination and data handoff is amortized over many iterations.

A paper on warp scheduling demonstrated that warp specialization can achieve nearly perfect overlap of memory and compute. In this case, the GPU kernel had distinct memory and compute phases such that memory and compute took turns being the bottleneck. By applying warp specialization, their workload transformed into a state in which both the SM’s memory subsystem and compute units were simultaneously busy almost the entire time.

![Figure 10-5. Three-role warp-specialized pipeline configuration with one set of warps for loading data, another set for compute, and another set for data storing (source: https://oreil.ly/xs7YN)](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-5.png)

The profiling showed that before using warp specialization, their L2 bandwidth utilization and Tensor Core utilization were out of phase. After warp specialization, L2 bandwidth and Tensor Core utilization became in-phase. This resulted in much higher effective throughput and showed that warp specialization can squeeze out the last bits of idle time—even for a well-tuned asynchronous pipeline that might be left on the table.

Another warp specialization pattern is a modification of the three-role warp-specialized pipeline. It assigns a set of warps to the memory loader as before, but then uses two sets of consumer warps that “ping-pong” between the roles of compute and memory-storer. This three-role warp-specialization architecture is exposed in CUTLASS as gemm_tma_warp-specialized_ping-pong and shown in Figure 10-6.

![Figure 10-6. Ping-pong architecture with three-role warp-specialized kernel (source: https://oreil.ly/xs7YN)](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-6.png)

Here, the consumer MMAs are overlapping and include a small amount of post-MMA wrap-up, or *epilogue,* processing. This per-MMA epilogue cleanup is required before launching the next MMA. Specifically, the epilogue can include accumulating, scaling, writing back to global memory, or shuffling results to another warp. Additionally, the epilogue can perform housekeeping like advancing tile pointers, updating loop counters, and signaling that this tile is done so the next TMA load or MMA can kick off.

> While not shown here, there is also an equivalent prologue processing step that happens before the MMA operation. In this case, the prologue phase would fill the pipeline with a few TMA requests to move data into registers before the MMA consumers can start doing useful work on the Tensor Cores.

In practice, warp specialization is extremely effective at squeezing out the last bits of performance. In fact, FlashAttention v3 attributes its speedups partially to warp-specialized pipelines that overlap GEMM and softmax computations—along with data transfers—to keep all hardware units busy. This helps achieve near-peak FLOPS for attention computations due to aggressive overlap of compute and TMA-driven data movement.

In addition, the PyTorch compiler (covered in Chapters 13 and 14) generates kernels that use warp specialization to schedule separate warps for loading and computing data. It also uses low-cost barriers to synchronize the warps, similar to the CUDA Pipeline API implementation detailed in the next section. The PyTorch compiler system also integrates with CUTLASS’s ping-pong GEMM. Both torch.compile and Triton may generate warp specialized kernels for supported operations. However, they apply warp specialization selectively based on heuristics and do not enable warp specialization for every operator.

> Use warp specialization for imbalanced or latency-hiding scenarios—especially when a single warp’s compute is not enough to hide memory-load latency. However, if a kernel is small—or extremely memory bound—sticking to a simpler double-buffering scheme may produce similar benefits without the extra code complexity.

### Using CUDA Pipeline API for Warp Specialization

Warp specialization builds on the CUDA Pipeline API by allowing specialized warps to communicate using fine-grained producer and consumer primitives. These calls avoid full block barriers while composing naturally with asynchronous copies such as cuda::memcpy_async.

The key advantage of using the CUDA Pipeline API’s producer and consumer calls (e.g., pipe.producer_acquire(), pipe.producer_commit(), pipe.consumer_wait(), and pipe.consumer_release()) is that they synchronize only the specific warps or stages that actually need to hand off data. This is in contrast to forcing every thread in a block to wait.

A block-wide barrier would stall every warp—even those that are not involved with the producer-consumer pipeline. All execution in that block must pause until every thread reaches the barrier, as shown in Figure 10-7.

By comparison, the Pipeline API maintains per-stage state internally. When a producer warp finishes its asynchronous copy and calls pipe.producer_commit, only the warps that call pipe.consumer_wait will block until the data is ready. Other warps in the block can continue running any work that does not depend on that stage. In practice, the CUDA Pipeline API reduces idle time and decreases stalled warps because it eliminates the need to pause the entire block with a barrier. With pipelines you coordinate producer and consumer handoffs at a finer granularity than a hand-coded async-copy sequence (e.g., PTX cp.async + __syncthreads()).

You can implement warp specialization with the CUDA Pipeline API in a three-role pattern. A loader warp produces inputs for a compute warp, the compute warp consumes those inputs and produces results, and a storer warp consumes those results and writes them out. The pipeline object is block scoped, and it tracks the stage order internally.

![Figure 10-7. Block-wide barrier prevents threads from proceeding until they synchronize and load the new data](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-7.png)

Here is an example of a warp-specialized, three-role kernel that computes tiles of data using a loader warp (warp_id 0), compute warp (warp_id 1), and storer warp (warp_id 2):

```
// warp specialized roles using the CUDA Pipeline API
#include <cuda/pipeline>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
// smem_bytes = 3×TILE_SIZE^2×sizeof(float) (three roles, single-pipeline)
// 3 tiles * 112 * 112 * 4 byte per float =
// 150,528 bytes < 227,328 bytes (~227 KB)
// per-block dynamic SMEM limit on Blackwell
// Choose TILE_* so total shared memory per block is ≤ the device’s
// reported per-block shared memory limit.
// Query cudaDevAttrMaxSharedMemoryPerBlockOption and set
// cudaFuncAttributeMaxDynamicSharedMemorySize accordingly.
#define TILE_SIZE 112
// Example tile compute: C = A x B for one TILE_SIZE by TILE_SIZE tile
__device__ void compute_full_tile(const float* __restrict__ A_tile,
                                  const float* __restrict__ B_tile,
                                  float* __restrict__ C_tile,
                                  int lane_id) {
    for (int idx = lane_id; idx < TILE_SIZE * TILE_SIZE; idx += warpSize) {
        int row = idx / TILE_SIZE;
        int col = idx % TILE_SIZE;
        float acc = 0.0f;
        #pragma unroll
        for (int k = 0; k < TILE_SIZE; ++k) {
            acc += A_tile[row * TILE_SIZE + k] * B_tile[k * TILE_SIZE + col];
        }
        C_tile[idx] = acc;
    }
}
extern "C"
__global__ void warp_specialized_pipeline_kernel(
        const float* __restrict__ A_global,
        const float* __restrict__ B_global,
        float* __restrict__ C_global,
        int numTiles) {
    thread_block cta = this_thread_block();
    // three square tiles in dynamic shared memory: A, B, and C
    extern __shared__ float shared_mem[];
    float* A_tile = shared_mem;
    float* B_tile = A_tile + TILE_SIZE * TILE_SIZE;
    float* C_tile = B_tile + TILE_SIZE * TILE_SIZE;
    // three stage pipeline shared by the block
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 3>
        pipe_state;
    auto pipe = cuda::make_pipeline(cta, &pipe_state);
    int warp_id  = threadIdx.x >> 5;
    int lane_id  = threadIdx.x & 31;
    int warps_per_block = blockDim.x >> 5;
    // grid wide warp indexing for persistent tiling
    int totalWarps = gridDim.x * warps_per_block;
    int global_warp = warp_id + blockIdx.x * warps_per_block;
    for (int tile = global_warp; tile < numTiles; tile += totalWarps) {
        size_t offset = static_cast<size_t>(tile) * TILE_SIZE * TILE_SIZE;
        if (warp_id == 0) {
            // loader produces A_tile and B_tile
            pipe.producer_acquire();

            auto bytes = TILE_SIZE * TILE_SIZE * sizeof(float);
            cuda::memcpy_async(cta, A_tile, A_global + offset,
                               cuda::aligned_size_t<32>{bytes}, pipe);
            cuda::memcpy_async(cta, B_tile, B_global + offset,
                               cuda::aligned_size_t<32>{bytes}, pipe);
            pipe.producer_commit();
        }
        if (warp_id == 1) {
            // wait for A_tile/B_tile
            pipe.consumer_wait();
            // compute C_tile from A_tile, B_tile
            compute_full_tile(A_tile, B_tile, C_tile, lane_id);
            // finished consuming A/B; free that stage
            pipe.consumer_release();
            // publish C_tile for the storer warp
            pipe.producer_acquire();
            pipe.producer_commit();
        }
        if (warp_id == 2) {
            // store consumes computed C_tile
            // Wait for the committed stage before consuming
            pipe.consumer_wait();
            for (int idx=lane_id; idx<TILE_SIZE*TILE_SIZE; idx+=warpSize) {
                C_global[offset + idx] = C_tile[idx];
            }
            pipe.consumer_release();
        }
    }
    // launch with dynamic shared memory size equal to
    // 3 * TILE_SIZE * TILE_SIZE * sizeof(float)
    // When launching with dynamic shared memory >48 KB
    // You need to set cudaFuncAttributeMaxDynamicSharedMemorySize
    // for the kernel
}
```

Here each warp works in a distinct role. The loader warp calls producer_acquire(), performs two cooperative copies with cuda::memcpy_async into A_tile and B_tile, then calls producer_commit(). The compute warp calls pipe.consumer_wait() to observe the newly committed data and immediately calls consumer_release() to free that stage for reuse.

The compute warp then becomes a producer for the next handoff by calling producer_acquire(), computing C_tile, and calling producer_commit(). The storer warp calls consumer_wait() to observe the computed C_tile and writes it to global memory in a warp-striped loop, then calls consumer_release(). This sequence uses a single block-scoped pipeline with no explicit stage numbers, and it avoids any block-wide __syncthreads.

> This kernel runs as a persistent kernel across many tiles to amortize launch overhead. More on persistent kernels in a bit.

In short, using the CUDA Pipeline API together with cooperative groups allows fine-grained, SM-wide producer-consumer handoffs without any explicit __sync threads() calls. Table 10-3 compares three implementations: a naive tiled kernel, a two-stage double-buffered GEMM using double_buffered_pipeline, and our warp-specialized pipeline kernel warp_specialized_pipeline.

Table 10-3. Comparison of three implementations: a naive tiled kernel, a two-stage double-buffered GEMM, and a warp-specialized pipeline kernel

| Metric | Naive tiling | Two-stage, double-buffered pipeline (double_buffered_pipeline) | Warp-specialized pipeline (warp_specialized_pipeline) |
| --- | --- | --- | --- |
| Kernel execution time | 41.3 ms | 20.5 ms (2.01× faster versus naive) | 18.4 ms (10.2% speedup versus two-stage) |
| Warp execution efficiency | 68% | 92% (+24% versus naive) | 96% (+4% versus two-stage) |
| Shared memory stall latency/warp sync stalls | High | Low | Minimal |
| L2 throughput | 80 GB/s | 155 GB/s (+94% versus naive) | 165 GB/s (+6.45% versus two-stage) |
| Throughput scalability | Scales up to only 2–3 warps per SM | Scales well up to ~6 warps per SM | Scales nearly linearly to the SM’s warps (e.g., 64 warps) |
| DRAM bytes read versus SM cycles | Poor overlap | Great overlap | Excellent overlap |
| Instruction count | 1.7 B | 1.05 B (–38% versus naive) | ~1.00 B (–5% versus two-stage) |

Here, the double-buffered pipeline finishes GEMM in 20.5 ms, whereas the warp-specialized version completes in just 18.4 ms. This 10.2% improvement is a result of the warp-specialized kernel only stalling the consumer warp (e.g., compute). The other warps (e.g., loader and storer) progress independently. In the previous two-stage, double-buffered kernel, every thread participates in the consumer phase. As such, consumer_wait() effectively stalls the entire thread block.

This finer-grained, per-warp synchronization eliminates the implicit full-block wait and allows all three warps (loader, compute, and storer) to overlap continuously. As a result, average SM utilization rises from roughly 92% in the double-buffered design to about 96% in the warp-specialized version—and warp-stall cycles drop to near-zero.

From a scalability standpoint, Nsight Compute shows that the naive tiling kernel saturates after just two to three active warps per SM. This is because each tile load must complete before any computation can start.

The two-stage, double-buffered kernel improves on this by overlapping loads and compute. This implementation scales up to 6 warps per SM before shared-memory or register limits become an issue.

In contrast, the warp_specialized_pipeline scales almost linearly as long as you can assign additional warps for load, compute, and store. On Blackwell, for instance, you can keep up to the architectural limit of 64 resident warps per SM. (Actual residency depends on registers, shared memory, and block size.)

As Table 10-3 shows, both the double-buffered and warp-specialized approaches substantially outperform the naive tiled kernel. The double_buffer_pipeline halves the runtime by overlapping tile loads and computation, while the warp_specialized_pipeline adds another 10.2% speedup by avoiding any implicit block-wide waits. Only the dedicated “compute” warp ever stalls.

Instruction counts drop from 1.7 billion in the naive version to 1.05 billion in the two-stage pipeline, a 38% reduction, and further to ~1.00 billion in the warp-specialized kernel for an additional 4.76% reduction.

L2 load throughput climbs from 80 GB/s in naive tiling to 155 GB/s in the two-stage approach (+94%) and then to 165 GB/s in the warp-specialized kernel (+6.45% versus two-stage). This is because this warp-specialized kernel dedicates one warp to loading each tile into shared memory once—and then multicasts that single copy to all compute lanes. This eliminates any remaining redundant L2 reads. As such, after tiling and double-buffering, nearly all redundancy is already removed from the pipeline.