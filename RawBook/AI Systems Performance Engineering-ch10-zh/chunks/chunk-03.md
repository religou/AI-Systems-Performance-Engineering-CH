Instead of falling back to global loads and stores, you launch a thread block cluster whose blocks each work on a portion of the data, then synchronize and share using DSMEM. All communication happens at SM-to-SM speed, which avoids expensive HBM traffic.

In short, DSMEM is a faster, more structured alternative to using global memory for block-to-block communication. But its scope is limited to the blocks in the same cluster. DSMEM is ideal for persistent kernels that require frequent, low-latency inter-block coordination, attention mechanisms, or other multistage algorithms in LLMs in which every block must exchange intermediate state before proceeding—as well as any workload that benefits from keeping interim results on-chip rather than writing back to and reloading from global memory.

> The portable maximum cluster size is 8 blocks. Some GPUs support larger non-portable cluster sizes (e.g., up to 16 CTAs) when explicitly enabled with the cudaFuncAttributeNonPortableClusterSizeAllowed attribute. This increases your DSMEM footprint at the cost of occupying more SMs in the cluster.

### Scratch Memory

Scratch memory is the low-level hardware infrastructure that underpins DSMEM and thread block cluster synchronization, whereas DSMEM is the shared data buffer that is implicitly visible to your CUDA code, scratch memory is a separate on-chip SRAM region used by the GPU to track coordination metadata such as barrier state, group progress, and access flags for DSMEM.

You do not access scratch memory directly from your kernels. Instead, the GPU manages it automatically. When you launch a thread block cluster, the hardware allocates a portion of scratch memory to maintain cluster-wide barrier counters (cluster .sync() state), track which blocks have arrived at DSMEM operations or synchronization points, and coordinate safe access to the distributed shared regions across SMs.

Because scratch memory is optimized for these metadata operations, it enables cluster-level barriers and DSMEM accesses to complete very quickly. If the metadata size exceeds available scratch SRAM for very large clusters or complex synchronization patterns, the GPU transparently spills some state into local memory, which is backed by L1/L2 caches. This spill ensures correctness while still preserving as much on-chip efficiency as possible.

In short, DSMEM is the abstraction you use to share data between blocks and scratch memory in the behind-the-scenes facility that makes those DSMEM operations fast and scalable. Together, they allow a thread block cluster to behave like a single logical unit that breaks the traditional barrier between thread blocks. This improves performance for workloads that need tightly coordinated parallelism across multiple thread blocks.

### Launching a Thread Block Cluster

Let’s illustrate how to use thread block clusters in practice. First, we need to launch a kernel with a cluster dimension that we haven’t used before.

In CUDA, this is done with an extended launch API that allows specifying the cluster size. For example, if we want clusters of two blocks, also called a *thread block pair*, or *CTA pair*, we can configure a special launch attribute, as shown here:

```
// Host code: launch a kernel with thread block clusters of size 2
cudaLaunchConfig_t config{};
config.gridDim = dim3(128, 1, 1);
config.blockDim = dim3(256, 1, 1);
config.dynamicSmemBytes = 0;
cudaLaunchAttribute attr{};
attr.id = cudaLaunchAttributeClusterDimension;
attr.val.clusterDim.x = 2;
attr.val.clusterDim.y = 1;
attr.val.clusterDim.z = 1;
config.attrs = &attr;
config.numAttrs = 1;
// Allow non-portable cluster sizes if you intend to use > 8 later
cudaFuncSetAttribute(MyClusterKernel,
                     cudaFuncAttributeNonPortableClusterSizeAllowed,
                     1);
cudaLaunchKernelEx(&config, MyClusterKernel, args, nullptr);
```

In this host code, we use cudaLaunchKernelEx to launch MyClusterKernel with clusters of 2 thread blocks (so 64 clusters total since 128 blocks ÷ 2 per cluster). The clusterDim is set to 2, meaning each cluster will contain 2 thread blocks. This replaces the traditional <<<gridDim, blockDim>>> syntax for cases where we want cluster-cooperative kernels. Under the hood, the CUDA runtime ensures those paired blocks are scheduled such that they can communicate with DSMEM.

> Remember that each thread block cluster needs to physically fit into the SMs and resources that you specify. The portable maximum cluster dimension is 8 thread blocks. To launch 16 on supported Blackwell parts, you must enable the non-portable attribute and size blocks so the cluster stays resident on the SMs. Keep this in mind when sizing your thread block clusters.

### Coordinating Thread Block Clusters with Cooperative Groups API

To coordinate work across multiple thread blocks in a cluster, you first obtain a handle to that group of blocks using cooperative_groups::this_cluster(). This cluster handle lets you perform hardware-accelerated cluster-wide barriers—and directly access another block’s shared memory.

This all happens without leaving the kernel or resorting to global-memory flags. Here is an example kernel that sums a local value from each thread block into thread block 0’s shared memory:

```
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__global__ void MyClusterKernel(/* args */) {
    // 1. Form a cluster group for all thread blocks in this thread block cluster
    cg::cluster_group cluster = cg::this_cluster();
    // 2. Allocate the same extern shared buffer in each block
    extern __shared__ int shared_buffer[];
    // 3. Each block learns its rank and the total cluster size
    int clusterRank = cluster.block_rank();
    int clusterSize = cluster.num_blocks();
    // 4. Initialize this block’s portion of shared memory (a simple local sum)
    int localSum = threadIdx.x;
    shared_buffer[threadIdx.x] = localSum;
    // 5. Barrier across all blocks in the cluster; no block proceeds until
    //    every block reaches this point and has written its shared_buffer.
    cluster.sync();
    // 6. Map a pointer to block 0’s shared_buffer so block can write into it
    // Pointers returned by cluster.map_shared_rank
    // refer to the remote CTA’s shared memory
    // and support remote atomics
    // and memory operations within
    // the cluster
    // Pair updates with cluster.sync at well defined point
    int* remote_buffer = cluster.map_shared_rank(shared_buffer, 0);
    if (clusterRank != 0) {
        // 7. Nonzero blocks atomically add their local shared_buffer[0] into
        //    rank 0’s buffer. This atomicAdd is routed on‐chip over DSMEM
        //    (not through DRAM.)
        atomicAdd(&remote_buffer[0], shared_buffer[0]);
    }
    // 8. Another cluster‐wide barrier ensures all atomic adds have completed
    cluster.sync();
    // 9. Finally, block 0 (rank 0, thread 0) can read the combined result
    if (clusterRank == 0 && threadIdx.x == 0) {
        printf("Combined sum in cluster[0]: %d\n", shared_buffer[0]);
    }
}
```

Here, we obtain a cluster handle by calling cg::this_cluster(). This returns a cluster_group object that represents exactly those blocks launched together as one thread block cluster. The runtime guarantees that every thread block in this cluster group is resident simultaneously—otherwise, the launch will fail.

A uniform shared memory allocation is implicitly provided. Each block in the cluster must declare the same size for extern __shared__ int shared_buffer[]. The DSMEM hardware will then logically combine each block’s shared memory into one virtual address space. And because all blocks reserve the same shared_buffer size, a pointer to the buffer of thread block 0, or rank 0, will coordinate properly across SMs.

> When using thread block clusters, make sure to organize each thread block’s DSMEM accesses just like global-memory coalescing. In other words, have each warp read or write contiguous, 32-byte-aligned sectors. That way, the hardware can route cluster-wide DSMEM transfers without bank conflicts or unexpected serialization. In practice, lay out your tile data so that the warp i in thread block j always touches a unique, aligned range. This will avoid memory bank contention and keep DSMEM data transfers performing at full speed.

In the previous example code, we also see that cluster.sync() is used as a cluster-wide barrier across thread blocks in the cluster. Unlike __syncthreads(), which only synchronizes threads within a single block, cluster.sync() synchronizes all threads in every block of this cluster.

This barrier is implemented in hardware and generally has lower latency than a grid-level synchronization, because it coordinates only the blocks in the cluster. This means you can synchronize blocks frequently with minimal impact—as long as they are within a cluster.

And there is no risk of deadlock caused by missing blocks since CUDA enforces that the grid launches properly and fits on the GPU. As such, there is no scenario in which some blocks reach the barrier and then hang forever waiting for “missing” blocks that never started. All blocks are already present, so the barrier completes cleanly once every block calls it.

In the previous code block, you see that cross-block shared memory access happens using map_shared_rank(). After the first barrier, every block’s shared_buffer[] is initialized. To get a pointer into block 0’s shared memory, the other blocks call int* remote_buffer = cluster.map_shared_rank(shared_buffer, 0), which specifies the thread block id (0) in the cluster.

The GPU hardware automatically translates that pointer so any load or store goes over the on-chip DSMEM network rather than out to global DRAM. As such, any write operation into remote_buffer—either an atomic or regular write—will update block 0’s shared memory directly on-chip. It’s worth highlighting that the performance characteristics of remote DSMEM accesses differ from local shared memory. Remote loads and stores benefit from coalesced, aligned 32-byte segments.

For instance, in the previous code block, when blocks 1...n need to add their local values into block ID 0’s shared buffer, they call atomicAdd(&remote_buffer[0], shared_buffer[0]). And because remote_buffer was obtained with cluster.map_shared_rank(), the atomicAdd goes directly into block 0’s SMEM over the on-chip DSMEM network. In other words, there is no round trip to DRAM or L2 cache. Every write happens at on-chip speeds.

In the previous code example, you see that once all blocks have performed their atom icAdd, a second cluster.sync() ensures that block 0 sees the complete sum in its own shared_buffer[0] before proceeding.

In short, cooperative_groups::this_cluster() plus cluster.sync() and cluster .map_shared_rank() give you a simple, efficient way to synchronize and share data across multiple thread blocks in a thread block cluster. All of these operations occur on-chip, avoid global-memory round trips, and enable fine-grained cooperation among blocks. This combination of thread block clusters and DSMEM provides much higher performance than any global-memory fallback or manual atomic-flag approach.

### Thread Block Pair

With modern NVIDIA, GPUs let you coschedule exactly two thread blocks in a cluster, or a thread block pair (aka *CTA pair*), across SMs within a single GPC. By grouping thread blocks in a cluster (e.g., a 2-block cluster, as shown in Figure 10-13), kernels that share data can use TMA to move tiles into each block’s shared memory.

![Figure 10-13. Thread block pair combines two thread blocks](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-13.png)

A single thread block might lack the registers or shared-memory capacity to process a very large tile (for example, a 256 × 256 matrix subtile) by itself. By pairing two thread blocks on nearby SMs within a GPC and using DSMEM, those two blocks can split the work on one large tile yet share data through a unified shared-memory region, as shown in Figure 10-14.

![Figure 10-14. Thread block pair (aka CTA pair) loading tiles as operands for an A * B matrix multiply (source: https://oreil.ly/kEZsv)](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-14.png)

Here, each thread block in the pair can load a fraction of the operand tile (e.g., 128 × 16) into its on-chip SMEM for the matrix multiply. In addition, each thread block holds part of the accumulator (e.g., 128 × 256) in Tensor Memory (TMEM). This allows the two thread blocks in the CTA pair to collaborate on a single tile, as shown in Figure 10-15.

![Figure 10-15. Thread block pair with Tensor Cores and TMEM](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-15.png)

Thread block pairs allow larger matrix multiplies that span two physical SMs. Using a pair of thread blocks effectively doubles the tile size since each SM handles half of the tile’s data.

The SM hardware shares operand data between the SMs using DSMEM. DSMEM reduces duplicate loads, improves data reuse, and increases arithmetic intensity. Figure 10-16 shows this data sharing between SMs in a thread block cluster using DSMEM.

![Figure 10-16. Sharing data between SMs in a thread block cluster using DSMEM](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-16.png)

CUDA provides enhanced multicast barrier and synchronization primitives for these multi-SM operations. The pair synchronizes with each other using the lightweight cluster.sync() call. This way, when one SM finishes using a tile, the adjacent SM in the CTA pair can consume it.

Each tile is fed into the pair of thread blocks without redundant global-memory transactions. This better occupies otherwise idle registers, shared-memory banks, and Tensor Cores. And a thread block pair achieves higher Tensor Core utilization by doubling the threads and shared memory available for one tile.

Any boost in performance will happen transparently. From a programmer’s perspective, you simply launch a cluster of two blocks and treat it as one large thread block split across two thread blocks. In CUTLASS, for instance, this is exposed as a 2-SM UMMA (Unified MMA) GEMM operation that uses DSMEM and cluster barriers for inter-CTA data sharing. You simply request two SMs for a single GEMM tile, and CUTLASS handles the cluster launch, DSMEM setup, and interthread block synchronization automatically.

If you request a tile size that a single thread block can’t handle (say, 128 × 128 for FP16 Tensor Core operations), CUTLASS will automatically allocate two thread blocks as a pair. The SMs split the work and rely on DSMEM + cluster.sync() under the hood to share the tile. This way, you can do pairwise overlapping of UMMA operations with DSMEM.

In short, thread block pairs and DSMEM provide parallelism in the form of multi-SM cooperation. They provide fast, on-chip data sharing and synchronization across thread blocks. This eliminates many scenarios that previously required global-memory handoffs or additional kernel launches. This benefits many multithread block algorithms and simplifies their implementation.

## Reducing Global Memory Traffic with Thread Block Clusters

From a performance perspective, DSMEM can significantly cut down redundant global-memory traffic and enable higher effective bandwidth. For instance, in a tiled GEMM where multiple thread blocks within a cluster share chunks of the A or B matrix, one block can load a tile from global memory and multicast the tile to other blocks in the cluster using the TMA.

The TMA engine supports a multicast copy mode that feeds directly into DSMEM when thread blocks belong to the same cluster. A single TMA transfer from global memory can place data into each participating block’s shared memory simultaneously. This avoids redundant DRAM fetches.

With TMA multicast, the GPU ensures that L2-cached data is broadcast into the shared memory of each cluster member (thread block) in one pass. As a result, you avoid repeated global-memory loads of the same tile. This improves bandwidth utilization and shrinks DRAM traffic—especially when many blocks need the same input, as shown in Figure 10-17.

![Figure 10-17. For these four (2 × 2) thread clusters, each tile is loaded once and multicast into the shared memory of all CTAs in each cluster (source: https://oreil.ly/kEZsv)](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-17.png)

Here, the TMA engine performs a single multicast from global memory into DSMEM, broadcasting the tile to every thread block’s SMEM in the cluster and eliminating redundant DRAM reads. TMA multicast is configured through a tensor-map descriptor and issued as cp.async.bulk.tensor targeting shared::cluster. Fortunately, higher-level libraries like CUTLASS/cuTe and Triton generate these multicast operations, tensor-map descriptors, and bulk tensor copies for you. Specifically, these libraries can issue the relevant PTX instructions including cp.async.bulk.tensor with a tensor-map operand. They can issue PTX instructions that target shared::cluster for multicast. In these cases, the hardware delivers the tile to each thread block’s SMEM.

Another common use case is multiblock reductions or scans. Instead of having each block write out its partial sum or scan result to global memory and then launching a separate kernel to combine them, thread blocks in a thread block cluster can write their partial results into DSMEM.

Within that same cluster, you can perform a fast on-chip reduction or prefix-sum across these partial results using shared memory and a cluster-level barrier, cluster.sync(). Only the final result needs to go to global memory. This greatly reduces global memory reads and writes.

By combining thread block clusters, DSMEM, and TMA multicast, you can build multistage, fine-grained pipelines that keep most data exchanges on-chip. Whether you are sharing tiles for GEMM, accumulating partial sums for a reduction, or performing a multiblock scan, these mechanisms let you minimize HBM round trips and maximize arithmetic intensity.

> A block scoped cuda::pipeline does not synchronize other blocks; cluster wide distribution uses TMA multicast or DSMEM with cluster.sync.

Consider two thread blocks, CTA 0 and CTA 1, that both need the same tile of matrix A. Without DSMEM, each CTA issues its own global-memory load, which wastes DRAM bandwidth by fetching the identical data twice.

With DSMEM, however, CTA 0 loads the tile once into its shared memory and then multicasts it over the on-chip DSMEM network so that CTA 1 can read it directly from shared memory. If more than two CTAs form a cluster, the same tile will be shared across all thread blocks in the cluster, but this still requires only one global HBM load. Table 10-4 shows a comparison using a cluster of two thread blocks versus two independent thread blocks.

Table 10-4. Performance impact of DSMEM on a two-block workload

| Metric | Two independent CTAs (no DSMEM) | CTA pair with DSMEM (cluster of 2) |
| --- | --- | --- |
| Global load transactions | 2× (each thread block loads tile) | 1× (tile loaded once) |
| L2 cache hit rate | 50% | 85% |
| Inter-CTA data reuse | N/A (no reuse) | Significant (tile reused by CTA 1) |
| Effective DRAM BW per CTA | 300 GB/s | 150 GB/s (50% less) |
| Kernel time (relative) | 1.0× | 0.6× (40% speedup) |

Here, we see a 40% speedup in kernel execution when using a thread block pair (aka *CTA pair*) with DSMEM. This is consistent with the bandwidth savings as well. The DRAM bandwidth per thread block drops by 50% from 300 GB/s to 150 GB/s since each tile is fetched only once. The L2 hit rate jumps from 50% to 85% since the TMA hardware is performing a multicast copy on-chip—and therefore avoiding multiple loads from global memory.

In short, by multicasting shared data, thread block clusters allow multiple blocks to reuse data at on-chip speeds. This leads to substantial speedups for memory-bound workloads.

It’s important to note that both DSMEM and L2 operate in parallel. This provides two “lanes” for inter-block data sharing. This dual-path design combines DSMEM’s ultra-fast cluster-wide communication with L2’s broader caching coverage. In other words, DSMEM access is bypassing the L2 cache for cluster-local addresses. Instead, DSMEM uses the dedicated SM-to-SM network, as shown in Figure 10-18.

![Figure 10-18. DSMEM uses an SM-to-SM thread block cluster-local network between two thread block clusters for its data exchange](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-18.png)

Remote DSMEM accesses are routed using the thread block cluster interconnect and are distinct from global-memory traffic. For example, when a thread block pulls a tile using DSMEM, it uses low-latency shared-memory transfers. This ensures that interthread block data sharing in a thread block cluster is as fast as on-chip shared memory.

If a tile is not present in DSMEM, it must be fetched from global memory (following the normal cache hierarchy that may hit in L2, for example). This way, if a tile has already been evicted from DSMEM, for instance, the thread block can still retrieve the tile data from the L2 cache instead of going all the way back to global DRAM.

As a result, memory stalls are rare, occupancy remains high, and overall execution runs significantly faster than the unoptimized, two-independent-thread-blocks, non-DSMEM implementation.

## Designing Efficient Algorithms with Thread Block Clusters

Thread block clusters enable new strategies for parallelizing workloads that previously required global memory communication or multiple kernel launches. For example, imagine a large matrix multiplication in which the output tile is too large to be handled by a single thread block due to shared memory limits.

In the past, you might split the work across two thread blocks, but then each block would have to exchange partial results with global memory, which is relatively slow and inefficient. Otherwise, you’d have to launch a separate reduction kernel to combine the outputs of the two thread blocks.

With thread block clusters and DSMEM, those two blocks can form a cluster and directly share a joint region of on-chip shared memory to seamlessly combine their results using hardware-supported primitives. The DSMEM hardware allows an SM to perform loads/stores/atomics to another SM’s shared memory through a fast network.

It’s important to carefully design algorithms to use thread block clusters effectively. Synchronization overhead is low, but it is still nonzero. Performing very fine-grained data sharing might not pay off.

Thread block cluster barriers work like warp-level intrinsics (e.g., __shfl_sync()), introduced in Chapter 7, in that every participant must arrive at the synchronization point together. When you call cluster.sync(), all thread blocks in the cluster must reach that line before any block can continue.

If one block finishes its work early, it simply waits. If a block never reaches the barrier, because its threads took a different branch, for instance, the entire cluster will deadlock.

In other words, just as warp intrinsics demand that all threads in a warp execute the same instruction path to avoid divergence, thread block clusters demand that all blocks follow the same control flow up to each cluster.sync() call.

Because of this lockstep requirement, fine-grained data sharing between blocks must be balanced against the overhead and risk of deadlock. Synchronization itself is inexpensive when done correctly, but if even one block bypasses or delays reaching a cluster.sync(), performance may be impacted—or, even worse, the kernel might hang.

You should structure code such that every block arrives at each barrier in unison—just as every thread in a warp must arrive at the barrier with warp-level intrinsics. This is essential to take full advantage of the thread block clusters’ low-latency, on-chip communication without falling into deadlocks.

Typically, thread block clusters perform the best when each block has a sizable amount of work that can run independently until a synchronization or data exchange point is needed. Once the synchronization happens and the data is transferred on-chip, the thread blocks can continue processing.

Thread block clusters are especially effective for block-sparse matrix operations—common in sparse attention, model pruning, and compression in LLMs. In these cases, the blocks process different nonzero regions and share their boundary data.

Thread block clusters are also useful for multiphase reductions such as the softmax and normalization steps in transformer layers. These require a final combine step using the partial results of each thread block in the thread block cluster.

More generally, large GEMMs benefit from thread block clusters when they exceed the resources of a single block. GEMMs, of course, are central to the transformer attention and multilayer perceptron (MLP) layers and embedding lookups common in modern LLMs.

Larger cluster sizes can reduce overall occupancy, however. For instance, a cluster of 16 blocks might monopolize 16 SMs for one task. This could leave fewer SMs for other tasks needed by that kernel launch. It’s recommended to start with small clusters, two- or four-thread block clusters, unless a bigger cluster is needed for special cases. As always, you should profile with your specific workload to confirm that sharing on-chip resources across thread block clusters outweighs the potential loss of parallelism.

> When using cluster launches, verify active blocks and cluster residency with Nsight Compute launch statistics and the standard occupancy APIs, such as cudaOccupancyMaxActiveBlocksPer Multiprocessor. For nonportable cluster sizes, set cudaFuncAttri buteNonPortableClusterSizeAllowed (or pass cudaLaunchAttrib uteNonPortableClusterSizeAllowed using cudaLaunchKernelEx attributes). Otherwise the launch may fail or measure low occupancy.

## Warp Specialization with Thread Block Clusters

Let’s now revisit warp specialization using a thread block cluster with the CUDA Pipeline API. We use a thread-block scoped pipeline as the cluster leader to stage the copies and perform cluster-wide barriers with DSMEM. This way, every block in the cluster consumes the same input tiles without reloading them from global memory.

Roles are assigned per warp inside each block. The leader block’s loader warp performs the cooperative copies into its shared memory once per tile. After a cluster-wide barrier publishes those tiles, every block uses a compute warp to read the leader’s tiles through DSMEM and compute a disjoint band of rows. A storer warp in each block writes that band back to global memory. The block-scoped pipeline is used only by the leader for the asynchronous copies. Other blocks do not wait on that pipeline. Here is the code:

```
// Warp specialization across a thread-block cluster
// using DSMEM and a block-scoped pipeline
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <algorithm>
namespace cg = cooperative_groups;
#define TILE_SIZE 128
#define TILE_ELEMS (TILE_SIZE * TILE_SIZE)
// Compute a band of rows of the TILE_SIZE×TILE_SIZE product from DSMEM sources.
// Each lane processes rows [row_begin, row_end) in a 32-way striped loop.
__device__ void compute_rows_from_ds(const float* __restrict__ A_src,
                                     const float* __restrict__ B_src,
                                     float* __restrict__ C_dst,
                                     int row_begin, int row_end,
                                     int lane_id) {
    for (int row = row_begin + lane_id; row < row_end; row += warpSize) {
        for (int col = 0; col < TILE_SIZE; ++col) {
            float acc = 0.0f;
            #pragma unroll
            for (int k = 0; k < TILE_SIZE; ++k) {
                acc += A_src[row * TILE_SIZE + k] * B_src[k * TILE_SIZE + col];
            }
            C_dst[row * TILE_SIZE + col] = acc;
        }
    }
}
extern "C"
__global__ void warp_specialized_cluster_pipeline(
    const float* __restrict__ A_global,
    const float* __restrict__ B_global,
    float* __restrict__ C_global,
    int numTiles) {
    thread_block cta = this_thread_block();
    cluster_group  cluster = this_cluster();
    extern __shared__ float shared_mem[];
    float* A_tile_local = shared_mem;
    float* B_tile_local = A_tile_local + TILE_ELEMS;
    float* C_tile_local = B_tile_local + TILE_ELEMS;
    // Block-scoped pipeline used only by the cluster leader
    // to stage asynchronous copies
    __shared__
    cuda::pipeline_shared_state<cuda::thread_scope_block, 2> pipe_state;
    auto pipe = cuda::make_pipeline(cta, &pipe_state);
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int cluster_rank      = cluster.block_rank();
    const dim3 cluster_dims     = cluster.dim_blocks();
    const int  blocks_in_cluster = cluster_dims.x * cluster_dims.y *
                                   cluster_dims.z;
    // 1D cluster arrangement along x;
    // each iteration processes one tile per cluster
    auto loader = cooperative_groups::tiled_partition<32>(cta);
    for (int tile = blockIdx.x / cluster_dims.x; tile < numTiles;
         tile += gridDim.x / cluster_dims.x) {
        const size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;
        // Leader block’s loader warp stages A and B once for the entire cluster
        if (cluster_rank == 0 && warp_id == 0) {
            pipe.producer_acquire();
            cuda::memcpy_async(loader, A_tile, A_global + offset,
                               cuda::aligned_size_t<32>{TILE_ELEMS*sizeof(float)},
                               pipe);
            cuda::memcpy_async(loader, B_tile, B_global + offset,
                               cuda::aligned_size_t<32>{TILE_ELEMS*sizeof(float)},
                               pipe);
            pipe.producer_commit();
            // Make loads visible to leader CTA before publishing cluster-wide
            // wait for committed stage before publishing
            pipe.consumer_wait();
            pipe.consumer_release();
        }
        // Publish the leader’s tiles to every block via DSMEM
        cluster.sync();
        const float* A_src = cluster.map_shared_rank(A_tile_local, 0);
        const float* B_src = cluster.map_shared_rank(B_tile_local, 0);
        // Divide rows among blocks in the cluster
        const int rows_per_block = (TILE_SIZE + blocks_in_cluster - 1)
                                   / blocks_in_cluster;
        const int row_begin = std::min(cluster_rank * rows_per_block, TILE_SIZE);
        const int row_end   = std::min(row_begin + rows_per_block, TILE_SIZE);
        // Compute warp produces block’s band of rows into local shared memory
        if (warp_id == 1) {
            pipe.producer_acquire();
            compute_rows_from_ds(A_src, B_src, C_tile_local, row_begin, row_end,
                                 lane_id);
            pipe.producer_commit(); // publish C_tile_local
        }
        // Storer warp writes this block’s rows back to global memory
        if (warp_id == 2) {
            pipe.consumer_wait(); // observe C_tile_local
            for (int row = row_begin + lane_id; row < row_end; row += warpSize) {
                for (int col = 0; col < TILE_SIZE; ++col) {
                    C_global[offset + row * TILE_SIZE + col] =
                        C_tile_local[row * TILE_SIZE + col];
                }
            }
            pipe.consumer_release();
        }
        // All blocks finish this tile before the leader reuses its buffers
        cluster.sync();
    }
    // dynamic shared memory size: 3 * TILE_ELEMS * sizeof(float)
}
```

Here, the leader performs one pair of cooperative copies of each tile into its shared memory using a block-scoped pipeline. The cluster-wide barrier makes the data visible to all cluster members through DSMEM.