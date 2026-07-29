与其退回到全局加载和存储，你可以启动一个线程块簇，让其中的每个块处理数据的一部分，然后使用 DSMEM 进行同步与共享。所有通信都以 SM 间（SM-to-SM）速度完成，从而避免了昂贵的 HBM 流量。

简而言之，DSMEM 是一种比使用全局内存进行块间通信更快、更结构化的替代方案。但它的作用域仅限于同一个簇内的块。DSMEM 非常适合以下场景：需要频繁、低延迟块间协调的持久化核函数、注意力机制，或 LLM 中的其他多阶段算法（每个块在继续之前都必须交换中间状态）——以及任何受益于将中间结果保留在片上、而非写回全局内存再重新加载的工作负载。

> 可移植的最大簇大小为 8 个块。某些 GPU 在通过 cudaFuncAttributeNonPortableClusterSizeAllowed 属性显式启用时，支持更大的非可移植簇大小（例如最多 16 个 CTA）。这会增大你的 DSMEM 占用量，代价是占用簇中更多的 SM。

### 暂存内存

暂存内存（scratch memory）是支撑 DSMEM 与线程块簇同步的底层硬件基础设施；DSMEM 是对你的 CUDA 代码隐式可见的共享数据缓冲区，而暂存内存则是 GPU 用来跟踪协调元数据（如屏障状态、组进度以及 DSMEM 访问标志）的一块独立的片上 SRAM 区域。

你不会从核函数中直接访问暂存内存，而是由 GPU 自动管理它。当你启动一个线程块簇时，硬件会分配一部分暂存内存，用来维护簇级屏障计数器（cluster.sync() 状态）、跟踪哪些块已到达 DSMEM 操作或同步点，并协调跨 SM 对分布式共享区域的安全访问。

由于暂存内存针对这些元数据操作做了优化，它使簇级屏障和 DSMEM 访问能够非常快速地完成。如果对于非常大的簇或复杂的同步模式，元数据大小超过了可用的暂存 SRAM，GPU 会透明地将部分状态溢出到本地内存中（由 L1/L2 缓存支撑）。这种溢出在保证正确性的同时，仍尽可能保留片上效率。

简而言之，DSMEM 是你用来在块之间共享数据的抽象，暂存内存则是幕后使这些 DSMEM 操作快速且可扩展的设施。二者结合，使一个线程块簇能够表现得像一个单一的逻辑单元，打破了线程块之间的传统壁垒。这为那些需要跨多个线程块紧密协调并行的工作负载提升了性能。

### 启动线程块簇

让我们说明如何在实践中使用线程块簇。首先，我们需要用一个此前未使用过的簇维度（cluster dimension）来启动核函数。

在 CUDA 中，这通过一个允许指定簇大小的扩展启动 API 来完成。例如，如果我们想要两个块组成的簇，也称为*线程块对*（thread block pair）或 *CTA pair*，可以配置一个特殊的启动属性，如下所示：

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

在这段主机代码中，我们使用 cudaLaunchKernelEx 来启动 MyClusterKernel，每个簇包含 2 个线程块（因此总共 64 个簇，因为 128 个块 ÷ 每簇 2 个）。clusterDim 被设为 2，意味着每个簇将包含 2 个线程块。对于我们想要簇协作核函数的场景，这取代了传统的 <<<gridDim, blockDim>>> 语法。在底层，CUDA 运行时确保这些配对的块被调度到能够通过 DSMEM 通信的位置。

> 请记住，每个线程块簇都需要在物理上装入你所指定的 SM 与资源。可移植的最大簇维度为 8 个线程块。要在受支持的 Blackwell 型号上启动 16 个块，你必须启用非可移植属性，并合理设置块大小，使簇能够常驻于 SM 上。在为线程块簇设定大小时请牢记这一点。

### 用协作组 API 协调线程块簇

要在簇中的多个线程块之间协调工作，你首先要使用 cooperative_groups::this_cluster() 获得该组块的句柄。这个簇句柄让你能够执行硬件加速的簇级屏障——并直接访问另一个块的共享内存。

这一切都无需离开核函数、也无需借助全局内存标志。下面是一个示例核函数，它将每个线程块的一个本地值累加到线程块 0 的共享内存中：

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

在这里，我们通过调用 cg::this_cluster() 获得一个簇句柄。它返回一个 cluster_group 对象，正好表示那些作为一个线程块簇一起启动的块。运行时保证该簇组中的每个线程块都同时常驻——否则启动将失败。

系统会隐式提供一致的共享内存分配。簇中的每个块都必须为 extern __shared__ int shared_buffer[] 声明相同的大小。DSMEM 硬件随后会在逻辑上将每个块的共享内存合并为一个虚拟地址空间。而且由于所有块都预留了相同的 shared_buffer 大小，指向线程块 0（即 rank 0）缓冲区的指针将能够跨 SM 正确协调。

> 使用线程块簇时，务必像全局内存合并一样组织每个线程块的 DSMEM 访问。换言之，让每个 warp 读写连续的、32 字节对齐的扇区。这样，硬件就能在没有 bank 冲突或意外串行化的情况下路由簇级 DSMEM 传输。在实践中，布局你的分块数据，使线程块 j 中的 warp i 始终访问一个唯一的、对齐的范围。这将避免内存 bank 争用，并使 DSMEM 数据传输保持全速运行。

在前面的示例代码中，我们还看到 cluster.sync() 被用作跨簇内线程块的簇级屏障。与只同步单个块内线程的 __syncthreads() 不同，cluster.sync() 同步该簇中每个块的所有线程。

这个屏障由硬件实现，通常比网格级同步具有更低的延迟，因为它只协调簇中的块。这意味着你可以频繁地同步块而影响极小——只要它们在同一个簇内。

而且不存在因块缺失而导致死锁的风险，因为 CUDA 强制要求网格正确启动并装入 GPU。因此，不会出现某些块到达屏障后永远挂起、等待那些从未启动的“缺失”块的情形。所有块都已存在，因此一旦每个块都调用它，屏障就会干净地完成。

在前面的代码块中，你看到跨块共享内存访问是通过 map_shared_rank() 完成的。在第一个屏障之后，每个块的 shared_buffer[] 都已初始化。为了获得指向块 0 共享内存的指针，其他块调用 int* remote_buffer = cluster.map_shared_rank(shared_buffer, 0)，其中指定了簇中的线程块 id（0）。

GPU 硬件会自动转换该指针，使任何加载或存储都经由片上 DSMEM 网络，而不是发往全局 DRAM。因此，对 remote_buffer 的任何写操作——无论是原子写还是普通写——都会直接在片上更新块 0 的共享内存。值得强调的是，远程 DSMEM 访问的性能特征不同于本地共享内存。远程加载和存储受益于合并的、对齐的 32 字节段。

例如，在前面的代码块中，当块 1...n 需要将它们的本地值累加到块 ID 0 的共享缓冲区时，它们调用 atomicAdd(&remote_buffer[0], shared_buffer[0])。而由于 remote_buffer 是通过 cluster.map_shared_rank() 获得的，这个 atomicAdd 会经由片上 DSMEM 网络直接进入块 0 的 SMEM。换言之，不存在到 DRAM 或 L2 缓存的往返。每次写入都以片上速度完成。

在前面的代码示例中，你看到一旦所有块都执行了它们的 atomicAdd，第二个 cluster.sync() 会确保块 0 在继续之前，在自己的 shared_buffer[0] 中看到完整的和。

简而言之，cooperative_groups::this_cluster() 加上 cluster.sync() 和 cluster.map_shared_rank() 为你提供了一种简单、高效的方式，在一个线程块簇中的多个线程块之间进行同步和数据共享。所有这些操作都在片上进行，避免了全局内存往返，并支持块之间的细粒度协作。这种线程块簇与 DSMEM 的结合提供的性能，远高于任何全局内存回退方案或手动原子标志方法。

### 线程块对

借助现代 NVIDIA，GPU 让你能够在单个 GPC 内的多个 SM 上，恰好协同调度簇中的两个线程块，即线程块对（thread block pair，即 *CTA pair*）。通过将线程块编组为一个簇（例如图 10-13 所示的 2 块簇），共享数据的核函数可以使用 TMA 将分块搬移到每个块的共享内存中。

![图 10-13. 线程块对由两个线程块组合而成](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-13.png)

单个线程块可能缺乏足够的寄存器或共享内存容量来独自处理一个非常大的分块（例如一个 256 × 256 的矩阵子块）。通过在一个 GPC 内相邻的 SM 上配对两个线程块并使用 DSMEM，这两个块可以分担一个大分块上的工作，同时通过一个统一的共享内存区域共享数据，如图 10-14 所示。

![图 10-14. 线程块对（即 CTA pair）为 A * B 矩阵乘法加载分块作为操作数（来源：https://oreil.ly/kEZsv）](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-14.png)

在这里，对中的每个线程块都可以将操作数分块的一部分（例如 128 × 16）加载到其片上 SMEM 中以进行矩阵乘法。此外，每个线程块在 Tensor Memory（TMEM）中持有累加器的一部分（例如 128 × 256）。这使得 CTA pair 中的两个线程块能够协作处理单个分块，如图 10-15 所示。

![图 10-15. 配合 Tensor Core 与 TMEM 的线程块对](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-15.png)

线程块对允许跨两个物理 SM 的更大矩阵乘法。使用一对线程块实际上使分块大小翻倍，因为每个 SM 处理分块数据的一半。

SM 硬件使用 DSMEM 在两个 SM 之间共享操作数数据。DSMEM 减少了重复加载，改善了数据复用，并提高了算术强度。图 10-16 展示了在一个线程块簇中使用 DSMEM 在 SM 之间进行的这种数据共享。

![图 10-16. 在一个线程块簇中使用 DSMEM 在 SM 之间共享数据](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-16.png)

CUDA 为这些多 SM 操作提供了增强的多播屏障与同步原语。这一对块使用轻量级的 cluster.sync() 调用彼此同步。这样，当一个 SM 用完一个分块时，CTA pair 中相邻的 SM 就可以消费它。

每个分块都被馈入这一对线程块，而无需冗余的全局内存事务。这更好地占用了原本闲置的寄存器、共享内存 bank 和 Tensor Core。而且线程块对通过为一个分块提供翻倍的线程和共享内存，实现了更高的 Tensor Core 利用率。

任何性能提升都会透明地发生。从程序员的角度看，你只需启动一个两块的簇，并把它当作一个被拆分到两个线程块上的大型线程块来对待。例如在 CUTLASS 中，这被暴露为一个 2-SM UMMA（Unified MMA）GEMM 操作，它使用 DSMEM 和簇屏障来进行 CTA 间的数据共享。你只需为单个 GEMM 分块请求两个 SM，CUTLASS 就会自动处理簇启动、DSMEM 设置和线程块间同步。

如果你请求一个单个线程块无法处理的分块大小（比如用于 FP16 Tensor Core 操作的 128 × 128），CUTLASS 会自动将两个线程块分配为一对。这些 SM 会分担工作，并在底层依靠 DSMEM + cluster.sync() 来共享分块。这样，你就可以用 DSMEM 对 UMMA 操作进行成对重叠。

简而言之，线程块对与 DSMEM 以多 SM 协作的形式提供并行性。它们在线程块之间提供快速的片上数据共享与同步。这消除了许多以往需要全局内存交接或额外核函数启动的场景。这使许多多线程块算法受益，并简化了它们的实现。

## 用线程块簇减少全局内存流量

从性能角度看，DSMEM 可以显著削减冗余的全局内存流量，并实现更高的有效带宽。例如，在一个分块化 GEMM 中，簇内的多个线程块共享 A 矩阵或 B 矩阵的若干块，其中一个块可以从全局内存加载一个分块，并使用 TMA 将该分块多播到簇中的其他块。

TMA 引擎支持一种多播复制模式，当线程块属于同一个簇时，它可以直接馈入 DSMEM。一次从全局内存发起的 TMA 传输可以同时将数据放入每个参与块的共享内存中。这避免了冗余的 DRAM 取数。

借助 TMA 多播，GPU 确保被 L2 缓存的数据在一趟传输中被广播到每个簇成员（线程块）的共享内存中。因此，你避免了对同一分块的重复全局内存加载。这提高了带宽利用率，并缩减了 DRAM 流量——尤其是在许多块需要相同输入时，如图 10-17 所示。

![图 10-17. 对于这四个（2 × 2）线程簇，每个分块只加载一次，并被多播到每个簇中所有 CTA 的共享内存中（来源：https://oreil.ly/kEZsv）](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-17.png)

在这里，TMA 引擎执行一次从全局内存到 DSMEM 的多播，将分块广播到簇中每个线程块的 SMEM，并消除冗余的 DRAM 读取。TMA 多播通过一个张量映射描述符（tensor-map descriptor）配置，并作为面向 shared::cluster 的 cp.async.bulk.tensor 发出。幸运的是，CUTLASS/cuTe 和 Triton 等更高层的库会替你生成这些多播操作、张量映射描述符和批量张量复制。具体来说，这些库可以发出相关的 PTX 指令，包括带有张量映射操作数的 cp.async.bulk.tensor。它们可以发出面向 shared::cluster 进行多播的 PTX 指令。在这些情况下，硬件会将分块传递到每个线程块的 SMEM。

另一个常见的用例是多块归约或扫描。与其让每个块把它的部分和或扫描结果写出到全局内存、然后再启动一个单独的核函数来合并它们，一个线程块簇中的线程块可以把它们的部分结果写入 DSMEM。

在同一个簇内，你可以使用共享内存和一个簇级屏障 cluster.sync()，对这些部分结果执行快速的片上归约或前缀和。只有最终结果需要写到全局内存。这极大地减少了全局内存的读写。

通过将线程块簇、DSMEM 和 TMA 多播结合起来，你可以构建多阶段、细粒度的流水线，使大多数数据交换保持在片上。无论你是在为 GEMM 共享分块、为归约累加部分和，还是执行多块扫描，这些机制都能让你最小化 HBM 往返并最大化算术强度。

> 块作用域的 cuda::pipeline 不会同步其他块；簇级分发使用 TMA 多播或配合 cluster.sync 的 DSMEM。

考虑两个线程块 CTA 0 和 CTA 1，它们都需要 A 矩阵的同一个分块。没有 DSMEM 时，每个 CTA 都会发起自己的全局内存加载，通过两次取回相同的数据而浪费了 DRAM 带宽。

而有了 DSMEM，CTA 0 只需将该分块加载一次到它的共享内存中，然后经由片上 DSMEM 网络将其多播，使 CTA 1 可以直接从共享内存中读取它。如果一个簇由两个以上的 CTA 组成，同一个分块将在簇中所有线程块之间共享，但这仍然只需要一次全局 HBM 加载。表 10-4 展示了使用两个线程块的簇与两个独立线程块的对比。

表 10-4. DSMEM 对两块工作负载的性能影响

| 指标 | 两个独立 CTA（无 DSMEM） | 配合 DSMEM 的 CTA pair（2 块簇） |
| --- | --- | --- |
| 全局加载事务 | 2×（每个线程块加载分块） | 1×（分块只加载一次） |
| L2 缓存命中率 | 50% | 85% |
| CTA 间数据复用 | N/A（无复用） | 显著（分块被 CTA 1 复用） |
| 每 CTA 有效 DRAM 带宽 | 300 GB/s | 150 GB/s（减少 50%） |
| 核函数时间（相对） | 1.0× | 0.6×（加速 40%） |

在这里，我们看到使用配合 DSMEM 的线程块对（即 *CTA pair*）时，核函数执行有 40% 的加速。这与带宽的节省也是一致的。每线程块的 DRAM 带宽从 300 GB/s 下降 50% 到 150 GB/s，因为每个分块只被取回一次。L2 命中率从 50% 跃升到 85%，因为 TMA 硬件在片上执行了一次多播复制——从而避免了从全局内存的多次加载。

简而言之，通过多播共享数据，线程块簇让多个块能够以片上速度复用数据。这为访存受限的工作负载带来了可观的加速。

需要指出的是，DSMEM 和 L2 是并行工作的。这为块间数据共享提供了两条“通道”。这种双路径设计将 DSMEM 超快的簇级通信与 L2 更广泛的缓存覆盖结合起来。换言之，对于簇内本地地址，DSMEM 访问会绕过 L2 缓存。取而代之，DSMEM 使用专用的 SM 间网络，如图 10-18 所示。

![图 10-18. DSMEM 使用两个线程块簇之间的一个 SM 间、簇内本地网络进行数据交换](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-18.png)

远程 DSMEM 访问使用线程块簇互连进行路由，并与全局内存流量相区别。例如，当一个线程块使用 DSMEM 拉取一个分块时，它使用低延迟的共享内存传输。这确保了一个线程块簇中线程块间的数据共享与片上共享内存一样快。

如果某个分块不在 DSMEM 中，就必须从全局内存取回（遵循可能命中 L2 的正常缓存层次结构）。这样，如果一个分块已经从 DSMEM 中被逐出，线程块仍可以从 L2 缓存中取回该分块数据，而不必一路回到全局 DRAM。

因此，内存停顿很少见，占用率保持在高位，整体执行速度显著快于未经优化的、两个独立线程块的非 DSMEM 实现。

## 用线程块簇设计高效算法

线程块簇为并行化那些以往需要全局内存通信或多次核函数启动的工作负载提供了新的策略。例如，设想一个大型矩阵乘法，其输出分块因共享内存限制而过大，无法由单个线程块处理。

在过去，你可能会把工作拆分到两个线程块上，但那样每个块就都不得不通过全局内存来交换部分结果，而这相对缓慢且低效。否则，你就得启动一个单独的归约核函数来合并这两个线程块的输出。

有了线程块簇和 DSMEM，这两个块就可以组成一个簇，并直接共享一块联合的片上共享内存区域，使用硬件支持的原语无缝地合并它们的结果。DSMEM 硬件允许一个 SM 通过一个快速网络对另一个 SM 的共享内存执行加载/存储/原子操作。

精心设计算法以有效使用线程块簇很重要。同步开销很低，但它仍然非零。执行非常细粒度的数据共享可能得不偿失。

线程块簇屏障的工作方式类似于第 7 章介绍的 warp 级内建函数（例如 __shfl_sync()），即每个参与者都必须一起到达同步点。当你调用 cluster.sync() 时，簇中的所有线程块都必须到达那一行，任何块才能继续。

如果一个块提前完成了它的工作，它就只是等待。如果某个块因为它的线程走了不同的分支等原因而始终未到达屏障，整个簇就会死锁。

换言之，正如 warp 内建函数要求一个 warp 中的所有线程执行相同的指令路径以避免发散，线程块簇要求所有块在每个 cluster.sync() 调用之前都遵循相同的控制流。

由于这种锁步（lockstep）要求，块之间的细粒度数据共享必须与开销和死锁风险相权衡。正确执行时同步本身是廉价的，但如果哪怕只有一个块绕过或延迟到达 cluster.sync()，性能就可能受影响——或者更糟，核函数可能会挂起。

你应该组织代码，使每个块都齐步到达每个屏障——正如使用 warp 级内建函数时一个 warp 中的每个线程都必须到达屏障一样。这对于充分利用线程块簇低延迟的片上通信而不陷入死锁至关重要。

通常，当每个块都有相当数量的工作、可以独立运行直到需要一个同步或数据交换点时，线程块簇表现最佳。一旦同步发生、数据在片上完成传输，线程块就可以继续处理。

线程块簇对块稀疏矩阵操作尤其有效——这类操作常见于 LLM 中的稀疏注意力、模型剪枝和压缩。在这些情况下，各个块处理不同的非零区域，并共享它们的边界数据。

线程块簇对于多阶段归约也很有用，例如 transformer 层中的 softmax 和归一化步骤。这些操作需要使用线程块簇中每个线程块的部分结果进行一个最终合并步骤。

更一般地，当大型 GEMM 超出单个块的资源时，它们会从线程块簇中受益。当然，GEMM 是现代 LLM 中常见的 transformer 注意力、多层感知机（MLP）层和嵌入查找的核心。

不过，更大的簇大小可能会降低整体占用率。例如，一个 16 块的簇可能会为一个任务独占 16 个 SM。这可能会让该次核函数启动所需的其他任务可用的 SM 更少。建议从小簇开始，即两块或四块的簇，除非特殊情况需要更大的簇。一如既往，你应该用你的具体工作负载做性能分析，以确认跨线程块簇共享片上资源的收益胜过潜在的并行性损失。

> 使用簇启动时，请用 Nsight Compute 的启动统计信息和标准占用率 API（如 cudaOccupancyMaxActiveBlocksPer Multiprocessor）来验证活跃块和簇常驻情况。对于非可移植簇大小，请设置 cudaFuncAttri buteNonPortableClusterSizeAllowed（或使用 cudaLaunchKernelEx 属性传递 cudaLaunchAttrib uteNonPortableClusterSizeAllowed）。否则启动可能会失败，或测得的占用率很低。

## warp 专门化与线程块簇结合

现在让我们用一个线程块簇配合 CUDA Pipeline API 来重新审视 warp 专门化。我们使用一个线程块作用域的流水线作为簇的主导者（leader）来暂存复制，并使用 DSMEM 执行簇级屏障。这样，簇中的每个块都消费相同的输入分块，而无需从全局内存重新加载它们。

角色是在每个块内部按 warp 分配的。主导块的加载器 warp 每个分块执行一次协作式复制到它的共享内存中。在一个簇级屏障发布这些分块之后，每个块使用一个计算 warp 通过 DSMEM 读取主导者的分块，并计算一个不相交的行带。每个块中的一个存储器 warp 将该行带写回全局内存。块作用域的流水线仅由主导者用于异步复制。其他块不会在该流水线上等待。代码如下：

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

在这里，主导者使用一个块作用域的流水线，将每个分块进行一对协作式复制到它的共享内存中。簇级屏障通过 DSMEM 使这些数据对所有簇成员可见。
