# 第 11 章　核间流水线化、同步与 CUDA 流序内存分配

到目前为止，我们一直聚焦于**核内**（intra-kernel）的工具——cuda::pipeline 双缓冲、warp 专门化（loader/compute/storer warp）、持久化核函数以及带 DSMEM/TMA 的线程块簇——以便让 SM 在单个核函数内保持繁忙。本章我们保留这些核函数，并展示如何借助 CUDA streams、events 与流序内存分配器（stream-ordered memory allocator）在多个核函数与批次之间做流水线化。简而言之，第 10 章聚焦于在一个核函数**内部**隐藏延迟；本章则展示如何在核函数**之间**、以及在 GPU 与主机之间隐藏延迟。

这种核间并发（inter-kernel concurrency）对于在真实（real-world）工作负载中让 GPU 的所有引擎保持繁忙至关重要。要在现代 GPU 上达到峰值利用率，我们需要让 GPU 的计算引擎（compute engine）与直接内存访问（direct memory access，DMA）引擎保持繁忙并并行运行。

CUDA streams 为这种核间并发提供了基础。通过组合异步内存操作、细粒度同步（fine-grained synchronization）以及 CUDA Graphs（本章简要引入，下一章详述），你可以构建出高度高效、避免主机侧停顿的流水线。

## 用 CUDA Streams 重叠核函数执行

一个 CUDA stream 是一串操作——核函数启动、内存拷贝与内存分配——它们按照发起（issue）的顺序执行。设想从 CPU 使用 2 个 stream 向 GPU 启动 5 个核函数，如图 11-1 所示。

![图 11-1. 从 CPU 向 GPU 上运行的两个 stream 启动五个核函数](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-1.png)

这里可以看到 ker_A 与 ker_B 在 stream 2 上运行，而 ker_1、ker_2 与 ker_3 在 stream 1 上运行。只要硬件资源允许，所有核函数都可以彼此重叠——并且可以跨 CUDA streams 重叠。

在 stream 异步执行核函数操作的同时，CPU 能够继续执行工作（cpu_code_1 与 cpu_code_2）。在这两个 CUDA stream 上启动这五个核函数的代码如下所示：

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

ker_1 被入队到 stream1，随后控制权立即返回 CPU。cpu_code_1() 在主机上运行，与此同时 ker_1 在 GPU 上执行。与此并行，我们把 ker_A 入队到 stream2、把 ker_B 入队到 stream1。接着我们把 ker_2 入队到 stream2，穿插执行 cpu_code_2，再把 ker_3 入队到 stream1。最后，我们在每个 stream 上做同步，等待所有工作完成，然后销毁 stream 以清理资源。

这个示例突出展示了五次不同的核函数执行跨两个不同的 stream 相互重叠。稍微增加一点复杂度，并在第 10 章的基础上构建，下面是同一个 warp 专门化流水线示例，但改用了 CUDA streams：

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

这里，我们复用了 warp 专门化流水线，并展示了 stream 如何再加一层重叠：当 stream 1 在批次 n 上做计算时，stream 2 正在批次 b+1 上执行 DMA 加载。与此同时，它还能把批次 b−1 拷回。核函数内部的 cuda::pipeline 重叠保持不变。

我们会在本章后面继续叠加线程块簇（thread block cluster）来构建更高的复杂度，但让我们先深入了解 stream 如何帮助把计算与数据传输重叠起来。这有助于夯实 CUDA streams 的基础知识及其在基于 GPU 的性能工程中的作用。

## 用 Streams 重叠计算与数据传输

例如，你可以把每次核函数启动与内存拷贝入队到各自的 stream 中。这让 SM 得以执行核函数，而两个专用 DMA 引擎（一个负责主机 → 设备传输，一个负责设备 → 主机传输）并发地搬运数据。

由于 SM 计算流水线独立于这两个 DMA 引擎运行，你可以用 CUDA streams 将核函数的计算与两路数据传输完全重叠。然而，如果计算已被某个占满全部 SM 吞吐的核函数完全打满——或者某条拷贝流水线正以过度、额外的重叠使内存带宽饱和——那么这么做并不会提升性能。

当计算与内存吞吐都饱和时，你会开始看到两个并发操作各自只跑到 50%（举例而言），因为它们都在争抢同一资源。你可以对 GPU 利用率做剖析，以识别这些饱和阈值。

例如，设想一个把工作切分成批次的 AI 模型训练或推理工作负载。此时，你会在 stream 0 上启动批次 0 的核函数，与此同时 stream 1 调用 cudaMemcpyAsync() 把批次 1 从主机拷贝到设备，如图 11-2 所示。

![图 11-2. 三路重叠的时间线](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-2.png)

在至少有两个拷贝引擎（deviceProp.asyncEngine Count()）的现代 GPU 上，你可以把它扩展为三路重叠：stream 0 运行批次 0 的核函数，stream 1 把批次 1 从主机拷贝到设备，stream 2 把上一批次的结果写回主机。这可以进一步扩展到更多 stream。这种模式把数据传输延迟隐藏在计算之后，反之亦然，从而让 GPU 的所有引擎保持繁忙、把空闲时间降到最低。

在实践中，你的核函数必须满足若干要求才能实现这种并发行为。首先，异步传输中使用的任何主机指针都必须是页锁定（page-locked），即固定（pinned）的。如果你对可分页内存（pageable memory）调用 cudaMemcpyAsync()，运行时会在主机侧执行一次到固定内存的暂存拷贝，这会阻塞发起调用的主机线程以及入队所在的 stream，直到暂存完成。

这会使该次传输无法异步进行。虽然这会阻塞发起调用的主机线程，但 GPU 仍可以在其他 stream 中重叠计算与拷贝。只是那一次特定的传输不会正确重叠。要在你的 stream 中实现完全异步的传输，你必须使用固定主机内存（pinned host memory）。

> 在 PyTorch 的 DataLoader 中设置 pin_memory=True，就是在把你的主机缓冲区页锁定，以便数据能通过 DMA 直接传输到 GPU 内存。这让拷贝得以与计算重叠，并立即把控制权交还主机。DMA 引擎可以把传输与计算重叠起来。可分页内存会强制一次隐藏的暂存拷贝，从而破坏该次传输的重叠。

其次，你应当使用异步的分配与释放例程 cudaMallocAsync() 与 cudaFreeAsync()，而不是同步且阻塞的 cudaMalloc() 或 cudaFree() 调用。像 PyTorch 这样的框架提供了使用 CUDA 异步、流序内存分配器的选项。其思路是：在分配内存时不要停顿所有活跃的 stream。那样做对性能会非常糟糕。

异步的流序分配器允许每个 stream 请求或归还设备内存，而无需等待其他 stream。使用异步分配器可确保一个 stream 中的内存操作不会停顿其他 stream 中的操作。这将避免不必要的全局同步。让我们在下一节探讨流序内存分配器。

PyTorch 默认的 CUDA 缓存分配器是感知 stream 的（stream-aware），在正常运行时（例如从其缓存中服务分配请求），它会避免设备级同步（device-wide synchronization）。只有当它必须用 cudaMalloc 向操作系统申请更多内存时，才会发生同步。在实践中这意味着大多数张量的分配与释放不会阻塞其他 stream。启用 cudaMallocAsync 后端还能在许多工作负载中进一步减少碎片化（fragmentation）、提升复用，接下来你就会看到这一点。

## 流序内存分配器

在 PyTorch 中，你可以在启动 PyTorch 脚本之前设置环境变量 PYTORCH_ALLOC_CONF=backend:cudaMallocAsync 来启用 CUDA 的流序分配器。如果设置了该变量，PyTorch 的张量内存分配（cudaMallocAsync()）与释放（cudaFreeAsync()）操作会按照它们被调用的顺序入队到各自独立的 CUDA stream 中。当没有设置这个环境变量时，PyTorch 会使用它自己的缓存分配器。

如果你使用传统的 cudaMalloc(...)，请记住它是一个阻塞的、设备级的操作，会在返回前同步整个设备。这会停顿其他 stream 中的工作，因为每一次分配都会强制整个 GPU 停顿，直到内存被预留完毕。这会暂停所有 stream、限制并行性，并摧毁你工作负载的性能。

相比之下，使用带 cudaMallocAsync(...) 的流序分配器只是把分配请求记录到将要使用它的那个 CUDA stream 中——无论该 stream 正在执行核函数还是内存操作。它不会阻塞其他 stream。这样，内存管理就绝不会把正在给这些核函数供料的 stream 串行化。

> PyTorch 中使用的 CUDA 流序分配器避免了全局设备锁，并降低了分配开销。

在实践中，stream 0 可能正在批次 N 上执行一个 attention 核函数，stream 1 把批次 N+1 从主机拷贝到设备，而 stream 2 为批次 N+2 入队一个 cudaMallocAsync(...)。由于 cudaMallocAsync(...) 只是把它的工作追加到 stream 2 的队列中，stream 0 与 stream 1 得以不被打断地继续运行。

有了感知 stream 的分配器，为每个小批量（mini-batch）分配 GPU 内存都不会阻塞其他 stream。这对于要为每个批次分配暂存空间的 LLM 流水线很重要。即便在剧烈的内存变动之下，异步分配器也能防止停顿。

> 如果你的流水线为每个小批量分配一个暂存缓冲区（scratch buffer）——这在 LLM 训练与推理中很常见——那么使用流序内存分配器尤其重要。例如，LLM 流水线中的每个小批量都需要自己的临时工作区来存放 attention 的键/值或中间激活缓冲区。在这种情况下，你通常会调用分配器在 GPU 上预留那块“暂存缓冲区”。

分配请求由每个设备各自的内存池（memory pool）来满足。你可以用 cudaMemPoolSetAttribute() 调节内存池的释放阈值，在“把内存归还操作系统”与“为了性能而复用内存”之间权衡。阈值越高，内存池就会把内存保留得越久。这会减少内存被归还给操作系统的次数。这样就减少了对操作系统的调用，并通过避免反复的内存分配与释放来获得更好的性能。

下面的示例展示了如何用带 cudaMallocAsync 与 cudaFreeAsync 的流序内存分配器实现基于 stream 的重叠，并演示了 cudaMemPoolSetAttribute() 的用法。它突出展示了内存分配、数据传输与核函数执行如何借助 CUDA streams 完全流水线化：

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

这里，我们创建了两个 CUDA stream（stream1 与 stream2），并用 cudaMallocAsync 分配设备内存，确保每个 stream 都拥有自己的流序内存缓冲区。随后我们为两个相互独立的数据块发起工作。

在 stream1 上，我们执行一次从主机到设备（H2D）的异步拷贝，启动一个计算核函数，然后再把结果从设备异步拷贝回主机（D2H）。与此同时，我们在 stream2 上对第二个数据块做同样的事情。

由于这些操作发起在不同的 stream 上，GPU 设备会在它们之间重叠工作。stream1 执行核函数的同时，stream2 上正在进行 H2D 拷贝。一旦 stream1 的核函数完成，它就可以把数据拷回主机（D2H），并与 stream2 的核函数执行相重叠。

这里，得益于 CUDA streams 与流序内存分配，内存分配与核函数计算相互重叠。本示例所示的错峰调度减少了空闲时间、最大化了吞吐（throughput）。若没有流序分配，你要么不得不预先分配全部内存——增加内存占用——要么承受沉重的同步惩罚。

有了 cudaMallocAsync，内存管理被无缝地整合进了 CUDA streams。这让每个 stream 的分配与释放得以进行，而不会触发全局的设备同步。

此外，流序分配器让你能够为变长缓冲区——例如 token 缓存或中间激活——发起细粒度的内存请求。随后你可以立即启动依赖这些缓冲区的核函数。这一切都发生在同一个 stream 内。

> 在实践中，达到峰值吞吐需要仔细调节数据块大小，并保持在 GPU 的并发上限之内。现代 GPU 设备提供多个拷贝引擎，能够重叠主机到设备（H2D）与设备到主机（D2H）的传输。查询 deviceProp.asyncEngineCount 以确定你的设备支持多少个拷贝引擎，从而据此规划重叠。

现代 GPU 对一台设备上所有 SM 能够并发运行的核函数数量有一个硬性上限（最多 128 个驻留网格上限）。正如第 5 章所讨论的，现代 GPU 的上限是每台设备 128 个并发执行的核函数。一旦超过活跃核函数的上限，额外的核函数启动就会排队等待，直到某个 SM 上腾出一个槽位。

另外请记住，共享同一个 SM 的核函数只有在其寄存器、共享内存与线程块需求的总和能装进该 SM 的资源上限时，才会一起执行。平衡数据块（tile）大小、启动顺序与每个核函数的资源占用至关重要。

如果数据块太小，你会让拷贝引擎与 SM 资源得不到充分利用。如果数据块太大，或同时入队的核函数太多，你就会超出核函数槽位或耗尽每个 SM 的资源。这会导致停顿。

简而言之，只要调节得当，CUDA streams 与流序内存分配器（cudaMallocAsync）相结合，就能确保数据传输、核函数执行与内存管理无缝重叠。这让多个 DMA 引擎与 SM 保持繁忙，而不产生不必要的排队。

## 在 LLM 中使用 CUDA Streams 与流序内存分配器

CUDA streams 的非阻塞行为与流序内存分配器相结合，对 LLM 训练与推理工作负载至关重要。这些工作负载会跨多个 stream 重叠计算与数据搬运，以提高 GPU 利用率、降低端到端延迟。

此外，LLM 会使用即时的“暂存内存”分配，这正是由上一节讨论的流序内存分配器来支撑的。例如，在运行一个 transformer 层时，你常常需要额外的共享内存或设备内存，即所谓的 *scratch memory*，来存放一次矩阵乘法的结果，然后再把它送入 softmax 操作。

由于 LLM 工作负载中不同的小批量在长度（token 数）上可能各不相同，你会希望使用流序内存分配器，专门为每个输入批次在 GPU 上提供一块全新的暂存缓冲区。这样，你为该批次的中间计算分配的空间恰好够用——一个字节都不多。

如果你用旧的、阻塞式的分配 API（cudaMalloc(...) 与 cudaFree(...)）来即时分配这些暂存缓冲区，那么每一次分配或释放都会与整个 GPU 同步，因为调用 cudaMalloc(...) 会强制一次全局设备同步。如此一来，在所有待处理的核函数与拷贝完成之前，任何重叠都无从谈起。

> 全局设备同步对性能是绝对灾难性的。避免在你的 CUDA streams 中使用像 cudaMalloc() 与 cudaFree() 这样的阻塞调用。优先使用 events 与 stream 等待。而且绝对要避免用 cudaStreamSynchronize(0) 在默认 stream 0 上做同步！

在这样一条流水线里——一个 stream 正忙于为批次 *N* 运行 attention 核函数，另一个 stream 正在为 attention 核函数准备批次 N+1——在第二个 stream 上调用阻塞式的 cudaMalloc(...) 会停顿所有 stream。在分配器完成之前，每个 SM 实际上都被暂停了。这会彻底抹掉你原本指望在数据传输、计算与内存管理之间实现的任何重叠。

解决办法是使用带 cudaMallocAsync() 与 cudaFreeAsync() 的流序分配器。这些 API 在 stream 层面把分配与释放设备内存区域的工作入队。因此，它们只在 stream 层面同步——而非设备层面。

例如，设想有一个 stream 需要一块 16 MB 的暂存缓冲区，用于在一批输入数据上做 attention。它会调用 cudaMallocAsync(&scratchPtr, scratch Bytes, stream1)，这会把该分配请求记录到它的操作队列中，但并不会强制任何其他 stream 等待，如图 11-3 所示。

![图 11-3. 流序内存分配](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-3.png)

其他 stream 会继续启动核函数、拷贝数据，或做它们正在做的任何事——即便 stream 1 的分配还在途（in-flight）中。而一旦 CUDA 运行时在幕后预留好内存，stream 1 就能再次推进，并把 attention 核函数启动到那块新分配的区域中——这一切都不会让任何其他 stream 停顿。

> 与传统的 cudaMalloc 不同，cudaMallocAsync 不会停顿其他 stream。每次分配只在其自身的 stream 内同步。

在 LLM 训练与推理的语境下，这一点尤为宝贵，因为变长序列常常导致暂存缓冲区大小的波动。如果批次 *N* 每个序列有 512 个 token，而批次 N+1 有 1,024 个 token，那么你的 attention 模块为批次 N+1 需要的空间会比批次 *N* 更多，因此复用批次 *N* 的分配是不够的。有了 cudaMallocAsync()，你可以为更大的缓冲区入队一次非阻塞的分配，而不必把所有其他 stream 都拖停。

另外，典型 LLM 的自回归逐 token 生成（又称 *decoding*）阶段会用到一个不断增长的键/值缓存。每生成一个 token，就需要向每个序列各自的缓冲区追加新的 KV 对。随着缓冲区增长，你需要重新分配或扩展暂存区域。cudaMallocAsync(...) 让你可以在运行 attention 核函数的同一个 stream 中完成这件事。与此同时，上游的数据加载与下游的结果拷回操作会在它们各自的 stream 中并行地继续推进。

CUDA streams 在 LLM 语境下的另一种用法是大型 LLM 中的按层流水线化（layerwise pipelining）。假设你把一个大型的、基于 transformer 的 LLM 模型切成两半，让它们在不同的 CUDA stream 上运行——stream 0 运行第 0–5 层，stream 1 运行第 6–11 层。在这两半之间，你需要中间缓冲区来存放激活。

每当 stream 0 在一个小批量上完成其工作，它就可能调用 cudaMallocAsync(...) 为下一批次的激活抓取一块新缓冲区。由于该调用不会同步设备，stream 1 可以在上一批次的结果上继续计算第 6–11 层，同时 stream 0 为下一批次的输入分配内存。

相比之下，如果你在那条流水线里使用了传统的 cudaMalloc(...)，那么每当你为下一个小批量或一个扩大的 KV 缓存分配一块新的暂存区域时，整个 GPU 都会暂停，直到分配完成。那会破坏跨 stream 重叠计算与数据搬运的任何可能。

总结一下，在 LLM 语境下，你经常需要为 attention、层归一化、softmax、KV 缓存或中间激活准备临时缓冲区。它们被统称为 *scratch buffer*（暂存缓冲区）。

在相互独立的 stream 中用 cudaMallocAsync(...) 与 cudaFreeAsync(...) 来管理这些暂存缓冲区，可确保内存管理绝不会强制一次全局的、跨 stream 的停顿。相反，分配会入队到与你的核函数或拷贝操作相同的 stream 中。

这让所有其他 stream 得以继续运行，并让你的 attention 核函数、数据传输以及任何主机侧工作尽可能地重叠。这在大规模、实时的 LLM 工作负载中最大化了 GPU 利用率。

## 传统默认流

当你没有显式创建或指定 stream 时，操作会进入传统默认流（legacy default stream），常被称为 *stream 0*。默认情况下，stream 0 有两个值得强调的重要行为：

*与自身的隐式同步* 入队到 stream 0 的任意两个操作会严格地一个接一个执行。你无法在 stream 0 中重叠两个核函数，或重叠一次拷贝与一个核函数，因为 stream 0 会串行化它自己的所有命令。

*与其他 stream 的隐式同步* 在传统默认流模型中，任何启动到 stream 0 的操作，在开始之前都会等待其他每一个 stream 中先前入队的所有工作完成。反过来，任何启动到非默认 stream 的操作，也会阻塞，直到 stream 0 中先前的所有工作完成。实际上，stream 0 就像整个 GPU 上的一道全局“屏障”。即便你把命令发起到不同的 stream 上，一旦你向 stream 0 提交某个操作，它就会迫使其他每一个 stream 停顿，直到 stream 0 赶上进度，反之亦然。这对性能非常糟糕，应尽可能避免。

由于这些隐式依赖，把所有工作都放进 stream 0 会阻止任何形式的并发。例如，核函数与拷贝引擎无法重叠。如此一来，你的 GPU 就会把时钟周期浪费在空闲地等待默认流屏障清空上。

要释放真正的并行性，除了那些确实需要与其他每一个 stream 串行化的操作（这种情况相对罕见）之外，你应当避免把任何东西放进 stream 0。

## 现代每线程默认流

为缓解传统默认流的“全局屏障”行为，CUDA 引入了每线程默认流（per-thread default stream），有时缩写为 PTDS（与传统 stream 多年来给我们带来的创伤后应激障碍（posttraumatic stress disorder，PTSD）相对）。

在每线程默认流语义下，每个 CPU 线程的默认流都是相互独立的。换句话说，当启用每线程默认流时，每个主机线程都拥有它自己的隐式“stream 0”。

入队到线程 A 默认流的操作不会等待线程 B 默认流中的工作。只要硬件资源允许，它们就会并发运行。同样地，线程 B 默认流中的操作也不会等待线程 A 的默认流，如此类推。

PTDS 被广泛用于多线程 CUDA 应用中，以规避“主机级屏障”问题。要启用 PTDS，你可以用 nvcc --default-stream per-thread 编译代码，或设置 CUDA_API_PER_THREAD_DEFAULT_STREAM=1 环境变量（在包含任何 CUDA 头文件之前）。

> 一旦 PTDS 生效，每个主机 CPU 线程的默认流就表现得像一个用户创建的 stream，不会与其他线程的默认流隐式同步。如果你在同一进程中把 PTDS 与传统默认流混用，PTDS stream 仍会与传统默认流同步。

有了 PTDS，任何不带显式 stream 参数的核函数启动、拷贝或分配都会进入一个线程本地的队列。只有同一主机线程默认流内的命令会串行化，而它们绝不会对属于其他线程的 stream 施加隐式的全局屏障。

简而言之，通过启用每线程默认流，传统默认流的同步屏障就被移除了。每个主机线程的默认流绝不会等待其他线程的 stream。这允许跨线程完全重叠多次核函数启动。

而且，如果你从不同的 CPU 线程发起核函数启动（或内存拷贝）而不指定显式 stream，那么只要资源允许，这些操作就会在 GPU 上重叠。这在图 11-4 中有所展示。

![图 11-4. 时间线，展示在启用 PTDS 的情况下，从不同线程在各自默认流上发起、跨相互独立的 CUDA stream 并发运行的多个 GPU 核函数](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-4.png)

## 显式（非默认）流 vs 默认流

依赖默认流行为最终会引发问题。它总是会。任何入队到传统默认流（stream 0）的工作都会隐式地等待——并阻塞——其他每一个 stream，反之亦然。

在性能关键的代码中，最好创建并使用你自己的、非默认的、显式且具名的 stream，这样就不会有东西意外地落进 stream 0。如果你不小心在 stream 0 上用了一个核函数——或往里拷了数据——你就可能停顿其他每一个活跃的 stream。许多库，如 cuBLAS、Thrust 等，都接受一个显式的 stream 参数。建议你始终创建显式 stream 并使用它们。

> 在 PyTorch 中，操作会在底层被调度到非默认 stream 上，以避免意料之外的同步。例如，PyTorch 对 cuDNN、cuBLAS 等的内部调用使用它们自己的 stream，以避免阻塞默认 stream 0。此外，PyTorch 的分布式后端会把 NCCL 通信操作启动到独立的 CUDA stream 上，而非默认 stream。这让它得以（举例而言）把梯度通信与计算重叠起来。另外，NCCL 的通信操作常常运行在一个高优先级 stream 上，这一点我们稍后会讲到。

通过管理你自己的 stream——或使用每线程默认流——你就保有了对并发的掌控。下面是一个在 CUDA C++ 中创建显式、非默认 stream 的示例（我们会在第 13 章和第 14 章展示如何在 PyTorch 中使用 stream）：

```
cudaStream_t streamA, streamB;
cudaStreamCreateWithFlags(&streamA, cudaStreamNonBlocking);
cudaStreamCreateWithFlags(&streamB, cudaStreamNonBlocking);
myKernel<<<grid, block, 0, streamA>>>(...);                 // streamA
cudaMemcpyAsync(dest, src, size, cudaMemcpyHostToDevice, streamB); // streamB
```

这里，streamA 与 streamB 可以自由重叠。然而，在传统默认流模型下（PTDS 未启用），任何随后进入 stream 0 的调用都会迫使 streamA 与 streamB 都等待，直到 stream 0 清空。

同样地，如果 stream 0 仍有待处理的任务，那么入队到 streamA 或 streamB 的任何工作都会阻塞。要避免这些隐藏的全局屏障，就让 stream 0 保持空闲，只把它用于初始设置、最终清理等一次性操作。

简而言之，启用每线程默认流，让每个 CPU 线程的默认流不再与任何其他线程的默认流同步。然后为所有性能关键的核函数与拷贝创建并使用显式 stream（如 streamA 与 streamB）。

同时做到这两点，你入队到显式 stream 中的任何东西就都不会意外地与另一个显式 stream、另一个线程的默认流，或传统默认 stream 0 中的工作相撞。这确保了安全、可预测的重叠，而没有隐式同步。用 cudaStreamNonBlocking 创建 stream 可确保它们不与传统默认流同步。这是避免隐藏屏障所必需的。

## 默认流使用最佳实践

由于默认流可能对性能造成困扰，让我们来强调每种 stream——传统默认流、每线程默认流与显式（非默认）流——的同步特性：

*传统默认流（*cudaStreamLegacy*）* 它会阻塞其他每一个 stream，也会被其他每一个 stream 阻塞。如果你需要任何形式的并发，就不要在这里发起工作。

*每线程默认流（*cudaStreamPerThread*）* 每个主机线程的默认流都是私有的。它仍会串行化自己的命令，但不会等待或阻塞任何其他线程的默认流或显式 stream。

*显式流（用* cudaStreamCreateWithFlags() *创建）* 显式流是相互独立的队列，只有当你显式插入依赖（例如使用 cudaStreamWaitEvent()）时才会彼此同步。

下面给出一些最佳实践，帮助你把握默认流与显式流（非默认流）的用法：

*绝不要把性能关键的核函数放进 stream 0（传统默认流），除非你有意串行化所有 GPU 工作* 哪怕只有一个走散的核函数或拷贝进入 stream 0，也会拖住所有其他活跃的流。举例来说，较老的 CUDA API 可能会隐式使用 stream 0。例如，当你在不指定显式流的情况下调用 CUDA 驱动 API 时，很可能就会用到传统流。这也是应当迁移到较新 API、或始终显式指定流的又一个理由。

*启用每线程默认流* 如果你的应用使用多个 CPU 线程、每个线程都在向 GPU 排入工作，就应使用 PTDS。这可以避免这些线程的默认流之间出现隐藏的主机端屏障。在现代系统上，你也可以用 cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) 来创建一个不与 stream 0 同步的流。

*创建并管理显式的、非默认的流* 显式流是一项最佳实践，因为它们允许核函数与内存拷贝重叠。始终向 <<<...>>>()、cudaMemcpyAsync() 或 cudaMallocAsync() 传入一个非默认的 cudaStream_t。这样可以保证不会有隐式的默认流同步来干扰你的流水线化工作流。

*使用* cudaStreamWaitEvent() *来协调细粒度依赖，而不是* cudaStreamSynchronize() 你应当使用 stream 事件，而不是在默认流上使用 cudaStreamSynchronize()。只在定义明确的全局节点（例如一个模型训练 epoch 的结尾）才调用 cudaStreamSynchronize()，以避免拖住不相关的流。

*对你的流标志保持显式* 如果你要启用 PTDS，请在任何 CUDA 调用之前设置好设备标志。否则你仍处于传统默认流模式。任何对 stream 0 的涉及都会制造一道全局屏障。

简而言之，传统默认流（stream 0）始终扮演一道全局屏障。你排入其中的任何工作都会等待、并强制所有其他流完成；而只要 stream 0 有挂起的工作，所有其他流也都会等待。

要避免这些看不见的停顿，就不要把性能关键的核函数或拷贝放进 stream 0。相反，应使用 cudaStreamCreateWithFlags 创建你自己的具名流，并把一切都启动到这些流中，让它们能够独立运行。

如果你的程序有多个 CPU 线程——每个线程都发出各自的 GPU 工作——你还应启用每线程默认流。启用 PTDS 后，每个线程的“默认”流就不再与其他线程的默认流、也不再与 stream 0 同步。

这样一来，即使那些没有显式创建新流的代码，也不会意外阻塞任何其他线程的工作。在所有情况下，只要你想让两个操作重叠（例如一次拷贝和一个核函数），就应给它们各自独立的、显式的非默认流。如此便可避开 stream 0 的隐藏全局同步规则，让 GPU 以最大并行度运行。

## 用 events 与 callbacks 做细粒度同步

即便多个流与 DMA 引擎能够重叠，也总有一些时候，某个流的操作必须等待另一个流。设想一对生产者—消费者流。生产者 stream 0 正在加载并准备数据，供消费者 stream 1 处理。

在这种情形下，很容易想到在主机端同步这两个流、用一次完整的 cudaDeviceSynchronize() 阻塞 CPU。然而，这会阻塞 CPU，直到 GPU 上所有流的所有操作全部完成。这对性能非常不利。你也可以使用 cudaStreamSynchronize()（如图 11-5 所示），但这会一直阻塞，直到该流队列中的所有操作完成。

![图 11-5. 使用 cudaStreamSynchronize() 会阻塞 CPU，直到该 stream 中的所有操作都已同步](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-5.png)

相反，你可以使用 CUDA events，为流提供一种粒度细得多的同步机制。有了 CUDA events，你在产出数据的流中记录一个 cudaEvent_t，然后让消费者流去等待这个事件。

例如，你会在 stream 0 上启动一个生产者核函数，并在数据就绪可供消费时调用 cudaEventRecord(doneEvent, stream0)。随后在 stream 1 中，你会在启动消费者核函数之前调用 cudaStreamWaitEvent(stream1, doneEvent, 0)。这样一来，只有 stream 1 会停下来等待该事件被记录——主机线程和所有其他流都会继续执行，如图 11-6 所示。

![图 11-6. 使用 CUDA events 进行细粒度同步](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-6.png)

除了用 CUDA events 在流间协调之外，你还可以用它们在 GPU 设备与 CPU 主机之间通信。为此，你会在主机端用 cudaLaunchHostFunc() 注册一个主机回调。

假设你需要在某个 GPU 核函数于 stream 0 中一完成，就立刻在主机端回收一个自定义内存池。在这种情况下，你会在主机端用 cudaLaunchHostFunc() 注册一个回调，并指定你所关心的事件。然后在 stream 0 上启动该核函数。

当核函数完成时，它会记录该事件，主机随即运行回调函数、更新 CPU 上的内存池。这一切都无需轮询循环，也无需一次完整的 cudaDeviceSynchronize()。

一旦 GPU 工作完成，CUDA 运行时便会在一个主机线程上执行回调函数。这让 CPU 保持空闲，直到你恰好需要它的那一刻——从而避免浪费主机周期、也避免阻塞不相关的 GPU 工作。

你不应在用 cudaLaunchHostFunc() 启动的主机回调内部调用任何设备端 GPU API。举例来说，如果你在回调内部调用 cudaMemcpy()，就可能死锁，因为该回调运行在由 CUDA 运行时管理的主机线程上。在其中调用 CUDA API 会导致死锁，因为设备可能正在等待该回调完成。

> 回调应当仅限于 CPU 端任务，例如释放或回收主机内存。这条建议可以避免这样一种环形死锁：回调试图在 GPU 上启动新工作，而与此同时 GPU 正在等待该回调完成。

## 用 CUDA events 做跨流同步

当多个流并行运行时，我们常常需要在它们之间进行协调与同步。CUDA events 是跨流同步（cross-stream synchronization）的主要机制，既不会拖住 CPU，也不会拖住整个设备。

事件就像一个标记，由某个流在特定节点记录下来。其他流、甚至主机，都可以等待这个标记，从而得知某个事件何时发生（例如某个核函数已完成）。与会一直阻塞到所有流完成的完整 cudaDeviceSynchronize() 不同，事件允许在流之间进行细粒度的排序。

流可用于确保每个 transformer 层的数据在计算之前都已就位。它们还能改进流水线并行——确保 GPU 0 已产出某个张量之后，GPU 1 才去消费它。而这一切发生的同时，其他独立工作都不会闲置。

例如，设想一组四个流（Streams A–D），它们对操作进行流水线化，并在某些情况下彼此依赖。我们可以用 CUDA events 来强制这些依赖（见图 11-7）。

为确保 stream B 在 stream A 产出数据之前不会开始处理，我们可以在 stream A 中记录一个事件，让 stream B 先等待它再继续。类似地，stream D 可以如图所示等待 stream B。通过这样串联事件，我们在流之间维持正确的排序，同时仍能并发地运行不同的批次。

这种基于事件的同步在深度学习框架中被大量用于让梯度计算与 all-reduce 通信重叠。计算流在梯度就绪时记录一个事件，通信流等待该事件以启动一次 NCCL 操作，从而让通信与剩余计算重叠。

简而言之，CUDA events 轻量且针对设备信号做了优化。当某个流到达其命令队列中的特定节点时便会记录一个事件。而其他流可以高效地轮询/等待它。它们让我们能够在流之间编排复杂的依赖图，而无需强制全局等待。它们还提供了在多流 LLM 工作负载中实现流水线化执行所必需的控制力。

![图 11-7. 用 events 同步 CUDA streams（source: https://oreil.ly/MynOA）](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-7.png)

> NVIDIA 持续改进事件的时间粒度、并降低事件记录开销。因此，事件也是性能剖析的良好选择，因为它们会在 GPU 时间线上记录事件时间戳、可用于测量执行时间等。

## 把核内 warp 专门化与核间 CUDA streams 结合的流水线（Pipelining with Warp Specialization (Intra-Kernel) and CUDA Streams (Inter-Kernel)）

第 10 章演示了如何用 warp 专门化和 CUDA Pipeline API 在单个核函数内隐藏内存延迟。在那一节里，我们启动了一个单一的、常驻的、warp 专门化的核函数，其中每个线程块被拆分为三种不同类型的 warp：loader、compute 和 storer warp。这些 warp 共享一块连续的共享内存，以及一个两阶段（双缓冲）的 CUDA Pipeline（<cuda::pipeline>）对象来协调各自的工作。

让我们保留那个 warp 专门化的设备核函数，现在改用各自独立的 CUDA streams 驱动多个批次通过它。其结果是一个如下的两级流水线：

*核内重叠（每个块内部）* loader ↔ compute ↔ storer 使用各自独立的 pipeline 并发运行。

*核间重叠（跨批次之间）* 当 stream 0 在批次 *t* 上计算时，stream 1 DMA 加载批次 t+1，而 stream 2 返回批次 t–1 的结果。这用到了非阻塞流、固定主机内存和流序内存分配器，以便分配/拷贝不会串行化其他工作。

如果多个块需要同一个 tile，第 10 章描述的线程块簇 + DSMEM 路径可以消除跨块的冗余全局加载。不过，协作式/簇启动会与其他协作式核函数串行化。这里所用的流模式针对的是主机 ↔ 设备的重叠，并适用于非协作式核函数。

> 如果你的瓶颈是设备上的 tile 复用，就使用线程块簇。如果瓶颈是主机 ↔ 设备的通信与批处理，就使用 CUDA streams。我们稍后会展示如何把二者结合，但这是初步决策的一个良好起点。

下一个核函数每个块使用三个 warp（0 = loader，1 = compute，2 = storer）和两个 pipeline 来乒乓推进各阶段。因此，它需要 6 * TILE_SIZE * sizeof(float) 的动态共享内存（[A0|B0|C0|A1|B1|C1]）。值得注意的是，我们改为使用两个独立的、块作用域的 pipeline，每个深度为 2。一个用于 loader → compute（pipe_lc）的交接，另一个用于 compute → storer（pipe_cs）。

此外，我们使用双缓冲共享内存，这样核函数就能在计算 tile *i* 的同时加载 tile i+1、并存储 tile i–1。这就是为什么在接下来的代码里你会看到两个 cuda::pipeline 对象，以及共享内存中的“[A|B|C] × 2 stages”。这两个 pipeline 是一个核内选择，用于加深每个核函数内部的重叠。（注：你可以对任一种核函数风格使用流。）代码如下：

```
#include <cooperative_groups.h>     // thread_block, etc.
#include <cuda/pipeline>            // CUDA Pipeline API
namespace cg = cooperative_groups;
// shmem bytes = 2(stages)×3(buffers)×TILE_SIZE×sizeof(float)=6×TILE_SIZE×4
//             = 6 * 1024 * 4 = 24,576 B (24 KB) << 227 KB SMEM
// We keep this at 1024 (versus going higher) as a safe starting point
// (good occupancy balance)
#define TILE_SIZE 1024  // one tile = 1,024 floats per buffer (1-Dimension)
    // Alignment / size guards for vectorized copies
    static_assert((TILE_SIZE % (32 * 4)) == 0,
                  "TILE_SIZE must be multiple of 128 for float4 vectorization");
    // If you cannot guarantee 16B alignment or sizes, handle
    //  the tail/ragged edges with a fallback 4B loop.
// Three warps per block: 0 loads, 1 computes, 2 stores.
// Two block-scoped pipelines (each depth=2) implement ping-pong across tiles.
__global__ void warp_specialized_two_pipelines(
    const float* __restrict__ A_global,
    const float* __restrict__ B_global,
    float*       __restrict__ C_global,
    int          numTiles)
{
    thread_block cta = this_thread_block();
    // Stage s∈{0,1}: [A_s | B_s | C_s], each length TILE_SIZE
    extern __shared__ float shared_mem[];
    // loader -> compute pipeline (2 in-flight stages)
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> state_lc;
    auto pipe_lc = cuda::make_pipeline(cta, &state_lc);
    // compute -> storer pipeline (2 in-flight stages)
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2> state_cs;
    auto pipe_cs = cuda::make_pipeline(cta, &state_cs);
    const int warp_id = threadIdx.x >> 5;   // 0,1,2
    const int lane_id = threadIdx.x & 31;
    // Prime first tile handled by this block
    const int first = blockIdx.x;
    if (warp_id == 0 && first < numTiles) {
        const int stage0 = first & 1;
        float* A0 = shared_mem + stage0 * 3 * TILE_SIZE;
        float* B0 = A0 + TILE_SIZE;
pipe_lc.producer_acquire();
#pragma unroll
for (int chunk = 0; chunk < TILE_SIZE; chunk += 32 * 4) {
    cuda::memcpy_async(
        cta,
        reinterpret_cast<float4*>(A0 + chunk) + lane_id,
        reinterpret_cast<const float4*>(A_global + size_t(first) *
          TILE_SIZE + chunk) + lane_id,
        sizeof(float4),
        pipe_lc);
    cuda::memcpy_async(
        cta,
        reinterpret_cast<float4*>(B0 + chunk) + lane_id,
        reinterpret_cast<const float4*>(B_global + size_t(first) *
          TILE_SIZE + chunk) + lane_id,
        sizeof(float4),
        pipe_lc);
}
pipe_lc.producer_commit();
    }
    // Walk a strided set of tiles; ping-pong stage by tile parity.
    for (int tile = blockIdx.x; tile < numTiles; tile += gridDim.x) {
        const size_t offset = size_t(tile) * TILE_SIZE;
        const int    stage  = tile & 1;
        float* A_buf = shared_mem + stage * 3 * TILE_SIZE;
        float* B_buf = A_buf      + TILE_SIZE;
        float* C_buf = B_buf      + TILE_SIZE;
        // Loader prefetches next tile while compute/store work on current.
        if (warp_id == 0) {
            const int next = tile + gridDim.x;
            if (next < numTiles) {
                const int next_stage = next & 1;
                float* A_next = shared_mem + next_stage * 3 * TILE_SIZE;
                float* B_next = A_next + TILE_SIZE;
                pipe_lc.producer_acquire();
                #pragma unroll
                for (int chunk = 0; chunk < TILE_SIZE; chunk += 32 * 4) {
                    cuda::memcpy_async(
            cta,
                        reinterpret_cast<float4*>(A_next + chunk) + lane_id,
                        reinterpret_cast<const float4*>(A_global + size_t(next) *
                          TILE_SIZE + chunk) + lane_id,
                        sizeof(float4),
                        pipe_lc);
                    cuda::memcpy_async(
                        cta,
                        reinterpret_cast<float4*>(B_next + chunk) + lane_id,
                        reinterpret_cast<const float4*>(B_global + size_t(next) *
                          TILE_SIZE + chunk) + lane_id,
                        sizeof(float4),
                        pipe_lc);
                }
                pipe_lc.producer_commit();
            }
        }
        // Compute consumes loader output and signals storer
        if (warp_id == 1) {
            pipe_lc.consumer_wait();
            #pragma unroll
            for (int chunk = 0; chunk < TILE_SIZE; chunk += 32) {
                C_buf[chunk+lane_id] = A_buf[chunk+lane_id]+B_buf[chunk+lane_id];
            }
            pipe_lc.consumer_release();
            pipe_cs.producer_acquire();
            pipe_cs.producer_commit();
        }
        // Storer waits on compute and writes back
        if (warp_id == 2) {
            pipe_cs.consumer_wait();
            #pragma unroll
            for (int chunk = 0; chunk < TILE_SIZE; chunk += 32) {
                C_global[offset + chunk + lane_id] = C_buf[chunk + lane_id];
            }
            pipe_cs.consumer_release();
        }
    }
    // Launch requirements: blockDim.x = 96 (3 warps), gridDim as needed,
    // dynamic shared memory = 6 * TILE_SIZE * sizeof(float).
}
```

在此，我们要指出一些值得关注的性能要点。首先，我们要求 A0、B0、A_global + base 和 B_global + base 满足 16 字节对齐。如果某条路径不是 16 字节对齐的，我们就对未对齐的序言/收尾退回到 4 字节的循环。

> 每 lane 16 字节（float4/int4）符合现代 GPU 的最佳实践，可在 warp 层级达成 128 字节的合并访问。这有助于降低流水线开销。

其次，我们假设 TILE_SIZE 是 128（32 lanes × 4 floats）的倍数。如果不是，就用标量循环处理尾部。此外，我们对 float4 使用 cuda::memcpy_async。这既保留了 pipeline 的异步语义，又把每个 tile 的拷贝计数从每 lane 4× 降到 1×。

下面是主机端驱动程序，它在非阻塞流之间轮转各批次，并使用固定主机内存和流序内存分配器（cudaMallocAsync ÷ cudaFreeAsync）。它会启动前一段代码中的 warp_specialized_two_pipelines：

```
#include <cstdio>
#include <cuda_runtime.h>
#define TILE_SIZE   1024
#define NUM_STREAMS 2
#define BATCHES     8
// Device kernel
__global__ void warp_specialized_two_pipelines(
    const float* __restrict__ A_global,
    const float* __restrict__ B_global,
    float*       __restrict__ C_global,
    int          numTiles);
int main() {
    // Create nonblocking streams (do NOT use legacy default stream)
    cudaStream_t s[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);
    // Pinned host buffers so cudaMemcpyAsync truly overlaps
    float *hA = nullptr, *hB = nullptr, *hC = nullptr;
    const size_t bytesPerBatch = TILE_SIZE * sizeof(float);
    cudaMallocHost(&hA, BATCHES * bytesPerBatch);
    cudaMallocHost(&hB, BATCHES * bytesPerBatch);
    cudaMallocHost(&hC, BATCHES * bytesPerBatch);
    // Initialize inputs
    for (int b = 0; b < BATCHES; ++b) {
        for (int i = 0; i < TILE_SIZE; ++i) {
            hA[b * TILE_SIZE + i] = float(i);
            hB[b * TILE_SIZE + i] = 1.0f;
        }
    }
    // Enqueue batches in a round-robin across streams
    for (int b = 0; b < BATCHES; ++b) {
        cudaStream_t st = s[b % NUM_STREAMS];
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        cudaMallocAsync(&dA, bytesPerBatch, st);   // stream-ordered allocator
        cudaMallocAsync(&dB, bytesPerBatch, st);
        cudaMallocAsync(&dC, bytesPerBatch, st);
        cudaMemcpyAsync(dA, hA + size_t(b) * TILE_SIZE, bytesPerBatch,
                        cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(dB, hB + size_t(b) * TILE_SIZE, bytesPerBatch,
                        cudaMemcpyHostToDevice, st);
        const dim3 block(96);         // 3 warps: loader/compute/storer
        const dim3 grid(1);
        const size_t shmem = 6 * TILE_SIZE * sizeof(float); //[A0|B0|C0|A1|B1|C1]
        // Each batch is one tile in this 1-D example
        warp_specialized_two_pipelines<<<grid, block, shmem, st>>>(
            dA, dB, dC, /*numTiles=*/1);
        cudaMemcpyAsync(hC + size_t(b) * TILE_SIZE, dC, bytesPerBatch,
                        cudaMemcpyDeviceToHost, st);
        cudaFreeAsync(dA, st);        // stream-ordered free
        cudaFreeAsync(dB, st);
        cudaFreeAsync(dC, st);
    }
    // Clean up
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(s[i]);
        cudaStreamDestroy(s[i]);
    }
    cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC);
    return 0;
}
```

在这里，每个流承载着自己的一串序列：allocate → H2D → kernel → D2H → free。因为分配与拷贝是按流顺序排入的，且主机缓冲区是固定的，所以 GPU 的拷贝引擎可以让 H2D/D2H 与另一个流的 SM 计算重叠。这就是核间层，它与第 10 章引入的、由 warp 专门化所产生的核内重叠互为补充。

具体来说，这个例子展示了共享内存持有两组、每组三个长度均为 TILE_SIZE 的缓冲区，从而在准备下一个 tile 的同时可以使用其中一组。一个两阶段的 <cuda::pipeline> 在各 tile 之间提供双缓冲，并强制正确的排序，使 loader、compute 和 storer warp 能够在不同的 tile 上重叠。核函数在循环之前先给第一个 tile 预热，然后在计算 tile i 的同时预取 tile i 加 gridDim.x。

单独来看，这个 warp 专门化核函数通过在每个 tile 内重叠三种角色来隐藏内存延迟。然而，单个 warp 专门化核函数一次只能处理一批 tile。

为了让 GPU 忙于处理多个批次——并跨这些批次重叠 H2D 传输、核函数计算与 D2H 传输，我们在各自独立的 CUDA streams 中启动这同一个 warp 专门化核函数的多个实例。这将为每个批次使用流序内存分配器。

这第二层流水线化，即核间并发，让我们能够把接连而来的小批量（mini-batch）一波波送过 GPU。这样一来，当一个批次的核函数在 stream 0 中计算时，另一个批次的数据仍在 stream 1 中抵达。而与此同时，前一个批次的结果可能正在 stream 2 中流回主机。

在实践中，我们会挑选少量的流，两个或三个，并循环使用它们。每个流用 cudaMallocAsync 分配自己的设备缓冲区，用 cudaMemcpyAsync 异步拷贝输入数据，启动 warp 专门化核函数来处理这些缓冲区，把结果异步拷回主机，然后用 cudaFreeAsync 释放缓冲区。

因为这些操作中的每一个都被排入了特定的流，所以它们都可以与其他流中的等价操作重叠。在带有多个拷贝引擎和流序内存分配器的现代 GPU 上，这一模式能够通过同时压满芯片的每一个部分，来大幅提升利用率。实际能达成的重叠受限于拷贝引擎数量（查询 cudaDeviceProp::asyncEngineCount）、带宽以及核函数占用率。

一旦某个核函数中的 loader warp 正在等待一次全局内存取数，下一批次的 H2D 拷贝就已在拷贝引擎上在途（in-flight）。一旦前一批次的 storer warp 正在回写全局内存，另一个流中的分配器就可以为下一批次抓取内存，而无需强制一次全局同步。

本质上，第 10 章的 warp 专门化例子教会了我们如何让单个核函数通过在一个线程块内重叠 load/compute/store 来隐藏其内存延迟。本章的多流例子在此基础上更进一步，展示了如何通过在整条流水线上重叠主机 ↔ 设备传输、计算和设备 ↔ 主机传输，把许多这样的核函数——每个处理一个不同的批次——同时送入 GPU。

现代 GPU 拥有多个拷贝引擎，并执行异步内存分配。它们与 Tensor Core 一道并行工作，构成一种两级流水线化策略——核内的 warp 专门化与核间的流。这些机制让我们的核函数能够为 LLM 工作负载逼近硬件的峰值利用率。

## warp 专门化 + 线程块簇 + CUDA streams（Warp Specialization with Thread Block Clusters and CUDA Streams）

现在我们要把本章——以及此前各章——的所有内容整合起来，通过多个流驱动多个在途（in-flight）的小批量，而每个流都启动一个协作式的、线程块簇化的、warp 专门化的核函数。这代表了在最新 GPU 硬件上进行 CUDA 性能优化的巅峰。

> 尽管我们出于完整性考虑覆盖这一主题，但需要指出的是，由于其复杂性，这种技术组合在非常专门化的研究项目和对超低延迟敏感的推理引擎之外很少见到。不过它仍然值得讲述，因为它把我们目前学到的许多概念融进了同一个代码示例。

第 10 章展示了在单个线程块内部的 warp 专门化，也展示了用 DSMEM 把 warp 专门化与线程块簇结合起来。这样一来，一个主导线程块只加载一次 tile——而线程块簇中的每个块都从这份共享的片上副本进行计算。这消除了重复的全局加载。

这个例子复用了完全相同的模式：主导块用一个块作用域的 pipeline 来暂存这些拷贝。所有块都通过 cluster.map_shared_rank 进行读取，因此它继承了数据复用的收益。

在前面那个带 CUDA streams 的 warp 专门化例子中，每个线程块都在自己的共享内存区域内负责全部三个流水线阶段——loader、compute 和 storer。现在，让我们把那个较早的实现扩展为使用一个线程块簇 pipeline。这样一来，loader、compute 和 storer warp 就分布在整个线程块簇上。这与之前那个局限于单个线程块的实现形成对比。

线程块簇加 warp 专门化并配合 CUDA streams 的例子代码如下。在这个例子中，我们取 NUM_STREAMS = 2，以便主机可以在各自独立的 CUDA streams 中排入两次独立的启动。我们复用第 10 章“warp 专门化与线程块簇”一节中相同的 warp_specialized_cluster_pipeline 实现，并加入 CUDA streams：

```
// Warp specialization across a thread block cluster using DSMEM
// and a block scoped pipeline, launched from multiple CUDA streams
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <cooperative_groups.h>
#include <algorithm>
namespace cg = cooperative_groups;
#define TILE_SIZE   128   // 3×TILE_ELEMS×4 bytes=196,608 bytes < 227 KB SMEM
#define TILE_ELEMS  (TILE_SIZE * TILE_SIZE)
#define NUM_STREAMS 2
#define CLUSTER_BLOCKS 4  // blocks per cluster along x (tune to device)
// ---- Device helpers ----
__device__ void compute_rows_from_ds(const float* __restrict__ A_src,
                                     const float* __restrict__ B_src,
                                     float*       __restrict__ C_dst,
                                     int row_begin, int row_end, int lane_id)
{
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
// Clustered, warp-specialized kernel (leader loads once, others consume DSMEM)
extern "C"
__global__ void warp_specialized_cluster_pipeline(
    const float* __restrict__ A_global,
    const float* __restrict__ B_global,
    float*       __restrict__ C_global,
    int numTiles)
{
    thread_block cta      = this_thread_block();
    cluster_group cluster = this_cluster();
    extern __shared__ float smem[];
    float* A_tile_local = smem;
    float* B_tile_local = A_tile_local + TILE_ELEMS;
    float* C_tile_local = B_tile_local + TILE_ELEMS;
    // Leader uses a block-scoped pipeline to stage A/B exactly once per tile
    __shared__ cuda::pipeline_shared_state<cuda::thread_scope_block, 2>
        pipe_state;
    auto pipe = cuda::make_pipeline(cta, &pipe_state);
    const int lane_id = threadIdx.x & 31;
    const int warp_id = threadIdx.x >> 5;
    const int cluster_rank       = cluster.block_rank();
    const dim3 cluster_dims      = cluster.dim_blocks();
    const int blocks_in_cluster  = cluster_dims.x*cluster_dims.y*cluster_dims.z;
    // Each iteration handles one tile per cluster (1-D cluster along x)
    auto loader = cg::tiled_partition<32>(cta);
    for (int tile = blockIdx.x / cluster_dims.x; tile < numTiles;
         tile += gridDim.x / cluster_dims.x) {
        const size_t offset = static_cast<size_t>(tile) * TILE_ELEMS;
        // Leader block’s loader warp stages A and B once for the entire cluster
        if (cluster_rank == 0 && warp_id == 0) {
            pipe.producer_acquire();
            cuda::memcpy_async(loader, A_tile_local, A_global + offset,
                               cuda::aligned_size_t<32>{TILE_ELEMS
                               * sizeof(float)}, pipe);
            cuda::memcpy_async(loader, B_tile_local, B_global + offset,
                               cuda::aligned_size_t<32>{TILE_ELEMS
                               * sizeof(float)} pipe);
            pipe.producer_commit();
            // Make visible inside leader before publishing
            pipe.consumer_wait();
            pipe.consumer_release();
        }
        // Publish to all blocks in the cluster
        cluster.sync();
        const float* A_src = cluster.map_shared_rank(A_tile_local, 0);
        const float* B_src = cluster.map_shared_rank(B_tile_local, 0);
        // Divide rows among blocks in the cluster
        const int rows_per_block = (TILE_SIZE + blocks_in_cluster - 1)
                                    / blocks_in_cluster;
        const int row_begin = std::min(cluster_rank * rows_per_block, TILE_SIZE);
        const int row_end   = std::min(row_begin + rows_per_block, TILE_SIZE);
        // Compute warp produces this block’s band into local SMEM
        if (warp_id == 1) {
            compute_rows_from_ds(A_src, B_src, C_tile_local,
                                 row_begin, row_end, lane_id);
        }
        // Ensure the storer sees computed rows
        cta.sync();
        // Storer warp writes band back to global
        if (warp_id == 2) {
            for (int row = row_begin + lane_id; row < row_end; row += warpSize){
                for (int col = 0; col < TILE_SIZE; ++col) {
                    C_global[offset + row * TILE_SIZE + col] =
                        C_tile_local[row * TILE_SIZE + col];
                }
            }
        }
        // Done with this tile; allow leader to reuse buffers
        cluster.sync();
    }
    // Dynamic shared memory size: 3 * TILE_ELEMS * sizeof(float)
}
// ---- Host driver: stream-staged batches + clustered kernel launch ----
void launch_warp_specialized_cluster_pipeline_multistream(
    const float* h_A,        // Host input A: length = numBatches * batchLength
    const float* h_B,        // Host input B: length = numBatches * batchLength
    float*       h_C,        // Host output C: length = numBatches * batchLength
    int          batchLength,// elems per batch; must be multiple of TILE_ELEMS
    int          numBatches)
{
    // Basic validation
    if (batchLength % TILE_ELEMS != 0) {
        fprintf(stderr, "batchLength must be a multiple of TILE_ELEMS (%d)\n",
                TILE_ELEMS);
        return;
    }
    const int numTiles = batchLength / TILE_ELEMS;
    // Nonblocking streams avoid legacy default-stream barriers
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i)
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    // Size and launch geometry
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    // grid.x must be a multiple of CLUSTER_BLOCKS
    const int blocksPerGrid = prop.multiProcessorCount * CLUSTER_BLOCKS;
    const dim3 blockDim(96);  // 3 warps: loader/compute/storer
    const size_t shmemBytes = 3ull * TILE_ELEMS * sizeof(float);
    // Cluster attributes
    cudaLaunchAttribute attr[2]{};
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim = dim3(CLUSTER_BLOCKS, 1, 1);
    attr[1].id = cudaLaunchAttributeNonPortableClusterSizeAllowed;
    attr[1].val.nonPortableClusterSizeAllowed = 1;
    // Enqueue batches round-robin across streams
    for (int b = 0; b < numBatches; ++b) {
        cudaStream_t st = streams[b % NUM_STREAMS];
        float *dA = nullptr, *dB = nullptr, *dC = nullptr;
        const size_t bytes = static_cast<size_t>(batchLength) * sizeof(float);
        // Stream-ordered allocator avoids global sync
        cudaMallocAsync(&dA, bytes, st);
        cudaMallocAsync(&dB, bytes, st);
        cudaMallocAsync(&dC, bytes, st);
        // H2D (host pointers should be pinned for true overlap)
        cudaMemcpyAsync(dA, h_A + static_cast<size_t>(b) * batchLength, bytes,
                        cudaMemcpyHostToDevice, st);
        cudaMemcpyAsync(dB, h_B + static_cast<size_t>(b) * batchLength, bytes,
                        cudaMemcpyHostToDevice, st);
        // Clustered launch config
        void* args[] = { &dA, &dB, &dC, (void*)&numTiles };
        cudaLaunchConfig_t cfg{};
        cfg.gridDim  = dim3(blocksPerGrid);
        cfg.blockDim = blockDim;
        cfg.dynamicSmemBytes = shmemBytes;
        cfg.stream = st;
        cfg.attrs = attr;
        cfg.numAttrs = 2;
        // Launch the clustered kernel (cooperative/cluster launch)
        cudaKernel_t k;
        cudaGetKernel(&k, warp_specialized_cluster_pipeline);
        void* fptr = reinterpret_cast<void*>(k);
        cudaLaunchKernelExC(&cfg, fptr, args);
        // D2H and cleanup
        cudaMemcpyAsync(h_C + static_cast<size_t>(b) * batchLength, dC, bytes,
                        cudaMemcpyDeviceToHost, st);
        cudaFreeAsync(dA, st);
        cudaFreeAsync(dB, st);
        cudaFreeAsync(dC, st);
    }
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
}
```

> 这个例子使用了一次用 cudaLaunchKernelExC 配置的簇启动。协作式启动（包括簇启动）需要足够的资源，才能按启动约束让其各块保持常驻。这往往会限制并发，但在资源允许时，协作式核函数仍可与其他工作交错进行。并发并不被保证，因为它依赖于拓扑与启动方式。（注：在资源限制与启动约束的前提下，其他流中的数据传输与核函数仍可重叠。）

与第 10 章一样，我们在启动核函数时指定一个簇维度，使各个块加入同一个 cluster_group，从而可以访问彼此的分布式共享内存。主导块（leader block）通过块作用域的 cuda::thread_scope_block 暂存其拷贝，而所有块都使用 cluster.map_shared_rank 读取主导块的 tile。

借助簇作用域与分布式共享内存，加载阶段在主导块中每个簇只运行一次，而计算与存储则在簇内各个块上并发执行。与之前一样，每个 warp 的 warp_id 决定其角色，warp 以持久化方式循环遍历所有 tile，在同步的轮转中依次执行取数、计算与存储。

通过加入 CUDA streams，并在各自独立的 CUDA streams 中启动本核函数的独立副本（NUM_STREAMS = 2），我们让 GPU 持续忙于处理多批输入数据。在每个 stream 中，我们执行以下步骤：

1. 用 cudaMallocAsync 为每个线程块分配按批次划分的设备缓冲区。

2. 用 cudaMemcpyAsync 将输入从主机暂存到设备。

3. 用 cudaLaunchKernelExC 启动带簇的 warp 专门化核函数。

4. 用 cudaMemcpyAsync 将输出拷回主机。

5. 用 cudaFreeAsync 释放设备缓冲区。

由于每个 stream 都会将自己的协作式启动连同异步拷贝和释放一并入队，即便一次只运行一个协作式核函数，主机也能让多个 mini-batch 保持在途（in flight）。请记住，GPU 会将协作式核函数启动串行化，因为每个协作式核函数会同时占满所有线程块（这一限制会在代码示例之后再次强调）。

使用多个 stream（NUM_STREAMS = 2）仍然让我们能够把主机端的分配与拷贝和上一个核函数的执行重叠起来。例如，当 stream 0 的线程块簇处理 tile *n* 时，stream 1 可以异步分配缓冲区（cudaMallocAsync）并把 tile n+1 拷入设备内存，而 stream 2 则可以把 tile n–1 的结果写回主机。

> 协作式核函数必须占满它在 GPU 上所需的每一个线程块槽位（CTA）。这会阻止任何其他协作式启动同时运行，因为所有这些 CTA 资源都已被占用。

在实践中，stream 0 发出其协作式启动并开始执行，而 stream 1 可以立即将自己的启动入队。但第二个启动会一直处于挂起状态，直到 stream 0 的核函数结束。

不过，一旦 stream 0 完成，stream 1 的启动会立即开始。这是因为它的输入早已由 cudaMemcpyAsync 暂存完毕——而它的缓冲区也已由 cudaMallocAsync 分配好。

将线程块簇、核内 warp 专门化（三阶段的 cuda::thread_scope_cluster）与核间 CUDA streams 结合起来，我们便构建出跨越多个线程块的两层流水线。这种双层流水线能把硬件推向峰值。

在第一层，线程块簇流水线让 loader、compute 与 storer warp 跨所有线程块协作。线程块簇可以利用 TMA 多播（multicast），把一次全局内存 tile 传输复制到簇内每个块的共享内存中。多播局限于线程块簇内部。本质上，线程块簇对每个 tile 只取一次。在第二层，多个 stream 让主机端的分配、拷贝和核函数启动彼此相互隐藏。

简而言之，这些组合技术确保内存延迟在网格级和主机到设备级都被隐藏起来。这会把现代 GPU 上的硬件利用率推向接近峰值——并为 LLM 工作负载最大化吞吐。

> 在许多场景中，像 DRAM 带宽这样的内存层级瓶颈，以及像 NCCL all-reduce 这样的模型并行通信，会主导整个工作负载。因此，一旦你的核函数已接近饱和 HBM 带宽，并且你的 stream 已经隐藏了 CPU → GPU 延迟，那么在此之上再叠加 warp 专门化、进而再叠加线程块簇所能多换来的那几个百分点的 SM 利用率，往往难以抵消其高昂的工程成本。在实践中，大多数真实（real-world）LLM 训练与推理工作负载，从前面介绍的更简单设计（例如双缓冲核函数，或配合 CUDA streams 的两阶段流水线）中获得的收益已然足够。

## 多 GPU 计算与数据传输的重叠（借助 CUDA Streams）

在跨多个 GPU 训练或服务 LLM 时，CUDA streams 让你能够重叠本地计算、点对点传输、集合通信、数据准备与内存管理，从而让任何设备都不空闲。例如，假设你把一个 transformer 模型切分到两个 GPU 上，让 GPU 0 处理第 0–3 层，GPU B 处理第 4–7 层。在 GPU A 上，你可能会这样写：

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

与此同时，在 CPU 上，你已经解码或准备好了下一个微批量（N+1），并发出了一次 cudaMemcpyAsync，将其拷入一块固定主机缓冲区。这样一来，当 GPU A 的 stream 0 完成批次 *N* 的前向传播时，批次 N+1 的 CPU 到 GPU 拷贝就能立即开始，无需等待。

同时，你调用 cudaStreamWaitEvent(stream1_A, eventA, 0)，以确保 stream1_A 一直等待到 outputActivationsA 就绪后，才发起直接经由 NVLink/PCIe 拷入 GPU B 内存的操作，如下所示：

```
// Stream 1 on GPU A: wait for layer computation, then copy to GPU B
cudaStreamWaitEvent(stream1_A, eventA, 0);
cudaMemcpyPeerAsync(
    destActivationsB, /*dstGPU=*/1,
    outputActivationsA, /*srcGPU=*/0,
    bytes, stream1_A);
```

一旦你记录了 eventA，GPU A 的 stream0 就能立即启动批次 N+1 的下一个前向核函数，并用 cudaMemcpyAsync 把它的输入从固定主机缓冲区拷入设备内存而不发生阻塞。在 GPU B 上，你在拷贝开始或结束时记录一个相匹配的事件：

```
// Stream 1 on GPU A: after peer copy starts,
// record eventFromA
cudaEventRecord(eventFromA, stream1_A);
```

然后在 GPU B 上，你等待数据到达并运行第 4–7 层，如下所示：

```
// Stream 1 on GPU B: wait for data arrival, then run layers 4–7
cudaStreamWaitEvent(stream1_B, eventFromA, 0);
myTransformerLayers4to7<<<gridB, blockB, 0, stream1_B>>>(
    destActivationsB, nextActivationsB);
```

这种显式的事件—等待模式保证 GPU B 的第 4–7 层计算只在点对点拷贝完成之后才开始。与此同时，GPU B 的 stream0_B 可以并行地预取权重或执行其他准备工作。

这里使用的 P2P 传输运行在 GPU A 的 DMA 拷贝引擎上，不占用 GPU A 的任何 SM。这样，GPU A 的计算 stream 就能立即推进到批次 N+1，而无需去管理这次数据传输。

其结果是一个三路重叠：GPU A 的 SM 可以在 stream0_A 中立即开始批次 N+1，GPU A 的对端 DMA 引擎可以在 stream1_A 中把批次 *N* 的激活值传送给 GPU B，而 GPU B 的 SM 可以在 stream0_B 中运行批次 *N*，或在其 stream1_B 中开始批次 *N*。通过把工作划分到不同的 stream 中，并在主机上使用固定内存，我们把 P2P 与 H2D 延迟隐藏在持续进行的计算和数据准备之后。

当需要在众多 GPU 之间同步梯度或广播参数时，NCCL 会使用低占用率的设备核函数在大规模下处理通信，这些核函数驱动 GPUDirect P2P 或通过 NVLink、PCIe 或 InfiniBand 进行 RDMA。在底层，NCCL 会把张量拆分成多个连续的分块。

这一设计表明，每个 GPU 都可以拥有多个 stream，包括一条用于主工作的计算流（compute stream）、一条用于接收数据的接收流，以及一条用于归约的通信流（communication stream）。为每种角色使用专用 stream——并给通信流更高的优先级——能够实现最优重叠。事实上，像 PyTorch 这样的框架通常会在这些专用 stream 上运行 NCCL 集合操作，并为网络传输赋予更高的优先级。

> PyTorch 和 NCCL 都使用专用的高优先级 stream，把通信与计算密集型操作交错进行。这样一来，它们就不会被排在大型计算核函数之后而延迟。

NCCL 会根据 NVLink 或 PCIe 拓扑选择环形（ring）或树形算法。考虑一个四 GPU 环形结构，如图 11-8 所示，其中有四个分块（1–4）。

![图 11-8. 四 GPU 环形结构中的分块 all-reduce](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-8.png)

在这种环形 all-reduce 中，每个 GPU 把分块 *i* 发送给它的一个邻居（k → k+1），同时从它的另一个邻居（k−1 → k）接收分块 i−1，并使用设备核函数在 NVLink 或 PCIe 上搬运和归约这些分块。

通过对这些分块的发送与接收进行流水线化，NCCL 让 NVLink 保持完全饱和。当分块 *i* 从 GPU 0 → GPU 1 传输时，分块 i−1 从 GPU 1 → GPU 2 移动，依此类推。这将空闲间隙降到最低。在代码中，你会看到类似下面这样的写法：

```
// In a high-priority communication stream on each GPU:
cudaEventRecord(eventK, computeStream);
cudaStreamWaitEvent(commStream, eventK, 0);
ncclAllReduce( // this is asynchronous
    gradBuffer, gradBuffer, numElements,
    ncclFloat, ncclSum, comm, commStream);
```

由于 NCCL 只使用少量 SM 线程块，它以低占用率的方式在 SM 上编排跨 NVLink 或 NVSwitch 的分块发送/接收 + 归约。与此同时，例如正在运行第 k+1 层反向传播的主计算 stream 可以继续运行。当 all-reduce 完成时，会记录事件，如下所示：

```
cudaEventRecord(eventAllReduceDone, commStream); // signals collective completion
cudaStreamWaitEvent(computeStream, eventAllReduceDone, 0);
// Now apply optimizer updates on computeStream...
```

请记住，像 all-reduce 这样的 NCCL 集合操作是异步的。它们会立即返回控制权。通过在调用之后记录一个事件，并在计算 stream 中等待它，你就能确保计算 stream 直到归约真正完成后才恢复。

> NCCL 的核函数经过刻意优化，能够在低占用率下、每个 GPU 只用少量 SM 资源的情况下达到最大带宽。在实践中，得益于为互连（interconnect）调优过的低占用率核函数，NCCL 每个 GPU 只用少量线程块就能饱和 NVLink 或 PCIe 带宽。此外，如果你的网络硬件支持，你还可以把其中一些集合聚合操作卸载到 NVIDIA SHARP。这将释放出更多 SM 资源来执行更有用的计算工作。

NIXL 为跨 NVLink、RDMA 与存储后端的点对点及分离式传输提供了统一的高吞吐 API，因此你通常会使用 NIXL 的 API，而不是自己去调用 cudaMemcpyPeerAsync。NIXL 能够跨不同的内存层级和互连快速且异步地搬运数据。当你用 nixlCommStream 调用一个 NIXL 操作时，数据会被分块，并使用可用的最快传输机制（例如 GPUDirect RDMA 或 NVLink）在网络上流水线化传输。

与 NCCL 一样，NIXL 也可以为分块的点对点及集合数据传输使用专用的高优先级 stream。这可以减少排队延迟，使它们的拷贝命令在发出后不久便命中 GPU 的 DMA 引擎——很可能抢占优先级更低的工作。

把这些传输 stream 标记为高优先级，并不意味着它们会预先占用 SM 资源。它只是保证当拷贝命令准备就绪时，它们会先于其他低优先级操作被发出。

在实践中，拷贝引擎随后会利用你的计算核函数尚未占用的那部分 DRAM 到 DRAM 带宽。因此，它们实际上只用空闲的内存织构（memory-fabric）带宽在互连上搬运数据。

这一设计最大化了重叠：当一个 stream 在拷贝引擎上驱动 cudaMemcpyPeerAsync 传输或 NCCL 归约核函数时，另一个 stream 的 SM 核函数可以继续处理前向/反向工作。而第三个 stream 可以执行异步分配或事件等待。这让所有硬件单元都保持忙碌且互不争用。

用 cudaMemcpyPeerAsync 启动的点对点传输完全运行在 GPU 的拷贝引擎上，因此只消耗 DRAM 到 DRAM 带宽，而不占用任何 SM 周期。而像 all-reduce 这样的集合操作则被实现为设备核函数，它们使用少量 SM，同时把互连带宽推向峰值。

与此同时，用于激活值或混合精度梯度的临时缓冲区会被异步地分配和释放，以避免全局停顿。例如，在每个 GPU 的计算 stream 内部，你会调用以下代码：

```
cudaMallocAsync(&tempBuffer, tempBytes, computeStream);
// ...use tempBuffer in kernels...
cudaFreeAsync(tempBuffer, computeStream);
```

由于流序内存分配器记录这些操作时不会强制进行设备级同步，CUDA 在分配或释放内存时不会暂停其他 stream，例如 P2P、NCCL 和 NIXL stream。这确保了缓冲区管理绝不会打断重叠进行的计算与通信流水线。

> 这正是我们前面讨论过的流序内存分配器——只不过现在应用到了多个 GPU 上。使用它有助于防止内存管理成为你分布式工作负载的瓶颈。

在你把一次迭代（例如训练或推理迭代）的前向传播、反向传播、P2P 拷贝、NCCL all-reduce、内存释放以及事件等待操作入队之后，你可以把这条执行链捕获进单个 CUDA Graph。

具体来说，当你调用 cudaStreamBeginCapture 时，你入队到该 stream 的每一个操作（例如核函数启动、计算、通信、数据传输、事件等）都会作为节点插入到图中。这可以包括 ncclAllReduce()、cudaMemcpyAsync、cudaMemcpyPeerAsync、cudaMallocAsync、cudaFreeAsync、cudaEventRecord、cudaStreamWaitEvent——以及它们之间的依赖关系。你用 cudaStreamEndCapture(stream, &graph) 结束捕获。代码大致如下：

```
cudaStreamBeginCapture(streamA, cudaStreamCaptureModeGlobal);
computeGradientsLayer1<<<... , 0, streamA>>>(...);
ncclAllReduce(..., comm, streamB);
computeGradientsLayer2<<<... , 0, streamA>>>(...);
ncclAllReduce(..., comm, streamB);
cudaStreamEndCapture(streamA, &graph);
```

在捕获提交到多个 stream 的工作时，请使用 cudaStreamCaptureMode Global，这样同一线程中入队到任何参与 stream 的操作都会被记录进同一个捕获会话——并且跨流依赖也会被保留。否则，只有入队到传给 cudaStream BeginCapture 的那个 stream 的操作才会被捕获。

随后你可以用 cudaGraphLaunch() 实例化并启动该图。这将以接近零的 CPU 开销重放整个 DAG。通过把整个多 GPU 迭代一次性捕获下来，你消除了每个操作的入队与启动开销。

此外，CUDA Graphs 支持条件节点（例如梯度裁剪），因此不常发生的分支也保留在图的逻辑内部。我们会在下一章更详细地介绍 CUDA Graphs，但理解它们与 CUDA streams 的关系很重要，包括 cudaStreamBeginCapture 和 cudaStreamEndCapture。

> 像 PyTorch 这样的现代框架把这些复杂性大多隐藏了起来。例如，PyTorch 的 DistributedDataParallel 会自动把数据传输、计算与通信调度到各自独立的 stream 中。它还使用 CUDA Graphs 来降低每次迭代的开销。尽管它可以使用 CUDA Graphs 来降低开销，但你必须通过使用捕获安全（capture-safe）的代码和 API 显式启用这一点，因为它并非自动启用。

虽然你并不需要显式使用这些 CUDA stream 的开始与结束 API 来构建 CUDA Graph，但当你希望在把操作入队到 stream 的同时捕获图时，它们会很方便（我们会在下一章更详细地介绍 CUDA Graphs）。

CUDA streams 通过在 CPU、DMA 引擎与 SM 之间重叠工作，让多 GPU 系统中的流水线得以精细调优。在 CPU 一侧，把数据用 cudaMemcpyAsync 拷入固定主机缓冲区，可以在 GPU 执行当前工作负载的同时为下一批次暂存数据，从而确保批次 N+1 的输入在批次 *N* 完成之前就已就绪。与此同时，GPU 之间的点对点传输（使用 cudaMemcpyPeerAsync）可以被调度到专用 stream 上，并用 cudaStreamWaitEvent 同步，以便在不停顿正在进行的计算的前提下交接激活值或梯度。

如前所述，像 NCCL 或 NIXL 这样的集合通信库是使用独立 stream 的低占用率通信核函数。这样一来，当一个 GPU 在归约梯度或广播参数时，同一个 GPU 上的其他 stream 可以继续执行本地计算核函数（例如后续各层的前向或反向传播）。此外，按 stream 顺序使用 cudaMallocAsync 和 cudaFreeAsync，可以避免为临时缓冲区做全局同步，让分配与释放能够与计算和通信并发进行。

把整个迭代（例如 CPU 暂存、P2P 传输、计算核函数、NCCL 调用、分配、释放以及事件/等待）捕获进一个 CUDA Graph，可以进一步降低 CPU 开销。图一旦实例化，调用 cudaGraphLaunch() 就会一次性重放所记录的 DAG。这消除了每次调用的入队开销，并自动保留所有依赖关系。

综合起来，这些技术确保每个 GPU 的 SM 流水线、拷贝引擎和互连链路都保持忙碌。当一个 stream 运行矩阵乘核函数时，另一个执行点对点拷贝，第三个执行集合通信，而 CPU 则为下一个微批量暂存数据。

简而言之，你可以用 CUDA streams 来编排跨多个层次的工作，并重叠计算、内存操作与数据传输。这种方法通过把点对点和集合通信延迟隐藏在活跃计算之后，把每一个硬件单元都推向极限。总体而言，CUDA streams 为许多 AI 工作负载（包括 LLM 训练与推理）最大化了吞吐、利用率和效率。

## 程序化依赖启动

另一种核间流水线化与通信的方式称为*程序化依赖启动*（Programmatic Dependent Launch，PDL）。PDL 让一个核函数能够在设备上直接触发另一个核函数的执行，使用同一条 CUDA stream——并且无需 CPU 参与。例如，作为主核函数（primary kernel）的 Kernel A 可以触发作为次级核函数（secondary kernel）的 Kernel B，而后者正在等待来自 Kernel A 的信号。

这一触发甚至可以在 Kernel A 执行完成之前发生。为此，它使用 cudaTriggerProgrammaticLaunchCompletion()，如图 11-9 所示。这里，Kernel B 通过调用 cudaGridDependencySynchronize() 等待 Kernel A。

![图 11-9. 使用 PDL 从 Kernel A 启动 Kernel B——让 Kernel B 的执行与 Kernel A 的收尾（epilogue）部分重叠（在同一 CUDA stream 中）](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-9.png)

利用 PDL 提供的构件，Kernel A 可以在其*收尾*（epilogue，即收官）阶段就发信号让 Kernel B 执行。这样，Kernel A 就能与 Kernel B 并行执行一小段时间。需要重点注意的是，在 Kernel A 尚未产出并同步好 Kernel B 所需的全部数据（例如 L2/共享内存/全局内存）之前，Kernel A 不应触发 Kernel B 执行。

> 依赖核函数的数据依赖应在其继续处理之前，在 L2、共享内存、全局内存等处可见。

下面的代码展示了 Kernel A 使用 cudaTriggerProgrammaticLaunchCompletion() 向 Kernel B 发信号，表明其主要工作已经完成。这同时也通知 Kernel B：所有必要的全局内存刷新都已发生——可以安全地继续：

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

这里，Kernel B 一直在 cudaGridDependencySynchronize() 上等待，直到它收到来自 Kernel A 的这一程序化启动完成信号。一旦交接发生，这两个核函数就可以重叠执行。通过把 Kernel A 中的触发、Kernel B 中的同步以及主机端的启动属性配对起来，这段代码在 Kernel A 与 Kernel B 之间实现了尽可能多的重叠。

正如本例所示，使用 PDL 时你需要创建一个 cudaLaunchConfig_t，并在 cudaLaunchKernelExC() 这一 CUDA 调用中使用特殊属性。具体来说，在主机端，PDL 是通过为 Kernel B 的启动配置一个 cudaLaunchConfig_t，并启用 cudaLaunchAttributeProgrammaticStreamSerialization 属性来开启的，以允许提前的、重叠的分派。

用 cudaLaunchAttributeProgrammaticStream Serialization 调用 cudaLaunchKernelExC() 会告诉 CUDA 运行时：即便 Kernel A 尚未完全完成，把 Kernel B 入队也是安全的。随后用 cudaLaunchKernelExC() 携带这些扩展属性执行实际启动，它依赖触发机制来完成交接。

## 综合 PDL、线程块簇与 warp 专门化

让我们把三种正交的技术——PDL、warp 专门化流水线以及线程块簇——汇聚到一个执行模型中。

PDL 提供了让一个核函数发信号表明其序言（prologue）完成、并触发一个依赖核函数执行的机制。随后它会逐渐收尾，而依赖核函数则逐渐启动并执行。

warp 专门化把每个线程块细分为生产者 warp 与消费者 warp。具体来说，生产者 warp 使用 TMA 异步传输 tile，而计算 warp 执行乘加累加（matrix-multiply-accumulate，MMA）操作。

而线程块簇保证多个线程块运行在邻近的 SM 组上。这有利于多 SM 协作以及面向大规模工作负载的共享内存多播。

综合来看，这种核间与核内技术的组合隐藏了 DRAM 延迟，减少了核函数边界开销，并最大化了 Tensor Core 利用率。它们构建出一个具有以下特征的流水线：

*核内流水线化* warp 专门化确保在每个线程块内部，数据传输（生产者 warp）与计算（消费者 warp）通过多阶段流水线完全重叠。

*核间重叠* PDL 允许一个依赖 GEMM 核函数（它可能作用于神经网络中的下一层）的序言，在主核函数完成数据（权重）预取后立即开始——而无需等待整个线程块拆除。

*块间协作* 线程块簇让多组线程块得以协调预取（例如多播），并在 SM 之间执行动态负载均衡。这样，生产者与消费者任务都能在整个簇范围内均匀分布。

例如，你可以在同一个核函数内把模型权重的搬运（数据传输）与 GEMM（计算）重叠起来，使得一个 tile 在被计算的同时，下一个 tile 正在途中。一个 warp 专门化的多阶段软件流水线（阶段 0… N）可以使用 PDL 和 *mbarrier* 原语来协调这些角色，如图 11-10 所示。

![图 11-10. 结合 PDL、线程块簇与 TMA 多播的 warp 专门化多阶段流水线，用于实现高性能的核间与核内 GEMM](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-10.png)

下面是一个 CUDA C++ 示例，展示如何用一个简单的 TMA 风格异步拷贝 + 计算流水线，把 PDL、线程块簇与 warp 专门化结合起来。具体来说，主核函数 primary_gemm 使用 cudaTriggerProgrammatic LaunchCompletion() 向次级核函数发信号，表明所有内存刷新都已完成、数据已准备好被消费。于是，次级核函数 secondary_gemm 现在可以安全地从 cudaGridDependencySynchronize() 处继续，如下所示：

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

在主机上，你把主核函数（primary_gemm）启动到一条非阻塞流中，并创建一个带有 ProgrammaticStreamSerialization 属性的 cudaLaunchConfig_t。你用它来为次级核函数（secondary_gemm）调用 cudaLaunchKernelExC。

在 primary_gemm 内部，一次对 cudaTriggerProgrammaticLaunchCompletion() 的调用标志着内存刷新的完成，并允许依赖核函数的处理开始——甚至在主核函数完全拆除之前。

依赖核函数随后调用 cudaGridDependencySynchronize()。这会等待来自主核函数的信号和必要的内存刷新，从而使它能够与主核函数的收尾并行地开始执行。

在每个线程块内部，我们使用 warp 专门化来重叠数据搬运与计算。通过把每个块拆分为单个“生产者” warp 和多个“消费者” warp，生产者发出 cuda::memcpy_async 调用把数据拷入共享内存。这利用 TMA 多播把单次 DMA 传输广播给簇内所有线程块，如图 11-11 所示。

![图 11-11. TMA 多播：主导 CTA 将 cp.async.bulk.tensor 发射到 DSMEM（簇共享内存）；跟随 CTA 使用 cluster.map_shared_rank 消费这些 tile；cuda::memcpy_async 驱动 TMA](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-11.png)

当生产者 warp 用 TMA 多播加载数据时，消费者会在一个 C++ 块作用域屏障（cuda::barrier<cuda::thread_scope_block>）上自旋，然后再执行其矩阵乘步骤（do_compute()）。这让每个 tile 的加载与其乘加融合（FMA）操作能够在一个细粒度的多阶段软件流水线中交错进行，从而在核函数内部隐藏全局内存延迟。

为了在 SM 之间协调工作，我们用 __cluster_dims__(2,1,1) 标注这两个核函数。这会把成对的线程块分组到邻近的 SM 上。一次对 cluster.sync()（协作组对 PTX 的 mbarrier 指令的封装）的调用充当簇级屏障和共享内存栅栏。这样，簇内所有线程块都能看到相同的 TMA 加载数据，并能动态地重新平衡剩余的 tile。这种块间协作避免了 SM 空闲，并增强了 warp 专门化流水线的收益。

> 生产级核函数通常对二维 tile 使用 Tensor Memory Accelerator（TMA）的 cp.async.bulk.tensor，并在多个线程块需要同一 tile 时跨线程块簇进行多播。当在 Blackwell 上做分块拷贝时，可考虑使用基于描述符的 TMA 归约操作（例如 cp.reduce.async.bulk(.tensor)）来做即时归约。当把小型归约融合进数据搬运流水线时，优先使用基于描述符的加载/存储加上 TMA 归约操作。这将降低寄存器压力并改善重叠。

最后，让我们再回顾一下这条高性能 GEMM 流水线的三个特征：

*核内流水线化* 在每个线程块内部通过 warp 专门化实现，把工作细分开来，使生产者 warp 发出 cuda::memcpy_async 的 tile，而消费者 warp 在块作用域屏障上自旋。这让数据传输与计算操作完全重叠。

*核间流水线化（Inter-kernel pipelining）* 使用 PDL，它让下一个 GEMM 核函数的处理能够在主核函数被完全拆除之前就开始（借助 cudaTriggerProgrammaticLaunchCompletion() → cudaGridDependencySynchronize() 机制）。这掩盖了核函数启动开销。

*块间协作（Interblock cooperation）* 通过用 __cluster_dims__ 注解核函数，这段代码创建了被协同调度到邻近 SM 上的线程块对。再配合调用 cluster.sync()（基于 mbarrier 的簇屏障），这些线程块共享 TMA 多播数据，并在整个网格上动态负载均衡工作。

简而言之，通过把 warp 级流水线化与 TMA、簇级同步与多播，以及使用 PDL 的核间重叠层层叠加，你就构建出一条高度重叠的流水线，它隐藏 DRAM 延迟、掩盖核函数启动开销，并最大化 Tensor Core 利用率。其结果是在每个阶段都得到高性能 GEMM：在 warp 内拷贝中、在跨线程块屏障中，以及在核函数交接期间。它们协同运作，让硬件持续繁忙、避免停顿。

## 关键要点

本章涵盖了一些与 CUDA streams、流序内存分配器（stream-ordered memory allocator）、基于事件的同步、核间流水线化、线程块簇（thread block cluster）以及 PDL 相关的进阶主题。它们有助于为推理和训练工作负载构建高效的流水线。以下是一些关键要点：

*显式流与默认流（Explicit versus default streams）* 避免使用传统默认流（legacy default stream）（stream 0），它会串行化所有工作并充当全局屏障。取而代之，创建显式的非阻塞流（cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)），使核函数与拷贝能够并发运行，而不会产生隐藏的同步。

*流序内存分配器（Stream-ordered memory allocator）* 使用 cudaMallocAsync 和 cudaFreeAsync 在特定 stream 内分配/释放设备内存。这个非阻塞分配器把请求记录到该 stream 的队列中，避免全局设备同步，并让分配能够与在途（in-flight）的核函数和拷贝相重叠。

*重叠 H2D、计算与 D2H（Overlapping H2D, compute, and D2H）* 通过在不同 stream 上入队异步的主机到设备拷贝（cudaMemcpyAsync）、核函数启动以及设备到主机拷贝，你可以实现三路重叠。当一个 stream 运行核函数时，另一个 stream 可以把下一批数据拷贝到设备（H2D），而第三个 stream 则可以把结果拷回主机（D2H）。这隐藏了延迟并减少了空闲时段。

*用于细粒度同步的 CUDA 事件（CUDA events for fine-grained synchronization）* 使用 cudaEventRecord 和 cudaStreamWaitEvent 来协调跨 stream 的生产者—消费者依赖，而不必让整个 GPU 或 CPU 停顿。事件让消费者 stream 能够精确地等待，直到生产者 stream 完成一次拷贝或核函数，从而保持最大并发度。

*核间流水线化（Inter-kernel pipelining）* 把核函数内部的 warp 专门化（多角色）、两阶段（双缓冲）流水线与多 stream 启动结合起来。在不同的 stream 中启动一个 warp 专门化核函数（loader → compute → storer）的多个实例，向 GPU 持续投喂相继的小批量（mini-batch）。这将核内内存/计算重叠与核间并发结合在了一起。

*线程块簇配合 streams（Thread block clusters with streams）* 把核内 warp 专门化流水线扩展为覆盖整个网格的线程块簇（协作式启动），可让 loader/compute/storer warp 跨越多个块协同工作。在多个 stream 中启动这些协作式核函数，使得后续批次的主机端分配与拷贝能够在某个协作式核函数执行期间进行。

*核内信号与借助 PDL 的重叠（In-kernel signaling and overlap with PDL）* 核函数 A 在其数据写入被刷新后调用 cudaTriggerProgrammaticLaunchCompletion()，核函数 B 则使用 cudaGridDependencySynchronize() 来等待该信号。这让核函数 B 的序言（prologue）能够开始，并与 A 的收尾（epilogue）相重叠——无需 CPU 介入。

*主机端 PDL 设置（Host-side PDL setup）* 主机通过 cudaLaunchKernelExC() 配置核函数 B 的启动，并用一个设置了 cudaLaunchAttributeProgrammaticStreamSerialization 的 cudaLaunchConfig_t。这让驱动能够提前入队 B 并最大化核间重叠。PDL 在主核函数中使用 cudaTriggerProgrammaticLaunchCompletion，在依赖核函数中使用 cudaGridDependencySynchronize，并在主机端使用 cudaLaunchAttributeProgrammaticStreamSerialization 启动属性。

## 结论

总而言之，基于 CUDA streams 的核间并发已经从一种手动优化演变为现代 AI 框架与 GPU 运行时所使用的自动特性。通过理解诸如 streams、资源占用（resource occupancy）与同步点等核心原理，开发者可以最大化 GPU 利用率。

核间并发对于在现代工作负载中最大化 GPU 利用率至关重要，它通过在单个 GPU 内部以及跨多个 GPU 之间重叠核函数执行与数据传输来实现这一点。随着硬件不断增加更多并行性——而软件抽象掉越来越多的调度复杂性——理解如何最大化并发，是从你的高性能 AI 系统硬件中榨取最大价值的关键一环。

本章演示了如何在多个 CUDA stream 之间编排核函数、内存操作与分配，从而让所有 GPU 硬件单元都保持活跃运行。这些硬件单元包括计算流水线、DMA 引擎与互连（interconnect）。CUDA streams 充当基础机制，把核函数、内存操作与分配入队到相互独立的队列中，让 GPU 的计算引擎与 DMA 引擎能够同时运行。

通过避开默认流的隐藏屏障、利用流序分配器、用事件做精确同步，以及把核内 warp 专门化与核间多 stream 流水线相结合（并扩展到第 10 章的线程块簇），你即使面对复杂的 LLM 工作负载，也能达到接近峰值的利用率。

在多 GPU 场景中，跨不同 stream 重叠点对点（peer-to-peer，P2P）传输、集合通信（collective communication）与计算，可进一步压缩空闲时间。我们回顾了第 10 章，展示了如何用 CUDA Graphs 捕获这些 stream 工作流，以降低重复迭代时的 CPU 开销。

在下一章，我们将进一步深入，并在这些原理之上，引入动态核函数编排以及借助动态并行（dynamic parallelism）与 CUDA Graphs 的元调度。我们将在运行时协调整条核函数与数据搬运的流水线——以适应不断变化的工作负载。这种动态资源均衡与设备端编排，将把我们推向大规模 AI 系统性能优化的下一个层次。
