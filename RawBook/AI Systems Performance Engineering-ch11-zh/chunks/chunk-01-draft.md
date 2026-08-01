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

此外，LLM 会使用即时的“暂存内存”分配，这正是由上一节讨论的流序内存分配器来支撑的。例如，在运行一个 transformer 层时，你常常需要额外的共享内存或设备内存，即所谓的 _scratch memory_，来存放一次矩阵乘法的结果，然后再把它送入 softmax 操作。

由于 LLM 工作负载中不同的小批量在长度（token 数）上可能各不相同，你会希望使用流序内存分配器，专门为每个输入批次在 GPU 上提供一块全新的暂存缓冲区。这样，你为该批次的中间计算分配的空间恰好够用——一个字节都不多。

如果你用旧的、阻塞式的分配 API（cudaMalloc(...) 与 cudaFree(...)）来即时分配这些暂存缓冲区，那么每一次分配或释放都会与整个 GPU 同步，因为调用 cudaMalloc(...) 会强制一次全局设备同步。如此一来，在所有待处理的核函数与拷贝完成之前，任何重叠都无从谈起。

> 全局设备同步对性能是绝对灾难性的。避免在你的 CUDA streams 中使用像 cudaMalloc() 与 cudaFree() 这样的阻塞调用。优先使用 events 与 stream 等待。而且绝对要避免用 cudaStreamSynchronize(0) 在默认 stream 0 上做同步！

在这样一条流水线里——一个 stream 正忙于为批次 _N_ 运行 attention 核函数，另一个 stream 正在为 attention 核函数准备批次 N+1——在第二个 stream 上调用阻塞式的 cudaMalloc(...) 会停顿所有 stream。在分配器完成之前，每个 SM 实际上都被暂停了。这会彻底抹掉你原本指望在数据传输、计算与内存管理之间实现的任何重叠。

解决办法是使用带 cudaMallocAsync() 与 cudaFreeAsync() 的流序分配器。这些 API 在 stream 层面把分配与释放设备内存区域的工作入队。因此，它们只在 stream 层面同步——而非设备层面。

例如，设想有一个 stream 需要一块 16 MB 的暂存缓冲区，用于在一批输入数据上做 attention。它会调用 cudaMallocAsync(&scratchPtr, scratch Bytes, stream1)，这会把该分配请求记录到它的操作队列中，但并不会强制任何其他 stream 等待，如图 11-3 所示。

![图 11-3. 流序内存分配](AI%20Systems%20Performance%20Engineering-ch11_images/figure-11-3.png)

其他 stream 会继续启动核函数、拷贝数据，或做它们正在做的任何事——即便 stream 1 的分配还在途（in-flight）中。而一旦 CUDA 运行时在幕后预留好内存，stream 1 就能再次推进，并把 attention 核函数启动到那块新分配的区域中——这一切都不会让任何其他 stream 停顿。

> 与传统的 cudaMalloc 不同，cudaMallocAsync 不会停顿其他 stream。每次分配只在其自身的 stream 内同步。

在 LLM 训练与推理的语境下，这一点尤为宝贵，因为变长序列常常导致暂存缓冲区大小的波动。如果批次 _N_ 每个序列有 512 个 token，而批次 N+1 有 1,024 个 token，那么你的 attention 模块为批次 N+1 需要的空间会比批次 _N_ 更多，因此复用批次 _N_ 的分配是不够的。有了 cudaMallocAsync()，你可以为更大的缓冲区入队一次非阻塞的分配，而不必把所有其他 stream 都拖停。

另外，典型 LLM 的自回归逐 token 生成（又称 _decoding_）阶段会用到一个不断增长的键/值缓存。每生成一个 token，就需要向每个序列各自的缓冲区追加新的 KV 对。随着缓冲区增长，你需要重新分配或扩展暂存区域。cudaMallocAsync(...) 让你可以在运行 attention 核函数的同一个 stream 中完成这件事。与此同时，上游的数据加载与下游的结果拷回操作会在它们各自的 stream 中并行地继续推进。

CUDA streams 在 LLM 语境下的另一种用法是大型 LLM 中的按层流水线化（layerwise pipelining）。假设你把一个大型的、基于 transformer 的 LLM 模型切成两半，让它们在不同的 CUDA stream 上运行——stream 0 运行第 0–5 层，stream 1 运行第 6–11 层。在这两半之间，你需要中间缓冲区来存放激活。

每当 stream 0 在一个小批量上完成其工作，它就可能调用 cudaMallocAsync(...) 为下一批次的激活抓取一块新缓冲区。由于该调用不会同步设备，stream 1 可以在上一批次的结果上继续计算第 6–11 层，同时 stream 0 为下一批次的输入分配内存。

相比之下，如果你在那条流水线里使用了传统的 cudaMalloc(...)，那么每当你为下一个小批量或一个扩大的 KV 缓存分配一块新的暂存区域时，整个 GPU 都会暂停，直到分配完成。那会破坏跨 stream 重叠计算与数据搬运的任何可能。

总结一下，在 LLM 语境下，你经常需要为 attention、层归一化、softmax、KV 缓存或中间激活准备临时缓冲区。它们被统称为 _scratch buffer_（暂存缓冲区）。

在相互独立的 stream 中用 cudaMallocAsync(...) 与 cudaFreeAsync(...) 来管理这些暂存缓冲区，可确保内存管理绝不会强制一次全局的、跨 stream 的停顿。相反，分配会入队到与你的核函数或拷贝操作相同的 stream 中。

这让所有其他 stream 得以继续运行，并让你的 attention 核函数、数据传输以及任何主机侧工作尽可能地重叠。这在大规模、实时的 LLM 工作负载中最大化了 GPU 利用率。

## 传统默认流

当你没有显式创建或指定 stream 时，操作会进入传统默认流（legacy default stream），常被称为 _stream 0_。默认情况下，stream 0 有两个值得强调的重要行为：

_与自身的隐式同步_ 入队到 stream 0 的任意两个操作会严格地一个接一个执行。你无法在 stream 0 中重叠两个核函数，或重叠一次拷贝与一个核函数，因为 stream 0 会串行化它自己的所有命令。

_与其他 stream 的隐式同步_ 在传统默认流模型中，任何启动到 stream 0 的操作，在开始之前都会等待其他每一个 stream 中先前入队的所有工作完成。反过来，任何启动到非默认 stream 的操作，也会阻塞，直到 stream 0 中先前的所有工作完成。实际上，stream 0 就像整个 GPU 上的一道全局“屏障”。即便你把命令发起到不同的 stream 上，一旦你向 stream 0 提交某个操作，它就会迫使其他每一个 stream 停顿，直到 stream 0 赶上进度，反之亦然。这对性能非常糟糕，应尽可能避免。

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
