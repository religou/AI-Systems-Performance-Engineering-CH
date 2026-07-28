# Chapter 7. Profiling and Tuning GPU Memory Access Patterns

# 第 7 章 GPU 内存访问模式的剖析与调优

As AI models grow in size and complexity, a GPU’s memory system often becomes the bottleneck that stands between theoretical compute capability and real-world performance. As you saw in Chapter 6, modern NVIDIA GPUs combine thousands of simple, throughput-optimized cores with specialized Tensor Cores. They also include high-bandwidth memory (HBM), coherent CPU-GPU unified memory address space (e.g., Grace Blackwell Superchip), on-chip shared memory, caches, and specialized direct memory access (DMA) engines like the Tensor Memory Accelerator (TMA).

随着 AI 模型规模与复杂度不断增长，GPU 的内存系统往往会成为横亘在理论计算能力与真实性能之间的瓶颈。正如你在第 6 章所见，现代 NVIDIA GPU 将数千个简单、面向吞吐优化的核心与专用的 Tensor Core 组合在一起。它们还包含高带宽内存（HBM）、CPU 与 GPU 一致的统一内存地址空间（例如 Grace Blackwell Superchip）、片上共享内存、缓存，以及像 Tensor Memory Accelerator（TMA）这样的专用直接内存访问（DMA）引擎。

In this chapter, you’ll see various CUDA C++ and PyTorch optimization techniques to align data structures for efficient memory access, eliminate redundant data loads, and overlap data transfers with computation using hardware.

本章将展示各种 CUDA C++ 与 PyTorch 优化技巧，用于对齐数据结构以实现高效内存访问、消除冗余的数据加载，以及借助硬件把数据传输与计算重叠起来。

Through concrete before-and-after examples of matrix multiplies, tensor operations, and more, you’ll see how small changes in memory access patterns, tiling strategies, and asynchronous data transfers can reduce wasted bandwidth, boost arithmetic efficiency, and transform kernels from memory bound to compute bound.

通过矩阵乘法、张量运算等一系列具体的前后对比示例，你将看到内存访问模式、分块策略与异步数据传输上的微小改动如何减少浪费的带宽、提升算术效率，并把核函数从访存受限（memory-bound）转变为计算受限（compute-bound）。

By the end of this chapter, you’ll know how to write CUDA kernels that can better utilize the GPU’s memory hierarchy and hardware-optimized data transfer engines.

读完本章后，你将懂得如何编写能更好地利用 GPU 内存层级（memory hierarchy）与硬件优化数据传输引擎的 CUDA 核函数。

## Coalesced Versus Uncoalesced Global Memory Access

## 合并访问与非合并全局内存访问

The memory access pattern of your code can greatly impact performance. Global memory accesses are fastest when threads in a warp access contiguous memory addresses that the hardware can combine into fewer, larger transactions. If threads access scattered or misaligned addresses, the device cannot coalesce requests into the minimal number of cache line transactions, which on modern GPUs are 128-byte lines composed of four 32-byte sectors. This results in many more memory transactions retrieving unused data, which quickly eats up memory bandwidth.

代码的内存访问模式会极大地影响性能。当一个 warp（线程束）中的线程访问连续的内存地址，且硬件能够把它们合并为更少、更大的事务时，全局内存（global memory）访问速度最快。如果线程访问的是分散或未对齐的地址，设备就无法把请求合并（coalesce）为最少数量的缓存行（cache line）事务——在现代 GPU 上，缓存行是由四个 32 字节扇区（sector）组成的 128 字节缓存行。其结果是产生多得多的内存事务去取回未被使用的数据，很快便耗尽内存带宽。

On a Blackwell GPU, per-device HBM3e bandwidth is up to 8 TB/s. Within the Grace Blackwell GB200 and GB300 (two GPU superchips), this increases 16 TB/s across both GPUs. Using uncoalesced memory accesses will leave most of this bandwidth unused due to excess memory transactions and stalls.

在 Blackwell GPU 上，单设备 HBM3e 带宽可达 8 TB/s。在 Grace Blackwell GB200 和 GB300（双 GPU 超级芯片）内部，跨两颗 GPU 的带宽提升到 16 TB/s。使用非合并（uncoalesced）内存访问会因为过多的内存事务与停顿，而让这些带宽的绝大部分闲置浪费。

In the uncoalesced case, each thread in a warp loads from scattered addresses. This results in many separate memory transactions. Even if threads in a warp access consecutive addresses, if the first address isn’t 128-byte aligned, the warp’s request will span two 128-byte cache lines.

在非合并的情形下，一个 warp 中的每个线程都从分散的地址加载。这会产生许多彼此独立的内存事务。即便一个 warp 中的线程访问的是连续地址，只要第一个地址不是 128 字节对齐的，该 warp 的请求就会横跨两个 128 字节缓存行。

For example, if a warp’s first thread starts at an address that isn’t 128-byte aligned, the warp’s memory request will cross a cache-line boundary, often resulting in two 128-byte transactions instead of one. In that case the warp may fetch an extra sector beyond the optimal four sectors, for a total of five sectors across the two lines. This is a waste of bandwidth. Whether a misaligned, contiguous 128-byte warp load touches 5× 32 B sectors or 8× 32 B sectors depends on the start offset. Aligned accesses keep it to 4× 32 B sectors.

举例来说，如果一个 warp 的首个线程从一个非 128 字节对齐的地址开始，该 warp 的内存请求就会跨越一个缓存行边界，往往导致产生两个 128 字节事务而非一个。在这种情况下，该 warp 可能会多取一个扇区，超出最优的四个扇区，于是两条缓存行上总共取到五个扇区。这是对带宽的浪费。一个未对齐、连续的 128 字节 warp 加载究竟触及 5× 32 B 扇区还是 8× 32 B 扇区，取决于起始偏移量。对齐的访问则能将其保持在 4× 32 B 扇区。

In the coalesced case, however, threads load from consecutive addresses combined into a single wide transaction. Figure 7-1 compares coalesced and uncoalesced memory access.

然而在合并的情形下，线程从连续地址加载，这些加载被合并成单个宽事务。图 7-1 对比了合并与非合并的内存访问。

![Figure 7-1. Comparing coalesced versus uncoalesced memory access pattern](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-1.png)

![图 7-1. 对比合并与非合并的内存访问模式](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-1.png)

In kernel code, this problem typically appears as _strided_ or irregular indexing such that each thread reaches into different cache lines. When a kernel’s threads fetch data with strided or irregular indices, the GPU issues many small, uncoalesced global-memory transactions rather than a handful of full‐width loads.

在核函数代码里，这个问题通常表现为*跨步*（strided）或不规则的索引，使得每个线程都伸向不同的缓存行。当一个核函数的线程用跨步或不规则的索引取数据时，GPU 会发出许多小而非合并的全局内存事务，而不是少数几个全宽加载。

In Nsight Compute, the Memory Workload Analysis section will show lower Global Memory Load Efficiency, higher DRAM sector read counts, and average sectors per request above 4.0 when uncoalesced patterns are present. This is because more sectors are being fetched than a properly coalesced memory access pattern, and it indicates that you’re wasting bandwidth by fetching mostly unused bytes. And DRAM throughput percentage will remain well below peak. This confirms that your warp is spending cycles stalled on memory rather than driving the ALUs.

在 Nsight Compute 中，当存在非合并模式时，内存工作负载分析（Memory Workload Analysis）小节会显示更低的全局内存加载效率（Global Memory Load Efficiency）、更高的 DRAM 扇区读取计数，以及高于 4.0 的每次请求平均扇区数。这是因为相比恰当合并的内存访问模式，此时取回了更多扇区，表明你正因为取回的大多是无用字节而浪费带宽。同时 DRAM 吞吐百分比会一直远低于峰值。这印证了你的 warp 正把周期消耗在等待内存上，而不是驱动 ALU。

To break out of this memory‐bound regime, you can reorganize your data so that each warp’s 32 threads load contiguous elements. Either index the array with input[idx] where idx = blockIdx.x \* blockDim.x + threadIdx.x or switch to a structure‐of‐arrays (SoA) layout so that thread i always touches element i. The difference between an array of structures (AoS) and an SoA is shown in Figure 7-2.

要摆脱这种访存受限的境地，你可以重新组织数据，让每个 warp 的 32 个线程加载连续元素。要么用 input[idx] 来索引数组（其中 idx = blockIdx.x \* blockDim.x + threadIdx.x），要么切换到数组结构体（Structure of Arrays，SoA）布局，使得线程 i 始终触及元素 i。结构体数组（Array of Structures，AoS）与 SoA 之间的区别如图 7-2 所示。

![Figure 7-2. Array of structures (AoS) versus structure of arrays (SoA)](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-2.png)

![图 7-2. 结构体数组（AoS）与数组结构体（SoA）](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-2.png)

Once you make the change, the hardware will automatically combine the warp’s global memory loads into fewer, wider transactions with more usable (less wasted) data being returned. The Nsight Compute counters will immediately show improvement.

一旦做出改动，硬件就会自动把该 warp 的全局内存加载合并为更少、更宽的事务，返回更多可用（更少浪费）的数据。Nsight Compute 的计数器会立刻显示出改善。

Let’s demonstrate this with an example. The following before-and-after code demonstrates how restructuring global memory accesses from a strided pattern to a contiguous pattern produces a significant performance gain.

我们用一个例子来演示这一点。下面的前后对比代码展示了如何把全局内存访问从跨步模式重构为连续模式，从而带来显著的性能提升。

_Before example (C++): uncoalesced strided access._ In this example, each thread copies from an input array with a stride of 2, causing misaligned memory accesses:

_前置示例（C++）：非合并的跨步访问。_ 在这个示例中，每个线程都以步长 2 从输入数组拷贝数据，导致未对齐的内存访问：

```
#include <cuda_runtime.h>
#include <iostream>

__global__ void uncoalescedCopy(const float* __restrict__ in,
float* __restrict__ out,
int N, int stride) {
    // n = 1048576, stride = 2
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Loads from in[] with a stride, causing
        // multiple memory segments to be fetched
        out[idx] = in[idx * stride];
    }
}

int main() {
    const int N = 1 << 20;
    const int stride = 2;

    float* h_in = nullptr;
    float* h_out = nullptr;
    cudaMallocHost(&h_in, N * stride * sizeof(float));
    cudaMallocHost(&h_out, N * sizeof(float));

    for (int i = 0; i < N * stride; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * stride * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * stride * sizeof(float),
      cudaMemcpyHostToDevice);

    // Number of threads per block (multiple of 32)
    const int threadsPerBlock = 256;

    // Number of blocks per grid
    const int blocksPerGrid = (N + threadsPerBlock - 1) /
      threadsPerBlock;

    uncoalescedCopy<<<blocksPerGrid,
      threadsPerBlock>>>(d_in, d_out, N, stride);
    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    return 0;
}
```

```
#include <cuda_runtime.h>
#include <iostream>

__global__ void uncoalescedCopy(const float* __restrict__ in,
float* __restrict__ out,
int N, int stride) {
    // n = 1048576, stride = 2
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Loads from in[] with a stride, causing
        // multiple memory segments to be fetched
        out[idx] = in[idx * stride];
    }
}

int main() {
    const int N = 1 << 20;
    const int stride = 2;

    float* h_in = nullptr;
    float* h_out = nullptr;
    cudaMallocHost(&h_in, N * stride * sizeof(float));
    cudaMallocHost(&h_out, N * sizeof(float));

    for (int i = 0; i < N * stride; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in,  N * stride * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * stride * sizeof(float),
      cudaMemcpyHostToDevice);

    // Number of threads per block (multiple of 32)
    const int threadsPerBlock = 256;

    // Number of blocks per grid
    const int blocksPerGrid = (N + threadsPerBlock - 1) /
      threadsPerBlock;

    uncoalescedCopy<<<blocksPerGrid,
      threadsPerBlock>>>(d_in, d_out, N, stride);
    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    return 0;
}
```

The CUDA C++ kernel issues global memory loads at addresses separated by a stride > 1, which is, by definition, noncontiguous. This causes each warp to generate multiple small transactions instead of a single wide transaction.

这个 CUDA C++ 核函数在彼此相距一个 stride > 1 的地址上发出全局内存加载，按定义即是非连续的。这导致每个 warp 生成多个小事务，而非单个宽事务。

_Before example (PyTorch)._ In PyTorch, an analogous situation can be created using a strided index for a gather operation:

_前置示例（PyTorch）。_ 在 PyTorch 中，可以用带跨步索引的 gather 操作制造出类似情形：

```
import torch

def uncoalesced_copy(input_tensor, stride):
    # Flatten to 1D so we know exactly
    #   which dimension we're indexing
    flat_tensor = input_tensor.contiguous().view(-1)

    # Generate indices with a fixed stride to gather
    assert flat_tensor.numel() % stride == 0,"stride must divide tensor length"
    idx = torch.arange(0, flat_tensor.numel(), stride,
        device=flat_tensor.device, dtype=torch.long)
    # index_select uses a gather kernel that issues uncoalesced loads
    return torch.index_select(flat_tensor, 0, idx)

# Usage
n, stride = 1 << 20, 2
inp = torch.arange(n * stride, device='cuda',
                   dtype=torch.float32)
out = uncoalesced_copy(inp, stride)
```

```
import torch

def uncoalesced_copy(input_tensor, stride):
    # Flatten to 1D so we know exactly
    #   which dimension we're indexing
    flat_tensor = input_tensor.contiguous().view(-1)

    # Generate indices with a fixed stride to gather
    assert flat_tensor.numel() % stride == 0,"stride must divide tensor length"
    idx = torch.arange(0, flat_tensor.numel(), stride,
        device=flat_tensor.device, dtype=torch.long)
    # index_select uses a gather kernel that issues uncoalesced loads
    return torch.index_select(flat_tensor, 0, idx)

# Usage
n, stride = 1 << 20, 2
inp = torch.arange(n * stride, device='cuda',
                   dtype=torch.float32)
out = uncoalesced_copy(inp, stride)
```

This PyTorch snippet uses torch.index_select with a strided index pattern, which causes the underlying GPU gather kernel to perform uncoalesced loads. Specifically, a warp of 32 threads will access addresses that are stride \* 4 bytes apart.

这段 PyTorch 代码使用带跨步索引模式的 torch.index_select，导致底层的 GPU gather 核函数执行非合并加载。具体来说，一个含 32 个线程的 warp 会访问彼此相距 stride \* 4 字节的地址。

This does not allow a single wide transaction and instead generates 32 separate loads. Each thread loads a value from inp that is far apart in memory from the value loaded by the next thread, which prevents coalescing. Figure 7-3 shows the coalesced versus strided access pattern—as well as random access.

这就无法完成单个宽事务，反而生成 32 次独立的加载。每个线程从 inp 加载的值，与相邻线程加载的值在内存中相距甚远，从而阻碍了合并。图 7-3 展示了合并访问与跨步访问模式——以及随机访问。

![Figure 7-3. Coalesced, strided, and random memory access patterns](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-3.png)

![图 7-3. 合并、跨步与随机内存访问模式](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-3.png)

After running the C++ and PyTorch codes on a GPU, we measure performance metrics. In the uncoalesced version, each warp’s memory request is broken into 8 separate 32-byte sectors on average. Since each 128-byte cache line fetch is split into 4 separate 32-byte sectors (more on this number 4 in a bit), the access pattern spans two lines per warp to retrieve the 8 separate 32-byte sectors.

在 GPU 上运行这些 C++ 与 PyTorch 代码后，我们测量性能指标。在非合并版本中，每个 warp 的内存请求平均被拆成 8 个独立的 32 字节扇区。由于每次 128 字节缓存行取数会被拆为 4 个独立的 32 字节扇区（稍后详述这个数字 4），该访问模式每个 warp 会横跨两条缓存行，以取回那 8 个独立的 32 字节扇区。

In the unoptimized version, each warp’s uncoalesced loads break a single logical request into up to eight separate 32-byte sectors, ballooning transaction counts and starving the memory pipeline. As a result, Nsight Compute reports only about 25% of sustained peak dynamic random-access memory (DRAM) throughput.

在未优化版本中，每个 warp 的非合并加载会把单个逻辑请求拆成多达八个独立的 32 字节扇区，令事务数量急剧膨胀，并使内存流水线陷入“饥饿”。结果是，Nsight Compute 报告的持续动态随机存取存储器（DRAM）吞吐仅约为峰值的 25%。

Now, let’s optimize this by coalescing the memory accesses and making threads read contiguous elements so each warp issues fewer, larger transactions.

现在，让我们通过合并内存访问、让线程读取连续元素来优化它，从而使每个 warp 发出更少、更大的事务。

_After example (C++): coalesced access._ By simply removing the stride (or setting it to 1), each thread copies a contiguous element. This alignment of thread accesses allows the hardware to coalesce memory requests into full 128-byte transactions:

_后置示例（C++）：合并访问。_ 只需去掉步长（或将其设为 1），每个线程就拷贝一个连续元素。这种线程访问的对齐让硬件能把内存请求合并为完整的 128 字节事务：

```
#include <cuda_runtime.h>
#include <iostream>

__global__ void coalescedCopy(const float* __restrict__ in,
                             float* __restrict__ out,
                             int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Contiguous load
        // Threads copy neighboring elements
        out[idx] = in[idx];
    }
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    // 1) Allocate pinned host memory
    float *h_in = nullptr, *h_out = nullptr;
    cudaMallocHost(&h_in,  bytes);  // page-locked host alloc
    cudaMallocHost(&h_out, bytes);

    // 2) Initialize input
    for (int i = 0; i < n; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    // 3) Allocate device memory
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in,  bytes);
    cudaMalloc(&d_out, bytes);

    // 4) Copy to device
    cudaMemcpy(d_in, h_in, bytes,
      cudaMemcpyHostToDevice);

    // 5) Launch kernel
    dim3 block(256);

    dim3 grid((n + block.x - 1) / block.x);
    coalescedCopy<<<grid, block>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    // 6) Copy back to host
    cudaMemcpy(h_out, d_out, bytes,
      cudaMemcpyDeviceToHost);

    // 7) Clean up
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);

    return 0;
}
```

```
#include <cuda_runtime.h>
#include <iostream>

__global__ void coalescedCopy(const float* __restrict__ in,
                             float* __restrict__ out,
                             int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Contiguous load
        // Threads copy neighboring elements
        out[idx] = in[idx];
    }
}

int main() {
    const int n = 1 << 20;
    const size_t bytes = n * sizeof(float);

    // 1) Allocate pinned host memory
    float *h_in = nullptr, *h_out = nullptr;
    cudaMallocHost(&h_in,  bytes);  // page-locked host alloc
    cudaMallocHost(&h_out, bytes);

    // 2) Initialize input
    for (int i = 0; i < n; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    // 3) Allocate device memory
    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in,  bytes);
    cudaMalloc(&d_out, bytes);

    // 4) Copy to device
    cudaMemcpy(d_in, h_in, bytes,
      cudaMemcpyHostToDevice);

    // 5) Launch kernel
    dim3 block(256);

    dim3 grid((n + block.x - 1) / block.x);
    coalescedCopy<<<grid, block>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    // 6) Copy back to host
    cudaMemcpy(h_out, d_out, bytes,
      cudaMemcpyDeviceToHost);

    // 7) Clean up
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFreeHost(h_in);
    cudaFreeHost(h_out);

    return 0;
}
```

We leave out the coalesced PyTorch implementation for brevity since it’s already built into PyTorch. A coalesced version in PyTorch would simply do something like out = inp.clone(). This copies contiguous elements efficiently. In fact, clone() on a contiguous tensor uses a vectorized memory copy under the hood, analogous to our coalesced kernel.

为简洁起见，我们省略了合并版本的 PyTorch 实现，因为它已内置于 PyTorch 中。PyTorch 里的合并版本只需做类似 out = inp.clone() 的操作即可。这会高效地拷贝连续元素。事实上，在连续张量上调用 clone() 会在底层使用向量化内存拷贝，与我们的合并核函数类似。

> You can also use torch.compile with the default TorchInductor backend to reduce redundant copies and to fuse adjacent operations when safe. clone() is already a device-to-device copy, so it’s typically not fused into a separate custom kernel. Enable autotuning (e.g., mode="max-autotune") to help TorchInductor pick coalesced and vectorized schedules when shapes are stable. We’ll cover the PyTorch compiler in depth in Chapters 13 and 14.

> 你也可以使用带默认 TorchInductor 后端的 torch.compile 来减少冗余拷贝，并在安全时融合相邻算子。clone() 本身已是设备到设备的拷贝，因此通常不会被融合进一个单独的自定义核函数。启用自动调优（autotuning，例如 mode="max-autotune"）有助于在形状稳定时让 TorchInductor 选择合并且向量化的调度方案。我们将在第 13 章和第 14 章深入讲解 PyTorch 编译器。

With coalesced access, or no stride, each warp’s threads access adjacent addresses. In this case, the hardware coalesces each warp’s loads into the minimum number of 128-byte transactions—often one 128-byte transaction per warp when the first address is 128-byte aligned, or two if the access straddles a boundary. Table 7-1 shows the dramatic improvements resulting from this optimization.

在合并访问下，或者说无步长时，每个 warp 的线程访问相邻地址。此时，硬件把每个 warp 的加载合并为最少数量的 128 字节事务——当首地址为 128 字节对齐时，往往是每个 warp 一个 128 字节事务；若访问跨越了边界，则为两个。表 7-1 展示了此优化带来的巨大改善。

_Table 7-1. Coalesced versus uncoalesced memory access performance_

_表 7-1. 合并与非合并内存访问的性能对比_

| Metric                        | Before (uncoalesced) | After (coalesced) |
| ----------------------------- | -------------------- | ----------------- |
| DRAM throughput (% of peak)   | 25%                  | 90% (3.6×)        |
| Global Memory Load Efficiency | 23%                  | 99%               |
| Average sectors per request   | 8.0                  | 4.0 (optimal)     |
| SM Active %                   | 62%                  | 99%               |
| Kernel execution time (ms)    | 4.8 ms               | 1.3 ms (3.7×)     |

| 指标                      | 优化前（非合并） | 优化后（合并） |
| ------------------------- | ---------------- | -------------- |
| DRAM 吞吐（占峰值百分比） | 25%              | 90% (3.6×)     |
| 全局内存加载效率          | 23%              | 99%            |
| 每次请求平均扇区数        | 8.0              | 4.0（最优）    |
| SM 活跃百分比             | 62%              | 99%            |
| 核函数执行时间（ms）      | 4.8 ms           | 1.3 ms (3.7×)  |

> Note: The numeric values in all metrics tables are illustrative to explain the concepts. For actual benchmark results on different GPU architectures, see the GitHub repository.

> 注：所有指标表格中的数值均为示意性质，用于阐释概念。不同 GPU 架构上的实际基准测试结果，请参见 GitHub 仓库。

After fixing the data layout and coalescing global memory accesses, DRAM throughput rises from 25% to 90% of peak, about 3.6× higher. Kernel execution time improves by about 3.7×, from 4.8 ms to 1.3 ms, as fewer stalls free the SMs to make forward progress.

在修正数据布局并合并全局内存访问之后，DRAM 吞吐从峰值的 25% 升至 90%，约提高 3.6×。核函数执行时间改善约 3.7×，从 4.8 ms 降到 1.3 ms，因为更少的停顿让 SM 得以持续向前推进。

Global Memory Load Efficiency rises from 23% to 99%, meaning nearly every fetched byte is useful. At the same time, Average Sectors per Request falls to about 4.0.

全局内存加载效率从 23% 升至 99%，意味着几乎每个取回的字节都是有用的。与此同时，每次请求平均扇区数降至约 4.0。

> A value near 4.0 means the warp’s loads are fully coalesced. With the first address 128-byte aligned, all 32 threads map to one 128-byte line. This line has four 32-byte sectors, so the metric reports an average of 4.0 sectors per request.

> 接近 4.0 的值意味着该 warp 的加载已完全合并。当首地址为 128 字节对齐时，全部 32 个线程都映射到一条 128 字节缓存行。这条缓存行有四个 32 字节扇区，因此该指标报告每次请求平均 4.0 个扇区。

In the worst case with strided or scattered access, the value can approach 32 because activity is reported in 32-byte sectors at L2. Values above 4.0 indicate uncoalesced or misaligned access. In an unoptimized version, uncoalesced loads can break a single logical access into many sectors per request, approaching the worst case.

在最坏情况下，即跨步或分散访问时，该值可能接近 32，因为在 L2 处活动是以 32 字节扇区为单位报告的。高于 4.0 的值表明存在非合并或未对齐的访问。在未优化版本中，非合并加载会把单个逻辑访问拆成每次请求许多扇区，逼近最坏情况。

Here we are at about 4.0 sectors per request, which indicates that requests map cleanly to 128-byte lines with no unused sectors. With fewer memory stalls, SM Active percent improves from 62% to 99%.

这里我们处于每次请求约 4.0 个扇区的水平，表明请求干净地映射到 128 字节缓存行，没有未使用的扇区。随着内存停顿减少，SM 活跃百分比从 62% 提升到 99%。

In summary, by aligning each warp’s threads on successive addresses, the GPU’s memory controller can service each warp with a few large transactions instead of dozens of tiny ones. This boosts the Global Memory Load Efficiency (fraction of each transaction that returns useful data) to near 100% and raises SM Active % by keeping warps busy instead of idle.

总而言之，通过让每个 warp 的线程对齐到连续地址，GPU 的内存控制器就能用少数几个大事务、而非数十个微小事务来服务每个 warp。这会把全局内存加载效率（每个事务中返回有用数据的比例）提升到接近 100%，并通过让 warp 保持忙碌而非空闲来抬高 SM 活跃百分比。

And with that, you’ve seen how coalescing reshapes your thread-to-address mapping so an entire warp’s loads line up on the GPU’s 128-byte segments. However, even a fully coalesced warp still issues 32 individual 4-byte reads under the hood, one per thread, forcing the hardware to stitch them back together into each 128-byte transaction.

至此，你已经看到合并如何重塑线程到地址的映射，使整个 warp 的加载对齐到 GPU 的 128 字节段上。然而，即便是完全合并的 warp，底层仍会发出 32 次独立的 4 字节读取——每个线程一次——迫使硬件把它们重新拼合成每个 128 字节事务。

To eliminate that last layer of inefficiency, we turn to vectorized memory access: having each thread fetch a wider, aligned chunk (e.g., a 16-byte float4) in a single instruction so that a coalesced warp issues exactly four 128-byte transactions, not 32. Let’s dive into how to pack your per-thread loads into CUDA’s built-in vector types.

要消除这最后一层低效，我们转向向量化内存访问：让每个线程在单条指令中取回一个更宽、对齐的数据块（例如一个 16 字节的 float4），使一个合并的 warp 恰好发出四个 128 字节事务，而非 32 个。下面我们深入探讨如何把每线程的加载打包进 CUDA 的内建向量类型。

## Vectorized Memory Access

## 向量化内存访问

While memory coalescing is a runtime hardware optimization on NVIDIA GPUs, vectorized memory access is a compile-time strategy in which each load or store instruction explicitly fetches multiple contiguous elements (e.g., float4, or 16 bytes) per thread. This reduces instruction count and eliminates stitching overhead.

内存合并是 NVIDIA GPU 上的一种运行期硬件优化，而向量化内存访问则是一种编译期策略——每条加载或存储指令显式地为每个线程取回多个连续元素（例如 float4，即 16 字节）。这既降低了指令数量，又消除了拼合开销。

Efficient global‐memory access on Blackwell relies on matching your loads to the GPU’s native 128-byte transaction size. When each thread reads only a 4-byte float, a 32-thread warp still has to stitch together 32 4-byte requests to fill one 128-byte line.

Blackwell 上高效的全局内存访问依赖于让你的加载与 GPU 原生的 128 字节事务大小相匹配。当每个线程只读取一个 4 字节的 float 时，一个 32 线程的 warp 仍需把 32 个 4 字节请求拼合起来才能填满一条 128 字节缓存行。

In ideal coalesced cases, a warp that reads 32 4-byte words aligned to a 128-byte boundary maps to exactly four 32-byte sectors. Nsight Compute captures this waste in the average sectors per request metric, which can climb well above 4.0 when your accesses are misaligned, strided, or scattered. As such, you’ll see Global Memory Load Efficiency drop as bandwidth goes underutilized.

在理想的合并情形下，一个读取 32 个对齐到 128 字节边界的 4 字节字的 warp 恰好映射到四个 32 字节扇区。Nsight Compute 通过每次请求平均扇区数这一指标捕捉到这种浪费——当你的访问未对齐、跨步或分散时，该值可能远超 4.0。这样一来，你会看到全局内存加载效率随着带宽被闲置而下降。

The fix is to bundle each thread’s work into a larger, naturally aligned vector that maps cleanly onto those 128-byte transactions. CUDA’s built-in float4 type does exactly that: it packs four 4-byte floats into a 16-byte struct guaranteed by the compiler to be 16-byte aligned. For illustration purposes, it looks something like the following code:

修复之道是把每个线程的工作打包成一个更大、天然对齐的向量，让它干净地映射到那些 128 字节事务上。CUDA 内建的 float4 类型正是如此：它把四个 4 字节的 float 打包进一个 16 字节的结构体，编译器保证其为 16 字节对齐。为便于说明，它大致如下所示：

```
// Example of a custom float4 for illustration purposes.
// Note: it's recommended to use the built-in CUDA float4 type.
// Note: if N is not divisible by 4, handle the last N % 4 elements
// (e.g., a short scalar cleanup) or assert divisibility in host code. The
// input and output pointers must be 16-byte aligned for float4 loads/stores.
struct my_float4 {
    float x;  // 4 bytes
    float y;  // 4 bytes
    float z;  // 4 bytes
    float w;  // 4 bytes
};
```

```
// Example of a custom float4 for illustration purposes.
// Note: it's recommended to use the built-in CUDA float4 type.
// Note: if N is not divisible by 4, handle the last N % 4 elements
// (e.g., a short scalar cleanup) or assert divisibility in host code. The
// input and output pointers must be 16-byte aligned for float4 loads/stores.
struct my_float4 {
    float x;  // 4 bytes
    float y;  // 4 bytes
    float z;  // 4 bytes
    float w;  // 4 bytes
};
```

When all 32 threads in a warp issue a float4 load, they together fetch 32 × 16 bytes = 512 bytes of contiguous data, which the Blackwell memory controller then splits into exactly four 128-byte transactions (512 bytes ÷ 128 bytes per transaction = 4 transactions). This preserves the ideal 4.0 sectors per 128-byte transaction (discussed earlier re: coalesced memory access) because the warp requests 512 bytes, and the hardware services the request with four aligned 128-byte transactions for a total of 16 sectors, each fully utilized.

当一个 warp 中全部 32 个线程都发出一次 float4 加载时，它们合起来取回 32 × 16 字节 = 512 字节的连续数据，Blackwell 内存控制器随后把它恰好拆成四个 128 字节事务（512 字节 ÷ 每事务 128 字节 = 4 个事务）。这就保持了理想的每个 128 字节事务 4.0 个扇区（前文讨论合并内存访问时提到过），因为该 warp 请求 512 字节，硬件用四个对齐的 128 字节事务、共计 16 个扇区来服务此请求，且每个扇区都被完全利用。

Compared to an unaligned or strided case that can inflate sector counts, vectorized float4 loads reduce per-thread load instructions by 4×. This helps maintain the ideal 4.0 sectors per request when alignment is satisfied.

与可能令扇区数膨胀的未对齐或跨步情形相比，向量化的 float4 加载把每线程的加载指令减少了 4×。在满足对齐时，这有助于维持理想的每次请求 4.0 个扇区。

> High-level GPU compilers like the PyTorch compiler can often generate vectorized memory operations when alignment and contiguity requirements are met. You can encourage this behavior by using per-thread tiles that match vector widths. As such, frameworks can usually achieve a similar reduction in memory transactions under the hood. However, it’s still beneficial to align data properly from the start.

> 像 PyTorch 编译器这样的高层 GPU 编译器，在满足对齐与连续性要求时，往往能生成向量化的内存操作。你可以通过使用与向量宽度相匹配的每线程分块来促成这一行为。因此，各类框架通常也能在底层实现类似的内存事务缩减。不过，从一开始就恰当地对齐数据仍然是有益的。

As a result of this vectorized memory access optimization, global memory load efficiency increases toward 100%. Sectors per transaction remain at 4.0 when each 128-byte transaction is fully utilized, while the warp issues multiple aligned transactions to serve a wider request.

作为这项向量化内存访问优化的结果，全局内存加载效率朝 100% 攀升。当每个 128 字节事务被完全利用时，每事务扇区数保持在 4.0，同时该 warp 发出多个对齐事务来服务一个更宽的请求。

Vectorized loads reduce the number of memory instructions per byte moved and often increase effective bandwidth. To benefit from vectorized loads, data pointers must be aligned to the vector width. The CUDA runtime and driver allocation functions, such as cudaMalloc return device pointers aligned to at least 256 bytes.

向量化加载降低了每移动一字节所需的内存指令数，往往还能提升有效带宽。要从向量化加载中获益，数据指针必须对齐到向量宽度。CUDA 运行时与驱动的分配函数（例如 cudaMalloc）返回的设备指针至少对齐到 256 字节。

The base alignment is sufficient for 16-byte and 32-byte vector alignment at the allocation boundary. Adding an element offset can break alignment if the offset is not a multiple of the vector width. The following cast asserts the intended type to the compiler and assumes the pointer is already correctly aligned:

这一基础对齐足以在分配边界处满足 16 字节和 32 字节的向量对齐。如果加上的元素偏移量不是向量宽度的整数倍，添加偏移量可能会破坏对齐。下面的类型转换向编译器断言了预期类型，并假设指针已经正确对齐：

```
auto ptr4 = reinterpret_cast<const float4*>(ptr);
```

```
auto ptr4 = reinterpret_cast<const float4*>(ptr);
```

Make sure the pointer value is a multiple of 16 bytes (4 floats) before casting. This is because cudaMalloc() returns at least 256-byte alignment. Adding an element offset can break alignment if the offset is not a multiple of 4 floats. Correct alignment is a precondition for vectorized loads.

在转换之前，务必确保指针值是 16 字节（4 个 float）的整数倍。这是因为 cudaMalloc() 返回至少 256 字节的对齐。如果加上的元素偏移量不是 4 个 float 的整数倍，添加偏移量可能会破坏对齐。正确的对齐是向量化加载的前提条件。

With a base address aligned to 32 bytes, each thread reads 32 bytes per iteration, typically compiling into two 16-byte vector load instructions per thread (Hopper) or one 32-byte load instruction (Blackwell). A 32-thread warp therefore requests 1,024 bytes. Each 128-byte cache line comprises four 32-byte sectors. As such, the warp generates eight aligned 128-byte transactions for a total of 32 sectors when accesses are aligned properly. This way, each access falls on a natural boundary and avoids split transactions.

在基址对齐到 32 字节的情况下，每个线程每次迭代读取 32 字节，通常在 Hopper 上编译为每线程两条 16 字节向量加载指令，或在 Blackwell 上编译为一条 32 字节加载指令。于是一个 32 线程的 warp 请求 1,024 字节。每条 128 字节缓存行由四个 32 字节扇区组成。因此，当访问恰当对齐时，该 warp 生成八个对齐的 128 字节事务，共计 32 个扇区。这样一来，每次访问都落在天然边界上，避免了被拆分的事务。

It’s worth noting that CUDA, as of this writing, does not provide a built-in 8-float aggregate CUDA vector type (e.g., eight combined float32s) in its <vector_types.h>. You can define your own struct of 8 float values, then load it as two float4 values per thread. You would use 32-byte alignment on the type with alignas(32). This way, the pointer value is a multiple of 32 bytes. Global memory vector instructions on modern GPUs typically load 16 bytes per thread (Hopper) or 32 bytes per thread (Blackwell). In the case of Hopper’s 16 bytes per thread, an 8-float aggregate load will compile into two 16-byte loads per thread to include all 32 bytes.

值得注意的是，截至本文撰写时，CUDA 在其 <vector_types.h> 中并未提供内建的 8-float 聚合 CUDA 向量类型（例如八个组合的 float32）。你可以自定义一个含 8 个 float 值的结构体，然后每线程将其作为两个 float4 值加载。你会用 alignas(32) 为该类型施加 32 字节对齐。这样一来，指针值就是 32 字节的整数倍。现代 GPU 上的全局内存向量指令通常每线程加载 16 字节（Hopper）或每线程 32 字节（Blackwell）。在 Hopper 每线程 16 字节的情形下，一次 8-float 聚合加载会编译为每线程两次 16 字节加载，以涵盖全部 32 字节。

Blackwell, on the other hand, would compile into a single 32-byte load per thread when using proper 32-byte aligned data.

而 Blackwell 在使用恰当的 32 字节对齐数据时，则会编译为每线程一次 32 字节加载。

With 32-byte per-thread loads (e.g., 8 floats × 4 bytes per float), a warp moves 1,024 bytes (1,024 bytes = 32 threads per warp × 32 bytes per thread). This typically maps to 8 × 128-byte lines since each line consists of four 32-byte sectors. With 16-byte per-thread loads (e.g., four float4s), a warp moves 512 bytes (512 bytes = 32 threads per warp × 16 bytes per thread) or 4 × 128-byte lines. When the base pointer and per-thread stride are naturally aligned, the hardware coalescer combines lane accesses into full lines and avoids split sectors. This type of mechanical sympathy is crucial to achieving peak throughput. Therefore, for 256-bit loads on Blackwell, you should enforce 32-byte alignment to maintain peak performance.

在每线程 32 字节加载（例如 8 个 float × 每 float 4 字节）时，一个 warp 移动 1,024 字节（1,024 字节 = 每 warp 32 个线程 × 每线程 32 字节）。由于每条缓存行由四个 32 字节扇区组成，这通常映射到 8 × 128 字节缓存行。在每线程 16 字节加载（例如四个 float4）时，一个 warp 移动 512 字节（512 字节 = 每 warp 32 个线程 × 每线程 16 字节），即 4 × 128 字节缓存行。当基指针与每线程步长天然对齐时，硬件合并器会把各通道（lane）的访问合并为完整缓存行，避免拆分扇区。这种“机械同理心”（mechanical sympathy）对于达到峰值吞吐至关重要。因此，对于 Blackwell 上的 256 位加载，你应当强制 32 字节对齐以维持峰值性能。

Let’s see an example of scalar versus vectorized memory access in a simple vector copy kernel.

我们来看一个简单向量拷贝核函数中标量与向量化内存访问的对比示例。

_Before example (C++): scalar copy._ Each thread copies one float:

_前置示例（C++）：标量拷贝。_ 每个线程拷贝一个 float：

```
#include <cuda_runtime.h>

__global__ void copyScalar(
const float* __restrict__ in,
float* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Scalar load: 4-byte copy per thread
        out[idx] = in[idx];
    }
}

int main() {
    const int N = 1 << 20;

    float* h_in = nullptr;
    float* h_out = nullptr;
    cudaMallocHost(&h_in, N * sizeof(float));
    cudaMallocHost(&h_out, N * sizeof(float));

    for (int i = 0; i < N; ++i) h_in[i] = float(i);

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256), grid((N + 255) / 256);
    copyScalar<<<grid, block>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaFree(d_in); cudaFree(d_out);
    cudaFreeHost(h_in);

    cudaFreeHost(h_out);
    return 0;
}
```

```
#include <cuda_runtime.h>

__global__ void copyScalar(
const float* __restrict__ in,
float* __restrict__ out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Scalar load: 4-byte copy per thread
        out[idx] = in[idx];
    }
}

int main() {
    const int N = 1 << 20;

    float* h_in = nullptr;
    float* h_out = nullptr;
    cudaMallocHost(&h_in, N * sizeof(float));
    cudaMallocHost(&h_out, N * sizeof(float));

    for (int i = 0; i < N; ++i) h_in[i] = float(i);

    float *d_in, *d_out;
    cudaMalloc(&d_in, N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256), grid((N + 255) / 256);
    copyScalar<<<grid, block>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    cudaFree(d_in); cudaFree(d_out);
    cudaFreeHost(h_in);

    cudaFreeHost(h_out);
    return 0;
}
```

_Before example (PyTorch)_. A scalar elementwise copy in PyTorch could be done with a Python loop for illustration:

_前置示例（PyTorch）。_ 在 PyTorch 中，一个标量逐元素拷贝可以用 Python 循环来演示：

```
import torch

def copy_scalar(inp: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(inp)
    flat_in = inp.view(-1)
    flat_out = out.view(-1)
    for i in range(flat_in.numel()):
        # Each iteration issues a 4-byte load on the GPU
        # This is extremely slow. DO NOT DO THIS!
        # Use vectorized operations to avoid Python loops
        # on GPU tensors as shown in optimized version
        flat_out[i] = flat_in[i]
    return out

# Usage
N = 1 << 20
inp = torch.arange(N, device='cuda', dtype=torch.float32)
out = copy_scalar(inp)
```

```
import torch

def copy_scalar(inp: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(inp)
    flat_in = inp.view(-1)
    flat_out = out.view(-1)
    for i in range(flat_in.numel()):
        # Each iteration issues a 4-byte load on the GPU
        # This is extremely slow. DO NOT DO THIS!
        # Use vectorized operations to avoid Python loops
        # on GPU tensors as shown in optimized version
        flat_out[i] = flat_in[i]
    return out

# Usage
N = 1 << 20
inp = torch.arange(N, device='cuda', dtype=torch.float32)
out = copy_scalar(inp)
```

With scalar 4-byte loads, a warp often issues one 128-byte transaction for 128 bytes when aligned. Otherwise, it uses two transactions if the access straddles a 128-byte boundary. Using float4 (16 bytes per thread) means each warp issues 4× more data per memory instruction—512 bytes per warp—split into four 128-byte transactions. This allows the transfer to complete in fewer instructions. Next, let’s optimize by using vector loads.

在标量 4 字节加载下，当对齐时，一个 warp 往往为 128 字节发出一个 128 字节事务。否则，若访问跨越了 128 字节边界，就会使用两个事务。使用 float4（每线程 16 字节）意味着每个 warp 每条内存指令发出的数据量多 4×——每 warp 512 字节——拆成四个 128 字节事务。这让传输能以更少的指令完成。接下来，我们通过使用向量加载来优化。

_After example (C++): vectorized copy._ Each thread copies a float4 (16 bytes):

_后置示例（C++）：向量化拷贝。_ 每个线程拷贝一个 float4（16 字节）：

```
#include <cuda_runtime.h>

// 16-byte (128-bit) vector copy: one float4 per thread
static_assert(alignof(float4) == 16, "float4 alignment must be 16 bytes");
__global__ void copyVector16B(const float4* __restrict__ in,
                              float4* __restrict__ out,
                              int N4)  // number of float4 elements
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N4) {
        // Per-thread 16B load+store.
        // On sm_90, NVCC emits ld.global.v4.f32 / st.global.v4.f32.
        // for 16-byte vector loads and stores
        // On sm_90 (Hopper) (Blackwell), NVCC emits
        // ld.global.v8.f32 and st.global.v8.f32 for 32-byte aligned data
        out[idx] = in[idx];
    }

}

int main() {
    const int N  = 1 << 20;   // total number of floats
    const int N4 = N / 4;     // total number of float4s (16B chunks)

    float4 *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in,  N4 * sizeof(float4));   // >=256B aligned
    cudaMalloc(&d_out, N4 * sizeof(float4));

    dim3 block(256);
    dim3 grid((N4 + block.x - 1) / block.x);
    copyVector16B<<<grid, block>>>(d_in, d_out, N4);
    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
```

```
#include <cuda_runtime.h>

// 16-byte (128-bit) vector copy: one float4 per thread
static_assert(alignof(float4) == 16, "float4 alignment must be 16 bytes");
__global__ void copyVector16B(const float4* __restrict__ in,
                              float4* __restrict__ out,
                              int N4)  // number of float4 elements
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N4) {
        // Per-thread 16B load+store.
        // On sm_90, NVCC emits ld.global.v4.f32 / st.global.v4.f32.
        // for 16-byte vector loads and stores
        // On sm_90 (Hopper) (Blackwell), NVCC emits
        // ld.global.v8.f32 and st.global.v8.f32 for 32-byte aligned data
        out[idx] = in[idx];
    }

}

int main() {
    const int N  = 1 << 20;   // total number of floats
    const int N4 = N / 4;     // total number of float4s (16B chunks)

    float4 *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in,  N4 * sizeof(float4));   // >=256B aligned
    cudaMalloc(&d_out, N4 * sizeof(float4));

    dim3 block(256);
    dim3 grid((N4 + block.x - 1) / block.x);
    copyVector16B<<<grid, block>>>(d_in, d_out, N4);
    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
```

Here we use CUDA’s built-in float4 type from <cuda_runtime.h>. We launch N/4 threads to maintain 16-byte alignment and issue true vector loads/stores, and each thread copies one float4 (16 bytes). This means each thread handles four floats, and each warp handles 128 floats total in four transactions. cudaMalloc returns pointers aligned to at least 256 bytes which satisfies the float4 (16-byte) requirement and helps with 32-byte aligned vectors when you use 32-byte alignment for the data’s starting address. It’s important to note that misaligned casts can forfeit vectorization. Following is an example that takes advantage of Blackwell’s support for 32-byte vectorized loads:

这里我们使用来自 <cuda_runtime.h> 的 CUDA 内建 float4 类型。我们启动 N/4 个线程以维持 16 字节对齐并发出真正的向量加载/存储，每个线程拷贝一个 float4（16 字节）。这意味着每个线程处理四个 float，每个 warp 在四个事务中总共处理 128 个 float。cudaMalloc 返回至少对齐到 256 字节的指针，这既满足 float4（16 字节）的要求，也有助于在你为数据起始地址采用 32 字节对齐时使用 32 字节对齐的向量。需要注意的是，未对齐的类型转换会葬送向量化。下面是一个利用 Blackwell 对 32 字节向量化加载支持的示例：

```
// --- Blackwell-only variant: 32-byte per-thread vector copy
// (PTX: ld.global.v8.f32)
// Requires 32B alignment.
#include <cuda_runtime.h>

// Proof of 32-byte alignment
static_assert(alignof(float8) == 32, "float8 alignment must be 32 bytes");
struct alignas(32) float8 { float v[8]; };

__global__ void copyVector32B(const float8* __restrict__ in,
                              float8* __restrict__ out, int N8) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N8) {
        // Per-thread 32B load+store
        // NVCC emits ld.global.v8.f32/st.global.v8.f32 on sm_100
        out[idx] = in[idx];
    }
}

int main() {
    const int N = 1 << 20;

    const int N8 = N / 8;

    float8 *d_in, *d_out;
    // cudaMalloc returns pointers aligned to ≥256 bytes
    // This will allow the compiler to emit
    // optimized ld.global.v8.f32 and st.global.v8.f32 instructions
    cudaMalloc(&d_in,  N8 * sizeof(float8));
    cudaMalloc(&d_out, N8 * sizeof(float8));

    dim3 block(256), grid((N8 + block.x - 1) / block.x);
    copyVector32B<<<grid, block>>>(d_in, d_out, N8);
    cudaDeviceSynchronize();

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
```

```
// --- Blackwell-only variant: 32-byte per-thread vector copy
// (PTX: ld.global.v8.f32)
// Requires 32B alignment.
#include <cuda_runtime.h>

// Proof of 32-byte alignment
static_assert(alignof(float8) == 32, "float8 alignment must be 32 bytes");
struct alignas(32) float8 { float v[8]; };

__global__ void copyVector32B(const float8* __restrict__ in,
                              float8* __restrict__ out, int N8) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N8) {
        // Per-thread 32B load+store
        // NVCC emits ld.global.v8.f32/st.global.v8.f32 on sm_100
        out[idx] = in[idx];
    }
}

int main() {
    const int N = 1 << 20;

    const int N8 = N / 8;

    float8 *d_in, *d_out;
    // cudaMalloc returns pointers aligned to ≥256 bytes
    // This will allow the compiler to emit
    // optimized ld.global.v8.f32 and st.global.v8.f32 instructions
    cudaMalloc(&d_in,  N8 * sizeof(float8));
    cudaMalloc(&d_out, N8 * sizeof(float8));

    dim3 block(256), grid((N8 + block.x - 1) / block.x);
    copyVector32B<<<grid, block>>>(d_in, d_out, N8);
    cudaDeviceSynchronize();

    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
```

_After example (PyTorch)_. We can simulate vectorized copy by using a tensor view and clone:

_后置示例（PyTorch）。_ 我们可以通过使用张量视图与 clone 来模拟向量化拷贝：

```
import torch

def copy_vectorized(inp: torch.Tensor) -> torch.Tensor:
    # Reshape into groups of 4 floats for bulk copy
    vec = inp.view(-1, 4)
    # clone() on a contiguous CUDA tensor performs a device-to-device copy
    # using optimized runtime paths such as cudaMemcpyAsync().

    out_vec = vec.clone()
    return out_vec.view(-1)

# Usage
N = 1 << 20
inp = torch.arange(N, device='cuda', dtype=torch.float32)
out = copy_vectorized(inp)
```

```
import torch

def copy_vectorized(inp: torch.Tensor) -> torch.Tensor:
    # Reshape into groups of 4 floats for bulk copy
    vec = inp.view(-1, 4)
    # clone() on a contiguous CUDA tensor performs a device-to-device copy
    # using optimized runtime paths such as cudaMemcpyAsync().

    out_vec = vec.clone()
    return out_vec.view(-1)

# Usage
N = 1 << 20
inp = torch.arange(N, device='cuda', dtype=torch.float32)
out = copy_vectorized(inp)
```

Calling clone() on the reshaped tensor causes PyTorch to perform contiguous copies. This is in contrast to copying element by element. By using float4 vector loads, each thread moves 16 bytes per instruction, which is optimized for Hopper. (Blackwell supports 32-byte vector loads per thread.) A warp issues 512 bytes for that load, typically split into four 128-byte transactions when aligned. The benefit comes from fewer instructions per byte moved, better sector utilization, and proper alignment to maximize memory throughput on modern GPUs. This shows far better utilization of memory bandwidth, as shown in Table 7-2.

在重塑后的张量上调用 clone() 会让 PyTorch 执行连续拷贝。这与逐元素拷贝形成对比。通过使用 float4 向量加载，每个线程每条指令移动 16 字节，这一点针对 Hopper 做了优化。（Blackwell 支持每线程 32 字节向量加载。）一个 warp 为该次加载发出 512 字节，当对齐时通常拆成四个 128 字节事务。收益来自每移动一字节所需指令更少、更好的扇区利用率，以及恰当的对齐，从而在现代 GPU 上最大化内存吞吐。这体现出对内存带宽好得多的利用率，如表 7-2 所示。

> Blackwell doubles the number of bytes for vector loads/stores per thread relative to Hopper. With CUDA 13, Blackwell can perform 32-byte vector loads and stores instead of 16 bytes on Hopper. This requires that the compiler can prove 32-byte alignment. Otherwise, it may split the per-thread load/store into two 16-byte instructions. This will increase the number of instructions required to load/store the same amount of data which will negatively impact performance.

> Blackwell 相对于 Hopper，把每线程向量加载/存储的字节数翻了一倍。借助 CUDA 13，Blackwell 可以执行 32 字节的向量加载与存储，而不是 Hopper 上的 16 字节。这要求编译器能够证明 32 字节对齐。否则，它可能把每线程的加载/存储拆成两条 16 字节指令。这会增加加载/存储相同数据量所需的指令数，从而对性能产生负面影响。

By using properly aligned float4 vector loads, each thread moves 16 bytes per instruction. This shows far better utilization of memory bandwidth, as shown in the results in Table 7-2.

通过使用恰当对齐的 float4 向量加载，每个线程每条指令移动 16 字节。这体现出对内存带宽好得多的利用率，如表 7-2 的结果所示。

_Table 7-2. Nsight Compute metrics for scalar versus vectorizing memory access_

_表 7-2. 标量与向量化内存访问的 Nsight Compute 指标对比_

| Metric                        | Before (scalar) | After (vectorized)        |
| ----------------------------- | --------------- | ------------------------- |
| Global Memory Load Efficiency | 28%             | 97%                       |
| Average sectors per request   | 31.8            | 4.0                       |
| DRAM throughput (% of peak)   | 25%             | 90% (3.6×)                |
| Kernel execution time         | 4.2 ms          | 1.2 ms (3.5× improvement) |

| 指标                      | 优化前（标量） | 优化后（向量化）    |
| ------------------------- | -------------- | ------------------- |
| 全局内存加载效率          | 28%            | 97%                 |
| 每次请求平均扇区数        | 31.8           | 4.0                 |
| DRAM 吞吐（占峰值百分比） | 25%            | 90% (3.6×)          |
| 核函数执行时间            | 4.2 ms         | 1.2 ms（改善 3.5×） |

These metrics confirm the improvement. Global Memory Load Efficiency jumped from 28% to 97% (~3.5×), and percentage of peak DRAM throughput increased from 25% to 90% (~3.6×), trimming the overall kernel runtime by ~3.5×.

这些指标印证了改善。全局内存加载效率从 28% 跃升至 97%（约 3.5×），占峰值的 DRAM 吞吐百分比从 25% 提升到 90%（约 3.6×），把核函数整体运行时间削减了约 3.5×。

This optimization trimmed the kernel execution runtime by roughly 3.5× from 4.2 ms down to 1.2 ms. Global Memory Load Efficiency jumps from 28% (scalar) to 97% (vectorized), meaning that almost every fetched 128-byte transaction is now used.

这项优化把核函数执行运行时间削减了大约 3.5×，从 4.2 ms 降到 1.2 ms。全局内存加载效率从 28%（标量）跃升至 97%（向量化），意味着现在几乎每个取回的 128 字节事务都被用上。

The average sectors per request drops toward the 4.0 sector-per-line ideal value. This indicates that each 128-byte line is fully utilized. Each warp-wide float4 instruction issues four aligned 128-byte transactions to move 512 bytes. As such, every transaction is fully utilized.

每次请求平均扇区数朝着每缓存行 4.0 扇区的理想值下降。这表明每条 128 字节缓存行都被完全利用。每条 warp 级（warp-wide）的 float4 指令发出四个对齐的 128 字节事务来移动 512 字节。因此，每个事务都被完全利用。

The reduction in memory requests and the replacement of four scalar loads with one vector load increase global memory load efficiency from 28% to 97% and raise sustained DRAM throughput from about 25% to about 90% of peak. In this case, warps are doing far fewer memory operations and get more data with each operation, so they spend much less time waiting on memory and more time executing useful work. The end result is a significantly faster kernel.

内存请求的减少，以及用一次向量加载替代四次标量加载，把全局内存加载效率从 28% 提升到 97%，并把持续 DRAM 吞吐从约峰值的 25% 抬高到约 90%。在这种情况下，warp 执行的内存操作大为减少，且每次操作获取更多数据，因此它们花在等待内存上的时间大大减少，而花在执行有用工作上的时间更多。最终结果是一个显著更快的核函数。

> Vectorization reduces instruction count. Coalescing and alignment improve the number of sectors per request.

> 向量化减少指令数量。合并与对齐则改善每次请求的扇区数。

Let’s quickly compare coalesced global memory access from the previous section to vectorized memory access discussed here. Coalescing makes sure threads in a warp all hit one big contiguous block. Vectorizing makes sure each thread fetches a wide chunk that exactly maps onto those big blocks. The differences are summarized in Table 7-3.

我们快速对比一下上一节的合并全局内存访问与这里讨论的向量化内存访问。合并确保一个 warp 中的线程全部命中一个大的连续块。向量化确保每个线程取回一个恰好映射到那些大块上的宽数据块。二者的区别汇总于表 7-3。

_Table 7-3. Comparing coalesced memory access (previous section) and vectorized memory access_

_表 7-3. 对比合并内存访问（上一节）与向量化内存访问_

| Aspect                   | Coalesced warp load (interthread)                                                   | Vectorizing (intrathread)                                                                                                            |
| ------------------------ | ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Granularity              | Threads ↔ contiguous memory addresses                                               | Per thread ↔ vector loads                                                                                                            |
| Typical fix              | Contiguous indexing (stride = 1)                                                    | Use float4, a custom 32-byte struct with explicit 32-byte alignment, or a 32-byte-aligned vector type (e.g., Blackwell and CUDA 13+) |
| Transactions/warp        | 32 threads × 4 bytes → 128 bytes, equal to 4 32-byte sectors if aligned (4 sectors) | 32 threads × 16 bytes → 512 bytes, equal to 4 128-byte transactions, which correspond to 16 32-byte sectors (16 sectors)             |
| Instruction count change | Same number of loads                                                                | 1 vector load instead of 4 scalars                                                                                                   |
| Benefit                  | Fewer scattered sectors                                                             | Perfectly sized, fewer sectors                                                                                                       |

| 方面           | 合并 warp 加载（线程间，interthread）                                | 向量化（线程内，intrathread）                                                                                 |
| -------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| 粒度           | 线程 ↔ 连续内存地址                                                  | 每线程 ↔ 向量加载                                                                                             |
| 典型修复方式   | 连续索引（stride = 1）                                               | 使用 float4、带显式 32 字节对齐的自定义 32 字节结构体，或 32 字节对齐的向量类型（例如 Blackwell 与 CUDA 13+） |
| 每 warp 事务数 | 32 线程 × 4 字节 → 128 字节，对齐时等于 4 个 32 字节扇区（4 个扇区） | 32 线程 × 16 字节 → 512 字节，等于 4 个 128 字节事务，对应 16 个 32 字节扇区（16 个扇区）                     |
| 指令数量变化   | 加载数量相同                                                         | 1 次向量加载替代 4 次标量加载                                                                                 |
| 收益           | 更少的分散扇区                                                       | 大小完美、更少的扇区                                                                                          |

By combining both strategies, you ensure that warps hit contiguous blocks such that each thread fetches a wide, aligned vector. This helps you push your kernel’s memory bandwidth utilization to its peak and unlock the full power of your GPU.

通过组合这两种策略，你就能确保 warp 命中连续块，同时让每个线程取回一个宽而对齐的向量。这有助于你把核函数的内存带宽利用率推向峰值，并释放 GPU 的全部威力。

It’s worth noting that, prior to Blackwell and CUDA 13, global memory vector loads were limited to 16 bytes (128 bits) per thread. However, Blackwell and CUDA 13 added 32-byte (256-bit) load/store instructions and data types for specific vector types with 32-byte alignment.

值得注意的是，在 Blackwell 与 CUDA 13 之前，全局内存向量加载在每个线程上被限制为 16 字节（128 位）。不过，Blackwell 与 CUDA 13 为具有 32 字节对齐的特定向量类型新增了 32 字节（256 位）的加载/存储指令与数据类型。

When available, prefer these wider 32-byte types and instructions for user-defined 8-float aggregates. This will reduce the number of instructions needed to load and store wider 32-byte aligned data.

在可用时，对于用户自定义的 8 个 float 的聚合体，应优先使用这些更宽的 32 字节类型与指令。这样可以减少加载和存储更宽的 32 字节对齐数据所需的指令数量。

Custom 8-float aggregates will still compile into two 16-byte loads unless you explicitly use the 32-byte-aligned types that map to a single 32-byte instruction.

自定义的 8 个 float 聚合体仍会被编译成两条 16 字节加载，除非你显式使用映射到单条 32 字节指令的 32 字节对齐类型。

> Even though a Blackwell thread can load a full 32-byte vector type, the memory coalescer can still only service requests in 128-byte chunks, or four 32-byte sectors. On Blackwell, a 32-thread warp moving 32 B per thread transfers 1024 B (8 × 128 B lines). Hopper’s 16 B/thread variant moves 512 B (4 × 128 B). Note that the transaction count scales with bytes moved. Both are fully efficient when properly aligned: 16-byte (Hopper) or 32-byte (Blackwell and beyond).

> 尽管 Blackwell 线程可以加载一个完整的 32 字节向量类型，内存合并器（memory coalescer）仍然只能以 128 字节为单位、也就是四个 32 字节扇区（sector）来处理请求。在 Blackwell 上，一个 32 线程的 warp 每线程搬运 32 B，共传输 1024 B（8 × 128 B 缓存行）。Hopper 的 16 B/线程 变体搬运 512 B（4 × 128 B）。注意事务数量随搬运的字节数一起增长。只要正确对齐，两者都是完全高效的：16 字节（Hopper）或 32 字节（Blackwell 及更新架构）。

## Tiling and Data Reuse Using Shared Memory

## 使用共享内存进行分块与数据复用

A common performance pitfall is repeatedly reading the same data from global memory. _Tiling_ is a technique to avoid this by loading chunks of data into faster on-chip shared memory—and reusing those chunks across many threads.

一个常见的性能陷阱是反复从全局内存读取相同的数据。_分块_（tiling）是一种避免这种情况的技术：把数据块加载到更快的片上共享内存中——并让这些数据块在许多线程之间复用。

For example, a naive matrix multiplication of size _N_ × _N_ might load each element of matrix A from HBM _N_ times, once for each row of B it multiplies with. This results in N–1 redundant loads per element. And on Blackwell, which can easily execute tens of teraFLOPS (TFLOPS), redundant loads can waste memory bandwidth, which could otherwise be feeding more math operations to the GPU SMs.

例如，一个 _N_ × _N_ 大小的朴素矩阵乘法可能会把矩阵 A 的每个元素从 HBM 加载 _N_ 次，每次对应它要相乘的 B 的一行。这会导致每个元素有 N–1 次冗余加载。而在 Blackwell 上，它可以轻松执行数十 teraFLOPS（TFLOPS），冗余加载会浪费内存带宽，而这些带宽本可以为 GPU 的 SM 提供更多数学运算。

Tiling eliminates this waste by having each thread block pull a small submatrix (a _tile_) of A and B into shared memory exactly once. It then reuses the cached values across all threads for multiple multiply-accumulate operations. In our next example, we’ll use a 32 × 32 tile, which is a common choice that fits well in shared memory.

分块通过让每个线程块只把 A 和 B 的一个小子矩阵（一个*瓦片*，tile）拉入共享内存恰好一次，来消除这种浪费。随后它在所有线程之间复用这些被缓存的值，进行多次乘加运算。在下一个示例中，我们会使用 32 × 32 的瓦片，这是一个能很好地放入共享内存的常见选择。

Threads within a block can cooperatively load the tile into shared memory, then call \_\_syncthreads() to synchronize the data. Then the threads perform parallel matrix-multiply computations using the data in shared memory. This amortizes the global memory access cost over many threads and computations. It’s worth noting that these tile loads are also arranged to be coalesced. Specifically, each warp loads a full 128-byte segment from global memory into shared memory—consistent with the coalescing example from earlier.

一个线程块内的线程可以协作地把瓦片加载到共享内存中，然后调用 \_\_syncthreads() 来同步数据。之后，线程们使用共享内存中的数据执行并行的矩阵乘法计算。这样就把全局内存访问的开销摊分到了许多线程和计算之上。值得注意的是，这些瓦片加载也被安排成合并的（coalesced）。具体来说，每个 warp 从全局内存把一个完整的 128 字节段加载到共享内存中——与前面的合并示例保持一致。

By reading each element from DRAM only once (into shared memory) and reusing it for many calculations, we reduce global memory traffic. Let’s illustrate this with an _N_ × _N_ matrix multiplication example. First, consider a naive implementation.

通过把每个元素从 DRAM 只读取一次（读入共享内存）并复用它进行许多次计算，我们减少了全局内存流量。让我们用一个 _N_ × _N_ 矩阵乘法示例来说明这一点。首先，考虑一个朴素实现。

_Before example (CUDA C++): naive matrix multiply_. Each thread computes one element of the result matrix C, reading entire rows of A and columns of B from global memory for each output:

_前例（CUDA C++）：朴素矩阵乘法_。每个线程计算结果矩阵 C 的一个元素，为每个输出从全局内存读取 A 的整行和 B 的整列：

```
#include <cuda_runtime.h>
#include <iostream>

__global__ void naiveMatMul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            // Each thread loads A[row, k] and B[k, col]
            // from global memory for every k.
            // This is very memory heavy.
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;

    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * N * sizeof(float);

    float* h_A = nullptr;
    float* h_B = nullptr;
    float* h_C = nullptr;
    cudaMallocHost(&h_A, N*N * sizeof(float));
    cudaMallocHost(&h_B, N*N * sizeof(float));
    cudaMallocHost(&h_C, N*N * sizeof(float));

    for (int i = 0; i < N*N; ++i) { h_A[i] = 1.0f; h_B[i] = 1.0f; }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (N + 31) / 32);
    naiveMatMul<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    return 0;
}
```

```
#include <cuda_runtime.h>
#include <iostream>

__global__ void naiveMatMul(const float* A, const float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            // Each thread loads A[row, k] and B[k, col]
            // from global memory for every k.
            // This is very memory heavy.
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;

    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * N * sizeof(float);

    float* h_A = nullptr;
    float* h_B = nullptr;
    float* h_C = nullptr;
    cudaMallocHost(&h_A, N*N * sizeof(float));
    cudaMallocHost(&h_B, N*N * sizeof(float));
    cudaMallocHost(&h_C, N*N * sizeof(float));

    for (int i = 0; i < N*N; ++i) { h_A[i] = 1.0f; h_B[i] = 1.0f; }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (N + 31) / 32);
    naiveMatMul<<<grid, block>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    return 0;
}
```

This CUDA C++ kernel issues global memory loads for every multiplication inside the inner loop. Each thread reads A[row, k] and B[k, col] from DRAM for every _k_, causing massive redundant traffic. The result is a heavily memory-bound kernel with low SM utilization and frequent stalls waiting on global memory. Here is the PyTorch version of the naive matrix multiplication:

这个 CUDA C++ 核函数在内层循环里为每一次乘法都发起全局内存加载。每个线程为每一个 _k_ 从 DRAM 读取 A[row, k] 和 B[k, col]，造成海量的冗余流量。其结果是一个严重访存受限（memory-bound）的核函数，SM 利用率很低，并且频繁因等待全局内存而停顿。下面是朴素矩阵乘法的 PyTorch 版本：

```
import torch

def naive_matmul(A, B):
    N = A.size(0)
    C = torch.zeros((N, N), device='cuda')
    for i in range(N):
        for j in range(N):
            # Each dot product loads A[i,:] B[:,j] from global memory repeatedly
            C[i, j] = (A[i, :] * B[:, j]).sum()
    return C
```

```
import torch

def naive_matmul(A, B):
    N = A.size(0)
    C = torch.zeros((N, N), device='cuda')
    for i in range(N):
        for j in range(N):
            # Each dot product loads A[i,:] B[:,j] from global memory repeatedly
            C[i, j] = (A[i, :] * B[:, j]).sum()
    return C
```

```
# Usage
N = 1024
A = torch.ones((N, N), device='cuda', dtype=torch.float32)
B = torch.ones((N, N), device='cuda', dtype=torch.float32)
C = naive_matmul(A, B)
```

```
# Usage
N = 1024
A = torch.ones((N, N), device='cuda', dtype=torch.float32)
B = torch.ones((N, N), device='cuda', dtype=torch.float32)
C = naive_matmul(A, B)
```

This PyTorch implementation uses nested Python loops. While the innermost operations are offloaded to GPU as elementwise multiply and sum operations, it still triggers repeated global memory loads under the hood for each dot product. This mimics the memory-bound behavior of the naive CUDA kernel since the GPU spends most cycles waiting on memory rather than computing multiplications.

这个 PyTorch 实现使用了嵌套的 Python 循环。虽然最内层的操作会作为逐元素乘法和求和操作卸载到 GPU 上，它在底层仍然会为每个点积触发重复的全局内存加载。这模拟了朴素 CUDA 核函数的访存受限行为，因为 GPU 把大部分周期都花在等待内存上，而不是计算乘法。

> This PyTorch code is extremely slow on purpose to illustrate the extreme case of performing redundant global memory access loads inside of a loop. In practice, frameworks like PyTorch use optimized kernels for this type of operation.

> 这段 PyTorch 代码故意写得极其缓慢，以此说明在循环内执行冗余全局内存访问加载的极端情形。在实践中，像 PyTorch 这样的框架会对这类操作使用优化过的核函数。

Now, let’s apply tiling to improve this. We divide the matrices into 32 × 32 tiles. 32 × 32 is a convenient tile size since it aligns with a warp size of 32, it fits well in shared memory, and it maps to a full warp of 32 threads reading each row. This allows each warp to collaboratively load and process one single row (per tile) at a time.

现在，让我们应用分块来改进它。我们把矩阵划分成 32 × 32 的瓦片。32 × 32 是一个方便的瓦片尺寸，因为它与 32 的 warp 大小对齐，能很好地放入共享内存，并且映射到一个完整的 32 线程 warp 来读取每一行。这让每个 warp 能够协作地一次加载并处理（每个瓦片的）单独一行。

As such, each thread block loads one 32 × 32 tile of A and one 32 × 32 tile of B into shared memory, performs the 32 × 32 matrix multiplies, and accumulates the results. This way, each element of A and B is loaded from HBM only once per tile instead of 32 times in the naive version. The optimized version using shared memory to cache tiles for matrix multiplies is shown here:

因此，每个线程块把 A 的一个 32 × 32 瓦片和 B 的一个 32 × 32 瓦片加载到共享内存中，执行 32 × 32 的矩阵乘法，并累加结果。这样，A 和 B 的每个元素每个瓦片只从 HBM 加载一次，而不是像朴素版本那样加载 32 次。下面展示了使用共享内存缓存瓦片以进行矩阵乘法的优化版本：

```
#include <cuda_runtime.h>
#include <iostream>
#define TILE_SIZE 32

__global__ void tiledMatMul(const float* A, const float* B, float* C, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // compute partial results using the tile
    // in shared memory
    for (int t = 0; t < N; t += TILE_SIZE) {
        // Cooperative load of a tile of A and B into shared memory
        // Load tile A with boundary check
        if (row < N && (t + threadIdx.x) < N) {
            sA[threadIdx.y][threadIdx.x] = A[row * N + t + threadIdx.x];
        } else {

            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile B with boundary check
        if ((t + threadIdx.y) < N && col < N) {
            sB[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // We will optimize this later to use a
        // thread-block-scoped cooperative-groups barrier
        __syncthreads();

        // Compute using the tile loaded in shared memory
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        // We will optimize this later to use a
        // thread-block-scoped cooperative-groups barrier
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * N * sizeof(float);

    float* h_A = nullptr;
    float* h_B = nullptr;
    float* h_C = nullptr;
    cudaMallocHost(&h_A, N*N * sizeof(float));
    cudaMallocHost(&h_B, N*N * sizeof(float));
    cudaMallocHost(&h_C, N*N * sizeof(float));

    for (int i = 0; i < N*N; ++i) { h_A[i] = 1.0f; h_B[i] = 1.0f; }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    tiledMatMul<<<grid, block>>>(d_A, d_B, d_C, N);

    // synchronize the kernel with the device
    // for timing accuracy
    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
```

```
#include <cuda_runtime.h>
#include <iostream>
#define TILE_SIZE 32

__global__ void tiledMatMul(const float* A, const float* B, float* C, int N) {
    __shared__ float sA[TILE_SIZE][TILE_SIZE];
    __shared__ float sB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    // compute partial results using the tile
    // in shared memory
    for (int t = 0; t < N; t += TILE_SIZE) {
        // Cooperative load of a tile of A and B into shared memory
        // Load tile A with boundary check
        if (row < N && (t + threadIdx.x) < N) {
            sA[threadIdx.y][threadIdx.x] = A[row * N + t + threadIdx.x];
        } else {

            sA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load tile B with boundary check
        if ((t + threadIdx.y) < N && col < N) {
            sB[threadIdx.y][threadIdx.x] = B[(t + threadIdx.y) * N + col];
        } else {
            sB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // We will optimize this later to use a
        // thread-block-scoped cooperative-groups barrier
        __syncthreads();

        // Compute using the tile loaded in shared memory
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += sA[threadIdx.y][k] * sB[k][threadIdx.x];
        }
        // We will optimize this later to use a
        // thread-block-scoped cooperative-groups barrier
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    const int N = 1024;
    size_t bytes = N * N * sizeof(float);

    float* h_A = nullptr;
    float* h_B = nullptr;
    float* h_C = nullptr;
    cudaMallocHost(&h_A, N*N * sizeof(float));
    cudaMallocHost(&h_B, N*N * sizeof(float));
    cudaMallocHost(&h_C, N*N * sizeof(float));

    for (int i = 0; i < N*N; ++i) { h_A[i] = 1.0f; h_B[i] = 1.0f; }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);
    tiledMatMul<<<grid, block>>>(d_A, d_B, d_C, N);

    // synchronize the kernel with the device
    // for timing accuracy
    cudaDeviceSynchronize();

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
```

In this tiled kernel, each block cooperatively loads a 32 × 32 tile of A (into sA) and a 32 × 32 tile of B (into sB) from global memory. These loads happen in the first two lines inside the loop and are followed by \_\_syncthreads() to ensure the tile is fully loaded before use.

在这个分块核函数中，每个线程块协作地从全局内存把 A 的一个 32 × 32 瓦片（加载到 sA）和 B 的一个 32 × 32 瓦片（加载到 sB）加载进来。这些加载发生在循环内部的前两行，随后是 \_\_syncthreads()，以确保瓦片在使用前已完全加载。

Then the block performs 32 × 32 multiply-accumulate operations using the shared memory tiles. This inner loop, running over k in steps of 32, reuses each loaded value 32×, yielding a 32× reduction in global memory reads for those elements.

然后，线程块使用这些共享内存瓦片执行 32 × 32 次乘加运算。这个以 32 为步长在 k 上运行的内层循环，把每个加载进来的值复用 32×，从而使这些元素的全局内存读取减少 32×。

After finishing all tile iterations, each thread writes its result to C. The result is a dramatic 8× reduction in global memory accesses per thread, as you’ll see in a bit.

在完成所有瓦片迭代后，每个线程把它的结果写入 C。其结果是每个线程的全局内存访问量大幅减少 8×，你稍后就会看到。

> These examples are focused on memory access patterns and are using FP32 CUDA cores—not the reduced-precision Tensor Cores. Chapter 9 demonstrates the use of reduced-precision computations (e.g., 16-bit, 8-bit, 4-bit) to improve performance even further.

> 这些示例聚焦于内存访问模式，使用的是 FP32 CUDA 核心——而非低精度的 Tensor Core。第 9 章会演示使用低精度计算（例如 16 位、8 位、4 位）来进一步提升性能。

This PyTorch version manually tiles the matrices and invokes torch.mm on each tile. PyTorch’s torch.mm leverages NVIDIA’s cuBLAS and CUTLASS, which implement shared memory tiling and reuse internally at the C++ level. Here is the PyTorch version of the tiled matrix multiply:

这个 PyTorch 版本手动对矩阵进行分块，并对每个瓦片调用 torch.mm。PyTorch 的 torch.mm 利用了 NVIDIA 的 cuBLAS 和 CUTLASS，它们在 C++ 层面内部实现了共享内存分块与复用。下面是分块矩阵乘法的 PyTorch 版本：

```
import torch

def tiled_matmul(A, B, tile_size=32):
    N = A.size(0)
    C = torch.zeros((N, N), device='cuda')
    for i in range(0, N, tile_size):
        for j in range(0, N, tile_size):
            C_block = torch.zeros((tile_size, tile_size), device='cuda')
            for k in range(0, N, tile_size):
                A_block = A[i:i+tile_size, k:k+tile_size]
                B_block = B[k:k+tile_size, j:j+tile_size]
                # torch.mm uses an optimized kernel (likely tiling internally)
                C_block += torch.mm(A_block, B_block)
            C[i:i+tile_size, j:j+tile_size] = C_block

    return C

# Usage
N = 1024
A = torch.ones((N, N), device='cuda', dtype=torch.float32)
B = torch.ones((N, N), device='cuda', dtype=torch.float32)
C = tiled_matmul(A, B)
```

```
import torch

def tiled_matmul(A, B, tile_size=32):
    N = A.size(0)
    C = torch.zeros((N, N), device='cuda')
    for i in range(0, N, tile_size):
        for j in range(0, N, tile_size):
            C_block = torch.zeros((tile_size, tile_size), device='cuda')
            for k in range(0, N, tile_size):
                A_block = A[i:i+tile_size, k:k+tile_size]
                B_block = B[k:k+tile_size, j:j+tile_size]
                # torch.mm uses an optimized kernel (likely tiling internally)
                C_block += torch.mm(A_block, B_block)
            C[i:i+tile_size, j:j+tile_size] = C_block

    return C

# Usage
N = 1024
A = torch.ones((N, N), device='cuda', dtype=torch.float32)
B = torch.ones((N, N), device='cuda', dtype=torch.float32)
C = tiled_matmul(A, B)
```

> With a 32 × 32 tile, threads in a warp will contend for the same shared-memory bank when they all access the same column. In practice, you can avoid this by padding the tile to 33 columns—or using techniques like swizzling—so that each access falls in a different bank. We will apply this optimization in the next section. For now, let’s focus on tiling as an optimization on its own.

> 在 32 × 32 的瓦片下，当一个 warp 中的线程都访问同一列时，它们会争用同一个共享内存 bank。在实践中，你可以通过把瓦片填充到 33 列——或者使用像混洗（swizzling）这样的技术——来避免这一点，使得每次访问落在不同的 bank 上。我们会在下一节应用这项优化。目前，先让我们把分块本身作为一项优化来关注。

The performance impact of tiling is significant. By structuring the computation in tiles and reusing data in shared memory, we reduce DRAM traffic and raise arithmetic intensity because each byte retrieved from memory is used for many more floating-point operations. Along with fewer global memory transactions, we also observe higher achieved occupancy. Table 7-4 compares key metrics before and after applying shared memory tiling.

分块带来的性能影响是显著的。通过以瓦片方式组织计算并在共享内存中复用数据，我们减少了 DRAM 流量并提高了算术强度（arithmetic intensity），因为从内存取回的每个字节都被用于更多的浮点运算。除了更少的全局内存事务，我们还观察到更高的实测占用率（occupancy）。表 7-4 比较了应用共享内存分块前后的关键指标。

_Table 7-4. Performance improvement with shared memory tiling_

_表 7-4. 使用共享内存分块带来的性能提升_

| Metric                               | Before (naive kernel) | After (tiled kernel) | Notes                                                                                                                                                                                                                            |
| ------------------------------------ | --------------------- | -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DRAM throughput (% of peak)          | 90%                   | 25% (3.6× less)      | Naive uses more bandwidth, but it's wasteful. The tile implementation is lower because it's more efficient after tiling due to fewer redundant loads.                                                                            |
| Achieved occupancy (%)               | 42%                   | 89%                  | More resident warps make forward progress with fewer stalls.                                                                                                                                                                     |
| Floating-point throughput (GFLOPS)   | 15 GFLOPS             | 170 GFLOPS           | Small matrix size limits the absolute value. Reuse increases sustained compute. (The absolute GFLOPS numbers are low because the matrix size (N = 1,024) is small and the kernel is memory limited.)                             |
| Global memory load sectors           | 9800                  | 1200                 | Each element is fetched once per tile instead of 32 times (once per thread). Total number of 32-byte sector reads issued by the kernel. The decrease reflects the elimination of redundant loads thanks to shared-memory tiling. |
| Shared memory throughput (% of peak) | 52%                   | 99%                  | Tiled kernel access pattern avoids memory-bank conflicts on sA and sB.                                                                                                                                                           |

| 指标                         | 之前（naive kernel） | 之后（tiled kernel） | 备注                                                                                                                                                 |
| ---------------------------- | -------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- |
| DRAM 吞吐（占峰值百分比）    | 90%                  | 25%（减少 3.6×）     | 朴素版本使用了更多带宽，但很浪费。分块实现的值更低，是因为分块后由于冗余加载更少而更高效。                                                           |
| 实测占用率（%）              | 42%                  | 89%                  | 更多驻留 warp 能以更少的停顿向前推进。                                                                                                               |
| 浮点吞吐（GFLOPS）           | 15 GFLOPS            | 170 GFLOPS           | 较小的矩阵尺寸限制了其绝对值。复用提升了持续算力。（绝对 GFLOPS 数值偏低是因为矩阵尺寸（N = 1,024）较小且核函数受内存限制。）                        |
| 全局内存加载扇区数           | 9800                 | 1200                 | 每个元素每个瓦片只被取一次，而不是取 32 次（每线程一次）。这是核函数发起的 32 字节扇区读取的总数。这一下降反映了得益于共享内存分块而消除了冗余加载。 |
| 共享内存吞吐（占峰值百分比） | 52%                  | 99%                  | 分块核函数的访问模式避免了对 sA 和 sB 的内存 bank 冲突。                                                                                             |

Staging 32 × 32 tiles in shared memory makes sure that each element is fetched once from DRAM and reused across all threads in a block. DRAM throughput falls from 90% to 25%, a reduction of about 3.6×, which is expected when redundant traffic is removed.

把 32 × 32 的瓦片暂存到共享内存中，确保了每个元素从 DRAM 取一次并在一个线程块内的所有线程之间复用。DRAM 吞吐从 90% 降到 25%，大约减少了 3.6×，这在移除冗余流量后是符合预期的。

The drop in DRAM throughput is desirable because the kernel now performs more work per byte moved. Arithmetic intensity increases, and sustained floating-point throughput rises from 15 GFLOPS to 170 GFLOPS, a gain of nearly 11×.

DRAM 吞吐的下降是我们想要的，因为核函数现在每搬运一个字节所完成的工作更多了。算术强度上升，持续浮点吞吐从 15 GFLOPS 升到 170 GFLOPS，提升近 11×。

> The optimized 170 GFLOPS is far below Blackwell’s theoretical FP32 peak near 80 TFLOPS. This is expected for small tiles and frequent memory access. The important result is the 11× improvement after removing the memory bottleneck. Larger problems or more compute work per byte would move performance closer to peak.

> 优化后的 170 GFLOPS 远低于 Blackwell 接近 80 TFLOPS 的理论 FP32 峰值。这对于小瓦片和频繁的内存访问来说是符合预期的。重要的结果是消除内存瓶颈后带来的 11× 提升。更大的问题规模或每字节更多的计算量会让性能更接近峰值。

This shift in arithmetic intensity moves the kernel from memory bound toward compute bound, which is ideal because performance becomes limited by abundant floating-point throughput rather than comparatively scarce chip bandwidth. Global memory load sectors drop substantially from 9,800 down to 1,200 because each element is fetched once into shared memory and reused across threads. This eliminates redundant loads. The tiled kernel also accesses sA and sB in a way that avoids shared-memory bank conflicts, which is why shared memory throughput approaches 100%.

这种算术强度上的转变把核函数从访存受限推向计算受限（compute-bound），这是理想的，因为性能此时受限于充裕的浮点吞吐，而不是相对稀缺的芯片带宽。全局内存加载扇区数从 9,800 大幅下降到 1,200，因为每个元素只被取一次到共享内存中并在线程之间复用。这消除了冗余加载。分块核函数还以一种避免共享内存 bank 冲突（bank conflict）的方式访问 sA 和 sB，这就是共享内存吞吐接近 100% 的原因。

Overall, we shifted this kernel from memory bound to compute bound, which was the goal. We successfully relieved memory pressure, freed up the memory bus for other useful work, and achieved higher compute throughput.

总的来说，我们把这个核函数从访存受限转变为了计算受限，而这正是目标所在。我们成功缓解了内存压力，为其他有用工作腾出了内存总线，并实现了更高的算力吞吐。

We also see a large increase in achieved occupancy percentage from 42% to 89%. This metric nearly doubles because the tiled kernel keeps allowing more resident warps to make progress with fewer stalls. As such, the SMs remain busy more consistently.

我们还看到实测占用率百分比从 42% 大幅提升到 89%。这个指标几乎翻倍，是因为分块核函数持续允许更多驻留 warp 以更少的停顿向前推进。因此，SM 更持续地保持繁忙。

By introducing shared memory tiling, we increased per-thread work without increasing per-thread resource usage beyond available registers and shared memory. This helped our kernel achieve higher occupancy and utilization, which is important on GPUs like Blackwell, which offer a very large register file (65,536 32-bit registers per SM) that threads can exploit up to the 255-register per-thread limit without spilling.

通过引入共享内存分块，我们在不让每线程资源使用超过可用寄存器和共享内存的前提下，增加了每线程的工作量。这帮助我们的核函数达到了更高的占用率和利用率，这在像 Blackwell 这样的 GPU 上很重要——它提供了非常大的寄存器文件（每个 SM 有 65,536 个 32 位寄存器），线程可以在不发生溢出的情况下利用到每线程 255 个寄存器的上限。

> Recall that each thread can use up to 255 registers. Our tiling kernel implementation stays within this limit, avoids register spilling, and preserves high performance.

> 请回想，每个线程最多可以使用 255 个寄存器。我们的分块核函数实现保持在这个上限之内，避免了寄存器溢出（register spilling），并保持了高性能。

We can compare the arithmetic intensity of the naive and tiled kernels by examining their sustained FLOP rates and average DRAM bandwidth. The naive version ran at 15 GFLOPS while moving 10 GB/s of data on average, or 1.5 FLOPS per byte. The tiled implementation, by contrast, sustained 170 GFLOPS while moving 21 GB/s on average, or 8 FLOPS/byte.

我们可以通过考察朴素核函数和分块核函数的持续 FLOP 速率以及平均 DRAM 带宽，来比较它们的算术强度。朴素版本以 15 GFLOPS 运行，同时平均搬运 10 GB/s 的数据，即每字节 1.5 FLOPS。相比之下，分块实现持续达到 170 GFLOPS，同时平均搬运 21 GB/s，即 8 FLOPS/字节。

It might be tempting to increase the tile size further to 64 × 64 or 128 × 128 in order to reduce memory traffic even more and increase data reuse. Just remember that larger tiles consume more on-chip resources, including both registers and shared memory. This leaves less capacity for additional thread blocks on each SM.

也许有人会想把瓦片尺寸进一步增大到 64 × 64 或 128 × 128，以便进一步减少内存流量并增加数据复用。但要记住，更大的瓦片会消耗更多片上资源，包括寄存器和共享内存。这会为每个 SM 上的额外线程块留下更少的容量。

For instance, Blackwell GPUs provide up to 228 KB of allocatable shared memory per SM. This can easily accommodate a 64 × 64 tile of 4-byte floats in which each A and B tile requires 16 KB (16,384 bytes = 64 × 64 floats × 4 bytes per float). However, in addition to shared-memory usage, you need to budget for the additional registers per thread.

例如，Blackwell GPU 每个 SM 提供多达 228 KB 的可分配共享内存。这可以轻松容纳一个由 4 字节 float 构成的 64 × 64 瓦片，其中每个 A 瓦片和 B 瓦片各需要 16 KB（16,384 字节 = 64 × 64 个 float × 每个 float 4 字节）。然而，除了共享内存的使用之外，你还需要为每线程的额外寄存器留出预算。

Each Blackwell SM provides up to 64K registers (32-bit each) in total with a per-thread maximum of 255 registers. When you launch a 32 × 32 thread block (1,024 threads) to compute a 64 × 64 tile, each thread handles a 2 × 2 subtile, which produces 4 outputs.

每个 Blackwell SM 总共提供多达 64K 个寄存器（每个 32 位），每线程最大为 255 个寄存器。当你启动一个 32 × 32 的线程块（1,024 个线程）来计算一个 64 × 64 的瓦片时，每个线程处理一个 2 × 2 的子瓦片，产生 4 个输出。

For a 64 × 64 tile size, you need the following: four accumulator registers (one per output), two registers for A-tile elements (reused across two outputs), two registers for B-tile elements (same reuse), and ~4 registers for loop counters, thread indices, and address arithmetic. In total, this is ~12 registers per thread.

对于 64 × 64 的瓦片尺寸，你需要如下寄存器：四个累加器寄存器（每个输出一个）、两个用于 A 瓦片元素的寄存器（在两个输出间复用）、两个用于 B 瓦片元素的寄存器（同样复用），以及约 4 个用于循环计数器、线程索引和地址运算的寄存器。总计约为每线程 12 个寄存器。

12 registers per thread × 1,024 threads per block = 12,288 registers needed per thread block. And since each Blackwell SM has 65,536 32-bit registers (maximum 255 registers per thread), in theory, up to 5 blocks’ worth of registers can fit (≈61,440 registers). However, the SM can support only 2,048 concurrent threads (64 warps). So, in practice, occupancy is limited by the smaller limits of registers, shared memory, warps, and threads. For this configuration, two blocks per SM will saturate the 2,048-thread limit.

每线程 12 个寄存器 × 每块 1,024 个线程 = 每个线程块需要 12,288 个寄存器。由于每个 Blackwell SM 有 65,536 个 32 位寄存器（每线程最多 255 个寄存器），理论上可以容纳多达 5 个线程块所需的寄存器（≈61,440 个寄存器）。然而，SM 只能支持 2,048 个并发线程（64 个 warp）。所以在实践中，占用率受寄存器、共享内存、warp 和线程这几者中较小的限制所约束。对于这种配置，每个 SM 两个线程块就会占满 2,048 线程的上限。

> If your kernel uses additional registers for double-buffering registers, vectorized loads, etc., you can recalculate the occupancy with cudaOccupancyMaxPotentialBlockSize or Nsight Compute’s occupancy report.

> 如果你的核函数为双缓冲（double buffering）寄存器、向量化加载等使用了额外的寄存器，你可以用 cudaOccupancyMaxPotentialBlockSize 或 Nsight Compute 的占用率报告重新计算占用率。

For a 64 × 64 tile size, 22 registers per thread are needed for a total of 22,528 registers for a 32 × 32 thread block (1,024 threads). In this case, you can only fit up to two thread blocks per Blackwell SM. This translates to a maximum occupancy of only 2,048 threads—hitting the SM’s thread limit with only two blocks (out of the 16 maximum concurrent blocks per SM).

对于 64 × 64 的瓦片尺寸，每线程需要 22 个寄存器，一个 32 × 32 线程块（1,024 个线程）总共需要 22,528 个寄存器。在这种情况下，每个 Blackwell SM 最多只能容纳两个线程块。这意味着最大占用率只有 2,048 个线程——只用两个线程块就撞上了 SM 的线程上限（每个 SM 最多 16 个并发线程块中的两个）。

A reduction in concurrent thread blocks caused by using a larger tile size will lower occupancy and hurt performance if you exceed the hardware’s limits. In practice, your tile dimensions must fit within the GPU’s shared-memory and register budgets to maintain high occupancy and throughput.

使用更大的瓦片尺寸导致并发线程块减少，如果你超出了硬件的限制，这会降低占用率并损害性能。在实践中，你的瓦片维度必须能放入 GPU 的共享内存和寄存器预算内，以维持高占用率和吞吐。

For a Blackwell GPU with ~228 KB of allocatable shared memory per SM, a 64 × 64 tile (~16 KB per input matrix tile) might still fit, but doubling the tile dimensions squares the reuse factor while quadrupling shared memory usage. There are diminishing returns and possible trade-offs.

对于每个 SM 有约 228 KB 可分配共享内存的 Blackwell GPU，一个 64 × 64 的瓦片（每个输入矩阵瓦片约 16 KB）也许仍然放得下，但把瓦片维度翻倍会让复用系数平方增长，同时让共享内存使用量翻两番。这里存在收益递减以及可能的权衡。

It is recommended to experiment with different tile dimensions to balance on-chip reuse against resource limits. A 32 × 32 tile is a solid starting point on modern NVIDIA GPUs, but depending on your shared-memory and register usage, you may find that a slightly smaller or larger tile delivers better throughput.

建议尝试不同的瓦片维度，以在片上复用与资源限制之间取得平衡。在现代 NVIDIA GPU 上，32 × 32 的瓦片是一个可靠的起点，但取决于你的共享内存和寄存器使用情况，你可能会发现稍小或稍大的瓦片能带来更好的吞吐。

> Libraries like CUTLASS also include profilers that automate this exploration, letting you find the optimal tile size for your kernel and hardware.

> 像 CUTLASS 这样的库也包含能自动化这种探索的剖析器（profiler），让你为你的核函数和硬件找到最优的瓦片尺寸。

In short, we transformed a naive, global-memory-only matrix multiplication into a tiled implementation that uses shared memory. This enabled cooperative data reuse, reduced the number of DRAM transactions, and boosted arithmetic intensity. Both the CUDA C++ and PyTorch implementations benefited from this tiling technique.

简而言之，我们把一个仅使用全局内存的朴素矩阵乘法转变成了使用共享内存的分块实现。这实现了协作式数据复用，减少了 DRAM 事务的数量，并提升了算术强度。CUDA C++ 和 PyTorch 两种实现都从这项分块技术中获益。

It’s worth noting that the tiling techniques we applied manually are exactly what high-performance GPU libraries do under the hood. NVIDIA’s CUTLASS library, for instance, provides templated components to implement general matrix multiplies (GEMMs) with multiple layers of tiling. These CUTLASS components load fragments of matrices into registers and shared memory—and then compute partial results much like our previous 32 × 32 tile example.

值得注意的是，我们手动应用的分块技术，正是高性能 GPU 库在底层所做的事情。举例来说，NVIDIA 的 CUTLASS 库提供了模板化组件，用多层分块来实现通用矩阵乘法（GEMM）。这些 CUTLASS 组件把矩阵的片段加载到寄存器和共享内存中——然后计算部分结果，很像我们前面 32 × 32 瓦片的示例。

In fact, NVIDIA’s optimized cuBLAS and cuDNN libraries use similar blocking strategies at the thread, warp, and block levels to achieve near-peak throughput. NVIDIA even announced a Python-first API in early 2025 called cuTile that lets programmers describe these tile shapes in a more convenient Pythonic way. In fact, NVIDIA has developed a Tile-based intermediate representation (IR) called TileIR to support cuTile and facilitate automatic compilation and tuning.

事实上，NVIDIA 优化过的 cuBLAS 和 cuDNN 库在线程、warp 和线程块层面使用了类似的分块策略，以达到接近峰值的吞吐。NVIDIA 甚至在 2025 年初宣布了一个 Python 优先的 API，称为 cuTile，让程序员能以更方便的 Python 风格描述这些瓦片形状。事实上，NVIDIA 已经开发了一种基于瓦片的中间表示（intermediate representation，IR），称为 TileIR，来支持 cuTile 并促进自动编译与调优。

Other high-performance libraries encapsulate these tiling patterns as well. For instance, NVIDIA’s CUTLASS C++ and Python libraries expose templated tile iterators and profilers. And PyTorch-based compilers like TorchInductor (using the OpenAI Triton library) generate tiled kernels automatically when shapes and alignment permit. These libraries lower the barrier to using these tiling optimizations— and reduce the amount of boilerplate code.

其他高性能库同样封装了这些分块模式。例如，NVIDIA 的 CUTLASS C++ 和 Python 库暴露了模板化的瓦片迭代器与剖析器。而像 TorchInductor（使用 OpenAI 的 Triton 库）这样基于 PyTorch 的编译器，会在形状和对齐允许时自动生成分块核函数。这些库降低了使用这些分块优化的门槛——并减少了样板代码的数量。

The key idea is that reusing data in registers/shared memory as much as possible before going back to DRAM is fundamental, and libraries encapsulate this. So whenever possible, leverage these highly optimized libraries (or refer to them) for performant tiling patterns.

关键思想在于：在返回 DRAM 之前尽可能多地在寄存器/共享内存中复用数据，这是根本性的，而各种库正封装了这一点。所以只要有可能，就利用（或参考）这些高度优化的库来获得高性能的分块模式。

For instance, if you use torch.mm in PyTorch or cublasSgemm in your code, under the covers it’s doing exactly this kind of tiling and memory coalescing. This is why our PyTorch example saw the same benefits automatically.

例如，如果你在 PyTorch 中使用 torch.mm，或在代码中使用 cublasSgemm，其底层做的正是这种分块与内存合并。这就是为什么我们的 PyTorch 示例会自动获得同样的收益。

In practice, you would use high-performance libraries like cuBLAS and PyTorch’s torch.matmul, which already implement tiling and other optimizations in C++. In production code, directly using torch.mm or torch.matmul would produce the same benefits—and possibly more, thanks to highly tuned kernels.

在实践中，你会使用像 cuBLAS 和 PyTorch 的 torch.matmul 这样的高性能库，它们已经用 C++ 实现了分块和其他优化。在生产代码中，直接使用 torch.mm 或 torch.matmul 会产生同样的收益——甚至更多，这要归功于高度调优的核函数。

> While you can definitely reuse existing tiling libraries and frameworks, understanding how they work, as we’ve done here, is invaluable for when you need to diagnose performance issues and possibly write your own custom kernels for specialized situations that these libraries and frameworks don’t cover. Just don’t forget to give back to the community as they’ve given you a lot!

> 虽然你完全可以复用现有的分块库和框架，但像我们这里所做的那样理解它们的工作原理是非常宝贵的——当你需要诊断性能问题，或者可能需要为这些库和框架未覆盖的专门场景编写自己的自定义核函数时尤其如此。只是别忘了回馈社区，因为他们给了你很多！

As mentioned earlier in this section, when using a 32 × 32 tile, the threads in a warp will contend for the same shared-memory bank when they access the same column. Let’s explain this issue—as well as some optimizations—in the next section.

正如本节前面所提到的，在使用 32 × 32 的瓦片时，当一个 warp 中的线程访问同一列时，它们会争用同一个共享内存 bank。让我们在下一节解释这个问题——以及一些优化方法。

## Avoid Shared-Memory Bank Conflicts

## 避免共享内存 bank 冲突

On modern NVIDIA GPUs, including Blackwell, shared memory has 32 banks with a 4-byte bank width (i.e., addresses map mod 32). As such, a warp that strides by 128 B (32 floats × 4 B) maps all threads to the same bank. If multiple threads in a warp access the same bank, a _bank conflict_ occurs. This forces the memory accesses to serialize, which negates the speed advantage of shared memory.

在包括 Blackwell 在内的现代 NVIDIA GPU 上，共享内存有 32 个 bank，bank 宽度为 4 字节（即地址按 mod 32 映射）。因此，一个以 128 B（32 个 float × 4 B）为步长的 warp 会把所有线程映射到同一个 bank。如果一个 warp 中的多个线程访问同一个 bank，就会发生 _bank 冲突_（bank conflict）。这会迫使内存访问串行化，从而抵消掉共享内存的速度优势。

In code, bank conflicts often occur when threads access a shared-memory array with a stride that causes them to fall into the same bank. Figure 7-4 shows two examples of conflict-free memory bank accesses.

在代码中，bank 冲突常常发生在线程以某个步长访问共享内存数组、导致它们落入同一个 bank 时。图 7-4 展示了两个无冲突内存 bank 访问的示例。

![Figure 7-4. No bank conflicts](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-4.png)

![图 7-4. 无 bank 冲突](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-4.png)

Here, there are no two threads accessing the same memory bank concurrently. This is ideal. Figure 7-5 shows examples of both a 2-way and 16-way bank conflict.

这里，没有两个线程并发访问同一个内存 bank。这是理想情况。图 7-5 展示了 2 路和 16 路 bank 冲突的示例。

![Figure 7-5. 2-way and 16-way bank conflicts](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-5.png)

![图 7-5. 2 路和 16 路 bank 冲突](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-5.png)

Here, multiple threads are accessing the same memory bank, which will cause conflicts and impact performance. A classic example of a bank conflict is a naive matrix transpose that uses a 32 × 32 shared memory tile. If 32 threads each read tile[i] [threadIdx.x] such that the same column index (threadIdx.x) is read across different rows (i), all 32 threads in the warp will each access memory addresses that lie in the same shared-memory bank, causing a 32-way bank conflict.

这里，多个线程正在访问同一个内存 bank，这会造成冲突并影响性能。bank 冲突的一个经典例子是使用 32 × 32 共享内存瓦片的朴素矩阵转置。如果 32 个线程各自读取 tile[i][threadIdx.x]，使得同一个列索引（threadIdx.x）在不同行（i）之间被读取，那么这个 warp 中的全部 32 个线程各自访问的内存地址都落在同一个共享内存 bank 中，造成 32 路 bank 冲突。

Specifically, during a matrix transpose, you are reading down the same column of a row-major tile by holding the column index constant (threadIdx.x) and varying the row index (i) across threads. And because each row is 128 bytes apart in memory (32 columns × 4 bytes per column = 128 bytes), the accessed memory addresses will differ by exact multiples of 128 bytes.

具体来说，在矩阵转置期间，你是在通过固定列索引（threadIdx.x）、在线程间改变行索引（i）来沿着一个行主序瓦片的同一列向下读取。而由于每一行在内存中相距 128 字节（32 列 × 每列 4 字节 = 128 字节），被访问的内存地址会恰好相差 128 字节的整数倍。

Remember that the address-to-bank mapping repeats every 128 bytes because there are 32 banks with a 4-byte bank width (per access). Therefore, accessing memory addresses offset by 128 bytes will always land back in bank 0—hence, the full 32-way bank conflict.

请记住，由于有 32 个 bank、bank 宽度为 4 字节（每次访问），地址到 bank 的映射每 128 字节重复一次。因此，访问相差 128 字节的内存地址总会落回 bank 0——于是产生了完整的 32 路 bank 冲突。

> There is one exception worth noting: If all 32 threads in a warp access the exact same address in the same memory bank, the hardware will broadcast the value to all threads in a single cycle. This avoids the bank conflict. Any other scenario in which two or more different memory addresses are accessing the same bank will cause a bank conflict and serialize the memory accesses.

> 有一个值得注意的例外：如果一个 warp 中的全部 32 个线程访问同一个内存 bank 中的完全相同的地址，硬件会在单个周期内把该值广播（broadcast）给所有线程。这就避免了 bank 冲突。任何其他情形——即两个或更多不同的内存地址访问同一个 bank——都会造成 bank 冲突并使内存访问串行化。

Another common pitfall is using a stride equal to the number of memory banks, 32 in this case. For instance, if you stride your index by exactly 32 floats, each 4 bytes, then every thread’s address ends up differing by multiples of 128 bytes. In this case, all threads map to bank 0, as shown in this code:

另一个常见陷阱是使用等于内存 bank 数量（这里是 32）的步长。例如，如果你以恰好 32 个 float（每个 4 字节）为步长来访问索引，那么每个线程的地址最终会相差 128 字节的倍数。在这种情况下，所有线程都映射到 bank 0，如以下代码所示：

```
// Allocate a shared buffer big enough for several warps (warpCount)
__shared__ float arr[32 * warpCount];

// Each thread reads from arr[threadIdx.x * 32]
float x = arr[threadIdx.x * 32];
```

```
// Allocate a shared buffer big enough for several warps (warpCount)
__shared__ float arr[32 * warpCount];

// Each thread reads from arr[threadIdx.x * 32]
float x = arr[threadIdx.x * 32];
```

Here, threadIdx.x _ 32 in floats becomes (threadIdx.x _ 32 _ 4) bytes. Because 32 _ 4 = 128, every thread’s memory-load address is threadIdx.x _ 128 bytes. And threadIdx.x _ 128 mod 128 = 0 for all threads. They all hit bank 0 simultaneously, creating a 32-way bank conflict.

这里，以 float 为单位的 threadIdx.x _ 32 变成了 (threadIdx.x _ 32 _ 4) 字节。因为 32 _ 4 = 128，所以每个线程的内存加载地址是 threadIdx.x _ 128 字节。而对所有线程来说，threadIdx.x _ 128 mod 128 = 0。它们全都同时命中 bank 0，造成 32 路 bank 冲突。

When this happens, the hardware must serialize what should have been 32 parallel reads into a sequence of single-bank accesses. In Nsight Compute (Shared Memory section), you will see an increased bank conflict count and lower shared-memory efficiency. At the same time, the shared‐memory throughput is a fraction of its expected bandwidth. In Nsight Systems, you’ll see warps are waiting on long bank-conflict stalls rather than doing useful work.

当这种情况发生时，硬件必须把本应是 32 次并行读取的操作串行化为一系列单 bank 访问。在 Nsight Compute（Shared Memory 部分）中，你会看到 bank 冲突计数增加、共享内存效率降低。与此同时，共享内存吞吐只是其预期带宽的一小部分。在 Nsight Systems 中，你会看到 warp 因长时间的 bank 冲突停顿而等待，而不是在做有用的工作。

Bank conflicts force what should be parallel on-chip shared-memory accesses to replay one by one, wiping out any speedup you expected from buffering and often yielding disappointing, lower-than-anticipated performance. If your kernel isn’t accelerating as it should, bank conflicts are a likely culprit.

bank 冲突迫使本应并行的片上共享内存访问逐个重放，抹掉了你期望从缓冲中获得的任何加速，并常常带来令人失望的、低于预期的性能。如果你的核函数没有像预期那样加速，bank 冲突很可能是罪魁祸首。

> Always choose your stride and data layout so that threads in the same warp hit different banks and avoid that serializing bottleneck.

> 始终选择你的步长和数据布局，让同一个 warp 中的线程命中不同的 bank，从而避免那个串行化瓶颈。

The solution is to adjust data layouts in shared memory to avoid conflicts. A common technique is _padding_ shared arrays so that each row, or each memory-access pattern, maps to different banks. For instance, if you have a 32 × 32 shared tile, you can declare it as [32][33] by adding one extra padding column so that each row now occupies 33 floats. This extra 1-element offset means that when thread k of a warp accesses tile[i][k], successive rows start at addresses that shift across shared-memory banks. This keeps all threads from hitting the same bank, as shown in Figure 7-6.

解决办法是调整共享内存中的数据布局以避免冲突。一种常见技术是对共享数组进行*填充*（padding），使每一行、或每一种内存访问模式映射到不同的 bank。例如，如果你有一个 32 × 32 的共享瓦片，你可以通过增加一个额外的填充列，把它声明为 [32][33]，这样每一行现在占据 33 个 float。这额外的 1 个元素偏移意味着，当一个 warp 的线程 k 访问 tile[i][k] 时，连续的行会从跨越不同共享内存 bank 的地址开始。这就使所有线程不再命中同一个 bank，如图 7-6 所示。

![Figure 7-6. Avoiding bank conflicts with padding](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-6.png)

![图 7-6. 用填充避免 bank 冲突](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-6.png)

By changing the stride to 33, no two threads in a warp will contend for the same bank when accessing a given column. This eliminates what would have been a 32-way bank conflict.

通过把步长改为 33，一个 warp 中没有任何两个线程在访问某一给定列时会争用同一个 bank。这消除了本会发生的 32 路 bank 冲突。

The padding adds a negligible overhead, ~3% more memory for a 32-wide tile, but it completely eliminates the conflicts, which greatly improves performance. And remember that Blackwell has > 200 KB shared memory per SM. A 3% memory overhead is only 1 KB for a 32 × 32 tile. This is worth the performance increase.

这项填充增加的开销可以忽略不计——对于一个 32 宽的瓦片约多用 3% 的内存，但它完全消除了冲突，从而大幅提升性能。而且请记住，Blackwell 每个 SM 有 > 200 KB 的共享内存。对于一个 32 × 32 的瓦片，3% 的内存开销只有 1 KB。这为换取性能提升是值得的。

Let’s show an example of removing shared-memory bank conflicts for a simple transpose kernel. In this example, each thread accesses shared memory addresses that fall into the same shared-memory bank as other threads. This causes the memory access to serialize, prevents parallelism, and shows slow performance. Here is the native implementation of the transpose kernel that incurs bank conflicts.

让我们展示一个针对简单转置核函数消除共享内存 bank 冲突的示例。在这个示例中，每个线程访问的共享内存地址与其他线程落入同一个共享内存 bank。这导致内存访问串行化，阻碍并行，并表现出缓慢的性能。下面是会引发 bank 冲突的转置核函数的原始实现。

_Before example (C++): naive transpose with bank conflicts:_

_前例（C++）：会引发 bank 冲突的朴素转置：_

```
#include <cuda_runtime.h>
#define TILE_DIM 32

__global__ void transposeNaive(const float *idata, float *odata, int width) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // threads in a warp write a row
    tile[threadIdx.y][threadIdx.x] = idata[y * width + x];
    __syncthreads();

    // Read from shared memory with transposed indices
    // This is a classic case of all threads in a warp
    // hitting the same bank causing a bank conflict
    // Read transposed from shared memory and write out
    odata[x * width + y] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    const int N = 1024;
    size_t size = N * N * sizeof(float);
    float *h_idata = (float*)malloc(size);
    float *h_odata = (float*)malloc(size);
    // Initialize input h_idata...
    float *d_idata, *d_odata;
    cudaMalloc(&d_idata, size);
    cudaMalloc(&d_odata, size);
    cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(N / TILE_DIM, N / TILE_DIM);
    transposeNaive<<<grid, block>>>(d_idata, d_odata, N);
    cudaDeviceSynchronize();

    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);
    return 0;
}
```

```
#include <cuda_runtime.h>
#define TILE_DIM 32

__global__ void transposeNaive(const float *idata, float *odata, int width) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // threads in a warp write a row
    tile[threadIdx.y][threadIdx.x] = idata[y * width + x];
    __syncthreads();

    // Read from shared memory with transposed indices
    // This is a classic case of all threads in a warp
    // hitting the same bank causing a bank conflict
    // Read transposed from shared memory and write out
    odata[x * width + y] = tile[threadIdx.x][threadIdx.y];
}

int main() {
    const int N = 1024;
    size_t size = N * N * sizeof(float);
    float *h_idata = (float*)malloc(size);
    float *h_odata = (float*)malloc(size);
    // Initialize input h_idata...
    float *d_idata, *d_odata;
    cudaMalloc(&d_idata, size);
    cudaMalloc(&d_odata, size);
    cudaMemcpy(d_idata, h_idata, size, cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(N / TILE_DIM, N / TILE_DIM);
    transposeNaive<<<grid, block>>>(d_idata, d_odata, N);
    cudaDeviceSynchronize();

    cudaFree(d_idata);
    cudaFree(d_odata);
    free(h_idata);
    free(h_odata);
    return 0;
}
```

In this kernel, the tile write into shared memory is row-major (tile[ty][tx]) and therefore coalesced such that each warp writes a full row of 32 floats to shared memory in a contiguous manner. (Note: Writing the tile to shared memory with column-major tile[tx][ty] would make the warp stride by 128 bytes and trigger a 32-way bank conflict.) In contrast, the tile read is transposed since each thread reads a value where threadIdx.y is the row index and threadIdx.x is the column index.

在这个核函数中，写入共享内存的瓦片写操作是行主序的（tile[ty][tx]），因此是合并的，使得每个 warp 以连续方式把一整行 32 个 float 写入共享内存。（注：以列主序 tile[tx][ty] 写入瓦片到共享内存会使 warp 以 128 字节为步长，并触发 32 路 bank 冲突。）相比之下，瓦片的读操作是转置的，因为每个线程读取的值中 threadIdx.y 是行索引、threadIdx.x 是列索引。

As such, for a given warp, fixed threadIdx.y across threads 0–31, all threads access tile[constant_row][varying_col], meaning the same row index is used across different columns. This means all 32 addresses are in the same row of the tile array, which occupies a single memory bank segment and causes a 32-way bank conflict. As a result, those reads are serialized by a factor of 32.

因此，对于一个给定的 warp，若各线程的 threadIdx.y 固定、跨越线程 0–31，那么所有线程访问的都是 tile[constant_row][varying_col]，也就是在不同的列上使用同一个行索引。这意味着这 32 个地址全部落在 tile 数组的同一行内，而该行占据单个内存 bank 段，从而引发一次 32 路 bank 冲突（bank conflict）。结果，这些读取被序列化，速度降为原来的 1/32。

> Remember that Blackwell’s shared memory has 32 banks. This number exactly matches the number of threads in a warp, 32. So if all threads index into the same bank, as happens when you fix the row and vary the column, you will always get a full 32-way conflict. This one-to-one correspondence means any misaligned access pattern at warp granularity will force every thread to serialize through the same bank—regardless of the GPU’s architectural advancements.

> 请记住，Blackwell 的共享内存有 32 个 bank。这个数字恰好等于一个 warp 中的线程数 32。因此，如果所有线程都索引到同一个 bank——正如固定行、变化列时所发生的那样——你总会得到一次完整的 32 路冲突。这种一一对应关系意味着：在 warp 粒度上任何未对齐的访问模式，都会迫使每个线程串行地通过同一个 bank——无论 GPU 架构如何演进都是如此。

In this case, 32 threads attempt to read from 32 different addresses, all in bank 0. This results in heavy serialization of memory accesses and poor performance. Let’s apply padding to remove the conflicts.

在这种情况下，32 个线程试图从 32 个不同的地址读取，而这些地址全都在 bank 0 中。这导致内存访问被严重序列化、性能低下。下面我们应用填充（padding）来消除这些冲突。

_After example (C++): padded transpose (avoiding bank conflicts)._ Here, we add a small padding, an extra unused column, so that each row of shared memory starts at a different bank modulo. This way, the bank-index collisions are eliminated by the offset:

_后例（CUDA C++）：带填充的转置（避免 bank 冲突）_。这里我们加入一小段填充，即一个额外的、未使用的列，使共享内存的每一行都从不同的 bank 取模值开始。这样，bank 索引冲突就被这个偏移量消除了：

```
#include <cuda_runtime.h>
#define TILE_DIM 32
#define PAD 1  // padding columns to avoid bank conflicts

__global__ void transposePadded(const float *idata, float *odata, int width) {
    // Each row is TILE_DIM+1 elements to shift bank mapping
    __shared__ float tile[TILE_DIM][TILE_DIM + PAD];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    tile[threadIdx.y][threadIdx.x] = idata[y * width + x];
    // We will optimize this later to use a thread-block-scoped
    // cooperative-groups barrier
    __syncthreads();

    odata[x * width + y] = tile[threadIdx.x][threadIdx.y];
}
```

```
#include <cuda_runtime.h>
#define TILE_DIM 32
#define PAD 1  // padding columns to avoid bank conflicts

__global__ void transposePadded(const float *idata, float *odata, int width) {
    // Each row is TILE_DIM+1 elements to shift bank mapping
    __shared__ float tile[TILE_DIM][TILE_DIM + PAD];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    tile[threadIdx.y][threadIdx.x] = idata[y * width + x];
    // We will optimize this later to use a thread-block-scoped
    // cooperative-groups barrier
    __syncthreads();

    odata[x * width + y] = tile[threadIdx.x][threadIdx.y];
}
```

(The host code and setup are the same as in the naive version and omitted for brevity.)

（主机端代码与初始化设置与朴素版本相同，为简洁起见此处省略。）

> PyTorch itself doesn’t expose any high-level APIs for shared-memory padding, so you would have to implement those manually in CUDA kernels and load them with torch.utils.cpp_extension). However, in practice, PyTorch relies on optimized libraries like cuDNN and cuBLAS, which implement techniques under the hood to avoid bank conflicts and maximize throughput.

> PyTorch 本身并不暴露任何用于共享内存填充的高层 API，因此你必须在 CUDA 核函数中手动实现，并用 torch.utils.cpp_extension 加载它们。不过在实践中，PyTorch 依赖 cuDNN 和 cuBLAS 等优化过的库，这些库在底层实现了各种技术来避免 bank 冲突并最大化吞吐。

Padding each shared-memory row with an extra element makes each row 33 floats long. Padding by 1 changes the indexing math so that tile[row][col] addresses differ in the lower 5 bits for each thread, rather than all threads sharing the same 5-bit bank index. This ensures each thread index maps to a different bank.

给每一行共享内存填充一个额外元素，会让每行变成 33 个 float。填充 1 会改变索引运算，使得 tile[row][col] 的地址在每个线程的低 5 位上各不相同，而不再是所有线程共享同一个 5 位 bank 索引。这确保了每个线程索引都映射到不同的 bank。

In the padded kernel, when a warp reads tile[threadIdx.y][threadIdx.x], the addresses for threads 0–31 span multiple banks rather than all hitting bank 0. Now that each thread in the warp accesses distinct banks on reads, bank conflicts are eliminated entirely, as shown in Table 7-5.

在带填充的核函数中，当一个 warp 读取 tile[threadIdx.y][threadIdx.x] 时，线程 0–31 的地址会跨越多个 bank，而不再全部命中 bank 0。既然 warp 中的每个线程在读取时都访问不同的 bank，bank 冲突就被彻底消除了，如表 7-5 所示。

_Table 7-5. Effect of eliminating shared-memory bank conflicts_

_表 7-5. 消除共享内存 bank 冲突的效果_

| Metric                                         | Before (no padding) | After (padding)         |
| ---------------------------------------------- | ------------------- | ----------------------- |
| Shared-memory load bank conflicts              | 4.8 million         | 0                       |
| Shared-memory utilization (percentage of peak) | 52%                 | 100%                    |
| Warp stall fraction (memory)                   | 38%                 | 0.5%                    |
| Kernel execution time (ms)                     | 4 ms                | 1.3 ms (3× improvement) |

| 指标                           | 前（无填充） | 后（有填充）        |
| ------------------------------ | ------------ | ------------------- |
| 共享内存加载 bank 冲突         | 4.8 million  | 0                   |
| 共享内存利用率（占峰值百分比） | 52%          | 100%                |
| warp 停顿占比（内存）          | 38%          | 0.5%                |
| 核函数执行时间（ms）           | 4 ms         | 1.3 ms（提升 3 倍） |

The Nsight Compute metrics in Table 7-5 confirm the impact: shared-memory load bank conflicts dropped to 0. Shared memory throughput, according to Nsight Compute, rose from 52% to 100% after eliminating conflicts. The warp stall fraction, the percentage of cycles warps spend stalled on shared-memory reads, dropped from ~38% to near 0%.

表 7-5 中的 Nsight Compute 指标印证了这一影响：共享内存加载 bank 冲突降到了 0。根据 Nsight Compute，消除冲突后共享内存吞吐从 52% 上升到 100%。warp 停顿占比——即 warp 因等待共享内存读取而停顿的周期百分比——从约 38% 降到接近 0%。

Here, we have successfully eliminated bank conflicts and unlocked the full performance of the on-chip shared memory. This change improved the kernel’s execution time by about 3× from 4 ms to 1.3 ms since shared-memory accesses went from serialized to fully parallel.

至此，我们成功消除了 bank 冲突，释放出片上共享内存的全部性能。由于共享内存访问从序列化变为完全并行，这一改动把核函数的执行时间从 4 ms 缩短到 1.3 ms，提升了约 3 倍。

This restores full parallelism for the shared-memory accesses. The cost of padding, 1 extra float per 32, is trivial (~3% memory overhead) compared to the performance gain.

这恢复了共享内存访问的完全并行性。填充的代价——每 32 个 float 多出 1 个——微不足道（约 3% 的内存开销），相比之下所获得的性能收益相当可观。

An effective alternative to padding is _swizzling_. Swizzling is a compile-time index transformation that “scrambles” the linear index used for shared memory so that sequential threads map to different banks. For example, one can XOR the index with a bit mask or use a modulo offset to achieve a conflict-free pattern.

除了填充之外，还有一种有效的替代方案是*混洗*（swizzling）。混洗是一种编译期的索引变换，它把用于共享内存的线性索引“打乱”，使相邻的线程映射到不同的 bank。例如，可以把索引与一个位掩码做 XOR，或使用一个取模偏移，来实现无冲突的访问模式。

Swizzling avoids the slight memory overhead of padding while still ensuring perfect bank parallelism. Padding is simpler to implement, but swizzling can achieve the same goal with zero memory overhead—and it’s fun to say!

混洗在保证完美 bank 并行的同时，避免了填充带来的那一点点内存开销。填充实现起来更简单，但混洗能以零内存开销达到同样的目标——而且这个词念起来也挺有意思！

> NVIDIA’s CUTLASS library and other high-performance CUDA-based libraries use index swizzling in their tile iterators to ensure threads map to separate banks, avoid bank conflicts, and optimize shared-memory usage.

> NVIDIA 的 CUTLASS 库以及其他高性能的基于 CUDA 的库，会在其瓦片迭代器中使用索引混洗，以确保线程映射到各自独立的 bank、避免 bank 冲突并优化共享内存的使用。

In summary, when using shared memory, it’s important that threads in a warp access different memory banks in parallel rather than queueing up on the same bank. Techniques like padding and swizzling improve shared-memory load/store efficiency, yielding higher throughput and better performance whenever shared memory is used.

总之，在使用共享内存时，重要的是让一个 warp 中的线程并行访问不同的内存 bank，而不是在同一个 bank 上排队。填充和混洗这类技术能提升共享内存加载/存储效率，只要用到共享内存，就能带来更高的吞吐和更好的性能。

Next, let’s explore a technique to avoid shared memory altogether and communicate directly between threads.

接下来，让我们探索一种彻底避免共享内存、在线程之间直接通信的技术。

## Warp Shuffle Intrinsics: Avoid Shared Memory and Explicit Synchronization

## Warp Shuffle 内建函数：避免共享内存与显式同步

The preceding avoiding-bank-conflicts technique assumes that we use shared memory for communication between threads. But what if we could avoid shared memory altogether—and its bank conflict issues?

前面这种避免 bank 冲突的技术，假设我们使用共享内存在线程之间通信。但如果我们能够彻底避开共享内存——以及它带来的 bank 冲突问题——又会怎样呢？

NVIDIA GPUs support warp-synchronous primitives that allow threads in the same warp to exchange data through registers instead of shared memory. In fact, these primitives work only within a single warp such that threads exchange data with their 31 siblings. So no memory banks are involved for this intrawarp communication— and therefore no bank conflicts are possible.

NVIDIA GPU 支持 warp 同步原语，允许同一个 warp 中的线程通过寄存器而不是共享内存来交换数据。事实上，这些原语只在单个 warp 内工作，线程与它的 31 个同胞交换数据。因此这种 warp 内通信不涉及任何内存 bank——也就不可能出现 bank 冲突。

The most common are the **shfl_sync intrinsics (shuffle). **shfl_sync lets you broadcast a value from one thread to all other threads in a warp. You can also perform warp-level reductions without ever writing to shared memory. And remember that these intrinsics let threads exchange values through registers (instead of shared memory), which completely eliminates shared-memory bank conflicts.

其中最常见的是 **shfl_sync 内建函数（shuffle）。**shfl_sync 让你可以把一个线程的值广播（broadcast）给 warp 中的所有其他线程。你还可以在完全不写入共享内存的情况下执行 warp 级归约（reduction）。请记住，这些内建函数让线程通过寄存器（而非共享内存）交换数据，从而彻底消除了共享内存 bank 冲突。

> Modern GPUs will automatically broadcast a single shared-memory value if all threads access the exact same memory address. This will avoid a bank conflict in this special, single-address case. Warp shuffles, on the other hand, use broadcasting to avoid bank issues for the more general, arbitrary, multivalue pattern of data access.

> 如果所有线程访问的都是完全相同的内存地址，现代 GPU 会自动广播这一个共享内存值。在这种特殊的单地址情形下，这可以避免一次 bank 冲突。而 warp shuffle 则是利用广播，来避免更一般的、任意的、多值数据访问模式下的 bank 问题。

Imagine you need to sum the 32 per-thread partial results within a single warp. The naive shared-memory approach would have each thread write its value into a shared array, call a synchronization barrier, and then read and accumulate all 32 entries. This risks bank conflicts and adds additional synchronization overhead.

设想你需要在单个 warp 内对 32 个每线程的部分结果求和。朴素的共享内存做法会让每个线程把自己的值写入一个共享数组，调用一次同步屏障，然后读取并累加全部 32 个条目。这既有 bank 冲突的风险，又增加了额外的同步开销。

With **shfl_sync, you can do a tree-style reduction entirely in registers. For example, let’s use a convenience variant called **shfl_down_sync to perform a butterfly-style reduction such that each thread reads the value held by another thread that is offset a number of lanes away, as shown here:

借助 **shfl_sync，你可以完全在寄存器中完成一次树形归约。例如，我们使用一个便捷变体 **shfl_down_sync 来执行蝶形归约，使每个线程读取另一个偏移若干通道（lane）之外的线程所持有的值，如下所示：

```
unsigned mask = __activemask();
float val = threadVal;  // each thread’s partial sum

// Perform butterfly reduction: exchange values with increasingly distant lanes
for (int offset = 16; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(mask, val, offset);
    val += other;
}

// After the loop, lane 0 holds the warp’s total sum
```

```
unsigned mask = __activemask();
float val = threadVal;  // each thread’s partial sum

// Perform butterfly reduction: exchange values with increasingly distant lanes
for (int offset = 16; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(mask, val, offset);
    val += other;
}

// After the loop, lane 0 holds the warp’s total sum
```

Here, each \_\_shfl_down_sync directly reads another lane’s register, halving the active threads each step, until lane 0 accumulates the full sum. Because all communication stays in registers, there’s no shared-memory traffic and thus zero bank conflicts or extra synchronizations.

在这里，每次 \_\_shfl_down_sync 都直接读取另一个通道的寄存器，每一步都把活跃线程数减半，直到通道 0 累加出完整的总和。由于所有通信都停留在寄存器中，没有共享内存流量，因此零 bank 冲突、也无需额外同步。

In short, \_\_shfl_sync and its variants perform operations entirely within the warp’s execution lanes. This avoids bank conflicts because no shared memory is used. And it’s often faster since it uses registers and cuts down on the number of shared-memory instructions.

简而言之，\_\_shfl_sync 及其变体完全在 warp 的执行通道内完成操作。这避免了 bank 冲突，因为根本不使用共享内存。而且它通常更快，因为它使用寄存器并减少了共享内存指令的数量。

Many high-performance warp-level reductions bypass shared memory entirely by using shuffle intrinsics, which exchange values directly through registers in just a few instructions and incur no bank conflicts. CUDA’s Cooperative Groups API builds on these primitives. This API provides calls like thread_group.shuffle() to simplify intrawarp communication. Helper intrinsics like \_\_reduce_sync() ultimately compile down to these shuffle patterns for intra-warp data exchange on modern NVIDIA GPUs.

许多高性能的 warp 级归约完全绕开共享内存，转而使用 shuffle 内建函数——它们仅用寥寥几条指令就通过寄存器直接交换值，且不产生任何 bank 冲突。CUDA 的 Cooperative Groups API 正是构建在这些原语之上。该 API 提供了 thread_group.shuffle() 这样的调用来简化 warp 内通信。像 \_\_reduce_sync() 这样的辅助内建函数，最终在现代 NVIDIA GPU 上会被编译为这些 shuffle 模式，用于 warp 内的数据交换。

> Shuffles work within a single warp only. For interwarp exchange on modern GPU architectures, also consider thread-block-wide reductions using cooperative groups or thread-block-cluster-level communication when supported. We will discuss these concepts in Chapter 10.

> Shuffle 仅在单个 warp 内工作。在现代 GPU 架构上进行 warp 间交换时，还可以考虑使用 cooperative groups 的线程块级归约，或在受支持时使用线程块簇（thread block cluster）级别的通信。我们将在第 10 章讨论这些概念。

Remember, however, that shuffles are limited to the 32 threads within a single warp. Whenever you need to pass data across warps, you must still fall back on shared or global memory with proper synchronization.

不过请记住，shuffle 局限于单个 warp 内的这 32 个线程。每当你需要在 warp 之间传递数据时，仍然必须退回到共享内存或全局内存，并配合适当的同步。

In Chapter 10, we’ll explore interwarp patterns like cooperative group (CG) synchronization primitives, multiwarp shuffle methods, and thread block clusters (CTA clusters). All of these ultimately use the same hardware intrinsics under the hood. Mastering both intrawarp and interwarp techniques is essential because memory-related bottlenecks remain one of the most common causes of GPU performance issues.

在第 10 章，我们会探讨 warp 间的模式，如 cooperative group（CG）同步原语、多 warp shuffle 方法，以及线程块簇（CTA cluster）。所有这些底层最终都使用同一套硬件内建函数。掌握 warp 内和 warp 间这两类技术都至关重要，因为与内存相关的瓶颈始终是 GPU 性能问题最常见的根源之一。

## Read-Only Data Caches

## 只读数据缓存

When all threads read the same values or a thread rereads data that does not change, failing to use the GPU’s caching mechanisms can bottleneck performance. For example, consider a large lookup table such as an embedding vector in a natural language processing (NLP) model that is read-only during inference. Many threads might need to access this vector in parallel. A naive implementation might fetch from global memory each time, even though the data is immutable and could be cached on-chip.

当所有线程读取相同的值，或某个线程反复读取不会改变的数据时，如果不利用 GPU 的缓存机制，就会成为性能瓶颈。例如，考虑一个大型查找表，比如自然语言处理（natural language processing，NLP）模型中的嵌入向量，它在推理期间是只读的。许多线程可能需要并行访问这个向量。朴素的实现可能每次都从全局内存取数，尽管这些数据是不可变的、本可以缓存在片上。

Note that the read-only cache we refer to here is different from the 64 KB constant memory cache discussed previously in the GPU memory hierarchy section. It’s a large cache for immutable data. The constant memory cache, on the other hand, is too small for big arrays. Modern architectures rely on the larger L1/read-only cache for these types of embedding-vector lookups—rather than trying to squeeze this data into the small 64 KB constant memory cache.

请注意，我们这里所说的只读缓存，不同于前面在 GPU 内存层级一节讨论过的 64 KB 常量内存缓存。它是一个面向不可变数据的大容量缓存。而常量内存缓存对于大数组来说太小了。现代架构依赖更大的 L1/只读缓存来处理这类嵌入向量查找——而不是试图把这些数据硬塞进小小的 64 KB 常量内存缓存里。

On modern GPUs, global memory loads are automatically cached in L2 and often L1. You can use const **restrict** qualified pointers to define your function arguments as non-coherent and read-mostly (versus read-only). For read-only data, the compiler may route loads through this read-only path when it can prove immutability, nonaliasing, and safety. This lets the compiler/hardware route unchanging data through the read-only L1 cache, which has lower latency—especially for broadcast accesses—and doesn’t evict other cached data.

在现代 GPU 上，全局内存加载会被自动缓存到 L2、且通常也会缓存到 L1。你可以使用 const **restrict** 限定的指针，把函数参数声明为非一致性的（non-coherent）、以读为主（read-mostly，相对于只读而言）。对于只读数据，当编译器能够证明其不可变性、无别名性和安全性时，它可以把加载路由到这条只读路径上。这让编译器/硬件能够把不变的数据经由只读 L1 缓存传送——该缓存延迟更低，尤其是对广播访问而言——而且不会驱逐其他已缓存的数据。

> In modern CUDA, you usually don’t need to call **ldg() explicitly. If a pointer is const **restrict**, the compiler may use the read-only data cache for global memory loads when it can provide safety. The older **ldg() intrinsic is still available for explicit control, but it’s generally not needed with modern compilers.

> 在现代 CUDA 中，你通常不需要显式调用 **ldg()。如果一个指针是 const **restrict**，当编译器能够保证安全时，它可能会为全局内存加载使用只读数据缓存。较老的 **ldg() 内建函数仍然可用于显式控制，但在现代编译器下一般并不需要。

A common performance pitfall is forgetting to tell the compiler that a buffer is truly read-only, which means it won’t use the read-only (non-coherent) data path and won’t route those loads through the specialized read-only cache. Instead, every access becomes a plain global memory load, resulting in redundant DRAM traffic and spurious cache misses.

一个常见的性能陷阱是忘记告诉编译器某个缓冲区确实是只读的，这意味着它不会使用只读（非一致性）数据路径，也不会把那些加载经由专门的只读缓存来路由。相反，每一次访问都变成一次普通的全局内存加载，导致冗余的 DRAM 流量和不必要的缓存未命中。

When you profile such a kernel, you’ll spot the same addresses fetched repeatedly from off-chip memory, see no \_\_ldg() operations in the instruction stream, observe a surprisingly low L2 hit rate for that array, and measure elevated DRAM throughput reminiscent of an uncached workload.

当你剖析这样一个核函数时，你会发现相同的地址被反复从片外内存取回、在指令流中看不到任何 \_\_ldg() 操作、观察到该数组出人意料地低的 L2 命中率，并测得偏高的 DRAM 吞吐——就像一个未缓存的工作负载。

The solution is to leverage the read-only path by marking data as const **restrict**, or by explicitly using the \_\_ldg() intrinsic to load it. This tells the hardware that the data will not be modified, allowing it to be cached in the specialized read-only cache, which sits alongside L1 and has lower latency for broadcast loads.

解决办法是通过把数据标记为 const **restrict**，或显式使用 \_\_ldg() 内建函数来加载它，从而利用只读路径。这告诉硬件该数据不会被修改，允许它被缓存到专门的只读缓存中——该缓存紧邻 L1，对广播加载具有更低的延迟。

When a warp issues a constant‐cache load (**constant** or uniform \_\_ldg() in older GPUs), the hardware can service all 32 lanes with a single transaction if they hit the same address. By broadcasting that value to every thread, it uses only one cycle instead of doing 32 separate loads. This warp-wide broadcast reduces both latency and memory bandwidth usage for uniform data like lookup tables, coefficients, etc. This lets you fetch a shared constant, for free essentially, once per warp rather than 32 times per warp (1 per thread).

当一个 warp 发起一次常量缓存加载（在较老的 GPU 上为 **constant** 或统一的 \_\_ldg()）时，如果这些访问命中同一地址，硬件可以用一次事务服务全部 32 个通道。通过把该值广播给每个线程，它只用一个周期，而不必执行 32 次单独的加载。对于查找表、系数等统一数据，这种 warp 级（warp-wide）广播同时降低了延迟和内存带宽占用。这让你几乎可以“免费”地每个 warp 取一次共享常量，而不是每个 warp 取 32 次（每线程 1 次）。

As an example of caching benefits using the standard read-only cache, suppose we have a kernel that, for each thread, looks up a value from a table of size T = 1,024 and writes it to an output. This simulates an embedding-lookup pattern and is shown here in both CUDA C++ and PyTorch:

作为使用标准只读缓存所带来的缓存收益的一个例子，假设我们有一个核函数，它为每个线程从一个大小为 T = 1,024 的表中查找一个值并写入输出。这模拟了一种嵌入查找模式，下面同时用 CUDA C++ 和 PyTorch 给出：

```
#include <cuda_runtime.h>
#define T 1024

__global__ void naiveLookup(const float* table, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // __ldg not used here, each access goes to
        // global memory without using read-only cache
        int t = idx % T;
        out[idx] = table[t];
    }
}

int main() {
    const int N = 1 << 20;

    float* h_table = nullptr;
    float* h_out = nullptr;
    cudaMallocHost(&h_table, T * sizeof(float));
    cudaMallocHost(&h_out, N * sizeof(float));

    for (int i = 0; i < T; ++i) h_table[i] = float(i);

    float *d_table, *d_out;
    cudaMalloc(&d_table, T * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_table, h_table, T * sizeof(float),
        cudaMemcpyHostToDevice);

    dim3 block(256), grid((N + 255) / 256);
    naiveLookup<<<grid, block>>>(d_table, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_table);
    cudaFree(d_out);
    cudaFreeHost(h_table);
    cudaFreeHost(h_out);

    return 0;
}
```

```
#include <cuda_runtime.h>
#define T 1024

__global__ void naiveLookup(const float* table, float* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // __ldg not used here, each access goes to
        // global memory without using read-only cache
        int t = idx % T;
        out[idx] = table[t];
    }
}

int main() {
    const int N = 1 << 20;

    float* h_table = nullptr;
    float* h_out = nullptr;
    cudaMallocHost(&h_table, T * sizeof(float));
    cudaMallocHost(&h_out, N * sizeof(float));

    for (int i = 0; i < T; ++i) h_table[i] = float(i);

    float *d_table, *d_out;
    cudaMalloc(&d_table, T * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_table, h_table, T * sizeof(float),
        cudaMemcpyHostToDevice);

    dim3 block(256), grid((N + 255) / 256);
    naiveLookup<<<grid, block>>>(d_table, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_table);
    cudaFree(d_out);
    cudaFreeHost(h_table);
    cudaFreeHost(h_out);

    return 0;
}
```

Here is a rough PyTorch equivalent to the naive version shown in CUDA C++:

下面是与上面 CUDA C++ 朴素版本大致对应的 PyTorch 实现：

```
import torch

def vectorized_lookup(table, N):
    flat = table.view(-1)
    T = flat.size(0)

    # build indices [0,1,2,...,N-1] % T all on GPU
    idx = torch.arange(N, device=flat.device) % T

    # one gather kernel does all N loads in parallel
    return flat.index_select(0, idx)

# Usage
T = 1024
N = 1 << 20
table = torch.arange(T, dtype=torch.float32,
                     device='cuda')
out = vectorized_lookup(table, N)
```

```
import torch

def vectorized_lookup(table, N):
    flat = table.view(-1)
    T = flat.size(0)

    # build indices [0,1,2,...,N-1] % T all on GPU
    idx = torch.arange(N, device=flat.device) % T

    # one gather kernel does all N loads in parallel
    return flat.index_select(0, idx)

# Usage
T = 1024
N = 1 << 20
table = torch.arange(T, dtype=torch.float32,
                     device='cuda')
out = vectorized_lookup(table, N)
```

In these naive versions, each lookup likely hits in L2 after the first use (since L2 will cache it), but there is no use of the specialized read-only cache. The hardware may still treat it as normal global data, which could evict other useful data or not take full advantage of broadcast caching if multiple threads read the same table[t] in a warp.

在这些朴素版本中，每次查找在首次使用之后很可能会命中 L2（因为 L2 会缓存它），但完全没有用到专门的只读缓存。硬件仍可能把它当作普通的全局数据来对待，这可能驱逐其他有用的数据，或者在一个 warp 中多个线程读取同一个 table[t] 时无法充分利用广播缓存。

Now we optimize the kernel by marking the table as const **restrict**. This will hint to the hardware that it should use the read-only cache path, which is shown here:

现在我们通过把 table 标记为 const **restrict** 来优化这个核函数。这会提示硬件应当使用只读缓存路径，如下所示：

```
#include <cuda_runtime.h>
#define T 1024

__global__ void lookup(const float* __restrict__ table,
        float* out, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int t = idx % T;
        // Compiler can turn this into a load from the
        // read-only cache for faster loads
        out[idx] = table[t];
    }
}

int main() {
    const int N = 1 << 20;

    float* h_table = nullptr;
    float* h_out = nullptr;
    cudaMallocHost(&h_table, T * sizeof(float));
    cudaMallocHost(&h_out, N * sizeof(float));

    for (int i = 0; i < T; ++i) h_table[i] = float(i);

    float *d_table, *d_out;
    cudaMalloc(&d_table, T * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_table, h_table, T * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256), grid((N + 255) / 256);
    lookup<<<grid, block>>>(d_table, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_table);
    cudaFree(d_out);
    cudaFreeHost(h_table);
    cudaFreeHost(h_out);

    return 0;
}
```

```
#include <cuda_runtime.h>
#define T 1024

__global__ void lookup(const float* __restrict__ table,
        float* out, int N) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        int t = idx % T;
        // Compiler can turn this into a load from the
        // read-only cache for faster loads
        out[idx] = table[t];
    }
}

int main() {
    const int N = 1 << 20;

    float* h_table = nullptr;
    float* h_out = nullptr;
    cudaMallocHost(&h_table, T * sizeof(float));
    cudaMallocHost(&h_out, N * sizeof(float));

    for (int i = 0; i < T; ++i) h_table[i] = float(i);

    float *d_table, *d_out;
    cudaMalloc(&d_table, T * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_table, h_table, T * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256), grid((N + 255) / 256);
    lookup<<<grid, block>>>(d_table, d_out, N);
    cudaDeviceSynchronize();
    cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_table);
    cudaFree(d_out);
    cudaFreeHost(h_table);
    cudaFreeHost(h_out);

    return 0;
}
```

> In a PyTorch setting, you could implement the same trick by writing a small CUDA extension with torch.utils.cpp_extension that declares your embedding table pointer as const **restrict**. You could also use the PyTorch compiler to optimize this code. We’ll cover the PyTorch compiler in depth in Chapters 13 and 14.

> 在 PyTorch 环境下，你可以通过用 torch.utils.cpp_extension 编写一个小的 CUDA 扩展来实现同样的技巧，在其中把你的嵌入表指针声明为 const **restrict**。你也可以使用 PyTorch 编译器来优化这段代码。我们将在第 13 章和第 14 章深入介绍 PyTorch 编译器。

In short, marking the table const **restrict** tells the compiler that these values are immutable and not aliased, which permits the use of the read-only data path when safe to do so. This increases cache hit rate and reduces off-chip traffic. Table 7-6 shows Nsight Compute measurements before and after this change.

简而言之，把 table 标记为 const **restrict** 是在告诉编译器这些值是不可变的、且不存在别名，这就允许在安全的情况下使用只读数据路径。这提高了缓存命中率并减少了片外流量。表 7-6 展示了此改动前后的 Nsight Compute 测量结果。

_Table 7-6. Benefit of read-only cache_

_表 7-6. 只读缓存的收益_

| Metric                        | Before (no**restrict**) | After (**restrict** used) |
| ----------------------------- | ----------------------- | ------------------------- |
| Global Memory Load Efficiency | 52%                     | 97%                       |
| DRAM throughput (GB/s)        | 600 GB/s                | 200 GB/s                  |
| L2 read throughput (GB/s)     | 1,500 GB/s              | 1,800 GB/s                |
| SM Active %                   | 45%                     | 93%                       |
| Kernel execution time (ms)    | 2.5 ms                  | 1.0 ms                    |

| 指标                                              | 前（无**restrict**） | 后（使用**restrict**） |
| ------------------------------------------------- | -------------------- | ---------------------- |
| 全局内存加载效率（Global Memory Load Efficiency） | 52%                  | 97%                    |
| DRAM 吞吐（GB/s）                                 | 600 GB/s             | 200 GB/s               |
| L2 读吞吐（GB/s）                                 | 1,500 GB/s           | 1,800 GB/s             |
| SM 活跃百分比（SM Active %）                      | 45%                  | 93%                    |
| 核函数执行时间（ms）                              | 2.5 ms               | 1.0 ms                 |

After adding const **restrict**, the kernel time improves by about 2.5×, from 2.5 milliseconds to 1.0 millisecond, because warps spend less time stalled on DRAM. SM Active % rises from 45% to 93%, indicating that nearly every cycle has active work rather than idle memory waits.

加入 const **restrict** 之后，核函数时间提升了约 2.5 倍，从 2.5 毫秒降到 1.0 毫秒，因为 warp 因等待 DRAM 而停顿的时间更少了。SM 活跃百分比从 45% 上升到 93%，表明几乎每个周期都有活跃的工作，而不是空闲地等待内存。

DRAM throughput falls from 600 GB/s to 200 GB/s, while L2 read throughput increases from 1,500 GB/s to 1,800 GB/s as more requests are satisfied in cache. Global Memory Load Efficiency increases from 52% to 97%, confirming that most fetched cache lines carry useful data.

DRAM 吞吐从 600 GB/s 降到 200 GB/s，而 L2 读吞吐从 1,500 GB/s 提升到 1,800 GB/s，因为更多请求在缓存中得到了满足。全局内存加载效率从 52% 提升到 97%，证实了大多数取回的缓存行携带的都是有用数据。

> Nsight Systems presents a timeline view of overall GPU activity, while Nsight Compute reports per-kernel metrics such as SM Active %. Use Nsight Compute when you need quantitative per-kernel analysis.

> Nsight Systems 呈现的是整体 GPU 活动的时间线视图，而 Nsight Compute 报告的是逐核函数的指标，如 SM 活跃百分比。当你需要定量的逐核函数分析时，请使用 Nsight Compute。

Because more traffic is served by on-chip caches rather than traveling to DRAM, the compute units remain fed and arithmetic intensity increases. This balance is what moves a kernel from the memory-bound regime toward the compute-bound regime. Texture objects and surface objects are also available for read-only and read/write access patterns with strong two-dimensional or three-dimensional locality. By binding an array to a cudaTextureObject_t and using tex1Dfetch or tex2D fetches, the hardware can exploit spatial locality with high cache hit rates and features such as wrapping and interpolation. Surface objects allow writes on similar access patterns.

由于更多流量由片上缓存服务、而不必远赴 DRAM，计算单元得以持续获得数据供给，算术强度（arithmetic intensity）随之提高。正是这种平衡把一个核函数从访存受限（memory-bound）区间推向计算受限（compute-bound）区间。对于具有很强二维或三维局部性的只读及读/写访问模式，还可以使用纹理对象（texture object）和表面对象（surface object）。通过把一个数组绑定到 cudaTextureObject_t 并使用 tex1Dfetch 或 tex2D 取数，硬件可以以很高的缓存命中率利用空间局部性，并支持环绕（wrapping）和插值等特性。表面对象则允许在类似的访问模式上进行写入。

> While texture reference and surface reference APIs are deprecated, texture object and surface object APIs remain supported and are appropriate for access patterns with two- or three-dimensional locality. However, for most AI workloads involving 1D data, using the read-only data cache with constant memory is much simpler and preferred.

> 尽管纹理引用与表面引用 API 已被弃用，纹理对象与表面对象 API 仍受支持，适用于具有二维或三维局部性的访问模式。不过，对于大多数涉及一维数据的 AI 工作负载，使用只读数据缓存配合常量内存要简单得多，也是更受青睐的做法。

In summary, mark read-only data as const **restrict** to tap into the low-latency read-only cache, cutting DRAM traffic and lifting SM activity. Consider texture or surface memory whenever your access pattern has 2D/3D locality that a regular cache might not handle optimally. Together, these techniques collapse memory stalls, boost cache utilization, and unlock substantial performance gains for memory-bound kernels.

总之，把只读数据标记为 const **restrict**，以接入低延迟的只读缓存，削减 DRAM 流量并提升 SM 活跃度。每当你的访问模式具有普通缓存可能无法最优处理的 2D/3D 局部性时，就考虑使用纹理或表面内存。这些技术合在一起，能够坍缩内存停顿、提升缓存利用率，并为访存受限的核函数释放出可观的性能收益。

## Asynchronous Memory Prefetching and Tensor Memory Accelerator

## 异步内存预取与 Tensor Memory Accelerator

In earlier sections, we saw how coalescing dozens of 4-byte loads into a single 128-byte transaction significantly improved global-load efficiency and cut wasted sectors per request. Yet even a perfectly coalesced load still stalls a warp for the full DRAM round trip.

在前面的小节中，我们看到把几十次 4 字节加载合并成一次 128 字节事务，如何显著提升了全局加载效率并削减了每次请求中被浪费的扇区。然而，即便是完美合并的加载，仍会让一个 warp 停顿整整一趟 DRAM 往返的时间。

On Blackwell, for instance, a full DRAM round-trip is on the order of hundreds of cycles before any computation can begin. To hide that latency, we need to overlap data transfer with compute. This overlap is what hides most of the DRAM latency.

以 Blackwell 为例，在任何计算开始之前，一趟完整的 DRAM 往返都在数百个周期的量级。为了隐藏这段延迟，我们需要让数据传输与计算重叠。正是这种重叠隐藏了大部分 DRAM 延迟。

CUDA’s Pipeline API together with the Tensor Memory Accelerator (TMA) hardware engine take this idea to the thread-block level. Instead of having each warp use the SM’s load and store (LD/ST) units to fetch data from global memory, you can invoke the TMA engine to asynchronously fetch an entire tile from global memory into shared memory, as shown in Figure 7-7.

CUDA 的 Pipeline API 连同 Tensor Memory Accelerator（TMA）硬件引擎，把这一思路带到了线程块层面。你不必让每个 warp 使用 SM 的加载与存储（LD/ST）单元去从全局内存取数，而是可以调用 TMA 引擎，把一整个瓦片从全局内存异步取入共享内存，如图 7-7 所示。

![Figure 7-7. TMA asynchronously fetching data from global HBM into shared memory](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-7.png)

![图 7-7. TMA 从全局 HBM 异步取数据到共享内存](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-7.png)

To start the TMA transfer, you can call cuda::memcpy_async(). On modern GPU architectures, cuda::memcpy_async() together with cuda::pipeline exposes the hardware engines for asynchronous global to shared transfers. This includes the TMA when available. This will use TMA’s on-chip DMA engine to perform the asynchronous bulk data transfer. This is implemented in CUDA as follows:

要启动 TMA 传输，你可以调用 cuda::memcpy_async()。在现代 GPU 架构上，cuda::memcpy_async() 连同 cuda::pipeline 一起，暴露了用于全局到共享异步传输的硬件引擎。这在可用时包含 TMA。它会使用 TMA 的片上 DMA 引擎来执行异步的批量数据传输。在 CUDA 中的实现如下：

```
cuda::memcpy_async(sharedBuf,
               globalPtr + offset,
               cuda::aligned_size_t<16>(bytes),
               pipe);
pipe.producer_commit();
```

```
cuda::memcpy_async(sharedBuf,
               globalPtr + offset,
               cuda::aligned_size_t<16>(bytes),
               pipe);
pipe.producer_commit();
```

And while TMA handles the bulk copy, including coalescing, strided transfers, and even multidimensional transfers, the kernel computes the previous tile. This is called _double buffering_, or _ping-ponging_.

而当 TMA 处理批量拷贝时——包括合并、跨步传输，乃至多维传输——核函数则在计算上一个瓦片。这被称为*双缓冲*（double buffering），或*乒乓*（ping-ponging）。

By implementing double buffering with TMA, the SM’s load/store units are now free to do real work because TMA’s DMA engine moves data for us in the background. In effect, data movement becomes asynchronous such that while TMA streams in the next tile of data, the SM’s warps compute on the previous tile. This overlap is what hides the 800-cycle DRAM latency.

通过用 TMA 实现双缓冲，SM 的加载/存储单元现在可以腾出来做真正的工作，因为 TMA 的 DMA 引擎在后台替我们搬运数据。实际效果是，数据搬运变成了异步的——当 TMA 把下一个瓦片的数据流式送入时，SM 的 warp 正在计算上一个瓦片。正是这种重叠隐藏了那 800 个周期的 DRAM 延迟。

Specifically, TMA is capable of 1D–5D bulk copies and arbitrary strides between global and shared memory without blocking the SM instruction pipeline. By offloading these transfers from the SM to TMA, your kernel issues far fewer LD/ST instructions, eliminates extra synchronization, and lets the warp schedulers spend almost every cycle on useful computation instead of waiting on memory.

具体来说，TMA 能够在全局内存与共享内存之间进行 1D–5D 的批量拷贝以及任意跨步，而不阻塞 SM 的指令流水线。通过把这些传输从 SM 卸载到 TMA，你的核函数发出的 LD/ST 指令大为减少，省去了额外的同步，并让 warp 调度器几乎把每个周期都花在有用的计算上，而不是等待内存。

Here is a code snippet showing how to initiate an asynchronous copy from global to shared memory using the CUDA C++ Pipeline API and the TMA hardware engine:

下面这段代码展示了如何使用 CUDA C++ Pipeline API 和 TMA 硬件引擎发起一次从全局到共享内存的异步拷贝：

```
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>

#define TILE_SIZE 1024  // example tile size
// User-provided compute function operating on a shared-memory tile
__device__ void processTile(const float* tile);

__global__ void kernelWithTMA(const float* __restrict__ global_ptr,
                              int nTiles) {
    // Two ping-pong buffers in shared memory
    // On Blackwell (CUDA 13+), prefer 32B alignment for 256-bit vectors
    // (v8.f32 / double4).
    __shared__ __align__(VEC_BYTES) float tile0[TILE_SIZE];
    __shared__ __align__(VEC_BYTES) float tile1[TILE_SIZE];

float* tiles[2] = { tile0, tile1 };

    // Alignment / size guards for vectorized copies
    // --- choose vector width -------------------------------------------------
    #ifndef VEC_BYTES
      // Prefer 32B vectors on CUDA 13+/Blackwell.
      // Define -DUSE_256B_VEC to force.
      #if defined(USE_256B_VEC) || (defined(CUDA_VERSION)&&CUDA_VERSION >= 13000)
        #define VEC_BYTES 32      // 256-bit: v8.f32, double4

      #else
        #define VEC_BYTES 16      // 128-bit: v4.f32, float4
      #endif
    #endif
    constexpr int VEC_ELEMS = VEC_BYTES / sizeof(float); // 4 (128b) or 8 (256b)
    constexpr int WARP      = 32;

    // Alignment / size guards for vectorized copies/compute
    static_assert((TILE_SIZE % (WARP * VEC_ELEMS)) == 0,
                  "TILE_SIZE must be a multiple of WARP_SIZE * VEC_ELEMS");
    // Also guarantee the tile byte count
    //  is a multiple of the vector width
    static_assert(((TILE_SIZE * sizeof(float)) % VEC_BYTES) == 0,
                  "Tile byte size must be a multiple of VEC_BYTES");

    // On Blackwell (CUDA 13+), prefer 32B alignment
    // for 256-bit vectors (v8.f32 / double4).

    assert((reinterpret_cast<std::uintptr_t>(global_ptr) % VEC_BYTES) == 0);

    size_t bytes = TILE_SIZE * sizeof(float);
    // Block-scoped pipeline for TMA
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope_block, 2> state;

    auto pipe =
        cuda::make_pipeline(cuda::this_thread_block(),
            &state);

    // Prime pipeline with the first async copy into tile0
    pipe.producer_acquire();
    cuda::memcpy_async(tiles[0],
                       global_ptr + 0 * TILE_SIZE,
                       cuda::aligned_size_t<32>{bytes},
                       pipe);
    pipe.producer_commit();

    // Loop over the remaining tiles
    for (int t = 1; t < nTiles; ++t) {
        // Wait for the previous copy to finish, then compute on it
        pipe.consumer_wait();
        processTile(tiles[(t - 1) & 1]);
        pipe.consumer_release();

        // Enqueue the next async copy into the alternate buffer
        pipe.producer_acquire();
        cuda::memcpy_async(tiles[t & 1],
                           global_ptr + t * TILE_SIZE,
                           cuda::aligned_size_t<32>{bytes},,
                           pipe);
        pipe.producer_commit();
    }

    // Final wait and compute on the last tile
    pipe.consumer_wait();
    processTile(tiles[(nTiles - 1) & 1]);
    pipe.consumer_release();
}
```

```
#include <cuda/pipeline>
#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>

#define TILE_SIZE 1024  // example tile size
// User-provided compute function operating on a shared-memory tile
__device__ void processTile(const float* tile);

__global__ void kernelWithTMA(const float* __restrict__ global_ptr,
                              int nTiles) {
    // Two ping-pong buffers in shared memory
    // On Blackwell (CUDA 13+), prefer 32B alignment for 256-bit vectors
    // (v8.f32 / double4).
    __shared__ __align__(VEC_BYTES) float tile0[TILE_SIZE];
    __shared__ __align__(VEC_BYTES) float tile1[TILE_SIZE];

float* tiles[2] = { tile0, tile1 };

    // Alignment / size guards for vectorized copies
    // --- choose vector width -------------------------------------------------
    #ifndef VEC_BYTES
      // Prefer 32B vectors on CUDA 13+/Blackwell.
      // Define -DUSE_256B_VEC to force.
      #if defined(USE_256B_VEC) || (defined(CUDA_VERSION)&&CUDA_VERSION >= 13000)
        #define VEC_BYTES 32      // 256-bit: v8.f32, double4

      #else
        #define VEC_BYTES 16      // 128-bit: v4.f32, float4
      #endif
    #endif
    constexpr int VEC_ELEMS = VEC_BYTES / sizeof(float); // 4 (128b) or 8 (256b)
    constexpr int WARP      = 32;

    // Alignment / size guards for vectorized copies/compute
    static_assert((TILE_SIZE % (WARP * VEC_ELEMS)) == 0,
                  "TILE_SIZE must be a multiple of WARP_SIZE * VEC_ELEMS");
    // Also guarantee the tile byte count
    //  is a multiple of the vector width
    static_assert(((TILE_SIZE * sizeof(float)) % VEC_BYTES) == 0,
                  "Tile byte size must be a multiple of VEC_BYTES");

    // On Blackwell (CUDA 13+), prefer 32B alignment
    // for 256-bit vectors (v8.f32 / double4).

    assert((reinterpret_cast<std::uintptr_t>(global_ptr) % VEC_BYTES) == 0);

    size_t bytes = TILE_SIZE * sizeof(float);
    // Block-scoped pipeline for TMA
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope_block, 2> state;

    auto pipe =
        cuda::make_pipeline(cuda::this_thread_block(),
            &state);

    // Prime pipeline with the first async copy into tile0
    pipe.producer_acquire();
    cuda::memcpy_async(tiles[0],
                       global_ptr + 0 * TILE_SIZE,
                       cuda::aligned_size_t<32>{bytes},
                       pipe);
    pipe.producer_commit();

    // Loop over the remaining tiles
    for (int t = 1; t < nTiles; ++t) {
        // Wait for the previous copy to finish, then compute on it
        pipe.consumer_wait();
        processTile(tiles[(t - 1) & 1]);
        pipe.consumer_release();

        // Enqueue the next async copy into the alternate buffer
        pipe.producer_acquire();
        cuda::memcpy_async(tiles[t & 1],
                           global_ptr + t * TILE_SIZE,
                           cuda::aligned_size_t<32>{bytes},,
                           pipe);
        pipe.producer_commit();
    }

    // Final wait and compute on the last tile
    pipe.consumer_wait();
    processTile(tiles[(nTiles - 1) & 1]);
    pipe.consumer_release();
}
```

Immediately after kernel launch, each block allocates two shared-memory tiles (tile0, tile1) and constructs a block-scoped pipeline so all threads in the thread block coordinate their asynchronous DMA, as shown here:

核函数启动后，每个线程块立即分配两个共享内存瓦片（tile0、tile1），并构造一个块级作用域的流水线，使线程块中的所有线程协同其异步 DMA，如下所示：

```
__shared__ cuda::pipeline_shared_state<
    cuda::thread_scope_block, 2> state;

auto pipe =
    cuda::make_pipeline(cuda::this_thread_block(),
        &state);
```

```
__shared__ cuda::pipeline_shared_state<
    cuda::thread_scope_block, 2> state;

auto pipe =
    cuda::make_pipeline(cuda::this_thread_block(),
        &state);
```

To prime the pipeline, we submit an asynchronous copy using the TMA, which coalesces even strided or multidimensional transfers and streams bytes from global memory into tiles[0] in the background, as seen here:

为了给流水线预热，我们使用 TMA 提交一次异步拷贝，它甚至能合并跨步或多维传输，并在后台把字节从全局内存流式送入 tiles[0]，如下所示：

```
pipe.producer_acquire();
cuda::memcpy_async(tiles[0],
                   global_ptr + 0 * TILE_SIZE,
                   cuda::aligned_size_t<32>{bytes},
                   pipe);
pipe.producer_commit();
```

```
pipe.producer_acquire();
cuda::memcpy_async(tiles[0],
                   global_ptr + 0 * TILE_SIZE,
                   cuda::aligned_size_t<32>{bytes},
                   pipe);
pipe.producer_commit();
```

Before using that data, we do the following to ensure we block just long enough for the TMA to finish the copy:

在使用这些数据之前，我们执行下面的操作，以确保恰好阻塞足够长的时间、等 TMA 完成拷贝：

```
pipe.consumer_wait();
processTile(tiles[0]);
pipe.consumer_release();
```

```
pipe.consumer_wait();
processTile(tiles[0]);
pipe.consumer_release();
```

Inside the main loop, we alternate buffers using pipe.consumer_wait(); + processTile() on the previous tile, pipe.consumer_release(), pipe.producer_acquire() + new cuda::memcpy_async() into the other tile, and pipe.producer_commit(). Then we repeat!

在主循环内部，我们通过以下步骤交替缓冲区：对上一个瓦片执行 pipe.consumer_wait(); + processTile()，然后 pipe.consumer_release()、pipe.producer_acquire() + 一次新的 cuda::memcpy_async() 写入另一个瓦片，以及 pipe.producer_commit()。然后循环往复！

By ping-ponging between tile0 and tile1, each new memcpy_async overlaps processTile on the previous buffer. The load and store units on the SM experience lower pressure with more instruction issue slots available for computation. At the same time, the TMA moves data in parallel. This eliminates redundant global memory loads, reduces synchronization overhead, and keeps warps busy rather than stalled on memory.

通过在 tile0 与 tile1 之间乒乓（ping-pong），每一次新的 memcpy_async 都与对上一个缓冲区的 processTile 重叠。SM 上的加载和存储单元压力更低，从而有更多指令发射槽可用于计算。与此同时，TMA 并行地搬运数据。这消除了冗余的全局内存加载、减少了同步开销，并让 warp 保持忙碌，而不是停顿在内存上。

Asynchronous prefetching from global memory to shared memory hides latency behind compute. Threads preload upcoming data into shared memory while they compute on data that was previously loaded.

从全局内存到共享内存的异步预取，把延迟隐藏在计算之后。线程在对先前加载的数据进行计算的同时，把即将用到的数据预加载到共享内存中。

This pattern is especially effective in memory-bound loops and tensor computations. On modern GPUs, the TMA can stream the next tile for a matrix multiply while the current tile is being processed.

这种模式在访存受限的循环和张量计算中尤其有效。在现代 GPU 上，TMA 可以在当前瓦片正被处理时，为矩阵乘法流式送入下一个瓦片。

> TMA is the preferred path for tiled bulk copies when moving 2D and N-dimensional tiles between global memory and SMEM/TMEM. Prefer cuda::memcpy_async with cuda::pipeline at block scope; on Hopper/Blackwell the implementation will leverage TMA (cp.async.bulk.\* family) when alignment and direction permit (e.g., global memory ↔ shared memory).

> 在全局内存与 SMEM/TMEM 之间搬运 2D 和 N 维瓦片时，TMA 是分块批量拷贝的首选路径。优先在块作用域使用 cuda::memcpy_async 配合 cuda::pipeline；在 Hopper/Blackwell 上，只要对齐和方向允许（例如全局内存 ↔ 共享内存），其实现就会利用 TMA（cp.async.bulk.\* 系列）。

We trade many scattered global reads for one coalesced global copy plus many fast shared-memory loads. This is favorable given the gap between DRAM and on-chip SRAM. Table 7-7 summarizes Nsight Compute metrics before and after a TMA-based double-buffering implementation.

我们用一次合并的全局拷贝加上许多次快速的共享内存加载，换掉了许多次分散的全局读取。鉴于 DRAM 与片上 SRAM 之间的差距，这样做是划算的。表 7-7 汇总了基于 TMA 的双缓冲实现前后的 Nsight Compute 指标。

_Table 7-7. Comparing naive kernel (no prefetch) to TMA-accelerated double buffering_

_表 7-7. 朴素核函数（无预取）与 TMA 加速的双缓冲的对比_

| Metric                        | Before (no prefetch) | After (async prefetch) |
| ----------------------------- | -------------------- | ---------------------- |
| Global Memory Load Efficiency | 23%                  | 99%                    |
| Average sectors per request   | 6.4                  | 4.0                    |
| DRAM throughput (% of peak)   | 25%                  | 90%                    |
| SM Active %                   | 62%                  | 100%                   |
| Kernel execution time         | 18 ms                | 7 ms                   |

| 指标                                              | 前（无预取） | 后（异步预取） |
| ------------------------------------------------- | ------------ | -------------- |
| 全局内存加载效率（Global Memory Load Efficiency） | 23%          | 99%            |
| 每次请求的平均扇区数                              | 6.4          | 4.0            |
| DRAM 吞吐（占峰值百分比）                         | 25%          | 90%            |
| SM 活跃百分比（SM Active %）                      | 62%          | 100%           |
| 核函数执行时间                                    | 18 ms        | 7 ms           |

Here, we see SM Active % approaches 100%, which shows that the SMs have active warps for nearly all cycles. Global Memory Load Efficiency increases from 23% to 99%, meaning nearly every fetched byte is useful.

这里我们看到 SM 活跃百分比接近 100%，这表明 SM 在几乎所有周期都有活跃的 warp。全局内存加载效率从 23% 提升到 99%，意味着几乎每一个取回的字节都是有用的。

Average sectors per request falls from 6.4 to about 4.0, which indicates that requests map cleanly to 128-byte lines at the cache. DRAM throughput rises from 25% to 90% of peak, and overall time improves from 18 milliseconds to 7 milliseconds, about 2.6× faster. These results confirm that offloading bulk copies to the TMA and ping-ponging shared memory buffers keep the GPU busy and hide most of the DRAM latency behind useful work.

每次请求的平均扇区数从 6.4 降到约 4.0，这表明请求干净地映射到了缓存中的 128 字节缓存行。DRAM 吞吐从峰值的 25% 上升到 90%，整体时间从 18 毫秒改善到 7 毫秒，大约快了 2.6 倍。这些结果证实：把批量拷贝卸载给 TMA 并对共享内存缓冲区做乒乓，能让 GPU 保持忙碌，并把大部分 DRAM 延迟隐藏在有用的工作之后。

NVIDIA’s CUDA pipeline API plus TMA is a textbook example of hardware-software codesign. The Pipeline API specifically exposes TMA’s capabilities—and the TMA hardware supports exactly the asynchronous, coalesced, multidimensional copies that cuda::memcpy_async needs.

NVIDIA 的 CUDA pipeline API 加上 TMA，是软硬件协同设计（codesign）的教科书式范例。Pipeline API 专门暴露了 TMA 的能力——而 TMA 硬件恰好支持 cuda::memcpy_async 所需的那种异步、合并、多维的拷贝。

The API and the TMA DMA engine were developed hand in hand so you can express high-level pipeline operations that map closely to the hardware transfer capabilities.

这套 API 与 TMA DMA 引擎是携手开发出来的，因此你可以表达出与硬件传输能力紧密对应的高层流水线操作。

This allows the efficient overlap of memory movement with compute to boost performance.

这使得内存搬运与计算能够高效重叠，从而提升性能。

> For almost all cases, you should write your CUDA kernels with the highest-level and most-recent APIs available for global-to-shared tiling. This includes the CUDA Pipeline API (cuda::memcpy_async). These APIs and libraries are constantly improving and will transparently leverage the latest hardware features like TMA for bulk, strided, and 2D/3D transfers. In addition, they enable advanced performance optimization features like multicast with thread block clusters. When using these APIs, you get all of this “for free"-without requiring code changes.

> 几乎在所有情况下，对于全局到共享的分块，你都应当使用可用的最高层、最新的 API 来编写 CUDA 核函数。这包括 CUDA Pipeline API（cuda::memcpy_async）。这些 API 和库在不断改进，会透明地利用像 TMA 这样的最新硬件特性来进行批量、跨步以及 2D/3D 传输。此外，它们还启用了诸如配合线程块簇的多播（multicast）等高级性能优化特性。使用这些 API 时，你能“免费”获得所有这一切——而无需改动代码。

In summary, when memory access limits your kernel’s performance, offload and overlap data movement by combining careful tiling, double buffering, and TMA-driven asynchronous prefetching. By staging tiles in shared memory and using cuda::memcpy_async alongside pipe.producer_commit and pipe.consumer_wait, you hand off coalesced, multidimensional DMA transfers to the TMA, offloading the global to shared memory transfer.

总之，当内存访问限制了你核函数的性能时，通过结合精心的分块、双缓冲和 TMA 驱动的异步预取，来卸载并重叠数据搬运。通过把瓦片暂存在共享内存中，并在 pipe.producer_commit 和 pipe.consumer_wait 旁边使用 cuda::memcpy_async，你就把合并的、多维的 DMA 传输交给了 TMA，从而卸载了全局到共享内存的传输。

Using TMA to offload memory transfers helps to relieve pressure on the SM’s load/store units to help keep the computation pipeline full. As such, the SM focuses on compute, while shared memory traffic uses the on-chip TMA path. On Blackwell’s massive-bandwidth HBM3e fabric, these techniques are essential to hide DRAM latency, sustain peak throughput, and turn memory-bound kernels into near-compute-bound workhorses.

使用 TMA 卸载内存传输，有助于缓解 SM 加载/存储单元的压力，从而帮助保持计算流水线满载。如此一来，SM 专注于计算，而共享内存流量走片上 TMA 路径。在 Blackwell 那具备巨大带宽的 HBM3e 结构上，这些技术对于隐藏 DRAM 延迟、维持峰值吞吐，以及把访存受限的核函数变成近乎计算受限的主力干将，都是必不可少的。

## Key Takeaways

## 关键要点

Optimizing memory access patterns on GPUs—through coalescing, data reuse, and asynchronous transfers—can shift a kernel from being memory bound to approaching the hardware’s peak capabilities. Small code changes to better align with GPU architecture (such as proper thread grouping, using shared memory, avoiding bank conflicts) can yield massive performance gains. Here are the key takeaways from this chapter:

在 GPU 上优化内存访问模式——通过合并、数据复用和异步传输——能够把一个核函数从访存受限转变为逼近硬件的峰值能力。为更好地契合 GPU 架构而做的小小代码改动（例如恰当的线程分组、使用共享内存、避免 bank 冲突），就能带来巨大的性能收益。以下是本章的关键要点：

_Global-memory coalescing_

_全局内存合并_

Coalescing is achieved when each warp’s accesses fall within as few 128-byte cache lines as possible. Arrange your data and thread indexes so that each warp’s threads read consecutive 4-byte words, letting the hardware fuse them into a few 128-byte transactions. Coalesced memory loads maximize effective DRAM bandwidth, or Global Memory Load Efficiency, and minimize average sectors per request down to the optimal 4.0 value. When using modern versions of Nsight Compute, you can also use sm**sass*data_bytes_mem*\* counters and gpu**dram_throughput.avg.pct_of_peak_sustained_elapsed to profile and optimize memory coalescing.

当每个 warp 的访问落在尽可能少的 128 字节缓存行内时，就实现了合并。请合理组织你的数据和线程索引，使每个 warp 内的线程读取连续的 4 字节字，从而让硬件把它们融合成少数几个 128 字节事务。合并的内存加载能最大化有效 DRAM 带宽（即全局内存加载效率，Global Memory Load Efficiency），并把平均每次请求的扇区数降到最优的 4.0。使用较新版本的 Nsight Compute 时，你还可以使用 sm**sass*data_bytes_mem*\* 计数器以及 gpu**dram_throughput.avg.pct_of_peak_sustained_elapsed 来剖析并优化内存合并。

_Vectorized loads/stores_

_向量化加载/存储_

Use built-in vector types such as float4 for 16-byte vectors. On Blackwell with CUDA 13+, prefer 32-byte per-thread vectors when 32-byte alignment is provable. This includes double4 or a custom struct alignas(32) { float v[8]; }. This reduces instructions per byte and keeps sectors/request at the ideal 4.0 when properly aligned. This way, each thread moves as many elements as possible in one instruction. The number of 128-byte transactions per warp scales with the total bytes requested. Be mindful of alignment: ensure your arrays are allocated with at least 16-byte alignment for float4, which cudaMalloc does by default using 256-byte alignment, typically. Misaligned vector accesses will forfeit these benefits.

使用内置向量类型（如 float4）来处理 16 字节向量。在 Blackwell 上配合 CUDA 13+，当能够证明 32 字节对齐成立时，优先采用每线程 32 字节向量，这包括 double4 或自定义结构体 alignas(32) { float v[8]; }。这样能降低每字节的指令数，并在正确对齐时把每次请求的扇区数保持在理想的 4.0。如此一来，每个线程可以在一条指令中搬运尽可能多的元素。每个 warp 的 128 字节事务数量与请求的总字节数成正比。要注意对齐：确保你的数组以至少 16 字节对齐来分配以配合 float4，而 cudaMalloc 默认通常就以 256 字节对齐来完成这一点。未对齐的向量访问会让这些收益付诸东流。

_Bank-conflict avoidance_

_避免 bank 冲突_

Pad your shared-memory arrays (e.g., make rows 33 floats wide for 32-thread warps) so that no two threads hit the same bank in the same cycle. Removing bank conflicts restores full shared-memory throughput. Try swizzling for a slightly more memory-efficient implementation than padding.

对共享内存数组做填充（例如，为 32 线程的 warp 把每行做成 33 个 float 宽），使任意两个线程不会在同一周期命中同一个 bank。消除 bank 冲突可恢复共享内存的全部吞吐。可以尝试用混洗（swizzling）来实现比填充稍微更省内存的方案。

_Shared-memory tiling and data reuse_

_共享内存分块与数据复用_

Stage working sets in on-chip shared memory (e.g., tiling a matrix in 32 × 32 blocks) so each element is fetched once from DRAM but used many times on the SM. This raises arithmetic intensity and shifts kernels toward being compute bound.

把工作集暂存到片上共享内存中（例如把矩阵按 32 × 32 的块做分块），使每个元素只从 DRAM 取一次，却能在 SM 上被多次使用。这能提升算术强度，并把核函数推向计算受限。

_Read-only data cache_

_只读数据缓存_

Mark small, static lookup tables or coefficients as const **restrict** so the compiler can route loads through the read-only data path when applicable. Uniform broadcasts are lower-latency than DRAM, avoid redundant transactions, and can be served from on-chip cache.

把小而静态的查找表或系数标记为 const **restrict**，以便编译器在适用时把这些加载路由到只读数据路径。相比 DRAM，统一广播的延迟更低，能避免冗余事务，并可由片上缓存来提供服务。

_Overlap host–GPU copies with streams_

_用流重叠主机—GPU 拷贝_

Allocate your host buffers as page-locked (“pinned”) memory, and use cudaMemcpyAsync on multiple streams to overlap H2D/D2H transfers with kernel execution. Pinned memory enables asynchronous DMA transfers, and multiple streams allow copies to overlap with kernel execution to hide PCIe or NVLink transfer latency. Prefer cudaMemcpyAsync with explicit streams and events to overlap H2D/D2H and kernels. Remember that pageable (non-pinned) memory will disable DMA overlap. You should verify whether pageable or pinned memory is used, as observed transfer rates will vary widely depending on this configuration.

把主机缓冲区分配为页锁定（“固定”，pinned）内存，并在多个流上使用 cudaMemcpyAsync，以便把 H2D/D2H 传输与核函数执行重叠。固定内存可启用异步 DMA 传输，而多个流则让拷贝与核函数执行重叠，从而隐藏 PCIe 或 NVLink 的传输延迟。优先使用带显式流和事件的 cudaMemcpyAsync 来重叠 H2D/D2H 与核函数。请记住，可分页（非固定）内存会禁用 DMA 重叠。你应当核实使用的是可分页还是固定内存，因为观测到的传输速率会随这一配置而大相径庭。

_Asynchronous prefetch with TMA + Pipeline API_

_用 TMA + Pipeline API 做异步预取_

Use the C++ libcu++ barrier and pipeline APIs (cuda::barrier and cuda::pipeline) with cuda::memcpy_async to drive TMA (cp.async.bulk.tensor) for global memory → shared memory bulk copies when alignment and scope requirements are met. This offloads coalesced, strided (even multidimensional) copies into shared memory and overlaps them with computation with double buffering. This will reduce pressure on the SM’s LD/ST units and let the SM focus on compute.

当满足对齐与作用域要求时，使用 C++ libcu++ 的 barrier 与 pipeline API（cuda::barrier 与 cuda::pipeline）配合 cuda::memcpy_async 来驱动 TMA（cp.async.bulk.tensor），完成全局内存 → 共享内存的批量拷贝。这会把合并的、跨步的（乃至多维的）拷贝卸载到共享内存中，并借助双缓冲（double buffering）与计算重叠。这将减轻 SM 的 LD/ST 单元压力，让 SM 专注于计算。

It’s important to note that the libcu++ pipeline APIs and TMA continue to evolve. Prefer the staged forms (e.g., producer_acquire(), producer_commit(), consumer_wait(), consumer_release()) for two-stage ping-pong buffers. Use a block-scoped pipeline (e.g., cuda::thread_scope_block) or block-scoped barrier (e.g., cuda::barrier[cuda::thread_scope_block](cuda::thread_scope_block)) unless you specifically need cluster-scoped or grid-scoped copies.

值得注意的是，libcu++ 的 pipeline API 与 TMA 仍在持续演进。对于两阶段乒乓（ping-pong）缓冲，优先采用分阶段形式（例如 producer_acquire()、producer_commit()、consumer_wait()、consumer_release()）。除非你确实需要簇作用域或网格作用域的拷贝，否则请使用块作用域的 pipeline（例如 cuda::thread_scope_block）或块作用域的 barrier（例如 cuda::barrier[cuda::thread_scope_block](cuda::thread_scope_block)）。

_Profile to guide you_

_用剖析来指引你_

Rely on Nsight Compute metrics like Global Memory Load Efficiency, average sectors per request, shared-memory bank conflicts, SM Active %, warp stall reasons, etc. Also, review Nsight Systems timelines to visualize overlaps and stalls, pinpoint bottlenecks, and verify each optimization.

依靠 Nsight Compute 指标，如全局内存加载效率、平均每次请求的扇区数、共享内存 bank 冲突、SM 活跃百分比、warp 停顿原因等。此外，查看 Nsight Systems 时间线，以可视化重叠与停顿、定位瓶颈，并验证每一项优化。

## Conclusion

## 结论

With your memory-access pipeline now firing on all cylinders with coalesced global memory loads, conflict-free tiling, vectorized fetches, read-only caching, and TMA-driven prefetching, you’ve removed the biggest data-movement bottlenecks and freed the SMs to run at full speed.

如今，你的内存访问流水线已经全速运转——合并的全局内存加载、无冲突的分块、向量化的取数、只读缓存以及 TMA 驱动的预取，你已经消除了最大的数据搬运瓶颈，让 SM 得以全速运行。

Throughout this chapter we’ve relied on Nsight Compute and Nsight Systems to expose exactly where warps were starving for data. We also used them to confirm, step by step, that each optimization really did reduce stalls, collapse wasted transactions, and boost sustained bandwidth. Those tools remain your north star whenever you tune a new kernel.

本章自始至终都依靠 Nsight Compute 与 Nsight Systems，精确暴露 warp 在何处因缺数据而“挨饿”。我们还用它们逐步确认：每一项优化确实减少了停顿、消解了浪费的事务，并提升了持续带宽。每当你调优一个新核函数时，这些工具始终是你的指路明星。

In the next chapter, we’ll cover some fundamental latency-hiding techniques in CUDA and GPU programming. These techniques include tuning occupancy, increasing warp efficiency, avoiding warp divergence, and exposing instruction-level parallelism.

在下一章，我们将介绍 CUDA 与 GPU 编程中一些基础的延迟隐藏技术。这些技术包括调优占用率、提升 warp 效率、避免 warp 分歧，以及暴露指令级并行。
