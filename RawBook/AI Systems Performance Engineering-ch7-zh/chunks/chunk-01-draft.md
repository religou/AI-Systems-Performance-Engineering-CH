# 第 7 章 GPU 内存访问模式的剖析与调优

随着 AI 模型规模与复杂度不断增长，GPU 的内存系统往往会成为横亘在理论计算能力与真实性能之间的瓶颈。正如你在第 6 章所见，现代 NVIDIA GPU 将数千个简单、面向吞吐优化的核心与专用的 Tensor Core 组合在一起。它们还包含高带宽内存（HBM）、CPU 与 GPU 一致的统一内存地址空间（例如 Grace Blackwell Superchip）、片上共享内存、缓存，以及像 Tensor Memory Accelerator（TMA）这样的专用直接内存访问（DMA）引擎。

本章将展示各种 CUDA C++ 与 PyTorch 优化技巧，用于对齐数据结构以实现高效内存访问、消除冗余的数据加载，以及借助硬件把数据传输与计算重叠起来。

通过矩阵乘法、张量运算等一系列具体的前后对比示例，你将看到内存访问模式、分块策略与异步数据传输上的微小改动如何减少浪费的带宽、提升算术效率，并把核函数从访存受限（memory-bound）转变为计算受限（compute-bound）。

读完本章后，你将懂得如何编写能更好地利用 GPU 内存层级（memory hierarchy）与硬件优化数据传输引擎的 CUDA 核函数。

## 合并访问与非合并全局内存访问

代码的内存访问模式会极大地影响性能。当一个 warp（线程束）中的线程访问连续的内存地址，且硬件能够把它们合并为更少、更大的事务时，全局内存（global memory）访问速度最快。如果线程访问的是分散或未对齐的地址，设备就无法把请求合并（coalesce）为最少数量的缓存行（cache line）事务——在现代 GPU 上，缓存行是由四个 32 字节扇区（sector）组成的 128 字节缓存行。其结果是产生多得多的内存事务去取回未被使用的数据，很快便耗尽内存带宽。

在 Blackwell GPU 上，单设备 HBM3e 带宽可达 8 TB/s。在 Grace Blackwell GB200 和 GB300（双 GPU 超级芯片）内部，跨两颗 GPU 的带宽提升到 16 TB/s。使用非合并（uncoalesced）内存访问会因为过多的内存事务与停顿，而让这些带宽的绝大部分闲置浪费。

在非合并的情形下，一个 warp 中的每个线程都从分散的地址加载。这会产生许多彼此独立的内存事务。即便一个 warp 中的线程访问的是连续地址，只要第一个地址不是 128 字节对齐的，该 warp 的请求就会横跨两个 128 字节缓存行。

举例来说，如果一个 warp 的首个线程从一个非 128 字节对齐的地址开始，该 warp 的内存请求就会跨越一个缓存行边界，往往导致产生两个 128 字节事务而非一个。在这种情况下，该 warp 可能会多取一个扇区，超出最优的四个扇区，于是两条缓存行上总共取到五个扇区。这是对带宽的浪费。一个未对齐、连续的 128 字节 warp 加载究竟触及 5× 32 B 扇区还是 8× 32 B 扇区，取决于起始偏移量。对齐的访问则能将其保持在 4× 32 B 扇区。

然而在合并的情形下，线程从连续地址加载，这些加载被合并成单个宽事务。图 7-1 对比了合并与非合并的内存访问。

![图 7-1. 对比合并与非合并的内存访问模式](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-1.png)

在核函数代码里，这个问题通常表现为*跨步*（strided）或不规则的索引，使得每个线程都伸向不同的缓存行。当一个核函数的线程用跨步或不规则的索引取数据时，GPU 会发出许多小而非合并的全局内存事务，而不是少数几个全宽加载。

在 Nsight Compute 中，当存在非合并模式时，内存工作负载分析（Memory Workload Analysis）小节会显示更低的全局内存加载效率（Global Memory Load Efficiency）、更高的 DRAM 扇区读取计数，以及高于 4.0 的每次请求平均扇区数。这是因为相比恰当合并的内存访问模式，此时取回了更多扇区，表明你正因为取回的大多是无用字节而浪费带宽。同时 DRAM 吞吐百分比会一直远低于峰值。这印证了你的 warp 正把周期消耗在等待内存上，而不是驱动 ALU。

要摆脱这种访存受限的境地，你可以重新组织数据，让每个 warp 的 32 个线程加载连续元素。要么用 input[idx] 来索引数组（其中 idx = blockIdx.x * blockDim.x + threadIdx.x），要么切换到数组结构体（Structure of Arrays，SoA）布局，使得线程 i 始终触及元素 i。结构体数组（Array of Structures，AoS）与 SoA 之间的区别如图 7-2 所示。

![图 7-2. 结构体数组（AoS）与数组结构体（SoA）](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-2.png)

一旦做出改动，硬件就会自动把该 warp 的全局内存加载合并为更少、更宽的事务，返回更多可用（更少浪费）的数据。Nsight Compute 的计数器会立刻显示出改善。

我们用一个例子来演示这一点。下面的前后对比代码展示了如何把全局内存访问从跨步模式重构为连续模式，从而带来显著的性能提升。

*前置示例（C++）：非合并的跨步访问。* 在这个示例中，每个线程都以步长 2 从输入数组拷贝数据，导致未对齐的内存访问：

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

这个 CUDA C++ 核函数在彼此相距一个 stride > 1 的地址上发出全局内存加载，按定义即是非连续的。这导致每个 warp 生成多个小事务，而非单个宽事务。

*前置示例（PyTorch）。* 在 PyTorch 中，可以用带跨步索引的 gather 操作制造出类似情形：

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

这段 PyTorch 代码使用带跨步索引模式的 torch.index_select，导致底层的 GPU gather 核函数执行非合并加载。具体来说，一个含 32 个线程的 warp 会访问彼此相距 stride * 4 字节的地址。

这就无法完成单个宽事务，反而生成 32 次独立的加载。每个线程从 inp 加载的值，与相邻线程加载的值在内存中相距甚远，从而阻碍了合并。图 7-3 展示了合并访问与跨步访问模式——以及随机访问。

![图 7-3. 合并、跨步与随机内存访问模式](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-3.png)

在 GPU 上运行这些 C++ 与 PyTorch 代码后，我们测量性能指标。在非合并版本中，每个 warp 的内存请求平均被拆成 8 个独立的 32 字节扇区。由于每次 128 字节缓存行取数会被拆为 4 个独立的 32 字节扇区（稍后详述这个数字 4），该访问模式每个 warp 会横跨两条缓存行，以取回那 8 个独立的 32 字节扇区。

在未优化版本中，每个 warp 的非合并加载会把单个逻辑请求拆成多达八个独立的 32 字节扇区，令事务数量急剧膨胀，并使内存流水线陷入“饥饿”。结果是，Nsight Compute 报告的持续动态随机存取存储器（DRAM）吞吐仅约为峰值的 25%。

现在，让我们通过合并内存访问、让线程读取连续元素来优化它，从而使每个 warp 发出更少、更大的事务。

*后置示例（C++）：合并访问。* 只需去掉步长（或将其设为 1），每个线程就拷贝一个连续元素。这种线程访问的对齐让硬件能把内存请求合并为完整的 128 字节事务：

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

为简洁起见，我们省略了合并版本的 PyTorch 实现，因为它已内置于 PyTorch 中。PyTorch 里的合并版本只需做类似 out = inp.clone() 的操作即可。这会高效地拷贝连续元素。事实上，在连续张量上调用 clone() 会在底层使用向量化内存拷贝，与我们的合并核函数类似。

> 你也可以使用带默认 TorchInductor 后端的 torch.compile 来减少冗余拷贝，并在安全时融合相邻算子。clone() 本身已是设备到设备的拷贝，因此通常不会被融合进一个单独的自定义核函数。启用自动调优（autotuning，例如 mode="max-autotune"）有助于在形状稳定时让 TorchInductor 选择合并且向量化的调度方案。我们将在第 13 章和第 14 章深入讲解 PyTorch 编译器。

在合并访问下，或者说无步长时，每个 warp 的线程访问相邻地址。此时，硬件把每个 warp 的加载合并为最少数量的 128 字节事务——当首地址为 128 字节对齐时，往往是每个 warp 一个 128 字节事务；若访问跨越了边界，则为两个。表 7-1 展示了此优化带来的巨大改善。

*表 7-1. 合并与非合并内存访问的性能对比*

| 指标 | 优化前（非合并） | 优化后（合并） |
| --- | --- | --- |
| DRAM 吞吐（占峰值百分比） | 25% | 90% (3.6×) |
| 全局内存加载效率 | 23% | 99% |
| 每次请求平均扇区数 | 8.0 | 4.0（最优） |
| SM 活跃百分比 | 62% | 99% |
| 核函数执行时间（ms） | 4.8 ms | 1.3 ms (3.7×) |

> 注：所有指标表格中的数值均为示意性质，用于阐释概念。不同 GPU 架构上的实际基准测试结果，请参见 GitHub 仓库。

在修正数据布局并合并全局内存访问之后，DRAM 吞吐从峰值的 25% 升至 90%，约提高 3.6×。核函数执行时间改善约 3.7×，从 4.8 ms 降到 1.3 ms，因为更少的停顿让 SM 得以持续向前推进。

全局内存加载效率从 23% 升至 99%，意味着几乎每个取回的字节都是有用的。与此同时，每次请求平均扇区数降至约 4.0。

> 接近 4.0 的值意味着该 warp 的加载已完全合并。当首地址为 128 字节对齐时，全部 32 个线程都映射到一条 128 字节缓存行。这条缓存行有四个 32 字节扇区，因此该指标报告每次请求平均 4.0 个扇区。

在最坏情况下，即跨步或分散访问时，该值可能接近 32，因为在 L2 处活动是以 32 字节扇区为单位报告的。高于 4.0 的值表明存在非合并或未对齐的访问。在未优化版本中，非合并加载会把单个逻辑访问拆成每次请求许多扇区，逼近最坏情况。

这里我们处于每次请求约 4.0 个扇区的水平，表明请求干净地映射到 128 字节缓存行，没有未使用的扇区。随着内存停顿减少，SM 活跃百分比从 62% 提升到 99%。

总而言之，通过让每个 warp 的线程对齐到连续地址，GPU 的内存控制器就能用少数几个大事务、而非数十个微小事务来服务每个 warp。这会把全局内存加载效率（每个事务中返回有用数据的比例）提升到接近 100%，并通过让 warp 保持忙碌而非空闲来抬高 SM 活跃百分比。

至此，你已经看到合并如何重塑线程到地址的映射，使整个 warp 的加载对齐到 GPU 的 128 字节段上。然而，即便是完全合并的 warp，底层仍会发出 32 次独立的 4 字节读取——每个线程一次——迫使硬件把它们重新拼合成每个 128 字节事务。

要消除这最后一层低效，我们转向向量化内存访问：让每个线程在单条指令中取回一个更宽、对齐的数据块（例如一个 16 字节的 float4），使一个合并的 warp 恰好发出四个 128 字节事务，而非 32 个。下面我们深入探讨如何把每线程的加载打包进 CUDA 的内建向量类型。

## 向量化内存访问

内存合并是 NVIDIA GPU 上的一种运行期硬件优化，而向量化内存访问则是一种编译期策略——每条加载或存储指令显式地为每个线程取回多个连续元素（例如 float4，即 16 字节）。这既降低了指令数量，又消除了拼合开销。

Blackwell 上高效的全局内存访问依赖于让你的加载与 GPU 原生的 128 字节事务大小相匹配。当每个线程只读取一个 4 字节的 float 时，一个 32 线程的 warp 仍需把 32 个 4 字节请求拼合起来才能填满一条 128 字节缓存行。

在理想的合并情形下，一个读取 32 个对齐到 128 字节边界的 4 字节字的 warp 恰好映射到四个 32 字节扇区。Nsight Compute 通过每次请求平均扇区数这一指标捕捉到这种浪费——当你的访问未对齐、跨步或分散时，该值可能远超 4.0。这样一来，你会看到全局内存加载效率随着带宽被闲置而下降。

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

当一个 warp 中全部 32 个线程都发出一次 float4 加载时，它们合起来取回 32 × 16 字节 = 512 字节的连续数据，Blackwell 内存控制器随后把它恰好拆成四个 128 字节事务（512 字节 ÷ 每事务 128 字节 = 4 个事务）。这就保持了理想的每个 128 字节事务 4.0 个扇区（前文讨论合并内存访问时提到过），因为该 warp 请求 512 字节，硬件用四个对齐的 128 字节事务、共计 16 个扇区来服务此请求，且每个扇区都被完全利用。

与可能令扇区数膨胀的未对齐或跨步情形相比，向量化的 float4 加载把每线程的加载指令减少了 4×。在满足对齐时，这有助于维持理想的每次请求 4.0 个扇区。

> 像 PyTorch 编译器这样的高层 GPU 编译器，在满足对齐与连续性要求时，往往能生成向量化的内存操作。你可以通过使用与向量宽度相匹配的每线程分块来促成这一行为。因此，各类框架通常也能在底层实现类似的内存事务缩减。不过，从一开始就恰当地对齐数据仍然是有益的。

作为这项向量化内存访问优化的结果，全局内存加载效率朝 100% 攀升。当每个 128 字节事务被完全利用时，每事务扇区数保持在 4.0，同时该 warp 发出多个对齐事务来服务一个更宽的请求。

向量化加载降低了每移动一字节所需的内存指令数，往往还能提升有效带宽。要从向量化加载中获益，数据指针必须对齐到向量宽度。CUDA 运行时与驱动的分配函数（例如 cudaMalloc）返回的设备指针至少对齐到 256 字节。

这一基础对齐足以在分配边界处满足 16 字节和 32 字节的向量对齐。如果加上的元素偏移量不是向量宽度的整数倍，添加偏移量可能会破坏对齐。下面的类型转换向编译器断言了预期类型，并假设指针已经正确对齐：

```
auto ptr4 = reinterpret_cast<const float4*>(ptr);
```

在转换之前，务必确保指针值是 16 字节（4 个 float）的整数倍。这是因为 cudaMalloc() 返回至少 256 字节的对齐。如果加上的元素偏移量不是 4 个 float 的整数倍，添加偏移量可能会破坏对齐。正确的对齐是向量化加载的前提条件。

在基址对齐到 32 字节的情况下，每个线程每次迭代读取 32 字节，通常在 Hopper 上编译为每线程两条 16 字节向量加载指令，或在 Blackwell 上编译为一条 32 字节加载指令。于是一个 32 线程的 warp 请求 1,024 字节。每条 128 字节缓存行由四个 32 字节扇区组成。因此，当访问恰当对齐时，该 warp 生成八个对齐的 128 字节事务，共计 32 个扇区。这样一来，每次访问都落在天然边界上，避免了被拆分的事务。

值得注意的是，截至本文撰写时，CUDA 在其 <vector_types.h> 中并未提供内建的 8-float 聚合 CUDA 向量类型（例如八个组合的 float32）。你可以自定义一个含 8 个 float 值的结构体，然后每线程将其作为两个 float4 值加载。你会用 alignas(32) 为该类型施加 32 字节对齐。这样一来，指针值就是 32 字节的整数倍。现代 GPU 上的全局内存向量指令通常每线程加载 16 字节（Hopper）或每线程 32 字节（Blackwell）。在 Hopper 每线程 16 字节的情形下，一次 8-float 聚合加载会编译为每线程两次 16 字节加载，以涵盖全部 32 字节。

而 Blackwell 在使用恰当的 32 字节对齐数据时，则会编译为每线程一次 32 字节加载。

在每线程 32 字节加载（例如 8 个 float × 每 float 4 字节）时，一个 warp 移动 1,024 字节（1,024 字节 = 每 warp 32 个线程 × 每线程 32 字节）。由于每条缓存行由四个 32 字节扇区组成，这通常映射到 8 × 128 字节缓存行。在每线程 16 字节加载（例如四个 float4）时，一个 warp 移动 512 字节（512 字节 = 每 warp 32 个线程 × 每线程 16 字节），即 4 × 128 字节缓存行。当基指针与每线程步长天然对齐时，硬件合并器会把各通道（lane）的访问合并为完整缓存行，避免拆分扇区。这种“机械同理心”（mechanical sympathy）对于达到峰值吞吐至关重要。因此，对于 Blackwell 上的 256 位加载，你应当强制 32 字节对齐以维持峰值性能。

我们来看一个简单向量拷贝核函数中标量与向量化内存访问的对比示例。

*前置示例（C++）：标量拷贝。* 每个线程拷贝一个 float：

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

*前置示例（PyTorch）。* 在 PyTorch 中，一个标量逐元素拷贝可以用 Python 循环来演示：

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

在标量 4 字节加载下，当对齐时，一个 warp 往往为 128 字节发出一个 128 字节事务。否则，若访问跨越了 128 字节边界，就会使用两个事务。使用 float4（每线程 16 字节）意味着每个 warp 每条内存指令发出的数据量多 4×——每 warp 512 字节——拆成四个 128 字节事务。这让传输能以更少的指令完成。接下来，我们通过使用向量加载来优化。

*后置示例（C++）：向量化拷贝。* 每个线程拷贝一个 float4（16 字节）：

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

*后置示例（PyTorch）。* 我们可以通过使用张量视图与 clone 来模拟向量化拷贝：

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

在重塑后的张量上调用 clone() 会让 PyTorch 执行连续拷贝。这与逐元素拷贝形成对比。通过使用 float4 向量加载，每个线程每条指令移动 16 字节，这一点针对 Hopper 做了优化。（Blackwell 支持每线程 32 字节向量加载。）一个 warp 为该次加载发出 512 字节，当对齐时通常拆成四个 128 字节事务。收益来自每移动一字节所需指令更少、更好的扇区利用率，以及恰当的对齐，从而在现代 GPU 上最大化内存吞吐。这体现出对内存带宽好得多的利用率，如表 7-2 所示。

> Blackwell 相对于 Hopper，把每线程向量加载/存储的字节数翻了一倍。借助 CUDA 13，Blackwell 可以执行 32 字节的向量加载与存储，而不是 Hopper 上的 16 字节。这要求编译器能够证明 32 字节对齐。否则，它可能把每线程的加载/存储拆成两条 16 字节指令。这会增加加载/存储相同数据量所需的指令数，从而对性能产生负面影响。

通过使用恰当对齐的 float4 向量加载，每个线程每条指令移动 16 字节。这体现出对内存带宽好得多的利用率，如表 7-2 的结果所示。

*表 7-2. 标量与向量化内存访问的 Nsight Compute 指标对比*

| 指标 | 优化前（标量） | 优化后（向量化） |
| --- | --- | --- |
| 全局内存加载效率 | 28% | 97% |
| 每次请求平均扇区数 | 31.8 | 4.0 |
| DRAM 吞吐（占峰值百分比） | 25% | 90% (3.6×) |
| 核函数执行时间 | 4.2 ms | 1.2 ms（改善 3.5×） |

这些指标印证了改善。全局内存加载效率从 28% 跃升至 97%（约 3.5×），占峰值的 DRAM 吞吐百分比从 25% 提升到 90%（约 3.6×），把核函数整体运行时间削减了约 3.5×。

这项优化把核函数执行运行时间削减了大约 3.5×，从 4.2 ms 降到 1.2 ms。全局内存加载效率从 28%（标量）跃升至 97%（向量化），意味着现在几乎每个取回的 128 字节事务都被用上。

每次请求平均扇区数朝着每缓存行 4.0 扇区的理想值下降。这表明每条 128 字节缓存行都被完全利用。每条 warp 级（warp-wide）的 float4 指令发出四个对齐的 128 字节事务来移动 512 字节。因此，每个事务都被完全利用。

内存请求的减少，以及用一次向量加载替代四次标量加载，把全局内存加载效率从 28% 提升到 97%，并把持续 DRAM 吞吐从约峰值的 25% 抬高到约 90%。在这种情况下，warp 执行的内存操作大为减少，且每次操作获取更多数据，因此它们花在等待内存上的时间大大减少，而花在执行有用工作上的时间更多。最终结果是一个显著更快的核函数。

> 向量化减少指令数量。合并与对齐则改善每次请求的扇区数。

我们快速对比一下上一节的合并全局内存访问与这里讨论的向量化内存访问。合并确保一个 warp 中的线程全部命中一个大的连续块。向量化确保每个线程取回一个恰好映射到那些大块上的宽数据块。二者的区别汇总于表 7-3。

*表 7-3. 对比合并内存访问（上一节）与向量化内存访问*

| 方面 | 合并 warp 加载（线程间，interthread） | 向量化（线程内，intrathread） |
| --- | --- | --- |
| 粒度 | 线程 ↔ 连续内存地址 | 每线程 ↔ 向量加载 |
| 典型修复方式 | 连续索引（stride = 1） | 使用 float4、带显式 32 字节对齐的自定义 32 字节结构体，或 32 字节对齐的向量类型（例如 Blackwell 与 CUDA 13+） |
| 每 warp 事务数 | 32 线程 × 4 字节 → 128 字节，对齐时等于 4 个 32 字节扇区（4 个扇区） | 32 线程 × 16 字节 → 512 字节，等于 4 个 128 字节事务，对应 16 个 32 字节扇区（16 个扇区） |
| 指令数量变化 | 加载数量相同 | 1 次向量加载替代 4 次标量加载 |
| 收益 | 更少的分散扇区 | 大小完美、更少的扇区 |

通过组合这两种策略，你就能确保 warp 命中连续块，同时让每个线程取回一个宽而对齐的向量。这有助于你把核函数的内存带宽利用率推向峰值，并释放 GPU 的全部威力。
