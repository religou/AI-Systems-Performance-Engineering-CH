因此，对于一个给定的 warp，若各线程的 threadIdx.y 固定、跨越线程 0–31，那么所有线程访问的都是 tile[constant_row][varying_col]，也就是在不同的列上使用同一个行索引。这意味着这 32 个地址全部落在 tile 数组的同一行内，而该行占据单个内存 bank 段，从而引发一次 32 路 bank 冲突（bank conflict）。结果，这些读取被序列化，速度降为原来的 1/32。

> 请记住，Blackwell 的共享内存有 32 个 bank。这个数字恰好等于一个 warp 中的线程数 32。因此，如果所有线程都索引到同一个 bank——正如固定行、变化列时所发生的那样——你总会得到一次完整的 32 路冲突。这种一一对应关系意味着：在 warp 粒度上任何未对齐的访问模式，都会迫使每个线程串行地通过同一个 bank——无论 GPU 架构如何演进都是如此。

在这种情况下，32 个线程试图从 32 个不同的地址读取，而这些地址全都在 bank 0 中。这导致内存访问被严重序列化、性能低下。下面我们应用填充（padding）来消除这些冲突。

*后例（CUDA C++）：带填充的转置（避免 bank 冲突）*。这里我们加入一小段填充，即一个额外的、未使用的列，使共享内存的每一行都从不同的 bank 取模值开始。这样，bank 索引冲突就被这个偏移量消除了：

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

（主机端代码与初始化设置与朴素版本相同，为简洁起见此处省略。）

> PyTorch 本身并不暴露任何用于共享内存填充的高层 API，因此你必须在 CUDA 核函数中手动实现，并用 torch.utils.cpp_extension 加载它们。不过在实践中，PyTorch 依赖 cuDNN 和 cuBLAS 等优化过的库，这些库在底层实现了各种技术来避免 bank 冲突并最大化吞吐。

给每一行共享内存填充一个额外元素，会让每行变成 33 个 float。填充 1 会改变索引运算，使得 tile[row][col] 的地址在每个线程的低 5 位上各不相同，而不再是所有线程共享同一个 5 位 bank 索引。这确保了每个线程索引都映射到不同的 bank。

在带填充的核函数中，当一个 warp 读取 tile[threadIdx.y][threadIdx.x] 时，线程 0–31 的地址会跨越多个 bank，而不再全部命中 bank 0。既然 warp 中的每个线程在读取时都访问不同的 bank，bank 冲突就被彻底消除了，如表 7-5 所示。

*表 7-5. 消除共享内存 bank 冲突的效果*

| 指标 | 前（无填充） | 后（有填充） |
| --- | --- | --- |
| 共享内存加载 bank 冲突 | 4.8 million | 0 |
| 共享内存利用率（占峰值百分比） | 52% | 100% |
| warp 停顿占比（内存） | 38% | 0.5% |
| 核函数执行时间（ms） | 4 ms | 1.3 ms（提升 3 倍） |

表 7-5 中的 Nsight Compute 指标印证了这一影响：共享内存加载 bank 冲突降到了 0。根据 Nsight Compute，消除冲突后共享内存吞吐从 52% 上升到 100%。warp 停顿占比——即 warp 因等待共享内存读取而停顿的周期百分比——从约 38% 降到接近 0%。

至此，我们成功消除了 bank 冲突，释放出片上共享内存的全部性能。由于共享内存访问从序列化变为完全并行，这一改动把核函数的执行时间从 4 ms 缩短到 1.3 ms，提升了约 3 倍。

这恢复了共享内存访问的完全并行性。填充的代价——每 32 个 float 多出 1 个——微不足道（约 3% 的内存开销），相比之下所获得的性能收益相当可观。

除了填充之外，还有一种有效的替代方案是*混洗*（swizzling）。混洗是一种编译期的索引变换，它把用于共享内存的线性索引“打乱”，使相邻的线程映射到不同的 bank。例如，可以把索引与一个位掩码做 XOR，或使用一个取模偏移，来实现无冲突的访问模式。

混洗在保证完美 bank 并行的同时，避免了填充带来的那一点点内存开销。填充实现起来更简单，但混洗能以零内存开销达到同样的目标——而且这个词念起来也挺有意思！

> NVIDIA 的 CUTLASS 库以及其他高性能的基于 CUDA 的库，会在其瓦片迭代器中使用索引混洗，以确保线程映射到各自独立的 bank、避免 bank 冲突并优化共享内存的使用。

总之，在使用共享内存时，重要的是让一个 warp 中的线程并行访问不同的内存 bank，而不是在同一个 bank 上排队。填充和混洗这类技术能提升共享内存加载/存储效率，只要用到共享内存，就能带来更高的吞吐和更好的性能。

接下来，让我们探索一种彻底避免共享内存、在线程之间直接通信的技术。

## Warp Shuffle 内建函数：避免共享内存与显式同步

前面这种避免 bank 冲突的技术，假设我们使用共享内存在线程之间通信。但如果我们能够彻底避开共享内存——以及它带来的 bank 冲突问题——又会怎样呢？

NVIDIA GPU 支持 warp 同步原语，允许同一个 warp 中的线程通过寄存器而不是共享内存来交换数据。事实上，这些原语只在单个 warp 内工作，线程与它的 31 个同胞交换数据。因此这种 warp 内通信不涉及任何内存 bank——也就不可能出现 bank 冲突。

其中最常见的是 __shfl_sync 内建函数（shuffle）。__shfl_sync 让你可以把一个线程的值广播（broadcast）给 warp 中的所有其他线程。你还可以在完全不写入共享内存的情况下执行 warp 级归约（reduction）。请记住，这些内建函数让线程通过寄存器（而非共享内存）交换数据，从而彻底消除了共享内存 bank 冲突。

> 如果所有线程访问的都是完全相同的内存地址，现代 GPU 会自动广播这一个共享内存值。在这种特殊的单地址情形下，这可以避免一次 bank 冲突。而 warp shuffle 则是利用广播，来避免更一般的、任意的、多值数据访问模式下的 bank 问题。

设想你需要在单个 warp 内对 32 个每线程的部分结果求和。朴素的共享内存做法会让每个线程把自己的值写入一个共享数组，调用一次同步屏障，然后读取并累加全部 32 个条目。这既有 bank 冲突的风险，又增加了额外的同步开销。

借助 __shfl_sync，你可以完全在寄存器中完成一次树形归约。例如，我们使用一个便捷变体 __shfl_down_sync 来执行蝶形归约，使每个线程读取另一个偏移若干通道（lane）之外的线程所持有的值，如下所示：

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

在这里，每次 __shfl_down_sync 都直接读取另一个通道的寄存器，每一步都把活跃线程数减半，直到通道 0 累加出完整的总和。由于所有通信都停留在寄存器中，没有共享内存流量，因此零 bank 冲突、也无需额外同步。

简而言之，__shfl_sync 及其变体完全在 warp 的执行通道内完成操作。这避免了 bank 冲突，因为根本不使用共享内存。而且它通常更快，因为它使用寄存器并减少了共享内存指令的数量。

许多高性能的 warp 级归约完全绕开共享内存，转而使用 shuffle 内建函数——它们仅用寥寥几条指令就通过寄存器直接交换值，且不产生任何 bank 冲突。CUDA 的 Cooperative Groups API 正是构建在这些原语之上。该 API 提供了 thread_group.shuffle() 这样的调用来简化 warp 内通信。像 __reduce_sync() 这样的辅助内建函数，最终在现代 NVIDIA GPU 上会被编译为这些 shuffle 模式，用于 warp 内的数据交换。

> Shuffle 仅在单个 warp 内工作。在现代 GPU 架构上进行 warp 间交换时，还可以考虑使用 cooperative groups 的线程块级归约，或在受支持时使用线程块簇（thread block cluster）级别的通信。我们将在第 10 章讨论这些概念。

不过请记住，shuffle 局限于单个 warp 内的这 32 个线程。每当你需要在 warp 之间传递数据时，仍然必须退回到共享内存或全局内存，并配合适当的同步。

在第 10 章，我们会探讨 warp 间的模式，如 cooperative group（CG）同步原语、多 warp shuffle 方法，以及线程块簇（CTA cluster）。所有这些底层最终都使用同一套硬件内建函数。掌握 warp 内和 warp 间这两类技术都至关重要，因为与内存相关的瓶颈始终是 GPU 性能问题最常见的根源之一。

## 只读数据缓存

当所有线程读取相同的值，或某个线程反复读取不会改变的数据时，如果不利用 GPU 的缓存机制，就会成为性能瓶颈。例如，考虑一个大型查找表，比如自然语言处理（natural language processing，NLP）模型中的嵌入向量，它在推理期间是只读的。许多线程可能需要并行访问这个向量。朴素的实现可能每次都从全局内存取数，尽管这些数据是不可变的、本可以缓存在片上。

请注意，我们这里所说的只读缓存，不同于前面在 GPU 内存层级一节讨论过的 64 KB 常量内存缓存。它是一个面向不可变数据的大容量缓存。而常量内存缓存对于大数组来说太小了。现代架构依赖更大的 L1/只读缓存来处理这类嵌入向量查找——而不是试图把这些数据硬塞进小小的 64 KB 常量内存缓存里。

在现代 GPU 上，全局内存加载会被自动缓存到 L2、且通常也会缓存到 L1。你可以使用 const __restrict__ 限定的指针，把函数参数声明为非一致性的（non-coherent）、以读为主（read-mostly，相对于只读而言）。对于只读数据，当编译器能够证明其不可变性、无别名性和安全性时，它可以把加载路由到这条只读路径上。这让编译器/硬件能够把不变的数据经由只读 L1 缓存传送——该缓存延迟更低，尤其是对广播访问而言——而且不会驱逐其他已缓存的数据。

> 在现代 CUDA 中，你通常不需要显式调用 __ldg()。如果一个指针是 const __restrict__，当编译器能够保证安全时，它可能会为全局内存加载使用只读数据缓存。较老的 __ldg() 内建函数仍然可用于显式控制，但在现代编译器下一般并不需要。

一个常见的性能陷阱是忘记告诉编译器某个缓冲区确实是只读的，这意味着它不会使用只读（非一致性）数据路径，也不会把那些加载经由专门的只读缓存来路由。相反，每一次访问都变成一次普通的全局内存加载，导致冗余的 DRAM 流量和不必要的缓存未命中。

当你剖析这样一个核函数时，你会发现相同的地址被反复从片外内存取回、在指令流中看不到任何 __ldg() 操作、观察到该数组出人意料地低的 L2 命中率，并测得偏高的 DRAM 吞吐——就像一个未缓存的工作负载。

解决办法是通过把数据标记为 const __restrict__，或显式使用 __ldg() 内建函数来加载它，从而利用只读路径。这告诉硬件该数据不会被修改，允许它被缓存到专门的只读缓存中——该缓存紧邻 L1，对广播加载具有更低的延迟。

当一个 warp 发起一次常量缓存加载（在较老的 GPU 上为 __constant__ 或统一的 __ldg()）时，如果这些访问命中同一地址，硬件可以用一次事务服务全部 32 个通道。通过把该值广播给每个线程，它只用一个周期，而不必执行 32 次单独的加载。对于查找表、系数等统一数据，这种 warp 级（warp-wide）广播同时降低了延迟和内存带宽占用。这让你几乎可以“免费”地每个 warp 取一次共享常量，而不是每个 warp 取 32 次（每线程 1 次）。

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

在这些朴素版本中，每次查找在首次使用之后很可能会命中 L2（因为 L2 会缓存它），但完全没有用到专门的只读缓存。硬件仍可能把它当作普通的全局数据来对待，这可能驱逐其他有用的数据，或者在一个 warp 中多个线程读取同一个 table[t] 时无法充分利用广播缓存。

现在我们通过把 table 标记为 const __restrict__ 来优化这个核函数。这会提示硬件应当使用只读缓存路径，如下所示：

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

> 在 PyTorch 环境下，你可以通过用 torch.utils.cpp_extension 编写一个小的 CUDA 扩展来实现同样的技巧，在其中把你的嵌入表指针声明为 const __restrict__。你也可以使用 PyTorch 编译器来优化这段代码。我们将在第 13 章和第 14 章深入介绍 PyTorch 编译器。

简而言之，把 table 标记为 const __restrict__ 是在告诉编译器这些值是不可变的、且不存在别名，这就允许在安全的情况下使用只读数据路径。这提高了缓存命中率并减少了片外流量。表 7-6 展示了此改动前后的 Nsight Compute 测量结果。

*表 7-6. 只读缓存的收益*

| 指标 | 前（无 __restrict__） | 后（使用 __restrict__） |
| --- | --- | --- |
| 全局内存加载效率（Global Memory Load Efficiency） | 52% | 97% |
| DRAM 吞吐（GB/s） | 600 GB/s | 200 GB/s |
| L2 读吞吐（GB/s） | 1,500 GB/s | 1,800 GB/s |
| SM 活跃百分比（SM Active %） | 45% | 93% |
| 核函数执行时间（ms） | 2.5 ms | 1.0 ms |

加入 const __restrict__ 之后，核函数时间提升了约 2.5 倍，从 2.5 毫秒降到 1.0 毫秒，因为 warp 因等待 DRAM 而停顿的时间更少了。SM 活跃百分比从 45% 上升到 93%，表明几乎每个周期都有活跃的工作，而不是空闲地等待内存。

DRAM 吞吐从 600 GB/s 降到 200 GB/s，而 L2 读吞吐从 1,500 GB/s 提升到 1,800 GB/s，因为更多请求在缓存中得到了满足。全局内存加载效率从 52% 提升到 97%，证实了大多数取回的缓存行携带的都是有用数据。

> Nsight Systems 呈现的是整体 GPU 活动的时间线视图，而 Nsight Compute 报告的是逐核函数的指标，如 SM 活跃百分比。当你需要定量的逐核函数分析时，请使用 Nsight Compute。

由于更多流量由片上缓存服务、而不必远赴 DRAM，计算单元得以持续获得数据供给，算术强度（arithmetic intensity）随之提高。正是这种平衡把一个核函数从访存受限（memory-bound）区间推向计算受限（compute-bound）区间。对于具有很强二维或三维局部性的只读及读/写访问模式，还可以使用纹理对象（texture object）和表面对象（surface object）。通过把一个数组绑定到 cudaTextureObject_t 并使用 tex1Dfetch 或 tex2D 取数，硬件可以以很高的缓存命中率利用空间局部性，并支持环绕（wrapping）和插值等特性。表面对象则允许在类似的访问模式上进行写入。

> 尽管纹理引用与表面引用 API 已被弃用，纹理对象与表面对象 API 仍受支持，适用于具有二维或三维局部性的访问模式。不过，对于大多数涉及一维数据的 AI 工作负载，使用只读数据缓存配合常量内存要简单得多，也是更受青睐的做法。

总之，把只读数据标记为 const __restrict__，以接入低延迟的只读缓存，削减 DRAM 流量并提升 SM 活跃度。每当你的访问模式具有普通缓存可能无法最优处理的 2D/3D 局部性时，就考虑使用纹理或表面内存。这些技术合在一起，能够坍缩内存停顿、提升缓存利用率，并为访存受限的核函数释放出可观的性能收益。

## 异步内存预取与 Tensor Memory Accelerator

在前面的小节中，我们看到把几十次 4 字节加载合并成一次 128 字节事务，如何显著提升了全局加载效率并削减了每次请求中被浪费的扇区。然而，即便是完美合并的加载，仍会让一个 warp 停顿整整一趟 DRAM 往返的时间。

以 Blackwell 为例，在任何计算开始之前，一趟完整的 DRAM 往返都在数百个周期的量级。为了隐藏这段延迟，我们需要让数据传输与计算重叠。正是这种重叠隐藏了大部分 DRAM 延迟。

CUDA 的 Pipeline API 连同 Tensor Memory Accelerator（TMA）硬件引擎，把这一思路带到了线程块层面。你不必让每个 warp 使用 SM 的加载与存储（LD/ST）单元去从全局内存取数，而是可以调用 TMA 引擎，把一整个瓦片从全局内存异步取入共享内存，如图 7-7 所示。

![图 7-7. TMA 从全局 HBM 异步取数据到共享内存](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-7.png)

要启动 TMA 传输，你可以调用 cuda::memcpy_async()。在现代 GPU 架构上，cuda::memcpy_async() 连同 cuda::pipeline 一起，暴露了用于全局到共享异步传输的硬件引擎。这在可用时包含 TMA。它会使用 TMA 的片上 DMA 引擎来执行异步的批量数据传输。在 CUDA 中的实现如下：

```
cuda::memcpy_async(sharedBuf,
               globalPtr + offset,
               cuda::aligned_size_t<16>(bytes),
               pipe);
pipe.producer_commit();
```

而当 TMA 处理批量拷贝时——包括合并、跨步传输，乃至多维传输——核函数则在计算上一个瓦片。这被称为*双缓冲*（double buffering），或*乒乓*（ping-ponging）。

通过用 TMA 实现双缓冲，SM 的加载/存储单元现在可以腾出来做真正的工作，因为 TMA 的 DMA 引擎在后台替我们搬运数据。实际效果是，数据搬运变成了异步的——当 TMA 把下一个瓦片的数据流式送入时，SM 的 warp 正在计算上一个瓦片。正是这种重叠隐藏了那 800 个周期的 DRAM 延迟。

具体来说，TMA 能够在全局内存与共享内存之间进行 1D–5D 的批量拷贝以及任意跨步，而不阻塞 SM 的指令流水线。通过把这些传输从 SM 卸载到 TMA，你的核函数发出的 LD/ST 指令大为减少，省去了额外的同步，并让 warp 调度器几乎把每个周期都花在有用的计算上，而不是等待内存。

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

核函数启动后，每个线程块立即分配两个共享内存瓦片（tile0、tile1），并构造一个块级作用域的流水线，使线程块中的所有线程协同其异步 DMA，如下所示：

```
__shared__ cuda::pipeline_shared_state<
    cuda::thread_scope_block, 2> state;

auto pipe =
    cuda::make_pipeline(cuda::this_thread_block(),
        &state);
```

为了给流水线预热，我们使用 TMA 提交一次异步拷贝，它甚至能合并跨步或多维传输，并在后台把字节从全局内存流式送入 tiles[0]，如下所示：

```
pipe.producer_acquire();
cuda::memcpy_async(tiles[0],
                   global_ptr + 0 * TILE_SIZE,
                   cuda::aligned_size_t<32>{bytes},
                   pipe);
pipe.producer_commit();
```

在使用这些数据之前，我们执行下面的操作，以确保恰好阻塞足够长的时间、等 TMA 完成拷贝：

```
pipe.consumer_wait();
processTile(tiles[0]);
pipe.consumer_release();
```

在主循环内部，我们通过以下步骤交替缓冲区：对上一个瓦片执行 pipe.consumer_wait(); + processTile()，然后 pipe.consumer_release()、pipe.producer_acquire() + 一次新的 cuda::memcpy_async() 写入另一个瓦片，以及 pipe.producer_commit()。然后循环往复！

通过在 tile0 与 tile1 之间乒乓（ping-pong），每一次新的 memcpy_async 都与对上一个缓冲区的 processTile 重叠。SM 上的加载和存储单元压力更低，从而有更多指令发射槽可用于计算。与此同时，TMA 并行地搬运数据。这消除了冗余的全局内存加载、减少了同步开销，并让 warp 保持忙碌，而不是停顿在内存上。

从全局内存到共享内存的异步预取，把延迟隐藏在计算之后。线程在对先前加载的数据进行计算的同时，把即将用到的数据预加载到共享内存中。

这种模式在访存受限的循环和张量计算中尤其有效。在现代 GPU 上，TMA 可以在当前瓦片正被处理时，为矩阵乘法流式送入下一个瓦片。

> 在全局内存与 SMEM/TMEM 之间搬运 2D 和 N 维瓦片时，TMA 是分块批量拷贝的首选路径。优先在块作用域使用 cuda::memcpy_async 配合 cuda::pipeline；在 Hopper/Blackwell 上，只要对齐和方向允许（例如全局内存 ↔ 共享内存），其实现就会利用 TMA（cp.async.bulk.* 系列）。

我们用一次合并的全局拷贝加上许多次快速的共享内存加载，换掉了许多次分散的全局读取。鉴于 DRAM 与片上 SRAM 之间的差距，这样做是划算的。表 7-7 汇总了基于 TMA 的双缓冲实现前后的 Nsight Compute 指标。

*表 7-7. 朴素核函数（无预取）与 TMA 加速的双缓冲的对比*

| 指标 | 前（无预取） | 后（异步预取） |
| --- | --- | --- |
| 全局内存加载效率（Global Memory Load Efficiency） | 23% | 99% |
| 每次请求的平均扇区数 | 6.4 | 4.0 |
| DRAM 吞吐（占峰值百分比） | 25% | 90% |
| SM 活跃百分比（SM Active %） | 62% | 100% |
| 核函数执行时间 | 18 ms | 7 ms |

这里我们看到 SM 活跃百分比接近 100%，这表明 SM 在几乎所有周期都有活跃的 warp。全局内存加载效率从 23% 提升到 99%，意味着几乎每一个取回的字节都是有用的。

每次请求的平均扇区数从 6.4 降到约 4.0，这表明请求干净地映射到了缓存中的 128 字节缓存行。DRAM 吞吐从峰值的 25% 上升到 90%，整体时间从 18 毫秒改善到 7 毫秒，大约快了 2.6 倍。这些结果证实：把批量拷贝卸载给 TMA 并对共享内存缓冲区做乒乓，能让 GPU 保持忙碌，并把大部分 DRAM 延迟隐藏在有用的工作之后。

NVIDIA 的 CUDA pipeline API 加上 TMA，是软硬件协同设计（codesign）的教科书式范例。Pipeline API 专门暴露了 TMA 的能力——而 TMA 硬件恰好支持 cuda::memcpy_async 所需的那种异步、合并、多维的拷贝。

这套 API 与 TMA DMA 引擎是携手开发出来的，因此你可以表达出与硬件传输能力紧密对应的高层流水线操作。

这使得内存搬运与计算能够高效重叠，从而提升性能。

> 几乎在所有情况下，对于全局到共享的分块，你都应当使用可用的最高层、最新的 API 来编写 CUDA 核函数。这包括 CUDA Pipeline API（cuda::memcpy_async）。这些 API 和库在不断改进，会透明地利用像 TMA 这样的最新硬件特性来进行批量、跨步以及 2D/3D 传输。此外，它们还启用了诸如配合线程块簇的多播（multicast）等高级性能优化特性。使用这些 API 时，你能“免费”获得所有这一切——而无需改动代码。

总之，当内存访问限制了你核函数的性能时，通过结合精心的分块、双缓冲和 TMA 驱动的异步预取，来卸载并重叠数据搬运。通过把瓦片暂存在共享内存中，并在 pipe.producer_commit 和 pipe.consumer_wait 旁边使用 cuda::memcpy_async，你就把合并的、多维的 DMA 传输交给了 TMA，从而卸载了全局到共享内存的传输。

使用 TMA 卸载内存传输，有助于缓解 SM 加载/存储单元的压力，从而帮助保持计算流水线满载。如此一来，SM 专注于计算，而共享内存流量走片上 TMA 路径。在 Blackwell 那具备巨大带宽的 HBM3e 结构上，这些技术对于隐藏 DRAM 延迟、维持峰值吞吐，以及把访存受限的核函数变成近乎计算受限的主力干将，都是必不可少的。

## 关键要点

在 GPU 上优化内存访问模式——通过合并、数据复用和异步传输——能够把一个核函数从访存受限转变为逼近硬件的峰值能力。为更好地契合 GPU 架构而做的小小代码改动（例如恰当的线程分组、使用共享内存、避免 bank 冲突），就能带来巨大的性能收益。以下是本章的关键要点：

*全局内存合并*
