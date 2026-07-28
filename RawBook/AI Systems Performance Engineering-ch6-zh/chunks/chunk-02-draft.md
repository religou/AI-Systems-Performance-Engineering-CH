```
// 2d_kernel.cu

#include <cuda_runtime.h>
#include <iostream>

//-------------------------------------------------------
// Kernel: my2DKernel running on the device (GPU)
//   - input  : device pointer to float array of size width×height
//   - width  : number of columns
//   - height : number of rows
//-------------------------------------------------------
__global__ void my2DKernel(float* input, int width, int height) {
    // Compute 2D thread coordinates
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Only process valid pixels
    if (x < width && y < height) {
        int idx = y * width + x;
        input[idx] *= 2.0f;
    }
}

int main() {
    // Image dimensions
    const int width  = 1024;
    const int height = 1024;
    const int N      = width * height;

    // 1) Allocate and initialize host image
    float* h_image = nullptr;
    cudaMallocHost(&h_image, N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        h_image[i] = 1.0f;  // e.g., initialize all pixels to 1.0f
    }

    // 2) Allocate device image and copy data to device
    cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    float* d_image = nullptr;
    cudaMallocAsync(&d_image, N * sizeof(float), s);
    cudaMemcpyAsync(d_image, h_image, N * sizeof(float),
                    cudaMemcpyHostToDevice, s);

    my2DKernel<<<blocksPerGrid2D, threadsPerBlock2D,
                 0, s>>>(d_image, width, height);

    cudaMemcpyAsync(h_image, d_image, N * sizeof(float),
                    cudaMemcpyDeviceToHost, s);
    cudaStreamSynchronize(s);

    cudaFreeAsync(d_image, s);
    cudaStreamDestroy(s);
```

这里再次给出完整的核函数（设备端）与调用（主机端）代码。同样的模式可以直接推广到 3D：只需对 blocksPerGrid 和 threadsPerBlock 都使用 dim3(x, y, z)，即可把体数据直接映射到 GPU 的线程层级上。

> 本书大多数情况下对 blocksPerGrid 和 threadsPerBlock 使用 1D 或 2D（分块）取值。在 1D 情形下，你可以把 blocksPerGrid 和 threadsPerBlock 定义为简单的常量，而不必使用 dim3。

### 异步内存分配与内存池

如前面示例所示，标准的 cudaMalloc/cudaFree 调用是同步的，且开销相对较大。它们需要一次完整的设备同步（相对较慢），并涉及 mmap/ioctl 等操作系统级调用来管理 GPU 内存。

这种操作系统级交互会引发内核态上下文切换和驱动开销，因此相较纯设备端操作而言相对较慢。为此，建议使用异步版本 cudaMallocAsync 和 cudaFreeAsync，以在 GPU 上实现更高效的内存分配。

默认情况下，CUDA 运行时维护一个全局的 GPU 内存池。当你异步释放内存时，它会回到内存池中，供后续分配复用。cudaMallocAsync 和 cudaFreeAsync 在底层就使用了 CUDA 内存池（memory pool）。

内存池会回收已释放的内存缓冲区，避免为分配新内存而反复进行操作系统调用。例如，在一个长时间运行的训练循环中，它通过复用先前已释放的块（而不是每次迭代都新建），有助于随时间推移减少内存碎片化。许多高性能库和运行时（如 PyTorch）默认启用了内存池。

事实上，PyTorch 使用一个自定义的内存缓存分配器（caching allocator），通过 PYTORCH_ALLOC_CONF（旧称 PYTORCH_CUDA_ALLOC_CONF）进行配置。PyTorch 的内存缓存分配器在思路上与 CUDA 的内存池类似：它复用 GPU 内存，避免在——例如长时间运行的训练循环的每次迭代中——为每个新创建的 PyTorch 张量都调用同步的 cudaMalloc 操作所带来的开销。

在需要频繁进行细粒度分配的 CUDA 应用中，使用基于内存池的异步例程——cudaMallocAsync 和 cudaFreeAsync——要比使用传统的同步 cudaMalloc/cudaFree 高效得多，后者会引发整设备同步，甚至操作系统级调用。要使用流序（stream-ordered）分配，先创建一个非阻塞流：

```
cudaStream_t stream1;
cudaStreamCreateWithFlags(&stream1, cudaStreamNonBlocking);
```

> 使用显式 CUDA 流是重叠传输、核函数与内存操作的最佳实践。可以把每个流看作一个隔离的通道，它在自身的操作之间强制保持顺序。此外，建议用 cudaStreamCreateWithFlags(..., cudaStreamNonBlocking) 创建非阻塞流，以避免遗留的默认流屏障。我们将在第 11 章更详细地探讨多流重叠技术与最佳实践。

然后，每当你需要一个包含 N 个 float 的缓冲区时，就在该流上使用 cudaMallocAsync 和 cudaFreeAsync 进行分配与释放，如下所示：

```
float* d_buf = nullptr;
cudaMallocAsync(&d_buf, N * sizeof(float), stream1);

// ... launch kernels into stream1 that use d_buf ...
myKernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_buf, N);

// Free is deferred until all work in stream1 completes—
cudaFreeAsync(d_buf, stream1);
```

这些 API 从每个设备的内存池中分配，但会尊重你传入的流的顺序，因此释放会被推迟，直到该流的工作完成。而且由于 cudaFreeAsync 只等待 stream1 完成，所以既不需要开销高昂的全局 cudaDeviceSynchronize，也不会与其他流发生隐式同步。其结果是：当你的代码发起成千上万——甚至数百万——次分配/释放循环时，分配开销大幅降低，同时减少碎片化并平滑延迟尖峰。总体而言，相较传统的 cudaMalloc 和 cudaFree，这种模式减少了全局同步与碎片化。

你还可以进一步调节从设备内存池进行的流序分配的行为——例如，设置 cudaMemPoolAttrReleaseThreshold，提示内存池在尝试释放之前应保留多少预留内存。你也可以使用 cudaMemPoolTrimTo 主动归还内存。这些手段有助于在 GPU 总内存占用与碎片化之间取得平衡。

对于简单的一次性缓冲区，阻塞式的 cudaMalloc 和 cudaFree 或许就够用了。然而在更复杂、长时间运行、反复分配和释放内存的循环中，改用专用流上的 cudaMallocAsync 和 cudaFreeAsync 并利用其内存池，将带来更一致的性能和更高的吞吐量。

改用专用流上的 cudaMallocAsync 和 cudaFreeAsync 并利用其内存池，将带来更一致的性能和更高的吞吐量。你还可以用 cudaMemPoolSetAttribute 进一步调节内存池行为（例如调整 cudaMemPoolAttrReleaseThreshold），以 _调节释放阈值，在最小内存占用与低碎片化之间取得恰当的权衡。_

### 理解 GPU 内存层级

到目前为止，我们一直在较高层面上宽泛地讨论内存分配，且通常是从全局内存进行分配。这些分配来自某个流的内存池——包括默认的流 0 内存池。

然而实际上，GPU 提供了一个多级内存层级（memory hierarchy），有助于在容量与速度之间取得平衡。该层级包括寄存器、共享内存、缓存、全局内存，以及 Blackwell 及更新 GPU 上专用的 TMEM。TMEM（稍后详述）是一块专用的每 SM 约 256 KB 的片上内存，供 Blackwell 第五代 Tensor Core 指令（tcgen05.\*）使用。它无法从 CUDA C++ 中直接以指针寻址。相反，数据搬运由 TMA 硬件（global memory ↔ SMEM）以及 tcgen05 Tensor Core 数据搬运指令（SMEM ↔ TMEM，隐式地使用张量描述符）来编排。

全局内存（HBM 或 DRAM）容量大、位于片外，且相对较慢。寄存器很小、位于片上，且极快。L1 缓存、L2 缓存与共享内存则介于两者之间。缓存和共享内存的好处在于，它们能隐藏访问大容量片外内存存储的相对较长的延迟。GPU 内存层级（含 CPU）的高层视图如图 6-10 所示。

![图 6-10. GPU 内存层级（含 CPU）](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-10.png)

TMEM 是一块专用的每 SM 256 KB 缓冲区，以每秒数十太字节的带宽透明地与 Tensor Core 通信。这减少了 Tensor Core 对全局内存的依赖。图 6-11 展示了 TMEM 与 SMEM 一起为 Tensor Core 提供服务，以计算 C = A × B 矩阵乘法。

![图 6-11. TMEM 与 SMEM 为 Tensor Core 提供服务以计算 C = A × B 矩阵乘法](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-11.png)

在这里，操作数 B 来自 SMEM。操作数 A 位于 TMEM（不过它也可能位于 SMEM）。累加器同样位于 TMEM。分块（tile）通过 TMA（例如 cuda::memcpy_async）从全局内存经 L2 缓存流向 SMEM。

操作数在 SMEM 与 TMEM 之间的移动，则是通过诸如 unified matrix-multiply-accumulate（UMMA）和 tcgen05.mma 之类的 Tensor Core 指令隐式完成的。

表 6-5 展示了 Blackwell GPU 各级内存及其特性。随后对内存层级的每一级进行说明。

_表 6-5. Blackwell 内存层级及其特性_

| 级别                    | 作用范围                | 容量                                                                        | 延迟                                                                                                                                                       | 带宽（近似）                      |
| ----------------------- | ----------------------- | --------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| 寄存器                  | 每线程（位于 SM 上）    | 每 SM 64 K 个 32-bit 寄存器（每线程最多 255 个）                            | 接近寄存器延迟（寄存器读写为单周期，基本零开销）                                                                                                           | 每 SM 数十 TB/s（寄存器堆端口）   |
| 共享内存与 L1 缓存      | 每 SM                   | 228 KB（227 KB usable）共享内存 + 其余作为 L1/数据缓存                      | ~20–30 周期（L1/共享内存基准）                                                                                                                             | 每 SM TB/s 级（无 bank 冲突时）   |
| TMEM                    | 每 SM                   | 每 SM 256 KB SRAM，专供 Tensor Core                                         | ~10 周期（SM 上的专用 SRAM）                                                                                                                               | 与 Tensor Core 之间 TB/s 级的通信 |
| 常量内存缓存            | 每 SM                   | 约 8 KB 缓存，服务于 64 KB 的 **constant** 空间                             | ~1 周期（warp 广播）。命中缓存且一个 warp 内所有线程访问同一地址时，凭借常量缓存与广播行为，速度可快如寄存器。发生分歧或未命中时则会串行化，或产生更高延迟 | TB/s 级（广播吞吐）               |
| L2 缓存                 | 全 GPU 范围（所有 SM）  | 共 126 MB                                                                   | ~200 周期                                                                                                                                                  | 多 TB/s 总带宽                    |
| 本地内存                | 每线程（溢出到 DRAM）   | 近乎无限（由全局内存支撑）                                                  | 100s → 1,000 周期（类 DRAM）                                                                                                                               | ~8 TB/s（HBM3e）                  |
| 全局内存（HBM 或 DRAM） | 全设备范围（片外 DRAM） | 每块 Blackwell B200 GPU 最高 180 GB（每块 Blackwell B300 GPU 最高 ~288 GB） | 100s → 1,000 周期（全局内存延迟）                                                                                                                          | 总计 ~8 TB/s                      |

由此你可以看出，为什么最大化在寄存器、共享内存以及 L1/L2 缓存中的数据复用——并最小化对全局内存和本地内存（由全局内存支撑）的依赖——对于高吞吐的 GPU 核函数至关重要。下面对该层级的每一级再作一些说明：

_寄存器_

在 Blackwell 上，每个线程的旅程都从寄存器堆（register file）开始，这是一块微小的 SM 上 SRAM 阵列，用于保存每个线程的局部变量，且几乎不增加延迟。每个 SM 拥有 64 K 个 32-bit 寄存器（共 256 KB），但硬件对每个线程最多只暴露 255 个寄存器。

由于读写都在单个周期内完成，且几乎不与其他任何东西争用，寄存器带宽每 SM 可达每秒数十太字节。然而，如果你的核函数需要更多寄存器——无论是因为大量线程局部变量还是编译器临时变量——溢出部分就会溢入本地内存（映射到片外 DRAM），并带来数百乃至上千周期的延迟。这块本地内存如图 6-12 所示。

![图 6-12. 每线程的本地内存](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-12.png)

_共享内存与 L1 数据缓存_

再上一层是统一的 L1/数据缓存与共享内存块。这是每 SM 256 KB 的 SM 上 SRAM，你可以在用户管理的共享内存（每块最多 228 KB）之间动态划分它——在 Blackwell 这类具有统一 L1/纹理/共享内存的架构上，使用 cudaFuncSetAttribute() 配合 cudaFuncAttributePreferredSharedMemoryCarveout 来选择内存划分（carveout）。每块最大动态共享内存为 227 KB（CUDA 为每块保留 1KB），且每 SM 可分配的共享内存总量也受此上限约束。

这里的访问大约耗费 20–30 周期，但如果你设计线程块时避免了 bank 冲突，就能达到每秒太字节级的吞吐。线程块共享内存如图 6-13 所示。

![图 6-13. 线程块共享内存](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-13.png)

_TMEM_

TMEM 是每 SM 的一块专用片上内存（Blackwell 上为 256 KB），供 Tensor Core 专属的运算与指令使用，包括 unified matrix-multiply-accumulate 和 tcgen05（详见第 10 章）。它不是 CUDA C++ 中普通的指针可寻址空间。相反，传输由 Tensor Memory Accelerator（TMA）借助描述符来编排。这使开发者无需手动管理与 Tensor Core 之间的数据流。例如，某些算术操作数驻留在共享内存中，而累加器则驻留在 TMEM 中。随后由 TMA 负责在全局内存、共享内存与 TMEM 内存之间搬运数据以完成计算。

_常量内存缓存_

对于微小的只读表，Blackwell 提供了每 SM 约 8 KB 的常量内存缓存，位于 64 KB 的 **constant** 空间之前。当一个 warp 中全部 32 个线程加载同一地址时，该缓存会在单个周期内广播该值。

分歧的读取则会跨 lane 串行化。它非常适合共享小型查找表，例如旋转位置编码、带线性偏置的注意力（Attention with Linear Biases，ALiBi）斜率、LayerNorm 的 γ/β 向量以及嵌入量化缩放因子。这些都能在每个线程间共享，而无需产生全局内存流量。

_L2 缓存_

在片上 SRAM 之外是 L2 缓存，这是一块 126 MB 的全 GPU 范围缓冲区，把所有 SM 与片外 HBM3e 粘合在一起。延迟接近 200 周期，聚合带宽达每秒数十太字节，L2 吸收来自 L1 的溢出。

有了 L2，数据可以由一个线程块取回，再由其他线程块复用，而无需再次访问 DRAM。为了最大化 L2 的收益，应把你的全局加载组织成 128 字节的合并访问事务，使其干净地映射到缓存行上。稍后我们会展示具体做法。

> 把你的全局加载组织成 128 字节对齐、合并的分段，使其干净地映射到缓存行上。这可以避免拆分事务，并最大化利用 L2 与 DRAM 带宽。

_全局内存（HBM 或 DRAM）_

全局内存层级——本地溢出空间与 HBM——位于片外。任何溢出的寄存器或超大的自动数组都驻留在本地内存中，尽管 HBM3e 带宽高达 ~8 TB/s，仍要付出完整的 DRAM 延迟（数百乃至一千多周期）。

对于 Blackwell，HBM3e 层级提供最高 180 GB 的全设备范围存储，总带宽 ~8 TB/s。然而其高延迟使它成为整条链路中最慢的一环。每设备的全局内存如图 6-14 所示。

![图 6-14. 每设备的全局内存，即 HBM](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-14.png)

借助 Nsight Compute 之类的工具跟踪溢出与缓存命中率，你可以让核函数尽可能贴近该内存层级的片上峰值运行。这些工具能帮助你有效地在寄存器、shared/L1、常量缓存与 L2 缓存之间编排数据。像 Blackwell 这样的现代 GPU 允许核函数开发者利用内存层级——使用 L2 缓存和统一的 L1/共享内存来缓冲并合并对 HBM 的访问，我们很快就会看到。

> Blackwell B200 对外呈现为一块单一 GPU，构建于统一的全局地址空间之上。然而，它由两个光罩受限裸片（reticle-limited die）组成，二者通过 10 TB/s 的芯片间互连连接。每个裸片连接到四个 HBM3e 堆栈，共计八个 HBM3e 堆栈。不过，从开发者的视角看，HBM 内存访问在这一合并地址空间中是一致的，但了解这一架构的底层细节仍然是值得的。

内存层级中各级的一致性点（point of coherency，PoC）取决于你的需求，以及线程进行通信所处的层级。它通常发生在以下层级：线程、线程块（又称 _CTA_）、线程块簇（又称 _CTA 簇_）、设备或系统，如图 6-15 所示。

![图 6-15. GPU 线程的内存一致性点](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-15.png)

总而言之，理解 GPU 的内存层级并恰当地针对每一级进行优化非常重要。这样做，你就能构建 CUDA 核函数以最大化数据局部性、隐藏内存访问延迟、提高占用率，并充分发挥 Blackwell 强大的并行计算能力，稍后我们会加以探讨。首先，让我们讨论 NVIDIA 的统一内存——鉴于 Grace Hopper 与 Grace Blackwell 的 CPU-GPU 统一超级芯片设计，理解它十分重要。

### 统一内存

统一内存（Unified Memory，又称 CUDA 托管内存（Managed Memory））为你提供一个横跨 CPU 与 GPU 的单一、一致的地址空间，因此你不再需要费心管理分开的主机与设备缓冲区，也不必发起显式的 cudaMemcpy 调用。在底层，CUDA 运行时为每一次 cudaMallocManaged() 分配都以页面作为支撑，这些页面能够在连接你的 CPU 与 GPU 的任何互连上按需迁移，如图 6-16 所示。

尽管访问统一内存对开发者极为友好，它却可能引发 CPU 与 GPU 之间不期而至的按需页迁移。这会带来隐藏的延迟与执行停顿。例如，如果一个 GPU 线程访问当前驻留在 CPU 内存中的数据，GPU 就会发生缺页，并在数据经由 NVLink-C2C 互连传输期间等待。统一内存的性能在很大程度上取决于底层硬件。

在传统 PCIe 或早期 NVLink 系统上，这类迁移以相对较低的带宽进行——常常使缺页触发的传输比手动 cudaMemcpy 还慢。但在 Grace Hopper 与 Grace Blackwell Superchip 上，NVLink-C2C 结构在 CPU 的 HBM 与 GPU 的 HBM3e 之间提供最高 ~900 GB/s 的带宽。因此，缺页驱动的迁移在速度上要接近设备原生水平得多——尽管它们仍带有非零的延迟。

尽管如此，核函数启动期间任何意外的缺页都会在运行时把所需页面搬到位的过程中使 GPU 停顿。为避免这些“意外”停顿，你可以事先用 cudaMemPrefetchAsync() 预取内存，如图 6-17 所示。

这会提示驱动在你启动核函数之前，把指定范围搬到目标 GPU（或 CPU）上，从而把代价高昂的首次访问（first-touch）迁移转化为可重叠的异步传输。你还可以给出内存建议（memory advice），如下面的代码所示：

```
cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, gpuId);
cudaMemAdvise(ptr, size, cudaMemAdviseSetReadMostly, gpuId);
```

在这里，你可以用 PreferredLocation 告诉驱动你将主要在何处使用数据，用 ReadMostly 表示数据在很大程度上是只读的，如图 6-18 所示。

![图 6-16. CPU-GPU 统一内存的自动页迁移](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-16.png)

![图 6-17. 用 cudaMemPrefetchAsync() 通过 NVLink-C2C 将数据从 CPU 流式传输到 GPU](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-17.png)

![图 6-18. 指定“首选位置”以告诉 CUDA 驱动数据的主要用途（例如，对基本只读的工作负载使用 ReadMostly）](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-18.png)

你还可以调用下面的代码，让第二块 GPU 映射这些页面，而不会在启动时触发迁移：

```
cudaMemAdvise(ptr, size, cudaMemAdviseSetAccessedBy,
   otherGpuId);
```

默认情况下，任何 CUDA 流或设备核函数都可能在一次托管分配上触发缺页。这会导致意外的迁移和隐式同步。如果你知道某个缓冲区在同一时间只会在一个流/GPU 上使用，把它附着到那个流，就能让迁移与其他流中的操作相重叠。调用下面的代码可将该内存范围绑定到指定的流：

```
cudaStreamAttachMemAsync(stream, ptr, 0,
    cudaMemAttachSingle);
```

在这种情况下，只有该流中的操作才会缺页并迁移其页面。这可以防止其他流意外地在它上面停顿。因此，把某个范围附着到特定的流会推迟其迁移，使之只与该流的工作相重叠。这就避免了跨流同步。

> 在没有 NVLink-C2C 的多 GPU 系统中，你还可以使用 cudaMemcpyPeerAsync()，或预取到某个特定设备，以把数据固定在最近的 NUMA 本地 GPU 内存中，从而避免缓慢的远程访问。

简而言之，显式预取托管内存并提供内存建议，能够消除统一内存带来的大多数“意外”停顿。数据在核函数运行时已经就位，而不是让 GPU 暂停以按需取数。

借助主动预取、有针对性的内存建议以及流附着等技术，统一内存可以在保留统一地址空间之简洁性的同时，交付非常接近手动 cudaMemcpy 的性能。

### 维持高占用率与 GPU 利用率

GPU 通过并发运行大量 warp 来维持性能，这样当一个 warp 因等待数据而停顿时，另一个 warp 就能运行。这种在 warp 之间快速切换的能力使 GPU 得以隐藏内存延迟。正如我们前面所述，一个 SM 的容量中被活跃 warp 实际占据的比例，被称为 _占用率_。

如果占用率很低（只有寥寥几个活跃 warp），当一个 warp 在等待内存时，SM 可能就会空闲。这会导致 SM 利用率低下。在 Blackwell 上，凭借其庞大的寄存器堆（每 SM 64K 个寄存器），实现高占用率会容易一些，它可以在不溢出的情况下支撑许多 warp。

> 正如你前面所见，一个 warp 中的每个线程最多可使用 255 个寄存器。务必使用你的剖析工具来检查实际占用率——并据此调整核函数的块大小和寄存器用量。

反过来，高占用率（每 SM 有许多活跃 warp）会让 GPU 计算单元保持繁忙，因为当一个 warp 在等待内存访问时，其他 warp 会切入 SM 并执行。这就掩盖了漫长的内存访问延迟。这通常被称为 _隐藏延迟_。

让我们展示一个能提升占用率、并最终提升 GPU 利用率、吞吐量与整体核函数性能的示例。这是 CUDA 性能优化最基本的规则之一：启动足够的并行工作以充分占满 GPU。

如果你的实际占用率（正在使用的硬件线程槽的比例）远低于 GPU 的上限且性能不佳，首要的补救办法就是增加并行度——使用更多的块或线程，使占用率在现代 GPU 上接近 80%–100% 的区间。

反之，如果占用率已经处于中到高的水平，但核函数受内存吞吐所限，把它推到 100% 可能也无济于事。你通常只需要恰好足够多的 warp 来隐藏延迟，超过之后瓶颈可能就在别处了（例如内存带宽）。

为说明占用率的影响，考虑一个非常简单的操作：把两个长度为 _N_ 的向量相加（计算 C = A + B）。我们将考察两种核函数实现：addSequential 和 addParallel。addSequential 使用单个线程（或单个 warp）在循环中把全部 _N_ 个元素相加。addParallel 使用许多线程，使加法在整个数组上并发完成。

在串行版本中，一个 GPU 线程串行地处理整个工作负载，如下所示：

```
#include <cuda_runtime.h>

const int N = 1'000'000;

// Single thread does all N additions
__global__ void addSequential(const float* A,
                              const float* B,
                                    float* C,
                              int N)
{
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < N; ++i) {
            C[i] = A[i] + B[i];
        }
    }
}

int main()
{
    // Allocate and initialize host
    float* h_A = nullptr;
    float* h_B = nullptr;
    float* h_C = nullptr;
    cudaMallocHost(&h_A, N * sizeof(float));
    cudaMallocHost(&h_B, N * sizeof(float));
    cudaMallocHost(&h_C, N * sizeof(float));

    for (int i = 0; i < N; ++i) {
        h_A[i] = float(i);
        h_B[i] = float(i * 2);
    }

    // Allocate device
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, N * sizeof(float));

    // Copy inputs to device
    cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch: one thread
    // Note: This kernel assumes <<<1,1>>>
    // (one block, one thread).
    // Do not change the launch config when running this example.
    addSequential<<<1,1>>>(d_A, d_B, d_C, N);

    // Ensure completion before exit
    cudaDeviceSynchronize();

    // Copy d_C => h_C (back to host)
    cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
```

在这个单线程版本中，GPU 庞大的资源大多处于闲置状态。只有一个 warp，甚至只有该 warp 中的一个线程在工作，而其他所有线程都闲着。结果是占用率极低，并最终导致性能低下。

还必须小心，避免在 PyTorch 这类高级库和框架中间接执行低效的 GPU 代码。例如，下面这段朴素的 PyTorch 代码错误地用一个 Python for 循环执行逐元素操作，一个接一个地在 GPU 上发起了 _N_ 次独立的加法操作：

```
import torch

N = 1_000_000
A = torch.arange(N, dtype=torch.float32, device='cuda')
B = 2 * A
C = torch.empty_like(A)

# Ensure all previous work is done
torch.cuda.synchronize()

# Naive, Sequential GPU operations - DO NOT DO THIS
with torch.inference_mode(): # avoids unnecessary autograd graph construction
    # This launches N tiny GPU operations serially
    for i in range(N):
        C[i] = A[i] + B[i]

torch.cuda.synchronize()
```

这段代码实际上把 GPU 当成了一个标量的、非并行的处理器。它的占用率极低，与前面原生的 addSequential CUDA C++ 代码类似。

让我们优化这段 CUDA 核函数与 PyTorch 代码，实现向量加法操作的并行版本。图 6-19 展示了向量化加法操作的工作方式。

![图 6-19. 向量化加法在向量各元素间并行进行](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-19.png)

在下面的 CUDA C++ 代码中，我们启动足够多的线程以覆盖所有元素（<<< (N+255)/256, 256 >>>），使得每块 256 个线程在所需数量的块上并行处理 _N_ 个元素：

```
#include <cuda_runtime.h>

const int N = 1'000'000;

// One thread per element
__global__ void addParallel(const float* __restrict__ A,
                            const float* __restrict__ B,
                                  float* __restrict__ C,
                            int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }

}

int main()
{
  // Allocate and initialize host (pinned for faster DMA)
  float* h_A = nullptr;
  float* h_B = nullptr;
  float* h_C = nullptr;
  cudaMallocHost(&h_A, N * sizeof(float));
  cudaMallocHost(&h_B, N * sizeof(float));
  cudaMallocHost(&h_C, N * sizeof(float));
  for (int i = 0; i < N; ++i) { h_A[i] = float(i); h_B[i] = float(2*i); }

  // Create a non-blocking stream and allocate device buffers
  cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
  float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
  cudaMallocAsync(&d_A, N * sizeof(float), s);
  cudaMallocAsync(&d_B, N * sizeof(float), s);
  cudaMallocAsync(&d_C, N * sizeof(float), s);

  // Async HtoD copies on the same stream
  cudaMemcpyAsync(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice, s);
  cudaMemcpyAsync(d_B, h_B, N*sizeof(float), cudaMemcpyHostToDevice, s);

  // Launch (same stream)
  int threads = 256;
  int blocks  = (N + threads - 1) / threads;
  addParallel<<<blocks, threads, 0, s>>>(d_A, d_B, d_C, N);

  // Async DtoH copy and stream sync
  cudaMemcpyAsync(h_C, d_C, N*sizeof(float), cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);

  // Cleanup (stream-ordered free)
  cudaFreeAsync(d_A, s); cudaFreeAsync(d_B, s); cudaFreeAsync(d_C, s);
  cudaStreamDestroy(s);
  cudaFreeHost(h_A); cudaFreeHost(h_B); cudaFreeHost(h_C);
  return 0;
}
```

当 _N_ 足够大时，GPU 利用率的差异非常显著。现在让我们优化这段 PyTorch 代码——它启动一个单一的向量化核函数（A + B），像前面优化过的 addParallel CUDA C++ 示例那样，在 GPU 上并发调动许多线程。下面是 PyTorch 代码的并行版本：

```
# add_parallel.py
import torch

N = 1_000_000
A = torch.arange(N, dtype=torch.float32, device='cuda')
B = 2 * A

torch.cuda.synchronize()

# Proper parallel approach using vectorized operation
# Launches a single GPU kernel that adds all elements in parallel
C = A + B

torch.cuda.synchronize()
```

> 在实践中，当你使用向量化的张量操作时，PyTorch 这类高级框架会做出正确的处理。只需注意：在 GPU 操作外围引入 Python 层的循环会使工作串行化，并对性能产生负面影响。如有可能应加以避免。除非你在编写某种新颖的东西，否则几乎总能找到一个经过优化的 PyTorch 原生实现——包括 PyTorch 编译器生成的代码。

为了量化使用并行实现相较串行实现的性能影响，我们可以用 Nsight Systems 和 Nsight Compute 来测量两种方法的核函数总执行时间、GPU 利用率、占用率以及 warp 执行效率指标。下面是 Nsight Systems（nsys）和 Nsight Compute（ncu）命令：

```
# Sequential add
nsys profile \
  --stats=true \
  -t cuda,nvtx \
  -o sequential_nsys_report \
  ./add_sequential.py

ncu \
 --section SpeedOfLight \
 --metrics
     sm__warps_active.avg.pct_of_peak_sustained_active,gpu__time_duration.avg \
 --target-processes all \
 --print-summary per-gpu \
 -o sequential_ncu_report \
 ./add_sequential.py

# Parallel add
nsys profile \
  --stats=true \
  -t cuda,nvtx \
  -o parallel_nsys_report \
  ./add_parallel.py

ncu \
 --section SpeedOfLight \
 --metrics sm__warps_active.avg.pct_of_peak_sustained_active \
 --target-processes all \
 --print-summary per-gpu \

 -o parallel_ncu_report \
 ./add_parallel.py
```

我们用 nsys 来揭示时间花在了哪里，以及 GPU 是被“饿着”还是被阻塞。然后用 ncu 来解释核函数为何呈现这样的表现——也许是由于占用率低，等等。

如果你只运行 nsys，可能会错过细粒度的核函数低效之处。而如果你只运行 ncu，则无从得知你的核函数是否被足够快地喂入了数据。表 6-6 展示了统一的结果。

_表 6-6. 串行与并行 CUDA 核函数的对比_

| 指标                 | add_sequential | add_parallel |
| -------------------- | -------------- | ------------ |
| 核函数执行时间（ms） | 48.21          | 2.17         |
| GPU 利用率           | 1.5%           | 95%          |
| 实际占用率           | 1.3%           | 38.7%        |
| warp 执行效率        | 3.1%           | 100%         |

> 其他剖析工具可能对这些指标使用不同的标签。例如，Nsight Systems 报告的是整体的“GPU Utilization”，而 Nsight Compute 提供的是每核函数的“SM Active %”指标——但两者都反映了 GPU 的 SM 被活跃 warp 占据的充分程度。

正如预期，从单线程、单 warp 的实现转向完全并行、多 warp 的实现，将平均占用率从 1.3% 提升到了 ~38.7%。这将运行时间缩短了约 22×，从 48.21 ms 降至 2.17 ms。
