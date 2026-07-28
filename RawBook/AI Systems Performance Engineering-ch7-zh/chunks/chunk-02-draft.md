值得注意的是，在 Blackwell 与 CUDA 13 之前，全局内存向量加载在每个线程上被限制为 16 字节（128 位）。不过，Blackwell 与 CUDA 13 为具有 32 字节对齐的特定向量类型新增了 32 字节（256 位）的加载/存储指令与数据类型。

在可用时，对于用户自定义的 8 个 float 的聚合体，应优先使用这些更宽的 32 字节类型与指令。这样可以减少加载和存储更宽的 32 字节对齐数据所需的指令数量。

自定义的 8 个 float 聚合体仍会被编译成两条 16 字节加载，除非你显式使用映射到单条 32 字节指令的 32 字节对齐类型。

> 尽管 Blackwell 线程可以加载一个完整的 32 字节向量类型，内存合并器（memory coalescer）仍然只能以 128 字节为单位、也就是四个 32 字节扇区（sector）来处理请求。在 Blackwell 上，一个 32 线程的 warp 每线程搬运 32 B，共传输 1024 B（8 × 128 B 缓存行）。Hopper 的 16 B/线程 变体搬运 512 B（4 × 128 B）。注意事务数量随搬运的字节数一起增长。只要正确对齐，两者都是完全高效的：16 字节（Hopper）或 32 字节（Blackwell 及更新架构）。

## 使用共享内存进行分块与数据复用

一个常见的性能陷阱是反复从全局内存读取相同的数据。*分块*（tiling）是一种避免这种情况的技术：把数据块加载到更快的片上共享内存中——并让这些数据块在许多线程之间复用。

例如，一个 *N* × *N* 大小的朴素矩阵乘法可能会把矩阵 A 的每个元素从 HBM 加载 *N* 次，每次对应它要相乘的 B 的一行。这会导致每个元素有 N–1 次冗余加载。而在 Blackwell 上，它可以轻松执行数十 teraFLOPS（TFLOPS），冗余加载会浪费内存带宽，而这些带宽本可以为 GPU 的 SM 提供更多数学运算。

分块通过让每个线程块只把 A 和 B 的一个小子矩阵（一个*瓦片*，tile）拉入共享内存恰好一次，来消除这种浪费。随后它在所有线程之间复用这些被缓存的值，进行多次乘加运算。在下一个示例中，我们会使用 32 × 32 的瓦片，这是一个能很好地放入共享内存的常见选择。

一个线程块内的线程可以协作地把瓦片加载到共享内存中，然后调用 __syncthreads() 来同步数据。之后，线程们使用共享内存中的数据执行并行的矩阵乘法计算。这样就把全局内存访问的开销摊分到了许多线程和计算之上。值得注意的是，这些瓦片加载也被安排成合并的（coalesced）。具体来说，每个 warp 从全局内存把一个完整的 128 字节段加载到共享内存中——与前面的合并示例保持一致。

通过把每个元素从 DRAM 只读取一次（读入共享内存）并复用它进行许多次计算，我们减少了全局内存流量。让我们用一个 *N* × *N* 矩阵乘法示例来说明这一点。首先，考虑一个朴素实现。

*前例（CUDA C++）：朴素矩阵乘法*。每个线程计算结果矩阵 C 的一个元素，为每个输出从全局内存读取 A 的整行和 B 的整列：

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

这个 CUDA C++ 核函数在内层循环里为每一次乘法都发起全局内存加载。每个线程为每一个 *k* 从 DRAM 读取 A[row, k] 和 B[k, col]，造成海量的冗余流量。其结果是一个严重访存受限（memory-bound）的核函数，SM 利用率很低，并且频繁因等待全局内存而停顿。下面是朴素矩阵乘法的 PyTorch 版本：

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

这个 PyTorch 实现使用了嵌套的 Python 循环。虽然最内层的操作会作为逐元素乘法和求和操作卸载到 GPU 上，它在底层仍然会为每个点积触发重复的全局内存加载。这模拟了朴素 CUDA 核函数的访存受限行为，因为 GPU 把大部分周期都花在等待内存上，而不是计算乘法。

> 这段 PyTorch 代码故意写得极其缓慢，以此说明在循环内执行冗余全局内存访问加载的极端情形。在实践中，像 PyTorch 这样的框架会对这类操作使用优化过的核函数。

现在，让我们应用分块来改进它。我们把矩阵划分成 32 × 32 的瓦片。32 × 32 是一个方便的瓦片尺寸，因为它与 32 的 warp 大小对齐，能很好地放入共享内存，并且映射到一个完整的 32 线程 warp 来读取每一行。这让每个 warp 能够协作地一次加载并处理（每个瓦片的）单独一行。

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

在这个分块核函数中，每个线程块协作地从全局内存把 A 的一个 32 × 32 瓦片（加载到 sA）和 B 的一个 32 × 32 瓦片（加载到 sB）加载进来。这些加载发生在循环内部的前两行，随后是 __syncthreads()，以确保瓦片在使用前已完全加载。

然后，线程块使用这些共享内存瓦片执行 32 × 32 次乘加运算。这个以 32 为步长在 k 上运行的内层循环，把每个加载进来的值复用 32×，从而使这些元素的全局内存读取减少 32×。

在完成所有瓦片迭代后，每个线程把它的结果写入 C。其结果是每个线程的全局内存访问量大幅减少 8×，你稍后就会看到。

> 这些示例聚焦于内存访问模式，使用的是 FP32 CUDA 核心——而非低精度的 Tensor Core。第 9 章会演示使用低精度计算（例如 16 位、8 位、4 位）来进一步提升性能。

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

> 在 32 × 32 的瓦片下，当一个 warp 中的线程都访问同一列时，它们会争用同一个共享内存 bank。在实践中，你可以通过把瓦片填充到 33 列——或者使用像混洗（swizzling）这样的技术——来避免这一点，使得每次访问落在不同的 bank 上。我们会在下一节应用这项优化。目前，先让我们把分块本身作为一项优化来关注。

分块带来的性能影响是显著的。通过以瓦片方式组织计算并在共享内存中复用数据，我们减少了 DRAM 流量并提高了算术强度（arithmetic intensity），因为从内存取回的每个字节都被用于更多的浮点运算。除了更少的全局内存事务，我们还观察到更高的实测占用率（occupancy）。表 7-4 比较了应用共享内存分块前后的关键指标。

*表 7-4. 使用共享内存分块带来的性能提升*

| 指标 | 之前（naive kernel） | 之后（tiled kernel） | 备注 |
| --- | --- | --- | --- |
| DRAM 吞吐（占峰值百分比） | 90% | 25%（减少 3.6×） | 朴素版本使用了更多带宽，但很浪费。分块实现的值更低，是因为分块后由于冗余加载更少而更高效。 |
| 实测占用率（%） | 42% | 89% | 更多驻留 warp 能以更少的停顿向前推进。 |
| 浮点吞吐（GFLOPS） | 15 GFLOPS | 170 GFLOPS | 较小的矩阵尺寸限制了其绝对值。复用提升了持续算力。（绝对 GFLOPS 数值偏低是因为矩阵尺寸（N = 1,024）较小且核函数受内存限制。） |
| 全局内存加载扇区数 | 9800 | 1200 | 每个元素每个瓦片只被取一次，而不是取 32 次（每线程一次）。这是核函数发起的 32 字节扇区读取的总数。这一下降反映了得益于共享内存分块而消除了冗余加载。 |
| 共享内存吞吐（占峰值百分比） | 52% | 99% | 分块核函数的访问模式避免了对 sA 和 sB 的内存 bank 冲突。 |

把 32 × 32 的瓦片暂存到共享内存中，确保了每个元素从 DRAM 取一次并在一个线程块内的所有线程之间复用。DRAM 吞吐从 90% 降到 25%，大约减少了 3.6×，这在移除冗余流量后是符合预期的。

DRAM 吞吐的下降是我们想要的，因为核函数现在每搬运一个字节所完成的工作更多了。算术强度上升，持续浮点吞吐从 15 GFLOPS 升到 170 GFLOPS，提升近 11×。

> 优化后的 170 GFLOPS 远低于 Blackwell 接近 80 TFLOPS 的理论 FP32 峰值。这对于小瓦片和频繁的内存访问来说是符合预期的。重要的结果是消除内存瓶颈后带来的 11× 提升。更大的问题规模或每字节更多的计算量会让性能更接近峰值。

这种算术强度上的转变把核函数从访存受限推向计算受限（compute-bound），这是理想的，因为性能此时受限于充裕的浮点吞吐，而不是相对稀缺的芯片带宽。全局内存加载扇区数从 9,800 大幅下降到 1,200，因为每个元素只被取一次到共享内存中并在线程之间复用。这消除了冗余加载。分块核函数还以一种避免共享内存 bank 冲突（bank conflict）的方式访问 sA 和 sB，这就是共享内存吞吐接近 100% 的原因。

总的来说，我们把这个核函数从访存受限转变为了计算受限，而这正是目标所在。我们成功缓解了内存压力，为其他有用工作腾出了内存总线，并实现了更高的算力吞吐。

我们还看到实测占用率百分比从 42% 大幅提升到 89%。这个指标几乎翻倍，是因为分块核函数持续允许更多驻留 warp 以更少的停顿向前推进。因此，SM 更持续地保持繁忙。

通过引入共享内存分块，我们在不让每线程资源使用超过可用寄存器和共享内存的前提下，增加了每线程的工作量。这帮助我们的核函数达到了更高的占用率和利用率，这在像 Blackwell 这样的 GPU 上很重要——它提供了非常大的寄存器文件（每个 SM 有 65,536 个 32 位寄存器），线程可以在不发生溢出的情况下利用到每线程 255 个寄存器的上限。

> 请回想，每个线程最多可以使用 255 个寄存器。我们的分块核函数实现保持在这个上限之内，避免了寄存器溢出（register spilling），并保持了高性能。

我们可以通过考察朴素核函数和分块核函数的持续 FLOP 速率以及平均 DRAM 带宽，来比较它们的算术强度。朴素版本以 15 GFLOPS 运行，同时平均搬运 10 GB/s 的数据，即每字节 1.5 FLOPS。相比之下，分块实现持续达到 170 GFLOPS，同时平均搬运 21 GB/s，即 8 FLOPS/字节。

也许有人会想把瓦片尺寸进一步增大到 64 × 64 或 128 × 128，以便进一步减少内存流量并增加数据复用。但要记住，更大的瓦片会消耗更多片上资源，包括寄存器和共享内存。这会为每个 SM 上的额外线程块留下更少的容量。

例如，Blackwell GPU 每个 SM 提供多达 228 KB 的可分配共享内存。这可以轻松容纳一个由 4 字节 float 构成的 64 × 64 瓦片，其中每个 A 瓦片和 B 瓦片各需要 16 KB（16,384 字节 = 64 × 64 个 float × 每个 float 4 字节）。然而，除了共享内存的使用之外，你还需要为每线程的额外寄存器留出预算。

每个 Blackwell SM 总共提供多达 64K 个寄存器（每个 32 位），每线程最大为 255 个寄存器。当你启动一个 32 × 32 的线程块（1,024 个线程）来计算一个 64 × 64 的瓦片时，每个线程处理一个 2 × 2 的子瓦片，产生 4 个输出。

对于 64 × 64 的瓦片尺寸，你需要如下寄存器：四个累加器寄存器（每个输出一个）、两个用于 A 瓦片元素的寄存器（在两个输出间复用）、两个用于 B 瓦片元素的寄存器（同样复用），以及约 4 个用于循环计数器、线程索引和地址运算的寄存器。总计约为每线程 12 个寄存器。

每线程 12 个寄存器 × 每块 1,024 个线程 = 每个线程块需要 12,288 个寄存器。由于每个 Blackwell SM 有 65,536 个 32 位寄存器（每线程最多 255 个寄存器），理论上可以容纳多达 5 个线程块所需的寄存器（≈61,440 个寄存器）。然而，SM 只能支持 2,048 个并发线程（64 个 warp）。所以在实践中，占用率受寄存器、共享内存、warp 和线程这几者中较小的限制所约束。对于这种配置，每个 SM 两个线程块就会占满 2,048 线程的上限。

> 如果你的核函数为双缓冲（double buffering）寄存器、向量化加载等使用了额外的寄存器，你可以用 cudaOccupancyMaxPotentialBlockSize 或 Nsight Compute 的占用率报告重新计算占用率。

对于 64 × 64 的瓦片尺寸，每线程需要 22 个寄存器，一个 32 × 32 线程块（1,024 个线程）总共需要 22,528 个寄存器。在这种情况下，每个 Blackwell SM 最多只能容纳两个线程块。这意味着最大占用率只有 2,048 个线程——只用两个线程块就撞上了 SM 的线程上限（每个 SM 最多 16 个并发线程块中的两个）。

使用更大的瓦片尺寸导致并发线程块减少，如果你超出了硬件的限制，这会降低占用率并损害性能。在实践中，你的瓦片维度必须能放入 GPU 的共享内存和寄存器预算内，以维持高占用率和吞吐。

对于每个 SM 有约 228 KB 可分配共享内存的 Blackwell GPU，一个 64 × 64 的瓦片（每个输入矩阵瓦片约 16 KB）也许仍然放得下，但把瓦片维度翻倍会让复用系数平方增长，同时让共享内存使用量翻两番。这里存在收益递减以及可能的权衡。

建议尝试不同的瓦片维度，以在片上复用与资源限制之间取得平衡。在现代 NVIDIA GPU 上，32 × 32 的瓦片是一个可靠的起点，但取决于你的共享内存和寄存器使用情况，你可能会发现稍小或稍大的瓦片能带来更好的吞吐。

> 像 CUTLASS 这样的库也包含能自动化这种探索的剖析器（profiler），让你为你的核函数和硬件找到最优的瓦片尺寸。

简而言之，我们把一个仅使用全局内存的朴素矩阵乘法转变成了使用共享内存的分块实现。这实现了协作式数据复用，减少了 DRAM 事务的数量，并提升了算术强度。CUDA C++ 和 PyTorch 两种实现都从这项分块技术中获益。

值得注意的是，我们手动应用的分块技术，正是高性能 GPU 库在底层所做的事情。举例来说，NVIDIA 的 CUTLASS 库提供了模板化组件，用多层分块来实现通用矩阵乘法（GEMM）。这些 CUTLASS 组件把矩阵的片段加载到寄存器和共享内存中——然后计算部分结果，很像我们前面 32 × 32 瓦片的示例。

事实上，NVIDIA 优化过的 cuBLAS 和 cuDNN 库在线程、warp 和线程块层面使用了类似的分块策略，以达到接近峰值的吞吐。NVIDIA 甚至在 2025 年初宣布了一个 Python 优先的 API，称为 cuTile，让程序员能以更方便的 Python 风格描述这些瓦片形状。事实上，NVIDIA 已经开发了一种基于瓦片的中间表示（intermediate representation，IR），称为 TileIR，来支持 cuTile 并促进自动编译与调优。

其他高性能库同样封装了这些分块模式。例如，NVIDIA 的 CUTLASS C++ 和 Python 库暴露了模板化的瓦片迭代器与剖析器。而像 TorchInductor（使用 OpenAI 的 Triton 库）这样基于 PyTorch 的编译器，会在形状和对齐允许时自动生成分块核函数。这些库降低了使用这些分块优化的门槛——并减少了样板代码的数量。

关键思想在于：在返回 DRAM 之前尽可能多地在寄存器/共享内存中复用数据，这是根本性的，而各种库正封装了这一点。所以只要有可能，就利用（或参考）这些高度优化的库来获得高性能的分块模式。

例如，如果你在 PyTorch 中使用 torch.mm，或在代码中使用 cublasSgemm，其底层做的正是这种分块与内存合并。这就是为什么我们的 PyTorch 示例会自动获得同样的收益。

在实践中，你会使用像 cuBLAS 和 PyTorch 的 torch.matmul 这样的高性能库，它们已经用 C++ 实现了分块和其他优化。在生产代码中，直接使用 torch.mm 或 torch.matmul 会产生同样的收益——甚至更多，这要归功于高度调优的核函数。

> 虽然你完全可以复用现有的分块库和框架，但像我们这里所做的那样理解它们的工作原理是非常宝贵的——当你需要诊断性能问题，或者可能需要为这些库和框架未覆盖的专门场景编写自己的自定义核函数时尤其如此。只是别忘了回馈社区，因为他们给了你很多！

正如本节前面所提到的，在使用 32 × 32 的瓦片时，当一个 warp 中的线程访问同一列时，它们会争用同一个共享内存 bank。让我们在下一节解释这个问题——以及一些优化方法。

## 避免共享内存 bank 冲突

在包括 Blackwell 在内的现代 NVIDIA GPU 上，共享内存有 32 个 bank，bank 宽度为 4 字节（即地址按 mod 32 映射）。因此，一个以 128 B（32 个 float × 4 B）为步长的 warp 会把所有线程映射到同一个 bank。如果一个 warp 中的多个线程访问同一个 bank，就会发生 *bank 冲突*（bank conflict）。这会迫使内存访问串行化，从而抵消掉共享内存的速度优势。

在代码中，bank 冲突常常发生在线程以某个步长访问共享内存数组、导致它们落入同一个 bank 时。图 7-4 展示了两个无冲突内存 bank 访问的示例。

![图 7-4. 无 bank 冲突](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-4.png)

这里，没有两个线程并发访问同一个内存 bank。这是理想情况。图 7-5 展示了 2 路和 16 路 bank 冲突的示例。

![图 7-5. 2 路和 16 路 bank 冲突](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-5.png)

这里，多个线程正在访问同一个内存 bank，这会造成冲突并影响性能。bank 冲突的一个经典例子是使用 32 × 32 共享内存瓦片的朴素矩阵转置。如果 32 个线程各自读取 tile[i][threadIdx.x]，使得同一个列索引（threadIdx.x）在不同行（i）之间被读取，那么这个 warp 中的全部 32 个线程各自访问的内存地址都落在同一个共享内存 bank 中，造成 32 路 bank 冲突。

具体来说，在矩阵转置期间，你是在通过固定列索引（threadIdx.x）、在线程间改变行索引（i）来沿着一个行主序瓦片的同一列向下读取。而由于每一行在内存中相距 128 字节（32 列 × 每列 4 字节 = 128 字节），被访问的内存地址会恰好相差 128 字节的整数倍。

请记住，由于有 32 个 bank、bank 宽度为 4 字节（每次访问），地址到 bank 的映射每 128 字节重复一次。因此，访问相差 128 字节的内存地址总会落回 bank 0——于是产生了完整的 32 路 bank 冲突。

> 有一个值得注意的例外：如果一个 warp 中的全部 32 个线程访问同一个内存 bank 中的完全相同的地址，硬件会在单个周期内把该值广播（broadcast）给所有线程。这就避免了 bank 冲突。任何其他情形——即两个或更多不同的内存地址访问同一个 bank——都会造成 bank 冲突并使内存访问串行化。

另一个常见陷阱是使用等于内存 bank 数量（这里是 32）的步长。例如，如果你以恰好 32 个 float（每个 4 字节）为步长来访问索引，那么每个线程的地址最终会相差 128 字节的倍数。在这种情况下，所有线程都映射到 bank 0，如以下代码所示：

```
// Allocate a shared buffer big enough for several warps (warpCount)
__shared__ float arr[32 * warpCount];

// Each thread reads from arr[threadIdx.x * 32]
float x = arr[threadIdx.x * 32];
```

这里，以 float 为单位的 threadIdx.x * 32 变成了 (threadIdx.x * 32 * 4) 字节。因为 32 * 4 = 128，所以每个线程的内存加载地址是 threadIdx.x * 128 字节。而对所有线程来说，threadIdx.x * 128 mod 128 = 0。它们全都同时命中 bank 0，造成 32 路 bank 冲突。

当这种情况发生时，硬件必须把本应是 32 次并行读取的操作串行化为一系列单 bank 访问。在 Nsight Compute（Shared Memory 部分）中，你会看到 bank 冲突计数增加、共享内存效率降低。与此同时，共享内存吞吐只是其预期带宽的一小部分。在 Nsight Systems 中，你会看到 warp 因长时间的 bank 冲突停顿而等待，而不是在做有用的工作。

bank 冲突迫使本应并行的片上共享内存访问逐个重放，抹掉了你期望从缓冲中获得的任何加速，并常常带来令人失望的、低于预期的性能。如果你的核函数没有像预期那样加速，bank 冲突很可能是罪魁祸首。

> 始终选择你的步长和数据布局，让同一个 warp 中的线程命中不同的 bank，从而避免那个串行化瓶颈。

解决办法是调整共享内存中的数据布局以避免冲突。一种常见技术是对共享数组进行*填充*（padding），使每一行、或每一种内存访问模式映射到不同的 bank。例如，如果你有一个 32 × 32 的共享瓦片，你可以通过增加一个额外的填充列，把它声明为 [32][33]，这样每一行现在占据 33 个 float。这额外的 1 个元素偏移意味着，当一个 warp 的线程 k 访问 tile[i][k] 时，连续的行会从跨越不同共享内存 bank 的地址开始。这就使所有线程不再命中同一个 bank，如图 7-6 所示。

![图 7-6. 用填充避免 bank 冲突](AI%20Systems%20Performance%20Engineering-ch7_images/figure-7-6.png)

通过把步长改为 33，一个 warp 中没有任何两个线程在访问某一给定列时会争用同一个 bank。这消除了本会发生的 32 路 bank 冲突。

这项填充增加的开销可以忽略不计——对于一个 32 宽的瓦片约多用 3% 的内存，但它完全消除了冲突，从而大幅提升性能。而且请记住，Blackwell 每个 SM 有 > 200 KB 的共享内存。对于一个 32 × 32 的瓦片，3% 的内存开销只有 1 KB。这为换取性能提升是值得的。

让我们展示一个针对简单转置核函数消除共享内存 bank 冲突的示例。在这个示例中，每个线程访问的共享内存地址与其他线程落入同一个共享内存 bank。这导致内存访问串行化，阻碍并行，并表现出缓慢的性能。下面是会引发 bank 冲突的转置核函数的原始实现。

*前例（C++）：会引发 bank 冲突的朴素转置：*

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

在这个核函数中，写入共享内存的瓦片写操作是行主序的（tile[ty][tx]），因此是合并的，使得每个 warp 以连续方式把一整行 32 个 float 写入共享内存。（注：以列主序 tile[tx][ty] 写入瓦片到共享内存会使 warp 以 128 字节为步长，并触发 32 路 bank 冲突。）相比之下，瓦片的读操作是转置的，因为每个线程读取的值中 threadIdx.y 是行索引、threadIdx.x 是列索引。
