在实践中，两阶段（two-stage）双缓冲（double buffering）流水线非常适合均匀分块（tiling）的 GEMM 工作负载。它更简单的生产者/消费者模型能把大部分 DRAM 延迟隐藏在计算之下。而 warp 专门化（warp specialization）方法则针对不规则或更深的流水线做了优化，例如融合注意力核函数。原因在于，每个 warp 都能持续执行分配给它的角色——加载、计算或存储——而不必迫使块内其余部分停顿。

### PyTorch、CUDA Pipeline API 与 warp 专门化

PyTorch 的公开 API 并不直接暴露 cuda::pipeline，但当你调用 torch.compile 时，编译器会生成核函数（用 OpenAI 的 Triton 语言实现），这些核函数所用的优化实现了与 <cuda/pipeline> 原语和 warp 专门化的生产者/消费者等价的功能。

因此，虽然你看不到显式生成的 cuda::pipeline 调用，但你会看到底层的指令与屏障。这些优化有助于提升占用率（occupancy）并增大数据传输/计算的重叠。换句话说，你无需自己编写任何 CUDA 代码，就能获得与手写 <cuda/pipeline> 实现相同的低延迟、高吞吐行为。

PyTorch 由 TorchInductor 生成的融合注意力核函数，使用了带生产者组与消费者组的 warp 专门化。例如，考虑 PyTorch 中如下所示的三个独立 GPU 核函数：

```
scores = torch.matmul(queries, keys.transpose(-2, -1))
probabilities = F.softmax(scores, dim=-1)
context = torch.matmul(probabilities, values)
```

你可以在 PyTorch 中用一行代码将其重写，如下所示：

```
context = torch.nn.functional.scaled_dot_product_attention(queries,
   keys, values)
```

在底层，加载 warp 将分块取入共享内存，计算 warp 处理这些分块，存储 warp 将结果写回。这消除了 matmul 与 softmax 阶段之间对全局内存的昂贵往返（round trip）。

总体而言，torch.compile 非常适合性能敏感且不规则的工作负载。像 Blackwell 这样的现代 GPU 拥有充裕的 SM 资源和高内存带宽。因此，warp 专门化之类的技术能最大限度地减少空闲周期，并允许在各 warp 之间实现近乎线性的扩展——至少在 SM 或内存带宽被完全占满之前如此。

> 即便是像 FlashAttention-3 这样高度优化的核函数，使用 warp 专门化重叠也只能达到峰值 FP16 FLOPS 的约 ~75%。这表明你不必达到 100% 的计算利用率，也能取得显著的优化里程碑。

通过将加载、计算和存储解耦为独立的阶段，流水线模型最大化了吞吐（throughput）与资源利用率，使其成为长时间运行核函数的首选方法，包括诸如 transformer 注意力、融合算子流水线以及自定义任务调度器等处理过程。这些核函数利用细粒度的 warp 间通信、深度流水线化以及线性的 warp 扩展来提供高性能。

## 持久化核函数与巨型核函数

*持久化核函数*（persistent kernel），也称为*持久化线程*（persistent threads），颠覆了通常“每个任务一个核函数”的做法。你不必启动许多各自带来显著开销的小核函数，而是可以启动单个长时间运行的核函数，其线程持续从全局内存或共享内存中的共享生产者—消费者队列拉取工作。

当持久化线程循环运行时，它们会在数据块到达时立即处理——通常借助内存拷贝或主机信号——而不退出核函数。这完全避免了重复的核函数启动开销。例如，一个持久化核函数可能用一个线程块的 Warp 0 将数据从全局内存（或 CPU 主机内存）拷贝到共享内存。与此同时，Warp 1 计算上一批数据。这是 GPU 上软件流水线化的一种形式。

例如，考虑有 1,000 个微小的、彼此独立的任务。传统上，人们可能会启动 1,000 个独立的核函数。每个核函数只短暂占用几个 SM，随即退出。

在实践中，GPU 会为每个微小核函数反复加速启动，之后再减速退出。这会在各次启动之间让大多数 SM 处于空闲，无法充分利用硬件。

改用持久化核函数后，你转而启动一个大网格，其设计目标是在整个工作负载期间都让 GPU 保持忙碌。例如，在一块拥有 132 个 SM 的 GPU 上，这可能意味着为每个 SM 启动一个块、每个块 256 个线程。总共就是 33,792 个线程。随后每个线程执行大致如下所示的代码：

```
__device__ int g_index;  // global counter for next task; initialize to 0 on host
__global__ void persistentKernel(Task* tasks, int totalTasks) {
    // Every thread loops, atomically grabbing next task index until none remain
    while (true) {
        int idx = atomicAdd(&g_index, 1);
        if (idx >= totalTasks) break;
        processTask(tasks[idx]);
    }
}
```

在启动这个核函数之前，主机会在设备内存中设置 g_index = 0。然后它会像下面这样调用该核函数：

```
cudaMemset(&g_index, 0, sizeof(int));
// one block per SM on a 132 SM GPU
int blocks         = 132;
int threadsPerBlock = 256;
persistentKernel<<<blocks, threadsPerBlock>>>(d_tasks,
    totalTasks);
cudaDeviceSynchronize();
```

现在，你不再为每个单独任务支付启动开销，而只需为启动 persistentKernel 支付一次开销。假设启动大约需要 0.02 ms，而 1,000 个任务中每个运行 0.1 ms，那么这些核函数会背靠背运行，总工作量约为 100 ms。

相比之下，背靠背运行 1,000 个微小核函数、每个启动耗时 0.02 ms，会在执行时间上额外增加 20 ms 的开销——这与全部 1,000 个任务共计 100 ms 的总运行时间是分开的。简而言之，将小任务整合进一个持久化循环，可以削减数十毫秒的启动开销。

使用持久化核函数，GPU 保持高度利用——几乎所有 SM 都在并发处理任务——因为有数万个线程可用于处理这 ~1,000 个任务。相反，顺序启动 1,000 个微小核函数会让 GPU 在任一时刻都有大量部分未被充分利用。在我们的例子中，这相当于平均只使用了 GPU ~35% 的容量。

在这个持久化核函数场景中，每个 SM 运行一个 256 线程的块（在最多 2,048 线程中），因此每个 SM 都在工作。这样一来，尽管每个 SM 自身的占用率相对较低，仅为 12%（256 ÷ 2048 线程），却有近 100% 的 SM 处于活跃状态。

以这种方式使用持久化核函数，GPU 可以维持较高的实测 SM Active percent，但受寄存器和共享内存用量的制约。请记得始终用 Nsight Compute 加以验证。

在现代 GPU 上，持久化核函数尤其有效，因为其更大的共享内存容量和扩展的寄存器文件让每个线程能在片上保留更多中间状态。线程可以在其他 warp 进行计算时，使用 TMA 为即将到来的任务预取（prefetch）张量分块。因此，在某些线程处理一个任务的同时，其他线程使用 TMA 为即将到来的任务预取数据——而不会用内存传输指令拖累 SM 的计算流水线。

### 持久化核函数的常见工作负载

当你有许多小的或大小不均的任务、若分别处理会带来高启动开销时，持久化核函数便大放异彩。它们支持动态负载均衡。这让更快的线程能够继续循环、抓取更多工作。使用持久化核函数，任何 SM 都不会过早地陷入空闲。

这种模式常见于不规则工作负载，例如图遍历、自定义批处理变换，以及 LLM 推理中常见的逐 token 操作。在这些情形中，每个任务的耗时可能差异很大。

不过，持久化核函数也有缺点。首先，你必须使用原子操作显式地管理任务队列与同步。如果许多线程试图同时递增同一个计数器，这可能引入竞争。

调试单个巨型的持久化循环，比调试多个小核函数更为复杂。这是因为单个发散线程或一个意料之外的分支就可能导致整个核函数挂起。此外，一个持久化核函数可以无限期地独占 GPU。因此，如果需要并发运行其他工作负载，你必须谨慎地分配 stream 或划分资源。

简而言之，通过把 GPU 变成一个不断抓取并处理任务的动态“工作线程”池，持久化核函数相比朴素核函数可以大幅提高整体吞吐（例如 2–3×）。在现代 GPU 硬件上，这种方法消除了启动开销，最大化了 SM 占用率，并且——当与协作组（cooperative groups）或线程块簇（thread block cluster，稍后介绍）结合时——能在多阶段流水线中始终把数据保持在片上。

越来越多的框架和库正在使用持久化核函数与巨型核函数，以避免容量浪费并提升推理等延迟敏感型工作负载的性能。关键在于消除重复启动，并在 GPU 上使用设备端任务队列（device-side queue），让 SM 始终被占满并执行有用的工作。

> 在撰写本文时，由于调度的复杂性，PyTorch 尚不会自动把整个多阶段工作负载融合进一个核函数。因此，要充分获得持久化核函数与巨型核函数的收益，需要自定义 CUDA 代码或专用编译器。尽管如此，对于多阶段算法，重构为持久化巨型核函数可以带来显著的性能提升——前提是你妥善处理各处同步并避免死锁（deadlock）。

### 面向推理的巨型核函数

此外，一种源自大规模推理的持久化核函数现代做法称为*巨型核函数*（megakernel）。巨型核函数将跨层——甚至跨 GPU——的整段操作序列融合进单个大型核函数。如图 10-8 所示，持久化巨型核函数已被证明通过消除重复的核函数启动开销，相比传统的逐层启动可将延迟降低 1.2× 到 6.7×。

![图 10-8. 巨型核函数相对于 vLLM 和 SGLang 的解码吞吐提升（来源：https://oreil.ly/2aZiF）](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-8.png)

### 持久化核函数与 warp 专门化

warp 专门化通常与持久化核函数一起使用，在这些场景中线程会在相对较长的时间内执行多次迭代。这样便可实现更深的流水线、更好的重叠，以及对长生命周期资源的高效利用。对于运行时间较短的核函数，持久化核函数与 warp 专门化所增加的代码复杂性可能得不偿失。

而持久化核函数调度的一个局限，是要为持久化核函数找到足够的 SM 来利用。如果太多 SM 被另一个核函数占用，可能就没有足够的资源来启动持久化核函数。这在尝试跨 SM 调度和负载均衡工作时会带来挑战。

为了便于实现持久化核函数（从而实现 warp 专门化），现代 GPU 支持*线程块簇*（thread block cluster）——由于线程块也称为协作线程阵列（cooperative thread array，CTA），因此它也被称为*协作线程阵列（CTA）簇*。我们会在后续小节讨论线程块（CTA）簇，简而言之，它们让你把多个线程块组合成占用 GPU 上多个邻近 SM 的“簇”。

## 协作组

协作组（cooperative groups）让你能够以任意粒度定义并同步线程组。例如，你可以用单个线程、warp、分块、块和簇来创建组，如图 10-9 所示。

![图 10-9. 使用协作组在不同粒度的线程之间进行同步](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-9.png)

协作组提供了安全、可复用的集合操作，如 sync、broadcast 和 reduce。这与使用临时凑合的同步屏障形成对比。通常，线程只能使用 __syncthreads() 在自己的块内同步——例如，并没有针对整个网格的内置全局屏障。

协作组为你提供核函数内部的细粒度同步。该 API 非常适合跨 warp、块和簇协调多阶段流水线。要在你的核函数中使用 Cooperative Groups API，只需包含 <cooperative_groups.h>，获取组对象，然后在这些组之间进行同步与协调。

该 API 包含诸如 cg::this_thread_block()、cg::tiled_partition() 和 cg::this_cluster() 之类的调用。随后你调用 group.sync()——或类似的集合操作——来协调这些组中的线程。

要以协作模式启动核函数，你使用 cudaLaunchCooperativeKernel()。在协作模式下，CUDA 会确保你启动的网格能够并发驻留——否则启动失败。因此，建议始终使用 cudaOccupancyMaxActiveBlocksPerMultiprocessor 来确定协作式网格（cooperative grid）的规模，并对网格大小进行钳制以避免启动失败。在核函数内部，你随后可以调用下面的代码来实现一个覆盖所有线程块的屏障：

```
cooperative_groups::this_grid().sync(); // or grid.sync()
```

在这种情况下，任何块中的任何线程都不能越过该点，直到每个块中的每个线程都到达该点为止。这个屏障让你能够把一个核函数拆分为顺序的多个阶段，而无需结束启动或将控制权交还给主机。

例如，考虑 LLM 中常见的 softmax 算法，它有两个阶段：先在整个数组上做归约以计算聚合和，然后是使用该聚合和的逐元素后续计算。传统上，你会启动一个核函数做归约，将结果拷回主机内存或全局内存，再启动第二个核函数从主机或全局内存消费该结果，然后计算 softmax。这需要大量相对缓慢的内存搬移。

有了协作组，你可以在一个核函数中完成这两个阶段：每个块计算它的部分和，由一个块把这些部分和聚合成最终结果，所有块调用 grid.sync() 等待聚合完成，然后所有线程进入第二阶段。随后每个线程会从寄存器或共享内存——而非从全局内存——读取聚合和。

CUDA 保证协作式启动中的每个块都同时驻留在 GPU 上。如果你请求的块数超过了能够并发容纳的数量，启动就会直接失败。

由于协作式核函数必须以 GPU 能够并发运行的网格大小来启动，grid.sync() 不会因等待不存在的块而挂起。换句话说，CUDA 运行时保证所有线程块都处于活跃状态并会到达 grid.sync() 屏障。如果核函数启动过大而无法一次性运行，它只会启动失败。因此，检查 cudaLaunchCooperativeKernel() 的返回状态非常重要。

例如，如果一块 GPU 有 132 个 SM，而你的核函数占用的资源使得每个 SM 能运行四个块，那么网格必须不大于 528 个块才能成功。如果超出该上限，协作式启动就会直接失败。请使用 cudaOccupancyMaxActiveBlocksPerMultiprocessor 为协作式核函数确定网格规模，并在假定已取得进展之前检查 cudaLaunchCooperativeKernel 的返回状态。

在 Blackwell 之前，开发者常将 cudaLaunchCooperativeKernel 与全局内存原子操作或标志位配合使用，来协调那些并不共享片上内存的多个块。这种方法虽然可行，但迫使中间结果被搬移到全局内存。这带来了额外的 HBM 流量。

在 Blackwell 上，线程块簇与 DSMEM 提供了一种效率高得多的替代方案。线程块簇可以在片上 SRAM 中共享数据——并且无需全局内存往返即可同步。我们将在本章后面介绍线程块簇与 DSMEM。

当你需要在单次核函数启动内实现一个真正的、覆盖所有线程块的屏障时，就应当使用协作式核函数。此外，当你希望把中间结果保留在快速内存（例如寄存器或共享内存）中、而不是反复写入和读取全局内存时，它们也很有用。建议你把网格大小限制在 GPU 的最大并发块容量以内——或选择更大的块——以减少总块数并避免核函数启动失败。

协作式核函数的缺点在于你的网格大小受容量限制。这可能迫使你使用更少、更大的块——或依赖线程块簇（本章后面介绍）。

尽管所有块都并发运行，协作式屏障仍要求每个块中的每个线程都调用 grid.sync()。如果任何单个线程跳过——或永远无法到达——grid.sync() 调用（例如由于一条发散的 if 语句），那么每个确实调用了 grid.sync() 的其他线程都将永远等待下去。这会导致死锁。

简而言之，协作组核函数让你能把整个 GPU 当作一个单一的协作资源来对待，以 grid.sync() 充当全局屏障。这对于需要全局同步与数据共享的多阶段算法非常理想。只是要记住，grid.sync() 是一种相对重量级的同步，其开销高于块级或簇级屏障。这是因为它必须在运行于多个 SM 上的所有线程块之间进行协调。因此，你应当谨慎使用 grid.sync()，并且至少要确保你的核函数在两次重量级同步调用之间执行了相当多的工作。

> 对于只需要有限的跨块协调的情形，使用全局内存原子操作或每块标志位可能比依赖整网格屏障更安全、更简单。不过，正如我们稍后将讨论的，它们的效率不及使用线程块簇与 DSMEM。

### 协作式网格同步与持久化核函数

如果你的工作负载偶尔需要在持久化循环内部使用跨线程块屏障来聚合部分结果，你可以通过调用 grid.sync() 把持久化核函数与协作组结合起来。这将提供一个覆盖整个网格的屏障，从而避免结束核函数并不得不重新启动它。以这种方式，一条由归约和其他全局步骤组成的多阶段流水线便可完全保持在设备上。

例如，考虑一个工作负载，它在 1,000 次迭代中反复执行两种不同的计算，且每种计算因跨块数据依赖而需要一个全局屏障。朴素的实现可能每次迭代启动两个独立的核函数，导致 2,000 次核函数启动。在单个 stream 中，第二个核函数会自动等待第一个——无需显式的主机端同步——但你仍要支付启动开销。相反，你可以把所有内容融合进一个协作式、持久化的核函数，如下所示：

```
#include <cuda_runtime.h>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__device__ inline float someComputationA(float x) { return x * 2.0f; }
__device__ inline float someComputationB(float a, float b) { return a + b; }
__global__ void combinedKernel(float* __restrict__ dataA,
                               float* __restrict__ dataB,
                               int N,
                               int iterations) {
    cg::grid_group grid = cg::this_grid();                 // whole-grid group
    const int tid = blockIdx.x * blockDim.x + threadIdx.x; // thread's linear id
    const int stride = gridDim.x * blockDim.x;             // total grid threads
    for (int it = 0; it < iterations; ++it) {
        // Stage 1: update all elements of A using a grid-stride loop
        for (int i = tid; i < N; i += stride) {
            dataA[i] = someComputationA(dataA[i]);
        }
        grid.sync();  // global barrier; also provides memory visibility
        // Stage 2: use finished A to update B (again grid-stride)
        for (int i = tid; i < N; i += stride) {
            const float mid = dataA[i];
            dataB[i] = someComputationB(mid, dataB[i]);
        }
        grid.sync();  // barrier before next iteration
    }
}
```

在主机端，你只需调用这个合并了协作式与持久化特性的核函数，如下所示：

```
// ... allocate/copy dA, dB, set N, iterations, choose blockSize
cudaDeviceProp prop{};
cudaGetDeviceProperties(&prop, 0);
if (!prop.cooperativeLaunch) {
    // Fallback: run two kernels per iteration instead of grid.sync()
    // (see alternative below)
}
// Compute the maximum grid size allowed for a cooperative kernel
int maxBlocksPerSm = 0;
const int blockSize = 256;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSm,
                                              combinedKernel,
                                              blockSize, 0);
const int maxCoopGrid = maxBlocksPerSm * prop.multiProcessorCount;
// Choose grid size, but clamp to cooperative limit
int gridSize = (N + blockSize - 1) / blockSize;
if (gridSize > maxCoopGrid) gridSize = maxCoopGrid;
// Launch cooperatively
void* args[] = { &dA, &dB, &N, &iterations };
cudaLaunchCooperativeKernel((void*)combinedKernel, gridSize, blockSize, args);
cudaDeviceSynchronize();
```

这个单一核函数取代了原本需要的 2 × 1,000 = 2,000 次核函数启动，并避免了重复的启动开销。在循环内部，grid.sync() 确保了正确的次序（每个块都先完成 Stage 1，然后任何块才开始 Stage 2——并且在进入下一次迭代之前如此），而无需任何主机端同步。

逐线程和逐块的数据可以跨越 grid.sync() 保留在寄存器/共享内存中。对于跨块的数据交换，则使用全局内存（或者线程块簇与分布式共享内存，我们稍后会介绍）。与此同时，由于工作是在 GPU 上循环进行的，首次调用之后就不再有启动开销。

协作式核函数需要设备支持（cooperativeLaunch），并且必须用 cudaLaunchCooperativeKernel 启动。网格中的所有块都必须并发驻留。请使用 CUDA Occupancy API 来确定并钳制网格大小，使所有 CTA 都能共同驻留。否则启动将失败。一个例子是当 ((N + threads - 1) ÷ threads) 超过协作容量时。

> 使用 CUDA Occupancy API 来确定协作式网格的规模。

现代 GPU 可以并发运行数千个数量级的线程块（前提是每个块只使用适量的资源，包括寄存器和共享内存），因此有相当大的余量。不过，在采用这种实现之前，你仍应验证块大小和资源消耗。

在这个核函数示例中，核函数在两次迭代之间从不将控制权交还给主机。因此，对于外层循环它表现得像一个持久化核函数，但在内层阶段又强制了全局同步点。

这种组合模式对于 LLM 推理中的多步算法尤其强大，其中每一层可能都需要一次归约或归一化（Stage 1），随后是一次逐元素变换（Stage 2）。通过把所有内容都装进一个协作式持久化核函数，你消除了所有核间往返，并最大化了片上数据局部性。

### 何时结合使用持久化核函数与协作组

一种常见的最佳实践是，在资源允许的情况下，为每个 SM 启动一个线程块的持久化核函数。这样，每个 SM 都有一个驻留的线程块，反复处理来自工作队列的任务。这最大化了占用率，并让 SM 不至于陷入空闲。这些线程块随后会使用 grid.sync() 在各阶段之间协调。然而，在协作式策略与持久化策略之间——以及是否将二者结合——做决定时，请提出以下问题：

*你是否有多个需要全局同步的顺序阶段？* 如果是，就使用协作式核函数（或协作式持久化核函数），以便在各阶段之间调用 grid.sync()。

*你是否有许多因启动开销而受损的小任务或不规则任务？* 如果是，就使用持久化核函数，让你的线程在一个共享队列上循环，而无需返回主机。

*你是否能够承担为这一工作负载独占整个网格（或一个线程块簇）？* 如果是，协作式持久化核函数或许能带来最佳性能。

*你是否需要与其他工作共享 SM？* 如果是，就考虑使用线程块簇（下一节介绍），以便持久化或协作式核函数不为其他工作负载服务。

简而言之，当你使用单个核函数既在任务上持久化循环、又包含 grid.sync() 调用来在各阶段之间同步所有线程块时，你就消除了通常在独立核函数启动之间发生的暂停和额外的内存传输。

在现代 GPU 上，这意味着数据在整个计算过程中都保留在共享内存或寄存器里。这与在每个阶段之后把数据写回全局内存形成对比。其结果是，GPU 几乎始终忙于做有用的工作——达到接近其硬件峰值极限的性能。

一个重要的注意事项：协作式核函数会占用网格中的所有 SM，因此在这些 SM 上不能并发运行其他核函数。如果你需要协同调度（coscheduled）其他工作，例如在另一个 stream 上的异步数据预取或较低优先级的推理核函数，你可能需要把 GPU 划分为线程块簇，这将在下一节介绍。

## 线程块簇与分布式共享内存

*协作组*是一种软件层面的抽象，它提供了一个 API，让你能把核函数的线程切分为任意的集合以进行同步和数据搬移。这包括 warp、分块、线程块——甚至整个网格或多设备网格。

相比之下，线程块簇（或协作线程阵列 [CTA] 簇）是一种硬件层面的层次结构。它把 SM 的一个子集授予你的协作式网格——并把其余部分留给其他核函数使用。这缓解了单个核函数独占 GPU 的风险。GPU 保证这些线程块会被协同调度到同一个 GPU 处理集群（GPC）上，如图 10-10 所示。

![图 10-10. 多个线程块簇被保证协同调度到同一个 GPC 或 GPC 分区上](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-10.png)

GPC 是一组邻近 SM 的集合。GPU 把线程块调度到各个 GPC 上，类似于它把一个线程块的线程调度到同一个 SM 上。

> 在像 NVIDIA Blackwell B200/300 和 GB200/300 这样的多晶片模块上，实际上存在多个 GPC“分区”——每个晶片一个 GPC 分区。由于 Blackwell 是双晶片 GPU，它有两个 GPC 分区。请记住，Blackwell 的两个 GPU 晶片通过 NV-HBI 连接，并作为单个 CUDA 设备呈现，跨晶片具有完整的缓存一致性。L2 缓存在两个晶片之间也是一致的。因此，这些晶片构成了一个组合的逻辑 GPC，从而由架构为你处理各个独立的 GPC 分区。

GPU 提供分布式共享内存（distributed shared memory，DSMEM，将在下一节讨论），供线程块簇跨这些块使用。它还支持使用 Cluster Group API 的簇级屏障（cluster.sync()）。

这个簇级屏障让你只同步线程块的一个子集，而不阻塞整个 GPU。线程块簇让你能够启动一个协作式核函数，把网格细分为更小的块组，如图 10-11 所示。

![图 10-11. 对于这四个（2 × 2）线程簇，A 和 B 的每个分块被同时加载到两个线程块中（来源：https://oreil.ly/kEZsv）](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-11.png)

在每个组内，调用 cluster.sync() 提供一个局部屏障。这让簇内的块能够通过专用的片上资源共享数据，而不必独占每一个 SM。在现代 GPU 上，你可以使用 DSMEM，它允许线程块簇共享一块连续的片上 SRAM 区域。这在同一簇内的块之间实现了低延迟通信，并有原生硬件支持。

线程块簇把线程块的一个子集分组，并在每个簇内调用 cluster.sync() 以“局部”地在这些线程块之间同步。虽然线程块内的线程传统上可以使用共享内存进行协作，但在现代 GPU 上，它们现在还可以使用线程块簇和 DSMEM 相互协作。

换句话说，如果没有线程块簇，只有同一线程块内的线程才能在共享内存中共享数据。不同的线程块则必须使用全局内存和全局屏障（例如 grid.sync()）来协调，而这会让所有线程停顿并限制可扩展性。

而且，此外，正如上一节所述，不同线程块中的线程此前除了使用全局内存或用于粗粒度、覆盖整个网格的同步的 grid.sync() 之外，无法高效地共享状态和同步。遗憾的是，全局内存和 grid.sync() 都相对缓慢，因此可能成为瓶颈。

线程块簇由 GPU 硬件原生支持，包括*簇启动控制*（cluster launch control）。簇启动控制是一种硬件层面的机制，用于启动和调度持久化的线程块簇。具体而言，它允许一个持久化核函数（及其线程块簇）维持均衡的工作负载——即便某些 SM 已被占用。这为高效的 warp 专门化实现奠定了基础。

借助硬件支持的通信与同步构件，线程块簇可以以底层 PTX 指令和 CUDA intrinsic 的形式使用覆盖整个簇的屏障进行同步。因此，由于线程块之间有硬件支持的屏障同步，簇内的块进行屏障同步可以比使用 grid.sync() 的完整网格同步快得多。

### 线程块交错

在直接的网格启动中，线程块按严格的行主序或列主序处理分块。这可能导致较早的块驱逐掉稍后的块将会需要的数据——从而造成糟糕的复用和额外的内存流量。相反，你希望同一波（wave）内的分块 A 和 B 能从 L2 缓存中读出。

为了绕过这种低效，你可以使用线程块交错（thread block swizzling）。类似于用交错来优化内存访问并避免共享内存 bank 冲突，你可以使用线程块交错来避免以低效的行主序和列主序分配分块，如图 10-12 所示。

![图 10-12. 线程块交错，在单一波中从 L2 缓存读取分块 A 和 B](AI%20Systems%20Performance%20Engineering-ch10_images/figure-10-12.png)

线程块交错让同一波所需的 A 矩阵和 B 矩阵的分块都留在 L2 中以实现最大复用。当应用于持久化和分块的 GEMM 工作负载时，这类交错可以通过减少内存未命中和带宽压力，带来两位数的性能提升。线程块交错是一种简单却强大的模式重排技术，它把核函数的启动次序与缓存局部性对齐。

有了对线程块簇的硬件支持、相对大量的 SM 以及改进的流水线调度，现代 GPU 可以承载带有精细 warp 专门化流水线的大型持久化核函数，从而优化核函数性能。

简而言之，线程块簇建立在协作组模型之上，允许块的一个子集在不锁死整个 GPU 的情况下进行同步和状态共享。这让你能够在一次启动内构建多阶段、细粒度的流水线，而不锁死设备的其余部分。这就让剩余的 SM 得以自由执行独立的工作。

在线程块簇内部，有两种在线程块之间共享状态的关键机制：DSMEM 和暂存内存（scratch memory）。接下来我们逐一介绍。

### 分布式共享内存

分布式共享内存（DSMEM）把 __shared__ 内存的概念从单个线程块扩展到横跨整个线程块簇。在传统核函数中，每个线程块都有自己私有的 __shared__ 区域，且其他线程块无法访问。

然而有了 DSMEM，同一簇内的多个块可以读写一个在逻辑上把它们所有本地 __shared__ 区域组合起来的共享内存空间。实际上，簇的共享内存被拼接成一个分布式的片上缓冲区。

通过把块间数据保留在片上内存中，DSMEM 显著提高了有效算术强度（arithmetic intensity），因为块之间可以交换中间结果或执行归约，而无需往返全局内存。当数据集对单个块的共享内存来说过大时，这尤其有价值。
