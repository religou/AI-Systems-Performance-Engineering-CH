```
# two_stage_pipeline.cu
#include <cuda/pipeline>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
extern "C" __global__
void stage_ab_tiles(const float* __restrict__ globalA,
                    const float* __restrict__ globalB,
                    float* __restrict__ outC,
                    int tile_elems,
                    int num_tiles) {
     // Alignment / size guards for vectorized copies (runtime parameter)
    assert((tile_elems % (32 * 4)) == 0 &&
           "tile_elems must be multiple of 128 for float4 vectorization");
    // If you cannot guarantee 16B alignment or sizes, handle
    // the tail/ragged edges with a fallback 4B loop.
  extern __shared__ float smem[];
  auto block = cg::this_thread_block();
  // Shared buffers for double buffering of A and B
  float* A0 = smem + 0 * tile_elems;
  float* A1 = smem + 1 * tile_elems;
  float* B0 = smem + 2 * tile_elems;
  float* B1 = smem + 3 * tile_elems;
  constexpr auto scope = cuda::thread_scope_block;
  constexpr int stages = 2;
  __shared__ cuda::pipeline_shared_state<scope, stages> pstate;
  auto pipe = cuda::make_pipeline(block, &pstate);
  // Prime the pipeline with tile 0
  pipe.producer_acquire();
  cuda::memcpy_async(block, A0, globalA + 0 * tile_elems,
                     cuda::aligned_size_t<32>{tile_elems * sizeof(float)}, pipe);
  cuda::memcpy_async(block, B0, globalB + 0 * tile_elems,
                     cuda::aligned_size_t<32>{tile_elems * sizeof(float)}, pipe);
  pipe.producer_commit();
  for (int t = 1; t < num_tiles; ++t) {
    // Stage the next A and B tiles
    pipe.producer_acquire();
    cuda::memcpy_async(block, (t & 1) ? A1 : A0,
                       globalA + t * tile_elems,
                       cuda::aligned_size_t<32>{tile_elems*sizeof(float)}, pipe);
    cuda::memcpy_async(block, (t & 1) ? B1 : B0,
                       globalB + t * tile_elems,
                       cuda::aligned_size_t<32>{tile_elems*sizeof(float)}, pipe);
    pipe.producer_commit();
    // Consume the previously staged tiles
    pipe.consumer_wait();
    float* prevA = (t & 1) ? A0 : A1;
    float* prevB = (t & 1) ? B0 : B1;
    // Perform compute using prevA and prevB
    pipe.consumer_release();
  }
  // Consume the final staged tiles
  pipe.consumer_wait();
  int last = (num_tiles - 1) & 1;
  float* lastA = last ? A1 : A0;
  float* lastB = last ? B1 : B0;
  // Perform compute using lastA and lastB
  pipe.consumer_release();
}
```

启动这个核函数时，将动态共享内存大小设置为 4 x tile_elems x sizeof(float)，以便在共享内存中为 A0、A1、B0、B1 分配空间。这种双缓冲模式确保一旦某个分块驻留在共享内存中，Tensor Core 就能立即开始处理它。与此同时，cuda::memcpy_async 会并行地把下一个分块取入共享内存。由于 TMEM 为 Tensor Core 指令提供了片上数据缓冲区、而共享内存提供了暂存空间，你可以将 FP16、FP8 或 FP4 分块完全在片上暂存并复用。其结果是：当流水线调优得当、分块与拷贝的规模设置合适时，停顿更少。cuda::memcpy_async 能让从 HBM 到共享内存的传输与计算重叠，使核函数保持忙碌。这有助于把访存延迟隐藏在计算之后。

### TF32 与自动混合精度（PyTorch）

Tensor Core 最初是为 FP16 设计的，但它们也支持 TF32——一种介于 FP32 与 FP16 之间的格式。TF32 采用与 FP32 相同的 8 位指数位（exponent），以及与 FP16 相同的 10 位尾数位（mantissa）。TF32 在 Tensor Core 上的吞吐远高于 FP32 在 CUDA core 上的吞吐，同时保留了 FP32 的指数范围。在 PyTorch 中，启用 TF32 只需在代码里设置如下内容：

```
import torch
torch.set_float32_matmul_precision('high') # {'highest'|'high'|'medium'}
```

一旦设置了这些标志，torch.matmul 和 torch.nn.Linear 等高层操作就会自动以 TF32 Tensor Core 核函数执行，而不再以 FP32 在标准 CUDA core 上执行。

除 TF32 外，PyTorch 的自动混合精度（automatic mixed precision，AMP）还能为每个操作选择最优精度（FP16 或 BF16），并将结果以 FP32 累加以保证稳定性。BF16 有助于规避 FP16 的上溢（overflow）问题。默认情况下，CUDA autocast 使用 float-16。只需传入 dtype=torch.bfloat16，即可在支持的 GPU 上选用 BF16。例如，你可以用上下文管理器包裹模型代码，如下所示：

```
with torch.amp.autocast("cuda", dtype=torch.bfloat16):
    output = model(input)
```

在底层，TorchInductor（见第 13、14 章）会自动融合这些精度转换，以确保：大型 GEMM 操作在 Tensor Core 上以 FP16 或 TF32 运行、累加保持在 FP32 以获得数值稳定性、诸如层归一化（layer normalization）和 softmax 之类的小型“敏感”核函数以 FP32 运行，以及 GradScaler 在使用 FP16 训练时防止下溢（underflow）。注意，BF16 拥有 FP32 的指数范围。因此，用 BF16 训练时通常不需要 GradScaler。

在 PyTorch 中，这些混合精度决策已集成进编译器，因此你无需手动干预即可获得最优的 dtype 选择（例如计算用 FP16/FP8、累加用 FP32）。这一点在图 9-7 中以混合精度矩阵乘加（matrix multiply-accumulate，MMA）的形式展示。

这条自动混合精度流水线以最小的代码改动最大化了算术强度。融合后的 Tensor Core 核函数通过在共享内存（如操作数）和 TMEM（如累加器）中暂存并复用数据，尽量减少往返 HBM 的次数。

![图 9-7. 混合精度与矩阵乘加（MMA）](AI%20Systems%20Performance%20Engineering-ch9_images/figure-9-7.png)

在使用前文所述的结构化稀疏、或极低精度（FP8/FP4）时，务必保持足够大的批大小或分块粒度，让 TMEM 和 Tensor Core 保持满负荷。小批次会带来开销，包括格式转换、稀疏索引处理、不规则内存访问模式等。这会削弱实际获得的加速。

例如，使用 FP8 或 2:4 稀疏时，批大小为 1 可能几乎看不到收益，因为固定开销没有被摊薄。相比之下，批大小为 128 或 256 会充分利用 TMEM 流水线，产生接近峰值的吞吐。

### BF16/FP16、FP8 与 FP4 低精度

BF16/FP16（半精度，half-precision）已在多代 GPU 上得到支持，而现代 GPU 上的 Tensor Core 往往能维持超过 90% 的 BF16/FP16 峰值吞吐，约为 FP32 峰值吞吐的 4×。这是因为硬件在每个周期都并行发射大量 BF16/FP16 FMA 运算。

FP16 训练使用比 FP32 更窄的 5 位指数位，因此除非施加损失缩放（loss scaling），否则极小的梯度值会下溢到零。损失缩放在反向传播（backpropagation）期间维持数值稳定性。这种缩放可以是静态的，也可以是动态的。

相比之下，BF16 与 FP32 的 8 位指数范围一致，天生就能避免下溢。因此它极少（如果有的话）需要损失缩放。这简化了混合精度工作流，并且在现代 GPU 上往往能提升训练精度。

> 在现代 GPU 上，训练通常首选 BF16，因为它能维持与 FP32 相当的精度，又不必承受 FP16 所要求的损失缩放的复杂性。

要把吞吐推得更高，可以使用 FP8。把 16 位权重减少 50% 降到 8 位后，你将内存流量减半——并使每次 HBM 事务加载的权重数量翻倍。实践中，采用 FP32 或 TF32 累加的 FP8 矩阵乘可达到 BF16/FP16 TFLOPS 的 2–3×——前提是模型因量化误差（quantization errors）造成的轻微精度损失仍在可接受范围内。

为应对极低精度下的精度问题，Transformer Engine 既支持 FP8，也支持 NVIDIA 带微缩放（micro-scaling）的 4 位 NVFP4 格式。NVFP4 采用两级缩放，将按微块缩放（per-microblock scaling）与一个更高层级的缩放相结合，使模型在用 4 位存储权重的同时仍能保持精度。此外，Blackwell B200 的 NVFP4 采用激进的微缩放量化，提供 10 petaFLOPS（稠密），而 FP32 峰值约为 80 teraFLOPS（稠密）。这意味着每权重的理论吞吐提升约两个数量级。而 Blackwell 的 B300（Ultra）以 15 petaFLOPS（稠密）的 NVFP4 算力，比 B200 高出 50%。

如果你的模型在校准（calibrate）后能容忍精度下降，那么在受支持的硬件上，NVFP4 核函数可以提供远高于 FP32 的吞吐，但精度必须针对每个模型逐一验证。

而且由于精度如此之低，每个 SM 的 256 KB TMEM 能容纳更大的 FP4 分块（例如 256 × 256），这进一步提升了片上复用并改善性能。注意，所有低精度 → 累加的转换都是自动完成的。核函数从 HBM 读取 FP4 输入，Tensor Core 执行 FP4 × FP4 乘法，MMA API 则把结果累加进 BF16/FP16 或 FP32 累加器（accumulator）。

每降低一档精度，每字节的运算数就翻倍或翻四倍，因而提升了算术强度。当 TMEM/TMA 让访存与计算重叠时，这些低精度格式会把原本访存受限的核函数变成完全计算受限的核函数。这充分发挥了现代 GPU 中每 GPU 数 PFLOPS 的 Tensor Core 引擎。

### INT8 低精度与用于推理的 DP4A 指令

LLM 推理场景通常能容忍现代 GPU 所支持的低精度 INT8 量化（quantization）——在常规 CUDA core 上使用 DP4A（SIMD 点积指令）、在 Tensor Core 上使用整数矩阵乘加（MMA）指令。在指令层面，DP4A 每条指令执行四次 INT8 乘加（MAC）运算，而每条 FP32 融合乘加（FMA）指令只做一次。

由于 INT8 的权重流量为每元素一字节，而非 FP32 的四字节，权重的内存流量下降了 75%。凭借更高的 INT8 Tensor Core 峰值吞吐和更低的内存流量，INT8 推理工作负载可以显著超越 FP32。这是因为使用 INT8 权重时，每个 GPU 每秒能从内存处理约 4× 的数据。这得益于 TMEM 与 TMA 让数据和计算完美重叠——并尽可能高效地喂给 Tensor Core。

### 深入 Transformer Engine 与 TMEM

现代 NVIDIA GPU 内置了 Transformer Engine，它把面向低精度格式的 Tensor Core 硬件支持与用于缩放和转换的软件运行时结合在一起。cuBLASLt、cuDNN、CUTLASS 或 OpenAI Triton 中的核函数会执行 cp.async 指令或 TMA 传输，把数据搬入共享内存。随后，Tensor Core 指令会隐式地在共享内存与 TMEM 之间搬移操作数。

请记住，TMEM 是每个 SM 256 KB 的 SRAM 缓冲区，Transformer Engine 和 Tensor Core 用它来存储结果（而非用寄存器）。实践中，你从不显式分配 TMEM。这一切都由硬件处理。例如，调用 Tensor Core 的 MMA 操作时，硬件会处理所有的内存分配与数据传输。

借助 MMA 指令，每个 warp 都直接驱动 Tensor Core 执行高吞吐的混合精度 MMA 操作。这些操作管理片段加载（fragment loads）、寄存器映射以及混合精度 MMA 运算。

> 截至本文写作时，PyTorch 的 INT8 量化支持通过 TorchAO 和各厂商后端提供。量化模块使用专用的 INT8 核函数运行。使用 cuBLASLt 或 CUTLASS 进行底层 INT8 GEMM 可以确保 Tensor Core 的利用率。

每当你启动基于 Tensor Core 的核函数或一个 GEMM 库函数（如 CUTLASS），其实现都会自动通过共享内存和 TMEM 管理操作数搬移。这让 Tensor Core 始终装满待处理的分块。（注意，应用代码不会直接分配 TMEM。）

Transformer Engine 的工作流很直接。首先，你的核函数发出一个 MMA 调用或启动一个 CUTLASS GEMM。接着，Transformer Engine 的固件安排 TMA（或 cuda::memcpy_async）把权重和激活从 HBM 拷贝进共享内存（SMEM）。随后，Tensor Core 指令（如 tcgen05.mma）在 MMA 流水线期间隐式地在 SMEM 与 TMEM 之间搬移操作数。理想情况下，权重是 FP8 或 FP4，激活在可能时被转换为 FP8/FP4——否则，激活可以保留为 FP16/FP32 格式。

Tensor Core MMA 操作以低精度执行，例如 FP8 × FP8 配合更高精度累加，或 FP16 × FP16 配合 FP32 累加。部分和以更高精度（如 BF16、FP16、FP32）在 TMEM 中累加，具体取决于核函数。累加器状态驻留在 TMEM 中。该状态通过 tcgen05 加载和存储接口访问。硬件透明地管理这些搬移。

如果你构建自定义的分块循环，就可以让数据搬移与 Tensor Core 计算重叠。你可以使用 cuda::memcpy_async 和 CUDA Pipeline API 来做到这一点，如下面的代码所示：

```
#include <cuda/pipeline>
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
extern "C" __global__
void double_buffer_a(const float* __restrict__ globalA,
                     int tile_elems,
                     int numTiles) {
  __shared__ float tileA0[TILE][TILE];
  __shared__ float tileA1[TILE][TILE];
  auto block = cg::this_thread_block();
  constexpr auto scope = cuda::thread_scope_block;
  constexpr int stages = 2;
  __shared__ cuda::pipeline_shared_state<scope, stages> pstate;
  auto pipe = cuda::make_pipeline(block, &pstate);
  // Prime pipeline with tile 0
  pipe.producer_acquire();
  cuda::memcpy_async(block,
                     &tileA0[0][0],
                     globalA + 0 * tile_elems,
                     cuda::aligned_size_t<32>{tile_elems * sizeof(float)}
                     pipe);
  pipe.producer_commit();
  for (int t = 1; t < numTiles; ++t) {
    // Stage next tile into the alternate buffer
    pipe.producer_acquire();
    float* nxtA = (t & 1) ? &tileA1[0][0] : &tileA0[0][0];
    cuda::memcpy_async(block,
                       nxtA,
                       globalA + t * tile_elems,
                       cuda::aligned_size_t<32>{tile_elems * sizeof(float)}
                       pipe);
    pipe.producer_commit();
    // Consume the previously staged tile
    pipe.consumer_wait();
    float* curA = ((t - 1) & 1) ? &tileA1[0][0] : &tileA0[0][0];
    pipe.consumer_release();
  }
  // Consume the final staged tile
  pipe.consumer_wait();
  float* lastA = ((numTiles - 1) & 1) ? &tileA1[0][0] : &tileA0[0][0];
  // Use lastA with your compute
  pipe.consumer_release();
}
```

由于 TMEM 是 Tensor Core 指令专用的片上缓冲区，数据得以贴近计算单元。当 Tensor Core 处理当前分块时，cuda::memcpy_async 会把下一个分块从 HBM 流式送入共享内存。

这种重叠有助于隐藏访存延迟，并能在流水线调优得当时让 Tensor Core 保持忙碌。Transformer Engine、TMEM 和 TMA 之间的这种协作可以大幅提升算术强度，并在优化良好的情况下逼近 speed-of-light 效率。

> 尽管加载和存储操作相对于调用它的 warp 是同步的，但计算与数据搬移的重叠应当来自 CUDA Pipeline API。与 wait/release 等流水线原语搭配使用时，cuda::memcpy_async 会映射到 Tensor Memory Accelerator（TMA），对于大批量张量传输应始终优先使用它。cp.async 仅保留给 TMA 无法表达的小众场景。不过，这类场景很少见。你还应确保在使用数据之前拷贝已经完成。

## 使用 CUTLASS 获得最优算术强度与 Tensor Core 性能

自行利用这些优化最简单的途径之一，就是使用 NVIDIA 的 CUTLASS 库。有了 CUTLASS，你只需写一个模板化调用，它就会自动应用许多高级优化。

CUTLASS 应用的一些优化包括：共享内存分块、异步内存传输，以及借助 TMEM 每 SM 256 KB 缓冲区实现的双缓冲。这样一来，无需任何手动核函数调优，你的 Tensor Core 就能以接近峰值的吞吐运行。

> CUTLASS 还实现了 warp 专门化（warp specialization），这是一种高性能 GPU 优化技术，我们将在下一章讨论。

例如，假设你想计算一个 GEMM，C = A * B，输入为半精度、输出也为半精度，并视情况以 FP16 或 FP32 累加。你无需编写手工调优的 MMA 循环，只需引入 CUTLASS 并实例化一个模板，如下面的代码所示：

```
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
using Gemm = cutlass::gemm::device::Gemm<
  cutlass::half_t,  // A (FP16)
  cutlass::layout::RowMajor,
  cutlass::half_t,  // B (FP16)
  cutlass::layout::ColumnMajor,
  cutlass::half_t,  // C / output (FP16)
  cutlass::layout::RowMajor,
  float, // accumulator (FP32 accumulate)
  cutlass::arch::OpClassTensorOp,
  cutlass::arch::Sm100 // e.g., Blackwell B200
>;
// ... (allocate device pointers A_d, B_d, C_d,
// set up dimensions M,N,K, and strides lda, ldb, ldc) ...
Gemm gemm_op;
cutlass::Status status = gemm_op(
    { M, N, K },              // GEMM shape
    float(1.0f)               // alpha
    A_d, lda,                 // A pointer + leading dimension
    B_d, ldb,                 // B pointer + leading dimension
    float(0.0f),              // beta
    C_d, ldc                  // C pointer + leading dimension
);
```

当你编译并运行这段代码时，CUTLASS 会自动完成几件关键的事。首先，CUTLASS 会选择分块，以平衡寄存器压力、共享内存容量和 Tensor Core 利用率。在现代 GPU 上，TMEM 与共享内存和 L1 并存。CUTLASS 在共享内存中暂存分块，并使用与 TMEM 交互的 Tensor Core 指令来存储累加器数据。分块形状是凭经验、按核函数逐一选定的。例如，它可能选择 128 × 128 或 256 × 128 这样的分块大小。这些都能放入 TMEM 每 SM 256 KB 的缓冲区，并在整个 Tensor Core 计算过程中保持在片上。

取决于精度，一个 256 × 512 的分块会占满每 SM 256 KB 的 TMEM 预算，因为 256 × 512 个元素 × 每元素 2 字节 = 256 KiB。而 256 × 256 个元素 × 每元素 4 字节 = 256 KB。更大的分块能提升单块吞吐，但会减少每个 SM 上并发分块的数量。在较小的 GEMM 上，这可能导致利用不足。反过来，非常小的分块则以牺牲算术强度换取并行度。

随后，CUTLASS 发出异步内存拷贝（cp.async 或 TMA），把每个分块从 DRAM 流式送入共享内存。cp.async 指令把数据从全局内存暂存到共享内存，而不使用每线程寄存器（或可选地使用 L1 缓存），如图 9-8 所示。缓存行为通过 cp.async 修饰符控制，或通过使用 TMA 进行大批量张量传输来控制。

![图 9-8. 使用异步内存拷贝指令（cp.async）把数据从全局内存加载到共享内存，而不涉及寄存器堆，也可选地不涉及 L1 缓存](AI%20Systems%20Performance%20Engineering-ch9_images/figure-9-8.png)

CUTLASS 使用 cp.async 或 TMA（cp.async.bulk.tensor）把分块从全局 DRAM 暂存进 SMEM。随后，Tensor Core 的 tcgen05.mma 指令从 SMEM 读取操作数，并把结果隐式累加进 TMEM。这在共享内存中创建了一块软件管理的暂存区，用于双缓冲。这样一来，在 Tensor Core 处理当前分块的同时，TMA 已经在把下一个分块取入共享内存。

借助 CUDA Pipeline API 和 warp 专门化的计算阶段（将在下一章讨论），CUTLASS 让所有 Tensor Core 流水线保持忙碌。它以你指定的精度累加部分和（例如输入为 FP16 或 FP8 时用 FP32），以确保数值保真度——然后以合并访问的方式把结果从 TMEM 写出到共享内存或全局内存。

> 在有益时，CUTLASS 还会利用线程块簇（thread block clusters），通过跨多个 SM 分块来得到更大的有效分块。我们将在下一章介绍线程块簇。

由于所有这些复杂性都被隐藏起来，CUTLASS 给了你一个直接替换式（drop-in）、高性能的 GEMM 核函数，其表现可与手工调优的 MMA 核函数媲美——在整体 Tensor Core 利用率和性能上，往往与手写版本相差不到几个百分点，如表 9-1 所示。

表 9-1. 手工调优的 MMA 与 CUTLASS 核函数的性能与资源用量对比

| 指标 | 手工调优的 MMA 核函数 | CUTLASS GEMM |
| --- | --- | --- |
| Tensor Core 利用率 | 98% | 98% |
| 每线程寄存器数 | ~52 | ~60（略高） |
| 每线程块（CTA）共享内存 | ~2 KB | ~4 KB |
| 开发投入 | 高 | 低（简单的模板配置） |

> 注：所有指标表格中的数值仅为说明概念的示意值。不同 GPU 架构上的实际基准测试结果，参见 GitHub 仓库。

这里，两者都使用 FP16 输入配合 FP32 累加。而且两者都以最大化 Tensor Core 利用率为目标。如表所示，CUTLASS 在约 2% 的差距内追平甚至超过手工调优的 MMA 性能。尽管在这个例子中 CUTLASS 多用了几个寄存器、并把共享内存翻了一倍，但它仍远在硬件限制之内。这些微小的增加不影响占用率。

> 寄存器和共享内存用量上的这些微小差异，源于 CUTLASS 为了灵活性而对核函数做了泛化。虽然可以通过手工调优把这些差异优化掉，但在大多数情况下，额外的复杂性很可能不值得——而且 CUTLASS 的性能与手工调优版本几乎完全相同。

它只需要几行模板代码，而不是数周的底层调优。此外，CUTLASS 模板已经支持 FP4、FP8、FP16 和 TF32 操作数类型。而且它们能把 bias-add（偏置加）和激活等常见后处理操作融合进同一个核函数。

> 另外请记住，CUTLASS 模板会透明地使用线程块对（thread block pairs）、多 SM 分块，以及带分布式共享内存（DSMEM）的 TMA 多播（multicast）来最大化数据复用，如前所述——这些将在下一章详细介绍。

这与编写自定义 MMA 核函数形成对比：后者需要手动选择分块大小、编写异步拷贝循环、管理双缓冲、实现 warp 专门化流水线，以及线程块簇分块。所有这一切，CUTLASS 都会自动替你完成。

诸如 cuBLAS 之类的优化库就构建在 CUTLASS 之上。而像 PyTorch 这样的高层库会为许多核函数调用这些优化库。在前面的融合注意力示例中，我们展示了 TorchInductor 分派了一个 CUTLASS 融合注意力核函数，它使用了完全相同的双缓冲 TMEM 流水线。这带来了 98% 的 Tensor Core 利用率和近乎为零的内存停顿。

> 随着 PyTorch 和其他高层库中越来越多的算子在底层采用 CUTLASS，你无需自己编写任何 CUDA C++ 代码，就能利用这些相同的优化。

仍然可能存在你需要手写 MMA 核函数的场景——例如，当你需要一种高度专门化的数据布局，或一种 CUTLASS 尚未支持的独特融合模式时。

在这些情况下，你就需要自己实现这份复杂性。你首先要选择一个能放入 TMEM 的分块大小（例如 128 × 128 FP16），然后使用 <cuda/pipeline> 为每个分块执行异步内存拷贝（cp.async）指令。

接着，你需要实现 warp 专门化的 MMA 循环，并对 TMEM 做双缓冲以隐藏 DRAM 延迟。最后，你要把任何自定义的后处理步骤（如 softmax 和逐元素非线性）交错编排——如果可能，全部放在同一个循环内。

不过，对于几乎每一种标准 GEMM 或融合注意力场景，CUTLASS 及构建于其上的库都是推荐的做法。

它基于模板的设计、针对特定 GPU 的调优，以及对 TMEM 和 TMA 流水线的内建支持，通常能在受支持的形状上实现较高的 Tensor Core 利用率。这让你能以最小的开发投入，在某些情况下达到 96%–98% 的 Tensor Core 利用率。

依托 CUTLASS 的自动优化，你可以把时间花在模型架构、数值精度策略和端到端性能上。CUTLASS 让你有信心：你的底层张量操作会以接近峰值的算术强度运行，并针对你的特定 GPU 硬件做了优化。

> NVIDIA 会持续更新 CUTLASS、cuBLAS 等库，以利用 FP8、FP4、线程块簇对、TMEM 等最新硬件特性。使用这些库能让你免于为每一代新 GPU 重写新核函数。切换到新的 GPU 架构时，请始终检查是否有新版 CUTLASS。

## 用于微优化的内联 PTX 与 SASS 调优

对于愿意越过 C++、深入底层微优化（microoptimization）的人，CUDA 允许内联 Parallel Thread Execution（PTX）代码和 SASS（NVIDIA 的汇编语言），以榨出那些可能被留在桌面上的最后一点性能。

这是真正的高阶领域，因为 CUDA 编译器在优化上已经相当出色。但在某些极端情况下，你可以手工调度汇编指令——或使用专用指令——在非常特定的场景下换取一小部分性能提升。

有了 PTX 和 Streaming Assembler（SASS），你还能启用高层 CUDA 语言尚未暴露的特性。现代 GPU 通常不会引入激进的新汇编指令，但它们确实提供了定制调优的机会。例如，你可以调整 GPU 的缓存策略、修改 CPU-GPU 统一内存访问的协调方式，以及实现其他细粒度的微优化。

> PTX（读作“pee-tex”）是一种底层的并行线程执行虚拟机，把 GPU 暴露为一台并行计算设备。它为 NVIDIA GPU 提供编程模型和指令。高层编译器（如 CUDA C++）生成 PTX 指令，再被翻译为原生的目标架构指令。SASS 则是真正在 NVIDIA GPU 硬件上原生执行的底层汇编语言。

举个例子，设想有一段代码，你明知某个特定的指令序列会是最优的，但编译器却没有生成该特定序列。常见场景包括：使用没有直接 CUDA 内建函数的 GPU 指令、在特定访问上施加内存加载修饰符（缓存提示，cache hints）、在精确的位置插入内存栅栏或屏障，或手动重排指令以避免流水线停顿。

另一种用途是读取特殊寄存器或状态，例如 SM ID、warp lane ID 等，它们可能没有高层 API。内联 PTX 让你可以用 asm() 语句把汇编直接嵌入你的 CUDA C++ 代码。你可以通过为汇编代码指定输入和输出来混用 C++ 与 PTX。编译器随后会把你的 PTX 指令并入最终的 SASS。

我们来看一个简单的例子，它用一条内联 PTX 指令把一个全局内存地址预取到 L2 缓存。这里我们使用 PTX 指令 cp.async.bulk.prefetch.global 在核函数一侧进行预取：

```
__global__ void PrefetchExample(const float *in, float *out) {
    // ... assume idx is our thread’s data index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Manually prefetch the next cache line (128B) of in[] into L2:
    // Prefetch 128B from global to L2.
    // Address must be 16B-aligned
    // and size is a 16B multiple.
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                 :: "l"(in + idx + 32), "n"(128));
    float x = in[idx];
    // (do some work here before using in[idx+32] to give time for prefetch)
    out[idx] = x;
}
```

在这个片段中，内联 PTX cp.async.bulk.prefetch.L2.global [%0] 使用我们提供的地址操作数（in + idx + 32 字节，即向前 32 个 float），并向 L2 发起一次预取。我们把它标记为 volatile，以确保编译器不会把它优化掉。

这些 PTX 指令会被注入机器码。像这样使用内联汇编能给我们非常细粒度的控制。例如，我们可以预取到 L2 或 L1（通过使用 .L1），或选择距离（本例中为向前 32 个 float）。

这本质上就是 __prefetch_async 很可能编译后所对应的形式。更一般地，我们可以用内联 PTX 来控制普通加载的缓存行为。例如，我们可能写 asm("ld.global.cg.f32 %0, [%1];" : "=f"(val) : "l"(ptr)) 来以 .cg（“cache global”）修饰符加载一个 float。

在某些架构上，这意味着我们希望把数据缓存到 L2，但绕过 L1 缓存。如果我们知道某次访问正在冲刷 L1、而我们更愿意只用 L2，这会有帮助。通常，编译器的默认选择可能是缓存到 L1（.ca），但我们可以用 PTX 覆盖编译器的决定。

> 在现代架构上做 L2 预取时，凡可用之处请使用 cp.async.bulk.prefetch.tensor.L2。相比使用未文档化的内建函数，这是更可取的做法。无论如何，知道存在这一能力都是有用的。

内联 PTX 有帮助的另一个方面是指令调度（instruction scheduling）。默认情况下，编译器会以它认为最优的顺序发射指令。但你可能会发现某种情况，你想更有效地交错各项操作。

例如，假设你有两个独立的内存加载，然后要两次使用这些结果。编译器可能会发射 load1、然后 use1、然后 load2、然后 use2。但也许更好的指令调度是先执行 load1 和 load2（背靠背），再使用两个结果。这样可以让内存延迟重叠。

通过为这些加载编写内联 PTX，你可以强制它们提前执行，然后再做计算。这是前面讨论过的手动提升 ILP 的一种形式。实践中，编译器在这里通常已经做得不错，因为现代编译器会尝试用其他独立指令填满加载延迟。但内联 PTX 与 SASS 汇编能给你确定性。

在共享 CPU 与 GPU 内存的现代 CPU-GPU 超级芯片（superchip）上，如果你在管理一个 GPU 轮询由 CPU 更新的某个内存位置的工作负载，这种细粒度控制可能会有用。这里，你可以把适当的内存栅栏（如 membar.sys 或 __threadfence_system()）与加载和存储上所需的缓存操作符搭配使用，以在预期的作用域上确保一致性。这是高层 CUDA 可能不会直接暴露的东西。

> PTX 通常是前向兼容（forward compatible）的；然而，SASS 汇编会随每一代 GPU 架构而变化。

你还可以用内联 PTX 来利用特殊寄存器。例如，尽管 SM ID（线程块正在其上运行的 SM）没有 C++ 内建函数，你可以用 asm("mov.u32 %0, %smid;" : "=r"(smid)) 来获取 SM ID。这种灵活性对调试和工作划分很有用。

例如，一些开发者在持久化核函数中使用 %smid，让每个 SM 只有一个块去做某些工作。这实际上执行了手动的 SM 划分，超出了 CUDA C++ API 所提供的能力。

如果你的代码在算法层面已经优化得很好，那么内联 PTX/SASS 带来的收益往往只是渐进式的，在大多数情况下量级为几个百分点。例如，在一个访存受限的核函数中，你可以精心地展开并调度指令，以减少 load-to-use 延迟气泡，通过用 PTX 实现两条独立的加载流，或许能看到 5%–10% 的加速。在这种情况下，编译器可能会更保守。

在计算受限的场景中，你可能会用内联汇编去使用一条更快的数学指令，而不是一条更精确的指令。CUDA 为此提供了快速数学内建函数，包括 __sinf()。

在 C++ 内建函数可用之前，编写矩阵乘核函数的开发者有时会嵌入 PTX 指令来使用 Tensor Core。如今，我们已经有了用于此目的的更高层内建函数。但简而言之，汇编让你一旦得知某项硬件特性就能立即使用它——而无需等待 CUDA 支持它。
