# 第 9 章 提升 CUDA 核函数效率与算术强度

即便你已经通过大规模并行和高 ILP 完全隐藏了延迟，核函数的性能仍可能受限于它每次访存所完成的有用工作量。*算术强度*（arithmetic intensity），也称为*运算强度*（operational intensity），衡量的是从内存传输的每字节数据上执行了多少次浮点运算，即每字节 FLOPS。

新一代 GPU 的计算吞吐正在远远超越内存带宽的增长。这一日益扩大的差距意味着，提高算术强度比以往任何时候都更为关键。更高的算术强度表明核函数在每取回一个字节时完成了更多计算，而这正是充分发挥 GPU 计算能力的关键所在。

算术强度是 Roofline 性能模型（Roofline model）中的一项关键指标。Roofline 模型是一个实用的可视化工具，它以核函数性能（FLOPs/sec）对算术强度（FLOPs/byte）作图。它展示了内存带宽和计算吞吐的硬件上限（屋顶，roofs），使我们能够判断一个核函数是访存受限（memory bound，性能受限于内存传输）还是计算受限（compute bound，性能受限于 ALU 吞吐）。

在实践中，你可以使用 Nsight Compute 之类的工具生成 roofline 图，它包含一个 Roofline 分析视图。借助这些工具，你可以确认核函数最初是访存受限还是计算受限——然后在做优化的过程中持续剖析并验证改进效果。

我们的目标是把核函数推向计算受限的区间，充分发挥 GPU 不断增长的计算能力。Roofline 性能模型能够恰当地指引你的优化朝这一目标推进。

正如前一章所示，roofline 图用一条水平线表示硬件的峰值计算吞吐（屋顶，roof）——而从原点出发的一条对角线则表示受内存带宽限制的峰值可达吞吐。核函数的算术强度决定了它落在 x 轴上的位置，其性能则可与这些上限作对比，如图 9-1 所示。

![图 9-1. Roofline 模型示例（GFLOP/s 对以 FLOPs/byte 为单位的算术强度）](AI%20Systems%20Performance%20Engineering-ch9_images/figure-9-1.png)

一个算术强度低（即每移动一字节数据只做很少数学运算）的核函数会是访存受限的。此时，核函数的速度被硬件的内存带宽所限制，因为 GPU 大部分时间都在等待数据，而不是在做数值运算。

反之，一个算术强度很高（即每移动一字节做许多次 FLOPs）的核函数则会是计算受限的，因为它把 ALU 和 Tensor Core 用到了接近峰值的水平。此时，核函数的内存带宽占用便退居次要地位。

我们的目标始终是在可能的情况下提高算术强度，即在每传入/传出全局内存的一字节数据上完成更多的计算工作（每字节 FLOPs）。你可以借助多种技术来提高算术强度，例如用循环分块（loop tiling）来复用数据、用片上 L1/共享内存来实现复用，以及把多个核函数融合为一个，从而避免中间结果被写回全局内存。

> 现代编译器框架，例如 PyTorch 的 TorchInductor，会自动完成其中的一部分优化，把计算保留在 GPU 上、减少片外内存流量并提高有效算术强度。然而作为开发者，你有时仍需手动组合这些技术，或编写自定义 CUDA 核函数，以确保数据在被逐出缓存之前得到最优复用等等。

你还可以使用更低精度的数据类型（FP16、FP8、FP4）来减少内存传输量——并利用 Tensor Core 来提高每秒 FLOPs。两者结合，将提高每字节 FLOPs 比率，从而提升算术强度。接下来，我们讨论其中的一些技术。

请记住，并非每种工作负载都能轻松提高其算术强度。它受算法特性的约束。不过，你应当留意任何可以改进算法、复用数据、融合操作以及增大批大小的机会，从而在不改变算法结果（例如精度）的前提下提高算术强度。

## 多级微分块与软件预取

正如第 7 章所讨论的，分块（tiling，又称 *chunking* 或 *blocking*）与数据复用是提高算术强度的有效手段。在那一章中，我们展示了把 A 和 B 的一个小子矩阵（tile）载入共享内存，如何让从全局内存取回的每一字节能够以静态随机存取存储器（static random-access memory，SRAM）的速度用于许多次乘加运算。

每当你重构代码，使每个元素只加载一次却被使用几十甚至几百次时（如分块的情形），你就把每字节 FLOPs 比率乘上了复用因子（reuse factor）。举例来说，在典型的矩阵乘中，A 和 B 的一个 32 × 32 的 tile 会为共享内存中的每个元素产生 1,024（1,024 = 32 × 32）次独立的乘法。这样一来，相比每次运算都直接从 DRAM 取回每个元素，算术强度便得到了提升。

除了简单的共享内存分块之外，你还可以用多级分块（multilevel tiling）进一步提高强度并暴露更多 ILP。使用多级分块时，在把一个 tile 暂存进共享内存后，让每个线程用 float4、<half2> 等向量化类型把微分块（microtiles）载入寄存器。这样，重复的运算就完全发生在寄存器中。多级分块的一个示例如图 9-2 所示。

![图 9-2. 全局内存（DRAM）、共享内存（SMEM）与寄存器之间的多级分块](AI%20Systems%20Performance%20Engineering-ch9_images/figure-9-2.png)

这种 SM 内部的复用（register → SMEM → DRAM）减小了每一层级的工作集——并最大限度地降低了片外流量。一如既往，务必在填充共享内存时合并（coalesce）全局读取，并对共享数据进行填充/交错（pad/swizzle）以避免内存 bank 冲突，正如我们在第 7 章所讲的那样。

在现代 GPU 上，这些内层循环的分块步骤往往通过使用 MMA 片段 API（fragment APIs）来覆盖完成。硬件使用 Tensor Core 指令在共享内存和 Tensor Memory（TMEM）之间隐式地搬运数据。TMEM 的使用由编译器和库来管理。在现代 GPU 上，tcgen05 指令在共享内存和 TMEM 之间隐式地暂存数据。它们使用一个独立的 TMEM 地址空间。不过在实现某些算法并需要显式控制时，开发者仍然可以用 cp.async 或 TMA 手动把 tile 搬入共享内存。

一项与之密切相关的技术是软件预取（software prefetching），它常以双缓冲（double buffering）的形式实现。举例来说，与其等到当前 tile 的计算完成，你可以为下一个 tile 发起异步加载，把它读入共享内存。这会让 DRAM → 共享内存（SMEM）的传输与正在进行的算术运算相互重叠。精心设计的预取可以显著减少停顿时间并提升吞吐。其思想是让数据传输与计算相互重叠，从而使 ALU 永远不会因等待数据而空转。

当在 Grace Blackwell 这类 CPU-GPU 超级芯片（superchip）上使用统一内存（unified memory）时，你可以用 cudaMemPrefetchAsync() 来提示某个 tile 很快就会被用到。这会向运行时提示，让它通过 NVLink-C2C 迁移页面。不过预取只是一个提示，并非保证。你仍然需要确保正在重叠传输并适当地同步，以避免缺页（page fault）停顿。以这种方式让数据搬运与计算相互重叠，可确保每当需要新的 tile 时 ALU 都能持续获得数据供给。这进一步隐藏了内存延迟——并提升了实际达到的 FLOPS。

> 统一内存简化了开发，但未必能带来最佳性能。资深用户往往更倾向于使用显式的 cudaMemcpy 或固定内存（pinned memory）分配，以彻底避免页迁移（page migration）开销。

简而言之，来自 DRAM 的每一字节在片上层级（寄存器或共享内存）被复用的次数越多，你的算术强度就越高。而更高的算术强度会把核函数推得更接近计算受限的区间。

## 使用线程块簇进行分块

在现代 GPU 上，你可以借助来自协作组（Cooperative Groups）的 CUDA 线程块簇（thread block clusters，将在第 10 章讨论）来扩展分块复用的思路。它们允许多个线程块使用分布式共享内存（distributed shared memory，DSMEM）来共享数据，如图 9-3 所示。

![图 9-3. 一个 CGA 内的多个 CTA（线程块）共享的 DSMEM](AI%20Systems%20Performance%20Engineering-ch9_images/figure-9-3.png)

我们会在下一章详细介绍 CGA 与线程块簇，但这里值得先提一提，因为它们可以直接提高算术强度。举例来说，一个由四个线程块组成的簇可以利用 Tensor Memory Accelerator（TMA）的多播（multicast）特性协同加载一个 tile，如图 9-4 所示，该图用四个 CTA 来演示这一机制。

![图 9-4. 对于这四个（2 × 2）线程块簇，A 和 B 的每个 tile 都通过多播被同时载入四个 CTA（线程块）（来源：https://oreil.ly/EEO_O）](AI%20Systems%20Performance%20Engineering-ch9_images/figure-9-4.png)

每个 tile 被划分到这四个线程块上，从而使该 tile 的全局内存流量在整个簇上被均摊。这些 tile 只取回一次，随后被全部四个线程块复用。

当四个 CTA 通过多播复用同一份数据时，线程块簇在 2 × 2 的簇中可将全局内存流量最多降低到原来的 1/4（4×）。此外，线程块簇通过降低分母（即从全局内存搬运的字节数）来提高每个 GPU 的算术强度。这类簇的一种特化形式称为*线程块对*（thread block pairs），稍后会在使用 Tensor Core 分块的语境中讨论。

> 当你选择使用超出默认可移植上限（8 个 CTA）的不可移植簇尺寸时，Blackwell 支持最多 16 个线程块的线程块簇。要启用这一点，请在核函数上设置 cudaFuncAttributeNonPortableClusterSizeAllowed 属性。更大的簇可以提高复用，但可能降低占用率，因此在启用 16 之前请先做剖析。这可以支持更大的跨 SM tile，从而最大化数据复用（16×）并以相近的倍数提高算术强度。

## 核函数融合

提高算术强度的另一种方式是把多个操作——或多次循环迭代——融合为一次操作。通过把多个核函数融合在一起，从内存加载的数据在被写回之前可以用于多次计算和迭代。

与之类似，上一节讨论的循环展开（loop unrolling）允许单个线程在每个已加载的数据元素上执行更多计算，但代价是更多的寄存器占用。过度融合会增加每线程的寄存器压力并降低占用率，因此这里存在一个权衡。

请务必对融合后的核函数进行剖析。如果寄存器占用变得过高并开始溢出（spilling）到本地内存，融合带来的收益可能会被额外的内存流量所抵消。不过，如果你找到了恰当的平衡点，就能提高每移动一字节所做的 FLOPS，而当内存带宽是限制因素时这是有益的。

现代深度学习框架可以通过其即时编译器（JIT）和图优化器自动完成融合与展开。举例来说，PyTorch 的 torch.compile，尤其是 TorchInductor，可以自动融合一连串的逐元素操作（elementwise operations）。我们会在第 13 章和第 14 章介绍 PyTorch 编译器。

> *逐元素操作*（elementwise operations），也称为*逐点操作*（pointwise operations），对张量的每个元素独立地施加一个简单的计算。

融合这些逐元素操作，通过把中间值保留在片上而消除了不必要的内存流量。这提高了每从全局内存取回一字节所完成的工作量——从而提升算术强度。

例如，朴素实现会启动两个核函数。第一个核函数读取 x 并把 y 写入全局内存。第二个核函数读取 y 并写入 z：

```
y = sin(x);
z = sqrt(y);
```

在这里，每个元素被触碰两次：一次在 sin(x) 之后，一次在 sqrt(y) 之后。因此，每个核函数的算术强度都非常低，因为它对每个元素——每次加载/存储操作——只执行一个昂贵的数学函数（一条多周期的 ALU 指令）。相比之下，融合核函数在单趟中完成同一组操作：

```
z[i] = sqrt(sin(x[i]));
```

每个 x[i] 只加载一次，然后 sin 和 sqrt 都在寄存器中完成，只有最终的 z[i] 才写入内存。由于中间量 y 从不写出到全局内存，有效的每字节 FLOPS 大幅跃升，使该操作更接近计算屋顶（compute roof）。

> 一条经验法则：如果数据会被同一个线程块内的线程读取超过一次，那么通常值得把数据暂存进共享内存，以消除冗余的全局加载。这有助于把你的核函数从访存受限区间提升到计算受限区间，从而更好地利用充裕的 GPU FLOPS。

融合减少了全局内存流量并提高了算术强度，因为现在每个元素在每次读写内存操作时都要经历两个数学运算。在我们的例子中，每个元素的 FLOPS 翻了一番（sin + sqrt），而由于没有中间写入，内存流量大约减半。这带来了显著更高的算术强度，即 FLOPS/byte。

为把这一点讲透，让我们用一个具体的例子来演示算术强度。假设我们想对一个 2D 张量 x（形状 [batch, hidden]）中每个长度为 hidden 的行做 L2 归一化。对每一行 b，计算单个范数 norm_b = sqrt(Σ_i x[b,i]*x[b,i] + ε)，然后对所有 i 写入 y[b,i] = x[b,i] / norm_b。

一个朴素实现会在一个核函数里做平方，在第二个核函数里把每一行归约为一个标量，再在第三个核函数里做除法。这将需要多次核函数启动，并向 HBM 写入中间结果。

假设这 4 个核函数中的每一个都需要 1 次 FLOP 的计算。那么，这 4 个核函数各自的算术强度就是 1 FLOP 对 12 字节（2 次 float 读取、1 次 float 写入），即 0.083 FLOPS/byte。

作为替代，我们可以把这 4 个核函数融合成一个核函数，从而提高算术强度。手动融合的核函数代码如下所示：

```
__global__ void fusedL2Norm(const float* __restrict__ x,
                            float* __restrict__ y,
                            int hidden) {
  extern __shared__ float sdata[];        // reduction buffer
  const int batch   = blockIdx.x;         // one block per batch row
  const int tid     = threadIdx.x;
  const float* batch_ptr = x + size_t(batch) * hidden;
  // 1) Accumulate sum of squares into shared memory
  float local = 0.f;
  for (int i = tid; i < hidden; i += blockDim.x) {
    float v = batch_ptr[i];
    local   = fmaf(v, v, local);          // v*v + local
  }
  sdata[tid] = local;
  __syncthreads();
  // 2) Parallel reduction to sdata[0]
  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) sdata[tid] += sdata[tid + offset];
    __syncthreads();
  }
  // 3) Normalize (guard tiny norms)
  float norm = sqrtf(sdata[0]);
  float inv = rsqrtf(sdata[0]); // prefer inverse
  float* out_batch = y + size_t(batch) * hidden;
  for (int i = tid; i < hidden; i += blockDim.x) {
    // multiply by inverse (rsqrt) vs. divide by sqrt
    out_batch[i] = batch_ptr[i] * inv;
  }
}
```

在这个融合核函数中，每个线程会遍历它负责的那一段 x[b,*] 两次——一次用于累加局部的平方和，一次用于写出归一化后的输出——因此每个元素的全局流量约为 12 字节（两次读取 + 一次写入）。就每个元素而言，该核函数在归约阶段做约 1 次乘法 + 1 次加法，在归一化阶段做 1 次乘法。

sqrt 和 rsqrt 的开销被均摊到整行之上。就 roofline 定位而言，一个保守的算术强度约为 ≈3 FLOPs / 12 bytes ≈ 0.25 FLOPs/byte（再加上来自每行 sqrt 的微小 1/(hidden * 12) 贡献）。这让我们能够通过给每个线程分配多个元素来提高 ILP，从而隐藏 sqrt 和 rsqrt 的延迟。

> 此外，如上面的代码所示，我们计算逆平方根（rsqrtf）并做乘法，而不是做除法。这是一种常见的微优化——尤其适用于热点内层循环。其思路是用高吞吐的乘法指令流来替换缓慢的除法指令流。我们同时也把 sqrtf 换成了更廉价的 rsqrtf 近似。这些都属于微优化，因为总体上这条流水线是访存受限而非计算受限——但它仍值得指出。这里还有另一项未展示的优化。它的做法是在线程块内由单个线程做一次 rsqrtf/sqrtf，然后用共享内存把这个标量结果广播给其他线程。这对提升性能的影响更大。关于这项优化的更多细节，请参见本书的 GitHub 仓库。

与向 HBM 写入中间结果的朴素三核函数流水线（平方 → 归约 → 除法）相比，融合版本至少省去了一次全局的写/读往返和一次启动屏障。因此，尽管其每元素的 FLOP/byte 仅约为 0.25，它在实践中的强度和运行时间都要好得多。这归功于延迟的节省和缓存局部性的改善。

在实践中，通过把三个独立核函数合并为一个，这个单一的融合核函数由于更高的算术强度（FLOPS/byte）、更好的缓存局部性以及更低的启动开销，其执行速度比一系列独立核函数更快。

融合不仅提高了算术强度、把核函数更多地推向 roofline 的计算受限一侧，还节省了内存带宽。在朴素的多核函数版本中，我们不得不把中间结果写入全局内存，并在下一个核函数中把它们读回。而在融合版本中，中间结果（例如 ai*ai）从不需要离开线程的寄存器。

在这段代码示例中，加法可以直接使用那些寄存器来计算总和。随后 sqrt 可以使用这个总和——全程无需任何额外的全局内存流量。只有最终结果才被写回全局内存。

因此，融合核函数或许能以 12 字节的往返全局内存的数据搬运达成 4 FLOPS，而朴素的未融合方式在计入中间内存搬运之后，则要以加载和存储 36 字节才达成 4 FLOPS。这意味着更少的 DRAM 流量和更低的延迟。

这个简单的例子展示了融合如何提高核函数的算术强度和整体性能。让我们再看另一种提高算术强度的方式——利用 GPU 的 Tensor Core 硬件单元。

> 最先进的 GPU 核函数通过纵向融合（vertical fusion）来获得更高的算术强度，它把作用于同一数据的顺序操作组合起来——同时还借助横向融合（horizontal fusion），它把跨数据的并行操作组合起来。像 NVIDIA 的 CUTLASS 或 OpenAI 的 Triton（已集成进 PyTorch 的编译器后端 TorchInductor）这样的库，可以帮助你利用 Tensor Core、TMEM、TMA 等非常高效地实现这些不同类型的融合核函数。

## 结构化稀疏

在现代 GPU 上，2:4 结构化稀疏（structured sparsity）由 Sparse Tensor Cores 和 cuSPARSELt 在硬件中加速。2:4 意味着每连续四个权重中恰有两个非零。制造这种稀疏有时称为*剪枝*（pruning）。

通过把一半权重剪枝成 2:4 模式，现在每次内存加载所交付的、真正参与乘法的非零值就翻了一番。换句话说，你不再取回那些结果为零的权重。因此，你也不会在明知为零的东西上浪费一次矩阵乘操作，如图 9-5 所示。

![图 9-5. 2:4 结构化稀疏](AI%20Systems%20Performance%20Engineering-ch9_images/figure-9-5.png)

结构化稀疏是在模型训练完成之后应用的。模型被剪枝并针对推理做优化。剪枝与格式转换在 cuSPARSELt 之类的软件栈以及框架工具中完成。请注意，Transformer Engine 会加速受支持的稀疏执行，但在转换过程中并不强制稀疏。

剪枝与格式转换在软件中处理——通常通过 cuSPARSELt 和框架工具。在 PyTorch 中，使用 to_sparse_semi_structured() 把训练好的稠密模块转换为 2:4 稀疏格式，然后在 Sparse Tensor Cores 上部署稀疏 GEMM。

一旦你的模型完成转换，它就会调用运行在 Sparse Tensor Cores 上的优化稀疏 GEMM 核函数，而非标准核函数。对许多推理工作负载而言，Sparse Tensor Cores 相较其稠密版本可接近达到 2× 的加速——尤其在提交大批量输入时，因为这些批量能均摊核函数启动开销。

> 批处理（batching）是一种非常常见且实用的提高算术强度的方式。与其一次处理一项（连带所有相关的内存 I/O 等等），你可以在一趟中处理多项，从而使内存访问（例如加载权重等）在多次计算上被均摊。

这为稀疏加速的矩阵乘提供了足够的并行工作量，以隐藏处理索引或压缩表示所带来的任何开销。在较小的批量中，这类开销可能占主导地位，并限制你所观察到的加速幅度。

2:4 稀疏在使用大型矩阵乘时会产生最大收益，而大型矩阵乘在 LLM 这类基于 transformer 的模型中很常见。这是因为硬件能够充分利用专用的 Sparse Tensor Cores。这些 Sparse Tensor Cores 直接在硬件中对半宽数据进行运算。它跳过零值，并在相同的周期预算内对非零元素完成两倍的工作。

由于 Blackwell 上的计算能力增长得比 HBM 带宽更快，结构化稀疏是保持计算受限的绝佳方式。即便在一个 NVLink-C2C 让 GPU 能以极高吞吐从 CPU 内存流式取数的 Grace Blackwell 系统上，你仍然希望在每个已加载的 tile 上最大化每字节 FLOPS。

例如，通过以 2:4 模式剪枝掉 50% 的权重，你就确保了一半的内存流量永远都不需要。这会立即减少全局内存读取，并使有效算术强度提高近 2×。

NVIDIA GPU 在硬件中实现这种 2:4 结构化稀疏，使得每 16 个元素的块可以将 8 个元素置零。这正是用于把稀疏矩阵的 Tensor Core 吞吐翻倍的模式。在撰写本书时，没有其他任意稀疏模式能在硬件中获得这种特殊加速。

> 稀疏带来的加速以模型精度得到保持为前提。在实践中，剪枝之后进行微调（fine-tune）或仔细校准（calibrate）很重要。如此一来，你才能把精度损失降到最低。

在应用稀疏之前，重要的是先落实前面介绍过的那些基础优化：合并所有全局加载、用分块复用数据，以及融合逐点操作以消除多余的内存往返。一旦这些基础到位并经过验证，结构化稀疏就能为推理再提供一层加速。

> 结构化稀疏通常适用于推理工作负载。在训练期间，梯度并不从 2:4 稀疏中获益。此外，在梯度更新中维持稀疏很复杂。因此，建议将其用于那些你已预先剪枝并校准过模型的部署场景。NVIDIA 的 2:4 稀疏 Tensor Core 特性主要用于推理。训练方面的支持有限，且依赖于模型和框架。在依赖它之前，请先在你的软件栈中验证支持情况。

## 重算与内存权衡

此外，与其存储或加载预先算好的值（例如 x²），当内存带宽是瓶颈时，不妨考虑按需重算它们。举例来说，在寄存器中反复计算 x*x 往往比从全局内存加载一个先前算好的 x² 更快。对廉价表达式的持续重算可以提高算术强度，并且在内存紧张时是一项有用的技术。

许多 LLM 推理引擎使用这项技术来节省内存。与其把大型激活张量（activation tensors）存进 HBM 稍后再读回，它们可以即时重算某些层和激活。这类似于模型训练语境中的*激活检查点*（activation checkpointing）。

重算提高了有效的 FLOPS/byte，并能把大模型塞进更小的内存中。此外，重算为更大的批大小腾出了内存，并以少量额外的 FLOPS 换取内存流量的显著减少。

## PyTorch 与算术强度

在 PyTorch 中，这些思路的许多都会被自动应用。如前所述，PyTorch 编译器（将在第 13 章和第 14 章讨论）可以自动融合一连串逐元素操作——甚至一些归约。它利用执行图层面的优化来把数据保留在 GPU 上并尽可能地复用。

由于它在底层使用 cuDNN、cuBLAS 这类优化过的库，当你用 torch.matmul 执行矩阵运算时，PyTorch 会替你完成分块并使用共享内存。此外，PyTorch 的 scaled_dot_product_attention (SPDA)（应为 SDPA）可能会根据张量的形状和 dtype 分派到 FlashAttention、memory-efficient 或 cuDNN 后端。要控制后端选择，可以使用例如 torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION)。作为一名注重性能的开发者，你应当了解这些优化，以及如何验证它们何时被启用。

需要指出的是，尽管 PyTorch 能够识别并编译大多数操作，某些非标准操作或自定义 CUDA 操作可能得不到融合。在这些情况下，仍可能需要手动优化，例如融合、分块等等。

> 如果你在编写 PyTorch 代码，请优先使用融合操作以及执行多重计算的优化库调用，而非一长串单独的核函数启动。在实践中，这意味着使用 torch.nn.functional 激活或 torch.matmul 这样的高层操作，而不是在 Python 中编写许多细小的逐元素核函数。这些库会为这类高层操作调用高效的核函数。而编译器知道如何把它们与周围的操作高效地融合。

PyTorch 的嵌套张量（nested tensors），或称不规则张量（ragged tensors），让你无需填充就能表示一批变长输入。每个嵌套张量把变长序列打包进单个高效的底层缓冲区。NestedTensor 暴露了常规的张量接口，但它消除了不必要的零填充。因此，全局内存加载变得更高效，因为取回的每一字节都在计算中是有用的。

嵌套张量对于序列长度各异的 LLM 很有用。使用嵌套张量时，注意力和批量矩阵乘之类的操作只会从内存中取回必要的数据。这在 Roofline 模型上把你的核函数推得更接近计算受限区间，并有助于减少访存受限造成的停顿。其结果是更高的持续吞吐——尤其在对内存敏感的工作负载上。

> 在实践中，嵌套张量需要就算子覆盖度和性能特性做仔细验证。支持情况依工作负载而定，加速幅度依形状而定。你可以用有代表性的序列长度分布和注意力模式来验证端到端收益。请同时剖析内存流量和核函数时间。

简而言之，PyTorch 暴露了多种机制来提高核函数的算术强度。重要的是理解这些选项，并判断哪种最适合你的工作负载。在现代 GPU 上提高算术强度的另一种有效方法是使用降精度的 Tensor Core。接下来我们就介绍这些机制。

## 混合精度与利用 Tensor Core

现代 NVIDIA GPU 在 Tensor Core 中实现了 TF32、FP16、FP8、FP4 和 INT8 等降精度计算。Blackwell 中的每个 SM 都有一块 256 KB 的片上 TMEM，专用于 Tensor Core 数据。它还有一个专门的 TMA 单元，用于在全局内存和共享内存之间异步拷贝 tile。随后 Tensor Core 指令（例如 tcgen05.mma）在共享内存和 TMEM 之间隐式地搬运操作数与累加器。这一设计以高吞吐向 Tensor Core 供给数据，并最大限度地减少停顿。相较于前几代直接在寄存器中累加的 GPU，Blackwell 基于 TMEM 的累加器有助于减轻寄存器压力。

在正确使用时，这些特性可以通过把算术强度（每字节 FLOPS）提高 2×、4× 乃至 8×，将一个曾经访存受限、张量密集的核函数转变为完全计算受限的核函数。你可以通过监视 Nsight Compute 中的 Roofline 图和 Stall Stats 来验证其影响。

Nsight Compute 的 *Speed of Light* 分析会展示访存受限的停顿原因，例如“Memory Throttle”和缓存未命中。当使用带低精度格式的 Tensor Core 时，这些都会显著下降。而且 Nsight Compute 集成了一张 Roofline 图，用来交叉核对算术强度是否已提高到能把你的核函数推向计算屋顶。

当你从 FP32 转向 TF32、FP16、FP8 或 FP4 这类低精度的 Tensor Core 核函数时，Nsight Compute 的 warp 停顿指标（Warp Stall）通常会显示与内存相关的停顿减少，同时依赖类或流水线类停顿相对增加。这表明算术强度提高了，执行从访存受限转向了计算受限。

### 用 TMEM 与 TMA 供给 Tensor Core

高吞吐张量计算的核心是 TMEM，即每个 SM 一块 256 KB 的 SRAM 缓冲区。不过在较高的层面上，程序员并不显式地分配或管理 TMEM。当你使用 Tensor Core 操作时，TMEM 由硬件或库来处理。TMEM 如图 9-6 所示。

![图 9-6. TMEM 通过累加部分结果（而非使用寄存器）来支撑 Tensor Core](AI%20Systems%20Performance%20Engineering-ch9_images/figure-9-6.png)

在底层，Blackwell 使用 tcgen05.mma 指令，它以 TMEM 作为操作数和累加器的存储。CUTLASS 和库核函数通过核函数配置以及 Parallel Thread Execution（PTX）汇编来管理所需的分配与使用。因此，Transformer Engine 使用 TMEM 来存放部分结果。这减轻了 MMA 对寄存器的依赖。

像 CUTLASS 这样的高层 API 会自动替你处理所有这些复杂性。在可能的情况下请使用 CUTLASS 及其他高层库，因为 CUTLASS 使用 tcgen05.* 这些 PTX 指令，它们实现了 Tensor Core 矩阵操作以及内存加载/存储接口。每当你用 CUDA MMA 内建函数或 CUTLASS GEMM 启动一次 Tensor Core MMA 操作时，其实现都会通过共享内存和 TMEM 管理操作数的搬运。TMA 在全局内存和共享内存之间流式传输 tile，而 Tensor Core 指令则在共享内存和 TMEM 之间隐式地搬运操作数。

> Nsight Compute 内置了 roofline 与 speed-of-light 分析，用以确认你的核函数在采用低精度 Tensor Core 路径后是否已从访存受限转向计算受限。

随后 Tensor Core 指令作为 MMA 流水线的一部分，在共享内存和 TMEM 之间搬运操作数。这一切都在幕后发生，无需显式的用户代码。如此一来，数据便被暂存到 Tensor Core 所需之处。要执行这次从全局内存到共享内存的数据传输，可以使用 TMA，或使用配合 CUDA Pipeline API（<cuda/pipeline>）的 cuda::memcpy_async。在代码中，用 cuda::memcpy_async 和 CUDA Pipeline API 实现一条简单的两级流水线大致如下所示：
