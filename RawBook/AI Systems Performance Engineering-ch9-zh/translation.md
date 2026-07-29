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

> 在有益时，CUTLASS 还会利用线程块簇，通过跨多个 SM 分块来得到更大的有效分块。我们将在下一章介绍线程块簇。

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

> 另外请记住，CUTLASS 模板会透明地使用线程块对、多 SM 分块，以及带分布式共享内存的 TMA 多播来最大化数据复用，如前所述——这些将在下一章详细介绍。

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

对于愿意越过 C++、深入底层微优化（microoptimization）的人，CUDA 允许内联 Parallel Thread Execution（PTX）代码和 SASS（NVIDIA 的汇编语言），以榨出那些原本可能被白白浪费的最后一点性能。

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

在共享 CPU 与 GPU 内存的现代 CPU-GPU 超级芯片上，如果你在管理一个 GPU 轮询由 CPU 更新的某个内存位置的工作负载，这种细粒度控制可能会有用。这里，你可以把适当的内存栅栏（如 membar.sys 或 __threadfence_system()）与加载和存储上所需的缓存操作符搭配使用，以在预期的作用域上确保一致性。这是高层 CUDA 可能不会直接暴露的东西。

> PTX 通常是前向兼容（forward compatible）的；然而，SASS 汇编会随每一代 GPU 架构而变化。

你还可以用内联 PTX 来利用特殊寄存器。例如，尽管 SM ID（线程块正在其上运行的 SM）没有 C++ 内建函数，你可以用 asm("mov.u32 %0, %smid;" : "=r"(smid)) 来获取 SM ID。这种灵活性对调试和工作划分很有用。

例如，一些开发者在持久化核函数中使用 %smid，让每个 SM 只有一个块去做某些工作。这实际上执行了手动的 SM 划分，超出了 CUDA C++ API 所提供的能力。

如果你的代码在算法层面已经优化得很好，那么内联 PTX/SASS 带来的收益往往只是渐进式的，在大多数情况下量级为几个百分点。例如，在一个访存受限的核函数中，你可以精心地展开并调度指令，以减少 load-to-use 延迟气泡，通过用 PTX 实现两条独立的加载流，或许能看到 5%–10% 的加速。在这种情况下，编译器可能会更保守。

在计算受限的场景中，你可能会用内联汇编去使用一条更快的数学指令，而不是一条更精确的指令。CUDA 为此提供了快速数学内建函数，包括 __sinf()。

在 C++ 内建函数可用之前，编写矩阵乘核函数的开发者有时会嵌入 PTX 指令来使用 Tensor Core。如今，我们已经有了用于此目的的更高层内建函数。但简而言之，汇编让你一旦得知某项硬件特性就能立即使用它——而无需等待 CUDA 支持它。

Nsight Compute 有助于指导汇编调优。通过查看“SASS throughput”（SASS 吞吐）指标与“Warp Stall Reasons”（warp 停顿原因），你可能会发现大量停顿源于内存依赖。你可以尝试按前文所述重排这些加载操作。

改动之后，你会期望看到更少的“Stall Memory Dependency”（内存依赖停顿）——或许还有更高的指令发射率，从而每周期执行更多指令。请注意，汇编级调整是费时费力的——而且会降低代码可移植性。

通常只有在那些你已用尽更高层次优化手段的极热内层循环里，这么做才值得。此外，一旦更换 GPU 世代，就可能需要重新调优，并验证你的假设是否依然成立，包括内存延迟、缓存行为等。

为了说明可能的微优化，设想这样一个场景：某个核函数已经相当优化，但仍残留一个瓶颈——一段做整数索引计算与内存加载的紧凑循环。

你可以用内联 PTX（inline PTX），通过 mad.wide.u32 在一条指令内计算出字节地址（base + index * stride）。接着，以 ld.global.cg 发射该加载，从而对这种流式访问模式绕过 L1。其结果是：循环使用更少的指令，并避免 L1 驱逐。在本例中，我们把该核函数从 1.07 ms 压到 1.0 ms，挤出了约 7% 的加速。表 9-2 汇总了一个假想的前后对比：优化后的 CUDA C++ 与带手工调度和缓存提示的手写 PTX。

表 9-2. 优化后的 CUDA C++ 与带手工调度和缓存提示的手写 PTX 对比

| 版本 | Warp 停顿（内存） | 发射 IPC | 核函数耗时 | 加速比 |
| --- | --- | --- | --- | --- |
| 优化后的 C++（编译器调度） | 占周期的 35% | 1.5 | 1.07 ms | 1.0×（基准） |
| 手工调优的 PTX（手工调度与提示） | 占周期的 20% | 1.6 | 1.00 ms | 1.07× |

这里可以看到，调优之后，通过重叠加载，我们核函数的内存停顿下降了，缓存提示也削减了部分延迟。此外，我们提高了 ILP 与每周期指令数（IPC）。这使得整体核函数执行时间净改善了 7%。

这些数字符合在一个已优化核函数上进行手工汇编所能预期的结果。在某些情况下，如果编译器做出了糟糕的选择，收益可能更大——而你可以把修正实现出来。反之，若编译器已经选出了最优实现，收益本质上就是零。用错误的汇编排序反而可能损害性能，因此必须对每一处改动都做实验与剖析。

内联 PTX 与 SASS 汇编可视为最后手段的优化工具。它以复杂度为代价，提供了终极的控制力。建议仅在以下情形动用这一最后手段：你需要某个 CUDA C++ 中无法访问的硬件特性或指令——或者你已精确定位到一小段代码，其编译器调度尚有改进空间。这类例子包括自定义内存访问模式（缓存提示、预取）、细粒度同步或栅栏，以及在新指令被官方正式支持之前抢先使用它们。

在应用内联汇编时，用剖析来验证其效果尤为重要。你希望看到停顿原因或指令数如预期般下降。同时，你应把这类代码隔离出来并写好文档。它很可能需要针对新的 GPU 架构做更新——尤其是当你使用 SASS 汇编时，因为它并不总是前向兼容。

> 虽然 PTX 比 SASS 更稳定，但某些硬件变更出于性能考虑仍可能要求你更新内联 PTX。

内联 PTX/SASS 调优属于专家级微调，用于榨出额外的延迟余量或强制某种特定调度。它能带来适度的加速并启用某些自定义行为，但应在用尽所有其他高层次优化之后再考虑。举例来说，你可能想为那些运行数百万次的关键循环手工打造汇编。至少，这是一种精确了解硬件如何执行你代码的有效途径。

简而言之，谨慎使用内联 PTX/SASS，并勤于剖析。收益是真实的，但通常是增量式的。维护成本则高得多。对大多数使用场景而言，依赖 CUDA 内置的优化——或像 CUB 与 Thrust 这样高度调优的库——很可能就已经足够。但知道在必要时你可以下沉到汇编、对 GPU 取得完全控制，这一点很有价值。

## DeepSeek 使用内联 PTX 进行内存分配优化

一个广为人知的自定义 PTX 案例来自 DeepSeek 的 DeepEP 专家并行（expert parallelism）库。该库使用了一条定制（bespoke）的 PTX 指令 ld.global.nc.l1::no_allocate.l2::256b，通过绕过 L1 缓存分配、保护关键数据，并利用 256 字节的 L2 缓存块，来优化全局内存访问。它非常适合把大数据集直接流式送入 L2 缓存，而不扰乱 L1 缓存中频繁访问的内存操作。

这条指令并不属于 NVIDIA 官方的 PTX ISA 规范，而是由 DeepSeek 工程师“out-of-doc”（未见于文档）发掘出来的，用以在其受美国出口管制的 Hopper GPU 的 H800 变体上微调缓存行为。

我们来逐段拆解 ld.global.nc.l1::no_allocate.l2::256b 这条 PTX 指令。ld.global.nc 前缀发射一次非一致（noncoherent，nc）的全局内存加载（ld.global），而修饰符 l1::no_allocate 与 l2::256b 则指示硬件避免把数据分配进 L1 缓存（l1::no_allocate）。取而代之的是，它每次取 256 字节进入 L2（l2::256b）。

通过绕过 L1，这些加载可以把大块数据直接流入 L2，而不驱逐那些常被使用的、驻留在 L1 的数据。当你有热点工作集（working set）需要保留在 L1 中以获得低延迟内存访问时，这一点至关重要。

在实践中，诸如专家并行 all-to-all 通信核函数之类的流式工作负载能从这种做法中获益，因为它们往往在每次 dispatch 时对大块连续缓冲区恰好只读取一次。如果这些加载经过 L1，就可能驱逐那些仍被 SM 积极使用的较早缓存行。

通过以 256 字节对齐的块直接取入 L2，这条指令减少了不必要的 L1 流量。这有助于为访存受限的通信与计算操作同时维持高吞吐。

然而，这样使用 PTX 会带来一些风险，因为 ld.global.nc.l1::no_allocate.l2::256b 并不保证在各 GPU 世代间保持稳定。因此，依赖它的代码在未来架构上可能失效或产生错误结果。

DeepSeek 的 DeepEP 配置甚至包含了一个构建期开关 DISABLE_AGGRESSIVE_PTX_INSTRS=1，用于在出现兼容性问题时禁用这些激进指令。尽管 DeepEP 定制的 PTX 技巧能带来显著加速，但内联 PTX/SASS 仍应谨慎使用，并在每次升级到新的 GPU 架构时充分测试。

## 关键要点

你看到了如何通过把工作从缓慢的全局内存转移到更快的片上资源与计算单元，来发现并消除 GPU 核函数瓶颈。遵循“剖析、诊断、优化、再剖析”的循环，你就能把核函数从利用率不足或访存受限（memory bound）转变为计算饱和、高吞吐的例程。以下这些技术将帮助你充分发挥 GPU 的全部威力：

*用分块与融合提高算术强度* 为了提升每字节传输所产生的 FLOPS，用多级分块把数据暂存到共享内存与寄存器中。例如，把一个 32 × 32 的子矩阵加载进 SMEM，使每个元素在众多 FMA 中被复用。通过融合逐元素操作来合并连续的核函数，让中间结果留在片上。借助 CUDA Pipeline API 的 cuda::memcpy_async 进行异步内存加载，从而实现软件预取，让数据搬运与计算相互重叠。这样就能隐藏 DRAM 延迟。

*善用混合精度、Tensor Cores 与 Transformer Engine* 从 FP32 降到 TF32/BF16/FP16/FP8/FP4 可将权重流量削减 2× 到 8×。这相应地提升了算术强度。在 PyTorch 中，你可以用 torch.set_float32_matmul_precision('high') 与 torch.cuda.amp 来启用混合精度。在现代 GPU 上，TMEM 与 TMA 引擎会把数据块流式送往 Tensor Core。现代 GPU 还提供了 Transformer Engine，专门为 AI 工作负载利用这些更低精度。这可以进一步提高吞吐。

*利用 CUTLASS 获得高性能 GEMM 与融合核函数* 与其手工编写 MMA 循环，不如实例化 CUTLASS GEMM 模板，让它自动管理双缓冲、TMEM 暂存与 Tensor Core 流水线。这将产出一个性能与手工调优核函数相差不过几个百分点的核函数。像 cuBLAS 与 TorchInductor 这样的高层框架已经依赖 CUTLASS。只有当你需要非标准布局或独特的融合模式时，才需要自定义。

*PyTorch 特有的最佳实践* 优先使用内置张量操作，如 torch.matmul、融合注意力和嵌套张量，因为 PyTorch 通常会随时间从 CUDA 继承 Tensor Core 及其他硬件优化。而如果你为 PyTorch 编写自定义核函数扩展，请沿用同样的策略：尽量减少寄存器使用、对齐数据以实现合并内存加载、瞄准 Tensor Core。这样才能确保你的核函数与原生核函数处于同等水平。

通过遵循“剖析、诊断、应用有针对性优化”的系统化工作流——从占用率调优、warp 级精化，到 ILP，再到借助分块、融合与 Tensor Core 实现高算术强度——你就能把一个访存受限的 GPU 核函数转变为计算受限的核函数。这能带来巨大的加速，而在强访存受限的工作负载中，有时甚至是数量级的收益。

## 结论

在本章中，你了解了与高级内存和计算硬件特性相关的优化技术，例如 TMA、TMEM、Transformer Engine 与 Tensor Cores。同样的原则可从单 GPU 扩展到多 GPU 集群。首先，使用系统级剖析器（如 NVIDIA Nsight Systems）来关联跨多个 GPU 的 CPU 活动、GPU 核函数以及 NVLink/NVSwitch 流量。然后使用 Nsight Compute 深入钻研每个核函数的细节。

通过系统化地剖析、消除主导性停顿并掌握高级硬件特性，你可以把一个访存受限的工作负载转变为计算受限的工作负载——往往能带来巨大的加速，包括在强访存受限路径上的数量级改善。

而即便是像 NVIDIA GB200/GB300 NVL72 这样具备高 NVLink/NVSwitch 带宽的超大规模多 GPU 系统，每 GPU 的算术强度优化仍是消除大多数瓶颈的关键。那些每字节做太少工作的核函数会需要过多的内存搬运、使互连饱和，并把内存带宽压到极限。

这种互连与内存带宽的饱和，会在我们的核函数尚未能充分利用众多互连 GPU（例如 NVL72 中的 72 个）的计算能力之前就早早发生。因此，提高核函数的算术强度是让多 GPU 训练与推理工作负载高效扩展的关键。

在下一章中，我们将继续深入探讨诸如持久化核函数（persistent kernels）、巨型核函数（megakernels）、warp 专门化、协作组与线程块簇等技术。这些思想是大多数现代 LLM 运行时优化的基础，因此理解它们的实现——以及如何将它们应用到你自己的底层系统优化工作中——非常重要。
