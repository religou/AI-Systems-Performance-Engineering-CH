# 第 2 章　AI 系统硬件概览

设想一下，把相当于一整台超级计算机的 AI 硬件浓缩进一个机架（rack）之中。NVIDIA 最新的架构做到的正是这件事。本章将深入剖析 NVIDIA 如何把 CPU 与 GPU 融合成强大的超级芯片（superchip），再用超高速互连把几十颗这样的芯片连成一体，从而打造出一台盒装 AI 超级计算机（AI supercomputer-in-a-box）。我们会先探讨最基础的硬件构件——Grace CPU 与 Blackwell GPU——看看它们的紧密集成与巨大的内存池如何让 AI 工程师的工作变得更轻松。

接着我们把视野向外扩展，看看那张把 72 块 GPU 连接起来、使其宛如一台机器的网络结构（fabric）。一路上，我们会重点介绍这套系统在算力、内存容量和能效上的飞跃，正是它们赋予了系统超能力。读到本章结尾，你会体会到这套前沿硬件是如何让训练和部署此前看似不可能实现的数万亿参数模型成为现实的。

## CPU 与 GPU 超级芯片

NVIDIA 扩展 AI 算力的思路，始于把单颗 CPU 与 GPU 合二为一的超级芯片模块。从 Hopper 一代开始，NVIDIA 就把一颗基于 ARM 的 CPU 与一块或多块 GPU 封装在同一单元内，并用高速接口将它们紧密连接。其结果是一个像统一计算引擎那样运作的单一模块。

超级芯片的首个实现是 Grace Hopper（GH200），它把一颗 Grace CPU 与一块 Hopper GPU 配成一对。接下来是 Grace Blackwell（GB200）Superchip，它在同一封装内把一颗 Grace CPU 与两块 Blackwell GPU 配成一组。如图 2-1 所示，Grace CPU 位于模块中央，两侧环绕着两颗 Blackwell GPU 裸片（die）。

![Figure 2-1. NVIDIA Grace Blackwell Superchip module containing one Grace CPU (center) and two Blackwell B200 GPUs (top left and right) on a single module with a shared unified memory space and connected by a custom high-speed link called NVLink-C2C (chip-to-chip)](AI%20Systems%20Performance%20Engineering-ch2_images/figure-2-1.png)

_图 2-1. NVIDIA Grace Blackwell 超级芯片模块，在单个模块上包含一颗 Grace CPU（中央）和两块 Blackwell B200 GPU（左上和右上），共享统一内存空间，并通过一条名为 NVLink-C2C（chip-to-chip，片间）的定制高速链路相连_

在传统系统中，CPU 与 GPU 各自拥有独立的内存池，并通过相对缓慢的总线（如 PCIe）通信，这意味着数据必须在两者之间来回拷贝。NVIDIA 的超级芯片用一条名为 NVLink-C2C（chip-to-chip，片间）的定制高速链路把 CPU 与 GPU 连在一起，从而消除了这一屏障。

在 GB200 Superchip 中，NVLink-C2C 在 Grace CPU 与 Blackwell GPU 之间提供高达 ~900 GB/s 的带宽。相比之下，PCIe Gen5 x16（Blackwell B200）单方向约为 64 GB/s，PCIe Gen6 x16（Blackwell Ultra B300）单方向约为 128 GB/s。NVLink-C2C 的互连速度比典型的 PCIe 快一个数量级。而且，更重要的是，它是缓存一致（cache-coherent）的。

缓存一致性（cache coherency）意味着 CPU 与 GPU 共享一个一致的、统一的内存架构。因此，它们始终看到相同的数值。在实践中，超级芯片上的 Grace CPU 与各 Blackwell GPU 都能直接访问彼此的内存，就好像它们同属一个巨大的内存池。GPU 可以读写存放在 CPU 内存中的数据，反之亦然，而无需显式拷贝。NVIDIA 通常把这种统一内存（unified memory）架构称为统一 CPU-GPU 内存（Unified CPU-GPU Memory）或扩展 GPU 内存（Extended GPU Memory，EGM），它实际上模糊了 CPU 内存与 GPU 内存之间的界线。

每颗 Grace Blackwell Superchip 都携带着海量内存。Grace CPU 挂载了数百 GB 的 LPDDR5X DRAM，而每块 Blackwell GPU 都拥有自己的高速、高带宽内存（high-bandwidth memory，HBM）堆栈。

在 GB200 Superchip 中，Grace CPU 提供高达 ~480 GB 的 LPDDR5X，速率高达 ~500 TB/s，两块 Blackwell GPU 合计贡献高达 ~384 GB 的 HBM3e 内存（每块 GPU 共 192 GB）。总体而言，一颗 GB200 Superchip 对外暴露大约 ~900 GB 一致的统一内存，供 GPU 和 CPU 在统一地址空间中访问。

简而言之，每颗超级芯片都拥有近 1 TB 的快速统一内存可供调用。这对于庞大的 AI 模型而言是一次颠覆性变革。在较老的系统中，单块 GPU 的内存可能被限制在 < 100 GB，这意味着比它更大的模型必须被切分或卸载到更慢的存储上。而在这里，GPU 可以无缝地把 CPU 的内存当作扩展来使用。

如果某个神经网络层或一张大型嵌入表放不进 GPU 本地的 HBM，它可以驻留在 CPU 的内存中，GPU 依然能够跨 NVLink-C2C 对其进行处理。从程序员的角度看，统一虚拟地址空间与一致性简化了正确性的保证。不过，为了性能，应当使用异步预取（prefetch）和分级流水线等技术，显式地管理数据的放置与移动。通过 NVLink-C2C 访问 LPDDR5X，其延迟高于直接访问 HBM，带宽也大约低一个数量级。

GPU 内存仍然比 CPU 内存快得多、也离 GPU 核心更近——你可以把 CPU 内存看作一块容量很大但速度略慢的扩展。访问 LPDDR5X 中的数据不像访问 GPU 上的 HBM 那样快，带宽大约低一个数量级（10×），延迟也更高。聪明的运行时会把最常用的数据保留在 HBM 中，而把 CPU 的 LPDDR5X 用于溢出数据或对速度不那么敏感的数据。关键在于，溢出不再需要走到 NVMe SSD 或跨越网络。

GPU 从 CPU RAM 取数据的速度或许可达 900 GB/s（每方向 450 GB/s），虽然比 HBM 慢，但远快于从 NVMe SSD 存储读取。这种灵活性至关重要，因为这意味着一个体量比如说 500 GB 的模型（对单块 GPU 的 HBM 而言太大了）依然可以被完整放入一个超级芯片模块之内，可访问合计 192（可用 180）GB 的 HBM 和 ~500 GB 的 CPU 内存。这样的模型无需在多块 GPU 之间切分即可运行。GPU 只会在需要时透明地从 CPU 内存中拉取额外的数据。

本质上，只要整个模型能装进超级芯片上 CPU + GPU 合计的内存，内存容量就不再是承载超大模型的硬性限制。许多研究者都曾在模型放不进 GPU 时遭遇过令人头疼的“内存不足（out of memory，OOM）”错误——这套架构正是为大幅拓宽这一边界而设计的。

### NVIDIA Grace CPU

Grace CPU 本身也绝非等闲之辈。它是一颗由 NVIDIA 面向带宽与能效定制设计的 ARM Neoverse V2 CPU。它在超级芯片中的职责是处理通用任务、预处理数据并把数据喂给 GPU，以及管理挂载在其上的海量内存。它的时钟频率并不高，但凭借巨大的内存带宽——对其 LPDDR5X 内存高达 ~500 GB/s——以及大量缓存（包括超过 100 MB 的 L3 缓存）来加以弥补。

其设计理念是：在向 GPU 铲送数据时，CPU 绝不应成为瓶颈。它可以从存储中流式读取数据，或者即时执行诸如分词、数据增强之类的数据变换——通过 NVLink-C2C 极其高效地喂给 GPU。如果你工作负载中的某一部分更适合放在 CPU 上，Grace 核心可以承担这部分工作，并让结果立即可被 GPU 访问。

这是一种和谐的耦合：在 GPU 相对薄弱的领域（如随机内存访问或控制流繁重的代码），CPU 扩展了 GPU 的能力；而在 CPU 力不从心的数值运算上，GPU 则负责加速。

CPU 与 GPU 之间的低延迟链路意味着它们可以在没有常见开销的情况下交换任务。举例来说，从 CPU 启动一个 GPU 内核（kernel）可以比在传统系统上快得多，因为命令不必穿越缓慢的 PCIe 总线。CPU 和 GPU 本质上位于同一块板上。这就好比调用一个快速的本地函数，而不是较慢的远程函数。接下来，我们来谈谈 Blackwell GPU——超级芯片中那台蛮力十足的引擎。

### NVIDIA Blackwell“双裸片”GPU

Blackwell 是 NVIDIA 为这一代 GPU 取的代号，无论在算力还是内存上，它都相较上一代 Hopper（H100）GPU 实现了显著飞跃。Blackwell B200 与 B300“Ultra”GPU 并不是单颗芯片。相反，它们采用多芯片模块（multichip module，MCM）设计，在单个模块内放置两块 GPU 裸片。因此，Blackwell 被称为双裸片 GPU（dual-die，见图 2-2）。

> 尽管本节深入探讨双裸片架构的细节，但本书其余部分会把 Blackwell 合并后的两块 GPU 裸片统称为“Blackwell GPU”。

这种芯粒（chiplet）方式，把本应是一整块巨大 GPU 的东西拆分成较小的 GPU 裸片——再用一条超高速的封装内裸片间互连（die-to-die interconnect）把它们连接起来。为什么要这么做？因为单块整体式裸片受制于制造工艺，在硅片上能做出的芯片尺寸是有上限的。通过把两块物理 GPU 裸片合并进单个模块，NVIDIA 可以把该模块的晶体管总预算翻倍。

![Figure 2-2. Blackwell dual-die multichip module (MCM) design](AI%20Systems%20Performance%20Engineering-ch2_images/figure-2-2.png)

_图 2-2. Blackwell 双裸片多芯片模块（MCM）设计_

对于 Blackwell B200 MCM，每块 GPU 裸片拥有约 104 billion 晶体管和 96 GB HBM3e 内存。合并后的 GPU 模块每块 B200 GPU 大约有 208 billion 晶体管，共 192（可用 180）GB 内存。相比之下，Hopper H100 GPU 拥有 ~80 billion 晶体管和 80 GB HBM3（对比 Blackwell 的 HBM3e）内存。因此，Blackwell 的 B200 在晶体管数量上翻了一倍多，内存容量提升约 ~2.4×。

Blackwell 的两块 GPU 裸片使用一条专用的高速 10 TB/s 裸片间互连来通信，该互连名为 NV-HBI（High-Bandwidth Interface，高带宽接口）。这使得模块内的两块 GPU 裸片能作为单一的统一 GPU 运作。运行其上的软件层只会看到一块 GPU。

从系统的视角看，一块 Blackwell GPU 就是一个单一的模块（或设备），拥有一个大内存池（192〔可用 180〕GB HBM3e）和大量执行单元，但在底层它其实是两块芯片协同工作。NVIDIA 的软件与调度机制会确保工作在两块 GPU 裸片之间均衡分配，且内存访问保持一致。这让开发者可以在很大程度上忽略这种复杂性，因为如 NVIDIA 所期望的那样，它们表现得就像一块 GPU。

每块 Blackwell B200 GPU 模块拥有 192（可用 180）GB HBM3e 内存，跨两块 GPU 裸片合计（每块 96 GB），并划分为 8 层堆栈（8-Hi）。一个 8-Hi HBM3e 堆栈由八块 DRAM 裸片垂直堆叠而成——每块 3 GB——每个堆栈合计 24 GB。

B200 GPU 使用八个这样的堆栈（每块裸片四个）来提供 192（可用 180）GB（192 GB = 8 个堆栈 × 每堆栈 24 GB）的封装内内存。与上一代 Hopper GPU 相比，这提高了每块 GPU 的堆栈数量与容量——为模型参数、激活值、梯度和输入数据留出了更多余量。

> 由于纠错码（ECC）、系统固件占用、制造限制及其他阻碍芯片对外暴露完整 192 GB 的因素，每块 B200 的 192 GB HBM3e 内存中只有 180 GB 可用。因此，谈到 Blackwell B200 的可用内存时，我们会引用 180 GB 而非完整的 192 GB。

内存速度也更快，因为 Blackwell 的 B200 HBM3e 每块 GPU 的聚合带宽高达约 8 TB/s。作为对比，Hopper 使用上一代的 HBM3，每块 GPU 提供 ~3.35 TB/s。因此，Blackwell 的内存带宽吞吐大约比 Hopper 高 2.4×。

以每秒 8 TB 的速率喂送数据，Blackwell GPU 核心得以持续忙于处理巨型矩阵，而不必频繁停顿等待数据。NVIDIA 还强化了片上缓存，Blackwell 拥有共计 126 MB 的 L2 缓存（每块裸片 63 MB）。这块缓存是 GPU 上一块虽小但极快的内存，用于存放最近使用过的数据。

通过把 L2 缓存容量相较 Hopper 的 50 MB L2 缓存提高到 2.5× 以上，Blackwell 可以把更多神经网络权重或中间结果保留在片上，从而避免额外往返 HBM。这再次有助于确保 GPU 的计算单元很少因缺数据而挨饿。

接下来，我们展示 Blackwell GPU 是如何与一组专用的降精度 Tensor Core，以及 NVIDIA 提供的、面向 Transformer 优化的硬件和软件 API（称为 Transformer 引擎）配对的。诸如 PyTorch 这样的框架和 vLLM 这样的推理引擎，通过使用 CUDA、CUTLASS 和 OpenAI 的 Triton 等库来支持这些优化，我们会在后续章节中讨论它们。

> 请记住，本书其余部分把 Blackwell 的双裸片 GPU 直接称为“Blackwell GPU”。

### NVIDIA GPU Tensor Core 与 Transformer 引擎

说到计算单元，Blackwell 引入了专门针对 AI 工作负载的增强。其核心是 NVIDIA 的 Tensor Core（张量核心）技术和 Transformer 引擎（Transformer Engine，TE）。Tensor Core 是位于 GPU 每个流式多处理器（streaming multiprocessor，SM）内部的专用单元，能够以极高的速度执行矩阵乘法运算。

Tensor Core 在此前几代中就已存在，但 Blackwell 的 Tensor Core 支持更多的数值格式，包括 8 位和 4 位浮点这类极低精度的格式。低精度背后的思想很简单：用更少的比特来表示数字，你就能在同一时间执行更多运算——更不用说由于用更少的比特表示相同的数字，你的内存也更“耐用”了。这一切的前提是你的算法能够容忍数值精度上的少许损失。如今，大量 AI 算法在设计之初就已考虑到低精度数值格式。

NVIDIA 首创了 TE，用以在深度学习中自动调整并使用混合精度（mixed precision）：关键层使用更高精度（FP16 或 BF16），不那么关键的层使用 FP8。TE 会自动优化精度上的平衡，目标是在更低精度下保持模型的准确率。

在 Hopper 一代，TE 首次引入了 FP8 支持，其吞吐量比 FP16 翻了一倍。Blackwell 更进一步，引入了 NVIDIA FP4（NVFP4）——一种 4 位浮点格式，所用比特数是 FP8 的一半。FP4 如此之小，以至于有潜力将 FP8 的计算吞吐再翻一番。图 2-3 展示了 FP8 与 FP4 相对于 FP16 的加速比。

整个 NVL72 机架（72 块 GPU）在 4 位精度下的理论 Tensor Core 吞吐量超过 1.4 exaFLOPS（即 1.4 × 10^18）。这是一个令人瞠目的数字，足以让这一个机架跻身世界最快超级计算机之列——尽管是在 FP4 低精度下。即便真实世界的工作负载并不总能触及这一峰值，但这份能力实实在在地存在，着实令人惊叹。

现代 GPU 使用的 TE 增加了对 NVFP4 的支持，并改进了缩放与校准。在实践中，你通过在 PyTorch 等框架中使用 TE 的内核和模块来采用它。这样，FP8 和 NVFP4 便会在能够保持准确率时被应用。这在所有框架中并不都是完全自动的逐层决策。

![Figure 2-3. Relative speedup of FP8 and FP4 compared to FP16](AI%20Systems%20Performance%20Engineering-ch2_images/figure-2-3.png)

_图 2-3. FP8 与 FP4 相对于 FP16 的加速比_

更高级的技术包括：在训练和推理期间，为神经网络的每一层动态改变其精度。目标是为每一层使用仍能保持模型准确率的最低精度。举例来说，TE 可能会把神经网络的最初几层保持在 FP16，因为早期层可能对噪声较为敏感。但根据启发式规则，它可以决定对更耐受的后续层——或者对高精度并不那么关键的巨型嵌入矩阵——使用 FP8 或 FP4。

所有这些都可以在 NVIDIA 的库和 PyTorch 这类 AI 框架的底层悄然发生。作为用户，你只需启用混合精度，得到的结果便是一个几乎“免费”的巨大加速。我们会在第 9 章讨论混合精度，但你只需知道，如今许多大语言模型（large language model，LLM）正是出于这个原因而使用混合精度。相较 FP16 和 FP32，这些降精度提升了训练速度——并减少了准确率损失。Blackwell 的设计初衷正是让 FP8 和 FP4 既易于使用又高效。

这些降精度格式同样降低了内存占用。相较 FP8，使用 FP4 把每个参数所需的内存减半（而 FP8 又把 FP16 的内存占用减半），这意味着你可以把更大的模型塞进 GPU 的内存里。

NVIDIA 实际上已经押注 AI 的未来在于更低精度的算术，并赋予了 Blackwell 在这方面出类拔萃的能力。这对于超大模型的推理服务尤为关键，因为在那里，吞吐量（每秒 token 数）与延迟至关重要。

为了说明从 Hopper 到 Blackwell 的代际飞跃，NVIDIA 报告称：对于一个庞大的 1.8 万亿参数 MoE（mixture of experts，混合专家）模型，基于 H100 的系统每块 GPU 只能生成约 3.4 个 token/秒——而且首个 token 的延迟超过 5 秒。这对于交互式使用而言太慢了。

基于 Blackwell 的系统（NVL72）运行同一模型，每块 GPU 约达 150 个 token/秒，首 token 延迟低至 ~50 毫秒。这大约是相较 Hopper 一代约 30× 的实时吞吐量提升。NVL72 让这个庞大的模型能够提供实时响应——从而为更多低延迟的应用场景打开了大门。

这一加速源自原始 FLOPS，是更快的 GPU、更低精度（FP4）的使用，以及 NVLink 互连持续为 GPU 供给数据这几方面的结合。它彰显了一种横跨计算与通信的整体设计，如何能够转化为真实世界的性能收益。

本质上，Blackwell GPU 比它们的前辈更强大、更聪明，也被数据喂得更饱。得益于 Tensor Core、TE 和低精度，它们处理数学运算的速度更快。此外，凭借巨大的内存带宽、大容量缓存和 NVLink，系统架构确保了数据能被迅速就位。

在继续之前，我们先快速讨论一下 GPU 内部的层级结构，因为这对后续理解性能调优很有帮助。

### 流式多处理器、线程与线程束

与前几代一样，每块 Blackwell GPU 都由许多流式多处理器（SM）组成。如图 2-4 所示，可以把它们想象成 GPU 的“核心”。

![Figure 2-4. Comparing CPU cores to GPU cores (source: https://oreil.ly/003EH, https://oreil.ly/Z25Tf)](AI%20Systems%20Performance%20Engineering-ch2_images/figure-2-4.png)

_图 2-4. 对比 CPU 核心与 GPU 核心（source: https://oreil.ly/003EH, https://oreil.ly/Z25Tf）_

每个 SM 都包含一批算术单元（用于 FP32、INT32 等）、用于矩阵运算的 Tensor Core、用于内存操作的加载/存储单元，以及一些用于超越函数运算等任务的特殊功能单元。GPU 还拥有自己的一小块超快内存，包括寄存器、共享内存和 L1 缓存。

SM 以固定大小的分组执行线程，这些分组称为线程束（warp），每个线程束恰好包含 32 个线程，它们步调一致地执行完全相同的指令。这被称为单指令多线程（single instruction, multiple threads，SIMT）执行模型。

SM 并行执行许多活跃的线程束，以帮助掩盖某个线程等待从全局内存访问数据时的延迟。设想一个 SM 同时有几十个线程束（数百个线程）在运行。如果一个线程束正在等待内存读取，另一个线程束便可以运行。这被称为延迟隐藏（latency hiding）。我们会在全书中反复回到延迟隐藏这一话题。这是你的调优工具箱中一件非常重要的性能优化利器。

像 Blackwell 这样的高端 GPU 会拥有数百个 SM。每个 SM 都能并发运行数千个线程。这就是我们如何在单块 GPU 上获得数万个活跃线程的原因。正如前面提到的，所有这些 SM 共享一块 126 MB 的 L2 缓存，并共享连接到 HBM 的内存控制器。如图 2-5 所示，内存层级依次为：寄存器（每线程）→ 共享内存（每线程块，位于各 SM 上）→ L1 缓存（每 SM）→ L2 缓存（在 GPU 上所有 SM 之间共享）→ HBM 内存（片外）。

![Figure 2-5. GPU memory hierarchy](AI%20Systems%20Performance%20Engineering-ch2_images/figure-2-5.png)

_图 2-5. GPU 内存层级_

为了获得最佳性能，数据需要尽可能保留在该层级的高处。如果每个操作即便以 8 TB/s 的速率也都要走到 HBM，那么由于访问片外内存的延迟增加，GPU 会过于频繁地停顿。通过把可复用的数据保留在 SM 本地内存或 L2 缓存中，GPU 就能达成惊人的吞吐量。Blackwell 架构把缓存和带宽翻倍，其目标正是让这头 GPU 巨兽吃得饱、跑得欢。

作为性能工程师，我们会看到许多例子：一个内核的性能既受制于计算，也受制于内存流量与吞吐量。NVIDIA 显然是这样设计 Blackwell 的——对许多 AI 工作负载而言，FLOPS 与内存带宽之间的平衡都恰到好处。

Blackwell 的设计平衡了计算与内存，使得对许多 AI 内核而言，GPU 能够以最少的停顿持续计算。在实践中，经过良好优化的稠密数学运算可以复用片上内存中的数据，从而在不严重受内存限制的情况下逼近峰值 FLOPS。

所有这些意味着：只要代码经过良好优化，GPU 往往会忙于计算，而不是等待数据。需要注意的是，某些操作（如巨大的归约或随机内存访问）仍可能受内存限制，但更新换代后的 GPU、内存和互连硬件让这个问题不再那么突出。

## 超大规模网络：把众多 GPU 视为一体

把两块 GPU 和一颗 CPU 塞进一颗超级芯片，为我们带来了一个极其强大的节点。下一个挑战是把许多这样的超级芯片连接起来，以横向扩展到规模更大的模型训练。

NVIDIA 提供了一种使用 GB200/GB300 Superchip 的大型机架配置，称为 NVL72 系统。NVL72 指的是一套拥有 72 块 Blackwell GPU——以及 36 颗 Grace CPU——并全部通过 NVLink 互连的系统。这本质上就是装进单个机架里的一台 AI 超级计算机。

如图 2-6 所示，GB200/GB300 NVL72 由 18 个计算节点构成，每个节点包含两颗 GB200/GB300 Superchip，因此每个计算节点共有四块 Blackwell GPU + 两颗 Grace CPU。

![Figure 2-6. A 1U compute tray within the GB200/GB300 NVL72 rack with two Grace Blackwell Superchips (source: developer.nvidia.com)](AI%20Systems%20Performance%20Engineering-ch2_images/figure-2-6.png)

_图 2-6. GB200/GB300 NVL72 机架内的一个 1U 计算托盘（compute tray），含两颗 Grace Blackwell 超级芯片（source: developer.nvidia.com）_

在这里，每颗超级芯片模块都有一颗 Grace CPU 和两块 Blackwell GPU（每块 B200 都是一个双裸片 MCM）。NVL72 把 18 个这样的托盘连在一起。通过把这 18 个计算节点连接起来，GB200/GB300 NVL72 将 72 块 Blackwell GPU（18 个节点 × 4 块 GPU）和 36 颗 Grace CPU（18 个节点 × 2 颗 CPU）连成一体，构成一个强大的、统一的 CPU-GPU 集群。

NVL72 有趣的一点在于：在单个 NVLink 域内，每块 GPU 都能通过 NVLink Switch 网络结构以极高速度与任意另一块 GPU 通信。NVIDIA 通过在 GPU 上采用 NVLink 5 连接与一款名为 NVSwitch 的专用交换芯片相结合，实现了这一点。

### NVLink 与 NVSwitch

每块 Blackwell GPU 对外暴露 18 个 NVLink 5 端口。每块 GPU 的聚合双向 NVLink 带宽为 1.8 TB/s（18 条 NVLink 链路 × 100 GB/s 双向），NVL72 把所有端口都接入 NVLink Switch System。每个 NVLink 交换托盘提供 144 个 NVLink 端口，速率为 100 GB/s。在这九个托盘上，每块 GPU 的 18 条 NVLink 5 链路各接到一颗 NVSwitch 芯片，因此 72 块 GPU 以完整的对分带宽（bisection bandwidth）实现全互连。每块 GPU 的聚合双向 NVLink 5 带宽为 1.8 TB/s（18 条 NVLink 链路 × 100 GB/s 双向）。

这是 Hopper GPU 所用上一代每块 GPU NVLink 带宽的两倍。Hopper H100 使用 18 个 NVLink 4 端口，但其速率只有 NVLink 5 的一半。跨 NVLink 的 GPU 间延迟处于个位数微秒级别。

这些 GPU 通过 NVSwitch 芯片连成一张网络。NVSwitch 本质上是一块类似于网络交换机的交换芯片，但它专为 NVLink 而打造。这意味着任意 GPU 都能以完整的对分带宽，在 NVLink Switch System 中经过一级交换到达任意另一块 GPU。这种“单级”特性在单个 NVL72 机架内成立，因为每块 GPU 都用其 18 条 NVLink 链路连接到 18 颗 NVSwitch 芯片，从而实现经由单个交换机的路径。图 2-7 展示了 NVL72 中使用的一个 NVLink Switch 托盘。

![Figure 2-7. One NVLink Switch tray inside the NVL72 (source: https://oreil.ly/h7seG)](AI%20Systems%20Performance%20Engineering-ch2_images/figure-2-7.png)

_图 2-7. NVL72 内部的一个 NVLink Switch 托盘（source: https://oreil.ly/h7seG）_

每个交换托盘包含两颗 NVSwitch 芯片和多个高速端口。如图 2-8 所示，NVL72 机架由 9 个这样的交换托盘和 18 个计算托盘组成。

![Figure 2-8. NVSwitch System of nine trays inside an NVL72 rack (source: https://oreil.ly/h7seG)](AI%20Systems%20Performance%20Engineering-ch2_images/figure-2-8.png)

_图 2-8. NVL72 机架内由九个托盘构成的 NVSwitch System（source: https://oreil.ly/h7seG）_

由于这 9 个交换托盘每个都包含两颗 NVSwitch 芯片，NVL72 系统中共有 18 颗 NVSwitch 芯片。这张网络被组织成一个全交叉开关（full crossbar），使得每块 GPU 都连接到每一颗 NVSwitch，且每一颗 NVSwitch 都连接到每一块 GPU。这在任意一对 GPU 之间提供了一条高带宽路径。

每个交换托盘对外暴露 144 个 NVLink 端口，以完整连接每块 GPU 上的 18 条 NVLink 链路。具体而言，每块 GPU 用其 18 条 NVLink 链路连接到 18 颗 NVSwitch 芯片（每颗交换芯片一条链路）。这意味着任意 GPU 都能一跳（GPU → NVSwitch → GPU）到达任意另一块 GPU，而且沿途拥有巨大的带宽。图 2-9 展示了完整的 NVL72 架构，含 72 块全互连的 GPU（36 颗 GB200 超级芯片）和 18 颗 NVSwitch。

![Figure 2-9. Each GPU connects to each NVSwitch (one link for each switch)](AI%20Systems%20Performance%20Engineering-ch2_images/figure-2-9.png)

_图 2-9. 每块 GPU 都连接到每一颗 NVSwitch（每颗交换芯片一条链路）_

在一个 NVL72 机架内，整张 72-GPU 网络的聚合对分带宽约为 130 TB/s。作为参照，这比同等规模下即便顶级的 InfiniBand 集群也要高出许多倍。这一设计对外呈现出一张全互连的高带宽网络结构，并在各 GPU 之间提供全局地址空间。这既支持高效的集合通信与单边操作，又保留了软件对同步与一致性的显式控制。

### 多 GPU 编程

从编程模型的角度看，一块 GPU 可以借助点对点（peer-to-peer）以及分区全局地址空间（partitioned global address space，PGAS）模型，通过 NVLink 直接访问另一块 GPU 的内存，例如 NVIDIA SHMEM（NVSHMEM）——NVIDIA 的 GPU 加速版 OpenSHMEM 实现。这里存在一个全局地址空间，但 GPU 缓存在各 GPU 之间并非全局一致。只有跨 NVLink-C2C 的 CPU–GPU 路径才是缓存一致的。诸如 NCCL 和 NVSHMEM 之类的软件栈提供了正确进行多 GPU 访问所需的同步与排序。硬件缓存一致性与软件同步技术相结合，使得 NVL72 本质上可被视为一块巨大的 GPU。

远程直接内存访问（remote direct memory access，RDMA）是一种网络技术，能够在跨 InfiniBand 以及 RDMA over Converged Ethernet（RoCE）传输的主机之间实现直接、零拷贝的内存传输。InfiniBand 与 RoCE 的可选远程原子操作由 InfiniBand Trade Association（IBTA）定义。

GPUDirect RDMA 是 NVIDIA 对 RDMA 协议的实现，它借助 nvidia-peermem 驱动，使网络接口控制器（network interface controller，NIC）能够注册 GPU 内存，并直接对 GPU 内存进行 RDMA 读写。这让 GPU 无需牵涉 CPU 便能跨节点交换数据并执行原子操作。这也让 NIC 能够对 GPU 内存执行直接 DMA，而无需经由主机 RAM 中转。

跨节点的远程原子操作与单边操作由上层库（如 NVSHMEM）提供，它们在 RDMA 传输之上实现了这些语义。需要注意的是，GPUDirect RDMA 提供的是直接的数据通路，而非原子 API 本身。分布式训练与推理工作负载需要在许多 GPU 之间频繁地同步与交换信息。

传统上，这些 GPU 分处不同的计算节点和机架。因此，同步只能通过 InfiniBand、以太网等相对缓慢的网络链路进行。当横向扩展到许多 GPU 以支撑大型 AI 模型时，这往往成为瓶颈。

而在 NVL72 系统中，这些交换是通过 NVLink 和 NVSwitch 以超高速进行的。这意味着你可以在极小的通信开销下，把训练作业或推理集群扩展到 72 块 GPU。而且，由于 GPU 彼此等待数据的时间大幅减少，整体吞吐量在多达 72 块 GPU 的范围内几乎呈线性扩展。
