编译后的 FSDP 会借助 AOT Autograd 和 Inductor 的内存规划器融合前向与反向传播，并在各模型分片之间复用缓冲区。因此，每块 GPU 上只驻留处于活跃状态的参数切片以及最少量的中间内存缓冲区。与 DDP 或即时执行模式（eager mode）相比，这降低了峰值内存占用。

内存节省来自：避免冗余的梯度存储、复用中间分配，以及在各分片间将通信与计算重叠。这些手段使得更大的模型能够放入每块 GPU。如果你不逐一包装子模块，FSDP 会退回到把所有参数当作一个大桶来处理。这仍然可以工作，但会限制内存收益与重叠潜力。因此，建议将 torch.compile 与按模块的 FSDP 包装器结合使用，以获得最高的速度和内存效率——在大规模训练作业中尤其如此。

> 一旦出现问题，调试可能非常复杂——在更大的集群/配置中会更加复杂。在将 FSDP 与 torch.compile 搭配使用时，务必先在较小的配置上测试。

_在使用自定义及第三方 CUDA C++ 和 Triton 算子时监控性能权衡_ 如果你依赖某个 PyTorch 并不了解的自定义或第三方 CUDA 扩展，Dynamo 会产生一次图中断（graph break），因为它无法推断该算子的行为——也无法判断它是否安全。如果该算子对性能至关重要，可以考虑用 Triton 以 Python 重写这个自定义算子。

PyTorch 支持 torch.library.triton_op() API，它让你能够把 Triton 核函数（kernel）作为自定义算子无缝集成到 PyTorch 中。这使编译器得以窥探 Triton 代码内部并执行优化。在深入 Triton 之前，我们先快速总结一下如何调试各个编译器阶段、图中断以及编译器性能。

> 如今许多流行的第三方库要么提供了 Triton 实现，要么为其算子提供了 Dynamo/FX 包装器。在自己动手写之前，先查一查这些是否已经存在。

## 调试编译器阶段、图中断与性能

你可以在运行时通过设置各种环境变量（如 TORCH*LOGS、TORCH_COMPILE_DEBUG 和 TORCHDYNAMO_REPRO*\*）来记录并调试不同类型的编译器事件。这些事件包括图中断、重新编译（recompilation）、保护条件（guard）以及其他编译器决策。设置 TORCH_LOGS 的一个示例如下所示（常用取值见表 14-1）：

```
# "graph_breaks", "dynamo", "aot_graphs", "inductor",
# "graph_outputs", "graph_code", "dynamic", "perf_hints",
# "output_code", "recompiles", "guards", etc.
TORCH_LOGS="graph_breaks" python train.py
```

这会让 PyTorch 在每次发生图中断时打印相关信息。为了总结各种日志选项，你可以将 TORCH_LOGS 设置为下列取值来调试 torch.compile，涵盖各个阶段（TorchDynamo、AOT Autograd 和 TorchInductor）、图、图中断、生成的代码、性能、重新编译与保护条件——以及编译器决策和性能，如表 14-1 所示。

表 14-1. torch.compile 的日志选项

| TORCH_LOGS 取值 | 说明                                             |
| --------------- | ------------------------------------------------ |
| graph_breaks    | 记录图中断事件                                   |
| dynamo          | 来自 TorchDynamo 的详细日志                      |
| aot_graphs      | 来自 AOT Autograd 的详细日志                     |
| inductor        | 来自 TorchInductor 的详细日志                    |
| graph_outputs   | 显示编译后的 FX 图                               |
| graph_code      | 转储 TorchDynamo 生成的每个 FX 图的 Python 代码  |
| dynamic         | 追踪关于动态形状的决策，以及维度何时被标记为动态 |
| perf_hints      | 显示错失的性能优化机会                           |
| output_code     | 打印每个编译图生成的代码                         |
| recompiles      | 记录触发重新编译的原因                           |
| guards          | 记录保护条件及保护条件的求值                     |

如果你怀疑子图的切分方式——以及编译了哪些形状——存在问题，这些设置会很有用。有了这些设置，你无需改动代码就能获得大量内部调试信息。

> 请做好输出非常冗长的准备。例如，当你只想调试图中断时，建议从仅设置 "graph_breaks" 开始。

在底层，设置 TORCH_LOGS 相当于使用 torch.\_logging.set_logs() API。不过，将 TORCH_LOGS 作为环境变量在外部配置有时更为简便。

另外请记住，你还可以设置 TORCH_COMPILE_DEBUG=1 来启用 TorchInductor 的调试模式。这会记录 FX 图、TorchInductor IR、生成的 Triton 代码，以及一份带可视化的 HTML 报告（前提是已安装 Graphviz）。

你也可以设置 TORCHDYNAMO_REPRO_AFTER 和 TORCHDYNAMO_REPRO_LEVEL，强制 TorchDynamo 在每个阶段之后转储其图。它还会针对未编译的即时执行版本代码进行运行时对比。

此外，还可以使用一个名为 tlparse 的工具来遍历编译日志。追踪日志对于调试编译事件（例如重新编译）以及生成缺陷报告都很有用。

要启用追踪日志，请通过 TORCH*TRACE 环境变量指定 \_trace-log* 目录。然后在该 _trace-log_ 目录上运行 tlparse，即可生成如下所示的栈帧树状表示：

```
- /workspace/networks/layers/transformer.py:634 in forward
  .../torch/nn/modules/module.py in _wrapped_call_impl
 .../torch/nn/modules/module.py in _call_impl
  - [2/2] [2/3] ../torch/_dynamo/convert_frame.py in __call__
  - /workspace/networks/layers/transformer.py:753 in forward
    - [8/2] [8/3] .../torch/_dynamo/convert_frame.py in __call__
...
```

此外，你可以使用 Perfetto UI 来展示追踪时间线的可视化。而且由于追踪带来的开销极小，甚至可以在生产环境中启用 TORCH_TRACE。

现在，让我们更深入地探讨 TorchInductor 所使用的 OpenAI Triton 语言与编译器。我们将编写一些基础与进阶的 Triton 核函数，然后把它们注册到 PyTorch。

## 用 OpenAI Triton 编写自定义核函数

到目前为止，我们只是简单提及了 OpenAI 开源的 Triton 语言与编译器。现在是深入探讨的时候了，因为 TorchInductor 使用 Triton 作为其后端代码生成的实现——也因为在 OpenAI 这类大公司的支持下，Triton 正日益流行。

如前所述，Inductor 在底层使用 Triton 来生成经过优化的 GPU 核函数。通过检视、理解并定制这些核函数，你可以把性能进一步推高到超出 TorchInductor 所能产出的水平。在 PyTorch 与 NVIDIA GPU 环境中，学习 Triton 对性能优化至关重要。

从高层看，OpenAI Triton 是一种开源、Python 原生的领域特定语言（domain-specific language，DSL），用于以熟悉的 Python 编写 GPU 核函数。Triton 还包含一个 JIT 编译器，可将 Triton 代码直接转换为 NVIDIA PTX 代码。换言之，Triton 让你能够用 Python 创建高性能的自定义 GPU 算子——而无需手写 CUDA C++。Triton 与 PyTorch 保持紧密集成，使其成为该生态系统中编写自定义 GPU 核函数的首选。

用 Triton 编写 GPU 核函数比用 CUDA C++ 更为熟悉、也更简单。对那些倾向于留在 Python 中、快速迭代、不愿操心复杂 C++ 模板或细致内存管理的研究者而言，尤其如此。在 PyTorch 和 Triton 这类专注 GPU 性能的编译器已经存在的时代，他们根本无需再使用 C++。

> NVIDIA 已经注意到了这一趋势。2025 年，他们发布了以 Python 为中心的 CUDA 库（例如 cuTile、CuTe Python DSL、CUTLASS Python DSL 以及作为 numpy 替代品的 cuPyNumeric）。这些本质上是与 Triton 竞争的库。与 torch.compile 的集成仍在持续演进，而在本文写作之时，TorchInductor 仍以 Triton 作为其主要的 GPU 代码生成路径。

虽然 PyTorch 的 torch.compile 自动化了大量的核函数生成工作，但自定义 Triton 核函数仍能榨出最后一点性能——尤其是对于超出 TorchInductor 当前覆盖范围的算子，如复杂的稀疏模式和新型层类型。有时确实可以击败 TorchInductor 生成代码的性能——如果你具备领域特定的知识，就更是如此。然而，这非常高阶，并且需要持续维护，还可能因支持新硬件而需要重写。

现在，让我们从一段简短的 Triton 编程入门开始。随后我们会深入一些有趣的 Triton 主题，包括访问共享内存（shared memory）、向 PyTorch 注册 Triton 核函数、自动调优（autotuning）核函数启动参数以及剖析。接着我们会进一步覆盖诸如 warp 专门化（warp specialization）和软件流水线（software pipelining，例如双缓冲（double buffering））等进阶的 Triton 主题。

### Triton 编程模型

Triton 采用单程序多数据（single-program, multiple-data，SPMD）模型，而非 CUDA 的 SIMT 模型。这一点很重要，因为 Triton 有意抽象掉了 CUDA 指令和线程的底层细节。

Triton 核函数（又称 _程序_）在更高的层面运作：它把程序的多个实例运行在各自独立的线程块（thread block，又称 _协作线程数组_，即 cooperative thread arrays，CTA）上，并以此作为基本的计算单元。这与 CUDA 核函数形成对比——后者运行在线程块内的单个线程上。

> 社区往往把 Triton _kernel_ 和 Triton _program_ 混用——通常更偏爱 Triton _kernel_，因此本书大多数情况下使用 Triton _kernel_（核函数）。

你用 Triton 的 Python DSL 编写 Triton 核函数。然后 Triton JIT 编译器把该核函数编译成 GPU 代码，从而运行该核函数的众多并行实例。每个程序实例都映射到一个 CUDA 线程块。

Triton 核函数（又称 _程序_）通过用 @triton.jit 装饰一个 Python 函数来定义。在核函数内部，你使用来自 triton.language 模块（通常别名为 tl）的特殊原语来操作内存指针、执行向量化的加载/存储，并使用 tl.program_id 和块偏移算术来计算每个程序各自的索引。

Triton 的 SPMD 模型意味着你通常处理的是向量化操作，例如把两个 tl.arange 向量相加。Triton 编译器会把向量化的 SPMD 代码映射到 CUDA 块中的各个线程上。它并不保证元素与线程之间是一一对应的映射。

使用 Triton 时，你无需显式管理单个线程或 warp，因为它的编译器会替你完成这些工作。下面是一个简单的 Triton 核函数，它把两个大小相等（本例中为 n_elements）的向量相加：

```
import triton
import triton.language as tl
BLOCK_SIZE = 1024
@triton.jit
def vector_add_kernel(x_ptr,y_ptr,out_ptr,n_elements,BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)              # unique program ID for each block
    block_start = pid * BLOCK_SIZE
    # each program handles BLOCK_SIZE elements
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard against out-of-bounds
    # (if n is not divisible by BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)        # masked load
    y = tl.load(y_ptr + offsets, mask=mask)
    result = x + y
    tl.store(out_ptr + offsets, result, mask=mask) # masked store
```

在这里你可以看到，Triton 抽象掉了线程和 warp。注意，BLOCK_SIZE 是一个编译期常量，用于定义每个程序实例处理多少个元素。每个 CUDA 块的线程数由核函数配置通过 num_warps 控制，并不等于 BLOCK_SIZE。

具体来说，在前面的代码中，tl.arange(0, BLOCK_SIZE) 返回一个大小为 BLOCK_SIZE 的索引向量（[0, 1, ..., BLOCK_SIZE-1]）。我们把 pid \* BLOCK_SIZE（即 block_start）加到该索引向量上，从而推导出运行在某个线程块上的这个核函数实例进入每个向量的实际索引 x_ptr + offsets 和 y_ptr + offsets。

假设我们启动了足够多的 Triton 核函数实例，以覆盖每个向量中元素的总数 n_elements，那么该核函数就会把两个向量 x_ptr 和 y_ptr 的每一个元素相加，并把结果存入 out_ptr。本质上，Triton 让你能以张量化的方式编写核函数逻辑。

例如在这里，我们一次性对整整一个索引块（offsets）进行操作。Triton 编译器负责把这项工作拆分到实际的 GPU 线程之间，并确保内存访问（tl.load 和 tl.store）在可能时被合并（coalesce）。

要启动该 Triton 核函数的实例，需传入一个 grid 函数，它根据 meta['BLOCK_SIZE'] 计算程序实例的数量：

```
import triton
def grid(meta):
    return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
vector_add_kernel[grid](x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE=1024)
```

在这里，代码使用一个掩码（mask）来避免当 n_elements 不是 BLOCK_SIZE 的整数倍时发生越界内存访问。这与前面讲 CUDA 的章节类似——那时我们在核函数中使用 if (idx < N) 来避免越界索引错误。

> 在加载/存储中使用掩码，是一种巧妙而便利的边界条件处理方式，无需显式检查或 if/else 分支。

在底层，Triton 会把这个程序转换为 NVIDIA PTX，使每个程序使用单个 CUDA 线程块。每个程序都映射到一个 CUDA 线程块。tl.arange 在程序内生成逐通道（per-lane）的索引，编译器再把这个向量化的索引空间映射到块中的各个线程上。你也可以以直观的方式管理矩阵运算所需的多维索引。Triton 会自动为你处理算术与内存操作的向量化。

简言之，Triton 让你在获得 Python 的开发效率的同时，享有经过优化的 CUDA C++ 核函数的性能。它还允许你下沉到底层优化，以操控并充分利用整个内存层次结构（例如共享内存分块等），下一节我们会演示这一点。

### 在 Triton 中访问共享内存

高效的 Triton 核函数会利用 L2 缓存和每个 SM 上由软件管理的共享内存。使用共享内存时，每个线程块会把来自矩阵 A 和 B 的一个分块（tile）加载到共享内存中。这与每个线程反复从全局内存加载相同数值形成对比。

随后，核函数会复用这些分块进行多次计算。这样能更好地利用片上内存缓存，并减少在全局 HBM 与寄存器（register）之间往返的数据量。

Triton 并不暴露显式的共享内存分配器。相反，它使用张量描述符（tl.make_tensor_descriptor(...)）以及一条按预期形状与步长构建的异步流水线，把分块暂存在片上共享内存中。这样，你就可以在一个流水线化的 tl.range(..., num_stages=...) 内部通过这些描述符发起加载与存储。该循环会下降（lowering）为 cp.async、TMA 和屏障。

### 向 PyTorch 注册自定义核函数

写好一个 Triton 核函数后，你可以使用 torch.library.triton_op 把它注册为 PyTorch 中的自定义算子。这让 Triton 核函数对 torch.compile 可见，而不是被当作一个可能回退到即时执行模式的不透明黑盒算子。这样一来，编译器就了解该 Triton 核函数、会在图捕获期间将其纳入，并把它与图的其余部分一起优化。由此便可实现诸如融合之类的额外优化。

注册 Triton 核函数有助于在把自定义 Triton 核函数/程序与 PyTorch 编译器搭配使用时避免图中断。下面是一个从 PyTorch 中注册并调用 Triton 核函数 vector_add_kernel 的示例：

```
import torch
import triton
import triton.language as tl
from torch.library import triton_op, wrap_triton
from torch import Tensor
# Triton compute kernel
@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, out_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)
# Register as a Triton-backed PyTorch op
@triton_op("my_triton_lib::vector_add", mutates_args=())
def vector_add(x: Tensor, y: Tensor) -> Tensor:
    assert x.device.type == "cuda" and y.device.type == "cuda"
    n = x.numel()
    out = torch.empty_like(x)
    # Compute grid size
    def grid_fn(meta):
        return (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    # Wrap and launch the Triton kernel
    wrap_triton(vector_add_kernel)[grid_fn](x, y, out, n, BLOCK_SIZE=1024)
    return out
# Usage
a = torch.randn(10_000, device="cuda")
b = torch.randn(10_000, device="cuda")
c = torch.ops.my_triton_lib.vector_add(a, b)
```

在这里，triton_op("my_triton_lib::vector_add", mutates_args=()) 向 PyTorch 注册了算子名称和（为空的）变异元数据。接着，wrap_triton(vector_add_kernel) 把原始的 Triton 核函数包装成一个可调用对象，使编译器能够在 torch.compile 图内对其进行内联和优化。随后，编译器会在 torch.compile 图的其余部分中对该核函数进行融合、重排和内联。

注册仅前向的算子很直接。然而，要利用 PyTorch 的自动微分以获得完整的训练支持，你通常需要实现并注册一份自定义的反向计算。否则，你就得用现有的可微原语把它组合出来。

要支持训练，请使用 vector_add.register_autograd(backward, setup_context=setup_context) 注册一个自动微分公式。如果你愿意，也可以把逻辑包装进一个 torch.autograd.Function，并同时注册前向和反向参数。不过，register_autograd 才是获得 torch.compile 可组合性的推荐路径。

> 如果 OpenAI Triton 不支持你所需的某项功能——或者未能提供你所期望的性能——你可以借助像 CUTLASS 这样的库用 CUDA C++ 重写该核函数以获得更高效率。之后，我们会以类似的方式把该 CUDA C++ 扩展注册到 PyTorch，包括为反向传播注册自动微分梯度计算。

### 调优核函数启动参数

对许多核函数而言，Triton 程序通常每块使用 4 个 warp，即 128 个线程。然而，随着现代 GPU 硬件每个 SM 的共享内存和寄存器文件更大，你通常可以把 num_warps 提高到每块 8 或 16 个 warp。例如，你可以在 BLOCK_SIZE >= 2048 时把 num_warps 提高到 8，在 BLOCK_SIZE >= 4096 时提高到 16。

warp 的数量取决于你的核函数能否在不引发过度争用的情况下利用这种并行性。最优设置取决于核函数的算术强度和内存访问模式。

考虑如下方式启动核函数：my_kernel[grid](..., num_warps=8)。在这种情况下，我们为每个 Triton 核函数指定 8 个 warp（256 个线程）。这种配置对计算密集型核函数通常有效。然而，受内存带宽限制，内存受限型核函数可能仍会在约 4 个 warp 处触顶。

对于内存受限型核函数，每个线程块使用更多的 warp 有助于通过并行执行更多操作来隐藏内存延迟。但每个线程块的 warp 过多可能引发争用或缓存抖动。

新一代 GPU 拥有更多的 SM 和更宽的内存总线。这让我们能够把每块的 warp 数从默认的 4 个提高到 8 或 16 个。这有助于提升占用率（occupancy）、掩盖更多内存访问延迟，并使可用的内存与计算达到饱和。

为每个核函数手动探索 BLOCK_SIZE 与 num_warps 的各种组合会很繁琐。因此，通常最好使用 Triton 内置的自动调优器。它会替你基准测试并自动挑选最优的 BLOCK_SIZE、num_warps、分块大小以及其他参数。下一节我们就来探讨自动调优器。

### 自动调优 Triton 核函数

GPU 核函数性能对编译期参数高度敏感，例如分块维度、warp 数量、循环展开阶段，以及对寄存器和共享内存等片上资源的使用。Triton 内置的自动调优器让你能够用 @triton.autotune 装饰一个 triton.jit 核函数，从而自动搜索这些最优设置。你可以传入一组 triton.Config 对象，用以描述 BLOCK_SIZE、num_warps、num_stages、分块大小以及其他核函数元参数的不同候选组合。

在首次调用核函数期间，Triton 会对每一种配置组合进行 JIT 编译并基准测试。请务必在这次初始调用时使用具有代表性的输入负载，因为 Triton 会根据从输入特征（如输入大小/形状）派生出的键，把最快的配置缓存起来，用于该输入。

此后所有具有相同输入特征的调用都会自动复用已缓存的（最快的）配置。这样，你只需为每种输入大小/形状支付一次自动调优的成本——并可在之后的核函数调用中立即受益于最优配置。

如果 Triton 检测到新的输入形状，它会针对新的输入特征再次遍历 triton.Config 对象，执行一次新的自动调优过程。它会再次为该输入选出最佳配置，并将其缓存以供后续核函数调用使用。

为避免次优的调优结果，建议你用贴近生产负载的真实且有代表性的输入来预热自动调优器。这样，Triton 便能用一份紧密反映你生产输入的最优配置来填充缓存。

> 你可以通过向 @triton.autotune(key_fn=...) 提供自定义的 key_fn，把输入元数据（如张量形状）映射到自定义缓存键，从而针对特定的输入形状和负载覆盖最优设置。这是一项高阶技巧，能让你对不同类型输入负载的缓存配置拥有更多控制权。

在选择可能的核函数配置时，值得记住的是：更大的分块和更多的 warp 会以消耗每个线程块更多的寄存器和共享内存为代价，提升算术强度。换言之，提高计算与内存之比会限制占用率，因为资源需求增加，能在每个 SM 上执行的线程块变少。

反过来，使用更小的分块和更少的 warp 会减少每线程的工作量与数据复用，但允许每个 SM 上并发活跃更多的块和 warp。这以更低的算术强度为代价换取更高的占用率。

简言之，最优的权衡取决于你的输入矩阵维度以及 GPU 的具体资源上限。手动调优既耗时又易出错。Triton 的自动调优器会以数据驱动的方式在真实负载上自动处理这种复杂性，确定手动搜索可能错过的最优配置。使用更高的 num_warps（例如 8–16）以及多阶段流水线，往往能在 Blackwell 上使 tcgen05.\* 路径达到饱和。建议尽可能多地使用自动调优。

## 进阶 Triton 核函数实现

为巩固这些概念，接下来给出一些自包含的 Triton 核函数示例，涉及 warp 专门化以及数据传输/计算的异步双缓冲。它们展示了如何用 Triton 把高层 Python 代码转化为高度优化的 GPU 核函数。

### 用 Triton 实现 warp 专门化

TorchInductor 可以为其生成的许多 GPU 核函数瞄准 Triton 的 warp 专门化支持。它会尝试把每个线程块的 warp 拆分为“生产者”（内存）和“消费者”（计算）两种角色，做法是发出带 warp_specialize=True 的 tl.range() 循环，类似下面所示的示例：

```
// warp_specialize=True is supported on modern GPUs
// Use it together with num_stages > 1
// to enable producer/consumer warp partitioning
// and overlap
for k in tl.range(0, K_tiles, _warn_unused=False, warp_specialize=True):
    # loop body
    ...
```

内存 warp 会在另一个 warp 计算当前分块的同时预取下一个分块。这会将内存延迟与计算重叠，从而带来更高的吞吐量。warp 专门化与基于描述符的 TMA 拷贝协同工作。你也可以在自己的自定义 Triton 核函数中使用它，方法是向 tl.range() 传入 warp_specialize=True，如代码所示。

你还可以通过 Triton 自动调优配置来驱动 warp 专门化，做法是在 triton.Config 中设置 num_consumer_groups>0（例如 2）和 num_buffers_warp_spec（例如 3），如下面的代码片段所示。这会让生产者和消费者始终保持忙碌。如果提供了这些值，TorchInductor 会在底层使用它们：

```
triton.Config(
    { 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,
      'num_warps': 8, 'num_stages': 2,
      'num_consumer_groups': 2, '
      num_buffers_warp_spec': 3 }
)
```

这种专门化方法对于在 GEMM 中反复迭代一个较大 _K_ 维度的长时间运行循环尤其有效。这种专职分工的方式让内存子系统和 ALU 始终保持忙碌，从而最大化硬件利用率。

### 分块与持久化 GEMM 核函数（Triton）

这个 Triton 核函数高效地计算矩阵乘法（C = A \* B），因为每次核函数启动都通过内部对 K 维度循环来完成全部工作，而不是为每个 K 分块启动多个核函数。这样，我们只需支付一次启动开销，并且 warp 会一直保持忙碌，直到每个分块都完成。下面的示例在一次启动内对 K 进行分块，但不会在多个输出分块之间复用同一个线程块：

```
@triton.jit
def tiled_gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Tiled GEMM with Triton tensor descriptors + autotuning.

    This is the BASIC PRODUCTION example showing:
    1. Tensor descriptors (maps to TMA on Blackwell)
    2. Autotuning across block sizes
    3. Standard 2D grid decomposition
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    m0 = pid_m * BLOCK_M
    n0 = pid_n * BLOCK_N
    offs_m = m0 + tl.arange(0, BLOCK_M)
    offs_n = n0 + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # On Blackwell, descriptor .load/.store map to TMA
    # tl.dot lowers to UMMA (tcgen05) with accumulators in TMEM.
    A_desc = tl.make_tensor_descriptor(
        A_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        block_shape=[BLOCK_M, BLOCK_K],
    )
    B_desc = tl.make_tensor_descriptor(
        B_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        block_shape=[BLOCK_K, BLOCK_N],
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    K_tiles = (K + BLOCK_K - 1) // BLOCK_K
    if K_tiles == 0:
        c_ptrs = C_ptr + (offs_m[:, None] * stride_cm
                          + offs_n[None, :] * stride_cn)
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, acc, mask=c_mask)
        return
    k0 = 0
    if (m0 + BLOCK_M <= M) and (k0 + BLOCK_K <= K):
        a_cur = A_desc.load([m0, k0])
    else:
        col_ids = k0 + offs_k
        row_offsets = offs_m[:, None] + tl.zeros((BLOCK_M, BLOCK_K),
                                                  dtype=offs_m.dtype)
        col_offsets = col_ids[None, :] + tl.zeros((BLOCK_M, BLOCK_K),
                                                   dtype=col_ids.dtype)
        a_cur = tl.load(
            A_desc,
            offsets=(row_offsets, col_offsets),
            boundary_check=(0, 1),
            padding_option="zero",
        )
    if (n0 + BLOCK_N <= N) and (k0 + BLOCK_K <= K):
        b_cur = B_desc.load([k0, n0])
    else:
        row_ids = k0 + offs_k
        row_offsets = row_ids[:, None] + tl.zeros((BLOCK_K, BLOCK_N),
                                                   dtype=row_ids.dtype)
        col_offsets = offs_n[None, :] + tl.zeros((BLOCK_K, BLOCK_N),
                                                  dtype=offs_n.dtype)
        b_cur = tl.load(
            B_desc,
            offsets=(row_offsets, col_offsets),
            boundary_check=(0, 1),
            padding_option="zero",
        )
    for kt in tl.range(0, K_tiles, num_stages=2):
        k0 = kt * BLOCK_K
        acc += tl.dot(a_cur, b_cur)
        next_k = k0 + BLOCK_K
        if next_k < K:
            if (m0 + BLOCK_M <= M) and (next_k + BLOCK_K <= K):
                a_cur = A_desc.load([m0, next_k])
            else:
                col_ids = next_k + offs_k
                row_offsets = offs_m[:, None] + tl.zeros((BLOCK_M, BLOCK_K),
                                                          dtype=offs_m.dtype)
                col_offsets = col_ids[None, :] + tl.zeros((BLOCK_M, BLOCK_K),
                                                           dtype=col_ids.dtype)
                a_cur = tl.load(
                    A_desc,
                    offsets=(row_offsets, col_offsets),
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
            if (n0 + BLOCK_N <= N) and (next_k + BLOCK_K <= K):
                b_cur = B_desc.load([next_k, n0])
            else:
                row_ids = next_k + offs_k
                row_offsets = row_ids[:, None] + tl.zeros((BLOCK_K, BLOCK_N),
                                                           dtype=row_ids.dtype)
                col_offsets = offs_n[None, :] + tl.zeros((BLOCK_K, BLOCK_N),
                                                          dtype=offs_n.dtype)
                b_cur = tl.load(
                    B_desc,
                    offsets=(row_offsets, col_offsets),
                    boundary_check=(0, 1),
                    padding_option="zero",
                )
    # Store results with masking
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

def persistent_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=torch.float32)
    MT = triton.cdiv(M, 128)
    NT = triton.cdiv(N, 128)
    grid = lambda META: (min(65536, MT * NT),)  # bound launch overhead
    matmul_kernel_persistent[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C
```

在这里，核函数在 M×N 分块上启动一个二维网格，并在单次核函数启动内部执行完整的 K 循环。这减少了启动开销，并在 K 较大时能提升利用率，但代价是在单个核函数中占用资源的时间更长。每个程序（线程块）把 A 和 B 的分块加载到共享内存中，并用 tl.dot 计算这些分块的部分点积。Triton 以 FP32 累加结果。而在 Blackwell 上，Triton 会把 tl.dot 下降为 tcgen05 和 UMMA，以调用张量核心（Tensor Core）。张量核心随后会在专用的 TMEM 中累加结果，而不是在通用寄存器中。

> 在 Triton 中，最好使用张量描述符来表达由共享内存支撑的分块搬运。例如 desc=tl.make_tensor_descriptor(...)。在现代 GPU 上，这些张量描述符调用会映射到基于 TMA 的硬件操作，使用异步、合并的传输。

Triton 会为提升效率而展开/向量化这些循环与计算。此外，当数据类型和分块形状受支持时（例如 FP16/BF16，或用于 FP32 的 TF32），Triton 会把 tl.dot 下降为张量核心指令。下面是一个用于启动该 Triton 核函数的简单 Python 包装器：
