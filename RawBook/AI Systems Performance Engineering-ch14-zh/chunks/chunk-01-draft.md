# 第 14 章 PyTorch 编译器、OpenAI Triton 与 XLA 后端

在第 13 章中，我们讨论了优化与调优基于 PyTorch 的训练和推理工作负载的多种方法，并简要介绍了 PyTorch 编译器，以及它如何自动完成核函数融合（kernel fusion）等核函数级技术，从而在几乎不改动代码的情况下提升性能。

本章我们将深入这套动态的 PyTorch 编译栈，涵盖 TorchDynamo、提前自动微分（Ahead-of-Time Autograd，AOT Autograd）、PrimTorch 中间表示（intermediate representation，IR）（又称 _Prims_ 或 _Prims IR_）等组件，以及 TorchInductor、加速线性代数（Accelerated Linear Algebra，XLA）、OpenAI 的 Triton 生态等编译器后端（compiler backend）。PyTorch 编译栈如图 14-1 所示。

我们还会介绍用于调试编译流水线的工具，以及在多 GPU、多节点集群上扩展 PyTorch 的库。随后，我们将探究 torch.compile 的底层工作原理，以及如何高效处理动态形状（dynamic shapes）与可变序列长度。

我们还将考察 PyTorch 编译器与 OpenAI Triton 生态的集成。我们的目标是在不牺牲 PyTorch 灵活的即时执行（eager execution）开发体验的前提下，加速并扩展我们的 PyTorch 模型与应用。

![图 14-1. PyTorch 编译栈概览](AI%20Systems%20Performance%20Engineering-ch14_images/figure-14-1.png)

## PyTorch 编译器深入剖析

正如第 13 章所述，PyTorch 的 torch.compile 会编译你的 PyTorch 代码（以及模型），带来显著的加速。多数情况下，只需一行代码即可完成，如下所示。我们会在后文逐步讲解各种选项：

```
compiled_model = torch.compile(model,
  mode="max-autotune",
#  ...
)
```

本节拆解 PyTorch 编译流水线的各个步骤，包括 TorchDynamo 的图捕获、AOT Autograd 对前向/反向图的联合优化、PrimTorch IR，以及 TorchInductor 的代码生成（code generation）。这条流水线负责为目标 GPU 硬件生成优化后的核函数，如图 14-2 所示。

![图 14-2. PyTorch 编译器流水线（来源：https://oreil.ly/55JDn）](AI%20Systems%20Performance%20Engineering-ch14_images/figure-14-2.png)

### 用于字节码捕获与图提取的 TorchDynamo

TorchDynamo（简称 Dynamo）是 torch.compile 的第一个阶段。它挂接到 Python 的帧求值（frame-evaluation）机制中，在字节码层面拦截模型执行。

Dynamo 挂接到 CPython 的帧求值机制，识别出会产生张量的字节码区域，并为这些区域构建执行图，然后用所选后端执行编译后的图。不受支持的代码则留待以即时执行方式运行。

正是这种拦截与重写机制，使 TorchDynamo 能够把一连串 PyTorch 操作捕获为图表示，以便后续步骤——即接下来几节介绍的 AOT Autograd 与 PrimTorch IR——对其进行优化。

TorchDynamo 利用 CPython 帧求值 API（PEP 523）以最小的开销安全地捕获操作。通常，Python 解释器会逐个执行每个操作；但启用 Dynamo 后，解释器会把执行重定向到 Dynamo，由它先将张量操作聚合成图，再执行。这样便能进行核函数融合之类的全图优化，从而减少每个操作的 Python 及主机端开销。

编译后的图可以把许多操作融合进一个或少数几个核函数，而不必为每个操作都启动一个新的 GPU 核函数并承担逐操作的 Python 开销。这减少了分派开销，并改善了内存访问模式。

在第 13 章中，我们看到 Python 开销与大量小操作会成为模型的瓶颈；也看到 TorchDynamo 通过在可能时把小操作合并成更大的单元来解决这一问题——前提是这一串操作对图友好，且不会引发过多图中断（graph break）。

当 torch.compile 启用 TorchDynamo 后，它会检查每一条 Python 字节码指令。每当遇到 PyTorch 张量操作（如算术运算或神经网络层）时，它并不立即执行，而是把该操作作为一个节点追加到 FX 图（FX graph）中，并将已捕获区域的执行推迟给编译后端。TorchDynamo 会持续这一过程，直到遇到它无法处理的代码。此时便发生一次图中断（稍后详述），不受支持的操作则以常规的、未编译的即时执行模式（eager mode）运行。

TorchDynamo 会尽量把你程序中尽可能大的部分编译进单个图，但当遇到不受支持的结构（如复杂的控制流或非 PyTorch 库调用）而无法继续时，它会回退到 Python 即时执行。这样做是为了保证正确性。

在一段不受支持的操作结束之后，Dynamo 会从该点起把后续操作重新捕获进一个新图。这种编译与非编译混合执行的方式让你兼得两者之长：在需要之处保留 PyTorch 的灵活性，而将其余部分统统编译以换取速度。

> 应尽可能避免图中断，因为它们会打断全图优化，并限制编译器带来的性能收益。

你可以使用 torch.compiler.set_stance("fail_on_recompile") 强制 Dynamo 抛出错误，以捕捉不安全的重新编译（recompilation）。它会记录重新编译的原因，帮助你调试图中断，让你能够充分了解图为何被拆分。代码如下所示：

```
// fail on graph breaks, recompiles
torch.compiler.set_stance("fail_on_recompile")
compiled_model = torch.compile(model,
  mode="max-autotune",
  ...
)
```

这样，你就可以重构导致中断的代码路径——或者用 torch.\_dynamo.allow_in_graph() 把它们标记为预期中的图边界，后文会讲到。一旦图变得干净，你就可以切回 torch.compiler.set_stance("eager_on_recompile")。请记住，改回该设置后，若随后发生图中断，TorchDynamo 会静默地回退到即时执行模式。

TorchDynamo 捕获的产物称为你代码的 _FX 图_。FX 是一种中间表示（IR），其中每个节点都是对某个 PyTorch aten 算子或内置 Python 函数的调用。例如，考虑一个简单的 Python 函数，如下所示：

```
def f(x, y):
    z = x.sin() + y
    return z.sum()
```

此时，TorchDynamo 会生成一个大致等价于以下伪代码的 FX 图：

```
graph():  # pseudo-code for FX IR
    %x : Tensor = Placeholder[target=x]
    %y : Tensor = Placeholder[target=y]
    %sin : Tensor = torch.sin(%x)
    %add : Tensor = torch.add(%sin, %y)
    %sum : Tensor = torch.sum(%add)
    return %sum
```

这些 FX 图节点对应于 Placeholder 输入以及对 aten::sin、aten::add、aten::sum 等原始 ATen 操作的调用，清晰地表示了计算图结构。一旦构建完成，该 FX 图便交给 AOT Autograd，以进行前向与反向传播的联合追踪。

AOT Autograd 生成的前向与反向合并追踪随后被送往 TorchInductor 或 XLA 等后端编译器，以执行核函数融合并生成优化后的设备代码。TorchDynamo 自身则保持与框架无关，专注于准确、低开销的图捕获，并把所有繁重的优化都交给下游的编译器阶段。

TorchDynamo 会在影响图追踪的 Python 值上插入保护条件（guard）——例如张量形状、dtype，甚至全局变量。这些保护条件确保：如果某个被假定为常量的东西（如张量形状或 dtype）发生变化，编译后的图便会失效，并在必要时重新编译。这样便能稳健地处理动态形状与模型改动——代价是在假设被违反时会产生额外的重新编译。

> 建议在使用 PyTorch 编译器时监控图重新编译的次数。

在默认的 dynamic=None 下，编译器会根据观察到的形状进行特化（specialization）。当遇到动态形状时，它会重新编译一个更具动态性的核函数；如果之后又检测到新的动态来源，还可能再次重新编译。

实际上，这意味着在第一个变化形状的输入上你会看到一次额外编译——而且只有一次额外编译。如果你已经知道哪些输入维度会变化，可以在运行模型之前使用 torch.\_dynamo.mark_dynamic(tensor, dim)，从而提前避免这次初始编译。

你还可以用 torch.compiler.set_stance() 进一步调整编译器的“立场”（stance），以改变 TorchDynamo 何时以及如何回退到即时执行或重新编译。立场决定了编译器在面对错误或回退时的容忍度或严格程度，让你能更好地在开发者反馈与不间断执行之间权衡。以下是撰写本文时可用的立场：

default 编译器会尽量编译它能编译的部分，遇到不受支持的代码时静默回退到即时执行。这是标准的默认设置，产生正常的编译行为与回退。

fail_on_recompile 一旦遇到图中断或不受支持的操作，编译器就抛出错误。这在开发阶段很有用，便于你捕捉意外的中断或未编译的代码路径。

eager_on_recompile 当需要重新编译时，以即时执行模式运行代码。如果已有编译后的代码被缓存（且对该输入有效），则仍应使用它。

force_eager 使用即时执行模式，并忽略所有 torch.compile 指令。

通过捕获整段操作序列，TorchDynamo 能够执行核函数融合之类的全图优化，从而减少启动开销并尽量减少内存搬运。它还消除了这些被融合操作的 Python 层开销。图 14-3 对比了即时执行与编译两种模式，也包括编译器缓存。

编译器可以把许多操作融合进单个核函数，而不必分别执行每个小操作和每次核函数启动。这减少了 CPU–GPU 同步点，并改善了内存局部性。否则，GPU 会被大量细粒度操作、繁重的同步以及过多的全局内存访问拖成瓶颈。

![图 14-3. PyTorch 即时执行与编译模式对比](AI%20Systems%20Performance%20Engineering-ch14_images/figure-14-3.png)

TorchDynamo 会持续捕获，直到需要一次图中断为止。图中断可能由不受支持的 Python 结构触发（如某些控制流，例如使用 Python bool 而非张量操作的 if 语句），也可能由不受支持的操作触发。发生图中断时，当前图段结束，Dynamo 对无法捕获的代码回退到即时执行模式。之后，一旦回到可追踪的代码，TorchDynamo 就会开始一段新的图追踪。

你可以把逻辑表达为纯 PyTorch 张量操作（包括配合掩码使用的 torch.where()），从而在 PyTorch 中成功实现对编译器图友好的条件判断。下面这个例子用纯张量操作的掩码方式替换了一个对图不友好的 Python if 语句：

```
# Compute a boolean mask from a data-dependent tensor condition
# mask shape (batch_size,1) broadcastable to x’s
mask = x.sum(dim=1, keepdim=True) > 0
# Use torch.where to select element-wise between two tensor expressions
# picks f(x) where mask is True, otherwise g(x)
out = torch.where(mask, f(x), g(x))
```

这样就避免了会引发图中断的 Python if x.sum() > 0: 语句，做法是完全停留在 PyTorch 对图友好的张量操作之内。在这种情况下，TorchDynamo 会捕获整段序列，包括掩码和 torch.where()，而不会中断图。

随着每次新版本发布，PyTorch 都会扩充可在不引发图中断的情况下被捕获的操作范围。例如，torch.cond 这类高级条件原语可以在图中捕获某些 if/else 逻辑。具体来说，两个分支都会被追踪并编译。torch.cond 要求一个布尔标量谓词；两个分支必须返回相同的结构与 dtype；并且形状在运行时必须一致。依赖数据的分支往往会导致图中断。

> 建议通过用 torch.where() 等张量操作重构代码来尽量减少图中断，从而最大化 TorchDynamo 能够捕获的连续区域。

### AOT Autograd 对前向与反向传播的融合

一旦 TorchDynamo 为尽可能多的前向传播捕获了 FX 图，下一个编译器阶段便是 AOT Autograd。AOT Autograd 会以“函数式”模式让 Dynamo 捕获的前向图通过 PyTorch 的自动微分引擎运行，以记录反向操作。静态反向图正是这样产生的（这与依赖 PyTorch 默认自动微分引擎逐个执行反向传播操作形成对比）。

本质上，AOT Autograd 会生成一个联合的前向–反向图，随后便可作为整体进行优化与融合。而且它保证前向与反向结果与即时执行模式一致。

AOT Autograd 的工作方式是让前向图通过自动微分引擎进行追踪，以捕获梯度计算。它实际上会用 torch.autograd.forward_ad（或类似技术）运行前向图，记录反向计算所需的操作。其结果是一个前向与反向合并的图。这个合并图随后可以用公共子表达式消除（common-subexpression elimination）等手段优化，之后再由 TorchInductor 或 XLA 等后端编译。

通过提前规划前向与反向，PyTorch 编译器可以整体地跨越前向与反向传播的边界进行融合。这带来了跨这两个阶段的操作的提前融合。例如，它可以在可能时把前向传播中的一个逐元素操作与反向传播中相应的逐元素梯度计算融合进同一个核函数。

在安全的情况下，编译器还可以通过在前向与反向计算之间复用中间结果来消除开销。对于反向操作可能主导整体运行时间的模型训练工作负载来说，这能大幅提升性能。

若没有 AOT Autograd，PyTorch 就需要用默认的 PyTorch 自动微分引擎单独执行每个反向操作——与前向传播相互独立。有了 AOT Autograd，执行图便得到整体优化。

AOT Autograd 产生的联合图保证计算出相同的结果（这也是为什么当图无法保证正确性时需要图中断的原因）。由于前向与反向的整段操作序列都已知，这个联合图可以被进一步优化。

通过跨前向与反向传播融合操作和内存使用，我们减少了内存访问与核函数启动开销。当你对涉及梯度的操作和工作负载（如模型训练）调用 torch.compile 时，PyTorch 的编译模式会在底层自动使用 AOT Autograd。

简而言之，torch.compile 利用 AOT Autograd 在模型训练期间提前计算梯度。它能无缝处理大多数自动微分操作，并保证结果与即时执行模式一致；它还跨前向与反向传播复用缓冲区，以降低峰值内存占用。虽然这个阶段对最终用户基本不可见，但它是让现代 AI 模型训练工作负载获得大幅加速的关键组件。

### PrimTorch IR（Prims）精简算子集

在把图交给底层代码生成器之前，PyTorch 会执行一次中间表示（IR）变换，在部分文档和源码中称为 PrimTorch IR（Prims）。PrimTorch IR 是一种把图中操作的种类精简为一小组核心“原始”操作的 IR，这也是 _PrimTorch_ 这一名称的由来。

作为背景，PyTorch 的完整 API 拥有数千（> 2,000）个操作。PrimTorch IR 把它精简为一组小得多的原始操作，让编译器可以聚焦其上。实际上，PrimTorch IR 定义了约 250 个原始操作，如基本算术、归约、复制、reshape 等。

许多复杂或高级的 PyTorch aten 操作都可以借助 PrimTorch IR 分解（decomposition）为这些原始操作。例如，像 x.add\_(y) 这样的“原地”PyTorch 操作会被下降（lowering）为一次函数式 add，再跟一次显式地复制回 x 存储的操作，如下所示：

```
%z    = aten::add(x, y)
%copy = aten::copy_(x, %z)    # writes z’s data into x
```

这里，IR 中包含一个独立的 aten::copy\_ 节点，而不是某种特殊的原地变更。这使所有张量更新都变得显式，并通过把变更当作普通的复制操作来简化下游的编译器核函数。

具体而言，通过把原地变更转换为函数式操作外加独立的复制节点，编译器就不必再推理别名或隐藏的副作用。因此，融合过程与内存规划算法可以在纯函数式的数据流上运作，这就允许更激进的核函数融合以及可预测的高性能代码生成。

PrimTorch IR 还有助于消除图中的别名和变更等问题。它会尽可能把操作转换为不执行原地更新的形式，因为这些原地更新会使优化复杂化。PrimTorch IR 这一处理阶段的输出是一个只包含 aten IR 与 PrimTorch IR 操作的 FX 图。随后，该 FX 图便可交由后端进行下降。

通过这种 IR 标准化，PrimTorch IR 为编译器后端提供了一个稳定、简化的目标接口。像 TorchInductor 这样的后端无需实现成千上万个 PyTorch 操作，只需支持那 250 个原始操作即可——其他更高级的操作都由这些原始操作派生而来。这大大降低了复杂度。

而且，由于大多数高级操作（如 2,000 多个 PyTorch 操作）都能被分解为现有的原始操作，即便 PyTorch 为适配新算法、新模型和新技术而不断新增操作，PrimTorch IR 这组原始操作的集合演进得也相对缓慢。

> 稳定的 PrimTorch IR 接口意味着，例如要支持一款新的加速器，开发者只需实现那 250 个原始操作，而不必实现成千上万个 ATen 操作。

小结一下目前为止的流水线：TorchDynamo 捕获一个图，AOT Autograd 加入反向传播并形成一个联合的前向–反向图，然后 PrimTorch IR 把操作规范化为一组更精简、更原始的集合。到这一步，我们就以核心操作的形式获得了整个计算——包括前向与反向传播——的一个相当标准、与设备无关的表示。现在，让我们转向由编译器后端执行的实际代码生成阶段。

### TorchInductor 后端代码生成

torch.compile 栈的最后一个阶段是编译器后端。TorchInductor（简称 Inductor）是 PyTorch 的默认编译器后端。Inductor 接收由 aten 与 PrimTorch IR 操作组成的、经过优化并合并的前向与反向 FX 图，为目标硬件（包括 NVIDIA GPU、AMD GPU、CPU 等）生成高性能代码。

> XLA 是一个面向非 CUDA 硬件的备选后端。它主要通过 OpenXLA 项目用于 Google Cloud TPU，但也可以支持其他采用 XLA IR 的加速器。例如，Meta 自研的推理 ASIC——Meta 训练与推理加速器（Meta Training & Inference Accelerator，MTIA）就使用 XLA。此外，AWS 定制的 Inferentia 与 Trainium 加速器芯片在其开源的 AWS Neuron SDK 中通过 XLA 编译器运行 PyTorch。不过，NVIDIA GPU 通常使用 TorchInductor。

TorchInductor 的工作方式是把图 _下降_ 为一种循环级、随运行定义（define-by-run）的 IR，再将其编译为高效代码。在内部，Inductor 把操作表示为对多维数据的循环。它会在可能时自动把 FX 图中的节点分组为融合的循环块，每一组在代码生成时都会成为一个核函数。

Inductor 还支持符号化形状（symbolic shapes），从而允许动态维度。这种 IR 比原生 CUDA 稍高级，因为它把逐元素计算之类的东西表示为对张量索引的循环。该 IR 可供用户检视，有助于调试甚至扩展。

TorchInductor 的 IR 还可用于配合 torch.export() 和 AOTInductor 的提前编译流程。AOTInductor 会编译由 torch.export 产生的产物，用于打包和部署等 AOT 场景，从而可以跨多次运行保存并复用编译后的代码。图 14-4 在 torch.compile() 的语境中突出展示了导出流程。

![图 14-4. PyTorch 编译与导出（TorchDynamo → AOT Autograd → PrimTorch IR → TorchInductor → Triton/LLVM NVPTX）；通过 torch.export/AOTInductor 导出；形状为静态时使用 CUDA Graphs](AI%20Systems%20Performance%20Engineering-ch14_images/figure-14-4.png)

对于 NVIDIA GPU 后端，TorchInductor 使用 OpenAI 的 Triton JIT 编译器来生成实际的 GPU 核函数。Triton 是一种用 Python 编写的、类似 CUDA 的领域特定语言（domain-specific language，DSL）。Triton 还包含针对其 DSL 的编译器（稍后我们会更多地介绍 Triton）。

TorchInductor 会把它的循环级 IR 翻译成 Triton 代码，然后用 Triton 编译器借助 LLVM 直接把 Triton 代码转换为 NVIDIA PTX。请记住，PTX 是 NVIDIA 为其 GPU 设计的低层指令集架构（instruction set architecture，ISA）。

> 重要的是，Triton 使用 LLVM NVPTX 下降到 NVIDIA PTX，而不会调用 NVCC 来编译核函数。这种方式让 TorchInductor 能够即时生成为你特定模型或算法量身定制的自定义核函数。

循环级 IR 用 Python 实现，因而易于检视和扩展。例如，假设某个图中有一个操作 z = x.permute(1,0) + x[2,:]，Inductor 可能会用下面的 IR 来表示它：

```
def inner_fn(index: List[sympy.Expr]):
    i1, i0 = index  # index variables for dims
    tmp0 = ops.load("x", i1 + i0*size1)         # x[i1, i0]
    tmp1 = ops.load("x", 2*size1 + i0)          # x[2, i0]
    return ops.add(tmp0, tmp1)                 # elementwise add
torchinductor.ir.Pointwise(
    device=torch.device("cuda"), dtype=torch.float32,
    inner_fn=inner_fn, ranges=[size0, size1]
)
```

这里，size0 和 size1 是输入 x 的各个维度，而 inner_fn 描述了如何计算输出的一个元素。Pointwise 节点表示一个在这些范围上的嵌套循环，它逐元素地应用 inner_fn 来产生输出。

这是一种随运行定义风格的 IR。运行这段 IR，实际上就是在执行进行迭代并调用 ops.load 与 ops.add 的 Python。随后，Inductor 会使用 Triton JIT 编译器和 LLVM 生成对应的 NVIDIA PTX 代码。

> 使用 torch.library.wrap_triton 配合 triton_op，可以把一个 Triton 核函数注册为具备自动微分与伪张量（fake-tensor）支持的一等 PyTorch 算子。这意味着你可以编写一个 Triton 核函数，并让 TorchInductor 把它作为你模型图的一部分来优化。

### 使用 TorchInductor 进行自动调优

TorchInductor 内置了一个基于 Triton 自动调优（autotuning）能力构建的自动调优器（我们将在后面的小节中介绍）。这个自动调优器会为每个生成的 GPU 核函数找到最佳的启动配置。自动调优得到的配置会按核函数分别缓存，使得后续运行无需重做调优步骤。

第一次用 TorchInductor 后端编译代码时，它会花费额外时间，使用不同的块大小、分块大小等对各种核函数变体进行基准测试。Inductor 会挑选最快的变体并在之后一直使用它。核函数自动调优会增加初始的编译时延迟，但由此得到的核函数在运行时高度优化。

> 如果你还记得上一章的内容，这种激进的自动调优被称为 max-autotune 编译器模式。这是最耗时的编译器模式——而这正是其底层所发生的事情。

除了核函数融合与自动调优，TorchInductor 还会施加许多底层优化。这些优化包括：用于减少循环中复杂索引运算的索引化简、生成代码内部的公共子表达式消除，以及用于复用缓冲区、减少分配的高效内存规划。

TorchInductor 还使用 CUDA Graphs 在运行时捕获核函数序列，以极小的 CPU 开销实现更快的图重放。默认情况下，Inductor 会尝试把它生成的核函数包进一个 CUDA Graph，以减少每次迭代的启动开销。这对推理尤其有利——或者在运行任何含有大量核函数的代码或模型时也是如此。

> 第 13 章介绍的 reduce-overhead 与 max-autotune 编译器模式会触发对 CUDA Graphs 的使用。然而，CUDA Graphs 要求静态形状，因此在启用动态形状编译时不会使用它们。换句话说，如果用 dynamic=True 启用了动态形状，TorchInductor 就不会使用 CUDA Graphs。此外，当你需要自动调优但不想进行 CUDA Graph 捕获时，可以使用 max-autotune-no-cudagraphs。一般来说，先从默认模式开始，再用 max-autotune 以可观的编译时间为代价，为大型/关键工作负载提供额外加速。对于较小的模型，你可能看不到太多收益。

TorchInductor 的最终产物是为你的工作负载生成的、高度优化的设备专用代码。在许多情况下，Inductor 能达到接近、甚至超过手工调优库的性能。例如，Inductor 可以把一整串逐元素操作（包括激活和逐点变换）融合进单个核函数；它甚至能把某些“矩阵乘法后接逐元素操作（如 bias-add + 激活）”的模式融合进一次启动。这类工作手工完成相对困难——而且需要持续维护。

PyTorch 编译器使用启发式方法和后端，可能会把某些操作（如大型 GEMM（通用矩阵乘法，general matrix multiply））路由到 cuBLAS/cuBLASLt 或 CUTLASS 等高性能库，甚至能为可融合的模式发出 Triton 核函数。实际上，TorchInductor 会选择并缓存表现最佳——或已知对给定形状最优——的那条路径。

对于 transformer 模型，TorchInductor 会把围绕大型 GEMM 的 layernorm 和残差连接逐元素操作融合起来——同时仍用 cuBLAS 来完成实际的 GEMM 计算本身。或者，对于内存访问不规则的模型，Inductor 的自定义 Triton 核函数可以只做必要的工作，从而胜过现有库的核函数——而这类情形并不能从 cuBLAS 或 CUTLASS 这样的通用库中获益。

在现代 GPU 上，PyTorch 编译器可以配合 NVIDIA 的 Transformer Engine（TE）处理某些 transformer 块和层。不过，当你调用 torch.compile 时，PyTorch 并不会自动替换为 NVIDIA TE 核函数。TE 是一个独立的库，你必须通过它的模块或融合算子显式使用。但当你调用 TE API 时，torch.compile 可以围绕它们进行编译和融合。这与生成的 Triton 核函数相辅相成，以提供最高性能。

> 只有当你打算在模型中直接调用 TE 模块（例如 transformer_engine.pytorch.layers）时，才需要安装 NVIDIA Transformer Engine。torch.compile 不会自动把 TE 核函数替换进一个普通的 PyTorch 模型。

本质上，TorchInductor 承担了把代码和模型转化为高性能 GPU 核函数的繁重工作。随着 PyTorch 的每次发布，硬件覆盖范围在扩大，新的优化技术也在不断涌现。例如，PyTorch 提供了 FlexAttention，这是一个新的注意力算子，TorchInductor 可以把它编译成接近 FlashAttention 性能的融合核函数。具体来说，经测量，FlexAttention 的融合核函数在前向和反向传播中都能达到现代 FlashAttention 性能的至多约 85–90%，同时提供更多灵活性，包括块稀疏（block-sparsity）和自定义掩码。要启用这些快速路径，请把下列 torch.backend.cuda 属性设为 True：

```
# Ensure SDPA fast paths are enabled
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
```

使用 torch.nn.attention.flex_attention 时，请确保你的输入满足快速路径的约束条件。这样，TorchInductor 才能发出融合的 Triton 核函数。

TorchInductor 和 Triton 在现代 GPU 上支持自动化的 warp 专门化（warp specialization）。编译器会在认为有益时有选择地启用它。warp 专门化可以通过 num_consumer_groups 和 num_buffers_warp_spec 等 Triton 元参数进行调优。这些优化会进一步提升 GEMM 吞吐量。Triton 的自动 warp 专门化在包括 Blackwell 在内的现代 GPU 目标上支持 TMA 和张量描述符 API（例如 tcgen05）。

> 建议使用基于描述符的分块加载/存储来映射到 TMA 并降低寄存器压力。相比手写的 tl.load 循环，这种方法更受推荐。

简而言之，许多模型在使用 torch.compile 后运行得明显更快，尽管确切的收益取决于你模型的特性。第一次运行 torch.compile 时，你要付出编译与自动调优的成本，但后续运行会使用缓存的图和核函数，实现闪电般的执行速度。

### 动态形状与可变序列长度

LLM 训练与推理的一大挑战在于序列输入的尺寸可变。在传统的编译器和加速器中，变化的形状往往会导致重新编译，否则就需要把输入填充到一个固定的公共尺寸。本节讨论 torch.compile 中的动态形状追踪如何处理变长序列。

所幸，PyTorch 编译栈的设计能够优雅地处理动态形状。具体来说，它通过使用 SymPy 库以符号方式表示未知维度，使模型能够接受不同的输入尺寸而无需每次都重新编译，这一点我们稍后会讲到。

如果观察到某些维度的大小发生变化，PyTorch 编译器会自动把它们标记为动态。TorchInductor 一开始采用静态假设，之后若检测到形状可变，便会在重新编译时进行泛化。对于第一个新形状，你通常会看到一次额外编译。通过预先设置 dynamic=True 标志，你会强制编译器从一开始就把所有维度都视为动态。不过请记住，设置 dynamic=True 会禁用 CUDA Graphs。更推荐的做法是只用 torch.\_dynamo.mark_dynamic() 标记代码中已知会变化的维度。

TorchDynamo 和 TorchInductor 会在追踪动态维度时插入类似 sequence_length <= 256（或你指定的任意范围）这样的保护条件，以生成能适配多种尺寸及尺寸范围的代码。例如，如果某个输出尺寸为 x.size(0) + y.size(0)，Inductor 可以把它表示为一个符号表达式，并确保生成的代码对任何满足保护条件的取值都能工作。

当 Dynamo 遇到 sequence_length 的新形状时，它会设置一个新的保护条件（如 sequence_length <= 1024），并在这个新假设下编译核函数——从那时起把该维度当作动态处理。之后，如果出现更长的序列而违反了该保护条件，编译器会重新编译一个能处理更大范围的新版本图。随着时间推移，它会为每个不同的形状范围逐步积累起一份已编译核函数的缓存。

你也可以用 torch.\_dynamo.mark_dynamic(tensor, dim) 手动在张量上标记预期的动态维度，以提前避免一次重新编译。你还可以使用 torch.compiler.set_stance()，它让你能够调整如何处理重新编译。例如，你可以使用 eager-on-recompile 立场，在若干次重新编译之后回退到即时执行模式。避免重新编译的最佳实践我们稍后讨论。
