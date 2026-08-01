TorchInductor 会在首次重新编译（recompilation）之后尝试对形状做泛化处理，而不是针对每一个新形状反复特化（specialization）。例如，它会在生成的核函数内部用 if 语句发出条件代码，从而让同一个核函数能覆盖一段 sequence_lengths 区间而不报错。这样就减少了为每一种尺寸单独编译的必要。

> 某些具有数据相关输出秩（data-dependent output rank）——或极其复杂索引——的操作，仍可能触发形状特化。在这些情况下，编译器会插入更多保护条件（guard）；如果这些保护条件频繁失效，你可能会看到伴随 mark_dynamic() 或 set_stance() 出现的频繁重新编译。

作为背景，一种简单但低效地处理变长序列、又无需支持动态形状（dynamic shapes）的做法，是把所有输入序列填充到批次中的最大长度。这样你就能对所有输入使用同一套静态计算。虽然填充简化了实现，但当输入长度差异很大时它非常低效，因为大量算力被浪费在毫无意义的填充 token 上。

如果最大长度远大于所有输入的平均长度，填充会损害 GPU 利用率。然而借助动态形状编译，我们可以让编译器生成只迭代到每个输入实际序列长度的代码。动态形状让你避免为变长输入做过度填充。

我们来看一个典型的基于文本的生成式 AI 场景：随着生成过程推进，序列长度会不断增长。使用动态形状编译能够持续优于即时执行（eager execution）——即便序列长度不断增加。

相反，如果为了使用静态形状而把所有输入都填充到 2 的幂次长度，就会引入大量被浪费的计算，并因张量尺寸更大而增加编译时间。换句话说，使用动态形状能带来更好的编译期性能与运行期性能，且使用更简单，因为你不必手动填充输入。

> 建议按尺寸对输入分桶（bucket），以限制不同形状的数量。这样就能对剩余的变化范围启用动态形状。这种混合方法既避免了过度的重新编译，又减少了填充浪费。

有了动态形状，你可以只编译一次，然后在不同形状的输入上复用同一个已编译模型。只要这些变化落在支持的范围内，一个已编译模型就能处理多种配置。

在内部，TorchInductor 使用 SymPy 库以符号方式表示动态维度。它会把这些符号沿 IR 传播，从而使 z.size(0) = x.size(0) + y.size(0) 这样的表达式能被符号化地处理。Inductor 会把各种条件归约为保护表达式。

如果某个保护条件因维度落在预期范围之外——或某个数据相关条件发生变化——而失效，Inductor 就会触发一次重新编译。本质上，TorchInductor 试图为一段尺寸区间编译出一个通用核函数，而不是只针对单一固定尺寸。

动态形状在近期版本中已显著改进。不过，如果编译器无法对某些操作做符号化处理，这些操作仍可能强制形状特化。在这种情况下，编译器可能插入更多保护条件；若这些条件频繁失效，就会导致频繁重新编译，从而抵消编译带来的收益。

数据相关的控制流仍会触发特化。对变化的序列长度使用动态形状，但不要用于真正数据相关的分支。

值得注意的是，在撰写本书时，CUDA Graph 回放要求静态形状（以及固定的内存地址）。而且对已实例化的图仅支持有限的参数更新。内存地址与核函数启动拓扑必须与捕获时保持兼容。因此，启用动态形状通常会禁用这些区域的图捕获。这会使编译器无法获得 CUDA Graph 带来的性能收益，包括降低核函数启动开销。

> 如果你指定了 reduce-overhead 编译模式却又设置了 dynamic=True，那么 reduce-overhead 提供的 CUDA Graph 优化将不会生效，因为你已声明形状可以变化。启用动态形状会改变保护条件与内存规划，进而禁用图捕获。实践中，只在形状稳定时使用 mode="reduce-overhead" 以获得 CUDA Graph。对于可变序列长度，优先选择 mode="default" 或 mode="max-autotune-no-cudagraphs"，并在 ±10–20% 范围内分桶/填充以限制重新编译。

建议对你的系统做剖析，判断动态形状对你的用例是否值得启用。在某些情况下，填充到固定尺寸、配合 CUDA Graph 使用静态形状，并通过免去为每个唯一长度重新编译来获得更高吞吐，可能是更好的选择。在另一些情况下，动态形状则更优。

你应当剖析不同的方案，找出最适合你的做法。在这样做时，务必监控内存使用。支持动态形状的代码会因需要额外的保护条件和为最大区间生成的通用代码，而带来略高的内存占用。

> 一条经验法则是：如果你的序列长度只有 10%–20% 的波动，那么定长填充很可能对你更有利。

简而言之，支持动态形状意味着你不必为 LLM 模型中常见的变长输入而禁用 torch.compile。通过支持动态形状，PyTorch 编译器能够跨不同输入尺寸执行核函数融合与其他优化。

### 禁用 PyTorch 编译器并回退到即时执行模式

如果你想在不改动代码的情况下完全禁用 torch.compile——这对性能 A/B 测试和隔离问题很有用——可以用 @torch.compiler.disable 装饰器来禁用该函数的编译。若需区域级作用域控制，可以把 torch.compiler.set_stance() 当作上下文管理器使用。这会强制代码以即时执行模式（eager mode）运行。例如，你可能希望对复杂的数据加载或一次性的初始化逻辑禁用编译，让已编译的图专注于计算部分。这对处理那些与追踪（tracing）配合不佳的代码同样有用，我们稍后会讲到。

或者，你也可以像下面这样直接改用 eager 后端：torch.compile(model, backend="eager")。这会让你的代码回退到即时执行模式运行。这样你就能轻松地在已编译模式与即时执行模式之间调试并比较正确性/性能结果。

torch.compiler.disable() 与 torch.compiler.set_stance() 是宝贵的“逃生舱”，可用于某些操作无法与 PyTorch 编译配合工作——或你仅仅出于性能原因不想让它们进入图中的场景。说到性能，接下来我们就来探讨如何利用 PyTorch 编译器日志改进已编译图与代码的性能。

### 性能提示与调试生成的代码

另一个极其有用、值得启用的日志选项是 TORCH*LOGS="perf_hints"。这些日志会向你展示错失的性能优化机会。例如，如果某个模式无法被融合——或某个 CUDA Graph 无法被使用——它会记录一条提示，如“\_PerfHint: CUDA Graph not used because input is mutated*”或“_PerfHint: fell back to eager for random op_”等。这些提示会指引你了解可能限制代码或模型性能的因素。

要做更深入的性能调试与调优，你很可能想看到 TorchInductor 生成的确切代码。有几种方式可以检视这些代码。首先，你可以设置 TORCH_LOGS="output_code" 来打印每个已编译图生成的代码。这会显示所生成核函数的原始源代码。如有需要，你甚至可以修改源代码并进一步优化。

你还可以通过设置 TORCH*COMPILE_DEBUG=1 来启用 TorchInductor 的调试模式。当你在启用调试模式下运行程序时，Inductor 会创建一个调试目录（例如 /tmp/torchinductor*<pid>/...），其中包含 FX 图（.fx）、诸如 outputcode.py、_fx_graph_runnable.py_ 等 Inductor 产物、IR 转储，以及生成的 Triton 源代码。

在阅读生成的 .triton 代码时，你可能会注意到 Triton 特有的构造——在高级场景下甚至会看到原始 PTX。如果你还检视调试产物中已编译的 PTX，你可能会在 tl.dot 被下降（lowering）为 Tensor Core 操作的地方看到 mma.sync 指令。这些日志、工具和产物对性能调优极其有用，因为它们让你精确看到编译器在做什么。理解这些内容有助于你确认编译器确实应用了核函数融合、warp 专门化（warp specialization）或双缓冲（double buffering）等优化。如果你发现了低效之处，就可以针对你的具体用例手动编写一个自定义 Triton 核函数。

> 如果你乐于分享，甚至可以把你的自定义核函数贡献回 PyTorch 与 Triton 生态，因为很可能别人也能从你的优化中受益。

### 调试数值正确性与精度

尽管非常罕见，torch.compile 仍有可能产生与即时执行模式在数值上不同的结果。如果你怀疑编译器存在 bug，在向社区反馈并创建 GitHub issue 之前，有几种策略可用于验证并收集数据。

首先，你可以使用 PyTorch 的 minifier 工具来创建可复现的脚本。PyTorch 提供了 TorchDynamo minifier 工具和 TorchInductor minifier 工具，它们会尝试把你的程序缩减到仍能复现该错误的最小版本。如有需要，为 PyTorch 团队创建一个小巧、可复现的脚本会非常有帮助。若真到了这一步，你就把这个文件附在你的 GitHub issue 上。

此外，你可以配置 TorchDynamo 在编译器栈的每一层调试数值精度。为帮助判断数值差异是在哪里引入的，你可以在编译期间设置以下环境变量，将即时执行模式与不同编译阶段进行比较，从而隔离问题究竟出在 TorchDynamo、AOT Autograd 还是 TorchInductor：

```
# Dump the outputs after each compilation stage
TORCHDYNAMO_REPRO_AFTER="aot"
TORCHDYNAMO_REPRO_LEVEL=4
```

这些设置会让 TorchDynamo 在每个阶段之后转储图——并以即时执行模式运行每个图以作对比。这有助于精确定位是哪个阶段引入了错误。

具体来说，设置 TORCHDYNAMO_REPRO_AFTER="aot" 会告诉 TorchDynamo 在 AOT Autograd 阶段之后转储 FX 图，并触发生成复现该错误脚本的逻辑。这与在最初的 Dynamo 捕获之后生成复现脚本形成对比。

使用 TORCHDYNAMO_REPRO_LEVEL=4，TorchDynamo 会以即时执行模式运行每个被转储的图，并将其输出与已编译版本进行比较。一旦检测到任何数值不匹配，它就会中止并保存一个最小复现脚本。

> PyTorch 编译器团队很乐意修复正确性 bug，所以如果你确实发现了真正的错误，请在 GitHub 上报告该问题。务必通过设置 TORCHDYNAMO_REPRO_AFTER="aot" 和 TORCHDYNAMO_REPRO_LEVEL=4 来附上已最小化的可复现产物集。

如果使用随机数（种子）或随机序列，你应确保它们被一致地生成。默认情况下，TorchInductor 可能不会产生与即时执行模式完全相同的随机种子或序列。原因之一是，被融合或被重排的核函数可能不会按即时执行模式那样预期的顺序生成数字。

如有需要，你可以设置 torch.\_inductor.config.fallback_random=True，强制 TorchInductor 完全像即时执行模式那样生成随机数。这会带来轻微的性能损失，但在使用 PyTorch 编译器时，为保证数值正确性（numerical correctness）它可能是必要的。

数值差异也可能源自浮点精度。例如，如果你使用 PyTorch 自动混合精度（automatic mixed precision，AMP）或 BF16，融合核函数中的运算顺序可能与即时执行未融合的序列相比引入细微的数值差异。

虽然这类差异很少影响收敛，但在某些情况下确实会。如果你怀疑存在与精度相关的不稳定，可以尝试禁用 torch.compile，并以完整 FP32 运行模型以隔离问题。你还可以使用 torch.set_float32_matmul_precision('highest') 来控制 TF32 的使用，以及在完整 FP32 矩阵乘法与最大数值精度之间权衡精度与性能。

同样重要的是要理解，使用混合精度（例如 FP16/BF16）可能带来细小差异。你可以通过设置 torch.use_deterministic_algorithms(True) 来强制确定性行为。这会使 PyTorch 在使用非确定性操作时抛出错误。虽然 torch.compile 在设计上确实减少了一些非确定性来源，但在调试期间启用此标志仍是良好实践。

不过要记住，并非所有操作都有确定性实现。例如，依赖 cuBLAS 的默认 torch.matmul() 操作就没有确定性实现。

具体来说，cuBLAS 实现依赖 split-K 之类的并行优化，这可能以不同顺序归约操作。其结果是浮点结果在多次运行之间无法按位复现。

因此，启用此设置可能导致你的代码失败，除非存在可用的回退替代方案。要对像 torch.matmul() 这类依赖 cuBLAS 的操作强制完全确定性，你需要调用 torch.use_deterministic_algorithms(True) 并将 CUBLAS_WORKSPACE_CONFIG 设置为固定大小，如下所示：

```
# Set this before starting the Python/PyTorch process
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # or :16:8
# Use this with the PyTorch process
torch.use_deterministic_algorithms(True)
```

这里，第一个值（例如 4096 或 16）选择 cuBLAS 工作区缓冲的字节大小，并向上舍入到某个内部分桶。第二个值（例如 8）选择保留多少个这样的缓冲。按文档所述设置为 :4096:8 或 :16:8 以强制使用确定性算法。

要在 torch.use_deterministic_algorithms(True) 下强制 cuBLAS 使用确定性算法，请按文档将 CUBLAS_WORKSPACE_CONFIG 设置为受支持的值，如 :4096:8 或 :16:8。如果你强制确定性却不设置此项，PyTorch 会在运行时对那些原本会选择非确定性 cuBLAS 算法的操作抛出错误。

> 请始终在你的实际硬件和模型配置上测试确定性，以确认可复现性。

此外，对于关键工作负载，你可以通过设置诸如 torch.\_inductor.config.triton.cudagraphs=False 之类的标志，临时禁用某些编译器优化，以便更好地隔离差异的成因。这会禁用对 TorchInductor 生成的 Triton 核函数的 CUDA Graph 捕获。

调试 PyTorch 编译器优化需要略有不同的思维方式，因为除了底层生成的代码之外，你还要透过日志和图可视化审视元层面（meta-level）的执行步骤。像 torch.\_dynamo.explain() 这样的工具能提供代码如何被转换为图、图中断（graph break）与子图的高层概览，而各种 TORCH_LOGS 选项则让你得以窥见编译器所做的决策——以及它生成的确切代码。

简而言之，借助这些组合起来的工具与调试机制，你可以迭代地消除图中断，并确保你的模型和代码被完整捕获与优化。这份付出是值得的，因为一个编译良好的模型能显著优于其即时执行的对应版本——尤其是对大型 LLM 架构而言，每一点性能提升都会累积起来。

## 解释与最小化图中断

在使用 torch.compile 时，诊断性能与正确性需要专门的工具。在本节中，我们将向你展示如何使用各种工具与最佳实践来调试并精确定位过多的图中断。这些工具包括 torch.\_dynamo.explain()、用于记录编译器决策的环境变量，以及调试所捕获的图与它们所生成核函数的最佳实践。

### 图中断与 TorchDynamo explain()

当 TorchDynamo 无法继续把一段连续的操作序列捕获进单个图时，就会发生图中断。发生这种情况时，它会对这部分代码回退到即时执行。

图中断是性能的大敌。每一次中断都意味着一个优化过的图被截断——并引入更多 Python 开销。如果你编译了一个模型却只看到不大的加速，很可能是频繁的图中断阻止了大型融合图的形成。理想情况下，我们希望中断越少越好——最好整个模型或整个训练步骤只有一个大图。

涉及集合通信（例如 all-gather、reduce-scatter 等）的复杂图往往需要图中断。图 14-5 展示了 PyTorch FSDP 策略中由集合通信导致的图中断。

![图 14-5. PyTorch FSDP 中由通信层引起的图中断（来源：https://oreil.ly/TJW42）](AI%20Systems%20Performance%20Engineering-ch14_images/figure-14-5.png)

PyTorch 提供了 torch.\_dynamo.explain() 来帮助分析和调试图中断。当你用你的模型和示例输入调用这个调试函数时，它会在 TorchDynamo 内运行该模型，并返回一份报告，说明生成了多少个图、中断发生在何处、以及它们为何发生，如下所示，随后附上详细的图中断分析与解释：

```
import torch._dynamo as dynamo
def toy_example(a, b):
    x = a / (torch.abs(a) + 1)
    print("woo")         # a print statement in the model
    if b.sum() < 0:      # dynamic control flow depending on data
        b = -b
    return x * b
explanation = dynamo.explain(toy_example)(torch.randn(10), torch.randn(10))
print(explanation)
Graph Count: 3
Graph Break Count: 2
Op Count: 5
Break Reasons:
  Break Reason 1:
    Reason: builtin: print [...ConstantVariable] False
    User Stack:
      <frame at toy_example: line 3, in toy_example>
  Break Reason 2:
    Reason: generic_jump TensorVariable()
    User Stack:
      <frame at toy_example: line 5, in toy_example>
Ops per Graph:
  ...
```

这里，这份解释显示 TorchDynamo 将代码分割为三个图段，中间有两次图中断。注意输出中的“User Stack”部分，它指向问题发生的具体代码行。这对于精确定位导致图中断的代码非常有用。

第一次中断由第 3 行附近的 print("woo") 引起。由于 print() 具有向 stdio 写入文本的“副作用”，它无法被捕获。因此，Dynamo 把图拆分为两个图：print() 之前与之后。

第二次图中断由第 5 行附近的数据相关控制流逻辑 if b.sum() < 0: 引起。由于这一特定场景中使用了数据相关的动态控制流逻辑，Dynamo 无法在单个图中处理它——这也是前一节提到的一项限制。

在你没有从 PyTorch 编译器获得预期性能时，对你的模型——配以有代表性的输入——运行 dynamo.explain() 是首先要做的事情之一。它能让你快速概览生成了多少个图——以及为何无法只生成一个大图。

一旦你理解了成因，就可以重构代码，逐个解决图中断。在前面的例子中，你可以移除 print()，或用诸如 if not torch.\_dynamo.is_compiling() 之类的保护把它包起来，以避免在追踪期间执行，如下所示：

```
import torch
def model(a, b):
    x = a / (torch.abs(a) + 1)
    # avoid during compiling/tracing
    if not torch._dynamo.is_compiling():
        print("do not print during tracing/compiling")
    if b.sum() < 0:
        b = -b
    return x * b
explanation = dynamo.explain(model)(torch.randn(10),
  torch.randn(10))
print(explanation)
```

如前所述，如果你的模型确实需要数据相关的分支，你可以把它们包在 torch.cond() 中。这会把“真”分支和“假”分支都捕获为图子例程，如下所示：

```
import torch
def model_cond(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Compute x as before
    x = a / (torch.abs(a) + 1)
    # Retain the compile-time check as a
    #   Python-level guard
    # Avoid side-effects during tracing/compilation
    if not torch._dynamo.is_compiling():
        print("do not print during tracing/compiling")
    # Handle the data-dependent sign flip on b
    b = torch.cond(
        b.sum() < 0,  # predicate (0-dim bool tensor)
        lambda b: -b, # true_fn: flip sign
        lambda b: b,  # false_fn: leave unchanged
        (b,)          # operands tuple
    )
    return x * b
# Generate and print the Dynamo explanation just like before
explanation = dynamo.explain(model_cond)(torch.randn(10), torch.randn(10))
print(explanation)
```

这里，谓词 b.sum() < 0 必须是一个 Python bool 或一个单元素的 torch.bool 张量。true_fn 与 false_fn 是接受相同操作数（此处即 (b,)）并返回形状和 dtype 相同的张量的可调用对象。

这段代码把 Dynamo 的编译期检查（dynamo.is_compiling()）保留为一个 Python if，因为它在运行时并非数据相关，而且我们希望避免在追踪期间产生副作用（例如 print）。

注意，torch.cond() 目前只接受张量谓词，要求两个分支具有相同的输入并返回形状和 dtype 完全一致的单个张量，且不允许原地修改或任意副作用。

相比之下，你可以像前面描述的那样，使用 torch.where() 的纯张量掩码方法。这不会带来上述任何限制，也能避免图中断，因此在你不需要 torch.cond() 的完整表达能力时，它是更简单、更可靠的选择。这段代码如下所示：

```
import torch
import torch._dynamo as dynamo
def model_where(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Compute x as before
    x = a / (torch.abs(a) + 1)
    # Preserve compile-time guard to avoid side-effects during tracing
    if not torch._dynamo.is_compiling():
        print("do not print during tracing/compiling")
    # Data-dependent branch expressed using torch.where
    b = torch.where(
        b.sum() < 0,  # predicate: a 0-dim bool tensor
        -b,           # true branch: flip sign
        b             # false branch: unchanged
    )
    return x * b
# Display the Dynamo explanation just as before
explanation = dynamo.explain(model_where)(torch.randn(10), torch.randn(10))
print(explanation)
```

这里，torch.where(condition, input, other) 返回一个张量，在 condition 为 True 处从 input 选取元素、在 condition 为 False 处从 other 选取元素。由于 b.sum() < 0 产生一个 0 维布尔张量，它可以被广播到 b 的所有元素上。这样就用一次向量化的符号翻转，替代了逐元素的 Python if。

> 使用 torch.where() 可以在已编译和已追踪的流水线中避免图中断。这让 TorchDynamo 能够内联优化这些操作。

使用 torch.compiler.set_stance("fail_on_recompile") 也很有帮助，它会在代码无法被干净地捕获为完整图时强制报错并拒绝运行。这在开发期间很有用，因为它让你能在编译期提前捕捉到图中断，而不是悄悄回退到较慢的 PyTorch 即时执行。

> torch.compiler.set_stance("fail_on_recompile") 也适合加入你的 CI 构建，以捕捉开发过程后期引入的任何图中断。在项目的整个生命周期中，拥有健壮且持续的性能回归测试极其重要。

### 最小化图的重新编译

除了图中断之外，你还应监控重新编译的次数。如果 TorchDynamo 的保护条件不断使输入张量形状等失效，它可能会多次编译同一个图。如果某个张量的形状在运行时发生变化，保护条件就会失效并触发一次重新编译。如果你看到的重新编译多于预期，就要排查是哪个保护条件（形状、dtype 等）导致的——并解决该问题。

通常，你会注意到重新编译正在发生，因为迭代会持续很慢——即便在最初的预热/编译迭代之后也是如此。幸运的是，你可以让 PyTorch 用 TORCH_LOGS="graph_breaks,recompiles,guards" 记录每次保护条件求值以及任何触发的重新编译。

如果你观察到频繁的保护条件失效，这往往意味着某个 Python 侧的常量——如随机数种子、时间戳或随循环变化的值——在每次迭代都在变化，从而不断使保护条件失效并触发重新编译。在这种情况下，你需要确保这些值要么被设为静态，要么用前面介绍的动态形状 API（例如 torch.\_dynamo.mark_dynamic）来处理。这将有助于避免不必要且过度的重新编译。

根据不同情况，有几种常见机制可用于最小化图的重新编译。首先，对于刚才提到的常量场景，你可以把该常量作为张量传入代码块，以防止编译器对其取值设置保护而反复失效。

其次，如前所述，你可以用 torch.\_dynamo.mark*dynamic(tensor, dim) 标记你已知会变化的动态维度，以先发制人地避免重新编译。另一个选项是使用 torch.compiler.set_stance("eager_on_recompile")，在 \_N* 次重新编译之后回退到即时执行模式，从而避免反复重新编译。这实际上为重新编译次数设置了上限。

另一个选项是使用 torch.\_dynamo.allow_in_graph，显式地把那部分图标记为安全。我们将在下一节更深入地探讨这一技巧。

### 用 allow_in_graph 把函数和代码块标记为安全

当 TorchDynamo 因某个函数或代码块使用了不受支持的操作（例如）而不知如何处理它时，你可以用 torch.\_dynamo.allow_in_graph——作为 Python 装饰器或上下文管理器——来修饰该函数或包裹该代码，告诉 Dynamo 它没有副作用。这样做之后，Dynamo 会以更宽松的分析与接受策略把该代码纳入追踪。allow_in_graph 会绕过 Dynamo 的一些安全检查。因此，请优先修复图中断的根本原因。

这是一项高级特性，应谨慎使用。你实际上是在承诺该函数是纯函数、对相同的输入张量总是返回相同的输出张量、仅依赖其张量输入、且没有副作用。如果使用不当，你可能会悄无声息地得到错误结果。然而，当使用得当时，如果某个特定函数或代码块本可安全追踪却导致了图中断，它可以成为拯救性能的关键。

一般来说，你应当谨慎地使用 allow_in_graph。它是供高级用户覆盖 Dynamo 保守本性的工具——但只有在你完全确定该函数没有可能影响代码正确性的副作用或隐藏状态时才应使用。

### 处理图中断的技巧

图中断限制了编译器执行大型优化的能力，例如把许多核函数融合成数量更少、更高效的核函数。这会迫使 PyTorch 对图的某些部分回退到较慢的即时执行。

理解什么会触发图中断——以及如何防止它们——至关重要。以下是图中断的一些常见成因，以及如何最小化它们的技巧：

_避免原地操作和意外的修改_ TorchDynamo 可以借助一种名为 _functionalization（函数化）_ 的机制处理某些修改，它会为追踪把原地操作转换为非原地操作。但某些原地操作仍可能导致图中断。如果你看到关于修改的中断原因，如“mutation on data”或“modifying a global”，请尝试重写该部分以避免原地操作。通常，如果“原地”正是问题所在，你只需把原地的 x.relu\_() 重写为非原地的 x = x.relu() 即可避免图中断。

_优先使用 PyTorch 的数据结构、集合类型和张量操作，而非等价的 Python 实现_ 在函数内部向一个 Python 张量列表追加元素会让 TorchDynamo 困惑，因为它不太能很好地追踪不断增长的列表。请尝试预分配张量，或使用像 torch.stack() 这样的张量操作，而不是动态构建 Python（非 PyTorch）列表。调用许多 Python 库——包括 I/O 操作、print、logging 以及 math.\* 函数——极有可能导致图中断。建议把这些从性能关键的代码路径中移除。

> 始终建议尽可能使用 Python 数据结构、集合类型和张量操作的 PyTorch 等价物。它们针对 PyTorch 编译、GPU 处理以及分布式数据传输做了大量优化，而这些在基于 PyTorch 的 AI 应用与模型中十分常见。

_尽可能避免数据相关的控制流_ 如果你有 if tensor.sum() > 0: 这类逻辑，TorchDynamo 无法轻易追踪它，因为该条件在编译期是未知的。它需要根据首次运行选择其中一个分支、对该条件设置保护、并对后续调用强制执行这个保护。由于这样做是不正确的，Dynamo 会创建一次图中断。

PyTorch 支持一个名为 torch.cond() 的高层操作，用于在图中捕获某些动态流。它可以封装 if/else 语句，使两个分支都被编译。不过，它要求条件是一个张量，并且通常最适合诸如依赖参数的开关这类情形，而非任意的 Python 逻辑。

除此之外，大多数数据相关的控制流仍会中断图。请继续在可能时优先使用张量操作（torch.where()、掩码等）。如果 torch.cond() 与重构都不可行，你可能只能接受图中断及其带来的性能影响。

_理解在 PyTorch DDP 下重叠与同步子图的性能特征_ PyTorch 的 DDP 通过在同步点（包括 all-reduce 桶）显式中断图来与 TorchDynamo 协同工作。你可能会在 explain 输出中看到与 allreduce 或 torch.distributed 操作相关的中断。这是预期之内的，因为 PyTorch 可能会单独编译每个梯度桶的归约，以便让它能把通信与反向计算相重叠。

如果你想保持计算-通信重叠，就无法在 DDP 通信边界处避免图中断。PyTorch 的编译器与 DDP 会有意在每个 all-reduce 桶处插入中断，从而让梯度同步发生在各子图之间。这让一个桶的通信可以与下一个桶的反向计算相重叠。

虽然这确实会阻止形成单一的整体图，但它保持了性能。TorchDynamo + DDP 的运行性能与即时执行模式的 DDP 相近。在大规模下，它甚至能优于即时执行的 DDP。所以，尽管你无法消除这些通信图中断，它们对于实现正确且高效、并带有恰当重叠的分布式训练是必要的。

_用 PyTorch FSDP 包裹图子模块_ PyTorch 通过使用 use_original_params=True 支持在已编译模式下运行 FSDP。一条最佳实践是把子模块（如每个 transformer 块）各自包裹进它们自己的 FSDP 子模块。Dynamo 随后会在每个 FSDP 子模块边界处创建显式的图中断。这让每个分片的通信可以与计算相重叠，类似于前面为 DDP 描述的分桶策略。
