TorchInductor attempts to generalize shapes after the first recompilation instead of repeatedly specializing on each new shape. For instance, it will emit conditional code inside the generated kernel using an if statement so that one kernel works for a range of sequence_lengths without erroring out. This reduces the need for separate compilation for every single size.

> Certain operations with data-dependent output ranks—or extremely complex indexing—may still trigger shape specialization. In these cases, the compiler will insert more guards—and if those are frequently violated, you might see frequent recompilations with mark_dynamic() or set_stance().

For context, a simple but inefficient way to handle variable-length sequences without supporting dynamic shapes is to pad all input sequences to the max length in the batch. This way, you can use one static computation for all inputs. While padding simplifies the implementation, it is inefficient when input lengths vary widely since a lot of compute is wasted on the meaningless padding tokens.

Padding can hurt GPU utilization if the maximum length is much bigger than the average length of all the inputs. With dynamic shape-compilation, however, we can let the compiler generate code that only iterates up to the actual sequence length of each input. Dynamic shapes let you avoid excessive padding for variable lengths.

Let’s look at a typical text-based generative AI scenario in which sequence lengths continue to grow as the generation progresses. Compiling with dynamic shapes can consistently outperform eager execution—even as sequence length increases.

In contrast, if one were to pad everything to a power-of-two length to use static shapes, it would introduce a lot of wasted computation and increase compile time due to larger tensor sizes. In other words, using dynamic shapes provides better compile-time performance and runtime performance and easier usage since you don’t have to manually pad the inputs.

> It’s recommended to bucket inputs by size in order to limit the number of distinct shapes. This will enable dynamic shapes for the remaining variability. This hybrid approach avoids excessive recompilations while still reducing padding waste.

With dynamic shapes, you can compile once and use the same compiled model on inputs of different shapes. If the variations are within the supported range, one compiled model can handle multiple configurations.

Internally, TorchInductor uses the SymPy library to represent dynamic dimensions symbolically. It will propagate these symbols through the IR so that an expression like z.size(0) = x.size(0) + y.size(0) can be handled symbolically. Inductor will reduce conditions to guard expressions.

If a guard fails because the dimension fell outside an expected range—or a data-dependent condition changed, Inductor will trigger a recompile. In essence, TorchInductor attempts to compile a general kernel for a range of sizes instead of a single fixed size.

Dynamic shape has significantly improved in recent releases. However, certain operations may force shape specialization if the compiler can’t handle them symbolically. In this case, the compiler might insert more guards, which, if violated often, could lead to frequent recompilations and negate the benefits of compiling.

Data-dependent control flow still triggers specialization. Use dynamic shapes for varying sequence lengths but not for truly data-dependent branches.

It’s worth noting that as of this writing, CUDA Graph replay requires static shapes (and fixed memory addresses). And only limited parameter updates are supported on instantiated graphs. Memory addresses and kernel-launch topology must remain compatible with capture. As such, enabling dynamic shapes will typically disable graph capture for those regions. This prevents the compiler from gaining the performance benefits of CUDA Graphs, including reduced kernel-launch overhead.

> If you specify the reduce-overhead compiler mode but also set dynamic=True, the CUDA Graph optimization from reduce-overhead won’t apply since you are specifying that the shapes can vary. Enabling dynamic shapes will change guards and memory planning, which will disable graph capture. In practice, use mode="reduce-overhead" only with stable shapes to get CUDA Graphs. For variable sequence lengths, prefer mode="default" or mode="max-autotune-no-cudagraphs" and bucket/pad within ±10–20% to limit recompiles.

It’s recommended to profile your system to see if dynamic shapes are worth using for your use case. In certain cases, it might be better to pad to a fixed size, use static shapes with CUDA Graphs, and achieve higher throughput by not having to recompile for each unique length. In other cases, dynamic shapes will be better.

You should profile different approaches to find what works best for you. When you do this, be sure to monitor memory usage. Code that supports dynamic shapes will incur a slightly higher memory footprint due to the additional guards and generalized code needed for the maximum range.

> A rule of thumb is that if your sequence lengths vary by only 10%–20%, you will likely benefit from fixed-length padding.

In short, dynamic shape support means you don’t have to disable torch.compile for variable-length inputs common in LLM models. By supporting dynamic shapes, the PyTorch compiler can perform kernel fusion and other optimizations across different input sizes.

### Disabling the PyTorch Compiler and Reverting Back to Eager Mode

If you want to completely disable torch.compile without changing your code—useful for A/B testing performance and isolating issues—you can use the @torch.compiler.disable decorator to disable compilation for that function. For region-scoped control, use torch.compiler.set_stance() as a context manager. This will force the code to run in eager mode. For example, you might want to disable compilation for complex data loading or one-time initialization logic to keep the compiled graph focused on computations. This is also useful around code that does not work well with tracing, as we’ll cover in a bit.

Or, you can simply change to use the eager backend as follows: torch.compile(model, backend="eager"). This will revert your code to run in eager mode. This lets you easily debug and compare correctness/performance results between compiled and eager modes.

torch.compiler.disable() and torch.compiler.set_stance() are a valuable escape hatches when certain operations don’t work with PyTorch compile—or you simply don’t want them in the graph for performance reasons. Speaking of performance, let’s explore ways to improve the performance of our compiled graphs and code using the PyTorch compiler logs.

### Performance Hints and Debugging Generated Code

Another extremely useful logging option to enable is TORCH_LOGS="perf_hints". These logs will show you missed performance-optimization opportunities. For example, if a certain pattern could not be fused—or if a CUDA Graph could not be used—it will log a hint like “*PerfHint: CUDA Graph not used because input is mutated*” or “*PerfHint: fell back to eager for random op*,” etc. These hints guide you on what might be limiting the performance of your code or model.

For deeper performance debugging and tuning, you likely want to see the exact code that TorchInductor generates. There are a couple of ways to inspect the code. First, you can set TORCH_LOGS="output_code" to print the generated code for each compiled graph. This will show the raw source code for the generated kernels. You can even modify the source code and further optimize, if needed.

You can also enable TorchInductor’s debug mode by setting TORCH_COMPILE_DEBUG=1. When you run your program with debug mode enabled, Inductor will create a debug directory (e.g., /tmp/torchinductor_<pid>/...) that contains the FX Graph (.fx), Inductor artifacts such as outputcode.py, *fx_graph_runnable.py*, IR dumps, and generated Triton sources.

When reading the generated .triton code, you may notice Triton-specific constructs —or even raw PTX in advanced cases. If you also inspect the compiled PTX in the debug artifacts, you may see mma.sync instructions where tl.dot is lowered to Tensor Core operations. These logs, tools, and artifacts are incredibly useful for performance tuning because they let you see exactly what the compiler is doing. Understanding these can help you verify that the compiler is applying optimizations like kernel fusion, warp specialization, or double buffering. If you spot an inefficiency, you can manually create a custom Triton kernel for your specific use case.

> If you’re feeling benevolent, you can even contribute your custom kernel back to the PyTorch and Triton ecosystems since it’s likely that somebody else can benefit from your optimization.

### Debugging Numerical Correctness and Accuracy

While very rare, it’s possible that torch.compile produces a result that is numerically different compared to eager mode. If you suspect a bug in the compiler, there are a few strategies to verify and collect data before notifying the community and creating a GitHub issue.

First, you can use PyTorch’s minifier tools to create reproducible scripts. PyTorch has a TorchDynamo minifier tool and TorchInductor minifier tool, which will try to reduce your program to the smallest version that still reproduces the error. It’s very helpful to create a small, reproducible script for the PyTorch team to use if needed. You would attach this file to your GitHub issue if it gets to this point.

Additionally, you can configure TorchDynamo to debug numerical accuracy at each layer of the compiler stack. To help determine where a numerical discrepancy is introduced, you can set the following environment variables during compilation to compare eager mode to the different compiler stages and isolate if the issue is in TorchDynamo, AOT Autograd, or TorchInductor:

```
# Dump the outputs after each compilation stage
TORCHDYNAMO_REPRO_AFTER="aot"
TORCHDYNAMO_REPRO_LEVEL=4
```

These settings will cause TorchDynamo to dump the graph after each stage—and run each graph in eager mode for comparison. This can help pinpoint which stage introduced the error.

Specifically, setting TORCHDYNAMO_REPRO_AFTER="aot" tells TorchDynamo to dump the FX Graph and trigger the logic to generate a script to reproduce the error after the AOT Autograd stage. This is in contrast to generating the reproduction script after the initial Dynamo capture.

Using TORCHDYNAMO_REPRO_LEVEL=4, TorchDynamo will run each dumped graph in eager mode and compare its outputs to the compiled version. This halts and saves a minimal reproduction script if any numeric mismatch is detected.

> The PyTorch compiler team loves fixing correctness bugs, so if you do find a true error, report the issue on GitHub. Make sure to include the minified reproducible set of artifacts by setting TORCHDYNAMO_REPRO_AFTER="aot" and TORCHDYNAMO_REPRO_LEVEL=4.

If using random numbers (seeds) or sequences, you should make sure they are being generated consistently. By default, TorchInductor might not produce the exact same random seed or sequence as with eager mode. One reason is that fused or reordered kernels may not generate numbers in the same, expected order as eager mode.

If needed, you can set torch._inductor.config.fallback_random=True to force TorchInductor to generate random numbers exactly like it would with eager mode. This will incur a slight performance hit, but it may be required for numerical correctness when using the PyTorch compiler.

Numerical differences can also stem from floating-point precision. For example, if you use PyTorch automatic mixed precision (AMP) or BF16, the order of operations in a fused kernel might introduce slight numerical differences versus eager’s unfused sequence.

While such differences rarely affect convergence, they can in some cases. If you suspect precision-related instability, try disabling torch.compile and run the model in full FP32 to isolate the issue. You can also use torch.set_float32_matmul_precision('highest') to control TF32 usage and the accuracy-performance trade-off for full FP32 matmuls and maximum numerical accuracy.

It’s also important to understand that small discrepancies may arise from using mixed precision (e.g., FP16/BF16). You can enforce deterministic behavior by setting torch.use_deterministic_algorithms(True). This causes PyTorch to throw an error if a nondeterministic operation is used. While torch.compile does reduce some sources of nondeterminism by design, it’s still good practice to enable this flag during debugging.

Keep in mind, however, that not all operations have deterministic implementations. For example, the default torch.matmul() operation that relies on cuBLAS does not have a deterministic implementation.

Specifically, the cuBLAS implementation relies on parallel optimizations like split-K, which can reduce operations in varying orders. This results in floating-point results that aren’t bitwise reproducible across runs.

As such, enabling this setting may cause your code to fail unless there is a fallback alternative available. To enforce full determinism for cuBLAS-dependent operations like torch.matmul(), you need to call torch.use_deterministic_algorithms(True) and set the CUBLAS_WORKSPACE_CONFIG to a fixed size, as shown here:

```
# Set this before starting the Python/PyTorch process
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # or :16:8
# Use this with the PyTorch process
torch.use_deterministic_algorithms(True)
```

Here, the first value (e.g., 4096 or 16) selects the size of the cuBLAS workspace buffer in bytes rounded to an internal bucket. The second value (e.g., 8) selects how many such buffers are reserved. Set either :4096:8 or :16:8 as documented to enforce deterministic algorithms.

To force cuBLAS to use deterministic algorithms under torch.use_deterministic_algorithms(True), set CUBLAS_WORKSPACE_CONFIG to a supported value like :4096:8 or :16:8, as documented. If you enforce determinism without setting this, PyTorch will raise at runtime for operations that would otherwise select nondeterministic cuBLAS algorithms.

> Always test determinism on your actual hardware and model configuration to confirm reproducibility.

Also, for critical workloads, you might temporarily disable certain compiler optimizations by setting flags like torch._inductor.config.triton.cudagraphs=False to better isolate the cause of a discrepancy. This disables CUDA Graph capture for TorchInductor-generated Triton kernels.

Debugging PyTorch compiler optimizations requires a slightly different mindset since you’re looking at the meta-level execution steps through logs and graph visualizations—in addition to the low-level generated code. Tools like torch._dynamo.explain() give a high-level overview of how your code is converted into graphs, graph breaks, and subgraphs, while the various TORCH_LOGS options let you peek into the decisions that the compiler makes—as well as the exact code that it generates.

In short, with these combined tools and debugging mechanisms, you can iteratively eliminate graph breaks and make sure your model and code are fully captured and optimized. The payoff is worth it, as a well-compiled model can significantly outperform its eager-execution counterpart—especially for large LLM architectures in which every bit of performance improvement will add up.

## Explaining and Minimizing Graph Breaks

When using torch.compile, diagnosing performance and correctness requires specialized tools. In this section, we’ll show you how to use various tools and best practices to debug and pinpoint excessive graph breaks. These include torch._dynamo.explain(), environment variables to log compiler decisions, and best practices for debugging both the captured graphs and the kernels that they generate.

### Graph Breaks and TorchDynamo explain()

A graph break occurs when TorchDynamo cannot continue capturing a continuous sequence of operations into a single graph. When this happens, it falls back to eager execution for this part of the code.

Graph breaks are the enemy of performance. Each break means an optimized graph is cut short—and more Python overhead is introduced. If you compile a model and see only modest speedups, it may be caused by frequent graph breaks that are preventing large, fused graphs. Ideally, we want as few breaks as possible—ideally one large graph for the whole model or whole training step.

Complex graphs that involve collective communications (e.g., all-gather, reduce-scatter, etc.) often require graph breaks. Figure 14-5 shows the graph breaks in PyTorch’s FSDP strategy due to collective communication.

![Figure 14-5. Graph breaks in PyTorch FSDP caused by communication layers (source: https://oreil.ly/TJW42)](AI%20Systems%20Performance%20Engineering-ch14_images/figure-14-5.png)

PyTorch provides torch._dynamo.explain() to help analyze and debug graph breaks. When invoking this debugging function with your model and example inputs, it will run the model within TorchDynamo and return a report of how many graphs were generated, where the breaks occurred, and why they happened, as shown here, followed by the detailed graph-break analysis and explanation:

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

Here, the explanation shows that TorchDynamo splits the code into three graph segments across two graph breaks. Note the “User Stack” portions of the output that point to the specific line of code where the issue happens. This is very useful for pinpointing the code causing the graph break.

The first break is caused by the print("woo") near line 3. Because print() has a “side effect” of writing text to stdio, it isn’t capturable. As such, Dynamo breaks the graph into two graphs: before and after the print().

The second graph break is caused by the dynamic control flow logic if b.sum() < 0: near line 5, which Dynamo couldn’t handle in a single graph because of the data-dependent dynamic control flow logic used in this specific scenario—and mentioned as a limitation in a previous section.

Using dynamo.explain() on your model—with representative inputs—is one of the first things to do if you’re not getting the performance you expect from the PyTorch compiler. It gives you a quick overview of how many graphs were made—and why it couldn’t make just one large graph.

Once you understand the causes, you can refactor the code to address the graph breaks one by one. In the preceding example, you can remove the print() or wrap it in a guard such as if not torch._dynamo.is_compiling() to avoid executing during tracing, as shown here:

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

As mentioned earlier, if your model truly needs data-dependent branches, you can wrap them in torch.cond(). This will capture both the “true” and “false” branches as graph subroutines, as shown here:

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

Here, the predicate b.sum() < 0 must be either a Python bool or a one-element torch.bool tensor. The true_fn and false_fn are callables taking the same operands (here, just (b,)) and returning tensors of the same shape and dtype.

This code keeps the Dynamo compile-time check (dynamo.is_compiling()) as a Python if since it’s not data-dependent at runtime and we want to avoid side-effects (e.g., print) during tracing.

Note that torch.cond() currently only accepts a tensor predicate, requires both branches to have the same inputs and return a single tensor of identical shape and dtype, and does not allow in-place mutations or arbitrary side-effects.

In contrast, you can use a pure-tensor masking approach with torch.where(), as described earlier. This will impose no such restrictions and avoids graph breaks, making it the simpler, more reliable choice when you don’t need the full expressivity of torch.cond(). This code is shown here:

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

Here, torch.where(condition, input, other) returns a tensor selecting elements from input where condition is True and from other where condition is False. Because b.sum() < 0 produces a 0-dimensional Boolean tensor, it can be broadcast across all elements of b. This allows a single, vectorized sign flip instead of an elementwise Python if.

> Using torch.where() can avoid graph breaks in compiled and traced pipelines. This allows TorchDynamo to optimize operations inline.

It’s also helpful to use torch.compiler.set_stance("fail_on_recompile") to force an error and refuse to run if the code is not cleanly capturable into a full graph. This is useful during development since it lets you catch graph breaks upfront at compile time instead of silently falling back to slower PyTorch eager execution.

> torch.compiler.set_stance("fail_on_recompile") is also useful to add in your CI build to catch any graph breaks introduced later in the development process. Having robust and continuous performance-regression tests is extremely important throughout the life of a project.

### Minimize Graph Recompilations

Besides graph breaks, you should also monitor the number of recompilations. TorchDynamo might be compiling the graph many times if its guards keep invalidating input tensor shapes, etc. If a tensor’s shape changes at runtime, the guard fails and triggers a recompile. If you see more recompiles than expected, investigate which guard (shape, dtype, etc.) is causing it—and address the issue.

Typically, you’ll notice recompilations happening because iterations will continue to be slow—even after the initial warm-up/compile iterations. Fortunately, you can have PyTorch log each guard evaluation and any trigger recompilation using TORCH_LOGS="graph_breaks,recompiles,guards".

If you observe frequent guard failures, it often means a Python-side constant, such as a random number seed, timestamp, or loop-varying value, is changing on every iteration—and continuously invalidating the guard and triggering a recompile. In this case, you’ll need to ensure those values are either made static or handled with the dynamic-shape APIs presented earlier (e.g., torch._dynamo.mark_dynamic). This will help avoid needless and excessive recompiles.

There are a few common mechanisms to minimize graph recompilations depending on the situation. First, for the constant scenario just mentioned, you can pass the constant into the code block as a tensor to prevent the compiler from guarding on the value and repeatedly failing.

Next, as mentioned earlier, you can mark dynamic dimensions that you know will change using torch._dynamo.mark_dynamic(tensor, dim) to preempt a recompile. Another option is to use torch.compiler.set_stance("eager_on_recompile") to avoid repeated recompiles by falling back to eager mode after *N* number of recompiles. This effectively caps the limit of recompilations.

Another option is to explicitly mark that part of the graph as safe using torch._dynamo.allow_in_graph. Let’s dive into this technique a bit more in the next section.

### Mark Functions and Code Blocks as Safe with allow_in_graph

When TorchDynamo doesn’t know how to handle a function or code block because it’s using unsupported operations, for example, you can decorate the function or wrap the code with torch._dynamo.allow_in_graph—as either a Python decorator or context manager—to tell Dynamo that it has no side effects. When you do this, Dynamo will then include the code in the trace using a more lenient analysis and acceptance policy. allow_in_graph bypasses some Dynamo safety checks. As such, prefer fixing the root cause of graph breaks first.

This is an advanced feature and should be used carefully. You are essentially promising that the function is pure, always returns the same output tensor for the same input tensor, depends only on its tensor inputs, and has no side effects. If used incorrectly, you may silently get the wrong results. However, when used correctly, it can be a performance lifesaver if a specific function or code block is causing a graph break even though it’s safe to be traced.

In general, you should use allow_in_graph sparingly. It’s a tool for power users to override Dynamo’s conservative nature—but only when you’re absolutely sure that the function does not have side effects or hidden state that could impact the code’s correctness.

### Tips for Handling Graph Breaks

Graph breaks limit the compiler’s ability to perform large optimizations such as fusing many kernels into a smaller number of efficient kernels. This forces PyTorch to fall back to slower eager execution for certain parts of the graph.

It’s critical to understand what triggers graph breaks—and how to prevent them. Here are some common causes of graph breaks and tips on how to minimize them:

*Avoid in-place operations and unexpected mutations* TorchDynamo can handle some mutations using a mechanism called *functionalization*, which converts in-place operations to out-of-place for tracing. But certain in-place operations might still cause a graph break. If you see a break reason about mutation, such as “mutation on data” or “modifying a global,” try to rewrite that part to avoid in-place operations. Often, you can simply rewrite in-place x.relu_() to out-of-place x = x.relu() to avoid a graph break if being in-place was causing the issue.

*Prefer PyTorch data structures, collections, and tensor operations over equivalent* *Python implementations* Appending to a Python list of tensors inside a function will confuse TorchDynamo since it doesn’t trace growing lists very well. Try to preallocate tensors or use tensor operations like torch.stack() instead of building Python (non-PyTorch) lists dynamically. Calls to many Python libraries, including I/O operations, print, logging, and math.* functions will most likely cause a graph break. It’s recommended to remove these from the performance-critical code paths.

> It’s always recommended to use the PyTorch equivalent of Python data structures, collections, and tensor operations whenever possible. These are heavily optimized for PyTorch compilation, GPU processing, and distributed data transfers, which are common in PyTorch-based AI applications and models.

*Avoid data-dependent control flow, if possible* If you have if tensor.sum() > 0: style logic, TorchDynamo cannot easily trace through this because the condition is unknown at compile time. It would need to choose one branch or the other based on the first run, guard on that condition, and enforce this guard for subsequent invocations. Since this is incorrect, Dynamo will create a graph break.

PyTorch supports a high-level operation called torch.cond() to capture certain dynamic flows in graphs. This can encapsulate if/else statements such that both branches are compiled. However, it requires the condition to be a tensor and typically works best for things like parameter-dependent switches rather than arbitrary Python logic.

Apart from this, most data-dependent control flow still breaks graphs. Continue to prefer tensor operations (torch.where(), masks, etc.) when possible. If neither torch.cond() nor refactoring is feasible, you may have to accept the graph break and its performance impact.

*Understand performance characteristics of overlapping and synchronizing subgraphs* *with PyTorch DDP* PyTorch’s DDP works with TorchDynamo by explicitly breaking graphs at synchronization points, including the all-reduce buckets. You might see breaks in the explain output related to allreduce or torch.distributed ops. This is expected, as PyTorch may compile each gradient bucket’s reduction separately so that it can remove overlap communication with backward computation.

You can’t avoid graph breaks at DDP communication boundaries if you want to preserve compute-communication overlap. PyTorch’s compiler and DDP intentionally insert breaks at each all-reduce bucket so that gradient synchronization happens between subgraphs. This lets one bucket’s communication overlap with the backward computation of the next bucket.

While this does prevent a single monolithic graph, it preserves performance. TorchDynamo + DDP runs with similar performance to eager-mode DDP. And it can even outperform eager DDP at scale. So, although you can’t eliminate these communication graph breaks, they are necessary to achieve correct and efficient distributed training with the proper overlap.

*Wrap graph submodules with PyTorch FSDP* PyTorch supports FSDP in compiled mode by using use_original_params=True. A best practice is to wrap submodules, like each transformer block, into their own FSDP submodule. Dynamo will then create explicit graph breaks at each FSDP submodule boundary. This allows each shard’s communication to overlap with computation, similar to the bucketization strategy described for DDP.