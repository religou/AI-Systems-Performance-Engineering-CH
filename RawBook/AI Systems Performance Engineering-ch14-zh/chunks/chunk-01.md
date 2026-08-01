# Chapter 14. PyTorch Compiler, OpenAI Triton, and XLA Backends

In Chapter 13, we discussed multiple ways to optimize and tune PyTorch-based training and inference workloads. We touched on the PyTorch compiler and how it automates kernel fusion and other kernel-level techniques to improve performance with very little changes to your code.

In this chapter, we dive deeper into the dynamic PyTorch compilation stack, including components like TorchDynamo, Ahead-of-Time Autograd (AOT Autograd), and PrimTorch Intermediate Representation (IR) (aka *Prims* or *Prims IR*)—as well as compiler backends like TorchInductor, Accelerated Linear Algebra (XLA), and OpenAI’s Triton ecosystem. The PyTorch compiler stack is shown in Figure 14-1.

We also cover tools for debugging the compilation pipeline as well as libraries for scaling PyTorch across multi-GPU and multinode clusters. We will then explore how torch.compile works under the hood and how to handle dynamic shapes and variable sequence lengths efficiently.

We will also examine the PyTorch compiler’s integration with the OpenAI Triton ecosystem. Our goal is to accelerate and scale our PyTorch models and applications without sacrificing the flexible, eager-execution development experience of PyTorch.

![Figure 14-1. Overview of PyTorch compiler stack](AI%20Systems%20Performance%20Engineering-ch14_images/figure-14-1.png)

## PyTorch Compiler Deep Dive

As described in Chapter 13, PyTorch’s torch.compile will compile your PyTorch code (and models) to produce significant speedups. In most cases, you can do this in just a single line of code, as shown next. We’ll talk about the different options as we go along:

```
compiled_model = torch.compile(model,
  mode="max-autotune",
#  ...
)
```

This section breaks down the PyTorch compilation pipeline steps, including TorchDynamo’s graph capture, AOT Autograd’s combined forward/backward graph optimization, PrimTorch IR, and TorchInductor’s code generation. This pipeline is responsible for producing optimized kernels for the target GPU hardware and is shown in Figure 14-2.

![Figure 14-2. PyTorch compiler pipeline (source: https://oreil.ly/55JDn)](AI%20Systems%20Performance%20Engineering-ch14_images/figure-14-2.png)

### TorchDynamo for Bytecode Capture and Graph Extraction

TorchDynamo, or just Dynamo, is the first stage of torch.compile. It hooks into Python’s frame-evaluation mechanism to intercept model execution at the bytecode level.

Dynamo hooks into CPython’s frame evaluation to identify tensor-producing bytecode regions and constructs an execution graph for those regions. It then executes the compiled graph using the chosen backend. Unsupported code is left to run eagerly.

This interception and rewriting mechanism is what lets TorchDynamo capture sequences of PyTorch operations into a graph representation that can be optimized by the next steps, AOT Autograd and PrimTorch IR, covered in the next sections.

TorchDynamo leverages the CPython Frame Evaluation API (PEP 523) to capture the operations safely—and with minimal overhead. Normally, the Python interpreter executes each operation one by one. With Dynamo enabled, however, the interpreter redirects its execution to Dynamo, which aggregates the tensor operations into a graph before executing them. This enables whole-graph optimizations like kernel fusion, which reduces per-operation Python and host-side overhead.

Instead of launching a new GPU kernel for each operation and paying per-operation Python overhead, the compiled graph can fuse many operations into one or a few kernels. This reduces dispatch overhead and improves memory access patterns.

In Chapter 13, we saw how Python overhead and many small operations can bottleneck models. We also saw that TorchDynamo addresses this issue by batching small operations into larger units when possible—assuming the chain of operations is graph-friendly and doesn’t cause too many graph breaks.

When TorchDynamo is enabled by torch.compile, it inspects each Python opcode. Whenever it encounters a PyTorch tensor operation such as arithmetic or a neural network layer, it doesn’t execute it immediately. Instead, Dynamo appends the operation as a node in an FX graph and defers execution to the compiled backend for captured regions. TorchDynamo continues this process until it hits some code it can’t handle. At this point, a graph break occurs (more on this in a bit), and the unsupported operations run in regular, noncompiled eager mode.

TorchDynamo tries to compile as large a portion of your program as possible into a single graph, but it will fall back to Python eager execution when it can’t proceed due to unsupported constructs such as complex control flow or non-PyTorch library calls. It does this to ensure correctness.

After a block of unsupported operations has passed, Dynamo will resume capturing subsequent operations into a new graph after that point. This mix of compiled and noncompiled execution gives you the best of both worlds: you keep PyTorch’s flexibility where needed but compile everything else for speed.

> You want to avoid graph breaks whenever possible, as they interrupt whole-graph optimizations and limit the performance benefits of the compiler.

You can use torch.compiler.set_stance("fail_on_recompile") to force Dynamo to raise an error to catch unsafe recompilations. It will log the reason for recompile and help you debug the graph break. This gives you full visibility into why your graph is splitting. The code is shown here:

```
// fail on graph breaks, recompiles
torch.compiler.set_stance("fail_on_recompile")
compiled_model = torch.compile(model,
  mode="max-autotune",
  ...
)
```

This way, you can refactor the code paths that are causing the break—or mark them as expected graph boundaries with torch._dynamo.allow_in_graph(), as you’ll see in a bit. Once your graph is clean, you can switch back to torch.compiler.set_stance("eager_on_recompile"). Remember, setting this back will cause TorchDynamo to silently fall back to eager mode if a subsequent graph break occurs.

The output of TorchDynamo’s capture is called an *FX Graph* of your code. FX is an intermediate representation (IR) in which each node is a call to a PyTorch aten operator or built-in Python function. For example, consider a simple Python function, as shown here:

```
def f(x, y):
    z = x.sin() + y
    return z.sum()
```

Here, TorchDynamo will produce an FX Graph roughly equivalent to the following pseudocode:

```
graph():  # pseudo-code for FX IR
    %x : Tensor = Placeholder[target=x]
    %y : Tensor = Placeholder[target=y]
    %sin : Tensor = torch.sin(%x)
    %add : Tensor = torch.add(%sin, %y)
    %sum : Tensor = torch.sum(%add)
    return %sum
```

The FX Graph nodes correspond to Placeholder inputs and calls to primitive ATen operations like aten::sin, aten::add, aten::sum, clearly representing the computation graph structure. Once constructed, this FX Graph is handed off to AOT Autograd for combined forward-pass and backward-pass tracing.

AOT Autograd’s generated forward and backward combined trace is then sent to a backend compiler such as TorchInductor or XLA to perform kernel fusion and generate optimized device code. TorchDynamo itself remains framework-agnostic and focuses on accurate, low-overhead graph capture. TorchDynamo delegates all heavy optimizations to downstream compiler stages.

TorchDynamo inserts guards on Python values—such as tensor shapes, dtypes, and even global variables—that affect the graph trace. These guards ensure that if something changes that was assumed to be constant (e.g., tensor shape or dtype), the compiled graph is invalidated and recompiled if needed. This allows robust handling of dynamic shapes and model modifications—at the cost of additional recompilations when assumptions are violated.

> It’s recommended to monitor the number of graph recompilations when using the PyTorch compiler.

Using the default dynamic=None, the compiler will specialize using the observed shapes. When it encounters dynamic shapes, it will recompile a more dynamic kernel. It may recompile again if it later detects additional sources of dynamism.

In practice, this means you’ll see one extra compilation on the first varying-shape input—and only one extra compile. If you already know which input dimensions will vary, you can use torch._dynamo.mark_dynamic(tensor, dim) before running your model. This will preempt the initial compile.

You can further tune the compiler’s “stance” with torch.compiler.set_stance() to change how and when TorchDynamo falls back to eager or recompiles. A stance governs the compiler’s tolerance or strictness in the face of errors or fallbacks. This gives you more control of the trade-off between developer feedback and uninterrupted execution. The following are the stances as of this writing:

default The compiler tries to compile what it can, silently falling back to eager execution when unsupported code is encountered. This is the standard, default setting, which causes normal compilation behavior and fallbacks.

fail_on_recompile If a graph break or unsupported operation is encountered, the compiler raises an error. This is useful during development when you want to catch unexpected breaks or uncompiled paths.

eager_on_recompile When a recompile is necessary, run the code in eager mode. If compiled code is already cached (and valid for the input), it should still be used.

force_eager Use eager mode and ignore all torch.compile directives.

By capturing whole sequences of operations, TorchDynamo can perform whole-graph optimizations like kernel fusion to reduce launch overhead and minimize memory movement. It also eliminates Python-layer overhead for those fused operations. Figure 14-3 compares eager versus compiled modes, including the compiler cache.

Instead of executing each small operation and kernel launch separately, the compiler can fuse many operations into a single kernel. This reduces CPU-GPU synchronization points and improves memory locality. Otherwise, the GPU would be bottlenecked with lots of fine-grained operations, heavy synchronization, and excessive global memory accesses.

![Figure 14-3. PyTorch eager versus compiled modes](AI%20Systems%20Performance%20Engineering-ch14_images/figure-14-3.png)

TorchDynamo continues capturing until a graph break is needed. A graph break can be triggered by unsupported Python constructs (like certain control flows, e.g., an if statement using a Python bool instead of tensor operations) or by unsupported operations. When a graph break occurs, the current graph segment ends, and Dynamo falls back to eager mode for the code that can’t be captured. After that, TorchDynamo will start a new graph trace once it returns to traceable code.

You can successfully implement compiler-graph-friendly conditionals in PyTorch by expressing the logic as pure PyTorch tensor operations, including torch.where() with a mask. Here is an example that replaces a non-graph-friendly Python if statement with a pure-tensor operation masking approach:

```
# Compute a boolean mask from a data-dependent tensor condition
# mask shape (batch_size,1) broadcastable to x’s
mask = x.sum(dim=1, keepdim=True) > 0
# Use torch.where to select element-wise between two tensor expressions
# picks f(x) where mask is True, otherwise g(x)
out = torch.where(mask, f(x), g(x))
```

This avoids a Python if x.sum() > 0: statement, which causes a graph break. It does this by staying entirely within PyTorch’s graph-friendly tensor operations. In this case, TorchDynamo captures the whole sequence, including the mask and torch.where() without breaking the graph.

With each new release, PyTorch expands which operations can be captured without causing a graph break. For instance, high-level conditional primitives like torch.cond can capture certain if/else logic in graphs. Specifically, both branches are traced and compiled. torch.cond requires a boolean scalar predicate. Both branches must return the same structure and dtypes. And shapes must be consistent at runtime. Data-dependent branches will often lead to graph breaks.

> It’s recommended to minimize graph breaks by refactoring your code using tensor operations like torch.where() to maximize the continuous regions that TorchDynamo can capture.

### AOT Autograd Fusion for Forward and Backward Passes

Once TorchDynamo has captured an FX Graph for as much of the forward pass as possible, the next compiler phase is AOT Autograd. AOT Autograd runs the Dynamo-captured forward graph through PyTorch’s autograd engine in “functional” mode to record the backward operations. This is how the static backward graph is produced (this is in contrast to relying on PyTorch’s default autograd engine to execute the backward-pass operations one by one).

In essence, AOT Autograd generates a joint forward-backward graph that can then be optimized and fused as a whole. And it guarantees the same forward and backward results as eager mode.

AOT Autograd works by tracing the forward graph through the autograd engine to capture the gradient computations. It effectively runs the forward graph with torch.autograd.forward_ad (or a similar technique) to record which operations are needed for the backward computations. The result is a combined forward and backward graph. The combined graph can then be optimized using common-subexpression elimination, etc. Later, it’s compiled by a backend like TorchInductor or XLA.

By planning both the forward and backward ahead of time, the PyTorch compiler can holistically fuse across the boundary of the forward and backward passes together. This results in ahead-of-time fusion of operations that span the two phases. For example, it can fuse an elementwise operation in the forward pass with the corresponding elementwise gradient computation in the backward pass, if possible, into one kernel.

The compiler can also eliminate overhead by reusing intermediate results between forward and backward computations when safe to do so. This can improve performance greatly for model training workloads in which backward operations can dominate the overall runtime.

Without AOT Autograd, PyTorch would need to execute each backward operation separately using the default PyTorch autograd engine—independently of the forward pass. With AOT Autograd, the graph of execution is optimized holistically.

The resulting joint graph produced by AOT Autograd is guaranteed to compute the same results (this is why graph breaks are needed when the graph can’t guarantee correctness). This joint graph can be further optimized because the whole sequence of operations, forward and backward, is known.

By fusing operations and memory usage across the forward and backward passes, we reduce memory accesses and kernel-launch overhead. PyTorch’s compiled mode automatically uses AOT Autograd under the hood when you call torch.compile for operations and workloads that involve gradients, such as model training.

In short, AOT Autograd is used by torch.compile to compute gradients ahead of time during model training. It handles most autograd operations seamlessly and guarantees the same results as eager mode. It reuses buffers across the forward and backward passes to reduce peak memory usage. And while this phase is mostly invisible to the end user, it’s a key component to enabling large speedups in modern AI model training workloads.

### PrimTorch IR (Prims) Simplified Operator Set

Before handing the graph over to a low-level code generator, PyTorch performs an intermediate representation (IR) transformation known as PrimTorch IR (Prims) in some documentation and source code. PrimTorch IR is an IR that reduces the variety of operations in the graph down to a smaller core set of “primitive” operations, hence the name *PrimTorch*.

For context, PyTorch has thousands (> 2,000) of operations in its full API. PrimTorch IR reduces this to a much smaller set of primitives on which the compiler can focus. In practice, PrimTorch IR defines around 250 primitive operations, such as basic arithmetic, reductions, copy, reshape, etc.

Many complex or high-level PyTorch aten operations can be decomposed into these primitives with PrimTorch IR. For instance, an “in-place” PyTorch operation like x.add_(y) is lowered into a functional add followed by an explicit copy back into x’s storage, as shown here:

```
%z    = aten::add(x, y)
%copy = aten::copy_(x, %z)    # writes z’s data into x
```

Here, the IR contains a separate aten::copy_ node instead of a special in-place mutation. This makes all tensor updates explicit and simplifies downstream compiler kernels by treating mutations as ordinary copy operations.

Specifically, by converting in-place mutations into functional operations plus distinct copy nodes, the compiler no longer has to reason about aliasing or hidden side effects. As a result, fusion passes and memory-planning algorithms can operate on a purely functional dataflow. This allows more aggressive kernel fusion and predictable, high-performance code generation.

PrimTorch IR also helps with things like eliminating aliasing and mutation in the graph. It tries to convert operations into a form that does not perform in-place updates when possible since these can complicate optimization. The output of the PrimTorch IR pass is an FX Graph that contains only aten IR and PrimTorch IR operations. The FX Graph is then ready for lowering by the backend.

By doing this IR standardization, PrimTorch IR provides a stable, simplified interface for compiler backends to target. Instead of having to implement thousands of PyTorch operations, a backend like TorchInductor only needs to support the 250 primitives—other higher-level operations are derived from the primitives. This greatly reduces complexity.

And since most high-level operations (e.g., 2,000+ PyTorch ops) can be decomposed into the existing primitives, the set of PrimTorch IR primitives evolves relatively slowly—even as PyTorch adds new operations to adapt to new algorithms, models, and techniques.

> The stable PrimTorch IR interface means that to support a new accelerator, for example, developers need only implement the 250 primitives rather than thousands of ATen operations.

To summarize the pipeline so far: TorchDynamo captures a graph, AOT Autograd adds a backward pass and forms a joint forward-backward graph, and then PrimTorch IR canonicalizes the operations to a leaner, more primitive set. At this point, we have a fairly standard, device-agnostic representation of the whole computation—including the forward and backward passes—in terms of core operations. Now let’s turn to the actual code-generation stage performed by the compiler backend.

### TorchInductor Backend Code Generation

The final stage of the torch.compile stack is the compiler backend. TorchInductor, or Inductor, is PyTorch’s default compiler backend. Inductor takes the optimized, joined forward and backward FX Graph consisting of aten + PrimTorch IR operations and generates high-performance code for the target hardware, including NVIDIA GPUs, AMD GPUs, CPUs, and others.

> XLA is an alternate backend targeting non-CUDA hardware. It’s mainly used for Google Cloud TPUs through the OpenXLA project. But it can support other accelerators that adopt XLA IR. For example, Meta’s in-house inference ASIC, Meta Training & Inference Accelerator (MTIA), uses XLA. Additionally, AWS’s custom Inferentia and Trainium accelerator chips run PyTorch with an XLA compiler in its open source AWS Neuron SDK. NVIDIA GPUs typically use TorchInductor, however.

TorchInductor works by *lowering* the graph to a loop-level, define-by-run IR that it compiles to efficient code. Internally, Inductor represents operations as loops over multidimensional data. It automatically groups nodes from the FX Graph into fused loop blocks when possible. Each group becomes a kernel during code generation.

Inductor also supports symbolic shapes, which allow dynamic dimensions. The IR is somewhat higher-level than raw CUDA, as it represents things like elementwise computations as loops over tensor indices. The IR is user-inspectable, which helps debugging and even extending.

TorchInductor’s IR can also be used for ahead-of-time compilation flows with torch.export() and AOTInductor. AOTInductor compiles artifacts produced by torch.export for AOT use cases such as packaging and deployment. This allows saving and reusing the compiled code across runs. Export is highlighted in the context of torch.compile() in Figure 14-4.

![Figure 14-4. PyTorch compile and export (TorchDynamo → AOT Autograd → PrimTorch IR → TorchInductor → Triton/LLVM NVPTX); export via torch.export/AOTInductor; CUDA Graphs are used when shapes are static](AI%20Systems%20Performance%20Engineering-ch14_images/figure-14-4.png)

For NVIDIA GPU backends, TorchInductor uses OpenAI’s Triton JIT compiler to generate the actual GPU kernels. Triton is a CUDA-like domain-specific language (DSL) written in Python. Triton also includes a compiler for its DSL (we’ll cover Triton more in a bit).

TorchInductor translates its loop-level IR into Triton code and then uses the Triton compiler to convert the Triton code into NVIDIA PTX directly using LLVM. Remember that PTX is NVIDIA’s low-level instruction set architecture (ISA) for its NVIDIA GPUs.

> Importantly, Triton lowers to NVIDIA PTX using LLVM NVPTX. It does not invoke NVCC for kernel compilation. This approach lets TorchInductor produce custom kernels on the fly that are tailored to your specific model or algorithm.

The loop-level IR is implemented in Python, which makes it easy to inspect and extend. For example, suppose a graph has an operation z = x.permute(1,0) + x[2,:]. Inductor might represent this operation with the following IR:

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

Here, size0 and size1 are the dimensions of the input, x. And inner_fn describes how to compute one element of the output. The Pointwise node represents a loop nest over those ranges that applies inner_fn elementwise to produce the output.

This is a define-by-run style IR. By running this IR, it’s executing Python that iterates and calls ops.load + ops.add. Inductor then generates the corresponding NVIDIA PTX code using the Triton JIT compiler and LLVM.

> Use torch.library.wrap_triton with triton_op to register a Triton kernel as a first-class PyTorch op with autograd and fake-tensor support. This means you can write a Triton kernel and have TorchInductor optimize it as part of your model graph.

### Autotuning with TorchInductor

TorchInductor includes an autotuner built on Triton’s autotuning capabilities, which we’ll describe in an upcoming section. The autotuner finds the best launch configuration for each generated GPU kernel. The autotuned configuration is cached per kernel so that subsequent runs don’t need to redo the tuning step.

The first time you compile code with the TorchInductor backend, it will spend extra time benchmarking different kernel variants using different block sizes, tile sizes, etc. Inductor picks the fastest variant and uses this going forward. Kernel autotuning increases initial compile-time latency, but the resulting kernels are highly optimized for runtime.

> If you recall from the last chapter, this aggressive autotuning is described as the max-autotune compiler mode. This is the most time-consuming compiler mode—and this is what’s happening under the hood.

Beyond kernel fusion and autotuning, TorchInductor applies many low-level optimizations. These include index simplification to reduce complex index arithmetic in loops, common-subexpression elimination within the generated code, and efficient memory planning to reuse buffers and reduce allocations.

TorchInductor also uses CUDA Graphs to capture sequences of kernels at runtime for faster graph replay with minimal CPU overhead. By default, Inductor will try to wrap its generated kernels into a CUDA Graph to reduce launch overhead on each iteration. This is especially beneficial for inference—or when running any code or model with many kernels.

> The reduce-overhead and max-autotune compiler modes, described in Chapter 13, trigger the use of CUDA Graphs. However, CUDA Graphs require static shapes, so they are not used when dynamic-shape compilation is enabled. Said differently, if dynamic shapes are enabled with dynamic=True, TorchInductor will not use CUDA Graphs. Also, you can use max-autotune-no-cudagraphs when you need autotuning without CUDA Graph capture. In general, start with the default mode and use max-autotune to provide additional speedup for large/critical workloads as the expense of significant compile time. You may not see much benefit for smaller models.

The end result of TorchInductor is highly optimized, device-specific code for your workload. In many cases, Inductor achieves performance close to, or even exceeding, hand-tuned libraries. For instance, Inductor can fuse an entire sequence of elementwise operations, including activations and pointwise transformations, into a single kernel. It can even fuse certain patterns of matrix multiplication followed by elementwise operations like bias-add + activation into one launch. This is relatively difficult to do by hand—and requires ongoing maintenance.

The PyTorch compiler uses heuristics and backends that may route certain operations (e.g., large GEMMs) to high-performance libraries such as cuBLAS/cuBLASLt or CUTLASS. It can even emit Triton kernels for fusable patterns. In practice, TorchInductor selects and caches whichever path performs best—or is known to be optimal for the given shapes.

For transformer models, TorchInductor will fuse the layernorm and residual connection elementwise operations around a large GEMM—while still using cuBLAS for the actual GEMM computation itself. Or, for models with irregular memory access, Inductor’s custom Triton kernel can outperform an existing library’s kernel by doing just the work that’s needed—and not benefit from a general-purpose library like cuBLAS or CUTLASS.

On modern GPUs, PyTorch’s compiler can work with NVIDIA’s Transformer Engine (TE) for certain transformer blocks and layers. However, PyTorch does not automatically substitute NVIDIA TE kernels when you call torch.compile. TE is a separate library that you must use explicitly via its modules or fused ops. But when you call TE APIs, torch.compile can compile and fuse around them. This complements the generated Triton kernels to provide maximum performance.

> Make sure to install NVIDIA Transformer Engine only if you plan to call TE modules directly in your model—for example, transformer_engine.pytorch.layers. torch.compile will not automatically swap TE kernels into a plain PyTorch model.

In essence, TorchInductor does the heavy lifting of turning code and models into high-performance GPU kernels. With each PyTorch release, hardware coverage is expanding and new optimized techniques are emerging. For example, PyTorch provides FlexAttention, a new attention operator that TorchInductor can compile into fused kernels approaching FlashAttention performance. Specifically, FlexAttention’s fused kernels have been measured to reach up to ~85–90% of modern FlashAttention performance in both forward and backward passes, while allowing more flexibility including block-sparsity and custom masks. To enable the fast paths, set the following torch.backend.cuda attributes to True:

```
# Ensure SDPA fast paths are enabled
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)
```

When using torch.nn.attention.flex_attention, make sure your inputs meet the fast-path constraints. This way, TorchInductor can emit the fused Triton kernels.

TorchInductor and Triton support automated warp specialization on modern GPUs. The compiler will enable it selectively when it’s deemed beneficial. Warp specialization can be tuned using Triton meta-parameters such as num_consumer_groups and num_buffers_warp_spec. These optimizations further improve GEMM throughput. Triton’s automatic warp specialization supports TMA and tensor descriptor APIs on modern GPU targets including Blackwell (e.g., tcgen05).

> It’s recommended to use descriptor-based tiled loads/stores to map to TMA and reduce register pressure. This method is preferred over manual tl.load loops.

In short, many models run significantly faster with torch.compile, though the exact gains depend on your models’ characteristics. The first time you run torch.compile, you pay a compile and autotune cost, but subsequent runs use the cached graph and kernels for lightning-fast execution.

### Dynamic Shapes and Variable Sequence Lengths

A major challenge with LLM training and inference is the variable-sized sequence inputs. In traditional compilers and accelerators, varying shapes often cause recompilation or otherwise require padding inputs to a fixed, common size. This section discusses how dynamic shape tracing works in torch.compile to handle variable-length sequences.

Fortunately, the PyTorch compiler stack is designed to handle dynamic shapes gracefully. Specifically, it allows models to accept different input sizes without recompiling every time by using the SymPy library to represent unknown dimensions symbolically, as we’ll cover in a bit.

The PyTorch compiler will automatically mark dimensions as dynamic if it observes changes in their size. TorchInductor starts with static assumptions and then generalizes on recompile if it detects shape variability. You typically see one extra compile for the first new shape. By setting the dynamic=True flag upfront, you will force the compiler to consider all dimensions as dynamic from the start. However, remember that setting dynamic=True will disable CUDA Graphs. Prefer marking the code with only known varying dimensions using torch._dynamo.mark_dynamic().

TorchDynamo and TorchInductor insert a guard-like sequence_length <= 256 (or whatever range you specify) during tracing on dynamic dimensions to generate code that works for a variety and range of sizes. For instance, if an output size is x.size(0) + y.size(0), Inductor can represent that as a symbolic expression and ensure the generated code works for any values that satisfy the guard conditions.

When Dynamo encounters a new shape for sequence_length, it sets a new guard such as sequence_length <= 1024 and compiles the kernels under this new assumption—treating the dimension as dynamic from that point on. Later, if a longer sequence is seen that violates the guard, the compiler will recompile a new version of the graph that handles the larger range. Over time, it builds up a cache of compiled kernels for each different shape range.

You can also manually mark expected dynamic dimensions on a tensor with torch._dynamo.mark_dynamic(tensor, dim) to preempt a recompile. You can also use torch.compiler.set_stance(), which lets you adjust how recompilations are handled. For instance, you can use an eager-on-recompile stance to fall back to eager mode after a certain number of recompiles. We’ll discuss best practices for avoiding recompiles in a bit.