Compiled FSDP fuses forward and backward passes and reuses buffers across model shards using AOT Autograd and Inductor’s memory planner. As such, only the active parameter slices and minimal intermediate memory buffers are resident on each GPU. This reduces peak memory usage compared to DDP or eager mode.

The memory savings comes from avoiding redundant gradient storage, reusing intermediate allocations, and overlapping communication with computation across shards. And these allow larger models to fit into each GPU. If you don’t wrap submodules individually, FSDP falls back to treating all parameters as one big bucket. This still works, but it limits memory benefits and overlap potential. As such, combining torch.compile with per-module FSDP wrappers is recommended for maximum speed and memory efficiency—especially on large-scale training jobs.

> Debugging can be very complex if issues arise—and even more complex in a larger cluster/configuration. Always test on a smaller configuration when using FSDP with torch.compile.

*Monitor performance tradeoffs when using custom and third-party CUDA C++ and* *Triton operations* If you rely on a custom or third-party CUDA extension that PyTorch doesn’t know about, Dynamo will create a graph break because it can’t reason about what that operation does—or whether it’s safe. If it’s performance-critical, consider rewriting the custom operation in Python using Triton.

PyTorch supports torch.library.triton_op() API that lets you integrate Triton kernels as custom operations into PyTorch seamlessly. This lets the compiler peek inside the Triton code to perform optimizations. Before diving into Triton, let’s quickly summarize how to debug various compiler phases, graph breaks, and compiler performance.

> Many popular third-party libraries now provide either a Triton implementation or Dynamo/FX wrappers for their operations. Check if these exist before writing your own.

## Debugging Compiler Phases, Graph Breaks, and Performance

You can log and debug different types of compiler events at runtime by setting various environment variables such as TORCH_LOGS, TORCH_COMPILE_DEBUG, and TORCHDYNAMO_REPRO_*. These include graph breaks, recompiles, guards, and other compiler decisions. An example of setting TORCH_LOGS is shown next (see Table 14-1 for common values):

```
# "graph_breaks", "dynamo", "aot_graphs", "inductor",
# "graph_outputs", "graph_code", "dynamic", "perf_hints",
# "output_code", "recompiles", "guards", etc.
TORCH_LOGS="graph_breaks" python train.py
```

This will cause PyTorch to print out whenever a graph break occurs. To summarize the different logging options, you can set TORCH_LOGS to the following to debug torch.compile, including the different phases (TorchDynamo, AOT Autograd, and TorchInductor), graphs, graph breaks, generated code, performance, recompiles, and guards—as well as compiler decisions and performance, as shown in Table 14-1.

Table 14-1. Logging options for torch.compile

| TORCH_LOGS value | Description |
| --- | --- |
| graph_breaks | Logs graph break events |
| dynamo | Verbose logging from TorchDynamo |
| aot_graphs | Verbose logging from AOT Autograd |
| inductor | Verbose logging from TorchInductor |
| graph_outputs | Shows the compiled FX graphs |
| graph_code | Dumps the Python code for each FX graph that TorchDynamo produces |
| dynamic | Traces decisions around dynamic shapes and when dimensions are marked as dynamic |
| perf_hints | Shows you the missed performance-optimization opportunities |
| output_code | Prints the generated code for each compiled graph |
| recompiles | Logs recompilation triggers |
| guards | Logs guards and guard evaluations |

These settings can be useful if you suspect an issue in how the subgraphs were segmented—and which shapes were compiled. With these settings, you will get a lot of internal debugging information without changing your code.

> Be prepared for very verbose output. It’s recommended to start with just "graph_breaks" when debugging just graph breaks, for example.

Under the hood, setting TORCH_LOGS is analogous to using the torch._logging.set_logs() API. However, setting TORCH_LOGS is sometimes easier to configure externally as an environment variable.

And remember that you can also set TORCH_COMPILE_DEBUG=1 to enable TorchInductor’s debug mode. This will log the FX Graph, the TorchInductor IR, the generated Triton code, and an HTML report with visualizations if Graphviz is installed.

You can also set TORCHDYNAMO_REPRO_AFTER and TORCHDYNAMO_REPRO_LEVEL to force TorchDynamo to dump its graph after each stage. It will also perform a runtime comparison against a noncompiled, eager-mode version of the code.

It’s also possible to trace through compilations logs using a tool called tlparse. Trace logs are useful for debugging compilation events (e.g., recompilations) as well as generating bug reports.

To enable trace logs, specify the *trace-log* directory using the TORCH_TRACE environment variable. Then run tlparse on the *trace-log* directory to produce a tree representation of stack frames as shown here:

```
- /workspace/networks/layers/transformer.py:634 in forward
  .../torch/nn/modules/module.py in _wrapped_call_impl
 .../torch/nn/modules/module.py in _call_impl
  - [2/2] [2/3] ../torch/_dynamo/convert_frame.py in __call__
  - /workspace/networks/layers/transformer.py:753 in forward
    - [8/2] [8/3] .../torch/_dynamo/convert_frame.py in __call__
...
```

In addition, you can use the Perfetto UI to display a trace timeline visualization. And since tracing incurs minimal overhead, it’s even possible to enable TORCH_TRACE in production.

Let’s now dive deeper into OpenAI’s Triton language and compiler used by TorchInductor. We’ll write some basic and advanced Triton kernels and then register them with PyTorch.

## Writing Custom Kernels with OpenAI Triton

Up until now, we’ve only briefly mentioned OpenAI’s open source Triton language and compiler. Now it’s time to dive deeper since TorchInductor uses Triton as its backend code-generation implementation—and because Triton is growing in popularity with backing from large companies like OpenAI.

As mentioned, Inductor uses Triton to generate optimized GPU kernels under the hood. By examining, understanding, and customizing these kernels, you can further improve performance beyond what TorchInductor could produce. Learning Triton is critical to performance optimizations in a PyTorch and NVIDIA GPU environment.

At a high level, OpenAI Triton is an open source, Python-native domain-specific language (DSL) for writing GPU kernels in familiar Python. Triton also includes a JIT compiler that converts Triton code into NVIDIA PTX code directly. In other words, Triton lets you create high-performance custom GPU operations in Python—without writing CUDA C++ by hand. Triton remains tightly integrated with PyTorch, making it the go-to choice for custom GPU kernels in this ecosystem.

Writing a GPU kernel in Triton is much more familiar and simpler than CUDA C++. This is especially true for researchers who prefer to stay in Python, iterate quickly, and not worry about complex C++ templates or detailed memory management. They simply don’t need to use C++ in an era when GPU-performance-focused compilers like PyTorch and Triton exist.

> NVIDIA has recognized this trend. In 2025, they announced Python-centric CUDA libraries (e.g., cuTile, CuTe Python DSL, CUTLASS Python DSL, and cuPyNumeric numpy replacement). These are essentially competing libraries to Triton. Integration with torch.compile continues to evolve, and as of this writing, TorchInductor still uses Triton as its primary GPU code generation path.

While PyTorch’s torch.compile automates a lot of kernel generation, custom Triton kernels can squeeze out the last drops of performance—especially for operations outside of TorchInductor’s current scope like complex sparse patterns and novel layer types. It’s sometimes possible to beat the performance of TorchInductor’s generated code—especially if you have domain-specific knowledge. However, this is very advanced and will require ongoing maintenance and potential rewrites for new hardware support.

Let’s now start with a quick Triton programming primer. Then we’ll dive into some interesting Triton topics, including accessing shared-memory, registering a Triton kernel with PyTorch, autotuning kernel-launch parameters, and profiling. Then we’ll progress to cover advanced Triton topics such as warp specialization and software pipelining (e.g., double buffering).

### Triton Programming Model

Triton uses a single-program, multiple-data (SPMD) model, as opposed to CUDA’s SIMT model. This is significant because Triton intentionally abstracts away the low-level details of CUDA instructions and threads.

Triton kernels (aka *programs*) operate at a higher level by running instances of the program on separate thread blocks (aka *cooperative thread arrays*, or CTAs) as the fundamental unit of compute. This is in contrast to CUDA kernels, which run on individual threads in a thread block.

> The community tends to use Triton *kernel* and Triton *program* interchangeably—typically preferring Triton *kernel*, so this book uses Triton *kernel* for the most part.

You write a Triton kernel with the Triton Python DSL. Then the Triton JIT compiler compiles the kernel into GPU code that runs many parallel instances of this kernel. Each program instance maps to a CUDA thread block.

Triton kernels (aka *programs*) are defined by decorating a Python function with @triton.jit. Within the kernel, you use special primitives from the triton.language module, commonly aliased as tl, to work with memory pointers, perform vectorized loads/stores, and compute per-program indices using tl.program_id and block offset arithmetic.

Triton’s SPMD model means you typically work with vectorized operations such as adding two tl.arange vectors. The Triton compiler maps vectorized SPMD code across the threads in a CUDA block. There is no guaranteed one-element-to-one-thread mapping.

You don’t explicitly need to manage individual threads or warps with Triton since its compiler does this for you. Here is a simple Triton kernel that adds two vectors of equal size, n_elements, in this case:

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

Here, you see that Triton abstracts away threads and warps. Note that BLOCK_SIZE is a compile-time constant that defines how many elements each program instance processes. The number of threads per CUDA block is controlled by the kernel’s configuration using num_warps and is not equal to BLOCK_SIZE.

Specifically, in the preceding code, tl.arange(0, BLOCK_SIZE) returns a vector of indices of size BLOCK_SIZE ([0, 1, ..., BLOCK_SIZE-1]). We add pid * BLOCK_SIZE, or block_start, to the vector of indices in order to derive the actual indices, x_ptr + offsets and y_ptr + offsets, into each vector for this instance of the kernel running on a thread block.

Assuming we launch enough Triton kernel instances to cover the total number of elements, n_elements, in each of the vectors, this kernel will add together every element of the two vectors, x_ptr and y_ptr, and store the result in out_ptr. In essence, Triton lets you write kernel logic in a tensorized manner.

Here, for example, we operate on a whole block of indices (offsets) at once. The Triton compiler takes care of splitting this work among actual GPU threads and makes sure that memory accesses (tl.load and tl.store) are coalesced when possible.

To launch instances of this Triton kernel, pass a grid function that computes the number of program instances from meta['BLOCK_SIZE']:

```
import triton
def grid(meta):
    return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
vector_add_kernel[grid](x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE=1024)
```

Here, the code uses a mask to avoid out-of-bounds memory access when n_elements isn’t a multiple of BLOCK_SIZE. This is similar to earlier chapters on CUDA in which we used if (idx < N) within our kernel to avoid out-of-bounds index errors.

> The use of a mask in loads/stores is a clever and convenient way to handle boundary conditions without requiring explicit checks or if/else branches.

Under the hood, Triton converts this program to NVIDIA PTX such that each program uses a single CUDA thread block. Each program maps to a CUDA thread block. tl.arange produces per-lane indices within the program, and the compiler maps this vectorized index space across the thread in the block. You can also manage multidimensional indices for matrix operations in a straightforward way. Triton will automatically handle vectorizing your arithmetic and memory operations for you.

In short, Triton gives you the productivity of Python with the performance of optimized CUDA C++ kernels. It also lets you drop down into low-level optimizations to manipulate and utilize the full memory hierarchy (e.g., shared-memory tiling, etc.), as we demonstrate in the next section.

### Accessing Shared Memory in Triton

Efficient Triton kernels take advantage of the L2 cache and software-managed shared memory on each SM. When using shared memory, each thread block loads a tile from both matrix A and B into shared memory. This is in contrast to each thread repeatedly loading the same values from global memory.

The kernel then reuses those tiles for multiple computations. This better utilizes the on-chip memory caches and reduces the amount of data traveling between global HBM and the registers.

Triton does not expose an explicit shared-memory allocator. Instead, it stages tiles in on-chip shared memory using tensor descriptors (tl.make_tensor_descriptor(...)) and an asynchronous pipeline using the intended shapes and strides. This way, you can issue loads and stores through those descriptors inside a pipelined tl.range(..., num_stages=...). This loop lowers to cp.async, TMA, and barriers.

### Registering Custom Kernels with PyTorch

After writing a Triton kernel, you can register it as a custom operation in PyTorch using torch.library.triton_op. This makes the Triton kernel visible to torch.compile without treating it as an opaque, black-box operation that could fall back to eager execution mode. This way, the compiler knows about the Triton kernel, includes it during graph capture, and optimizes it along with the rest of the graph. This allows additional optimizations such as fusion.

Registering the Triton kernel helps avoid graph breaks when using custom Triton kernels/programs with the PyTorch compiler. Here is an example of registering and calling the Triton kernel vector_add_kernel from PyTorch:

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

Here, triton_op("my_triton_lib::vector_add", mutates_args=()) registers the operator name and mutation metadata (empty) with PyTorch. Then wrap_triton (vector_add_kernel) wraps the raw Triton kernel into a callable that the compiler can inline and optimize within the torch.compile graph. The compiler will then fuse, reorder, and inline this kernel within the rest of the torch.compile graph.

Registering forward-only operations is straightforward. However, to leverage PyTorch’s automatic differentiation for full training support, you typically need to implement and register a custom backward computation. Otherwise, you need to compose it from existing differentiable primitives.

For training support, register an autograd formula using vector_add.register_autograd(backward, setup_context=setup_context). If you prefer, you can wrap the logic in a torch.autograd.Function and register both the forward and backward arguments. However, register_autograd is the recommended path for torch.compile composability.

> If OpenAI Triton doesn’t support something that you need—or doesn’t provide the performance that you expected—you can rewrite the kernel using CUDA C++ with a library like CUTLASS for efficiency. We would then register the CUDA C++ extension with PyTorch in a similar manner, including registering the autograd gradient computation for the backward pass.

### Tuning Kernel-Launch Parameters

Triton programs typically use 4 warps, or 128 threads, per block for many kernels. However, with modern GPU hardware’s larger shared memory and register file sizes per SM, you can typically push num_warps higher to 8 or 16 warps per block. For instance, you can increase num_warps to 8 when BLOCK_SIZE >= 2048 and to 16 when BLOCK_SIZE >= 4096.

The number of warps is dependent on whether your kernel can make use of the parallelism without causing excessive contention. The optimal setting depends on the kernel’s arithmetic intensity and memory access pattern.

Consider launching a kernel as follows: my_kernel[grid](..., num_warps=8). In this case, we are specifying 8 warps (256 threads) per Triton kernel. This configuration is typically effective for compute-heavy kernels. However, memory-bound kernels might still top out around 4 warps due to memory throughput limits.

For memory-bound kernels, using more warps per thread block can help hide memory latency by doing more in parallel. But too many warps per thread block can cause contention or cache thrashing.

New GPU generations are gaining more SMs and wider memory buses. This lets us increase the number of warps per block from the default 4 warps to 8 or 16 warps. This helps to increase occupancy, cover more memory-access latency, and saturate the available memory and compute.

Manually exploring combinations of BLOCK_SIZE and num_warps for each kernel can be tedious. As such, it’s usually best to use Triton’s built-in autotuner. This will benchmark and automatically pick the optimal BLOCK_SIZE, num_warps, tile size, and other parameters for you. Let’s explore the autotuner in the next section.

### Autotuning Triton Kernels

GPU kernel performance is highly sensitive to compile-time parameters such as tile dimensions, warp counts, loop unrolling stages, and the use of on-chip resources like registers and shared memory. Triton’s built-in autotuner automates the search for these optimal settings by letting you decorate a triton.jit kernel with @triton .autotune. You can pass in a list of triton.Config objects that describe the different candidate combinations of BLOCK_SIZE, num_warps, num_stages, tile size, and other kernel meta-parameters.

During the first kernel invocation, the Triton JIT-compiles and benchmarks each configuration combination. Be sure to use a representative input workload on this initial invocation, as Triton will cache the fastest configuration for that input using a key derived from its characteristics, such as input size/shape.

All subsequent calls that have these same input characteristics will automatically reuse the cached (fastest) configuration. This way, you only pay the autotuning cost once for each input size/shape—and immediately start benefiting from the optimal configuration in later kernel invocations.

If Triton detects a new input shape, it will perform another autotune process by iterating through the triton.Config objects using the new input characteristics. It will again choose the best configuration for this input and cache it for subsequent kernel invocations.

To avoid suboptimal tuning results, it’s recommended that you warm up the autotuner with realistic and representative inputs that closely match your production workload. This way, Triton populates the cache with an optimal configuration that closely reflects your production inputs.

> You can override the optimal settings for specific input shapes and workloads by supplying a custom key_fn to @triton.autotune(key_fn=...) that maps the input metadata (e.g., tensor shapes) to a custom cache key. This is an advanced technique that gives you more control of the cache configurations for different types of input workloads.

When choosing possible kernel configurations, it’s worth remembering that larger tiles and more warps will increase arithmetic intensity at the expense of consuming additional registers and shared memory per thread block. In other words, by increasing the compute-to-memory ratio, you limit occupancy since fewer thread blocks can execute on each SM due to the increased resource needs.

Conversely, using smaller tiles and fewer warps will reduce per-thread work and data reuse but allow more blocks and warps to be active on each SM concurrently. This improves occupancy at the expense of lower arithmetic intensity.

In short, the optimal trade-off depends on both your input-matrix dimensions and your GPU’s specific resource limits. Manually tuning is time-consuming and error-prone. Triton’s autotuner handles this complexity automatically using a data-driven approach on realistic workloads to determine the optimal configuration that a manual search might miss. Using higher num_warps (e.g., 8–16) and multistage pipelining will often saturate tcgen05.* paths on Blackwell. It’s recommended to use autotuning as much as possible.

## Advanced Triton Kernel Implementations

To solidify these concepts, next are some self-contained Triton kernel examples for warp specialization and asynchronous double buffering of data transfers/computations. These illustrate how you can implement Triton to transform high-level Python code into highly optimized GPU kernels.

### Warp Specialization with Triton

TorchInductor can target Triton’s warp specialization support for many of its generated GPU kernels. It will try to split each thread block’s warps into “producer” (memory) and “consumer” (compute) roles by emitting tl.range() loops with warp_specialize=True, similar to the example shown here:

```
// warp_specialize=True is supported on modern GPUs
// Use it together with num_stages > 1
// to enable producer/consumer warp partitioning
// and overlap
for k in tl.range(0, K_tiles, _warn_unused=False, warp_specialize=True):
    # loop body
    ...
```

The memory warp prefetches the next tile while another warp computes the current tile. This will overlap memory latency with computation to produce higher throughput. Warp specialization works hand-in-hand with descriptor-based TMA copies. You can also use this in your own custom Triton kernels by passing warp_specialize=True to tl.range(), as shown in the code.

You can also drive warp specialization through Triton autotune configs by setting num_consumer_groups>0 (e.g., 2) and num_buffers_warp_spec (e.g., 3) in triton .Config as shown in the following code snippet. This will keep the producers and consumers busy with work. If provided, TorchInductor will use these values under the hood:

```
triton.Config(
    { 'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,
      'num_warps': 8, 'num_stages': 2,
      'num_consumer_groups': 2, '
      num_buffers_warp_spec': 3 }
)
```

This specialization approach is especially effective for long-running loops that iterate over a large *K* dimension in a GEMM. This dedicated approach keeps both the memory subsystem and the ALUs busy at all times and maximizes hardware utilization.

### Tiled and Persistent GEMM Kernel (Triton)

This Triton kernel computes a matrix multiplication (C = A * B) efficiently since each kernel launch does all the work by looping over the K dimension internally, instead of launching multiple kernels for each K chunk. This way, we pay the launch overhead only once, and warps stay busy until every tile is done. The following example tiles over K inside one launch but does not reuse the same thread block across multiple output tiles:

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

Here, the kernel launches a 2-D grid over the M×N tiles and performs the full K-loop inside a single kernel launch. This reduces launch overhead and can increase utilization when K is large, but comes at the cost of holding resources longer in a single kernel. Each program (thread block) loads tiles of A and B into shared memory and computes a partial dot product of the tiles with tl.dot. Triton accumulates the results in FP32. And, on Blackwell, Triton lowers the tl.dot to tcgen05 and UMMA to engage the Tensor Cores. The Tensor Cores then accumulate results in specialized TMEM rather than general registers.

> It’s best to express shared-memory–backed tile movement in Triton using tensor descriptors. For instance, desc=tl.make_tensor_ descriptor(...). On modern GPUs, these tensor-descriptor calls map to TMA-based hardware operations using asynchronous, coalesced transfers.

Triton will unroll/vectorize these loops and computations for efficiency. In addition, when dtypes and tile shapes are supported (e.g., FP16/BF16, or TF32 for FP32), Triton lowers tl.dot to Tensor Core instructions. Here is a simple Python wrapper that launches this Triton kernel: