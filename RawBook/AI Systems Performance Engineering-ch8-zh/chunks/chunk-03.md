In SIMT execution, the warp must execute both paths serially: first, all threads that take the if branch execute while the others in that warp are masked off (idle). Then the threads that took else execute while the first group idles, as shown in Figure 8-7.

During the divergent sections, effectively only half, or some fraction, of the warp is doing useful work. This reduces overall throughput. In the case of a 50/50 split between if and else, the warp’s active utilization drops to 50% for that portion of code. If the split is 1 thread versus 31 threads, then 31/32 threads will be idle in one of the subbranches.

![Figure 8-7. Divergent versus nondivergent warp execution](AI%20Systems%20Performance%20Engineering-ch8_images/figure-8-7.png)

> If your kernel contains multiple divergence points or if each divergent branch carries heavier work, removing those branches can compound these gains. In the ideal case (e.g., a 50/50 split branch), removing that divergent branch can nearly double throughput up to ~2× speedup. Eliminating multiple heavy divergence paths can compound these gains.

The overall effect is that warp divergence causes some GPU cores to sit idle. This increases the total instruction count since each branch path is executed serially by different subsets of threads. Therefore, minimizing intrawarp divergence is a key to warp-level efficiency.

### Techniques to Avoid Warp Divergence

There are various best practices to minimize warp divergence, including restructuring conditions and separating branches into multiple kernels.

> Warp divergence only affects threads within a warp. Threads in different warps do not cause each other to stall.

Additionally, you can use warp-unanimous branches, warp-vote intrinsics, and predication. Let’s discuss each of these:

*Restructure conditions* Wherever possible, organize your computations so that threads in the same warp follow the same execution path. This might involve moving a divergent condition to a higher level outside of the inner loops—or into a separate kernel launch. You could also sort or group your data so that each warp handles more homogeneous cases.

For instance, if you have an array of values and you want to process negative values differently from nonnegative ones, a naive approach might put an if(x<0) inside the kernel that diverges per element. A smarter approach is to partition the data such that one kernel handles all negatives and another kernel handles nonnegatives. By aligning data with warps, you reduce the chance that a single warp has to split its execution.

*Separate into multiple kernels* Another approach is to separate the work into multiple kernel launches such that one kernel handles the if case and another handles the else case. You can use a prefix sum or compaction to distribute threads to the different kernels. This avoids divergence at the cost of launching more kernels and adding logic to distribute data between the kernels. This might be worth it if divergence is a big issue and the divergent sections are large enough in instruction count.

*Rewrite conditions to be warp-unanimous* In some cases, you can rewrite a condition to be *warp-unanimous*. This means that either all 32 threads in a warp satisfy the condition or none do. A common trick is to use the warp index in the condition statement. For instance, you first compute a warp ID as int warpId = threadIdx.x / 32;. Then half of your warps do task A and the other half do task B. For this, you write: if (warpId % 2 == 0) { /* Task A */ } else { /* Task B */ }.

In this case, all threads in a given warp either go into task A or task B together, and there’s no warp divergence since one warp executes one branch while another warp executes the other branch. But within a single warp, all threads agree on the branch to execute.

The slight overhead is that every warp still evaluates the branch condition, but since they all agree on it, each warp executes only one of the branches. This technique essentially trades some flexibility—since you’re constraining how work is divided—to gain warp coherence.

> Use warp-unanimous branches when you can divide work into coarse increments of 32 threads.

*Utilize warp-vote parallel algorithms* Warp intrinsics (__ballot_sync, __any_sync, __all_sync), cooperative-groups’ warp.ballot/any/all, and device-side vote masks (%WarpVote) let a warp collectively decide, or vote, which lanes need “special” work. The warp then dynamically delegates, repartitions, and compacts that work into one (or a few) lane(s) instead of diverging all 32 threads. This avoids per-lane branch divergence but introduces some potential load-imbalance trade-offs that you should profile for impact. We will cover warp intrinsics in more detail in a bit.

*Predicate short lanes* The CUDA compiler will sometimes *predicate* short conditional code. Predication means that the compiler converts an if into a boolean mask for each thread—and executes both paths for all threads. However, it only commits the results appropriate to each thread’s path.

Predication avoids divergent branching at the cost of doing extra work per thread. This is beneficial when the branches are very short and divergence is high. As a programmer, you can encourage predication by writing branch-free constructs, including the ?: ternary operator or bitwise logic tricks for simple conditions. For instance, consider this naive implementation, which uses an if/else statement:

```
if (x > 0)
    y = f(x);
else
    y = g(x);
```

Instead, you could write the following, which computes both f(x) and g(x) for all threads but multiplies each result by cond to select the result that matches the condition:

```
float cond = x > 0 ? 1.0f : 0.0f;
y = cond * f(x) + (1.0f - cond) * g(x);
```

Here, there’s no branch, thus no divergence, but we did extra work for each thread since both functions still run, including the one that wasn’t needed for a given case. As such, predication is worthwhile only if the extra work is cheaper than the cost of divergence would be. In practice, you should profile the effects of these optimizations. Specifically, if using predication, check metrics like “predicated-off threads” and overall instruction count. Profilers like Nsight Compute will show if predication is reducing warp stalls—or if it’s performing unnecessary work that reduces goodput.

Typically, predication is good for very short, simple branches with just a few instructions each—especially if many warps would diverge due to the given conditions. So if the branch involves a lot of work—or only a handful of threads diverge—predication could actually be worse. Use this technique carefully—and only for small conditional workloads.

> CUDA compilers will often apply simple predication automatically for you, but it’s still worth profiling for performance-critical kernels.

In short, minimizing warp divergence requires careful algorithm and data organization. Whenever possible, restructure your problem so that each warp follows a uniform execution path. This might mean splitting the kernel to handle different cases in separate launches, rearranging the data so that each warp processes similar items, or pulling divergent conditions out of inner loops.

When divergence is unavoidable (e.g., tree or graph workloads with inherently varying per-thread work), keep the divergent sections as short and infrequent as possible so that warps reconverge quickly. You can also leverage warp-level intrinsics like __any_sync, __ballot_sync, and other voting primitives to have threads agree on conditions together. You can also try to compact work into fewer lanes rather than branch all 32.

### Profiling and Detecting Warp Divergence

Nsight Compute’s Source Counters and Warp State stats will pinpoint issues like divergent warps. You can detect these by looking for high Branch Divergence metric values or poorly coalesced loads that will show many replay or L2 cache miss events.

Nsight Compute can also flag inefficiencies, including shared-memory bank conflicts and Tensor Core pipeline stalls. For example, if your kernel shows a high percentage of memory_throttle stalls, it might indicate that there is not enough memory-level parallelism for the GPUs, deep instruction pipelines. You can try increasing the number of independent memory accesses in flight—or use asynchronous copy instructions like cp.async and the CUDA Pipeline API (covered in Chapters 9 and 10)—to hide latency. A profiler will show symptoms of warp divergence in the form of low *warp execution efficiency*. This measures the average percentage of threads in a warp that are active. A warp execution efficiency of 30% means, on average, only 30% of the threads are doing useful work at any time. The rest were inactive due to divergence.

When you profile a kernel with divergent branches, the profiler will flag a high percentage of *predicated-off* instructions. These are instructions that are fetched and issued but do no work because their lane mask is disabled. You’ll also see an inflated “dynamic instruction” count compared to the data you actually processed. Together, those two numbers tell you that every warp is walking down all sides of your conditionals in turn and serially.

For instance, one subset of threads, or *lanes*, executes path A while the rest of the threads sit idle. Subsequently, the idle threads become active threads and execute path B. The result is a doubling of your instruction traffic and a serious reduction in your warp’s effective throughput, or goodput, since each inactive set of threads still fetches and issues instructions—even though they’re masked out.

This is somewhat easy to spot in source code, as any per-thread conditional, whether it’s a threshold check, a sparse‐data loop, or a data-dependent filter, will trigger this serialized, repeated execution. To reclaim performance, you must eliminate or flatten those divergent branches.

You can make the condition uniform across the warp, pull it out of tight loops, or replace it with arithmetic or lookup‐table techniques so that all 32 threads execute the same instruction stream and perform useful work. Let’s look at an example to make things clearer.

If you see low warp execution efficiency, inspect your kernel’s branches. It may be beneficial to refactor conditional logic into separate kernels or use warp-level primitives (e.g., ballot sync) to handle divergence more efficiently, as we’ll cover next.

### Using Predication to Minimize Divergence

Let’s show an example using predication to profile and eliminate warp divergence and branches. Consider a kernel that thresholds an array using if (x[i] > 0) y[i] = x[i]; else y[i] = 0;.

If half the values are positive and half are not, then most warps will have some threads take the if branch and some take the else branch. The warp will first execute the if branch instructions for the threads when the condition is true. It will simply mask out the other threads. It then runs again and executes the else branch for the remaining threads. So this took two sets of instructions to accomplish one logical set of operations. This decreases efficiency by 50%.

*Before example (CUDA C++).* In this example, if some threads in a warp satisfy X[i] > threshold and others do not, the warp will diverge. This is a clear example of a kernel that will cause warp branch divergence:

```
// threshold_naive.cu
__global__ void threshold_naive(const float* X, float* Y,
                                float threshold, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if (X[i] > threshold) {
            Y[i] = X[i];    // branch 1
        } else {
            Y[i] = 0.0f;    // branch 2
        }
    }
}
```

This results in executing the warp twice, one after another in serial. One execution will run with the assignment Y[i]=X[i] for one subset of threads, and the other execution will run with Y[i]=0 for the other subset of threads. The warp execution efficiency will be low—approximately 50% if half of the threads take each path.

*After example (CUDA C++).* Here is a divergence-reduced approach using predication. In this version, we use the ternary operator, which the compiler is likely to translate into a predicated move instruction (SEL/MOV based on condition) rather than an actual branch:

```
// threshold_predicated.cu
__global__ void threshold_predicated(const float* X, float* Y,
                                     float threshold, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = X[i];
        // Use a conditional move or multiplication by boolean
        float val = (x > threshold) ? x : 0.0f;
        Y[i] = val;
    }
}
```

In this case, all threads execute the same instruction sequence of computing val, then storing it. This avoids warp divergence because the control flow is uniform across the warp. But threads for which the condition is false will just set val=0. The result is that warp execution efficiency stays high since all threads follow one path.

> In simple cases, the CUDA NVCC compiler likely generated a predicated move for the ternary operator, as expected. In PTX/assembly, you would see the PTX @p predicate syntax to guard the write without splitting into separate warp paths.

*After example (PyTorch).* In PyTorch, the threshold operation can be done with vectorized operations as follows:

```
# threshold_op.py
import torch

X = torch.randn(N, device='cuda')
Y = torch.maximum(X, torch.zeros_like(X))  # equivalent to Y = X > 0 ? X : 0
torch.cuda.synchronize()
```

The torch.maximum with 0 will execute on the GPU without branching since it uses elementwise max, which is implemented in a vectorized manner. Libraries like PyTorch ensure these elementwise ops are divergence-free at the warp level by using predication or bitwise tricks under the hood:

```
# jit_threshold_op.py
import torch

# In PyTorch, we can compile and fuse this operation
# for even higher throughput
@torch.compile()
def threshold_op(X):
    return torch.maximum(X, torch.zeros_like(X))

X = torch.randn(N, device='cuda')
Y = threshold_op(X)
torch.cuda.synchronize()
```

> As you will see in Chapters 13 and 14, torch.compile uses TorchInductor to fuse many pointwise operations into kernels, which better match the performance of the manual CUDA C++ example. However, torch.compile does not guarantee optimal occupancy or ILP, so you may still need to profile and tune performance further.

Under the hood, these types of elementwise functions compile down to single-instruction, multiple data (SIMD)‐style predication or bitwise select operations on GPUs. This ensures that every thread in a warp follows the same instruction stream and avoids the serialization penalties of divergent control flow.

Overall, reducing warp divergence improves the warp’s efficiency and yields substantial speedups in which divergence was a major issue. Table 8-3 shows the results of reducing warp divergence by replacing branches with predicated operations.

Table 8-3. Profiling results of removing warp divergence

| Metric | Before | After |
| --- | --- | --- |
| Kernel execution time | 30 ms | 15 ms (–50%) |
| Dynamic instruction count | 600 M instructions | 300 M instructions (–50%) |
| Average warp branch-resolving stall latency | 200 cycles | 100 cycles (–50%) |
| Warp execution efficiency | 50% | 99% |
| Predicated-off threads (%) | 50% | 0% |

> Note: The numeric values in all metrics tables are illustrative to explain the concepts. For actual benchmark results on different GPU architectures, see the GitHub repository.

By replacing the two‐path branches with a single predicated max() operation, we saw dramatic improvements across all key Nsight Compute metrics. The kernel execution time measured by gpu__time_elapsed.avg dropped from 30 ms to 15 ms, effectively doubling throughput.

Warp execution efficiency climbed to nearly 100% since all threads in each warp stayed active with useful work. And the profiler further reports a 0% predicated-off ratio, indicating that no lanes were ever masked off under the predicated max() approach. This confirms that nearly all reconvergence overhead has been eliminated.

At the same time, the dynamic instruction count (smsp__inst_executed.sum) dropped by 50% from 600 million to 300 million instructions, since each warp no longer spent cycles executing both sides of the branch serially. The average warp branch-resolving stall latency (smsp__average_warp_latency_issue_stalled_branch_resolving) also halved, from 200 cycles down to 100 cycles.

> If your kernel contains multiple divergence points or if each divergent branch carries heavier work, removing those branches can compound these gains—potentially yielding more than a 2× speedup per branch eliminated.

Note that predication isn’t free. Some threads still compute values (e.g., val = x) and their results are never used. If each branch carries substantial work, computing both sides for every thread can actually cost more than a mild divergence.

In simple cases like our threshold example, predication wins, but you should always benchmark. Try both branch-based and predicated versions under Nsight Compute’s warp execution efficiency metric. If efficiency is low and each branch is light, predication will likely help. If a branch is heavy, allowing some divergence—or using a separate kernel—may be the better path. Ultimately, minimizing divergence is crucial for SIMT performance, so structure your algorithms to keep warps on a single path when possible, and isolate any remaining divergent logic into its own kernel or warp-sized region.

> CUDA compilers will often apply simple predication automatically for you, but it’s still worth hand-tuning and profiling for performance-critical kernels.

### Efficient Intrawarp Communication with Warp Intrinsics

When threads within a warp must share data to compute a reduction/aggregation, for instance, you can use warp shuffle intrinsics, including __shfl_sync and __shfl_down_sync (introduced in Chapter 6) to exchange values directly through registers. This is in contrast to staging them in shared memory and calling __syncthreads(), which negatively impacts performance.

Unlike the shared memory approach that generates extra L1/L2 traffic and requires a full-block barrier, shuffles move data only between registers and implicitly synchronize the 32 lanes in a warp at each instruction. This adds only a few clock cycles of latency.

For a full discussion, including performance comparisons, code examples, and how to implement warp-level reductions or prefix sums, see Chapter 6. If your cooperation spans multiple warps, you must still use block-level or grid-level synchronization (e.g., __syncthreads()) or use cooperative groups (covered in Chapter 10.) But for purely intrawarp communication, shuffles are almost always the fastest choice.

In short, if your algorithm requires threads in the same warp to share intermediate results or coordinate on a computation, reach for __shfl_sync and related warp-level options. Only use block-level shared memory and __syncthreads() if the cooperation truly spans multiple warps.

### PyTorch Considerations for Warp-Level Efficiency

At the pure Python level, PyTorch doesn’t expose warp intrinsics or allow you to directly control warp-level execution. However, as an end user of PyTorch, you indirectly benefit from these optimizations because many of PyTorch’s internal CUDA kernels use warp-level tricks.

For instance, the implementation of torch.sum(x, dim=0), which sums across a dimension of a tensor, will often use warp-level reductions when the dimension being reduced is small (e.g., within 32 elements). In such cases, each warp of threads cooperates using shuffle instructions to produce a partial sum without diverging or using shared memory. This logic is implemented inside the library so that you don’t have to implement it yourself.

If you write a custom CUDA kernel as a PyTorch extension, you can—and should—use these warp-level optimizations in your device code when possible. For instance, if your operation requires summing values within a warp or exchanging data between threads, prefer using intrinsics like __shfl_sync over naive shared memory approaches to increase speed and reduce overhead. Also be mindful of warp divergence in any custom CUDA code.

And if you find yourself using if statements per element, you can instead express the condition as a tensor-level operation using torch.where, for instance, on the whole tensor. This allows the library to handle the condition more efficiently—likely in a vectorized and warp-coherent way.

PyTorch’s JIT compiler and graph executor, discussed in Chapter 10, can fuse elementwise operations. This increases warp efficiency and arithmetic intensity by doing more work per kernel—and more work per byte of memory moved.

When multiple operations are fused into one kernel, a warp processes more work in a single pass. While kernel fusion can’t magically eliminate divergence inherent in the algorithm, it does mean that each warp does more useful work. As such, any divergence overhead is amortized over more computations.

For example, if you have a rectified linear unit (ReLU) operation followed by an abs() operation, a fused kernel could handle both in one pass. So even if there’s a branch, it’s handling the two operations together per warp.

As a CUDA programmer, you should avoid intrawarp divergence and use warp intrinsics for any collective operations within a warp. This ensures all 32 threads in the warp stay busy doing useful work in parallel.

As a PyTorch user, be aware that the library is already doing this for you under the hood. Your role is to write your computations in a way that allows these optimizations to be utilized. For instance, use built-in tensor operations (optimized with CUDA C++) instead of writing explicit Python loops with per-element conditionals. If you are using custom CUDA extensions, apply the same best practices we’ve discussed.

Each warp’s 32 lanes working in unison is the ideal. By minimizing divergence and using fast intrinsics for any needed communication, you ensure warp execution efficiency. These techniques often yield a modest, but meaningful, speedup—perhaps a 1.1× to 1.3× improvement. This can be significant in a tight kernel.

These optimizations also set the stage for instruction-level parallelism, which allows each thread to do more useful work. Let’s cover this next.

## Exposing Instruction-Level Parallelism

As we saw in the occupancy discussion, running many warps concurrently lets the SM’s scheduler switch away from any warp that stalls on a long‐latency operation such as a global‐memory load. In addition to launching enough threads, we can also exploit ILP within each warp so that a single warp does not need to wait for one instruction to complete before issuing the next.

You can rearrange or unroll your code so that each thread issues multiple independent operations (e.g., memory loads and arithmetic instructions) before consuming their results. This way, the GPU keeps its execution units busy while earlier instructions are still pending.

Leveraging ILP allows a single warp to issue certain independent instructions back-to-back, which improves latency hiding. For instance, a thread might load data and then perform unrelated arithmetic while waiting for that load to complete. Figure 8-8 shows an example of multiple instructions overlapping during each cycle.

![Figure 8-8. Overlapping with ILP](AI%20Systems%20Performance%20Engineering-ch8_images/figure-8-8.png)

By unrolling your loop body, you turn what was once “load → multiply → store” each iteration into a sequence that loads and multiplies multiple elements before looping back. For example, instead of:

```
for (int i = 0; i < N; ++i) {

    float ai = a[i];       // load
    float bi = b[i];       // load
    sum += ai * bi;        // multiply after both loads complete
}
```

You can unroll such that each loop iteration issues two independent multiply operations instead of one, as shown in the next code block. This gives the hardware an opportunity to execute the second multiply while the first multiply is still in progress—or while it’s waiting for its operands to load from memory.

This properly overlaps and hides latency. The result is a higher instructions-per-cycle (IPC) and better latency hiding as shown below by computing the two multiplies, ai0*bi0 and ai1*bi1, in parallel:

```
for (int i = 0; i + 1 < N; i += 2) {
    float ai0 = a[i];        // load a[i]
    float bi0 = b[i];        // load b[i]
    float ai1 = a[i + 1];    // load a[i+1]
    float bi1 = b[i + 1];    // load b[i+1]
    sum += ai0 * bi0 + ai1 * bi1;
}
```

Here, we’re loading pairs of values (a[i], b[i] and a[i+1], b[i+1]) before doing any multiplies. Unrolling exposes two independent multiply instructions per loop iteration. This way, the GPU can overlap those arithmetic operations and hide the latency of each operand load.

ILP does not directly change arithmetic intensity (FLOPS per byte). However, it raises overall throughput (FLOPS) by overlapping compute with memory or compute‐to‐compute dependencies. We explicitly unroll to increase independent work per thread and help dual-issue. In other words, ILP boosts throughput by time-multiplexing independent work—not by doing extra work.

> A common misconception is that increasing ILP will perform more operations. However, it doesn’t—it just keeps the GPU’s multiple functional units busy. ILP helps only if your program has idle issue slots due to latency. If your kernel is already fully utilizing the execution units on every cycle (e.g., a tight compute-bound loop with no stalls), increasing ILP won’t actually increase performance.

To encourage the compiler to increase, or expose, ILP, you can use #pragma unroll on short loops or tune -maxrregcount to allow the compiler to allocate more registers for holding intermediate values. This explicitly acknowledges that you want the compiler to increase register usage and potentially reduce occupancy.

In this way, ILP complements occupancy. High occupancy ensures there are many warps to switch to when one stalls, and ILP makes sure that a single warp can issue as many independent instructions as possible to fill the pipeline and hide latency. This is much like superscalar and out‐of‐order execution in CPUs, except it’s directed explicitly by the compiler’s scheduling of your CUDA code.

### Warp Scheduling and Dual Issue Instructions

Each SM contains multiple warp schedulers. For example, Blackwell SMs have four warp schedulers. Each scheduler can issue up to two independent instructions per cycle, as shown in Figure 8-9.

![Figure 8-9. Each warp scheduler can dispatch up to two instructions per cycle called “dual issue”](AI%20Systems%20Performance%20Engineering-ch8_images/figure-8-9.png)

In practice, this means that a single warp can overlap multiple independent operations across successive cycles—and sometimes even in the same cycle if they target different pipelines such as one math, one memory, or one special-function unit (SFU) pipeline. This means that multiple instructions can be in flight and progressing through the pipeline simultaneously. This overlap is the ILP that we discussed earlier.

> On Blackwell, the FP32 and INT32 pipelines have been merged into a single set of unified CUDA cores. As such, they can only execute either FP32 or INT32 instructions in a given cycle—not both at once. Each core must pick one data type each cycle. As a result, any ILP that depended on dual-issuing an INT and an FP in the same cycle will no longer work since Blackwell must choose one or the other per cycle on each core. As such, mixed INT32 and FP32 instruction streams no longer benefit from dual issue on a single core. Mixed streams should instead exploit warp/SM concurrency —and not rely on per-core dual issue. As such, you should prefer dual issue with two independent math and memory instructions (or two independent math instructions such as FP32 and FP16) in flight using ILP. This will increase instruction-issue efficiency.

ILP does not require that two instructions physically fire in the same cycle. Instead, it makes sure that while one instruction (e.g., long-latency global memory load) is still pending, the warp can immediately issue another independent instruction (e.g., arithmetic operation) on the next cycle.

As a result of this overlap, by the time the data returns from memory, the arithmetic work has already made progress. This effectively hides the load’s latency. On Blackwell, multiple instructions can be in flight per warp due to pipelining, and the schedulers issue to different execution units over successive cycles. In practice, ILP appears as a steady stream of independent issues rather than a fixed number of concurrent instructions per warp.

Consider a scenario in which each thread in your kernel loads two array elements—and then multiplies each by a different constant before combining the results. The multiply operation for the first element can execute while the second element’s load operation is still underway.

This overlapping pattern keeps the warps’ execution units busy instead of waiting. This raises overall throughput without changing how many loads and multiplies occur per element. Here is a concrete example of increasing ILP:

```
// Launch configuration: e.g., 256 threads per block, enough blocks to cover N
__global__ void independentOps(const float *a, const float *b,
                               float *out, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = a[i];
        float y = b[i];
        // Two independent operations (no dependency between u and v):
        float u = x * x;
        float v = y * y;
        // Dependent operation that uses both results:
        float sum = u + v;
        out[i] = sqrtf(sum);
    }
}
```

In this kernel, each thread loads two values x and y from different arrays and computes two results u and v. The calculations of u = x*x and v = y*y are independent of each other since neither uses the result of the other.

Structuring the code this way, instead of sum = x*x + y*y, gives the compiler an opportunity to arrange the instruction sequence to expose ILP and increase performance. Specifically, it can issue the instruction for u = x*x, then in the next cycle issue v = y*y before u has finished.

While the first multiplication is running—or waiting for a pipeline slot—the second one can run in parallel in another arithmetic unit. By the time it needs to do sum = u + v, likely both multiplications have been completed. The warp thus had multiple instructions in flight, which helps to hide the latency of those two multiply operations.