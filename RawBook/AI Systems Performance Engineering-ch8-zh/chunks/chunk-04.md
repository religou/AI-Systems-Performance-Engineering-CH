In contrast, consider the code here in which v’s computation was placed after using u. In this scenario, we’d create an unnecessary dependency chain that limits ILP:

```
float u = x * x;
float temp = u + 1.0f;    // some dependent use of u
float v = y * y;
...
```

The kernel computes u = x * x and immediately uses that result to form temp = u + 1.0f. Only *after* computing temp do you issue v = y * y, which does not depend on u or temp. Therefore, its placement after temp forces the GPU to wait until the u + 1.0f instruction finishes before it can issue the multiply for v.

In effect, you create a serial dependency chain: first x * x → u, then u + 1.0f → temp, and only then y * y → v. During the time the hardware is executing (or waiting for) the u + 1.0f operation, it cannot begin y * y even though v is mathematically independent. This ordering artificially limits instruction‐level parallelism because one independent multiply must park itself behind a dependent instruction.

By contrast, consider the following reordered and optimized code:

```
// Optimized ordering with better ILP:

float u = x * x;    // Independent multiply on x
float v = y * y;    // Independent multiply on y (issues immediately after u)
float temp = u + 1.0f;  // Only now use u
// ... use v or temp as needed ...
```

Here, we reordered the code such that both x * x and y * y occur back‐to‐back (i.e., float u = x * x; float v = y * y; float temp = u + 1.0f;). In this case, the GPU can issue the two independent multiplies in successive cycles without waiting for the u + 1.0f add to finish.

This means that while the result of x * x may still be lingering in a “pending” pipeline slot, the hardware can already start executing y * y. Only once both of those multiplies have calculated their results does the GPU perform temp = u + 1.0f.

In other words, you now have two multiplies in flight at once. This fills the execution units and hides latency. This improved ILP, issuing v = y * y as soon as possible, makes sure that the multiply on y overlaps with both the multiply on x and the later add—instead of being forced to wait. Consequently, the warp spends fewer cycles idle, which translates directly into higher throughput.

GPUs rely on the compiler, and sometimes the programmer’s hints, to schedule independent instructions. On modern CUDA compilers, simple patterns like the preceding are usually detected and scheduled optimally. But you can encourage ILP by structuring your code appropriately or using pragma compiler directives.

This example is simple, but it demonstrates the idea that you should not serialize independent work within a thread if you can help it. If a thread needs to load two arrays and do math on both, doing them in parallel, or interleaving them, is better than doing one after the other in a serial manner.

### ILP and Occupancy

On modern GPUs, the maximum resident warps per SM is 64 warps, or 2,048 threads (2,048 threads = 32 threads per warp × 64 warps). The warp schedulers can issue multiple instructions per cycle when dependencies permit. The INT32 and FP32 cores are unified and operate as either FP32 or INT32 in a given clock rather than both at once. The register file per SM is 256 KB, which is 64K 32-bit registers. This large register file is important for sustaining ILP without spilling, as we’ll see in a bit.

If you write a kernel with little to no ILP such that each thread issues exactly one operation at a time, you typically need on the order of 1,536 active threads (~48 warps, or 75% occupancy) to saturate the SM’s execution units (these are not hard limits but rather approximations).

By contrast, exposing even a modest amount of ILP can significantly lower the number of threads required. Table 8-4 summarizes how ILP reduces the threads/occupancy needed to saturate a Blackwell SM at different ILP values.

Table 8-4. How ILP reduces the threads/occupancy needed to saturate a Blackwell SM

| ILP (independent ops/thread) | Threads/SM for ~100% utilization | Occupancy (% of 2,048 threads) |
| --- | --- | --- |
| 1 (no ILP) | ~1,536 threads (48 warps) | 75% |
| 2 | ~1,024 threads (32 warps) | 50% |
| 3 | ~768 threads (24 warps) | 37.5% |
| 4 | ~512 threads (16 warps) | 25% |

Here, two-way ILP, in which each thread issues two independent operations back-to-back, often achieves full throughput with roughly 1,024 active threads (≈32 warps, or 50% occupancy). This is because each warp keeps two operations in flight whenever one is still pending.

Three-way ILP, on the other hand, can saturate with about 768 threads (≈24 warps, or 37.5% occupancy). A four-way ILP may need only 512 threads (≈16 warps, or 25% occupancy) to fill the pipelines. This is because each warp itself does the work of four independent operations.

> More in-flight operations per thread lets each warp keep the math pipelines busy—even at lower thread counts.

Clearly, increasing ILP lets you achieve peak compute throughput with fewer warps. This is especially valuable when your workload cannot launch huge numbers of threads—or when you want to free up on-chip resources for other tasks.

It’s important to note that there is a practical limit on how much ILP you can expose. Even though Blackwell SMs can keep many instructions in flight per warp, each warp can hold only a finite number of pending instructions.

On top of that, the compiler must schedule independent instructions without exceeding the available registers or overwhelming the GPU’s instruction decode bandwidth. The decode bandwidth is the maximum number of instructions per cycle that the instruction‐fetch and decode hardware can push to execution units. Once you hit those limits, adding more independent operations produces diminishing returns.

Once you have enough independent work to keep the issue slots and memory system busy, there is little practical benefit in increasing ILP further. Pushing to five-way or six-way ILP produces little gain and can even hurt performance due to the extra register pressure and instruction overhead.

> On Blackwell, the ideal ILP configuration is often between two and four independent operations per thread. At this point, its schedulers and caches are typically saturated. However, the exact point is kernel and workload dependent. Use Nsight Compute issue and stall metrics to decide where to stop.

### Loop Unrolling, Interleaving, and Compiler Hinting

To exploit ILP in your own kernels, look for opportunities in which each thread issues multiple independent arithmetic or memory instructions before any one result is needed. Common patterns include the following:

*Unrolling small loops* Unrolling loops allows each thread to perform *N* accumulations (e.g., 2× or 4×) with separate accumulator registers. For example, consider the following loop:

```
float sum = 0.0f;
for (int k = 0; k < 4; ++k) {
    float a = A[idx * 4 + k];
    sum += a * w[k];  // dependent on load a
}
```

You can rewrite this to take advantage of ILP, as shown next. This transformation creates four independent load-multiply pairs—one per a0..a3—that can be issued in rapid succession:

```
float a0 = A[idx * 4 + 0];
float a1 = A[idx * 4 + 1];
float a2 = A[idx * 4 + 2];
float a3 = A[idx * 4 + 3];
float sum0 = a0 * w[0];
float sum1 = a1 * w[1];
float sum2 = a2 * w[2];
float sum3 = a3 * w[3];
float sum = sum0 + sum1 + sum2 + sum3;
```

Here, all four loads are issued first. As each load completes, its corresponding multiply can execute and overlap subsequent loads with computations to hide the latency of the memory load. In other words, the GPU interleaves these operations such that while one pair is waiting on memory, another pair can be multiplying. This increases ILP and hides latency.

*Interleaving independent operations* You can interleave operations that use different data elements, as shown here:

```
float x = A[idx];
float y = B[idx];
// If you wrote float u = x * c; then float v = y * d;,
// the multiplies are independent.
float u = x * c;  // can start as soon as x is ready
float v = y * d;  // can start as soon as y is ready, overlapping with u
```

Here, you keep the execution units busy on every cycle by not letting one dependent instruction block the next independent instruction, etc. In this interleaved code, u = x*c and v = y*d are independent instructions and can be executed concurrently. The compiler will schedule them back-to-back such that one multiply can proceed while the other multiply is still completing. This is ILP in action.

*Using compiler hints* Compiler hints such as #pragma unroll on small loops can help the compiler explicitly create multiple arithmetic chains. If needed, you can adjust -maxrregcount to allow the compiler to use more registers per thread for the unrolled variables. By allowing more resources per thread, you are increasing ILP at the cost of fewer threads. This naturally trades off some occupancy for higher ILP.

In short, to hide latency and maximize throughput on Blackwell’s deep pipelines, you should combine high occupancy (e.g., enough warps/threads to hide stalls) with high ILP (e.g., multiple independent instructions per thread). Here, you make sure that even when one instruction is waiting (e.g., memory load or long-latency arithmetic), the warp can issue other instructions immediately. This will significantly increase the kernel’s achieved FLOPS.

> On large unrolls, watch instruction-fetch and issue stalls in Nsight Compute. Excessive unrolling can bloat code size—or saturate decode and dispatch—without further throughput gains.

### Profiling and Mitigating Register Pressure

However, be mindful of register pressure since ILP typically requires more registers to hold multiple independent values and results. If you unroll too much—or add too many parallel computations—you might increase register usage to the point that occupancy falls significantly. As always, finding the right balance is key.

The CUDA compiler does a good job of unrolling loops automatically. But if you push ILP too far by manually unrolling too many iterations using #pragma unroll, for instance, you may notice that occupancy drops.

In this case, Nsight Compute will show “Limited by Registers” within its Occupancy analysis. This is usually a clear signal that you’ve gone too far. If you see this, you should dial back the unrolling, reduce the number of independent accumulator variables, or relax your launch bounds. Specifically, you can reduce the required MinBlocksPerSM or increase MaxThreadsPerBlock so that each block can use more registers without spilling. In other words, allow a bit lower occupancy to avoid severe spilling. This way the register pressure eases and occupancy is recovered.

From a profiling perspective, if ILP is helping, you should see the “Stall: Exec Dependency” percentage drop since warps are spending less time waiting on prior instructions. And you should see higher instructions per cycle.

Nsight Compute reports an *Issue Efficiency* or *IPC* metric. This should rise as ILP increases. For example, you should see the issue efficiency or instructions-per-cycle metric rise as you increase ILP. For example, if you see the IPC metric increase from 1.0 to 1.8 per warp scheduler when you unroll, this indicates that the warp is now issuing close to two instructions per cycle on average instead of just one. However, the actual values depend on the instruction mix and the architecture scheduling constraints.

You might also see that you can achieve good performance at lower occupancy than before. The ideal scenario is that you have enough ILP to keep each warp busy—and enough warps to keep the SM busy. Between those two, you’re covering latency both within and across warps.

At this point, we’ve tackled warp-level parallelism and instruction-level parallelism. These are two ways to keep the GPU’s computational units busy. Next, we turn to measuring and improving arithmetic intensity. This is about maximizing the useful work done per memory access.

## Key Takeaways

In this chapter, you saw how to uncover and eliminate GPU kernel bottlenecks by moving work from slow global memory into faster on-chip resources and compute units. By following a cycle of profiling, diagnosing, optimizing, and reprofiling, you can transform kernels from underutilized or memory‐bound into compute‐saturated, high‐throughput routines. These techniques will help utilize the full power of your GPUs:

*Profiling with Nsight and* torch.profiler Start with Nsight Systems to visualize end‐to‐end timelines and reveal CPU‐GPU gaps, kernel overlaps, and NVTX spans. Then drill into individual kernels with Nsight Compute’s stall metrics, roofline analysis, and occupancy reports. In PyTorch code, you can insert profiler instrumentation (using the torch.profiler API, built on the Kineto library) to map Python-level operations directly to GPU activities. Remember that the PyTorch profiler’s with_flops estimates are formula-based on a limited set of operators. It does not read GPU hardware counters. For hardware counters, use Nsight Compute on the kernels of interest.

*Tuning occupancy to hide latency* Adequate warps per SM are essential to cover memory and instruction latencies. You can reduce per‐thread register/shared‐memory usage through code refactoring or compiler hints. You should also choose block sizes (e.g., 128–256 threads) that fit the SM resources. Doing these will increase your kernel’s achieved occupancy to an optimal range—around 50%–75%—and stop underutilization (e.g., ~32-48 warps on Blackwell).

*Minimizing warp divergence for SIMT efficiency* Because warps execute in lockstep, any branch divergence means masked lanes sit idle. Restructure kernels so that all threads in a warp follow the same path. Use predicated operations like ternary math and torch.maximum—or employ warp-vote intrinsics to compact active lanes. This will boost warp execution efficiency and reduce serialized execution.

*Recognizing performance regimes* Every kernel falls into one of four buckets: underutilized, latency bound, memory bound, or compute bound. This is based on FLOPS, bandwidth usage, and stall reasons. Understanding which regime applies will help direct the exact optimization strategy. For instance, a high Long Scoreboard stall means the workload is latency bound.

*Exposing ILP* By unrolling loops, breaking dependency chains, and prefetching data, you expose instruction-level parallelism so that each thread can issue multiple independent operations per cycle. This lets two- or four-way ILP reduce the warps needed to fill an SM in half—reducing execution-dependency stalls, raising IPC, and decreasing the “Stall: Exec Dependency” metric. Always profile different ILP depths alongside thread counts to find your kernel’s sweet spot.

*Iterative optimization and validation* After each change, including occupancy tweaks, ILP restructuring, tiling, or precision scaling, you should reprofile to confirm reduced memory stalls, fewer execution dependencies, and higher achieved occupancy or FLOPS. Always compare results to a FP32 baseline using asserts or numeric checks to guard against unacceptable accuracy regressions when lowering precision.

## Conclusion

You saw how effectively tuning GPU performance requires an iterative workflow. First, measure with profilers. Then identify your primary bottlenecks (compute, memory, interconnects, etc.). Last, apply the right optimizations and repeat. While each new GPU architecture brings improvements in compute and memory bandwidth, they also add complexity. You must constantly stay on top of these innovations in order to sustain peak performance.

We covered how to tune occupancy, avoid warp divergence, and increase ILP. You also saw how to use profilers to correlate CPU, kernel, and NVLink traffic across GPUs. Then we went into Nsight Compute for per-kernel details.

In the next chapters, we’ll extend this foundation by focusing on kernel-level efficiency to increase arithmetic intensity. To do this you will fully utilize the GPUs hardware-optimized resources such as Tensor Cores for compute, TMEM to service the Tensor Cores, and TMA for data transfers. TMA remains the preferred method for bulk copies from global to shared on modern GPUs that support it. Let’s continue pushing toward the hardware’s peak performance limits!