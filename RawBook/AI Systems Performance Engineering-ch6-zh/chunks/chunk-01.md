# Chapter 6. GPU Architecture, CUDA Programming, and Maximizing Occupancy

In this chapter, we’ll start by reviewing the single instruction, multiple-threads (SIMT) execution model and how warps, thread blocks, and grids map your GPU-based algorithms onto streaming multiprocessors (SMs).

We’ll review the SIMT execution model on modern NVIDIA GPUs, including how warps, thread blocks, and grids map to SMs. We’ll then dive into CUDA programming patterns, discuss the on-chip memory hierarchy (register file, shared/L1, L2, HBM3e), and demonstrate the GPUs asynchronous data transfer capabilities, including the Tensor Memory Accelerator (TMA) and the Tensor Memory (TMEM) that serves as the accumulator for Tensor Core operations.

We’ll also introduce roofline analysis to identify compute-bound versus memory-bound kernels. This will provide the fundamentals to push modern GPU systems toward their theoretical peak throughput ceilings.

## Understanding GPU Architecture

Unlike CPUs, which optimize for low-latency single-thread performance, GPUs are throughput‐optimized processors built to run thousands of threads in parallel. A simple CUDA programming flow between the CPU and GPU is shown in Figure 6-1.

![Figure 6-1. Simple CUDA programming flow](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-1.png)

Initially, the host loads data into CPU memory. It then copies the data from the CPU to the GPU memory. After calling the GPU kernel with the data in GPU memory, the CPU copies the results back from GPU memory to CPU memory. Now the results live back on the CPU for further processing.

GPUs rely on massive parallelism to hide data-transfer latency such as the CPU-GPU data transfer described in Figure 6-1. Each GPU comprises many SMs, which are roughly analogous to CPU cores but streamlined for parallelism. Each SM can track up to 64 warps (32‐thread groups) on Blackwell.

Each GPU includes many SMs—similar to CPU cores but optimized for throughput. On modern GPUs, each SM tracks up to 64 warps (2,048 threads) concurrently. Blackwell GPUs feature 64K 32-bit registers per SM (256 KB total) and a combined 256 KB L1 cache/shared memory per SM. Up to 228 KB (227 KB usable) of that SRAM can be configured as user-managed shared memory per SM. Any single thread block can request up to 227 KB of dynamic shared memory (1 KB is of the 228 KB is reserved by CUDA). These help the SMs support the GPU’s high amount of thread-level parallelism.

Within a Blackwell SM, multiple warp schedulers issue instructions to the available pipelines; four independent warp schedulers allow up to four warps to issue instructions to the available pipelines on every cycle. Furthermore, each scheduler supports dual-issue capable of issuing two independent instructions (e.g., one arithmetic and one memory operation) per warp. Note that the dual-issue must come from the same warp—and not across warps.

In the best case, one warp from each scheduler can issue an instruction concurrently each cycle, allowing four warps to execute in parallel per cycle. This further boosts throughput when instruction mixing is utilized, as shown in Figure 6-2.

![Figure 6-2. Blackwell SMs contain four independent warp schedulers, each capable of issuing one warp instruction per cycle with dual-issue of one math and one memory operation per scheduler](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-2.png)

Here, each SM is subdivided into four independent scheduling partitions—each with its own warp scheduler and dispatch logic. You can think of the SM as four “mini-SMs” sharing on-chip resources. This lets the hardware pick ready warps and issue instructions from up to four different warps each clock cycle.

Within each of the four “mini-SM” partitions, the scheduler can issue two instructions per cycle from the same warp: one arithmetic instruction (e.g., INT32, FP32, or Tensor Core) and one memory instruction (a load or store). This is why the scheduler is called *dual-issue*. Table 6-1 summarizes these numbers.

*Table 6-1. Key SM scheduler and instruction-issue limits (per clock cycle)*

| Metric | Value |
| --- | --- |
| Number of schedulers | Four |
| Maximum warps issued | Four (one per scheduler) |
| Maximum math operations | Four (one per scheduler’s arithmetic issue) |
| Maximum memory operations | Four (one per scheduler’s load/store issue) |

> Note: The numeric values in all metrics tables are illustrative to explain the concepts. For actual benchmark results on different GPU architectures, see the GitHub repository.

So in the best case you could dual-issue four math and four memory instructions across four warps every cycle. This would maximize both compute and memory throughput simultaneously. These numbers are a result of the SM’s four-way partitioning—as well as its ability to pick one warp per partition and issue two orthogonal instructions each cycle.

The Special Function Unit (SFU) sits alongside the INT32, FP32, and Tensor Core pipelines. They handle transcendental operations (e.g., sine, cosine, reciprocal, square root). However, they are not part of the dual-issue math and memory pair. SFUs use a dedicated SFU pipeline that runs independently of the main INT32/FP32 and load/store (LD/ST) pipelines.

Because SFUs occupy a separate pipeline and can execute in parallel when needed, the SM can continue issuing math and memory instructions without waiting for the slower functions to complete. This separation increases instruction‐level parallelism and overall throughput even further for mixed‐operation kernels. They keep complex math operations from stalling the core compute and memory pipelines.

Because there are four schedulers—and each can typically issue one warp instruction per cycle—up to four warps can make forward progress each cycle when there is sufficient independent work and issue-pairing. For instance, the memory operations can flow through the SM’s combined 16 load/store (LD/ST) pipelines (four LD/ST pipelines per scheduler). These will read or write data to L1/shared memory, L2 cache, or global memory (covered in an upcoming section).

> Exact LD/ST pipeline counts and pairings are not guaranteed. Rely on profiling counters to determine whether your kernel is limited by memory issue or compute issue. And consult the NVIDIA documentation for specifics of your architecture. The Blackwell tuning guide is a good place to start.

In short, GPUs excel at data-parallel workloads, including large matrix multiplies, convolutions, and other operations where the same instruction applies to many elements. Developers write kernels directly in CUDA C++ or indirectly through high-level frameworks like PyTorch and domain-specific, Python-based GPU languages like OpenAI’s Triton.

Before diving into kernel development and memory-access optimizations, let’s review the CUDA thread hierarchy and key terminology that underpins all of these practices.

### Threads, Warps, Blocks, and Grids

CUDA structures parallel work into a three-level hierarchy—threads, thread blocks (aka *cooperative thread arrays* [CTAs]), and grids—to balance programmability with massive throughput. At the lowest level, each thread executes your kernel code. You group threads into thread blocks of up to 1,024 threads each on modern GPUs. Thread blocks form a grid when you launch the kernel, as seen in Figure 6-3.

![Figure 6-3. Threads, thread blocks (aka CTAs), and grids](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-3.png)

By sizing your grid appropriately, you can scale to millions of threads without changing your kernel logic. CUDA’s runtime (and frameworks like PyTorch) handle scheduling and distribution across all SMs. Figure 6-4 shows another view of the thread hierarchy, including the CPU-based host, which invokes a CUDA kernel running on the GPU device.

![Figure 6-4. View of thread hierarchy, including the CPU-based host, which launches a kernel running on the GPU device](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-4.png)

Traditionally, threads from different thread blocks could not work with one another directly. However, modern GPU architectures and CUDA versions support thread block clusters. Threadblock clusters are groups of thread blocks that can communicate with one another across SMs.

Specifically, within a thread block cluster, threads in different thread blocks can access one another’s shared memory and use hardware-supported, cluster-scoped barriers. These allow for much larger compute operations, including matrix multiplies, which are very common in today’s massive LLM workloads. Thread block clusters share a distributed shared-memory (DSMEM) address space between SMs that participate in the thread block cluster, as shown in Figure 6-5.

![Figure 6-5. Hardware-supported DSMEM used in thread block clusters containing multiple thread blocks](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-5.png)

DSMEM is a hardware feature that links the shared-memory banks of all SMs into a thread block cluster over a fast on-chip interconnect. With DSMEM, the SMs share a combined multi-SM distributed shared-memory pool. This unification allows threads in different blocks to read, write, and atomically update one another’s shared buffers at on-chip speeds—and without using global memory bandwidth.

> We’ll cover advanced topics like thread block clusters and DSMEM in Chapter 10. These are an extremely important addition to modern GPU processing—and very important for an AI systems performance engineer to understand. For this chapter, our focus remains on intrablock shared-memory optimizations.

Within each thread block, threads share data using low-latency on-chip shared memory and synchronize with __syncthreads(). Because each barrier incurs overhead, you should minimize synchronization points, as shown in Figure 6-6.

![Figure 6-6. Synchronizing all threads within a thread block between two sections of code](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-6.png)

The goal is to minimize synchronization points. However, the GPU hardware will attempt to hide long-latency events such as global-memory loads, cache fills, and pipeline stalls by rapidly switching among warps.

Thread blocks are subdivided into warps of 32 threads that execute in lockstep under the SIMT model using a warp scheduler. This is shown in Figure 6-7.

![Figure 6-7. Warps (32 threads) advance as a whole with instructions managed by the warp scheduler](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-7.png)

Keeping more warps in flight is known as *high occupancy* on the SM. When your CUDA code allows high occupancy, it means that when one warp stalls, another is ready to run. This keeps the GPU’s compute units busy.

However, high occupancy must be balanced against per-thread resource limits, such as registers and shared memory. Spilling registers to slower memory can create new stalls. Profiling occupancy alongside register and shared-memory usage helps you choose a block size that maximizes throughput without triggering resource contention.

> We will cover occupancy tuning in Chapter 8, but it’s a key concept to understand in the context of SMs, warps, threads, etc.

Thread blocks execute independently and in no guaranteed order. This allows the GPU scheduler to dispatch them across all SMs and fully exploit hardware parallelism. This grid–block–warp hierarchy guarantees that your CUDA kernels will run unmodified on future GPU architectures with more SMs and threads.

Throughput also hinges on warp execution efficiency. Threads in a warp must follow the same control-flow path and perform coalesced memory accesses. If some threads diverge such that one branch takes the if path and others take the else path, the warp serializes execution, processing each branch path sequentially. This is called *warp divergence*, and it’s shown in Figure 6-8.

![Figure 6-8. SIMT warp divergence (left) versus uniformity (right)](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-8.png)

By masking inactive lanes and running extra passes to cover each branch, warp divergence multiplies the overall execution time by the number of branches. We’ll dive deeper into warp divergence in Chapter 8—as well as ways to detect, profile, and mitigate it.

> Divergence is an issue only for threads within a single warp. Different warps can follow different branches with no performance penalty.

### Choosing Threads-per-Block and Blocks-per-Grid Sizes

A critical aspect of GPU performance is choosing a thread block size that aligns with the hardware’s 32-thread warp size. As such, you typically pick thread block sizes that are exact multiples of 32. For example, a 256-thread block (8 warps = 256 ÷ 32) fully occupies each warp, whereas a 33-thread block will require two warp slots and use only 1/32 of the second warp’s lanes. This wastes parallelism opportunities since every warp occupies a scheduler slot whether it’s actively running 32 threads or just 1 thread.

Additionally, different GPU generations have different hardware limits, including maximum threads per SM and the number of registers per SM. This naturally limits the size of our blocks if we want to maintain good performance. For instance, too large a block might require too many registers, which will cause *register spilling* and decrease the kernel’s performance.

A large block might also require too much shared memory, which is finite in GPU hardware. Specifically, Blackwell provides only 228 KB (227 KB usable) per SM of shared memory addressable by all resident thread blocks running on the SM.

These hardware limits affect how many blocks/warps can be active on an SM at once. This is a measurement of occupancy, as we introduced earlier. Smaller blocks might enable higher occupancy if they allow more concurrent warps to run concurrently on the SM.

It’s important to understand the relative scale and hardware thread limits for your GPU generation, including number of threads, thread blocks, warps, and SMs. Figure 6-9 shows the relative scale of these resources, including their limits.

![Figure 6-9. Relative scale and hardware limits for threads on a Blackwell GPU](AI%20Systems%20Performance%20Engineering-ch6_images/figure-6-9.png)

Table 6-2 summarizes these GPU limits for the Blackwell B200 GPU. The rest of the limits are available on NVIDIA’s website. (Other GPU generations will have different limits, so be sure to check the exact specifications for your system.)

*Table 6-2. Thread-level and block-level limits (Blackwell B200)*

| Resource | Hardware limit | Notes |
| --- | --- | --- |
| Warp size | 32 threads | The fundamental SIMT execution unit is 32 threads (a warp). Always use a multiple of 32 to avoid waste. |
| Maximum threads per thread block | 1,024 threads | blockDim.x * blockDim.y * blockDim.z ≤ 1024. |
| Maximum warps per thread block | 32 warps | (1,024 threads ÷ 32 threads-per-warp) = 32 warps max per block. |

We already discussed the warp size limit of 32 threads, which encourages us to choose block dimensions that are multiples of 32 threads to create “full warps” and avoid underutilized warps. Note that each block can have up to 1,024 threads and, correspondingly, a block can contain only 32 warps. These limits affect your occupancy since, once a block is scheduled, each SM can host a limited number of warps and blocks simultaneously.

Additionally, there are per-SM limits, or *SM-resident limits* as they are commonly called, for the different GPU generations. These SM-resident Blackwell limits are summarized in Table 6-3.

*Table 6-3. SM-resident resource limits (Blackwell B200)*

| Resource (per SM) | Hardware limit | Notes |
| --- | --- | --- |
| Maximum resident warps per SM | 64 warps | Hardware can keep up to 64 warps in flight (64 × 32 threads = 2,048 threads). Note: This limit has held for many generations and remains true for Blackwell. |
| Maximum resident threads per SM | 2,048 threads | Equals 64 warps × 32 threads/warp. If each block uses 1,024 threads, then at most 2 such blocks (64 warps) can reside on one SM concurrently. Using smaller blocks (e.g., 256 threads) allows more blocks to reside on the SM (up to 8 blocks × 256 = 2,048 threads), which can increase occupancy and help hide latency—though too many tiny blocks can add scheduling overhead. |
| Maximum active blocks per SM | 32 blocks | At most, 32 thread blocks can be simultaneously resident on one SM (if blocks are smaller, more can fit up to this limit). |

Here, we see that the maximum number of concurrent warps per SM on Blackwell is 64. This hasn’t changed for recent GPU generations, so occupancy considerations carry over. Maximum active blocks on an SM is 32, and, correspondingly, maximum resident threads per SM is 2,048 threads. CUDA grids also have maximum dimensions, as shown in Table 6-4.

*Table 6-4. CUDA grid limits*

| Grid dimension | Limit | Notes |
| --- | --- | --- |
| Maximum blocks in X, Y, or Z | X: 2,147,483,647 blocks; Y: 65,535 blocks; Z: 65,535 blocks | A 3D grid can be as large as 2,147,483,647 × 65,535 × 65,535 blocks. |
| Maximum concurrent grids (kernels) | 128 grids | Up to 128 kernels can execute concurrently on one device (i.e., 128 grids resident at once). |

While it’s good to know the theoretical grid limits, you will typically be bound by the thread/block/per-SM limits shown previously. If you ever need more than 65,535 blocks in one dimension, you can launch a 2D or 3D grid to split your work across multiple kernel launches (multilaunch). We show an example of this in a later section. In practice, it’s rare to hit the grid size limit before hitting other resource limits.

### CUDA GPU Backward and Forward Compatibility Model

One of CUDA’s core strengths is its forward and backward compatibility model. Kernels compiled today will generally run unmodified on future GPU generations—as long as you include PTX in your binary for forward compatibility. If you ship only SASS for a single architecture (e.g., sm_90 for Hopper or sm_100 for Blackwell) without PTX, that binary will not forward-run on newer architectures. Family-specific targets such as sm_100f or compute_100f restrict portability to devices in the same feature family. It’s best to ship a fatbin that includes both generic cubin/PTX and family-specific cubins needed (e.g., optimizations, etc.).

You can verify compatibility by forcing PTX JIT compilation at load time by setting CUDA_FORCE_PTX_JIT=1 to JIT-compile the PTX and cache the result. If your binary lacks PTX, the kernel launch will fail. This forces you to rebuild with PTX support. This compatibility model is fundamental to the large CUDA ecosystem. It lets you target both legacy and cutting-edge hardware from a single codebase.

> To truly maintain both backward and forward compatibility across current and future GPU generations, you should compile with generic targets—or explicitly include the PTX. When you need specific optimizations from newer hardware features, you can use generation-specific targets. When doing this, be sure to provide fallback paths for other architectures.

## CUDA Programming Refresher

In CUDA C++, you define parallel work by writing kernels. These are special functions annotated with __global__ that execute on the GPU device. When you invoke a kernel from the CPU (host) code, you use the <<< >>> “chevron” syntax to specify how many threads should run—and how they’re organized—using two configuration parameters: blocksPerGrid for the number of thread blocks and threadsPerBlock for the number of threads within each block.

Here is a simple example that demonstrates the key components of a CUDA kernel and kernel launch. This kernel simply doubles every element in the input array in place so no additional memory is created—just the input array. Behind the scenes, CUDA compiles the __global__ function into GPU device code that can be executed by thousands or millions of lightweight threads in parallel:

```
//-------------------------------------------------------
// Kernel: myKernel running on the device (GPU)
//   - input : device pointer to float array of length N
//   - N   : total number of elements in the input
//-------------------------------------------------------

__global__ void myKernel(float* input, int N) {
    // Compute a unique global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only process valid elements
    if (idx < N) {
        input[idx] *= 2.0f;
    }
}

// This code runs on the host (CPU)
int main() {
    // 1) Problem size: one million floats
    const int N = 1'000'000;
    float *h_input = nullptr;
    float *d_input = nullptr;

    // 1) Allocate input float array of size N on host
    cudaMallocHost(&h_input, N * sizeof(float));

    // 2) Initialize host data (for example, all ones)
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;
    }

    // 3) Allocate device memory for input on the device
    cudaMalloc(&d_input, N * sizeof(float));

    // 4) Copy data from the host to the device
    cudaMemcpy(d_input, h_input, N * sizeof(float),
      cudaMemcpyHostToDevice);

    // 5) Choose kernel launch parameters

    // Number of threads per block (multiple of 32)
    const int threadsPerBlock = 256;

    // Number of blocks per grid (3,907 for N = 1000000)
    const int blocksPerGrid = (N + threadsPerBlock - 1) /
      threadsPerBlock;

    // 6) Launch myKernel across blocksPerGrid blocks
    // Each block has threadsPerBlock number of threads
    // Pass a reference to the d_input device array
    myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input,
      N);
    // 7) Wait for the kernel to finish running on device
    cudaDeviceSynchronize();

    // 8) When finished, copy the results
    //    (stored in d_input) from the device back to
    //     host (stored in h_input)

    cudaMemcpy(h_input, d_input, N * sizeof(float),
               cudaMemcpyDeviceToHost);

    // Cleanup: Free memory on the device and host
    cudaFree(d_input);
    cudaFreeHost(h_input);

    // return 0 for success!
    return 0;
```

> This code is not fully optimized. We will optimize performance as we continue through the book. But this gives you a simple, complete template to start building your own CUDA kernels.

Here, we are passing kernel input arguments, d_input and N, which are accessible inside the kernel function for processing. The processing is shared, in parallel, across many threads. This is by design.

The full data flow is as follows:

1. Allocate memory on the host (h_input).

2. Copy data from the host (h_input) to device (d_input) using cudaMemcpy with cudaMemcpyHostToDevice.

3. Run the kernel on the device with d_input.

4. Synchronize to ensure the kernel has finished executing on the device.

5. Transfer the results (d_input) from device to host (h_input) using cudaMemcpy with cudaMemcpyDeviceToHost.

6. Clean up memory on the device and host with cudaFree and cudaFreeHost.

You can pass additional, advanced, CUDA-specific parameters to your kernel at launch time with <<< >>>, including shared-memory size (and many others), but the two core launch parameters, blocksPerGrid and threadsPerBlock, are the foundation of any CUDA kernel invocation. In the next section, we will discuss how to best choose these launch parameter values.

And you might be wondering why we have to pass N, the size of the input array. This seems redundant since the kernel should be able to inspect the size of the array. However, this is the core difference between a GPU CUDA kernel function and a typical CPU function: a CUDA kernel function is designed to work inside of a single thread, alongside thousands of other threads, on a partition of the input data. As such, N defines the size of the partition that this particular kernel will process.

Combined with the built-in kernel variables blockDim (1 in this case since we’re passing a one-dimensional input array), blockIdx, and threadIdx, the kernel calculates the specific idx into the input array. This unique idx lets the kernel process every element of the input array cleanly and uniquely, in parallel, across many threads running across many different SMs simultaneously.

Note the bounds check if (idx < N). This is needed to avoid out-of-range access (bounds check) since N may not be an exact multiple of the block size. For instance, consider a scenario in which the input array is size 63, so N = 63. The warp scheduler will likely assign two warps (32 threads each) to process the 63 elements in the input array.

The first warp will run 32 instances of the kernel simultaneously to process elements 0–31 and never exceed N = 63. That’s straightforward. The second warp, running in parallel with the first warp, will expect to process elements 32–64. However, it will stop when it reaches N = 63.

Without the if (idx < N) bounds check, the second warp will try to process idx = 64, and it will throw an illegal memory access error (e.g., cudaErrorIllegalAddress). The bounds check ensures that every thread either works on a valid input element or exits immediately if its idx is out of range.

CUDA kernels execute asynchronously on the device without per‐thread exceptions; instead, any illegal operation (out-of-bounds access, misaligned access, etc.) sets a global fault flag for the entire launch. The host driver only checks that flag when you next call a synchronization or another CUDA API function, so errors surface lazily (e.g., as cudaErrorIllegalAddress or a generic launch failure).

This design keeps the GPU’s pipelines and interconnects fully occupied but requires you to explicitly synchronize and poll for errors on the host—usually with cudaGetLastError() and cudaDeviceSynchronize() immediately after kernel launches. This way, you catch faults as soon as they occur.

You will see a bounds check in a lot of CUDA kernels. If you don’t see it, you should understand why it’s not there. It’s likely there in some fashion—or the CUDA kernel developer can somehow guarantee the illegal memory access error will never happen.

And finally, we get to the actual kernel logic. After computing its unique index idx into the input array, this kernel (running separately on thousands of threads in parallel across many SMs) multiplies the value at index idx in the input array by 2. It then updates the value (in place) in the input array. In this specific kernel, no additional memory is needed except the temporary idx variable of type int.

### Configuring Launch Parameters: Blocks per Grid and Threads per Block

As discussed earlier, using a block size that’s a multiple of the warp size (32) is critical. A threadsPerBlock size of 256 (eight warps) is a common starting point to balance occupancy and resource usage. This will help us avoid partially filled warps during kernel execution, hide latency, and balance SMs and other hardware resources:

*Multiple of 32 threads*

Choosing a block size that is a multiple of 32 threads helps to avoid empty warp slots. Otherwise those underfilled warps occupy scarce scheduler resources— without contributing useful work.

*Latency hiding*

Hundreds of threads per SM are needed to hide DRAM and instruction‐latency stalls. If you launch, say, eight blocks of 256 threads on an SM with 2,048 threads of capacity, you can keep the pipeline busy without oversubscribing.

*Occupancy*

With 256 threadsPerBlock, for example, you need only eight warps per block. This tends to give good occupancy without running out of registers or shared memory per block.

> For modern GPUs like Blackwell, consider 256–512 threads per block to maximize occupancy while respecting register and shared-memory limits.

*Resource-balanced*

256 is small enough that you rarely exceed the 1,024-thread-per-block limit. And it’s large enough that you’re not leaving too many warps idle when threads in other warps stall.

Starting with threadsPerBlock=256, you can tune up or down (128, 512, etc.) based on your kernel’s register and shared-memory requirements—as well as occupancy characteristics.

For blocksPerGrid, you can base this on the number of N input elements and the value of threadsPerBlock. For instance, the blocksPerGrid is commonly set to (N + threadsPerBlock - 1) / threadsPerBlock to round up so that you cover all elements if N is not an exact multiple of threadsPerBlock. This is a common choice that guarantees every input element is covered by a thread. Here is the code that shows the calculation:

```
//-------------------------------------------------------
// Kernel: myKernel running on the device (GPU)
//   - input : device pointer to float array of length N
//   - N   : total number of elements in the input
//-------------------------------------------------------
__global__ void myKernel(float* input, int N) {
    // Compute a unique global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Only process valid elements
    if (idx < N) {
        input[idx] *= 2.0f;
    }
}

// This code runs on the host (CPU)
int main() {
    // 1) Problem size: one million floats
    const int N = 1'000'000;

    float* h_input = nullptr;
    cudaMallocHost(&h_input, N * sizeof(float));

    // Initialize host data (for example, all ones)
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f;
    }

    // Allocate device memory for input on the device (d_)
    float* d_input = nullptr;
    cudaMalloc(&d_input, N * sizeof(float));

    // Copy data from the host to the device using cudaMemcpyHostToDevice
    cudaMemcpy(d_input, h_input, N * sizeof(float),
      cudaMemcpyHostToDevice);

    // 2) Tune launch parameters
    const int threadsPerBlock = 256; // multiple of 32
    const int blocksPerGrid = (N + threadsPerBlock - 1) /
      threadsPerBlock; // 3,907, in this case

    // Launch myKernel across blocksPerGrid number of blocks
    // Each block has threadsPerBlock number of threads
    // Pass a reference to the d_input device array
    myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, N);
    // Wait for the kernel to finish running on the device
    cudaDeviceSynchronize();

    // When finished, copy results (stored in d_input) from device to host
    // (stored in h_input) using cudaMemcpyDeviceToHost
    cudaMemcpy(h_input, d_input, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Cleanup: Free memory on the device and host
    cudaFree(d_input);
    cudaFreeHost(h_input);

    return 0; // return 0 for success!
```

This is the same kernel as previously but calculates the blocksPerGrid and threadsPerBlock dynamically based on the size of N. Note the familiar if (idx < N) bounds check. This ensures that any “extra” threads in the final block that fall outside of N will simply do nothing—and not cause an illegal memory address error. Next, let’s explore multidimensional inputs like 2D images and 3D volumes.

### 2D and 3D Kernel Inputs

When your input data naturally lives in two dimensions (e.g., images), you can launch a 2D grid of 2D blocks. For example, here’s a kernel that processes a two-dimensional 1,024 × 1,024 matrix using a 16 × 16 dimensional thread block for a total of 256 threads: