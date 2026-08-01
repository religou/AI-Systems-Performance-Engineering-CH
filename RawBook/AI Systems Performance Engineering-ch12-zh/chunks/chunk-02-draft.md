实践中，这会在你的持久化核函数或动态核函数内部嵌入一个“图启动”节点，或调用设备侧图 API。时机到来时，GPU 会在它自己拥有的流上启动整张图，如图 12-5 所示。

设备端发起的图启动让数据驱动的工作流完全保留在 GPU 上。计算判定条件的责任落在你的核函数上，而不是 CPU。因此，它可以直接派生下一张图，消除 CPU 往返，并进一步降低延迟。

由于图已经驻留在 GPU 上、无需 CPU-GPU 握手，设备端发起的启动将主机调度从关键路径上移除，可在主机受限的循环中降低端到端延迟。实践中，设备端发起的 CUDA graph 启动相较等价的主机侧图启动，启动延迟大约低 2×。而且即便图的规模或复杂度增长，其开销依然保持平坦。

![图 12-5. 由 CUDA Graph 启动的操作序列（核函数与数据传输的节点）及其依赖（边）](AI%20Systems%20Performance%20Engineering-ch12_images/figure-12-5.png)

设备端启动的图延迟不受图中节点数量或并行分支数量的影响。这与主机端启动的图延迟形成对比——后者会因 CPU 调度开销而随图变大而增加。

此外，设备端启动在图宽度上扩展良好。随着并行节点增多，主机侧启动会承受额外的同步开销，但设备端启动延迟几乎保持平坦。

调试设备端启动的图可能有些棘手，但像 Nsight Systems 这样的工具会在 GPU 时间线上将子图显示为独立的工作流。建议在父核函数中于 cudaGraphLaunch 调用前后使用 NVTX 标记，以标注设备端启动发生的位置。这有助于验证图相对于父线程是否如预期般运行。

在你的设备端代码中，你用简单的 API cudaGraphLaunch(graphExec, stream) 来启动图。运行时使用特殊的、保留的 cudaStream_t 值来区分下列受支持的启动模式：“发射即忘”（cudaStreamGraphFireAndForget）、“尾部”（cudaStreamGraphTailLaunch）和“同级”（cudaStreamGraphFireAndForgetAsSibling）。这些模式会在 CUDA 流中自动强制执行正确的顺序，无需任何主机干预。

在发射即忘（fire-and-forget）启动中，子图立即开始执行，并与发起它的父核函数并发运行。父核函数不会等待子图结束——这很像启动一个独立的工作线程。发射即忘启动适用于在核函数内部派生异步任务。

> 一张图在其执行过程中最多可以有 120 个发射即忘图。

相比之下，设备端发起的图尾部启动（tail launch）会将图的执行推迟到发起它的核函数到达同步点或执行完成之后。这实际上把图排入队列，作为当前核函数的延续在其之后运行，如图 12-6 所示。

![图 12-6. 由某张图入队的尾部启动会一次一个、按入队顺序执行](AI%20Systems%20Performance%20Engineering-ch12_images/figure-12-6.png)

在实现 GPU 驻留的工作调度器时，尾部启动尤其强大。一个持久化的“调度器”核函数可以尾部启动一张图，然后在该图完成后重新启动自身。这一技术实际上在 GPU 上创建了一个循环，无需主机重新调用。为了重新启动自身，核函数调用 cudaGetCurrentGraphExec() 获取指向它自身正在执行的图的句柄。随后它使用 cudaGraphLaunch(..., cudaStreamGraphTailLaunch) 启动该图，把自己再次入队。

此外，尾部图可以执行额外的尾部启动。在这种情况下，新的尾部启动会先于前一张图的尾部启动执行，如图 12-7 所示。

![图 12-7. 由多张图入队的尾部启动](AI%20Systems%20Performance%20Engineering-ch12_images/figure-12-7.png)

> 你在一张 CUDA Graph 中最多可以有 255 个待处理的尾部启动排队。然而，就自尾部启动（例如一张图把自身入队以重新启动）而言，同一时刻只能有一个待处理的自尾部启动。

同级启动（sibling launch）是发射即忘的一种变体，其中被启动的图作为父图的对等体（peer）执行——而不是作为子图。此外，同级图运行在父图的流环境中。这意味着它立即且独立地运行，但不会延迟父图的任何尾部启动，如图 12-8 所示。

![图 12-8. 在父图流环境中的同级图启动](AI%20Systems%20Performance%20Engineering-ch12_images/figure-12-8.png)

对于这一模式，你可以使用 cudaGraphLaunch(graphExec, cudaStreamGraphFireAndForgetAsSibling) 以“同级模式”启动。这会将图作为当前图执行环境的同级提交。

在使用设备端发起的 CUDA Graphs 时，你需要谨慎管理依赖关系。例如，如果被启动的核函数必须消费来自该图的结果，那么尾部启动是合适的，因为父核函数会暂停，直到图的工作完成。相反，如果被启动的图更像是一个附带任务，那么发射即忘模式允许父核函数无需等待即可继续。

实践中，设备端发起的 CUDA Graphs 开启了新的模式。例如，设想一个 GPU 压缩流水线，其中一个核函数必须根据数据内容在不同的压缩算法之间做出选择。GPU 核函数无需结束并告诉 CPU 去启动所选的压缩核函数，而是可以直接启动一张预先录制的图，对应于比如“LZ 压缩”或“Huffman 压缩”。

在这个压缩示例中，GPU 从不空闲等待 CPU 做决定。让我们再看一个有用的模式，它把原子队列/计数器与设备端发起的、尾部启动的 CUDA Graphs 结合起来，用于持久化核函数内部的 LLM 推理调度。

### 用原子队列与设备端发起的 CUDA Graphs 实现核内持久化调度

我们可以把前文的原子计数器工作队列与设备端发起的图尾部启动结合起来。考虑一个 LLM 推理循环的用例，它使用一张 CUDA Graph 来执行解码——通过捕获一张包含 transformer 块前向传递（注意力 + 前馈）的图。

一个轻量、持久化的调度器核函数可以使用 atomicAdd(&queueHead,1) 来认领下一个工作项。随后它尾部启动预先捕获的解码 CUDA Graph 以计算输出，并立即循环回去处理队列中的下一项。

每当一张 CUDA Graph 完成时，核内调度器循环使用 atomicAdd(&queueHead,1) 抓取下一个索引，并尾部启动另一张解码图。这实际上创建了一个完全 GPU 驻留的调度器，既做决策又执行任务，全程不触及 CPU。

通过链接这些尾部启动，每个 token 都在设备上从头到尾处理，CPU 开销几乎为零。而且由于 CPU 从不重新进入关键路径，SM 保持完全利用，每 token 延迟下降，你还可以即时适配不同的序列长度和批大小。为此，你只需更新图参数或在预先录制的图之间切换。

### 条件图节点

在传统的 CUDA Graph 中，每个节点及其依赖在捕获时就已固定，从而迫使任何决策逻辑退回主机端。条件图节点（conditional graph node）打破了这种僵硬——它将分支决策推迟给 GPU 自身，依据的是与节点关联的一个小小的“条件句柄”。

随着图的执行，GPU 会评估该句柄，并选择性地运行其某个主体子图（body subgraph）之一（或循环遍历它），全程从不将控制权交还给 CPU。具体来说，条件图节点让你把控制流（IF、IF/ELSE、WHILE、SWITCH）直接嵌入 CUDA Graphs，在 GPU 设备上运行。条件图节点消除了主机往返，并可在现代 GPU 上带来显著的性能提升。

本质上，条件图节点让你基于设备核函数中计算出的值来控制图的执行——全程不涉及 CPU。这一能力使复杂的分支工作流可以实现为单次、可重复的图启动。CUDA Graphs 支持多种类型的条件节点，如图 12-9 所示：

_IF_ 当条件非零时，恰好执行其单主体图一次。

_IF/ELSE_ 通过指定两个主体图，当条件为真时选择其一，为假时选择另一。

_WHILE_ 只要条件保持非零，就反复执行其主体图，并在每次迭代后再次检查。

_SWITCH_ 持有 _N_ 个主体图，当条件等于 _i_ 时执行第 i 个；若条件 ≥ _N_，则完全跳过执行。

![图 12-9. 条件图节点的类型](AI%20Systems%20Performance%20Engineering-ch12_images/figure-12-9.png)

接下来是一个示例，展示如何创建并填充一个 IF 条件节点。注意 cudaGraphSetConditional 的使用，它写入控制 IF 节点的标志。在本例中，条件检查的是求和是否大于给定阈值。这样，如果数据满足给定条件（flag = 1u），就运行下一个子图。否则，如果条件不满足，条件节点不运行该子图：

```
#include <cuda_runtime.h>
#include <cstdio>
// Device kernel that computes and sets the condition handle
__global__ void setHandle(cudaGraphConditionalHandle handle,
                          int *data, size_t N) {
    // Example threshold
    constexpr int threshold   = 123456;
    // Test whether the sum of data exceeds a threshold
    //     using a custom reduce_sum() function
    //     (recommended to implement this with
    //     CUB's DeviceReduce::Sum routine.)
    unsigned int flag =
        (reduce_sum(data, N) > threshold) ? 1u : 0u;
    cudaGraphSetConditional(handle, flag);
}
// A simple body kernel that runs only if flag != 0
__global__ void bodyKernel() {
    printf("Conditional body executed on GPU!\n");
}
int main() {
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream,
        cudaStreamNonBlocking);
    // 1) Create the graph
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);
    // 2) Create a condition handle associated with graph
    cudaGraphConditionalHandle condHandle;
    cudaGraphConditionalHandleCreate(&condHandle, graph);
    // 3) Add the upstream kernel node to set the handle
    cudaGraphNode_t setNode;
    cudaKernelNodeParams setParams = {};
    setParams.func          = (void*)setHandle;
    setParams.gridDim       = dim3(1);
    setParams.blockDim      = dim3(32);
    // 4) Allocate input data
    constexpr size_t N              = 1 << 20;
    int*             d_data         = nullptr;
    cudaMalloc(&d_data, N * sizeof(int));

    void* setArgs[] = { &condHandle, &d_data, &N };
    setParams.kernelParams  = setArgs;
    cudaGraphAddKernelNode(&setNode, graph, nullptr, 0,
        &setParams);
    // 5) Add the IF conditional node
    cudaGraphNode_t condNode;
    cudaConditionalNodeParams ifParams = {};
    ifParams.handle = condHandle;
    ifParams.type   = cudaGraphCondTypeIf;
    // One-node body graph, in this case
    ifParams.size   = 1;
    cudaGraphAddConditionalNode(&condNode,
                                graph,
                                &setNode,
                                1,
                                &ifParams);
    // 6) Populate the body graph: one kernel that prints a message
    cudaGraph_t bodyGraph = ifParams.phGraphOut[0];
    cudaGraphNode_t bodyNode;
    cudaKernelNodeParams bodyParams = {};
    bodyParams.func         = (void*)bodyKernel;
    bodyParams.gridDim      = dim3(1);
    bodyParams.blockDim     = dim3(32);
    cudaGraphAddKernelNode(&bodyNode, bodyGraph, nullptr,
        0, &bodyParams);
    // 7) Instantiate, upload, and launch the graph on the device
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr,
    cudaGraphInstantiateFlagDeviceLaunch);
    cudaGraphUpload(graphExec, stream);
    cudaGraphLaunch(graphExec, stream);
    // 8) Wait for completion and clean up
    cudaStreamSynchronize(stream);
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    return 0;
}
```

这里，我们用 cudaGraphCreate 创建一张新的 CUDA Graph，用来容纳后续所有节点。随后我们用 cudaGraphConditionalHandleCreate 创建一个条件句柄。（这会将一个小整数值关联到图上，该值可在设备端设置。）

接着添加一个上游核函数 setHandle。这个核函数在单个线程上运行，以避免竞争条件。它随后调用 cudaGraphSetConditional 来写入控制 IF 节点的标志。

用 cudaGraphAddConditionalNode 添加 IF 条件节点——指定 cudaGraphCondTypeIf 和 size=1。这定义了我们计划支持多少个条件分支或迭代。

这里，我们分配一个空的子图（主体），当条件标志返回非零时执行它。主体图从 ifParams.phGraphOut[0] 取得，并通过添加 bodyKernel 来填充，后者在执行时只是打印一条消息。

图构建完成后，调用 cudaGraphInstantiate 生成一个可执行图对象。要从设备端代码启动图，你必须用 cudaGraphInstantiateFlagDeviceLaunch 标志实例化它，并在任何设备侧启动之前用 cudaGraphUpload 上传它。

在 CUDA 流上用 cudaGraphLaunch 启动会触发上游 set-handle 核函数、条件检查的执行，然后，如果标志已置位，触发主体核函数。而这一切都直接发生在 GPU 上，如图 12-10 所示。

![图 12-10. 若条件满足则进行额外处理（主体子图）](AI%20Systems%20Performance%20Engineering-ch12_images/figure-12-10.png)

随后我们用 cudaStreamSynchronize 同步流以等待完成。最后，我们通过销毁已实例化的图、图本身以及流来做清理。

为了尽量减少竞争条件，务必始终在单个线程中设置条件（例如 if (threadIdx.x == 0)）。并确保前置核函数刷新内存，使该值在条件节点执行前可见。

条件节点也可以嵌套。例如，一个 WHILE 节点的主体可以包含一个 IF 节点，如图 12-11 所示。这允许多层级决策逻辑而无需 CPU 跳转。

![图 12-11. 嵌套的条件图节点](AI%20Systems%20Performance%20Engineering-ch12_images/figure-12-11.png)

简而言之，你应当使用条件图节点将决策保留在 GPU 上、减少 CPU 开销，并直接在你的 CUDA Graph 中表达复杂的控制流。由于图的创建成本可以在多次迭代中摊薄，完全在设备端表示动态工作流可以带来显著的性能改善。

> 在本书撰写时，PyTorch 的 CUDA Graphs API 尚未提供直接用 Python 创建条件 CUDA graph 节点的方式。框架与工具中对条件图执行的支持仍在演进。在本书撰写时，如果你需要用 PyTorch 实现 if/while/switch 节点，此特性需要自定义的 C++ 集成。

## 动态并行

此前，我们看到了设备端发起的 CUDA Graph 启动如何以最小的 CPU 参与来捕获并重放固定的操作序列。但那个模型预期你的整个执行流程是提前已知的，而这并不总是可能。许多工作负载会在运行时根据输入数据、中间结果或问题复杂度改变形态。这正是动态并行（dynamic parallelism，DP）的用武之地。

DP 赋予你的 GPU 核函数为自身派生新工作的能力，而不必等待 CPU。CUDA Graphs 要求你提前知道整条流水线，而 DP 则让一个正在运行的“父”核函数检查它自己的输出，并即时决定接下来启动多少个“子”核函数。对于真正不规则的问题——层次化归约、自适应网格细化和图遍历——这是一个颠覆性的能力，因为后续任务的数量只有在你处理数据时才逐渐明朗。

设想一个推理流水线，它偶尔需要对某些输入进行一次特殊的“插件”模型评估。在 CPU 驱动的流程中，你会运行核函数 A，把它的结果拷回主机，决定是否启动核函数 B，然后发出该启动——在往返期间让 GPU 空闲。有了 DP，核函数 A 检查它的输出，如果条件成立，就直接在设备上启动核函数 B。整个决策与派发都发生在一个 GPU 驻留的工作负载内部，消除了空闲间隙并让 SM 保持忙碌。

在 LLM 的语境中，大多数 token 遵循标准的 transformer 路径，但有些需要一个辅助的注意力块。一个支持 DP 的 transformer 核函数可以在运行时检测出这些特殊 token，并仅对这些位置尾部启动额外的注意力核函数——无需主机干预，不浪费周期。NVIDIA 的库已经在自适应算法中为类似模式利用了 DP：随着数据在计算中流动，新的子任务动态涌现。

当你的性能剖析器时间线显示出像核函数 A → GPU 空闲间隙 → 核函数 B 这样的背靠背模式时，你就会知道 DP 适合你。这里，空闲时间对应的是 CPU 准备下一次启动。用设备侧启动替换那个间隙可以让每个 SM 保持占用，并大幅削减相依阶段之间的延迟。

当然，DP 的性能收益并非免费。每一次子启动都要使用 GPU 调度资源，并需要额外的栈空间。为避免“栈溢出”错误，你可能需要用 cudaDeviceSetLimit(cudaLimitStackSize, newSize) 提高运行时栈大小。如果你触及其默认限制，CUDA 会向你发出警告。

与此相关，CUDA 对可以有多少个子核函数启动处于待处理状态设有上限。默认情况下，CUDA 允许同一时刻有 2,048 个未完成的设备端启动。然而，这是可配置的。

如果一个父核函数因为使用一个大循环来启动数千个微小核函数而需要派生超过 2,048 个子核函数，你可以使用 API cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, newLimit) 提高这个限制。否则，超过默认的 2,048 限制会导致运行时错误。实践中，默认值对大多数用途通常已经足够。但对于极端情况，这是一个重要的考量。

> 在 GPU 上派生许多子核函数时，务必监控设备内存使用，因为每个待处理的子核函数启动都会保留资源，可能超过你的 GPU 硬件的硬性限制。

由于 DP 会增加一些指令级开销，它最好保留给这样的情形：静态编排（如持久化核函数、流或 CUDA Graphs）本会在 CPU 决定下一步时让 GPU 空闲。换句话说，当你的工作负载确实是一个固定的操作序列时，你通常更适合把它捕获为一张 CUDA Graph 并在设备侧重放。

当你的工作从数据自身动态涌现时，DP 让你把调度与执行完全保留在 GPU 上。对于嵌套的、数据相关的或不可预测的并行，这会带来更好的扩展性和更低的端到端延迟。

> 由于 DP 会带来每次启动的开销并消耗待处理启动槽位，当新的并行在运行时从数据中涌现、且无法表达为预先录制的 CUDA Graph 时，才使用 DP。相反，当控制流处于表达式级别且重复出现时，优先选择设备端启动的 CUDA Graphs，因为 CUDA Graphs 会摊薄这些成本。

让我们比较一个简单的父子工作流的两种实现。接下来展示的主机驱动版本启动一个父核函数，等待它完成，然后从 CPU 发出两个子核函数。这在那些决策间隙期间让 GPU 空闲：

```
// dp_host_launched.cu
#include <cuda_runtime.h>
__global__ void childKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * data[idx];
    }
}
__global__ void parentKernel(float* data, int N) {
    // Parent does setup work. Here, CPU decides on child launches.
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // maybe mark regions or compute flags here
    }
}
int main() {
    const int N = 1 << 20;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    // ... initialize d_data ...
    // 1) Launch parent and wait
    cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    parentKernel<<<1,1,0,s>>>(d_data, N);
    cudaStreamSynchronize(s);
    // 2) CPU splits work in half and launches children
    int half = N / 2;
    childKernel<<<(half+255)/256,256>>>(d_data,     half);
    childKernel<<<(half+255)/256,256>>>(d_data+half, half);
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
    cudaFree(d_data);
    return 0;
}
```

在主机驱动版本中，GPU 运行 parentKernel，然后在 CPU 依次准备并启动每个 childKernel 时空闲。注意父核函数之后以及两个子核函数之间显式的 cudaDevice Synchronize() 调用。这些调用导致了应被消除的空闲间隙，如图 12-12 所示。

![图 12-12. 由子核函数启动导致的空闲间隙](AI%20Systems%20Performance%20Engineering-ch12_images/figure-12-12.png)

相比之下，设备端启动的 DP 版本让父核函数在设备上派生它的子核函数。这种方式在父核函数与子核函数启动之间不需要任何主机同步。这样，父核函数的子核函数启动会隐式地将子核函数入队，并仅在最后同步，如下面的代码所示：

```
// dp_device_launched.cu
// Dynamic parallelism requires relocatable device code enabled with -rdc=true.
#include <cuda_runtime.h>
__global__ void childKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * data[idx];
    }
}
__global__ void parentKernel(float* data, int n) {
    // Launch children from a single thread to avoid duplicate launches.
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const int threadsPerBlock = 256;
        const int firstHalfCount  = n / 2;
        const int secondHalfCount = n - firstHalfCount;
        const int blocksFirst  = (firstHalfCount  + threadsPerBlock - 1)
                                  / threadsPerBlock;
        const int blocksSecond = (secondHalfCount + threadsPerBlock - 1)
                                  / threadsPerBlock;
        // Device launched child kernels.
        // No device side cudaDeviceSynchronize is needed.
        childKernel<<<blocksFirst,  threadsPerBlock>>>(data,
                                                       firstHalfCount);
        childKernel<<<blocksSecond, threadsPerBlock>>>(data + firstHalfCount,
                                                       secondHalfCount);
        // Parent kernel will not finish until both children finish.
    }
}
int main() {
    const int N = 1024 * 1024;    // 1M elements, avoids bit shifting for clarity
    float* d_data = nullptr;
    cudaMalloc(&d_data, N * sizeof(float));
    // Initialize to zero as a concrete, valid initialization.
    cudaMemset(d_data, 0, N * sizeof(float));
    // Launch parent on the default stream.
    parentKernel<<<1, 1>>>(d_data, N);
    // Wait for completion without cudaDeviceSynchronize. Sync stream instead.
    cudaStreamSynchronize(0);
    cudaFree(d_data);
    return 0;
}
```

这里，parentKernel 直接在 GPU 上发出两个子核函数启动。主机只提交一个核函数，然后等待一次。当父核函数完成时，设备运行时会确保所有已启动的子核函数都完成后再继续。

> 注：这个动态并行版本避免了任何对 cudaDeviceSynchronize() 的使用。它依赖于一条隐式规则：父核函数在其所有设备端启动的子核函数完成之前不会完成，而主机只需在流上等待。

采用设备侧 DP 方式，就没有用于 CPU 决策的空闲间隙。因此，它消除了相依阶段之间的延迟，并让 SM 端到端保持忙碌。这提升了 GPU 利用率，代价只是设备端产生的少量每次启动开销，如图 12-13 的时间线所示。

![图 12-13. 使用设备侧启动与动态并行则没有间隙](AI%20Systems%20Performance%20Engineering-ch12_images/figure-12-13.png)

在我们这个简单的双子核函数示例中，主机驱动版本发出三次独立的启动（1 个父 + 2 个子）。在这种情况下，GPU 在 CPU 决定何时启动每个子核函数时空闲。

这与设备驱动的 DP 版本形成对比——后者只为父核函数执行一次主机侧启动。父核函数随后在 GPU 上派生两个子核函数，无需进一步的主机干预。表 12-2 比较了主机驱动与设备驱动 DP 子核函数启动的性能。

表 12-2. 主机驱动与 GPU 驱动的嵌套子核函数启动性能对比（2 个子核函数）

| 指标                     | 之前（主机启动） | 之后（设备启动） |
| ------------------------ | ---------------- | ---------------- |
| 主机启动总数             | 3                | 1                |
| 每次调用的平均启动开销   | ~20 µs           | ~25 µs           |
| GPU 空闲周期（序列期间） | ~40%             | ~5%              |
| 整体执行时间             | 1.00 ms          | 0.75 ms          |

这里，我们看到通过把子核函数派发移入 GPU，DP 消除了大约 35% 的空闲时间，并将总运行时长减少约 25%。每次启动成本的轻微上升（20 µs → 25 µs）反映了 GPU 的设备端调度开销。然而，与移除多次 CPU-GPU 握手所带来的节省相比，这一开销可以忽略不计。

GPU 驱动启动的另一个好处是改善了数据局部性，因为中间结果在各阶段之间从不必拷回 CPU。在我们的示例中，父核函数计算出的数据可以立即被子核函数使用，而无需离开 GPU 内存。这避免了额外的内存传输并保留了缓存数据。没有 CPU 干预也意味着更少机会发生缓存逐出或对本可在 GPU 上重用的数据进行 DRAM 重取。

简而言之，DP 把一个走走停停、主机驱动的工作流转变为一条无缝的、GPU 驻留的流水线，维持高 SM 利用率并最小化主机-GPU 协调。并且请记住，动态并行和其他高级技术一样，应当针对影响进行测试。

尽管 DP 消除了 CPU 交互，但设备端发起的核函数启动的开销量级仍与主机启动大致相同。因此，并非所有算法都会看到收益。事实上，某些算法用 DP 甚至可能运行得更慢——尤其是当被启动核函数的工作量太小、不足以摊薄开销时。换句话说，对于某些小核函数，设备侧启动开销可能抵消 DP 的收益。

> 在做出更改之前和之后务必进行性能剖析。像 Nsight Compute 这样的工具可以剖析用 DP 启动的子核函数，以帮助量化它们的成本，并确保 GPU 驻留流水线的收益真正超过额外开销并提升吞吐量。

介绍完单 GPU 编排之后，我们接下来转向多 GPU 与多节点场景，在那里互连带宽与集合操作将我们的 roofline 考量扩展到集群层面。

## 跨多个 GPU 与集群节点编排（NVSHMEM）

当你从单个 GPU 扩展到多个时，核心目标依然不变：通过把数据移动隐藏在有用工作之后来让每个设备保持忙碌。一旦主机把任务分派给每个 GPU——无论是通过独立的 CPU 线程、异步启动，还是多 GPU 图——GPU 就会接管。当每个设备上的一个流驱动计算时，第二个流可以在 NVLink 或 PCIe 上进行点对点数据穿梭，完全无需涉及主机内存。

这意味着，在大规模下，你必须把点对点传输与计算重叠。需要注意的是，即便有 NVLink，其带宽和延迟也不等同于设备上的 HBM。因此这种通信必须通过重叠来隐藏。

实践中，随着集群规模增长，把工作与数据传输重叠对于让环境线性扩展绝对是必不可少的。对于简单直接的交接，你可以使用 GPUDirect Peer Access 在后台移动大块内存，如下所示：

```
cudaMemcpyPeerAsync(dest_ptr, dest_gpu, src_ptr, src_gpu, size, comm_stream);
```

当你需要集合通信（例如 PyTorch 分布式数据并行（DDP）中的梯度 all-reduce）时，你可以使用 NCCL 的异步集合调用，在一个独立的流上启动 NCCL 的非阻塞例程。NCCL 随后会把你的张量编排成环或树，以饱和每一条 NVLink 与 NVSwitch 路径——同时你的计算核函数继续在它们自己的流上运行。

如果你的 MPI 库是 CUDA-aware 的并能识别 GPU 设备指针，它会自动使用 GPUDirect RDMA，通过类似下面这样的调用在 InfiniBand 上发送数据：

```
MPI_Send(device_buf, count, MPI_FLOAT, peer_rank, ...);
```

MPI（以及 NCCL）中的这种 CUDA-awareness 意味着 GPU 数据通过 InfiniBand 上的 GPUDirect RDMA 直接跨网络移动，而无需经由主机内存中转。（注意，在一个节点内部，对等拷贝通过使用 GPUDirect Peer-to-Peer 的 NVLink 或 PCIe 进行。）

使用这些原语并避开 CPU 有助于降低数据传输延迟，并为 GPU 间传输达到近乎线速。因此，节点间的数据传输与通信可以正确地与 GPU 计算重叠。

### 用 NVSHMEM 做细粒度 GPU 间内存共享

对于需要超紧密、事件驱动协调的工作负载（例如动态任务队列和细粒度事件通知），NVIDIA 的 NVIDIA SHMEM（NVSHMEM）库是一个绝佳选择。它把每个 GPU 视为分区全局地址空间（PGAS）中的一个处理单元（PE）。

有了 PGAS，一个 GPU 可以从设备端代码直接写入另一个 GPU 的内存，绕过 CPU。延迟取决于互连，NVLink 通常低于 PCIe 或网络传输。下面是使用 NVSHMEM 的经典“发送并信号”模式：

```
#include <cstdio>
#include <cuda_runtime.h>
#include <nvshmem.h>
#include <nvshmemx.h>
// Device symbols for the symmetric buffers
__device__ int   *remote_flag;
__device__ float *remote_data;
//-----------------------------------------------------------------------------
// GPU 0: send data then signal GPU 1
//-----------------------------------------------------------------------------
__global__ void sender_kernel(float *local_data, int dest_pe) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float value = local_data[idx];
    // 1) Put the payload into remote_data[1] on dest_pe
    nvshmem_float_p(remote_data + 1, value, dest_pe);
    // 2) Wait for the RMA to complete before setting the flag
    nvshmem_quiet();
    // 3) Signal completion by setting remote_flag[0] = 1 on dest_pe
    nvshmem_int_p(remote_flag + 0, 1, dest_pe);
}
//-----------------------------------------------------------------------------
// GPU 1: wait for flag then consume payload
//-----------------------------------------------------------------------------
__global__ void receiver_kernel(float *recv_buffer) {
    // 1) Spin until remote_flag[0] == 1
    nvshmem_int_wait_until(remote_flag + 0,
        NVSHMEM_CMP_EQ, 1);
    // 2) Once flag is set, the payload at remote_data[1] is valid
    float val = remote_data[1];
    recv_buffer[0] = val * 2.0f;
}
//-----------------------------------------------------------------------------
// Host-side setup and teardown
//-----------------------------------------------------------------------------
int main(int argc, char **argv) {
    // 1) Initialize the NVSHMEM runtime
    nvshmem_init();
    // 2) Determine this PE’s rank and bind to the matching GPU
    int mype = nvshmem_my_pe();
    cudaSetDevice(mype);
    // 3) Allocate symmetric buffers on each PE
    //    - Two ints for the flag
    //    - Two floats for the data payload
    int   *flag_buf = (int*)   nvshmem_malloc(2 * sizeof(int));
    float *data_buf = (float*) nvshmem_malloc(2 * sizeof(float));
    // 4) Zero out flags on PE 0 and synchronize
    nvshmem_barrier_all();
    if (mype == 0) {
        int zeros[2] = {0, 0};
        cudaMemcpy(flag_buf, zeros, 2 * sizeof(int),
                   cudaMemcpyHostToDevice);
    }
    nvshmem_barrier_all();
    // 5) Register the device pointers for use in kernels
    cudaMemcpyToSymbol(remote_flag, &flag_buf, sizeof(int*));
    cudaMemcpyToSymbol(remote_data, &data_buf, sizeof(float*));
    // 6) Launch either the sender or receiver kernel
    dim3 grid(1), block(128);
    if (mype == 0) {
        // Example input buffer for the sender
        float *local_data;
        cudaMalloc(&local_data, 128 * sizeof(float));
        // ... initialize local_data as needed ...
        sender_kernel<<<grid, block>>>(local_data, 1);
        cudaFree(local_data);
    } else {
        float *recv_buffer;
        cudaMalloc(&recv_buffer, sizeof(float));
        receiver_kernel<<<grid, block>>>(recv_buffer);
        cudaFree(recv_buffer);
    }
    // 7) Wait for all GPU work to finish
    cudaDeviceSynchronize();
    // 8) Clean up NVSHMEM resources
    nvshmem_free(flag_buf);
    nvshmem_free(data_buf);
    nvshmem_finalize();
    return 0;
}
```

这里，单边远程内存操作完全发生在设备上。GPU/PE 0 把它的结果直接写入 GPU/PE 1 的内存，并在那里翻转一个标志。具体来说，GPU/PE 0 发出一个 nvshmem_float_p 将载荷数据直接写入 GPU/PE 1 的内存，调用 nvshmem_quiet() 以确保完成，然后使用 nvshmem_int_p 翻转一个标志。
