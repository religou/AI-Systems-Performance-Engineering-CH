# 术语表（glossary.md）— 第 11 章：核间流水线化、同步与 CUDA 流序内存分配

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具/命令名、API/函数名、Nsight 指标名、代码一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。承接第 6-10 章的既定译法。

## 章节主题

- Inter-Kernel Pipelining, Synchronization, and CUDA Stream-Ordered Memory Allocations → 核间流水线化、同步与 CUDA 流序内存分配
- inter-kernel pipelining / inter-kernel concurrency → 核间流水线化（inter-kernel pipelining）/ 核间并发（inter-kernel concurrency）
- intra-kernel pipelining → 核内流水线化（intra-kernel pipelining）— 承接第 10 章
- overlap / overlapping → 重叠（overlap）
- latency hiding → 延迟隐藏
- pipeline → 流水线（pipeline）
- warp → warp（线程束）— 首次可括注，后续用 warp
- SM (streaming multiprocessor) → 流多处理器（streaming multiprocessor，SM）
- DMA (direct memory access) engine → 直接内存访问（direct memory access，DMA）引擎
- compute engine → 计算引擎（compute engine）
- host / device → 主机（host）/ 设备（device）

## CUDA 流

- CUDA stream / CUDA streams → CUDA stream / CUDA streams — 保留（可括注“流”）
- stream 0 / default stream → stream 0 / 默认流（default stream）
- legacy default stream → 传统默认流（legacy default stream）
- per-thread default stream → 每线程默认流（per-thread default stream）
- explicit / nondefault stream → 显式流（explicit stream）/ 非默认流（nondefault stream）
- nonblocking stream → 非阻塞流（nonblocking stream）— 保留 nonblocking 拼写
- cudaStreamCreateWithFlags / cudaStreamNonBlocking → 保留
- cudaStreamSynchronize / cudaStreamDestroy / cudaStreamWaitEvent → 保留
- cudaDeviceSynchronize → 保留
- CUDA_LAUNCH_BLOCKING → 保留（环境变量）
- CUDA_MODULE_LOADING / PYTORCH... env var → 保留（环境变量名）
- serialize / serialization → 串行化（serialize）
- concurrency / concurrent → 并发（concurrency）
- queue → 队列（queue）

## 流序内存分配器

- stream-ordered memory allocator → 流序内存分配器（stream-ordered memory allocator）
- cudaMallocAsync / cudaFreeAsync → 保留
- cudaMalloc / cudaFree (blocking) → 保留（阻塞式）
- memory pool → 内存池（memory pool）
- fragmentation → 碎片化（fragmentation）
- pinned host memory / pageable memory → 固定主机内存（pinned host memory）/ 可分页内存（pageable memory）
- H2D / D2H / host-to-device / device-to-host → 保留 H2D/D2H；主机到设备 / 设备到主机
- cudaMemcpyAsync → 保留
- in-flight (kernels) → 在途（in-flight）
- scratch buffer → 暂存缓冲区（scratch buffer）

## 事件与同步

- CUDA event / cudaEventRecord / cudaEventCreateWithFlags / cudaEventDisableTiming → 保留
- cross-stream synchronization → 跨流同步（cross-stream synchronization）
- fine-grained synchronization → 细粒度同步（fine-grained synchronization）
- producer-consumer (dependency) → 生产者—消费者（producer-consumer）
- callback / cudaLaunchHostFunc → 回调（callback）/ 保留 cudaLaunchHostFunc
- synchronization barrier → 同步屏障（synchronization barrier）
- host thread → 主机线程（host thread）

## 核间流水线（承接第 10 章）

- warp specialization / warp-specialized → warp 专门化（warp specialization）
- producer / consumer / loader / compute / storer warp → 生产者 / 消费者 / loader / compute / storer warp
- double-buffered / two-stage / multistage → 双缓冲（double-buffered）/ 两阶段（two-stage）/ 多阶段（multistage）— 保留 multistage 拼写
- cuda::pipeline / cuda::memcpy_async / cuda::barrier → 保留
- thread block cluster → 线程块簇（thread block cluster）
- distributed shared memory (DSMEM) → 分布式共享内存（distributed shared memory，DSMEM）
- Tensor Memory Accelerator (TMA) / TMA multicast → Tensor Memory Accelerator（TMA）/ TMA 多播（multicast）
- cp.async.bulk.tensor / cp.reduce.async.bulk / cluster.map_shared_rank / cluster.sync → 保留
- CTA / leader CTA / follower CTA → CTA（线程块）/ 主导 CTA（leader CTA）/ 跟随 CTA（follower CTA）
- **cluster_dims** / mbarrier → 保留
- mini-batch / microbatch → 小批量（mini-batch）/ 微批量（microbatch）
- cooperative launch / cooperative kernel → 协作式启动（cooperative launch）/ 协作式核函数
- GEMM / FMA (fused-multiply-add) → 保留 GEMM；FMA（乘加融合，fused-multiply-add）

## 多 GPU

- multi-GPU → 多 GPU（multi-GPU）
- peer-to-peer (P2P) transfer → 点对点（peer-to-peer，P2P）传输
- cudaMemcpyPeerAsync → 保留
- collective communication / NCCL / all-reduce → 集合通信（collective communication）/ 保留 NCCL / all-reduce
- ring (all-reduce) → 环形（ring）
- interconnect / NVLink / NVSwitch → 互连（interconnect）/ 保留 NVLink、NVSwitch
- communication stream / compute stream → 通信流（communication stream）/ 计算流（compute stream）
- stream priority → 流优先级（stream priority）
- device-wide synchronization → 设备级同步（device-wide synchronization）
- chunked all-reduce / chunked P2P → 分块 all-reduce / 分块 P2P
- PyTorch distributed backend → PyTorch 分布式后端

## 程序化依赖启动（PDL）

- programmatic dependent launch (PDL) → 程序化依赖启动（programmatic dependent launch，PDL）
- primary kernel / secondary / dependent kernel → 主核函数（primary kernel）/ 次级核函数、依赖核函数（dependent kernel）
- cudaTriggerProgrammaticLaunchCompletion → 保留
- cudaGridDependencySynchronize → 保留
- cudaLaunchKernelExC / cudaLaunchConfig_t / cudaLaunchAttributeProgrammaticStreamSerialization → 保留
- prologue / epilogue → 序言（prologue）/ 收尾（epilogue）
- kernel-launch overhead / launch overhead → 核函数启动开销（kernel-launch overhead）
- torn down / teardown → 拆除（teardown）

## 其他

- CUDA Graphs → 保留（第 12 章详述）
- dynamic parallelism → 动态并行（dynamic parallelism）
- resource occupancy / occupancy → 资源占用（resource occupancy）/ 占用率（occupancy）
- Tensor Core → 保留
- Blackwell / Hopper / GB200 / NVL72 → 保留
- register pressure → 寄存器压力（register pressure）
- goodput / throughput → 有效吞吐（goodput）/ 吞吐（throughput）
- real-world → 真实（real-world）
- ultra-latency-sensitive → 对超低延迟敏感的（ultra-latency-sensitive）
