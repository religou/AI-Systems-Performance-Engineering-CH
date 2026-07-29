# 术语表（glossary.md）— 第 10 章：核内流水线化、warp 专门化与协作式线程块簇

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具/命令名、API/函数名、Nsight 指标名、代码一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。承接第 6/7/8/9 章的既定译法。

## 章节主题

- Intra-Kernel Pipelining, Warp Specialization, and Cooperative Thread Block Clusters → 核内流水线化、warp 专门化与协作式线程块簇
- intra-kernel pipelining → 核内流水线化（intra-kernel pipelining）
- inter-kernel pipelining → 核间流水线化（inter-kernel pipelining）— 第 11 章主题
- warp specialization / warp-specialized → warp 专门化（warp specialization）/ warp 专门化的
- producer-consumer (model/pattern) → 生产者—消费者（producer-consumer）模型/模式
- cooperative thread block clusters → 协作式线程块簇（cooperative thread block clusters）
- latency hiding → 延迟隐藏
- overlap (compute and data transfer) → 重叠（overlap，计算与数据传输）
- pipeline stage → 流水线阶段（pipeline stage）
- tile / tiling / microtile → 分块块（tile）/ 分块（tiling）/ 微分块块（microtile）
- warp → warp（线程束）— 首次可括注，后续用 warp
- SIMT (single instruction, multiple threads) → 单指令多线程（SIMT）
- streaming multiprocessor (SM) → 流多处理器（streaming multiprocessor，SM）
- GPC (Graphics Processing Cluster) → 保留 GPC

## CUDA Pipeline API 与双缓冲

- CUDA Pipeline API / C++ Pipeline API → CUDA Pipeline API — 保留
- cuda::pipeline / cuda::pipeline_shared_state / cuda::make_pipeline → 保留
- cuda::memcpy_async / cp.async → 保留
- double buffering / double-buffered → 双缓冲（double buffering）/ 双缓冲的
- two-stage / three-stage / multistage → 两阶段（two-stage）/ 三阶段（three-stage）/ 多阶段（multistage）
- producer_acquire() / producer_commit() / consumer_wait() / consumer_release() → 保留（Pipeline API 调用）
- prologue / steady state / epilogue → 序言（prologue）/ 稳态（steady state）/ 收尾（epilogue）
- ring buffer → 环形缓冲（ring buffer）
- prefetch → 预取（prefetch）
- __syncthreads() / block-wide barrier → 保留 __syncthreads() / 块级屏障（block-wide barrier）
- thread_scope_block → 保留
- cooperative tiling → 协作式分块（cooperative tiling）
- FlashAttention-3 → 保留

## 持久化核函数与巨型核函数

- persistent kernel → 持久化核函数（persistent kernel）
- megakernel → 巨型核函数（megakernel）
- work queue / work-stealing → 工作队列（work queue）/ 工作窃取（work-stealing）
- device-side queue → 设备端队列（device-side queue）
- launch overhead → 启动开销（launch overhead）
- kernel launch → 核函数启动
- grid-stride loop → 网格跨步循环（grid-stride loop）
- vLLM / SGLang → 保留
- decode throughput → 解码吞吐（decode throughput）
- persistentKernel / combinedKernel → 保留（示例核函数名）

## 协作组

- cooperative groups → 协作组（cooperative groups）
- grid synchronization / grid.sync() → 网格同步（grid synchronization）/ 保留 grid.sync()
- cluster.sync() → 保留
- cooperative grid / cooperative launch → 协作式网格（cooperative grid）/ 协作式启动（cooperative launch）
- cudaLaunchCooperativeKernel → 保留
- cooperativeLaunch (device attribute) → 保留
- cudaOccupancyMaxActiveBlocksPerMultiprocessor → 保留
- this_thread_block() / this_grid() / this_cluster() / tiled_partition → 保留
- grid-level / cluster-level / block-level synchronization → 网格级 / 簇级 / 块级同步
- deadlock → 死锁（deadlock）
- coscheduled / co-scheduled → 协同调度（coscheduled）

## 线程块簇与分布式共享内存

- thread block cluster → 线程块簇（thread block cluster）
- cooperative thread array (CTA) → 协作线程阵列（cooperative thread array，CTA）
- CTA → 保留（即线程块）
- CGA (cooperative grid array) → 保留 CGA
- distributed shared memory (DSMEM or DSM) → 分布式共享内存（distributed shared memory，DSMEM 或 DSM）
- Tensor Memory Accelerator (TMA) → Tensor Memory Accelerator（TMA）— 保留
- TMA multicast → TMA 多播（multicast）
- cp.async.bulk.tensor / multicast descriptor → 保留
- thread block swizzling → 线程块交错（thread block swizzling）
- scratch memory → 暂存内存（scratch memory）
- thread block pair (CTA pair) → 线程块对（thread block pair，即 CTA pair）
- map_shared_rank() / block_rank() / dim_blocks() → 保留
- cluster dimension / cluster size → 簇维度（cluster dimension）/ 簇大小（cluster size）
- portable / nonportable cluster size → 可移植 / 非可移植（nonportable）簇大小
- cudaFuncSetAttribute / cudaFuncAttributeNonPortableClusterSizeAllowed → 保留
- on-chip / off-chip → 片上（on-chip）/ 片外（off-chip）
- SM-to-SM network → SM 间网络（SM-to-SM network）
- multicast / broadcast → 多播（multicast）/ 广播（broadcast）
- inter-CTA / inter-block → CTA 间（inter-CTA）/ 块间（inter-block）
- leader block / follower block → 主导块（leader block）/ 跟随块（follower block）

## 性能与硬件（一律保留原文型号）

- SM Active % / warp execution efficiency / warp state stall % → SM Active % / warp 执行效率（warp execution efficiency）/ warp 状态停顿百分比
- L2 throughput / L2 cache hit rate → L2 吞吐 / L2 缓存命中率
- occupancy / resident warps → 占用率（occupancy）/ 驻留 warp（resident warps）
- HBM / DRAM / L1 / L2 → 保留
- memory bound / compute bound → 访存受限（memory bound）/ 计算受限（compute bound）
- arithmetic intensity → 算术强度（arithmetic intensity）
- Blackwell / B200 / Hopper / Ampere → 保留
- Tensor Cores / TMEM → 保留 Tensor Core / TMEM（张量内存）
- Nsight Compute / Nsight Systems → 保留
- NVCC / PTX / SASS → 保留
- cudaMemcpyAsync / cudaMallocAsync / cudaFreeAsync → 保留
- CUDA streams → CUDA streams — 保留
- multiheaded attention / multihead attention → 多头注意力（multiheaded attention）
- TorchInductor / PyTorch compiler → 保留
- GB200 / GB300 / NVL72 / NVLink / NVSwitch → 保留
- round trip → 往返（round trip）
- goodput / throughput → 有效吞吐（goodput）/ 吞吐（throughput）
