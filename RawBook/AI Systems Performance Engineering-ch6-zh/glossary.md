# 术语表（glossary.md）— 第 6 章：GPU 架构、CUDA 编程与最大化占用率

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具/命令名、API/函数名、代码一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。

## 章节主题

- GPU Architecture, CUDA Programming, and Maximizing Occupancy → GPU 架构、CUDA 编程与最大化占用率
- occupancy → 占用率（occupancy）
- achieved occupancy → 实际占用率（achieved occupancy）
- GPU utilization → GPU 利用率
- goodput / useful throughput → 有效吞吐（goodput）
- throughput-optimized → 面向吞吐优化的
- latency hiding → 延迟隐藏

## CUDA 执行模型与线程层级

- SIMT (single instruction, multiple threads) → 单指令多线程（SIMT）
- thread → 线程
- warp → 线程束（warp）— 首次括注，后续用 warp
- thread block → 线程块（thread block）
- cooperative thread array (CTA) → 协作线程阵列（cooperative thread array，CTA）
- thread block cluster / CTA cluster → 线程块簇（thread block cluster）/ CTA 簇
- grid → 网格（grid）
- kernel → 核函数（kernel）
- kernel launch → 核函数启动 / 启动核函数
- launch parameters → 启动参数（launch parameters）
- blocksPerGrid / threadsPerBlock → 保留（代码标识符，不翻译、不加空格）
- warp size (32 threads) → warp 大小（32 线程）
- warp scheduler → warp 调度器
- dual-issue → 双发射（dual-issue）
- instruction-level parallelism (ILP) → 指令级并行（instruction-level parallelism，ILP）
- warp divergence → warp 分歧（warp divergence）
- lockstep → 锁步（lockstep）
- barrier / synchronization → 屏障（barrier）/ 同步
- \_\_syncthreads() / cooperative groups → 保留（API 名）

## GPU 硬件（保留原文/型号）

- SM (streaming multiprocessor) → 保留（streaming multiprocessor，流式多处理器，SM）
- Blackwell / B200 / B300 → 保留
- Grace Hopper / Grace Blackwell / Superchip → 保留
- Tensor Core → 保留（Tensor Core）
- Special Function Unit (SFU) → 特殊功能单元（Special Function Unit，SFU）
- INT32 / FP32 / FP16 / FP8 / FP4 → 保留
- LD/ST (load/store) pipeline → 保留 LD/ST（load/store，加载/存储）流水线
- TMA (Tensor Memory Accelerator) → 保留（Tensor Memory Accelerator，TMA）
- TMEM (Tensor Memory) → 保留（Tensor Memory，TMEM）
- UMMA (unified matrix-multiply-accumulate) → 保留（unified matrix-multiply-accumulate，UMMA）
- tcgen05 / tcgen05.\* → 保留（指令名）
- reticle-limited die → 光罩受限裸片（reticle-limited die）
- chip-to-chip (C2C) interconnect → 芯片间（chip-to-chip，C2C）互连
- NVLink / NVLink-C2C → 保留

## GPU 内存层级（保留硬件名，译概念）

- memory hierarchy → 内存层级（memory hierarchy）
- register / register file → 寄存器 / 寄存器堆（register file）
- register pressure → 寄存器压力（register pressure）
- register spilling / spills → 寄存器溢出（register spilling）
- shared memory (SMEM) → 共享内存（shared memory，SMEM）
- DSMEM (distributed shared memory) → 分布式共享内存（distributed shared memory，DSMEM）
- L1 cache / L2 cache → 保留 L1 缓存 / L2 缓存
- global memory → 全局内存（global memory）
- local memory → 本地内存（local memory）
- constant memory → 常量内存（constant memory）
- HBM / HBM3e → 保留
- SRAM / DRAM → 保留
- coalesced access / coalescing → 合并访问（coalesced access）
- bank conflict → bank 冲突（bank conflict）
- cache line → 缓存行（cache line）
- point of coherency (PoC) → 一致性点（point of coherency，PoC）
- memory pool → 内存池（memory pool）
- fragmentation → 碎片 / 碎片化
- bounce buffer → 中转缓冲区（bounce buffer）
- pinned / page-locked memory → 固定内存（pinned memory）/ 页锁定内存

## 统一内存与数据迁移

- Unified Memory / CUDA Managed Memory → 统一内存（Unified Memory）/ CUDA 托管内存（Managed Memory）
- unified address space → 统一地址空间
- page migration → 页迁移（page migration）
- page fault → 缺页（page fault）
- on-demand migration → 按需迁移
- prefetch / prefetching → 预取
- memory advice → 内存建议（memory advice）
- first-touch → 首次访问（first-touch）
- NUMA-local → NUMA 本地

## CUDA API / 函数（一律保留原文，不翻译、不加空格）

- **global** / **restrict** / **constant** / **launch_bounds** → 保留
- <<< >>>（chevron 语法）→ 保留，称“<<< >>>（三尖括号/chevron）语法”
- blockIdx / threadIdx / blockDim / gridDim → 保留
- cudaMalloc / cudaFree / cudaMallocHost / cudaFreeHost → 保留
- cudaMallocAsync / cudaFreeAsync → 保留
- cudaMemcpy / cudaMemcpyAsync / cudaMemcpyHostToDevice / cudaMemcpyDeviceToHost → 保留
- cudaMallocManaged / cudaMemPrefetchAsync / cudaMemAdvise → 保留
- cudaMemAdviseSetPreferredLocation / cudaMemAdviseSetReadMostly / cudaMemAdviseSetAccessedBy → 保留
- cudaStreamAttachMemAsync / cudaMemAttachSingle → 保留
- cudaMemPoolSetAttribute / cudaMemPoolAttrReleaseThreshold / cudaMemPoolTrimTo → 保留
- cudaFuncSetAttribute / cudaDeviceSynchronize / cudaStreamSynchronize → 保留
- cudaGetLastError / cudaErrorIllegalAddress → 保留
- cudaOccupancyMaxPotentialBlockSize / Occupancy API → 保留 / 占用率 API
- stream / stream-ordered → 流（stream）/ 流序（stream-ordered）
- addSequential / addParallel / myKernel / my2DKernel → 保留（示例核函数名）

## 性能分析工具（保留原文）

- Nsight Systems (nsys) → 保留
- Nsight Compute (ncu) → 保留
- Compute Sanitizer → 保留
- NVTX (NVIDIA Tools Extension) → 保留（NVIDIA Tools Extension，NVTX）
- CUDA-GDB → 保留
- range replay / source correlation → 范围重放（range replay）/ 源码关联
- CI (continuous integration) → 持续集成（continuous integration，CI）

## Roofline 与性能特征

- roofline model → Roofline 模型（保留 Roofline）
- arithmetic intensity → 算术强度（arithmetic intensity）
- FLOPS / FLOP/s / TFLOPs → 保留
- compute-bound / memory-bound → 计算受限（compute-bound）/ 访存受限（memory-bound）
- operational point / operational intensity → 运算点 / 运算强度
- ceiling / roof → 上限 / 屋顶（roofline 语境）
- DRAM bandwidth utilization → DRAM 带宽利用率
- global load efficiency → 全局加载效率（global load efficiency）
- memory pipe utilization → 内存管线利用率

## 兼容性与编译

- backward / forward compatibility → 向后 / 向前兼容
- compute capability → 计算能力（compute capability）
- PTX → 保留
- SASS → 保留
- JIT (just-in-time) compilation → 即时编译（just-in-time，JIT）
- multilaunch → 多次启动（multilaunch）

## CUDA 编程通用

- host / device → 主机（host）/ 设备（device）
- host-to-device (H2D) / device-to-host (D2H) → 主机到设备（H2D）/ 设备到主机（D2H）
- CUDA C++ / Triton / OpenAI Triton → 保留
- PyTorch / TensorFlow / JAX → 保留
- caching allocator → 缓存分配器（caching allocator）
- inference_mode / autograd → 保留（PyTorch 概念/API）
- data-parallel → 数据并行
- 2D / 3D kernel → 2D / 3D 核函数

## 性能与通用（沿用 ch1–ch5）

- throughput / latency / bandwidth / utilization / bottleneck / overhead → 吞吐量 / 延迟 / 带宽 / 利用率 / 瓶颈 / 开销
- training / inference → 训练 / 推理
- transformer / attention / prefill / decode → 保留 transformer / 注意力（attention）/ 预填充（prefill）/ 解码（decode）
- LLM → 保留
- profiling → 性能剖析（profiling）
- benchmark → 基准测试（benchmark）
- Nsight / Blackwell tuning guide → 保留

## 固定表达（沿用前几章）

- rule of thumb → 经验法则
- under the hood → 底层 / 幕后
- best practices → 最佳实践
- sweet spot → 最佳平衡点
- out of the box → 开箱即用
- in flight → 在途 / 运行中（warp in flight → 在飞 warp / 运行中的 warp）
- speed of light（硬件极限比喻）→ “光速”（物理极限，保留引号）

## 本章需报告的低争议原文修正（翻译时按正确含义处理并汇报）

- Key Takeaways 中 "threads → locks → grids" 应为 "threads → blocks → grids"（原文误将 blocks 印成 locks）；译为“线程 → 线程块 → 网格”。
- "(1 KB is of the 228 KB is reserved by CUDA)" 原文多一个 "is"，本意为“228 KB 中有 1 KB 由 CUDA 保留”。
