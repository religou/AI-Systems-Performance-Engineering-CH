# 术语表（glossary.md）— 第 8 章：占用率调优、Warp 效率与指令级并行

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具/命令名、API/函数名、Nsight 指标名、代码一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。

## 章节主题

- Occupancy Tuning, Warp Efficiency, and Instruction-Level Parallelism → 占用率调优、Warp 效率与指令级并行
- occupancy / achieved occupancy → 占用率（occupancy）/ 实测占用率（achieved occupancy）
- occupancy tuning → 占用率调优
- warp efficiency / warp execution efficiency → warp 效率 / warp 执行效率（warp execution efficiency）
- instruction-level parallelism (ILP) → 指令级并行（instruction-level parallelism，ILP）
- latency hiding → 延迟隐藏
- throughput → 吞吐（throughput）
- warp → warp（线程束）— 首次可括注“线程束”，后续直接用 warp
- lane → 线程通道（lane）
- SIMT (Single Instruction, Multiple Threads) → 单指令多线程（SIMT）
- streaming multiprocessor (SM) → 流多处理器（streaming multiprocessor，SM）

## 剖析工具与诊断

- profiling / profiler → 剖析（profiling）/ 剖析器（profiler）
- bottleneck → 瓶颈
- Nsight Systems / Nsight Compute → 保留原文
- nsys / ncu → 保留原文
- timeline view → 时间线视图（timeline view）
- NVTX / NVTX ranges → 保留原文；NVTX 范围
- roofline / Roofline model / Roofline analysis → Roofline 模型 / Roofline 分析 — 保留 Roofline
- compute roof / memory roof → 计算屋顶（compute roof）/ 内存屋顶（memory roof）
- arithmetic intensity (FLOPS per byte) → 算术强度（arithmetic intensity，即每字节 FLOPS）
- PyTorch profiler / torch.profiler → PyTorch 剖析器 / torch.profiler — 保留 API 名
- Kineto / CUPTI (CUDA Profiling Tools Interface) → 保留原文
- Chrome tracing → 保留原文
- guided analysis rules → 引导式分析规则
- Memory Workload Analysis → 内存工作负载分析（Memory Workload Analysis）
- data pipeline / data loader → 数据管线（data pipeline）/ 数据加载器（data loader）
- double buffering → 双缓冲（double buffering）

## Warp 停顿原因（warp stall reasons）

- warp stall reasons / Warp State Statistics → warp 停顿原因（warp stall reasons）/ Warp 状态统计（Warp State Statistics）
- Long Scoreboard (stall) → Long Scoreboard 停顿 — 保留 Long Scoreboard
- Short Scoreboard (stall) → Short Scoreboard 停顿 — 保留 Short Scoreboard
- scoreboard → 记分板（scoreboard）
- Exec Dependency / execution-dependency stall → 执行依赖停顿（Exec Dependency）
- Not Selected (scheduler) → Not Selected（未被选中）— 保留 Not Selected
- Memory Throttle → 保留 Memory Throttle
- Math Pipe Throttle → 保留 Math Pipe Throttle
- Compute Unit Busy → 保留 Compute Unit Busy
- Memory Dependency → 内存依赖停顿（Memory Dependency）
- barrier stall / synchronization stall → 屏障停顿（barrier stall）/ 同步停顿
- instruction fetch stall → 取指停顿（instruction fetch stall）
- No Eligible / Idle → 保留 No Eligible / Idle
- eligible warps per cycle → 每周期可发射 warp 数（eligible warps per cycle）
- active warps per scheduler → 每调度器活跃 warp 数（active warps per scheduler）
- issue slot / issue-slot utilization → 发射槽（issue slot）/ 发射槽利用率
- instructions per cycle (IPC) → 每周期指令数（instructions per cycle，IPC）

## 性能区间（regimes）

- memory bound → 访存受限（memory bound）
- compute bound → 计算受限（compute bound）
- latency bound → 延迟受限（latency bound）
- underutilization / underutilizing the GPU → 利用不足（underutilization）/ GPU 利用不足
- memory bandwidth / peak HBM bandwidth → 内存带宽 / HBM 峰值带宽
- compute throughput / peak FLOPS → 计算吞吐 / 峰值 FLOPS
- limiting factor → 限制因素
- occupancy limiters → 占用率限制项（occupancy limiters）
- power limiter / HW Slowdown → 功耗限制器 / HW Slowdown — 保留 HW Slowdown

## 占用率调优

- launch configuration → 启动配置（launch configuration）
- block size / thread-block size → 线程块大小（block size）
- grid size → 网格大小（grid size）
- warps per SM / resident warps → 每 SM warp 数 / 驻留 warp 数（resident warps）
- register / register file → 寄存器（register）/ 寄存器堆（register file）
- register pressure / register spilling → 寄存器压力 / 寄存器溢出（register spilling）
- shared memory (SMEM) → 共享内存（shared memory，SMEM）
- local memory → 局部内存（local memory）
- **launch_bounds** → 保留
- Occupancy API / Occupancy Calculator → 占用率 API（Occupancy API）/ 占用率计算器（Occupancy Calculator）
- cudaOccupancyMaxPotentialBlockSize() / cudaOccupancyMaxActiveBlocksPerMultiprocessor() → 保留
- -maxrregcount / #pragma unroll → 保留
- compiler hints → 编译器提示（compiler hints）
- CUDA Graphs → 保留
- torch.compile / TorchInductor → 保留
- max-autotune / reduce-overhead → 保留（torch.compile 的 mode 取值）
- MinBlocksPerSM / MaxThreadsPerBlock → 保留

## Warp 分歧与谓词化

- warp divergence / branch divergence → warp 分歧（warp divergence）/ 分支分歧（branch divergence）
- intrawarp divergence → warp 内分歧（intrawarp divergence）
- divergent branch → 分歧分支
- control flow / control-flow path → 控制流 / 控制流路径
- masked off / inactive lanes → 被掩蔽（masked off）/ 非活跃通道
- warp-unanimous branch → warp 一致分支（warp-unanimous branch）
- predication / predicate / predicated-off → 谓词化（predication）/ 谓词（predicate）/ 谓词关闭（predicated-off）
- ternary operator → 三元运算符（ternary operator）
- warp-vote intrinsics → warp 投票内建函数（warp-vote intrinsics）
- **ballot_sync / **any_sync / \_\_all_sync → 保留
- warp intrinsics / warp shuffle → warp 内建函数（warp intrinsics）/ warp shuffle
- **shfl_sync / **shfl_down_sync → 保留
- \_\_syncthreads() → 保留
- cooperative groups → 协作组（cooperative groups）
- goodput → 有效吞吐（goodput）
- reconvergence → 重新收敛（reconvergence）
- torch.where / torch.maximum → 保留

## 指令级并行（ILP）

- dual issue / dual-issue instructions → 双发射（dual issue）
- warp scheduler → warp 调度器（warp scheduler）
- loop unrolling / #pragma unroll → 循环展开（loop unrolling）
- interleaving → 交错（interleaving）
- accumulator (register) → 累加器（accumulator）
- dependency chain → 依赖链（dependency chain）
- in-flight (instructions) → 在途（in-flight）指令
- pipeline / deep pipeline → 流水线（pipeline）/ 深流水线
- functional unit / execution unit → 功能单元（functional unit）/ 执行单元
- decode bandwidth / instruction decode → 译码带宽（decode bandwidth）/ 指令译码
- unified CUDA cores → 统一 CUDA 核心（unified CUDA cores）
- software pipelining / prefetching → 软件流水线（software pipelining）/ 预取（prefetching）
- cp.async / CUDA Pipeline API → 保留
- superscalar / out-of-order execution → 超标量（superscalar）/ 乱序执行（out-of-order execution）

## 硬件与架构（保留原文）

- Blackwell / B200 / B300 / GB200 / GB300 / GB10 → 保留
- Ampere / Hopper → 保留
- Grace Blackwell Superchip / DGX Spark / RTX PRO 6000 → 保留
- compute capability → 计算能力（compute capability）
- HBM / HBM3e → 保留
- NV-HBI (NVIDIA High-Bandwidth Interface) → 保留
- NVLink → 保留
- L1 / L2 cache → 保留（L1 / L2 缓存）
- Tensor Core / Tensor Cores → 保留
- TMEM (Tensor Memory) / TMA (Tensor Memory Accelerator) → 保留
- CUDA cores / FP32 / INT32 / FP16 / FP8 / FP4 → 保留
- MXFP8 / MXFP4 / NVFP4 / OCP MX formats → 保留
- tcgen05.mma / tcgen05.ld / tcgen05.st → 保留
- MMA (matrix multiply-accumulate) → 保留 MMA
- kernel fusion / kernel → 核函数融合（kernel fusion）/ 核函数（kernel）
- NVML (NVIDIA Management Library) → 保留
- nvidia-smi → 保留

## 度量名（Nsight 指标，一律保留原文）

- gpu\_\_time_elapsed.avg
- smsp\_\_inst_executed.sum
- smsp\_\_average_warp_latency_issue_stalled_branch_resolving
- SM Throughput % / SM Active Cycles
- with_flops / profile_memory
