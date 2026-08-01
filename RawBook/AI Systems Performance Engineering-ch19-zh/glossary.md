# 术语表（glossary.md）— 第 19 章：动态与自适应推理引擎优化

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具/命令名、API/函数名、环境变量、指标名、代码一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。承接第 6–18 章的既定译法。

## 章节主题

- Dynamic and Adaptive Inference Engine Optimizations → 动态与自适应推理引擎优化
- dynamic / adaptive → 动态（dynamic）/ 自适应（adaptive）
- inference engine → 推理引擎（inference engine）
- runtime → 运行时（runtime）
- prefill / decode / token / worker → 保留
- throughput / latency / goodput → 吞吐量 / 延迟 / 有效吞吐量（goodput）
- TTFT / TPOT / SLO / SLA / RPS → 保留

## 自适应并行策略

- adaptive parallelism strategies → 自适应并行策略
- tensor parallelism (TP) / pipeline parallelism (PP) → 张量并行（tensor parallelism，TP）/ 流水线并行（pipeline parallelism，PP）
- data parallelism / replica scaling → 数据并行（data parallelism）/ 副本扩展（replica scaling）
- expert parallelism → 专家并行（expert parallelism）
- hybrid (dynamic / autoswitching) → 混合（hybrid）/ 动态自动切换（autoswitching）
- hybrid-sharding pool → 混合分片池（hybrid-sharding pool）
- pipeline bubble → 流水线气泡（pipeline bubble）
- microbatch → 微批（microbatch）
- all-reduce / collective → 保留 all-reduce / 集合通信（collective）
- mixture-of-experts (MoE) → 保留 MoE（专家混合，mixture-of-experts）
- gating network / router / top-k experts → 门控网络（gating network）/ 路由器（router）/ top-k 专家
- preprovisioning → 预置（preprovisioning）

## 动态精度

- dynamic precision changes → 动态精度切换
- FP16 / BF16 / FP8 / FP4 / INT4 / INT8 / MXFP8 / NVFP4 → 保留
- precision mode → 精度模式（precision mode）
- AMP (automatic mixed precision) → 保留 AMP（自动混合精度，automatic mixed precision）
- torch.autocast / torch.float8_e4m3 / torch.float8_e5m2 → 保留
- Transformer Engine (TE) → 保留 Transformer Engine（TE）
- weight quantization / activation quantization → 权重量化 / 激活量化（activation quantization）
- hysteresis → 迟滞（hysteresis）
- exponential moving average (EMA) → 指数移动平均（exponential moving average，EMA）
- logit margin / confidence → logit 间距（logit margin）/ 置信度（confidence）
- flapping → 抖动（flapping）
- Tensor Core → 保留

## 核函数自动调优与占用率

- kernel autotuning → 核函数自动调优（kernel autotuning）
- self-attention / MLP path → 自注意力（self-attention）/ MLP 通路
- tile size / tiling → 分块大小（tile size）/ 分块（tiling）
- TILE_Q / TILE_K → 保留
- CUTLASS / Triton / cuBLAS / cuBLASLt → 保留
- GEMM → 保留
- shared-memory allocation → 共享内存分配（shared-memory allocation）
- occupancy-aware kernel selection → 占用率感知的核函数选择（occupancy-aware kernel selection）
- occupancy → 占用率（occupancy）
- SM / register / thread block → 保留 SM / 寄存器（register）/ 线程块（thread block）
- microbenchmark → 微基准测试（microbenchmark）
- programmatic dependent launch (PDL) → 保留 PDL（编程式依赖启动，programmatic dependent launch）
- domain-specific library/language (DSL) → 领域特定库/语言（DSL）

## KV 缓存优化

- speculative KV prefetching → 推测式 KV 预取（speculative KV prefetching）
- prefetch → 预取（prefetch）
- real-time KV cache compression → 实时 KV 缓存压缩
- policy switching → 策略切换（policy switching）
- KV cache / PagedAttention → 保留 KV 缓存（KV cache）/ PagedAttention
- eviction / retention → 逐出（eviction）/ 保留（retention）

## 运行时调优与内存分配

- reinforcement learning (RL) agent → 强化学习（reinforcement learning，RL）智能体
- reward / policy / state / action → 奖励（reward）/ 策略（policy）/ 状态（state）/ 动作（action）
- dynamic memory-allocation switching → 动态内存分配切换
- slab allocator / caching allocator / stream-ordered allocator → slab 分配器 / 缓存分配器（caching allocator）/ 流序分配器（stream-ordered allocator）
- fragmentation → 碎片化（fragmentation）
- memory pool → 内存池（memory pool）
- hot-swappable implementation → 可热插拔实现（hot-swappable implementation）
- hot swap → 热插拔（hot swap）
- continuous prewarming → 持续预热（continuous prewarming）
- CUDA Graphs → 保留
- time-series prediction / forecasting → 时间序列预测（time-series prediction）/ 预测

## 自适应批处理与调度

- adaptive batching → 自适应批处理（adaptive batching）
- chunked prefill scheduling → 分块 prefill 调度（chunked prefill）
- continuous batching → 连续批处理（continuous batching）
- congestion-aware / topology-aware scheduling → 拥塞感知（congestion-aware）/ 拓扑感知（topology-aware）调度
- NVLink / NVSwitch / topology / bandwidth → 保留 / 拓扑（topology）/ 带宽（bandwidth）
- link telemetry → 链路遥测（link telemetry）
- process-GPU mapping → 进程-GPU 映射（process-GPU mapping）
- NCCL / GPUDirect / RDMA / InfiniBand → 保留
- collective communication → 集合通信（collective communication）
- multinode / multirack → 多节点（multinode）/ 多机架（multirack）
- MoE expert rebalancing / regrouping → MoE 专家再平衡（rebalancing）/ 重新分组（regrouping）
- hotspot → 热点（hotspot）

## 其他自适应与动态技术

- dynamic early-exit network → 动态提前退出网络（dynamic early-exit network）
- input-aware layer skipping (DASH) → 输入感知的层跳过（input-aware layer skipping，DASH）
- speculative MoE expert routing → 推测式 MoE 专家路由
- dynamic token pruning (LazyLLM) → 动态 token 剪枝（dynamic token pruning）— LazyLLM 保留
- edge-oriented MoE memory budgeting → 面向边缘的 MoE 内存预算
- dynamic quantization / activation range adjustment → 动态量化 / 激活范围调整（activation range adjustment）
- early exit → 提前退出（early exit）
- layer skipping / token pruning → 层跳过 / token 剪枝（token pruning）

## 硬件/平台（保留原文）

- Blackwell / B200 / B300 / Hopper / H100 / A100 / GB200 / NVL72 → 保留
- GPU / CPU / HBM / DRAM / SM / NVLink / NVSwitch → 保留
- PyTorch / vLLM / SGLang / NVIDIA Dynamo / NCCL / Triton / CUTLASS / cuBLAS → 保留
