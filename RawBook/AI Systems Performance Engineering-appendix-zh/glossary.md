# 术语表（glossary.md）— 附录：AI 系统性能检查清单（175+ 项）

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具/命令名、API/函数名、环境变量、指标名、代码一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。承接第 1–20 章的既定译法（这是全书性能优化术语的综合清单）。

## 结构与体裁

- Appendix → 附录
- AI Systems Performance Checklist (175+ Items) → AI 系统性能检查清单（175+ 项）
- checklist / checklist item → 检查清单（checklist）/ 清单项
- 每个清单项为「_斜体引导语_ + 说明」结构：保留 `*...*` 斜体，引导语译为中文（内含 API/工具名保留原文），说明段落译为中文。
- mindset / best practices / tips → 心态（mindset）/ 最佳实践（best practices）/ 建议

## 通用性能术语

- performance tuning / cost optimization → 性能调优（performance tuning）/ 成本优化（cost optimization）
- throughput / latency / goodput → 吞吐量（throughput）/ 延迟（latency）/ 有效吞吐量（goodput）
- utilization / occupancy → 利用率（utilization）/ 占用率（occupancy）
- profiling / profiler → 剖析（profiling）/ 剖析器（profiler）
- bottleneck → 瓶颈（bottleneck）
- ROI / auto-tuning / autotuning → 保留 ROI / 自动调优（autotuning）
- hyperparameter → 超参数（hyperparameter）
- reproducibility → 可复现性（reproducibility）
- Bayesian optimization / reinforcement learning → 贝叶斯优化（Bayesian optimization）/ 强化学习（reinforcement learning，RL）
- activation checkpointing → 激活检查点（activation checkpointing）
- AMP (automatic mixed precision) → 保留 AMP（自动混合精度，automatic mixed precision）
- data prefetch → 数据预取（data prefetch）

## 硬件与架构

- system architecture / hardware planning → 系统架构 / 硬件规划
- CPU-GPU superchip / unified memory → CPU-GPU 超级芯片（superchip）/ 统一内存（unified memory）
- multi-GPU scaling / interconnect → 多 GPU 扩展 / 互连（interconnect）
- NVLink / NVSwitch / InfiniBand / PCIe / RDMA / GPUDirect → 保留
- topology / bandwidth → 拓扑（topology）/ 带宽（bandwidth）
- NUMA / MIG → 保留
- Blackwell / Hopper / B200 / B300 / GB200 / NVL72 / Grace / ARM → 保留

## 系统与驱动

- operating system / driver optimization → 操作系统 / 驱动优化
- kernel (OS) / huge pages / IRQ affinity → 内核（操作系统）/ 大页（huge pages）/ 中断亲和性（IRQ affinity）
- GPU resource management / scheduling → GPU 资源管理 / 调度（scheduling）
- Kubernetes / cgroups / container → 保留 / 容器（container）
- I/O optimization → I/O 优化
- data processing pipeline → 数据处理流水线（data processing pipeline）
- prefetch / caching / sharding → 预取 / 缓存 / 分片（sharding）

## GPU 编程与 CUDA 调优

- GPU programming / CUDA tuning → GPU 编程 / CUDA 调优
- kernel → 核函数（kernel）
- warp / thread block / grid / SM → 保留 warp / 线程块（thread block）/ 网格（grid）/ SM
- shared memory / registers / L2 cache → 共享内存（shared memory）/ 寄存器（registers）/ L2 缓存
- coalesced memory access → 合并内存访问（coalesced memory access）
- occupancy / bank conflict → 占用率 / bank 冲突（bank conflict）
- kernel scheduling / execution → 核函数调度 / 执行
- CUDA Graphs / streams / events → 保留 CUDA Graphs / 流（streams）/ 事件（events）
- kernel fusion / fused kernel → 核函数融合（kernel fusion）/ 融合核函数
- cuBLAS / cuDNN / CUTLASS / Triton / Nsight / nvcc / nvidia-smi → 保留

## 算术与精度

- arithmetic optimization → 算术优化
- reduced/mixed precision → 降低/混合精度（reduced/mixed precision）
- FP32 / TF32 / FP16 / BF16 / FP8 / FP4 / INT8 / INT4 → 保留
- Tensor Core / quantization → 保留 Tensor Core / 量化（quantization）
- arithmetic intensity / GEMM → 算术强度（arithmetic intensity）/ GEMM

## 分布式训练与推理

- distributed training / network optimization → 分布式训练 / 网络优化
- data/tensor/pipeline parallelism → 数据/张量/流水线并行
- all-reduce / collective / NCCL → 保留 all-reduce / 集合通信（collective）/ NCCL
- gradient accumulation / overlap → 梯度累积（gradient accumulation）/ 重叠（overlap）
- efficient inference / serving → 高效推理 / 服务（serving）
- prefill / decode / KV cache / batching → 保留 prefill / decode / KV 缓存（KV cache）/ 批处理（batching）
- continuous batching / PagedAttention / speculative decoding → 连续批处理（continuous batching）/ PagedAttention / 推测解码（speculative decoding）
- multinode inference → 多节点推理（multinode inference）
- vLLM / SGLang / TensorRT-LLM / NVIDIA Dynamo / PyTorch / TensorFlow / JAX / Keras → 保留
- MoE / TTFT / TPOT / SLO / OOM → 保留

## 功耗与散热

- power and thermal management → 功耗与散热管理
- power capping / DVFS / clock → 功耗封顶（power capping）/ DVFS / 时钟（clock）
- thermal throttling → 热降频（thermal throttling）
- energy efficiency / perf-per-watt → 能效（energy efficiency）/ 每瓦性能
