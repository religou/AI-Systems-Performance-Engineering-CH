# 术语表（glossary.md）— 第 13 章：PyTorch 的性能剖析、调优与扩展

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具/命令名、API/函数名、Nsight 指标名、代码一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。承接第 6–12 章的既定译法。

## 章节主题

- Profiling, Tuning, and Scaling PyTorch → PyTorch 的性能剖析、调优与扩展
- profiling → 剖析（profiling）/ 性能剖析
- bottleneck → 瓶颈（bottleneck）
- tuning → 调优（tuning）
- scaling → 扩展（scaling）
- overhead → 开销（overhead）
- throughput / latency → 吞吐量（throughput）/ 延迟（latency）
- runtime → 运行时（runtime）

## 剖析工具与 NVTX

- NVTX (NVIDIA Tools Extension) → NVTX（NVIDIA Tools Extension）— 保留 NVTX
- NVTX marker → NVTX 标记（NVTX marker）
- NVTX range → NVTX 区间（NVTX range）
- range_push / range_pop / record_function → 保留
- timeline / timeline view → 时间线（timeline）/ 时间线视图
- PyTorch profiler (Kineto) → PyTorch 剖析器（PyTorch profiler，Kineto）— 保留 Kineto
- Nsight Systems (nsys) → 保留
- Nsight Compute (ncu) → 保留
- Linux perf → 保留
- trace / trace export → 追踪（trace）/ 追踪导出
- shape recording → 形状记录（shape recording）
- graph break → 图中断（graph break）
- key_averages / self_cuda_time_total → 保留
- CUPTI → 保留
- op-level / operation-level → 算子级（op-level）
- GPU projection summary → GPU 投影汇总（GPU projection summary）
- self GPU time / child GPU time → 自身 GPU 时间（self GPU time）/ 子 GPU 时间（child GPU time）

## Roofline 与核函数分析

- roofline analysis / roofline chart → Roofline 分析 / Roofline 图
- arithmetic intensity → 算术强度（arithmetic intensity）
- memory bound / compute bound → 访存受限（memory bound）/ 计算受限（compute bound）
- occupancy / achieved occupancy → 占用率（occupancy）/ 实测占用率
- peak FLOPS / peak memory bandwidth → 峰值 FLOPS / 峰值内存带宽
- GEMM (general matrix multiply) → GEMM（通用矩阵乘法）— 保留 GEMM
- SM (streaming multiprocessor) → SM（流式多处理器）— 保留 SM
- warp → warp（线程束）— 首次可括注，后续用 warp
- stall / stall reason → 停顿（stall）/ 停顿原因
- Tensor Core → 保留

## PyTorch 编译器（torch.compile）

- PyTorch Compiler / torch.compile → PyTorch 编译器（PyTorch Compiler）/ 保留 torch.compile
- TorchInductor / TorchDynamo / AOTAutograd → 保留
- Triton / OpenAI Triton → 保留
- kernel fusion → 核函数融合（kernel fusion）
- custom kernel → 自定义核函数（custom kernel）
- compilation mode → 编译模式（compilation mode）
- default / reduce-overhead / max-autotune / max-autotune-no-cudagraphs → 保留（模式名）
- autotuning → 自动调优（autotuning）
- regional compilation → 区域化编译（regional compilation）
- compile time → 编译时间（compile time）
- dynamic shapes → 动态形状（dynamic shapes）
- guard / recompilation → 保护条件（guard）/ 重新编译（recompilation）

## 优化的注意力机制

- attention → 注意力（attention）
- Scaled Dot-Product Attention (SDPA) → 缩放点积注意力（Scaled Dot-Product Attention，SDPA）
- scaled_dot_product_attention → 保留
- FlashAttention / FlexAttention / FlexDecoding → 保留
- torch.nn.functional → 保留
- sparsity pattern → 稀疏模式（sparsity pattern）
- attention variant → 注意力变体（attention variant）

## torchao、量化、稀疏化、剪枝

- torchao (PyTorch Architecture Optimization) → 保留 torchao
- quantization → 量化（quantization）
- sparsity → 稀疏化（sparsity）
- pruning → 剪枝（pruning）
- INT8 / FP8 / FP16 / BF16 → 保留

## CUDA Streams（PyTorch 并发）

- CUDA stream → CUDA 流（CUDA stream）
- concurrency → 并发（concurrency）
- overlapping communication and computation → 通信与计算重叠
- stream synchronization → 流同步（stream synchronization）
- event / cudaEvent → 事件（event）/ 保留 cudaEvent
- torch.cuda.Stream / torch.cuda.Event / torch.cuda.stream → 保留
- record_stream / wait_stream / synchronize → 保留
- MoE (mixture of experts) → MoE（专家混合，mixture of experts）— 保留 MoE
- expert / router / dispatch / combine → 专家（expert）/ 路由器（router）/ dispatch、combine（保留）

## CUDA Graphs（PyTorch）

- CUDA Graph / CUDA Graphs → 保留
- kernel launch overhead → 核函数启动开销
- graph capture / graph replay → 图捕获（graph capture）/ 图重放（graph replay）
- preallocating memory / static memory → 预分配内存 / 静态内存
- torch.cuda.graph / torch.cuda.CUDAGraph / make_graphed_callables → 保留
- CUDA Graph Trees → 保留（PyTorch 编译器内部机制）
- pointer stability → 指针稳定性（pointer stability）

## 内存剖析与调优

- CUDA memory allocator / caching allocator → CUDA 内存分配器 / 缓存分配器
- PYTORCH_CUDA_ALLOC_CONF / expandable_segments → 保留
- memory fragmentation → 内存碎片（memory fragmentation）
- activation checkpointing → 激活检查点（activation checkpointing）
- torch.utils.checkpoint → 保留
- offloading → 卸载（offloading）
- CPU offload / NVMe offload → CPU 卸载 / NVMe 卸载
- SuperOffload → 保留
- superchip / Grace Hopper / GH200 / GB200 → 保留
- FSDP (Fully Sharded Data Parallel) → FSDP（完全分片数据并行，Fully Sharded Data Parallel）— 保留 FSDP
- shard / sharding → 分片（shard/sharding）
- pluggable memory allocator → 可插拔内存分配器（pluggable memory allocator）
- peer-to-peer DMA (P2P DMA) → 点对点 DMA（peer-to-peer DMA，P2P DMA）
- UCX → 保留
- PyTorch Symmetric Memory → PyTorch 对称内存（symmetric memory）

## 分布式与扩展

- PyTorch Distributed → 保留
- DDP (Distributed Data Parallel) → DDP（分布式数据并行，Distributed Data Parallel）— 保留 DDP
- tensor parallelism (TP) / pipeline parallelism (PP) → 张量并行（tensor parallelism，TP）/ 流水线并行（pipeline parallelism，PP）
- context parallel / context_parallel → 上下文并行（context parallel）/ 保留 context_parallel
- TorchTitan / AsyncTP / AutoParallel / SimpleFSDP → 保留
- all-reduce / all-gather / reduce-scatter → 保留
- gradient synchronization → 梯度同步（gradient synchronization）
- HTA (Holistic Trace Analysis) → 保留 HTA
- data input pipeline / data loader / DataLoader → 数据输入流水线 / 数据加载器（data loader）/ 保留 DataLoader

## 持续集成与基准测试

- continuous integration (CI) → 持续集成（continuous integration，CI）
- performance benchmarking → 性能基准测试（performance benchmarking）
- PyTorch HUD → 保留
- regression / performance regression → 回归（regression）/ 性能回归
- MLPerf / MLPerf Logging → 保留
- structured logging → 结构化日志（structured logging）

## 硬件/平台（保留原文）

- Blackwell / Hopper / Grace / GB200 / NVL72 / GH200 → 保留
- HBM / DRAM / L2 / NVLink / NVSwitch / InfiniBand / RDMA → 保留
- PyTorch / TorchInductor / Triton / vLLM → 保留
- nvidia-smi / CUDA → 保留
