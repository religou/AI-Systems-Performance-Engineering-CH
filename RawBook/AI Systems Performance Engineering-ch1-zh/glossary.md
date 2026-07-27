# 术语表（glossary.md）

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 理由/备注。
产品名、公司名、人名、硬件型号一律保留原文，首次出现按需在译文后括注原文。

## 核心角色与方法论

- AI systems performance engineer → AI 系统性能工程师 — 全书核心角色
- AI systems performance engineering → AI 系统性能工程 — 学科/领域名
- mechanical sympathy → 机械同理心（mechanical sympathy） — 作者核心隐喻，指深谙硬件而写软件
- (hardware-software) codesign → （软硬件）协同设计 — 反复出现的核心概念
- goodput → 有效吞吐（goodput） — 关键度量，首次括注原文；区别于 throughput
- full-stack → 全栈

## 性能与度量

- throughput → 吞吐量
- latency → 延迟
- bandwidth → 带宽
- utilization → 利用率
- bottleneck → 瓶颈
- benchmarking → 基准测试
- profiling → 性能剖析（剖析）
- profiler → 性能剖析器
- overhead → 开销
- FLOPS → FLOPS（保留）
- effective training time ratio → 有效训练时间比

## 模型与算法

- large language model (LLM) → 大语言模型（LLM）
- frontier model → 前沿模型
- dense model → 稠密模型
- sparse model → 稀疏模型
- mixture of experts (MoE) → 混合专家（MoE）
- expert (routing) → 专家（路由）
- parameter → 参数
- active parameters → 激活参数
- attention (mechanism) → 注意力（机制）
- FlashAttention → FlashAttention（保留）
- Multi-Head Latent Attention (MLA) → 多头潜在注意力（MLA）
- transformer → Transformer（保留）
- chain-of-thought → 思维链
- reinforcement learning → 强化学习
- quantization → 量化
- (knowledge) distillation → （知识）蒸馏
- pruning / sparsity → 剪枝 / 稀疏性
- speculative decoding → 推测解码

## 硬件与系统

- GPU / CPU → 保留
- kernel / GPU kernel / CUDA kernel → 内核 / GPU 内核 / CUDA 内核
- Tensor Cores → Tensor Core（张量核心）
- Transformer Engine → Transformer 引擎
- interconnect → 互连
- NVLink / NVSwitch → 保留
- superchip → 超级芯片
- rack → 机架
- ultrascale → 超大规模
- memory hierarchy → 内存层级
- HBM (HBM3e) → HBM（HBM3e，保留）
- Unified Memory → 统一内存
- NUMA → 保留
- exaFLOPS / petaFLOPS → 保留
- structured sparsity → 结构化稀疏

## 分布式与通信

- distributed training / inference → 分布式训练 / 推理
- collective (communication) → 集合（通信）
- all-reduce / all-to-all / all-gather → 保留（集合通信原语）
- data / tensor / pipeline / context / expert parallelism → 数据 / 张量 / 流水线 / 上下文 / 专家并行
- fully sharded data parallelism (FSDP) → 完全分片数据并行（FSDP）
- pipeline bubble → 流水线气泡
- disaggregated prefill/decode → 分离式预填充/解码（prefill 预填充、decode 解码）
- KV cache → KV 缓存
- RDMA / GPUDirect RDMA → 保留

## 工具与库（保留原文）

- CUDA, PyTorch, Triton, NCCL, NIXL, Dynamo, vLLM, SGLang, TensorRT, cuPyNumeric,
  Nsight Systems, Nsight Compute, MLPerf, 3FS, DeepGEMM, DeepEP, DualPipe, FlashMLA, EPLB

## 固定表达

- brute force / brute-force → 蛮力 / 蛮力式
- virtuous cycle → 良性循环
- speed of light（NVIDIA 语境）→ 光速（指理论硬件上限）
- AI supercomputer in a rack → 机架级 AI 超级计算机 / “一个机架里的 AI 超级计算机”
- return on investment (ROI) → 投资回报（ROI）
- out-of-memory (OOM) → 内存不足（OOM）
