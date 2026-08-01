# 术语表（glossary.md）— 第 15 章：多节点推理、并行、解码与路由优化

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具/命令名、API/函数名一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。承接第 6–14 章的既定译法。本章几乎无代码，以架构与策略论述为主。

## 章节主题

- Multinode Inference, Parallelism, Decoding, and Routing Optimizations → 多节点推理、并行、解码与路由优化
- inference / serving → 推理（inference）/ 服务（serving）
- multinode → 多节点（multinode）
- throughput / latency → 吞吐量（throughput）/ 延迟（latency）

## 分离式 Prefill/Decode 架构

- disaggregated prefill and decode → 分离式 prefill 与 decode（disaggregated prefill and decode）
- prefill / decode → 保留 prefill / decode（prefill 为预填充、decode 为解码，首次可括注）
- prefill-decode interference → prefill-decode 干扰
- PD (prefill-decode) → 保留 PD
- worker node / prefill pool / decode worker → 工作节点（worker node）/ prefill 池 / decode 工作节点
- TTFT (time to first token) → 首 token 时延（time to first token，TTFT）
- TPOT (time per output token) → 每输出 token 时延（time per output token，TPOT）
- KV cache → KV 缓存（KV cache）
- KV cache transfer / handoff → KV 缓存传输 / 交接（handoff）
- NIXL → 保留
- vLLM / SGLang / NVIDIA Dynamo → 保留
- cluster orchestrator → 集群编排器（cluster orchestrator）
- Kubernetes / intranode / internode → 保留 Kubernetes / 节点内（intranode）/ 节点间（internode）
- hidden state → 隐藏状态（hidden state）

## 并行策略

- parallelism strategy → 并行策略（parallelism strategy）
- tensor parallelism (TP) → 张量并行（tensor parallelism，TP）
- pipeline parallelism (PP) → 流水线并行（pipeline parallelism，PP）
- expert parallelism (EP) → 专家并行（expert parallelism，EP）
- data parallelism (DP) → 数据并行（data parallelism，DP）
- context / sequence parallelism → 上下文并行（context parallelism）/ 序列并行（sequence parallelism）
- hybrid parallelism → 混合并行（hybrid parallelism）
- MoE (mixture of experts) → MoE（专家混合，mixture of experts）— 保留 MoE
- expert / gating / router → 专家（expert）/ 门控（gating）/ 路由器（router）
- all-reduce / all-to-all → 保留
- pipeline bubble → 流水线气泡（pipeline bubble）
- microbatching → 微批处理（microbatching）
- sparse activation / sparse compute → 稀疏激活（sparse activation）/ 稀疏计算
- capacity / model capacity → 容量（capacity）/ 模型容量
- NVLink / NVSwitch / interconnect → 保留 / 互连（interconnect）
- Blackwell / NVL72 → 保留

## 投机解码与并行 token 生成

- speculative decoding → 投机解码（speculative decoding）
- parallel token generation → 并行 token 生成
- draft model / target model → 草稿模型（draft model）/ 目标模型（target model）
- two-model / draft-based → 双模型（two-model）/ 基于草稿（draft-based）
- self-speculative decoding → 自投机解码（self-speculative decoding）
- EAGLE / EAGLE-2 → 保留
- Medusa / Medusa heads / multiple heads → 保留 Medusa / Medusa 头（Medusa heads）/ 多头
- multitoken decoding → 多 token 解码（multitoken decoding）
- acceptance rate / verification → 接受率（acceptance rate）/ 验证（verification）
- autoregressive → 自回归（autoregressive）
- interleaving decode steps → 交错解码步骤
- constrained decoding → 受约束解码（constrained decoding）
- grammar / schema / JSON → 语法（grammar）/ 模式（schema）/ 保留 JSON

## MoE 动态路由

- dynamic routing → 动态路由（dynamic routing）
- expert communication optimization → 专家通信优化
- load balancing → 负载均衡（load balancing）
- capacity factor → 容量因子（capacity factor）
- expert replication → 专家复制（expert replication）
- adaptive expert routing → 自适应专家路由（adaptive expert routing）
- real-time monitoring → 实时监控（real-time monitoring）
- token drop / overflow → token 丢弃（token drop）/ 溢出（overflow）
- routing imbalance / hot expert → 路由不均衡 / 热点专家（hot expert）

## 硬件/平台（保留原文）

- GPU / HBM / DRAM / NVLink / NVSwitch / InfiniBand → 保留
- Blackwell / Hopper / GB200 / NVL72 → 保留
- vLLM / SGLang / NVIDIA Dynamo / NIXL → 保留
