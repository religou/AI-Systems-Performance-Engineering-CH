# 术语表（glossary.md）— 第 20 章：AI 辅助的性能优化与迈向千万级 GPU 集群的扩展

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具/命令名、API/函数名、代码一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。承接第 6–19 章的既定译法。

## 章节主题

- AI-Assisted Performance Optimizations → AI 辅助的性能优化
- Scaling Toward Multimillion GPU Clusters → 迈向千万级 GPU 集群的扩展
- 100-Trillion-Parameter Models → 100 万亿参数模型
- performance optimization → 性能优化
- GPU kernel → GPU 核函数（kernel）
- inference / training → 推理（inference）/ 训练（training）

## AI 发现算法与核函数优化

- AlphaTensor → 保留（Google DeepMind）
- AI-discovered algorithm → AI 发现的算法
- Strassen's algorithm → 保留 Strassen 算法
- subquadratic algorithm → 次二次算法（subquadratic algorithm）
- matrix multiplication / GEMM → 矩阵乘法（matrix multiplication）/ GEMM
- tensor decomposition → 张量分解（tensor decomposition）
- automated GPU kernel optimization → 自动化 GPU 核函数优化
- DeepSeek → 保留
- reinforcement learning (RL) → 强化学习（reinforcement learning，RL）
- reward / policy / agent → 奖励（reward）/ 策略（policy）/ 智能体（agent）
- Predibase → 保留
- inference-time / test-time compute → 推理时（inference-time）/ 测试时算力（test-time compute）

## 自我改进的 AI 智能体与智能编译器

- self-improving AI agent → 自我改进的 AI 智能体（self-improving AI agent）
- AI Futures Project → 保留
- agentic → 智能体式（agentic）
- smart compiler → 智能编译器（smart compiler）
- automated code optimization → 自动化代码优化
- superoptimization → 超优化（superoptimization）
- autotuning → 自动调优（autotuning）
- LLM / large language model → 保留 LLM（大语言模型，large language model）
- code generation / codegen → 代码生成（code generation）
- compiler pass / IR → 编译器 pass / 中间表示（intermediate representation，IR）
- MLIR / LLVM / XLA / Triton / CUTLASS → 保留

## AI 辅助的实时系统优化

- AI-assisted real-time system optimization → AI 辅助的实时系统优化
- real-time / runtime → 实时（real-time）/ 运行时（runtime）
- feedback loop → 反馈回路（feedback loop）
- telemetry → 遥测（telemetry）
- self-tuning / self-optimizing → 自调优（self-tuning）/ 自优化（self-optimizing）
- digital twin → 数字孪生（digital twin）

## 大规模扩展

- multimillion GPU cluster → 千万级 GPU 集群
- hyperscale / ultrascale → 超大规模（hyperscale/ultrascale）
- trillion-parameter → 万亿参数
- mixture-of-experts (MoE) → 保留 MoE（专家混合，mixture-of-experts）
- sparsity → 稀疏性（sparsity）
- power / energy efficiency → 功耗 / 能效（energy efficiency）
- datacenter / data center → 数据中心（data center）
- interconnect / fabric → 互连（interconnect）/ 网络结构（fabric）
- fault tolerance / reliability → 容错（fault tolerance）/ 可靠性（reliability）

## 硬件/平台（保留原文）

- Blackwell / B200 / B300 / Rubin / GB200 / NVL72 → 保留
- GPU / CPU / HBM / NVLink / NVSwitch / InfiniBand → 保留
- Google DeepMind / NVIDIA / OpenAI / Anthropic / Meta → 保留
- PyTorch / vLLM / NVIDIA Dynamo → 保留
- Tensor Core / FP8 / FP4 → 保留
