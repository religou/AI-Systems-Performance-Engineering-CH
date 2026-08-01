# 术语表（glossary.md）— 第 16 章：大规模推理的剖析、调试与调优

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具/命令名、API/函数名、环境变量、指标名、代码一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。承接第 6–15 章的既定译法（尤其第 15 章推理服务术语）。

## 章节主题

- Profiling, Debugging, and Tuning Inference at Scale → 大规模推理的剖析、调试与调优
- inference / serving → 推理（inference）/ 服务（serving）
- at scale → 大规模
- throughput / latency → 吞吐量（throughput）/ 延迟（latency）
- goodput → 有效吞吐量（goodput）
- SLO / SLA → 保留（服务级目标/协议）
- tail latency / p95 / p99 → 尾延迟（tail latency）/ p95 / p99

## 剖析、监控与调试

- system metrics / counters → 系统指标（system metrics）/ 计数器（counters）
- Nsight Systems / Nsight Compute → 保留
- Program Counter (PC) sampling → 程序计数器采样（Program Counter sampling，PC sampling）
- DCGM / Prometheus / Alertmanager / Grafana → 保留
- Kubernetes / ingress controller → 保留 / 入口控制器（ingress controller）
- SM utilization / GPU utilization → SM 利用率 / GPU 利用率
- OOM (out of memory) → OOM（内存溢出，out of memory）— 保留 OOM
- ECC / CRC / NVLink / PCIe / NVSwitch → 保留
- thermal/power throttling → 热/功耗降频（throttling）
- troubleshooting recipe → 排障方案（troubleshooting recipe）
- full-stack optimization → 全栈优化（full-stack optimization）
- correctness issue → 正确性问题（correctness issue）
- nvidia-smi / nsys / ncu → 保留

## 动态批处理、调度与路由

- dynamic batching → 动态批处理（dynamic batching）
- static batching → 静态批处理（static batching）
- continuous batching → 连续批处理（continuous batching）
- continuous scheduling → 连续调度（continuous scheduling）
- stall-free scheduling / chunked prefill → 无停顿调度（stall-free scheduling）/ 分块 prefill（chunked prefill）
- latency-aware scheduling → 延迟感知调度（latency-aware scheduling）
- dynamic routing → 动态路由（dynamic routing）
- FIFO (first-in-first-out) → 先进先出（first-in-first-out，FIFO）— 保留 FIFO
- head-of-line blocking → 队头阻塞（head-of-line blocking）
- batch / batch size / max_batch_delay_ms → 批（batch）/ 批大小 / 保留 max_batch_delay_ms
- prefill / decode / token → 保留（首次可括注 预填充/解码）
- KV cache / PagedAttention / RadixAttention → 保留 KV 缓存（KV cache）/ PagedAttention / RadixAttention
- preemption / eviction → 抢占（preemption）/ 逐出（eviction）
- microservice → 微服务（microservice）
- TTFT / TPOT → 首 token 时延（TTFT）/ 每输出 token 时延（TPOT）

## 系统级优化

- systems-level optimization → 系统级优化（systems-level optimization）
- overlapping communication and computation → 通信与计算重叠
- CUDA stream / event-based synchronization → CUDA 流 / 基于事件的同步
- nonblocking collectives → 非阻塞集合通信（nonblocking collectives）
- power and thermal constraints → 功耗与热约束
- error handling → 错误处理（error handling）
- KV cache offloading / memory pool allocation → KV 缓存卸载 / 内存池分配（memory pool allocation）
- NUMA / MIG → 保留
- CUDA Graphs / buffer pooling → 保留 CUDA Graphs / 缓冲池化（buffer pooling）

## 实时推理的量化

- quantization → 量化（quantization）
- FP16 / BF16 / FP8 / FP4 / INT4 / INT8 / NVFP4 → 保留
- reducing precision → 降低精度
- weight-only quantization → 仅权重量化（weight-only quantization）
- GPTQ / AWQ / SmoothQuant → 保留
- activation quantization → 激活量化（activation quantization）
- post-training quantization (PTQ) → 训练后量化（post-training quantization，PTQ）
- quantization-aware training (QAT) → 量化感知训练（quantization-aware training，QAT）
- quantize/dequantize → 量化/反量化（quantize/dequantize）
- fusing quantization-dequantization → 融合量化-反量化
- scaling factor / calibration → 缩放因子（scaling factor）/ 校准（calibration）
- Transformer Engine (TE) → 保留 Transformer Engine（TE）
- range analysis → 范围分析（range analysis）
- FlashAttention / FlashInfer → 保留

## 应用级优化

- application-level optimization → 应用级优化（application-level optimization）
- prompt compression → 提示压缩（prompt compression）
- prompt cleansing → 提示清洗（prompt cleansing）
- prefix caching / prefix cache → 前缀缓存（prefix caching）
- trie / radix tree → 字典树（trie）/ 基数树（radix tree）
- model cascading / tiered model deployment → 模型级联（model cascading）/ 分层模型部署（tiered model deployment）
- streaming responses → 流式响应（streaming responses）
- debouncing / request coalescing → 去抖（debouncing）/ 请求合并（request coalescing）
- token output limits / timeouts → token 输出上限 / 超时（timeout）
- knowledge distillation / pruning → 知识蒸馏（knowledge distillation）/ 剪枝（pruning）

## 硬件/平台（保留原文）

- Blackwell / Hopper / GB200 / GB300 / NVL72 → 保留
- GPU / HBM / DRAM / NVLink / NVSwitch / InfiniBand / Tensor Core / TMA → 保留
- vLLM / SGLang / NVIDIA Dynamo / NIXL / LMCache / Triton → 保留
