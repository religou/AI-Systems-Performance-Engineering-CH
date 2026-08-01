# 术语表（glossary.md）— 第 18 章：高级 Prefill-Decode 与 KV 缓存调优

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具/命令名、API/函数名、环境变量、指标名、代码一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。承接第 6–17 章的既定译法（尤其第 17 章分离式 prefill/decode 术语）。

## 章节主题

- Advanced Prefill-Decode and KV Cache Tuning → 高级 prefill-decode 与 KV 缓存调优
- prefill / decode / token / worker → 保留
- KV cache → 保留 KV 缓存（KV cache）
- disaggregation / disaggregated → 分离（disaggregation）/ 分离式
- goodput / throughput / latency → 有效吞吐量（goodput）/ 吞吐量（throughput）/ 延迟（latency）
- TTFT / TPOT / SLO / SLA → 保留
- compute bound / memory bound → 计算受限（compute bound）/ 访存受限（memory bound）

## 优化的 Decode 核函数

- optimized decode kernels → 优化的 decode 核函数
- FlashMLA / ThunderMLA / FlexDecoding / FlexAttention → 保留
- MLA (Multi-head Latent Attention) → 保留 MLA（多头潜在注意力，Multi-head Latent Attention）
- GQA (grouped-query attention) → 分组查询注意力（grouped-query attention，GQA）
- MHA / MQA → 多头注意力（MHA）/ 多查询注意力（multi-query attention，MQA）
- attention primitive / fused attention → 注意力原语（attention primitive）/ 融合注意力（fused attention）
- arithmetic intensity → 算术强度（arithmetic intensity）
- transformer / DeepSeek / Stanford / PyTorch → 保留
- BlockMask / block mask → 保留 BlockMask / 块掩码
- kernel → 核函数（kernel）
- GEMM → 保留

## KV 缓存利用与管理

- KV cache utilization / management → KV 缓存利用 / 管理
- disaggregated KV cache pool → 分离式 KV 缓存池（disaggregated KV cache pool）
- KV cache reuse / prefix sharing → KV 缓存复用 / 前缀共享（prefix sharing）
- prefix caching / prefix cache → 前缀缓存（prefix caching）
- multiturn conversation → 多轮对话（multiturn conversation）
- eviction → 逐出（eviction）
- KV cache memory layout → KV 缓存内存布局
- PagedAttention / paged KV → 保留 PagedAttention / 分页 KV
- unified view / unified memory → 统一视图 / 统一内存（unified memory）
- offload to persistent storage → 卸载到持久化存储
- superchip → 超级芯片（superchip）
- fault isolation → 故障隔离（fault isolation）

## Prefill 与 Decode 之间的快速 KV 缓存传输

- fast KV cache transfer → 快速 KV 缓存传输
- KV cache size → KV 缓存大小
- zero-copy GPU-to-GPU transfer → 零拷贝 GPU 到 GPU 传输（zero-copy GPU-to-GPU transfer）
- NIXL (NVIDIA Inference Xfer Library) → 保留
- GPUDirect RDMA / RDMA / UCX → 保留
- InfiniBand / NVLink / NVSwitch → 保留
- KV transpose / conversion → KV 转置（transpose）/ 转换
- collation / collating → 归并（collation）— 合并小的分页块
- NATS → 保留
- connector / data path design → 连接器（connector）/ 数据通路设计（data path design）
- Pipe / logical interface → 保留 Pipe / 逻辑接口
- UCX_RNDV_THRESH → 保留（环境变量）

## 并行策略与异构硬件

- tensor parallelism (TP) / pipeline parallelism (PP) → 张量并行（tensor parallelism，TP）/ 流水线并行（pipeline parallelism，PP）
- data parallelism (DP) / sequence parallelism (SP) → 数据并行（data parallelism，DP）/ 序列并行（sequence parallelism，SP）
- context parallelism (CP) → 上下文并行（context parallelism，CP）
- TP_p / TP_d / PP_p / PP_d / SP_p / SP_d / DP_p / DP_d / CP → 保留（下标 p=prefill, d=decode）
- microbatch → 微批（microbatch）
- pipeline bubble → 流水线气泡（pipeline bubble）
- heterogeneous hardware → 异构硬件（heterogeneous hardware）
- compute-optimized vs memory-optimized → 计算优化型 vs 内存优化型
- hybrid prefill with GPU-CPU collaboration → GPU-CPU 协同的混合 prefill
- FP8 / FP4 / precision → 保留

## SLO 感知的请求管理与容错

- SLO-aware request management → SLO 感知的请求管理
- fault tolerance → 容错（fault tolerance）
- early rejection / admission control → 早期拒绝（early rejection）/ 准入控制（admission control）
- quality of service (QoS) → 服务质量（quality of service，QoS）
- graceful degradation → 优雅降级（graceful degradation）
- failover / retry → 故障转移（failover）/ 重试（retry）
- instance flip → 实例翻转（instance flip）

## 动态调度与负载均衡

- dynamic scheduling / load balancing → 动态调度（dynamic scheduling）/ 负载均衡（load balancing）
- adaptive resource scheduling → 自适应资源调度（adaptive resource scheduling）
- hotspot prevention → 热点预防（hotspot prevention）
- TetriInfer / Mooncake / Arrow → 保留（研究原型/系统名）
- two-level scheduler → 两级调度器（two-level scheduler）
- feedback loop → 反馈回路（feedback loop）
- exponential moving average → 指数移动平均（exponential moving average）
- workload variation → 负载波动
- speculative decoding → 推测解码（speculative decoding）
- NVIDIA Dynamo / vLLM / SGLang → 保留

## 硬件/平台（保留原文）

- Blackwell / B200 / B300 / Grace-Blackwell / GB200 / NVL72 → 保留
- GPU / CPU / HBM / DRAM / Tensor Core / NVLink → 保留
- head dimension / hidden size / layers / heads → 头维度 / 隐藏维度（hidden size）/ 层数 / 头数
