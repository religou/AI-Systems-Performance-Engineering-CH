# 术语表（glossary.md）— 第 17 章：面向推理的分离式 Prefill 与 Decode 扩展

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具/命令名、API/函数名、环境变量、指标名、代码一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。承接第 6–16 章的既定译法（尤其第 15、16 章推理服务术语）。

## 章节主题

- Scaling Disaggregated Prefill and Decode for Inference → 面向推理的分离式 prefill 与 decode 扩展
- prefill / decode / token → 保留（首次可括注 预填充/解码）
- prefill-decode disaggregation / disaggregation → prefill-decode 分离（disaggregation）/ 分离
- colocated → 同置（colocated）
- monolithic (system/pipeline) → 单体（monolithic）
- inference / serving → 推理（inference）/ 服务（serving）
- at scale / ultrascale → 大规模 / 超大规模（ultrascale）
- throughput / latency → 吞吐量（throughput）/ 延迟（latency）
- goodput → 有效吞吐量（goodput）
- SLO / SLA → 保留（服务级目标/协议）
- tail latency / p90 / p95 / p99 → 尾延迟（tail latency）/ p90 / p95 / p99
- TTFT / TPOT → 首 token 时延（TTFT）/ 每输出 token 时延（TPOT）
- compute bound / memory (I/O) bound → 计算受限（compute bound）/ 访存受限（memory bound）
- FLOPS → 保留
- interference → 干扰（interference）
- head-of-line blocking → 队头阻塞（head-of-line blocking）
- FIFO → 保留（先进先出）
- request batching → 请求批处理（request batching）
- continuous batching → 连续批处理（continuous batching）

## 集群与工作节点

- prefill worker / decode worker → prefill 工作节点（prefill worker）/ decode 工作节点（decode worker）
- worker pool / cluster pool → 工作节点池（worker pool）/ 集群池
- decode-centric design → 以 decode 为中心的设计（decode-centric design）
- ingress → 请求入口（ingress）
- API router → API 路由器
- 2P1D configuration → 2P1D 配置 — 2 个 prefill、1 个 decode
- RPS (requests per second) → 每秒请求数（requests per second，RPS）
- session state management → 会话状态管理
- autoscaling → 自动伸缩（autoscaling）
- heterogeneous cluster / homogeneous deployment → 异构集群（heterogeneous cluster）/ 同构部署

## KV 缓存传输与互连

- KV cache → 保留 KV 缓存（KV cache）
- KV cache transfer / handoff → KV 缓存传输 / 移交（handoff）
- NIXL (NVIDIA Inference Xfer Library) → 保留
- GPUDirect RDMA / RDMA → 保留
- UCX → 保留
- zero-copy GPU-to-GPU transfer → 零拷贝 GPU 到 GPU 传输
- NVLink / NVSwitch / InfiniBand → 保留
- MNNVL (Multi-Node NVLink) → 保留
- SHARP → 保留
- AllGather / ReduceScatter / collectives → 保留 AllGather / ReduceScatter / 集合通信（collectives）
- interconnect → 互连（interconnect）
- prefix cache / prefix caching → 前缀缓存（prefix caching）

## 路由与调度策略

- routing policy / scheduling policy → 路由策略（routing policy）/ 调度策略
- disaggregated router → 分离式路由器（disaggregated router）
- offload → 卸载（offload）— 指把 prefill 交给 prefill 工作节点
- Round robin → 轮询（round robin）
- Least requests → 最少请求（least requests）
- prefix-aware routing → 前缀感知路由（prefix-aware routing）
- KV-aware / KV cache-aware routing → KV 感知路由（KV-aware routing）
- cache locality / cache hit → 缓存局部性（cache locality）/ 缓存命中（cache hit）
- routing factor → 路由因子（routing factor）
- latency cost / weight → 延迟代价（latency cost）/ 权重
- multifactor scoring → 多因子打分（multifactor scoring）
- prefill queue length → prefill 队列长度
- KV occupancy → KV 占用率（KV occupancy）
- memory bandwidth utilization → 内存带宽利用率
- QoS (quality of service) → 保留 QoS（服务质量，quality of service）
- early rejection → 早期拒绝（early rejection）
- priority → 优先级（priority）
- exponential moving average → 指数移动平均（exponential moving average）
- speculative decoding → 推测解码（speculative decoding）
- Dynamo GPU Planner → 保留

## 精度与并行

- tensor parallelism (TP) → 张量并行（tensor parallelism，TP）
- pipeline parallelism → 流水线并行（pipeline parallelism）
- TP=1 → 保留
- FP8 / FP4 / precision → 保留 FP8 / FP4 / 精度
- multi-headed attention (MHA) → 多头注意力（multi-headed attention，MHA）
- head dimension → 头维度（head dimension）
- arithmetic intensity → 算术强度（arithmetic intensity）
- fused kernels → 融合核函数（fused kernels）

## 剖析与调优（承接前章）

- Nsight Systems → 保留
- CUDA kernels / CUDA Graphs → 保留
- eager mode → 保留（即时模式）
- CUDA Graph capture → CUDA Graph 捕获
- GPU utilization → GPU 利用率
- --max-seq-len-to-capture / --max-num-seqs / --max-num-batched-tokens → 保留（vLLM 参数）

## 硬件/平台（保留原文）

- B200 / B300 / Blackwell / Blackwell Ultra → 保留
- Rubin CPX / Grace-Blackwell / Vera-Rubin → 保留
- Transformer Engine → 保留
- GPU / HBM / DRAM / Tensor Core → 保留
- vLLM / SGLang / NVIDIA Dynamo / LMCache → 保留
- DistServe → 保留（论文/系统名）
- MLPerf → 保留
- Llama 2 / Llama 3.1 → 保留
