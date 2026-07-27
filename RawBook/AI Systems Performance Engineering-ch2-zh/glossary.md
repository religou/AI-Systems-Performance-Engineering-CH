# 术语表（glossary.md）— 第 2 章：AI 系统硬件概览

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具名一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。

## 芯片与封装

- superchip → 超级芯片
- Grace CPU / Grace Hopper (GH200) / Grace Blackwell (GB200) Superchip → 保留（Superchip 译“超级芯片”）
- Blackwell / Hopper / Blackwell Ultra (B300) / B200 / H100 → 保留
- die → 裸片（die） — 半导体语境
- dual-die GPU → 双裸片 GPU（dual-die）
- multichip module (MCM) → 多芯片模块（MCM）
- chiplet → 芯粒
- transistor → 晶体管
- ARM Neoverse V2 → 保留
- NV-HBI (High-Bandwidth Interface) → 保留（高带宽接口）

## 内存与缓存

- HBM / HBM3e / HBM3 → 保留（高带宽内存）
- LPDDR5X → 保留
- unified memory → 统一内存
- Unified CPU-GPU Memory / Extended GPU Memory (EGM) → 统一 CPU-GPU 内存 / 扩展 GPU 内存（EGM）
- cache-coherent / cache coherency → 缓存一致 / 缓存一致性
- coherent interconnect → 一致性互连
- unified (virtual) address space → 统一（虚拟）地址空间
- memory hierarchy → 内存层级
- L2 / L3 cache → L2 / L3 缓存
- HBM stack / 8-Hi stack → HBM 堆栈 / 8 层堆栈（8-Hi）
- bandwidth → 带宽
- prefetch → 预取

## 互连与网络

- NVLink / NVLink 5 / NVLink-C2C (chip-to-chip) → 保留（C2C 即“片间”）
- NVSwitch → 保留
- die-to-die interconnect → 裸片间互连
- PCIe (Gen5/Gen6 x16) → 保留
- SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) → 保留
- in-network aggregation → 网络内聚合
- interconnect → 互连
- fabric → 网络结构 / 交换网络
- co-packaged optics → 共封装光学（co-packaged optics）
- point-to-point → 点对点

## 计算单元与精度

- Tensor Core → Tensor Core（张量核心）
- Transformer Engine (TE) → Transformer 引擎（TE）
- streaming multiprocessor (SM) → 流式多处理器（SM）
- thread / warp → 线程 / 线程束（warp）
- kernel / GPU kernel → 内核 / GPU 内核
- FP8 / FP4 / NVFP4 / FP16 / BF16 / FP64 → 保留
- reduced/low precision → 降精度 / 低精度
- mixed precision → 混合精度
- exaFLOPS / petaFLOPS → 保留
- structured sparsity → 结构化稀疏

## 机架、供电与散热

- rack → 机架
- tray / compute tray (1U) → 托盘 / 计算托盘（1U）
- NVL72 / GB200 NVL72 / GB300 NVL72 → 保留
- preintegrated rack appliance → 预集成机架一体机
- liquid cooling / air cooling → 液冷 / 风冷
- power distribution → 供电分配
- compute density → 计算密度
- kW / MW → 保留

## 系统与角色（沿用第 1 章）

- AI systems performance engineer(ing) → AI 系统性能工程师 / AI 系统性能工程
- (hardware-software) codesign → （软硬件）协同设计
- mechanical sympathy → 机械同理心（mechanical sympathy）
- throughput / latency / utilization / bottleneck → 吞吐量 / 延迟 / 利用率 / 瓶颈
- training / inference → 训练 / 推理
- large language model (LLM) → 大语言模型（LLM）
- mixture of experts (MoE) → 混合专家（MoE）
- ultrascale → 超大规模
- multi-trillion-parameter → 数万亿参数
- ROI → 投资回报（ROI）
- AI factory → AI 工厂
- out of memory (OOM) → 内存不足（OOM）
- NVMe SSD / RDMA → 保留

## 工具/框架（保留原文）

- CUDA, CUTLASS, Triton, PyTorch, vLLM, TensorRT

## 固定表达

- AI supercomputer-in-a-box / in a rack → 盒装 AI 超级计算机 / 机架级 AI 超级计算机
- speed of light（NVIDIA 语境）→ 光速（指理论硬件上限）
- game changer → 颠覆性变革 / 革命性突破
