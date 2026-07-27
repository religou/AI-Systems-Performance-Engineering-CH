# 术语表（glossary.md）— 第 3 章：面向 GPU 环境的操作系统、Docker 与 Kubernetes 调优

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具/命令名、代码一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。

## 操作系统与 CPU

- operating system (OS) → 操作系统（OS）
- NUMA (Non-Uniform Memory Access) → NUMA（非统一内存访问）
- NUMA node → NUMA 节点
- CPU pinning / affinity → CPU 绑定（pinning）/ 亲和性（affinity）
- thread / core → 线程 / 核心
- context switch → 上下文切换
- hugepages / Transparent Hugepages (THP) → 大页 / 透明大页（THP）
- page-locked / pinned memory → 页锁定 / 固定内存（pinned memory）
- pageable memory → 可分页内存
- swapping / swap → 交换（swap）
- virtual memory → 虚拟内存
- filesystem caching / write-back → 文件系统缓存 / 回写（write-back）
- C-states / CPU frequency → C-state / CPU 频率
- interrupt affinity / scheduler → 中断亲和性 / 调度器
- memory allocator → 内存分配器
- NIC (network interface card) → 网卡（network interface card，NIC）
- sysfs / cgroups → 保留

## GPU 驱动与运行时

- GPU driver → GPU 驱动
- CUDA Toolkit / Runtime → CUDA 工具包 / 运行时
- forward/backward compatibility → 向前 / 向后兼容
- persistence mode → 持久化模式（persistence mode）
- MPS (Multi-Process Service) → 多进程服务（Multi-Process Service，MPS）
- MIG (Multi-Instance GPU) → 多实例 GPU（Multi-Instance GPU，MIG）
- GPU clock speeds → GPU 时钟频率
- ECC (Error Correcting Code) → ECC（纠错码）
- memory oversubscription / fragmentation → 内存超额订阅 / 碎片化
- out-of-memory (OOM) / OOM Killer → 内存不足（OOM）/ OOM Killer
- NVML / nvidia-smi → 保留

## 库与框架（保留原文）

- CUDA, cuDNN, NCCL, PyTorch, JAX, Keras, TensorFlow, Triton, CUTLASS, cuPyNumeric, RAPIDS, Warp
- TorchDynamo, TorchInductor, AOTAutograd, XLA
- DataLoader（PyTorch，保留）；worker_init_fn / pin_memory / prefetch_factor / num_workers 等参数保留原文

## 容器与编排

- container / container runtime → 容器 / 容器运行时
- Docker → 保留
- NVIDIA Container Toolkit / Container Runtime → 保留
- overlay filesystem → 叠加文件系统（overlay）
- image (container) → 镜像
- Kubernetes (K8s) → 保留
- NVIDIA GPU Operator / device plugin → 保留（设备插件）
- Topology Manager → 拓扑管理器（Topology Manager）
- SLURM → 保留
- orchestration / scheduling → 编排 / 调度
- jitter → 抖动（jitter）
- resource guarantees / isolation → 资源保障 / 隔离
- multitenant → 多租户

## 性能与通用（沿用 ch1/ch2）

- throughput / latency / bandwidth / utilization / bottleneck / overhead → 吞吐量 / 延迟 / 带宽 / 利用率 / 瓶颈 / 开销
- prefetch / prefetching → 预取
- batching → 批处理
- overlap (communication/computation) → 重叠（通信/计算）
- data locality → 数据局部性
- training / inference → 训练 / 推理
- distributed training → 分布式训练
- RDMA / InfiniBand / GPUDirect → 保留
- MTU / TCP buffers → 保留
- (hardware-software) codesign → （软硬件）协同设计
- GPU / CPU / HBM / NVLink / NVSwitch → 保留
- DDP (DistributedDataParallel) → 保留

## 固定表达

- drop-in replacement → 即插即用替代品
- best practices → 最佳实践
- ready state → 就绪状态
