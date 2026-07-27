# 术语表（glossary.md）— 第 4 章：分布式网络通信调优

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具/命令名、代码一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。

## 章节主题

- Tuning Distributed Networking Communication → 分布式网络通信调优
- Overlapping communication and computation / pipelining → 通信与计算重叠 / 流水线化（pipelining）
- overlap → 重叠
- goodput (useful throughput) → 有效吞吐（goodput）

## 通信与集合操作

- collective communication → 集合通信
- point-to-point (P2P) communication → 点对点（point-to-point，P2P）通信
- all-reduce / all-gather / broadcast / reduce / reduce-scatter → 保留（集合通信操作名，如 all-reduce 不译）
- send / recv → 保留
- gradient accumulation → 梯度累积
- bucket / bucketing → 分桶（bucketing）
- compression / quantization / sparsification → 压缩 / 量化 / 稀疏化
- synchronization / barrier semantics → 同步 / 屏障语义（barrier）
- nonblocking / asynchronous → 非阻塞 / 异步
- CUDA stream → CUDA 流
- straggler → 掉队节点（straggler）

## 库、协议与硬件（保留原文）

- NVIDIA Magnum IO → 保留
- NCCL → 保留
- NIXL (NVIDIA Inference Xfer Library) → 保留（NVIDIA Inference Xfer Library，NIXL）
- RDMA (Remote Direct Memory Access) → 保留（远程直接内存访问，RDMA）
- GPUDirect RDMA / GPUDirect Storage (GDS) → 保留
- RoCE (RDMA over Converged Ethernet) → 保留
- InfiniBand / Ethernet / PCIe → 保留
- NVLink / NVSwitch / NVLink-C2C → 保留
- NVL72 → 保留
- SHARP (Scalable Hierarchical Aggregation and Reduction Protocol) → 保留（在网聚合协议）
- UCX / HPC-X / NVSHMEM / MPI / SHMEM → 保留
- NIC (network interface card) → 网卡（network interface card，NIC）
- multi-rail / rail → 多轨（multi-rail）/ 轨（rail）
- Grace Hopper / Grace Blackwell Superchip → 保留
- NVIDIA Dynamo → 保留
- BlueField / SNAP → 保留

## NCCL 相关

- topology awareness → 拓扑感知
- communicator → 通信器（communicator）
- Ring / Tree / NVLSTree / CollTree / CollNet / PAT (Parallel Aggregated Tree) → 保留（NCCL 算法名）
- ring all-reduce → 环形 all-reduce（ring all-reduce）
- hierarchical / intranode / internode → 分层 / 节点内（intranode）/ 节点间（internode）
- rank / world_size → 保留
- in-network aggregation / reduction → 在网聚合 / 归约
- zero-copy / user buffers → 零拷贝（zero-copy）/ 用户缓冲区（user buffers）
- persistent buffers / registration → 持久化缓冲区 / 注册（registration）

## 并行策略

- data parallelism (DP) → 数据并行（data parallelism，DP）
- DDP (Distributed Data Parallel) → 保留
- FSDP (Fully Sharded Data Parallelism) → 保留
- Horovod → 保留
- minibatch / batch size → 小批量（minibatch）/ 批大小（batch size）
- effective batch size → 有效批大小
- convergence → 收敛

## 推理与 NIXL

- disaggregated inference / serving → 分离式推理 / 分离式服务（disaggregated serving）
- prefill / decode → 预填充（prefill）/ 解码（decode）
- KV cache → KV 缓存
- TTFT (time to first token) → 首 token 时间（time to first token，TTFT）
- KV cache offloading → KV 缓存卸载（offloading）
- memory hierarchy → 内存层级
- object store (e.g., Amazon S3) → 对象存储（如 Amazon S3）
- callback → 回调（callback）
- interconnect routing → 互连路由

## 性能与通用（沿用 ch1/ch2/ch3）

- throughput / latency / bandwidth / utilization / bottleneck / overhead → 吞吐量 / 延迟 / 带宽 / 利用率 / 瓶颈 / 开销
- tail latency → 尾延迟
- training / inference → 训练 / 推理
- distributed training → 分布式训练
- data locality → 数据局部性
- prefetch / prefetching → 预取
- GPU / CPU / HBM / SM (streaming multiprocessor) → 保留（SM 首次括注 streaming multiprocessor，流式多处理器）
- (hardware-software) codesign → （软硬件）协同设计
- ultrascale → 超大规模
- speed of light（硬件极限比喻）→ “光速”（物理极限，保留引号）

## 固定表达

- drop-in replacement → 即插即用替代品
- best practices → 最佳实践
- under the hood → 底层 / 幕后
- sweet spot → 最佳平衡点
