# 术语表（glossary.md）— 第 5 章：基于 GPU 的存储 I/O 优化

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具/命令名、代码一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。

## 章节主题

- GPU-Based Storage I/O Optimizations → 基于 GPU 的存储 I/O 优化
- storage / data pipeline → 存储 / 数据流水线（data pipeline）
- data locality → 数据局部性
- input pipeline → 输入流水线
- goodput (useful throughput) → 有效吞吐（goodput）
- overlap I/O with compute → I/O 与计算重叠

## 存储与文件系统（保留原文/命令）

- NVMe / NVMe SSD → 保留
- NVMe over Fabrics (NVMe-oF) → 保留（NVMe over Fabrics，NVMe-oF）
- SSD / HDD → 保留
- PCIe → 保留
- RAID 0 → 保留
- filesystem → 文件系统
- parallel filesystem → 并行文件系统
- distributed filesystem → 分布式文件系统
- object store / object storage → 对象存储
- page cache → 页缓存（page cache）
- read-ahead → 预读（read-ahead）
- sequential / random read → 顺序读 / 随机读
- IOPS / throughput / latency / bandwidth → 保留 IOPS / 吞吐量 / 延迟 / 带宽
- checkpoint / checkpointing → 检查点 / 检查点保存（checkpointing）
- sharding / shard → 分片（sharding）/ 分片
- replication / compression → 复制（replication）/ 压缩
- staging → 暂存（staging）
- Lustre / GPFS / BeeGFS / WekaFS / IBM Storage Scale / VAST → 保留
- XFS / NFS / EFS / FSx → 保留
- Amazon S3 / FSx for Lustre → 保留
- s5cmd / aws s3 cp / blockdev / numactl → 保留（命令名）
- jumbo frames / MTU → 保留 / MTU
- blk-mq / mq-deadline / CFQ / BFQ / none scheduler → 保留（I/O 调度器名）
- io_uring / pread() / O_DIRECT → 保留
- noatime / rsize / wsize / actimeo / lookupcache → 保留（挂载选项）

## GPU 存储加速（保留原文）

- NVIDIA GDS (GPUDirect Storage) → 保留（GPUDirect Storage，GDS）
- GPUDirect RDMA → 保留
- RDMA (Remote Direct Memory Access) → 保留（远程直接内存访问，RDMA）
- RoCE → 保留
- InfiniBand → 保留
- bounce buffer / host bounce buffer → 中转缓冲区（bounce buffer）
- zero-copy → 零拷贝（zero-copy）
- pinned memory / page-locked → 固定内存（pinned memory）/ 页锁定
- DMA (direct memory access) → 保留（直接内存访问，DMA）
- cuda-checkpoint → 保留（命令名）
- gdsio → 保留（工具名）
- Magnum IO / NCCL / NIXL / NVSHMEM → 保留
- Grace Hopper / Grace Blackwell Superchip / GB200 / GB300 / NVL72 → 保留
- Blackwell / Rubin → 保留
- unified memory → 统一内存（unified memory）

## 数据流水线与框架（保留原文）

- DeepSeek / Fire-Flyer File System (3FS) → 保留（Fire-Flyer File System，3FS）
- FoundationDB / FUSE → 保留
- PyTorch / TensorFlow / JAX → 保留
- DataLoader / Dataset / IterableDataset / DistributedSampler → 保留（PyTorch API）
- num_workers / pin_memory / non_blocking / persistent_workers / prefetch_factor → 保留（参数名）
- collate_fn / collating → 保留 / 整理（collate）
- WebDataset / TFRecord / Arrow / Parquet → 保留（数据格式）
- NVIDIA DALI (Data Loading Library) → 保留（Data Loading Library，DALI）
- NVIDIA NeMo Curator → 保留
- data loader / data loading → 数据加载器 / 数据加载
- worker → 工作进程（worker）
- prefetch / prefetching → 预取
- multimodal → 多模态
- GIL (Global Interpreter Lock) → 保留（全局解释器锁，GIL）

## 并行与训练（沿用 ch4）

- data parallelism (DP) → 数据并行（data parallelism，DP）
- DDP (DistributedDataParallel) → 保留
- FSDP → 保留
- minibatch / batch size / effective batch size → 小批量（minibatch）/ 批大小（batch size）/ 有效批大小
- all-reduce / collective communication → 保留 all-reduce / 集合通信
- epoch → 轮次（epoch）
- convergence → 收敛

## 性能与通用（沿用 ch1-ch4）

- throughput / latency / bandwidth / utilization / bottleneck / overhead → 吞吐量 / 延迟 / 带宽 / 利用率 / 瓶颈 / 开销
- tail latency → 尾延迟
- training / inference → 训练 / 推理
- distributed training → 分布式训练
- prefetch → 预取
- GPU / CPU / HBM / SM (streaming multiprocessor) / NIC → 保留（SM 首次括注 streaming multiprocessor，流式多处理器；NIC 首次括注 network interface card，网卡）
- NUMA / IRQ affinity → 保留 NUMA / IRQ 亲和性
- roofline / roofline model → Roofline 模型（保留 Roofline）
- compute-bound / memory-bound / communication-bound → 计算受限 / 访存受限 / 通信受限
- (hardware-software) codesign → （软硬件）协同设计
- ultrascale → 超大规模
- speed of light（硬件极限比喻）→ “光速”（物理极限，保留引号）
- profiling → 性能剖析（profiling）
- Nsight Systems / Nsight Compute / PyTorch profiler → 保留

## 固定表达（沿用前几章）

- drop-in replacement → 即插即用替代品
- best practices → 最佳实践
- under the hood → 底层 / 幕后
- sweet spot → 最佳平衡点
- out of the box → 开箱即用
- rule of thumb → 经验法则
