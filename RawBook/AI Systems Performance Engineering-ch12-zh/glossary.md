# 术语表（glossary.md）— 第 12 章：动态调度、CUDA Graphs 与设备端发起的核函数编排

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具/命令名、API/函数名、Nsight 指标名、代码一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。承接第 6-11 章的既定译法。

## 章节主题

- Dynamic Scheduling, CUDA Graphs, and Device-Initiated Kernel Orchestration → 动态调度、CUDA Graphs 与设备端发起的核函数编排
- dynamic scheduling → 动态调度（dynamic scheduling）
- kernel orchestration / meta-scheduling → 核函数编排（kernel orchestration）/ 元调度（meta-scheduling）
- device-initiated / device-side / GPU-driven → 设备端发起（device-initiated）/ 设备侧（device-side）/ GPU 驱动（GPU-driven）
- host-driven / host-side → 主机驱动（host-driven）/ 主机侧（host-side）
- launch overhead → 启动开销（launch overhead）
- idle gap / idle time → 空闲间隙（idle gap）/ 空闲时间
- SM utilization / occupancy → SM 利用率 / 占用率（occupancy）
- warp → warp（线程束）— 首次可括注，后续用 warp
- back-to-back → 背靠背（back-to-back）
- runtime → 运行时（runtime）

## 动态调度与原子工作队列

- atomic work queue → 原子工作队列（atomic work queue）
- atomic counter / atomicAdd / atomic operation → 原子计数器（atomic counter）/ 保留 atomicAdd / 原子操作
- atomic transaction / contention / contention ratio → 原子事务 / 争用（contention）/ 争用率
- work stealing / load balancing → 工作窃取（work stealing）/ 负载均衡
- persistent kernel → 持久化核函数（persistent kernel）
- per-task runtime → 每任务运行时长
- globalIndex / work item → 保留 globalIndex / 工作项（work item）

## CUDA Graphs

- CUDA Graph / CUDA Graphs → CUDA Graph / CUDA Graphs — 保留
- graph capture / graph replay → 图捕获（graph capture）/ 图重放（graph replay）
- cudaStreamBeginCapture / cudaStreamEndCapture / cudaGraphLaunch / cudaGraphInstantiate → 保留
- cudaGraphExec / cudaGraphExecUpdate → 保留
- g.replay() / graph.replay() → 保留
- node / dependency → 节点（node）/ 依赖（dependency）
- memory pool / static memory pool → 内存池（memory pool）/ 静态内存池
- torch.cuda.graph_pool_handle() / torch.cuda.CUDAGraph → 保留
- pointer stability / pointer-stable → 指针稳定性（pointer stability）
- dynamic graph update → 动态图更新（dynamic graph update）
- device-initiated CUDA Graph launch → 设备端发起的 CUDA Graph 启动
- tail launch / sibling launch → 尾部启动（tail launch）/ 同级启动（sibling launch）
- conditional graph node / IF / WHILE node → 条件图节点（conditional graph node）/ IF、WHILE 节点
- body subgraph / nested → 主体子图（body subgraph）/ 嵌套（nested）
- in-kernel persistent scheduling → 核内持久化调度
- warm up / warmup → 预热（warm up）

## 动态并行

- dynamic parallelism (DP) → 动态并行（dynamic parallelism，DP）
- parent kernel / child kernel → 父核函数（parent kernel）/ 子核函数（child kernel）
- nested kernel launch → 嵌套核函数启动
- device launch / host launch → 设备端启动 / 主机端启动
- data locality → 数据局部性（data locality）
- cache eviction / DRAM refetch → 缓存逐出（cache eviction）/ DRAM 重取

## 多 GPU 编排（NVSHMEM）

- NVSHMEM → 保留
- one-sided communication → 单边通信（one-sided communication）
- GPU-to-GPU memory sharing → GPU 间内存共享
- symmetric memory / PGAS → 对称内存（symmetric memory）/ PGAS
- nvshmem_put / nvshmem_get / nvshmemx → 保留
- put / get (operation) → put / get 操作
- collective / all-reduce / NCCL → 集合（collective）/ 保留 all-reduce、NCCL
- ncclCommRegister / ncclCommInitRank → 保留
- fine-grained → 细粒度（fine-grained）
- overlap → 重叠（overlap）
- N-GPU scaling → N-GPU 扩展
- cluster node → 集群节点（cluster node）
- NVLink / NVSwitch / InfiniBand / RDMA → 保留

## Roofline 引导的调度

- Roofline-guided scheduling → Roofline 引导的调度
- arithmetic intensity → 算术强度（arithmetic intensity）
- memory bound / compute bound → 访存受限（memory bound）/ 计算受限（compute bound）
- hardware ceiling / roof → 硬件天花板（ceiling）/ 屋顶（roof）
- FLOPs/byte → 保留

## 硬件/平台（保留原文）

- Blackwell / Hopper / GB200 / NVL72 → 保留
- Tensor Core / TMA / DSMEM / CTA → 保留（CTA 可括注“线程块”）
- HBM / DRAM / L2 → 保留
- PyTorch / TorchInductor / vLLM → 保留
- Nsight Compute / Nsight Systems → 保留
