# 术语表（glossary.md）— 第 7 章：GPU 内存访问模式的剖析与调优

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具/命令名、API/函数名、代码一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。

## 章节主题

- Profiling and Tuning GPU Memory Access Patterns → GPU 内存访问模式的剖析与调优
- memory access pattern → 内存访问模式
- memory hierarchy → 内存层级（memory hierarchy）
- memory-bound / compute-bound → 访存受限（memory-bound）/ 计算受限（compute-bound）
- arithmetic intensity → 算术强度（arithmetic intensity）
- bandwidth utilization → 带宽利用率
- wasted / redundant bandwidth → 浪费的 / 冗余的带宽

## 合并访问与事务

- coalesced / uncoalesced memory access → 合并（coalesced）/ 非合并（uncoalesced）内存访问
- coalescing / memory coalescer → 合并（coalescing）/ 内存合并器（memory coalescer）
- global memory → 全局内存（global memory）
- cache line → 缓存行（cache line）
- sector (32-byte) → 扇区（sector）
- 128-byte transaction / 128-byte line → 128 字节事务 / 128 字节缓存行
- aligned / misaligned / alignment → 对齐 / 未对齐 / 对齐（alignment）
- 128-byte aligned → 128 字节对齐
- stride / strided access → 跨步（stride）/ 跨步访问
- contiguous → 连续
- Global Memory Load Efficiency → 全局内存加载效率（Global Memory Load Efficiency）
- DRAM throughput → DRAM 吞吐
- sectors per request → 每次请求的扇区数
- SM Active % → SM 活跃百分比（SM Active %）
- Memory Workload Analysis → 内存工作负载分析（Memory Workload Analysis）

## 数据布局与向量化

- Array of Structures (AoS) → 结构体数组（Array of Structures，AoS）
- Structure of Arrays (SoA) → 数组结构体（Structure of Arrays，SoA）
- vectorized memory access → 向量化内存访问
- vector load / vector type → 向量加载 / 向量类型
- float4 / float2 / int4 → 保留（向量类型）
- scalar load → 标量加载
- 32-byte (256-bit) load/store → 32 字节（256 位）加载/存储
- warp-wide → warp 级（warp-wide）
- interthread / intrathread → 线程间（interthread）/ 线程内（intrathread）

## 分块与共享内存

- tiling / tile → 分块（tiling）/ 瓦片（tile）
- data reuse → 数据复用
- shared memory (SMEM) → 共享内存（shared memory，SMEM）
- tiled kernel / naive kernel → 分块核函数 / 朴素核函数（naive kernel）
- matrix multiply / matmul / GEMM → 矩阵乘法 / GEMM
- __shared__ → 保留
- __syncthreads() → 保留
- register pressure / register spilling → 寄存器压力 / 寄存器溢出（register spilling）
- occupancy → 占用率（occupancy）

## Bank 冲突

- bank conflict → bank 冲突（bank conflict）
- shared-memory bank → 共享内存 bank
- 2-way / 16-way bank conflict → 2 路 / 16 路 bank 冲突
- padding → 填充（padding）
- swizzling / index swizzling → 混洗（swizzling）/ 索引混洗
- broadcast → 广播（broadcast）
- transpose → 转置（transpose）
- CUTLASS → 保留（库名）

## Warp Shuffle 与同步

- warp shuffle intrinsics → warp shuffle 内建函数（warp shuffle intrinsics）
- __shfl_sync / __shfl_down_sync / __shfl_xor_sync → 保留
- warp-level reduction → warp 级归约（reduction）
- lane / lane index → 通道（lane）/ 通道索引
- explicit synchronization → 显式同步
- cooperative groups → 保留（cooperative groups）

## 只读数据缓存

- read-only data cache → 只读数据缓存（read-only data cache）
- __restrict__ → 保留
- __ldg() intrinsic → 保留（__ldg 内建函数）
- texture object / surface object → 纹理对象（texture object）/ 表面对象（surface object）
- texture / surface reference → 纹理引用 / 表面引用
- cudaTextureObject_t / tex1Dfetch / tex2D → 保留
- constant memory / __constant__ → 常量内存（constant memory）
- L1 cache / L2 cache → 保留 L1 缓存 / L2 缓存
- L2 read throughput → L2 读吞吐
- spatial locality → 空间局部性
- cache hit rate → 缓存命中率

## 异步预取与 TMA

- Asynchronous Memory Prefetching → 异步内存预取
- Tensor Memory Accelerator (TMA) → 保留（Tensor Memory Accelerator，TMA）
- prefetch / prefetching → 预取
- double buffering → 双缓冲（double buffering）
- ping-pong / ping-ponging → 乒乓（ping-pong）
- cuda::memcpy_async → 保留
- CUDA Pipeline API / cuda::pipeline → 保留（CUDA Pipeline API）
- bulk copy / bulk transfer → 批量拷贝 / 批量传输
- multicast → 多播（multicast）
- thread block cluster → 线程块簇（thread block cluster）
- DMA engine → DMA 引擎
- overlap (transfer with compute) → （传输与计算的）重叠
- codesign / hardware-software codesign → 软硬件协同设计（codesign）

## 硬件与工具（保留原文）

- SM (streaming multiprocessor) → 保留（streaming multiprocessor，流式多处理器，SM）
- warp / thread / thread block / grid → warp（线程束）/ 线程 / 线程块 / 网格
- Blackwell / B200 / B300 / Hopper → 保留
- Grace Blackwell / GB200 / GB300 / Superchip → 保留
- Tensor Core → 保留
- HBM / HBM3e → 保留
- HBM3e bandwidth → HBM3e 带宽
- unified memory → 统一内存（unified memory）
- CUDA C++ / CUDA 13 → 保留
- PyTorch / TorchInductor / torch.compile → 保留
- cuDNN / cuBLAS → 保留
- Nsight Compute (ncu) / Nsight Systems (nsys) → 保留
- Nsight Compute metrics → Nsight Compute 指标
- GFLOPS / GB/s / TB/s / ms → 保留
- ALU → 保留

## CUDA 通用（沿用 ch6）

- kernel → 核函数（kernel）
- host / device → 主机（host）/ 设备（device）
- launch / kernel launch → 启动 / 核函数启动
- blockIdx / threadIdx / blockDim / gridDim → 保留
- <<< >>> → 保留（<<< >>>，三尖括号/chevron 语法）
- cudaMalloc / cudaMemcpy / cudaMemcpyAsync → 保留
- stream → 流（stream）
- intrinsic → 内建函数（intrinsic）
- profiling → 性能剖析（profiling）
- benchmark → 基准测试（benchmark）
- autotuning → 自动调优（autotuning）

## 固定表达（沿用前几章）

- before-and-after (example) → 前后对比（示例）
- rule of thumb → 经验法则
- under the hood → 底层 / 幕后
- best practices → 最佳实践
- out of the box → 开箱即用
- for free → “免费”（无需额外代价）
- textbook example → 教科书式的范例
