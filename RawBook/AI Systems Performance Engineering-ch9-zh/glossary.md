# 术语表（glossary.md）— 第 9 章：提升 CUDA 核函数效率与算术强度

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具/命令名、API/函数名、Nsight 指标名、代码一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。承接第 6/7/8 章的既定译法。

## 章节主题

- Increasing CUDA Kernel Efficiency and Arithmetic Intensity → 提升 CUDA 核函数效率与算术强度
- arithmetic intensity / operational intensity → 算术强度（arithmetic intensity）/ 运算强度（operational intensity）
- FLOPS per byte / FLOPs/byte → 每字节 FLOPS / FLOPs/byte — 保留原文比率写法
- Roofline model / roofline chart / Roofline analysis → Roofline 模型 / roofline 图 / Roofline 分析 — 保留 Roofline
- memory roof / compute roof → 内存屋顶（memory roof）/ 计算屋顶（compute roof）
- memory bound / compute bound / latency bound → 访存受限（memory bound）/ 计算受限（compute bound）/ 延迟受限（latency bound）
- ALU throughput → ALU 吞吐
- reuse factor / data reuse → 复用因子（reuse factor）/ 数据复用
- FMA / multiply-accumulate → FMA（乘加）/ 乘加运算（multiply-accumulate）

## 分块与预取

- tiling (aka chunking or blocking) → 分块（tiling，又称 chunking 或 blocking）
- multilevel tiling → 多级分块（multilevel tiling）
- microtiling / microtiles → 微分块（microtiling）/ 微分块块（microtiles）
- software prefetching → 软件预取（software prefetching）
- double buffering → 双缓冲（double buffering）
- vectorized types (float4, half2) → 向量化类型（vectorized types，如 float4、half2）— 类型名保留原文
- shared memory (SMEM) → 共享内存（shared memory，SMEM）
- registers / register file → 寄存器（register）/ 寄存器堆（register file）
- SRAM / static random-access memory → SRAM（静态随机存取存储器）
- coalesce / coalesced → 合并访问（coalesce）
- pad/swizzle / memory-bank conflicts → 填充/交错（pad/swizzle）/ 内存 bank 冲突
- cudaMemPrefetchAsync() / unified memory → 保留 API 名 / 统一内存（unified memory）
- page migration / page fault → 页迁移（page migration）/ 缺页（page fault）

## 线程块簇

- thread block clusters → 线程块簇（thread block clusters）
- Cooperative Groups → 协作组（Cooperative Groups）
- distributed shared memory (DSMEM) → 分布式共享内存（distributed shared memory，DSMEM）
- CTA (thread block) → CTA（线程块）— 保留 CTA
- CGA → 保留 CGA
- thread block pairs → 线程块对（thread block pairs）
- TMA multicast → TMA 多播（multicast）
- Tensor Memory Accelerator (TMA) → Tensor Memory Accelerator（TMA）— 保留

## 核函数融合

- kernel fusion / fuse → 核函数融合（kernel fusion）/ 融合
- elementwise operations / pointwise operations → 逐元素操作（elementwise operations）/ 逐点操作（pointwise operations）
- vertical fusion / horizontal fusion → 纵向融合（vertical fusion）/ 横向融合（horizontal fusion）
- loop unrolling → 循环展开（loop unrolling）
- register pressure / register spilling → 寄存器压力 / 寄存器溢出（register spilling）
- occupancy → 占用率（occupancy）
- L2-normalize → L2 归一化
- just-in-time (JIT) compiler / graph optimizer → 即时编译器（JIT）/ 图优化器

## 结构化稀疏

- structured sparsity / 2:4 structured sparsity → 结构化稀疏（structured sparsity）/ 2:4 结构化稀疏
- Sparse Tensor Cores → 保留 Sparse Tensor Cores（稀疏 Tensor Core）
- cuSPARSELt → 保留
- pruning → 剪枝（pruning）
- sparse GEMM → 稀疏 GEMM
- to_sparse_semi_structured() → 保留 API 名
- batching → 批处理（batching）
- calibrate / fine-tune → 校准（calibrate）/ 微调（fine-tune）

## 重算与内存权衡

- recomputation → 重算（recomputation）
- activation checkpointing → 激活检查点（activation checkpointing）
- recompute on demand → 按需重算
- activation tensors → 激活张量（activation tensors）

## PyTorch 与算术强度

- torch.compile / TorchInductor → 保留
- torch.matmul / torch.nn.Linear / torch.nn.functional → 保留
- scaled_dot_product_attention (SDPA) → 保留 API 名 — 注：原文写作 SPDA，系笔误，正确缩写为 SDPA；译文保留原文 SPDA 并可括注（应为 SDPA）
- FlashAttention / memory-efficient / cuDNN backend → 保留 FlashAttention / memory-efficient 后端 / cuDNN 后端
- torch.nn.attention.sdpa_kernel(SDPBackend.FLASH_ATTENTION) → 保留
- cuDNN / cuBLAS / cuBLASLt → 保留
- nested tensors / ragged tensors / NestedTensor → 嵌套张量（nested tensors）/ 不规则张量（ragged tensors）/ 保留 NestedTensor
- padding / zero-padding → 填充（padding）/ 零填充

## 混合精度与 Tensor Core

- mixed precision / automatic mixed precision (AMP) → 混合精度（mixed precision）/ 自动混合精度（automatic mixed precision，AMP）
- Tensor Cores → 保留 Tensor Core（张量核心）— 首次可括注，后续用 Tensor Core
- TMEM / Tensor Memory → Tensor Memory（TMEM，张量内存）— 保留 TMEM
- TMA → Tensor Memory Accelerator（TMA）
- tcgen05 / tcgen05.mma / tcgen05.ld / tcgen05.st → 保留（PTX 指令）
- MMA (matrix multiply-accumulate) → MMA（矩阵乘加，matrix multiply-accumulate）
- MMA fragment APIs → MMA 片段 API（fragment APIs）
- Parallel Thread Execution (PTX) → 保留 PTX（Parallel Thread Execution）
- TF32 / FP32 / FP16 / BF16 / FP8 / FP4 / INT8 → 保留（精度格式名）
- half-precision → 半精度（half-precision）
- NVFP4 / MXFP8 / MXFP4 / OCP MX → 保留（微缩放格式名）
- micro-scaling / microscaling / per-microblock scaling → 微缩放（micro-scaling）/ 按微块缩放
- autocast / GradScaler / loss scaling → 保留 autocast / GradScaler / 损失缩放（loss scaling）
- torch.set_float32_matmul_precision('high') → 保留
- exponent / mantissa → 指数位（exponent）/ 尾数位（mantissa）
- underflow / overflow → 下溢（underflow）/ 上溢（overflow）
- accumulator / accumulation → 累加器（accumulator）/ 累加
- DP4A (SIMD dot-product) → 保留 DP4A（SIMD 点积指令）
- MAC (multiply-accumulate) → MAC（乘加）
- petaFLOPS / teraFLOPS / TFLOPS / PFLOPS → 保留（petaFLOPS、teraFLOPS 等）
- dense / sparse → 稠密（dense）/ 稀疏（sparse）
- Transformer Engine → 保留
- Speed of Light (analysis) → 保留 Speed of Light（分析）
- Memory Throttle → 保留 Memory Throttle
- Warp Stall metrics → warp 停顿指标（Warp Stall）
- quantization / quantization error → 量化（quantization）/ 量化误差
- TorchAO → 保留

## CUTLASS

- CUTLASS → 保留
- GEMM → 保留 GEMM（通用矩阵乘）
- warp specialization → warp 专门化（warp specialization）
- templated call / template → 模板化调用 / 模板（template）
- hand-tuned MMA kernel → 手工调优的 MMA 核函数
- cp.async / cp.async.bulk.tensor → 保留（PTX 指令）
- bias-add / activation fusion → bias-add（偏置加）/ 激活融合
- drop-in → 直接替换式（drop-in）
- Sm100 / OpClassTensorOp / half_t / RowMajor / ColumnMajor → 保留（CUTLASS 模板参数）

## 内联 PTX 与 SASS

- inline PTX / inline assembly → 内联 PTX（inline PTX）/ 内联汇编
- SASS (Streaming Assembler) → 保留 SASS（NVIDIA 汇编语言）
- microoptimization / micro-optimization → 微优化（microoptimization）
- asm() / asm volatile → 保留
- cache hints / cache modifiers → 缓存提示（cache hints）/ 缓存修饰符
- ld.global.cg / .cg (cache global) / .ca / .L1 / .L2 → 保留（PTX 缓存修饰符）
- mad.wide.u32 → 保留
- memory fences / membar.sys / __threadfence_system() → 内存栅栏（memory fences）/ 保留 membar.sys、__threadfence_system()
- special registers / %smid / SM ID → 特殊寄存器 / 保留 %smid / SM ID
- fast math intrinsics / __sinf() / __prefetch_async → 保留（快速数学内建函数）
- instruction scheduling → 指令调度（instruction scheduling）
- load-to-use latency → load-to-use 延迟
- forward compatible / forward-compatible → 前向兼容（forward compatible）
- CUB / Thrust → 保留

## DeepSeek 案例

- DeepSeek / DeepEP → 保留
- expert parallelism / expert-parallel all-to-all → 专家并行（expert parallelism）/ 专家并行 all-to-all
- ld.global.nc.l1::no_allocate.l2::256b → 保留（PTX 指令）
- noncoherent (nc) → 非一致（noncoherent，nc）
- out-of-doc / bespoke → 未见于文档（out-of-doc）/ 定制（bespoke）
- H800 / Hopper → 保留
- DISABLE_AGGRESSIVE_PTX_INSTRS=1 → 保留（编译期开关）
- working set → 工作集（working set）

## 硬件/平台（一律保留原文）

- Blackwell / B200 / B300 (Ultra) / Hopper / Ampere → 保留
- Grace Blackwell / GB200 / GB300 / NVL72 → 保留
- NVLink / NVSwitch / NVLink-C2C / NV-HBI → 保留
- HBM / DRAM / L1 / L2 cache / SM (streaming multiprocessor) → 保留 HBM/DRAM/L1/L2；SM → 流多处理器（SM）
- CPU-GPU superchip → CPU-GPU 超级芯片（superchip）
- reticle-limited die → 受限于光罩尺寸的裸片
