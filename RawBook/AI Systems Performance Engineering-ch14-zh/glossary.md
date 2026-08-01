# 术语表（glossary.md）— 第 14 章：PyTorch 编译器、OpenAI Triton 与 XLA 后端

供各分块 subagent 共用，确保术语全局一致。格式：原文 → 推荐译法 — 备注。
公司名、人名、产品/硬件型号、库/工具/命令名、API/函数名、环境变量、代码一律保留原文；术语/缩写首次出现在译文后括注原文，全书只标注一次。承接第 6–13 章的既定译法（尤其第 13 章的编译器/Triton 术语）。

## 章节主题

- PyTorch Compiler, OpenAI Triton, and XLA Backends → PyTorch 编译器、OpenAI Triton 与 XLA 后端
- compiler backend → 编译器后端（compiler backend）
- deep dive → 深入剖析
- code generation / codegen → 代码生成（code generation）

## PyTorch 编译器栈

- PyTorch Compiler / torch.compile → PyTorch 编译器 / 保留 torch.compile
- TorchDynamo → 保留
- bytecode capture → 字节码捕获（bytecode capture）
- graph extraction → 图提取（graph extraction）
- FX graph / FX → FX 图（FX graph）— 保留 FX
- AOT Autograd → 保留（提前自动微分）
- forward/backward pass → 前向/反向传播
- PrimTorch / Prims / PrimTorch IR → 保留
- IR (intermediate representation) → 中间表示（intermediate representation，IR）
- operator set / decomposition → 算子集（operator set）/ 分解（decomposition）
- TorchInductor → 保留
- lowering → 下降 / 下译（lowering）
- guard → 保护条件（guard）
- recompilation / recompile → 重新编译（recompilation）
- eager mode / eager execution → 即时执行模式（eager mode）/ 即时执行 — 统一用「即时执行」，勿混用「eager 执行」

## 自动调优与动态形状

- autotuning → 自动调优（autotuning）
- dynamic shapes → 动态形状（dynamic shapes）
- variable sequence length → 可变序列长度
- specialization → 特化（specialization）
- symbolic shapes → 符号化形状（symbolic shapes）

## 图中断与调试

- graph break → 图中断（graph break）
- explain() / TorchDynamo explain() → 保留
- allow_in_graph / mark functions as safe → 保留 allow_in_graph
- recompilations → 重新编译
- TORCH*LOGS / TORCH_COMPILE_DEBUG / TORCHDYNAMO_REPRO*\* → 保留（环境变量）
- perf_hints / graph_breaks / aot_graphs / inductor / output_code / recompiles / guards → 保留（TORCH_LOGS 取值）
- numerical correctness / accuracy → 数值正确性（numerical correctness）/ 精度
- generated code → 生成的代码

## OpenAI Triton

- OpenAI Triton / Triton → 保留
- Triton programming model → Triton 编程模型
- triton.language / tl → 保留
- @triton.jit / @triton.autotune → 保留
- kernel / custom kernel → 核函数（kernel）/ 自定义核函数
- shared memory → 共享内存（shared memory）
- program / program_id / pid → 保留
- block / tile → 块（block）/ 分块（tile）
- BLOCK_SIZE / num_warps / num_stages → 保留（启动参数）
- launch parameter → 启动参数（launch parameter）
- registering custom kernels → 注册自定义核函数
- torch.library / custom op → 保留 torch.library / 自定义算子（custom op）

## 高级 Triton 核函数

- warp specialization → warp 专门化（warp specialization）
- warp_specialize → 保留
- tiled GEMM / persistent GEMM → 分块 GEMM（tiled GEMM）/ 持久化 GEMM（persistent GEMM）
- GEMM (general matrix multiply) → GEMM（通用矩阵乘法，general matrix multiply）— 保留 GEMM
- software pipelining → 软件流水线（software pipelining）
- double buffering → 双缓冲（double buffering）
- Proton / Triton Proton Profiler → 保留
- occupancy → 占用率（occupancy）
- register / register pressure → 寄存器（register）/ 寄存器压力
- SM (streaming multiprocessor) → SM（流式多处理器）— 保留 SM
- warp → warp（线程束）— 首次可括注，后续用 warp
- TMA (Tensor Memory Accelerator) → 保留 TMA

## XLA 后端

- PyTorch XLA / XLA Backend → 保留
- OpenXLA → 保留
- HLO / StableHLO → 保留
- TPU → 保留
- graph compilation / lazy tensor → 图编译 / 惰性张量（lazy tensor）

## 硬件/平台（保留原文）

- Blackwell / Hopper / GB200 / H100 / TPU → 保留
- CUDA / cuDNN / cuBLAS / CUTLASS → 保留
- HBM / DRAM / L2 / SRAM → 保留
- PyTorch / TorchInductor / Triton / JAX / TensorFlow → 保留
- FlashAttention / transformer_engine → 保留
