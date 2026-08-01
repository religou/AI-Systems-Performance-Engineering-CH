# 第 14 章 审校报告（critique.md）

对 `bilingual.md`（PyTorch 编译器、OpenAI Triton 与 XLA 后端）逐块比对英文原文与中文译文的审校结果。整体译文质量**良好（good）**：准确性高、术语总体贴合 `glossary.md`、代码块未被误译。主要问题集中在**跨分块重复括注**（4 个分块各自首次括注，导致同一术语在全章多次括注），需按“只在全书首次出现处括注”的规则清理。

---

## 1. DUPLICATE ANNOTATIONS（重复括注 —— 保留首次，删除后续）

规则：`中文（English）` 括注全书只保留**首次出现**，后续出现只留中文。以下按术语列出「保留」与「应删除」位置（附短引用便于定位）。

### graph break —— 图中断（graph break）

- **保留**：L81「……且不会引发过多**图中断（graph break）**。」
- 删除：L763「……代码如何被转换为图、**图中断（graph break）**与子图的高层概览……」→ 图中断
- 删除：L1137「Dynamo 会产生一次**图中断（graph break）**，因为它无法推断……」→ 图中断

### eager execution —— 即时执行（eager execution）

- **保留**：L19「……灵活的**即时执行（eager execution）**开发体验……」
- 删除：L561「……持续优于**即时执行（eager execution）**——即便序列长度……」→ 即时执行

### eager mode —— 即时执行模式（eager mode）

- **保留**：L85「……以常规的、未编译的**即时执行模式（eager mode）**运行。」
- 删除：L621「这会强制代码以**即时执行模式（eager mode）**运行。」→ 即时执行模式
- 删除：L1125「与 DDP 或**即时执行模式（eager mode）**相比……」→ 即时执行模式
- 备注：`eager execution` 与 `eager mode` 概念同一，glossary 要求统一「即时执行」。可考虑整章 eager 概念仅括注一次（首次 L19），其余全部去括注。

### recompilation —— 重新编译（recompilation）

- **保留**：L101「……以捕捉不安全的**重新编译（recompilation）**。」
- 删除：L545「TorchInductor 会在首次**重新编译（recompilation）**之后……」→ 重新编译
- 删除：L1153「……图中断、**重新编译（recompilation）**、保护条件（guard）……」→ 重新编译

### guard —— 保护条件（guard）

- **保留**：L175「……插入**保护条件（guard）**——例如张量形状……」
- 删除：L549「编译器会插入更多**保护条件（guard）**；如果这些保护条件……」→ 保护条件
- 删除：L1153「……重新编译（recompilation）、**保护条件（guard）**以及其他编译器决策。」→ 保护条件

### specialization —— 特化（specialization）

- **保留**：L183「……根据观察到的形状进行**特化（specialization）**。」
- 删除：L545「……而不是针对每一个新形状反复**特化（specialization）**。」→ 特化

### autotuning —— 自动调优（autotuning）

- **保留**：L439「内置了一个基于 Triton **自动调优（autotuning）**能力构建的自动调优器……」
- 删除：L1291「……向 PyTorch 注册 Triton 核函数、**自动调优（autotuning）**核函数启动参数……」→ 自动调优

### lowering —— 下降（lowering）

- **保留**：L313「……会被**下降（lowering）**为一次函数式 add……」
- 删除：L649「你可能会在 tl.dot 被**下降（lowering）**为 Tensor Core 操作的地方……」→ 下降
- 删除：L1427「该循环会**下降（lowering）**为 cp.async、TMA 和屏障。」→ 下降

### warp specialization —— warp 专门化（warp specialization）

- **保留**：L505「……支持自动化的 **warp 专门化（warp specialization）**。」
- 删除：L649「……核函数融合、**warp 专门化（warp specialization）**或双缓冲……」→ warp 专门化
- 删除：L1291「……诸如 **warp 专门化（warp specialization）**和软件流水线……」→ warp 专门化
- 删除：L2513「你还可以使用 **warp 专门化（warp specialization）**，Triton 会借此……」→ warp 专门化

### double buffering —— 双缓冲（double buffering）

- **保留**：L649「……warp 专门化（warp specialization）或**双缓冲（double buffering）**等优化。」
- 删除：L1291「……软件流水线（software pipelining，例如**双缓冲（double buffering）**）……」→ 双缓冲
- 删除：L2279「这个示例展示了如何用 Triton 实现**双缓冲（double buffering）**。」→ 双缓冲

### software pipelining —— 软件流水线（software pipelining）

- **保留**：L1291「……和**软件流水线（software pipelining）**……」（注意此处 double buffering 已是重复，见上）
- 删除：L2279「双缓冲是**软件流水线（software pipelining）**的一种两阶段形式……」→ 软件流水线

### shared memory —— 共享内存（shared memory）

- **保留**：L1291「……包括访问**共享内存（shared memory）**……」
- 删除：L2259「……相对较大的寄存器文件、**共享内存（shared memory）**和 L2 缓存。」→ 共享内存
- 备注：L1419/L1423 的「共享内存」（无括注）在 L1291 之后，处理正确。

### occupancy —— 占用率（occupancy）

- **保留**：L1553「这有助于提升**占用率（occupancy）**、掩盖更多内存访问延迟……」
- 删除：L2537「……更低的**占用率（occupancy）**以及更差的整体性能。」→ 占用率

### tile —— 分块（tile）

- **保留**：L1419「……把来自矩阵 A 和 B 的一个**分块（tile）**加载到共享内存中。」
- 删除：L2003「……为每个输出**分块（tile）**都启动多个核函数。」→ 分块

### coalesce —— 合并（coalesce）

- **保留**：L1377「……在可能时被**合并（coalesce）**。」
- 删除：L2695「确保内存访问是**合并（coalesced）**的……」→ 合并

### domain-specific language —— 领域特定语言（domain-specific language，DSL）

- **保留**：L383「Triton 是一种……**领域特定语言（domain-specific language，DSL）**。」
- 删除：L1275「OpenAI Triton 是一种……**领域特定语言（domain-specific language，DSL）**……」→ 领域特定语言（DSL）

---

## 2. TERMINOLOGY（术语）

- [terminology] TMA 首个「展开式」括注出现过晚。`TMA` 在 L505/L507/L1427 等处已多次裸用，但全称 `张量内存加速器（Tensor Memory Accelerator，TMA）` 直到 L2533 才给出。建议把全称括注移到**首次出现处（L505）**，后续只用 TMA。
- [terminology] `Tensor Core` 译名不一致：L649 用英文「Tensor Core」（未括注），而 L1935/L1943/L1959 用「张量核心（Tensor Core）」。建议统一，且在**首次出现（L649）**处括注一次「张量核心（Tensor Core）」，后续用「张量核心」。
- [terminology] `low-level ISA` | current: "低层指令集架构"（L387）| fix: "底层指令集架构" —— 全章其余处 "under the hood/low-level" 多译作「底层」，此处「低层」不一致。

---

## 3. ACCURACY（准确性）

整体准确，仅少量可优化的细微偏差：

- [accuracy] "rounded to an internal bucket" | current: "并**向上舍入**到某个内部分桶"（L745）| fix: "并**舍入**到某个内部分桶" —— 原文只说 rounded，未指定向上取整。
- [accuracy] "an autotuner built on Triton's autotuning capabilities, which we'll describe in an upcoming section" | current: "基于 Triton 自动调优能力构建的自动调优器**（我们将在后面的小节中介绍）**"（L439）| fix: 建议明确「后面的小节」介绍的是 **Triton 自动调优**本身，例如「……构建的自动调优器；Triton 的自动调优将在后面的小节中介绍」，以免读者误以为介绍的是 TorchInductor 的调优器。

---

## 4. READABILITY（可读性）

- [readability] L1291 括注嵌套过深、信息密度过高：「……诸如 warp 专门化（warp specialization）和软件流水线（software pipelining，例如双缓冲（double buffering））等进阶的 Triton 主题。」按第 1 节删除重复括注后，可简化为：「……诸如 warp 专门化、软件流水线（software pipelining，例如双缓冲）等进阶的 Triton 主题。」可读性显著改善。
- [readability] L1137「_在使用自定义及第三方 CUDA C++ 和 Triton 算子时监控性能权衡_」后紧接正文无空格/停顿，属原文小标题+正文合排格式，符合前后体例，可保留。
- 说明：全章 CJK–Latin 间距总体处理规范，未发现连续中文字符间的多余空格或普遍缺失空格的情况。

---

## 5. CODE / STRUCTURE（代码/结构）

- 未发现代码块内容被误译或改动：所有 fenced code（Python/Triton/伪代码/环境变量）英文块与中文块内容一致，代码内注释保留英文。
- 内联标识符（`torch.compile`、`torch._dynamo.mark_dynamic`、`num_warps`、`BLOCK_SIZE`、`tl.dot`、`backend="eager"`、`use_original_params=True` 等）均正确保留原文，未被翻译。
- 备注（非译文问题，原文即如此）：`matmul_kernel_persistent` 的封装函数 `persistent_matmul` 调用时未传入 kernel 声明的 `NUM_STAGES: tl.constexpr`；此为英文源码本身的缺陷，译文未引入新问题，无需在译稿中改动。

---

## 汇总

| 类别                  | 数量（约）                            | 说明                                          |
| --------------------- | ------------------------------------- | --------------------------------------------- |
| DUPLICATE ANNOTATIONS | 15 个术语组、约 26 处应删除的重复括注 | 主要问题，源于 4 分块各自首次括注             |
| TERMINOLOGY           | 3                                     | TMA 括注位置、Tensor Core 译名统一、低层→底层 |
| ACCURACY              | 2                                     | 均为轻微措辞偏差                              |
| READABILITY           | 1（+1 说明）                          | L1291 嵌套括注                                |
| CODE / STRUCTURE      | 0                                     | 代码与标识符处理正确                          |

**总体质量：good。** 译文忠实、术语规范、代码零误译；主要待办是按“首次括注”规则批量清理跨分块的重复括注，并统一 TMA/Tensor Core 的括注位置与译名。
