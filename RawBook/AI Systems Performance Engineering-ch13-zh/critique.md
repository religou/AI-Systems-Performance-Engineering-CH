# 第 13 章审校记录（critique）

审校对象：`bilingual.md`（英中对照，3189 行）。本文件仅记录问题，不修改任何译文文件。
格式：`[category] block-hint: "<英文原文短引>" | current: "<当前中文>" | fix: "<修正后中文>"`

---

## 总体结论

译文整体**准确、通顺、术语规范**，可评为 **good / 可接受**。数值、单位、表格百分比全部核对无误（如 248ms→173ms ~30%、Table 13-4、Table 13-6 的 43.8%/37.5%/16.7%/2.1%、19.5ms=81.3% 等），未发现方向性误译或逻辑反转。代码块正确保持英文原样、未被翻译。**主要问题集中在“重复标注”**——同一术语的英文括注按全书规则应只在首次出现时给出，但因分 6 个 chunk 翻译，出现了大量跨 chunk 的重复括注，需清理。其次是少量术语一致性问题（eager、FFN）。

---

## DUPLICATE ANNOTATIONS（核心问题：保留首次，删除后续括注，仅去括号、保留中文）

规则：术语的“中文（English）”括注全书只保留**首次出现**处；后续同术语出现时删除括注（保留中文译名）。若删括后出现“中文紧邻 ASCII 字母”，需补回 CJK↔Latin 空格。

- **剖析 / profiling**：保留 L7；删 L1333「做性能剖析（profiling）」→ 做性能剖析
- **PyTorch 剖析器 / PyTorch profiler**：保留 L61（PyTorch 剖析器（PyTorch profiler，Kineto））；删 L1333「PyTorch 剖析器（PyTorch profiler）」
- **CUDA 流 / CUDA stream**：保留 L11；删 L1203、L1309
- **并发 / concurrency**：保留 L11；删 L1203、L1501
- **吞吐量 / throughput**：保留 L99；删 L1525
- **延迟 / latency**：保留 L99；删 L1309「H2D 延迟（latency）」
- **PyTorch 编译器 / PyTorch Compiler**：保留 L263；删 L753（H2 标题）、L757、L2747
- **图中断 / graph break**：保留 L263；删 L1119、L2549、L2553
- **图捕获 / graph capture**：保留 L997；删 L1553「捕获（graph capture）」（注：同行的（graph replay）为首次，保留）
- **GEMM / 通用矩阵乘法**：保留 L349（GEMM（通用矩阵乘法，general matrix multiply））；删 L793「GEMM（通用矩阵乘法）」及 “### …通用矩阵乘法（GEMM）…” 标题内括注
- **MoE / 专家混合**：保留 L115（专家混合（mixture-of-experts，MoE））；删 L1957「MoE（专家混合，mixture of experts）」
- **全局追踪分析 / HTA**：保留 L15（Holistic Trace Analysis，HTA）；删 L87「全局追踪分析（Holistic Trace Analysis）」
- **FSDP / 完全分片数据并行**：保留 L15；删 L1917 注释、L2573「FSDP（完全分片数据并行，Fully Sharded Data Parallel）」
- **DDP / 分布式数据并行**：保留 L15；删 L2541、L2557
- **张量并行 / tensor parallelism, TP**：保留 L2171；删 L2743
- **流水线并行 / pipeline parallelism, PP**：保留 L1521；删 L2175、L2743
- **算术强度 / arithmetic intensity**：保留 L517；删 L2517「提高算术强度（arithmetic intensity）」→ 提高算术强度
- **内存碎片 / memory fragmentation**：保留 L1835；删 L1901、L2937
- **激活检查点 / activation checkpointing**：保留 L1835；删 L1921
- **卸载 / offloading**：保留 L1835（内存卸载（offloading））；删 L1957「卸载（offloading）」
- **分片 / shard**：保留 L2021（分片（shard））；删 L2573

**迟到/多余括注（术语在前文已多次无括注使用，此处才补括注，建议直接去括）：**

- L1501「专家（expert）」：专家一词在剖析章节早已多次使用，删括注 → 专家
- L2561「核函数融合（kernel fusion）」：核函数融合前文已多次出现（如区域化编译处），删括注 → 核函数融合

---

## TERMINOLOGY / 一致性

- [terminology] eager 译法不统一：L267 "fell back to eager execution" | current: "回退到了 eager 执行" | fix: 统一为「回退到即时执行」（正文 L805 已确立「即时执行模式（eager mode）」为标准译名；L2791「即时执行（eager）」亦为此风格）。全章“eager 执行 / eager 模式”与“即时执行 / 即时模式”混用，建议统一到「即时（eager）…」体系。
- [terminology] FFN 译名不一致且重复：L349 "feed-forward network (FFN)" | current: "前馈网络（feed-forward network，FFN）" ；L1929 "Feedforward Neural Network (FFN)" | current: "前馈神经网络（Feedforward Neural Network，FFN）" | fix: 统一中文为「前馈网络」，L1929 删括注（原文大小写虽不同，但同一 FFN 概念，全书择一译名）。

---

## ACCURACY

- 未发现实质性误译。抽查数值/单位/表格全部正确（matmul 128 次；dispatch 18.7ms + combine 12.1ms = 30.8ms；Table 13-4 Baseline 50/70/60% 访存受限、Optimized 85/40/80% 计算受限；138ms = 60.5+58.3+19.2；Table 13-6 百分比；SimpleFSDP ~28% / ~69%；NVLink 1.8 TB/s 双向≈单向 900 GB/s 等）。
- 唯一可归入准确性的边缘项即上文 eager 的一致性问题（属表述统一而非误译）。

---

## READABILITY

- 整体地道流畅。少量可选润色（非必须）：
    - [readability] L2xxx "This is doubly bad" | current: "这是双重的坏事" | fix: 「这会造成双重损害」（更书面）。
- 提醒：执行“去重复括注”后，逐处扫描是否出现“中文紧邻 ASCII 字母/数字”而缺空格（沿用第 11 章经验），如「做性能剖析profiling…」类，需补回 CJK↔Latin 半角空格。

---

## CODE / STRUCTURE

- 代码围栏正确：所有 ```代码块均保持英文原样、成对出现（英文↔英文），未被翻译；行内标识符（如 `torch.compile`、`bucket_cap_mb`、`PYTORCH_ALLOC_CONF`、`torch.allclose`）均保留原文。
- 表格结构、图注编号（图 13-1…13-7）、引用块 `>` 均与原文对应，无缺行/错位。
- 说明：代码内 `PYTORCH_ALLOC_CONF` 与词汇表的 `PYTORCH_CUDA_ALLOC_CONF` 不同，但属源码原文，保持不译正确，非错误。

---

## 计数摘要

- DUPLICATE ANNOTATIONS：约 23 个术语组（含 2 处迟到括注）——**主要问题**
- TERMINOLOGY：2 项（eager、FFN）
- ACCURACY：0 项实质错误
- READABILITY：1 项可选 + 去括后空格复检提醒
- CODE / STRUCTURE：0 项错误

**结论**：译文质量 good / 可接受；核心待办是按“全书只标注一次”规则清理重复英文括注，并统一 eager / FFN 译法。
