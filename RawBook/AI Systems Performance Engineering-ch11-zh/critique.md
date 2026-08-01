# 第 11 章审校报告（critique.md）

对照 `bilingual.md` 逐块核对 `translation.md`，并以 `glossary.md` 为术语基准。总体译文准确、通顺、完整，代码块与原文一致，未见漏译或跳段。以下仅列出可回源核实的问题。

## 准确性

- MMA 展开语义不完整：「计算 warp 执行**乘加累加**（matrix-multiply-accumulate，MMA）操作」（"Combining PDL and Thread Block Clusters" 一节） — 原文 `matrix-multiply-accumulate` 指“**矩阵**乘累加”，译文「乘加累加」漏掉了 matrix/矩阵，易与普通乘加混淆 → 建议改为「矩阵乘累加（matrix-multiply-accumulate，MMA）」或「矩阵乘加累加」。

## 可读性

- 「依赖默认流行为最终会引发问题。**它总是会。**」（"Default Versus Explicit (Nondefault) Streams" 开头） — 「它总是会。」在中文中略显突兀、意思不完整（原文 "It always does." 强调“无一例外总会出问题”）→ 建议改为「而且屡试不爽。」或「每次都是如此。」。
- 「有一个硬性上限（**最多 128 个驻留网格上限**）」（"Stream-Ordered Memory Allocator" 一节，讲并发核函数硬限） — 「最多…上限」语义重复 → 建议改为「（最多 128 个驻留网格）」。

## 术语一致性

重复括注（专业术语全书应只在首次出现时括注原文一次；下列术语在正文中被多次括注）：

- **in-flight / 在途**：`在途（in-flight）` 至少出现 4–5 次重复括注，如「stream 1 的分配还在途（in-flight）中」「下一批次的 H2D 拷贝就已在拷贝引擎上在途（in-flight）」「多个在途（in-flight）的小批量」「让多个 mini-batch 保持在途（in flight）」及 Key Takeaways「与在途（in-flight）的核函数…」→ 仅保留首次括注，其余用「在途」。
- **mini-batch / 小批量**：`小批量（mini-batch）` 多次括注，如「为每个小批量（mini-batch）分配 GPU 内存」「把接连而来的小批量（mini-batch）一波波送过 GPU」及 Key Takeaways「相继的小批量（mini-batch）」→ 仅首次保留。
- **real-world / 真实**：`真实（real-world）` 出现两次括注（章首「在真实（real-world）工作负载中」与 "Warp Specialization with Thread Block Clusters" 收尾注「大多数真实（real-world）LLM…工作负载」）→ 仅首次保留。
- **interconnect / 互连**：`互连（interconnect）` 出现两次括注（NCCL 注「为互连（interconnect）调优过的…核函数」与结论「DMA 引擎与互连（interconnect）」）→ 仅首次保留。
- **stream-ordered memory allocator / 流序内存分配器**：章首已括注一次，Key Takeaways 正文又出现「流序内存分配器（stream-ordered memory allocator）」重复括注 → 去除后者括注。
- **thread block cluster / 线程块簇**：章首已括注一次，Key Takeaways 正文又出现「线程块簇（thread block cluster）」重复括注 → 去除后者括注。
- **prologue / epilogue（序言 / 收尾）**：`收尾（epilogue）` 在图 11-9 题注、PDL 正文「收尾（epilogue，即收官）阶段」及 Key Takeaways 多次括注；`序言（prologue）` 在正文与 Key Takeaways 各括注一次 → 各仅保留首次括注。

术语内部不一致：

- **leader block**：同一节内混用「主导块」与「主导线程块」（"Warp Specialization with Thread Block Clusters" 一节：「主导块用一个块作用域的 pipeline」vs「主导线程块（leader block）通过块作用域的…」/「一个主导线程块只加载一次 tile」）→ 统一为「主导线程块」（与 glossary「主导 CTA（leader CTA）」对应）。

---

说明：原文自身的若干笔误/不一致（如 `improver` 应为 improve、`cudaMallocAsync ÷ cudaFreeAsync`、`deviceProp.asyncEngine Count()` 中的空格、`scratch Bytes`、`PYTORCH_ALLOC_CONF`、多 GPU 段「GPU 0…GPU B」及 batch `n / b+1 / b−1` 混用）译文均按要求忠实保留或按正确含义翻译，不计为问题。
