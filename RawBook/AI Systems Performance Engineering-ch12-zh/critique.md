# 审校报告（critique.md）— 第 12 章 动态调度、CUDA Graphs 与设备端发起的核函数编排

对照 `bilingual.md` 逐块核对 `translation.md`，并以 `glossary.md` 为术语基准。整体译文质量高、术语基本统一、数字与 CUDA API 名准确。以下仅记录可回源证伪的问题。

## 准确性

- 破损标识符 `cudaDevice Synchronize()`：位于「注意父核函数之后以及两个子核函数之间显式的 cudaDevice Synchronize() 调用」 — 该处把 API 名写成了带空格的 `cudaDevice Synchronize()`（原文因排版换行同样断成两截）。作为代码标识符应无空格 → 建议改为 `cudaDeviceSynchronize()`。（注：译文在别处已修复类似断裂，如 `atomic_transactions_per_request`、`cudaGraphInstantiateFlagDeviceLaunch`、`cudaGraphLaunch(instance, stream)`，此处宜保持一致。）

## 完整性

- 表 12-1 单元格半译：表格右侧多个单元格保留了英文原文未译，与相邻已译单元格风格不一致。「300 separate kernel launches」「300 cudaDeviceSynchronize calls」「100 graph replays（每次迭代 1 次）」「~0.75 ms（25% faster）」 — 数字与单位均正确，但描述性文字未统一翻译 → 建议统一为「300 次独立的核函数启动」「300 次 cudaDeviceSynchronize 调用」「100 次图重放（每次迭代 1 次）」「~0.75 ms（快 25%）」。
- 章末代码块内容缺失：结尾「…通过类似下面这样的调用在 InfiniBand 上发送数据：」之后的代码块为空且未闭合，缺少原文的 `MPI_Send(device_buf, count, MPI_FLOAT, peer_rank, ...);` 一行 → 建议补入该行并闭合代码块。（说明：`bilingual.md` 亦在同一处被截断，疑为上游切分/截断所致；若源材料本身完整，请据源补齐。）

## 术语一致性

- dynamic parallelism：`动态并行（dynamic parallelism）` — 重复括注。首次出现在开篇导言「设备侧的图启动与动态并行（dynamic parallelism）」已括注一次；后于「## 动态并行」小节再次括注为「动态并行（dynamic parallelism，DP）」。同一术语全书应只括注一次 → 建议在首次（导言）即括注为「动态并行（dynamic parallelism，DP）」并引入缩写，后文一律只用「动态并行」或「DP」，删去小节处的重复括注。
- pointer stability / pointer-stable：`指针稳定（pointer-stable）` 与 `指针稳定性（pointer stability）` — 在同一条注释段落内对同一概念出现两处括注（「进行指针稳定（pointer-stable）的 CUDA 图捕获」与「这满足了指针稳定性（pointer stability）的要求」）→ 建议仅保留一处英文括注（如首处 `指针稳定性（pointer stability）`），另一处去掉括注。
- operational intensity：`运算强度（operational intensity）` — 位于导言「提高核函数的运算强度（operational intensity）」。glossary 将同一 roofline 概念（arithmetic intensity）统一为「算术强度」；此处采用「运算强度」可能与全书 roofline 术语不一致 → 建议统一为「算术强度」（若确需保留原词差异，可作「运算强度（operational intensity）」但需确认全书 roofline 章节译法一致）。
