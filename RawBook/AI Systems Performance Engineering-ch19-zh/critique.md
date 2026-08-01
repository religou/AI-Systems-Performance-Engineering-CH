# 第 19 章译文审校报告（诊断）

> 审校对象：`AI Systems Performance Engineering-ch19-zh/translation.md`（对照 `bilingual.md` 英文源与 `glossary.md`、`prompt.md` 约定）。
> 本报告仅为诊断，不修改任何译文文件。行号均指 `translation.md`。
>
> **总体结论：译文质量高。** 术语准确、数字与表格无误、图注格式规范、中英混排空格干净、代码块基本保持原样。主要问题集中在**重复括注**（同一术语在全章多个 chunk 中被反复标注中文（English）），其次是少量**术语渲染不一致**。准确性与可读性问题极少。

---

## [duplicate-annotation] 重复括注（最高优先级）

依据 `glossary.md` 规则「术语全书只标注一次」，同一术语应仅在**首次出现**处保留 `中文（English）` 括注，后续出现只保留中文。以下术语被跨 chunk 重复标注，应保留首次、删除后续括注。

1. **pipeline bubble** — 保留 L29 `*流水线气泡*（pipeline bubble）`；删除后续括注。
    - 重复处 L1214：`流水线气泡（pipeline bubble）就会出现` → 改为 `流水线气泡就会出现`
    - 理由：首次已在 L29 定义，后续为重复标注。

2. **occupancy** — 保留 L342 `GPU 占用率（occupancy）`；删除后续括注。
    - 重复处 L392：`流式多处理器（streaming multiprocessor，SM）占用率（occupancy）` → `……SM）占用率`
    - 重复处 L1232：`占用率（occupancy）` → `占用率`
    - 理由：`occupancy` 首次已定义，L392/L1232 为重复。

3. **out-of-memory / OOM** — 保留 L39 `内存溢出（out-of-memory，OOM）`；删除后续括注。
    - 重复处 L623：`内存不足（out-of-memory，OOM）错误` → `内存不足错误`（并见 consistency 第 3 条：统一「内存溢出/内存不足」）
    - 理由：OOM 首次已定义。

4. **register** — 保留 L342 `寄存器（register）`；删除后续括注。
    - 重复处 L394：`寄存器（register）、共享内存和 warp` → `寄存器、共享内存和 warp`
    - 理由：重复标注。

5. **thread block** — 保留 L283 `线程块（thread block）`；删除后续括注。
    - 重复处 L394：`每个线程块（thread block）所用的共享内存量` → `每个线程块所用的共享内存量`
    - 理由：重复标注。

6. **self-attention** — 保留 L287 `自注意力（self-attention）`；删除后续括注。
    - 重复处 L396：`由于自注意力（self-attention）本质上是二次复杂度` → `由于自注意力本质上是二次复杂度`
    - 理由：重复标注。

7. **speculative KV prefetching** — 保留 L470 `推测式 KV 预取（speculative KV prefetching）`；删除后续括注。
    - 重复处 L822：`启用或禁用推测式 KV 预取（speculative KV prefetching）` → `……推测式 KV 预取`
    - 理由：重复标注。

8. **reinforcement learning / RL** — 保留 L5 `强化学习（reinforcement learning，RL）`；删除后续括注。
    - 重复处 L800：`我们可以用强化学习（reinforcement learning，RL）来调优` → `……用强化学习来调优`
    - 理由：RL 首次已定义。

9. **recency bias** — 保留 L593 `近因偏好（recency bias）`；删除后续括注（并见 consistency 第 1 条）。
    - 重复处 L784：`带近因偏置（recency bias）或滑动窗口注意力` → `带近因偏好或滑动窗口注意力`
    - 理由：重复标注且译名不一致。

10. **speculative decoding** — 保留 L816 `推测解码（speculative decoding）`；删除后续括注（并见 consistency 第 2 条）。
    - 重复处 L1176：`推测式解码（speculative decoding）的草稿模型` → `推测解码的草稿模型`
    - 理由：重复标注且译名不一致。

11. **fragmentation** — 保留 L906 `碎片化（fragmentation）`；删除后续括注。
    - 重复处 L1069：`不会引发内存碎片化（fragmentation）或分配延迟尖峰` → `……内存碎片化或分配延迟尖峰`
    - 理由：重复标注。

12. **caching allocator** — 保留 L918 `缓存分配器（caching allocator）`；删除后续括注。
    - 重复处 L1063：`默认缓存分配器（caching allocator）` → `默认缓存分配器`
    - 理由：重复标注。

13. **congestion-aware** — 保留 L1357 `拥塞感知（congestion-aware）`；删除后续括注。
    - 重复处 L1555：`动态的拥塞感知（congestion-aware）调度` → `动态的拥塞感知调度`
    - 理由：重复标注。

14. **topology-aware** — 保留 L1357 `拓扑感知（topology-aware）`；删除后续括注。
    - 重复处 L1623：`拥塞感知、拓扑感知（topology-aware）调度` → `拥塞感知、拓扑感知调度`
    - 理由：重复标注。

---

## [consistency] 术语渲染不一致

1. **recency bias**：`近因偏好`（L593，首次）vs `近因偏置`（L784）。
    - 英文：`recency bias`
    - 建议：统一为**首次出现**的 `近因偏好`，将 L784 的 `近因偏置` 改为 `近因偏好`。
    - 理由：同一术语两种译名会削弱一致性；以首次出现为准。

2. **speculative decoding**：`推测解码`（L816、L818、L820、L1698）vs `推测式解码`（L478 图注、L1176）。
    - 英文：`speculative decoding`
    - 建议：统一为占多数且首个带括注的 `推测解码`；将 L478、L1176 的 `推测式解码` 改为 `推测解码`。
    - 理由：多数用法为 `推测解码`，且与首次定义一致（注意与术语表内 `推测式 KV 预取`、`推测式 MoE 专家路由` 并不冲突，那是不同词条）。

3. **out-of-memory (OOM)**：`内存溢出`（L39）vs `内存不足`（L623）。
    - 英文：`out-of-memory`
    - 建议：统一为一种译名。就语义而言 `内存不足/内存耗尽` 比 `内存溢出`（overflow）更贴切，建议将 L39 的 `内存溢出` 改为 `内存不足`；或反之统一到 `内存溢出`。至少两处须一致。
    - 理由：同一缩写 OOM 出现两种展开译名。

---

## [terminology] 术语/约定偏差

1. **唯一被翻译的代码注释**（L1085）。
    - 英文（代码块内）：`# Monkey-patch the model's attention forward to use the new library`
    - 当前中文：`# 对模型的注意力 forward 打猴子补丁，改用新库`
    - 建议：按 `prompt.md`「代码保持原样」约定，本行为全章唯一被翻译的代码注释，其余代码注释均保留英文。建议还原为英文以保持一致，或在全章统一策略。
    - 理由：与「代码 verbatim」约定及本章其余代码块处理方式不一致（译文本身语义正确，属约定一致性问题）。

---

## [accuracy] 准确性

整体准确性优秀：抽查 表 19-1/19-2/19-3 数值（如 ~1.8×、~0.25×（25%）、~3.5%、<0.1%；64×64/48/85/8.2 等）与英文源逐格一致；`log₂(72)`、`71 = 72 – 1`、`~720 GB` 等数字/单位无误；图 19-1…19-14 图注均为 `图 19-N.` 格式且 `来源：` 使用规范；环境变量、dtype、API（`NCCL_CROSS_NIC`、`torch.autocast`、`cudaMemAdvise`、`E4M3`/`E5M2`、`amax_history_len=1024` 等）保持原样。仅一处语义可推敲：

1. L39 将 OOM 译为 `内存溢出`。
    - 英文：`out-of-memory (OOM) errors`
    - 当前中文：`内存溢出（out-of-memory，OOM）错误`
    - 建议：`out-of-memory` 指「内存不足/耗尽」，`内存溢出` 更接近 overflow；建议改为 `内存不足`（与 consistency 第 3 条合并处理）。
    - 理由：措辞更贴合原意。

---

## [readability] 可读性

译文行文流畅、断句自然、长句拆分得当，未见明显生硬或欧化表达。仅一处源文残句被忠实保留、读来略突兀：

1. L126 保留了英文源中的孤立残片 `4-bit targets.`。
    - 英文（源即为断句）：`… reduce memory down to 20% of the baseline. 4-bit targets. The speedup is around 3.5×`
    - 当前中文：`……可将内存减少到基线的 20%。4 位目标。加速比约为 3.5×`
    - 建议（可选）：`4 位目标。` 系忠实翻译英文源的残句，语义悬空。若允许轻微顺滑，可并入前句或删去该残片；若严格对齐源文，则维持现状。
    - 理由：属英文源本身的破碎片段，非误译；仅影响观感，优先级低。

---

## 统计汇总

| 类别                   | 数量                              |
| ---------------------- | --------------------------------- |
| [duplicate-annotation] | 14                                |
| [consistency]          | 3                                 |
| [terminology]          | 1                                 |
| [accuracy]             | 1（且与 consistency 第 3 条重叠） |
| [readability]          | 1（源文残句，可选）               |
| **合计**               | **20**                            |
