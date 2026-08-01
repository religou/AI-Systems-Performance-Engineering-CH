# 审校报告 — 第 15 章《多节点推理、并行、解码与路由优化》

对照 `bilingual.md` 逐块比对英文原文与中文译文。整体质量高：技术数字、单位、逻辑基本无误，prefill/decode/token/MoE 等按要求保留英文，术语与前序章节一致。主要问题集中在**跨分块的重复括注**，另有少量准确性与一致性小瑕疵。以下按类别列出。

---

## 1. DUPLICATE ANNOTATIONS（重复括注 — 只保留首次）

本章分 3 块翻译，出现若干术语在后续再次括注英文。规则：仅首次出现处括注，后文删除括注、只保留中文（或纯英文缩写）。

| 术语                                  | 首次（保留括注）                                                     | 后续（删除括注）                                                              |
| ------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **容量因子 capacity factor**          | 「高性能 MoE 服务系统会暴露一个**容量因子（capacity factor）**参数」 | 「使用会触发溢出机制的**容量因子（capacity factor）**」→ 改为「_容量因子_」   |
| **负载均衡 load balancing**           | 「高效的专家并行推理需要精心的**负载均衡（load balancing）**」       | 「通过添加一个**负载均衡（load balancing）**损失项」→ 改为「负载均衡」        |
| **draft-and-verify**                  | 「以及"草稿-验证"**（draft-and-verify）**等技术」                    | 「也称为"起草-验证"**（draft-and-verify）**方案」→ 删除括注（另见术语不一致） |
| **变体自动扩缩器 variant autoscaler** | 正文「一个名为 \*变体自动扩缩器**\*（variant autoscaler）**的组件」  | 图 15-4 标题「变体自动扩缩器**（variant autoscaler）**」→ 删除括注            |

低优先级（图注中的重复，镜像英文图题，可选处理）：

- **intranode / internode**：首次在图 15-2 标题「（节点内，intranode）…（节点间，internode）」；图 15-3 标题再次括注「节点内（Intranode）…节点间（Internode）」→ 可删后者括注。
- **MoE / 专家混合**：首行已括注「MoE（专家混合，mixture of experts）」；图 15-6 标题「专家混合（MoE）通信」重复配对（镜像英文图题，可保留）。

---

## 2. ACCURACY（准确性）

- [accuracy] "outperform float global collectives" | current: "并胜过**浮点**全局集合通信" | fix: "并胜过**扁平（非分层）**全局集合通信"
    - 原文语境为分层 vs. 非分层的 all-to-all 调度，"float" 极可能是 "flat" 的笔误；直译为"浮点"（floating point）在此处语义错误，与前文"蝶形/分层调度"对立面不符。

- [accuracy] "rely on **real-time metrics** like per-GPU utilization and per-expert token counts" | current: "依赖诸如…之类的**实时指标（real-time monitoring）**来持续测量" | fix: "依赖诸如…之类的**实时指标**来持续测量"
    - 括注英文 "real-time monitoring" 与相邻中文"实时指标"（= real-time metrics）不对应，属错位括注；应删除该括注（该术语首次英文 monitoring 概念此处并未出现）。

---

## 3. TERMINOLOGY（术语）

- [terminology] "draft-and-verify" | current: 前文译「草稿-验证」、后文译「起草-验证」 | fix: 统一为「起草-验证」或「草稿-验证」（择一，建议「起草-验证」以对应动词 draft）。

- [terminology] "goodput" | current: 「有效吞吐（goodput）」与「有效吞吐量」混用 | fix: 统一为「有效吞吐量（goodput）」。
    - 例：「**有效吞吐（goodput）**请求量最高提升到 7.4×」 vs. 「提升了利用率与**有效吞吐量**（即 _goodput_）」。

- [terminology] "KV cache" | current: 表 15-1 上下文并行行「通过拆分 **KV cache**，支持处理…」 | fix: 「通过拆分 **KV 缓存**，支持处理…」
    - 正文统一用「KV 缓存」，表格单元格残留英文 "KV cache"，前后不一致。

- [terminology] "adaptive expert routing" | current: 正文括注为「**自适应路由**（adaptive routing）」 | fix: 按术语表宜为「自适应专家路由（adaptive expert routing）」（小瑕疵；若确指 gate 决策可保留 adaptive routing）。

- [terminology] "self-speculative decoding" | current: 「其中一种方法就是**自投机解码**，也称为…」首次出现未括注英文 | fix: 首次出现处补括注「自投机解码（self-speculative decoding）」（术语表要求首次标注）。

---

## 4. CODE / STRUCTURE（结构一致性）

- [structure] 图注来源标签中英不一致：
    - 图 15-6：「（**来源：**https://oreil.ly/pzn5t）」已译；
    - 图 15-7「（**source:** …q1AEf）」、图 15-9「（**source:** …uG07b）」、图 15-10「（**source:** …MJMOQ）」仍为英文。
    - fix: 统一为「来源：」。

---

## 5. READABILITY（可读性）

- [readability] 自投机解码段存在**原文自带的重复句**（source error），译文忠实复制：
    - 「这样产生的草稿输出，很像那个更小的草稿模型会产生的输出。…这样产生的草稿输出很像那个更小的草稿模型会产生的输出——并且同样能取得约 2× 的加速。」
    - 与英文 "This produces draft output much like the smaller draft model would…" 的重复对应；忠实但读感冗余，可合并为一句（如需与原文严格对齐则保留）。属可选优化，非错误。

---

## 小结

共计约 **15 处**问题：DUPLICATE ANNOTATIONS 6 处（其中 4 处高置信、2 处图注低优先级）、ACCURACY 2 处、TERMINOLOGY 5 处、STRUCTURE 1 处、READABILITY 1 处（源文冗余）。整体译文**质量优良**：技术论断、数字（1.8 TB/s、130 TB/s、576 GPU、7.4×/12.6×、2–3.6× 等）、单位与逻辑均准确，术语与术语表高度吻合，英文保留策略执行到位、CJK-Latin 间距规范。最需修复的是跨分块重复括注（容量因子、负载均衡、draft-and-verify、变体自动扩缩器）以及 "float→flat" 这一处语义误译；其余为一致性与小术语打磨，不影响理解。
