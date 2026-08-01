# 第 20 章译文审校报告（critique.md）

审校对象：`AI Systems Performance Engineering-ch20-zh\translation.md`（对照 `bilingual.md` 英文源）
章节：第 20 章《AI 辅助的性能优化与迈向千万级 GPU 集群的扩展》（全书最后一章，展望性/总结性，8 图、少量代码、无表格）

**总体评价**：译文整体质量高。FLOPS 数量级（3 × 10²³、2 × 10²⁵、10²⁷–10²⁸）、参数量（100 万亿、200 万亿、320 亿、1 万亿）、矩阵维度（2 × 2）、加速比（10%–20%、15%、1.1–2.1×、3×、25×、30×、100×）、图注编号（图 20-1 至 20-8）与「来源：」格式均准确。代码块与 prompt 中的标识符（`def relative_positional(...)`、`qk_scale = sm_scale * 1.44269504`、`cudaGraphInstantiate()` 等）一字未改。术语大体遵循术语表。主要问题集中在：一处数量级误译（「万万亿」）、若干重复括注需按「保留首次」处理，以及少量翻译腔。以下按类别分组。

---

## [accuracy] 准确性

### A1（重要）「multitrillion-parameter」误译为「万万亿参数」——数量级错误

- **英文**：① "These provide low-latency, real-time inference for multitrillion-parameter models."　② "...accessible multitrillion-parameter models, you could be one of the enablers..."
- **当前中文**：① 「它们为**万万亿**参数模型提供低延迟、实时的推理。」　② 「……**万万亿**参数模型触手可及的时代……」（均在「结论」小节）
- **建议修正**：改为「**数万亿**参数模型」（两处均改）。
- **理由**：multitrillion = 数（多）万亿；「万万亿」在中文读作 10⁴ × 10¹² = 10¹⁶，与原意（数万亿，约 10¹²–10¹³ 量级）不符，且与上下文「已突破万亿门槛，迈向数十/数百万亿」矛盾。

### A2（低争议·须汇报）源文双重否定错误已被静默修正

- **英文**：_"Algorithms need to be clever about **not avoiding** unnecessary work through sparsity, lower precision, and better optimizers."_（源文含双重否定笔误，字面意为「不去避免」，与全句意图相反）
- **当前中文**：「算法需要在通过稀疏性、更低精度与更好的优化器来**避免**不必要的工作方面做得更聪明。」
- **说明**：译者已按正确意图删去「not」，译为「避免」，处理得当。此处仅向用户报告该低争议源文错误已被修正，无需改动译文。

### A3（低争议·须汇报）源文专有名疑似有误——GRPO

- **英文**：_"...called Group Relative **Preference** Optimization (GRPO)..."_
- **当前中文**：「……名为 Group Relative Preference Optimization（GRPO）……」（原样保留，符合「专有名保留原文」约定）
- **说明**：GRPO 通行全称为 Group Relative **Policy** Optimization（策略而非偏好）。源文很可能有误。译文按约定保留原文，处理正确；此处仅提示用户注意源文可能的名称错误，是否更正取决于是否忠实源文。

### A4（低争议）源文括号未闭合，译文已补齐

- **英文**：_"(Note that 1.44269504 = 1/ln(2). Using this value ... when forming qk. In addition to correctness, the generated kernel also achieved a 1.1–2.1× speedup..."_（左括号 "(Note" 一直未闭合）
- **当前中文**：「（注意 1.44269504 = 1/ln(2)。……相应地对相对位置项进行缩放。）除了正确性之外……」
- **说明**：译者在「缩放」后补上右括号，断句合理，属恰当处理。仅作汇报。

---

## [duplicate-annotation] 重复括注（保留首次，删除后续英文括注）

> 规则：同一术语的 `中文（English）` 括注全书只保留**首次**；以下均为**第二次**出现，应删去其英文括注（保留中文）。

### D1 核函数（kernel）

- **首次（保留）**：开篇「生成比人工手写更快的**核函数（kernel）**」。
- **重复（删括注）**：「关键要点」小节——「对矩阵乘法、注意力等核心**核函数（kernel）**的 AI 辅助发现与优化」→ 改为「核心核函数」。

### D2 矩阵乘法（matrix multiplication）

- **首次（保留）**：开篇「即便是在**矩阵乘法（matrix multiplication）**这类核心运算中」。
- **重复（删括注）**：MoE 段——「吞吐量既取决于**矩阵乘法（matrix multiplication）**速度」→ 改为「矩阵乘法速度」。

### D3 自动调优（autotuning）

- **首次（保留）**：智能编译器小节——「核函数启动参数的**自动调优（autotuning）**」。
- **重复（删括注）**：实时系统小节的 Note 引用块——「拥抱 AI 辅助与**自动调优（autotuning）**的人」→ 改为「自动调优」。

### D4 超大规模（ultrascale）——重复且形式不一致

- **首次（保留）**：AlphaTensor 小节——「对于**超大规模（ultrascale）**性能工程师而言」。
- **重复（删括注）**：实时 RL 段——「**超大规模（hyperscale/ultrascale）**系统由数百个……」→ 改为「超大规模系统」。
- **附注**：两处括注形式不同（前 `（ultrascale）`、后 `（hyperscale/ultrascale）`），删去后者即可一并解决不一致问题。

### D5 副驾（copilot）

- **首次（保留）**：DeepSeek 小节——「未来的工作流可能会与 AI **副驾（copilot）**搭档」。
- **重复（删括注）**：「关键要点」小节——「AI **副驾（copilot）**还会监控系统日志」→ 改为「AI 副驾」。

---

## [readability] 可读性 / 翻译腔（低优先）

### R1 「专家人类」生硬

- **英文**："an art reserved for expert humans called _CUDA Ninjas_"
- **当前中文**：「一门只属于被称为 _CUDA Ninjas_ 的**专家人类**的艺术」
- **建议**：改为「……被称为 _CUDA Ninjas_ 的**专家**的艺术」或「……**顶尖专家**……」。
- **理由**：「专家人类」为直译，中文不自然。

### R2 「飞行中批处理」偏字面

- **英文**："in-flight batching"（结论小节）
- **当前中文**：「**飞行中批处理**（in-flight batching）」
- **建议**：改为「**运行中批处理**（in-flight batching）」或保留英文「in-flight batching」。
- **理由**：此处 in-flight 指「处理进行中」，「飞行中」易引起误解。

### R3 「这听上去很奇幻」register 略偏

- **英文**："This sounds fanciful, but it's plausible..."
- **当前中文**：「这听上去很**奇幻**，但……可信」
- **建议**：改为「这听起来有些**异想天开**，但……并非空谈」之类。
- **理由**：「奇幻」偏文学化，技术说明文语域宜用「异想天开／天马行空」。

### R4 「书中所有的招数」易生歧义

- **英文**："using every trick in the book—and then some"
- **当前中文**：「将需要用尽**书中所有的招数**——再加上一些也许尚未被发现的招数」
- **建议**：改为「将需要**使出浑身解数（用尽一切可用手段）**——再加上一些尚未被发现的招数」。
- **理由**：every trick in the book 为习语（一切招数），直译「书中」易被误读为「本书」。

---

## [consistency] 一致性（低优先）

### C1 KV cache 中英混用

- **现象**：正文一处保留英文「**KV cache** 放置」，结论小节另作「分页 **KV 缓存**（paged KV cache）」。英文源两处均为 "KV cache"。
- **建议**：统一为其一（建议正文首次出现处也用「KV 缓存」，或全书统一保留「KV cache」）。属轻微不一致，不影响理解。

---

## 高质量之处（保留，无需改动）

- FLOPS 科学计数法、参数量、加速比、图号与「来源：」格式全部精确无误。
- 代码块、prompt 内标识符、公式（1/ln(2)、qk 缩放式）逐字保留。
- 术语首次括注规范（GEMM、subquadratic algorithm、reinforcement learning，RL、mixture-of-experts、mechanistic interpretability、iterated distillation and amplification、weight streaming、activation offloading、goodput 等均恰当且仅注一次）。
- 长句拆分自然，「brave new AI world→美丽新 AI 世界」「reduces time-to-insight→缩短从问题到洞见的时间」等处理到位；两处不同并行列表（含 model / 含 pipeline）分别忠实源文，未混淆。

```

```
