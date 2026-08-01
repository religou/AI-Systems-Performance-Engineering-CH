# 第 18 章译文审校报告（critique.md）

审校对象：`AI Systems Performance Engineering-ch18-zh/translation.md`（对照 `bilingual.md` 英文块）
主题：高级 prefill-decode 与 KV 缓存调优
说明：本报告仅为诊断，不修改任何译文文件。行号引用自 `bilingual.md`，并附中文原句以便在 `translation.md` 中定位。

---

## 总体评价

译文整体质量**很高**：术语把握准确、代码与环境变量原样保留、图注编号（图 18-N.）与「来源：」格式规范、两张表格（表 18-1、18-2）以及全部关键数字/单位/符号（250,000、80 层、32 头、128 维、4,096、8,192、655,360、1.31 MB、328 GB、100–150 GB、40,000、20–35%、29%、1.4×/20%、2.35×、5.6×、30%、25%、7,500-token、20 ms→8 ms、TP/PP/SP/DP/CP 取值等）均与英文**精确一致**，未发现数值、逻辑反转或增删义的实质性错误。

主要可改进项集中在**重复括注（keep-first）**与**术语一致性**两类，均为低风险的规范化修订。下面按类别列出。

---

## 一、[duplicate-annotation] 同一术语重复括注（应保留首次、删除其后括注）

> 规则：`中文（English）` 每个术语全书只标注一次。以下为重复标注，建议**保留首次**，其余仅删括注（中文照留）。

### 1. 超级核函数（mega kernel / megakernel）

- **首次（保留）** 第 7 行：「单次 decode 的"**超级核函数（mega kernel）**"」
- **重复（删括注）** 第 63 行：「一个完全融合的注意力 decode "**超级核函数（megakernel）**"」
- 建议：第 63 行改为「…decode "超级核函数"…」。
- 附带：见「[consistency] 第 3 条」——「要点回顾」处 megakernel 被译为「巨型核函数」，应统一为「超级核函数」。
- 理由：同一概念（mega kernel）括注两次且英文写法不一致（mega kernel vs megakernel）。

### 2. 分离（disaggregation）

- **首次（保留）** 第 139 行：「**分离（disaggregation）**要求你把 KV 缓存视为跨集群的一等共享资源。」
- **重复（删括注）** 第 911 行：「尽管核心的**分离（disaggregation）**逻辑聚焦于在多块 GPU 之间分配工作…」
- 建议：第 911 行改为「核心的分离逻辑…」。
- 理由：disaggregation 已在第 139 行首次括注。

### 3. 容错（fault tolerance）

- **首次（保留）** 第 223 行：「这提升了**容错（fault tolerance）**能力。」
- **重复（删括注）** 第 1065 行：「**容错（fault tolerance）**是运行一个健壮推理系统的另一个方面。」
- 建议：第 1065 行改为「容错是运行一个健壮推理系统的另一个方面。」
- 理由：同一术语二次括注（后者位于「容错」小节开头）。

### 4. 超级芯片（superchip）

- **首次（保留）** 第 871 行：「混合 prefill 在像 Grace Blackwell 这样的 CPU-GPU **超级芯片（superchip）**上更为常见…」
- **重复（删括注）** 第 911 行：「…朝着 Grace Blackwell 这类紧耦合的 CPU-GPU **超级芯片（superchip）**设计发展…」
- 建议：第 911 行改为「…CPU-GPU 超级芯片设计发展…」。
- 理由：superchip 已在第 871 行首次括注。

### 5. 会合（rendezvous）

- **首次（保留）** 第 531 行：「…并促使在更大的缓冲区上进行**会合（rendezvous）**：」
- **重复（删括注）** 第 561 行：「…以便大的 KV 缓冲区使用**会合（rendezvous）**、小的 KV 缓冲区使用急切模式（eager）。」
- 建议：第 561 行改为「…大的 KV 缓冲区使用会合、小的 KV 缓冲区使用急切模式…」。

### 6. 急切模式（eager）

- **首次（保留）** 第 531 行：「这将减少**急切模式（eager）**碎片化…」
- **重复（删括注）** 第 561 行：「…小的 KV 缓冲区使用**急切模式（eager）**。」
- 建议：第 561 行改为「…小的 KV 缓冲区使用急切模式。」
- 说明：第 5、6 两条同处第 531/561 行，可在第 561 行一并删去两处括注。

> 已核对为**仅一次**、无需处理的括注（正确）：`散布（scatter）`（103）、`逐出（eviction）`（183）、`归并（collation）`（387）、`早期拒绝（early rejection）`（923）、`有效吞吐量（goodput）`（935）、`访存受限（memory bound）`（23）、`算术强度（arithmetic intensity）`（39）、`注意力原语（attention primitive）`（71）、`多轮对话（multiturn conversation）`（179）、`前缀缓存（prefix caching）`（255）、`工作队列元素（work queue elements，WQEs）`（447）、`事件栅栏（event fences）`（447）、`准入控制（admission control）`（931）、`服务质量（QoS）`（971）、`自适应调度/负载均衡`（1113）、`热点（hotspot）`（1153）、`两级调度器（two-level scheduler）`（1161）、`负载卸载（load shedding）`（1229）、`反馈回路（feedback loop）`（1289）。

---

## 二、[terminology] 术语一致性

### 1. 「decode」被误译为「解码」（应保留英文 decode）

术语表明确规定 `decode → 保留`。以下 3 处将独立出现的 decode/decoding 译成了「解码」，与全章其余 decode 用法不一致：

- **第 741 行**：「因此，我们选择张量并行，因为它更适合我们的**解码**工作负载。」→ 建议「…更适合我们的 **decode** 工作负载。」
- **第 801 行**（表 18-2，PP_d 行）：「对于单 token **解码**，流水线并行会增加气泡…」→ 建议「对于单 token **decode**…」
- **第 1201 行**：「Arrow 可以把更多 GPU worker 分配给**解码**阶段…」→ 建议「…分配给 **decode** 阶段…」
- 说明：第 721 行的「推测**解码**（speculative decoding）」为术语表认可译法，**无需改动**。
- 理由：保持 decode 全章一致，避免 decode / 解码 混用。

### 2.（低优先）CP/SP/DP 首次出现缺少英文括注

- 第 763 行列举并行方案时为「张量（TP）、流水线（PP）、数据（DP）、序列（SP，即跨 GPU 拆分序列）」，其中 **CP（context parallelism）** 全章正文未按术语表给出「上下文并行（context parallelism，CP）」的首次括注（仅在表 18-1 中以符号 `CP` 出现）；SP、DP 亦未给出完整英文括注。
- 建议：可在首次正文出现处（或表 18-1 前后）补一次完整括注；因 TP/PP 已建立缩写模式，此为可选优化，优先级低。

---

## 三、[consistency] 一致性

### 1. 「disaggregation」译法混用：分离 vs 分离化

术语表标准为 `disaggregation → 分离`。但正文有 **9 处**使用「分离化」（第 609、629、665、677、681、689、693、725、819 行），而其他位置（如第 139、911 行及「分离消除了干扰，而自适应分离消除了失衡」）用「分离」/「分离式」。

- 例：第 629 行「**分离化**的一个强大优势是…」；第 725 行「**分离化**使得混合这些方法成为可能…」。
- 建议：统一为术语表的「**分离**」（动词/名词语境）与「**分离式**」（定语语境），将「分离化」改为「分离」。
- 理由：同一核心术语应全章统一，避免读者困惑。

### 2. 「prompt」译法混用：提示 vs 提示词

「prompt」在多数位置译为「提示词」，但有 **8 处**译为「提示」（第 159、239、251、283、307、375、379、419 行）。

- 例：第 375 行「prefill 的输出主要由所有**提示** token 的 KV 缓存构成」；第 379 行「…**提示**为 1,000 token 的模型」；第 419 行「prefill worker 在完成**提示**后…」。
- 建议：统一为「提示词」（第 239 行「共享系统**提示**」中的「系统提示」为习惯术语，可保留）。
- 理由：全章 prompt 用词一致。

### 3. 「megakernel」译法混用：超级核函数 vs 巨型核函数

- 正文两处作「超级核函数」（第 7、63 行），但**「要点回顾」**小节 "unified megakernels" 译为「统一**巨型核函数**」。
- 建议：统一为「超级核函数」（并配合「[duplicate-annotation] 第 1 条」处理括注）。

### 4.（低优先）「fabric」译法混用

- "fabric" 分别译为「网络架构」（UCX 网络架构）、「结构」（NVSwitch 结构）、「网络结构」（RoCE/IB 无损设置所在的网络结构）。
- 建议：可统一为「网络架构」或「网络结构」，优先级低。

---

## 四、[accuracy] 需向用户汇报的低争议源错误修正（译文已正确处理）

### 1. 英文原文「Two notable innovations」实列举三项——译文已修正为「三项」

- 第 23 行英文：「**Two** notable innovations in this space are FlashMLA (DeepSeek), ThunderMLA (Stanford), and FlexDecoding (PyTorch).」——原文写「Two」却列出**三**项。
- 译文（正确）：「该领域中有**三项**值得关注的创新：FlashMLA（DeepSeek）、ThunderMLA（Stanford）以及 FlexDecoding（PyTorch）。」
- 结论：这是译者对英文原文低争议错误的**正确修正**，依 `prompt.md` 约定在此**汇报**，无需改动译文。

### 2.（观察，非译误）源文 MLA 全称前后不一致

- 第 33 行英文将 FlashMLA 展开为「**Flash Multi-Latent Attention**」，而第 39 行及术语表为「**Multi-head Latent Attention**」。译文忠实保留了两处英文原样（第 39 行括注「MLA（多头潜在注意力，Multi-head Latent Attention）」）。
- 结论：属英文源文自身不一致，译文处理无误，仅作记录，**无需改动**。

---

## 五、经核对确认无误的重点项（高质量）

- **表 18-1 / 表 18-2**：四列结构、表头、符号（TP_p、PP_p、SP_p、CP、DP_p、TP_d、PP_d、SP_d、DP_d）、取值（2、1、1 (or 2)、1 (default) ... N (number of GPUs)）与说明文本，逐格核对**完全准确**。
- **数值/单位/百分比**：全章核对无一处偏差（含 KV 占用推导链 4,096→8,192→655,360→1.31 MB→328 GB）。
- **代码块 / 环境变量**：`enable_pd`、`transfer_channel: "nixl"`、`pd_buffer_size: 1073741824`、`UCX_RNDV_THRESH=16384`、`UCX_TLS=...`、`PYTHONHASHSEED=0`、`qos.yaml`（0.10/0.30/0.60、priority 100/50/10）等**原样保留**，注释中文化恰当。
- **图注**：图 18-1 至 18-10 编号、译文与「来源：」+URL（`https://oreil.ly/2xtK-`、`1-Ti0`、`_3KGj`）**格式规范、无误**。
- **保留术语**：FlashMLA、ThunderMLA、FlexDecoding、FlexAttention、PagedAttention、NIXL、NATS、UCX、GPUDirect RDMA、InfiniBand、NVLink、NVSwitch、Splitwise、HexGen-2、TetriInfer、Mooncake、Arrow、Grace Blackwell、GB200 NVL72、Llama 2 70B、H100/A100/B200 等**一律正确保留**。

---

## 六、修复优先级建议（Top）

1. **[duplicate-annotation]** 删除 6 处重复括注（超级核函数、分离、容错、超级芯片、会合、急切模式）——保留首次。
2. **[consistency]** 统一「disaggregation」：将 9 处「分离化」改为「分离/分离式」。
3. **[terminology]** 将第 741、801、1201 行的独立「解码」改回「decode」。
4. **[consistency]** 统一「prompt」：8 处「提示」改为「提示词」（保留「系统提示」）。
5. **[consistency]** 「要点回顾」的「巨型核函数」统一为「超级核函数」。
6. **[accuracy-汇报]** 已向用户汇报：原文「Two」→ 译文「三项」为正确的源错误修正。
