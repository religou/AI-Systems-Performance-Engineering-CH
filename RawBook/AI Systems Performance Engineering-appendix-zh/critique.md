# 审校诊断报告 — 附录《AI 系统性能检查清单（175+ 项）》

对照英文原文（`bilingual.md` 中的英-中配对块）逐块审校了本附录 19 个小节、175+ 条清单项的中文译文（`translation.md`）。以下按类别列出发现，每条给出【类别】、英文原文、当前中文、建议修改与一句话理由。行号均指向 `translation.md`，便于定位修改。

> 说明：斜体标记 `_..._` 与 `*..*` 混用（第 1–15 节多用 `_`，第 12、13、14、16、17 节多用 `*`）按任务要求**不视为错误**，未列入。

---

## 总体评价

译文整体质量**很高**：术语密度极大，但数字、单位、比值、命令、环境变量、API 标识符几乎全部精确无误且原样保留（已逐项核对，见下"准确性"节结论）。句式通顺、指令性强，基本无翻译腔。主要问题集中在 **跨清单项的重复括注**（术语在多个清单项中反复出现，首现之后仍再次括注原文），以及少量术语渲染不一致。这些正是本附录体裁（同一批全栈术语在 175+ 项中反复出现）最容易累积的问题。

---

## 一、[duplicate-annotation] 重复括注（同一术语被 `中文（English）` 标注多次 —— 按"全文只标注一次/保留首现"规则，删除后续括注）

### 1. Multi-Instance GPU，MIG
- **首现（保留）** 第 147 行：`如果使用多实例 GPU（Multi-Instance GPU，MIG）分区`
- **重复（删括注）** 第 167 行：`_多实例 GPU（Multi-Instance GPU，MIG）_ 使用 MIG 把高端 GPU 划分为更小的实例……`
- **建议**：第 167 行清单项引导语改为 `_多实例 GPU（MIG）_` 或直接 `_多实例 GPU_`，删去 `（Multi-Instance GPU，MIG）`。
- **理由**：MIG 全称首次已在第 147 行括注；第 167 行为同一缩写的第二次完整括注。

### 2. 统一内存 / unified memory
- **首现（保留）** 第 25 行：`……统一内存（unified memory）、NVLink/NVSwitch 互连……`
- **重复（删括注）** 第 287 行：`如果使用统一内存（Unified Memory），就用显式预取（cudaMemPrefetchAsync）……`
- **建议**：第 287 行改为 `如果使用统一内存，就用显式预取（cudaMemPrefetchAsync）……`
- **理由**：重复括注；且大小写不一致（unified memory vs Unified Memory）。删后仅保留首现即可（`cudaMemPrefetchAsync`、`Unified Memory` 作为 CUDA 特性名在第 97 行另有 `CUDA Unified Memory`，属 API 保留，不受影响）。

### 3. 自动混合精度，AMP
- **首现（保留）** 第 19 行：`……先摘取像启用 AMP（自动混合精度，automatic mixed precision）和数据预取……`
- **重复（删括注）** 第 349 行：`……如今借助自动混合精度（automatic mixed precision，AMP），这已是大多数框架中的标准做法。`
- **建议**：第 349 行改为 `……如今借助自动混合精度（AMP），这已是……` 或直接 `……借助 AMP，这已是……`。
- **理由**：AMP 全称首次已在第 19 行括注。

### 4. 激活检查点，activation checkpointing
- **首现（保留）** 第 15 行：`……尝试启用激活检查点（activation checkpointing）。`
- **重复（删括注）** 第 351 行：`……并考虑用激活检查点（activation checkpointing）来降低极深网络中的内存占用。`
- **建议**：第 351 行改为 `……并考虑用激活检查点来降低……`（第 29、19 行等处已正确地不再括注，保持一致）。
- **理由**：首现在第 15 行已括注。

### 5. 张量内存加速器，Tensor Memory Accelerator（TMA）
- **首现（保留）** 第 295 行：`……优先使用张量内存加速器（Tensor Memory Accelerator）；对于细粒度的分阶段拷贝……`
- **重复（删括注）** 第 357 行：`_利用 Tensor Core 与张量内存加速器（Tensor Memory Accelerator，TMA）_ ……`
- **建议**：把缩写并入首现 —— 第 295 行改为 `张量内存加速器（Tensor Memory Accelerator，TMA）`；第 357 行清单项引导语改为 `_利用 Tensor Core 与张量内存加速器（TMA）_`（删去完整英文）。
- **理由**：缩写 TMA 在第 277 行"intro"（`cp.async/TMA`）就已出现，读者在首现处更需要完整全称+缩写；保留首现（295）、去重后续（357）。

### 6. epoch（每轮 / 轮次）
- **首现（保留）** 第 199 行：`……并加快每轮（epoch）的启动时间。`
- **重复（删括注）** 第 233 行：`……多次复用同一数据集（称为轮次，_epochs_）……`
- **建议**：第 233 行删去 `_epochs_` 括注（改为 `……多次复用同一数据集（称为一轮，epoch，即 epoch）` 不必要；直接 `……多次复用同一数据集（称为"轮次"）……`）。同时统一"每轮/轮次"二选一（见【consistency】第 3 条）。
- **理由**：epoch 首次已在第 199 行括注；此处为第二次括注且译词不一致。

### 7. tile / tiling（切块 / 分块）
- **出现 1** 第 277 行：`……把数据切块（tile）进共享内存……`
- **出现 2** 第 283 行：`……例如矩阵的分块（tiles）……这种流行的分块（tiling）技术……`
- **建议**：统一译词后只括注一次。建议统一为"分块"，在第 277 行首现处括注 `分块（tiling）`，第 283 行两处均用"分块"且不再括注。
- **理由**：同一 tile/tiling 概念在两个块内被括注 3 次，且中文在"切块""分块"间摇摆（详见【consistency】第 2 条）。

---

## 二、[terminology] / [consistency] 术语与一致性

### 1. KV cache 术语渲染不一致
- 术语表规定 `KV cache → KV 缓存（KV cache）`，首现已在第 93 行正确括注（`KV 缓存（KV cache）`）。
- **不一致点** 第 457 行仍使用英文原形："`……例如陈旧的注意力 KV cache 条目或不常访问的模型权重……`"
- **建议**：第 457 行 `KV cache` → `KV 缓存`，与第 421、429、433 等处统一。
- **理由**：同一术语在译文中中英混用（KV 缓存 vs KV cache），应统一为"KV 缓存"。

### 2. tile/tiling 中文译词不一致（切块 vs 分块）
- 第 277 行"切块"、第 283 行"分块"。
- **建议**：全篇统一为"分块"（配合上文【duplicate-annotation】第 7 条）。
- **理由**：同一概念两种译词，影响一致性。

### 3. epoch 中文译词不一致（每轮 vs 轮次）
- 第 199 行"每轮"、第 233 行"轮次"。
- **建议**：二者择一（建议"轮/一轮"作名词、"每轮"作分配语境），保持全篇一致。
- **理由**：同一 epoch 概念两种译词。

### 4. occupancy 首现未括注（低优先）
- 术语表要求 `occupancy → 占用率（occupancy）` 首现括注，但译文全篇"占用率"（第 291、293、311、341、357 等）**从未括注 occupancy**。
- **建议**：在首个自然出现处（第 291 行 `实现良好的占用率与资源的平衡`）补 `占用率（occupancy）`。
- **理由**：与"全文标注一次"规则一致——应恰好标注一次，而非零次。

### 5. gradient accumulation 括注位置偏后（低优先）
- 首次出现于第 237 行（`……或者对更小的批做梯度累积……`，未括注），而括注出现在更靠后的第 351 行（`梯度累积（gradient accumulation）`）。
- **建议**：将括注移到首现（第 237 行），第 351 行不再括注。
- **理由**：括注应落在首次出现处。

### 6. NIXL 展开位置偏后（低优先）
- NIXL 首次出现于第 307 行（`……在可用处使用 NIXL……`，未展开），完整名在第 429 行才给出（`NVIDIA Inference Xfer Library（NIXL）`）。
- **建议**：可接受现状（两处分属 CUDA 节与推理节）；如追求严格，可在第 307 行首现处补全称。
- **理由**：不影响理解，仅体例上首现未展开。

### 7. 小节标题 "GPU 编程与 CUDA 调优优化" 略显冗余（低优先）
- 第 275 行上方 H2：英文 `GPU Programming and CUDA Tuning Optimizations` → `## GPU 编程与 CUDA 调优优化`。
- **建议**：可作 `## GPU 编程与 CUDA 调优`（术语表即用"CUDA 调优"），"调优优化"读来重复。
- **理由**："调优"与"优化"语义叠加。此为风格建议，非错误。

---

## 三、[readability] 可读性

### 1. 第 291 行 "warp 在飞（in flight）"
- 英文：`enough warps in flight to cover latency`
- 当前：`……以确保有足够多的 warp 在飞（in flight）来掩盖延迟……`
- **建议**：`……确保有足够多的 warp 处于运行中（in flight）来掩盖延迟……`（或"在途""并发驻留"）。
- **理由**："在飞"为生硬直译，"处于运行中/在途"更自然。

### 2. 第 311 行 "把性能白白留在桌上"
- 英文：`ensure you're not leaving performance on the table`
- 当前：`……以确保你没有把性能白白留在桌上。`
- **建议**：`……以确保你没有白白浪费性能。`
- **理由**：`leave … on the table` 为英语习语，直译"留在桌上"是翻译腔。

---

## 四、[accuracy] 准确性

逐块核对了全部数字、单位、比值、命令与标识符，**未发现实质性误译、漏译、增译或逻辑反转**。已核实且正确的关键点（抽样）：

- 比值/倍数：`1.8 TB/s`、`超过 PCIe Gen5 的 14×`（第 71 行）、`900 GB/s`（第 89/111 行）、`~130 TB/s`（第 111 行）、`最多 7 个 MIG 切片`（第 167 行）、`1.5–3.5×`（第 349 行）、`2–4×`（INT8 相较 FP16）、`FP4 吞吐是 FP8 两倍`（第 355 行）、`2:4 结构化稀疏 → 50% 清零`（第 355/389 行）—— 均与英文一致。
- 单位/换算：`2 MB 页`（第 141 行）、`MTU 9000`（第 399 行）、`100 Gbps → 12.5 GB/s = 100 Gb/s ÷ 每字节 8 比特`（第 267 行）、`NCCL_BUFFSIZE 1 MB → 4 MB`、`NCCL_NTHREADS 4 → 8 或 16`（第 411 行）、`85°C`（第 473 行）、`功耗上限 100% → 80%，功耗少 20% 而速度几乎不变`（第 479 行）、`90% vs 两块各 45%`（第 475 行）—— 均正确。
- 命令/环境变量/API 原样保留且未改动：`nvidia-smi -pm 1 / -lgc/-lmc / -pl / dmon`、`cudaMallocAsync`、`cudaMemPrefetchAsync`、`cudaMemcpyAsync`、`torch.set_float32_matmul_precision("high")`、`torch.backends.cudnn.benchmark=True`、`NCCL_COLLNET_ENABLE=1`、`SHARP_COLL_LOCK_ON_COMM_INIT=1`、`SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD=0`、`torch._dynamo.mark_dynamic()`、`torch.library.triton_op` 等 —— 全部逐字符正确。

- **忠实保留的原文特点（非错误，供知悉）**：第 233 行英文原文写作 "`embeddings and V cache`"（疑为作者笔误，应为 KV cache），译文忠实保留为"V cache"，处理得当。

---

## 五、修改优先级汇总

**建议按此顺序应用（均为块内改动，不改变块数，保持 1:1 对齐）：**

| 优先级 | 行号 | 类别 | 修改要点 |
|---|---|---|---|
| 高 | 167 | duplicate-annotation | 删 MIG 完整括注（保留 147） |
| 高 | 349 | duplicate-annotation | 删 AMP 完整括注（保留 19） |
| 高 | 351 | duplicate-annotation | 删 activation checkpointing 括注（保留 15） |
| 高 | 287 | duplicate-annotation | 删 unified memory 括注（保留 25） |
| 高 | 357 | duplicate-annotation | 删 TMA 括注（并把 TMA 并入首现 295） |
| 中 | 233 | duplicate-annotation | 删 epoch 括注（保留 199） |
| 中 | 277/283 | duplicate/consistency | tile 统一"分块"，仅括注一次 |
| 中 | 457 | terminology | `KV cache` → `KV 缓存` |
| 低 | 291 / 311 | readability | "在飞"→"处于运行中"；"留在桌上"→"白白浪费" |

---

### 统计
- **[duplicate-annotation]**：7 组（MIG、unified memory、AMP、activation checkpointing、TMA、epoch、tile/tiling）
- **[terminology] / [consistency]**：7 项（KV cache 混用、切块/分块、每轮/轮次、occupancy 未括注、gradient accumulation 括注位置、NIXL 展开位置、标题"调优优化"冗余）
- **[readability]**：2 项（第 291、311 行）
- **[accuracy]**：0 项实质错误（另有 1 处忠实保留的原文笔误 V cache，处理得当）

**合计**：16 项（其中高优先 5 项、中 3 项、低 8 项）。
