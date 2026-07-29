# 审校诊断报告 — 第 10 章

对照 `bilingual.md`（英文块 + 中文块逐块交替）、`translation.md` 与 `glossary.md` 完成审校。五张性能表（表 10-1..10-5）的数字、单位、百分比与核函数名（double_buffered_pipeline、warp_specialized_pipeline、warp_specialized_cluster_pipeline 等）经逐项核对均准确无误；未发现事实性偏移、漏译或跳段。以下仅记录可回源核实的问题。

## 术语一致性

用户明确要求核查"重复括注"。glossary.md 规定专业术语全文只在首次出现时括注原文一次。以下术语存在多处重复括注：

- **cooperative groups / 协作组**：首次已在开篇括注（第 5 行"带网格级与簇级同步的协作组（cooperative groups）"）。其后重复括注：第 483 行"当与协作组（cooperative groups）或线程块簇"、第 505 行"协作组（cooperative groups）让你能够以任意粒度" → 后续两处删去括注，仅保留"协作组"。

- **thread block cluster / 线程块簇**：首次已在第 5 行括注"线程块簇（thread block cluster，又称 …）"。其后重复：第 483 行"线程块簇（thread block cluster，稍后介绍）"、第 501 行"*线程块簇*（thread block cluster）" → 后续两处删去括注。

- **distributed shared memory / DSMEM**：首次已在第 5 行完整括注"分布式共享内存（distributed shared memory，DSMEM 或 DSM）"。第 652 行再次给出完整英文括注"分布式共享内存（distributed shared memory，DSMEM，将在下一节讨论）"，第 688 行又出现"分布式共享内存（DSMEM）" → 第 652、688 两处应删去括注，直接用"分布式共享内存"或"DSMEM"。

- **double buffering / 双缓冲**：首次已在第 21 行括注"双缓冲（double buffering）"。第 394 行重复"两阶段（two-stage）双缓冲（double buffering）流水线" → 删去"（double buffering）"括注。

- **warp specialization / warp 专门化**：首次已在第 21 行括注"warp 专门化（warp specialization）"。第 394 行重复"而 warp 专门化（warp specialization）方法" → 删去括注。（第 394 行整句集中出现了 double buffering、warp specialization 的冗余重复括注，建议整体清理。）

- **prefetch / 预取**：首次已在第 48 行括注"异步预取（prefetch）"。第 471 行重复"使用 TMA 为即将到来的任务预取（prefetch）张量分块" → 删去括注。

## 可读性

- "借助现代 NVIDIA，GPU 让你能够在单个 GPC 内的多个 SM 上，恰好协同调度簇中的两个线程块"（"线程块对"小节起始）：英文原文"With modern NVIDIA, GPUs let you…"本身逗号位置有误，中文照搬后"借助现代 NVIDIA，"读来突兀、指代悬空 → 建议改为"借助现代 NVIDIA GPU，你可以在单个 GPC 内的多个 SM 上恰好协同调度簇中的两个线程块"。

- "相反，顺序启动 1,000 个微小核函数会让 GPU 在任一时刻都有大量部分未被充分利用"（"持久化核函数"场景）："有大量部分未被充分利用"结构生硬 → 建议改为"会让 GPU 在任一时刻都有大部分处于未充分利用的状态"。

## 术语一致性（内部统一，次要）

- **leader / 主导块**：glossary.md 定为"主导块（leader block）"。译稿在"warp 专门化与线程块簇结合"小节将单独的"cluster leader"译为"簇的主导者（leader）"，而同段其后又用"主导块的加载器 warp" → 建议统一为"主导块"，即"作为簇的主导块"。

- **band of rows / 行带**：同一小节先后译为"不相交的行带"（"计算一个不相交的行带"）与"一段不同的行区间"（"计算一段不同的行区间"） → 建议统一为"行带"或"行区间"其一。
