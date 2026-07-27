# 审校报告（critique.md）— 第 2 章：AI 系统硬件概览

审校范围：原文 `AI Systems Performance Engineering-ch2.md` 对照译稿 `translation.md`，以 `glossary.md` 为术语基准。

总体评价：译文准确、通顺、完整，第一人称语气与专业热情保持良好。逐段核对了全部硬件数字/单位/型号/精度，均与原文一致；三个分块的两处衔接（“多 GPU 编程”节内、“Vera Rubin Superchip”节内）无重复、无漏译；10 张图片引用与 `%20` 编码路径、图注、来源 URL、引用块、粗体列表均完整保留。下列为可回源证伪的问题，多为术语一致性与文面格式，无实质性错译。

## 格式

- 年份括号全角/半角不一致：标题 `### Vera Rubin Superchip（2026）` 使用全角 `（2026）`，而 `### Rubin Ultra 与 Vera Rubin Ultra (2027)` 与 `### Feynman GPU (2028) 与每年翻一番` 使用半角 `(2027)` / `(2028)` → 三处统一为半角（或全角）括号，建议统一为半角以匹配英文型号语境，如 `（2026）` 改为 `(2026)`。

## 术语一致性

- billion：译文保留英文“billion”与中文混排 — “约 104 billion 晶体管”“大约有 208 billion 晶体管”“~80 billion 晶体管”三处，与同章 trillion 已译为“万亿”（如“1.8 万亿参数”）的处理不一致，读来突兀 → 统一改为中文数量，如“约 1040 亿个晶体管”“约 2080 亿个晶体管”“约 800 亿个晶体管”。

- NIC（network interface controller / card）：同一缩写出现两种中文译法并被重复括注 — “多 GPU 编程”节译为“网络接口控制器（network interface controller，NIC）”，“多机架与存储通信”节又译为“网络接口卡（Network Interface Card，NIC）”，其后正文再用“网卡” → 全章统一为一种中文译法（建议“网卡”或“网络接口卡”），缩写 NIC 仅在首次出现处括注一次，后续不再括注、不再变换中文词。

- HBM：已在“高带宽内存（high-bandwidth memory，HBM）”首次括注，但“Vera Rubin Superchip”节“更高带宽的 GPU 高带宽内存（HBM）”再次括注 → 依“全文只标注一次”原则删去后一处括注，直接用 HBM。

- superchip / Superchip：正文统一译“超级芯片”，但标题 `### Vera Rubin Superchip（2026）` 保留英文 Superchip（观察项）。此处作为产品名（VR200）保留可接受，与 glossary“产品/型号保留原文”一致；但同为路线图小节的 `### Blackwell Ultra 与 Grace Blackwell Ultra` 已含中文成分，建议确认标题风格是否需与之统一，或对该英文标题在正文首现处补一次中文对照，以免读者困惑。

## 可读性

- “以 8 TB/s 的速率喂送数据……”等段落整体流畅，无需拆分；主要可读性问题即上文“billion”中英混排一处 → 见术语一致性建议。

## 存疑（源文侧，低优先级，非译文缺陷）

- `~500 TB/s`：原文 “Grace CPU provides up to ~480 GB of LPDDR5X at up to ~500 TB/s” 疑为源文笔误（后文“NVIDIA Grace CPU”节将同一 LPDDR5X 带宽写作 “~500 GB/s”）。译文“速率高达 ~500 TB/s”忠实还原源文，处理正确；若后续与作者/编辑核对确认源文有误，可在译文加译注说明，此处无需改动译文。

---

## 主 agent 核实与处理

- 格式（年份括号）：**已修正** `### Vera Rubin Superchip（2026）` → `(2026)`，与 (2027)/(2028) 统一为半角。
- 术语（billion 中英混排）：**已修正** 104/208/~80 billion → 约 1040 亿 / 约 2080 亿 / 约 800 亿个晶体管。
- 术语（NIC）：**已统一** 为“网卡”，首次出现处括注 `网卡（network interface card，NIC）`（“多 GPU 编程”节），后文“多机架与存储通信”节删去重复括注，正文一律用“网卡”。
- 术语（HBM 重复括注）：**已删除** “Vera Rubin”节的重复括注，改为“带宽更高的 GPU HBM”。
- 术语（Vera Rubin Superchip 标题）：**保留** 英文 Superchip——作为产品线名（VR200），与正文“Grace Blackwell（GB200）Superchip”保留英文的处理一致；仅统一了括号。
- 存疑（~500 TB/s）：**最终已按用户确认更正为 ~500 GB/s**（源文 `ch2.md` 与译文 `translation.md` 同步修改）。经核 PDF，源书该处写 ~500 TB/s，而下一节将同一 LPDDR5X 带宽写作 ~500 GB/s（LPDDR5X 的合理量级为 GB/s），属源书内部不一致的低争议笔误；已两处一并更正并重新生成双语版。
- 另：翻译阶段 subagent 已修正低争议错误——把 OCR 误写的 `(oF)` 修正为 `NVMe-oF`，并去除一处赘余逗号。
