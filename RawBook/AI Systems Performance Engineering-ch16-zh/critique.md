# 第 16 章译文审校报告（critique.md）

对照 `bilingual.md`（英文块 + 中文块）与 `glossary.md` 完成。整体质量高：术语基本贴合术语表、表格数字与公式全部完好、代码/标识符均保留原文。主要问题集中在**跨分块的重复括注**（应只保留首次），另有零星准确性与一致性问题。以下按类别列出，`[category]` 标注。

---

## 1. ACCURACY（准确性）

**[accuracy] 倍数表述错误**
"Batching 4–8 requests can often double or triple the throughput over 1–2 request batches on large models."
current: "把 4–8 个请求组成一个批，其吞吐量往往能比 1–2 个请求的批**高出两到三倍**。"
fix: "……其吞吐量往往能达到 1–2 个请求批的**两到三倍**（即翻一到两番）。"
说明：`double/triple` = 变为 2×/3×；"高出两到三倍" 会被读成 3×–4×，含义被夸大。

**[accuracy/consistency] scaling factor / scale 译名混用**
glossary：scaling factor → 缩放因子。正文对 "scaling factor" 多处用"缩放因子"，但对 "scale/scales" 用"缩放系数"（如 "维护一个独立的缩放系数"、"微调量化缩放系数"）。
fix：可接受（scale 对缩放系数、scaling factor 对缩放因子），但建议全章统一为"缩放因子"以免读者困惑。属轻微一致性问题。

> 说明：7 张表（16-1 ~ 16-7）逐一核对，数字/公式完好，无误：
>
> - `T(6K)=18,003,000`、`38,007,000`、`3×T(2K)=6,003,000`、`22,005,000`、降幅 ~42% 均正确；
> - `20K×(20K+1)÷2=200M`、`N(N+1)÷2`、`O(N²)`、`O(NW)`、`200M` 全部保留正确；
> - 表 16-3 阈值（85/95 °C、>100 errors/sec、≥1 等）与表 16-1/16-4 单元格翻译准确。
>   未发现丢义、增义、数字/单位错误或逻辑反转。

---

## 2. TERMINOLOGY（术语）

- prefill / decode / token / KV cache / OOM / TTFT / TPOT / FIFO 均正确保留英文，符合术语表。
- 标识符、命令、指标名（`DCGM_FI_DEV_*`、`vllm:gpu_prefix_cache_*`、`nvidia-smi -lgc`、`enable_prefix_caching=True` 等）一律保留原文，正确。
- goodput→有效吞吐量、tail latency→尾延迟、head-of-line blocking→队头阻塞、preemption→抢占、eviction→逐出、quantization→量化 等均与术语表一致。

未发现术语偏离或标识符被误译的情况（仅有上文 scale/缩放系数的轻微不统一）。

---

## 3. DUPLICATE ANNOTATIONS（重复括注 —— 仅保留首次，删除其后括注）

以下术语的 `中文（English）` 括注在全章出现 **多次**；按规则应仅在**首次**出现处保留，其余删除英文括注。

| 术语                                             | 保留（首次）                                                                       | 删除英文括注（后续重复）                                                                                                          |
| ------------------------------------------------ | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| dynamic batching / 动态批处理                    | 全栈章节 "运行时策略，如动态批处理（dynamic batching）"                            | Dynamic Batching 节 "动态批处理（dynamic batching）是请求批处理的一种特化形式" → 改为"动态批处理"                                 |
| continuous batching / 连续批处理                 | 表 16-2 "（例如连续批处理，continuous batching）"                                  | Continuous Batching 节 "连续批处理（continuous batching），也称为" → 改为"连续批处理"                                             |
| eviction / 逐出                                  | 前段 "调整缓存大小、逐出（eviction）策略"                                          | Continuous Batching 节 "它逐出（eviction）已完成的请求" → 改为"它逐出"                                                            |
| prefix caching / 前缀缓存                        | 开篇 "前缀缓存（prefix caching）"                                                  | Prompt Cleansing 节 "用前缀缓存（prefix caching）来提升大型系统提示的效率" → 改为"前缀缓存"                                       |
| application-level optimization / 应用级优化      | 开篇 "应用级优化（application-level optimization）"                                | 小节标题 "## 应用级优化（application-level optimization）" → 改为"## 应用级优化"                                                  |
| quantization-aware training (QAT) / 量化感知训练 | 全栈章节 "量化感知训练（quantization-aware training，QAT）"                        | PTQ Workflow 节 "与之相对的是量化感知训练（quantization-aware training，QAT）" → 改为"量化感知训练（QAT）"                        |
| nonblocking collectives / 非阻塞集合通信         | Overlapping 节首次 "使用多个 CUDA 流和非阻塞集合通信（nonblocking collectives）"   | 同节结尾 "包括 CUDA 流、非阻塞集合通信（nonblocking collectives）和 RDMA" → 改为"非阻塞集合通信"                                  |
| all-to-all / 全对全                              | "以应对额外的全对全（all-to-all）通信"                                             | Overlapping 节 "使用 InfiniBand 进行全对全（all-to-all）通信"（及其后各处）→ 改为"全对全"                                         |
| microbatch / 微批                                | Overlapping 节 "为不同微批（microbatch）执行计算"                                  | Prefix Caching 节 "自动把传入查询归入微批（microbatch）" → 改为"微批"（注：另有 microbatching→微批处理 首次括注，属不同词，保留） |
| rage click / 暴躁点击                            | Streaming 节 "这可能引发暴躁点击（rage clicking）"                                 | Streaming 节稍后 "由一个沮丧的"暴躁点击（rage click）"型用户造成的" → 改为"暴躁点击"                                              |
| debouncing / 去抖                                | Debouncing 节标题句 "称为 _去抖_（debouncing）和 _请求合并_（request coalescing）" | 同段 "通过在响应前去抖（debounce）……" → 改为"去抖"                                                                                |

> 复核为"仅一次、无需改动"的术语（正确）：goodput 有效吞吐量、tail latency 尾延迟、head-of-line blocking 队头阻塞、prompt compression 提示压缩、quantization 量化、KV cache KV 缓存、full-stack optimization 全栈优化、knowledge distillation 知识蒸馏、pruning 剪枝、buffer pooling 缓冲池化、prefix caching 之外的 prefix merge/prefix tree、model cascading 模型级联、request coalescing 请求合并、preemption 抢占、speculative decoding 推测解码、latency-aware/continuous/stall-free scheduling、FIFO 先进先出、graceful degradation 优雅降级、arithmetic intensity 算术强度 等。

---

## 4. READABILITY（可读性）

整体流畅、CJK-Latin 间距处理良好，无明显翻译腔。仅少量引号内标识符（如 "file0"、`<POLICY_A>`、读作"try"）未加中英空格，但均在引号/尖括号内，属可接受，无需修改。

未发现需要改写的明显可读性问题。

---

## 5. CODE / STRUCTURE（代码与结构）

- 两段 NVTX C++ 代码、CUDA stream 代码、vLLM/Dynamo 日志、RadixAttention 伪代码：中文块均**原样保留**英文代码（bilingual 的"译文"即原代码），正确，标识符未被翻译。
- 图题（图 16-1 ~ 16-10）、表题（表 16-1 ~ 16-7）、引用块（blockquote）结构与英文一一对应，未见错位。
- 表格列头/单元格对齐正确，Markdown 表格语法完好。

未发现结构性问题。

---

## 摘要（Summary）

共记录约 **13** 处问题：ACCURACY 2（1 处倍数表述夸大、1 处 scale/缩放系数轻微不统一）、TERMINOLOGY 0（仅并入上条）、DUPLICATE ANNOTATIONS 11（跨分块重复括注，应删除后续英文括注）、READABILITY 0、CODE/STRUCTURE 0。总体译文质量**优秀**：语义忠实、术语规范、表格数字与公式全部无误、代码与标识符处理正确；主要待办是清理跨分块重复的英文括注（每个术语只在首次出现处标注），以及修正"double/triple throughput"的倍数表述。修订工作量小，属收尾润色级别。
