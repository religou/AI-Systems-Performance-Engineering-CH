# 双语输出版本

双语输出是翻译流水线的最后一步——翻译、审校、精编全部完成后执行。工具链的职责分工：脚本负责解析结构、生成候选对齐、按映射拼接；Agent 负责在脚本标出的少量风险点上做局部确认。

**注意**：若双语版已生成，用户随后又提出审校或精编，此时操作对象还是 `translation.md`（而非 `bilingual.md`）。审校或精编完成后，再重新生成双语版、删除旧版，并告知用户这一行为。

## 工具：bilingual.py

路径：`scripts/bilingual.py`，依赖 `markdown-it-py>=4.0,<5`。

### dump

```bash
python3 {baseDir}/scripts/bilingual.py dump <file>
```

将 Markdown 解析为块列表（JSON），每块记录 `index`、`kind`（heading / flow / code / thematicBreak / html）、`heading_level`、`words`、`md`。

### analyze

```bash
python3 {baseDir}/scripts/bilingual.py analyze <source> <translation> [-o report.json] [--mapping-out mapping.json]
```

加载原文与译文（支持 Markdown 文件或 dump JSON），输出结构分析报告和候选分组映射。`analyze` 用动态规划在两组 block 序列之间寻找成本最低的全局对齐，考虑五种操作：匹配、原文合并（2→1）、译文拆分（1→2）、原文独有（1→0）、译文独有（0→1）。成本基于块类型、标题层级和词数区间。

输出报告的关键字段：

```json
{
  "ready": false,
  "risk_level": "medium",
  "stats": {"source_blocks": 82, "translation_blocks": 86, "review_regions": 3},
  "warnings": [
    {"kind": "possible-merge", "severity": "medium", "entry_index": 11,
     "source_indices": [10, 11], "translation_indices": [11],
     "reason": "short adjacent source flow block likely merged into one translation block"}
  ],
  "mapping": {
    "version": 2, "kind": "bilingual-mapping",
    "source": {"path": "...", "sha256": "...", "block_count": 82},
    "translation": {"path": "...", "sha256": "...", "block_count": 86},
    "entries": [
      {"source": [0], "translation": [0]},
      {"source": [10, 11], "translation": [11], "review": {"severity": "medium", "reason": "..."}},
      {"source": [], "translation": [5], "review": {"severity": "low", "reason": "translation-only heading"}}
    ]
  }
}
```

- **ready**：无 medium 及以上 severity warning 时为 `true`，可直接 `join`
- **warnings**：每个 warning 的 `entry_index` 直接指向 `mapping.entries` 中需复核的条目
- **mapping**：分组映射对象，`join` 的主输入。每个 entry 的 `source` 和 `translation` 为索引数组，组合决定对齐方式：

| source | translation | 含义 |
|--------|-------------|------|
| `[s]` | `[t]` | 1:1 |
| `[s1, s2]` | `[t]` | 两段原文并为一段译文 |
| `[s]` | `[t1, t2]` | 一段原文拆为两段译文 |
| `[s]` 或 `[]` | `[]` 或 `[t]` | 单侧独有（如精编阶段新增的小标题） |

带 `review` 字段的 entry 需 Agent 确认。修正时只改对应 entry 的 `source` / `translation` 数组或删除 `review`——不必重写整份映射。Agent 也可自行构造分组映射 JSON 直接给 `join`，无需经过 `analyze`。

### join

```bash
python3 {baseDir}/scripts/bilingual.py join <mapping> <source> <translation> -o <output>
```

按映射拼接双语 Markdown。输入接受分组映射（`analyze --mapping-out` 产物）或旧式 `[[src_idx, tgt_idx], ...]` 数组。

拼接前校验：

| 类型 | 检查项 | 不通过时 |
|------|--------|----------|
| 阻断 | 索引越界、重复索引、checksum / block_count 不匹配、不兼容的 block 类型配对 | 报错退出 |
| 警告 | 连续 ≥4 个单侧 entry、标题层级不一致、被跳过的非标题 block | 记入审计输出 |

分组映射的拼接规则：每个 entry 内先依次输出所有 source block，再依次输出所有 translation block。2:1 合并场景下两段原文连续出现，再跟合并后的一段译文——合并关系以完整局部呈现。

## Agent 工作流

1. `bilingual.py analyze source.md translation.md` 生成分析报告
2. 检查 `warnings`，按 `entry_index` 定位 `mapping.entries` 中需复核的条目，确认或修正
3. 确认后 `bilingual.py analyze ... --mapping-out mapping.json` 落盘，或直接保存修正后的 mapping JSON
4. `bilingual.py join mapping.json source.md translation.md -o bilingual.md`

若 `analyze` 输出不满足需求且修正成本过高，可回退到手工流程：分别 `dump` 两份文件 → 手工编写 mapping → `join`。

## 已知局限

- **无语义内容匹配**：DP 仅比较结构特征（块类型、标题层级、词数区间），不比较文本内容。相邻 block 结构 profile 相似时，对齐边界可能偏移 1–2 个 block。`warnings` 和 `review` 机制即为此而设——标出可疑区域交由 Agent 裁决。
- **合并边界不精确**：`analyze` 能标出"这里大概率有合并"，但具体哪几段对哪一段可能需要微调。
- **agent 复核是正常流程**：`ready: false` 或 `risk_level: medium/high` 时 Agent 应审阅对应区域。这不是工具失败，而是工具完成了"暴露风险、缩小范围"的职责。
- **陈旧映射**：`analyze --mapping-out` 产物含 SHA256 和 block 数量，`join` 会比对当前文件。若翻译或精编后文件有变，`join` 报错，需重新 `analyze`。
