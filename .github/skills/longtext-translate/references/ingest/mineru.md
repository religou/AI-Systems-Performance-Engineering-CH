# 文档 → Markdown 规范化

将 PDF、图片（PNG/JPG/WebP 等）、Word（.docx/.doc）等文档转换为翻译流程可用的本地 Markdown，若文档含图片则一并提取为本地文件。用户确认后进入翻译流程。

MinerU extract 支持的输入格式：PDF（.pdf）、图片（.png/.jpg/.jpeg/.webp/.gif/.bmp）、Word（.docx/.doc）、URL（远程文件）。

以下流程以 PDF 为主要示例。图片和 Word 文档的处理方式相同，仅 `--pages` 参数和 qpdf 页码范围拆分不适用。

## 总体流程

1. 向用户说明处理方案，取得同意。
2. 安装 CLI（如未安装），按需引导 Token 配置。
3. 运行 MinerU extract 提取。
4. 运行空行规范化脚本。
5. 将 Markdown 路径交给用户验收。

## 处理前确认

先检查当前环境是否已安装 MinerU CLI：

```bash
mineru-open-api version
```

### 向用户提问

使用 `AskUserQuestion` 工具一次性询问用户所有问题。

> 若当前 agent 并无该工具，请优先使用当前 agent 运行时提供的内置用户输入工具，例如 `request_user_input`、`clarify`、`ask_user`，或任何等效工具。如果实在没有这类工具，则输出带编号的纯文本消息，并要求用户针对每个问题回复所选编号/答案。

**全局需说明的信息**（放在提问之外）：

- MinerU 会将文档上传至 mineru.net 服务端处理，处理完成后不保留数据。
- 若当前环境未安装，将通过 `npm install -g mineru-open-api` 全局安装。
- MinerU extract 需要 API Token。若尚未配置，需用户自己新开终端完成一次性配置。

**问题一（页码确认）**—— 仅当用户为 PDF 指定了页码范围时出现。措辞参考：

> header: "页码范围"
> question: "你指定的页码范围是"12–32 页"。这是指哪种页码？"
> options:
>   - "物理页码 12–32" 说明: "PDF 阅读器显示的第 12–32 页（从第 1 页开始计数）"
>   - "印刷页码 12–32" 说明: "书页上印的页码 12–32"
>   - "我不确定" 说明: "请帮我探测页码偏移关系"

**问题二（处理方案）**：

> header: "提取方式"
> question: "你希望如何提取文档内容？"
> options:
>   - "使用 MinerU（推荐）" 说明: "自动提取为结构良好的 Markdown，格式还原度高。文档将上传至 mineru.net 服务端，处理完成后不保留数据"
>   - "我自行处理" 说明: "自行将文档转为 Markdown 后再翻译。选择此项流程终止"

选择"我自行处理"则流程终止，同时为用户提供如下建议：“可参考 OmniDocBench 排行榜（https://github.com/opendatalab/OmniDocBench）选择 PDF 处理工具。”

**若用户在页码确认环节选择"我不确定"**，用 pypdf 探测前几页的印刷页码，计算偏移量后向用户确认。探测脚本：

```bash
python3 -c "
from pypdf import PdfReader
r = PdfReader('file.pdf')
for i in range(min(5, len(r.pages))):
    print(f'物理第{i+1}页:', r.pages[i].extract_text()[:200])
"
```

若无法从文本推断印刷页码，请用户提供目标范围内一页的样张文字帮助定位。

#### AskUserQuestion 示例

仅作为示例；其他运行时请按其 schema 调整字段名。

```json
{
  "questions": [
    {
      "header": "页码范围",
      "multiSelect": false,
      "question": "你指定的页码范围是 \"12–32\"。这是指哪种页码？",
      "options": [
        {"label": "物理页码 12–32（推荐）", "description": "PDF 阅读器显示的第 12–32 页（从第 1 页开始计数）"},
        {"label": "印刷页码 12–32", "description": "书页上印的页码 12–32"},
        {"label": "我不确定", "description": "请帮我探测页码偏移关系"}
      ]
    }
  ]
}
```

## 环境准备

### 安装 CLI

若 `mineru-open-api version` 不可用：

```bash
npm install -g mineru-open-api
```

### 配置 Token

MinerU extract 需要 API Token。先检查是否已配置：

```bash
mineru-open-api auth --verify
```

若已配置，继续执行提取。

若未配置，agent **不能**在聊天中执行 `mineru-open-api auth`（该命令需要交互式输入，当前会话无法支持，会直接报错 EOF），也**不能**通过环境变量 `MINERU_TOKEN` 传递 Token（该方式会将 Token 暴露给 agent）。正确的做法是引导用户**新开一个终端窗口**，在本地终端中完成一次性配置：

> MinerU 需要 API Token 才能执行提取。请到 [mineru.net](https://mineru.net) 官网获取 Token，然后**新开一个终端窗口**运行以下命令完成一次性配置（Token 将写入 `~/.mineru/config.yaml`，后续所有会话永久生效）：
>
> ```
> mineru-open-api auth
> ```
>
> 完成后告诉我，我会继续处理。

用户确认后，再次 `mineru-open-api auth --verify` 验证通过则继续。

### 图片目录冲突处理

MinerU 固定将图片输出到 `images/` 子目录，无法自定义目录名。当 `-o` 直接指定为源文件所在目录时，若该目录下已有 `images/`（来自其他文档的提取），会发生图片混入。

Agent 在运行提取命令前需检查源文件目录下是否已有 `images/`：

- **无冲突**：`-o` 直接指定为源文件目录，一步到位。
- **有冲突**：先将 `-o` 指定到一个临时目录提取，完成后将 `images/` 重命名为 `{basename}-images/`，替换 Markdown 中的 `images/` 引用路径，再将产物移到源文件目录，清理临时目录。

## 命令模板

```bash
mineru-open-api extract "input.pdf" -o "{source_dir}" -f md --model vlm --language en
```

图片和 Word 文档同样适用，替换文件路径即可。

`-o` 直接指定为源文件所在目录（无图片冲突时）。MinerU 会在该目录下生成 `{basename}.md` 和 `images/` 子目录。若有图片目录冲突，按上文「图片目录冲突处理」操作。

可选参数：`--language`（默认 `ch`）、`--pages`（仅 PDF）。`--model` 固定 `vlm`。

**注意**：extract 的输出文件不自动包含页码范围信息。若使用了 `--pages`，提取完成后需手动将输出文件重命名为 `{basename}-pages-MM-NN.md`（命名规则见下文「输出命名」）。

### 页码范围（仅 PDF）

页码范围默认按 PDF 物理页码从 1 开始计数。页码范围含义应在「处理前确认」阶段已明确。

直接在提取时指定：

```bash
mineru-open-api extract "input.pdf" -o "{source_dir}" -f md --model vlm --language en --pages 10-25
```

若用户需要拆出指定页码范围的内容，使用 qpdf 工具：

```bash
qpdf input.pdf --pages . 10-25 -- output-pages-10-25.pdf
```

### 输出命名

产物保存在源文件同级目录，文件名从源文件 basename 派生：

```
papers/example.pdf
papers/example.md
```

页码范围在 basename 后追加页段信息：

```
papers/example-pages-10-25.md
```

## Markdown 空行规范化

提取完成后运行规范化脚本：

```bash
python3 {baseDir}/scripts/ingest/normalize_markdown.py --strip-details <markdown-file>
```

其中 `{baseDir}` 为本 SKILL.md 所在目录。

`--strip-details` 用于剥离 MinerU VLM 后端在图片/图表处插入的 `<details>` 折叠块（例如 `<details><summary>line</summary>...</details>`）。这些折叠块内是 VLM 对图片内容的文本分析，并非原文内容，翻译流程不需要它们。图片引用本身不受影响。

脚本执行以下规则：

- 连续空行最多保留一行。
- 分隔线、表格前后各保留一行空行（文档开头或结尾处省略）。
- 标题前保留一行空行（文档开头处省略）。
- 段落前后保留一行空行。
- 保留 YAML front matter 与正文之间的分隔空行。
- YAML front matter、fenced code block 和 display math block 内部原样保留。
- 开启 `--strip-details` 时，移除 `<details>...</details>` 块及其内部全部内容。

## 用户验收

### Agent 自查

规范化完成后，agent 先自行完成以下技术性检查，再向用户汇报：

1. **页码范围**（若用户指定）：提取的实际页码范围是否与用户需求一致。
2. **正文完整性**：内容是否连续、无明显截断或整段缺失。
3. **文字质量**：是否存在乱码、系统性 OCR 错误、artifact 字符。
4. **结构保真**：标题层级是否正确识别、段落边界是否清晰（无粘连或异常断行）。
5. **特殊内容**：检查图片引用路径是否存在、表格 Markdown 语法是否完整、公式分隔符是否闭合。

### Agent 汇报

逐项汇报检查结果，并对发现的问题分类处理：

- **阻断性问题**（页码范围错误、大面积乱码、整段缺失等）：提出具体修复方案（调整参数重新提取、切换提取模式等），**在修复前不进入翻译流程**。
- **需用户判断**（散在的 OCR 错误影响语义、标题归属模糊等）：标注具体位置和影响范围，请用户决定是否接受。
- **通过**：不影响翻译的细微排版问题，记录但不必阻拦。

汇报末尾给出明确判断：**建议进入翻译 / 建议重新提取 / 存在需你判断的问题**。

最后，执行完文档规范化后立即停止，不得在用户确认前推进任何后续翻译步骤。
