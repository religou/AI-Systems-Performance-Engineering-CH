# EPUB → Markdown 规范化

本文档说明如何将 EPUB 中的指定内容转换为可供翻译流程使用的干净 Markdown。输出应保留章节标题、正文段落、语义性强调、列表、引用和必要图片，并排除目录、导航、封面、版权页、脚注回链等干扰内容。

## 总体流程

1. 确认 EPUB 路径和用户要翻译的内容范围。
2. 读取 EPUB spine，定位目标章节或 XHTML 文件。
3. 确认抽取边界。
4. 运行提取脚本，生成 Markdown 和本地图片目录。
5. 对照 EPUB 内建目录（TOC）校验 Markdown 标题层级并修正，随后检查图片引用、乱码字符和残留 HTML。
6. 使用规范化后的 Markdown 进入翻译方案确认。

## 边界确认

用户给出“前三章”“第 4～107 页”等表达时，不要直接把它们等同于 EPUB 内部顺序。先读取 spine，确认哪些 XHTML 文件对应目标内容。

若目标范围包含下一章封面、版权页、目录或其他非正文内容，先说明观察到的边界情况，并让用户确认是严格按范围抽取，还是按正文内容边界抽取。

## 输出命名

规范化 Markdown 默认保存到源 EPUB 同级目录，文件名从 EPUB basename 派生。文件名只做轻度路径清理：替换路径不安全字符、压缩连续空白、去掉首尾空白，必要时截断过长文件名。

整本或单一主体内容：

```text
books/example.epub
books/example.md
books/example_images/
```

部分章节：

```text
books/example.epub
books/example-chapters-1-3.md
books/example-chapters-1-3_images/
```

Markdown 中图片使用本地相对路径，例如 `![](example_images/cover.jpg)`。每个规范化 Markdown 使用独立图片目录，避免不同抽取结果互相覆盖。

## 使用提取脚本

优先使用 `{baseDir}/scripts/ingest/epub.py`。`{baseDir}` 为此 `SKILL.md` 所在目录路径。

### 列出 spine

```bash
python3 {baseDir}/scripts/ingest/epub.py path/to/book.epub --list-spine
```

输出左侧序号就是后续 `--spine-start` 和 `--spine-end` 使用的 0-based spine 序号。

### 按 href 指定章节

```bash
python3 {baseDir}/scripts/ingest/epub.py \
  path/to/book.epub \
  --href Text/Introduction.xhtml \
  --href Text/Chapter1.xhtml \
  --href Text/Chapter1-text.xhtml \
  --output path/to/book-chapters-1-3.md \
  --image-dir-name book-chapters-1-3_images
```

### 按 spine 序号范围

```bash
python3 {baseDir}/scripts/ingest/epub.py \
  path/to/book.epub \
  --spine-start 4 \
  --spine-end 14 \
  --output path/to/book-chapters-1-3.md \
  --image-dir-name book-chapters-1-3_images
```

`--spine-start` 和 `--spine-end` 都是闭区间。执行前根据 `--list-spine` 的结果确认边界。

### 列出目录

```bash
python3 {baseDir}/scripts/ingest/epub.py path/to/book.epub --list-toc
```

输出 EPUB 内建的目录树，`#` 个数表示标题层级，`←` 后为对应的内部文件路径。

输出示例：

```
# 第一章 开端  ← Text/Chapter1.xhtml
## 第一节 背景  ← Text/Chapter1.xhtml#s1
## 第二节 方法  ← Text/Chapter1.xhtml#s2
# 第二章 发展  ← Text/Chapter2.xhtml
```

用于后续章节的标题层级校验。

## 标题层级校验

EPUB 中部分出版社使用 `<p class="title">` 等非语义标记表示标题，而非标准的 `<h1>`–`<h6>` 标签。提取脚本无法识别此类标题，导致章节标题在 Markdown 中变成普通文本。通过对照 EPUB 内建目录逐条校验并修正。

### 操作步骤

**① 获取 TOC**

```bash
python3 {baseDir}/scripts/ingest/epub.py path/to/book.epub --list-toc
```

**② 对照检查**

在提取后的 Markdown 中搜索每条 TOC 条目文本，检查：
- 文本是否存在于 Markdown 中（被排除的前端内容如目录、版权页例外）
- 是否已用正确层级的 `#`–`######` 标记
- TOC 标签与 Markdown 实际文本可能存在空白和换行差异（TOC 标签为单行，Markdown 中可能因原排版断成多行）；匹配时先忽略空白和换行差异
- 若 TOC 与 Markdown 文本仍不完全一致（如 TOC 使用短标题、正文标题带副标题或标点差异），不要机械按字符串全等判断；应结合 `←` 后的内部文件路径，回到对应 XHTML 文件中定位实际标题，再决定是否修正

**③ 汇报并等待确认**

以简洁格式列出需修正的条目：

```
检测到以下章节标题未使用 Markdown 标题格式，建议修正：

1. 第一章 开端  → # 第一章 开端
2. 第一节 背景  → ## 第一节 背景

以下 TOC 条目在 Markdown 中未找到（已排除的前端内容，无需处理）：
- 目录
- 版权页
```

**不得静默修改**：向用户汇报后，等待确认再修改文件。

**④ 执行修正**

用户确认后，在 Markdown 中定位并修正。修正原则：

- **层级以 TOC 为准**：TOC 中嵌套多深，Markdown 中就是几个 `#`
- **文本以原文实际标题和 Markdown 实际文本为准**：TOC 标签可能缩写、换行或省略标点；若 TOC 与 Markdown 不完全一致，应先回到对应 XHTML 文件确认原文标题，再以 Markdown 中与原文一致的完整文本为准
- **仅转换 TOC 中明确出现的标题**：不推测 TOC 未列的小标题。若原文存在 TOC 未覆盖的子标题（如小节内的小标题），它们在原书中本就是视觉层级，保留当前形态即可
- 无法确定是否应转换的条目，列出疑虑并交由用户决定

## 清理原则

干净 Markdown 的判断标准：读者打开文件时只看到目标正文和必要图片，不需要理解 EPUB 内部结构。

默认排除：

- 目录页、导航页、landmarks；
- 封面、版权页、广告页等非目标正文；
- EPUB 内部锚点和 wrapper，例如 `<span id="...">`、`<div class="image">`；
- 排版用途的空白块、水平线、CSS class、无意义 id。

默认保留：

- 正文章节标题和小标题；
- 正文段落、列表、引用；
- 粗体、斜体等影响语义或阅读的强调；
- 正文内图片，使用本地相对路径。

## 验证清单

抽取完成后检查：

- Markdown 文件路径和大小是否合理
- 章节标题是否正确转换为 `#`–`######` 格式（按「标题层级校验」流程操作）
- 图片目录路径和图片数量是否合理
- 图片引用是否全部存在
- 是否有 `�` 等乱码替换字符
- 是否有残留 HTML 标签
- 是否按用户确认的章节边界截断

若校验发现边界可疑，不要自行修正成看似合理的范围；先说明观察到的内容，例如“最后一页已经进入 Chapter 4 封面”，再请用户确认。
