# 大型术语表的按需加载

## 适用场景

仅当用户术语约束是**条目庞大的术语表文件**（经验阈值：约 200 条以上）时才走本流程。命令中直接写出的术语映射、保留原文要求、几十条的小术语表，仍按主流程完整收集并应用，不要套用本流程，避免徒增步骤。

## 为什么这样做

术语表的作用是保证源文里出现的术语被一致、正确地翻译。庞大术语表里绝大多数词条往往不会在当前文档出现。若全量读入上下文：

- 普通翻译模式下，主 agent 的上下文被大量用不到的词条占据。
- 长文本模式成本被放大——`glossary.md` 由每个 subagent 读取，一份庞大术语表会被分片数乘一遍。

## 核心原则

把用户的完整术语表当作**数据库**，只让**源文中实际出现的子集**（工作集）进入上下文。全量术语表始终留在文件里，用 shell 过滤，不读入 agent 上下文。

## 操作流程

**第一步，识别格式。** 用 `head` 看术语表前几行，判断分隔形式和源词所在列，不要读取整个文件：

```bash
head -n 5 <glossary-file>
```

**第二步，按源文筛选。** 根据格式选下方对应命令，把全量表过滤成只含源文出现词条的子集。命令逐行取源词，仅当源词出现在源文中才保留该行，全程在 shell 完成，术语表不进上下文。占位符：`<glossary-file>` 为术语表，`<source-file>` 为规范化后的源文 Markdown，`<output-dir>` 为当前输出目录。

TSV（制表符分隔，源词在首列）：

```bash
while IFS= read -r line; do
  term=${line%%$'\t'*}
  [ -n "$term" ] && grep -qiF -- "$term" <source-file> && printf '%s\n' "$line"
done < <glossary-file> > <output-dir>/glossary.filtered.tsv
```

CSV（逗号分隔，源词在首列，字段无内嵌逗号）：

```bash
while IFS= read -r line; do
  term=${line%%,*}
  [ -n "$term" ] && grep -qiF -- "$term" <source-file> && printf '%s\n' "$line"
done < <glossary-file> > <output-dir>/glossary.filtered.csv
```

Markdown 表格（`| 源 | 译 |`，源词在第一列）：

```bash
while IFS= read -r line; do
  term=$(printf '%s' "$line" | awk -F'|' 'NF>=3{gsub(/^[ \t]+|[ \t]+$/,"",$2); print $2}')
  [ -n "$term" ] && grep -qiF -- "$term" <source-file> && printf '%s\n' "$line"
done < <glossary-file> > <output-dir>/glossary.filtered.md
```

`源: 译` 行（半角或全角冒号均可）：

```bash
while IFS= read -r line; do
  term=$(printf '%s' "$line" | sed -E 's/：/:/; s/:.*//; s/^[ \t]+//; s/[ \t]+$//')
  [ -n "$term" ] && grep -qiF -- "$term" <source-file> && printf '%s\n' "$line"
done < <glossary-file> > <output-dir>/glossary.filtered.txt
```

格式不在上述之列时，沿用同一骨架，只替换 `term=` 取源词的方式即可。

**第三步，核对并加载子集。** 对比筛选前后条数，向用户说明保留了多少条（`<ext>` 替换为上一步实际输出的扩展名）：

```bash
echo "原始 $(grep -c . <glossary-file>) 条，筛后 $(grep -c . <output-dir>/glossary.filtered.<ext>) 条"
```

随后主流程中所有「用户术语约束」改用筛选后的子集。长文本模式下，该子集即并入 `glossary.md` 的用户约束部分。

## 匹配策略与注意

- 匹配**偏向纳入**：`grep -iF` 为固定字符串、忽略大小写。源词是目标词形变体的子串（如 `Dasein` 命中 `Daseins`）能保住召回；偶尔误纳无关词条只是多留几行，无害。
- 仍可能漏掉不规则形变（如 `mouse`/`mice`）。漏掉的词条按主流程「未覆盖的专业术语使用行业公认的标准译法」兜底，不影响正确性，只是该词不强制走用户指定译法。
- `-F` 固定字符串匹配能避免源词含正则元字符时误判，务必保留。
- 命令假定 UTF-8 locale（macOS 默认）；逐条 grep 对文档级源文足够快，数千条亦在数秒内完成。
