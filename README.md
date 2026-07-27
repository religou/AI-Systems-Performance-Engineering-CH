# AI 系统性能工程（中文翻译）

本仓库是 O'Reilly 图书 **《AI Systems Performance Engineering》**（作者 Chris Fregly，2026 年出版）的中文翻译项目。内容涵盖从硬件架构、操作系统与容器编排，到分布式网络通信、GPU 存储 I/O 等 AI 系统全栈性能工程主题。

> 版权说明：本仓库仅为学习与交流用途的翻译稿。原书版权归 O'Reilly Media 及作者所有（Copyright © 2026 Flux Capacitor, LLC，ISBN 979-8-341-62778-9）。请勿用于任何商业用途。

## 翻译进度

| 章节          | 标题                                               | 状态      |
| ------------- | -------------------------------------------------- | --------- |
| 前言          | 前言                                               | ✅ 已完成 |
| 第 1 章       | 引言与 AI 系统概览                                 | ✅ 已完成 |
| 第 2 章       | AI 系统硬件概览                                    | ✅ 已完成 |
| 第 3 章       | 面向 GPU 环境的操作系统、Docker 与 Kubernetes 调优 | ✅ 已完成 |
| 第 4 章       | 分布式网络通信调优                                 | ✅ 已完成 |
| 第 5 章       | 基于 GPU 的存储 I/O 优化                           | ✅ 已完成 |
| 第 6 章及以后 | —                                                  | ⏳ 待翻译 |

## 目录结构

```
RawBook/
├── AI Systems Performance Engineering-preface-zh/   # 前言（中文）
│   └── 00_前言.md
├── AI Systems Performance Engineering-chN-zh/       # 第 N 章译文与相关产物
│   ├── translation.md      # 完整中文译文
│   ├── bilingual.md        # 中英对照
│   ├── glossary.md         # 术语表
│   ├── prompt.md           # 翻译约定 / 提示词
│   ├── mapping.json        # 中英段落映射
│   └── chunks/             # 分块翻译的中间产物
└── AI Systems Performance Engineering-chN_images/   # 章节配图
```

每一章的目录下包含以下主要产物：

- **translation.md**：该章的完整中文译文。
- **bilingual.md**：中英文逐段对照，便于校对。
- **glossary.md**：该章的术语表，统一专有名词译法。
- **prompt.md**：翻译时使用的约定与提示词。
- **mapping.json**：源文与译文的段落映射关系。
- **chunks/**：长文分块翻译的中间稿。

## 翻译约定

- 代码块保持原文，不作翻译。
- 专有名词在首次出现时以“中文（English）”形式括注，后续仅用中文。
- 数字、单位、公式与图表编号严格与原文一致（图注格式为“图 N-M.”）。
- 中文与英文/数字之间加空格；两个中文字符之间不加空格。
- 图片路径中的 `%20` 等转义保持不变。

## 阅读方式

- 直接阅读各章目录下的 `translation.md` 获取中文译文。
- 需要对照原文时，阅读 `bilingual.md`。
