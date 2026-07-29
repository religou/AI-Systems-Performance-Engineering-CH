# 审校诊断报告 — 第 9 章《提升 CUDA 核函数效率与算术强度》

对照 bilingual.md 逐块核对。译文整体准确、通顺，术语基本贴合 glossary.md。主要问题集中在“专业术语首次出现括注、全文只标注一次”这一规则上（存在多处重复括注，且个别术语首次出现未括注），另有个别表格单元未完整汉化及一处英文习语直译。未发现事实/数字/单位/代码标识符层面的偏移。

## 完整性

- 表 9-2 单元格未完整汉化：`| 优化后的 C++（编译器调度） | 35% of cycles | 1.5 | 1.07 ms | 1.0× base |` 与 `| 手工调优的 PTX（手工调度与提示） | 20% of cycles | 1.6 | 1.00 ms | 1.07× |` — “Warp 停顿（内存）”列的 `35% of cycles`、`20% of cycles` 及加速比列的 `1.0× base` 仍为英文（对照表 9-1 中 `~60（略高）`、`低（简单的模板配置）` 等已汉化）。→ 建议改为 `占周期的 35%`／`占周期的 20%`、`1.0×（基准）`，与全表风格一致。

## 可读性

- “left on the table”直译为“留在桌面上”：“以榨出那些可能被**留在桌面上**的最后一点性能” — 该英文习语意为“尚未被榨取／白白浪费的收益”，直译为“留在桌面上”属翻译腔，易误解。→ 建议改为“以榨出那些**尚未被榨取的**最后一点性能”或“……**本可获得却被白白浪费的**最后一点性能”。

## 术语一致性

说明：以下均为“同一术语在译文中被多次括注原文”的情况，按 glossary.md“全文只标注一次”的规则，应仅保留首次出现处的括注，其余删除括注（保留中文即可）。

- microoptimization / micro-optimization：`深入底层微优化（microoptimization）的人`（PTX 节）与 `为了说明可能的微优化（microoptimization），设想` — 括注两次；且概念首次出现在更早的 rsqrt 边注“这是一种常见的**微优化**”处却未括注。→ 建议把括注移到首次出现（rsqrt 边注）处，后两处删除括注。
- forward compatible：`PTX 通常是前向兼容（forward compatible）的` 与 `因为它并不总是前向兼容（forward compatible）` — 括注两次。→ 保留首次，删除第二处括注。
- superchip：`Grace Blackwell 这类 CPU-GPU 超级芯片（superchip）上` 与 `现代 CPU-GPU 超级芯片（superchip）上` — 括注两次。→ 保留首次，删除第二处括注。
- multicast：`Tensor Memory Accelerator（TMA）的多播（multicast）特性` 与 `带分布式共享内存（DSMEM）的 TMA 多播（multicast）` — 括注两次。→ 保留首次，删除第二处括注。
- thread block pairs：`称为*线程块对*（thread block pairs）` 与 `CUTLASS 模板会透明地使用线程块对（thread block pairs）` — 括注两次。→ 保留首次，删除第二处括注。
- thread block clusters：`CUDA 线程块簇（thread block clusters，将在第 10 章讨论）` 与 `CUTLASS 还会利用线程块簇（thread block clusters）` — 括注两次。→ 保留首次，删除第二处括注。
- nested tensors：`PyTorch 的嵌套张量（nested tensors）` 与关键要点 `融合注意力和嵌套张量（nested tensors）` — 括注两次。→ 保留首次，删除关键要点处括注。
- elementwise operations：`自动融合一连串的逐元素操作（elementwise operations）`、定义块 `*逐元素操作*（elementwise operations）` 与关键要点 `通过融合逐元素操作（elementwise operations）来合并` — 括注三次。→ 保留首次；定义块可视体裁保留，但关键要点处应删除括注。
- software prefetching：`一项与之密切相关的技术是软件预取（software prefetching）` 与关键要点 `从而实现软件预取（software prefetching）` — 括注两次。→ 保留首次，删除关键要点处括注。
- multilevel tiling：`用多级分块（multilevel tiling）进一步提高强度` 与关键要点 `用多级分块（multilevel tiling）把数据暂存` — 括注两次。→ 保留首次，删除关键要点处括注。
- distributed shared memory (DSMEM)：首次为 `分布式共享内存（distributed shared memory，DSMEM）`，后文又出现 `带分布式共享内存（DSMEM）的 TMA 多播` — 第二处再次括注 DSMEM。→ 第二处删除括注，仅用“分布式共享内存”。
- MMA（matrix multiply-accumulate）：首次出现于 `MMA 片段 API（fragment APIs）`（未括注 MMA 本身），而括注延后到图 9-7 `矩阵乘加（matrix multiply-accumulate，MMA）`。→ 建议将 MMA 的括注前移至首次出现处，保持“首次即括注”。
