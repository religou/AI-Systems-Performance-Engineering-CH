当每个 warp 的访问落在尽可能少的 128 字节缓存行内时，就实现了合并。请合理组织你的数据和线程索引，使每个 warp 内的线程读取连续的 4 字节字，从而让硬件把它们融合成少数几个 128 字节事务。合并的内存加载能最大化有效 DRAM 带宽（即全局内存加载效率，Global Memory Load Efficiency），并把平均每次请求的扇区数降到最优的 4.0。使用较新版本的 Nsight Compute 时，你还可以使用 sm__sass_data_bytes_mem_* 计数器以及 gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed 来剖析并优化内存合并。

*向量化加载/存储*

使用内置向量类型（如 float4）来处理 16 字节向量。在 Blackwell 上配合 CUDA 13+，当能够证明 32 字节对齐成立时，优先采用每线程 32 字节向量，这包括 double4 或自定义结构体 alignas(32) { float v[8]; }。这样能降低每字节的指令数，并在正确对齐时把每次请求的扇区数保持在理想的 4.0。如此一来，每个线程可以在一条指令中搬运尽可能多的元素。每个 warp 的 128 字节事务数量与请求的总字节数成正比。要注意对齐：确保你的数组以至少 16 字节对齐来分配以配合 float4，而 cudaMalloc 默认通常就以 256 字节对齐来完成这一点。未对齐的向量访问会让这些收益付诸东流。

*避免 bank 冲突*

对共享内存数组做填充（例如，为 32 线程的 warp 把每行做成 33 个 float 宽），使任意两个线程不会在同一周期命中同一个 bank。消除 bank 冲突可恢复共享内存的全部吞吐。可以尝试用混洗（swizzling）来实现比填充稍微更省内存的方案。

*共享内存分块与数据复用*

把工作集暂存到片上共享内存中（例如把矩阵按 32 × 32 的块做分块），使每个元素只从 DRAM 取一次，却能在 SM 上被多次使用。这能提升算术强度，并把核函数推向计算受限。

*只读数据缓存*

把小而静态的查找表或系数标记为 const __restrict__，以便编译器在适用时把这些加载路由到只读数据路径。相比 DRAM，统一广播的延迟更低，能避免冗余事务，并可由片上缓存来提供服务。

*用流重叠主机—GPU 拷贝*

把主机缓冲区分配为页锁定（“固定”，pinned）内存，并在多个流上使用 cudaMemcpyAsync，以便把 H2D/D2H 传输与核函数执行重叠。固定内存可启用异步 DMA 传输，而多个流则让拷贝与核函数执行重叠，从而隐藏 PCIe 或 NVLink 的传输延迟。优先使用带显式流和事件的 cudaMemcpyAsync 来重叠 H2D/D2H 与核函数。请记住，可分页（非固定）内存会禁用 DMA 重叠。你应当核实使用的是可分页还是固定内存，因为观测到的传输速率会随这一配置而大相径庭。

*用 TMA + Pipeline API 做异步预取*

当满足对齐与作用域要求时，使用 C++ libcu++ 的 barrier 与 pipeline API（cuda::barrier 与 cuda::pipeline）配合 cuda::memcpy_async 来驱动 TMA（cp.async.bulk.tensor），完成全局内存 → 共享内存的批量拷贝。这会把合并的、跨步的（乃至多维的）拷贝卸载到共享内存中，并借助双缓冲（double buffering）与计算重叠。这将减轻 SM 的 LD/ST 单元压力，让 SM 专注于计算。

值得注意的是，libcu++ 的 pipeline API 与 TMA 仍在持续演进。对于两阶段乒乓（ping-pong）缓冲，优先采用分阶段形式（例如 producer_acquire()、producer_commit()、consumer_wait()、consumer_release()）。除非你确实需要簇作用域或网格作用域的拷贝，否则请使用块作用域的 pipeline（例如 cuda::thread_scope_block）或块作用域的 barrier（例如 cuda::barrier<cuda::thread_scope_block>）。

*用剖析来指引你*

依靠 Nsight Compute 指标，如全局内存加载效率、平均每次请求的扇区数、共享内存 bank 冲突、SM 活跃百分比、warp 停顿原因等。此外，查看 Nsight Systems 时间线，以可视化重叠与停顿、定位瓶颈，并验证每一项优化。

## 结论

如今，你的内存访问流水线已经全速运转——合并的全局内存加载、无冲突的分块、向量化的取数、只读缓存以及 TMA 驱动的预取，你已经消除了最大的数据搬运瓶颈，让 SM 得以全速运行。

本章自始至终都依靠 Nsight Compute 与 Nsight Systems，精确暴露 warp 在何处因缺数据而“挨饿”。我们还用它们逐步确认：每一项优化确实减少了停顿、消解了浪费的事务，并提升了持续带宽。每当你调优一个新核函数时，这些工具始终是你的指路明星。

在下一章，我们将介绍 CUDA 与 GPU 编程中一些基础的延迟隐藏技术。这些技术包括调优占用率、提升 warp 效率、避免 warp 分歧，以及暴露指令级并行。
