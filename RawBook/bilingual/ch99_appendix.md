# Appendix. AI Systems Performance Checklist (175+ Items)

# 附录 AI 系统性能检查清单（175+ 项）

This extensive checklist covers both broad process-level best practices and detailed, low-level tuning advice for AI systems performance engineers. Each of these checklist items serves as a practical reminder to squeeze maximum performance and efficiency out of AI systems.

这份详尽的检查清单（checklist）既涵盖宏观的进程级最佳实践（best practices），也包含面向 AI 系统性能工程师的细致底层调优建议。其中每一条清单项都是一个实用提醒，帮助你榨取 AI 系统的极致性能与效率。

Use this guide when debugging, profiling, analyzing, and tuning one’s AI systems. By systematically applying these tips—from low-level OS and CUDA tweaks up to cluster-scale optimizations—an AI systems performance engineer can achieve both lightning-fast execution and cost-effective operation on modern NVIDIA GPU hardware using many AI software frameworks, including CUDA, PyTorch, OpenAI’s Triton, TensorFlow, Keras, and JAX. The principles in this checklist will also apply to future generations of NVIDIA hardware, including their GPUs, ARM-based CPUs, CPU-GPU superchips, networking gear, and rack systems.

在调试、剖析（profiling）、分析与调优 AI 系统时使用本指南。通过系统性地套用这些建议——从底层 OS 与 CUDA 微调，直到集群级优化——AI 系统性能工程师就能在现代 NVIDIA GPU 硬件上，借助众多 AI 软件框架（包括 CUDA、PyTorch、OpenAI 的 Triton、TensorFlow、Keras 与 JAX）实现闪电般的执行速度与高性价比的运营。本清单中的原则同样适用于未来各代 NVIDIA 硬件，包括其 GPU、基于 ARM 的 CPU、CPU-GPU 超级芯片（superchip）、网络设备与机架系统。

## Performance Tuning and Cost Optimization Mindset

## 性能调优与成本优化心态

A pragmatic, documented loop—quick wins before deep work—turns engineering time into measurable ROI. Start by targeting the biggest runtime and cost drivers, and always profile before and after to verify impact.

一个务实且有据可查的循环——先拿下速赢，再做深水区工作——能把工程时间转化为可衡量的 ROI。先从最大的运行时与成本驱动因素入手，并且始终在优化前后进行剖析以验证效果。

Combine auto-tuning, framework upgrades, cloud pricing levers, and utilization dashboards for high-ROI wins, documenting results and favoring simple, maintainable fixes. Tune throughput-sensitive hyperparameters when accuracy allows. Here are some tips on the performance tuning and cost optimization mindset:

把自动调优（autotuning）、框架升级、云定价手段与利用率（utilization）仪表盘结合起来，去争取高 ROI 的胜果，记录结果并偏好简单、易维护的修复。在精度允许时，调优对吞吐量（throughput）敏感的超参数（hyperparameter）。以下是关于性能调优与成本优化心态的一些建议：

*Optimize the expensive first* Use the 80/20 rule. Find the top contributors to runtime and focus on those. If 90% of the time is in a couple of kernels or a communication phase, it’s better to optimize those deeply than to microoptimize something taking 1% of the time. Each chapter’s techniques should be applied where they matter most. For example, if your training is 40% data loading, 50% GPU compute, and 10% communication, then first fix data loading, as you can maybe halve the overhead. Then look at GPU kernel optimization.

_优先优化昂贵之处_ 运用 80/20 法则。找出运行时的最大贡献者并聚焦于它们。如果 90% 的时间花在少数几个核函数（kernel）或某个通信阶段，那么深度优化它们要比微优化只占 1% 时间的部分更划算。每一章的技术都应用在最要紧的地方。例如，如果你的训练是 40% 数据加载、50% GPU 计算、10% 通信，那么先修复数据加载，因为你或许能把其开销减半。然后再看 GPU 核函数优化。

*Profile before and after* Whenever you apply an optimization, measure its impact. This sounds obvious, but often tweaks are made based on theory and might not help—or even hurt—in practice. Consider a scenario where your workload is not memory-limited, but you decide to try enabling activation checkpointing for your training job. This may actually slow down the job by using extra compute to reduce memory. In other words, always compare key metrics like throughput, latency, and utilization before and after making changes. Use the built-in profilers for simple timing, such as average iteration time over 100 iterations.

_在优化前后进行剖析_ 每当你实施一项优化，都要衡量它的影响。这听起来显而易见，但很多调整是基于理论做出的，在实践中可能无益——甚至有害。设想这样一个场景：你的工作负载并非受内存限制，但你决定为训练作业尝试启用激活检查点（activation checkpointing）。这实际上可能因为用额外计算换取内存而拖慢作业。换言之，在做改动前后，始终对比吞吐量、延迟（latency）、利用率等关键指标。使用内置剖析器（profiler）做简单计时，例如 100 iterations 的平均迭代时间。

*Embrace adaptive autotuning feedback loops* Implement advanced autotuning frameworks that leverage real-time performance feedback—using techniques like reinforcement learning or Bayesian optimization—to dynamically adjust system parameters. This approach enables your system to continuously fine-tune settings in response to changing workloads and operating conditions.

_拥抱自适应自动调优反馈回路_ 实现先进的自动调优框架，利用实时性能反馈——运用强化学习（reinforcement learning，RL）或贝叶斯优化（Bayesian optimization）之类的技术——来动态调整系统参数。这种方式使你的系统能够持续微调设置，以应对不断变化的工作负载与运行条件。

*Budget for optimization time* Performance engineering is an iterative investment. There are diminishing returns—pick the low-hanging fruit like enabling AMP and data prefetch. These might give 2× easily. Harder optimizations like writing custom kernels might give smaller increments. Always weigh the engineering time versus the gain in runtime and cost saved. For large recurring jobs like training a flagship model, even a 5% gain can justify weeks of tuning since it saves maybe millions. For one-off or small workloads, focus on bigger wins and be pragmatic.

_为优化时间预留预算_ 性能工程是一项迭代式投资。它存在边际递减效应——先摘取像启用 AMP（自动混合精度，automatic mixed precision）和数据预取（data prefetch）这样的低垂果实。这些也许轻松就能带来 2× 的提升。而像编写自定义核函数这样更难的优化，带来的增量可能较小。始终权衡工程时间与运行时收益及节省的成本。对于像训练旗舰模型这类大型且反复运行的作业，哪怕 5% 的收益也可能值得数周的调优，因为它也许能节省数百万。对于一次性或小型工作负载，聚焦于更大的胜果并保持务实。

*Stay updated on framework improvements* Many optimizations we discussed, such as mixed precision, fused kernels, and distributed algorithms, continue to be improved in deep learning frameworks and libraries. Upgrading to the latest PyTorch or TensorFlow can sometimes yield immediate speedups as they incorporate new fused ops or better heuristics. Leverage these improvements, as they are essentially free gains. Read release notes for performance-related changes.

_紧跟框架改进_ 我们讨论过的许多优化，例如混合精度、融合核函数（fused kernel）和分布式算法，在深度学习框架与库中持续得到改进。升级到最新的 PyTorch 或 TensorFlow，有时能立即带来加速，因为它们纳入了新的融合算子或更优的启发式策略。善用这些改进，它们本质上是免费的收益。阅读发行说明，关注与性能相关的变更。

*Codesign collaboratively with vendors and community members* Stay connected with hardware vendors and the broader performance engineering community to align software optimizations with the latest hardware architectures. This codesign approach can reveal significant opportunities for performance gains by tailoring algorithms to leverage emerging hardware capabilities. Regularly review vendor documentation, participate in forums, and test beta releases of drivers or frameworks. These interactions often reveal new optimization opportunities and best practices that can be integrated into your systems. Integrating new driver optimizations, library updates, and hardware-specific tips can provide additional, sometimes significant, performance gains.

_与厂商和社区成员协同设计_ 与硬件厂商及更广泛的性能工程社区保持联系，使软件优化与最新的硬件架构对齐。这种协同设计（codesign）方式能通过定制算法以利用新兴硬件能力，揭示出可观的性能提升机会。定期查阅厂商文档、参与论坛、测试驱动或框架的 beta 版本。这些互动往往能揭示可整合进你系统的新优化机会与最佳实践。整合新的驱动优化、库更新与硬件专属技巧，能带来额外的、有时相当可观的性能收益。

*Leverage cloud flexibility for cost* If running in cloud environments, use cheaper spot instances or reserved instances wisely. They can drastically cut costs, but you may lose the spot instances with a few minutes’ notice. Also consider instance types, as sometimes a slightly older GPU instance at a fraction of the cost can deliver better price/performance if your workload doesn’t need the absolute latest. Our discussions on H800 versus H100 showed it’s possible to do great work on second-best hardware with effort. In the cloud, you can get similar trade-offs. Evaluate cost/performance by benchmarking on different instance configurations, including number of CPUs, CPU memory, number of GPUs, GPU memory, L1/L2 caches, unified memory, NVLink/NVSwitch interconnects, network bandwidth and latency, and local disk configuration. Calculate metrics like throughput per dollar to guide your optimization decisions.

_利用云的灵活性控制成本_ 如果在云环境中运行，明智地使用更便宜的竞价实例（spot instance）或预留实例（reserved instance）。它们能大幅削减成本，但你可能在几分钟的通知后就失去竞价实例。此外还要考虑实例类型，因为有时一个成本仅为零头的稍旧 GPU 实例，若你的工作负载并不需要绝对最新的硬件，反而能带来更好的性价比。我们对 H800 与 H100 的讨论表明，只要肯下功夫，用次优硬件也能做出出色的工作。在云上，你能获得类似的取舍空间。通过在不同实例配置上做基准测试来评估性价比，配置维度包括 CPU 数量、CPU 内存、GPU 数量、GPU 内存、L1/L2 缓存、统一内存（unified memory）、NVLink/NVSwitch 互连、网络带宽与延迟以及本地磁盘配置。计算诸如每美元吞吐量这样的指标，以指导你的优化决策。

*Monitor utilization metrics* Continuously monitor GPU utilization, SM efficiency, memory bandwidth usage, and, for multinode, network utilization. Set up dashboards using DCGM exporter, Prometheus, etc., so you can catch when any resource is underused. If GPUs are at 50% utilization, dig into why. It’s likely data waiting/stalling and slow synchronization communication. If the network is only 10% utilized but the GPU waits on data, maybe something else like a lock is the issue. These metrics help pinpoint which subsystem to focus on.

_监控利用率指标_ 持续监控 GPU 利用率、SM 效率、内存带宽使用情况，对于多节点还要监控网络利用率。使用 DCGM exporter、Prometheus 等搭建仪表盘，以便你能捕捉到任何资源被闲置的情况。如果 GPU 处于 50% 利用率，就深入探究原因。很可能是数据等待/停顿以及缓慢的同步通信。如果网络仅有 10% 利用率但 GPU 却在等待数据，也许问题出在别处，比如某个锁。这些指标有助于精确定位应聚焦哪个子系统。

*Iterate and tune hyperparameters for throughput* Some model hyperparameters, such as batch size, sequence length, and number of MoE active experts, can be tuned for throughput without degrading final accuracy. For example, larger batch sizes give better throughput but might require tuning the learning rate schedule to maintain accuracy. Don’t be afraid to adjust these to find a sweet spot of speed and accuracy. This is part of performance engineering too—sometimes the model or training procedure can be adjusted for efficiency, like using activation checkpointing or more steps of compute for the same effective batch. You might tweak the training learning rate schedule to compensate for this scenario.

_为吞吐量迭代并调优超参数_ 一些模型超参数，例如批大小（batch size）、序列长度以及 MoE 激活专家数，可以在不损害最终精度的前提下为吞吐量而调优。例如，更大的批大小能带来更好的吞吐量，但可能需要调整学习率调度以维持精度。不要害怕调整这些参数以找到速度与精度的最佳平衡点。这也是性能工程的一部分——有时可以为效率而调整模型或训练流程，比如使用激活检查点，或为相同的有效批量投入更多计算步。你或许会微调训练学习率调度来补偿这种情形。

*Document and reuse* Keep notes of what optimizations you applied and their impact. Document in code or in an internal wiki-like shared knowledge-base system. This builds a knowledge base for future projects. Many tips are reusable patterns, like enabling overlapping and particular environment variables that help on a cluster. Having this history can save time when starting a new endeavor or when onboarding new team members into performance tuning efforts.

_记录并复用_ 记下你实施了哪些优化及其影响。在代码中或在类似内部 wiki 的共享知识库系统中记录。这为未来项目积累了一个知识库。许多技巧都是可复用的模式，比如启用重叠（overlap）以及一些在集群上有帮助的特定环境变量。拥有这段历史，能在开启新工作或让新团队成员上手性能调优时节省时间。

*Balance optimizations with complexity* Aim for the simplest solution that achieves needed performance. For example, if native PyTorch with torch.compile meets your speed target, you might not need to write custom CUDA kernels. This will help avoid extra maintenance. Over-optimizing with highly custom code can make the system brittle. There is elegance in a solution that is both fast and maintainable. Thus, apply the least-intrusive optimization that yields the required gain, and escalate to more involved ones only as needed.

_在优化与复杂度之间取得平衡_ 力求用能达到所需性能的最简单方案。例如，如果原生 PyTorch 配合 torch.compile 就能满足你的速度目标，你或许无需编写自定义 CUDA 核函数。这有助于避免额外的维护。用高度定制的代码过度优化，会使系统变得脆弱。既快又易维护的方案自有其优雅。因此，采用能带来所需收益的、侵入性最小的优化，仅在必要时才升级到更复杂的方案。

*Optimize AI-driven performance* Leverage machine learning models to analyze historical telemetry data and predict system bottlenecks, enabling automated adjustments of parameters in real time to optimize resource allocation and throughput.

_优化 AI 驱动的性能_ 利用机器学习模型来分析历史遥测数据并预测系统瓶颈（bottleneck），从而实现实时的参数自动调整，以优化资源分配与吞吐量。

## Reproducibility and Documentation Best Practices

## 可复现性与文档最佳实践

Performance wins don’t stick unless they’re reproducible, versioned, and continuously checked, or they’ll regress quietly over time. Treat docs, CI benchmarks, and shared knowledge as the glue that preserves speedups and accelerates onboarding and audits.

性能成果若不能复现、版本化并持续检查，就难以稳固，否则它们会随时间悄然退化。把文档、CI 基准测试与共享知识当作黏合剂，用以保全加速成果、加快上手与审计。

Lock down versions, configs, and benchmarks in source control so experiments are repeatable and regressions traceable. Bring performance checks into CI/CD, instrument end-to-end monitoring and alerts, and pair optimization with security and thorough documentation to create a durable, auditable practice. The following is a list of tips to improve reproducibility and documentation:

在源码管理中锁定版本、配置与基准，使实验可重复、回归可追溯。将性能检查引入 CI/CD，对端到端监控与告警进行埋点，并将优化与安全及详尽文档配对，以打造一套持久、可审计的实践。以下是一些改善可复现性（reproducibility）与文档的建议：

*Rigorous version control* Maintain comprehensive version control for all system configurations, framework/driver versions, OS settings, optimization scripts, and benchmarks. Use Git (or a similar system) to track changes and tag releases. This way, experiments can be reproduced exactly—and performance regressions can be easily identified.

_严格的版本控制_ 对所有系统配置、框架/驱动版本、OS 设置、优化脚本与基准维护全面的版本控制。使用 Git（或类似系统）来跟踪变更并为发行版打标签。这样，实验就能被精确复现——而且性能回归也能轻松定位。

*Continuous integration for performance regression* Integrate automated performance benchmarks and real-time monitoring into your CI/CD pipelines. This ensures that each change—from code updates to configuration changes—is validated against a set of performance metrics, helping catch regressions early and maintaining consistent and measurable performance gains. Adopt industry-standard benchmarks, such as MLPerf, to establish a reliable performance baseline and track improvements over time.

_针对性能回归的持续集成_ 将自动化性能基准测试与实时监控整合进你的 CI/CD 流水线。这可确保每一次改动——从代码更新到配置变更——都对照一组性能指标进行验证，帮助尽早捕捉回归，并维持一致且可衡量的性能收益。采用业界标准基准，例如 MLPerf，来建立可靠的性能基线并跟踪长期的改进。

*End-to-end workflow optimization* Ensure that optimizations are applied holistically across the entire AI pipeline—from data ingestion and preprocessing through training and inference deployment. Coordinated, cross-system tuning can reveal synergies that isolated adjustments might miss, resulting in more significant overall performance gains.

_端到端工作流优化_ 确保优化整体性地贯穿整个 AI 流水线——从数据摄取与预处理，到训练与推理部署。协调一致的跨系统调优能揭示孤立调整可能错过的协同效应，从而带来更显著的整体性能收益。

*Automated monitoring and diagnostics* Deploy end-to-end monitoring solutions that collect real-time metrics across hardware, network, and application layers. Integrate these with dashboards, such as Prometheus/Grafana, and configure automated alerts to promptly detect anomalies, such as sudden drops in GPU utilization or spikes in network latency.

_自动化监控与诊断_ 部署端到端监控方案，跨硬件、网络与应用各层采集实时指标。将它们与仪表盘（例如 Prometheus/Grafana）集成，并配置自动化告警，以便迅速发现异常，例如 GPU 利用率骤降或网络延迟飙升。

*Fault tolerance and automated recovery* Incorporate fault tolerance into your system design by using distributed checkpointing, redundant hardware configurations, and dynamic job rescheduling. This strategy minimizes downtime and maintains performance even in the face of hardware or network failures.

_容错与自动恢复_ 通过使用分布式检查点、冗余硬件配置与动态作业重调度，将容错纳入你的系统设计。这一策略能将停机时间降至最低，即便面对硬件或网络故障也能维持性能。

*Compiler and build optimizations* Leverage aggressive compiler flags and profile-guided optimizations during the build process to extract maximum performance from your code. Regularly update and tune your build configurations, and verify the impact of each change through rigorous benchmarking to ensure optimal execution.

_编译器与构建优化_ 在构建过程中利用激进的编译器标志与基于剖析的优化（profile-guided optimizations），从你的代码中榨取最大性能。定期更新并调优你的构建配置，并通过严格的基准测试验证每次改动的影响，以确保最优的执行。

*Security, compliance, and performance* Integrate and codesign security, compliance, and performance. Regularly audit configurations, enforce access controls, and maintain industry-standard safeguards, including encryption, secure data channels, zero-trust networking, hardware security modules (HSMs), and secure enclaves. And make sure that performance tuning never compromises system security. Similarly, make sure security doesn’t incur unnecessary performance overhead.

_安全、合规与性能_ 将安全、合规与性能协同设计。定期审计配置、强制执行访问控制，并维护业界标准的防护措施，包括加密、安全数据通道、零信任网络、硬件安全模块（HSM）与安全飞地（secure enclave）。并确保性能调优绝不损害系统安全。同样，确保安全不会带来不必要的性能开销。

*Comprehensive documentation and knowledge sharing* Maintain detailed records of all optimization steps, system configurations, and performance benchmarks. Develop an internal knowledge base to facilitate team collaboration and rapid onboarding, ensuring that best practices are preserved and reused across projects.

_全面的文档与知识共享_ 对所有优化步骤、系统配置与性能基准维护详尽的记录。搭建一个内部知识库，以促进团队协作与快速上手，确保最佳实践在各项目间得以保全与复用。

*Future-proofing and scalability planning* Design modular, adaptable system architectures that can easily incorporate emerging hardware and software technologies. Continuously evaluate scalability requirements and update your optimization strategies to sustain competitive performance as your workload grows.

_面向未来与可扩展性规划_ 设计模块化、可适配的系统架构，使其能轻松纳入新兴的硬件与软件技术。持续评估可扩展性需求并更新你的优化策略，以在工作负载增长时维持有竞争力的性能。

## System Architecture and Hardware Planning

## 系统架构与硬件规划

Your hardware, interconnects, and data paths set the ceiling for performance and cost-efficiency—no software tweak can outrun a starved GPU. Plan for goodput per dollar/watt by matching accelerators, CPU/DRAM/I/O, and cooling/power to the workload to avoid bottlenecks from the start.

你的硬件、互连（interconnect）与数据路径设定了性能与性价比的上限——任何软件调整都跑不过一个被饿着的 GPU。以每美元/每瓦的有效吞吐量（goodput）为目标进行规划，使加速器、CPU/DRAM/I/O 与散热/供电与工作负载相匹配，从一开始就避免瓶颈。

Specifically, design for goodput—useful work per dollar/watt—and not just raw FLOPS. Match accelerators and interconnects to workload, right-size CPU/memory/I/O to keep GPUs fed, keep data local, and plan power/cooling so hardware sustains peak clocks. Evaluate scaling efficiency before adding more GPUs. Here are some tips for optimizing system architecture and improving hardware planning efficiency:

具体而言，应面向有效吞吐量——每美元/每瓦的有用工作——来设计，而不仅仅是原始 FLOPS。让加速器与互连匹配工作负载，合理配置 CPU/内存/I/O 以持续喂饱 GPU，保持数据本地化，并规划供电/散热，使硬件能维持峰值时钟。在增加更多 GPU 之前评估扩展效率。以下是优化系统架构与提升硬件规划效率的一些建议：

*Design for goodput and efficiency* Treat useful throughput as the goal. Every bit of performance gained translates to massive cost savings at scale. Focus on maximizing productive work per dollar/watt—and not just raw FLOPS.

_面向有效吞吐量与效率而设计_ 把有用吞吐量当作目标。在大规模下，每一点性能收益都会转化为巨额成本节省。聚焦于最大化每美元/每瓦的有效产出——而不仅仅是原始 FLOPS。

*Choose the right accelerator* Prefer modern GPUs for superior performance-per-watt and memory capacity. Newer architectures offer features like native FP8 and FP4 precision support—along with much faster interconnects. These produce big speedups over older-generation GPUs and systems.

_选对加速器_ 优先选用现代 GPU，以获得更优的每瓦性能与内存容量。较新的架构提供诸如原生 FP8 与 FP4 精度支持等特性——以及快得多的互连。相比上一代 GPU 与系统，这些能带来巨大的加速。

*Leverage high-bandwidth interconnects* Use systems with NVLink/NVSwitch, such as GB200/GB300 NVL72, instead of PCIe-only connectivity for multi-GPU workloads. NVLink 5 provides up to 1.8 TB/s bidirectional GPU-to-GPU bandwidth (over 14× PCIe Gen5), enabling near-linear scaling across GPUs. NVLink Switch domains can be scaled with second-level switches to connect up to 576 GPUs in one NVLink domain. This enables hierarchical collectives that stay on NVLink as long as possible before falling back to the inter-rack fabric.

_利用高带宽互连_ 对于多 GPU 工作负载，使用带 NVLink/NVSwitch 的系统，例如 GB200/GB300 NVL72，而非仅有 PCIe 的连接。NVLink 5 提供高达 1.8 TB/s 的双向 GPU 到 GPU 带宽（超过 PCIe Gen5 的 14×），实现跨 GPU 近乎线性的扩展。NVLink Switch 域可通过二级交换机扩展，在一个 NVLink 域内连接多达 576 个 GPU。这使得分层集合通信（collective）得以尽可能长时间地留在 NVLink 上，再回退到机架间的 fabric。

*Balance CPU/GPU and memory ratios* Provision enough CPU cores, DRAM, and storage throughput per GPU. For example, allocate ~1 fast CPU core per GPU for data loading and networking tasks. Ensure system RAM and I/O can feed GPUs at required rates on the order of hundreds of MB/s per GPU to avoid starvation.

_平衡 CPU/GPU 与内存比例_ 为每个 GPU 配备足够的 CPU 核心、DRAM 与存储吞吐量。例如，为数据加载与网络任务，每个 GPU 分配约 1 个快速 CPU 核心。确保系统 RAM 与 I/O 能以所需速率喂饱 GPU，量级约为每 GPU 数百 MB/s，以避免饥饿。

*Plan for data locality* If training across multiple nodes, minimize off-node communication. Whenever possible, keep tightly coupled workloads on the same NVLink/NVSwitch domain to exploit full bandwidth, and use the highest-speed interconnect that you have access to. Ideally, this is NVLink for intranode and intra-rack communication and InfiniBand for inter-rack communication.

_为数据本地性而规划_ 如果跨多节点训练，尽量减少节点外通信。只要可能，就把紧耦合的工作负载保持在同一个 NVLink/NVSwitch 域内以充分利用带宽，并使用你能访问到的最高速互连。理想情况下，节点内与机架内通信使用 NVLink，机架间通信使用 InfiniBand。

*Avoid bottlenecks in the chain* Identify the slowest link—be it CPU, memory, disk, or network—and scale it up. For instance, if GPU utilization is low due to I/O, invest in faster storage or caching rather than more GPUs. An end-to-end design where all components are well-matched prevents wasted GPU cycles.

_避免链条中的瓶颈_ 找出最慢的一环——无论是 CPU、内存、磁盘还是网络——并将其扩容。例如，如果 GPU 利用率因 I/O 而偏低，就投资更快的存储或缓存，而不是更多 GPU。一个所有组件都良好匹配的端到端设计能避免浪费 GPU 周期。

*Choose an appropriate cluster size* Beware of diminishing returns when adding GPUs. Past a certain cluster size, overheads can grow—ensure the speedup justifies the cost. It’s often better to optimize utilization on *N* GPUs by reaching 95% usage, for example, before scaling to 2*N* GPUs.

_选择合适的集群规模_ 警惕增加 GPU 时的边际递减。超过某个集群规模后，各种开销会增长——确保加速幅度对得起成本。例如，通常更好的做法是先在 N 个 GPU 上把利用率优化到 95%，再扩展到 2N 个 GPU。

*Design for cooling and power* Ensure the data center can handle GPU thermal and power needs. High-performance systems like GB200/GB300 have very high TDP. Provide adequate cooling (likely liquid-based) and power provisioning so the GPUs can sustain boost clocks without throttling.

_为散热与供电而设计_ 确保数据中心能应对 GPU 的散热与供电需求。像 GB200/GB300 这样的高性能系统具有非常高的 TDP。提供充足的散热（很可能是液冷）与供电，使 GPU 能维持加速时钟而不降频。

## Unified CPU-GPU “Superchip” Architecture

## 统一 CPU-GPU“超级芯片”架构

Unified memory and on-package links let you fit larger models and cut copy overhead when you place the right data in the right tier. Using Grace for preprocessing and HBM for “hot” tensors turns the superchip into a tightly coupled engine with fewer stalls.

当你把正确的数据放在正确的层级时，统一内存与封装内链路能让你装下更大的模型并削减拷贝开销。用 Grace 做预处理、用 HBM 存放“热”张量，能把超级芯片变成一个停顿更少的紧耦合引擎。

On Grace Blackwell Superchips, treat CPU and GPU as a shared-memory complex. Keep hot weights/activations in HBM and overflow or infrequent data in Grace LPDDR via NVLink-C2C. Use the on-package Grace CPU for preprocessing/orchestration and prefetch or pipeline-managed memory to hide latency for ultralarge models. Take advantage of the superchip architecture as follows:

在 Grace Blackwell 超级芯片上，将 CPU 与 GPU 视为一个共享内存复合体。把热权重/激活放在 HBM 中，把溢出或不常用的数据经 NVLink-C2C 放在 Grace LPDDR 中。使用封装内的 Grace CPU 做预处理/编排，并用预取或流水线管理的内存来为超大模型隐藏延迟。可按如下方式利用超级芯片架构：

*Utilize unified CPU-GPU memory* Exploit the Grace Blackwell (GB200/GB300) Superchip’s unified memory space. Two Blackwell GPUs and a 72-core Grace CPU share a coherent memory pool with NVLink-C2C (900 GB/s). Use the CPU’s large memory (e.g., 480 GB LPDDR5X) as an extension for oversize models while keeping “hot” data in the GPUs’ HBM for speed.

_利用统一 CPU-GPU 内存_ 善用 Grace Blackwell（GB200/GB300）超级芯片的统一内存空间。两个 Blackwell GPU 与一个 72 核 Grace CPU 通过 NVLink-C2C（900 GB/s）共享一个一致性内存池。使用 CPU 的大容量内存（例如 480 GB LPDDR5X）作为超大模型的扩展，同时把“热”数据保留在 GPU 的 HBM 中以获得速度。

*Place data for locality* Even with unified memory, prioritize data placement. Put model weights, activations, and other frequently accessed data on GPU HBM3e (which has much higher local bandwidth), and let infrequently used or overflow data reside in CPU RAM. This ensures the 900 GB/s NVLink-C2C link isn’t a bottleneck for critical data.

_为本地性而放置数据_ 即便有了统一内存，也要优先考虑数据放置。把模型权重、激活与其他高频访问数据放在 GPU HBM3e 上（其本地带宽高得多），并让不常用或溢出的数据驻留在 CPU RAM 中。这可确保 900 GB/s 的 NVLink-C2C 链路不会成为关键数据的瓶颈。

*Take advantage of CPU-GPU direct memory access when available* Use the GPU’s ability to directly access CPU memory on combined CPU-GPU superchips like the GB200 and GB300. GPUs can read and write Grace LPDDR memory coherently over NVLink-C2C without staging over host PCIe. Bandwidth and latency are still lower than HBM, so prefetch managed pointers, stage data, and pipeline transfers to hide latency. As such, it’s recommended to keep hot activations and KV cache in HBM and use CPU memory as a lower-tier cache with explicit prefetch.

_在可用时利用 CPU-GPU 直接内存访问_ 在像 GB200 与 GB300 这样的 CPU-GPU 合体超级芯片上，利用 GPU 直接访问 CPU 内存的能力。GPU 可以经 NVLink-C2C 一致性地读写 Grace LPDDR 内存，而无需通过主机 PCIe 中转。其带宽与延迟仍不及 HBM，因此要预取托管指针、暂存数据并流水线化传输以隐藏延迟。因此，建议把热激活与 KV 缓存（KV cache）保留在 HBM 中，而把 CPU 内存作为带显式预取的下层缓存来使用。

*Use the Grace CPU effectively* The on-package Grace CPU provides 72 high-performance cores—utilize them! Offload data preprocessing, augmentation, and other CPU-friendly tasks to these cores. They can feed the GPUs quickly using NVLink-C2C, essentially acting as an extremely fast I/O and compute companion for the GPU.

_高效使用 Grace CPU_ 封装内的 Grace CPU 提供 72 个高性能核心——用起来！把数据预处理、增强及其他适合 CPU 的任务卸载到这些核心上。它们能通过 NVLink-C2C 快速喂饱 GPU，本质上充当了 GPU 极快的 I/O 与计算伙伴。

*Plan for ultralarge models* For trillion-parameter model training that exceeds GPU memory, GB200/GB300 systems allow you to train using CPU memory as part of the model’s memory pool. Prefer framework caching allocators and use cudaMallocAsync in custom code to minimize fragmentation and enable graph capture. Use CUDA Unified Memory or managed memory APIs to handle overflow gracefully, and consider explicit prefetching (e.g., cudaMemPrefetchAsync) of upcoming layers from CPU → GPU memory to hide latency.

_为超大模型而规划_ 对于超出 GPU 内存的万亿参数模型训练，GB200/GB300 系统允许你把 CPU 内存作为模型内存池的一部分来进行训练。优先使用框架的缓存分配器，并在自定义代码中使用 cudaMallocAsync 来尽量减少碎片并支持图捕获（graph capture）。使用 CUDA Unified Memory 或托管内存 API 来优雅地处理溢出，并考虑对即将用到的层做从 CPU → GPU 内存的显式预取（例如 cudaMemPrefetchAsync）以隐藏延迟。

*Consider superchip-optimized algorithms* SuperOffload is an example of a superchip-optimized set of algorithms focused on improving efficiency of offload and tensor cast/copy strategies. Innovations include speculation-then-validation (STV), heterogeneous optimizer computation, and an ARM-based CPU optimizer. Designed specifically for NVIDIA superchips (e.g., Grace Hopper, Grace Blackwell, Vera Rubin), SuperOffload increases token-processing throughput and chip utilization relative to traditional offload strategies.

_考虑面向超级芯片优化的算法_ SuperOffload 就是一套面向超级芯片优化的算法示例，专注于改进卸载（offload）以及张量转换/拷贝策略的效率。其创新包括推测-然后-验证（speculation-then-validation，STV）、异构优化器计算，以及一个基于 ARM 的 CPU 优化器。SuperOffload 专为 NVIDIA 超级芯片（例如 Grace Hopper、Grace Blackwell、Vera Rubin）设计，相对于传统卸载策略，它提升了 token 处理吞吐量与芯片利用率。

## Multi-GPU Scaling and Interconnect Optimizations

## 多 GPU 扩展与互连优化

Scaling pays only when communication is fast and topology-aware—otherwise added GPUs just wait on one another. Lean on NVLink/NVSwitch bandwidth, modern collectives, and fabric-aware placement to approach linear speedups.

只有当通信足够快且具备拓扑感知时，扩展才有回报——否则增加的 GPU 只会彼此干等。依靠 NVLink/NVSwitch 带宽、现代集合通信算法与 fabric 感知的放置，来逼近线性加速。

Specifically, exploit NVLink/NVSwitch domains (e.g., NVL72) for near-linear scaling, and choose parallelism strategies that fit the fabric. Use topology-aware placement, updated NCCL collectives (e.g., PAT), and telemetry to verify you’re using the ~1.8 TB/s bidirectional throughput per-GPU bandwidth effectively. Plan hierarchical communications as you expand. The following are tips on utilizing multi-GPU scaling through interconnect and topology optimizations:

具体而言，利用 NVLink/NVSwitch 域（例如 NVL72）实现近乎线性的扩展，并选择契合 fabric 的并行策略。使用拓扑（topology）感知的放置、更新的 NCCL 集合通信（例如 PAT）与遥测，来验证你是否有效地用上了每 GPU 约 1.8 TB/s 的双向吞吐带宽。随着规模扩大，规划分层通信。以下是关于通过互连与拓扑优化来利用多 GPU 扩展的一些建议：

*Design for high-speed all-to-all topology* On NVL72 NVSwitch clusters with 72 fully interconnected GPUs, for example, any GPU can communicate with any other at full NVLink 5 speed. At the fabric level, the NVLink Switch domain is nonblocking. Application-level throughput can vary with concurrent traffic and path scheduling, so verify behavior using DCGM NVLink counters and Nsight Systems traces before assuming per-pair saturation. Take advantage of this topology by using parallelization strategies, such as data parallel, tensor parallel, and pipeline parallelism, that would be bottlenecked on lesser interconnects.

_为高速全互连拓扑而设计_ 例如在拥有 72 个全互连 GPU 的 NVL72 NVSwitch 集群上，任意 GPU 都能以完整的 NVLink 5 速度与任意其他 GPU 通信。在 fabric 层面，NVLink Switch 域是无阻塞的。应用层吞吐量会随并发流量与路径调度而变化，因此在假定逐对饱和之前，应使用 DCGM NVLink 计数器与 Nsight Systems 追踪来验证行为。通过使用数据并行、张量并行与流水线并行等在较弱互连上会遭遇瓶颈的并行化策略，来充分利用这一拓扑。

*Utilize topology-aware scheduling* Always colocate multi-GPU jobs within an NVLink Switch domain if possible. Keeping all GPUs of a job on the NVL72 fabric means near-linear scaling for communication-heavy workloads. Mixing GPUs across NVLink domains or standard networks will introduce bottlenecks and should be avoided for tightly coupled tasks.

_利用拓扑感知调度_ 只要可能，就始终把多 GPU 作业共置于同一个 NVLink Switch 域内。让一个作业的所有 GPU 都处于 NVL72 fabric 上，意味着通信密集型工作负载能实现近乎线性的扩展。跨 NVLink 域或跨标准网络混用 GPU 会引入瓶颈，对于紧耦合任务应予以避免。

*Leverage unprecedented bandwidth* Recognize that NVLink 5 has 900 GB/s per GPU in each direction, which doubles the per-GPU bandwidth versus the previous generation. An NVL72 rack provides ~130 TB/s total intra-rack bandwidth in aggregate. This drastically reduces communication wait times, as even tens of gigabytes of gradient data can be all-reduced in a few milliseconds at 1.8 TB/s. Design training algorithms, such as gradient synchronization and parameter sharding, to fully exploit this relatively free communication budget.

_利用前所未有的带宽_ 要认识到 NVLink 5 在每个方向上都为每 GPU 提供 900 GB/s，这使每 GPU 带宽相比上一代翻倍。一个 NVL72 机架总计提供约 130 TB/s 的机架内聚合带宽。这大幅缩短了通信等待时间，因为即便是数十 GB 的梯度数据，在 1.8 TB/s 下也能在几毫秒内完成 all-reduce。设计诸如梯度同步与参数分片（sharding）之类的训练算法，以充分利用这份相对免费的通信预算。

*Embrace modern collective algorithms* Use the latest NVIDIA NCCL library optimized for NVSwitch. Specifically, enable the parallel aggregated tree (PAT) algorithm, which was introduced for NVLink Switch topologies. This further reduces synchronization time by taking advantage of the NVL72 topology to perform reductions more efficiently than other tree/ring algorithms.

_拥抱现代集合通信算法_ 使用为 NVSwitch 优化的最新 NVIDIA NCCL 库。具体而言，启用为 NVLink Switch 拓扑引入的并行聚合树（parallel aggregated tree，PAT）算法。它通过利用 NVL72 拓扑以比其他树/环算法更高效地执行归约，从而进一步缩短同步时间。

*Consider fine-grained parallelism* With full-bandwidth all-to-all connectivity, consider fine-grained model parallelism that wasn’t feasible before. For example, layer-wise parallelism or tensor parallelism across many GPUs can be efficient when each GPU has 1.8 TB/s bidirectional throughput to every other. Previously, one might avoid excessive cross-GPU communication, but NVL72 allows aggressive partitioning of work without hitting network limits.

_考虑细粒度并行_ 有了全带宽的全互连连接，可以考虑以往不可行的细粒度模型并行。例如，当每个 GPU 到其他每个 GPU 都有 1.8 TB/s 的双向吞吐量时，跨众多 GPU 的逐层并行或张量并行都能很高效。以往人们可能会避免过多的跨 GPU 通信，但 NVL72 允许激进地切分工作而不触及网络上限。

*Monitor for saturation* Although NVL72 is extremely fast, keep an eye on link utilization in profiling. If your application somehow saturates the NVSwitch using extreme all-to-all operations, for example, you might need to throttle communication by aggregating gradients, etc. Use NVIDIA’s tools or NVSwitch telemetry to verify that communications are within the NVLink capacity, and adjust patterns if needed. For instance, you can stagger all-to-all exchanges to avoid network contention. DCGM exposes NVLink counters that can help verify link balance and detect hotspots during collectives.

_监控饱和情况_ 尽管 NVL72 极快，仍要在剖析中留意链路利用率。例如，如果你的应用以极端的全互连操作把 NVSwitch 用饱和了，你可能需要通过聚合梯度等方式来节流通信。使用 NVIDIA 的工具或 NVSwitch 遥测来验证通信处于 NVLink 容量之内，并在需要时调整模式。例如，你可以错开全互连交换以避免网络争用。DCGM 暴露的 NVLink 计数器有助于验证链路平衡，并在集合通信期间检测热点。

*Plan for future expansion* Be aware that NVLink Switch can scale beyond a single rack—up to 576 GPUs in one connected domain using second-level switches. If you operate at that ultrascale, plan hierarchical communication using local NVL72 inter-rack collectives first, then use inter-rack interconnects only when necessary. This helps to maximize intra-rack NVLink usage first. This ensures you’re using the fastest links before resorting to inter-rack InfiniBand hops.

_为未来扩展而规划_ 要知道 NVLink Switch 可以扩展到单机架之外——使用二级交换机在一个互连域内多达 576 个 GPU。如果你运行在这种超大规模上，就规划分层通信，先使用本地 NVL72 的机架间集合通信，仅在必要时再使用机架间互连。这有助于优先最大化机架内的 NVLink 使用。这确保你在诉诸机架间 InfiniBand 跳转之前，先用上最快的链路。

*Identify opportunities for federated and distributed optimizations* For deployments that span heterogeneous environments, such as multicloud or edge-to-cloud setups, adopt adaptive communication protocols and dynamic load balancing strategies. This minimizes latency and maximizes throughput across distributed systems, which ensures robust performance even when resources vary in capability and capacity.

_识别联邦与分布式优化的机会_ 对于横跨异构环境的部署，例如多云或边缘到云的架构，采用自适应通信协议与动态负载均衡策略。这能在分布式系统间将延迟降至最低并最大化吞吐量，从而即便在资源能力与容量各异时也能确保稳健的性能。

## Operating System and Driver Optimizations

## 操作系统与驱动优化

OS jitter, NUMA misses, and driver mismatches quietly drain throughput and create variability you can’t tune around. Hardening the stack (huge pages, affinities, consistent CUDA/driver, persistence) creates a stable, high-performance baseline.

OS 抖动、NUMA 未命中与驱动不匹配会悄然消耗吞吐量，并制造出你无法靠调优规避的波动。加固技术栈（大页、亲和性、一致的 CUDA/驱动、持久化）能打造一个稳定的高性能基线。

Run a lean, HPC-tuned Linux. Set NUMA/IRQ affinities and enable THP and high memlock. Keep NVIDIA drivers/CUDA consistent across nodes. Isolate system jitter, tune CPU libraries/storage, set container limits correctly, and keep BIOS/firmware/NVSwitch fabric up to date for predictable throughput. Here are some host, OS, and container optimizations that you should explore in your environment:

运行一套精简的、面向 HPC 调优的 Linux。设置 NUMA/IRQ 亲和性，并启用 THP 与高 memlock。在各节点间保持 NVIDIA 驱动/CUDA 一致。隔离系统抖动、调优 CPU 库/存储、正确设置容器（container）限制，并保持 BIOS/固件/NVSwitch fabric 为最新，以获得可预测的吞吐量。以下是一些你应在自己环境中探索的主机、OS 与容器优化：

*Use a Linux kernel tuned for HPC* Ensure your GPU servers run a recent, stable Linux kernel configured for high-performance computing. Disable unnecessary background services that consume CPU or I/O. Use the “performance” CPU governor—versus “on-demand” or “power-save”—to keep CPU cores at a high clock for feeding GPUs.

_使用为 HPC 调优的 Linux 内核_ 确保你的 GPU 服务器运行一个较新、稳定且为高性能计算配置的 Linux 内核。禁用消耗 CPU 或 I/O 的不必要后台服务。使用“performance”CPU 调速器——而非“on-demand”或“power-save”——以让 CPU 核心保持高时钟来喂饱 GPU。

*Disable swap for performance-critical workloads* Disable swap on training servers to avoid page thrashing, or, if swap must remain enabled, lock critical buffers using mlock or cudaHostAlloc to ensure they stay in RAM.

_为性能关键型工作负载禁用交换_ 在训练服务器上禁用交换（swap）以避免页面抖动；如果交换必须保持启用，则使用 mlock 或 cudaHostAlloc 锁定关键缓冲区，确保它们留在 RAM 中。

*Avoid memory fragmentation with aggressive preallocation* Preallocate large, contiguous blocks of memory for frequently used tensors to reduce runtime allocation overhead and fragmentation. This proactive strategy ensures more stable and efficient memory management during long training runs.

_用激进的预分配避免内存碎片_ 为高频使用的张量预分配大块连续内存，以减少运行时分配开销与碎片。这一前瞻性策略能在长时间训练运行中确保更稳定、更高效的内存管理。

*Optimize environment variables for CPU libraries* Fine-tune parameters, such as OMP_NUM_THREADS and MKL_NUM_THREADS, to better match your hardware configuration. Adjusting these variables can reduce thread contention and improve the parallel efficiency of CPU-bound operations.

_为 CPU 库优化环境变量_ 微调诸如 OMP_NUM_THREADS 与 MKL_NUM_THREADS 之类的参数，以更好地匹配你的硬件配置。调整这些变量能减少线程争用，并改善 CPU 密集型操作的并行效率。

*Design for NUMA awareness* For multi-NUMA servers, pin GPU processes/threads to the CPU of the local NUMA node. Use tools like numactl or taskset to bind each training process to the CPU nearest its assigned GPU. Similarly, bind memory allocations to the local NUMA node (numactl --membind) so host memory for GPU DMA comes from the closest RAM. This avoids costly cross-NUMA memory traffic that can halve effective PCIe/NVLink bandwidth.

_为 NUMA 感知而设计_ 对于多 NUMA 服务器，把 GPU 进程/线程钉到本地 NUMA 节点的 CPU 上。使用 numactl 或 taskset 之类的工具，把每个训练进程绑定到离其所分配 GPU 最近的 CPU。同样，把内存分配绑定到本地 NUMA 节点（numactl --membind），使供 GPU DMA 使用的主机内存来自最近的 RAM。这避免了代价高昂的跨 NUMA 内存流量，那可能会把有效的 PCIe/NVLink 带宽腰斩。

*Utilize IRQ affinity for network and GPU tasks* Explicitly bind NIC interrupts to CPU cores on the same NUMA node as the NIC, and similarly pin GPU driver threads to dedicated cores—including those from long-running services like the nvidia-persistence service daemon. This strategy minimizes cross-NUMA traffic and stabilizes performance under heavy loads.

_为网络与 GPU 任务利用 IRQ 亲和性_ 显式地把 NIC 中断绑定到与 NIC 同一 NUMA 节点的 CPU 核心上，并同样把 GPU 驱动线程钉到专用核心上——包括来自像 nvidia-persistence 服务守护进程这类长期运行服务的线程。这一策略能最小化跨 NUMA 流量，并在重负载下稳定性能。

*Enable transparent hugepages* Turn on transparent hugepages (THP) in always or madvise mode so that large memory allocations use 2 MB pages. This reduces TLB thrashing and kernel overhead when allocating tens or hundreds of GBs of host memory for frameworks. Verify THP is active by checking for /sys/kernel/mm/transparent_hugepage/enabled. With THP enabled, your processes are using hugepages for big allocations. Prefer THP in madvise mode if your workload is latency-critical and you observe jitter.

_启用透明大页_ 以 always 或 madvise 模式开启透明大页（THP），使大内存分配使用 2 MB 页。这在为框架分配数十或数百 GB 主机内存时，能减少 TLB 抖动与内核开销。通过检查 /sys/kernel/mm/transparent_hugepage/enabled 来验证 THP 处于激活状态。启用 THP 后，你的进程会为大分配使用大页。如果你的工作负载对延迟敏感且观察到抖动，则优先使用 madvise 模式的 THP。

*Increase max locked memory* Configure the OS to allow large pinned (aka page-locked) allocations. GPU apps often pin memory for faster transfers—set ulimit -l unlimited or a high value so your data loaders can allocate pinned buffers without hitting OS limits. This prevents failures or fallbacks to pageable memory, which would slow down GPU DMA.

_提高最大锁定内存_ 配置 OS 以允许大量固定（即页锁定）分配。GPU 应用常常固定内存以加快传输——设置 ulimit -l unlimited 或一个较高的值，使你的数据加载器能分配固定缓冲区而不触及 OS 限制。这可防止失败或回退到可分页内存，那会拖慢 GPU DMA。

*Use the latest NVIDIA driver and CUDA stack* Keep NVIDIA drivers and CUDA runtime up-to-date (within a tested stable version) on all nodes. New drivers can bring performance improvements and are required for new GPUs’ compute capabilities. Make sure all nodes have the same driver/CUDA versions to avoid any mismatches in multinode jobs. Enable persistence mode on GPUs at boot (nvidia-smi -pm 1) so the driver stays loaded and GPUs don’t incur re-init delays. Update the NVIDIA driver and toolkit on all nodes to inherit bug fixes and performance improvements.

_使用最新的 NVIDIA 驱动与 CUDA 栈_ 在所有节点上保持 NVIDIA 驱动与 CUDA 运行时为最新（在经过测试的稳定版本范围内）。新驱动能带来性能改进，且是新 GPU 计算能力所必需的。确保所有节点具有相同的驱动/CUDA 版本，以避免多节点作业中的任何不匹配。在启动时为 GPU 启用持久化模式（nvidia-smi -pm 1），使驱动保持加载状态、GPU 不再承受重新初始化的延迟。在所有节点上更新 NVIDIA 驱动与工具包，以继承缺陷修复与性能改进。

*Enable GPU persistence when using a MIG configuration* With persistence mode enabled, the GPU remains “warm” and ready to use, reducing startup latency for jobs. This is especially crucial if using a Multi-Instance GPU (MIG) partitioning—without persistence, MIG configurations would reset on every job, but keeping the driver active preserves the slices. Always configure persistence mode when using MIG.

_在使用 MIG 配置时启用 GPU 持久化_ 启用持久化模式后，GPU 保持“温热”并随时可用，从而降低作业的启动延迟。如果使用多实例 GPU（Multi-Instance GPU，MIG）分区，这一点尤为关键——没有持久化，MIG 配置会在每个作业时重置，而保持驱动活跃则能保全这些切片。使用 MIG 时始终配置持久化模式。

*Isolate system tasks* Dedicate a core—or small subset of cores—on each server for OS housekeeping, such as interrupt handling and background daemons. This way, your main CPU threads feeding the GPU are not interrupted. This can be done using CPU isolation or cgroup pinning. Eliminating OS jitter ensures consistent throughput.

_隔离系统任务_ 在每台服务器上划出一个核心——或一小组核心——用于 OS 内务处理，例如中断处理与后台守护进程。这样，你喂饱 GPU 的主 CPU 线程就不会被打断。这可以通过 CPU 隔离或 cgroup 钉绑来实现。消除 OS 抖动能确保一致的吞吐量。

*Optimize system I/O settings* If your workload does a lot of logging or checkpointing, mount filesystems with options that favor throughput. Consider using noatime for data disks and increase filesystem read-ahead for streaming reads. Ensure the disk scheduler is set appropriately to use mq-deadline or noop for NVMe SSDs to reduce latency variability.

_优化系统 I/O 设置_ 如果你的工作负载有大量日志或检查点写入，就用偏向吞吐量的选项挂载文件系统。考虑为数据盘使用 noatime，并为流式读取增大文件系统的预读。确保磁盘调度器设置得当，对 NVMe SSD 使用 mq-deadline 或 noop，以减少延迟波动。

*Perform regular maintenance* Keep BIOS/firmware updated for performance fixes. Some BIOS updates improve PCIe bandwidth or fix input–output memory management unit (IOMMU) issues for GPUs. Also, periodically check for firmware updates for NICs and NVSwitch/Fabric if applicable, as provided by NVIDIA, such as Fabric Manager upgrades, etc. Minor firmware tweaks can sometimes resolve obscure bottlenecks or reliability issues.

_执行定期维护_ 保持 BIOS/固件为最新以获得性能修复。有些 BIOS 更新能改善 PCIe 带宽，或修复 GPU 的输入输出内存管理单元（IOMMU）问题。此外，如适用，也要定期检查 NIC 与 NVSwitch/Fabric 的固件更新，如 NVIDIA 提供的更新，例如 Fabric Manager 升级等。细微的固件调整有时能解决晦涩的瓶颈或可靠性问题。

*Tune Docker and Kubernetes configurations for maximum performance* When running in containers, add options, such as --ipc=host for shared memory, and set --ulimit memlock=-1 to prevent memory locking issues. This guarantees that your containerized processes access memory without OS-imposed restrictions.

_调优 Docker 与 Kubernetes 配置以获得最大性能_ 在容器中运行时，添加诸如为共享内存的 --ipc=host 之类的选项，并设置 --ulimit memlock=-1 以防止内存锁定问题。这可保证你的容器化进程在不受 OS 施加限制的情况下访问内存。

## GPU Resource Management and Scheduling

## GPU 资源管理与调度

Smarter placement and partitioning raise utilization without buying new hardware—and protect predictability for mixed workloads. Respect topology, use MPS/MIG where appropriate, and control clocks/power to minimize contention and tail latency.

更聪明的放置与分区能在不购置新硬件的情况下提升利用率——并为混合工作负载保护可预测性。尊重拓扑，在适当之处使用 MPS/MIG，并控制时钟/功耗以最小化争用与尾延迟。

Schedule with GPU/NUMA/NVLink topology in mind, and use MPS or MIG to raise utilization for smaller jobs while retaining ECC and persistence for reliability. Lock clocks or power limit for stability when needed, avoid CPU oversubscription, and pack jobs intelligently to maximize ROI without contention. Here are some GPU resource management and scheduling tips:

在调度时把 GPU/NUMA/NVLink 拓扑纳入考量，并使用 MPS 或 MIG 为较小的作业提升利用率，同时保留 ECC 与持久化以保障可靠性。在需要时锁定时钟或设置功耗上限以求稳定，避免 CPU 超额订阅，并智能地打包作业，从而在无争用的情况下最大化 ROI。以下是一些 GPU 资源管理与调度建议：

*Topology-aware job scheduling* Ensure that orchestrators like Kubernetes and SLURM are scheduling containers on nodes that respect NUMA and NVLink boundaries to minimize cross-NUMA and cross-NVLink-domain memory accesses. This alignment reduces latency and improves overall throughput.

_拓扑感知的作业调度_ 确保像 Kubernetes 与 SLURM 这样的编排器把容器调度到尊重 NUMA 与 NVLink 边界的节点上，以最小化跨 NUMA 与跨 NVLink 域的内存访问。这种对齐能降低延迟并改善整体吞吐量。

*Multi-Process Service (MPS)* Enable NVIDIA MPS when running multiple processes on a single GPU to improve utilization. MPS allows kernels from different processes to execute concurrently on the GPU instead of time-slicing. This is useful if individual jobs don’t fully saturate the GPU—for example, running 4 training tasks on one GPU with MPS can overlap their work and boost overall throughput.

_多进程服务（Multi-Process Service，MPS）_ 在单个 GPU 上运行多个进程时启用 NVIDIA MPS 以提升利用率。MPS 允许来自不同进程的核函数在 GPU 上并发执行，而非时间分片。如果单个作业无法完全用满 GPU，这就很有用——例如，用 MPS 在一个 GPU 上运行 4 个训练任务，可以让它们的工作重叠并提升整体吞吐量。

*Multi-Instance GPU (MIG)* Use MIG to partition high-end GPUs into smaller instances for multiple jobs. If you have many light workloads like inferencing small models or running many experiments, you can slice a GPU to ensure guaranteed resources for each job. For instance, modern GPUs can be split into multiple MIG slices (up to 7). Do not use MIG for tightly coupled parallel jobs, as those benefit from full GPU access. Deploy MIG for isolation and maximizing GPU ROI when jobs are smaller than a full GPU.

_多实例 GPU_ 使用 MIG 把高端 GPU 划分为更小的实例以供多个作业使用。如果你有许多轻量工作负载，比如对小模型做推理或运行大量实验，就可以切分一个 GPU，为每个作业确保有保障的资源。例如，现代 GPU 可以被切分为多个 MIG 切片（最多 7 个）。不要对紧耦合的并行作业使用 MIG，因为那些作业得益于完整的 GPU 访问。当作业小于一整个 GPU 时，部署 MIG 以实现隔离并最大化 GPU 的 ROI。

*Persistence for MIG* Keep persistence mode on to maintain MIG partitions between jobs. This avoids repartitioning overhead and ensures subsequent jobs see the expected GPU slices without delay. Configure MIG at cluster boot and leave it enabled so that scheduling is predictable, as changing MIG config on the fly requires resetting the GPU, which can disrupt running jobs. Plan for maintenance windows as MIG device partitions are not persisted by the GPU across reboot. Use NVIDIA’s MIG Manager to automatically recreate the desired layout on boot.

_为 MIG 保持持久化_ 保持持久化模式开启，以在作业之间维持 MIG 分区。这可避免重新分区的开销，并确保后续作业无延迟地看到预期的 GPU 切片。在集群启动时配置 MIG 并保持其启用，使调度可预测，因为动态更改 MIG 配置需要重置 GPU，这可能扰乱正在运行的作业。要为维护窗口做好规划，因为 MIG 设备分区不会被 GPU 跨重启持久保留。使用 NVIDIA 的 MIG Manager 在启动时自动重建所需的布局。

*GPU clock and power settings* Consider locking GPU clocks to a fixed high frequency with nvidia-smi -lgc/-lmc if you need run-to-run consistency. By default, GPUs use auto boost, which is usually optimal, but fixed clocks can avoid any transient downclocking. In power-constrained scenarios, you might slightly underclock or set a power limit to keep GPUs in a stable thermal/power envelope—this can yield consistent performance if occasional throttling was an issue.

_GPU 时钟与功耗设置_ 如果你需要逐次运行的一致性，可考虑用 nvidia-smi -lgc/-lmc 把 GPU 时钟锁定到一个固定的高频率。默认情况下 GPU 使用自动加速，这通常是最优的，但固定时钟能避免任何瞬态降频。在功耗受限的场景中，你或许会略微降频或设置功耗上限，使 GPU 处于稳定的散热/功耗包络内——如果偶发降频是个问题，这能带来一致的性能。

*ECC memory* Keep ECC enabled on data center GPUs for reliability unless you have a specific reason to disable it. The performance cost is minimal—on the order of a few percent loss in bandwidth and memory—but ECC catches memory errors that could otherwise corrupt a long training job. Most server GPUs ship with ECC on by default. Leave it on to safeguard multiweek training.

_ECC 内存_ 在数据中心 GPU 上保持 ECC 启用以求可靠，除非你有特定理由要禁用它。其性能代价很小——量级约为带宽与内存上百分之几的损失——但 ECC 能捕捉那些原本可能损坏长时间训练作业的内存错误。大多数服务器 GPU 出厂即默认开启 ECC。保持开启以护航长达数周的训练。

*Job scheduler awareness* Integrate GPU topology into your job scheduler, such as SLURM and Kubernetes. Configure the scheduler to allocate jobs on the same node or same NVSwitch group when low-latency coupling is needed. Use Kubernetes device plugins or SLURM Gres to schedule MIG slices for smaller jobs. A GPU-aware scheduler prevents scenarios like a single job spanning distant GPUs and suffering bandwidth issues.

_作业调度器的感知能力_ 把 GPU 拓扑整合进你的作业调度器，例如 SLURM 与 Kubernetes。配置调度器，在需要低延迟耦合时把作业分配到同一节点或同一 NVSwitch 组。使用 Kubernetes 设备插件或 SLURM Gres 为较小的作业调度 MIG 切片。一个 GPU 感知的调度器能避免诸如单个作业横跨相距较远的 GPU 而遭遇带宽问题之类的情况。

*CPU oversubscription* When scheduling jobs, account for the CPU needs of each GPU task, such as data loading threads, etc. Don’t pack more GPU jobs on a node than the CPUs can handle. It’s better to leave a GPU idle than to overload the CPU such that all GPUs become underfed. Monitor CPU utilization per GPU job to inform scheduling decisions.

_CPU 超额订阅（oversubscription）_ 在调度作业时，要考虑每个 GPU 任务的 CPU 需求，例如数据加载线程等。不要在一个节点上打包过多的 GPU 作业，以致超出 CPU 的处理能力。宁可让一块 GPU 闲置，也不要让 CPU 过载而导致所有 GPU 都供给不足（underfed）。监控每个 GPU 作业的 CPU 利用率，以便为调度决策提供依据。

*Use NVIDIA Fabric Manager for NVSwitch* On systems with NVSwitch, the GB200/GB300 NVL72 racks ensure NVIDIA Fabric Manager is running. It manages the NVSwitch topology and routing. Without it, multi-GPU communication might not be fully optimized or could even fail for large jobs. The Fabric Manager service typically runs by default on NVSwitch-equipped servers, but you should double-check that it’s enabled and running—especially after driver updates.

_为 NVSwitch 使用 NVIDIA Fabric Manager_ 在配备 NVSwitch 的系统上，GB200/GB300 NVL72 机架要确保 NVIDIA Fabric Manager 正在运行。它负责管理 NVSwitch 的拓扑与路由。若缺少它，多 GPU 通信可能无法充分优化，甚至在大型作业中直接失败。Fabric Manager 服务在配备 NVSwitch 的服务器上通常默认运行，但你应当再三确认它已启用并正在运行——尤其是在驱动更新之后。

*Job packing for utilization* Maximize utilization by intelligently packing jobs. For example, on a 4-GPU node, if you have two 2-GPU jobs that don’t use much CPU, running them together on the same node can save resources and even use the faster NVLink for communication if running together inside the same compute node or NVLink-enabled rack. Conversely, avoid colocating jobs that collectively exceed the memory or I/O capacity of the node. The goal is high hardware utilization without contention.

_打包作业以提升利用率_ 通过智能地打包作业来最大化利用率。例如，在一个 4-GPU 节点上，如果你有两个不怎么占用 CPU 的 2-GPU 作业，将它们放在同一节点上一起运行可以节省资源；若它们同处一个计算节点或支持 NVLink 的机架内一起运行，还能利用更快的 NVLink 进行通信。反过来，要避免把合计超出节点内存或 I/O 容量的作业放在一起。目标是在没有争用（contention）的前提下实现高硬件利用率。

## I/O Optimization

## I/O 优化

If data can’t keep up, GPUs idle—often the largest, cheapest speedups come from fixing input, not math. Parallelism, pinned memory, async transfers, and fast storage ensure the model is continuously fed.

如果数据供给跟不上，GPU 就会闲置——通常最大、最廉价的加速来自修复输入端，而非计算本身。并行化、固定内存（pinned memory）、异步传输和快速存储可确保模型被持续喂入数据。

Keep the GPUs fed by parallelizing data loaders, using pinned memory and async transfers, and storing data on fast NVMe—preferably with GPUDirect Storage. Stripe, cache, and compress wisely. Measure end-to-end throughput so I/O scales with cluster size, and write checkpoints/logs asynchronously. Here are some tips on I/O optimizations for your data pipeline:

通过并行化数据加载器、使用固定内存和异步传输，以及把数据存放在快速 NVMe 上（最好搭配 GPUDirect Storage）来让 GPU 持续有数据可用。明智地做条带化（stripe）、缓存与压缩。测量端到端吞吐量，使 I/O 能随集群规模扩展，并异步写入检查点/日志。以下是针对数据流水线的一些 I/O 优化建议：

*Load data in parallel* Use multiple workers/threads to load and preprocess data for the GPUs. The default of one to two data loader workers may be insufficient. Profile and increase the number of data loader processes/threads using PyTorch DataLoader(num_workers=N), for example, until the data input is no longer the bottleneck. High core-count CPUs exist to feed those GPUs, so make sure you utilize them.

_并行加载数据_ 使用多个工作进程/线程为 GPU 加载和预处理数据。默认的一到两个数据加载器工作进程可能不够用。对其进行剖析，并逐步增加数据加载进程/线程的数量，例如使用 PyTorch 的 DataLoader(num_workers=N)，直到数据输入不再是瓶颈为止。高核心数的 CPU 正是为喂饱这些 GPU 而存在的，所以务必充分利用它们。

*Pin host memory for I/O* Enable pinned (aka page-locked) memory for data transfer buffers. Many frameworks have an option like PyTorch’s pin_memory=True for its DataLoader to allocate host memory that the GPU can DMA from directly. Using pinned memory significantly improves H2D copy throughput. Combine this with asynchronous transfers to overlap data loading with computation.

_为 I/O 固定主机内存_ 为数据传输缓冲区启用固定（亦称页锁定，page-locked）内存。许多框架都提供类似 PyTorch DataLoader 的 pin_memory=True 的选项，用于分配 GPU 可直接 DMA 读取的主机内存。使用固定内存能显著提升 H2D 拷贝吞吐量。将其与异步传输结合，让数据加载与计算重叠。

*Overlap compute and data transfers* Pipeline your input data. While the GPU is busy computing on batch N, load and prepare batch N+1 on the CPU and transfer it in the background using CUDA streams and nonblocking cudaMemcpyAsync. This double buffering hides latency—the GPU ideally never waits for data. Ensure your training loop uses asynchronous transfers. For example, in PyTorch, you can copy tensors to GPU with non_blocking=True. Asynchronous transfer allows the CPU to continue running while the data transfer is in progress in the background. This will improve performance by overlapping computation with data transfer.

_重叠计算与数据传输_ 对输入数据做流水线化。当 GPU 正忙于计算第 N 批时，在 CPU 上加载并准备第 N+1 批，并使用 CUDA 流和非阻塞的 cudaMemcpyAsync 在后台完成传输。这种双缓冲（double buffering）能隐藏延迟——理想情况下 GPU 永不等待数据。确保你的训练循环使用异步传输。例如在 PyTorch 中，你可以用 non_blocking=True 将张量拷贝到 GPU。异步传输允许 CPU 在数据传输于后台进行时继续运行，从而通过让计算与数据传输重叠来提升性能。

*Use fast storage (NVMe/SSD)* Store training data on fast local NVMe SSDs or a high-performance parallel filesystem. Spinning disks will severely limit throughput. If available, enable GPUDirect Storage (GDS) so that GPUs can stream data directly from NVMe or network storage—bypassing the CPU. This further reduces I/O latency and CPU load when reading large datasets. For large datasets, consider each node having a local copy or shard of the data. If using network storage, prefer a distributed filesystem like Lustre with striping or an object store that can serve many clients in parallel.

_使用快速存储（NVMe/SSD）_ 将训练数据存放在快速的本地 NVMe SSD 或高性能并行文件系统上。机械硬盘会严重限制吞吐量。若条件允许，启用 GPUDirect Storage（GDS），使 GPU 能直接从 NVMe 或网络存储流式读取数据——绕过 CPU。这在读取大型数据集时能进一步降低 I/O 延迟和 CPU 负载。对于大型数据集，可考虑让每个节点持有一份本地副本或数据分片。若使用网络存储，优先选用像带条带化的 Lustre 这样的分布式文件系统，或选用能并行服务众多客户端的对象存储。

*Tune I/O concurrency and striping* Avoid bottlenecks from single-file access. If one large file is used by all workers, stripe it across multiple storage targets or split it into chunks so multiple servers can serve it. For instance, break datasets into multiple files and have each data loader worker read different files simultaneously. This maximizes aggregate bandwidth from the storage system.

_调优 I/O 并发与条带化_ 避免单文件访问带来的瓶颈。如果一个大文件被所有工作进程使用，就把它跨多个存储目标做条带化，或将其切分成多个块，让多台服务器都能提供服务。例如，把数据集拆成多个文件，让每个数据加载器工作进程同时读取不同文件。这样可最大化存储系统的聚合带宽。

*Optimize small files access* If your dataset consists of millions of small files, mitigate metadata overhead. Opening too many small files per second can overwhelm the filesystem’s metadata server. Solutions pack small files into larger containers, such as tar or RecordIO files; use data ingestion libraries that batch reads; or ensure metadata caching is enabled on clients. This reduces per-file overhead and speeds up epoch start times.

_优化小文件访问_ 如果你的数据集由数百万个小文件组成，要设法缓解元数据开销。每秒打开太多小文件会压垮文件系统的元数据服务器。解决办法包括：将小文件打包进更大的容器中，例如 tar 或 RecordIO 文件；使用批量读取的数据摄取库；或确保在客户端启用元数据缓存。这能降低每文件的开销并加快每轮（epoch）的启动时间。

*Use client-side caching when available* Take advantage of any caching layer. If using NFS, increase the client cache size and duration. For distributed filesystems, consider a caching daemon or even manually caching part of the dataset on a local disk. The goal is to avoid repeatedly reading the same data from a slow source. If each node processes the same files at different times, a local cache can drastically cut redundant I/O.

_在可用时使用客户端缓存_ 善用任何缓存层。若使用 NFS，增大客户端缓存的大小与保留时长。对于分布式文件系统，可考虑使用缓存守护进程，甚至手动将数据集的一部分缓存到本地磁盘。目标是避免反复从慢速源读取相同的数据。如果每个节点在不同时刻处理相同的文件，本地缓存能大幅削减冗余 I/O。

*Compress data wisely* Store the dataset compressed if I/O is the bottleneck, but use lightweight compression, such as LZ4 or Zstd fast mode. This trades some CPU to reduce I/O volume. If the CPU becomes the bottleneck due to decompression, consider multithreaded decompression or offloading to accelerators. Also, overlap decompression with reading by using one thread to read compressed data and another thread to decompress the data in parallel. Modern GPUs can perform on-the-fly data decompression using GPU computing resources (or specialized decoders for image/visual data) when paired with GPUDirect Storage and the cuFile I/O stack.

_明智地压缩数据_ 若 I/O 是瓶颈，则以压缩形式存储数据集，但要使用轻量级压缩，例如 LZ4 或 Zstd 的快速模式。这以少量 CPU 换取更小的 I/O 数据量。如果解压导致 CPU 成为瓶颈，可考虑多线程解压或将其卸载到加速器上。此外，通过用一个线程读取压缩数据、另一个线程并行解压的方式，让解压与读取重叠。现代 GPU 在搭配 GPUDirect Storage 与 cuFile I/O 栈时，可利用 GPU 计算资源（或针对图像/视觉数据的专用解码器）进行实时数据解压。

*Measure throughput and eliminate bottlenecks* Continuously monitor the data pipeline’s throughput. If GPUs aren’t near 100% utilization and you suspect input lag, measure how many MB/s you’re reading from disk and how busy the data loader cores are. Tools like dstat or NVIDIA’s DCGM can reveal if GPUs are waiting on data. Systematically tune each component by bumping up prefetch buffers, increasing network buffer sizes, optimizing disk RAID settings, etc. Do this until the input pipeline can feed data as fast as GPUs consume it. Often, these optimizations raise GPU utilization from ~70% to > 95% on the same hardware by removing I/O stalls.

_测量吞吐量并消除瓶颈_ 持续监控数据流水线的吞吐量。如果 GPU 利用率没有接近 100% 且你怀疑输入端存在滞后，就测量你从磁盘读取了多少 MB/s，以及数据加载器核心有多繁忙。像 dstat 或 NVIDIA 的 DCGM 这样的工具可以揭示 GPU 是否在等待数据。系统性地逐一调优各组件：加大预取缓冲区、增加网络缓冲区大小、优化磁盘 RAID 设置等。持续这样做，直到输入流水线能以 GPU 消耗数据的速度供给数据。通常，这些优化能在同一硬件上通过消除 I/O 停顿把 GPU 利用率从约 70% 提升到 > 95%。

*Scale I/O for multinode* At cluster scale, ensure the storage system can handle aggregate throughput. For example, 8 GPUs consuming 200 MB/s each is 1.6 GB/s per node. Across 100 nodes, that’s 160 GB/s needed. Very few central filesystems can sustain this. Mitigate by sharding data across storage servers, using per-node caches, or preloading data onto each node’s local disk. Trading off storage space for throughput (e.g., multiple copies of data) is often worth it to avoid starving expensive GPUs.

_为多节点扩展 I/O_ 在集群规模下，要确保存储系统能承受聚合吞吐量。例如，8 块 GPU 每块消耗 200 MB/s，就是每节点 1.6 GB/s。横跨 100 个节点，就需要 160 GB/s。极少有中心式文件系统能持续支撑这一水平。缓解办法有：将数据跨存储服务器分片、使用每节点缓存，或将数据预加载到每个节点的本地磁盘。用存储空间换吞吐量（例如保存数据的多份副本）往往是值得的，可避免让昂贵的 GPU 挨饿。

*Minimize checkpointing and logging overhead* Write checkpoints and logs efficiently. Use asynchronous writes for checkpoints if possible, or write to local disk, then copy to network storage to avoid stalling training. Compress checkpoints or use sparse storage formats to reduce size. Limit logging frequency on each step by aggregating iteration statistics and logging only every Nth iteration rather than every iteration. This will greatly reduce I/O overhead.

_尽量降低检查点与日志的开销_ 高效地写入检查点和日志。若可能，对检查点使用异步写入，或先写到本地磁盘再拷贝到网络存储，以免阻塞训练。压缩检查点或使用稀疏存储格式来减小体积。通过聚合迭代统计并每隔 N 次迭代才记录一次（而非每次迭代都记录）来限制每步的日志频率。这将大幅降低 I/O 开销。

You can also suspend a running GPU process with cuda-checkpoint and Checkpoint/Restore in Userspace (CRIU) to persist the process image. When ready to resume, the CUDA driver can restore device memory and CUDA state—even on to other GPUs of the same device type. Treat this as complementary to your model’s state-dict or sharded checkpoint files rather than a replacement.

你也可以用 cuda-checkpoint 配合用户空间中的检查点/恢复（Checkpoint/Restore in Userspace，CRIU）挂起一个正在运行的 GPU 进程，以持久化其进程镜像。当准备恢复时，CUDA 驱动可以还原设备内存和 CUDA 状态——甚至可以还原到同一设备类型的其他 GPU 上。应把它视为对模型 state-dict 或分片检查点文件的补充，而非替代。

## Data Processing Pipelines

## 数据处理流水线

The format, layout, and locality of data determine how smoothly the pipeline runs at scale. Binary formats, sharding, caching, and prioritized threads turn I/O from a bottleneck into a steady stream.

数据的格式、布局和局部性（locality）决定了流水线在规模化时能否顺畅运行。二进制格式、分片、缓存以及提升优先级的线程，能把 I/O 从瓶颈变成稳定的数据流。

Convert datasets to binary or memory-mapped formats, shard across storage and nodes, and raise thread priorities or move simple augments to the GPU to prevent stalls. Cache hot data/KV states, prefetch and buffer aggressively, and size batches to keep the pipeline smooth from disk to device. The following are tips for improving your data processing:

将数据集转换为二进制或内存映射（memory-mapped）格式，跨存储与节点做分片，并提升线程优先级或将简单的数据增强（augment）移到 GPU 上，以防止停顿。缓存热数据/KV 状态，激进地做预取与缓冲，并合理设置批大小，使流水线从磁盘到设备始终顺畅。以下是改进数据处理的一些建议：

*Use binary data formats* Convert datasets to binary formats, such as TFRecords, LMDB, or memory-mapped arrays. This conversion reduces the overhead associated with handling millions of small files and accelerates data ingestion.

_使用二进制数据格式_ 将数据集转换为二进制格式，例如 TFRecords、LMDB 或内存映射数组。这种转换能减少处理数百万个小文件所带来的开销，并加速数据摄取。

*Tune the file system* In addition to mounting file systems with noatime and increasing read-ahead, consider sharding data across multiple storage nodes to distribute I/O load and prevent bottlenecks on a single server.

_调优文件系统_ 除了以 noatime 挂载文件系统并增大预读（read-ahead）之外，还可考虑将数据跨多个存储节点分片，以分散 I/O 负载，防止单台服务器成为瓶颈。

*Disable hyperthreading for CPU-bound workloads* For data pipelines that are heavily CPU-bound, disabling hyperthreading can reduce resource contention and lead to more consistent performance. This is especially beneficial on systems where single-thread performance is critical.

_为 CPU 密集型工作负载禁用超线程_ 对于严重受 CPU 限制的数据流水线，禁用超线程（hyperthreading）可以减少资源争用，带来更稳定的性能。这在单线程性能至关重要的系统上尤其有益。

*Elevate thread priorities* Increase the scheduling priority of data loader and preprocessing CPU threads using tools, such as chrt or pthread_setschedparam. By giving these threads higher priority, you ensure that data is fed to the GPU with minimal latency, reducing the chance of pipeline stalls.

_提升线程优先级_ 使用 chrt 或 pthread_setschedparam 等工具，提高数据加载器和预处理 CPU 线程的调度优先级。通过给这些线程更高的优先级，你能确保以最小的延迟向 GPU 供给数据，降低流水线停顿的概率。

*Cache frequently used data* Leverage operating system page caches or a dedicated RAM disk to cache frequently accessed data. This approach is especially beneficial in applications like NLP, where certain tokens or phrases are accessed repeatedly, reducing redundant processing and I/O overhead.

_缓存频繁使用的数据_ 利用操作系统页缓存（page cache）或专用的 RAM 磁盘来缓存频繁访问的数据。这种方法在如 NLP 这类应用中尤为有益，因为某些 token 或短语会被反复访问，从而减少冗余处理和 I/O 开销。

*Prefetch and buffer data* Always load data ahead of the iteration that needs it. Use background data loader threads or processes, such as PyTorch DataLoader with prefetch_factor. For distributed training, use DistributedSampler to ensure each process gets unique data to avoid redundant I/O.

_预取并缓冲数据_ 始终在需要某次迭代的数据之前就把它加载好。使用后台数据加载器线程或进程，例如带 prefetch_factor 的 PyTorch DataLoader。对于分布式训练，使用 DistributedSampler 来确保每个进程获得各自独有的数据，以避免冗余 I/O。

*Parallelize data transformations* If CPU preprocessing—such as image augmentation and text tokenization—is heavy, distribute it across multiple worker threads/processes. Profile to ensure the CPU isn’t the bottleneck while GPUs wait. If it is, either increase workers or move some transforms to GPU, as libraries like NVIDIA’s DALI can do image operations on a GPU asynchronously.

_并行化数据变换_ 如果 CPU 预处理（例如图像增强和文本分词）负担很重，就将其分散到多个工作线程/进程上。做剖析以确保 CPU 不会成为瓶颈而让 GPU 干等。如果确实如此，要么增加工作进程数，要么把部分变换移到 GPU 上——像 NVIDIA 的 DALI 这类库就能在 GPU 上异步完成图像操作。

*Cache model states and outputs* When inferencing with LLMs, it’s beneficial to cache the embeddings and V cache for frequently seen tokens to avoid having to recompute them repeatedly. Similarly, if an LLM training job reuses the same dataset multiple times (called *epochs*), you should leverage OS page cache or RAM to store the hot data.

_缓存模型状态与输出_ 使用 LLM 推理时，缓存频繁出现的 token 的嵌入（embeddings）和 V cache 是有益的，可避免反复重新计算它们。类似地，如果一个 LLM 训练作业多次复用同一数据集（称为轮次），你应当利用 OS 页缓存或 RAM 来存储热数据。

*Shard data across nodes* In multinode training, give each node a subset of data to avoid every node reading the entire dataset from a single source. This scales out I/O. Use a distributed filesystem or manual shard assignment with each node reading different files. This speeds things up and naturally aligns with data parallelism since each node processes its own data shard. DeepSeek’s Fire-Flyer File System (3FS) is one example of a distributed dataset sharding filesystem. DeepSeek’s 3FS achieves multiterabyte-per-second throughput by distributing dataset shards across NVMe SSDs on each node—while minimizing traditional caching. This design feeds each GPU with local high-speed data, avoiding I/O bottlenecks.

_跨节点分片数据_ 在多节点训练中，给每个节点分配一部分数据，以避免每个节点都从单一源读取整个数据集。这能横向扩展 I/O。使用分布式文件系统或手动分片分配，让每个节点读取不同的文件。这既能加速，也天然契合数据并行，因为每个节点处理属于自己的数据分片。DeepSeek 的 Fire-Flyer File System（3FS）就是一个分布式数据集分片文件系统的例子。DeepSeek 的 3FS 通过将数据集分片分散到每个节点的 NVMe SSD 上——同时尽量减少传统缓存——实现了每秒数 TB 的吞吐量。这种设计以本地高速数据喂给每块 GPU，避免了 I/O 瓶颈。

*Monitor pipeline and adjust batch size* Sometimes increasing batch size will push more work onto GPUs and less frequent I/O, improving overall utilization—but only up to a point as it affects convergence. Conversely, if GPUs are waiting on data often, and you cannot speed I/O, you might actually decrease batch size to shorten each iteration and thus reduce idle time or do gradient accumulation of smaller batches such that data reads are more continuous. Find a balance where GPUs are nearly always busy.

_监控流水线并调整批大小_ 有时增大批大小会把更多工作压给 GPU 并降低 I/O 频率，从而提升整体利用率——但只在一定程度内有效，因为它会影响收敛（convergence）。反过来，如果 GPU 经常在等数据而你又无法加快 I/O，你实际上可以减小批大小以缩短每次迭代，从而减少空闲时间；或者对更小的批做梯度累积，使数据读取更连续。要找到一个让 GPU 几乎始终繁忙的平衡点。

*Apply data augmentation on GPU* If augmentation is simple but applied to massive data, like adding noise or normalization, it might be worth doing on GPU to avoid saturating CPU. GPUs are often underutilized during data loading, so using a small CUDA kernel to augment data after loading can be efficient. But be careful not to serialize the pipeline. Use streams to overlap augmentation of batch *N*+1 while batch *N* is training.

_在 GPU 上应用数据增强_ 如果增强操作很简单但作用于海量数据，例如加噪声或归一化，那么放在 GPU 上做也许值得，可避免占满 CPU。GPU 在数据加载期间往往未被充分利用，因此用一个小的 CUDA kernel 在加载后对数据做增强会很高效。但要小心不要让流水线串行化。使用流（streams）让第 _N_+1 批的增强与第 _N_ 批的训练重叠。

Utilize GPU-accelerated libraries like NVIDIA DALI to perform these tasks asynchronously. This helps maintain a smooth and high-throughput data pipeline.

利用像 NVIDIA DALI 这样的 GPU 加速库来异步执行这些任务。这有助于维持一条顺畅且高吞吐的数据流水线。

*Focus on end-to-end throughput (e.g., tokens per second)* Remember that speeding up model compute doesn’t help if your data pipeline cuts throughput in half. Always profile end-to-end, not just the training loop isolated. Use Nsight Systems and Nsight Compute to measure kernel timelines and stalls, or the PyTorch profiler for framework-level attribution. Then compare iteration time with synthetic versus real data to see how much overhead data loading introduces. Aim for less than 10% overhead from ideal. If it’s more than that, invest time in pipeline optimization; it often yields large “free” speedups in training.

_聚焦端到端吞吐量（例如每秒 token 数）_ 记住，如果你的数据流水线把吞吐量砍掉一半，那么加速模型计算也无济于事。始终做端到端剖析，而不仅仅是孤立地剖析训练循环。使用 Nsight Systems 和 Nsight Compute 来测量 kernel 时间线和停顿，或使用 PyTorch profiler 做框架级归因。然后对比使用合成数据与真实数据时的迭代时间，看看数据加载引入了多少开销。目标是控制在相对理想值 10% 以内的开销。若超过这个数，就投入时间做流水线优化；它往往能带来巨大的“免费”训练加速。

## Performance Profiling, Debugging, and Monitoring

## 性能剖析、调试与监控

You can’t optimize what you don’t measure; profiling reveals if you’re compute-bound, memory-bound, I/O-bound, or network-bound so you target the right fix. Continuous telemetry and regression tests keep wins from eroding as code, drivers, and data evolve.

你无法优化你没有测量的东西；剖析能揭示你是受计算限制、受内存限制、受 I/O 限制还是受网络限制，从而让你对症下药。持续的遥测（telemetry）与回归测试能防止已有成果随着代码、驱动和数据的演进而被侵蚀。

Specifically, use Nsight Systems/Compute and framework profilers with NVTX to determine whether you’re compute-bound, memory-bound, I/O-bound, or communication-bound. Trim Python overhead, watch utilization gaps, balance work across ranks, track memory/network/disk health, and gate changes with performance regression tests and alerts. Use the following guidance to profile, monitor, and debug the performance of your AI workloads:

具体来说，使用 Nsight Systems/Compute 和搭配 NVTX 的框架剖析器，来判断你是受计算限制、受内存限制、受 I/O 限制还是受通信限制。削减 Python 开销，关注利用率的空档，在各 rank 之间平衡工作，跟踪内存/网络/磁盘的健康状况，并用性能回归测试和告警来把关变更。使用以下指南来剖析、监控和调试你的 AI 工作负载性能：

*Profile to find bottlenecks and root cause analysis* Regularly run profilers on your training/inference jobs. Use NVIDIA Nsight Systems to get a timeline of CPU and GPU activity. You can also use Nsight Compute or the PyTorch profiler to drill down into kernel efficiency. Identify whether your job is compute bound, memory bound, or waiting on I/O/communication. Target your optimizations accordingly. For example, if your workload is memory bound, focus on reducing memory traffic rather than implementing compute-bound optimizations. Combine with machine-learning–driven analytics to predict and preempt performance bottlenecks. This can help in automating fine-tuning adjustments in real time. When using GPUDirect Storage, enable GDS tracing to correlate cuFile activity with kernel gaps.

_剖析以定位瓶颈并做根因分析_ 定期在你的训练/推理作业上运行剖析器。使用 NVIDIA Nsight Systems 获取 CPU 与 GPU 活动的时间线。你还可以使用 Nsight Compute 或 PyTorch profiler 深入到 kernel 效率层面。判断你的作业是受计算限制、受内存限制，还是在等待 I/O/通信。据此有针对性地做优化。例如，如果你的工作负载受内存限制，就聚焦于减少内存流量，而不是去做受计算限制的优化。结合由机器学习驱动的分析来预测并抢先化解性能瓶颈。这有助于实时自动化微调调整。使用 GPUDirect Storage 时，启用 GDS 追踪，以将 cuFile 活动与 kernel 空档关联起来。

*Eliminate Python overhead* Profile your training scripts to identify Python bottlenecks—such as excessive looping or logging—and replace them with vectorized operations or optimized library calls. Minimizing Python overhead helps ensure that the CPU does not become a hidden bottleneck in the overall system performance.

_消除 Python 开销_ 剖析你的训练脚本以找出 Python 瓶颈——例如过多的循环或日志记录——并用向量化操作或经优化的库调用替换它们。尽量降低 Python 开销有助于确保 CPU 不会成为整体系统性能中的隐藏瓶颈。

*Measure GPU utilization and idle gaps* Continuously monitor GPU utilization, SM efficiency, memory bandwidth usage, etc. If you notice periodic drops in utilization, correlate them with events. For example, a drop in utilization every 5 minutes might coincide with checkpoint saving. Such patterns point to optimization opportunities, such as staggering checkpoints and using asynchronous flushes. Utilize tools like DCGM or nvidia-smi in daemon mode to log these metrics over time.

_测量 GPU 利用率与空闲空档_ 持续监控 GPU 利用率、SM 效率、内存带宽使用率等。如果你注意到利用率周期性下降，就把它们与事件关联起来。例如，每 5 分钟一次的利用率下降可能与检查点保存同时发生。这类模式指向优化机会，例如错开检查点并使用异步刷写。使用像 DCGM 或以守护进程模式运行的 nvidia-smi 之类的工具，随时间记录这些指标。

*Use NVTX markers* Instrument your code with NVTX ranges or framework profiling APIs to label different phases, including data loading, forward pass, backward pass, etc. These markers show up in the Nsight Systems or Perfetto timeline and help you attribute GPU idle times or latencies to specific parts of the pipeline. This makes it easier to communicate to developers which part of the code needs attention. For PyTorch, you can use torch.profiler.record_function().

_使用 NVTX 标记_ 用 NVTX 区间或框架剖析 API 为代码插桩，以标注不同阶段，包括数据加载、前向传播、反向传播等。这些标记会显示在 Nsight Systems 或 Perfetto 的时间线中，帮助你把 GPU 空闲时间或延迟归因到流水线的具体部分。这让你更容易向开发者说明代码的哪一部分需要关注。对于 PyTorch，你可以使用 torch.profiler.record_function()。

*Utilize kernel profiling and analysis tools beyond just the PyTorch profiler* For performance-critical kernels, use Nsight Compute to examine kernel-level metrics like occupancy and throughput, or Nsight Systems to analyze GPU/CPU timelines and overlap. Check achieved occupancy, memory throughput, and instruction throughput. Look for signs of memory bottlenecks, such as memory bandwidth near the hardware maximum. This helps to identify memory-bound workloads. The profiler’s “Issues” section often directly suggests if a kernel is memory bound or compute bound and why. Use this feedback to guide code changes, such as improving memory coalescing if global load efficiency is low.

_善用超越 PyTorch profiler 的 kernel 剖析与分析工具_ 对于性能关键的 kernel，使用 Nsight Compute 来考察 kernel 级指标，如占用率和吞吐量，或使用 Nsight Systems 来分析 GPU/CPU 时间线与重叠。检查实际达成的占用率、内存吞吐量和指令吞吐量。留意内存瓶颈的迹象，例如内存带宽接近硬件上限。这有助于识别受内存限制的工作负载。剖析器的“Issues”部分通常会直接提示某个 kernel 是受内存限制还是受计算限制，以及原因。利用这些反馈来指导代码修改，例如在全局加载效率偏低时改进内存合并。

*Check for warp divergence* Use the profiler to see if warps are diverging, as it can show branch efficiency and divergent branch metrics. Divergence means some threads in a warp are inactive due to branching, which hurts throughput. If significant, revisit the kernel code to restructure conditionals or data assignments to minimize intrawarp divergence and ensure that each warp handles uniform work.

_检查 warp 分化_ 使用剖析器查看 warp 是否发生分化，因为它能显示分支效率和分化分支（divergent branch）指标。分化意味着一个 warp 中的部分线程因分支而处于非活动状态，这会损害吞吐量。如果分化显著，就重新审视 kernel 代码，重构条件判断或数据分配，以最小化 warp 内分化，并确保每个 warp 处理均匀的工作。

*Verify load balancing* In multi-GPU jobs, profile across ranks. Sometimes one GPU (rank 0) does extra work like aggregating stats and data gathering—and often becomes a bottleneck. Monitor each GPU’s timeline. If one GPU is consistently lagging, distribute that extra workload. For example, you can have the nonzero ranks share the I/O and logging responsibilities. Ensuring that all GPUs/ranks have similar workloads avoids the slowest rank dragging the rest.

_核实负载均衡_ 在多 GPU 作业中，跨 rank 做剖析。有时某块 GPU（rank 0）会做额外的工作，如汇总统计和收集数据——并常常因此成为瓶颈。监控每块 GPU 的时间线。如果某块 GPU 持续落后，就把那部分额外工作分摊出去。例如，你可以让非零 rank 分担 I/O 和日志记录的职责。确保所有 GPU/rank 拥有相近的工作量，可避免最慢的 rank 拖累其余部分。

*Monitor memory usage* Track GPU memory allocation and usage over time. Ensure you are not near OOM, which can cause the framework to unexpectedly swap tensors to host, which will cause huge slowdowns. If memory usage climbs iteration by iteration, you have likely identified leaks. In this case, profile with tools like torch.cuda.memory_summary() and Nsight Systems’ GPU memory trace to analyze detailed allocations. On the CPU side, monitor for paging, as your process’s resident memory (RES) should not exceed physical RAM significantly. If you see paging, reduce dataset preload size or increase RAM.

_监控内存使用_ 随时间跟踪 GPU 内存的分配与使用。确保你没有逼近 OOM，否则框架可能会意外地将张量换出到主机，从而导致巨大的性能下降。如果内存使用逐次迭代持续攀升，你很可能已经发现了泄漏。这种情况下，用 torch.cuda.memory_summary() 和 Nsight Systems 的 GPU 内存追踪等工具做剖析，以分析详细的分配情况。在 CPU 一侧，监控换页（paging），因为你进程的常驻内存（RES）不应显著超过物理 RAM。如果你看到换页，就减小数据集预加载规模或增加 RAM。

*Monitor network and disk* For distributed jobs, use OS tools to monitor network throughput and disk throughput. Ensure the actual throughput matches expectations. For example, on a 100 Gbps link, you should see 12.5 GB/s (12.5 GB/s = 100 Gb/s ÷ 8 bits per byte) if fully utilized. If not, the network might be a bottleneck or misconfigured. Similarly, monitor disk I/O on training nodes. If you see spikes of 100% disk utilization and GPU idle, you likely need to buffer or cache data better.

_监控网络与磁盘_ 对于分布式作业，使用操作系统工具监控网络吞吐量和磁盘吞吐量。确保实际吞吐量符合预期。例如，在 100 Gbps 链路上，如果被完全利用，你应当看到 12.5 GB/s（12.5 GB/s = 100 Gb/s ÷ 每字节 8 比特）。如果达不到，网络可能是瓶颈或配置有误。同样地，监控训练节点上的磁盘 I/O。如果你看到磁盘利用率飙升到 100% 而 GPU 空闲，那你很可能需要更好地缓冲或缓存数据。

*Set up alerts for anomalies* In a production or long-running training context, set up automated alerts or logs for events like GPU errors, such as ECC errors, device overheating, etc. This will help identify abnormally slow iterations. For example, NVIDIA’s DCGM can watch health metrics, and you can trigger actions if a GPU starts throttling or encountering errors. This helps catch performance issues—like a cooling failure causing throttling—immediately rather than after the job finishes.

_为异常设置告警_ 在生产或长时间运行的训练场景中，为诸如 GPU 错误（如 ECC 错误、设备过热等）之类的事件设置自动化告警或日志。这有助于识别异常缓慢的迭代。例如，NVIDIA 的 DCGM 可以监视健康指标，当某块 GPU 开始降频或出现错误时，你可以触发相应动作。这有助于立即捕捉性能问题——比如散热故障导致降频——而不是等作业结束后才发现。

*Perform regression testing* Maintain a set of benchmark tasks to run whenever you change software, including CUDA drivers, CUDA versions, AI framework versions, or even your training code. Compare performance to previous runs to catch regressions early. It’s not uncommon for a driver update or code change to inadvertently reduce throughput—a quick profiling run on a standard workload will highlight this so you can investigate. For example, maybe a kernel is accidentally not using Tensor Cores anymore. This is something to look into for sure.

_执行回归测试_ 维护一组基准任务，每当你更改软件（包括 CUDA 驱动、CUDA 版本、AI 框架版本，甚至你的训练代码）时都运行它们。将性能与之前的运行做对比，以尽早发现回退。驱动更新或代码改动无意中降低吞吐量的情况并不罕见——在标准工作负载上快速跑一次剖析就能凸显这一点，让你去排查。例如，也许某个 kernel 意外地不再使用 Tensor Core 了。这绝对是值得深究的事情。

## GPU Programming and CUDA Tuning Optimizations

## GPU 编程与 CUDA 调优优化

Aligning kernels with the memory hierarchy and hardware features is where large, durable gains come from. Fusion, Tensor Cores, CUDA Graphs, and compiler paths (e.g., torch.compile and OpenAI’s Triton) convert launch overhead into useful math.

让 kernel 与内存层次结构及硬件特性对齐，正是巨大而持久的收益之所在。融合（fusion）、Tensor Core、CUDA Graphs 以及编译器路径（例如 torch.compile 和 OpenAI 的 Triton）能把启动开销转化为有用的计算。

Optimize for the memory hierarchy: coalesce global loads, tile into shared memory, manage registers/occupancy, and overlap transfers (e.g., cp.async/TMA) with compute. Prefer tuned libraries and CUDA Graphs, leverage torch.compile and OpenAI’s Triton for fusion, and validate scalability with roofline analysis and PTX/SASS inspection. The following are some GPU and CUDA programming optimization tips and techniques:

面向内存层次结构做优化：合并全局加载、把数据分块（tiling）进共享内存、管理寄存器/占用率，并让传输（例如 cp.async/TMA）与计算重叠。优先使用经调优的库和 CUDA Graphs，借助 torch.compile 和 OpenAI 的 Triton 做融合，并用 roofline 分析以及 PTX/SASS 检查来验证可扩展性。以下是一些 GPU 与 CUDA 编程的优化建议与技巧：

*Understand GPU memory hierarchy* Keep in mind the tiered memory structure of GPUs—registers per thread, shared memory/L1 cache per block/SM, L2 cache across SM, and global HBM. Maximize data reuse in the higher tiers. For example, use registers and shared memory to reuse values and minimize accesses to slower global memory. A good kernel ensures the vast majority of data is either in registers or gets loaded from HBM efficiently using coalescing and caching.

_理解 GPU 内存层次结构_ 牢记 GPU 的分层内存结构——每线程的寄存器、每块/每 SM 的共享内存/L1 缓存、跨 SM 的 L2 缓存，以及全局 HBM。在更高层级中最大化数据复用。例如，用寄存器和共享内存来复用数值，尽量减少对较慢全局内存的访问。一个好的 kernel 能确保绝大多数数据要么在寄存器中，要么通过合并与缓存高效地从 HBM 加载。

*Coalesce global memory accesses* Ensure that threads in the same warp access contiguous memory addresses so that the hardware can service them in as few transactions as possible. Strided or scattered memory access by warp threads will result in multiple memory transactions per warp, effectively wasting bandwidth. Restructure data layouts or index calculations so that whenever a warp loads data, it’s doing so in a single, wide memory transaction.

_合并全局内存访问_ 确保同一 warp 中的线程访问连续的内存地址，从而让硬件能以尽可能少的事务为它们服务。warp 线程的跨步（strided）或分散（scattered）内存访问会导致每个 warp 产生多次内存事务，实质上浪费带宽。重构数据布局或索引计算，使 warp 每次加载数据时都以单次、宽幅的内存事务完成。

*Use shared memory for data reuse* Shared memory is like a manually managed cache with very high bandwidth. Load frequently used data—such as tiles of matrices—into shared memory. And have threads operate on those tiles multiple times before moving on. This popular tiling technique greatly cuts down global memory traffic. Be cautious of shared-memory bank conflicts. Organize shared-memory access patterns or pad data to ensure threads aren’t contending for the same memory bank, which would serialize accesses and reduce performance.

_使用共享内存实现数据复用_ 共享内存就像一块带宽极高、需手动管理的缓存。将频繁使用的数据——例如矩阵的分块——加载到共享内存中，并让线程在转向下一步之前对这些分块多次运算。这种流行的分块技术能大幅削减全局内存流量。要当心共享内存的 bank 冲突。组织好共享内存的访问模式或对数据做填充（pad），以确保线程不会争用同一内存 bank，否则会使访问串行化并降低性能。

*Optimize memory alignment* Align data structures to 128 bytes whenever possible, especially for bulk memory copies or vectorized loads. Misaligned accesses can force multiple transactions even if theoretically coalesced. Using vectorized types like float2 and float4 for global memory I/O can help load/store multiple values per instruction, but ensure your data pointer is properly aligned to the vector size.

_优化内存对齐_ 尽可能将数据结构对齐到 128 字节，尤其是在做批量内存拷贝或向量化加载时。未对齐的访问即便理论上可以合并，也可能被迫产生多次事务。对全局内存 I/O 使用像 float2 和 float4 这样的向量化类型，有助于每条指令加载/存储多个值，但要确保你的数据指针已正确对齐到向量大小。

*Minimize memory transfers* Only transfer data to the GPU when necessary and in large chunks. Consolidate many small transfers into one big transfer if you can. For example, if you have many small arrays to send each iteration, pack them into one buffer and send once. Small, frequent cudaMemcpy can become a bottleneck. If using Unified Memory, use explicit prefetch (cudaMemPrefetchAsync) to stage data on GPU before it’s needed, avoiding on-demand page faults during critical compute sections.

_尽量减少内存传输_ 只在必要时、且以大块的方式向 GPU 传输数据。如果可以，就把许多小传输合并为一次大传输。例如，如果你每次迭代都要发送许多小数组，就把它们打包进一个缓冲区一次性发送。频繁的小额 cudaMemcpy 会成为瓶颈。如果使用统一内存，就用显式预取（cudaMemPrefetchAsync）在数据被需要之前把它暂存到 GPU 上，从而避免在关键计算段中出现按需缺页（page fault）。

*Avoid excessive temporary allocations* Frequent allocation and freeing of GPU memory can hurt performance. For example, frequently using cudaMalloc/cudaFree or device malloc in kernels will cause extra overhead. Instead, reuse memory buffers or use memory pools available within most DL frameworks, like PyTorch, that implement a GPU caching allocator. If writing custom CUDA code, consider using cudaMallocAsync with a memory pool or manage a pool of scratch memory yourself to avoid the overhead of repetitive alloc/free.

_避免过多的临时分配_ 频繁分配和释放 GPU 内存会损害性能。例如，频繁使用 cudaMalloc/cudaFree 或在 kernel 中使用设备端 malloc 都会带来额外开销。相反，应复用内存缓冲区，或使用大多数 DL 框架（如 PyTorch）内置的内存池，它们实现了 GPU 缓存分配器。如果编写自定义 CUDA 代码，可考虑使用带内存池的 cudaMallocAsync，或自行管理一块暂存内存池，以避免反复分配/释放的开销。

*Balance threads and resource use* Achieve a good occupancy-resource balance. Using more threads for higher occupancy helps hide memory latency, but if each thread uses too many registers or too much shared memory, occupancy drops. Tune your kernel launch parameters—including threads per block—to ensure you have enough warps in flight to cover latency, but not so many that each thread is starved of registers or shared memory. In kernels with high instruction-level parallelism (ILP), reducing register usage to boost occupancy might actually hurt performance. The optimal point is usually in the middle of the occupancy spectrum, as maximum occupancy is not always ideal. Use the NVIDIA Nsight Compute Occupancy Calculator to experiment with configurations.

_平衡线程与资源使用_ 实现良好的占用率与资源的平衡。用更多线程换取更高占用率有助于隐藏内存延迟，但如果每个线程使用过多寄存器或过多共享内存，占用率就会下降。调优你的 kernel 启动参数——包括每块线程数——以确保有足够多的 warp 处于运行中（in flight）来掩盖延迟，但又不至于多到让每个线程都缺乏寄存器或共享内存。在具有高指令级并行（ILP）的 kernel 中，为提升占用率而减少寄存器使用反而可能损害性能。最优点通常位于占用率区间的中段，因为最高占用率并不总是理想的。使用 NVIDIA Nsight Compute 的占用率计算器（Occupancy Calculator）来试验各种配置。

*Monitor register and shared-memory usage* Continuously monitor per-thread register and shared-memory consumption using profiling tools like Nsight Compute. If the occupancy is observed to be below 25%, consider increasing the number of threads per block to better utilize available hardware resources. However, verify that this adjustment does not cause excessive register spilling by reviewing detailed occupancy reports and kernel execution metrics. Register spilling can lead to additional memory traffic and degrade overall performance.

_监控寄存器与共享内存使用_ 使用像 Nsight Compute 这样的剖析工具，持续监控每线程的寄存器与共享内存消耗。如果观测到占用率低于 25%，可考虑增加每块的线程数以更好地利用可用硬件资源。但要通过查看详细的占用率报告和 kernel 执行指标，确认这一调整不会导致过度的寄存器溢出（register spilling）。寄存器溢出会引发额外的内存流量并拖累整体性能。

*Overlap memory transfers with computation* Overlap memory transfers with computation whenever possible. Use cudaMemcpyAsync in multiple CUDA streams to prefetch while kernels run. Prefer the Tensor Memory Accelerator for bulk movement to shared memory, and use cp.async for fine-grained staged copies and prefetch. These approaches effectively mask global memory latency by overlapping data transfers with computation, making sure the GPU cores remain fully utilized without waiting for memory operations to complete.

_让内存传输与计算重叠_ 尽可能让内存传输与计算重叠。在多个 CUDA 流中使用 cudaMemcpyAsync，以在 kernel 运行时进行预取。对于到共享内存的批量搬移，优先使用张量内存加速器（Tensor Memory Accelerator，TMA）；对于细粒度的分阶段拷贝和预取，使用 cp.async。这些方法通过让数据传输与计算重叠，有效地掩盖了全局内存延迟，确保 GPU 核心保持充分利用，而不必等待内存操作完成。

*Use bulk prefetching when possible* For predictable patterns, prefetch into L2 using the PTX cp.async.bulk.prefetch.tensor.[1–5]d.L2.global* (or the prefetch.global.L2 family), and use TMA (e.g., cp.async.bulk.tensor) to stage blocks into shared memory. You can also use cp.async to stage global memory into shared memory asynchronously and overlap copy with compute. You can also explicitly load data into registers ahead of use. These proactive methods reduce the delay caused by global memory accesses and make sure that critical data is available in faster, lower-latency storage—such as registers or shared memory—right when it’s needed, thus minimizing execution stalls and improving overall kernel efficiency.

_在可能时使用批量预取_ 对于可预测的模式，使用 PTX 的 cp.async.bulk.prefetch.tensor.[1–5]d.L2.global\*（或 prefetch.global.L2 系列）将数据预取进 L2，并使用 TMA（例如 cp.async.bulk.tensor）将数据块暂存进共享内存。你也可以用 cp.async 把全局内存异步暂存进共享内存，让拷贝与计算重叠。你还可以显式地在使用之前把数据加载进寄存器。这些主动式方法减少了全局内存访问造成的延迟，确保关键数据在需要的那一刻就已就位于更快、更低延迟的存储中——例如寄存器或共享内存，从而最小化执行停顿并提升整体 kernel 效率。

*Utilize cooperative groups* Utilize CUDA’s cooperative groups to achieve efficient, localized synchronization among a subset of threads rather than enforcing a full block-wide barrier. This technique enables finer-grained control over synchronization, reducing unnecessary waiting times and overhead. By grouping threads that share data or perform related computations, you can synchronize only those threads that require coordination, which can lead to a more efficient execution pattern and better overall throughput.

_利用协作组_ 利用 CUDA 的协作组（cooperative groups）在一部分线程之间实现高效的局部同步，而不必强制施加覆盖整个线程块的屏障。这一技术能对同步实现更细粒度的控制，减少不必要的等待时间和开销。通过把共享数据或执行相关计算的线程编成一组，你可以只同步那些确实需要协调的线程，从而带来更高效的执行模式和更好的整体吞吐量。

*Optimize warp divergence* Structure your code so that threads within a warp follow the same execution path as much as possible. Divergence can double the execution time for that warp—for example, half the warp (16 threads) taking one branch and half the warp (16 threads) taking another branch. If you have branches that some data rarely triggers, consider “sorting” or grouping data so warps handle uniform cases such that all are true or all are false. Use warp-level primitives like ballot and shuffle to create branchless solutions for certain problems. Treat a warp as the unit of work, and aim for all 32 threads to do identical work in lockstep for maximum efficiency.

_优化 warp 分化_ 组织你的代码，使一个 warp 内的线程尽可能遵循相同的执行路径。分化会让该 warp 的执行时间翻倍——例如半个 warp（16 个线程）走一个分支，另半个 warp（16 个线程）走另一个分支。如果有些分支只被少量数据偶尔触发，可考虑对数据做“排序”或分组，让 warp 处理均匀的情形，使其全部为真或全部为假。使用像 ballot 和 shuffle 这样的 warp 级原语，为某些问题构造无分支（branchless）的解法。把一个 warp 视为工作单元，力求全部 32 个线程步调一致（lockstep）地做完全相同的工作，以获得最高效率。

*Leverage warp-level operations* Use CUDA’s warp intrinsics to let threads communicate without going to shared memory when appropriate. For example, use __shfl_sync to broadcast a value to all threads in a warp or to do warp-level reductions—like summing registers across a warp—instead of each thread writing to shared memory. These intrinsics bypass slower memory and can speed up algorithms like reductions or scans that can be done within warps. By processing these tasks within a warp, you avoid the latency associated with shared memory and full-block synchronizations.

_善用 warp 级操作_ 使用 CUDA 的 warp 内建函数（intrinsics），在合适时让线程无需经过共享内存即可通信。例如，使用 \_\_shfl_sync 将一个值广播到 warp 中的所有线程，或做 warp 级归约（reduction）——如跨一个 warp 对寄存器求和——而不是让每个线程都写入共享内存。这些内建函数绕过了较慢的内存，能加速像归约或扫描（scan）这类可在 warp 内完成的算法。通过在一个 warp 内处理这些任务，你避免了与共享内存和整块同步相关的延迟。

*Use CUDA streams for concurrency* Within a single process/GPU, launch independent kernels in different CUDA streams to overlap their execution if they don’t use all resources. Overlap computation with computation—e.g., one stream computing one part of the model while another stream launches an independent kernel like data preprocessing on GPU or asynchronous memcpy. Be mindful of dependencies and use CUDA events to synchronize when needed. Proper use of streams can increase GPU utilization by not leaving any resource idle—especially if you have some kernels that are light.

_使用 CUDA 流实现并发_ 在单个进程/GPU 内，如果各独立 kernel 不会占用所有资源，就把它们发射到不同的 CUDA 流中，以重叠它们的执行。让计算与计算重叠——例如，一个流计算模型的一部分，另一个流发射一个独立的 kernel，如在 GPU 上做数据预处理或异步 memcpy。要留意依赖关系，并在需要时用 CUDA 事件来同步。恰当地使用流能通过不让任何资源闲置来提升 GPU 利用率——尤其是当你有一些较轻的 kernel 时。

*Prefer library functions* Wherever possible, use NVIDIA’s optimized libraries, such as cuBLAS, cuDNN, Thrust, and NCCL, for core math and collective operations. For point-to-point GPU data movement in distributed inference, use NIXL where available. You can also use NVSHMEM when you need fine-grained GPU-initiated transfers. These are heavily optimized for each GPU architecture and often approach theoretical “speed of light” peaks. This will save you the trouble of reinventing them. For example, use cuBLAS GEMM for matrix multiplies rather than a custom kernel, unless you have a very special pattern. The libraries also handle new hardware features transparently. AI frameworks like PyTorch (and its compiler) use these optimized libraries under the hood.

_优先使用库函数_ 只要可能，就使用 NVIDIA 经优化的库，如 cuBLAS、cuDNN、Thrust 和 NCCL，来完成核心数学与集合通信操作。在分布式推理中做 GPU 点对点数据搬移时，在可用处使用 NIXL。当你需要细粒度、由 GPU 发起的传输时，也可以使用 NVSHMEM。这些库针对每一代 GPU 架构都做了大量优化，往往能逼近理论上的“光速”（speed of light）峰值。这为你省去了重新造轮子的麻烦。例如，除非你有非常特殊的模式，否则做矩阵乘法请用 cuBLAS GEMM 而非自定义 kernel。这些库还会透明地处理新的硬件特性。像 PyTorch（及其编译器）这样的 AI 框架在底层就使用这些经优化的库。

*Use CUDA Graphs for repeated launches* If you have a static training loop that is launched thousands of times, consider using CUDA Graphs to capture and launch the sequence of operations as a graph. This can significantly reduce CPU launch overhead for each iteration, especially in multi-GPU scenarios where launching many kernels and memcpy’s can put extra pressure on the CPU and incur additional latency.

_为重复发射使用 CUDA Graphs_ 如果你有一个会被发射数千次的静态训练循环，可考虑使用 CUDA Graphs 将一连串操作捕获并作为一张图来发射。这能显著降低每次迭代的 CPU 启动开销，尤其是在多 GPU 场景下——发射大量 kernel 和 memcpy 会给 CPU 带来额外压力并引入额外延迟。

*Check for scalability limits* As you optimize a kernel, periodically check how it scales with problem size and across architectures. A kernel might achieve great occupancy and performance on a small input but not scale well to larger inputs, as it may start thrashing L2 cache or running into memory-cache evictions. Use roofline analysis. Compare achieved FLOPS and bandwidth to hardware limits to ensure you’re not leaving performance on the table.

_检查可扩展性上限_ 在优化一个 kernel 时，要定期检查它如何随问题规模以及跨架构扩展。一个 kernel 可能在小输入上达到很高的占用率与性能，却无法很好地扩展到更大的输入，因为它可能开始抖动（thrash）L2 缓存或遭遇内存缓存驱逐。使用 roofline 分析。将实际达成的 FLOPS 和带宽与硬件上限做对比，以确保你没有白白浪费性能。

*Inspect PTX and SASS for advanced kernel analysis* For performance-critical custom CUDA kernels, use Nsight Compute to examine the generated PTX and SASS. This deep dive can reveal issues like memory bank conflicts or redundant computations, guiding you toward targeted low-level optimizations.

_检查 PTX 和 SASS 以做高级 kernel 分析_ 对于性能关键的自定义 CUDA kernel，使用 Nsight Compute 来考察生成的 PTX 和 SASS。这种深入剖析能揭示诸如内存 bank 冲突或冗余计算之类的问题，引导你做有针对性的底层优化。

*Use the PyTorch compiler* Take advantage of PyTorch’s torch.compile to fuse Python-level operations into optimized kernels through TorchInductor. The compiler can also reduce launch overhead by integrating CUDA Graphs. Typical gains of about 10%–40% are common once the optimizations are warmed up. This eliminates interpreter overhead and unlocks compiler-level optimizations.

_使用 PyTorch 编译器_ 善用 PyTorch 的 torch.compile，通过 TorchInductor 将 Python 层级的操作融合为经优化的 kernel。该编译器还能通过集成 CUDA Graphs 来降低启动开销。一旦优化预热完毕，约 10%–40% 的典型收益很常见。这消除了解释器开销并解锁了编译器层级的优化。

In practice, enabling torch.compile has produced substantial speedups (e.g., 20%–50% on many models) by automatically combining kernels and utilizing NVIDIA GPU hardware (e.g., Tensor Cores) more efficiently. Always test compiled mode on your model. While it can massively boost throughput, you should ensure compatibility and correctness before deploying. When graphs are stable, enable CUDA Graphs to reduce per-iteration CPU overhead. Keep static memory pools to satisfy pointer-stability constraints.

在实践中，启用 torch.compile 通过自动组合 kernel 并更高效地利用 NVIDIA GPU 硬件（例如 Tensor Core），已带来可观的加速（例如在许多模型上达到 20%–50%）。始终在你的模型上测试编译模式。虽然它能大幅提升吞吐量，但你应在部署前确保兼容性与正确性。当图稳定后，启用 CUDA Graphs 以降低每次迭代的 CPU 开销。保持静态内存池以满足指针稳定性（pointer-stability）约束。

*Plan for dynamic shapes* If your input sizes vary, use torch._dynamo.mark_dynamic() to annotate dynamic dimensions or export shape-polymorphic graphs with torch.export(), and then compile. Control recompilation behavior with torch.compiler.set_stance() using "fail_on_recompile" and torch._dynamo.error_on_graph_break() to surface problematic shape churn in testing and CI. Use static shapes where possible to enable CUDA Graphs to reduce per-iteration CPU overhead.

_为动态形状做规划_ 如果你的输入尺寸会变化，就用 torch.\_dynamo.mark_dynamic() 标注动态维度，或用 torch.export() 导出形状多态（shape-polymorphic）的图，然后再编译。用带 "fail_on_recompile" 的 torch.compiler.set_stance() 以及 torch.\_dynamo.error_on_graph_break() 来控制重编译行为，从而在测试与 CI 中暴露有问题的形状抖动（shape churn）。在可能处使用静态形状，以便启用 CUDA Graphs 来降低每次迭代的 CPU 开销。

*Leverage Triton kernels using* torch.compile If PyTorch doesn’t fuse an operation well, consider writing a custom GPU kernel in Triton and integrating it. PyTorch makes it easy to register a custom GPU kernel with torch.library.triton_op.

*借助* torch.compile *使用 Triton 核函数* 如果 PyTorch 没能很好地融合某个运算，可以考虑用 Triton 编写一个自定义 GPU 核函数并将其集成进来。PyTorch 让你能方便地用 torch.library.triton_op 注册自定义 GPU 核函数。

*Use autotuning when available* Enable library autotuning features to maximize low-level performance. For example, set torch.backends.cudnn.benchmark=True when input sizes are fixed. This lets NVIDIA’s cuDNN library try multiple convolution algorithms and pick the fastest one for your hardware. The one-time overhead leads to optimized kernels that can accelerate training and inference. If exact reproducibility isn’t required, allow nondeterministic algorithms by disabling cudnn.deterministic to unlock these faster implementations.

*在可用时启用自动调优* 开启库的自动调优功能以最大化底层性能。例如，当输入尺寸固定时设置 torch.backends.cudnn.benchmark=True。这会让 NVIDIA 的 cuDNN 库尝试多种卷积算法，并为你的硬件挑选最快的一种。这一次性开销会换来经过优化的核函数，从而加速训练与推理。如果不要求严格可复现，可以通过禁用 cudnn.deterministic 来允许非确定性算法，从而解锁这些更快的实现。

*Leverage the read-only path* Mark frequently used constants or coefficients as read-only so the GPU can cache them in the dedicated L1 read-only cache. In CUDA C++, you can use const __restrict__ pointers to hint that data is immutable. On modern GPU architectures, the compiler generates cached global loads for const __restrict__ qualified pointers. When using AI frameworks and libraries, make sure that lookup tables or static weights are on the device and treated as constant. This optimization reduces global memory traffic and latency for those values, as each SM can quickly fetch them from cache instead of repeatedly accessing slow DRAM.

*善用只读路径* 把频繁使用的常量或系数标记为只读，这样 GPU 就能把它们缓存在专用的 L1 只读缓存中。在 CUDA C++ 中，你可以使用 const __restrict__ 指针来提示数据不可变。在现代 GPU 架构上，编译器会为带 const __restrict__ 限定的指针生成带缓存的全局加载。使用 AI 框架和库时，确保查找表或静态权重位于设备上并被当作常量处理。这一优化能降低这些值的全局内存流量与延迟，因为每个 SM 都能从缓存中快速取用它们，而不必反复访问缓慢的 DRAM。

## Kernel Scheduling and Execution Optimizations

## 核函数调度与执行优化

Launch overhead and unnecessary syncs create idle gaps that crush throughput. Fusing small kernels and using persistent/dynamic strategies keeps the device busy and latency hidden.

启动开销与不必要的同步会制造空闲间隙，严重拖垮吞吐量。融合小核函数、采用持久化/动态策略能让设备保持忙碌并隐藏延迟。

Keep the device busy by minimizing synchronizations, fusing small kernels, and using persistent kernels when launching the same work repeatedly. For irregular tasks, consider GPU dynamic parallelism—but use it judiciously to avoid adding overhead. The following are tips on improving kernel scheduling and execution:

通过尽量减少同步、融合小核函数、以及在反复启动相同工作时使用持久化核函数（persistent kernel），让设备保持忙碌。对于不规则任务，可以考虑 GPU 动态并行（dynamic parallelism）——但要审慎使用，以免徒增开销。以下是改进核函数调度与执行的一些建议：

*Minimize GPU synchronization calls* Avoid unnecessary global synchronizations that stall GPU progress. Excessive use of cudaDeviceSynchronize() or blocking GPU operations (like synchronous memory copies) will insert idle gaps where neither the CPU nor GPU can do useful work. Synchronize only when absolutely needed. For instance, synchronize when transferring final results or when debugging. By letting asynchronous operations queue up, you keep the GPU busy and the CPU free to prepare further work. This leads to a more continuous execution pipeline.

*尽量减少 GPU 同步调用* 避免不必要的全局同步，以免阻塞 GPU 的推进。过度使用 cudaDeviceSynchronize() 或阻塞式 GPU 操作（如同步内存拷贝）会插入空闲间隙，使 CPU 与 GPU 都无法做有用的工作。只在绝对必要时才同步。例如，在传输最终结果或调试时再同步。通过让异步操作排队进行，你能让 GPU 保持忙碌、CPU 得以腾出手来准备后续工作。这会带来更连续的执行流水线。

*Fuse small kernels to amortize launch overhead* If you have many tiny GPU kernels launching back-to-back, consider merging their operations to run in a single kernel where possible. Every kernel launch has a fixed cost on the order of tens of microseconds, so combining operations through manual CUDA kernel fusion, XLA fusion, or tools like NVIDIA CUTLASS/Triton for custom ops can improve throughput. Fused kernels spend more time doing actual work and less time in launch overhead or memory round trips. This is especially helpful in inference or preprocessing pipelines where chains of elementwise ops can be executed in one go. Try torch.compile(mode="reduce-overhead") first. The compiler can fuse operation chains and wrap steady regions in CUDA Graphs. This will reduce CPU launch overhead. For unfused hotspots, consider migrating them to Triton kernels and using asynchronous TMA and automatic warp specialization where applicable.

*融合小核函数以摊薄启动开销* 如果你有许多相继背靠背启动的微小 GPU 核函数，尽量考虑把它们的运算合并到单个核函数中执行。每次核函数启动都有一个数十微秒量级的固定成本，因此通过手工 CUDA 核函数融合（kernel fusion）、XLA 融合、或借助 NVIDIA CUTLASS/Triton 等工具编写自定义算子来合并运算，都能提升吞吐量。融合后的核函数把更多时间花在实际计算上，减少了启动开销或内存往返的时间。这在推理或预处理流水线中尤其有用，因为一连串逐元素运算可以一次性执行完毕。先试试 torch.compile(mode="reduce-overhead")。编译器能融合运算链，并把稳态区域包进 CUDA Graphs。这会降低 CPU 侧的启动开销。对于未融合的热点，可考虑将其迁移到 Triton 核函数，并在适用处使用异步 TMA 与自动 warp 特化。

*Utilize GPU dynamic parallelism for intra-GPU work scheduling* Utilize CUDA’s Dynamic Parallelism to let GPU kernels launch other kernels from the GPU without returning to the CPU. In scenarios with unpredictable or iterative work, such as an algorithm that needs to spawn additional tasks based on intermediate results, dynamic parallelism cuts latency by removing the CPU launch bottleneck. For example, a parent kernel can divide and launch child kernels for further processing directly on the device. This keeps the entire workflow on the GPU, avoiding CPU intervention and enabling better overlap and utilization. Use this judiciously, however, as it can introduce its own overhead if overused.

*利用 GPU 动态并行进行 GPU 内部的工作调度* 利用 CUDA 的动态并行，让 GPU 核函数无需返回 CPU 即可从 GPU 上启动其他核函数。在工作不可预测或迭代式的场景中（例如某算法需要根据中间结果派生额外任务），动态并行通过消除 CPU 启动瓶颈来削减延迟。例如，父核函数可以直接在设备上划分并启动子核函数以做进一步处理。这让整个工作流保持在 GPU 上，避免 CPU 介入，实现更好的重叠与利用率。但要审慎使用，因为过度使用会引入其自身的开销。

*Use persistent kernels for repeated workloads* Use a persistent kernel strategy when a workload involves launching identical kernels in rapid succession, such as processing a work queue or streaming batches with the same computation. A persistent kernel is launched once and remains active, reusing threads to handle many units of work in a loop, rather than launching a fresh kernel for each unit. This approach trades a more complex kernel design for significantly lower scheduling overhead. By keeping the kernel alive, you avoid repeated launch costs and can achieve higher sustained occupancy. High-performance distributed training and inference systems often employ this technique to maximize throughput and minimize latency for iterative tasks.

*对重复的工作负载使用持久化核函数* 当某个工作负载涉及快速连续地启动完全相同的核函数时（例如处理一个工作队列，或以相同计算流式处理各批数据），可采用持久化核函数策略。持久化核函数只启动一次并保持活跃，在循环中复用线程处理许多个工作单元，而不是为每个单元都重新启动一个核函数。这种方式以更复杂的核函数设计换取显著更低的调度开销。通过让核函数保持存活，你避免了反复的启动成本，并能获得更高的持续占用率。高性能的分布式训练与推理系统常采用这一技巧，以最大化吞吐量并最小化迭代任务的延迟。

*Evaluate thread block clusters* Thread block clusters to keep data close and reduce relaunch overheads. Up to 16 thread blocks can form a cluster on Blackwell (after increasing the non-portable limit). Use cluster-aware synchronization and shared-memory residency to improve locality in persistent-style designs. Profile occupancy vs. residency trade-offs with kernel-level profiling tools like Nsight Compute.

*评估线程块簇* 用线程块簇（thread block cluster）来让数据保持邻近并减少重启开销。在 Blackwell 上，最多可有 16 个线程块组成一个簇（在提高不可移植上限之后）。在持久化风格的设计中，使用簇感知的同步与共享内存驻留来改善局部性。用 Nsight Compute 等核函数级剖析工具来权衡占用率与驻留之间的取舍。

## Arithmetic Optimizations and Reduced/Mixed Precision

## 算术优化与降低/混合精度

Lower precisions and sparsity let you trade bits for big speed and memory wins—often with negligible accuracy impact. Mixed precision, TF32/FP8/INT8, and fused scaling exploit hardware math paths to raise throughput per dollar.

更低的精度与稀疏性让你能用比特换取巨大的速度与内存收益——往往对准确率影响可忽略。混合精度、TF32/FP8/INT8 与融合缩放利用硬件数学通路来提升每美元吞吐量。

Specifically, use mixed precision (BF16/FP16) and Tensor Cores for big gains, adopt TF32 for easy FP32 speedups, and evaluate FP8/FP4 where quality allows. Exploit structured sparsity, lower-precision gradients/communications, and INT8/INT4 quantization for inference—fusing scales/activations to preserve accuracy. The following optimization techniques apply to improving the performance of arithmetic computations and utilizing reduced/mixed precision:

具体而言，使用混合精度（BF16/FP16）与 Tensor Core 获取大幅收益，采用 TF32 轻松加速 FP32，并在质量允许处评估 FP8/FP4。利用结构化稀疏性、更低精度的梯度/通信，以及面向推理的 INT8/INT4 量化（quantization）——通过融合缩放/激活来保住准确率。以下优化技术适用于改进算术计算的性能并利用降低/混合精度：

*Use mixed-precision training* Leverage FP16 or BF16 for training to speed up math operations and reduce memory usage. Modern GPUs have Tensor Cores that massively accelerate FP16/BF16 matrix operations. Keep critical parts like the final accumulation or a copy of weights in FP32 for numerical stability, but run bulk computations in half-precision. This often gives about a 1.5–3.5× speedup (depending on the model and kernel mix, with larger gains on matmul-heavy workloads) with minimal accuracy loss and is now standard in most frameworks with automatic mixed precision (AMP).

*使用混合精度训练* 借助 FP16 或 BF16 进行训练，以加速数学运算并减少内存占用。现代 GPU 拥有 Tensor Core，能大幅加速 FP16/BF16 矩阵运算。把最终累加或权重副本等关键部分保留在 FP32 以保证数值稳定性，但让大批量计算以半精度运行。这通常能带来约 1.5–3.5× 的加速（取决于模型与核函数组合，在矩阵乘密集的工作负载上收益更大），且准确率损失极小；如今借助自动混合精度（AMP），这已是大多数框架中的标准做法。

*Embrace gradient accumulation and activation checkpointing* Detail the use of gradient accumulation to effectively increase the batch size without extra memory usage, and consider activation checkpointing to reduce memory footprint in very deep networks. These techniques are crucial when training models that approach or exceed GPU memory limits.

*采用梯度累积与激活检查点* 详细运用梯度累积（gradient accumulation）来在不额外占用内存的情况下有效增大批大小，并考虑用激活检查点来降低极深网络中的内存占用。在训练接近或超出 GPU 内存上限的模型时，这些技术至关重要。

*Favor BF16 instead of FP16 on newer hardware* If available, use BF16 instead of FP16, as it has a larger exponent range and doesn’t require loss scaling. Modern GPUs support BF16 Tensor Cores at the same speed as FP16. BF16 will simplify training by avoiding overflow/underflow issues while still gaining the performance benefits of half precision.

*在较新硬件上优先使用 BF16 而非 FP16* 如果可用，请使用 BF16 而非 FP16，因为它的指数范围更大，且不需要损失缩放。现代 GPU 支持 BF16 Tensor Core，速度与 FP16 相同。BF16 通过避免上溢/下溢问题来简化训练，同时仍能获得半精度带来的性能优势。

*Exploit FP8, novel precisions, and scaling techniques* On modern GPUs, FP8 Tensor Cores provide roughly double the math throughput of FP16 or BF16 on compute-bound kernels while, at the same time, reducing activation and weight bandwidth. Additionally, FP4 (NVFP4) Tensor Cores double the throughput of FP8 and are used for inference with micro tensor scaling (an error-correction technique to maintain accuracy) to raise token throughput. For training, use FP8 with the NVIDIA Transformer Engine and maintain FP16 or FP32 accumulators when required. For inference, evaluate FP8 first and adopt NVFP4 only after calibration shows acceptable quality for your task. It’s recommended to use hybrid FP8 (E4M3 for forward activations/weights and E5M2 for gradients) for training. Specifically, consider using E4M3 for the forward pass (e.g., activations and weights) and E5M2 for the backward pass (e.g., gradients). It’s often beneficial to use a delayed scaling window of 256–1024. For inference, consider NVFP4 after calibration. TE integrates with PyTorch and is supported by modern GPU hardware. Prefer framework TE kernels over ad-hoc FP8 custom operations. End-to-end speedup depends on kernel mix, memory bandwidth, and calibration, so validate accuracy and performance on your model and workload.

*利用 FP8、新型精度与缩放技术* 在现代 GPU 上，FP8 Tensor Core 在计算受限的核函数上提供大约两倍于 FP16 或 BF16 的数学吞吐量，同时降低激活与权重的带宽。此外，FP4（NVFP4）Tensor Core 的吞吐量是 FP8 的两倍，并配合微张量缩放（一种维持准确率的纠错技术）用于推理，以提升 token 吞吐量。对于训练，配合 NVIDIA Transformer Engine 使用 FP8，并在需要时保留 FP16 或 FP32 累加器。对于推理，先评估 FP8，只有在校准显示对你的任务质量可接受后再采用 NVFP4。训练时建议使用混合 FP8（前向激活/权重用 E4M3，梯度用 E5M2）。具体而言，可考虑前向传播（如激活与权重）用 E4M3，反向传播（如梯度）用 E5M2。使用 256–1024 的延迟缩放窗口通常是有益的。对于推理，可在校准后考虑 NVFP4。TE 与 PyTorch 集成，并受现代 GPU 硬件支持。优先使用框架内的 TE 核函数，而非临时拼凑的 FP8 自定义算子。端到端加速取决于核函数组合、内存带宽与校准，因此请在你的模型与工作负载上验证准确率与性能。

*Leverage Tensor Cores and the Tensor Memory Accelerator (TMA)* Make sure your custom CUDA kernels utilize Tensor Cores for matrix ops if possible. This might involve using CUTLASS templates for simplicity. By using Tensor Cores and TMA for asynchronous tensor movement to shared memory, you can achieve dramatic speedups for GEMM, convolutions, and other tensor operations—often reaching near-peak FLOPS of the GPU. Ensure your data is in FP16/BF16/TF32 as needed and aligned to Tensor Core tile dimensions, which are multiples of 8 or 16.

*利用 Tensor Core 与张量内存加速器（TMA）* 尽可能确保你的自定义 CUDA 核函数在矩阵运算中利用 Tensor Core。这可能涉及为简便起见使用 CUTLASS 模板。通过使用 Tensor Core 以及用于向共享内存异步搬运张量的 TMA，你能为 GEMM、卷积及其他张量运算实现显著加速——往往能接近 GPU 的峰值 FLOPS。确保你的数据按需处于 FP16/BF16/TF32，并对齐到 Tensor Core 的分块维度（8 或 16 的倍数）。

*Use TF32 for easy speedup* For 32-bit matrix multiplies, set torch.set_float32_matmul_precision("high") to enable TF32 (fast FP32) for operations that are numerically safe in PyTorch. Libraries like cuBLAS and cuDNN will automatically pick optimal Tensor Core code paths on modern GPU hardware. If you force full-precision FP32 with “highest” (instead of “high”), make sure to understand the performance impact.

*使用 TF32 轻松加速* 对于 32 位矩阵乘法，设置 torch.set_float32_matmul_precision("high") 以为 PyTorch 中数值上安全的运算启用 TF32（快速 FP32）。cuBLAS 与 cuDNN 等库会在现代 GPU 硬件上自动挑选最优的 Tensor Core 代码路径。如果你用 “highest”（而非 “high”）强制全精度 FP32，务必理解其对性能的影响。

*Exploit structured sparsity* Modern NVIDIA GPUs support 2:4 structured sparsity in matrix multiply, which zeros out 50% of weights in a structured pattern. This allows the hardware to double its throughput. Leverage this by pruning your model. If you can prune weights to meet the 2:4 sparsity pattern, your GEMMs can run ~2× faster for those layers. Use NVIDIA’s SDK or library support to apply structured sparsity and ensure the sparse Tensor Core paths are used. This can give a free speed boost if your model can tolerate or be trained with that sparsity, which often requires retraining with sparsity regularization.

*利用结构化稀疏性* 现代 NVIDIA GPU 在矩阵乘法中支持 2:4 结构化稀疏性，它以结构化模式将 50% 的权重清零。这让硬件的吞吐量翻倍。通过对模型剪枝来利用这一点。如果你能把权重剪枝到符合 2:4 稀疏模式，这些层的 GEMM 就能快约 2×。使用 NVIDIA 的 SDK 或库支持来施加结构化稀疏性，并确保用上了稀疏 Tensor Core 通路。如果你的模型能容忍这种稀疏性、或能以此训练，这就能带来免费的速度提升；这通常需要用稀疏性正则化重新训练。

*Reduce precision for gradients and activations when possible* Even if you keep weights at higher precision, consider compressing gradients or activations to lower precision. For instance, use FP16/BF16 or FP8 communication for gradients. Many frameworks support FP16 gradient all-reduce. Similarly, for activation checkpointing, storing activations in 16-bit instead of FP32 saves memory. Research continues on FP8 and FP4 optimizers and quantized gradients. These help maintain model quality while reducing memory and bandwidth costs. In bandwidth-limited environments, gradient compression in particular can be a game changer. DeepSeek demonstrated this by compressing gradients to train on constrained GPUs.

*在可能时降低梯度与激活的精度* 即便你把权重保持在更高精度，也可考虑把梯度或激活压缩到更低精度。例如，对梯度使用 FP16/BF16 或 FP8 通信。许多框架支持 FP16 梯度 all-reduce。类似地，对于激活检查点，以 16 位而非 FP32 存储激活能节省内存。关于 FP8 与 FP4 优化器及量化梯度的研究仍在继续。这些做法有助于在降低内存与带宽成本的同时保住模型质量。在带宽受限的环境中，梯度压缩尤其可能带来质变。DeepSeek 就通过压缩梯度在受限的 GPU 上训练，证明了这一点。

*Use custom quantization for inference* For deployment, use INT8 quantization wherever possible. INT8 inference on GPUs is extremely fast and memory-efficient. Use NVIDIA’s TensorRT or quantization tools to quantize models to INT8 and calibrate them. Many neural networks like transformers can run in INT8 with a negligible accuracy drop. The speedups can be 2–4× over FP16. On the newest GPUs, also explore and evaluate FP8 or INT4 for certain models to further boost throughput for inference.

*为推理使用自定义量化* 部署时，尽可能使用 INT8 量化。GPU 上的 INT8 推理极快且内存高效。使用 NVIDIA 的 TensorRT 或量化工具把模型量化到 INT8 并校准。许多神经网络（如 transformer）可以在准确率损失可忽略的情况下以 INT8 运行。相比 FP16，加速可达 2–4×。在最新的 GPU 上，还可为某些模型探索并评估 FP8 或 INT4，以进一步提升推理吞吐量。

*Fuse scaling and computing operations when possible* When using lower precision, remember to fuse operations to retain accuracy. For example, Blackwell’s FP4 “microscaling” suggests keeping a scale per group of values. Incorporate these fused operations by scaling and computing in one pass—rather than using separate passes, which could cause precision loss. Many of these are handled by existing libraries, so just use them rather than implementing them from scratch.

*在可能时融合缩放与计算操作* 使用更低精度时，记得融合运算以保住准确率。例如，Blackwell 的 FP4 “微缩放（microscaling）”建议为每一组数值各保留一个缩放因子。把这些融合运算纳入其中，在一趟中完成缩放与计算——而不是分成多趟，那样可能导致精度损失。其中许多都已由现有库处理，所以直接用它们即可，不必从零实现。

## Advanced Tuning Strategies and Algorithmic Tricks

## 高级调优策略与算法技巧

Algorithmic shifts routinely beat hardware upgrades on ROI by reducing work rather than pushing it faster. Autotuning, FlashAttention, overlap of comm/compute, and sharding unlock scale while cutting waste.

在 ROI 上，算法层面的改变常常胜过硬件升级，因为它减少的是工作量本身，而非把工作做得更快。自动调优、FlashAttention、通信/计算重叠与分片，能在削减浪费的同时释放规模化能力。

Specifically, autotune kernel and layer parameters, swap in fused/FlashAttention kernels, and overlap communication with computation in distributed training. Scale deep models with pipeline/tensor parallelism and ZeRO sharding, and consider asynchronous updates or pruning/sparsity to trade a little accuracy work for big throughput wins. The following are some advanced performance optimizations and algorithmic tricks:

具体而言，对核函数与层参数做自动调优，换入融合/FlashAttention 核函数，并在分布式训练中让通信与计算重叠。用流水线/张量并行与 ZeRO 分片来扩展深层模型，并考虑异步更新或剪枝/稀疏化，以少量准确率上的调整换取吞吐量的巨大提升。以下是一些高级性能优化与算法技巧：

*Autotune kernel parameters* Autotune your custom CUDA kernels for the target GPU. Choosing the correct block size, tile size, unroll factors, etc., can affect performance, and the optimal settings often differ between GPUs’ generations, such as Ampere, Hopper, Blackwell, and beyond. Use autotuning scripts or frameworks like OpenAI Triton—or even brute-force search in a preprocessing step—to find the best launch config. This can easily yield 20%–30% improvements that you’d miss with static “reasonable” settings. Use Triton features in your autotuning loop—for instance, set num_warps and num_stages, enable automatic warp specialization, and test asynchronous TMA layouts. Prefer tensor map descriptor APIs for shared-memory staging. Re-benchmark tile shapes when migrating to different hardware, as optimal choices will differ across GPU generations.

*自动调优核函数参数* 针对目标 GPU 自动调优你的自定义 CUDA 核函数。选择正确的块大小、分块大小、展开因子等都会影响性能，而最优设置往往在不同代 GPU（如 Ampere、Hopper、Blackwell 及更新代）之间各不相同。使用自动调优脚本或 OpenAI Triton 等框架——甚至在预处理阶段做暴力搜索——来找到最佳启动配置。这轻松就能带来 20%–30% 的提升，而静态的“合理”设置往往会与之失之交臂。在你的自动调优循环中使用 Triton 特性——例如设置 num_warps 与 num_stages、启用自动 warp 特化、测试异步 TMA 布局。优先使用张量映射描述符 API 来做共享内存暂存。迁移到不同硬件时要重新基准测试分块形状，因为最优选择会随 GPU 代际而不同。

*Use kernel fusion in ML workloads* Utilize fused kernels provided by deep learning libraries. For example, enabling fused optimizers will fuse elementwise ops like weight update, momentum, etc. This will also use fused multihead attention implementations and fused normalization kernels. NVIDIA’s libraries and some open source projects like Transformer Engine and FasterTransformer provide fused operations for common patterns, such as fused LayerNorm + dropout. These reduce launch overhead and use memory more efficiently.

*在 ML 工作负载中使用核函数融合* 利用深度学习库提供的融合核函数。例如，启用融合优化器会融合权重更新、动量等逐元素运算。这还会用到融合的多头注意力实现与融合的归一化核函数。NVIDIA 的库以及一些开源项目（如 Transformer Engine 与 FasterTransformer）为常见模式提供融合运算，例如融合的 LayerNorm + dropout。这些能减少启动开销并更高效地使用内存。

*Utilize memory-efficient attention like FlashAttention* Integrate advanced algorithms like FlashAttention for transformer models. FlashAttention computes attention in a tiled, streaming fashion to avoid materializing large intermediate matrices, drastically reducing memory usage and increasing speed—especially for long sequences. Replacing the standard attention with FlashAttention can improve both throughput and memory footprint, allowing larger batch sizes or sequence lengths on the same hardware.

*利用 FlashAttention 等内存高效注意力* 为 transformer 模型集成 FlashAttention 等先进算法。FlashAttention 以分块、流式的方式计算注意力，避免物化大型中间矩阵，从而大幅降低内存占用并提升速度——尤其是对长序列。用 FlashAttention 替换标准注意力可同时改善吞吐量与内存占用，让相同硬件上能用更大的批大小或序列长度。

*Overlap communication and computation* In distributed training, overlap network communication with GPU computation whenever possible. For example, with gradient all-reduce, launch the all-reduce asynchronously as soon as each layer’s gradients are ready, while the next layer is still computing the backward pass. This pipelining can hide all-reduce latency entirely if done right. Use asynchronous NCCL calls or framework libraries like PyTorch’s Distributed Data Parallel (DDP), which provide overlapping out of the box. This ensures the GPU isn’t idle waiting for the network.

*重叠通信与计算* 在分布式训练中，尽可能让网络通信与 GPU 计算重叠。例如，对于梯度 all-reduce，一旦每层的梯度就绪就异步启动 all-reduce，而下一层仍在计算反向传播。若处理得当，这种流水线化能完全隐藏 all-reduce 的延迟。使用异步 NCCL 调用，或使用 PyTorch 的分布式数据并行（Distributed Data Parallel，DDP）等框架库，它们开箱即提供重叠。这能确保 GPU 不会空等网络。

*Use pipeline parallelism for deep models* When model size forces you to pipeline across GPUs using tensor parallelism or pipeline parallelism, you can use enough microbatches to keep all pipeline stages busy. Exploit NVLink/NVSwitch to send activations quickly between stages. Overlap and reduce pipeline bubbles by using an interleaved schedule. Some frameworks automate this type of scheduling. The NVL72 fabric is especially helpful here, as even communication-heavy pipeline stages can exchange data at multiterabyte speeds, minimizing pipeline stalls.

*对深层模型使用流水线并行* 当模型规模迫使你用张量并行或流水线并行跨 GPU 切分时，可以用足够多的微批来让所有流水线阶段保持忙碌。利用 NVLink/NVSwitch 在各阶段之间快速传送激活。通过交错式调度来重叠并减少流水线气泡。一些框架会自动化这类调度。NVL72 网络结构在这里尤其有用，因为即便是通信密集的流水线阶段，也能以多太字节的速度交换数据，从而最小化流水线停顿。

*Utilize distributed optimizer sharding* Use a memory-saving optimization strategy like Zero Redundancy Optimizer (ZeRO), which shards tensors like optimizer states and gradients across GPUs instead of replicating them. This allows scaling to extreme model sizes by distributing the memory and communication load. It improves throughput by reducing per-GPU memory pressure, avoiding swapping to CPU, and reducing communication volume if done in chunks. Many frameworks like DeepSpeed and Megatron-LM provide this type of sharding. Leverage it for large models to maintain high speed without running OOM or hitting slowdown from swapping.

*利用分布式优化器分片* 使用像零冗余优化器（Zero Redundancy Optimizer，ZeRO）这样的省内存优化策略，它把优化器状态与梯度等张量跨 GPU 分片，而不是复制它们。这通过分摊内存与通信负载，让你能扩展到极大的模型规模。它通过降低每 GPU 的内存压力、避免换出到 CPU、以及在分块进行时减少通信量，来提升吞吐量。DeepSpeed 与 Megatron-LM 等许多框架都提供这类分片。对大模型加以利用，就能在不触发 OOM 或因换出而变慢的情况下保持高速。

*Train asynchronously when possible* If applicable, consider asynchronous updates. For example, you can use stale stochastic gradient descent (SGD) in which workers don’t always wait for one another to share updates. This approach can increase throughput, though it may require careful tuning to not impact convergence. Asynchronous training can provide large performance benefits if done properly.

*在可能时异步训练* 若适用，可考虑异步更新。例如，你可以使用陈旧的随机梯度下降（stochastic gradient descent，SGD），其中各 worker 并不总是互相等待以共享更新。这种方式能提升吞吐量，不过可能需要仔细调优以免影响收敛。异步训练若处理得当，可带来巨大的性能收益。

*Incorporate sparsity and pruning* Large models often have redundancy. Use pruning techniques during training to introduce sparsity, which you can exploit at inference—and partially during training if supported. Modern GPU hardware supports accelerated sparse matrix multiply (2:4), and future GPUs will likely extend this feature. Even if you leave training as dense and prune only for inference, a smaller model will run faster and use less memory. This increases cost-efficiency for model deployments. Explore the lottery ticket hypothesis, distillation, or structured pruning to maintain accuracy while trimming model size.

*引入稀疏性与剪枝* 大模型往往存在冗余。在训练期间使用剪枝技术引入稀疏性，你可以在推理时加以利用——如果支持，也可在训练中部分利用。现代 GPU 硬件支持加速的稀疏矩阵乘法（2:4），未来的 GPU 很可能扩展这一特性。即便你让训练保持稠密、仅为推理剪枝，更小的模型也会运行更快、占用更少内存。这提升了模型部署的成本效益。可探索彩票假说（lottery ticket hypothesis）、蒸馏或结构化剪枝，在缩减模型大小的同时保住准确率。

## Distributed Training and Network Optimization

## 分布式训练与网络优化

At cluster scale, the network becomes the limiter. Untreated, the network can break linear scaling and inflate costs. RDMA/Jumbo frames, hierarchical collectives, affinity, and compression protect bandwidth and tame latency.

在集群规模下，网络会成为瓶颈。若不加处理，网络会打破线性扩展并推高成本。RDMA/巨型帧、分层集合通信、亲和性与压缩能守护带宽并驯服延迟。

Use RDMA (InfiniBand/RoCE) when available; if on Ethernet, tune TCP buffers, enable jumbo frames, and select modern congestion control. Align NIC/CPU affinity, adjust NCCL threads/buffers (and SHARP/CollNet where supported), compress or accumulate gradients, and test the fabric to catch loss or misconfigurations. Follow this guidance to optimize your network for distributed environments such as multi-GPU and multinode model training:

在可用时使用 RDMA（InfiniBand/RoCE）；如果用以太网，则调优 TCP 缓冲区、启用巨型帧（jumbo frames）、并选择现代拥塞控制。对齐 NIC/CPU 亲和性，调整 NCCL 线程/缓冲区（以及在支持处的 SHARP/CollNet），压缩或累积梯度，并测试网络结构以捕捉丢包或配置错误。遵循以下指导，为多 GPU 与多节点模型训练等分布式环境优化你的网络：

*Use RDMA networking when available* Equip your multinode cluster with InfiniBand or RoCE for low latency and high throughput. Ensure NCCL and MPI are using RDMA for training. NCCL will autodetect InfiniBand and use GPUDirect RDMA if available. RDMA bypasses the kernel networking stack and can reduce latency significantly versus traditional TCP. If you only have Ethernet, enable RoCE on RDMA-capable NICs to get RDMA-like performance. On NVLink domain systems (NVL72, GB200/GB300, etc.), keep collectives on-fabric when possible. Reserve host networking for inter-island links. Align NCCL topology hints with your NVLink/NVSwitch domains.

*在可用时使用 RDMA 网络* 为你的多节点集群配备 InfiniBand 或 RoCE，以获得低延迟与高吞吐。确保 NCCL 与 MPI 在训练中使用 RDMA。NCCL 会自动检测 InfiniBand，并在可用时使用 GPUDirect RDMA。RDMA 绕过内核网络栈，相比传统 TCP 可显著降低延迟。如果你只有以太网，在支持 RDMA 的 NIC 上启用 RoCE，即可获得类 RDMA 的性能。在 NVLink 域系统（NVL72、GB200/GB300 等）上，尽可能让集合通信保持在网络结构内进行。把主机网络保留给孤岛间链路。让 NCCL 的拓扑提示与你的 NVLink/NVSwitch 域对齐。

*Tune the TCP/IP stack if using Ethernet* For TCP-based clusters, increase network buffer sizes. Raise /proc/sys/net/core/{r,w}mem_max and the autotuning limits (net.ipv4.tcp_{r,w}mem) to allow larger send/receive buffers. This helps saturate 10/40/100 GbE links. Enable jumbo frames (MTU 9000) on all nodes and switches to reduce overhead per packet, which improves throughput and reduces CPU usage. Also consider modern TCP congestion control like BBR for wide-area or congested networks.

*若使用以太网则调优 TCP/IP 栈* 对于基于 TCP 的集群，增大网络缓冲区。提高 /proc/sys/net/core/{r,w}mem_max 以及自动调优上限（net.ipv4.tcp_{r,w}mem），以允许更大的发送/接收缓冲区。这有助于跑满 10/40/100 GbE 链路。在所有节点与交换机上启用巨型帧（MTU 9000），以减少每个数据包的开销，从而提升吞吐量并降低 CPU 占用。对于广域或拥塞网络，也可考虑 BBR 等现代 TCP 拥塞控制。

*Assign CPU affinity for NICs* Pin network interrupts and threads to the CPU core(s) on the same NUMA node as the NIC. This avoids cross-NUMA penalties for network traffic and keeps the networking stack’s memory accesses local. Check /proc/interrupts and use irqaffinity settings to ensure, for example, your NIC in NUMA node 0 is handled by a core in NUMA node 0. This can improve network performance and consistency, especially under high packet rates.

*为 NIC 分配 CPU 亲和性* 把网络中断与线程绑定到与 NIC 同一 NUMA 节点上的 CPU 核心。这避免了网络流量的跨 NUMA 惩罚，并让网络栈的内存访问保持本地。检查 /proc/interrupts 并使用 irqaffinity 设置，例如确保处于 NUMA 节点 0 的 NIC 由 NUMA 节点 0 中的某个核心处理。这能改善网络性能与一致性，尤其是在高包速率下。

*Optimize NCCL environment variables for your environment* Experiment with NCCL parameters for large multinode jobs. For example, increase NCCL_NTHREADS, the number of CPU threads per GPU for NCCL, from the default 4 to 8 or 16 to drive higher bandwidth at the cost of more CPU usage.

*为你的环境优化 NCCL 环境变量* 对大型多节点作业调试 NCCL 参数。例如，把 NCCL_NTHREADS（每个 GPU 用于 NCCL 的 CPU 线程数）从默认的 4 提高到 8 或 16，以更高 CPU 占用为代价驱动更高带宽。

Increase NCCL_BUFFSIZE, the buffer size per GPU, from the default 1 MB to 4 MB or more for better throughput on large messages. If your cluster uses SHARP-capable switches, install the NCCL SHARP plugin and enable CollNet by setting NCCL_COLLNET_ENABLE=1, then use the SHARP plugin variables such as SHARP_COLL_LOCK_ON_COMM_INIT=1 and SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD=0 as documented. Expect speedups only when your reductions are large enough and the network fabric supports SHARP offload.

把 NCCL_BUFFSIZE（每个 GPU 的缓冲区大小）从默认的 1 MB 提高到 4 MB 或更大，以在大消息上获得更好的吞吐量。如果你的集群使用支持 SHARP 的交换机，安装 NCCL SHARP 插件并通过设置 NCCL_COLLNET_ENABLE=1 启用 CollNet，然后按文档使用 SHARP 插件变量，例如 SHARP_COLL_LOCK_ON_COMM_INIT=1 与 SHARP_COLL_NUM_COLL_GROUP_RESOURCE_ALLOC_THRESHOLD=0。只有当你的归约足够大、且网络结构支持 SHARP 卸载时，才应期待加速。

*Use gradient accumulation for slow networks* If your network becomes the bottleneck because you are scaling too many nodes linked by a moderate-performance interconnect, use gradient accumulation to perform fewer, larger all-reduce operations. Accumulate gradients over a few minibatches before syncing so that you communicate once for *N* batches instead of every batch. This trades a bit of extra memory and some model accuracy tuning for significantly reduced network overhead. It’s especially helpful when adding more GPUs yields diminishing returns due to communication costs.

*在慢速网络上使用梯度累积* 如果因为把过多节点通过中等性能的互连连接在一起而使网络成为瓶颈，可用梯度累积来执行更少、更大的 all-reduce 操作。在同步之前，先在若干个 minibatch 上累积梯度，这样你就每 *N* 个批通信一次，而不是每个批都通信。这以少量额外内存与一些模型准确率调优为代价，显著降低网络开销。当因通信成本导致增加更多 GPU 收益递减时，这尤其有用。

*Optimize all-reduce topologies* Ensure you’re using the optimal all-reduce algorithm for your cluster topology. NCCL will choose ring or tree algorithms automatically, but on mixed interconnects like GPUs connected by NVLink on each node and InfiniBand or Ethernet between nodes, hierarchical all-reduce can be beneficial. Hierarchical all-reduce will first perform the all-reduce operation within the node, then it will proceed across nodes. Most frameworks will perform NCCL-based hierarchical aggregations by default but verify by profiling. In traditional MPI setups, you may consider manually doing this same two-level reduction—first intranode and then internode.

*优化 all-reduce 拓扑* 确保你为集群拓扑使用了最优的 all-reduce 算法。NCCL 会自动选择环形或树形算法，但在混合互连（如每个节点内 GPU 通过 NVLink 相连、节点间用 InfiniBand 或以太网）上，分层 all-reduce 可能更有利。分层 all-reduce 会先在节点内执行 all-reduce 操作，然后再跨节点进行。大多数框架默认会执行基于 NCCL 的分层聚合，但请通过剖析加以验证。在传统 MPI 设置中，你可以考虑手工做同样的两级归约——先节点内、再节点间。

*Avoid network oversubscription* On multi-GPU servers, ensure the combined traffic of GPUs doesn’t oversubscribe the NIC. For example, eight GPUs can easily generate more than 200 Gbps of traffic during all-reduce, so having only a single 100 Gbps NIC will constrain you. Consider multiple NICs per node and 200/400 Gbps InfiniBand if scaling to many GPUs per node. Likewise, watch out for PCIe bandwidth limits if your NIC and GPUs share the same PCIe root complex.

*避免网络超额订阅* 在多 GPU 服务器上，确保各 GPU 的合并流量不会超额订阅 NIC。例如，八块 GPU 在 all-reduce 期间轻松就能产生超过 200 Gbps 的流量，因此仅有单块 100 Gbps 的 NIC 会成为掣肘。如果每节点扩展到很多 GPU，可考虑每节点配备多块 NIC 以及 200/400 Gbps 的 InfiniBand。同样地，如果你的 NIC 与 GPU 共享同一个 PCIe 根复合体，要留意 PCIe 带宽上限。

*Compress communication* Just as with single-node memory, consider compressing data for network transfer. Techniques include 16-bit or 8-bit gradient compression, quantizing activations for cross-node pipeline transfers, or even more exotic methods like sketching. If your network is the slowest component, a slightly higher compute cost to compress/decompress data can be worth it. NVIDIA’s NCCL doesn’t natively compress, but you can integrate compression in frameworks (e.g., gradient compression in Horovod or custom AllReduce hooks in PyTorch). This was one key to DeepSeek’s success—compressing gradients to cope with limited internode bandwidth.

*压缩通信* 就像单节点内存那样，可考虑为网络传输压缩数据。技术包括 16 位或 8 位梯度压缩、为跨节点流水线传输量化激活，甚至像草图（sketching）这样更奇特的方法。如果网络是最慢的环节，那么为压缩/解压数据略微增加一点计算成本可能是值得的。NVIDIA 的 NCCL 本身不做压缩，但你可以在框架中集成压缩（例如 Horovod 中的梯度压缩，或 PyTorch 中的自定义 AllReduce 钩子）。这是 DeepSeek 成功的关键之一——通过压缩梯度来应对受限的节点间带宽。

*Monitor network health* Ensure no silent issues are hampering your distributed training. Check for packet loss (which would show up as retries or timeouts—on InfiniBand, use counters for resend, and on Ethernet, check for TCP retransmits). Even a small packet loss can severely degrade throughput due to congestion control kicking in. Use out-of-band network tests (like iPerf or NCCL tests) to validate you’re getting expected bandwidth and latency. If not, investigate switch configurations, NIC firmware, or CPU affinity.

*监控网络健康* 确保没有静默的问题在拖累你的分布式训练。检查丢包（它会表现为重试或超时——在 InfiniBand 上使用重发计数器，在以太网上检查 TCP 重传）。由于拥塞控制会随之启动，即使是很小的丢包也会严重降低吞吐量。使用带外网络测试（如 iPerf 或 NCCL tests）来验证你获得了预期的带宽与延迟。如果没有，就排查交换机配置、NIC 固件或 CPU 亲和性。

## Efficient Inference and Serving

## 高效推理与服务

Serving is a cost-and-latency game—utilization rises through orchestration and batching, not just bigger GPUs. Specialized runtimes, KV cache strategies, and warmups keep throughput high without violating SLOs.

服务是一场关乎成本与延迟的博弈——利用率的提升靠的是编排与批处理，而不只是更大的 GPU。专用运行时、KV 缓存策略与预热能在不违反 SLO 的前提下保持高吞吐。

Orchestrate for demand with autoscaling, microservices, and dynamic/continuous batching to keep GPUs hot without violating latency SLOs. Use specialized runtimes (vLLM, SGLang, TensorRT-LLM), exploit NIXL and KV cache offloading for disaggregated serving, warm models, and isolate resources to control tail latency. Follow these techniques to improve model inference efficiency and performance:

用自动伸缩、微服务与动态/连续批处理来按需编排，让 GPU 保持火热而不违反延迟 SLO。使用专用运行时（vLLM、SGLang、TensorRT-LLM），利用 NIXL 与 KV 缓存卸载做分离式服务，预热模型，并隔离资源以控制尾延迟。遵循以下技术来提升模型推理效率与性能：

*Orchestrate dynamic resources efficiently* Integrate advanced container orchestration platforms, such as Kubernetes augmented with custom performance metrics. This enables dynamic scaling and balancing workloads based on live usage patterns and throughput targets.

*高效编排动态资源* 集成先进的容器编排平台，例如用自定义性能指标增强的 Kubernetes。这能基于实时使用模式与吞吐量目标来动态扩缩并均衡工作负载。

*Embrace serverless architectures for inference* Explore serverless architectures and microservice designs for inference workloads, which can handle bursty traffic efficiently and reduce idle resource overhead by scaling down when demand is low.

*为推理拥抱无服务器架构* 为推理工作负载探索无服务器（serverless）架构与微服务设计，它们能高效应对突发流量，并通过在需求低时缩容来减少空闲资源开销。

*Optimize batch and concurrency* For inference workloads, find the right batching strategy. For inference workloads, favor dynamic or continuous batching to automatically batch incoming requests. Larger batch sizes improve throughput by keeping the GPU busy, but too large can add latency. Also, run multiple inference streams in parallel if one stream doesn’t use all GPU resources—e.g., two concurrent inference batches to use both GPU SMs and Tensor Cores fully.

*优化批大小与并发* 对于推理工作负载，找到合适的批处理策略。对于推理工作负载，优先使用动态或连续批处理来自动地对到来的请求成批。更大的批大小通过让 GPU 保持忙碌来提升吞吐，但过大会增加延迟。此外，如果单个流用不完全部 GPU 资源，可并行运行多个推理流——例如用两个并发的推理批来充分利用 GPU 的 SM 与 Tensor Core。

*Leverage NIXL for distributed inference* When serving large models across GPUs or nodes, use the NVIDIA Inference Xfer Library to stream KV cache between prefill and decode workers over RDMA. In the case of NIXL, the large transformer-based KV cache is transferred between nodes. NIXL provides a high-throughput, low-latency API for streaming the KV cache from a prefill GPU to a decode GPU in a disaggregated LLM inference cluster. It does this using GPUDirect RDMA and optimal paths—and without involving the CPU. This reduces tail latency for disaggregated prefill decode serving across nodes.

*利用 NIXL 做分布式推理* 当跨 GPU 或跨节点服务大模型时，使用 NVIDIA Inference Xfer Library（NIXL）通过 RDMA 在 prefill 与 decode 工作节点之间流式传输 KV 缓存。在 NIXL 的场景中，基于 transformer 的大型 KV 缓存在节点之间传输。NIXL 提供了一个高吞吐、低延迟的 API，用于在分离式 LLM 推理集群中把 KV 缓存从 prefill GPU 流式传输到 decode GPU。它借助 GPUDirect RDMA 与最优路径来完成——且无需 CPU 介入。这降低了跨节点分离式 prefill decode 服务的尾延迟。

*Offload KV cache if necessary* If an LLM’s attention KV cache grows beyond GPU memory, use hierarchical offloading. NVIDIA Dynamo’s Distributed KV Cache Manager offloads less frequently accessed KV pages to CPU memory, SSD, or networked storage, while inference engines like TensorRT-LLM and vLLM support paged and quantized KV caches. Reuse caches to lower memory pressure and first-token latency. Validate end-to-end impact because offloaded misses introduce extra I/O latency. This allows inference on sequences that would otherwise exceed GPU memory—and with minimal performance hit thanks to fast NVMe and compute-I/O overlapping. Ensure your inference server is configured to use this if you expect very long prompts or chats. Offloading to disk is better than failing completely.

*必要时卸载 KV 缓存* 如果 LLM 的注意力 KV 缓存增长到超出 GPU 内存，使用分层卸载。NVIDIA Dynamo 的分布式 KV 缓存管理器会把访问频率较低的 KV 页卸载到 CPU 内存、SSD 或联网存储，而 TensorRT-LLM 与 vLLM 等推理引擎则支持分页与量化的 KV 缓存。复用缓存以降低内存压力与首 token 延迟。要验证端到端影响，因为被卸载后的未命中会引入额外的 I/O 延迟。这让原本会超出 GPU 内存的序列也能进行推理——并且得益于快速 NVMe 与计算-I/O 重叠，性能损失极小。如果你预期会有很长的提示词或对话，确保你的推理服务器配置为使用这一功能。卸载到磁盘总比彻底失败要好。

*Serve models efficiently* Use optimized model inference systems, such as vLLM, SGLang, NVIDIA Dynamo, and NVIDIA TensorRT-LLM for serving large models with low latency and high throughput. They should implement quantization, low-precision formats, fusion, highly optimized attention kernels, and other tricks to maximize GPU utilization during inference. These libraries should also handle tensor parallelism, pipeline parallelism, expert parallelism, context parallelism, speculative decoding, chunked prefill, disaggregated prefill/decode, and dynamic request batching—among many other high-performance features.

*高效地服务模型* 使用优化的模型推理系统，例如 vLLM、SGLang、NVIDIA Dynamo 与 NVIDIA TensorRT-LLM，以低延迟、高吞吐服务大模型。它们应实现量化、低精度格式、融合、高度优化的注意力核函数以及其他技巧，以在推理期间最大化 GPU 利用率。这些库还应处理张量并行、流水线并行、专家并行、上下文并行、推测解码（speculative decoding）、分块 prefill、分离式 prefill/decode 以及动态请求批处理——以及许多其他高性能特性。

*Monitor and tune for tail latency* In real-time services, both average latency and (long-)tail latency (99th percentile) matter. Profile the distribution of inference latencies. If the tail is high, identify outlier causes, such as unexpected CPU involvement, garbage-collection (GC) pauses, or excessive context switches. Pin your inference server process to specific cores, isolate it from noisy neighbors, and use real-time scheduling if necessary to get more consistent latency.

*针对尾延迟进行监控与调优* 在实时服务中，平均延迟与（长）尾延迟（第 99 百分位）都很重要。剖析推理延迟的分布。如果尾部偏高，找出离群成因，例如意料之外的 CPU 介入、垃圾回收（garbage collection，GC）停顿或过多的上下文切换。把你的推理服务器进程绑定到特定核心，将其与吵闹的邻居隔离，并在必要时使用实时调度以获得更一致的延迟。

*Warm up to avoid cold-start latency* Warm up the GPUs by loading the model into the GPU and running a few dummy inferences. This will avoid one-time, cold-start latency hits when the first real request comes into the inference server.

*预热以避免冷启动延迟* 通过把模型加载到 GPU 并运行几次虚拟推理来预热 GPU。这样当第一个真实请求到达推理服务器时，就能避免一次性的冷启动延迟冲击。

*Partition resources efficiently for quality of service (QoS)* If running mixed, heterogeneous workloads, such as training and inference—or models with different architectures—on the same infrastructure, consider partitioning resources to ensure the latency-sensitive inference gets priority. This could mean dedicating some GPUs entirely to inference or using MIG to give an inference service a guaranteed slice of a GPU if it doesn’t need a full GPU but requires predictable latency. Separate inference from training on different nodes if possible, as training can introduce jitter with heavy I/O or sudden bursts of communication.

*为服务质量（quality of service，QoS）高效地划分资源* 如果在同一套基础设施上运行混合的异构工作负载（例如训练与推理——或不同架构的模型），可考虑划分资源以确保延迟敏感的推理获得优先级。这可能意味着把一些 GPU 完全专用于推理，或者在推理服务不需要整块 GPU 但要求可预测延迟时，用 MIG 为它划出一块有保障的 GPU 切片。如果可能，把推理与训练分到不同节点上，因为训练会以繁重的 I/O 或突发的通信引入抖动。

*Utilize Grace CPU for inference preprocessing* In Grace Blackwell systems, the server-class CPU can handle preprocessing—such as tokenization and batch collation—extremely fast in the same memory space as the GPU. Offload such tasks to the CPU and have it prepare data in the shared memory that the GPU can directly use. This reduces duplication of buffers and leverages the powerful CPU to handle parts of the inference pipeline, freeing the GPU to focus on more compute-intensive neural-network computations.

*利用 Grace CPU 做推理预处理* 在 Grace Blackwell 系统中，服务器级 CPU 能在与 GPU 相同的内存空间中极快地处理预处理——例如分词与批归并。把这类任务卸载给 CPU，让它在 GPU 可直接使用的共享内存中准备好数据。这减少了缓冲区的重复，并借助强大的 CPU 来承担推理流水线的部分工作，从而解放 GPU 去专注于更计算密集的神经网络计算。

*Tune carefully for edge AI and latency-critical deployments* Extend performance tuning to the edge by leveraging specialized edge accelerators and optimizing data transfer protocols between central servers and edge devices. This will help achieve ultralow latency for time-sensitive applications.

*为边缘 AI 与延迟关键型部署精心调优* 通过利用专用的边缘加速器并优化中心服务器与边缘设备之间的数据传输协议，把性能调优延伸到边缘。这将有助于为时间敏感的应用实现超低延迟。

## Multinode Inference and Serving

## 多节点推理与服务

Disaggregating prefill/decode and sharding models lets you handle bigger contexts and more users with higher occupancy. Continuous batching and hierarchical memory/offload maintain flow even under long prompts and heavy concurrency.

把 prefill/decode 分离、并对模型分片，能让你以更高的占用率处理更大的上下文与更多用户。连续批处理与分层内存/卸载即便在长提示词与高并发下也能维持流动。

Specifically, disaggregate prefill and decode across devices, continuously pool tokens across requests, and shard oversized models via tensor/pipeline parallelism. Add hierarchical memory/offload for very long contexts so you serve more without OOMs, trading small latency for much higher capacity. The following performance tips apply to multinode inference and serving:

具体而言，把 prefill 与 decode 跨设备分离，跨请求持续地对 token 汇池，并通过张量/流水线并行对超大模型分片。为极长的上下文加上分层内存/卸载，这样你就能在不触发 OOM 的情况下服务更多请求，以少量延迟换取高得多的容量。以下性能建议适用于多节点推理与服务：

*Disaggregate inference pipelines* Separate the inference workflow into distinct phases, including the “prefill” phase that processes the input prompt through all model layers, and the iterative “decode” phase that generates outputs token by token. Allocate these phases to different resources to allow for independent scaling. This two-stage approach prevents faster tasks from being bottlenecked by slower ones. For large language models, one strategy is to run the full model to encode the prompt, then handle autoregressive decoding on a stage-wise basis, possibly with specialized workers for each phase. By disaggregating the pipeline, you ensure that GPUs continuously work on the portion of the task they’re most efficient at, avoiding head-of-line blocking, where one long generation stalls others behind it.

*分离推理流水线* 把推理工作流拆分为不同阶段，包括处理输入提示词并使其穿过所有模型层的“prefill”阶段，以及逐 token 生成输出的迭代式“decode”阶段。把这些阶段分配到不同资源上，以允许独立扩展。这种两阶段方式可防止较快的任务被较慢的任务拖成瓶颈。对于大语言模型，一种策略是运行完整模型来编码提示词，然后按阶段处理自回归解码，可能为每个阶段配备专用的工作节点。通过分离流水线，你能确保 GPU 持续处理其最擅长的那部分任务，避免队头阻塞——即一次长生成会拖住排在它后面的其他请求。

*Use continuous batch processing for LLMs* Move beyond simple request batching and use continuous batching strategies to maximize throughput under heavy loads. Traditional dynamic batching groups incoming requests and processes them as a batch to improve GPU utilization. Continuous batching takes this further by dynamically merging and splitting sequences of tokens across requests in real time. Systems like vLLM implement token pooling, where as soon as any thread is ready to generate the next token, it gets grouped with other ready threads to form a new batch. This approach keeps the GPU at high occupancy at all times and drastically reduces idle periods. The result is significantly higher token throughput and better latency consistency, especially when serving many concurrent users with varying sequence lengths.

*为 LLM 使用连续批处理* 超越简单的请求批处理，使用连续批处理策略以在重负载下最大化吞吐量。传统动态批处理把到来的请求分组并作为一个批处理，以提升 GPU 利用率。连续批处理更进一步，在实时中动态地跨请求合并与拆分 token 序列。vLLM 等系统实现了 token 汇池：只要任一线程准备好生成下一个 token，它就会与其他就绪的线程分到一组，组成一个新的批。这种方式让 GPU 始终保持高占用率，并大幅减少空闲时段。其结果是显著更高的 token 吞吐量与更好的延迟一致性，尤其是在服务许多序列长度各异的并发用户时。

*Shard models efficiently across GPUs and nodes* For models that are too large to fit into a single GPU’s memory, employ model-parallel inference techniques by partitioning the model across multiple GPUs or even multiple servers. This can be done with tensor parallelism, in which it splits each layer’s weights and computation across devices, or pipeline parallelism, which splits the model’s layers into segments hosted on different GPUs and streams the data through them sequentially. While model sharding introduces communication overhead and some added latency as data must flow between shards, it enables deployment of trillion-parameter models that would otherwise be impossible to serve. Ensure high-speed interconnects, such as NVLink or InfiniBand, between GPUs to make this feasible, and overlap communication with computation where possible. The key is to balance the load so all devices work in parallel and no single stage becomes a bottleneck.

*跨 GPU 与节点高效地对模型分片* 对于太大而无法装入单块 GPU 内存的模型，采用模型并行推理技术，把模型划分到多块 GPU 甚至多台服务器上。这可以用张量并行来完成，即把每一层的权重与计算切分到各设备上；也可以用流水线并行，即把模型各层切成托管在不同 GPU 上的段，并让数据顺序流经它们。虽然模型分片会引入通信开销和一些额外延迟（因为数据必须在各分片之间流动），但它使得部署原本无法服务的万亿参数模型成为可能。确保 GPU 之间有高速互连（如 NVLink 或 InfiniBand）以使其可行，并在可能处让通信与计算重叠。关键在于均衡负载，使所有设备并行工作，且没有任何单个阶段成为瓶颈。

*Offload memory for extended contexts* Use hierarchical memory strategies to support inference workloads that demand more memory than GPUs have available. Incorporate memory offloading when serving very large models or long sequence contexts, such as long multiturn conversations and large documents. Less frequently used data, such as old attention KV cache entries or infrequently accessed model weights, can be moved to CPU RAM or even NVMe storage when GPU memory gets tight. Modern inference frameworks can automatically swap out these tensors and bring them back on the fly when needed. While this introduces additional latency for cache misses, it prevents out-of-memory errors and allows you to handle extreme cases. By thoughtfully offloading and prefetching data, you trade a bit of speed for the ability to serve requests with large working sets, achieving a better overall throughput under memory constraints.

_为超长上下文卸载内存_ 采用分层内存策略，以支撑对内存的需求超出 GPU 可用容量的推理工作负载。当服务超大模型或超长序列上下文（如长多轮对话和大型文档）时，引入内存卸载（memory offloading）。较少使用的数据，例如陈旧的注意力 KV 缓存条目或不常访问的模型权重，可在 GPU 内存吃紧时移至 CPU RAM，甚至 NVMe 存储。现代推理框架能自动换出这些张量，并在需要时即时换回。虽然这会为缓存未命中引入额外延迟，但可避免内存不足（out-of-memory）错误，并让你能应对极端情况。通过审慎地卸载和预取数据，你以少许速度换取服务大工作集请求的能力，从而在内存受限的条件下获得更优的整体吞吐量。

## Power and Thermal Management

## 功耗与散热管理

Performance per watt is a first-class metric—thermal or power throttling erases tuning gains and shortens hardware life. Power caps, efficient packing, and proactive cooling stabilize clocks while cutting energy spend.

每瓦性能是一等指标——热降频（thermal throttling）或功耗降频会抹平调优收益，并缩短硬件寿命。功耗封顶（power capping）、高效的作业打包与主动散热能稳定时钟频率，同时削减能耗开销。

Track perf/watt and thermals alongside speed: cap power or underclock memory-bound workloads for better efficiency with minimal throughput loss. Proactively manage cooling, consolidate jobs to run GPUs near full, monitor per-GPU power draw, and schedule around energy price/renewables when it reduces cost. Here are some tips on managing your power and thermal characteristics of your AI systems:

在关注速度的同时追踪每瓦性能和热状态：对内存受限的工作负载施加功耗封顶或降低内存频率，以在几乎不损失吞吐量的前提下获得更高能效。主动管理散热，将作业整合以让 GPU 接近满载运行，监控每块 GPU 的功耗，并在能降低成本时围绕电价/可再生能源来调度。以下是管理 AI 系统功耗与散热特性的一些建议：

*Utilize efficient and environmentally friendly energy when possible* Track and optimize energy consumption alongside performance. In addition to managing power and thermal limits, monitor energy usage metrics and consider techniques that improve both performance and sustainability. For example, by implementing dynamic power capping or workload shifting based on renewable energy availability, you can reduce operational costs and carbon footprint. This dual focus reduces operational costs and supports responsible, environmentally friendly AI deployments.

_尽可能使用高效且环境友好的能源_ 在关注性能的同时追踪并优化能耗。除了管理功耗与散热上限外，还应监控能耗指标，并考虑同时改善性能与可持续性的技术。例如，通过实施动态功耗封顶或依据可再生能源可用性来转移工作负载，你可以降低运营成本和碳足迹。这种双重关注既降低运营成本，又支持负责任、环境友好的 AI 部署。

*Monitor thermals and clocks* Keep an eye on GPU temperature and clock frequencies during runs. If GPUs approach thermal limits (85°C in some cases), they may start throttling clocks, which reduces performance. Use nvidia-smi dmon or telemetry to see if clocks drop from their max. If you detect throttling, improve cooling, increase fan speeds, improve airflow, or slightly reduce the power limit to keep within a stable thermal envelope. The goal is consistent performance without thermal-induced dips.

_监控热状态与时钟频率_ 在运行期间密切关注 GPU 温度和时钟频率。如果 GPU 接近热限制（某些情况下为 85°C），它们可能开始对时钟降频，从而降低性能。使用 nvidia-smi dmon 或遥测数据查看时钟是否已从峰值回落。若检测到降频，应改善散热、提高风扇转速、改善气流，或略微降低功耗上限，以维持在稳定的热包络内。目标是获得不受热波动影响的稳定性能。

*Use energy-aware dynamic power management* Modern data centers are increasingly using energy-aware scheduling to adjust workloads based on real-time energy costs and renewable energy availability. Incorporating adaptive power capping and dynamic clock scaling can help optimize throughput per watt while reducing operational costs and carbon footprint.

_使用能源感知的动态功耗管理_ 现代数据中心越来越多地采用能源感知调度，依据实时能源成本和可再生能源可用性来调整工作负载。引入自适应功耗封顶和动态时钟调节，有助于优化每瓦吞吐量，同时降低运营成本和碳足迹。

*Optimize for perf/watt* In multi-GPU deployments where power budget is constrained (or energy cost is high), consider tuning for efficiency. Many workloads, especially memory-bound ones, can run at slightly reduced GPU clocks with negligible performance loss but noticeably lower power draw. For example, if a kernel is memory bound, locking the GPU at a lower clock can save power while not hurting runtime. This increases throughput per watt. Test a few power limits using nvidia-smi -pl to see if your throughput/watt improves. For some models, going from a 100% to 80% power limit yields nearly the same speed at 20% less power usage.

_针对每瓦性能优化_ 在功耗预算受限（或能源成本高）的多 GPU 部署中，可考虑面向能效进行调优。许多工作负载，尤其是内存受限的那些，可在略微降低的 GPU 时钟下运行，性能损失可忽略，但功耗明显下降。例如，如果某个 kernel 是内存受限的，将 GPU 锁定在较低时钟可在不损害运行时的前提下节省功耗。这提升了每瓦吞吐量。使用 nvidia-smi -pl 测试几个功耗上限，看看每瓦吞吐量是否改善。对某些模型而言，将功耗上限从 100% 降到 80%，能以少 20% 的功耗获得几乎相同的速度。

*Use adaptive cooling strategies* If running in environments with variable cooling or energy availability, integrate with cluster management to adjust workloads. For instance, schedule heavy jobs during cooler times of the day or when renewable energy supply is high—if that’s a factor for cost. Some sites implement policies to queue nonurgent jobs to run at night when electricity is cheaper. This doesn’t change single-job performance but significantly cuts costs.

_使用自适应散热策略_ 如果运行在散热或能源可用性可变的环境中，可与集群管理集成以调整工作负载。例如，将重负载作业安排在一天中较凉爽的时段，或可再生能源供应充足时运行——如果这对成本有影响的话。有些站点会实施策略，将非紧急作业排队到夜间电价更便宜时运行。这不会改变单个作业的性能，但能显著削减成本。

*Consolidate workloads* Run GPUs at high utilization rather than running many GPUs at low utilization. A busy GPU is more energy efficient in terms of work done per watt than an idle or lightly used GPU. This is because the baseline power is better amortized when the GPU is busy. It may be better to run one job after another on one GPU at 90% utilization than two GPUs at 45% each in parallel—unless you need to optimize for the smallest wall-clock time. Plan scheduling to turn off or idle whole nodes when not in use, rather than leaving lots of hardware running at low utilization.

_整合工作负载_ 让 GPU 高利用率运行，而不是让许多 GPU 都低利用率运行。就每瓦完成的工作量而言，繁忙的 GPU 比闲置或轻度使用的 GPU 更节能。这是因为当 GPU 繁忙时，基线功耗被更好地摊薄。让一个作业接一个作业在一块 GPU 上以 90% 利用率运行，可能优于两块 GPU 各以 45% 并行运行——除非你需要针对最短的墙钟时间（wall-clock time）进行优化。规划调度，在不使用时关闭或闲置整个节点，而不是让大量硬件在低利用率下持续运行。

*Configure cooling efficiently* For air-cooled systems, consider setting GPU fans to a higher fixed speed during heavy runs to preemptively cool the GPUs. Some data centers always run fans at the maximum to improve consistency. Ensure inlet temps in the data center are within specifications. Check periodically for dust or obstructions in server GPUs. Clogged fins can greatly reduce cooling efficiency. For water-cooled, ensure flow rates are optimal and water temperature is controlled.

_高效配置散热_ 对于风冷系统，可考虑在重负载运行期间将 GPU 风扇设为较高的固定转速，以预先冷却 GPU。有些数据中心始终以最高转速运行风扇以提高一致性。确保数据中心的进风温度在规格范围内。定期检查服务器 GPU 中是否有灰尘或阻塞物。堵塞的散热鳍片会大幅降低散热效率。对于水冷系统，确保流量最优且水温可控。

*Monitor power carefully* Use tools to monitor per-GPU power draw. nvidia-smi reports instantaneous draw, which helps in understanding the power profile of your workload. Spikes in power might correlate with certain phases. For example, the all-reduce phase might measure less compute load and less power, while dense layers will spike the load and power measurements. Knowing this, you can potentially sequence workloads to smooth power draw. This is important if operating the cluster on a constrained power circuit. In the power-constrained scenario, you may need to avoid running multiple power-spikey jobs simultaneously on the same node to avoid tripping power limits.

_仔细监控功耗_ 使用工具监控每块 GPU 的功耗。nvidia-smi 报告瞬时功耗，有助于理解工作负载的功耗特征。功耗峰值可能与特定阶段相关。例如，all-reduce 阶段的计算负载和功耗可能较低，而稠密层则会使负载和功耗测量值飙升。了解这一点后，你就有可能对工作负载排序以平滑功耗。若在受限的供电电路上运行集群，这一点尤为重要。在功耗受限的场景中，你可能需要避免在同一节点上同时运行多个功耗尖峰型作业，以免触发功耗上限。

*Improve job resilience for long-running jobs* If you are running a months-long training job or 24-7 inference job, consider the impact of thermals on hardware longevity. Running at 100% power and thermal limit constantly can marginally increase failure risk over time. In practice, data center GPUs are built for this type of resiliency, but if you want to be extra safe, running at 90% power target can reduce component stress with minimal slowdown. It’s a trade-off of longer training runs versus less wear on the hardware—especially if that hardware will be reused for multiple projects over a long period of time.

_提升长时间运行作业的韧性_ 如果你正在运行长达数月的训练作业或全天候（24-7）推理作业，应考虑热状态对硬件寿命的影响。持续在 100% 功耗和热限制下运行，会随时间略微增加故障风险。实践中，数据中心 GPU 就是为此类韧性而设计的，但如果你想格外稳妥，以 90% 功耗目标运行可在几乎不减速的情况下降低元件应力。这是在更长训练时间与更少硬件损耗之间的权衡——尤其是当该硬件将在很长一段时间内被多个项目复用时。

## Conclusion

## 结论

Treat the checklist as a repeatable playbook: profile, tune the right bottleneck at the right layer, and verify gains before scaling out. By methodically applying these practices—from OS and kernels to distributed comms and serving—you’ll achieve fast, cost-efficient, and reliable AI systems at any size.

把这份检查清单当作可复用的操作手册：剖析、在正确的层面调优正确的瓶颈，并在横向扩展前验证收益。通过系统性地应用这些实践——从操作系统和 kernel 到分布式通信与服务——你将在任何规模上都获得快速、成本高效且可靠的 AI 系统。

This list, while comprehensive, is not exhaustive. The field of AI systems performance engineering will continue to grow as hardware, software, and algorithms evolve. And not every best practice listed here applies to every situation. But, collectively, they cover the breadth of performance engineering scenarios for AI systems. These tips encapsulate much of the practical wisdom accumulated over years of optimizing AI system performance.

这份清单虽然全面，但并不详尽。随着硬件、软件和算法的演进，AI 系统性能工程领域将继续发展。而且这里列出的每条最佳实践并非适用于每种情况。但总体而言，它们覆盖了 AI 系统性能工程场景的广度。这些建议凝结了多年优化 AI 系统性能所积累的大量实用智慧。

When tuning your AI system, you should systematically go through each of the relevant categories listed in this chapter and run through each of the items in the checklist. For example, you should ensure the OS is tuned, confirm GPU kernels are efficient, check that you’re using libraries properly, monitor the data pipeline, optimize the training loop, tune the inference strategies, and scale out gracefully. By following these best practices, you can diagnose and resolve most performance issues and extract the maximum performance from your AI system.

在调优你的 AI 系统时，你应系统性地逐一梳理本章所列的各个相关类别，并逐条走查检查清单中的每一项。例如，你应确保操作系统已调优、确认 GPU kernel 高效、检查是否正确使用了各类库、监控数据流水线、优化训练循环、调优推理策略，并优雅地横向扩展。遵循这些最佳实践，你就能诊断并解决大多数性能问题，并从 AI 系统中榨取最大性能。

And remember that before you scale up your cluster drastically, you should profile on a smaller number of nodes and identify potential scale bottlenecks. For example, if you see an all-reduce collective operation already taking 20% of an iteration on 8 GPUs, it will only get worse at a larger scale—especially as you exceed the capacity of a single compute node or data center rack system, such as the Grace Blackwell GB200 and GB300 NVL72 and Vera Rubin VR200 and VR300 NVL systems.

还要记住，在大幅扩容集群之前，你应在较少节点上进行剖析，识别潜在的扩展瓶颈。例如，如果你看到某个 all-reduce 集合通信操作在 8 块 GPU 上已经占用了一次迭代的 20%，那么在更大规模下只会变得更糟——尤其是当你超出单个计算节点或数据中心机架系统的容量时，例如 Grace Blackwell GB200 和 GB300 NVL72 以及 Vera Rubin VR200 和 VR300 NVL 系统。

Keep this checklist handy and add to it as you discover new tricks. Combine these tips and best practices with the in-depth understanding from the earlier chapters, and you will design and run AI systems that are efficient, scalable, maintainable, cost-effective, and reliable.

随手备好这份检查清单，并在发现新技巧时不断补充。将这些建议与最佳实践与前面各章的深入理解相结合，你就能设计并运行高效、可扩展、可维护、成本高效且可靠的 AI 系统。

Now go forth and make your most ambitious ideas a reality. Happy optimizing!

现在，去把你最雄心勃勃的想法变为现实吧。祝优化愉快！
