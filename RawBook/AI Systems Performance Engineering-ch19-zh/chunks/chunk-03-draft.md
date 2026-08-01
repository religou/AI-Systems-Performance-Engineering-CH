```
# dynamic_quantized_cache.py
# Dynamic KV cache policy using Hugging Face Transformers' QuantizedCache (2025).
# Starts with int8 HQQ and drops to int4 when device memory is tight.
# Requires: transformers >= 4.55, hqq (for HQQ backend).
#
# This uses only public APIs:
#   - cache_implementation="quantized"
#   - cache_config as a dict
#
# References:
#   - KV cache strategies docs (QuantizedCache, HQQ/Quanto backends)
from __future__ import annotations
from typing import Dict, Optional
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
def make_cache_config(
    *,
    backend: str,
    nbits: int,
    device: torch.device,
    compute_dtype: torch.dtype = torch.float16,
    q_group_size: int = 64,
    residual_length: int = 128,
    axis_key: int = 1,
    axis_value: int = 1,
) -> Dict:
    """
    Build a cache_config dictionary accepted by Transformers' quantized cache.
    HQQ supports nbits in {2, 4, 8}; Quanto supports {2, 4}.
    axis_key/axis_value=1 are recommended for HQQ.
    """
    return {
        "backend": backend,                 # "HQQ" or "quanto"
        "nbits": int(nbits),
        "axis_key": axis_key,
        "axis_value": axis_value,
        "q_group_size": int(q_group_size),  # group size along head_dim
        "residual_length": int(residual_length), # recent tokens (orig precision)
        "compute_dtype": compute_dtype,     # dequantization compute dtype
    }
def _gpu_used_ratio() -> float:
    """
    Return fraction of device memory used as 1 - free/total.
    Uses CUDA driver info, which reflects true device state,
    not just the PyTorch allocator's reserved bytes.
    """
    free, total = torch.cuda.mem_get_info()
    return 1.0 - (free / total)
@torch.no_grad()
def generate_with_dynamic_quantized_cache(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    *,
    max_new_tokens: int = 256,
    chunk_tokens: int = 32,
    memory_threshold: float = 0.90,   # switch policy if used_ratio >= threshold
    backend: str = "hqq",             # "hqq" or "quanto"
    start_bits: int = 8,              # initial cache bit-width
    fallback_bits: int = 4,           # lower bit-width on pressure
    residual_length: int = 128,
) -> str:
    """
    Generate text in chunks while allowing mid-run policy changes.
    The policy applies to each chunk by choosing cache_config for that chunk.
    If memory is tight, we switch from int8 to int4 in subsequent chunks.
    """
    backend = backend.lower()
    assert backend in {"hqq", "quanto"}, "backend must be 'hqq' or 'quanto'"
    if backend == "quanto":
        assert start_bits in {2, 4} and fallback_bits in {2, 4},
               "Quanto supports only 2 or 4 bits"
    if backend == "hqq":
        assert start_bits in {2, 4, 8} and fallback_bits in {2, 4, 8},
               "HQQ supports 2, 4, or 8 bits"
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    generated_ids = inputs["input_ids"]   # [batch=1, seq_len]
    tokens_remaining = int(max_new_tokens)
    current_bits = int(start_bits)
    # Use EOS if available to terminate early.
    eos_id: Optional[int] = tokenizer.eos_token_id
    while tokens_remaining > 0:
        # Decide policy for this chunk based on current memory pressure.
        if torch.cuda.is_available():
            # Smooth the signal to avoid oscillation
            # when multiple processes are active.
            if 'used_ratio' in locals():
                used_ratio = 0.8 * used_ratio + 0.2 * _gpu_used_ratio()
            else:
                used_ratio = _gpu_used_ratio()
            if used_ratio >= memory_threshold:
                current_bits = min(current_bits, fallback_bits) # drop bits
                logging.info(f"Current bits {current_bits}")
        cache_cfg = make_cache_config(
            backend=backend,
            nbits=current_bits,
            device=device,
            compute_dtype=torch.bfloat16,
            q_group_size=64,
            residual_length=residual_length,
            axis_key=1,
            axis_value=1,
        )
        # Generate a small chunk with the chosen cache policy.
        this_chunk = min(chunk_tokens, tokens_remaining)
        out = model.generate(
            input_ids=generated_ids,
            max_new_tokens=this_chunk,
            do_sample=False,       # deterministic for clarity; adjust as needed
            use_cache=True,
            cache_implementation="quantized",          # select QuantizedCache
            cache_config=cache_cfg,                    # pass backend + settings
            pad_token_id=eos_id,
            return_dict_in_generate=False,        # we only need the tokens here
        )
        # 'out' is [1, old_len + this_chunk]; slice out newly generated suffix
        new_tokens = out[:, generated_ids.shape[1]:]
        generated_ids = out
        tokens_remaining -= new_tokens.shape[1]
        # Early termination if the model emitted EOS.
        if eos_id is not None and int(new_tokens[0, -1].item()) == eos_id:
            break
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)
if __name__ == "__main__":
    # Example usage. Replace with a model that supports your hardware.
    ckpt = "meta-llama/Llama-3.1-8B-Instruct"
    tok = AutoTokenizer.from_pretrained(ckpt)
    mdl = AutoModelForCausalLM.from_pretrained(ckpt,
                                            torch_dtype=torch.float16).to("cuda")
    text = generate_with_dynamic_quantized_cache(
        mdl,
        tok,
        "Explain attention key-value caches in one paragraph.",
        max_new_tokens=120,
        chunk_tokens=32,
        memory_threshold=0.90,
        backend="hqq",         # or "quanto" if you installed Quanto
        start_bits=8,
        fallback_bits=4,
        residual_length=128,
    )
    print(text)
```

这里，我们以一个用于适度压缩的 INT8 HQQ 缓存起步，当 GPU 实际空闲内存降至阈值以下时切换到 INT4。空闲内存通过 torch.cuda.mem_get_info() 测量，它反映的是设备真实的空闲与总量对比，从而为策略选择提供了正确的信号。

随后，我们以小分块（chunk）的方式生成 token，这样就能在分块之间安全地切换策略，而无需尝试改写某个已存在的缓存实例。这避免了访问私有属性或手动量化张量。缓存后端会在模型的前向传播（forward pass）内部完成这些工作。

> 如本例所示，建议在策略切换时记录一条事件或递增一个计数器。这样，你就能将压缩事件与任何精度或输出异常关联起来。

同样地，如果条件改善，你也可以动态关闭压缩。假设一段长对话刚刚结束，而下一个问题很短。系统可以决定停止压缩，甚至把某些缓存恢复到更高的精度——只要这样能带来更高质量的回复。不过差异很可能很小，因此未必值得这样做。

务必避免压缩状态的快速波动，因为过于频繁地开关压缩会导致性能抖动。为此，你可以在两次变更之间引入有意的延迟（即*迟滞（hysteresis）*与*冷却（cooldown）*）。例如，当切换到某种更高压缩率的策略后，就保持这一策略，直到 GPU 内存降到某给定阈值以下相当多为止。这样就能避免震荡与抖动。

具备这种灵活性是有用的，例如，你的服务有时会为付费用户优先保证最高质量（不压缩），而为免费层用户优先保证最高吞吐量（重度压缩）。策略也可以基于请求元数据来切换，包括用户的订阅类型。

任何关于缓存的讨论都离不开逐出（eviction）策略，例如最近最少使用（Least Recently Used，LRU）逐出。如果上下文长度变得过长，某些模型架构——例如带近因偏置（recency bias）或滑动窗口注意力（sliding-window attention）的架构——可能会选择完全丢弃或降采样非常旧的 token。滑动窗口注意力如图 19-5 所示。

![图 19-5. 滑动窗口注意力基于这样的直觉：最有用的 token 就是最近的那些](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-5.png)

虽然把上下文中较早的 token 做 LRU 逐出并不完全等同于压缩，但它是另一类可以在运行时动态选择的策略。例如，系统可以判定：在超过 2,048 个 token 之后，模型很可能不再需要更早的那些 token——这一判断可以基于某些启发式规则或一个更小的 LLM。

在这种情况下，系统可以开始丢弃那些更旧的 token——或者定期把它们压缩成一份更小的摘要。这已经开始涉足模型与算法领域了——需要更多的支持、维护与模型训练——但它是一种动态上下文管理的形式，在高级服务引擎中值得考虑。

简而言之，你应当考虑使用推理引擎所提供的量化缓存机制，因为它们可以处理诸多细节：维护量化尺度因子、与注意力核函数对接、以及在运行时监控 GPU 内存分配器。至少，当系统看到内存利用率接近某个上限时，把这一情况记录下来，并观察此时启用一项压缩策略是否能在不损害延迟的前提下避免 OOM。

> 在实践中，对 GPU 内存设置一个高水位阈值（例如 80%）来触发 8 位压缩，已被证明能有效防止生产环境中的 OOM 崩溃。

如果使用动态压缩策略是合理的，你就可以实现这个触发器。与任何量化与压缩策略一样，务必针对你所在领域测试它们对模型输出的影响。许多生成式任务能容忍激进的压缩，但始终有必要验证：使用 4 位相较于 8 位不会引入错误或意料之外的输出。

## 在运行时用强化学习智能体调优 AI 系统

到目前为止我们讨论的许多技术都涉及基于系统当前状态的决策。这些决策往往需要权衡取舍，例如速度与精度、吞吐量与延迟。与其不断收集越来越多的启发式规则来做决策，我们可以用强化学习（reinforcement learning，RL）来调优推理系统，配备一个 RL 智能体（agent）、环境（environment）、策略（policy）与奖励（reward）。

> 这是一种前沿方法。你应当先以更简单的启发式规则作为基线。等基础打稳后，再把 RL 作为一种增量式改进引入。

具体来说，我们的推理引擎监视服务器指标（环境），选择动作（策略）以在把延迟保持在目标之内的同时最大化吞吐量，并接收反馈（奖励）来引导持续改进。如此一来，系统便成为一个在线优化器——随着条件变化不断精炼其决策。

例如，可以搭建这样一个环境：每个 RL“步”（step）就是一次推理请求。而 RL 动作包括下列这些：

_动作 1_ 选择并行模式：单卡、TP、PP 与混合。

_动作 2_ 选择精度：全 FP8 还是混用 FP8 与 FP4。

_动作 3_ 调整批大小或批等待时间。

_动作 4_ 启用或禁用缓存压缩。

_动作 5_ 启用或禁用推测解码（speculative decoding）。

_动作 6_ 为推测解码选择一个更小的草稿模型（draft model）。

_动作 7_ 为推测解码选择一个更大的草稿模型。

_动作 8_ 启用或禁用推测式 KV 预取（speculative KV prefetching）。

……更多动作……

智能体所观测到的当前 RL 状态可能包括 GPU 利用率、内存利用率、平均延迟、请求队列长度等。RL 奖励随后通过刻画业务目标来定义，例如 reward = throughput - λ \* max(0, latency - SLA)。在这个奖励函数中，λ 是一个可调的惩罚权重，用于缩放你对延迟违约的惩罚力度。

> 建议对状态特征做归一化，这样 RL 智能体就不必自己去学习各特征的量纲尺度。这可以加快训练收敛。例如，你可以用一个最大值来缩放队列长度，等等。

这里，较大的 λ 会让智能体更看重不违反 SLA，而不是压榨出额外的吞吐量。较小的 λ 则允许它冒偶尔延迟超标的风险，以换取整体上更高的 token 速率。本质上，这个奖励函数会惩罚超出 SLA 的延迟，但在其他情况下则尽力提升吞吐量。

> 在实践中，起步时应让 λ 满足：λ ×（典型的延迟超标量）大致等于你愿意为之付出的吞吐量收益。例如，如果为换取 100 tokens/sec 而容忍 10 ms 的延迟，那就把 λ 设为使 10 ms × λ ≈ 100。

经过多次迭代，RL 智能体可以学会何时压缩缓存是有益的——例如当内存占用很高而延迟不会立即受影响时。或者它可以学会：当 GPU 利用率偏低但某一块 GPU 过载时，切换到流水线并行模式，等等。

这之所以有帮助，是因为 PP 把模型拆分为跨多个设备的顺序阶段，并将繁重的工作从瓶颈设备上重新分配出去——从而平滑利用率、避免单 GPU 热点（hotspot）。

智能体能够找到产生更好性能的反直觉配置。例如，它可能学到：对于超过某一长度的提示词，应同时启用 PP 与 FP4 压缩以获得最佳的 token 吞吐量；而对于较短的提示词，则学会仅用 FP8 下的纯张量并行。如果我们试图把这套逻辑编码为一组静态规则，就可能漏掉复杂的交互作用。

RL 智能体可以通过探索动作空间更容易地发现最优组合——并最终利用（exploit）这一最优配置，直到环境条件发生变化。此时，RL 会相应地调整推理系统，因为它始终在探索动作空间、尝试新的配置。

训练这样一个智能体可以离线完成，使用诸如 Hugging Face 的 Transformer RL（TRL）之类的模拟器与库。例如，我们可以在各种条件下从运行中的系统采集大量数据——然后在模拟中训练一个 RL 策略来预测结果。在非常高的层面上，RL 的奖励与更新循环大致如下面的伪代码所示：

```
# Pseudo structure for an RL-driven tuner
# This loop runs in separate thread/process alongside main inference service.
# e.g., {gpu_util:0.7, mem_util:0.9, avg_latency:120ms, req_queue:5}
state = get_system_state()
# e.g., 0 -> high precision, 1 -> low precision
action = rl_agent.select_action(state)
# Map action to actual parameter changes
if action == 0:
    precision_policy = "FP8"
else:
    precision_policy = "FP4"
# (We could have multiple actions, but single action for illustration)
apply_precision_policy(precision_policy)
# ... After the next token or set of tokens ...
new_state = get_system_state()
reward = compute_reward(old_state, new_state)
rl_agent.update(state, action, reward, new_state)
```

这里，该循环在推理服务器的后台持续运行。compute_reward 函数纳入了吞吐量（例如自上一步以来每秒的 token 数）与延迟指标。由于我们试图在吞吐量与延迟之间取得平衡，这是一个多目标优化（multi-objective optimization）问题，即同时优化多个目标。一种常见做法是用加权和把多个目标合并为单一目标。

> 为获得更高的灵活性，尤其是在存在不确定性时，你可以转而把多目标优化问题——或帕累托前沿（Pareto front）分析——建模为一个部分可观测决策过程。这让智能体能够自行学习在诸如吞吐量与延迟等目标之间的权衡策略。当单一的加权奖励不够用时，这会很有帮助。

这类多参数交互很难用基本的网格搜索方法来调优。因此，RL 以及诸如近端策略优化（proximal policy optimization，PPO）之类的优化技术最适合用来调优推理工作负载。PPO 以在连续动作空间中更稳定的学习而著称。它很适合在实时环境中做连续更新，因为它会逐步调整策略。这避免了剧烈震荡，而这对推理稳定性很重要。我们不希望智能体在每个请求上都在各决策之间反复横跳。

另一种减少震荡的技术叫做*阻尼（damping）*。它要求某个动作至少保持一段最短的时间——或至少作用于最少数量的请求。必要时你可以为关键的 SLO 违约情况覆盖阻尼，但这应当谨慎为之。

需要知道的是，RL 智能体在学习过程中可能做出不安全或次优的动作。为缓解这一点，你可以把动作空间约束到一组合理的取值范围内。同样建议：先用我们已经识别出的启发式规则设定一个良好的默认策略。智能体随后可以围绕这个初始默认策略进行微调。

或者，也可以在*影子模式（shadow mode）*下用一个包含探索阶段的在线系统来训练智能体。在探索阶段，系统偶尔会尝试一种随机或略作修改的策略以采集新数据。其余时候，它则利用当前的最优策略。

另一种技术是应用奖励塑形（reward shaping），它能防止智能体违反关键约束。例如，当延迟超过某个硬性上限时——或当某个糟糕的动作导致 OOM 错误时——RL 系统会给出一个很高的负奖励。

此外，你还可以把系统硬编码为规避不安全的动作——即便奖励暗示系统应当这样做。这设置了额外的安全防线，使得智能体自然的探索不会导致灾难性故障。这是一种把 RL 与基于规则的护栏相结合的实用做法。

设计一个恰当的奖励函数很重要。例如，如果我们关心的是在延迟上限之下的吞吐量，奖励会像这里的代码那样：

```
reward = tokens_per_second - 1000 * (1 if latency > SLA else 0)
```

这里，一旦超出延迟 SLA，就施加一个很大的惩罚。否则不施加任何惩罚。另一种选择是施加一个连续惩罚，其大小与延迟超出目标 SLA 的程度成正比。一个简单的连续惩罚奖励可以写成如下形式：

```
reward = tokens_per_second – λ * max(0.0, latency – SLA)
```

这里，λ 是你的惩罚权重，而 max(0.0, latency – SLA) 随你超出 SLA 的程度线性增长。这样，延迟超出目标越久，智能体收到的惩罚就越平滑地增大。这会产生更平滑的梯度与更渐进的权衡决策。在实践中，连续（软）惩罚往往比二值（硬）惩罚产生更稳定的策略。

> 建议先用一组简单、静态的启发式规则来做调优。一旦系统稳定，你就可以开始引入 RL 智能体，来处理那些启发式规则无法捕捉的更复杂的调优。

日志记录与可观测性很重要。你应当持续记录 RL 智能体所做的决策——以及决策的结果。例如，你应使用结构化日志——甚至计数器与遥测仪表盘——来实时追踪 state → action → reward 的序列。当智能体开始出现异常行为时，这将有助于调试。

同样建议提供一个逃生舱口，或称*断路开关（kill switch）*。这样，如果智能体开始做出明显糟糕的事情，比如持续使延迟变差，你就可以让系统回退到一个安全、静态的配置，同时你去诊断问题并在离线重新训练一个新策略。例如，如果启用智能体后 p95 延迟增加超过 50%，系统会自动禁用智能体的动作，并向值班人员发送告警。

尽管截至本文撰写时，基于 RL 的在线推理调优在现代推理服务引擎中尚未成为主流，但它才刚刚开始出现。随着这些技术的成熟，可以预期会有更多推理平台纳入自调优能力。这一点很重要，因为这些模型与系统正变得越来越复杂。在快速变化的条件下手动管理所有这些调优旋钮很困难——反正对人类来说是很困难的！

一个能实时自适应的智能体，是系统优化的自然演进。我们已经开始看到能自动达到专家级性能调优水平的自优化 AI 推理服务器。而它们做到这一点，仅仅是通过学习自身的实时遥测指标。

## 动态内存分配切换（slab、缓存与流序三者之间）

GPU 内存碎片化（fragmentation）与非最优的内存分配可能是无声的性能杀手。推理服务器每秒为众多对象——包括 token、中间激活等——分配并释放数千个张量。内存分配器所采用的策略会影响碎片化与分配延迟。

动态切换内存分配器意味着系统可以即时改变它分配内存的方式。例如，系统可以对某些分配大小使用 slab 分配器——或者切换为使用 CUDA 的流序（cudaMallocAsync）分配器。这一决策取决于观测到的分配模式与预期的内存碎片化。

默认情况下，PyTorch 使用一种被称为*带合并的最佳适配（best-fit with coalescing）*（即 BFC）的伙伴/最佳适配（buddy/best-fit）分配器变体。它抓取大块的 GPU 内存并将其细分以满足分配请求。这重用了空闲空间，并避免了频繁调用相对缓慢且同步的 cudaMalloc 与 cudaFree。

伙伴分配器（buddy allocator）把内存拆分成大小为 2 的幂次的块。slab 分配器则工作在伙伴系统之上，用以高效管理小的、固定大小的对象。它预分配 slab（即某一给定类型对象的集合），并在每个 slab 内维护一个空闲列表（free list），如图 19-6 所示。

![图 19-6. slab 分配器在每个预分配的 slab 内维护一个内存对象的空闲列表](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-6.png)

slab 分配器允许在不产生碎片化的情况下快速重用。伙伴分配器处理粗粒度的页分配，而 slab 分配器则优化细粒度的对象重用。

默认的 PyTorch 缓存分配器（caching allocator）对许多工作负载都工作良好。然而，如果分配模式因在大分配与小分配之间来回交替而变化剧烈，它就可能出现碎片化。在一个处理不同类型查询的长期运行的服务器中，碎片会逐渐累积。

在这种情况下，会有大量内存空闲，但这些内存不够连续，无法满足大张量分配。这会导致 OOM 错误——尽管从技术上说内存是可用的。

请记住，PyTorch 提供了 torch.cuda.memory_summary() 来评估内存碎片化，以及一个内置于 torch.profiler 的内存分析器（profile_memory=True）。你可以用它们来判定哪些操作分配了大量内存。此外，NVIDIA Nsight Systems 在时间线上提供 CUDA 内存事件与统一内存（Unified Memory）缺页追踪，而 Nsight Compute 则提供内存工作负载分析。

这些工具合在一起，让你能够随时间观察分配行为与碎片化效应。你可以在开发与测试阶段使用它们来启动你的内存分配器调优策略——包括接下来讨论的动态调优策略。

一种减少内存碎片化的暴力方法是定期重置分配器的状态。不过，一种更巧妙的方法是使用 CUDA 的流序缓存分配器 cudaMallocAsync，它在内部通过按分配大小分箱（直到某个上限）采用了类似的思路。但 slab 分配更进一步——它绝不混用不同大小。

cudaMallocAsync 的行为有点像 slab 分配器与伙伴系统的结合——而且它由 CUDA 驱动为你托管。这让你几乎不费力就能获得自定义分配器的大部分收益——如果你愿意，它也是一个很棒的、可以始终开启的默认内存分配器。

具体来说，cudaMallocAsync 使用流序内存池，当对内存的引用被释放时会自动回收内存。随后它会在幕后合并已释放的块，因为它知道内存释放的依赖顺序——这一点与标准分配器不同。

在 PyTorch 运行时中使用 cudaMallocAsync 时，你可以通过设置环境变量 PYTORCH_ALLOC_CONF=max_split_size_mb: <value> 来动态调整 max_split_size_mb。这可以在不同条件下调整切分大小。

例如，一个动态系统可以在预期出现大分配时增大 max_split_size_mb。这样它们就不会被切成小碎片。反之，当运行许多小请求时，系统可以减小 max_split_size_mb，以允许更多地重用大块。

> 切分大小过小会让分配器充斥大量微小的块，这会增加元数据开销与潜在的碎片化。切分大小过大会减少块数量（以及元数据），但当你只释放某个块的一部分时，可能会在内存中留下更大的、无人使用的“空洞”。

考虑这样一个场景：你的服务检测到了碎片化——也许是借助 PyTorch 的内存快照功能，它会展示由碎片化造成的空洞。在这种情况下，系统可以动态切换为使用 cudaMallocAsync，它能整合内存使用。

你应当使用内存监控工具来追踪——并记录——内存碎片化。例如，在 PyTorch 中，你可以使用 torch.cuda.memory_reserved() 与 torch.cuda.memory_allocated()。这里，reserved 内存是分配器所持有的 GPU 内存总量，而 allocated 则是其中实际被张量使用的量。

reserved 与 allocated 之间的巨大差距意味着碎片化，因为有大量内存被保留但未被使用。如果这一差距随时间增长，一种动态策略可以是：定期清空缓存以把所有未使用的内存归还给 GPU，甚至重启 worker 进程以彻底重置分配器。这些方法具有侵入性但很有效，有时会在生产环境中用于碎片化严重的长期运行进程。

> 只应在维护窗口期间使用清空、重启这类侵入式碎片整理方法——或者以跨整个集群滚动重启的方式来避免停机。如果你不得不诉诸这些破坏性机制，那你很可能面临一个更深层、需要被解决与优化的问题。

举例来说，要在 PyTorch 中实现动态分配切换，你可以先从 PyTorch 原生分配器起步。然后，如果你捕获到一个 OOM 错误，就可以改用基于 cudaMallocAsync 的分配器重试。

遗憾的是，CUDA 缓存分配器在 torch 首次被导入时——或在首次触及 CUDA 上下文时——就已创建。而 Python 的 importlib.reload 并不会卸载 C++ 扩展或拆除分配器。因此，在进程内即时更改 PYTORCH_ALLOC_CONF（旧称 PYTORCH_CUDA_ALLOC_CONF）并重新加载 Python 模块，并不会重新配置分配器。

不过，你可以派生（spawn）一个全新的进程，在其中于导入 torch 之前就设置好该环境变量。下面是一段代码，它捕获 OOM，在父进程中释放内存，然后带着 PYTORCH_ALLOC_CONF 启动一个干净的子进程。这有点取巧，但展示了你如何动态设置 backend:cudaMallocAsync 并用不同的分配器后端重跑同一次调用。接下来是一个 PyTorch 示例，它在代码捕获到 torch.cuda.OutOfMemoryError 时实现这一动态策略：

```
# dynamic_memory_allocator.py
# Retry generation in fresh process with cudaMallocAsync if first attempt OOMs.
# This is the only reliable way to change the CUDA allocator at runtime.
import os
import sys
import gc
import pickle
import tempfile
import subprocess
import importlib
from typing import Callable, Any
def _resolve_factory(factory_path: str) -> Callable[[], Any]:
    """
    Resolves a factory string like "my_package.my_mod:build_model" to a callable.
    The callable must return ready-to-use model with .generate(request) method.
    """
    module_name, func_name = factory_path.split(":", 1)
    module = importlib.import_module(module_name)  # safe to import, no torch yet
    return getattr(module, func_name)
def generate_with_allocator_retry(
    model_factory_path: str,
    request_object: Any,
    *,
    allocator_conf: str = "backend:cudaMallocAsync"
) -> Any:
    """
    Attempts model.generate(request_object) in the current process.
    On torch.cuda.OutOfMemoryError, retries in a fresh subprocess with
    PYTORCH_ALLOC_CONF set to allocator_conf. request_object and the
    returned value must be picklable.
    """
    # Import torch only inside function; avoid importing at module import time.
    import torch
    model_factory = _resolve_factory(model_factory_path)
    model = model_factory()  # user-supplied function builds model, moves to GPU
    try:
        # First attempt uses whatever allocator current process started with.
        return model.generate(request_object)
    except torch.cuda.OutOfMemoryError:
        # Free as much as possible in the parent before spawning the child.
        # Avoids compounding pressure when two processes momentarily overlap.
        try:
            del model
        finally:
            gc.collect()
            torch.cuda.empty_cache()
        # Serialize request to temp file and ask fresh interpreter to do work.
        with tempfile.TemporaryDirectory() as td:
            req_path = os.path.join(td, "request.pkl")
            out_path = os.path.join(td, "output.pkl")
            with open(req_path, "wb") as f:
                pickle.dump(request_object, f)
            # In the child, we want torch to see allocator config at import.
            env = os.environ.copy()
            env["PYTORCH_ALLOC_CONF"] = allocator_conf
            # Re-run this module as a helper child. The child will import torch
            # only after PYTORCH_ALLOC_CONF is set in its environment.
            cmd = [
                sys.executable,
                __file__,
                "--child",
                "--factory", model_factory_path,
                "--request", req_path,
                "--output", out_path,
            ]
            completed = subprocess.run(cmd, env=env, capture_output=True,
                                       text=True)
            if completed.returncode != 0:
                # Bubble up child stderr to aid debugging
                raise RuntimeError(
                  f"Retry failed with exit code {completed.returncode}\n"
                  f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
                )
            with open(out_path, "rb") as f:
                return pickle.load(f)
def _child_main(factory_path: str, request_path: str, output_path: str) -> None:
    """
    Child entrypoint: assumes PYTORCH_ALLOC_CONF is already present in env.
    Imports torch only now, builds the model, runs generate, and pickles result.
    """
    # Import torch after env var set by the parent’s subprocess.run(env=...).
    import torch
    model_factory = _resolve_factory(factory_path)
    model = model_factory()  # build the model inside the child process
    with open(request_path, "rb") as f:
        request_object = pickle.load(f)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--factory", type=str, default="")
    parser.add_argument("--request", type=str, default="")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()
    if args.child:
        _child_main(args.factory, args.request, args.output)
    else:
        print("This module is intended to be imported.")
```

这里，模型是在子进程中使用一个工厂函数构建的，从而确保在 PYTORCH_ALLOC_CONF 环境变量生效之前不会导入任何与 CUDA 相关的内容。代码使用 torch.cuda.empty_cache() 清空缓存并把未使用的内存归还给 GPU。这一模式保证了分配器配置会在子进程中导入 torch 之前被应用。它还避免了在运行时尝试卸载原生扩展——而这是 CPython 所不支持的。

在非 PyTorch 环境中，例如一个基于 C++ 的 LLM 推理引擎，你可以实现一个纯粹的 slab 分配器，允许针对特定分配大小进行配置。这类 slab 分配器会把内存预先划分为固定大小的“slab”。它对同一大小的重复分配非常高效，并且对该特定大小的分配几乎不产生碎片化。

在 LLM 服务器中，一种非常常见的技术是分配每个 token 的输出张量，例如每生成一个 token，就为该 token 的激活分配一个 [layers, hidden_dim] 张量。如果这些分配每次都是相同大小，比如 64 KB，那么为这个确切大小设置一个 slab 就是理想之选。

系统检测到自己在反复分配大量的 64 KB 张量——于是创建一个由专用 64 KB 块组成的“slab”。slab 分配器通常不会把内存归还给通用内存池，直到整个 slab 被释放为止。
