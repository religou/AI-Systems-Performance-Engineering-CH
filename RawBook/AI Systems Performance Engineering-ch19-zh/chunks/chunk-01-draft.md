# 第 19 章 动态与自适应推理引擎优化

在现代硬件上进行超大规模语言模型（large language model，LLM）推理，需要在运行时进行动态适配，才能在多变的条件下同时实现高吞吐量与低延迟。对模型服务优化采用静态的“一刀切”（one-size-fits-all）方案已不再够用。

取而代之，最先进的模型服务系统采用*自适应策略*，在运行中动态调整并行、数值精度、CUDA 核函数调度与内存使用。本章将探讨这些高级技术，包括动态并行切换、精度伸缩、实时缓存管理，以及基于强化学习（reinforcement learning，RL）的调优。

本章给出超大规模 LLM 推理的最佳实践，教你如何编排一个能够自我监测性能、并实时自适应以最大化效率的引擎。

## 自适应并行策略（TP、PP 与混合）

超大规模 LLM 需要模型并行——例如张量并行与流水线并行，或二者的混合方案——才能将计算分散到多块 GPU 上。每种方案各有利弊。表 19-1 总结了针对特定推理流量模式所推荐的并行策略。

表 19-1. 常见推理流量模式与推荐并行策略的对应汇总

| 推理流量模式                       | 推荐并行               | 依据                                                                                            |
| ---------------------------------- | ---------------------- | ----------------------------------------------------------------------------------------------- |
| 大量短请求（< 256 tokens，高 RPS） | 数据并行/副本扩展      | 最小化 GPU 间通信；每块 GPU 运行副本、各自处理相互独立的请求（前提是模型能装入单块 GPU 的内存） |
| 少量长请求（≥ 8k tokens，低并发）  | 流水线并行（配合微批） | 通过将层拆分到多块 GPU 上以降低单请求延迟                                                       |
| 混合负载（短请求 + 部分长请求）    | 混合动态（自动切换）   | 在单块 GPU 上运行小型对话，对长请求走流水线，以满足延迟 SLA                                     |
| 超大模型（> 1 块 GPU 内存）        | 张量 + 流水线混合      | 为装下模型所必需；在两个维度上平衡计算与内存                                                    |
| MoE 模型推理（稀疏专家选择）       | 专家并行               | 将各个专家分布到多块 GPU 上；每个请求只调用一部分专家，从而降低单设备的内存与计算负载           |

数据并行与副本扩展策略会在每块 GPU 上复制完整模型，并将进入的请求在这些副本之间做负载均衡。由于每块 GPU 各自独立处理不同的请求，单次推理无需 GPU 间同步。

对于大量中小规模输入，这能以极小的通信开销最大化吞吐量。然而，如果模型无法装入单块 GPU 的内存，数据并行便不可行。

张量并行（tensor parallelism，TP）是一种模型并行形式（与数据并行相对），它将模型矩阵（例如权重、层等）拆分到多块 GPU 上以加速矩阵乘法。不过，它会引入额外的 all-reduce 通信来保持各 GPU 的同步。

流水线并行（pipeline parallelism，PP）是另一种同样对模型进行拆分的模型并行形式。但它不拆分单个模型层与矩阵，而是把整层分配给不同的 GPU，以突破内存限制——前提是单层能装入一块 GPU。PP 会带来以顺序阶段延迟为形式的额外开销。这些延迟被称为*流水线气泡*（pipeline bubble），如图 19-1 所示。

专家并行（expert parallelism）用于 MoE（专家混合，mixture-of-experts）模型架构，为每个专家子网络分配各自的 GPU。随后，一个轻量的门控网络（gating network）将每个输入请求或 token 只导向由路由器（router）识别出的 top-k 个激活专家。此时，每块 GPU 只处理它所承载的那一部分专家。

![图 19-1. PP 造成的流水线气泡](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-1.png)

由于每个输入只激活少数几个专家，对于拥有大量专家、常被称为*宽*（wide）专家模型的模型，专家并行可降低单设备内存、推理时间与计算成本。这种基于路由器的条件式专家计算模式，会随着专家数量的增加而高效扩展。例如，DeepSeek-R1 共有 256 个专家，但在推理时路由器只选择 top 9 个专家（其中包含 1 个共享专家）。

传统上，并行化策略——包括将多种并行技术组合而成的混合策略——是在加载模型时预先选定并固定下来的。然而，为在动态负载下最大化性能，现代推理引擎可以在运行时根据输入特征选择不同的并行策略。

高性能的自适应推理系统利用运行时指标来动态选择 TP、PP 或混合方案。关键因素包括 batch size、序列长度与内存利用率——以及响应延迟与吞吐量要求。例如，超长提示可能被路由到 TP + PP 实例，因为这会把层分散到多块 GPU 上，以避免内存溢出（out-of-memory，OOM）错误。

与此同时，对延迟敏感的短请求会被路由到仅使用 TP 的模型实例，以避免流水线阶段开销。为支持这一点，你的服务引擎会维护多个预分片（presharded）的模型实例，每个都针对不同的负载画像做了优化，并将进入的查询动态分派给其并行策略最能满足该作业 SLO 的模型实例。

你也可以使用不同的分片数量。如图 19-2 所示，它在跨八块 GPU 的两种不同的混合 TP + PP 并行配置中使用了两种不同数量的 TP 分片。

![图 19-2. 为某个模型在八块 GPU 上预置两种不同的混合分片池（TP = 4, PP = 1 与 TP = 2, PP = 1）](AI%20Systems%20Performance%20Engineering-ch19_images/figure-19-2.png)

对长序列输入动态使用 PP，有助于避免由大型输入序列引发的 OOM 错误。反之，对于短提示与延迟敏感的查询，系统可以改为路由到为低延迟优化的张量并行模型实例。此时，请求便可避开 PP 的开销。

由于每个请求可以使用不同的并行策略，系统需要维护该模型的多个实例，供推理调度器/路由器选择。其中一个模型实例使用 TP 针对低延迟做优化，另一个实例则同时使用 TP 与 PP 针对高吞吐量与大型输入序列做优化。

之所以必须维护模型的多个实例，是因为在运行中动态重新分片（resharding）会严重扰乱 GPU 缓存。它还会给内存与网络子系统带来过大压力——尤其是在对超大模型重新分片时。

在运行时，每个查询会根据其长度与指定的服务级别协议（service-level agreement，SLA）被分派到最契合的模型实例（分片策略）。以 DeepSeek-R1 为例，它是一个约 6800 亿参数的稀疏专家混合模型，在各专家之间每个 token 只激活 370 亿参数。

为支持不同的负载画像，可以将 GPU 组织成逻辑 worker 池，每个池按特定并行策略预先分片——例如张量并行，或张量 + 流水线混合并行。

来看一个例子。假设我们有一台配备 8× GPU 的 Blackwell B200 服务器，HBM 内存共计 1,440 GB（1,440 GB = 每块 GPU 180 GB × 8 块 GPU），我们可以用跨四块 GPU 的四路 TP 来服务 DeepSeek-R1——让另外四块 GPU 处于空闲。

如果有单个查询携带极长的上下文（例如 > 100 万 tokens）到来，调度器可以生成一个两阶段流水线，使阶段 1 覆盖 GPU 0–3、阶段 2 覆盖 GPU 4–7。这实际上把每个阶段可用的 GPU 内存翻倍到约 720 GB 的总 HBM（180 GB 每块 GPU × 4 块 GPU）（720 GB 可用 HBM）。这有助于在处理大型输入时避免 OOM 错误。

反之，当数十个短的、延迟敏感的提示并发到来时，系统只将它们路由到张量并行实例。通过避免流水线气泡（即在填充与排空流水线阶段时出现的空闲期），这种配置能在所有可用 GPU 上提供尽可能低的单请求延迟。

要实现动态并行切换，你可以实现一个决策函数，检查诸如输入序列长度、GPU 内存使用与当前负载等运行时指标。你会用这些指标为每个请求选择分片最优的模型实例，如下所示：

```
def choose_worker_pool(seq_len, gpu_mem_util, concurrent_reqs):
    # For long contexts or high memory pressure,
    # use hybrid pipeline + tensor parallelism
    # (example thresholds shown here)
    if seq_len > 4096 or gpu_mem_util > 0.8:
        return "tp_pp_hybrid"
    # For many simultaneous small requests, stick with tensor parallelism
    if concurrent_reqs > 4:
        return "tensor_parallel"
    # Fallback to tensor-parallel for typical workloads
    return "tensor_parallel"
```

你会在 GPU 集群上预先启动多个模型副本——一些仅分片为 TP，另一些分片为 TP + PP——并让路由器根据输入与决策策略把每个查询发送到合适的副本。这种做法可确保大型、内存密集的作业获得完整的流水线支持，而短的、延迟敏感的调用则在仅 TP 的实例上运行，以避免不必要的流水线开销。

建议利用来自模型与硬件的遥测数据来指导并行切换。你可以实时监控 GPU 内存利用率、计算利用率以及互连（例如 NVLink/NVSwitch）流量来做出决策。如果你发现由于较长的流水线气泡导致 GPU 空闲——且你还有额外的内存余量——你可以把流水线收缩为更少的阶段，使每块 GPU 承担更多工作并保持繁忙。反之，如果某些阶段触及内存上限或计算瓶颈，你可以扩展为更多的流水线阶段——或提高张量并行度。这会把计算与内存占用分散到更多的 GPU 上。

关键在于动态调整张量拆分与流水线拆分之间的平衡，让每块 GPU 都保持良好利用。与此同时，你需要保持在内存约束之内并达成延迟目标。这是静态的一刀切配置所无法做到的。

## 动态精度切换

像 Blackwell 这样的现代 GPU 引入了对 8 位与 4 位浮点（FP8/FP4）Tensor Core 计算单元的支持。这些更低的精度带来了巨大的加速、内存节省与极小的质量损失。

动态精度切换是一种高级技术，推理引擎会在运行时根据模型置信度（confidence）或资源压力调整数值精度。其目标是在不明显损失质量的前提下提高吞吐量。在实践中，这意味着系统可能为了效率以 FP8 或 FP4 执行模型的某些部分，但在为稳定性所需时回退到更高精度（FP16/BF16）。

触发精度自适应的一个因素是*logit 锐度*（logit sharpness），即模型输出的置信度。例如，如果模型对下一个 token 的概率分布因为对某个特定 token 高度自信而呈现出极端的尖峰，那么低精度带来的微小数值误差不太可能改变结果。

如果下一个 token 的生成能够容忍低精度，引擎便会在接下来的几步中安全地使用 FP4 以获得速度提升。反之，如果由于高度不确定性使分布较为平坦，引擎则应坚持使用 FP8 或 FP16 以保持保真度。

推理引擎通过计算词表上 softmax 分布的香农熵（Shannon entropy）来量化不确定性。熵越低，表示预测越尖锐（越自信）。一个在留出验证集上调好的固定熵阈值，决定何时降到 FP4、何时为数值稳定性而保持在 FP8/FP16。其目标是在延迟收益与精度损失之间取得平衡。

> 使用能维持精度的最低精度，并在模型置信度（以最大 softmax 概率衡量）下降时回退到更高精度。

这利用了这样一个事实：大型 LLM 在生成确定性的续写（例如闭合引号或完成一个列表）时，往往会变得更加确定。在这些情况下，较低的精度通常就足够了。

另一个因素是内存压力。如果由于非常长的上下文——或大量并行请求，GPU 内存使用逼近上限，系统可以动态地把激活压缩到更低精度。

在内存紧张时，可以把注意力的 key/value 张量以 INT4 而非 INT8 存储。这会把内存占用减少 50%。不过，要确保使用 INT4 带来的量化误差不会在许多解码步骤中累积放大。建议定期重新评估输出质量。

例如，如果某次推理达到 KV 缓存占用 90% 内存的临界点，引擎可能决定把新的缓存条目从 INT8 量化到 INT4——甚至回溯性地压缩较旧的条目——以释放空间。这可以在不停止模型的情况下完成。此时，后续的注意力层只需读取 INT4 缓存值——仅带有轻微的量化误差。

将 4 位权重量化与 8 位激活相结合可以显著减少内存。例如，对于纯计算受限的核函数，FP8 激活可实现最高 2× 的吞吐量——尤其是在高带宽的现代 GPU 上。对于混合型或内存受限的负载，1.5× 是可以实现的。对激活使用 FP4 可以把内存节省推进得更远。不过，它可能引入略高的累积误差，需要仔细的逐层调优。

现代 GPU 提供原生的 FP8 与 FP4 Tensor Core。然而，截至本文撰写时，PyTorch 的 AMP（自动混合精度，automatic mixed precision）支持（torch.autocast）仍只面向 FP16 与 BF16。它并不面向 FP8 或 FP4。尽管 PyTorch 中存在 FP8 dtype（例如 torch.float8_e4m3 与 torch.float8_e5m2）以及带缩放的数学通路，但 AMP 并不管理它们。对于推理与训练，建议使用 NVIDIA 的 Transformer Engine（TE），并在适当时采用其 MXFP8 与 NVFP4。

> 对于延迟关键的 decode，在 Blackwell 上使用 AMP 时优先选择 BF16 而非 FP16。对于 FP8 通路，Transformer Engine 的 MXFP8 格式是 Blackwell 上推荐的默认选项。在 KV 缓存与轻量层上有选择地使用 NVFP4，并配合仔细的回归测试。记得在你的具体负载上逐层验证数值。

表 19-2 汇总了一些示例精度配置及其权衡。可以看到，更低的精度会减少内存并提高吞吐量。不过，会带来轻微的质量下降。

表 19-2. LLM 推理中各精度模式的近似权衡

| 精度模式             | 内存使用（相对） | 计算吞吐量   | 质量影响（精度变化）                 |
| -------------------- | ---------------- | ------------ | ------------------------------------ |
| FP16（基线）         | 1.0× (100%)      | 1.0×（基线） | 无影响（完全保真）                   |
| FP16 权重 + FP8 激活 | ~0.5× (50%)      | ~1.5×        | 可忽略（< 0.1%）                     |
| INT4 权重 + FP8 激活 | ~0.25× (25%)     | ~1.8×        | 质量下降 ~0.5%（计算与内存混合受限） |
| INT4 权重 + FP4 激活 | ~0.2× (20%)      | ~3.5×        | 下降 ~1%（需要仔细调优）             |

这里可以看到，使用 FP8 激活时，我们相对基线 FP16 获得了约 50% 的内存减少，这与把激活位宽降低 50% 的预期一致。此外，这里测得的 FP8 激活的质量损失可忽略不计（< 0.1%）。（注：降低精度带来的质量影响与模型相关、也与核函数相关。你应当用自己的数据与负载进行验证。）

当内存是主要瓶颈时，INT4 权重 + FP8 激活的工作流可产生约 ~1.8× 的基线吞吐量。INT4 权重 + FP4 激活可将内存减少到基线的 20%。4 位目标。加速比约为 3.5×，这与相对 FP16 的理论峰值 4× 提升一致。

动态精度切换的目标是在把输出质量保持在可接受范围内的同时最大化性能。理想情况下，核函数以尽可能最快的精度（例如 FP8 或 FP4）运行，只在必要时才回退到更高精度（例如 FP16）。在实践中，像 NVIDIA 面向 PyTorch 的 Transformer Engine 这样的库，允许在运行时对精度进行逐层控制。

线性层可能默认使用 FP8，但一个运行时钩子可以根据层的作用，把某层的精度提高到 FP16 或降低到 FP4。例如，FP4 可以应用于输出投影这类轻量层——在这些层上轻微的精度下降是可以容忍的——而 FP8 或 FP16 则可用于处理原始用户输入、并从更高精度中受益的早期层。

除了逐层的混合精度控制之外，你还可以使用一种更细粒度的优化策略——按 token 调整精度。这种方法让推理系统在预测有把握时以尽可能最快的模式（例如 FP8）运行。随后，当它更不确定时，便回退到 FP16 这类更高精度的模式。

在实践中，模型先以默认精度（例如 FP16）生成当前 token，然后基于诸如输出熵、最大 softmax 概率或 logit 方差等运行时指标来评估置信度。

如果模型对其预测高度自信，下一个 token 就可以用更低的精度处理。如果不确定性很高，系统则回退到更稳定的格式以维持输出质量。下面是演示该概念的示例代码：

```
import contextlib
import torch
# ----------------------------
# Safe Transformer Engine (TE) FP8 autocast import
# ----------------------------
try:
    # TE is only effective if your model actually uses TE-enabled layers
    # (e.g., Linear, LayerNorm wrappers).
    from transformer_engine.pytorch import fp8_autocast as _te_fp8_autocast
    # type: ignore
    _TE_AVAILABLE = True
except Exception:
    _TE_AVAILABLE = False
    # No-op stand-in so the code runs without TE installed. It never changes
    # numerical behavior.
    class _NullCtx(contextlib.ContextDecorator):
        def __init__(self, **_): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return False
    def _te_fp8_autocast(**_):
        return _NullCtx()
# ----------------------------
# Helper: choose the precision context *for this step* safely
# ----------------------------
def _precision_context_cuda(use_fp8: bool,
                            prefer_bfloat16: bool,
                            enable_fp8: bool):
    """
    Enter exactly one precision context. If FP8 isn't enabled or TE is
    missing/unused, fall back to AMP (BF16/FP16).
    """
    if use_fp8 and enable_fp8 and _TE_AVAILABLE:
        # Note: fp8_autocast affects only TE-enabled modules. Non-TE modules
        # run at their native dtypes.
        return _te_fp8_autocast(enabled=True)
    amp_dtype = torch.bfloat16 if prefer_bfloat16 else torch.float16
    return torch.autocast(device_type="cuda", dtype=amp_dtype)
def _precision_context(device: torch.device, use_fp8: bool,
                       prefer_bfloat16: bool, enable_fp8: bool):
    return _precision_context_cuda(use_fp8, prefer_bfloat16,
                                   enable_fp8) if device.type == "cuda"
                                   else contextlib.nullcontext()
# ----------------------------
# Main decode loop with smoothed, hysteretic precision switching
# ----------------------------
@torch.no_grad()
def decode_with_dynamic_precision(
    model,
    tokens: torch.Tensor,
    max_steps: int,
    *,
    device: torch.device = torch.device("cuda"),
    prefer_bfloat16: bool = True,      # B200: prefer BF16 over FP16 for AMP
    enable_fp8: bool = True,           # Allow FP8 when TE present
    enter_fp8_threshold: float = 6.0,  # hysteresis upper bound
                                       # (logit margin average)
    exit_fp8_threshold: float = 3.0,   # hysteresis lower bound (avoid flapping)
    reeval_interval: int = 8,          # compute/inspect confidence every N steps
                                       # to avoid per-step sync
    topk_dim: int = -1,                # last dimension holds vocabulary logits
    eos_id: int | None = None,
):
    """
    Autoregressive decode loop that *smoothly* switches between AMP (BF16/FP16)
    and FP8 (TE) without per-step host sync. Works even when TE is not
    installed; in that case, runs AMP only.
    - Confidence signal: mean(top1 - top2) logits margin across the batch.
    - Smoothing: EMA + interval re-evaluation to minimize CPU-GPU sync pressure.
    - Hysteresis: separate enter/exit thresholds to avoid precision flapping.
    """
    assert exit_fp8_threshold <= enter_fp8_threshold,
        "Hysteresis requires exit <= enter threshold"
    model.eval()
    tokens = tokens.to(device, non_blocking=True)
    # Internal state
    use_fp8: bool = False  # start in AMP.
                           # Upgrade to FP8 when sustained confidence permits
    ema_conf: torch.Tensor | None = None  # stays on device;
                                          # host consults only at intervals
    alpha = 0.2  # EMA smoothing factor for confidence
  # A tiny helper to update on-device EMA without host sync
    def _update_confidence_ema(logits: torch.Tensor) -> torch.Tensor:
        # logits: [B, vocab] or [B, T, vocab]. Use the last time-step if 3D.
        last = logits if logits.dim() == 2 else logits[:, -1, :]
        # Compute top-2 margin on-device
        top2 = torch.topk(last, k=2, dim=topk_dim).values  # [B, 2]
        margin = (top2[:, 0] - top2[:, 1]).mean()      # scalar tensor on device
        nonlocal ema_conf
        ema_conf = (1 - alpha)
                   * (ema_conf if ema_conf is not None else margin)+alpha*margin
        return ema_conf  # device scalar
    # Decode
    for step in range(max_steps):
        # 1) Precision context (exactly one).
        # No nested contexts, no leakage across iterations.
        with _precision_context(device, use_fp8, prefer_bfloat16, enable_fp8):
            # Forward pass (HF-style or plain)
            try:
                logits = model(input_ids=tokens)
                if hasattr(logits, "logits"):
                    logits = logits.logits
            except TypeError:
                logits = model(tokens)
            # 2) Pick next token from the *last* position
            last_step_logits = logits if logits.dim() == 2 else logits[:, -1, :]
            next_token = torch.argmax(last_step_logits, dim=-1,
                                      keepdim=True)  # [B, 1]
            tokens = torch.cat([tokens, next_token], dim=1)
        # 3) Update on-device EMA signal every step (no host sync yet)
        conf_dev = _update_confidence_ema(logits)
        # 4) Periodically re-evaluate precision choice on host
        # to avoid per-step sync
        if (step + 1) % reeval_interval == 0:
            conf_value = float(conf_dev)  # exactly one tiny sync every N steps
            if not use_fp8 and enable_fp8 and _TE_AVAILABLE
               and (conf_value > enter_fp8_threshold):
                use_fp8 = True
            elif use_fp8 and (conf_value < exit_fp8_threshold):
                use_fp8 = False
        # 5) EOS handling
        if eos_id is not None:
            if (tokens[:, -1] == eos_id).all():
                break
    return tokens
# ----------------------------
# Example (commented):
# ----------------------------
# model = ...  # your TE-enabled model (or any torch.nn.Module)
# input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
# out = decode_with_dynamic_precision(model, input_ids, max_steps=128,
# eos_id=tokenizer.eos_token_id)
# print(out.shape)
```

这里我们看到，截至本文撰写时，PyTorch autocast 只支持 FP16 与 BF16 这两种降精度格式。此时，你需要使用 Transformer Engine 库把受支持的模块路由到 FP8 核函数。

> 本例中使用的阈值（enter = 6.0, exit = 3.0）应当在验证集上用有代表性的提示进行校准，以防延迟收益影响精度。

这种模式创建了一种弹性的精度机制，并最大化吞吐量。当模型运行在可预测（例如低熵）区域时，例如生成标点或样板式的补全，它会继续使用 FP8 以最大化性能。当它进入更高熵的片段时，例如含糊的提示或决策点，它会回到 FP16 以保持数值精度。

当与现代 GPU 对低精度运算的支持相结合时，token 级的动态精度切换为高吞吐、延迟敏感的推理提供了一种自适应策略。它只在需要时才应用低精度，减少计算开销，并在许多不同的提示条件下维持响应质量。

## 面向 Transformer 自注意力与 MLP 通路的核函数自动调优

神经网络层在 GPU 上的性能，会因线程块（thread block）大小、分块维度、循环展开与内存访问模式等底层参数而剧烈变化。对于固定尺寸的模型，库通常只选择这些参数一次——往往使用通用的启发式方法或离线调优。

然而，在在线推理服务场景中，输入尺寸（包括序列长度与 batch size）会因请求而异。核函数自动调优（kernel autotuning）指的是一种运行时机制，它为当前负载选择——甚至即时（JIT）编译——最优的核函数变体。

在大型 Transformer 模型的语境下，推理的两大计算阶段是自注意力（self-attention）与前馈 MLP 层。两者都能从各自 GPU 核函数的自动调优中受益。我们逐一在核函数自动调优的语境下讨论它们。

考虑一个用 _H_ 个注意力头处理长度为 _L_ 的序列的注意力层。注意力有许多实现，包括标准注意力与经过优化的 FlashAttention——及其多个变体。

由于分块（tiling）、并行以及内存访问方面的改进，FlashAttention 及其变体在长序列上显著更快。不过，对于非常短的序列，它的开销可能超过收益。动态引擎可以根据序列长度 _L_ 在 FlashAttention 核函数与更简单的核函数之间切换。

例如，如果某个请求的 _L_ = 256 tokens，引擎可能使用一次性完成计算的直接核函数启动，通过全局内存读取来计算注意力，这对小 _L_ 已经足够。如果另一个请求带着 _L_ = 2,048 到来，它可以切换到 FlashAttention 专门的分块核函数——已知它通过在共享内存中复用数据、避免不必要的 HBM 数据取用，从而对大 _L_ 有更好的扩展性。这可以表示为一个基于输入序列长度的条件语句，如下所示：

```
// Note: example threshold shown here
if (seq_len < 256) {
    // global-memory version, best for small L
    attn_kernel = standard_attention_kernel;
} else {
    // tiled loads, best for large L
    attn_kernel = tiled_attention_kernel;
}
output = attn_kernel(Q, K, V, mask);
```

在幕后，attn_kernel 在完全不同的 CUDA 实现之间做选择。一种实现使用默认注意力核函数针对小输入做优化，另一种则使用分块核函数针对大上下文做优化。

理想的分块维度取决于你 GPU 的共享内存容量与计算资源。像 CUTLASS 与 OpenAI 的 Triton 这样的框架内置了自动调优器，会在初始化时——甚至在运行时自适应地——对一系列 (TILE_Q, TILE_K) 组合进行基准测试，以选出最快的变体。表 19-3 展示了不同分块大小在 Blackwell 级 GPU 上表现的示例。

表 19-3. 分块大小选择对共享内存占用、SM 占用率与实际吞吐量影响的示例（实际数值取决于线程块维度、时钟频率与其他微架构因素）

| 分块大小  | 共享内存 (KB) | 占用率 (%) | 吞吐量 (GOPS) |
| --------- | ------------- | ---------- | ------------- |
| 64 × 64   | 48            | 85         | 8.2           |
| 128 × 64  | 64            | 78         | 10.5          |
| 128 × 128 | 96            | 72         | 9.8           |
| 256 × 128 | 128           | 60         | 11.3          |

通过在运行时根据输入选择正确的变体，你可以避免一刀切方案带来的巨大性能悬崖。在实践中，你可能会在目标硬件上做基准测试，发现大约 _L_ = 128 是盈亏平衡点。

接下来，我们在自动调优的语境下分析前馈 MLP 核函数。前馈层本质上是大型矩阵乘法——具体来说，是中间夹着一个非线性激活的两个线性投影。

像 PyTorch 这样的现代 AI 框架使用高度优化的 GEMM 核函数，借助 cuBLAS 与 CUTLASS 这类优化过的 CUDA 库。对于给定的矩阵尺寸，这些库中往往有多个算法变体，它们使用不同的分块策略、不同的 Tensor Core 以及各自的回退通路。

例如，NVIDIA 的 cuBLAS 与 cuBLASLt 库可以对 GEMM 核函数进行自动调优——先尝试几种算法，再为给定维度挑选最快的算法。不过，这通常只发生在第一次遇到该形状的 GEMM 时——之后不再重新评估。

> 在 cuBLAS/cuBLASLt 或自定义核函数中可用之处，PDL（编程式依赖启动，programmatic dependent launch）可以减少启动间隙并改善稳态吞吐量。务必通过性能分析确认重叠效果。

在一个会遇到许多不同 batch size 的推理服务器中，可以显式地调用这类自动调优机制——或维护一个最优算法的缓存。例如，对于 MLP 中形状为 [batch_size, hidden_dim] x [hidden_dim, 4*hidden_dim] 的 GEMM，其最优核函数在 batch_size = 1 与 batch_size = 16 时可能不同。

引擎可以检测到新的 batch size，并用 cuBLASLt 或自定义实现对候选核函数运行一次快速的微基准测试（microbenchmark），以选出最快的核函数。之后使用该 batch size 的调用便可直接使用所选的核函数。

此外，一些推理框架与运行时使用 OpenAI 的 Triton GPU 核函数领域特定库（domain-specific library，DSL），以自动调优的分块大小即时编译注意力与 MLP 核函数。此时，运行时会生成核函数的若干个不同分块大小的变体（例如 128 × 128、64 × 256 等），并测量在实际硬件与输入形状下哪个表现更好。

你可以使用 Nsight Systems 这类工具，把不同的核函数变体并排进行经验性能分析。

具体而言，Nsight Systems 提供详细的 CUDA 时间线，包括 memcpy 与 NVLink 活动；Nsight Compute 则提供内存工作负载分析，帮助把缓存与内存行为归因到核函数位置。这在评估分块大小与共享内存权衡时尤其有用。此外，它常常能揭示诸如 L2 缓存未命中这类不明显的瓶颈，从而进一步指导你的调优决策。

> 由于 L2 缓存效应、内存 bank 冲突等原因，硬件在负载下有时会有些难以预测，因此经验调优总会胜过理论猜测。但用合理的理论值来开启调优过程是个不错的做法。

动态分块切换会影响 GPU 占用率（occupancy），在选择分块大小时应加以考虑。使用更大的分块可以增加复用并减少核函数启动开销，但也会使用更多寄存器（register）与共享内存。这可能会减少能够并发运行的线程块数量——从而降低占用率。一个合适的自动调优器会权衡这一取舍。

在注意力核函数中，更大的分块（例如 128 × 128）能最大化共享内存中的数据复用。这对长序列很理想，因为你发出的全局内存加载更少、摊薄了循环开销，并产生更高的持续吞吐量。

然而，对于较短的序列，同样的大分块会消耗过多共享内存，从而限制占用率，即每个 SM 上并发运行的线程块数量。通过为较短序列减小分块大小（例如 64 × 64），你可以释放共享内存，从而并行调度更多的块。这提升了 SM 占用率并降低了单核函数延迟。

通过根据输入序列长度自适应地调整分块大小，核函数在大多数情况下都能达到接近最优的占用率。有些系统甚至在运行时查询 CUDA Occupancy API，以动态选择诸如线程块大小之类的核函数启动参数。下面展示一个 C++ 中 Occupancy API 的示例，不过也有可用的 Python API：

```
// Pseudocode for occupancy-based launch configuration
int maxBlocks, bestThreads;
for (int threads = 64; threads <= 256; threads *= 2) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &maxBlocks, MyKernel, threads,
        sharedMemPerBlock(threads));
    // choose "threads" to maximize occupancy
    // (remember to not exceed the max threads per SM limit (e.g., 2,048)
    float occupancy = (float) maxBlocks * threads /
                              hardwareMaxThreadsPerSM;
}
```

这段伪 C++ 展示了如何为一个核函数评估不同的线程块大小。它检查在给定共享内存使用量的情况下，每个 SM 能运行多少个块。随后核函数启动会相应地调整线程数——或共享内存使用量。高性能框架与推理引擎会在内部用以下一组步骤将这类逻辑自动化：

_1. 测量负载_ 检查下一次模型前向的当前输入维度（batch size B、序列长度 L 等）。
