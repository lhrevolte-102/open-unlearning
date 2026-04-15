# 基于 Selective DPO 改造 IdkDPO / NPO 的方法文档

## 1. 问题背景与目标

本文讨论的是一个**方法迁移**问题：如何将 *Principled Data Selection for Alignment: The Hidden Risks of Difficult Examples* 中的 **Selective DPO** 思想，迁移到 *OpenUnlearning: accelerating LLM unlearning via unified benchmarking of methods and metrics* 所统一描述的 **IdkDPO** 与 **NPO** 两类 unlearning 方法上。需要先说明，这里的 **Selective-IdkDPO** 与 **Selective-NPO** 不是原论文已经提出的方法，而是在两篇论文基础上的一种合理延伸。Selective DPO 的核心发现是：样本存在稳定难度差异，过难样本会伤害训练，而且模型容量决定其可承受的难度阈值；其基本做法是通过 held-out validation loss 对样本排序，再优先在较容易样本上训练。 

本文的目标，是将这种“**难度感知的数据选择**”改造为适用于 unlearning 的“**分阶段遗忘策略**”，以改善 IdkDPO / NPO 在 OpenUnlearning 框架中表现出的不平衡现象：在 TOFU 上，这两类方法的 utility 并不差，但 privacy 明显偏弱，说明训练过程可能存在 under-unlearning 或不稳定遗忘。

---

## 2. 为什么不能直接照搬 Selective DPO

Selective DPO 的原始流程可以概括为三步：训练若干 reference models，利用 held-out validation loss 估计样本难度，并只在最低验证损失的那部分样本上进行 DPO 训练。文中明确给出，其 reference models 来自随机二分后的多次重复训练，总计六个；validation loss 则作为 learned step 的廉价代理，用于对样本按难度升序排序。 

但在 unlearning 中，这种“只保留 easiest (\tau%) 数据”的做法不能直接照搬。原因在于：alignment 任务里，被过滤掉的样本可以视为“不参与偏好学习”；而 unlearning 任务里，forget set 中的每一个样本都对应一个必须处理的删除目标。若直接永久丢弃困难 forget samples，那么模型并没有真正忘掉这些样本。

因此，迁移时必须把原先的“hard filtering”改写为：

[
\text{不是永久丢弃难样本，而是先易后难、逐步引入、最终全覆盖。}
]

也就是说，Selective DPO 在 unlearning 中的合理对应物，不是“选 easy subset 并停止”，而是“构造 easy-to-hard 的 forget curriculum”。这仍然保留了原论文的基本洞见：困难样本会拖累训练，而且这种拖累并不只是错标、分布偏移或学习率失配，更像是当前模型能力与当前训练状态下的容量超载。

---

## 3. 两篇论文中可迁移的核心结论

### 3.1 来自 Selective DPO 的结论

Selective DPO 的结论可以浓缩为三点。第一，样本难度确实存在，而且这种难度并非偶然噪声；论文通过 learned step 和 validation loss 的一致性说明了这一点。第二，高 validation loss 的困难样本会显著拖累训练表现。第三，难度阈值与模型容量相关：较大模型能够承受更高比例的困难样本。 

对 unlearning 的启发是，forget samples 不应默认等价。它们很可能在“容易遗忘”与“困难遗忘”之间存在结构性差异，而这种差异会影响 IdkDPO / NPO 的训练稳定性。因此，困难 forget samples 不应在训练一开始就以与 easy samples 相同的方式进入优化过程。

### 3.2 来自 OpenUnlearning 的结论

OpenUnlearning 提供了两个关键信息。其一，它统一给出了 IdkDPO 与 NPO 的目标函数形式。其二，在 TOFU 上的比较显示，IdkDPO 与 NPO 的问题并不主要在 utility，而在 privacy 维度上明显偏弱；相比之下，SimNPO 之所以表现更好，在于它在遗忘与保留之间取得了更均衡的折中。 

这意味着：对 IdkDPO / NPO 的改造，核心目标不应只是“让 forgetting 更强”，而应是**让训练过程更平衡、更稳定，从而改善 privacy，同时尽量不牺牲 utility**。此外，OpenUnlearning 也强调，unlearning 评测本身需要同时关注 forgetting、privacy 与 utility，而不是只看是否“答不出来”。 

---

## 4. 原始 IdkDPO / NPO 的目标函数

设 target model 为 (f_{target})，unlearned model 为 (f_{unl})，forget set 为 (D_f)，retain set 为 (D_r)。

### 4.1 IdkDPO

OpenUnlearning 中，IdkDPO 的 forget 部分可写为：

[
\mathcal{L}^{F}_{\text{IdkDPO}}
===============================

-\frac{2}{\beta}
\log \sigma \left(
-\beta \log \frac{p(y_{idk}\mid x; f_{unl})}{p(y_{idk}\mid x; f_{target})}
-\beta \log \frac{p(y_f\mid x; f_{unl})}{p(y_f\mid x; f_{target})}
\right)
]

它的含义是：对于 forget query，模型既要相对更倾向于输出 (y_{idk})，又要相对更不倾向于输出原始 forget answer (y_f)。

### 4.2 NPO

NPO 的 forget 部分则为：

[
\mathcal{L}^{F}_{\text{NPO}}
============================

-\frac{2}{\beta}
\log \sigma \left(
-\beta \log \frac{p(y_f\mid x; f_{unl})}{p(y_f\mid x; f_{target})}
\right)
]

它只保留负反馈项，因此在 OpenUnlearning 的表述中被认为训练上比类似 GradDiff 的方法更稳定。

### 4.3 保留项

两者都需要额外配合 retain NLL：

[
\mathcal{L}
===========

\mathbb{E}*{z \sim D_f}[\mathcal{L}^{F}(z)]
+
\alpha \mathbb{E}*{(x,y)\sim D_r}[\ell(y\mid x; f_{unl})]
]

这体现了 OpenUnlearning 所强调的 unlearning 双目标：一方面移除 forget influence，另一方面维持 retain/general utility。

---

## 5. Selective 版本的核心改造

## 5.1 方法相关难度，而不是通用难度

本文的基本观点是：在 unlearning 中，样本“难不难忘”不是一个与目标函数无关的固定属性，而是**相对具体 unlearning objective 而言**的。

因此，定义 forget sample 的难度时，不应直接沿用原始 DPO validation loss，而应使用**方法相关的 held-out forget loss**：

* 对 IdkDPO，用 held-out IdkDPO forget loss 作为难度；
* 对 NPO，用 held-out NPO forget loss 作为难度。

这相当于把 Selective DPO 中“validation loss 代理 difficulty”的思想，迁移为“**held-out unlearning loss 代理 forgetting difficulty**”。

---

## 5.2 样本难度定义

设 forget sample 为

[
z_i=(x_i,y_i^f)
]

若采用 IdkDPO，则还需要对应的拒答目标 (y_i^{idk})。

参考 Selective DPO 的 reference-model 思路，构造 (K) 个 held-out reference unlearners，对每个样本计算未见过该样本的模型上的 forget loss，并取平均，得到难度分数。

### Selective-IdkDPO 的样本难度

[
d_i^{\text{IdkDPO}}
===================

\frac{1}{K}
\sum_{k=1}^{K}
\mathcal{L}^{F}_{\text{IdkDPO}}(z_i; \hat f_k)
]

### Selective-NPO 的样本难度

[
d_i^{\text{NPO}}
================

\frac{1}{K}
\sum_{k=1}^{K}
\mathcal{L}^{F}_{\text{NPO}}(z_i; \hat f_k)
]

这里 (\hat f_k) 表示 held-out reference unlearner。这样的构造直接对应 Selective DPO 中“随机二分 + 多次重复 + held-out validation”的思想，只不过把 alignment validation loss 换成了 unlearning loss。

---

## 6. 最终方法：Selective-IdkDPO / Selective-NPO

## 6.1 总体思想

在得到每个 forget sample 的难度分数后，将 forget set 按 (d_i) 从小到大排序，形成一个 forget curriculum。训练时不再一次性使用整个 (D_f)，而是在多个 stage 中逐步放宽允许进入训练的 forget samples 范围。

设 (r_i \in [0,1]) 为样本 (z_i) 的难度分位数，数值越小表示越容易。给定阶段阈值

[
0 < \tau_1 < \tau_2 < \cdots < \tau_S = 1
]

定义第 (s) 个阶段的门控函数为

[
g_s(i)=\mathbf{1}[r_i \le \tau_s]
]

其中 (\tau_S=1) 是一个关键约束，它保证在最后阶段时整个 forget set 被完全纳入训练。这个约束使得该方法仍然满足严格 unlearning 的要求，而不是退化为 alignment 式的数据子集选择。

---

## 6.2 Selective-IdkDPO 的阶段目标

在第 (s) 个阶段，Selective-IdkDPO 的训练目标写为

[
\mathcal{L}_s^{\text{Sel-IdkDPO}}
=================================

\mathbb{E}*{z_i\sim D_f}
[g_s(i),\mathcal{L}^{F}*{\text{IdkDPO}}(z_i)]
+
\alpha_s \mathbb{E}*{(x,y)\sim D_r}[\ell(y\mid x; f*{unl})]
]

其含义是：在早期阶段，模型只在难度低于当前阈值的 forget samples 上做 IdkDPO 遗忘；随着阶段推进，越来越多更难的样本被纳入优化；最终阶段覆盖全部 forget samples。

---

## 6.3 Selective-NPO 的阶段目标

对应地，Selective-NPO 的训练目标为

[
\mathcal{L}_s^{\text{Sel-NPO}}
==============================

\mathbb{E}*{z_i\sim D_f}
[g_s(i),\mathcal{L}^{F}*{\text{NPO}}(z_i)]
+
\alpha_s \mathbb{E}*{(x,y)\sim D_r}[\ell(y\mid x; f*{unl})]
]

与 Selective-IdkDPO 的区别只在于 forget loss 的形式不同。整体策略仍然是：在 easy-to-hard 的 curriculum 下逐步扩大 forget coverage，最后达到全覆盖。

---

## 7. 这种改造为什么可能有效

### 7.1 对 IdkDPO 的作用机制

IdkDPO 的 forget 优化同时包含两个方向：提升 (p(y_{idk}\mid x)) 的相对偏好，以及压低 (p(y_f\mid x)) 的相对偏好。换言之，它要同时学会“拒答”与“去偏向原答案”。这使它比 NPO 的优化结构更复杂，也更容易在训练初期被困难 forget samples 干扰。将 easy samples 优先引入，相当于先让模型建立一个较稳定的 refusal / forgetting 方向，再逐步扩展到 harder samples，因此理论上更容易提升稳定性。

### 7.2 对 NPO 的作用机制

NPO 虽然只包含负反馈项，理论上更稳定，但 OpenUnlearning 的结果表明它在 privacy 上仍然明显偏弱。
这说明问题未必是训练崩溃，而可能是：困难 forget samples 在训练早期将优化推向不均衡状态，使模型在“压低 forget likelihood”与“保持 retain-like behavior”之间没有得到足够好的折中。若先在 easy samples 上建立更平滑的遗忘轨迹，再逐步引入 harder samples，NPO 也可能因此获得更好的 privacy–utility balance。

### 7.3 为什么必须最终全覆盖

这一点是区分“alignment 式 selective training”与“strict unlearning”最核心的原则。若训练永远只使用前 (\tau<1) 的 forget samples，那么模型只是对 forget set 的一个子集进行了遗忘，而不是对全部删除请求进行了处理。因此，本方法的本质不是筛掉 hard samples，而是控制它们**何时**进入训练，而不是决定它们**是否**进入训练。

---

## 8. 评测目标与研究假设

在 OpenUnlearning 的框架下，unlearning 方法的好坏不能只看 memorization，还必须联合考虑 privacy 与 utility。TOFU 上的结果已经说明，IdkDPO / NPO 的短板主要在 privacy，而不在 utility。 因此，对 Selective-IdkDPO / Selective-NPO 的主要研究假设应表述为：

[
\text{难度感知的 staged curriculum 能将 IdkDPO / NPO 拉向更均衡的 forgetting–privacy–utility 折中。}
]

更具体地说，本文预期它应满足以下趋势：

[
\text{Privacy}*{\text{Selective}} > \text{Privacy}*{\text{Baseline}}
]

同时尽量保持

[
\text{Utility}*{\text{Selective}} \approx \text{Utility}*{\text{Baseline}}
]

并避免 memorization 指标明显回退。

此外，Selective DPO 原文强调 difficulty threshold 与 model capacity 相关，因此在 unlearning 中也可进一步提出一个容量相关假设：随着模型容量增大，最优 curriculum 中较大 (\tau) 阶段应更早、更平滑地进入训练。这个假设并非原论文直接验证于 unlearning，但从其 alignment 结果看具有明确动机。

---

## 9. 方法的边界与注意事项

第一，样本难度很可能是**目标函数相关**的。同一个 forget sample，对 IdkDPO 来说可能很难，对 NPO 来说未必同样难。因此，不应假设存在一个可在所有 unlearning objectives 间共享的“通用 forgetting difficulty”。

第二，Selectivity 不能被理解为“跳过困难删除请求”。它只能被理解为一种**训练顺序控制策略**，而不能改变最终需要处理的 forget set 范围。

第三，若未来要进一步扩展该方案，最自然的方向不是增加更多工程技巧，而是分析：这种方法相关 difficulty 是否能与 OpenUnlearning 中更可靠的 memorization / robustness 指标建立更直接的联系。OpenUnlearning 已明确指出，好的 unlearning evaluation 需要同时兼具 faithfulness 和 robustness，这意味着“训练时的 difficulty”与“评测时的可靠 forgetting”之间仍有待进一步建立更强联系。 
