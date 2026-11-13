# Implementation clarifications

This note describes the [WAF implementation at commit 82c8cec][source] and clarifies diagnostic conventions used alongside the [paper and supplement][paper]. It is an implementation reference, not a replacement for the published record.

## Gating and historical results

AdaptiveBiMaPLe length/depth gates and CoOp scalar gates are learned global parameters, independent of the input image. Their cross-image constancy follows from the architecture. Conditional CoCoOp instead applies a linear gate head to normalized image features; a separate meta-network supplies the prompt bias. See [CoOp][coop] and [CoCoOp][cocoop].

Original ImageNet run records for seeds 1, 16, and 22 report conditional scalar CoCoOp gating with trainable gate-head parameters and `N_CTX=4`. Their recorded Base/Novel results reproduce the published ImageNet Table VI summaries at reported precision: original CoCoOp **73.02 ± 0.22** and conditional gating **72.98 ± 0.21**. These are the mean and sample standard deviation of three per-seed harmonic means, `2BN/(B+N)`.

## Soft effective length

For scalar gates, soft effective length is `sum_i sigmoid(g_i)`. Divide by the **resolved context-token count** to obtain mean activation; the sum is not a hard retained-token count.

| Model | Reported soft sum (V-B2) | Supported context count | Implied mean activation |
|---|---:|---|---:|
| CoOp | approximately 7.96 | 16 in the public gated ImageNet recipe | 0.4975 |
| Conditional CoCoOp | approximately 1.995 | 4 in the original ImageNet run records | 0.49875 |

The [CoOp recipe][recipe] selects `vit_b16.yaml`, which does not override the [defaults][defaults] `N_CTX=16` and `CTX_INIT=""`. The corresponding [CoCoOp configuration][ccconfig] specifies four tokens and the four-word initialization “a photo of a”. Both constructors can replace `N_CTX` with the word count of a nonempty `CTX_INIT`; use the resolved count for other configurations.

Thus, the printed denominators 8 and 2 are not the maxima for these 16- and 4-token configurations. The reported sums are consistent with mean activation near **0.5**, rather than near-maximum activation. This distinction concerns the denominators and their interpretation, not established errors in the measured sums. The original CoOp diagnostic run underlying V-B2 has not been separately matched to its resolved configuration; the 16-token interpretation follows the public recipe.

## Reading the diagnostics

The [conditional CoCoOp logger][ccstats] averages B×L scalar activations across images before computing `gate/std_p` across tokens. It therefore does not measure cross-image variance, even when B=1. Stable sums and small token-wise standard deviation alone do not establish parameter-update inactivity or functional input independence.

The [ABM diagnostic][diagnostic] uses pooled group L2 gradient norms. CoOp/CoCoOp training loggers use L2 divided by parameter count, **not RMS**. Their measurements follow backward, AMP unscaling where applicable, and optional gradient equalization; improved ABM's non-AMP debug norms also follow scaling/clipping. Parameter groups, normalization, and capture points matter when comparing values. Norm differences alone do not establish a unique causal optimization barrier. WAF's frozen-prompt setting and distinction from MoE remain relevant.

## Entropy and freeze variants

Define Bernoulli entropy as `H(p) = -sum[p log p + (1-p) log(1-p)]`. The [improved ABM helper][entropy] adds `-lambda H` to the minimized loss, encouraging high entropy for positive lambda. Its docstring and supplement D3(b) display the opposite sign. Separate, default-disabled CoOp/CoCoOp helpers add positive entropy; the CoCoOp helper returns zero for a conditional head. These are distinct interventions.

Clearing `.grad` before backward in the [base ABM loop][base] does not implement its commented freeze. The [improved warm-up][warmup], AlwaysFrozen, and FreezeEarly [variants][freeze] use `requires_grad` controls. AlwaysFrozen also changes initial gate logits.

The public code conditions CoCoOp gates directly on normalized image features and computes ABM depth smoothness on raw depth parameters; the descriptions in III-D/B3 and III-C/B2, respectively, differ. These reporting distinctions, the entropy sign, and effective-length denominators are noted for clarification of the publication record.

[source]: https://github.com/MiCrSYZ/when_adaptation_fails/tree/82c8cec11151f5c1a8e6f3e5e650d0c04661dab7
[paper]: https://arxiv.org/abs/2605.09549
[coop]: https://github.com/MiCrSYZ/when_adaptation_fails/blob/82c8cec11151f5c1a8e6f3e5e650d0c04661dab7/trainers/coop.py#L68-L124
[cocoop]: https://github.com/MiCrSYZ/when_adaptation_fails/blob/82c8cec11151f5c1a8e6f3e5e650d0c04661dab7/trainers/cocoop.py#L128-L238
[recipe]: https://github.com/MiCrSYZ/when_adaptation_fails/blob/82c8cec11151f5c1a8e6f3e5e650d0c04661dab7/scripts/coop-gating/base2new_train.sh
[defaults]: https://github.com/MiCrSYZ/when_adaptation_fails/blob/82c8cec11151f5c1a8e6f3e5e650d0c04661dab7/train.py#L131-L133
[ccconfig]: https://github.com/MiCrSYZ/when_adaptation_fails/blob/82c8cec11151f5c1a8e6f3e5e650d0c04661dab7/configs/trainers/CoCoOp/vit_b16_c4_ep10_batch1_ctxv1.yaml
[ccstats]: https://github.com/MiCrSYZ/when_adaptation_fails/blob/82c8cec11151f5c1a8e6f3e5e650d0c04661dab7/trainers/cocoop.py#L550-L573
[diagnostic]: https://github.com/MiCrSYZ/when_adaptation_fails/blob/82c8cec11151f5c1a8e6f3e5e650d0c04661dab7/diagnostics/diagnostic_experiments.py#L94-L121
[entropy]: https://github.com/MiCrSYZ/when_adaptation_fails/blob/82c8cec11151f5c1a8e6f3e5e650d0c04661dab7/trainers/improved_adaptive_bidir.py#L81-L121
[base]: https://github.com/MiCrSYZ/when_adaptation_fails/blob/82c8cec11151f5c1a8e6f3e5e650d0c04661dab7/trainers/adaptive_bidir_maple.py#L599-L618
[warmup]: https://github.com/MiCrSYZ/when_adaptation_fails/blob/82c8cec11151f5c1a8e6f3e5e650d0c04661dab7/trainers/improved_adaptive_bidir.py#L130-L138
[freeze]: https://github.com/MiCrSYZ/when_adaptation_fails/blob/82c8cec11151f5c1a8e6f3e5e650d0c04661dab7/trainers/implicit_reg_trainers.py#L537-L780
