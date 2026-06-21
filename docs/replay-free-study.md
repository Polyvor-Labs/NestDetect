# Replay-Free Continual Learning Study

## Scope

A bounded learner cannot preserve unlimited independent information without
revisiting data, growing state, compressing prior information, or accepting some
forgetting. NestDetect therefore treats replay-free learning as a memory-design
problem rather than an optimizer-only problem.

The evaluated alternatives preserve prior information through model state:

1. persistent CMS statistics and importance tensors;
2. protected or isolated classifier parameters;
3. separate routed detector memories;
4. compressed conditional parameter deltas.

## Experimental evidence

| Method | Old mAP50-95 | New mAP50-95 | Forgetting |
|---|---:|---:|---:|
| Plain HoPe, no replay | 0.0639 | 0.4349 | 0.1398 |
| CMS V1 | 0.0570 | 0.4100 | 0.1467 |
| CMS V2 | 0.0571 | 0.4058 | 0.1466 |
| CMS V3 | 0.2035 | 0.0101 | 0.0002 |
| CMS V4 | 0.2037 | 0.4097 | -0.00002 |
| CMS V5 | 0.2035 | 0.4099 | 0.00016 |

V3 demonstrates that strict stability can eliminate plasticity. V4 demonstrates
that separate complete representations can remove direct interference. V5
demonstrates that the plastic representation can be compressed without materially
changing the current result.

## CMS V5

V5 stores:

- one immutable persistent detector;
- exact expert buffers;
- low-rank factors for the plastic-minus-persistent parameter delta;
- a class-context routing mask.

```text
theta_expert(context) = theta_base + delta_theta_context
```

The rank-32 experiment stores 738,272 delta elements instead of 2,556,560 and has a
relative reconstruction error of 0.52%.

## What the result does not establish

- Routing is based on known class IDs rather than inferred context.
- Storage grows when unrelated experts are added.
- Only one incremental transition is tested.
- Positive backward transfer is not demonstrated by replay-free CMS V5.
- Inference requires persistent and plastic predictions.
- Object-detection evidence does not establish domain-general continual learning.

## Required follow-up experiments

1. Replace class-ID routing with a learned context encoder.
2. Evaluate at least five sequential class groups.
3. Report average accuracy, average forgetting, forward transfer, storage growth,
   and routing error across multiple seeds.
4. Compare against equal-memory replay, generative replay, adapters,
   hypernetworks, and joint-training upper bounds.
5. Add expert composition and uncertainty-based expert allocation.
6. Evaluate independently collected domains and additional modalities.

## References

- Nested Learning / HoPe: <https://arxiv.org/abs/2512.24695>
- Continual Learning with Hypernetworks: <https://arxiv.org/abs/1906.00695>
- CL-LoRA: <https://arxiv.org/abs/2505.24816>
- C-LoRA: <https://arxiv.org/abs/2502.17920>
- Rehearsal-Free Modular and Compositional Continual Learning:
  <https://arxiv.org/abs/2404.00790>
- Efficient Parameter Mining and Freezing for Continual Object Detection:
  <https://arxiv.org/abs/2402.12624>
- Incremental Learning of Object Detectors without Catastrophic Forgetting:
  <https://arxiv.org/abs/1708.06977>
