# Research Results

## Evaluation protocol

### Class sequence

| Stage | Classes |
|---|---|
| Base | person, chair, dining table |
| Incremental | laptop, book |

### Dataset sizes

| Component | Train images | Validation images |
|---|---:|---:|
| Base task | 132 | 25 |
| New-class task | 107 | 21 |
| Baseline replay memory | 104 | 22 |
| HoPe replay memory | 55 | 11 |
| Baseline incremental mixture | 211 | 43 |
| HoPe incremental mixture | 162 | 32 |

The baseline and HoPe experiments use different replay capacities. Their absolute
scores must therefore not be interpreted as a controlled architecture comparison.

### Head-preserving design

The initial design changed the pretrained 80-class detection head to three classes
and later to five classes. Those shape changes discarded classifier tensors and
made forgetting inseparable from classifier reinitialization.

The final protocol keeps the COCO80 head and activates class IDs 0, 56, 60, 63,
and 73. This design preserves compatible pretrained tensors across stages.

### Training

| Parameter | YOLO11n baseline | YOLO11n-HoPe |
|---|---:|---:|
| Base epochs | 8 | 20 |
| Incremental epochs | 15 | 25 |
| Image size | 480 | 480 |
| Batch size | 8 | 4 |
| Optimizer | AdamW | NestedAdamW |
| Frozen layers | 22 | 9 |
| Seed | 42 | 42 |
| Deterministic mode | Yes | Yes |

Training was performed on an AMD Ryzen 5 6600H CPU. Evaluation used an image size
of 640.

### Metrics

The primary metric is mAP50-95. Forgetting is:

```text
F = base old-class mAP50-95 - incremental old-class mAP50-95
```

A positive value indicates degradation. A value near zero indicates retention. A
negative value indicates that old-class performance improved after incremental
training.

Precision, recall, mAP50, and confusion matrices are also stored in `results/`.

## 1. YOLO11n baseline

| Model and group | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Base — old classes | 0.6480 | 0.5382 | 0.6281 | 0.3778 |
| No replay — old classes | 0.0867 | 0.8160 | 0.2987 | 0.1399 |
| No replay — new classes | 0.6716 | 0.6558 | 0.6781 | 0.5225 |
| Replay — old classes | 0.6717 | 0.5880 | 0.6538 | 0.3691 |
| Replay — new classes | 0.7891 | 0.6180 | 0.7329 | 0.5265 |

| Strategy | Before increment | After increment | Forgetting |
|---|---:|---:|---:|
| Without replay | 0.3778 | 0.1399 | 0.2379 |
| With replay | 0.3778 | 0.3691 | 0.0086 |

Replay reduced forgetting by:

```text
(0.2379 - 0.0086) / 0.2379 × 100% = 96.4%
```

New-class mAP50-95 was 0.5225 without replay and 0.5265 with replay. Replay did not
reduce new-class acquisition in this experiment.

![YOLO11n forgetting comparison](../results/baseline/forgetting-comparison.png)

## 2. YOLO11n-HoPe

| Strategy | Old mAP50-95 | New mAP50-95 | Forgetting |
|---|---:|---:|---:|
| HoPe base | 0.2037 | — | — |
| HoPe without replay | 0.0639 | 0.4349 | 0.1398 |
| HoPe with replay | 0.2572 | 0.4431 | -0.0535 |

Replay changed the old-class outcome from a 0.1398 decline to a 0.0535 gain. The
new-class score also increased by 0.0082.

These results establish the value of replay within the HoPe protocol. They do not
isolate the contribution of HoPe relative to standard YOLO11n because training
budgets, replay capacities, freeze policies, and optimizers differ.

## 3. Replay-free CMS ablations

| Strategy | Old mAP50-95 | New mAP50-95 | Forgetting |
|---|---:|---:|---:|
| Plain HoPe, no replay | 0.0639 | 0.4349 | 0.1398 |
| CMS V1 | 0.0570 | 0.4100 | 0.1467 |
| CMS V2, protected old channels | 0.0571 | 0.4058 | 0.1466 |
| CMS V3, strict parameter isolation | 0.2035 | 0.0101 | 0.0002 |
| CMS V4, routed detector memories | 0.2037 | 0.4097 | -0.00002 |
| CMS V5, conditional rank-32 delta | 0.2035 | 0.4099 | 0.00016 |

V1 and V2 learned the new task but did not retain the old task. V3 retained old
performance but lost almost all new-class capability. V4 resolved this
stability-plasticity conflict by routing old and new classes through separate
complete detector memories.

V5 compressed the plastic memory:

| Quantity | Value |
|---|---:|
| Changed tensors | 170 of 268 |
| Original delta elements | 2,556,560 |
| Stored delta elements | 738,272 |
| Delta compression | 3.46× |
| Relative reconstruction error | 0.52% |

Wrong-context probing reduced new-class mAP50-95 from 0.4099 to 0.0050, providing
evidence that the routing path is operational.

## 4. Replay-Fusion

Replay-Fusion uses the replay-trained HoPe model as a persistent teacher and the
no-replay model as a new-class specialist.

| Group | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Old classes | 0.7740 | 0.3944 | 0.5105 | 0.2573 |
| New classes | 0.6691 | 0.5096 | 0.6093 | 0.4487 |

Relative to the single HoPe replay model, the new-class score increased by 0.0056
and the old-class score increased by 0.00005. The fusion scale was checked on the
same validation split, so these differences require confirmation on a held-out
test set.

## 5. Consolidated findings

1. Fine-tuning only on new classes caused substantial catastrophic forgetting.
2. Balanced replay reduced YOLO11n baseline forgetting by approximately 96.4%.
3. Replay did not reduce new-class performance in either published replay
   protocol.
4. Preserving the COCO80 head removed a major experimental confound.
5. Parameter protection alone produced a stability-plasticity tradeoff.
6. Routed complete detector memories retained old performance while learning the
   new classes without old-image replay.
7. Low-rank conditional memory reduced storage while retaining V4-level accuracy.
8. The highest-scoring checkpoint remains replay-informed.

## 6. Limitations

- The study uses a small curated COCO subset.
- The primary experiments use one deterministic seed.
- Only one incremental stage is evaluated.
- Baseline and HoPe results use different training and replay budgets.
- Replay-Fusion lacks a separate held-out test set.
- No confidence intervals or statistical significance tests are reported.
- CMS V5 uses known class IDs rather than learned task-context inference.
- Routed memory inference evaluates more than one detector realization.
- The study does not establish performance in language, audio, reinforcement
  learning, or unrelated visual domains.

## 7. Result files

- [`results/baseline/`](../results/baseline/)
- [`results/hope/`](../results/hope/)
- [`results/cms/`](../results/cms/)
- [`results/replay-fusion/`](../results/replay-fusion/)
