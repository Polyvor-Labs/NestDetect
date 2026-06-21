# Architecture

## System overview

NestDetect provides two experimental tracks:

1. a YOLO11n replay baseline;
2. YOLO11n-HoPe with Self-Referential Titans, a Continuum Memory System (CMS),
   and optional routed detector memories.

Both tracks preserve the pretrained COCO80 detection head and activate only the
five target class IDs.

```text
COCO subset
    ├── base task: person, chair, dining table
    │       ├── base training
    │       └── balanced replay memory
    └── incremental task: laptop, book
            ├── no-replay control
            ├── replay training
            └── replay-free CMS ablations
```

## Package components

| Module | Responsibility |
|---|---|
| `config.py` | configuration loading and project-relative path resolution |
| `dataset.py` | YOLO dataset validation, remapping, and materialization |
| `memory.py` | deterministic balanced replay sampling |
| `training.py` | Ultralytics training adapter and checkpoint metadata |
| `hope.py` | HoPe, Self-Referential Titans, CMS, and routed detector memories |
| `conditional_memory.py` | low-rank conditional parameter regeneration |
| `consolidation.py` | persistent CMS state and parameter-importance artifacts |
| `optim.py` | NestedAdamW and M3 optimizers |
| `trainer.py` | HoPe-aware and continual-learning training logic |
| `evaluation.py` | metrics, forgetting, confusion matrices, and plots |
| `inference.py` | image and BGR-frame inference |
| `cli.py` | command-line interface |
| `app.py` | Streamlit and WebRTC application |

## Head-preserving class-incremental learning

Changing a COCO-pretrained detector from 80 classes to three classes and later to
five classes changes classifier tensor shapes. In the initial implementation,
those changes discarded pretrained classification tensors and confounded the
forgetting experiment.

NestDetect keeps `nc=80` throughout all stages:

```yaml
classes: [0, 56, 60, 63, 73]
```

This preserves compatible pretrained tensors, keeps head shapes stable across
tasks, and makes forgetting measurements reflect incremental training rather than
classifier reinitialization.

## Replay memory

Replay memory uses deterministic greedy balanced sampling. Each pass selects an
image for a class whose quota has not been met; multi-label images may satisfy
multiple class quotas.

Two capacities are used:

| Protocol | Train quota per old class | Validation quota per old class |
|---|---:|---:|
| YOLO11n baseline | 60 | 12 |
| YOLO11n-HoPe | 30 | 6 |

The separate capacities preserve the protocols used to produce the published
baseline and HoPe results.

## YOLO11n-HoPe

`configs/models/yolo11n-hope.yaml` replaces the P5 attention bottleneck with
`Hope2D`:

```text
image
  → YOLO11 convolutional backbone
  → SPPF
  → Hope2D
      ├── local depth-wise convolution
      ├── spatial tokenization
      ├── Self-Referential Titans
      └── sequential CMS
  → multi-scale detection neck
  → COCO80 detection head
```

At an input size of 480, the P5 feature map contains 225 spatial tokens. Keeping
HoPe at P5 bounds the cost of per-image fast-memory updates.

### Self-Referential Titans

The implementation adapts the HoPe sequence mechanism to spatial feature tokens.
It includes residual key, value, query, learning-rate, retention, and associative
memories; normalized keys and queries; chunk-wise fast updates; learned retention;
momentum; and DGD-style covariance decay.

Fast weights are created independently for each image and are reset on the next
forward call. Checkpoint parameters provide the meta-learned initial state and are
not mutated during inference.

### Continuum Memory System

The sequential CMS contains three residual MLP levels:

| Level | Optimizer update period | Role |
|---|---:|---|
| Fast | 1 batch | rapid adaptation |
| Medium | 4 batches | intermediate memory |
| Slow | 16 batches | persistent memory |

`NestedAdamW` accumulates gradients between scheduled updates and applies their
sum when a level is due.

## Replay-free routed memories

### CMS V4

CMS V4 uses two complete detector memories:

```text
input
  ├── immutable base detector ── person, chair, dining table
  └── no-replay detector ─────── laptop, book
                    class router
                         ↓
                  combined detections
```

Each memory retains its own backbone, neck, classifier, and box regressor. This
resolves the V3 failure mode, where protecting the shared detector retained old
classes but removed the capacity required to localize new classes.

### CMS V5

CMS V5 stores the plastic detector as a low-rank parameter delta from the
persistent base detector:

```text
base parameters + context-selected low-rank delta → plastic realization
```

The rank-32 checkpoint stores 738,272 delta elements instead of 2,556,560, a
3.46× compression. It retains V4-level accuracy with an approximately 6.9 MiB
checkpoint.

## Replay-Fusion

The release checkpoint uses the replay-trained HoPe detector as its persistent
teacher and the no-replay detector as a new-class specialist. New-class candidates
from both memories are fused by non-maximum suppression.

This model is replay-informed. It must not be described as replay-free.

## Inference

The application filters predictions to class IDs `[0, 56, 60, 63, 73]` and changes
the display name `dining table` to `table`. Loaded models are cached by resolved
path and file modification time.
