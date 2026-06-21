# Model Card

## Overview

NestDetect publishes five checkpoints for a five-class, class-incremental object
detection study. All checkpoints retain the COCO80 detection head and expose the
target classes `person`, `chair`, `table`, `laptop`, and `book`.

## Checkpoints

| Model | Size | Old mAP50-95 | New mAP50-95 | Forgetting |
|---|---:|---:|---:|---:|
| `nestdetect_final.pt` | 5.3 MiB | 0.3691 | 0.5265 | 0.0086 |
| `nestdetect_hope.pt` | 5.1 MiB | 0.2572 | 0.4431 | -0.0535 |
| `nestdetect_hope_cms_only_v4.pt` | 10.3 MiB | 0.2037 | 0.4097 | -0.00002 |
| `nestdetect_hope_cms_v5.pt` | 6.9 MiB | 0.2035 | 0.4099 | 0.00016 |
| `nestdetect_hope_cms_v4_replay_fusion.pt` | 10.3 MiB | 0.2573 | 0.4487 | -0.0536 |

The metrics come from the curated validation protocol described in
[research-report.md](research-report.md).

## Recommended use

- Use CMS V4 for the web application's replay-free prediction tests.
- Use Replay-Fusion for the strongest current validation score.
- Use CMS V5 when evaluating the replay-free parameter-memory design.
- Use `nestdetect_hope.pt` for a single-detector HoPe replay comparison.
- Use `nestdetect_final.pt` for the original YOLO11n replay baseline.

## Provenance

Replay-Fusion combines a replay-trained persistent detector with a no-replay
plastic specialist. It is not replay-free.

CMS V5 stores no old-task images in its checkpoint. It retains one immutable base
detector and a compressed low-rank delta for the plastic detector. Storage still
grows when additional independent experts are introduced, so it is not a
fixed-capacity solution to unbounded continual learning.

## SHA-256

| Checkpoint | SHA-256 |
|---|---|
| `nestdetect_final.pt` | `fc2bd88048960dca96044da9bd177aa0da9cbfae3ae17580e7076c126e081ff5` |
| `nestdetect_hope.pt` | `bc220eeb4fe73af4b362e88b4065000b824a0541432372cb94a337b8a18ec2ec` |
| `nestdetect_hope_cms_only_v4.pt` | `6642d4a67a5cdcddaa914a80dd7c0fc906a1a4ca9c608abac17bab11027041e0` |
| `nestdetect_hope_cms_v5.pt` | `838252bae3f5f9df59749e32693dd8e10a7aec9e132bd26af98ed954ae93666a` |
| `nestdetect_hope_cms_v4_replay_fusion.pt` | `7cc3eb592d566133f6ac62d9d4cb5d4a4e84d9f09828ca01285385e10a012db1` |

## Intended use

The checkpoints are intended for research, education, and demonstrations of
class-incremental object detection. They are not validated for safety-critical
deployment, surveillance decisions, biometric identification, or decisions
affecting rights and access to services.

## Limitations

- The dataset is a small curated COCO subset.
- The primary experiments use one deterministic seed.
- The study contains one incremental stage.
- Replay-Fusion was evaluated and tuned on the same validation split.
- No separately collected classroom-domain test set is included.
- CMS V5 uses known class IDs for routing rather than inferred task context.
- Reported metrics do not establish performance in other domains or modalities.
