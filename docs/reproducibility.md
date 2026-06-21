# Reproducibility

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest
```

Original environment:

- CPU: AMD Ryzen 5 6600H;
- RAM: 8 GB;
- Python: 3.12;
- Ultralytics: 8.4.72;
- PyTorch: 2.9.1;
- seed: 42;
- deterministic mode: enabled.

Library, hardware, and numerical-kernel differences may produce small metric
changes.

## 2. Build the source subsets

```bash
python scripts/prepare_coco_subset.py \
  --train-per-class 60 \
  --val-per-class 12 \
  --seed 42
```

This creates:

```text
datasets/step1_base/
datasets/step2_new_classes/
```

## 3. Build the experimental datasets

```bash
python scripts/build_datasets.py
```

The builder creates and validates:

```text
datasets/base_coco80/
datasets/new_coco80/
datasets/incremental_no_replay_coco80/
datasets/incremental_replay_baseline_coco80/
datasets/incremental_replay_hope_coco80/
```

The baseline mixture uses replay quotas 60/12. The HoPe mixture uses 30/6.

## 4. YOLO11n baseline

```bash
nestdetect train configs/baseline/base.yaml
nestdetect train configs/baseline/incremental-no-replay.yaml
nestdetect train configs/baseline/incremental-replay.yaml

python scripts/evaluate_baseline.py
python scripts/plot_results.py \
  --results-dir results/baseline \
  --title "NestDetect YOLO11n Baseline"
```

Expected primary results:

| Strategy | Old mAP50-95 | New mAP50-95 | Forgetting |
|---|---:|---:|---:|
| Without replay | 0.1399 | 0.5225 | 0.2379 |
| With replay | 0.3691 | 0.5265 | 0.0086 |

## 5. YOLO11n-HoPe

```bash
nestdetect train configs/hope/base.yaml
nestdetect train configs/hope/incremental-no-replay.yaml
nestdetect train configs/hope/incremental-replay.yaml

python scripts/evaluate_hope.py
python scripts/plot_results.py \
  --results-dir results/hope \
  --title "NestDetect YOLO11n-HoPe"
```

Expected primary results:

| Strategy | Old mAP50-95 | New mAP50-95 | Forgetting |
|---|---:|---:|---:|
| Without replay | 0.0639 | 0.4349 | 0.1398 |
| With replay | 0.2572 | 0.4431 | -0.0535 |

## 6. Persistent CMS ablations

Build the persistent CMS and importance artifact:

```bash
python scripts/consolidate_cms.py
```

Train V1-V3 and the head-wise ablation:

```bash
nestdetect train configs/ablations/cms-v1.yaml
nestdetect train configs/ablations/cms-v2.yaml
nestdetect train configs/ablations/cms-v3.yaml
nestdetect train configs/ablations/cms-headwise.yaml
```

Build routed CMS V4 and compressed CMS V5:

```bash
python scripts/build_cms_v4.py
python scripts/build_cms_v5.py \
  --max-rank 32 \
  --minimum-energy 0.999
```

Evaluate:

```bash
python scripts/evaluate_cms.py
python scripts/plot_results.py \
  --results-dir results/cms \
  --title "NestDetect Replay-Free CMS Ablations"
```

## 7. Replay-Fusion

After the HoPe replay and no-replay checkpoints exist:

```bash
python scripts/build_cms_v4.py \
  --persistent models/nestdetect_hope.pt \
  --plastic artifacts/checkpoints/incremental_hope_no_replay_coco80.pt \
  --output models/nestdetect_hope_cms_v4_replay_fusion.pt \
  --fuse-persistent-plastic-classes \
  --plastic-score-scale 1.0
```

Replay-Fusion metrics are stored in `results/replay-fusion/`. The release result
was evaluated on the same validation split used to inspect the fusion scale.

## 8. Run inference

```bash
nestdetect detect \
  --model models/nestdetect_hope_cms_v4_replay_fusion.pt \
  --source sample.jpg \
  --output results/detection_outputs/sample.jpg \
  --conf 0.25 \
  --imgsz 640
```

Or launch the application:

```bash
python -m streamlit run app.py
```

## 9. Output policy

- release checkpoints: `models/`;
- intermediate checkpoints and training runs: `artifacts/`;
- compact metrics and selected figures: `results/`;
- raw validation runs: `artifacts/evaluations/`;
- generated datasets and replay memory: `datasets/` and `memory/`, excluded from
  Git.
