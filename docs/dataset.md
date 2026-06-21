# Dataset

## Source

NestDetect uses a curated subset of Microsoft COCO 2017.

- Dataset: <https://cocodataset.org/>
- Paper: <https://arxiv.org/abs/1405.0312>
- Terms of use: <https://cocodataset.org/#termsofuse>

COCO images are not committed to this repository. The preparation script downloads
only the selected images.

## Class split

| Stage | Local source ID | COCO ID | Class |
|---|---:|---:|---|
| Base | 0 | 0 | person |
| Base | 1 | 56 | chair |
| Base | 2 | 60 | dining table |
| Incremental | 0 | 63 | laptop |
| Incremental | 1 | 73 | book |

The application displays `dining table` as `table`.

## Curated source subsets

| Dataset | Train images | Validation images | Selection seed |
|---|---:|---:|---:|
| Base task | 132 | 25 | 42 |
| New-class task | 107 | 21 | 142 |

Cross-stage target classes are excluded: base images do not contain retained
`laptop` or `book` annotations, and incremental images do not contain retained
`person`, `chair`, or `dining table` annotations.

## Replay protocols

| Protocol | Replay train images | Replay validation images | Final mixture |
|---|---:|---:|---:|
| YOLO11n baseline | 104 | 22 | 211 train / 43 validation |
| YOLO11n-HoPe | 55 | 11 | 162 train / 32 validation |

Both replay memories are selected deterministically from the base subset with seed
42. Multi-label images may contribute to more than one class quota.

## Area constraints

| Class | Minimum selection area | Minimum retained label area |
|---|---:|---:|
| person | 5% | 1% |
| chair | 5% | 1% |
| dining table | 8% | 2% |
| laptop | 8% | 2% |
| book | 5% | 1% |

These constraints reduce the dominance of extremely small target boxes.

## Local structure

```text
datasets/
├── step1_base/                       # source manifests and downloaded images
├── step2_new_classes/                # source manifests and downloaded images
├── base_coco80/                      # generated COCO80 base dataset
├── new_coco80/                       # generated COCO80 new-class dataset
├── incremental_no_replay_coco80/
├── incremental_replay_baseline_coco80/
└── incremental_replay_hope_coco80/
```


## Rebuild

```bash
python scripts/prepare_coco_subset.py
python scripts/build_datasets.py
```

The second command validates every generated dataset and prints image and instance
counts.
