# Research Results

The repository stores compact, publication-oriented result files:

- `baseline/`: YOLO11n replay and no-replay comparison;
- `hope/`: YOLO11n-HoPe replay and no-replay comparison;
- `cms/`: replay-free CMS V1-V5 ablations;
- `replay-fusion/`: final replay-informed release evaluation.

Each directory contains machine-readable CSV/JSON summaries and selected figures.
Raw validation runs, temporary evaluation views, and prediction batches are
generated under `artifacts/evaluations/` and are excluded from Git.

See [`docs/research-report.md`](../docs/research-report.md) for the consolidated
methodology, interpretation, limitations, and conclusions.
