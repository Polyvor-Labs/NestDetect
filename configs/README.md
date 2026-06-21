# Experiment Configurations

Configurations are grouped by research role:

- `baseline/`: YOLO11n base, no-replay, and replay experiments;
- `hope/`: YOLO11n-HoPe base, no-replay, and replay experiments;
- `ablations/`: persistent-memory and head-wise CMS ablations;
- `models/`: custom Ultralytics model definitions.

Run any training configuration through the package CLI:

```bash
nestdetect train configs/hope/base.yaml
```

All paths are resolved from the repository root. Generated checkpoints and
training runs are written to `artifacts/` unless the configuration explicitly
defines a release checkpoint under `models/`.
