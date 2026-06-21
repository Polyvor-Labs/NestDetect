# Published Checkpoints

| Checkpoint | Role | Replay-free |
|---|---|---|
| `nestdetect_final.pt` | YOLO11n replay baseline | No |
| `nestdetect_hope.pt` | YOLO11n-HoPe replay model | No |
| `nestdetect_hope_cms_only_v4.pt` | Default application model | Yes |
| `nestdetect_hope_cms_v5.pt` | Compressed conditional CMS research model | Yes |
| `nestdetect_hope_cms_v4_replay_fusion.pt` | Highest-scoring release model | No |

Each checkpoint has a matching `.metadata.json` file containing its SHA-256
digest, provenance, target classes, and evaluation metrics.

CMS V1-V3 and other intermediate ablation checkpoints are excluded from Git. They
can be regenerated with the configurations under `configs/ablations/`.
