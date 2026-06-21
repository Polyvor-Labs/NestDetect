from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .io_utils import ensure_dir
from .training import require_yolo


def _scalar(value: Any) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


@lru_cache(maxsize=8)
def _load_yolo_model(model_path: str, modified_ns: int) -> Any:
    del modified_ns
    YOLO = require_yolo()
    return YOLO(model_path)


def load_yolo_model(model_path: str | Path) -> Any:
    resolved = Path(model_path).expanduser().resolve()
    return _load_yolo_model(str(resolved), resolved.stat().st_mtime_ns)


def _extract_detections(
    result: Any,
    class_aliases: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    detections: list[dict[str, Any]] = []
    boxes = result.boxes
    if boxes is None:
        return detections
    for index in range(len(boxes)):
        class_id = int(_scalar(boxes.cls[index]))
        x1, y1, x2, y2 = [_scalar(value) for value in boxes.xyxy[index]]
        detections.append(
            {
                "no": index + 1,
                "class_id": class_id,
                "object": (class_aliases or {}).get(
                    result.names[class_id], result.names[class_id]
                ),
                "confidence": _scalar(boxes.conf[index]),
                "x1": round(x1, 2),
                "y1": round(y1, 2),
                "x2": round(x2, 2),
                "y2": round(y2, 2),
            }
        )
    return detections


def predict_image(
    model_path: str | Path,
    source: str | Path | Image.Image | np.ndarray,
    confidence: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str | int | None = None,
    classes: list[int] | None = None,
    class_aliases: dict[str, str] | None = None,
) -> tuple[Image.Image, list[dict[str, Any]]]:
    model = load_yolo_model(model_path)
    kwargs: dict[str, Any] = {
        "source": source,
        "conf": confidence,
        "iou": iou,
        "imgsz": imgsz,
        "verbose": False,
    }
    if device not in (None, ""):
        kwargs["device"] = device
    if classes is not None:
        kwargs["classes"] = classes
    results = model.predict(**kwargs)
    if not results:
        raise RuntimeError("The model returned no result")
    result = results[0]
    annotated = Image.fromarray(result.plot()[..., ::-1])
    detections = _extract_detections(result, class_aliases)
    return annotated, detections


def predict_bgr_frame(
    model_path: str | Path,
    frame: np.ndarray,
    confidence: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 320,
    device: str | int | None = None,
    classes: list[int] | None = None,
    class_aliases: dict[str, str] | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    model = load_yolo_model(model_path)
    kwargs: dict[str, Any] = {
        "source": frame,
        "conf": confidence,
        "iou": iou,
        "imgsz": imgsz,
        "verbose": False,
    }
    if device not in (None, ""):
        kwargs["device"] = device
    if classes is not None:
        kwargs["classes"] = classes
    results = model.predict(**kwargs)
    if not results:
        raise RuntimeError("The model returned no result")
    result = results[0]
    return np.ascontiguousarray(result.plot()), _extract_detections(result, class_aliases)


def detect_to_file(
    model_path: str | Path,
    source: str | Path,
    output_path: str | Path,
    **kwargs: Any,
) -> tuple[Path, list[dict[str, Any]]]:
    annotated, detections = predict_image(model_path, source, **kwargs)
    output = Path(output_path)
    ensure_dir(output.parent)
    annotated.save(output)
    return output, detections
