import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("glm_ocr_server.layout")


@dataclass
class LayoutBlock:
    type: str
    bbox: tuple[int, int, int, int]
    score: float


_LAYOUT_ENGINE: Any = None
_LAYOUT_ENGINE_ERROR: Optional[Exception] = None


def _clamp_bbox(
    bbox: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return (x1, y1, x2, y2)


def _bbox_from_any(value: Any) -> Optional[tuple[int, int, int, int]]:
    if value is None:
        return None

    # Already [x1, y1, x2, y2]
    if isinstance(value, (list, tuple)) and len(value) == 4 and all(
        isinstance(v, (int, float)) for v in value
    ):
        x1, y1, x2, y2 = value
        return (int(x1), int(y1), int(x2), int(y2))

    # Polygon [[x, y], ...]
    if isinstance(value, (list, tuple)) and value and all(
        isinstance(point, (list, tuple)) and len(point) >= 2 for point in value
    ):
        xs = [float(point[0]) for point in value]
        ys = [float(point[1]) for point in value]
        return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))

    if isinstance(value, dict):
        keys = {k.lower(): k for k in value.keys()}
        if {"x1", "y1", "x2", "y2"}.issubset(keys):
            return (
                int(value[keys["x1"]]),
                int(value[keys["y1"]]),
                int(value[keys["x2"]]),
                int(value[keys["y2"]]),
            )

    return None


def _normalize_label(label: str) -> str:
    lowered = (label or "text").strip().lower()
    mapping = {
        "text": "text",
        "title": "title",
        "list": "list",
        "table": "table",
        "formula": "formula",
        "equation": "formula",
        "figure": "figure",
        "image": "figure",
        "chart": "figure",
        "caption": "caption",
        "header": "header",
        "footer": "footer",
    }
    return mapping.get(lowered, lowered or "text")


def _extract_layout_blocks(raw_result: Any, width: int, height: int) -> list[LayoutBlock]:
    candidates: list[Any] = []
    if raw_result is None:
        return []
    if isinstance(raw_result, dict):
        for key in ("layout", "result", "results", "boxes", "data"):
            value = raw_result.get(key)
            if isinstance(value, list):
                candidates.extend(value)
        if not candidates:
            candidates = [raw_result]
    elif isinstance(raw_result, list):
        candidates = raw_result
    else:
        candidates = [raw_result]

    blocks: list[LayoutBlock] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue

        bbox = None
        for key in ("bbox", "box", "coordinate", "points", "poly"):
            bbox = _bbox_from_any(item.get(key))
            if bbox is not None:
                break

        if bbox is None and "region" in item:
            bbox = _bbox_from_any(item.get("region"))

        if bbox is None:
            continue

        label = "text"
        for key in ("type", "label", "category", "class_name", "name"):
            value = item.get(key)
            if value:
                label = str(value)
                break

        score = 1.0
        for key in ("score", "confidence", "prob"):
            value = item.get(key)
            if isinstance(value, (int, float)):
                score = float(value)
                break

        blocks.append(
            LayoutBlock(
                type=_normalize_label(label),
                bbox=_clamp_bbox(bbox, width, height),
                score=score,
            )
        )

    return blocks


def _load_paddle_layout_engine() -> Any:
    global _LAYOUT_ENGINE, _LAYOUT_ENGINE_ERROR

    if _LAYOUT_ENGINE is not None:
        return _LAYOUT_ENGINE
    if _LAYOUT_ENGINE_ERROR is not None:
        raise _LAYOUT_ENGINE_ERROR

    try:
        import paddleocr  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        _LAYOUT_ENGINE_ERROR = exc
        raise

    last_error: Optional[Exception] = None

    # Prefer newer classes if available.
    for class_name in ("PPStructureV3", "PPStructure"):
        cls = getattr(paddleocr, class_name, None)
        if cls is None:
            continue
        for kwargs in (
            {
                "show_log": False,
                "ocr": False,
                "table": False,
                "layout": True,
                "layout_model": "PP-DocLayoutV3",
            },
            {
                "show_log": False,
                "ocr": False,
                "table": False,
                "layout": True,
            },
            {
                "show_log": False,
                "ocr": False,
                "table": False,
            },
        ):
            try:
                _LAYOUT_ENGINE = cls(**kwargs)
                logger.info("Initialized paddle layout engine: %s", class_name)
                return _LAYOUT_ENGINE
            except Exception as exc:  # pragma: no cover - optional dependency
                last_error = exc

    _LAYOUT_ENGINE_ERROR = last_error or RuntimeError(
        "No compatible PaddleOCR layout class")
    raise _LAYOUT_ENGINE_ERROR


def _run_layout_engine(engine: Any, image: Image.Image) -> Any:
    array = np.array(image.convert("RGB"))
    if hasattr(engine, "predict"):
        output = engine.predict(array)
        if not isinstance(output, list):
            output = list(output)
        if len(output) == 1 and isinstance(output[0], dict) and "res" in output[0]:
            return output[0].get("res")
        return output
    if callable(engine):
        return engine(array)
    raise RuntimeError("Unsupported layout engine interface")


def _find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: Optional[int] = None
    for idx, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            runs.append((start, idx))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def _detect_columns_fallback(image: Image.Image) -> list[LayoutBlock]:
    width, height = image.size
    if width < 200 or height < 200:
        return [LayoutBlock(type="text", bbox=(0, 0, width, height), score=0.1)]

    gray = np.array(image.convert("L"))
    ink = gray < 245
    vertical_density = ink.sum(axis=0).astype(np.float32)

    win = max(7, width // 100)
    kernel = np.ones(win, dtype=np.float32) / float(win)
    smoothed = np.convolve(vertical_density, kernel, mode="same")

    low_threshold = max(2.0, float(height) * 0.01)
    valley_mask = smoothed <= low_threshold
    valleys = [
        run
        for run in _find_runs(valley_mask)
        if (run[1] - run[0]) >= max(8, int(width * 0.03))
    ]

    if not valleys:
        return [LayoutBlock(type="text", bbox=(0, 0, width, height), score=0.15)]

    splits = [0]
    for start, end in valleys:
        center = (start + end) // 2
        if center < int(width * 0.15) or center > int(width * 0.85):
            continue
        splits.append(center)
    splits.append(width)
    splits = sorted(set(splits))

    blocks: list[LayoutBlock] = []
    for left, right in zip(splits[:-1], splits[1:]):
        if (right - left) < max(20, int(width * 0.12)):
            continue
        blocks.append(
            LayoutBlock(type="text", bbox=(left, 0, right, height), score=0.2)
        )

    if not blocks:
        blocks = [LayoutBlock(type="text", bbox=(
            0, 0, width, height), score=0.1)]
    return blocks


def detect_layout_blocks(image: Image.Image, backend: str = "ppdoclayoutv3") -> list[LayoutBlock]:
    backend_name = (backend or "ppdoclayoutv3").strip().lower()
    width, height = image.size

    if backend_name in {"none", "off"}:
        return [LayoutBlock(type="text", bbox=(0, 0, width, height), score=1.0)]

    if backend_name != "ppdoclayoutv3":
        raise ValueError(f"Unsupported layout backend: {backend}")

    try:
        engine = _load_paddle_layout_engine()
        raw = _run_layout_engine(engine, image)
        blocks = _extract_layout_blocks(raw, width, height)
        if blocks:
            return blocks
        logger.warning(
            "Layout engine returned no blocks. Using fallback columns.")
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Layout engine unavailable, using fallback: %s", exc)

    return _detect_columns_fallback(image)
