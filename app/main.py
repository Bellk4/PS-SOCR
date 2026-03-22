import base64
import asyncio
import io
import json
import logging
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)

from app.layout_ppdoclayoutv3 import LayoutBlock, detect_layout_blocks

MODEL_ID = "zai-org/GLM-OCR"
ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_CACHE_DIR = Path(
    os.getenv("GLM_MODEL_CACHE", str(ROOT_DIR / "models" / "hf_cache"))
)
DEFAULT_DPI = 220
DEFAULT_MAX_NEW_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.0
DEFAULT_USE_LAYOUT = False
DEFAULT_LAYOUT_BACKEND = "ppdoclayoutv3"
DEFAULT_READING_ORDER = "auto"
DEFAULT_REGION_PADDING = 12
DEFAULT_MAX_REGIONS = 200
DEFAULT_REGION_PARALLELISM = 1
ALLOWED_TASKS = {"text", "table", "formula", "extract_json"}
ALLOWED_LINEBREAK_MODES = {"none", "paragraph", "compact"}
ALLOWED_LAYOUT_BACKENDS = {"ppdoclayoutv3", "none"}
ALLOWED_READING_ORDERS = {"auto", "ltr_ttb", "rtl_ttb", "vertical_rl"}

logger = logging.getLogger("glm_ocr_server")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def patch_transformers_video_auto_none_bug() -> None:
    try:
        from transformers.models.auto import video_processing_auto
    except Exception:
        return

    if getattr(video_processing_auto, "_glm_none_patch_applied", False):
        return

    fixed = 0
    for key, value in list(video_processing_auto.VIDEO_PROCESSOR_MAPPING_NAMES.items()):
        if value is None:
            video_processing_auto.VIDEO_PROCESSOR_MAPPING_NAMES[key] = ""
            fixed += 1

    video_processing_auto._glm_none_patch_applied = True
    if fixed:
        logger.warning(
            "Applied transformers video auto patch for %d entries", fixed)


def resolve_device(device: str) -> str:
    requested = (device or "auto").lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            logger.warning(
                "CUDA requested but unavailable. Falling back to CPU.")
            return "cpu"
        return "cuda"
    if requested == "cpu":
        return requested
    raise HTTPException(
        status_code=400, detail=f"Unsupported device: {device}")


class GlmRuntime:
    def __init__(self) -> None:
        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[AutoModelForImageTextToText] = None
        self.current_device: Optional[str] = None
        self._load_lock = asyncio.Lock()

    def _load_model(self, device: str) -> AutoModelForImageTextToText:
        if device == "cuda":
            try:
                return AutoModelForImageTextToText.from_pretrained(
                    MODEL_ID,
                    cache_dir=str(MODEL_CACHE_DIR),
                    torch_dtype="auto",
                    device_map="auto",
                )
            except ValueError as exc:
                if "requires `accelerate`" not in str(exc):
                    raise
                logger.warning(
                    "accelerate is missing. Falling back to CUDA load without device_map."
                )
                model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_ID,
                    cache_dir=str(MODEL_CACHE_DIR),
                    torch_dtype="auto",
                    device_map=None,
                )
                return model.to("cuda")

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            cache_dir=str(MODEL_CACHE_DIR),
            torch_dtype=torch.float32,
            device_map=None,
        )
        return model.to("cpu")

    async def ensure_loaded(self, device: str) -> None:
        async with self._load_lock:
            if self.processor is None:
                logger.info("Loading processor: %s", MODEL_ID)
                patch_transformers_video_auto_none_bug()
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        MODEL_ID,
                        cache_dir=str(MODEL_CACHE_DIR),
                    )
                except ImportError as exc:
                    if "Torchvision library" in str(exc):
                        raise RuntimeError(
                            "torchvision is required by GLM-OCR processor. "
                            "Install it with: pip install torchvision"
                        ) from exc
                    raise
                except TypeError as exc:
                    if "NoneType" not in str(exc):
                        raise
                    # Retry once after forcing the compatibility patch.
                    patch_transformers_video_auto_none_bug()
                    self.processor = AutoProcessor.from_pretrained(
                        MODEL_ID,
                        cache_dir=str(MODEL_CACHE_DIR),
                    )

            if self.model is not None and self.current_device == device:
                return

            if self.model is not None:
                logger.info(
                    "Switching model device from %s to %s", self.current_device, device
                )
                del self.model
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            logger.info("Loading model: %s (device=%s)", MODEL_ID, device)
            self.model = await asyncio.to_thread(self._load_model, device)
            self.current_device = device

    def get(self) -> tuple[AutoProcessor, AutoModelForImageTextToText, str]:
        if self.processor is None or self.model is None or self.current_device is None:
            raise RuntimeError("GLM runtime is not initialized")
        return self.processor, self.model, self.current_device


def load_pages(path: Path, dpi: int, crop_region: Optional[dict] = None) -> list[Image.Image]:
    def apply_crop_if_needed(image: Image.Image) -> Image.Image:
        """Apply crop to image if crop_region is specified"""
        if crop_region is None:
            return image

        # Get image dimensions
        img_width, img_height = image.size

        # Get crop coordinates (ensure they're within bounds)
        x1 = max(0, min(crop_region['x1'], img_width - 1))
        y1 = max(0, min(crop_region['y1'], img_height - 1))
        x2 = max(x1 + 1, min(crop_region['x2'], img_width))
        y2 = max(y1 + 1, min(crop_region['y2'], img_height))

        # Apply crop
        return image.crop((x1, y1, x2, y2))

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise RuntimeError(
                "pypdfium2 is required for PDF input. Install it with: pip install pypdfium2"
            ) from exc

        pages: list[Image.Image] = []
        scale = max(36, int(dpi)) / 72.0
        doc = pdfium.PdfDocument(str(path))
        selected_page = None
        if crop_region is not None and crop_region.get("page") is not None:
            try:
                selected_page = max(1, int(crop_region["page"]))
            except (TypeError, ValueError):
                selected_page = None
        try:
            for page_index in range(len(doc)):
                if selected_page is not None and (page_index + 1) != selected_page:
                    continue
                page = doc[page_index]
                bitmap = page.render(scale=scale)
                try:
                    image = bitmap.to_pil().convert("RGB")
                    image = apply_crop_if_needed(image)
                finally:
                    if hasattr(bitmap, "close"):
                        bitmap.close()
                    if hasattr(page, "close"):
                        page.close()
                pages.append(image)
        finally:
            if hasattr(doc, "close"):
                doc.close()
        return pages

    with Image.open(path) as image:
        converted_image = image.convert("RGB")
        cropped_image = apply_crop_if_needed(converted_image)
        return [cropped_image]


def save_temp_upload(upload_name: str, content: bytes) -> Path:
    suffix = Path(upload_name or "upload").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        return Path(tmp.name)


def save_temp_png(image: Image.Image) -> Path:
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        image.save(tmp_path, format="PNG")
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return Path(tmp_path)


def build_prompt(task: str, schema: Optional[str]) -> str:
    if task == "text":
        return "Text Recognition:"
    if task == "table":
        return "Table Recognition:"
    if task == "formula":
        return "Formula Recognition:"
    if task == "extract_json":
        if not schema:
            raise HTTPException(
                status_code=400,
                detail="schema is required when task=extract_json",
            )
        # Align with the prompt style in the official model card.
        return f"请按下列JSON格式输出图中信息:\n{schema}"
    raise HTTPException(status_code=400, detail=f"Unsupported task: {task}")


def is_cjk_char(ch: str) -> bool:
    if not ch:
        return False
    code = ord(ch)
    return (
        0x3040 <= code <= 0x30FF
        or 0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
    )


def join_soft_wrapped_line(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if is_cjk_char(left[-1]) and is_cjk_char(right[0]):
        return left + right
    return f"{left} {right}"


def is_hard_break(left: str, right: str) -> bool:
    if not left or not right:
        return True
    if left.endswith(("。", "！", "？", ".", "!", "?", "：", ":", "；", ";")):
        return True
    if "|" in left and "|" in right:
        return True
    if re.match(r"^(\d+[\.\)]|[（(]?\d+[）)]|[-*•・●○■□])\s*", right):
        return True
    return False


def normalize_linebreaks(text: str, mode: str) -> str:
    normalized_mode = (mode or "none").strip().lower()
    if normalized_mode == "none" or not text:
        return text

    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    if normalized_mode == "paragraph":
        merged: list[str] = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                if merged and merged[-1] != "":
                    merged.append("")
                continue
            if not merged or merged[-1] == "":
                merged.append(line)
                continue
            if is_hard_break(merged[-1], line):
                merged.append(line)
            else:
                merged[-1] = join_soft_wrapped_line(merged[-1], line)
        return "\n".join(merged).strip()

    if normalized_mode == "compact":
        non_empty = [line.strip() for line in lines if line.strip()]
        if not non_empty:
            return ""
        merged = non_empty[0]
        for line in non_empty[1:]:
            merged = join_soft_wrapped_line(merged, line)
        return merged.strip()

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported linebreak_mode: {mode}",
    )


def circled_number(num: int) -> Optional[str]:
    if num == 0:
        return "⓪"
    if 1 <= num <= 20:
        return chr(ord("①") + (num - 1))
    if 21 <= num <= 35:
        return chr(0x3251 + (num - 21))
    if 36 <= num <= 50:
        return chr(0x32B1 + (num - 36))
    return None


def normalize_textcircled_notation(text: str) -> str:
    if not text:
        return text

    def replace_match(match: re.Match[str]) -> str:
        raw = (match.group(1) or "").strip()
        if not raw.isdigit():
            return match.group(0)
        symbol = circled_number(int(raw))
        return symbol or match.group(0)

    # Convert both "$\\textcircled{1}$" and "\\textcircled{1}".
    text = re.sub(r"\$\s*\\textcircled\{(\d+)\}\s*\$", replace_match, text)
    text = re.sub(r"\\textcircled\{(\d+)\}", replace_match, text)
    return text


def normalize_text_output(text: str, task: str, linebreak_mode: str) -> str:
    normalized = text
    if task in {"text", "table"}:
        normalized = normalize_textcircled_notation(normalized)
    return normalize_linebreaks(normalized, linebreak_mode)


def clamp_bbox_with_padding(
    bbox: tuple[int, int, int, int],
    image: Image.Image,
    padding: int,
) -> tuple[int, int, int, int]:
    width, height = image.size
    x1, y1, x2, y2 = bbox
    pad = max(0, int(padding))
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(width, int(x2) + pad)
    y2 = min(height, int(y2) + pad)
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return (x1, y1, x2, y2)


def bbox_dict(bbox: tuple[int, int, int, int]) -> dict[str, int]:
    return {"x1": int(bbox[0]), "y1": int(bbox[1]), "x2": int(bbox[2]), "y2": int(bbox[3])}


def normalize_layout_label(block_type: str) -> str:
    lowered = (block_type or "text").strip().lower()
    if lowered in {"formula", "equation"}:
        return "formula"
    if lowered in {"table"}:
        return "table"
    if lowered in {"figure", "image", "chart"}:
        return "figure"
    return lowered or "text"


def resolve_effective_reading_order(
    blocks: list[LayoutBlock],
    requested_order: str,
) -> str:
    order = (requested_order or DEFAULT_READING_ORDER).strip().lower()
    if order and order != "auto":
        return order
    if not blocks:
        return "ltr_ttb"

    widths = [max(1, b.bbox[2] - b.bbox[0]) for b in blocks]
    heights = [max(1, b.bbox[3] - b.bbox[1]) for b in blocks]
    tall_ratio = sum(1 for w, h in zip(widths, heights)
                     if h > (w * 1.8)) / float(len(blocks))
    narrow_ratio = sum(1 for w, h in zip(widths, heights)
                       if w < (h * 0.65)) / float(len(blocks))

    centers = sorted(((b.bbox[0] + b.bbox[2]) // 2) for b in blocks)
    x_diffs = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
    largest_gap = max(x_diffs) if x_diffs else 0
    span = max(1, centers[-1] - centers[0]) if len(centers) > 1 else 1
    multi_column = (largest_gap / float(span)) > 0.35

    if tall_ratio > 0.55 and narrow_ratio > 0.45:
        return "vertical_rl"
    if multi_column:
        # For Japanese multi-column documents, right-to-left column order is common.
        return "rtl_ttb"
    return "ltr_ttb"


def sort_blocks_ltr_or_rtl(
    blocks: list[LayoutBlock],
    rtl: bool,
) -> list[LayoutBlock]:
    if not blocks:
        return []
    avg_h = sum(max(1, b.bbox[3] - b.bbox[1])
                for b in blocks) / float(len(blocks))
    row_tol = max(8.0, avg_h * 0.45)

    sorted_by_y = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))
    rows: list[list[LayoutBlock]] = []
    for block in sorted_by_y:
        if not rows:
            rows.append([block])
            continue
        current = rows[-1]
        avg_y = sum(item.bbox[1] for item in current) / float(len(current))
        if abs(block.bbox[1] - avg_y) <= row_tol:
            current.append(block)
        else:
            rows.append([block])

    ordered: list[LayoutBlock] = []
    for row in rows:
        ordered.extend(sorted(row, key=lambda b: b.bbox[0], reverse=rtl))
    return ordered


def sort_blocks_vertical_rl(blocks: list[LayoutBlock]) -> list[LayoutBlock]:
    if not blocks:
        return []
    avg_w = sum(max(1, b.bbox[2] - b.bbox[0])
                for b in blocks) / float(len(blocks))
    col_tol = max(8.0, avg_w * 0.5)

    sorted_by_x = sorted(blocks, key=lambda b: b.bbox[0], reverse=True)
    cols: list[list[LayoutBlock]] = []
    for block in sorted_by_x:
        if not cols:
            cols.append([block])
            continue
        current = cols[-1]
        avg_x = sum(item.bbox[0] for item in current) / float(len(current))
        if abs(block.bbox[0] - avg_x) <= col_tol:
            current.append(block)
        else:
            cols.append([block])

    ordered: list[LayoutBlock] = []
    for col in cols:
        ordered.extend(sorted(col, key=lambda b: b.bbox[1]))
    return ordered


def sort_layout_blocks(blocks: list[LayoutBlock], reading_order: str) -> list[LayoutBlock]:
    if reading_order == "vertical_rl":
        return sort_blocks_vertical_rl(blocks)
    if reading_order == "rtl_ttb":
        return sort_blocks_ltr_or_rtl(blocks, rtl=True)
    return sort_blocks_ltr_or_rtl(blocks, rtl=False)


def block_prompt_for_task(global_task: str, block_type: str, schema: Optional[str]) -> str:
    if global_task != "text":
        return build_prompt(global_task, schema)
    normalized_type = normalize_layout_label(block_type)
    if normalized_type == "table":
        return build_prompt("table", None)
    if normalized_type == "formula":
        return build_prompt("formula", None)
    return build_prompt("text", None)


def combine_block_texts(blocks: list[dict[str, Any]], linebreak_mode: str) -> str:
    parts: list[str] = []
    for block in blocks:
        text = str(block.get("text") or "").strip()
        if not text:
            continue
        block_type = normalize_layout_label(str(block.get("type") or "text"))
        if block_type == "table":
            parts.append(text)
            parts.append("")
            parts.append("")
        else:
            parts.append(text)
            parts.append("")
    if not parts:
        return ""
    combined = "\n".join(parts).strip()
    return normalize_linebreaks(combined, linebreak_mode)


def build_layout_preview_base64(
    page: Image.Image,
    blocks: list[dict[str, Any]],
) -> str:
    preview = page.copy()
    drawer = ImageDraw.Draw(preview)
    for item in blocks:
        bbox = item.get("bbox") or {}
        x1 = int(bbox.get("x1", 0))
        y1 = int(bbox.get("y1", 0))
        x2 = int(bbox.get("x2", x1 + 1))
        y2 = int(bbox.get("y2", y1 + 1))
        block_id = str(item.get("id") or "")
        block_type = str(item.get("type") or "text")
        drawer.rectangle((x1, y1, x2, y2), outline="#2563eb", width=2)
        if block_id:
            drawer.text((x1 + 2, y1 + 2),
                        f"{block_id}:{block_type}", fill="#dc2626")
    buffer = io.BytesIO()
    preview.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return encoded


def glm_infer(
    processor: AutoProcessor,
    model: AutoModelForImageTextToText,
    image_path: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    request_id: Optional[str] = None,
) -> tuple[str, str, bool]:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    inputs.pop("token_type_ids", None)

    generation_args: dict[str, Any] = {
        "max_new_tokens": max(1, int(max_new_tokens))}
    if temperature is not None and float(temperature) > 0:
        generation_args.update(
            {"do_sample": True, "temperature": float(temperature)})
    if request_id:
        generation_args["stopping_criteria"] = StoppingCriteriaList(
            [CancelStoppingCriteria(request_id)]
        )

    with torch.inference_mode():
        generated = model.generate(**inputs, **generation_args)
    input_len = inputs["input_ids"].shape[1]
    output = generated[0][input_len:]
    output_len = int(output.shape[0])
    raw_text = processor.decode(output, skip_special_tokens=False).strip()
    clean_text = processor.decode(output, skip_special_tokens=True).strip()
    truncated = output_len >= max(1, int(max_new_tokens))
    return raw_text, clean_text, truncated


RUNTIME = GlmRuntime()
GENERATE_SEMAPHORE = asyncio.Semaphore(1)
PROGRESS_STATE: dict[str, dict[str, Any]] = {}
MAX_PROGRESS_ENTRIES = 300
CANCEL_REQUESTS: set[str] = set()


def set_progress(
    request_id: str,
    state: str,
    message: str,
    current_page: int = 0,
    total_pages: int = 0,
    current_region: int = 0,
    total_regions: int = 0,
) -> None:
    PROGRESS_STATE[request_id] = {
        "request_id": request_id,
        "state": state,
        "message": message,
        "current_page": int(current_page),
        "total_pages": int(total_pages),
        "current_region": int(current_region),
        "total_regions": int(total_regions),
        "updated_at": time.time(),
    }
    if len(PROGRESS_STATE) > MAX_PROGRESS_ENTRIES:
        # Keep memory bounded by removing the oldest entries.
        oldest = sorted(PROGRESS_STATE.items(), key=lambda item: item[1]["updated_at"])[
            : len(PROGRESS_STATE) - MAX_PROGRESS_ENTRIES
        ]
        for key, _ in oldest:
            PROGRESS_STATE.pop(key, None)


def is_cancel_requested(request_id: str) -> bool:
    return request_id in CANCEL_REQUESTS


class CancelStoppingCriteria(StoppingCriteria):
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> bool:
        return is_cancel_requested(self.request_id)


def clear_cancel_request(request_id: str) -> None:
    CANCEL_REQUESTS.discard(request_id)


def request_cancel(request_id: str) -> dict[str, Any]:
    item = PROGRESS_STATE.get(request_id)
    if item is not None:
        current_state = str(item.get("state") or "")
        if current_state in {"done", "error", "canceled"}:
            return {
                "request_id": request_id,
                "accepted": False,
                "state": current_state,
                "message": "このリクエストは既に終了しています",
            }
    CANCEL_REQUESTS.add(request_id)
    if item is None:
        return {
            "request_id": request_id,
            "accepted": True,
            "state": "cancel_requested",
            "message": "中断要求を受け付けました",
        }

    set_progress(
        request_id,
        "cancel_requested",
        "中断要求を受け付けました",
        int(item.get("current_page") or 0),
        int(item.get("total_pages") or 0),
        int(item.get("current_region") or 0),
        int(item.get("total_regions") or 0),
    )
    return {
        "request_id": request_id,
        "accepted": True,
        "state": "cancel_requested",
        "message": "中断要求を受け付けました",
    }


app = FastAPI(
    title="GLM-OCR Local Server",
    description="FastAPI server for local GLM-OCR inference",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
async def startup_load_model() -> None:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    default_device = resolve_device("auto")
    await RUNTIME.ensure_loaded(default_device)
    logger.info(
        "Startup complete (device=%s, cache_dir=%s)",
        default_device,
        MODEL_CACHE_DIR,
    )


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = static_dir / "index.html"
    if not html_path.exists():
        raise HTTPException(
            status_code=500,
            detail="UI not found. Ensure app/static/index.html exists.",
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def status() -> dict[str, Any]:
    return {
        "cuda_available": torch.cuda.is_available(),
        "device_default": "cuda" if torch.cuda.is_available() else "cpu",
        "model": MODEL_ID,
        "model_cache_dir": str(MODEL_CACHE_DIR),
    }


@app.get("/api/progress/{request_id}")
async def progress(request_id: str) -> dict[str, Any]:
    item = PROGRESS_STATE.get(request_id)
    if item is None:
        raise HTTPException(status_code=404, detail="progress not found")
    return item


@app.post("/api/cancel/{request_id}")
async def cancel(request_id: str) -> dict[str, Any]:
    return request_cancel(request_id)


@app.post("/api/analyze")
async def analyze(
    file: UploadFile = File(...),
    device: str = Form("auto"),
    dpi: int = Form(DEFAULT_DPI),
    task: str = Form("text"),
    linebreak_mode: str = Form("none"),
    schema: Optional[str] = Form(None),
    max_new_tokens: int = Form(DEFAULT_MAX_NEW_TOKENS),
    temperature: float = Form(DEFAULT_TEMPERATURE),
    use_layout: bool = Form(DEFAULT_USE_LAYOUT),
    layout_backend: str = Form(DEFAULT_LAYOUT_BACKEND),
    reading_order: str = Form(DEFAULT_READING_ORDER),
    region_padding: int = Form(DEFAULT_REGION_PADDING),
    max_regions: int = Form(DEFAULT_MAX_REGIONS),
    region_parallelism: int = Form(DEFAULT_REGION_PARALLELISM),
    crop_region: Optional[str] = Form(None),
    request_id: Optional[str] = Form(None),
) -> dict[str, Any]:
    request_id = (request_id or "").strip() or uuid.uuid4().hex
    clear_cancel_request(request_id)
    set_progress(request_id, "preprocessing", "事前処理中", 0, 0)

    normalized_task = (task or "text").strip().lower()
    if normalized_task not in ALLOWED_TASKS:
        set_progress(request_id, "error", f"Unsupported task: {task}", 0, 0)
        raise HTTPException(
            status_code=400, detail=f"Unsupported task: {task}")
    normalized_linebreak_mode = (linebreak_mode or "none").strip().lower()
    if normalized_linebreak_mode not in ALLOWED_LINEBREAK_MODES:
        set_progress(
            request_id,
            "error",
            f"Unsupported linebreak_mode: {linebreak_mode}",
            0,
            0,
        )
        raise HTTPException(
            status_code=400, detail=f"Unsupported linebreak_mode: {linebreak_mode}"
        )
    normalized_layout_backend = (
        layout_backend or DEFAULT_LAYOUT_BACKEND).strip().lower()
    if normalized_layout_backend not in ALLOWED_LAYOUT_BACKENDS:
        set_progress(
            request_id,
            "error",
            f"Unsupported layout_backend: {layout_backend}",
            0,
            0,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported layout_backend: {layout_backend}",
        )
    normalized_reading_order = (
        reading_order or DEFAULT_READING_ORDER).strip().lower()
    if normalized_reading_order not in ALLOWED_READING_ORDERS:
        set_progress(
            request_id,
            "error",
            f"Unsupported reading_order: {reading_order}",
            0,
            0,
        )
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported reading_order: {reading_order}",
        )

    normalized_dpi = max(36, min(600, int(dpi or DEFAULT_DPI)))
    normalized_max_new_tokens = max(
        1, min(32768, int(max_new_tokens or DEFAULT_MAX_NEW_TOKENS)))
    normalized_region_padding = max(
        0, min(256, int(region_padding or DEFAULT_REGION_PADDING)))
    normalized_max_regions = max(
        1, min(1000, int(max_regions or DEFAULT_MAX_REGIONS)))
    normalized_region_parallelism = max(
        1,
        min(8, int(region_parallelism or DEFAULT_REGION_PARALLELISM)),
    )
    use_layout_mode = bool(use_layout)

    # Parse crop region if provided
    parsed_crop_region = None
    if crop_region and crop_region.strip():
        try:
            crop_data = json.loads(crop_region.strip())
            # Validate crop region format
            if not isinstance(crop_data, dict):
                raise ValueError("crop_region must be a JSON object")

            required_keys = {'x1', 'y1', 'x2', 'y2'}
            if not all(key in crop_data for key in required_keys):
                raise ValueError(
                    "crop_region must contain x1, y1, x2, y2 coordinates")

            # Convert to integers and validate
            parsed_crop_region = {
                'x1': int(crop_data['x1']),
                'y1': int(crop_data['y1']),
                'x2': int(crop_data['x2']),
                'y2': int(crop_data['y2'])
            }
            if 'page' in crop_data and crop_data['page'] is not None:
                parsed_crop_region['page'] = int(crop_data['page'])

            # Basic sanity check
            if (parsed_crop_region['x1'] >= parsed_crop_region['x2'] or
                    parsed_crop_region['y1'] >= parsed_crop_region['y2']):
                raise ValueError(
                    "Invalid crop region: x1 must be < x2 and y1 must be < y2")

        except json.JSONDecodeError as e:
            set_progress(request_id, "error",
                         f"Invalid JSON in crop_region: {e}", 0, 0)
            raise HTTPException(
                status_code=400, detail=f"Invalid JSON in crop_region: {e}")
        except (ValueError, KeyError) as e:
            set_progress(request_id, "error",
                         f"Invalid crop_region format: {e}", 0, 0)
            raise HTTPException(
                status_code=400, detail=f"Invalid crop_region format: {e}")

    try:
        prompt = build_prompt(normalized_task, schema)
        resolved_device = resolve_device(device)
        await RUNTIME.ensure_loaded(resolved_device)
        processor, model, actual_device = RUNTIME.get()
    except HTTPException as exc:
        clear_cancel_request(request_id)
        set_progress(request_id, "error", str(exc.detail), 0, 0)
        raise
    except Exception as exc:
        clear_cancel_request(request_id)
        set_progress(request_id, "error", str(exc), 0, 0)
        raise

    try:
        content = await file.read()
        input_path = save_temp_upload(file.filename or "upload.bin", content)
        pages = load_pages(input_path, normalized_dpi, parsed_crop_region)
    except HTTPException:
        clear_cancel_request(request_id)
        raise
    except Exception as exc:
        logger.exception("Failed to load input file")
        clear_cancel_request(request_id)
        set_progress(request_id, "error", f"事前処理エラー: {exc}", 0, 0)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if "input_path" in locals():
            Path(input_path).unlink(missing_ok=True)

    total_pages = len(pages)
    set_progress(
        request_id,
        "ocr",
        "OCR準備完了",
        0,
        total_pages,
        0,
        0,
    )

    def build_response(state: str, response_results: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "device": actual_device,
            "task": normalized_task,
            "linebreak_mode": normalized_linebreak_mode,
            "use_layout": use_layout_mode,
            "layout_backend": normalized_layout_backend,
            "reading_order": normalized_reading_order,
            "region_padding": normalized_region_padding,
            "max_regions": normalized_max_regions,
            "region_parallelism": normalized_region_parallelism,
            "state": state,
            "page_count": len(pages),
            "results": response_results,
        }

    def build_canceled_response(
        response_results: list[dict[str, Any]],
        completed_pages: int,
        current_region: int = 0,
        total_regions_for_page: int = 0,
    ) -> dict[str, Any]:
        set_progress(
            request_id,
            "canceled",
            "中断しました",
            max(0, completed_pages),
            total_pages,
            current_region,
            total_regions_for_page,
        )
        clear_cancel_request(request_id)
        return build_response("canceled", response_results)

    results: list[dict[str, Any]] = []
    try:
        for index, page in enumerate(pages, start=1):
            if is_cancel_requested(request_id):
                return build_canceled_response(results, index - 1)

            if not use_layout_mode:
                set_progress(
                    request_id,
                    "ocr",
                    f"{index}/{total_pages}ページをOCR中",
                    index,
                    total_pages,
                    0,
                    0,
                )
                page_path = save_temp_png(page)
                try:
                    async with GENERATE_SEMAPHORE:
                        raw_text, clean_text, truncated = await asyncio.to_thread(
                            glm_infer,
                            processor,
                            model,
                            str(page_path),
                            prompt,
                            normalized_max_new_tokens,
                            temperature,
                            request_id,
                        )
                finally:
                    page_path.unlink(missing_ok=True)

                if is_cancel_requested(request_id):
                    return build_canceled_response(results, index - 1)

                item: dict[str, Any] = {
                    "page": index,
                    "text": (
                        normalize_text_output(
                            clean_text,
                            normalized_task,
                            normalized_linebreak_mode,
                        )
                        if normalized_task != "extract_json"
                        else clean_text
                    ),
                    "raw": raw_text,
                    "json": None,
                    "truncated": bool(truncated),
                }
                if normalized_task == "extract_json":
                    try:
                        item["json"] = json.loads(clean_text)
                    except json.JSONDecodeError as exc:
                        item["error"] = f"JSON parse failed: {exc.msg}"
                results.append(item)
                continue

            set_progress(
                request_id,
                "ocr",
                f"{index}/{total_pages}ページのレイアウト解析中",
                index,
                total_pages,
                0,
                0,
            )
            raw_layout_blocks = await asyncio.to_thread(
                detect_layout_blocks,
                page,
                normalized_layout_backend,
            )
            padded_blocks = [
                LayoutBlock(
                    type=normalize_layout_label(block.type),
                    bbox=clamp_bbox_with_padding(
                        block.bbox, page, normalized_region_padding),
                    score=float(block.score),
                )
                for block in raw_layout_blocks
            ]
            if not padded_blocks:
                width, height = page.size
                padded_blocks = [LayoutBlock(
                    type="text", bbox=(0, 0, width, height), score=1.0)]

            effective_order = resolve_effective_reading_order(
                padded_blocks, normalized_reading_order)
            ordered_blocks = sort_layout_blocks(padded_blocks, effective_order)[
                :normalized_max_regions]
            total_regions_for_page = len(ordered_blocks)
            if total_regions_for_page == 0:
                width, height = page.size
                ordered_blocks = [LayoutBlock(
                    type="text", bbox=(0, 0, width, height), score=1.0)]
                total_regions_for_page = 1

            set_progress(
                request_id,
                "ocr",
                f"{index}/{total_pages} pages, 0/{total_regions_for_page} regions",
                index,
                total_pages,
                0,
                total_regions_for_page,
            )

            region_semaphore = asyncio.Semaphore(normalized_region_parallelism)

            async def infer_region(
                region_index: int,
                layout_block: LayoutBlock,
            ) -> tuple[int, dict[str, Any]]:
                item: dict[str, Any] = {
                    "id": f"b{region_index + 1}",
                    "type": normalize_layout_label(layout_block.type),
                    "bbox": bbox_dict(layout_block.bbox),
                    "text": "",
                    "raw": "",
                    "truncated": False,
                }
                if is_cancel_requested(request_id):
                    item["error"] = "canceled"
                    return region_index, item

                crop = page.crop(layout_block.bbox)
                crop_path = save_temp_png(crop)
                try:
                    region_prompt = block_prompt_for_task(
                        normalized_task,
                        layout_block.type,
                        schema,
                    )
                    async with region_semaphore:
                        raw_text, clean_text, truncated = await asyncio.to_thread(
                            glm_infer,
                            processor,
                            model,
                            str(crop_path),
                            region_prompt,
                            normalized_max_new_tokens,
                            temperature,
                            request_id,
                        )
                    item["raw"] = raw_text
                    item["text"] = (
                        clean_text
                        if normalized_task == "extract_json"
                        else normalize_text_output(
                            clean_text,
                            normalized_task,
                            "none",
                        )
                    )
                    item["truncated"] = bool(truncated)
                except Exception as exc:
                    item["error"] = str(exc)
                finally:
                    crop_path.unlink(missing_ok=True)
                return region_index, item

            block_results: list[Optional[dict[str, Any]]] = [
                None] * total_regions_for_page
            completed_regions = 0
            for start in range(0, total_regions_for_page, normalized_region_parallelism):
                if is_cancel_requested(request_id):
                    return build_canceled_response(
                        results,
                        index - 1,
                        completed_regions,
                        total_regions_for_page,
                    )
                batch = ordered_blocks[start: start +
                                       normalized_region_parallelism]
                batch_jobs = [
                    infer_region(start + offset, block) for offset, block in enumerate(batch)
                ]
                batch_outputs = await asyncio.gather(*batch_jobs)
                for output_index, block_item in batch_outputs:
                    block_results[output_index] = block_item
                    completed_regions += 1
                    set_progress(
                        request_id,
                        "ocr",
                        f"{index}/{total_pages} pages, {completed_regions}/{total_regions_for_page} regions",
                        index,
                        total_pages,
                        completed_regions,
                        total_regions_for_page,
                    )
                    if is_cancel_requested(request_id):
                        return build_canceled_response(
                            results,
                            index - 1,
                            completed_regions,
                            total_regions_for_page,
                        )

            page_blocks = [item for item in block_results if item is not None]
            combined_text = combine_block_texts(
                page_blocks,
                normalized_linebreak_mode if normalized_task != "extract_json" else "none",
            )
            combined_raw = "\n\n".join(
                str(item.get("raw") or "").strip() for item in page_blocks if item
            ).strip()
            page_item: dict[str, Any] = {
                "page": index,
                "text": combined_text,
                "raw": combined_raw,
                "json": None,
                "blocks": page_blocks,
                "reading_order": effective_order,
                "layout_preview_base64": build_layout_preview_base64(page, page_blocks),
            }
            block_errors = [
                f"{block.get('id')}: {block.get('error')}"
                for block in page_blocks
                if block.get("error")
            ]
            if block_errors:
                page_item["error"] = "\n".join(block_errors)
            if normalized_task == "extract_json":
                try:
                    page_item["json"] = json.loads(combined_text)
                except json.JSONDecodeError as exc:
                    page_item["error"] = (
                        f"{page_item.get('error', '')}\nJSON parse failed: {exc.msg}".strip(
                        )
                    )
            results.append(page_item)
    except HTTPException:
        clear_cancel_request(request_id)
        set_progress(request_id, "error", "APIエラー", 0, total_pages)
        raise
    except Exception as exc:
        clear_cancel_request(request_id)
        logger.exception("Inference failed")
        set_progress(request_id, "error", f"推論エラー: {exc}", 0, total_pages)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    set_progress(request_id, "done", "完了", total_pages, total_pages, 0, 0)
    clear_cancel_request(request_id)

    return build_response("done", results)
