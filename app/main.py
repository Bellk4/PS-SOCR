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
from typing import Any, Optional, cast

import torch
import httpx
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)

from .layout_ppdoclayoutv3 import LayoutBlock, detect_layout_blocks
from .oauth_integration import (
    OAUTH_BASE_PATH,
    has_local_oauth_session,
    is_oauth_public_path,
    register_oauth_routes,
)

MODEL_ID = "zai-org/GLM-OCR"
ROOT_DIR = Path(__file__).resolve().parents[1]
load_dotenv(ROOT_DIR / ".env")
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
AUTH0_DOMAIN = os.getenv("AUTH0_DOMAIN", "").strip()
AUTH0_USERINFO_URL = f"https://{AUTH0_DOMAIN}/userinfo" if AUTH0_DOMAIN else ""
AUTH_DISABLED = os.getenv("AUTH_DISABLED", "false").strip().lower() in {
    "1", "true", "yes", "on"}
ALLOW_USER_ID_QUERY_WITHOUT_BEARER = os.getenv(
    "ALLOW_USER_ID_QUERY_WITHOUT_BEARER", "false"
).strip().lower() in {"1", "true", "yes", "on"}

logger = logging.getLogger("glm_ocr_server")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

_VIDEO_AUTO_PATCHED_MODULE_IDS: set[int] = set()

# transformersのビデオ自動モジュールで、特定の環境でVIDEO_PROCESSOR_MAPPING_NAMESの値がNoneになる問題への互換パッチ


def patch_transformers_video_auto_none_bug() -> None:
    try:
        from transformers.models.auto import video_processing_auto
    except Exception:
        return

    module_id = id(video_processing_auto)
    if module_id in _VIDEO_AUTO_PATCHED_MODULE_IDS:
        return

    fixed = 0
    for key, value in list(video_processing_auto.VIDEO_PROCESSOR_MAPPING_NAMES.items()):
        if value is None:
            video_processing_auto.VIDEO_PROCESSOR_MAPPING_NAMES[key] = ("", "")
            fixed += 1

    _VIDEO_AUTO_PATCHED_MODULE_IDS.add(module_id)
    if fixed:
        logger.warning(
            "transformersビデオ自動パッチを %d 件に適用しました", fixed)

# デバイス指定を解決するユーティリティ関数


def resolve_device(device: str) -> str:
    requested = (device or "auto").lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        if not torch.cuda.is_available():
            logger.warning(
                "CUDAが要求されましたが利用できません。CPUにフォールバックします。")
            return "cpu"
        return "cuda"
    if requested == "cpu":
        return requested
    raise HTTPException(
        status_code=400, detail=f"サポートされていないデバイス: {device}")

# GLM-OCRのprocessor/modelを読み込み・再利用し、デバイス切替も管理するランタイム。


class GlmRuntime:
    # ランタイム状態（processor/model/現在デバイス）を初期化する。
    def __init__(self) -> None:
        self.processor: Optional[Any] = None
        self.model: Optional[Any] = None
        self.current_device: Optional[str] = None
        self._load_lock = asyncio.Lock()

    # 指定デバイス向けにGLMモデルをロードして返す。
    def _load_model(self, device: str) -> Any:
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
                    "accelerateがありません。device_mapなしでCUDA読み込みにフォールバックします。"
                )
                model = AutoModelForImageTextToText.from_pretrained(
                    MODEL_ID,
                    cache_dir=str(MODEL_CACHE_DIR),
                    torch_dtype="auto",
                    device_map=None,
                )
                return cast(Any, model).to("cuda")

        model = AutoModelForImageTextToText.from_pretrained(
            MODEL_ID,
            cache_dir=str(MODEL_CACHE_DIR),
            torch_dtype=torch.float32,
            device_map=None,
        )
        return cast(Any, model).to("cpu")

    # processor/modelを必要に応じてロードし、要求デバイスへ揃える。
    async def ensure_loaded(self, device: str) -> None:
        async with self._load_lock:
            if self.processor is None:
                logger.info("プロセッサを読み込み中: %s", MODEL_ID)
                patch_transformers_video_auto_none_bug()
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        MODEL_ID,
                        cache_dir=str(MODEL_CACHE_DIR),
                    )
                except ImportError as exc:
                    if "Torchvision library" in str(exc):
                        raise RuntimeError(
                            "GLM-OCRプロセッサにはtorchvisionが必要です。"
                            "pip install torchvision でインストールしてください。"
                        ) from exc
                    raise
                except TypeError as exc:
                    if "NoneType" not in str(exc):
                        raise
                    # 互換パッチ強制適用後に一度リトライする
                    patch_transformers_video_auto_none_bug()
                    self.processor = AutoProcessor.from_pretrained(
                        MODEL_ID,
                        cache_dir=str(MODEL_CACHE_DIR),
                    )

            if self.model is not None and self.current_device == device:
                return

            if self.model is not None:
                logger.info(
                    "モデルデバイスを %s から %s に切り替えています", self.current_device, device
                )
                del self.model
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            logger.info("モデルを読み込み中: %s (デバイス=%s)", MODEL_ID, device)
            self.model = await asyncio.to_thread(self._load_model, device)
            self.current_device = device

    # 初期化済みのprocessor/model/deviceを取得する。
    def get(self) -> tuple[Any, Any, str]:
        if self.processor is None or self.model is None or self.current_device is None:
            raise RuntimeError("GLMランタイムが初期化されていません")
        return self.processor, self.model, self.current_device

# 画像ファイルやPDFファイルからページを読み込むユーティリティ関数


def load_pages(path: Path, dpi: int, crop_region: Optional[dict] = None) -> list[tuple[int, Image.Image]]:

    # 単一ページ画像に対してcrop_regionを適用する。

    def apply_crop_if_needed(image: Image.Image) -> Image.Image:
        if crop_region is None or 'x1' not in crop_region:
            return image
        img_width, img_height = image.size
        x1 = max(0, min(crop_region['x1'], img_width - 1))
        y1 = max(0, min(crop_region['y1'], img_height - 1))
        x2 = max(x1 + 1, min(crop_region['x2'], img_width))
        y2 = max(y1 + 1, min(crop_region['y2'], img_height))
        return image.crop((x1, y1, x2, y2))

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        try:
            import pypdfium2 as pdfium
        except ImportError as exc:
            raise RuntimeError(
                "PDF入力にはpypdfium2が必要です。pip install pypdfium2 でインストールしてください。"
            ) from exc

        pages: list[tuple[int, Image.Image]] = []
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
                source_page_num = page_index + 1
                pdf_page = doc[page_index]
                bitmap = cast(Any, pdf_page).render(scale=scale)
                try:
                    image = bitmap.to_pil().convert("RGB")
                    image = apply_crop_if_needed(image)
                finally:
                    if hasattr(bitmap, "close"):
                        bitmap.close()
                    if hasattr(pdf_page, "close"):
                        pdf_page.close()
                pages.append((source_page_num, image))
        finally:
            if hasattr(doc, "close"):
                doc.close()
        return pages

    with Image.open(path) as image:
        converted_image = image.convert("RGB")
        cropped_image = apply_crop_if_needed(converted_image)
        return [(1, cropped_image)]

# アップロード内容を一時ファイルへ保存し、パスを返す。


def save_temp_upload(upload_name: str, content: bytes) -> Path:
    suffix = Path(upload_name or "upload").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        return Path(tmp.name)

# PIL画像を一時PNGとして保存し、パスを返す。


def save_temp_png(image: Image.Image) -> Path:
    fd, tmp_path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        image.save(tmp_path, format="PNG")
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise
    return Path(tmp_path)

# タスク種別に応じてモデルへ渡すプロンプト文字列を構築する。


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
                detail="task=extract_jsonの場合、schemaが必要です",
            )
        # 公式モデルカードのプロンプトスタイルに合わせる
        return f"以下のJSON形式で画像中の情報を出力してください:\n{schema}"
    raise HTTPException(status_code=400, detail=f"サポートされていないタスク: {task}")


# 文字がCJK（日本語・中国語系）かどうかを判定する。
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


# 改行折り返しされた2行を、文字種に応じて自然に連結する。
def join_soft_wrapped_line(left: str, right: str) -> str:
    if not left:
        return right
    if not right:
        return left
    if is_cjk_char(left[-1]) and is_cjk_char(right[0]):
        return left + right
    return f"{left} {right}"


# 行結合時に改行を維持すべき境界かを判定する。
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


# 指定モード（none/paragraph/compact）で改行を正規化する。
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
        merged_text: str = non_empty[0]
        for line in non_empty[1:]:
            merged_text = join_soft_wrapped_line(merged_text, line)
        return merged_text.strip()

    raise HTTPException(
        status_code=400,
        detail=f"サポートされていない改行モード: {mode}",
    )


# 数字を対応する丸数字Unicodeへ変換する。
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


# TeXの\textcircled記法を丸数字へ置換する。
def normalize_textcircled_notation(text: str) -> str:
    if not text:
        return text

    def replace_match(match: re.Match[str]) -> str:
        raw = (match.group(1) or "").strip()
        if not raw.isdigit():
            return match.group(0)
        symbol = circled_number(int(raw))
        return symbol or match.group(0)

    # "$\\textcircled{1}$" と "\\textcircled{1}" の両形式を変換する
    text = re.sub(r"\$\s*\\textcircled\{(\d+)\}\s*\$", replace_match, text)
    text = re.sub(r"\\textcircled\{(\d+)\}", replace_match, text)
    return text


# OCR出力テキストへタスク別の正規化処理を適用する。
def normalize_text_output(text: str, task: str, linebreak_mode: str) -> str:
    normalized = text
    if task in {"text", "table"}:
        normalized = normalize_textcircled_notation(normalized)
    return normalize_linebreaks(normalized, linebreak_mode)


# bboxへ余白を付けつつ、画像範囲内へクリップする。
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


# bboxタプルをAPI返却用dict形式へ変換する。
def bbox_dict(bbox: tuple[int, int, int, int]) -> dict[str, int]:
    return {"x1": int(bbox[0]), "y1": int(bbox[1]), "x2": int(bbox[2]), "y2": int(bbox[3])}


# レイアウト検出ラベルをOCR処理用の正規化ラベルへ寄せる。
def normalize_layout_label(block_type: str) -> str:
    lowered = (block_type or "text").strip().lower()
    if lowered in {"formula", "equation"}:
        return "formula"
    if lowered in {"table"}:
        return "table"
    if lowered in {"figure", "image", "chart"}:
        return "figure"
    return lowered or "text"


# auto指定時に領域形状・配置から実効読順を推定する。
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
        # 日本語の段組み文書では、右から左への段組み順が一般的
        return "rtl_ttb"
    return "ltr_ttb"


# 横書き向けに行単位へグルーピングし、左右順で並べ替える。
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


# 縦書き右列優先向けに列単位へグルーピングして上から並べる。
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


# 読順設定に応じて適切なソート関数へ振り分ける。
def sort_layout_blocks(blocks: list[LayoutBlock], reading_order: str) -> list[LayoutBlock]:
    if reading_order == "vertical_rl":
        return sort_blocks_vertical_rl(blocks)
    if reading_order == "rtl_ttb":
        return sort_blocks_ltr_or_rtl(blocks, rtl=True)
    return sort_blocks_ltr_or_rtl(blocks, rtl=False)


# 領域タイプと全体タスクから領域単位の推論プロンプトを決定する。
def block_prompt_for_task(global_task: str, block_type: str, schema: Optional[str]) -> str:
    if global_task != "text":
        return build_prompt(global_task, schema)
    normalized_type = normalize_layout_label(block_type)
    if normalized_type == "table":
        return build_prompt("table", None)
    if normalized_type == "formula":
        return build_prompt("formula", None)
    return build_prompt("text", None)


# 領域ごとのOCR文字列を結合し、改行モードを適用した本文を作る。
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


# レイアウト領域の矩形を描画したプレビュー画像をbase64で返す。
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


# 単一画像に対してGLM推論を実行し、raw/clean文字列とtruncatedを返す。
def glm_infer(
    processor: Any,
    model: Any,
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
# {session_id: response} VB.NET連携用
_SESSION_RESULTS: dict[str, dict[str, Any]] = {}
_SESSION_TIMESTAMPS: dict[str, float] = {}  # {session_id: stored_at}
_SESSION_USER_INDEX: dict[str, str] = {}  # {session_id: user_id}
MAX_SESSION_ENTRIES = 100
_LATEST_RESULT: Optional[dict[str, Any]] = None  # ブラウザ単体起動用（最新1件のみ保持）
# {connection_id: response}
_CONNECTION_RESULTS: dict[str, dict[str, Any]] = {}
_CONNECTION_TIMESTAMPS: dict[str, float] = {}  # {connection_id: stored_at}
_CONNECTION_USER_INDEX: dict[str, str] = {}  # {connection_id: user_id}
MAX_CONNECTION_ENTRIES = 300
_WS_CONNECTIONS: dict[str, WebSocket] = {}  # {connection_id: websocket}
# {user_id: response}
_USER_RESULTS: dict[str, dict[str, Any]] = {}
_USER_TIMESTAMPS: dict[str, float] = {}  # {user_id: stored_at}
MAX_USER_ENTRIES = 300


def normalize_connection_id(raw_value: Optional[str]) -> Optional[str]:
    token = (raw_value or "").strip()
    if not token:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,128}", token):
        return None
    return token


def normalize_user_id(raw_value: Optional[str]) -> Optional[str]:
    token = (raw_value or "").strip()
    if not token:
        return None
    return token


def store_scoped_result(
    store: dict[str, dict[str, Any]],
    timestamps: dict[str, float],
    key: str,
    value: dict[str, Any],
    max_entries: int,
) -> None:
    store[key] = value
    timestamps[key] = time.time()
    if len(store) <= max_entries:
        return
    overflow = len(store) - max_entries
    oldest = sorted(timestamps.items(), key=lambda x: x[1])[:overflow]
    for old_key, _ in oldest:
        store.pop(old_key, None)
        timestamps.pop(old_key, None)


def _find_latest_result_for_user_from_scoped_indexes(
    user_id: str,
) -> Optional[dict[str, Any]]:
    latest_key: Optional[str] = None
    latest_ts: float = -1.0

    for conn_id, owner_id in _CONNECTION_USER_INDEX.items():
        if owner_id != user_id:
            continue
        ts = _CONNECTION_TIMESTAMPS.get(conn_id, 0.0)
        if ts > latest_ts and conn_id in _CONNECTION_RESULTS:
            latest_ts = ts
            latest_key = conn_id

    if latest_key is not None:
        return _CONNECTION_RESULTS.get(latest_key)

    latest_session_key: Optional[str] = None
    latest_session_ts: float = -1.0
    for session_id, owner_id in _SESSION_USER_INDEX.items():
        if owner_id != user_id:
            continue
        ts = _SESSION_TIMESTAMPS.get(session_id, 0.0)
        if ts > latest_session_ts and session_id in _SESSION_RESULTS:
            latest_session_ts = ts
            latest_session_key = session_id

    if latest_session_key is not None:
        return _SESSION_RESULTS.get(latest_session_key)

    return None


def _find_latest_result_from_any_scope() -> Optional[dict[str, Any]]:
    latest_result: Optional[dict[str, Any]] = None
    latest_ts: float = -1.0

    for conn_id, ts in _CONNECTION_TIMESTAMPS.items():
        if ts <= latest_ts:
            continue
        result = _CONNECTION_RESULTS.get(conn_id)
        if result is None:
            continue
        latest_result = result
        latest_ts = ts

    for session_id, ts in _SESSION_TIMESTAMPS.items():
        if ts <= latest_ts:
            continue
        result = _SESSION_RESULTS.get(session_id)
        if result is None:
            continue
        latest_result = result
        latest_ts = ts

    if latest_result is not None:
        return latest_result
    return _LATEST_RESULT


LISTVIEW_COLUMNS = ["品番", "部品名", "材質", "処理", "個数"]


def normalize_field_name(name: str) -> str:
    if not name:
        return ""
    lowered = str(name).strip().lower()
    return re.sub(r"[\s_／/・\-　\(\)（）\[\]{}]+", "", lowered)


def _pick_first_value(row: dict[str, Any], aliases: set[str]) -> str:
    for key, value in row.items():
        if normalize_field_name(str(key)) not in aliases:
            continue
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _iter_row_dicts_from_page(page_result: dict[str, Any]) -> list[dict[str, Any]]:
    raw = page_result.get("json")
    rows: list[dict[str, Any]] = []
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                rows.append(item)
        return rows

    if not isinstance(raw, dict):
        return rows

    scalar_values = all(
        not isinstance(v, (dict, list))
        for v in raw.values()
    )
    if scalar_values:
        rows.append(raw)
        return rows

    for value in raw.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def _iter_text_lines_from_page(page_result: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    for key in ("text", "raw"):
        value = page_result.get(key)
        if not isinstance(value, str):
            continue
        for line in value.splitlines():
            stripped = line.strip()
            if stripped:
                lines.append(stripped)
    return lines


def _parse_listview_row_from_text_line(line: str) -> Optional[dict[str, str]]:
    if not line:
        return None

    # ヘッダー行は除外する（OCR空白揺れを normalize_field_name で吸収）。
    normalized = normalize_field_name(line)
    if all(token in normalized for token in ("品番", "部品名", "材質", "処理", "個数")):
        return None

    match = re.match(
        r"^(?P<hinban>\S+)\s+(?P<body>.+?)\s+(?P<qty>\d+)\s*本\s*$", line)
    if not match:
        return None

    hinban = match.group("hinban").strip()
    qty = match.group("qty").strip()
    body_tokens = [token for token in match.group("body").split() if token]
    if len(body_tokens) < 2:
        return None

    process = ""
    material = ""
    name = ""
    if len(body_tokens) >= 3:
        process = body_tokens[-1]
        material = body_tokens[-2]
        name = " ".join(body_tokens[:-2]).strip()
    else:
        material = body_tokens[-1]
        name = " ".join(body_tokens[:-1]).strip()

    if not (hinban and name and material and qty):
        return None

    return {
        "品番": hinban,
        "部品名": name,
        "材質": material,
        "処理": process,
        "個数": qty,
    }


def build_listview_rows(results: list[dict[str, Any]]) -> list[dict[str, str]]:
    if not results:
        return []

    aliases_hinban = {
        normalize_field_name(k)
        for k in ["品番", "部品番号", "部番", "part_no", "partnumber", "hinban", "shohincd"]
    }
    aliases_name = {
        normalize_field_name(k)
        for k in ["部品名", "品名", "名称", "name", "part_name", "partname", "shohinnm"]
    }
    aliases_material = {
        normalize_field_name(k)
        for k in ["材質", "材", "material", "zaishitsu"]
    }
    aliases_process = {
        normalize_field_name(k)
        for k in ["処理", "表面処理", "処置", "process", "shori"]
    }
    aliases_qty = {
        normalize_field_name(k)
        for k in ["個数", "数量", "数", "qty", "quantity", "kosu"]
    }

    normalized_rows: list[dict[str, str]] = []
    for page_result in results:
        page_rows = _iter_row_dicts_from_page(page_result)
        if page_rows:
            for row in page_rows:
                item = {
                    "品番": _pick_first_value(row, aliases_hinban),
                    "部品名": _pick_first_value(row, aliases_name),
                    "材質": _pick_first_value(row, aliases_material),
                    "処理": _pick_first_value(row, aliases_process),
                    "個数": _pick_first_value(row, aliases_qty),
                }
                if any(item.values()):
                    normalized_rows.append(item)
            continue

        # JSON行が無いページのみ、VB.NET移行期向けにテキスト行フォールバックを試す。
        for line in _iter_text_lines_from_page(page_result):
            fallback_item = _parse_listview_row_from_text_line(line)
            if fallback_item is not None:
                normalized_rows.append(fallback_item)

    return normalized_rows


def build_latest_result_payload(result: dict[str, Any]) -> dict[str, Any]:
    response_results = result.get("results", [])
    if not isinstance(response_results, list):
        response_results = []

    listview_rows = build_listview_rows(response_results)
    return {
        "results": response_results,
        "listview": {
            "columns": LISTVIEW_COLUMNS,
            "rows": listview_rows,
            "row_count": len(listview_rows),
        },
        "listview_rows": listview_rows,
    }


def _result_belongs_to_user(result: Optional[dict[str, Any]], user_id: str) -> bool:
    if result is None:
        return False
    owner_user_id = normalize_user_id(
        cast(Optional[str], result.get("user_id")))
    return owner_user_id == user_id

# リクエスト進捗状態を保存し、上限を超えた古い履歴を間引く。


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
        # 古いエントリを削除してメモリを制限する
        oldest = sorted(PROGRESS_STATE.items(), key=lambda item: item[1]["updated_at"])[
            : len(PROGRESS_STATE) - MAX_PROGRESS_ENTRIES
        ]
        for key, _ in oldest:
            PROGRESS_STATE.pop(key, None)


# 指定リクエストに中断要求が出ているかを返す。
def is_cancel_requested(request_id: str) -> bool:
    return request_id in CANCEL_REQUESTS


# 生成中に中断要求フラグを監視し、トークン生成を停止するStoppingCriteria。
class CancelStoppingCriteria(StoppingCriteria):
    # 中断監視対象のrequest_idを保持する。
    def __init__(self, request_id: str) -> None:
        self.request_id = request_id

    # 生成ループ中に中断要求を検知したらTrueを返して停止させる。
    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs: Any,
    ) -> torch.BoolTensor:
        return cast(
            torch.BoolTensor,
            torch.tensor(
                [is_cancel_requested(self.request_id)],
                device=input_ids.device,
                dtype=torch.bool,
            ),
        )


# 中断要求フラグをクリアする。
def clear_cancel_request(request_id: str) -> None:
    CANCEL_REQUESTS.discard(request_id)


# 中断要求を受け付け、進捗状態をcancel_requestedへ更新する。
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


@app.websocket("/ws/session")
async def ws_session(websocket: WebSocket) -> None:
    requested_id = normalize_connection_id(
        websocket.query_params.get("connection_id")
    )
    connection_id = requested_id or uuid.uuid4().hex

    await websocket.accept()

    old_socket = _WS_CONNECTIONS.get(connection_id)
    if old_socket is not None and old_socket is not websocket:
        try:
            await old_socket.close(code=1000)
        except Exception:
            pass

    _WS_CONNECTIONS[connection_id] = websocket
    _CONNECTION_TIMESTAMPS[connection_id] = time.time()
    await websocket.send_json({"type": "connected", "connection_id": connection_id})

    try:
        while True:
            message = await websocket.receive_text()
            _CONNECTION_TIMESTAMPS[connection_id] = time.time()
            if message == "ping":
                await websocket.send_json({"type": "pong", "connection_id": connection_id})
            elif message == "get_connection_id":
                await websocket.send_json(
                    {"type": "connected", "connection_id": connection_id}
                )
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("WebSocketセッション処理でエラーが発生しました")
    finally:
        _WS_CONNECTIONS.pop(connection_id, None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OAUTH_CLIENT = register_oauth_routes(app, ROOT_DIR, logger)

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


async def get_authenticated_user(request: Request) -> Optional[dict[str, Any]]:
    if AUTH_DISABLED:
        return None

    if OAUTH_CLIENT is not None and has_local_oauth_session(request):
        try:
            user = await OAUTH_CLIENT.get_user({"request": request})
            return cast(Optional[dict[str, Any]], user)
        except Exception as exc:
            logger.warning("OAuthユーザー取得に失敗しました: %s", exc)

    authorization = (request.headers.get("Authorization") or "").strip()
    if not authorization.lower().startswith("bearer "):
        return None
    if not AUTH0_USERINFO_URL:
        logger.warning("AUTH0_DOMAIN が未設定のため Bearer token を検証できません")
        return None

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                AUTH0_USERINFO_URL,
                headers={"Authorization": authorization},
            )
        if response.status_code != 200:
            logger.warning(
                "Auth0 userinfo 取得に失敗しました: status=%s body=%s",
                response.status_code,
                response.text,
            )
            return None
        payload = response.json()
        if not isinstance(payload, dict):
            return None
        return cast(Optional[dict[str, Any]], payload)
    except Exception as exc:
        logger.warning("Bearer token から OAuth ユーザー取得に失敗しました: %s", exc)
        return None


@app.get("/api/user/sessions")
async def get_user_sessions(
    request: Request,
    user_id: Optional[str] = Query(None, description="ユーザーID")
) -> dict[str, Any]:
    """
    指定されたユーザーIDに関連付けられたアクティブなconnection_idのリストを返す
    VB.NETアプリからユーザーのアクティブセッション一覧を取得するために使用
    """
    authenticated_user = await get_authenticated_user(request)
    auth_user_id = normalize_user_id(
        str((authenticated_user or {}).get("sub") or "")
    )
    query_user_id = normalize_user_id(user_id)
    target_user_id = auth_user_id or query_user_id
    if not target_user_id:
        raise HTTPException(status_code=400, detail="user_idパラメータが必要です")
    if auth_user_id and query_user_id and auth_user_id != query_user_id:
        raise HTTPException(
            status_code=403, detail="指定user_idへのアクセスは許可されていません")

    logger.info(f"/api/user/sessions called for user_id={target_user_id}")

    # アクティブなconnection_idとそのメタデータを返す
    active_sessions = []
    current_time = time.time()

    # _CONNECTION_TIMESTAMPSから、対象ユーザーのアクティブなセッションのみ取得
    for conn_id, timestamp in list(_CONNECTION_TIMESTAMPS.items()):
        if _CONNECTION_USER_INDEX.get(conn_id) != target_user_id:
            continue
        # 1時間以内のセッションのみ返す
        if current_time - timestamp < 3600:
            result = _CONNECTION_RESULTS.get(conn_id)
            active_sessions.append({
                "connection_id": conn_id,
                "timestamp": timestamp,
                "has_result": result is not None,
                "result_pages": len(result.get("results", [])) if result else 0
            })

    # タイムスタンプの降順でソート（新しいものが先）
    active_sessions.sort(key=lambda x: x["timestamp"], reverse=True)

    logger.info(
        f"/api/user/sessions returning {len(active_sessions)} sessions for user_id={target_user_id}")

    return {
        "user_id": target_user_id,
        "sessions": active_sessions,
        "has_latest_result": target_user_id in _USER_RESULTS,
        "latest_result_pages": len(
            (_USER_RESULTS.get(target_user_id) or {}).get("results", [])
        ) if target_user_id in _USER_RESULTS else 0,
    }


@app.middleware("http")
async def require_login_for_app(request: Request, call_next):
    if AUTH_DISABLED:
        return await call_next(request)

    if OAUTH_CLIENT is None or request.method == "OPTIONS":
        return await call_next(request)

    path = request.url.path or "/"
    if is_oauth_public_path(path):
        return await call_next(request)

    # 社内連携用途: Bearerなしでも user_id 指定で latest_result を許可するオプション
    if (
        ALLOW_USER_ID_QUERY_WITHOUT_BEARER
        and path == "/api/latest_result"
        and normalize_user_id(request.query_params.get("user_id"))
        and not request.headers.get("Origin")
        and not request.headers.get("Referer")
        and not has_local_oauth_session(request)
    ):
        return await call_next(request)

    user = await get_authenticated_user(request)
    if user:
        return await call_next(request)

    if path.startswith("/api/"):
        return JSONResponse(
            status_code=401,
            content={
                "detail": "ログインが必要です",
                "oauth_login_url": f"{OAUTH_BASE_PATH}/login",
            },
        )

    return RedirectResponse(url=f"{OAUTH_BASE_PATH}/login", status_code=307)


@app.on_event("startup")
# サーバー起動時にモデルキャッシュを準備し、既定デバイスで初期ロードする。
async def startup_load_model() -> None:
    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    default_device = resolve_device("auto")
    await RUNTIME.ensure_loaded(default_device)
    logger.info(
        "起動完了 (デバイス=%s, キャッシュディレクトリ=%s)",
        default_device,
        MODEL_CACHE_DIR,
    )


@app.get("/", response_class=HTMLResponse)
# UI本体（index.html）を返す。
async def index() -> HTMLResponse:
    html_path = static_dir / "index.html"
    if not html_path.exists():
        raise HTTPException(
            status_code=500,
            detail="UIが見つかりません。app/static/index.htmlが存在することを確認してください。",
        )
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/status")
# 実行環境状態（CUDA可否・既定デバイス・モデル情報）を返す。
async def status(request: Request) -> dict[str, Any]:
    authenticated_user = await get_authenticated_user(request)
    current_user_id = normalize_user_id(
        str((authenticated_user or {}).get("sub") or "")
    ) or None
    return {
        "cuda_available": torch.cuda.is_available(),
        "device_default": "cuda" if torch.cuda.is_available() else "cpu",
        "model": MODEL_ID,
        "model_cache_dir": str(MODEL_CACHE_DIR),
        "oauth_enabled": OAUTH_CLIENT is not None,
        "oauth_base_path": OAUTH_BASE_PATH if OAUTH_CLIENT is not None else None,
        "current_user_id": current_user_id,
    }


@app.get("/api/progress/{request_id}")
# request_id単位の進捗状態を返す。
async def progress(request_id: str) -> dict[str, Any]:
    item = PROGRESS_STATE.get(request_id)
    if item is None:
        raise HTTPException(status_code=404, detail="進捗情報が見つかりません")
    return item


@app.post("/api/cancel/{request_id}")
# 指定リクエストへ中断要求を送る。
async def cancel(request_id: str) -> dict[str, Any]:
    return request_cancel(request_id)


@app.get("/api/latest_result")
# session_id指定時（VB.NET連携）はセッション結果を返す。未指定時（ブラウザ単体）は直近の結果を返す。
async def latest_result(
    request: Request,
    connection_id: Optional[str] = Query(
        None, description="WebSocket接続ID（ブラウザインスタンス単位）"
    ),
    session_id: Optional[str] = Query(
        None, description="VB.NETアプリが発行したセッションID"),
    user_id: Optional[str] = Query(
        None, description="ログインユーザーID（Auth0 sub）"
    ),
) -> dict[str, Any]:
    authenticated_user = await get_authenticated_user(request)
    has_local_session = OAUTH_CLIENT is not None and has_local_oauth_session(
        request)
    auth_user_id = normalize_user_id(
        str((authenticated_user or {}).get("sub") or "")
    )
    query_user_id = normalize_user_id(user_id)

    # ログインセッションがあるブラウザでは、未認証扱いでの user_id 参照を許可しない。
    if has_local_session and query_user_id and not auth_user_id:
        raise HTTPException(
            status_code=401,
            detail="ログインセッションの検証に失敗しました。再ログインしてください",
        )

    # Bearerなし運用時は user_id 指定を必須にし、他スコープへのフォールバックを禁止
    if (
        ALLOW_USER_ID_QUERY_WITHOUT_BEARER
        and not auth_user_id
        and not query_user_id
    ):
        raise HTTPException(
            status_code=401,
            detail="Bearer未使用モードでは user_id パラメータが必要です",
        )

    target_user_id = auth_user_id or query_user_id
    if auth_user_id and query_user_id and auth_user_id != query_user_id:
        raise HTTPException(
            status_code=403, detail="指定user_idへのアクセスは許可されていません")

    if target_user_id:
        result = _USER_RESULTS.get(target_user_id)
        if result is not None:
            return build_latest_result_payload(result)
        scoped_result = _find_latest_result_for_user_from_scoped_indexes(
            target_user_id
        )
        if scoped_result is not None:
            return build_latest_result_payload(scoped_result)
        if _result_belongs_to_user(_LATEST_RESULT, target_user_id):
            return build_latest_result_payload(cast(dict[str, Any], _LATEST_RESULT))
        # user_id指定時は他スコープへフォールバックせず、ユーザー結果のみ返す。
        raise HTTPException(status_code=404, detail="指定user_idの解析結果がありません")

    normalized_connection_id = normalize_connection_id(connection_id)
    if normalized_connection_id:
        result = _CONNECTION_RESULTS.get(normalized_connection_id)
        if result is not None:
            return build_latest_result_payload(result)
        raise HTTPException(
            status_code=404, detail="指定connection_idの解析結果がありません")

    if session_id:
        # VB.NET連携モード: セッションキーで検索
        result = _SESSION_RESULTS.get(session_id)
        if result is not None:
            return build_latest_result_payload(result)
        if target_user_id:
            raise HTTPException(
                status_code=404, detail="指定session_idの解析結果がありません")
        logger.info(
            "/api/latest_result: session_id=%s not found, falling back to latest", session_id)
        if _LATEST_RESULT is not None:
            return build_latest_result_payload(_LATEST_RESULT)
        raise HTTPException(status_code=404, detail="まだ解析結果がありません")
    # ブラウザ単体モード: 直近の1件を返す
    if _LATEST_RESULT is None:
        raise HTTPException(status_code=404, detail="まだ解析結果がありません")
    return build_latest_result_payload(_LATEST_RESULT)


@app.post("/api/analyze")
# OCRリクエストを受け取り、前処理・推論・後処理を実行して結果を返す。
async def analyze(
    request: Request,
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
    session_id: Optional[str] = Form(None),
    connection_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
) -> dict[str, Any]:
    request_id = (request_id or "").strip() or uuid.uuid4().hex
    # session_idは Form または URL クエリパラメータのどちらからでも受け取る
    raw_session_id = session_id or request.query_params.get("session_id")
    normalized_session_id = (raw_session_id or "").strip() or None
    raw_connection_id = connection_id or request.query_params.get(
        "connection_id")
    normalized_connection_id = normalize_connection_id(raw_connection_id)
    authenticated_user = await get_authenticated_user(request)
    auth_user_id = normalize_user_id(
        str((authenticated_user or {}).get("sub") or "")
    )
    raw_user_id = user_id or request.query_params.get("user_id")
    query_user_id = normalize_user_id(raw_user_id)
    if auth_user_id and query_user_id and auth_user_id != query_user_id:
        raise HTTPException(
            status_code=403, detail="指定user_idへのアクセスは許可されていません"
        )
    # Bearerなし運用時は解析時にも user_id を必須化する。
    # これにより user_id 未保存のまま latest_result が404になる取りこぼしを防ぐ。
    if ALLOW_USER_ID_QUERY_WITHOUT_BEARER and not auth_user_id and not query_user_id:
        raise HTTPException(
            status_code=401,
            detail="Bearer未使用モードでは user_id パラメータが必要です",
        )
    # ログイン時はAuth0ユーザーID、未ログイン時は明示user_idを採用
    normalized_user_id = auth_user_id or query_user_id
    clear_cancel_request(request_id)
    set_progress(request_id, "preprocessing", "事前処理中", 0, 0)

    normalized_task = (task or "text").strip().lower()
    if normalized_task not in ALLOWED_TASKS:
        set_progress(request_id, "error", f"サポートされていないタスク: {task}", 0, 0)
        raise HTTPException(
            status_code=400, detail=f"サポートされていないタスク: {task}")
    normalized_linebreak_mode = (linebreak_mode or "none").strip().lower()
    if normalized_linebreak_mode not in ALLOWED_LINEBREAK_MODES:
        set_progress(
            request_id,
            "error",
            f"サポートされていない改行モード: {linebreak_mode}",
            0,
            0,
        )
        raise HTTPException(
            status_code=400, detail=f"サポートされていない改行モード: {linebreak_mode}"
        )
    normalized_layout_backend = (
        layout_backend or DEFAULT_LAYOUT_BACKEND).strip().lower()
    if normalized_layout_backend not in ALLOWED_LAYOUT_BACKENDS:
        set_progress(
            request_id,
            "error",
            f"サポートされていないレイアウトバックエンド: {layout_backend}",
            0,
            0,
        )
        raise HTTPException(
            status_code=400,
            detail=f"サポートされていないレイアウトバックエンド: {layout_backend}",
        )
    normalized_reading_order = (
        reading_order or DEFAULT_READING_ORDER).strip().lower()
    if normalized_reading_order not in ALLOWED_READING_ORDERS:
        set_progress(
            request_id,
            "error",
            f"サポートされていない読み取り順: {reading_order}",
            0,
            0,
        )
        raise HTTPException(
            status_code=400,
            detail=f"サポートされていない読み取り順: {reading_order}",
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

    # crop_regionが指定されている場合はパースする
    parsed_crop_region = None
    if crop_region and crop_region.strip():
        try:
            crop_data = json.loads(crop_region.strip())
            # crop_regionの形式を検証
            if not isinstance(crop_data, dict):
                raise ValueError("crop_regionはJSONオブジェクトでなければなりません")

            coord_keys = {'x1', 'y1', 'x2', 'y2'}
            has_coords = all(key in crop_data for key in coord_keys)
            has_page = 'page' in crop_data and crop_data['page'] is not None

            if not has_coords and not has_page:
                raise ValueError(
                    "crop_regionにはx1, y1, x2, y2座標またはpageが必要です")

            parsed_crop_region = {}
            if has_page:
                parsed_crop_region['page'] = int(crop_data['page'])

            if has_coords:
                # 整数に変換して検証
                parsed_crop_region.update({
                    'x1': int(crop_data['x1']),
                    'y1': int(crop_data['y1']),
                    'x2': int(crop_data['x2']),
                    'y2': int(crop_data['y2'])
                })
                # 基本的な妥当性チェック
                if (parsed_crop_region['x1'] >= parsed_crop_region['x2'] or
                        parsed_crop_region['y1'] >= parsed_crop_region['y2']):
                    raise ValueError(
                        "無効なクロップ範囲: x1はx2より小さく、y1はy2より小さい必要があります")

        except json.JSONDecodeError as e:
            set_progress(request_id, "error",
                         f"crop_regionの無効なJSON: {e}", 0, 0)
            raise HTTPException(
                status_code=400, detail=f"crop_regionの無効なJSON: {e}")
        except (ValueError, KeyError) as e:
            set_progress(request_id, "error",
                         f"無効なcrop_region形式: {e}", 0, 0)
            raise HTTPException(
                status_code=400, detail=f"無効なcrop_region形式: {e}")

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

    input_path: Optional[Path] = None
    try:
        content = await file.read()
        input_path = save_temp_upload(file.filename or "upload.bin", content)
        page_tuples = load_pages(
            input_path, normalized_dpi, parsed_crop_region)
    except HTTPException:
        clear_cancel_request(request_id)
        raise
    except Exception as exc:
        logger.exception("入力ファイルの読み込みに失敗しました")
        clear_cancel_request(request_id)
        set_progress(request_id, "error", f"事前処理エラー: {exc}", 0, 0)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if input_path is not None:
            input_path.unlink(missing_ok=True)

    total_pages = len(page_tuples)
    pages = [img for _, img in page_tuples]
    page_numbers = [num for num, _ in page_tuples]
    set_progress(
        request_id,
        "ocr",
        "OCR準備完了",
        0,
        total_pages,
        0,
        0,
    )

    # APIレスポンス本体を組み立てる。
    def build_response(state: str, response_results: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "request_id": request_id,
            "user_id": normalized_user_id,
            "session_id": normalized_session_id,
            "connection_id": normalized_connection_id,
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

    # 中断時の進捗反映とレスポンス構築を行う。
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
        for index, (source_page, page) in enumerate(zip(page_numbers, pages), start=1):
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
                    "page": source_page,
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

            # 単一レイアウト領域を切り出してOCRし、領域結果を返す。
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
                "page": source_page,
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
        logger.exception("推論に失敗しました")
        set_progress(request_id, "error", f"推論エラー: {exc}", 0, total_pages)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    set_progress(request_id, "done", "完了", total_pages, total_pages, 0, 0)
    clear_cancel_request(request_id)

    response = build_response("done", results)
    # user_id指定時（ログインユーザー単位）はそのキーで保存
    # connection_id指定時（WebSocket連携）はそのキーでも保存
    # session_id指定時（VB.NET連携）はそのキーでも保存
    # どちらも未指定時（ブラウザ単体）は最新で1件のみ上書き保存
    global _LATEST_RESULT
    stored = False
    if normalized_connection_id:
        store_scoped_result(
            _CONNECTION_RESULTS,
            _CONNECTION_TIMESTAMPS,
            normalized_connection_id,
            response,
            MAX_CONNECTION_ENTRIES,
        )
        if normalized_user_id:
            _CONNECTION_USER_INDEX[normalized_connection_id] = normalized_user_id
        stored = True
    if normalized_session_id:
        store_scoped_result(
            _SESSION_RESULTS,
            _SESSION_TIMESTAMPS,
            normalized_session_id,
            response,
            MAX_SESSION_ENTRIES,
        )
        if normalized_user_id:
            _SESSION_USER_INDEX[normalized_session_id] = normalized_user_id
        stored = True
    if normalized_user_id:
        store_scoped_result(
            _USER_RESULTS,
            _USER_TIMESTAMPS,
            normalized_user_id,
            response,
            MAX_USER_ENTRIES,
        )
        stored = True
    # user_id/connection/session 保存有無に関係なく、最新結果ミラーは更新する
    _LATEST_RESULT = response
    return response
