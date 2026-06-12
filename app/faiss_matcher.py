from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import Any, Optional


DEFAULT_MATCH_TOP_K = 30


class FaissCandidateMatcher:
    def __init__(
        self,
        root_dir: Path,
        model_cache_dir: Path,
        logger: Any,
        candidates_csv_path: Optional[Path] = None,
        embedding_model_id: Optional[str] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.model_cache_dir = Path(model_cache_dir)
        self.logger = logger
        self.candidates_csv_path = candidates_csv_path or Path(
            os.getenv("CANDIDATES_CSV", str(self.root_dir /
                      "\u5546\u54c1\u30ea\u30b9\u30c8.csv"))
        )
        self.embedding_model_id = embedding_model_id or os.getenv(
            "EMBEDDING_MODEL_ID", "intfloat/multilingual-e5-small"
        )
        self._candidates_cache: Optional[list[dict[str, str]]] = None
        self._embedding_model: Optional[Any] = None
        self._faiss_index: Optional[Any] = None
        self._faiss_candidates: list[dict[str, str]] = []

    @property
    def has_index(self) -> bool:
        return self._faiss_index is not None

    def load_candidates(self, force_reload: bool = False) -> list[dict[str, str]]:
        if self._candidates_cache is not None and not force_reload:
            return self._candidates_cache
        if not self.candidates_csv_path.exists():
            self.logger.warning(
                "Candidates CSV not found: %s", self.candidates_csv_path)
            self._candidates_cache = []
            return []

        rows: list[dict[str, str]] = []
        for enc in ("utf-8-sig", "utf-8", "shift_jis", "cp932"):
            try:
                with open(self.candidates_csv_path, encoding=enc, newline="") as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    for row in reader:
                        if len(row) < 4:
                            continue
                        shohincd = row[0].strip()
                        kikakucd = row[1].strip()
                        shohinnm = row[2].strip()
                        kikaku = row[3].strip()
                        rows.append(
                            {
                                "shohincd": shohincd,
                                "kikakucd": kikakucd,
                                "shohinnm": shohinnm,
                                "kikaku": kikaku,
                                "search_key": f"{shohinnm} {kikaku}".strip(),
                            }
                        )
                break
            except (UnicodeDecodeError, UnicodeError):
                rows = []
                continue

        self._candidates_cache = rows
        self.logger.info(
            "Loaded %d candidate rows from %s",
            len(rows),
            self.candidates_csv_path,
        )
        return rows

    def load_embedding_model(self) -> Any:
        if self._embedding_model is not None:
            return self._embedding_model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is required for candidate matching. "
                "Install it with: pip install sentence-transformers"
            ) from exc

        self.logger.info("Loading embedding model: %s",
                         self.embedding_model_id)
        self._embedding_model = SentenceTransformer(
            self.embedding_model_id,
            cache_folder=str(self.model_cache_dir),
        )
        self.logger.info("Embedding model ready: %s", self.embedding_model_id)
        return self._embedding_model

    def build_faiss_index(self, candidates: Optional[list[dict[str, str]]] = None) -> None:
        if candidates is None:
            candidates = self.load_candidates()
        if not candidates:
            self._faiss_index = None
            self._faiss_candidates = []
            return

        try:
            import faiss
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "faiss is required for candidate matching. "
                "Install it with: pip install faiss-cpu"
            ) from exc

        emb_model = self.load_embedding_model()
        texts = [
            f"passage: {candidate['search_key']}" for candidate in candidates]
        self.logger.info(
            "Building FAISS index for %d candidates...", len(texts))
        embeddings = emb_model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=256,
        )
        embeddings = np.array(embeddings, dtype="float32")
        dim = embeddings.shape[1]
        nlist = 100
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(
            quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index.add(embeddings)
        self._faiss_index = index
        self._faiss_candidates = list(candidates)
        self.logger.info(
            "FAISS index ready (%d candidates, dim=%d)", len(candidates), dim)

    def search_candidates(
        self,
        ocr_text: str,
        top_k: int,
    ) -> tuple[list[dict[str, str]], list[float]]:
        if self._faiss_index is None or not self._faiss_candidates:
            return [], []

        try:
            import numpy as np
        except ImportError:
            return [], []

        emb_model = self.load_embedding_model()
        query_vec = emb_model.encode(
            [f"query: {ocr_text}"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_vec = np.array(query_vec, dtype="float32")
        k = min(top_k, len(self._faiss_candidates))
        scores, indices = self._faiss_index.search(query_vec, k)
        candidates = [self._faiss_candidates[int(
            i)] for i in indices[0] if i >= 0]
        score_list = [float(scores[0][j])
                      for j, i in enumerate(indices[0]) if i >= 0]
        return candidates, score_list

    def status(self) -> dict[str, Any]:
        candidates = self.load_candidates()
        return {
            "count": len(candidates),
            "path": str(self.candidates_csv_path),
            "loaded": len(candidates) > 0,
            "index_built": self.has_index,
            "embedding_model": self.embedding_model_id,
        }
