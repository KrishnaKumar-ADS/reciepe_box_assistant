"""Local semantic recipe search using Sentence Transformers + FAISS.

Embeddings and FAISS files are generated locally, persisted under data/vector_index,
and reused until the source dataset fingerprint or model changes.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from .dataset import Recipe

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_VERSION = 1


def dataset_fingerprint(data_dir: Path, filenames: Iterable[str] | None = None) -> str:
    """Fast fingerprint invalidated by a changed JSON file name, size, or mtime."""
    names = sorted(filenames) if filenames is not None else sorted(p.name for p in data_dir.glob("*.json"))
    digest = hashlib.sha256()
    for name in names:
        path = data_dir / name
        if not path.is_file():
            continue
        stat = path.stat()
        digest.update(f"{name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("utf-8"))
    return digest.hexdigest()


def recipe_embedding_text(recipe: Recipe) -> str:
    """Build a compact searchable representation; exclude long instructions by default."""
    title = (recipe.title or "").strip()
    ingredients = "; ".join(str(item).strip() for item in recipe.ingredients if str(item).strip())
    # A short method excerpt adds useful preparation context without dominating tokens.
    method = " ".join(str(item).strip() for item in recipe.instructions[:2] if str(item).strip())
    method = method[:420]
    return f"Recipe: {title}. Ingredients: {ingredients}. Preparation: {method}"[:2200]


def _load_sentence_model(model_name: str, device: str | None = None):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised by CLI error path
        raise RuntimeError(
            "Sentence Transformers is not installed. Run: pip install -r requirements.txt"
        ) from exc
    # None lets Sentence Transformers choose CUDA when available, otherwise CPU.
    return SentenceTransformer(model_name, device=device)


def _try_import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except (ImportError, OSError):
        return None


@dataclass(slots=True)
class VectorIndexInfo:
    ready: bool
    count: int = 0
    model_name: str = DEFAULT_EMBEDDING_MODEL
    dimension: int = 0
    metric: str = "cosine"
    error: str | None = None


class RecipeVectorIndex:
    """Read-only semantic index. The encoder is loaded lazily on first search."""

    def __init__(self, index_dir: Path, manifest: dict[str, Any], recipe_ids: list[str],
                 vectors: np.ndarray | None, faiss_index: Any = None):
        self.index_dir = index_dir
        self.manifest = manifest
        self.recipe_ids = recipe_ids
        self.vectors = vectors
        self.faiss_index = faiss_index
        self.model_name = str(manifest["model_name"])
        self._model = None
        self._id_positions = {recipe_id: i for i, recipe_id in enumerate(recipe_ids)}

    @classmethod
    def load(cls, index_dir: Path, expected_fingerprint: str | None = None,
             expected_model: str = DEFAULT_EMBEDDING_MODEL) -> "RecipeVectorIndex | None":
        manifest_path = index_dir / "manifest.json"
        ids_path = index_dir / "recipe_ids.json"
        vectors_path = index_dir / "vectors.npy"
        faiss_path = index_dir / "recipes.faiss"
        if not (manifest_path.is_file() and ids_path.is_file()):
            return None
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("index_version") != INDEX_VERSION:
                return None
            if manifest.get("model_name") != expected_model:
                return None
            if expected_fingerprint and manifest.get("dataset_fingerprint") != expected_fingerprint:
                return None
            recipe_ids = json.loads(ids_path.read_text(encoding="utf-8"))
            vectors = None
            faiss_index = None
            faiss = _try_import_faiss()
            if faiss is not None and faiss_path.is_file():
                try:
                    faiss_index = faiss.read_index(str(faiss_path))
                    if int(faiss_index.ntotal) != len(recipe_ids):
                        faiss_index = None
                except Exception:
                    faiss_index = None
            if faiss_index is None and vectors_path.is_file():
                vectors = np.load(vectors_path, mmap_mode="r")
                if vectors.ndim != 2 or len(recipe_ids) != vectors.shape[0]:
                    return None
            elif faiss_index is None:
                return None
            return cls(index_dir, manifest, recipe_ids, vectors, faiss_index)
        except Exception:
            return None

    @property
    def info(self) -> VectorIndexInfo:
        return VectorIndexInfo(
            ready=True,
            count=len(self.recipe_ids),
            model_name=self.model_name,
            dimension=int(self.manifest.get("embedding_dimension", 0)),
        )

    def _get_model(self):
        if self._model is None:
            self._model = _load_sentence_model(self.model_name)
        return self._model

    def search(self, query: str, top_k: int = 100) -> list[tuple[str, float]]:
        query = (query or "").strip()
        if not query or not self.recipe_ids:
            return []
        model = self._get_model()
        query_vector = model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )
        query_vector = np.asarray(query_vector, dtype=np.float32)
        k = max(1, min(int(top_k), len(self.recipe_ids)))
        if self.faiss_index is not None:
            scores, positions = self.faiss_index.search(query_vector, k)
            return [
                (self.recipe_ids[int(position)], float(score))
                for score, position in zip(scores[0], positions[0])
                if int(position) >= 0
            ]
        # Exact cosine similarity fallback; vectors are L2-normalized and memory-mapped.
        if self.vectors is None:
            raise RuntimeError("Neither a readable FAISS index nor a NumPy vector matrix is available.")
        scores = np.asarray(self.vectors @ query_vector[0], dtype=np.float32)
        if k == len(scores):
            positions = np.argsort(scores)[::-1]
        else:
            positions = np.argpartition(scores, -k)[-k:]
            positions = positions[np.argsort(scores[positions])[::-1]]
        return [(self.recipe_ids[int(i)], float(scores[int(i)])) for i in positions]


def index_info(index_dir: Path, expected_fingerprint: str | None = None) -> VectorIndexInfo:
    index = RecipeVectorIndex.load(index_dir, expected_fingerprint=expected_fingerprint)
    return index.info if index else VectorIndexInfo(ready=False)


def build_vector_index(
    recipes: list[Recipe],
    index_dir: Path,
    source_fingerprint: str,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = 64,
    chunk_size: int = 1024,
    device: str | None = None,
    force: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
    encoder: Any = None,
) -> VectorIndexInfo:
    """Generate normalized recipe vectors once and persist a FAISS inner-product index."""
    if not recipes:
        raise ValueError("No recipes were loaded. Load the three dataset JSON files first.")
    existing = RecipeVectorIndex.load(index_dir, expected_fingerprint=source_fingerprint,
                                      expected_model=model_name)
    if existing and not force:
        return existing.info

    model = encoder or _load_sentence_model(model_name, device=device)
    dimension = int(model.get_sentence_embedding_dimension())
    if dimension <= 0:
        raise RuntimeError("The embedding model did not report a valid vector dimension.")

    index_dir.mkdir(parents=True, exist_ok=True)
    recipe_ids = [recipe.recipe_id for recipe in recipes]
    texts = [recipe_embedding_text(recipe) for recipe in recipes]

    # Build temporary artifacts, then publish the manifest last so incomplete builds
    # are never mistaken for a ready index.
    tmp_dir = Path(tempfile.mkdtemp(prefix="vector_build_", dir=str(index_dir)))
    try:
        tmp_vectors = tmp_dir / "vectors.npy"
        matrix = np.lib.format.open_memmap(
            tmp_vectors, mode="w+", dtype=np.float32, shape=(len(texts), dimension)
        )
        for start in range(0, len(texts), max(1, chunk_size)):
            end = min(start + max(1, chunk_size), len(texts))
            encoded = model.encode(
                texts[start:end], batch_size=max(1, batch_size),
                normalize_embeddings=True, convert_to_numpy=True,
                show_progress_bar=False,
            )
            encoded = np.asarray(encoded, dtype=np.float32)
            if encoded.shape != (end - start, dimension):
                raise RuntimeError(
                    f"Embedding model returned shape {encoded.shape}; expected {(end - start, dimension)}."
                )
            matrix[start:end] = encoded
            if progress_callback:
                progress_callback(end, len(texts))
        matrix.flush()
        del matrix

        tmp_ids = tmp_dir / "recipe_ids.json"
        tmp_ids.write_text(json.dumps(recipe_ids, ensure_ascii=False), encoding="utf-8")

        faiss = _try_import_faiss()
        tmp_faiss = tmp_dir / "recipes.faiss"
        has_faiss = False
        if faiss is not None:
            vectors = np.load(tmp_vectors, mmap_mode="r")
            faiss_index = faiss.IndexFlatIP(dimension)  # exact cosine search on normalized vectors
            add_batch = 10000
            for start in range(0, len(vectors), add_batch):
                faiss_index.add(np.asarray(vectors[start:start + add_batch], dtype=np.float32))
            faiss.write_index(faiss_index, str(tmp_faiss))
            del vectors  # Release the Windows file mapping before temporary cleanup.
            has_faiss = True

        manifest = {
            "index_version": INDEX_VERSION,
            "model_name": model_name,
            "embedding_dimension": dimension,
            "vector_count": len(recipes),
            "dataset_fingerprint": source_fingerprint,
            "similarity": "cosine_similarity_via_normalized_inner_product",
            "faiss_index": has_faiss,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        tmp_manifest = tmp_dir / "manifest.json"
        tmp_manifest.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        # Remove the previous commit marker before swapping artifacts. A concurrent
        # reader will temporarily see "index not ready" rather than mixed artifacts.
        old_manifest = index_dir / "manifest.json"
        if old_manifest.exists():
            old_manifest.unlink()
        # Store one vector representation only: FAISS is compact and fast; NumPy
        # is retained solely as a fallback when FAISS is unavailable.
        os.replace(tmp_ids, index_dir / "recipe_ids.json")
        if has_faiss:
            os.replace(tmp_faiss, index_dir / "recipes.faiss")
            stale_vectors = index_dir / "vectors.npy"
            if stale_vectors.exists():
                stale_vectors.unlink()
        else:
            os.replace(tmp_vectors, index_dir / "vectors.npy")
            stale_faiss = index_dir / "recipes.faiss"
            if stale_faiss.exists():
                stale_faiss.unlink()
        os.replace(tmp_manifest, index_dir / "manifest.json")

        return VectorIndexInfo(
            ready=True, count=len(recipes), model_name=model_name, dimension=dimension
        )
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
