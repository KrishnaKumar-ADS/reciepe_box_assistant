from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dataset_cache import load_dataset_cached  # noqa: E402
from src.vector_index import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL, build_vector_index, dataset_fingerprint,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create persistent local Sentence Transformer embeddings and a FAISS index for Recipe Box."
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size (default: 32).")
    parser.add_argument("--chunk-size", type=int, default=512, help="Texts encoded per write chunk (default: 512).")
    parser.add_argument("--model", default=DEFAULT_EMBEDDING_MODEL, help="Sentence Transformers model ID.")
    parser.add_argument("--device", default=None, help="Optional device, e.g. cpu or cuda. Auto-detected by default.")
    parser.add_argument("--force", action="store_true", help="Rebuild even if a matching index already exists.")
    args = parser.parse_args()

    data_dir = ROOT / "data"
    print("Loading Recipe Box records (the parsed SQLite cache is reused when current)...")
    recipes, filenames = load_dataset_cached(data_dir)
    if not recipes:
        print("No recipes found. Put all three Recipe Box JSON exports in the data/ directory first.")
        return 2
    print(f"Loaded {len(recipes):,} recipes from {len(filenames)} JSON exports.")
    fingerprint = dataset_fingerprint(data_dir, filenames)

    def progress(done: int, total: int) -> None:
        percent = 100.0 * done / max(total, 1)
        print(f"\rEmbedding recipes: {done:,}/{total:,} ({percent:5.1f}%)", end="", flush=True)

    print(f"Loading embedding model: {args.model}")
    result = build_vector_index(
        recipes=recipes,
        index_dir=data_dir / "vector_index",
        source_fingerprint=fingerprint,
        model_name=args.model,
        batch_size=max(1, args.batch_size),
        chunk_size=max(1, args.chunk_size),
        device=args.device,
        force=args.force,
        progress_callback=progress,
    )
    print("\nSemantic index ready:")
    print(f"  Recipes: {result.count:,}")
    print(f"  Model: {result.model_name}")
    print(f"  Dimensions: {result.dimension}")
    print(f"  Directory: {data_dir / 'vector_index'}")
    print("The generated index is local and does not use your Groq API key.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
