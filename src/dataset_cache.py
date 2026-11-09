"""Persistent SQLite cache for parsed recipe records.

The raw JSON files remain the source of truth. This cache only avoids reparsing them
on every Streamlit process restart and is invalidated when source file metadata changes.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from .dataset import Recipe, discover_dataset_files, load_dataset
from .normalization import unique_ingredient_keys
from .vector_index import dataset_fingerprint


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def _write_cache(path: Path, fingerprint: str, recipes: list[Recipe], filenames: list[str]) -> None:
    temp_path = path.with_suffix(".building.sqlite3")
    if temp_path.exists():
        temp_path.unlink()
    connection = _connect(temp_path)
    try:
        connection.executescript("""
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE recipes (
                ordinal INTEGER PRIMARY KEY,
                recipe_id TEXT NOT NULL,
                title TEXT NOT NULL,
                ingredients_json TEXT NOT NULL,
                instructions_json TEXT NOT NULL,
                instructions_original_json TEXT,
                source_file TEXT NOT NULL,
                source_url TEXT NOT NULL,
                image_url TEXT NOT NULL
            );
            CREATE INDEX idx_recipes_id ON recipes(recipe_id);
        """)
        connection.executemany(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            [("dataset_fingerprint", fingerprint), ("filenames", json.dumps(filenames))],
        )
        def iter_rows():
            for ordinal, recipe in enumerate(recipes):
                yield (
                    ordinal, recipe.recipe_id, recipe.title,
                    json.dumps(recipe.ingredients, ensure_ascii=False),
                    json.dumps(recipe.instructions, ensure_ascii=False),
                    (None if isinstance(recipe.instructions_original, list)
                     and recipe.instructions_original == recipe.instructions
                     else json.dumps(recipe.instructions_original, ensure_ascii=False)),
                    recipe.source_file, recipe.source_url, recipe.image_url,
                )
        connection.executemany(
            "INSERT INTO recipes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", iter_rows()
        )
        connection.commit()
    finally:
        connection.close()
    # On Windows, close SQLite connections before replacing the previous cache.
    if path.exists():
        path.unlink()
    temp_path.replace(path)


def _read_cache(path: Path, fingerprint: str) -> tuple[list[Recipe], list[str]] | None:
    if not path.is_file():
        return None
    try:
        connection = sqlite3.connect(path)
        try:
            meta = dict(connection.execute("SELECT key, value FROM metadata"))
            if meta.get("dataset_fingerprint") != fingerprint:
                return None
            filenames = json.loads(meta.get("filenames", "[]"))
            cursor = connection.execute(
                "SELECT recipe_id, title, ingredients_json, instructions_json, "
                "instructions_original_json, source_file, source_url, image_url "
                "FROM recipes ORDER BY ordinal"
            )
            recipes = []
            for row in cursor:
                ingredients = json.loads(row[2])
                instructions = json.loads(row[3])
                original = json.loads(row[4]) if row[4] is not None else instructions
                recipes.append(Recipe(
                    recipe_id=row[0], title=row[1], ingredients=ingredients,
                    instructions=instructions, instructions_original=original,
                    source_file=row[5], source_url=row[6], image_url=row[7],
                    raw=None, ingredient_keys=unique_ingredient_keys(ingredients),
                ))
        finally:
            connection.close()
        return recipes, filenames
    except (sqlite3.Error, ValueError, TypeError, OSError, json.JSONDecodeError):
        return None


def load_dataset_cached(data_dir: Path, cache_path: Path | None = None) -> tuple[list[Recipe], list[str]]:
    files = discover_dataset_files(data_dir)
    filenames = [path.name for path in files]
    if not files:
        return [], []
    fingerprint = dataset_fingerprint(data_dir, filenames)
    cache_path = cache_path or (data_dir / ".cache" / "recipes.sqlite3")
    cached = _read_cache(cache_path, fingerprint)
    if cached is not None:
        return cached
    recipes, loaded_names = load_dataset(data_dir)
    _write_cache(cache_path, fingerprint, recipes, loaded_names)
    return recipes, loaded_names
