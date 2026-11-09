"""Load the three Recipe Box JSON exports without mutating source records."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .normalization import unique_ingredient_keys

# The current official archive includes the ``nosource`` segment in each
# export filename. Older mirrors/manual exports may omit it, so accept both.
EXPECTED_FILES = (
    "recipes_raw_nosource_fn.json",
    "recipes_raw_nosource_epi.json",
    "recipes_raw_nosource_ar.json",
)
DATASET_FILE_GROUPS = (
    ("recipes_raw_nosource_fn.json", "recipes_raw_fn.json"),
    ("recipes_raw_nosource_epi.json", "recipes_raw_epi.json"),
    ("recipes_raw_nosource_ar.json", "recipes_raw_ar.json"),
)
DATASET_URL = "https://eightportions.com/recipes_raw.zip"


def has_complete_dataset(filenames: Iterable[str]) -> bool:
    """Return True when one export from each of the three source groups exists."""
    available = set(filenames)
    return all(any(name in available for name in group) for group in DATASET_FILE_GROUPS)


def _source_prefix(source_file: str) -> str:
    """Return a compact, stable source code (fn, epi, or ar) for recipe IDs."""
    stem = Path(source_file).stem
    if stem.startswith("recipes_raw_"):
        stem = stem[len("recipes_raw_"):]
    if stem.startswith("nosource_"):
        stem = stem[len("nosource_"):]
    return stem


@dataclass(slots=True)
class Recipe:
    recipe_id: str
    title: str
    ingredients: list[str]
    instructions: list[str]
    instructions_original: Any
    source_file: str
    source_url: str
    image_url: str
    raw: dict[str, Any] | None
    ingredient_keys: list[str]

    @property
    def instructions_text(self) -> str:
        return "\n".join(self.instructions)


def _first_value(record: dict[str, Any], names: Iterable[str], default: Any = None) -> Any:
    for name in names:
        if name in record and record[name] not in (None, "", []):
            return record[name]
    return default


def _to_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        out = []
        for item in value:
            if isinstance(item, dict):
                text = _first_value(item, ("text", "instruction", "step", "name", "ingredient"))
                if text is None:
                    text = " ".join(str(v) for v in item.values() if v is not None)
                if str(text).strip():
                    out.append(str(text).strip())
            elif str(item).strip():
                out.append(str(item).strip())
        return out
    if isinstance(value, dict):
        return _to_text_list(list(value.values()))
    text = str(value).strip()
    if not text:
        return []
    # Keep one original paragraph intact; step counting is handled separately.
    return [text]


def _is_recipe_record(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    keys = {str(k).lower() for k in value}
    has_title = bool(keys & {"title", "name", "recipe_name"})
    has_ingredients = bool(keys & {"ingredients", "ingredient", "ingredient_list"})
    has_method = bool(keys & {"instructions", "directions", "steps", "method", "preparation"})
    return has_title and (has_ingredients or has_method)


def _records_from_payload(payload: Any) -> list[tuple[str | None, dict[str, Any]]]:
    if isinstance(payload, list):
        return [(None, x) for x in payload if isinstance(x, dict)]
    if not isinstance(payload, dict):
        return []
    if _is_recipe_record(payload):
        return [(None, payload)]
    # Common wrappers used by JSON exports.
    for wrapper in ("recipes", "data", "items", "results"):
        nested = payload.get(wrapper)
        if isinstance(nested, (list, dict)):
            return _records_from_payload(nested)
    rows: list[tuple[str | None, dict[str, Any]]] = []
    for key, value in payload.items():
        if isinstance(value, dict) and _is_recipe_record(value):
            rows.append((str(key), value))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    rows.append((str(key), item))
    return rows


def _stable_id(record: dict[str, Any], source_file: str, source_key: str | None,
               title: str, ingredients: list[str], instructions: list[str], source_url: str) -> str:
    original = _first_value(record, ("recipe_id", "id", "_id", "uuid"))
    if original is None and source_key:
        original = source_key
    if original is not None:
        raw_id = re.sub(r"[^A-Za-z0-9_-]+", "-", str(original)).strip("-")[:48] or "record"
        prefix = _source_prefix(source_file)
        return f"{prefix}-{raw_id}"
    stable_material = source_url or json.dumps(
        {"title": title, "ingredients": ingredients, "instructions": instructions,
         "source": Path(source_file).stem},
        ensure_ascii=False, sort_keys=True,
    )
    digest = hashlib.sha256(stable_material.encode("utf-8")).hexdigest()[:16]
    prefix = _source_prefix(source_file)
    return f"{prefix}-{digest}"


def parse_recipe_record(record: dict[str, Any], source_file: str,
                        source_key: str | None = None) -> Recipe | None:
    title_value = _first_value(record, ("title", "name", "recipe_name"), "Untitled recipe")
    title = str(title_value).strip() or "Untitled recipe"
    raw_ingredients = _first_value(record, ("ingredients", "ingredient", "ingredient_list"), [])
    ingredients = _to_text_list(raw_ingredients)
    original_instructions = _first_value(
        record, ("instructions", "directions", "steps", "method", "preparation"), []
    )
    instructions = _to_text_list(original_instructions)
    # For ordinary string-list exports, share the same list rather than duplicating
    # every instruction string in memory. The original structure remains preserved.
    if isinstance(original_instructions, list) and all(isinstance(item, str) for item in original_instructions):
        instructions = original_instructions
    source_url = str(_first_value(record, ("url", "source_url", "recipe_url", "source"), "") or "")
    image_url = str(_first_value(record, ("image", "image_url", "picture", "photo"), "") or "")
    recipe_id = _stable_id(record, source_file, source_key, title, ingredients,
                           instructions, source_url)
    return Recipe(
        recipe_id=recipe_id,
        title=title,
        ingredients=ingredients,
        instructions=instructions,
        instructions_original=original_instructions,
        source_file=source_file,
        source_url=source_url,
        image_url=image_url,
        # Parsed fields below preserve assignment-required source content. Avoid retaining
        # the full raw dictionary for every recipe: it duplicates large JSON payloads.
        raw=None,
        ingredient_keys=unique_ingredient_keys(ingredients),
    )


def load_json_file(path: Path) -> list[Recipe]:
    try:
        text = path.read_text(encoding="utf-8-sig")
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse {path.name} as JSON: {exc}") from exc
    records = _records_from_payload(payload)
    recipes = []
    for source_key, record in records:
        parsed = parse_recipe_record(record, path.name, source_key)
        if parsed and (parsed.ingredients or parsed.instructions):
            recipes.append(parsed)
    return recipes


def discover_dataset_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    selected: list[Path] = []
    recognized_names: set[str] = set()
    for group in DATASET_FILE_GROUPS:
        recognized_names.update(group)
        match = next((data_dir / name for name in group if (data_dir / name).is_file()), None)
        if match is not None:
            selected.append(match)
    # Also accept additional JSON exports placed in the data folder, while
    # avoiding duplicate ingestion when both filename variants are present.
    extras = sorted(
        p for p in data_dir.glob("*.json")
        if p.name not in recognized_names
    )
    return selected + extras


def load_dataset(data_dir: Path) -> tuple[list[Recipe], list[str]]:
    files = discover_dataset_files(data_dir)
    if not files:
        return [], []
    recipes: list[Recipe] = []
    loaded_names = []
    for path in files:
        parsed = load_json_file(path)
        recipes.extend(parsed)
        loaded_names.append(path.name)
    # Stable deduplication by ID; do not merge or rewrite original raw records.
    unique: dict[str, Recipe] = {}
    for recipe in recipes:
        unique.setdefault(recipe.recipe_id, recipe)
    return list(unique.values()), loaded_names
