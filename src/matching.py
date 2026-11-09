"""Deterministic, explainable pantry matching over existing Recipe Box records.

Pantry matching intentionally scans the complete loaded recipe collection. It does
not rely on a nearest-neighbour candidate pool, because a semantic top-k search can
hide eligible records and is not appropriate for hard allergen/step constraints.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .analytics import count_instruction_steps
from .dataset import Recipe
from .normalization import ingredient_key
from .search import violates_restrictions


_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}

_PANTRY_INTRO = re.compile(
    r"\b(?:i\s+have|i['’]?ve\s+got|available\s+ingredients(?:\s+are)?|"
    r"ingredients\s+i\s+have(?:\s+are)?|my\s+ingredients\s+are)\b\s*:?\s*(.+)",
    re.IGNORECASE | re.DOTALL,
)


@dataclass(slots=True)
class MatchResult:
    recipe: Recipe
    matched: list[str]
    missing: list[str]
    score: float  # Pantry ingredient coverage percentage; retained for UI compatibility.
    steps: int

    @property
    def matched_count(self) -> int:
        return len(self.matched)

    @property
    def additional_count(self) -> int:
        return len(self.missing)

    @property
    def coverage_percent(self) -> float:
        return self.score


@dataclass(slots=True)
class MatchStats:
    total_recipes: int
    excluded_by_restrictions: int
    excluded_by_step_limit: int
    eligible_recipes: int
    matching_recipes: int


def parse_available_ingredients(text_or_items: str | Iterable[str]) -> list[str]:
    """Normalize a user pantry list, including natural-language 'and' lists.

    Example: ``eggs, onions, tomatoes, and rice`` becomes
    ``['egg', 'onion', 'tomato', 'rice']``. Quantities/preparation words are
    stripped by ``ingredient_key``; quantities are not treated as verified stock.
    """
    if isinstance(text_or_items, str):
        raw_parts = re.split(r"[,;\n]+|\s+\band\b\s+|\s+\bor\b\s+", text_or_items, flags=re.IGNORECASE)
    else:
        raw_parts = [str(item) for item in text_or_items]

    output: list[str] = []
    seen: set[str] = set()
    for raw in raw_parts:
        part = re.sub(r"^\s*(?:(?:and|or)\s+)+|(?:\s+(?:and|or))\s*$", "", str(raw), flags=re.IGNORECASE)
        part = part.strip(" \t\r\n.,:;-")
        key = ingredient_key(part)
        if key and key not in seen:
            seen.add(key)
            output.append(key)
    return output


def _parse_limit_number(value: str) -> int | None:
    value = value.strip().lower()
    if value.isdigit():
        return int(value)
    return _NUMBER_WORDS.get(value)


def parse_pantry_request(message: str) -> dict | None:
    """Parse a pantry-style request into deterministic constraints.

    Returns ``None`` when the text is not a pantry request. The extraction stops
    at sentence boundaries or explicit task/constraint clauses, so trailing words
    such as 'suggest five recipes' never become fake ingredient names.
    """
    match = _PANTRY_INTRO.search(message or "")
    if not match:
        return None

    ingredient_text = match.group(1).strip()
    # Common phrases such as "I have no idea" are not ingredient lists.
    if re.match(r"(?:no\s+idea|no\s+clue|a\s+question|a\s+recipe)\b", ingredient_text, re.IGNORECASE):
        return None

    # A full stop / semicolon after the pantry list commonly starts the task request.
    ingredient_text = re.split(r"(?<=[.!?;])\s+", ingredient_text, maxsplit=1)[0]

    # Also support a one-sentence form: "I have eggs, rice, no peanuts, max 6 steps".
    ingredient_text = re.split(
        r"(?:,\s*|\s+)(?:(?:with\s+)?(?:no|without|exclude|excluding|avoid)|"
        r"at\s+most|maximum(?:\s+of)?|max(?:imum)?(?:\s+of)?|no\s+more\s+than|up\s+to|"
        r"prefer|prioritize|prioritise|rank|sort|suggest|recommend|show|find|give\s+me)\b",
        ingredient_text,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    available = parse_available_ingredients(ingredient_text)
    if not available:
        return {"available": [], "max_steps": None, "limit": 5, "prefer_fewer_missing": True}

    step_match = re.search(
        r"\b(?:at\s+most|maximum(?:\s+of)?|max(?:imum)?(?:\s+of)?|no\s+more\s+than|up\s+to)\s+"
        r"(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
        r"(?:instruction\s+)?steps?\b",
        message,
        re.IGNORECASE,
    )
    max_steps = _parse_limit_number(step_match.group(1)) if step_match else None

    result_match = re.search(
        r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+"
        r"(?:existing\s+)?(?:recipes|dishes|results|matches)\b",
        message,
        re.IGNORECASE,
    )
    requested_limit = _parse_limit_number(result_match.group(1)) if result_match else 5
    requested_limit = max(1, min(requested_limit or 5, 25))

    return {
        "available": available,
        "max_steps": max_steps,
        "limit": requested_limit,
        "prefer_fewer_missing": True,
    }


def match_recipes_detailed(
    recipes: list[Recipe],
    available: list[str] | Iterable[str],
    excluded: Iterable[str] = (),
    max_steps: int | None = None,
    limit: int = 10,
    prefer_fewer_missing: bool = True,
) -> tuple[list[MatchResult], MatchStats]:
    """Compare pantry ingredients against every loaded recipe and report counts.

    Ranking is lexicographic, matching the assignment's stated priorities:
      1. More available ingredients matched.
      2. Fewer additional recipe ingredients required (when preferred).
      3. Fewer instruction steps.
      4. Stable title and recipe-ID ordering.

    Hard restrictions and the maximum-step limit are filters, never score penalties.
    """
    available_keys = parse_available_ingredients(available)
    restriction_list = list(excluded)
    total = len(recipes)

    if not available_keys:
        return [], MatchStats(total, 0, 0, 0, 0)

    results: list[MatchResult] = []
    excluded_restrictions = 0
    excluded_steps = 0
    eligible_count = 0

    for recipe in recipes:
        # A forbidden ingredient always excludes the recipe, before ranking.
        if violates_restrictions(recipe, restriction_list):
            excluded_restrictions += 1
            continue

        steps = count_instruction_steps(recipe)
        if max_steps is not None and steps > max_steps:
            excluded_steps += 1
            continue

        eligible_count += 1
        recipe_keys = set(recipe.ingredient_keys)
        # Exact canonical ingredient-name matching avoids false equivalence,
        # such as dairy milk being treated as coconut milk.
        matched = sorted(key for key in available_keys if key in recipe_keys)
        if not matched:
            continue

        missing = sorted(key for key in recipe_keys if key not in available_keys)
        coverage = 100.0 * len(matched) / len(available_keys)
        results.append(MatchResult(recipe, matched, missing, coverage, steps))

    if prefer_fewer_missing:
        results.sort(key=lambda item: (
            -item.matched_count,
            item.additional_count,
            item.steps,
            item.recipe.title.casefold(),
            item.recipe.recipe_id,
        ))
    else:
        results.sort(key=lambda item: (
            -item.matched_count,
            item.steps,
            item.additional_count,
            item.recipe.title.casefold(),
            item.recipe.recipe_id,
        ))

    stats = MatchStats(
        total_recipes=total,
        excluded_by_restrictions=excluded_restrictions,
        excluded_by_step_limit=excluded_steps,
        eligible_recipes=eligible_count,
        matching_recipes=len(results),
    )
    return results[: max(0, int(limit))], stats


def match_recipes(
    recipes: list[Recipe],
    available: list[str],
    excluded: Iterable[str] = (),
    max_steps: int | None = None,
    limit: int = 10,
    prefer_fewer_missing: bool = True,
) -> list[MatchResult]:
    """Backward-compatible convenience wrapper returning only ranked matches."""
    results, _ = match_recipes_detailed(
        recipes,
        available,
        excluded=excluded,
        max_steps=max_steps,
        limit=limit,
        prefer_fewer_missing=prefer_fewer_missing,
    )
    return results


def format_pantry_match_answer(
    results: list[MatchResult],
    available: list[str],
    exclusions: Iterable[str],
    max_steps: int | None,
    stats: MatchStats,
    limit: int,
) -> str:
    """Render a transparent, assignment-ready ranked pantry answer."""
    available_count = len(available)
    lines = [
        f"### Pantry matches — {len(results)} of up to {limit} requested recipes",
        f"I compared **all {stats.total_recipes:,} loaded recipe records**. "
        f"After hard filters, **{stats.eligible_recipes:,}** remained eligible and "
        f"**{stats.matching_recipes:,}** eligible recipes used at least one available ingredient.",
    ]

    filters = []
    if exclusions:
        filters.append("exclude: " + ", ".join(sorted(set(exclusions))))
    if max_steps is not None:
        filters.append(f"at most {max_steps} instruction steps")
    if filters:
        lines.append("**Hard constraints:** " + "; ".join(filters) + ".")
    lines.append(
        "**Ranking rule:** first maximize the number of your available ingredients used; "
        "then minimize additional ingredients needed; then prefer fewer instruction steps. "
        "Recipe title and stable ID break any remaining ties."
    )

    if not results:
        lines.append(
            "**No eligible matches were found.** The hard constraints were not relaxed. "
            "Try adding pantry ingredients or explicitly changing a restriction/step limit."
        )
        if exclusions:
            lines.append(f"Recipes blocked by the hard ingredient exclusions: {stats.excluded_by_restrictions:,}.")
        if max_steps is not None:
            lines.append(f"Recipes blocked by the step limit: {stats.excluded_by_step_limit:,}.")
        return "\n\n".join(lines)

    lines.append("**Ranked existing recipes**")
    for rank, item in enumerate(results, 1):
        matched_text = ", ".join(item.matched) if item.matched else "none"
        missing_text = ", ".join(item.missing) if item.missing else "none"
        lines.extend([
            f"\n**{rank}. {item.recipe.title}** — `{item.recipe.recipe_id}`",
            f"- **Matched ingredients ({item.matched_count}/{available_count}):** {matched_text}",
            f"- **Additional required ingredients ({item.additional_count}):** {missing_text}",
            f"- **Instruction steps:** {item.steps}" + (f"/{max_steps}" if max_steps is not None else ""),
            f"- **Why this rank:** uses {item.matched_count} of your {available_count} ingredients, "
            f"requires {item.additional_count} additional ingredient names, and has {item.steps} steps.",
        ])

    lines.append(
        "**Matching limitations:** ingredients are compared by normalized name, not quantity. "
        "The step count follows the dataset's stored instruction structure. Peanut exclusion "
        "checks the ingredient text in each record; it is not a medical or cross-contact safety guarantee."
    )
    if exclusions:
        lines.append("Active hard exclusions: " + ", ".join(sorted(set(exclusions))) + ".")
    return "\n".join(lines)
