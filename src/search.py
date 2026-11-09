"""Deterministic lexical recipe retrieval and follow-up restriction parsing."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from .dataset import Recipe
from .normalization import (
    RESTRICTION_GROUPS, ingredient_matches, ingredient_key, normalize_phrase,
    restriction_terms, words, _STOP_WORDS, _singularize,
)


@dataclass(slots=True)
class SearchHit:
    recipe: Recipe
    score: float
    matched_terms: list[str]
    semantic_score: float | None = None
    lexical_score: float = 0.0


def expand_restrictions(restrictions: Iterable[str]) -> set[str]:
    return restriction_terms(restrictions)


def _matches_dairy_category(key: str) -> bool:
    # Plant-based milks/creams/butters/cheeses are not dairy by category alone.
    plant_prefixes = ("almond ", "cashew ", "coconut ", "oat ", "soy ", "soya ",
                      "rice ", "hemp ", "pea ", "sunflower ", "vegan ", "plant based ")
    if key.startswith(plant_prefixes):
        return False
    non_dairy_butters = ("peanut butter", "almond butter", "cashew butter", "sunflower butter")
    if any(key == item or key.startswith(item + " ") for item in non_dairy_butters):
        return False
    dairy_names = {
        "milk", "whole milk", "skim milk", "condensed milk", "evaporated milk",
        "cream", "heavy cream", "whipping cream", "double cream", "half and half",
        "butter", "unsalted butter", "salted butter", "buttermilk", "ghee",
        "cheese", "cream cheese", "sour cream", "yogurt", "yoghurt", "whey", "casein",
        "ricotta", "mozzarella", "parmesan", "cheddar", "feta", "goat cheese",
    }
    if key in dairy_names:
        return True
    return any(key.endswith(" " + item) for item in ("milk", "cream", "cheese", "yogurt", "yoghurt"))


def ingredient_violations(ingredient_text: str, restrictions: Iterable[str]) -> list[str]:
    restriction_list = list(restrictions)
    terms = expand_restrictions(restriction_list)
    if not terms:
        return []
    key = ingredient_key(ingredient_text)
    if not key:
        return []
    has_dairy_category = any(normalize_phrase(item) == "dairy" for item in restriction_list)
    violations = []
    if has_dairy_category and _matches_dairy_category(key):
        violations.append("dairy")
    plant_prefixes = ("almond ", "cashew ", "coconut ", "oat ", "soy ", "soya ",
                      "rice ", "hemp ", "pea ", "sunflower ", "vegan ", "plant based ")
    for term in terms:
        if has_dairy_category and key.startswith(plant_prefixes):
            continue
        if ingredient_matches(key, term):
            # Don't let a broad dairy token treat nut butter as dairy.
            if has_dairy_category and term == "butter" and key.startswith(("peanut butter", "almond butter", "cashew butter", "sunflower butter")):
                continue
            violations.append(term)
    return sorted(set(violations))


def violates_restrictions(recipe: Recipe, restrictions: Iterable[str]) -> list[str]:
    restriction_list = list(restrictions)
    violations = set()
    for line in recipe.ingredients:
        violations.update(ingredient_violations(line, restriction_list))
    return sorted(violations)


def _search_terms(text: str) -> list[str]:
    result = []
    for token in words(text):
        if token in _STOP_WORDS or len(token) < 2:
            continue
        token = _singularize(token)
        if token not in result:
            result.append(token)
    return result


def _lexical_score(recipe: Recipe, query: str, terms: list[str]) -> tuple[float, list[str]]:
    title_tokens = set(_search_terms(recipe.title))
    ingredient_tokens = set(_search_terms(" ".join(recipe.ingredient_keys)))
    instruction_tokens = set(_search_terms(recipe.instructions_text))
    matched = [t for t in terms if t in title_tokens or t in ingredient_tokens or t in instruction_tokens]
    score = 0.0
    for term in terms:
        if term in title_tokens:
            score += 5.0
        if term in ingredient_tokens:
            score += 2.5
        if term in instruction_tokens:
            score += 0.35
    if query.strip().lower() in recipe.title.lower():
        score += 8.0
    return score, matched


def search_recipes(recipes: Iterable[Recipe], query: str,
                   restrictions: Iterable[str] = (), limit: int = 10,
                   vector_index=None, recipe_lookup: dict[str, Recipe] | None = None) -> list[SearchHit]:
    """Search recipes using hybrid semantic/lexical retrieval when an index is supplied.

    FAISS retrieves a compact candidate set; deterministic lexical signals rerank it.
    Hard restrictions are always applied to the original ingredient lines after retrieval.
    If no vector index is ready, the original deterministic lexical scan is retained.
    """
    terms = _search_terms(query)
    if not query.strip():
        return []
    recipe_list = recipes if isinstance(recipes, list) else list(recipes)
    restriction_list = list(restrictions)

    if vector_index is not None:
        # Start with a practical semantic candidate pool. If hard restrictions leave
        # fewer than the requested number of eligible recipes, widen to the full
        # collection before claiming that no/too few matches exist. FAISS IndexFlatIP
        # is exact, so the full-pool pass provides an exhaustive eligibility check.
        candidate_count = min(len(recipe_list), max(1000, limit * 100))
        by_id = recipe_lookup or {recipe.recipe_id: recipe for recipe in recipe_list}

        def rerank(semantic_results: list[tuple[str, float]]) -> list[SearchHit]:
            candidates: list[SearchHit] = []
            max_lexical = 1.0
            for recipe_id, semantic_score in semantic_results:
                recipe = by_id.get(recipe_id)
                if recipe is None or violates_restrictions(recipe, restriction_list):
                    continue
                lexical_score, matched = _lexical_score(recipe, query, terms)
                max_lexical = max(max_lexical, lexical_score)
                candidates.append(SearchHit(
                    recipe=recipe, score=0.0, matched_terms=matched,
                    semantic_score=float(semantic_score), lexical_score=lexical_score,
                ))
            for hit in candidates:
                # Cosine is returned by FAISS IndexFlatIP because stored vectors are
                # normalized. Map it to [0, 1], then blend with a bounded lexical signal.
                semantic_component = (max(-1.0, min(1.0, hit.semantic_score or 0.0)) + 1.0) / 2.0
                lexical_component = min(hit.lexical_score / max_lexical, 1.0)
                exact_title_bonus = 0.08 if query.strip().casefold() in hit.recipe.title.casefold() else 0.0
                hit.score = 0.78 * semantic_component + 0.22 * lexical_component + exact_title_bonus
            candidates.sort(key=lambda h: (-h.score, h.recipe.title.casefold(), h.recipe.recipe_id))
            return candidates

        try:
            semantic_results = vector_index.search(query, top_k=candidate_count)
        except Exception:
            semantic_results = []
        if semantic_results:
            candidates = rerank(semantic_results)
            if len(candidates) < limit and candidate_count < len(recipe_list):
                # Query the remaining vector space before reporting a no-match or
                # returning an artificially short list after hard-filtering.
                try:
                    semantic_results = vector_index.search(query, top_k=len(recipe_list))
                    candidates = rerank(semantic_results)
                except Exception:
                    pass
            return candidates[:limit]

    if not terms:
        # Semantic retrieval can handle conversational phrasing with no useful
        # keyword tokens; lexical-only fallback cannot.
        return []
    hits: list[SearchHit] = []
    for recipe in recipe_list:
        if violates_restrictions(recipe, restriction_list):
            continue
        score, matched = _lexical_score(recipe, query, terms)
        if not matched:
            continue
        hits.append(SearchHit(recipe=recipe, score=score, matched_terms=matched,
                              semantic_score=None, lexical_score=score))
    hits.sort(key=lambda h: (-h.score, h.recipe.title.casefold(), h.recipe.recipe_id))
    return hits[:limit]


def _split_restrictions(value: str) -> list[str]:
    """Split a compact restriction phrase into canonical ingredient/category names."""
    value = re.sub(r"\b(?:please|also|too|as well|either|any|the|some)\b", " ", value, flags=re.I)
    parts = re.split(r",|\band\b|\bor\b", value, flags=re.I)
    cleaned: list[str] = []
    for part in parts:
        part = re.sub(
            r"\b(?:i|am|m|cannot|can't|can not|do not|don't|have|eat|consume|include|"
            r"avoid|exclude|excluding|without|no|allergic|to|with|actually|now|again|"
            r"idea|what|cook|cooking|recipe|recipes|dinner|tonight)\b",
            " ",
            part,
            flags=re.I,
        )
        part = re.sub(r"[^a-zA-Z -]", " ", part).strip().lower()
        part = re.sub(r"\s+", " ", part)
        if part:
            cleaned.append(part)
    return cleaned


def parse_restriction_update(message: str, active: Iterable[str]) -> tuple[list[str], list[str], bool]:
    """Return (new active restrictions, added, changed).

    Supports explicit restriction statements while avoiding false positives from
    numerical constraints such as "no more than six steps". The ingredient phrase
    ends before a following step/count/preference clause, so that clause is never
    accidentally stored as an allergen restriction.
    """
    current = list(dict.fromkeys(str(item) for item in active if str(item).strip()))
    lower = (message or "").lower()
    changed = False

    allow_patterns = [
        r"(?:i am |i'm |im )?(?:okay|fine|good) with (.+?)(?: now| again|$|[.!?,])",
        r"(.+?) (?:is|are) (?:okay|fine|allowed)(?: now| again|$|[.!?,])",
        r"remove (?:the )?(.+?) restriction(?:$|[.!?,])",
        r"(?:you can include|allow|remove the restriction on|no longer avoid) (.+?)(?:$|[.!?,])",
    ]
    for pattern in allow_patterns:
        match = re.search(pattern, lower)
        if match:
            allowed = _split_restrictions(match.group(1))
            for item in allowed:
                normalized = normalize_phrase(item)
                before = len(current)
                current = [r for r in current if normalize_phrase(r) != normalized]
                changed = changed or len(current) != before
            return current, [], changed

    # Do not interpret "no more than 6 steps" as a food restriction.
    restriction_pattern = re.compile(
        r"\b(?:allergic\s+to|cannot\s+have|can't\s+have|can\s+not\s+have|"
        r"cannot\s+eat|can't\s+eat|can\s+not\s+eat|don't\s+include|do\s+not\s+include|"
        r"don't\s+use|do\s+not\s+use|exclude|excluding|avoid|without|remove|no)\s+"
        r"(?!more\s+than\b|less\s+than\b|fewer\s+than\b)(.+?)"
        r"(?=$|[.!?;]|\s+(?:and\s+)?(?:at\s+most|maximum|max(?:imum)?(?:\s+of)?|"
        r"no\s+more\s+than|up\s+to|prefer|prioritize|prioritise|rank|sort|suggest|"
        r"recommend|show|find|give\s+me)\b)",
        re.IGNORECASE,
    )

    added: list[str] = []
    for match in restriction_pattern.finditer(lower):
        for item in _split_restrictions(match.group(1)):
            key = normalize_phrase(item)
            if not key:
                continue
            group = next((g for g in RESTRICTION_GROUPS if normalize_phrase(g) == key), None)
            value = group or key
            if value not in current:
                current.append(value)
                added.append(value)
                changed = True

    return current, added, changed


def restriction_only_message(message: str) -> bool:
    """True when a turn is primarily adding/removing ingredient restrictions."""
    lower = (message or "").lower()
    _, added, changed = parse_restriction_update(message, [])
    if changed or added:
        return True
    allow_markers = (
        "okay with", "fine with", "good with", "is okay", "are okay",
        "is fine", "remove the restriction", "restriction on",
    )
    return any(marker in lower for marker in allow_markers)

