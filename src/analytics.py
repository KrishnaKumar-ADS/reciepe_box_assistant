"""Exact statistics over the complete filtered in-memory recipe set."""
from __future__ import annotations

import re
import statistics
from collections import Counter
from typing import Iterable

from .dataset import Recipe
from .normalization import ingredient_frequency, extract_vocab_matches
from .search import violates_restrictions


def count_instruction_steps(recipe: Recipe) -> int:
    """Step rule: list items count directly; strings use numbered lines, then
    non-empty lines, then sentence-ending punctuation as a last resort.
    """
    original = recipe.instructions_original
    if isinstance(original, list):
        return sum(1 for item in original if str(item).strip())
    if isinstance(original, dict):
        return sum(1 for item in original.values() if str(item).strip())
    text = str(original or "").strip()
    if not text:
        return 0
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    numbered = [line for line in lines if re.match(r"^(?:\d+\s*[.)]|step\s+\d+|[-*•])\s*", line, re.I)]
    if len(numbered) >= 2:
        return len(numbered)
    if len(lines) >= 2:
        return len(lines)
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    return max(1, len(sentences))


def build_vocabulary(recipes: Iterable[Recipe]) -> set[str]:
    vocab: set[str] = set()
    for recipe in recipes:
        vocab.update(recipe.ingredient_keys)
    return vocab


def filter_recipes_for_ingredients(recipes: list[Recipe], ingredients: list[str] | None = None,
                                   restrictions: Iterable[str] = ()) -> list[Recipe]:
    terms = [x for x in (ingredients or []) if x]
    filtered = [r for r in recipes if not violates_restrictions(r, restrictions)]
    if terms:
        filtered = [r for r in filtered if all(term in r.ingredient_keys for term in terms)]
    return filtered


def count_recipes_with(recipes: list[Recipe], ingredients: list[str],
                       restrictions: Iterable[str] = ()) -> dict:
    filtered = [r for r in recipes if not violates_restrictions(r, restrictions)]
    terms = [x for x in ingredients if x]
    if not terms:
        return {"count": 0, "denominator": len(filtered), "percentage": 0.0, "matched": []}
    matched = filter_recipes_for_ingredients(filtered, terms)
    denominator = len(filtered)
    percentage = (len(matched) / denominator * 100.0) if denominator else 0.0
    return {"count": len(matched), "denominator": denominator,
            "percentage": percentage, "matched": matched}


def ingredient_statistics(recipes: list[Recipe], restrictions: Iterable[str] = ()) -> dict:
    filtered = [r for r in recipes if not violates_restrictions(r, restrictions)]
    counts = [len(set(r.ingredient_keys)) for r in filtered]
    return {
        "recipe_count": len(filtered),
        "mean": statistics.mean(counts) if counts else 0.0,
        "median": statistics.median(counts) if counts else 0.0,
        "min": min(counts) if counts else 0,
        "max": max(counts) if counts else 0,
        "distribution": counts,
    }


def common_ingredients(recipes: list[Recipe], n: int = 10,
                       restrictions: Iterable[str] = ()) -> list[tuple[str, int]]:
    filtered = [r for r in recipes if not violates_restrictions(r, restrictions)]
    counts = ingredient_frequency(filtered)
    return counts.most_common(n)


def rank_by_steps(recipes: list[Recipe], ingredients: list[str] | None = None,
                  ascending: bool = True, n: int = 5,
                  restrictions: Iterable[str] = ()) -> list[tuple[Recipe, int]]:
    terms = ingredients or []
    filtered = filter_recipes_for_ingredients(recipes, terms, restrictions)
    ranked = [(r, count_instruction_steps(r)) for r in filtered]
    ranked.sort(key=lambda pair: ((pair[1] if ascending else -pair[1]),
                                  pair[0].title.casefold(), pair[0].recipe_id))
    return ranked[:n]


def interpret_analytics_question(question: str, recipes: list[Recipe],
                                 restrictions: Iterable[str] = ()) -> dict | None:
    """Parse supported analytics intents before doing any full-corpus work.

    This early intent gate is important for semantic chat: ordinary recipe questions
    must not build/scan the ingredient vocabulary on every turn.
    """
    lower = question.lower()
    is_mean = any(word in lower for word in ("average", "mean", "median")) and "ingredient" in lower
    is_common = any(word in lower for word in (
        "common ingredient", "most common", "popular ingredient", "frequent ingredient"
    ))
    is_rank = any(w in lower for w in ("fewest", "least", "most", "rank", "shortest", "longest")) and "step" in lower
    is_count = any(w in lower for w in ("how many", "count", "percentage", "percent", "what proportion"))

    if not (is_mean or is_common or is_rank or is_count):
        return None

    if is_mean:
        result = ingredient_statistics(recipes, restrictions)
        return {"kind": "ingredient_stats", "result": result, "filters": list(restrictions)}

    if is_common:
        top_match = re.search(r"\b(?:top|first)\s+(\d+)\b", lower)
        n = max(1, min(30, int(top_match.group(1)))) if top_match else 10
        result = common_ingredients(recipes, n=n, restrictions=restrictions)
        return {"kind": "common_ingredients", "result": result,
                "recipe_count": len(recipes) - sum(bool(violates_restrictions(r, restrictions)) for r in recipes),
                "filters": list(restrictions)}

    # Only ingredient-filter queries need to construct/scan the vocabulary.
    vocabulary = build_vocabulary(recipes)
    terms = extract_vocab_matches(question, vocabulary)

    if is_rank:
        top_match = re.search(r"\b(?:top|five|ten)\s+(\d+)\b", lower)
        n = int(top_match.group(1)) if top_match else (5 if "five" in lower else 10 if "ten" in lower else 5)
        ascending = not any(w in lower for w in ("most steps", "longest", "highest number"))
        analyzed_count = len(filter_recipes_for_ingredients(recipes, terms, restrictions))
        result = rank_by_steps(recipes, ingredients=terms, ascending=ascending,
                               n=max(1, min(50, n)), restrictions=restrictions)
        return {"kind": "rank_by_steps", "result": result, "ingredients": terms,
                "ascending": ascending, "filters": list(restrictions),
                "analyzed_count": analyzed_count}

    if is_count and terms:
        result = count_recipes_with(recipes, terms, restrictions)
        return {"kind": "count", "result": result, "ingredients": terms,
                "filters": list(restrictions)}

    return None


def format_analytics_answer(analysis: dict) -> str:
    kind = analysis["kind"]
    if kind == "count":
        result = analysis["result"]
        terms = " and ".join(analysis["ingredients"])
        return (
            f"**{result['count']:,} recipes** contain {terms} together, out of "
            f"**{result['denominator']:,} recipes analyzed** "
            f"({result['percentage']:.2f}%).\n\n"
            "Calculation: matching recipe count ÷ filtered dataset size × 100. "
            "Each recipe is counted once; duplicate appearances of an ingredient "
            "within one recipe do not increase its count. Ingredient names are "
            "normalized conservatively after removing quantities and preparation words."
        )
    if kind == "ingredient_stats":
        r = analysis["result"]
        return (
            f"Across **{r['recipe_count']:,} recipes**, the mean number of distinct "
            f"normalized ingredient names is **{r['mean']:.2f}**, and the median is "
            f"**{r['median']:.2f}** (minimum {r['min']}, maximum {r['max']}). "
            "Each recipe contributes one count after name-level normalization."
        )
    if kind == "common_ingredients":
        lines = [f"Analyzed **{analysis['recipe_count']:,} recipes**. Most common ingredients:"]
        lines.extend(f"{i}. **{name}** — {count:,} recipes" for i, (name, count) in enumerate(analysis["result"], 1))
        return "\n".join(lines) + "\n\nCounts are recipe-level presence counts, not total mentions."
    if kind == "rank_by_steps":
        direction = "fewest" if analysis["ascending"] else "most"
        scope = f" containing {', '.join(analysis['ingredients'])}" if analysis["ingredients"] else " in the filtered dataset"
        lines = [f"Top recipes with the {direction} instruction steps{scope}:",
                 f"Analyzed **{analysis['analyzed_count']:,} eligible recipes** from the loaded collection before selecting the top results."]
        lines.extend(f"{i}. **{recipe.title}** (`{recipe.recipe_id}`) — {steps} steps"
                     for i, (recipe, steps) in enumerate(analysis["result"], 1))
        lines.append("\nStep-count rule: list instructions count as list items; string instructions use numbered lines, then non-empty lines, then sentence boundaries.")
        return "\n".join(lines)
    return "I could not map that request to a supported dataset calculation."
