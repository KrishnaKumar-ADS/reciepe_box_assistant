"""Dataset-first substitution evidence discovery and guarded Groq adaptation."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from .dataset import Recipe
from .normalization import ingredient_key, normalize_phrase
from .search import violates_restrictions
from .analytics import count_instruction_steps


@dataclass(slots=True)
class CandidateEvidence:
    ingredient: str
    support_count: int
    recipe_ids: list[str]
    titles: list[str]
    shared_ingredients: list[str]
    indirect: bool = True


# Broad food-function labels guide the model but are not proof of substitution.
ROLE_HINTS = {
    "egg": "binding and structure; in baking also moisture and emulsification",
    "butter": "fat, moisture, richness, and flavor",
    "oil": "fat and moisture",
    "cream": "fat, richness, and creamy texture",
    "milk": "liquid moisture and dairy flavor",
    "yogurt": "moisture, acidity, and tenderizing",
    "cheese": "salt, fat, umami, and texture",
    "flour": "structure and thickening",
    "sugar": "sweetness, browning, and moisture retention",
    "banana": "moisture, sweetness, and binding in baked goods",
}

# Compatibility is purposefully conservative. A candidate still needs support
# in retrieved peer recipes and is never labeled as a proven equivalent.
ROLE_CANDIDATES = {
    "egg": {"banana", "applesauce", "yogurt", "pumpkin puree", "mashed potato", "tofu"},
    "butter": {"oil", "olive oil", "coconut oil", "margarine", "shortening"},
    "cream": {"coconut milk", "coconut cream", "milk", "yogurt", "cashew"},
    "milk": {"soy milk", "almond milk", "oat milk", "coconut milk", "water", "broth"},
    "cheese": {"nutritional yeast", "tofu", "cashew", "miso"},
    "sour cream": {"yogurt", "coconut yogurt", "cream cheese"},
    "yogurt": {"sour cream", "buttermilk", "applesauce"},
    "flour": {"oat flour", "almond flour", "rice flour", "cornstarch"},
    "sugar": {"honey", "maple syrup", "brown sugar", "date"},
}


def _role_for(target: str) -> str:
    key = normalize_phrase(target)
    for base, role in ROLE_HINTS.items():
        if base in key:
            return role
    return "ingredient-specific flavor, moisture, and texture contribution (not established by this dataset)"


def find_substitution_evidence(recipes: list[Recipe], base_recipe: Recipe,
                               target_ingredient: str,
                               restrictions: Iterable[str] = (),
                               limit: int = 8) -> tuple[list[CandidateEvidence], list[tuple[Recipe, int]]]:
    target = ingredient_key(target_ingredient)
    base_keys = set(base_recipe.ingredient_keys)
    comparable = base_keys - {target}
    peers: list[tuple[Recipe, int, float]] = []
    for recipe in recipes:
        if recipe.recipe_id == base_recipe.recipe_id or target in recipe.ingredient_keys:
            continue
        if violates_restrictions(recipe, restrictions):
            continue
        peer_keys = set(recipe.ingredient_keys)
        overlap = comparable & peer_keys
        if len(overlap) < 2:
            continue
        union = comparable | peer_keys
        jaccard = len(overlap) / len(union) if union else 0.0
        # Similar titles add a small tie-break signal for related recipe types.
        title_tokens = set(base_recipe.title.lower().split()) & set(recipe.title.lower().split())
        peers.append((recipe, len(overlap), jaccard + 0.02 * len(title_tokens)))
    peers.sort(key=lambda x: (-x[2], -x[1], x[0].title.casefold()))
    peer_rows = peers[: max(30, limit * 5)]

    counts: Counter[str] = Counter()
    evidence: dict[str, list[Recipe]] = defaultdict(list)
    shared: dict[str, set[str]] = defaultdict(set)
    for peer, overlap_count, _ in peer_rows:
        for candidate in peer.ingredient_keys:
            if candidate in base_keys or candidate == target:
                continue
            # Candidate must be a plausible name-level role peer, not a random
            # ingredient. This map is an eligibility gate, not a factual claim.
            allowed = ROLE_CANDIDATES.get(target, set())
            if not any(candidate == normalize_phrase(x) or normalize_phrase(x) in candidate for x in allowed):
                continue
            counts[candidate] += 1
            evidence[candidate].append(peer)
            shared[candidate].update(set(base_recipe.ingredient_keys) & set(peer.ingredient_keys))

    candidates = []
    for name, count in counts.most_common():
        if count < 1:
            continue
        sources = evidence[name][:5]
        candidates.append(CandidateEvidence(
            ingredient=name, support_count=count,
            recipe_ids=[x.recipe_id for x in sources],
            titles=[x.title for x in sources],
            shared_ingredients=sorted(shared[name])[:8],
            indirect=True,
        ))
        if len(candidates) >= limit:
            break
    return candidates, [(recipe, overlap) for recipe, overlap, _ in peer_rows[:10]]


def build_updated_ingredient_list(recipe: Recipe, target_ingredient: str,
                                  replacement: str) -> list[str]:
    target = ingredient_key(target_ingredient)
    replacement_key = ingredient_key(replacement)
    updated = []
    replaced = False
    for line in recipe.ingredients:
        if ingredient_key(line) == target and not replaced:
            updated.append(f"{replacement} (suggested replacement for {line}; quantity needs testing)")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.append(f"{replacement} (suggested replacement; original ingredient not matched exactly)")
    return updated
