"""Conservative ingredient normalization helpers.

Normalization is intentionally name-level. It does not infer nutritional,
allergen-safety, or quantity equivalence.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

_WORD_RE = re.compile(r"[a-z]+(?:'[a-z]+)?")
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "to", "for", "with", "without",
    "about", "into", "from", "on", "in", "at", "by", "as", "is", "are",
    "was", "were", "be", "being", "been", "i", "me", "my", "we", "you",
    "your", "it", "this", "that", "these", "those", "recipe", "recipes",
    "please", "find", "show", "suggest", "give", "make", "need", "want",
    "have", "has", "had", "contain", "contains", "containing", "both",
    "using", "use", "used", "some", "any", "what", "which", "are", "there",
    "how", "many", "much", "percentage", "percent", "top", "few", "fewest",
    "most", "least", "rank", "ranked", "among", "all", "dataset", "data",
}

# Only well-justified name aliases are collapsed. Distinct products/forms remain
# distinct (e.g. milk is not collapsed with milk powder or coconut milk).
ALIASES = {
    "green onion": "scallion",
    "green onions": "scallion",
    "spring onion": "scallion",
    "spring onions": "scallion",
    "scallions": "scallion",
    "bell peppers": "bell pepper",
    "sweet peppers": "bell pepper",
    "chickpeas": "chickpea",
    "garbanzo beans": "chickpea",
    "garbanzo bean": "chickpea",
    "courgette": "zucchini",
    "aubergine": "eggplant",
    "caster sugar": "superfine sugar",
    "heavy cream": "cream",
    "whipping cream": "cream",
    "double cream": "cream",
    "whole milk": "milk",
    "skim milk": "milk",
    "nonfat milk": "milk",
    "confectioners sugar": "powdered sugar",
    "icing sugar": "powdered sugar",
    "plain flour": "all purpose flour",
    "all-purpose flour": "all purpose flour",
}

_PREP_WORDS = {
    "chopped", "minced", "diced", "sliced", "divided", "melted", "softened",
    "room", "temperature", "freshly", "fresh", "finely", "roughly", "coarsely",
    "grated", "shredded", "crushed", "peeled", "seeded", "deveined", "trimmed",
    "rinsed", "drained", "thawed", "cooked", "uncooked", "boneless", "skinless",
    "large", "medium", "small", "optional", "packed", "heaping", "level",
    "to", "taste", "plus", "extra", "for", "garnish", "as", "needed",
    "whole", "skim", "nonfat", "lowfat", "reduced", "fat",
}

_UNITS = {
    "teaspoon", "teaspoons", "tsp", "tablespoon", "tablespoons", "tbsp",
    "cup", "cups", "ounce", "ounces", "oz", "pound", "pounds", "lb", "lbs",
    "gram", "grams", "g", "kg", "kilogram", "milliliter", "milliliters", "ml",
    "liter", "liters", "l", "pinch", "dash", "clove", "cloves", "can", "cans",
    "package", "packages", "pkg", "slice", "slices", "piece", "pieces", "bunch",
    "bunches", "sprig", "sprigs", "head", "heads", "stick", "sticks", "jar", "jars",
    "container", "containers", "quart", "quarts", "pint", "pints", "gallon", "gallons",
}

RESTRICTION_GROUPS = {
    "dairy": {
        "milk", "cream", "heavy cream", "whipping cream", "butter", "cheese",
        "yogurt", "yoghurt", "whey", "casein", "buttermilk", "ghee", "ricotta",
        "mozzarella", "parmesan", "cheddar", "sour cream", "cream cheese",
    },
    "gluten": {
        "wheat", "barley", "rye", "semolina", "farina", "spelt", "couscous",
        "bulgur", "malt", "breadcrumbs", "bread flour", "all purpose flour",
    },
    "peanut": {"peanut", "peanuts", "peanut butter", "peanut oil"},
    "egg": {"egg", "eggs", "egg white", "egg whites", "egg yolk", "egg yolks"},
    "tree nuts": {
        "almond", "almonds", "walnut", "walnuts", "pecan", "pecans", "cashew",
        "cashews", "pistachio", "pistachios", "hazelnut", "hazelnuts", "macadamia",
        "brazil nut", "pine nut", "pine nuts",
    },
    "nuts": {
        "almond", "almonds", "walnut", "walnuts", "pecan", "pecans", "cashew",
        "cashews", "pistachio", "pistachios", "hazelnut", "hazelnuts", "macadamia",
        "brazil nut", "pine nut", "pine nuts",
    },
    "soy": {"soy", "soya", "soy sauce", "tofu", "tempeh", "edamame"},
    "seafood": {
        "fish", "salmon", "tuna", "cod", "shrimp", "prawn", "prawns", "crab",
        "lobster", "clam", "clams", "oyster", "oysters", "scallop", "scallops",
        "mussels", "anchovy", "anchovies",
    },
    "pork": {"pork", "ham", "bacon", "prosciutto", "pancetta"},
}


def words(text: str) -> list[str]:
    return _WORD_RE.findall(str(text).lower().replace("’", "'"))


def _singularize(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("oes"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def normalize_phrase(text: str) -> str:
    """Normalize a free-form ingredient or restriction name to a canonical key."""
    value = str(text or "").lower().replace("’", "'")
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"^\s*(?:\d+\s+\d+/\d+|\d+/\d+|\d+\.\d+|\d+)\s*", "", value)
    value = re.sub(r"^\s*(?:one|two|three|four|five|half|quarter)\s+", "", value)
    tokens = words(value)
    filtered: list[str] = []
    for token in tokens:
        if token in _UNITS or token in _PREP_WORDS:
            continue
        # Ignore common packaging/ingredient-preparation descriptors.
        if token in {"organic", "kosher", "seasoned", "unsalted", "salted", "lowfat", "nonfat"}:
            continue
        filtered.append(_singularize(token))
    phrase = " ".join(filtered).strip()
    return ALIASES.get(phrase, phrase)


def ingredient_key(line: object) -> str:
    """Convert one raw ingredient line into a conservative searchable name."""
    if isinstance(line, dict):
        for key in ("ingredient", "name", "text", "original"):
            if line.get(key):
                line = line[key]
                break
        else:
            line = " ".join(str(v) for v in line.values() if v is not None)
    value = str(line or "").strip()
    if not value:
        return ""
    # Parenthetical alternatives are retained in the raw source but excluded from
    # the canonical primary ingredient to avoid assuming an alternative was used.
    value = re.split(r"\s+(?:or|and/or)\s+", value, maxsplit=1, flags=re.I)[0]
    value = re.sub(r"[,;].*$", "", value)
    value = re.sub(r"\s+\([^)]*\)", "", value)
    return normalize_phrase(value)


def restriction_terms(restrictions: Iterable[str]) -> set[str]:
    expanded: set[str] = set()
    for item in restrictions:
        key = normalize_phrase(item)
        group = next((g for g in RESTRICTION_GROUPS if key == normalize_phrase(g)), None)
        if group:
            expanded.update(normalize_phrase(x) for x in RESTRICTION_GROUPS[group])
        else:
            expanded.add(key)
    return {x for x in expanded if x}


def ingredient_matches(key: str, restriction: str) -> bool:
    """Name-level match; handles a restriction appearing inside a compound name."""
    key = normalize_phrase(key)
    restriction = normalize_phrase(restriction)
    if not key or not restriction:
        return False
    if key == restriction:
        return True
    return bool(re.search(rf"(?<![a-z]){re.escape(restriction)}(?![a-z])", key))


def extract_vocab_matches(text: str, vocabulary: set[str]) -> list[str]:
    """Find exact vocabulary phrases in a user question using n-grams."""
    toks = [_singularize(t) for t in words(text)]
    found: list[str] = []
    for size in range(min(5, len(toks)), 0, -1):
        for start in range(0, len(toks) - size + 1):
            phrase = " ".join(toks[start : start + size])
            phrase = ALIASES.get(phrase, phrase)
            if phrase in vocabulary and phrase not in found:
                found.append(phrase)
    # Prefer longer terms and avoid returning their single-word components.
    chosen: list[str] = []
    for candidate in found:
        if not any(candidate in other.split() and candidate != other for other in chosen):
            chosen.append(candidate)
    return chosen


def unique_ingredient_keys(lines: Iterable[object]) -> list[str]:
    result = []
    seen = set()
    for line in lines:
        key = ingredient_key(line)
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def ingredient_frequency(records: Iterable[object]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(set(getattr(record, "ingredient_keys", [])))
    return counter
