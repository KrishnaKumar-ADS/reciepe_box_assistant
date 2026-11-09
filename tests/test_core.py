from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analytics import (
    count_instruction_steps, count_recipes_with, ingredient_statistics,
    rank_by_steps,
)
from src.dataset import load_json_file, parse_recipe_record
from src.matching import match_recipes, parse_available_ingredients
from src.normalization import ingredient_key, normalize_phrase
from src.search import parse_restriction_update, search_recipes, violates_restrictions
from src.substitution import find_substitution_evidence


def make_recipe(title, ingredients, instructions, recipe_id):
    return parse_recipe_record(
        {
            "id": recipe_id,
            "title": title,
            "ingredients": ingredients,
            "instructions": instructions,
            "url": f"https://example.test/{recipe_id}",
        },
        "recipes_raw_fn.json",
    )


@pytest.fixture
def recipes():
    return [
        make_recipe("Penne Pasta", ["200 g penne", "2 cloves garlic", "1 onion", "1 cup cream", "salt"],
                    ["Boil the pasta.", "Cook garlic and onion.", "Add cream and combine."], "p1"),
        make_recipe("Tomato Rice", ["rice", "tomatoes", "onions", "garlic", "olive oil"],
                    ["Wash the rice.", "Cook rice with vegetables."], "p2"),
        make_recipe("Banana Bread", ["flour", "banana", "sugar", "egg", "butter"],
                    ["Mash banana.", "Mix ingredients.", "Bake until done."], "p3"),
        make_recipe("Coconut Curry", ["coconut milk", "chickpeas", "tomato", "onion", "garlic"],
                    "Heat oil. Add onion and garlic. Stir in tomato and coconut milk. Simmer.", "p4"),
    ]


def test_ingredient_normalization_keeps_distinct_milks():
    assert ingredient_key("1 cup chopped green onions") == "scallion"
    assert ingredient_key("1 cup coconut milk") == "coconut milk"
    assert normalize_phrase("eggs") == "egg"


def test_official_nosource_exports_are_recognized_and_use_clean_id_prefix():
    from src.dataset import EXPECTED_FILES, has_complete_dataset

    assert EXPECTED_FILES == (
        "recipes_raw_nosource_fn.json",
        "recipes_raw_nosource_epi.json",
        "recipes_raw_nosource_ar.json",
    )
    assert has_complete_dataset(EXPECTED_FILES)
    # Backward compatibility for mirrors/manual exports using the old names.
    assert has_complete_dataset((
        "recipes_raw_fn.json", "recipes_raw_epi.json", "recipes_raw_ar.json"
    ))
    assert not has_complete_dataset(("recipes_raw_nosource_fn.json",))


def test_keyed_json_load_and_stable_id(tmp_path: Path):
    data = {
        "abc123": {
            "title": "Example Soup",
            "ingredients": ["1 cup water", "salt"],
            "instructions": ["Boil water.", "Add salt."],
        }
    }
    file = tmp_path / "recipes_raw_epi.json"
    file.write_text(json.dumps(data), encoding="utf-8")
    first = load_json_file(file)[0]
    second = load_json_file(file)[0]
    assert first.recipe_id == second.recipe_id
    assert first.recipe_id == "epi-abc123"
    assert first.title == "Example Soup"
    assert first.ingredients == ["1 cup water", "salt"]


def test_search_returns_title_match(recipes):
    hits = search_recipes(recipes, "penne pasta")
    assert hits
    assert hits[0].recipe.title == "Penne Pasta"


def test_dairy_restriction_blocks_dairy_but_not_coconut_milk(recipes):
    assert violates_restrictions(recipes[0], ["dairy"])
    assert not violates_restrictions(recipes[3], ["dairy"])
    hits = search_recipes(recipes, "tomato", restrictions=["dairy"])
    assert all(not violates_restrictions(hit.recipe, ["dairy"]) for hit in hits)


def test_restriction_followup_is_persisted_and_can_be_removed():
    active, added, changed = parse_restriction_update("I cannot have dairy", [])
    assert active == ["dairy"]
    assert added == ["dairy"]
    assert changed
    active, added, changed = parse_restriction_update("Dairy is okay now", active)
    assert active == []
    assert changed


def test_statistics_count_both_ingredients(recipes):
    result = count_recipes_with(recipes, ["garlic", "onion"])
    assert result["count"] == 3
    assert result["denominator"] == 4
    assert result["percentage"] == 75.0


def test_mean_median_and_step_count(recipes):
    assert count_instruction_steps(recipes[0]) == 3
    stats = ingredient_statistics(recipes)
    assert stats["recipe_count"] == 4
    assert stats["mean"] == 5.0
    assert stats["median"] == 5.0
    ranked = rank_by_steps(recipes, ["tomato"], ascending=True, n=2)
    assert ranked[0][1] <= ranked[1][1]


def test_matching_respects_exclusions_and_shows_missing(recipes):
    available = parse_available_ingredients("eggs, onions, tomatoes, rice")
    results = match_recipes(recipes, available, excluded=["peanuts"], max_steps=3, limit=5)
    assert results
    tomato_rice = next(item for item in results if item.recipe.title == "Tomato Rice")
    assert "rice" in tomato_rice.matched
    assert "olive oil" in tomato_rice.missing


def test_matching_no_match_does_not_relax_hard_step_limit(recipes):
    results = match_recipes(recipes, ["banana"], excluded=["dairy"], max_steps=1)
    assert results == []


def test_substitution_candidate_must_come_from_dataset(recipes):
    base = next(r for r in recipes if r.title == "Banana Bread")
    candidates, peers = find_substitution_evidence(recipes, base, "egg", limit=10)
    # The tiny synthetic set does not contain a role-compatible banana-bread peer,
    # so the safe behavior is no unsupported candidate.
    assert candidates == []



def test_name_level_counts_do_not_conflate_coconut_milk_and_milk(recipes):
    from src.analytics import count_recipes_with
    result = count_recipes_with(recipes, ["milk"])
    assert result["count"] == 0
    available = parse_available_ingredients("milk")
    matches = match_recipes(recipes, available, limit=10)
    assert all("coconut milk" not in item.matched for item in matches)


def test_no_idea_is_not_parsed_as_an_ingredient_restriction():
    from src.search import restriction_only_message
    active, added, changed = parse_restriction_update("I have no idea what to cook", [])
    assert active == []
    assert added == []
    assert changed is False
    assert restriction_only_message("I have no idea what to cook") is False


def test_plural_natural_language_search_matches_singular_ingredient(recipes):
    hits = search_recipes(recipes, "tomatoes")
    assert any(hit.recipe.title == "Tomato Rice" for hit in hits)



def test_download_and_extract_recognizes_official_archive_filenames(tmp_path: Path, monkeypatch):
    import io
    import zipfile
    from src import dataset_download

    archive_bytes = io.BytesIO()
    payload = json.dumps({"sample-key": {
        "title": "Archive Test Recipe",
        "ingredients": ["1 cup rice"],
        "instructions": "Cook the rice.",
    }}).encode("utf-8")
    with zipfile.ZipFile(archive_bytes, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in (
            "recipes_raw_nosource_fn.json",
            "recipes_raw_nosource_epi.json",
            "recipes_raw_nosource_ar.json",
        ):
            archive.writestr(f"dataset/{name}", payload)
    blob = archive_bytes.getvalue()

    class FakeResponse(io.BytesIO):
        headers = {"Content-Length": str(len(blob))}

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            self.close()

    monkeypatch.setattr(dataset_download.urllib.request, "urlopen", lambda *a, **k: FakeResponse(blob))
    paths = dataset_download.download_and_extract(tmp_path)

    assert {path.name for path in paths} == {
        "recipes_raw_nosource_fn.json",
        "recipes_raw_nosource_epi.json",
        "recipes_raw_nosource_ar.json",
    }
    from src.dataset import load_dataset
    recipes, filenames = load_dataset(tmp_path)
    assert len(recipes) == 3
    assert set(filenames) == {path.name for path in paths}
    assert all(recipe.recipe_id.startswith(("fn-", "epi-", "ar-")) for recipe in recipes)


def test_downloader_reuses_already_extracted_official_files(tmp_path: Path, monkeypatch):
    from src.dataset_download import download_and_extract

    for name in (
        "recipes_raw_nosource_fn.json",
        "recipes_raw_nosource_epi.json",
        "recipes_raw_nosource_ar.json",
    ):
        (tmp_path / name).write_text("{}", encoding="utf-8")

    def should_not_download(*args, **kwargs):
        raise AssertionError("Downloader should reuse files already extracted")

    monkeypatch.setattr("src.dataset_download.urllib.request.urlopen", should_not_download)
    paths = download_and_extract(tmp_path)
    assert len(paths) == 3


def test_hybrid_search_finds_semantic_paraphrase_and_keeps_hard_exclusions(recipes):
    # Candidate order/scores are deliberately returned by a fake semantic index.
    class FakeVectorIndex:
        def search(self, query, top_k=100):
            return [
                (recipes[3].recipe_id, 0.91),
                (recipes[0].recipe_id, 0.89),
            ]

    semantic_hits = search_recipes(
        recipes, "something warming with a coconut base", limit=5,
        restrictions=["dairy"], vector_index=FakeVectorIndex(),
    )
    assert semantic_hits
    assert semantic_hits[0].recipe.title == "Coconut Curry"
    assert all(not violates_restrictions(hit.recipe, ["dairy"]) for hit in semantic_hits)
    assert semantic_hits[0].semantic_score == 0.91


def test_embedding_index_persists_and_reuses_fake_encoder(tmp_path: Path, recipes):
    import numpy as np
    from src.vector_index import RecipeVectorIndex, build_vector_index

    class FakeEncoder:
        def get_sentence_embedding_dimension(self):
            return 3

        def encode(self, texts, **kwargs):
            rows = []
            for text in texts:
                lower = str(text).lower()
                vector = [1.0, 0.0, 0.0] if "coconut" in lower or "biryani" in lower else [0.0, 1.0, 0.0]
                rows.append(vector)
            arr = np.asarray(rows, dtype=np.float32)
            return arr / np.linalg.norm(arr, axis=1, keepdims=True)

    index_dir = tmp_path / "vector_index"
    fingerprint = "test-fingerprint"
    fake = FakeEncoder()
    result = build_vector_index(
        recipes[:2] + [recipes[3]], index_dir, fingerprint,
        encoder=fake, batch_size=2, chunk_size=2,
    )
    assert result.ready and result.count == 3 and result.dimension == 3

    loaded = RecipeVectorIndex.load(index_dir, expected_fingerprint=fingerprint)
    assert loaded is not None
    loaded._model = fake
    semantic = loaded.search("biryani", top_k=2)
    assert semantic[0][0] == recipes[3].recipe_id
    assert (index_dir / "manifest.json").is_file()
    assert (index_dir / "recipe_ids.json").is_file()

    # Different source fingerprint must invalidate the persisted index.
    assert RecipeVectorIndex.load(index_dir, expected_fingerprint="changed") is None

    class MustNotEncode:
        def get_sentence_embedding_dimension(self):
            raise AssertionError("Existing matching index should not re-embed the corpus")

    reused = build_vector_index(
        recipes[:2] + [recipes[3]], index_dir, fingerprint,
        encoder=MustNotEncode(), batch_size=2, chunk_size=2,
    )
    assert reused.count == 3


def test_non_analytics_chat_does_not_scan_the_full_vocabulary(recipes, monkeypatch):
    from src import analytics

    def should_not_build(_recipes):
        raise AssertionError("Normal recipe chat must not scan the entire vocabulary")

    monkeypatch.setattr(analytics, "build_vocabulary", should_not_build)
    assert analytics.interpret_analytics_question("Find a penne pasta recipe", recipes) is None


def test_dataset_sqlite_cache_is_reused_without_reparsing_json(tmp_path: Path, monkeypatch):
    from src import dataset_cache

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    payload = {
        "recipe-1": {
            "title": "Cached Soup",
            "ingredients": ["1 cup rice", "salt"],
            "instructions": ["Boil rice.", "Season."],
        }
    }
    (data_dir / "recipes_raw_nosource_fn.json").write_text(json.dumps(payload), encoding="utf-8")
    cache_path = data_dir / ".cache" / "recipes.sqlite3"

    first, names = dataset_cache.load_dataset_cached(data_dir, cache_path)
    assert first[0].title == "Cached Soup"
    assert cache_path.is_file()

    def should_not_reparse(_data_dir):
        raise AssertionError("Expected the persisted SQLite cache to be reused")

    monkeypatch.setattr(dataset_cache, "load_dataset", should_not_reparse)
    second, second_names = dataset_cache.load_dataset_cached(data_dir, cache_path)
    assert second[0].recipe_id == first[0].recipe_id
    assert second_names == names


def test_hybrid_search_expands_pool_before_reporting_restriction_no_match():
    # More than 1,000 nearest semantic candidates are blocked by the active
    # restriction; eligible recipes exist just beyond the initial candidate pool.
    many = [
        make_recipe(f"Pasta recipe {i}", ["pasta", "cream"], ["Cook pasta."], f"id{i}")
        for i in range(1000)
    ]
    many.extend(
        make_recipe(f"Pasta recipe {i}", ["pasta", "tomato", "coconut milk"], ["Cook pasta."], f"id{i}")
        for i in range(1000, 1005)
    )

    class OrderedVectorIndex:
        def __init__(self):
            self.requested = []

        def search(self, query, top_k=100):
            self.requested.append(top_k)
            return [(recipe.recipe_id, 0.9 - index / 100000.0)
                    for index, recipe in enumerate(many[:top_k])]

    fake_index = OrderedVectorIndex()
    hits = search_recipes(
        many, "pasta", restrictions=["dairy"], limit=2,
        vector_index=fake_index,
    )
    assert fake_index.requested == [1000, 1005]
    assert len(hits) == 2
    assert all("coconut milk" in hit.recipe.ingredients for hit in hits)
