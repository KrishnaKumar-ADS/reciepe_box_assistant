from __future__ import annotations

from src.dataset import parse_recipe_record
from src.matching import (
    format_pantry_match_answer,
    match_recipes_detailed,
    parse_pantry_request,
)
from src.search import parse_restriction_update, restriction_only_message


def make_recipe(title, ingredients, instructions, recipe_id):
    return parse_recipe_record(
        {
            "id": recipe_id,
            "title": title,
            "ingredients": ingredients,
            "instructions": instructions,
        },
        "recipes_raw_nosource_fn.json",
    )


def test_assignment_pantry_prompt_extracts_only_four_available_ingredients():
    prompt = (
        "I have eggs, onions, tomatoes, and rice. Suggest five recipes using as many of these as "
        "possible, with no peanuts and at most six instruction steps. Prefer recipes that need "
        "the fewest additional ingredients."
    )
    pantry = parse_pantry_request(prompt)
    assert pantry == {
        "available": ["egg", "onion", "tomato", "rice"],
        "max_steps": 6,
        "limit": 5,
        "prefer_fewer_missing": True,
    }


def test_step_limit_is_not_misread_as_an_ingredient_restriction():
    prompt = (
        "I have eggs, onions, tomatoes, and rice. Suggest five recipes with no peanuts "
        "and at most six instruction steps."
    )
    active, added, changed = parse_restriction_update(prompt, [])
    assert active == ["peanut"]
    assert added == ["peanut"]
    assert changed is True


def test_exclude_eggs_too_is_a_restriction_followup():
    assert restriction_only_message("Exclude eggs too") is True
    active, added, changed = parse_restriction_update("Exclude eggs too", ["peanut"])
    assert active == ["peanut", "egg"]
    assert added == ["egg"]
    assert changed is True


def test_no_idea_is_not_a_pantry_request_or_restriction():
    assert parse_pantry_request("I have no idea what to cook") is None
    assert parse_restriction_update("I have no idea what to cook", []) == ([], [], False)


def test_pantry_ranking_prioritizes_more_available_ingredients_before_extra_count():
    recipes = [
        make_recipe(
            "Four Pantry Ingredients, Many Additions",
            ["egg", "onion", "tomato", "rice", "flour", "sugar", "oil", "salt", "pepper"],
            ["Step one.", "Step two."],
            "four-match",
        ),
        make_recipe(
            "Three Pantry Ingredients, No Additions",
            ["egg", "onion", "rice"],
            ["Step one."],
            "three-match",
        ),
    ]
    results, stats = match_recipes_detailed(
        recipes,
        ["egg", "onion", "tomato", "rice"],
        excluded=[],
        max_steps=6,
        limit=5,
    )
    assert results[0].recipe.recipe_id == "fn-four-match"
    assert results[0].matched_count == 4
    assert results[1].matched_count == 3
    assert stats.total_recipes == 2
    assert stats.eligible_recipes == 2
    assert stats.matching_recipes == 2


def test_hard_peanut_and_six_step_constraints_are_filters_and_followup_can_add_egg():
    recipes = [
        make_recipe(
            "Egg Rice",
            ["eggs", "onions", "tomatoes", "rice"],
            ["Cook rice.", "Cook eggs.", "Combine."],
            "egg-rice",
        ),
        make_recipe(
            "Vegetable Rice",
            ["onions", "tomatoes", "rice", "oil"],
            ["Cook rice.", "Cook vegetables."],
            "veg-rice",
        ),
        make_recipe(
            "Peanut Rice",
            ["eggs", "onions", "tomatoes", "rice", "peanut oil"],
            ["Cook rice.", "Add vegetables."],
            "peanut-rice",
        ),
        make_recipe(
            "Seven Step Rice",
            ["eggs", "onions", "tomatoes", "rice"],
            [f"Step {number}." for number in range(1, 8)],
            "seven-step",
        ),
    ]

    initial, initial_stats = match_recipes_detailed(
        recipes,
        ["eggs", "onions", "tomatoes", "rice"],
        excluded=["peanut"],
        max_steps=6,
        limit=5,
    )
    assert "fn-egg-rice" in [item.recipe.recipe_id for item in initial]
    assert "fn-peanut-rice" not in [item.recipe.recipe_id for item in initial]
    assert "fn-seven-step" not in [item.recipe.recipe_id for item in initial]
    assert initial_stats.excluded_by_restrictions == 1
    assert initial_stats.excluded_by_step_limit == 1

    followup_active, added, _ = parse_restriction_update("Exclude eggs too", ["peanut"])
    assert added == ["egg"]
    after_followup, _ = match_recipes_detailed(
        recipes,
        ["eggs", "onions", "tomatoes", "rice"],
        excluded=followup_active,
        max_steps=6,
        limit=5,
    )
    assert "fn-egg-rice" not in [item.recipe.recipe_id for item in after_followup]
    assert "fn-veg-rice" in [item.recipe.recipe_id for item in after_followup]


def test_ranked_answer_contains_ids_matched_missing_step_counts_and_ranking_rule():
    recipe = make_recipe(
        "Tomato Rice",
        ["tomatoes", "rice", "olive oil"],
        ["Wash rice.", "Cook with tomatoes.", "Season."],
        "tomato-rice",
    )
    results, stats = match_recipes_detailed(
        [recipe], ["tomatoes", "rice", "eggs", "onions"], excluded=["peanut"], max_steps=6
    )
    answer = format_pantry_match_answer(
        results, ["tomato", "rice", "egg", "onion"], ["peanut"], 6, stats, 5
    )
    assert "fn-tomato-rice" in answer
    assert "Matched ingredients (2/4)" in answer
    assert "olive oil" in answer
    assert "**Instruction steps:** 3/6" in answer
    assert "maximize the number" in answer
    assert "all 1 loaded recipe records" in answer
