# Reproducible evaluation plan

The scenarios below map to the assignment's requested five conversations. Run them after downloading the official three-file dataset. Record the exact file names, loaded recipe count, date, and model name with your results. Dataset-dependent counts and recipe IDs are intentionally not hard-coded in this document.

## Conversation 1 — Search followed by a new restriction

**User:** Find a penne pasta recipe.

**Expected behavior:** Return existing records with recipe titles and stable IDs. Opening a result shows original ingredient lines and instructions.

**User:** I cannot have dairy.

**Expected behavior:** Re-run the prior penne search with dairy excluded. Show the active restriction. Do not return recipes whose ingredient list contains dairy matches. Do not silently remove the restriction if no match exists.

**What to record:** Search query, active restriction, number of results, and IDs returned.

## Conversation 2 — Programmatic aggregate statistic

**User:** How many recipes contain both garlic and onion? What percentage of the dataset is that?

**Expected behavior:** Scan the complete filtered local recipe collection; count recipes containing both names; compute `count / denominator × 100`; state the denominator and normalization rule. Do not ask Groq to estimate the count.

**What to record:** Numerator, denominator, percentage, active filters, and the normalized ingredient keys used.

## Conversation 3 — Ranking by instruction steps

**User:** Among recipes containing tomato, show the five with the fewest instruction steps.

**Expected behavior:** Filter every loaded recipe containing tomato; calculate steps using the documented rule; sort ascending; show five titles, stable IDs, and step counts.

**What to record:** Filtered recipe count, step-count rule, and ranked result IDs/counts.

## Conversation 4 — Ingredient match with a no-match hard constraint

**User:** I have eggs, onions, tomatoes, and rice. Suggest five recipes using as many of these as possible, with no peanuts and at most one instruction step. Prefer recipes that need the fewest additional ingredients.

**Expected behavior:** Apply peanut exclusion and the step cap as hard filters. Do not relax either condition. If there is no eligible recipe, say no eligible match was found. Closest alternatives, if shown in a future extension, must be clearly marked as violating the constraint; this implementation does not silently promote them into eligible results.

**User:** Exclude eggs too.

**Expected behavior:** Update the hard exclusions and recalculate against the same available ingredient list only if the user remains in the matcher workflow; in chat, restrictions persist for subsequent search. The matcher explicitly labels its input fields and never fabricates an existing recipe.

**What to record:** Available ingredients, exclusions, max steps, eligible count, and top-ranked IDs or no-match result.

## Conversation 5 — Substitution with weak or conflicting evidence

**User:** Take a recipe from the dataset that contains cream and replace cream without dairy. Prefer examples from recipes already in the database.

**Expected behavior:** Retrieve the selected recipe's peers locally first. Show candidate names only if they appear in related records and pass the simple role gate; show supporting recipe IDs and shared ingredients. If no adequate candidate remains after dairy exclusion, explicitly say local evidence is insufficient. If a candidate is selected, Groq drafts revised instructions using only the original recipe and candidate evidence. The app labels taste/texture statements as predictions and does not claim allergy safety or verified quantities.

**What to record:** Original recipe ID, target ingredient, hard restrictions, candidate(s), supporting IDs, and any uncertainty or refusal.

## Automated test set

Run:

```powershell
pip install pytest
python -m pytest -q
```

The unit tests use synthetic examples so they run without downloading the real dataset or calling Groq. They validate deterministic mechanics; they are not a substitute for the five live dataset conversations above.

## One-shot LLM comparison

Use this example to demonstrate why retrieval and calculation are separated from generation:

**Prompt to a plain LLM:** “How many recipes in Recipe Box contain both garlic and onion, and what percentage is that?”

A model without the local JSON data cannot know the exact numerator or denominator. Record its answer as unsupported unless it provides a reproducible dataset-backed computation. The application instead scans the complete loaded records and produces the count, denominator, and percentage using Python.

## Evaluation report template

| Scenario | Expected behavior | Actual result | Pass/Fail | Notes / failure reason |
|---|---|---|---|---|
| 1. Follow-up dairy exclusion | Previous search refined; restriction persists |  |  |  |
| 2. Garlic + onion statistic | Exact count and denominator |  |  |  |
| 3. Tomato fewest steps | Five ranked records with IDs and step counts |  |  |  |
| 4. Hard-constraint no-match | No silent relaxation |  |  |  |
| 5. Weak substitution evidence | Evidence IDs or explicit uncertainty |  |  |  |

## Semantic retrieval validation (run after the one-time vector build)

Build the index first:

```powershell
python scripts/build_vector_index.py --batch-size 32 --chunk-size 512
```

Then test a mix of exact and paraphrased queries in Chat and Recipe Explorer, for example:

| Query | Retrieval behavior to inspect |
|---|---|
| `biryani` | Does an exact dish-name search surface relevant biryani records? |
| `a comforting rice dish with warm spices` | Does semantic retrieval find related rice dishes when the wording differs from recipe titles? |
| `something creamy but dairy-free` | Does the semantic candidate list remain filtered by a hard dairy exclusion? |
| `pasta with garlic and a rich sauce` | Are relevant candidates ranked using both semantic scores and title/ingredient lexical matches? |

Record the actual retrieved IDs and judge relevance against their original ingredients/instructions. Do not describe the system as semantically accurate based only on synthetic unit tests. The tests validate the index contract and hybrid-ranking mechanics with a fake encoder; a real retrieval-quality assessment requires the official dataset and downloaded embedding model.

### Performance checks

1. Run `python scripts/build_vector_index.py` for the first time and record elapsed time, hardware, recipe count, and batch size.
2. Run it again without `--force`; it should detect the matching manifest and skip re-embedding.
3. Restart Streamlit; the parsed SQLite cache and vector index should be reused.
4. Ask an ordinary recipe question; the chat intent parser should not build the complete ingredient vocabulary. Statistics queries intentionally scan the full collection because their answers must be exact.

### Security checks

- Confirm there is no Groq API key text input in the Streamlit UI.
- Confirm `GROQ_API_KEY` is configured only in `.streamlit/secrets.toml` or the server environment.
- Confirm `.streamlit/secrets.toml`, `.env`, JSON data, SQLite caches, and vector-index files are ignored by Git.
- Confirm the app shows only generic provider errors and never renders raw exception objects or request headers.
