# Detailed project build prompt

Build a complete, interview-ready Recipe Box conversational assistant using Streamlit, Python, the Eight Portions Recipe Box dataset, local Sentence Transformers embeddings, FAISS, and Groq for constrained language generation.

## Functional requirements

1. Use all three Recipe Box JSON exports and preserve stable recipe IDs, original titles, ingredients, instructions, source filenames, and source URLs.
2. Build a persistent local embedding index using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). Embed recipe title, ingredients, and a bounded excerpt of instruction text. Normalize vectors and index them with FAISS `IndexFlatIP` so similarity is exact cosine on normalized vectors.
3. Create embeddings once in batches and save the index, ordered recipe ID mapping, and a manifest containing model ID, vector dimensions, record count, and source dataset fingerprint. Reuse the index on later launches. Rebuild only when the dataset fingerprint/model changes or an explicit force-rebuild is requested.
4. Persist a compact SQLite cache of parsed recipe records so that future Streamlit process launches do not repeatedly parse the large JSON exports. Preserve original ingredients and instructions in the cache; raw JSON dictionaries should not be duplicated in memory.
5. Implement hybrid retrieval: FAISS semantic candidate retrieval followed by deterministic lexical reranking (title, ingredient, and instruction matches). Apply hard restrictions against original ingredient lines after retrieval. If embeddings are not ready, clearly label and use a deterministic lexical fallback.
6. Do not use the LLM for dataset counts, percentages, averages, medians, rankings, or factual recipe attributes. Compute these in Python over the full applicable filtered collection.
7. Ingredient matching must retrieve existing recipes only, clearly show matched/missing ingredients, explain scores, enforce hard exclusions and step limits, and state when quantities were not verified.
8. Substitution must use local related-recipe evidence first, show evidence recipe IDs, disclose indirect evidence and uncertainty, obey hard restrictions, and never claim allergy safety from recipe text. Groq may draft updated ingredients/instructions only from the selected recipe and retrieved evidence.
9. Support follow-up turns and retain active restrictions until explicitly changed or removed.
10. Include tests, reproducible evaluation scenarios, a README, and an AI-tool disclosure.

## Groq and API key security

- Groq is only for language generation, never retrieval or calculations.
- Default to a currently supported Groq production model (currently `openai/gpt-oss-120b`) and expose the model selection as a setting.
- Read `GROQ_API_KEY` only from `st.secrets` or the server environment.
- Do not render a key input in the UI, echo credentials, log request headers, print raw SDK exceptions, or place keys in source files.
- Ignore `.streamlit/secrets.toml`, `.env`, downloaded JSON data, parsed caches, and vector index artifacts in Git.

## UI requirements

Use Streamlit with a simple white/light design, restrained green/neutral accents, readable typography, ample whitespace, subtle borders, and responsive layout. No dark theme, gradients, neon colors, or busy animations. The interface should have four clear sections: Chat, Ingredient Matcher, Recipe Explorer & Statistics, and Substitution Lab. Display whether semantic retrieval is active and provide a clear one-time index-build action.

## Deliverables

Provide the complete runnable source project, `requirements.txt`, `scripts/download_dataset.py`, `scripts/build_vector_index.py`, tests, `README.md`, `EVALUATION.md`, `.gitignore`, and `.streamlit/secrets.toml.example`. Do not pretend the third-party dataset or embeddings are bundled when they are not. All generated vector/cache files must remain local and reproducible.
