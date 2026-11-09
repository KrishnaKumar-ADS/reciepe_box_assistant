# Local verification results

Date: 2026-11-09

## Automated checks

```text
21 passed
```

Command:

```powershell
python -m pytest -q
```

The current unit tests cover:

- Stable IDs and official `nosource_` JSON export filenames.
- Deterministic lexical search, hybrid semantic reranking, and candidate-pool expansion when hard restrictions exclude initial nearest neighbors.
- Hard restriction handling for dairy, including coconut milk distinction.
- Dataset statistics, ranking, and ingredient matching.
- Evidence-gated substitution behavior.
- Follow-up restriction updates.
- Downloader extraction and reuse of already extracted exports.
- Persistent SQLite parsed-record cache reuse.
- Persistent vector-index artifact creation, loading, dataset-fingerprint invalidation, and skip-on-rebuild behavior through a deterministic fake encoder.
- Normal recipe chat does not trigger a full ingredient-vocabulary scan.

## Scope and limitations of this verification

The automated tests use small synthetic recipe records. They do not measure semantic relevance or indexing speed on the full Recipe Box data. The three official JSON exports were not included in the source ZIP, and the Sentence Transformers, FAISS, Streamlit, and Groq runtime packages were not installed in the build environment. Therefore a full Streamlit launch, a real model download, a real 125k-record embedding build, a Groq API call, and live retrieval-quality measurements still need to be performed on the user's Windows machine after installing `requirements.txt` and downloading the dataset.

The vector index builder uses the real `SentenceTransformer` API and `faiss-cpu` when installed. The unit test injects a deterministic fake encoder so index persistence and retrieval contracts can be tested without downloading a model.
