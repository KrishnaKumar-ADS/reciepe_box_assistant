# Recipe Box Assistant

A local-first recipe discovery and pantry-matching application built with **Python, Streamlit, Groq, Sentence Transformers, and FAISS**. It lets users search a large recipe collection in natural language, find recipes that use ingredients already in their kitchen, calculate collection statistics, and explore ingredient substitutions using evidence from related recipes.

The application is designed to be easy to run on a local Windows machine. Recipe records, chat history, embeddings, and search indexes are stored locally. Groq is used for natural-language explanations and substitution drafts; deterministic Python code handles recipe filtering, pantry ranking, and dataset calculations.

---

## 1. What the application does

| Feature | Description |
|---|---|
| Recipe chat | Search the local recipe collection using ordinary questions and follow-up messages. |
| Recipe source details | Open a result to see its stable recipe ID, original ingredient list, instructions, source file, and original source link when available. |
| Chat history and memory | Keep separate conversations, save preferences and ingredient exclusions, and return to earlier chats. |
| Pantry matching | Compare the ingredients a user has against the loaded recipes, apply hard exclusions and an optional instruction-step limit, and return a ranked list. |
| Dataset analytics | Calculate counts, percentages, common ingredients, ingredient statistics, and instruction-step rankings from loaded records. |
| Substitution lab | Select an ingredient in an existing recipe, inspect related recipe evidence for alternatives, and generate an adapted recipe draft. |
| Hybrid search | Use local Sentence Transformer embeddings and FAISS when the index is ready, with a keyword-based fallback. |
| Local persistence | Store chat sessions, parsed recipe cache, and vector-search artifacts on the machine running the app. |

## 2. Understanding the application screen

The interface has a **left sidebar** for conversations, tools, dataset management, and the Groq model, plus a **main workspace** for chatting or using a selected recipe tool.

### 2.1 Sidebar — chats

| Field or button | What it does | How to use it |
|---|---|---|
| **New chat** | Creates a separate conversation with its own chat context, exclusions, and memory note. | Select it when starting a different task. |
| **Search chat history** | Filters the visible conversation list by conversation title. | Type part of a title to find a previous chat. |
| **Conversation title** | Opens the selected saved conversation in the main workspace. | Select a title to continue that conversation. |
| **× beside a conversation** | Deletes that conversation. | Use the × next to the conversation you want to remove. |

Up to 30 matching conversations are shown in the sidebar, ordered by most recently updated. Saved chat data is capped at the 60 most recently updated conversations.

### 2.2 Sidebar — Manage current chat

Open the **Manage current chat** expander to access these controls:

| Field or button | What it does |
|---|---|
| **Conversation name** | The editable title for the current conversation. |
| **Save name** | Saves the title entered in the Conversation name field. |
| **Clear messages** | Removes the current chat's messages, clears its previous recipe-search query and active exclusions, and resets its title. The saved memory note remains available. |
| **Delete current chat** | Deletes the selected conversation and opens a new empty conversation. |
| **Export this conversation** | Downloads the current conversation as a JSON file, including its saved chat record and message details. |

### 2.3 Sidebar — Chat memory & preferences

These settings apply to the currently selected conversation, not every conversation.

| Field or button | What to enter or do | Example |
|---|---|---|
| **Active exclusions** | Comma-separated ingredients or categories that must be excluded from future recipe searches in this chat. | `peanuts, dairy, shellfish` |
| **Remember for this chat** | A short preference note to help guide future conversational answers. | `Prefer vegetarian dinners; avoid very spicy dishes.` |
| **Save memory & exclusions** | Saves the exclusion list and memory note locally. | Select after editing either field. |
| **Clear memory** | Removes the saved preference note for this conversation. | Select to clear the note. |
| **Reset context** | Clears active exclusions and the previous recipe-search query for this conversation. | Select when you want to reset search context. |

Memory is a preference aid, not a source of recipe facts. Explicit restrictions in a new request take priority over saved preferences.

### 2.4 Sidebar — Recipe tools

| Tool | Purpose |
|---|---|
| **Ingredient matcher** | Opens a structured form for matching pantry ingredients to recipes. |
| **Substitution lab** | Opens the workflow for finding a recipe ingredient alternative using related local recipes as evidence. |

Use the **Back to chat** control in a tool view to return to the conversation screen.

### 2.5 Sidebar — Recipe collection

This section manages the recipe data used by search, pantry matching, and analytics.

| Field or button | What it does | When to use it |
|---|---|---|
| **Import dataset JSON files** | Accepts one or more `.json` recipe export files and saves them in the local `data/` folder. | Use when the dataset files are already on the computer. |
| **Import selected files** | Imports the JSON files selected in the uploader and reloads the local dataset cache. | Select after choosing the export files. |
| **Download official dataset** | Downloads and extracts the Recipe Box archive into `data/`. | Use for the first setup when the data files are not already available. |
| **Build semantic embeddings** | Creates the local Sentence Transformer embeddings and FAISS index. | Use after loading the dataset to enable semantic retrieval. |

The sidebar reports whether the collection is **not loaded**, **partial**, or **full**. The full collection is identified by the presence of all three expected source exports. A partial collection can still be searched, but its results and statistics cover only the files that were loaded.

After the data is loaded, the main screen's **Collection details** expander shows the number of recipes loaded, the number of JSON exports, the number of distinct ingredients, and the filenames used.

### 2.6 Sidebar — AI provider

| Field | Description |
|---|---|
| **Groq model** | Selects the model used for natural-language explanations and adapted-recipe drafting. The current identifiers configured in `app.py` are `openai/gpt-oss-120b` (default), `openai/gpt-oss-20b`, and `qwen/qwen3.8-27b`; availability depends on the Groq service and account. |
| **Groq connected / not configured** | Indicates whether the application found a Groq API key. The key itself is never displayed in the interface. |

The application can still load recipes, perform keyword retrieval, calculate dataset statistics, and run deterministic pantry matching without a Groq key. Groq is needed for generated conversational explanations and for the **Generate adapted recipe** action in the substitution lab.

### 2.7 Main workspace — chat screen

| Screen block or control | What it shows or does |
|---|---|
| **Semantic search ready / Keyword fallback** | Shows whether the local FAISS index is available or the application is using keyword search. |
| **Groq connected / Groq not configured** | Shows the provider connection state. |
| **Collection details** | Expands to show dataset counts and loaded JSON filenames. |
| **Try asking** | Example prompts that can be selected to populate the message field in a new conversation. |
| **Conversation messages** | Displays the user and assistant messages for the selected chat. |
| **Recipe result card** | Expands to show the recipe ID, source file, original ingredients, original instructions, and source page link when available. |
| **Attach files** | Adds reference documents to the next message only. |
| **Message Recipe Box Assistant…** | The text area where a user enters a question or instruction. |
| **Send** | Submits the message and displays the application's processing status. |

#### Supported chat attachments

The attachment uploader accepts **PDF, TXT, Markdown (`.md`), JSON, and CSV** files. The current limits are:

- Maximum file size: **8 MB per file**.
- PDF extraction: up to the first **40 pages** of each PDF.
- Extracted text: up to **14,000 characters per file** and **24,000 characters total per message**.

Attachments are read for the current turn and are not persisted as files in chat history. When Groq is configured, extracted attachment text may be sent to Groq as part of that request. Do not attach confidential documents unless you are comfortable with that processing.

## 3. Using the Ingredient matcher

Select **Ingredient matcher** in the sidebar. This screen is intended for a structured pantry search.

| Field | What to enter | Example |
|---|---|---|
| **Ingredients you have** | Enter ingredients available in the kitchen. Separate items with commas, semicolons, or new lines. | `eggs, onions, tomatoes, rice` |
| **Hard exclusions** | Enter ingredients or categories that must not appear in a selected recipe. | `peanuts, dairy` |
| **Limit instruction steps** | Enable this checkbox to apply a maximum number of instruction steps. | Enable for a recipe with a short method. |
| **Maximum steps** | Set the upper step limit. This field is enabled only when Limit instruction steps is checked. The default is 6; allowed values are 1–100. | `6` |
| **Prefer fewer additional ingredients** | When selected, recipes needing fewer ingredients beyond the pantry list are prioritized after the number of pantry matches. | Leave selected to minimize extra shopping. |
| **Find matching recipes** | Runs the match across the complete loaded recipe collection and displays up to five results. | Select after filling the form. |

### How the pantry ranking works

The matcher uses deterministic Python logic rather than asking the language model to choose the winners. It scans the complete collection currently loaded and applies hard filters before ranking:

1. Recipes containing a hard-excluded ingredient are removed.
2. When the step limit is enabled, recipes above that limit are removed.
3. The remaining recipes are ranked by the number of pantry ingredients they match.
4. When **Prefer fewer additional ingredients** is selected, recipes requiring fewer additional ingredients are ranked next.
5. Instruction-step count, recipe title, and stable recipe ID provide subsequent tie-breakers.

Each result explains its **recipe ID**, matched pantry ingredients, additional ingredients, instruction-step count, and why it appears in the ranking. The result summary also reports how many loaded recipes were checked and how many remained after the filters.

Ingredient matching is based on normalized ingredient names. It does not verify the quantities available in the user's kitchen, and it does not establish allergy safety or protection from cross-contact. Always inspect the original ingredient list and product labels where an allergy or medical restriction is involved.

## 4. Using the Evidence-based substitution lab

Select **Substitution lab** in the sidebar. The workflow uses the local recipe collection to identify possible replacement ingredients and supporting recipe examples.

| Field or control | What to enter or select | Purpose |
|---|---|---|
| **Find an existing recipe** | Enter a recipe title or relevant search phrase. | Finds candidate recipes in the loaded collection. |
| **Search recipes** | Runs the search and displays candidate source recipes. | Starts the substitution workflow. |
| **Choose the exact recipe** | Select the recipe by its title and stable recipe ID. | Identifies the original recipe to adapt. |
| **Ingredient to replace** | Select the ingredient line from the chosen recipe. | Specifies the ingredient to change. |
| **Hard restrictions** | Enter comma-separated ingredients or categories that must be excluded from replacement candidates. | Filters unsuitable candidates. |
| **Find local evidence** | Searches related recipes in the local dataset. | Produces candidate replacement ingredients and supporting recipes. |
| **Dataset-supported candidate** | Select one of the candidates and review how often it appears in related recipes. | Chooses the candidate for the draft. |
| **Supporting recipe evidence** | Expand or inspect the recipe titles and IDs supporting the selected candidate. | Makes the basis for the candidate visible. |
| **Generate adapted recipe** | Requests revised instructions from Groq using the original recipe and selected local evidence. | Produces the adaptation draft. |

The dataset does not contain an authoritative ingredient-substitution table. A candidate found in related recipes is **indirect evidence**, not proof that the ingredients are interchangeable. The generated method is a draft: quantities, taste, texture, and cooking behavior may need testing. The application does not guarantee that a substitution is allergen-safe.

## 5. Running the application on Windows

Follow these steps on a Windows 10/11 computer.

### Prerequisites

- Python **3.11 or newer** installed and available through the Windows Python launcher (`py`).
- Git installed, if obtaining the source using the clone command below.
- Internet access for the initial dataset download and Sentence Transformer model download.
- A Groq API key for generated chat explanations and substitution drafts. The local data-processing features can be explored without it.
- Enough free disk space for the recipe archive, Python dependencies, embedding model, and generated vector index.

### Step 1 — Download the project

Open **PowerShell** and run the following commands. Replace the example GitHub address with the repository address where this application is hosted:

```powershell
git clone https://github.com/<OWNER>/<REPOSITORY>.git
cd <REPOSITORY>
```

Alternatively, download the repository source as a ZIP file from GitHub, extract it, open PowerShell in the extracted project directory, and continue with Step 2.

### Step 2 — Create a Python virtual environment

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If Python 3.12 is not installed but another supported Python 3.11+ version is available, replace `3.12` with that installed version. If PowerShell prevents virtual-environment activation, run this in the same terminal and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

When activation succeeds, `(.venv)` appears at the beginning of the PowerShell prompt.

### Step 3 — Install the dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main dependencies include Streamlit (web interface), Groq (language-model API client), Sentence Transformers (local text embeddings), FAISS CPU (vector similarity search), NumPy, and pypdf (PDF text extraction).

### Step 4 — Configure the Groq API key

This step is required for generated conversational explanations and adapted-recipe drafting. It can be skipped to test local dataset loading, keyword search, analytics, and pantry matching.

Create the local Streamlit secrets file from the example:

```powershell
Copy-Item .streamlit\secrets.toml.example .streamlit\secrets.toml
notepad .streamlit\secrets.toml
```

Add the following line and replace the example value with your own Groq API key:

```toml
GROQ_API_KEY = "gsk_your_actual_key_here"
```

Save the file and close Notepad. The application reads this value on the server side and does not display it. As an alternative, the key can be set for the current PowerShell session:

```powershell
$env:GROQ_API_KEY = "gsk_your_actual_key_here"
```

### Step 5 — Download the Recipe Box dataset

The application needs recipe JSON files before it can search or calculate collection statistics. The easiest option is to run:

```powershell
python scripts/download_dataset.py
```

The script downloads the Recipe Box archive and extracts the three expected JSON exports into the project's `data/` directory:

```text
data/
├── recipes_raw_nosource_fn.json
├── recipes_raw_nosource_epi.json
└── recipes_raw_nosource_ar.json
```

The dataset can also be downloaded from inside the application using **Download official dataset** in the sidebar. If the JSON files are already available on the computer, use **Import dataset JSON files** and **Import selected files** instead.

The dataset is a separate third-party resource and is downloaded at setup time; it is not required to be bundled with the application source. Dataset source: [Recipe Box dataset](https://eightportions.com/datasets/Recipes/).

### Step 6 — Build the semantic search index

After the JSON exports are in `data/`, run:

```powershell
python scripts/build_vector_index.py
```

This step:

1. Loads and parses the recipe JSON records.
2. Loads the `sentence-transformers/all-MiniLM-L6-v2` embedding model.
3. Generates normalized recipe embeddings in batches.
4. Builds a local FAISS similarity index.
5. Saves the index and its metadata under `data/vector_index/`.

The first run downloads the embedding model and can take time depending on the computer. The generated embeddings are local and do **not** use the Groq API key. A matching index is reused on later runs. If an index cannot be built, the app can still use its keyword-search fallback.

Optional command-line settings are available:

```powershell
python scripts/build_vector_index.py --help
python scripts/build_vector_index.py --device cpu
python scripts/build_vector_index.py --batch-size 64 --chunk-size 1024
python scripts/build_vector_index.py --force
```

### Step 7 — Start the Streamlit application

From the project root, with the virtual environment activated, run:

```powershell
streamlit run app.py
```

If the `streamlit` command is not found, use:

```powershell
python -m streamlit run app.py
```

Streamlit prints a local address in the terminal. Open the displayed **Local URL**, normally:

```text
http://localhost:8501
```

Keep the PowerShell terminal open while using the application. To stop the server, return to that terminal and press `Ctrl+C`.

### Step 8 — Run the automated tests (optional)

The test dependency is listed in `requirements-dev.txt`. Install it and run the test suite:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

The tests cover core dataset/search behavior and pantry-request parsing, filtering, ranking, and follow-up handling.

## 6. Suggested first-use workflow

After the application opens in the browser:

1. Check the sidebar's **Recipe collection** status. It should show a full dataset once all three JSON exports are loaded.
2. Confirm **Semantic index ready** appears in the main screen or in the Recipe collection section. If it does not, select **Build semantic embeddings** in the sidebar or run the build command from Step 6.
3. Confirm the AI provider shows **Groq connected** if you configured an API key. The local recipe tools remain usable without it.
4. Start a new chat and try: `Find a penne pasta recipe.` Open a result card to inspect the original ingredients and instructions.
5. Try a follow-up restriction, for example: `I cannot have dairy.` The exclusion is retained for later recipe searches in that conversation.
6. Open **Ingredient matcher** and enter pantry ingredients, restrictions, and an optional step limit.
7. Open **Substitution lab**, search for a recipe, select an ingredient to replace, inspect the supporting recipes, and generate an adaptation draft if Groq is configured.

### Example pantry-matching request

The same type of request can also be entered directly in the main chat:

> I have eggs, onions, tomatoes, and rice. Suggest five recipes using as many of these as possible, with no peanuts and at most six instruction steps. Prefer recipes that need the fewest additional ingredients.

The response should provide a ranked set of recipe records with their IDs, matched pantry ingredients, additional ingredients, step counts, applied restrictions, and a summary of the loaded dataset searched. A follow-up such as `Exclude eggs too` updates the previous pantry search using the same pantry list and step limit, adds eggs to the hard exclusions, and recalculates the results.

## 7. How the project is organized

```text
recipe-box-assistant/
├── app.py                         # Streamlit application and UI
├── README.md                      # Setup and usage guide
├── requirements.txt               # Runtime dependencies
├── requirements-dev.txt           # Test dependencies
├── .gitignore                     # Local secrets and generated-data exclusions
├── .streamlit/
│   ├── config.toml                # Streamlit appearance/configuration
│   └── secrets.toml.example       # Example Groq API key configuration
├── data/
│   ├── README.md                  # Dataset download and expected filenames
│   └── .gitkeep                   # Keeps the empty data directory present
├── scripts/
│   ├── download_dataset.py        # Downloads and extracts the dataset
│   └── build_vector_index.py      # Builds local embeddings and FAISS index
├── src/
│   ├── analytics.py               # Dataset statistics and instruction-step analysis
│   ├── dataset.py                 # Recipe record parsing and stable IDs
│   ├── dataset_cache.py           # Parsed dataset cache
│   ├── dataset_download.py        # Dataset archive download/extraction
│   ├── groq_service.py            # Groq chat and adaptation client
│   ├── matching.py                # Deterministic pantry matching and ranking
│   ├── normalization.py           # Ingredient/name normalization
│   ├── search.py                  # Recipe retrieval and ingredient restrictions
│   ├── substitution.py            # Related-recipe evidence for substitutions
│   └── vector_index.py            # Embedding generation and FAISS operations
└── tests/
    ├── test_core.py               # Core data, search, and logic tests
    └── test_pantry_intent.py      # Pantry prompt and follow-up regression tests
```

### Main processing flow

1. **Load data:** `dataset.py` reads the three source JSON exports and converts records into consistent recipe objects while retaining their original text and source details.
2. **Cache records:** `dataset_cache.py` stores parsed records locally so the large source files do not need to be fully reparsed on every run.
3. **Prepare semantic search:** `vector_index.py` creates local embeddings and a FAISS index for natural-language retrieval.
4. **Search recipes:** `search.py` combines semantic candidates with lexical signals and applies ingredient restrictions against recipe ingredient lines.
5. **Match pantry ingredients:** `matching.py` scans the complete loaded collection and performs hard filtering and deterministic ranking.
6. **Calculate analytics:** `analytics.py` calculates results from the loaded recipe records in Python.
7. **Find substitution evidence:** `substitution.py` finds possible replacement ingredients based on related recipe records.
8. **Generate language:** `groq_service.py` uses Groq to explain retrieved evidence or draft an adapted method. The model is not the source of recipe IDs, deterministic pantry ranking, or collection calculations.

## 8. Local files and data handling

During use, the application may create these local files and folders:

| Path | Purpose |
|---|---|
| `.streamlit/secrets.toml` | Local Groq API key configuration. |
| `data/recipes_raw_*.json` | Downloaded or imported recipe source exports. |
| `data/.cache/recipes.sqlite3` | Parsed recipe cache. |
| `data/.cache/chat_sessions.json` | Saved conversations, chat memory, and conversation metadata. |
| `data/vector_index/` | Generated embedding vectors, FAISS index, and index metadata. |

Chat history and memory are stored on the computer running Streamlit. They are not automatically synchronized between different computers or browser sessions. Uploaded chat attachments are processed for their submitted turn and are not saved as attachment files by the application.

## 9. Important limitations

- Search and statistics operate on the JSON exports that are currently loaded. Load all three expected source files for full-dataset coverage.
- Pantry matching compares normalized ingredient names; it does not calculate whether the quantity in the kitchen is sufficient.
- The ingredient list in a recipe is not a medical or allergen-safety certification. Cross-contact and product-label details are outside the dataset's scope.
- Substitution candidates are inferred from patterns in related recipes. They are not guaranteed culinary equivalents, and generated quantities or cooking behavior may need testing.
- The dataset does not provide reliable standardized nutrition or cooking-time values for every recipe. The application avoids inventing these values when they are unavailable.
- Semantic retrieval requires the local embedding model and FAISS index. Keyword search remains available if semantic indexing is not ready.
- Generated conversational explanations and substitution drafts require a valid Groq API key and a model available to the configured account.

## 10. AI and component usage

- **Python** performs data parsing, ingredient normalization, hard-restriction checks, pantry matching/ranking, analytics, and local evidence collection.
- **Sentence Transformers** generates local text embeddings for recipe records and search queries.
- **FAISS** performs local vector similarity search over those embeddings.
- **Groq** generates natural-language explanations grounded in retrieved recipe evidence and drafts revised recipe instructions for the substitution workflow.
- **Streamlit** provides the browser-based interface and local interaction flow.

The application keeps deterministic recipe facts and calculations in Python and uses the language model for the language-generation tasks described above.
