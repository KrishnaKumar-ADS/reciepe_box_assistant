from __future__ import annotations

import json
import os
import re
import sys
import uuid
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.analytics import (  # noqa: E402
    common_ingredients,
    count_recipes_with,
    filter_recipes_for_ingredients,
    format_analytics_answer,
    ingredient_statistics,
    interpret_analytics_question,
    rank_by_steps,
)
from src.dataset import Recipe, has_complete_dataset  # noqa: E402
from src.dataset_cache import load_dataset_cached  # noqa: E402
from src.dataset_download import download_and_extract  # noqa: E402
from src.groq_service import DEFAULT_MODEL, GroqService  # noqa: E402
from src.matching import match_recipes, parse_available_ingredients  # noqa: E402
from src.normalization import ingredient_key  # noqa: E402
from src.search import (  # noqa: E402
    ingredient_violations,
    parse_restriction_update,
    restriction_only_message,
    search_recipes,
)
from src.substitution import (  # noqa: E402
    build_updated_ingredient_list,
    find_substitution_evidence,
)
from src.vector_index import (  # noqa: E402
    DEFAULT_EMBEDDING_MODEL,
    RecipeVectorIndex,
    build_vector_index,
    dataset_fingerprint,
)

DATA_DIR = ROOT / "data"
CHAT_STORE = DATA_DIR / ".cache" / "chat_sessions.json"
MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024
MAX_ATTACHMENT_CHARS = 14_000
MAX_TOTAL_ATTACHMENT_CHARS = 24_000

st.set_page_config(
    page_title="Recipe Box Assistant",
    page_icon="🍲",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Clean light interface inspired by focused chat workspaces. No dark surfaces or gradients.
st.markdown(
    """
    <style>
      :root { color-scheme: light; }
      .stApp { background: #ffffff; color: #202a23; }
      [data-testid="stHeader"] { background: rgba(255,255,255,.96); }
      [data-testid="stMainBlockContainer"] {
        max-width: 1050px; padding-top: 1.25rem; padding-bottom: 2.5rem;
      }
      [data-testid="stSidebar"] {
        background: #f7f8f6; border-right: 1px solid #e7eae5;
      }
      [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
      [data-testid="stSidebar"] * { color: #27332b; }
      h1, h2, h3 { color: #202b23; letter-spacing: -.025em; }
      h1 { font-size: 1.85rem; font-weight: 690; }
      h2 { font-size: 1.35rem; font-weight: 650; }
      h3 { font-size: 1.08rem; font-weight: 640; }
      p, li, label, .stMarkdown { color: #344139; }
      .brand-row { display:flex; align-items:center; gap:10px; margin:0 0 1.05rem 0; }
      .brand-mark {
        width:36px; height:36px; border-radius:11px; background:#eaf1e9;
        color:#355b3c; display:flex; align-items:center; justify-content:center;
        font-size:19px; border:1px solid #dce7da;
      }
      .brand-name { font-weight:720; font-size:1.05rem; color:#26352a; }
      .brand-sub { font-size:.72rem; color:#7b877d; margin-top:1px; }
      .eyebrow { color:#708174; font-size:.72rem; font-weight:750; letter-spacing:.1em; text-transform:uppercase; }
      .welcome-title { font-size:2rem; letter-spacing:-.04em; font-weight:720; color:#1e2a21; margin:.2rem 0 .35rem 0; }
      .welcome-copy { color:#7b877e; font-size:.97rem; margin-bottom:1rem; }
      .status-line { color:#738076; font-size:.78rem; margin:.2rem 0 .8rem 0; }
      .status-dot { display:inline-block; width:7px; height:7px; border-radius:50%; background:#6b9870; margin:0 6px 1px 0; }
      .status-dot-muted { display:inline-block; width:7px; height:7px; border-radius:50%; background:#b7beb8; margin:0 6px 1px 0; }
      .stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {
        border-radius:10px; border:1px solid #dce3da; background:#ffffff;
        color:#2e4032; font-weight:570; min-height:2.45rem; transition:background .12s ease;
      }
      .stButton > button:hover, .stFormSubmitButton > button:hover, .stDownloadButton > button:hover {
        background:#f3f7f1; border-color:#b8cbb6; color:#25452d;
      }
      .stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
        background:#365f40; border-color:#365f40; color:white;
      }
      .stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
        background:#2d5337; border-color:#2d5337; color:white;
      }
      [data-testid="stChatMessage"] {
        border:0; background:transparent; padding:.65rem .1rem; gap:.8rem;
      }
      [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] { line-height:1.65; }
      [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background:#f5f7f4; border:1px solid #edf0eb; border-radius:16px;
        padding:.75rem 1rem; margin:.55rem 0 .8rem auto; max-width:92%;
      }
      [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {
        margin:.45rem 0 1rem 0;
      }
      .stTextArea textarea, .stTextInput input, .stSelectbox [data-baseweb="select"] {
        border-radius:11px !important; border-color:#dce3da !important; background:#fff !important;
      }
      .stTextArea textarea:focus, .stTextInput input:focus {
        border-color:#83a287 !important; box-shadow:0 0 0 1px #83a287 !important;
      }
      [data-testid="stFileUploader"] section { border:1px dashed #cfd9ce; border-radius:12px; background:#fbfcfa; }
      [data-testid="stExpander"] { border:1px solid #e7ebe5; border-radius:11px; background:#fff; }
      [data-testid="stMetric"] { background:#fafbf9; border:1px solid #e9ede7; padding:11px 13px; border-radius:11px; }
      [data-testid="stStatusWidget"] { border-radius:12px; }
      a { color:#356744 !important; }
      hr { border-color:#e7ebe5; }
      footer { visibility:hidden; }
      .sidebar-label { color:#7b877d; font-size:.7rem; font-weight:750; letter-spacing:.1em; text-transform:uppercase; margin:.8rem 0 .45rem 0; }
      .tiny-note { color:#879188; font-size:.73rem; line-height:1.45; }
      .tool-banner { border:1px solid #e4eae1; background:#f8faf7; padding:14px 16px; border-radius:13px; }
      @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] { padding-left:1rem; padding-right:1rem; }
        .welcome-title { font-size:1.6rem; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_chat_session() -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex,
        "title": "New chat",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "messages": [],
        "active_restrictions": [],
        "last_search_query": "",
        "memory_note": "",
    }


def load_chat_sessions() -> dict[str, dict[str, Any]]:
    """Load local-only chat history. Uploaded document contents are never persisted."""
    CHAT_STORE.parent.mkdir(parents=True, exist_ok=True)
    if not CHAT_STORE.is_file():
        session = new_chat_session()
        return {session["id"]: session}
    try:
        payload = json.loads(CHAT_STORE.read_text(encoding="utf-8"))
        raw_sessions = payload.get("sessions", []) if isinstance(payload, dict) else []
        sessions: dict[str, dict[str, Any]] = {}
        for item in raw_sessions:
            if not isinstance(item, dict) or not item.get("id"):
                continue
            session = {
                "id": str(item["id"]),
                "title": str(item.get("title") or "New chat")[:90],
                "created_at": str(item.get("created_at") or now_iso()),
                "updated_at": str(item.get("updated_at") or now_iso()),
                "messages": item.get("messages", []) if isinstance(item.get("messages", []), list) else [],
                "active_restrictions": item.get("active_restrictions", []) if isinstance(item.get("active_restrictions", []), list) else [],
                "last_search_query": str(item.get("last_search_query") or ""),
                "memory_note": str(item.get("memory_note") or "")[:4000],
            }
            # Keep the on-disk chat record intentionally small and structurally safe.
            session["messages"] = [
                {
                    "role": msg.get("role", "assistant"),
                    "content": str(msg.get("content", ""))[:24000],
                    "recipe_ids": [str(x) for x in msg.get("recipe_ids", []) if isinstance(x, str)],
                    "attachments": [str(x)[:160] for x in msg.get("attachments", []) if isinstance(x, str)],
                    "timestamp": str(msg.get("timestamp") or now_iso()),
                }
                for msg in session["messages"]
                if isinstance(msg, dict) and msg.get("role") in {"user", "assistant"}
            ][-100:]
            sessions[session["id"]] = session
        if sessions:
            return sessions
    except Exception:
        # A corrupt local history must not prevent the recipe application from starting.
        pass
    session = new_chat_session()
    return {session["id"]: session}


def save_chat_sessions() -> None:
    """Persist chat text and explicit memory locally, using an atomic replace."""
    CHAT_STORE.parent.mkdir(parents=True, exist_ok=True)
    sessions = list(st.session_state.chat_sessions.values())
    sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
    payload = {"version": 1, "sessions": sessions[:60]}
    temp_path = CHAT_STORE.with_suffix(".tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temp_path, CHAT_STORE)


def get_active_session() -> dict[str, Any]:
    sessions = st.session_state.chat_sessions
    active_id = st.session_state.get("active_chat_id")
    if active_id not in sessions:
        if not sessions:
            session = new_chat_session()
            sessions[session["id"]] = session
        active_id = max(sessions.values(), key=lambda x: x.get("updated_at", ""))["id"]
        st.session_state.active_chat_id = active_id
    return sessions[active_id]


def touch_session(session: dict[str, Any]) -> None:
    session["updated_at"] = now_iso()
    save_chat_sessions()


@st.cache_resource(show_spinner="Loading recipe records…", max_entries=2)
def cached_dataset(file_signature: tuple[tuple[str, int, int], ...]):
    return load_dataset_cached(DATA_DIR)


@st.cache_resource(show_spinner=False, max_entries=2)
def cached_vector_index(fingerprint: str) -> RecipeVectorIndex | None:
    return RecipeVectorIndex.load(DATA_DIR / "vector_index", expected_fingerprint=fingerprint)


@st.cache_resource(show_spinner=False, max_entries=2)
def cached_recipe_lookup(signature: tuple[tuple[str, int, int], ...]) -> dict[str, Recipe]:
    records, _ = cached_dataset(signature)
    return {recipe.recipe_id: recipe for recipe in records}


@st.cache_resource(show_spinner=False, max_entries=2)
def cached_distinct_ingredient_count(signature: tuple[tuple[str, int, int], ...]) -> int:
    records, _ = cached_dataset(signature)
    return len({key for recipe in records for key in recipe.ingredient_keys})


def file_signature() -> tuple[tuple[str, int, int], ...]:
    items = []
    for path in sorted(DATA_DIR.glob("*.json")) if DATA_DIR.exists() else []:
        stat = path.stat()
        items.append((path.name, stat.st_size, stat.st_mtime_ns))
    return tuple(items)


def get_secret_key() -> str:
    """Read Groq credentials only on the server; never render or log the value."""
    key = os.getenv("GROQ_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("GROQ_API_KEY", "") or "").strip()
    except Exception:
        return ""


def parse_comma_list(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[,;\n]", text or "") if part.strip()]


def render_recipe_card(recipe: Recipe, show_instructions: bool = True) -> None:
    with st.expander(f"{recipe.title}  ·  {recipe.recipe_id}", expanded=False):
        st.caption(f"Recipe ID: {recipe.recipe_id}  |  Source file: {recipe.source_file}")
        if recipe.source_url.startswith(("http://", "https://")):
            st.markdown(f"[Original source page]({recipe.source_url})")
        st.markdown("**Original ingredients**")
        if recipe.ingredients:
            for line in recipe.ingredients:
                st.markdown(f"- {line}")
        else:
            st.caption("No ingredient list is available in this record.")
        if show_instructions:
            st.markdown("**Original instructions**")
            if recipe.instructions:
                for index, step in enumerate(recipe.instructions, 1):
                    st.markdown(f"{index}. {step}")
            else:
                st.caption("No instructions are available in this record.")


def extract_upload_text(uploaded_files: list[Any]) -> tuple[dict[str, str], list[str]]:
    """Extract bounded text for this turn only; raw uploads are not saved to disk."""
    documents: dict[str, str] = {}
    notes: list[str] = []
    total_chars = 0
    for uploaded in uploaded_files:
        name = Path(uploaded.name).name
        raw = uploaded.getvalue()
        if len(raw) > MAX_ATTACHMENT_BYTES:
            notes.append(f"{name}: skipped because it exceeds the 8 MB per-file limit.")
            continue
        suffix = Path(name).suffix.lower()
        text = ""
        try:
            if suffix == ".pdf":
                try:
                    from pypdf import PdfReader
                except ImportError:
                    notes.append(f"{name}: PDF reading is unavailable. Install the pypdf dependency and restart.")
                    continue
                reader = PdfReader(BytesIO(raw))
                pages = []
                for page in reader.pages[:40]:
                    pages.append(page.extract_text() or "")
                text = "\n\n".join(pages)
            elif suffix in {".txt", ".md", ".csv"}:
                text = raw.decode("utf-8-sig", errors="replace")
            elif suffix == ".json":
                parsed = json.loads(raw.decode("utf-8-sig"))
                text = json.dumps(parsed, ensure_ascii=False, indent=2)
            else:
                notes.append(f"{name}: unsupported file type.")
                continue
        except Exception:
            notes.append(f"{name}: could not extract readable text.")
            continue
        remaining = MAX_TOTAL_ATTACHMENT_CHARS - total_chars
        if remaining <= 0:
            notes.append("The total attachment text limit was reached; remaining files were skipped.")
            break
        text = text[: min(MAX_ATTACHMENT_CHARS, remaining)].strip()
        if not text:
            notes.append(f"{name}: no extractable text was found.")
            continue
        documents[name] = text
        total_chars += len(text)
    return documents, notes


def is_pantry_request(message: str) -> bool:
    return bool(re.search(
        r"\b(i have|i've got|ive got|available ingredients|ingredients i have|my ingredients are)\b",
        message,
        re.I,
    ))


def pantry_terms_from_message(message: str) -> list[str]:
    value = re.sub(
        r"^.*?\b(?:i have|i've got|ive got|available ingredients(?: are)?|ingredients i have(?: are)?|my ingredients are)\b",
        "",
        message,
        flags=re.I,
    )
    value = re.split(r"\b(?:but|and no|without|exclude|excluding|avoid|at most|maximum)\b", value, maxsplit=1, flags=re.I)[0]
    value = re.sub(r"\b(?:please suggest|suggest|find|show|give me|recipes?|dishes?)\b", "", value, flags=re.I)
    return parse_available_ingredients(value)


def process_chat_message(
    message: str,
    recipes: list[Recipe],
    groq: GroqService,
    *,
    active_restrictions: list[str],
    last_search_query: str,
    vector_index: RecipeVectorIndex | None = None,
    recipe_lookup: dict[str, Recipe] | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    memory_note: str = "",
    attachment_context: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Handle deterministic recipe intents first, then request a grounded explanation."""
    updated, added, restriction_changed = parse_restriction_update(message, active_restrictions)
    attachment_context = attachment_context or {}
    lower_message = message.lower()
    unsupported_attribute_terms = (
        "calorie", "calories", "nutrition", "nutritional value", "protein", "carbohydrate",
        "carbs", "fat content", "fiber", "fibre", "cooking time", "cook time",
        "prep time", "preparation time", "how long does it take", "minutes to cook",
    )
    if any(term in lower_message for term in unsupported_attribute_terms) and not attachment_context:
        return {
            "content": (
                "The Recipe Box records do not provide a reliable standardized value for that attribute. "
                "I won't guess nutrition values or cooking/preparation times. I can still show original ingredients "
                "and instructions for matching recipes."
            ),
            "recipe_ids": [],
            "active_restrictions": updated,
            "last_search_query": last_search_query,
        }

    analytics = interpret_analytics_question(message, recipes, updated) if recipes else None
    if analytics:
        answer = format_analytics_answer(analytics)
        if not has_complete_dataset(st.session_state.get("loaded_dataset_filenames", [])):
            answer += "\n\n**Dataset coverage:** This calculation covers only the JSON exports currently loaded."
        if updated:
            answer += "\n\n**Active exclusions:** " + ", ".join(updated) + "."
        return {"content": answer, "recipe_ids": [], "active_restrictions": updated,
                "last_search_query": last_search_query}

    # Ingredient availability requests use deterministic pantry ranking, not an LLM guess.
    if recipes and is_pantry_request(message):
        available = pantry_terms_from_message(message)
        max_steps_match = re.search(r"(?:at most|maximum of?|no more than)\s+(\d+)\s+steps?", lower_message)
        step_limit = int(max_steps_match.group(1)) if max_steps_match else None
        excluded = list(updated)
        results = match_recipes(recipes, available, excluded, max_steps=step_limit,
                                limit=5, prefer_fewer_missing=True)
        if not available:
            answer = "I couldn't identify the ingredient names. Please list them separated by commas, for example: `I have eggs, onions, tomatoes and rice`."
            return {"content": answer, "recipe_ids": [], "active_restrictions": updated,
                    "last_search_query": last_search_query}
        if not results:
            answer = (
                "No eligible recipe matched those pantry ingredients and hard restrictions. "
                "I did not relax the exclusions. You can remove a restriction explicitly or add more ingredients."
            )
            return {"content": answer, "recipe_ids": [], "active_restrictions": updated,
                    "last_search_query": last_search_query}
        lines = [
            f"I compared your **{len(available)} available ingredient names** against the loaded recipe collection. "
            "These are existing recipes, ranked by matched ingredients, missing ingredients, and instruction steps:"
        ]
        ids = []
        for rank, item in enumerate(results, 1):
            ids.append(item.recipe.recipe_id)
            lines.append(
                f"\n**{rank}. {item.recipe.title}** (`{item.recipe.recipe_id}`)  \n"
                f"Matched: {', '.join(item.matched) or 'none'}  \n"
                f"Additional required: {', '.join(item.missing) or 'none'}  \n"
                f"Instruction steps: {item.steps} · Score: {item.score:.1f}"
            )
        lines.append("\nIngredient quantities are not validated; this is a name-level comparison.")
        if updated:
            lines.append("\nHard exclusions applied: " + ", ".join(updated) + ".")
        return {"content": "\n".join(lines), "recipe_ids": ids,
                "active_restrictions": updated, "last_search_query": last_search_query}

    if restriction_only_message(message) and last_search_query:
        query = last_search_query
        banner = (
            "\n\nApplied to the previous search: **" + ", ".join(updated) + "**."
            if updated else "\n\nThe active ingredient exclusions have been updated."
        )
    else:
        query = message
        if restriction_changed and last_search_query and len(message.split()) <= 9:
            query = last_search_query
        else:
            last_search_query = query
        banner = ""

    if not recipes:
        if attachment_context and groq.available:
            fallback = "I could not find a local Recipe Box dataset record. I can answer only from the text in the file(s) you attached to this turn."
            answer = groq.grounded_explanation(
                message, {"uploaded_documents": attachment_context}, fallback,
                history=conversation_history, memory=memory_note,
            )
            return {"content": answer, "recipe_ids": [], "active_restrictions": updated,
                    "last_search_query": last_search_query}
        return {"content": "The recipe collection is not loaded yet. Add the three Recipe Box JSON exports in the sidebar, then ask again.",
                "recipe_ids": [], "active_restrictions": updated, "last_search_query": last_search_query}

    hits = search_recipes(recipes, query, restrictions=updated, limit=8,
                          vector_index=vector_index, recipe_lookup=recipe_lookup)
    if not hits and attachment_context and groq.available:
        fallback = (
            "No strong Recipe Box match was found. The following answer is limited to the uploaded document text; "
            "it does not establish a matching record in the dataset."
        )
        answer = groq.grounded_explanation(
            message,
            {"retrieved_recipes": [], "active_exclusions": updated,
             "uploaded_documents": attachment_context},
            fallback,
            history=conversation_history,
            memory=memory_note,
        )
        return {"content": answer + banner, "recipe_ids": [], "active_restrictions": updated,
                "last_search_query": last_search_query}

    if not hits:
        answer = (
            "I couldn't find a strong match in the loaded recipe records. Try a dish name, ingredient combination, "
            "or a more specific request."
        )
        if updated:
            answer = "I couldn't find a matching recipe under the current exclusions. I have not relaxed them. " + answer
        return {"content": answer + banner, "recipe_ids": [], "active_restrictions": updated,
                "last_search_query": last_search_query}

    preview = [{
        "recipe_id": hit.recipe.recipe_id,
        "title": hit.recipe.title,
        "matched_terms": hit.matched_terms,
        "ingredients": hit.recipe.ingredients[:12],
        "instructions": hit.recipe.instructions[:3],
        "semantic_score": round(hit.semantic_score, 4) if hit.semantic_score is not None else None,
    } for hit in hits[:5]]
    fallback = (
        f"I found {len(hits)} relevant recipes in the loaded dataset. Open a result below to inspect its original "
        f"ingredients and instructions.{banner}"
    )
    evidence: dict[str, Any] = {"active_exclusions": updated, "retrieved_recipes": preview}
    if attachment_context:
        evidence["uploaded_documents"] = attachment_context
    answer = groq.grounded_explanation(
        message, evidence, fallback,
        history=conversation_history,
        memory=memory_note,
    )
    if not has_complete_dataset(st.session_state.get("loaded_dataset_filenames", [])):
        answer += "\n\n**Dataset coverage:** Results are limited to the JSON exports currently loaded."
    if banner and banner not in answer:
        answer += banner
    if added:
        answer += "\n\nI'll keep these exclusions active for follow-up searches in this chat: " + ", ".join(added) + "."
    return {"content": answer, "recipe_ids": [hit.recipe.recipe_id for hit in hits],
            "active_restrictions": updated, "last_search_query": last_search_query}


def add_chat_message(session: dict[str, Any], role: str, content: str,
                     recipe_ids: list[str] | None = None,
                     attachments: list[str] | None = None) -> None:
    session["messages"].append({
        "role": role,
        "content": content,
        "recipe_ids": recipe_ids or [],
        "attachments": attachments or [],
        "timestamp": now_iso(),
    })
    session["messages"] = session["messages"][-100:]
    if role == "user" and session.get("title") == "New chat":
        compact = re.sub(r"\s+", " ", content).strip()
        session["title"] = (compact[:42] + "…") if len(compact) > 43 else (compact or "New chat")


def append_tool_result(session: dict[str, Any], content: str,
                       recipe_ids: list[str] | None = None) -> None:
    add_chat_message(session, "assistant", content, recipe_ids=recipe_ids)
    session["active_tool"] = ""
    touch_session(session)
    st.rerun()


def render_messages(session: dict[str, Any], recipes_by_id: dict[str, Recipe]) -> None:
    messages = session.get("messages", [])
    for message in messages:
        role = message.get("role", "assistant")
        with st.chat_message(role, avatar="🧑" if role == "user" else "🍲"):
            st.markdown(message.get("content", ""))
            attachments = message.get("attachments", [])
            if attachments:
                st.caption("Attached for this turn: " + " · ".join(attachments))
            for recipe_id in message.get("recipe_ids", [])[:8]:
                recipe = recipes_by_id.get(recipe_id)
                if recipe:
                    render_recipe_card(recipe)


def render_pantry_tool(recipes: list[Recipe], session: dict[str, Any]) -> None:
    st.markdown("### Ingredient matcher")
    st.caption("Find existing recipes using ingredients you have. Hard exclusions and optional step limits are applied before ranking.")
    with st.form("pantry_matcher_form"):
        available_text = st.text_area("Ingredients you have", placeholder="eggs, onions, tomatoes, rice", height=90)
        excluded_text = st.text_input("Hard exclusions", placeholder="peanuts, dairy")
        use_step_limit = st.checkbox("Limit instruction steps")
        max_steps = st.number_input("Maximum steps", min_value=1, max_value=100, value=6, disabled=not use_step_limit)
        prefer_fewer = st.checkbox("Prefer fewer additional ingredients", value=True)
        submitted = st.form_submit_button("Find matching recipes", type="primary", use_container_width=True)
    if submitted:
        available = parse_available_ingredients(available_text)
        excluded = parse_comma_list(excluded_text)
        results = match_recipes(
            recipes, available, excluded,
            max_steps=int(max_steps) if use_step_limit else None,
            limit=5, prefer_fewer_missing=prefer_fewer,
        )
        if not available:
            st.warning("Enter at least one ingredient name.")
        elif not results:
            append_tool_result(session, "No eligible recipe matched those pantry ingredients and exclusions. No hard constraint was relaxed.")
        else:
            lines = ["### Pantry match results\nExisting recipes ranked against your available ingredients:"]
            ids = []
            for i, item in enumerate(results, 1):
                ids.append(item.recipe.recipe_id)
                lines.append(
                    f"\n**{i}. {item.recipe.title}** (`{item.recipe.recipe_id}`)  \n"
                    f"Matched: {', '.join(item.matched) or 'none'}  \n"
                    f"Additional required: {', '.join(item.missing) or 'none'}  \n"
                    f"Instruction steps: {item.steps} · Score: {item.score:.1f}"
                )
            lines.append("\nQuantities are not verified; this is a name-level match.")
            append_tool_result(session, "\n".join(lines), ids)


def render_substitution_tool(recipes: list[Recipe], recipes_by_id: dict[str, Recipe],
                             groq: GroqService, session: dict[str, Any]) -> None:
    st.markdown("### Evidence-based substitution lab")
    st.caption("Find a recipe, choose the ingredient to replace, and inspect related local recipes before generating an adapted draft.")
    if st.button("← Back to chat", key="substitution_back"):
        session["active_tool"] = ""
        touch_session(session)
        st.rerun()

    state = st.session_state.get("substitution_tool_state", {})
    with st.form("substitution_find_recipe_form"):
        recipe_query = st.text_input("Find an existing recipe", value=state.get("query", ""), placeholder="banana bread")
        find_recipe = st.form_submit_button("Search recipes", type="primary")
    if find_recipe:
        hits = search_recipes(recipes, recipe_query, limit=10,
                              vector_index=st.session_state.get("vector_index"),
                              recipe_lookup=recipes_by_id)
        state = {"query": recipe_query, "recipe_ids": [h.recipe.recipe_id for h in hits],
                 "evidence": None, "output": None}
        st.session_state.substitution_tool_state = state
        if not hits:
            st.warning("No recipe was found. Try a different title or ingredient.")
        else:
            st.rerun()

    hits = [recipes_by_id[rid] for rid in state.get("recipe_ids", []) if rid in recipes_by_id]
    if hits:
        with st.form("substitution_evidence_form"):
            selected_recipe_id = st.selectbox(
                "Choose the exact recipe", [r.recipe_id for r in hits],
                format_func=lambda rid: f"{recipes_by_id[rid].title} · {rid}",
            )
            selected_recipe = recipes_by_id[selected_recipe_id]
            ingredient_lines = [line for line in selected_recipe.ingredients if ingredient_key(line)]
            if ingredient_lines:
                target_line = st.selectbox("Ingredient to replace", ingredient_lines)
            else:
                target_line = ""
                st.info("This recipe has no usable ingredient list.")
            restrictions_text = st.text_input("Hard restrictions", placeholder="dairy, peanuts")
            evidence_submit = st.form_submit_button("Find local evidence", type="primary", disabled=not bool(target_line))
        if evidence_submit:
            restrictions = parse_comma_list(restrictions_text)
            candidates, peers = find_substitution_evidence(
                recipes, selected_recipe, target_line, restrictions=restrictions, limit=10
            )
            candidates = [c for c in candidates if not ingredient_violations(c.ingredient, restrictions)]
            st.session_state.substitution_tool_state = {
                **state,
                "selected_recipe_id": selected_recipe_id,
                "target_line": target_line,
                "restrictions": restrictions,
                "candidates": candidates,
                "peers": peers,
                "output": None,
            }
            st.rerun()

    state = st.session_state.get("substitution_tool_state", {})
    if state.get("candidates"):
        selected_recipe = recipes_by_id.get(state.get("selected_recipe_id", ""))
        if not selected_recipe:
            return
        st.markdown(f"**Replacing:** {state['target_line']} in {selected_recipe.title}")
        st.caption(f"{len(state.get('peers', []))} related recipes were inspected. Evidence is indirect, not a proof of equivalence.")
        candidate_labels = [
            f"{candidate.ingredient} · found in {candidate.support_count} related recipe(s)"
            for candidate in state["candidates"]
        ]
        chosen_label = st.selectbox("Dataset-supported candidate", candidate_labels, key="substitution_candidate_choice")
        chosen_candidate = state["candidates"][candidate_labels.index(chosen_label)]
        with st.expander("Supporting recipe evidence", expanded=True):
            for title, recipe_id in zip(chosen_candidate.titles, chosen_candidate.recipe_ids):
                st.markdown(f"- {title} (`{recipe_id}`)")
            st.caption("The dataset does not explicitly label these ingredients as substitutes. Treat this as indirect evidence.")
        if not groq.available:
            st.info("Configure GROQ_API_KEY in Streamlit secrets or the server environment to generate revised instructions.")
        if st.button("Generate adapted recipe", type="primary", disabled=not groq.available, key="generate_substitution"):
            evidence = [
                {"recipe_id": rid, "title": title, "candidate_ingredient": chosen_candidate.ingredient,
                 "shared_ingredients": chosen_candidate.shared_ingredients}
                for rid, title in zip(chosen_candidate.recipe_ids, chosen_candidate.titles)
            ]
            try:
                with st.status("Preparing an evidence-based adaptation…", expanded=True) as status:
                    st.write("Using the original recipe and locally retrieved supporting records.")
                    adapted = groq.adapt_recipe(
                        {"title": selected_recipe.title, "recipe_id": selected_recipe.recipe_id,
                         "ingredients": selected_recipe.ingredients, "instructions": selected_recipe.instructions},
                        state["target_line"], chosen_candidate.ingredient,
                        state.get("restrictions", []), evidence,
                    )
                    status.update(label="Adaptation draft ready", state="complete", expanded=False)
                updated_ingredients = build_updated_ingredient_list(
                    selected_recipe, state["target_line"], chosen_candidate.ingredient
                )
                output = (
                    f"### Adapted recipe draft: {selected_recipe.title}\n\n"
                    "**Updated ingredient list**\n" + "\n".join(f"- {item}" for item in updated_ingredients)
                    + "\n\n**Revised instructions**\n" + adapted
                    + "\n\n_Note: This adaptation is based on indirect recipe evidence. Taste/texture comments are predictions, quantities need testing, and this is not an allergen-safety guarantee._"
                )
                append_tool_result(session, output, [selected_recipe.recipe_id, *chosen_candidate.recipe_ids])
            except Exception:
                st.error("The Groq request failed. Check provider availability and server-side configuration; the API key was not displayed.")
    elif state.get("selected_recipe_id"):
        st.warning("No sufficiently relevant local candidate was found under these restrictions. The system will not invent a dataset-supported substitute.")


# Initialise local chat history once per browser session.
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = load_chat_sessions()
if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = max(
        st.session_state.chat_sessions.values(), key=lambda x: x.get("updated_at", "")
    )["id"]
if "active_tool" not in st.session_state:
    st.session_state.active_tool = ""
if "upload_nonce" not in st.session_state:
    st.session_state.upload_nonce = 0

# Load server-side provider configuration.
api_key = get_secret_key()
model_options = [DEFAULT_MODEL, "openai/gpt-oss-20b", "qwen/qwen3.8-27b"]
if "selected_groq_model" not in st.session_state:
    st.session_state.selected_groq_model = DEFAULT_MODEL
model = st.session_state.selected_groq_model
groq = GroqService(api_key or None, model=model)

# Load the Recipe Box collection and its persistent semantic index.
DATA_DIR.mkdir(parents=True, exist_ok=True)
try:
    current_file_signature = file_signature()
    recipes, loaded_filenames = cached_dataset(current_file_signature)
    data_error = None
except Exception:
    current_file_signature, recipes, loaded_filenames = (), [], []
    data_error = "The local recipe cache could not be loaded. Check the JSON exports and restart the app."
recipes_by_id = cached_recipe_lookup(current_file_signature) if recipes else {}
distinct_ingredient_count = cached_distinct_ingredient_count(current_file_signature) if recipes else 0
full_dataset_loaded = has_complete_dataset(loaded_filenames)
current_fingerprint = dataset_fingerprint(DATA_DIR, loaded_filenames) if loaded_filenames else ""
try:
    vector_index = cached_vector_index(current_fingerprint) if current_fingerprint else None
except Exception:
    vector_index = None
st.session_state.vector_index = vector_index
st.session_state.loaded_dataset_filenames = loaded_filenames


# ------------------------------ Sidebar -----------------------------------
with st.sidebar:
    st.markdown(
        '<div class="brand-row"><div class="brand-mark">🍲</div><div><div class="brand-name">Recipe Box</div><div class="brand-sub">Your local cooking assistant</div></div></div>',
        unsafe_allow_html=True,
    )
    if st.button("＋  New chat", type="primary", use_container_width=True, key="new_chat_button"):
        session = new_chat_session()
        st.session_state.chat_sessions[session["id"]] = session
        st.session_state.active_chat_id = session["id"]
        st.session_state.active_tool = ""
        touch_session(session)
        st.rerun()

    st.markdown('<div class="sidebar-label">Your chats</div>', unsafe_allow_html=True)
    history_filter = st.text_input("Search chat history", placeholder="Search conversations", label_visibility="collapsed", key="history_filter")
    current_session = get_active_session()
    sorted_sessions = sorted(
        st.session_state.chat_sessions.values(),
        key=lambda x: x.get("updated_at", ""), reverse=True,
    )
    visible_sessions = [
        item for item in sorted_sessions
        if not history_filter.strip() or history_filter.casefold() in item.get("title", "").casefold()
    ]
    for item in visible_sessions[:30]:
        row_a, row_b = st.columns([0.84, 0.16], gap="small")
        is_current = item["id"] == current_session["id"]
        label = item.get("title", "New chat")
        if row_a.button(
            ("● " if is_current else "　") + (label[:30] + "…" if len(label) > 31 else label),
            key=f"open_chat_{item['id']}",
            type="primary" if is_current else "secondary",
            use_container_width=True,
            help=label,
        ):
            st.session_state.active_chat_id = item["id"]
            st.session_state.active_tool = ""
            st.rerun()
        if row_b.button("×", key=f"delete_chat_{item['id']}", help="Delete this conversation"):
            st.session_state.chat_sessions.pop(item["id"], None)
            if st.session_state.active_chat_id == item["id"]:
                replacement = new_chat_session()
                st.session_state.chat_sessions[replacement["id"]] = replacement
                st.session_state.active_chat_id = replacement["id"]
            save_chat_sessions()
            st.rerun()

    with st.expander("Manage current chat", expanded=False):
        rename_value = st.text_input("Conversation name", value=current_session.get("title", "New chat"), key=f"rename_{current_session['id']}")
        if st.button("Save name", key=f"save_rename_{current_session['id']}", use_container_width=True):
            current_session["title"] = rename_value.strip()[:90] or "New chat"
            touch_session(current_session)
            st.rerun()
        if st.button("Clear messages", key=f"clear_messages_{current_session['id']}", use_container_width=True):
            current_session["messages"] = []
            current_session["last_search_query"] = ""
            current_session["active_restrictions"] = []
            current_session["title"] = "New chat"
            touch_session(current_session)
            st.rerun()
        if st.button("Delete current chat", key=f"delete_current_{current_session['id']}", use_container_width=True):
            st.session_state.chat_sessions.pop(current_session["id"], None)
            replacement = new_chat_session()
            st.session_state.chat_sessions[replacement["id"]] = replacement
            st.session_state.active_chat_id = replacement["id"]
            save_chat_sessions()
            st.rerun()
        export_payload = json.dumps(current_session, ensure_ascii=False, indent=2)
        st.download_button(
            "Export this conversation", data=export_payload,
            file_name=f"recipe-chat-{current_session['id'][:8]}.json",
            mime="application/json", use_container_width=True,
        )

    with st.expander("Chat memory & preferences", expanded=False):
        st.caption("Memory is stored locally on this computer and applies only to this conversation.")
        restriction_value = ", ".join(current_session.get("active_restrictions", []))
        restriction_version = int(current_session.get("restriction_editor_version", 0))
        memory_version = int(current_session.get("memory_editor_version", 0))
        restriction_text = st.text_input(
            "Active exclusions", value=restriction_value,
            key=f"restriction_editor_{current_session['id']}_{restriction_version}",
            help="Comma-separated. These exclusions are applied to future recipe searches in this chat.",
        )
        memory_text = st.text_area(
            "Remember for this chat", value=current_session.get("memory_note", ""),
            key=f"memory_editor_{current_session['id']}_{memory_version}", height=100,
            placeholder="Example: Prefer vegetarian dinners; avoid very spicy dishes.",
        )
        if st.button("Save memory & exclusions", key=f"save_memory_{current_session['id']}", use_container_width=True):
            current_session["active_restrictions"] = parse_comma_list(restriction_text)
            current_session["memory_note"] = memory_text.strip()[:4000]
            touch_session(current_session)
            st.success("Chat memory saved locally.")
        memory_col_a, memory_col_b = st.columns(2)
        if memory_col_a.button("Clear memory", key=f"clear_memory_{current_session['id']}", use_container_width=True):
            current_session["memory_note"] = ""
            current_session["memory_editor_version"] = memory_version + 1
            touch_session(current_session)
            st.rerun()
        if memory_col_b.button("Reset context", key=f"reset_context_{current_session['id']}", use_container_width=True):
            current_session["active_restrictions"] = []
            current_session["last_search_query"] = ""
            current_session["restriction_editor_version"] = restriction_version + 1
            touch_session(current_session)
            st.rerun()

    st.markdown('<div class="sidebar-label">Recipe tools</div>', unsafe_allow_html=True)
    if st.button("🥕  Ingredient matcher", use_container_width=True, key="open_matcher"):
        st.session_state.active_tool = "matcher"
        st.rerun()
    if st.button("🔁  Substitution lab", use_container_width=True, key="open_substitution"):
        st.session_state.active_tool = "substitution"
        st.session_state.substitution_tool_state = {}
        st.rerun()

    st.markdown('<div class="sidebar-label">Recipe collection</div>', unsafe_allow_html=True)
    existing_files = list(DATA_DIR.glob("*.json"))
    st.caption(f"{len(loaded_filenames)} JSON export(s) loaded")
    if full_dataset_loaded:
        st.success(f"Full dataset · {len(recipes):,} recipes")
    elif recipes:
        st.warning(f"Partial dataset · {len(recipes):,} recipes")
    else:
        st.info("Recipe dataset not loaded")

    with st.form("dataset_upload_form", clear_on_submit=True):
        dataset_uploads = st.file_uploader(
            "Import dataset JSON files", type=["json"], accept_multiple_files=True,
            help="Select the three Recipe Box JSON exports. Files are saved in this app's local data folder.",
        )
        import_dataset = st.form_submit_button("Import selected files", use_container_width=True)
    if import_dataset and dataset_uploads:
        saved = 0
        for upload in dataset_uploads:
            safe_name = Path(upload.name).name
            if safe_name.lower().endswith(".json"):
                (DATA_DIR / safe_name).write_bytes(upload.getvalue())
                saved += 1
        if saved:
            st.success(f"Imported {saved} JSON file(s). Reloading the local cache…")
            st.cache_resource.clear()
            st.rerun()

    if not full_dataset_loaded:
        if st.button("Download official dataset", use_container_width=True, key="download_dataset"):
            with st.status("Downloading and extracting the Recipe Box dataset…", expanded=True) as status:
                progress = st.progress(0.0, text="Starting download…")
                def _download_progress(done: int, total: int) -> None:
                    ratio = min(done / max(total, 1), 1.0)
                    progress.progress(ratio, text=f"Downloaded {done / (1024 * 1024):,.1f} MB")
                try:
                    paths = download_and_extract(DATA_DIR, _download_progress)
                    status.update(label=f"Extracted {len(paths)} dataset export(s)", state="complete", expanded=False)
                    st.cache_resource.clear()
                    st.rerun()
                except Exception:
                    status.update(label="Dataset download failed", state="error", expanded=True)
                    st.error("Check the internet connection, free disk space, or use the JSON upload control above.")

    if recipes:
        if vector_index:
            st.success(f"Semantic index ready · {len(vector_index.recipe_ids):,} vectors")
            st.caption(f"{vector_index.model_name} · {vector_index.info.dimension} dimensions · local FAISS search")
        else:
            st.warning("Semantic embeddings not built yet. Keyword search is available as a fallback.")
            if st.button("Build semantic embeddings", use_container_width=True, key="build_embeddings"):
                with st.status("Building local recipe embeddings…", expanded=True) as status:
                    progress = st.progress(0.0, text="Loading embedding model…")
                    def _embedding_progress(done: int, total: int) -> None:
                        progress.progress(done / max(total, 1), text=f"Embedded {done:,} / {total:,} recipes")
                    try:
                        result = build_vector_index(
                            recipes, DATA_DIR / "vector_index", current_fingerprint,
                            model_name=DEFAULT_EMBEDDING_MODEL,
                            batch_size=32, chunk_size=512,
                            progress_callback=_embedding_progress,
                        )
                        status.update(label=f"Semantic index ready · {result.count:,} recipes", state="complete", expanded=False)
                        st.cache_resource.clear()
                        st.rerun()
                    except Exception:
                        status.update(label="Embedding build failed", state="error", expanded=True)
                        st.error("Embedding generation failed. Check dependencies, memory and terminal logs. No Groq key is used for embeddings.")

    st.markdown('<div class="sidebar-label">AI provider</div>', unsafe_allow_html=True)
    selected_model = st.selectbox("Groq model", model_options, index=model_options.index(model), key="groq_model_select")
    if selected_model != st.session_state.selected_groq_model:
        st.session_state.selected_groq_model = selected_model
        st.rerun()
    groq = GroqService(api_key or None, model=st.session_state.selected_groq_model)
    if groq.available:
        st.markdown('<div class="tiny-note"><span class="status-dot"></span>Groq connected · key loaded server-side</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="tiny-note"><span class="status-dot-muted"></span>Groq not configured · local retrieval still works</div>', unsafe_allow_html=True)
    st.markdown('<div class="tiny-note">Chat history and memory are saved locally. Uploaded chat files are used for the current turn only and are not stored by the app.</div>', unsafe_allow_html=True)

# ------------------------------ Main chat ---------------------------------
session = get_active_session()
if data_error:
    st.error(data_error)

if st.session_state.active_tool == "matcher":
    st.markdown('<div class="eyebrow">Recipe tools</div>', unsafe_allow_html=True)
    if st.button("← Back to chat", key="matcher_back"):
        st.session_state.active_tool = ""
        st.rerun()
    if recipes:
        render_pantry_tool(recipes, session)
    else:
        st.info("Load the Recipe Box JSON exports first.")
elif st.session_state.active_tool == "substitution":
    st.markdown('<div class="eyebrow">Recipe tools</div>', unsafe_allow_html=True)
    if recipes:
        render_substitution_tool(recipes, recipes_by_id, groq, session)
    else:
        st.info("Load the Recipe Box JSON exports first.")
else:
    st.markdown('<div class="eyebrow">Recipe Box · Local-first recipe intelligence</div>', unsafe_allow_html=True)
    st.markdown('<div class="welcome-title">What would you like to cook?</div>', unsafe_allow_html=True)
    st.markdown('<div class="welcome-copy">Ask about recipes, ingredients, substitutions or the collection. Answers are grounded in your local Recipe Box data.</div>', unsafe_allow_html=True)

    status_items = []
    if vector_index:
        status_items.append('<span class="status-dot"></span>Semantic search ready')
    elif recipes:
        status_items.append('<span class="status-dot-muted"></span>Keyword fallback')
    else:
        status_items.append('<span class="status-dot-muted"></span>Dataset not loaded')
    status_items.append('<span class="status-dot"></span>Groq connected' if groq.available else '<span class="status-dot-muted"></span>Groq not configured')
    st.markdown('<div class="status-line">' + ' &nbsp;&nbsp; · &nbsp;&nbsp; '.join(status_items) + '</div>', unsafe_allow_html=True)

    if recipes:
        with st.expander("Collection details", expanded=False):
            metric_a, metric_b, metric_c = st.columns(3)
            metric_a.metric("Recipes loaded", f"{len(recipes):,}")
            metric_b.metric("JSON exports", len(loaded_filenames))
            metric_c.metric("Distinct ingredients", f"{distinct_ingredient_count:,}")
            st.caption("Loaded files: " + " · ".join(loaded_filenames))
            if not full_dataset_loaded:
                st.warning("Only the currently loaded exports are included in search and statistics.")

    if not session.get("messages"):
        st.markdown("#### Try asking")
        starters = [
            "Find a penne pasta recipe",
            "I have eggs, onions, tomatoes and rice. Suggest recipes.",
            "How many recipes contain both garlic and onion?",
            "Which tomato recipes have the fewest instruction steps?",
        ]
        starter_cols = st.columns(2)
        for i, starter in enumerate(starters):
            if starter_cols[i % 2].button(starter, key=f"starter_{i}", use_container_width=True):
                st.session_state[f"composer_text_{session['id']}"] = starter
                st.rerun()
        if not recipes:
            st.info("Add the three Recipe Box JSON exports in the sidebar to enable local recipe search and statistics.")
    else:
        render_messages(session, recipes_by_id)

    if session.get("active_restrictions"):
        st.caption("Active exclusions for this chat: " + ", ".join(session["active_restrictions"]))

    st.divider()
    # Files are attached to the next submitted message only. No raw bytes or extracted text are persisted.
    upload_key = f"chat_upload_{session['id']}_{st.session_state.upload_nonce}"
    attach_col, attach_hint_col = st.columns([0.25, 0.75])
    with attach_col:
        with st.popover("＋ Attach files"):
            uploaded_files = st.file_uploader(
                "Choose files",
                type=["txt", "md", "pdf", "json", "csv"],
                accept_multiple_files=True,
                key=upload_key,
                help="PDF, TXT, Markdown, JSON and CSV. Maximum 8 MB per file.",
            )
    with attach_hint_col:
        if uploaded_files:
            st.caption("Attached: " + " · ".join(Path(file.name).name for file in uploaded_files))
        else:
            st.caption("Attach PDF, text, Markdown, JSON or CSV files to this message.")

    with st.form(f"chat_composer_{session['id']}", clear_on_submit=True):
        prompt_col, send_col = st.columns([0.86, 0.14])
        with prompt_col:
            prompt = st.text_area(
                "Message",
                key=f"composer_text_{session['id']}",
                height=76,
                placeholder="Message Recipe Box Assistant…",
                label_visibility="collapsed",
            )
        with send_col:
            submitted = st.form_submit_button("↑  Send", type="primary", use_container_width=True)

    if submitted and prompt.strip():
        message_text = prompt.strip()
        attachment_context, attachment_notes = extract_upload_text(uploaded_files or [])
        attachment_names = [Path(file.name).name for file in uploaded_files or []]
        prior_history = [
            {"role": msg["role"], "content": msg.get("content", "")}
            for msg in session.get("messages", [])[-8:]
            if msg.get("role") in {"user", "assistant"}
        ]
        add_chat_message(session, "user", message_text, attachments=attachment_names)
        if attachment_notes:
            message_text_for_processing = message_text + "\n\nAttachment processing notes: " + " ".join(attachment_notes)
        else:
            message_text_for_processing = message_text
        try:
            with st.status("Working on your request…", expanded=True) as status:
                st.write("Applying chat memory and active ingredient exclusions…")
                st.write("Searching the local recipe collection and checking source records…")
                response = process_chat_message(
                    message_text_for_processing,
                    recipes,
                    groq,
                    active_restrictions=list(session.get("active_restrictions", [])),
                    last_search_query=session.get("last_search_query", ""),
                    vector_index=vector_index,
                    recipe_lookup=recipes_by_id,
                    conversation_history=prior_history,
                    memory_note=session.get("memory_note", ""),
                    attachment_context=attachment_context,
                )
                if attachment_context and not groq.available:
                    response["content"] += (
                        "\n\n**Attachment privacy note:** I extracted the file text locally, but Groq is not configured, "
                        "so the attachment content was not sent to an AI provider and could not be analyzed."
                    )
                if attachment_notes:
                    response["content"] += "\n\n**Attachment notes:** " + " ".join(attachment_notes)
                status.update(label="Response ready", state="complete", expanded=False)
            session["active_restrictions"] = response.get("active_restrictions", session.get("active_restrictions", []))
            session["last_search_query"] = response.get("last_search_query", session.get("last_search_query", ""))
            add_chat_message(session, "assistant", response["content"], recipe_ids=response.get("recipe_ids", []))
            touch_session(session)
        except Exception:
            add_chat_message(
                session, "assistant",
                "Something went wrong while processing this request. Check the local dataset, embedding index and provider availability. The API key was not displayed or written to the chat.",
            )
            touch_session(session)
        st.session_state.upload_nonce += 1
        st.rerun()

    st.markdown(
        '<div class="tiny-note" style="text-align:center;margin-top:.55rem">Recipe facts and statistics come from the loaded dataset. Local embeddings power retrieval; Groq generates explanations. Uploaded files are not retained.</div>',
        unsafe_allow_html=True,
    )
