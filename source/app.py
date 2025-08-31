# -*- coding: utf-8 -*-
"""
App: MOOC exercise browser (front page)

What this page does (for users):
- Loads a cached index of all MOOC exercises (fast), with a manual "refresh" toggle.
- Lets you filter by course part ("Osa N") and free-text search.
- Clicking an exercise opens the detailed page with the task and AI tools.

Notes for developers:
- Index is cached on disk via utils/cache.py to avoid re-scraping on every run.
- Navigation: when a user clicks a button, we set query params and switch to pages/exercise.py.
- Keep CSS minimal and scoped; we purposely remove Streamlit's white markdown background.
"""

import json
import streamlit as st
from utils.cache import get_or_build_json, index_path

# Scraping helpers
from scraping.mooc_scraper import index_all_exercises, fetch_exercise_by_href

# ---------------------------------------------------------------------
# Page config: do this before any other Streamlit call to avoid warnings
# ---------------------------------------------------------------------
st.set_page_config(page_title="Tehtävälista", page_icon="📚", layout="wide")


# Sidebar: optional manual refresh + optional auto-expire
force_index = st.sidebar.toggle("Force re-scrape index", value=False, help="Ignore local index.json")
auto_expire_days = None

# ---------------------------------------------------------------------
# # Load index (offline-first)
# ---------------------------------------------------------------------
with st.spinner("Ladataan tehtävälista…"):
    idx = get_or_build_json(
        index_path(),
        builder=lambda: index_all_exercises(headless=True),
        force=force_index,
        max_age_days=auto_expire_days,
    )

items = idx.get("items", [])
st.success(f"Tehtäviä löydetty: {len(items)}kpl — lähde: {'välimuisti' if not force_index else 'uusi haku'}.")

# ---------------------------------------------------------------------
# Minimal CSS
# - We override Streamlit's markdown container background to be transparent,
#   which removes the "annoying white board".
# - `.card` is intentionally made transparent (no border/shadow) so that
#   <div class="card">Tehtävät...</div> shows text only.
# - Keep styles scoped to stable testids/classes to reduce breakage across
#   Streamlit versions.
# ---------------------------------------------------------------------
st.markdown(
    """
<style>
/* --- Sidebar sizing --- */
section[data-testid="stSidebar"] { min-width: 280px; width: 280px; }

/* --- Sidebar typography & items --- */
.sidebar-title {
  font-weight: 800; font-size: 16px; letter-spacing: .3px; text-transform: uppercase;
  color: #d13b3b; margin-bottom: .5rem;
}
.sb-group { font-weight: 700; margin: 12px 0 6px; color: #444; }
.sb-item { display:block; padding: 6px 10px; border-radius: 6px; margin: 2px 0; color:#222; text-decoration:none; }
.sb-item:hover { background: #f2f2f7; }
.sb-item.active { background: #e9eefc; color:#1d3bb3; font-weight: 700; }

/* --- Remove Streamlit's white background around Markdown blocks --- */
div[data-testid="stMarkdownContainer"] {
  background: transparent !important;   /* kills the white board */
  padding: 0 !important;
  box-shadow: none !important;
  border: none !important;
}

/* --- "Card" look is disabled on purpose (keep text only) --- */
.card {
  background: transparent !important;   /* was #fff */
  border: none !important;              /* was 1px solid #e8e8ef */
  box-shadow: none !important;          /* was subtle shadow */
  padding: 0 !important;                /* tighten spacing */
  margin-bottom: 1rem;
}
.card h4 { margin: 0 0 .75rem 0; }

/* --- Titles & utilities --- */
.section-title { font-size: 42px; font-weight: 800; margin: 0 0 10px 0; }
.subtle { color: #666; }

/* --- Buttons in grid --- */
.btn-grid .stButton>button { width: 100%; text-align: left; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------
# Cached network fetch of the full exercise index.
# st.cache_data is good if site changes frequently:
# -return the cached object from memory (within a session)
# -reload from disk if it’s a new session
# -rebuild by scraping if the cache file isn’t present
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_index():
    data = index_all_exercises(headless=True)
    # Expected schema: data["items"] -> list[dict]
    # keys: url, section_number, exercise_id, exercise_type
    return data["items"]

# ---------------------------------------------------------------------
# Session state initialization
# - Store selected sidebar part ("Osa N") and the clicked exercise href.
# ---------------------------------------------------------------------
if "selected_href" not in st.session_state:
    st.session_state.selected_href = None
if "selected_part" not in st.session_state:
    st.session_state.selected_part = "Osa 1"

# ---------------------------------------------------------------------
# Sidebar: two lists of parts (basic + advanced)
# - Buttons set selected_part and also clear any previously selected_href.
# - Using use_container_width=True for nicer look in sidebar.
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="sidebar-title">Ohjelmoinnin perusteet ja jatkokurssi 2025</div>',
        unsafe_allow_html=True,
    )

    basics_parts = [f"Osa {i}" for i in range(1, 8)]
    adv_parts    = [f"Osa {i}" for i in range(8, 15)]

    def render_list(parts):
        for p in parts:
            if st.button(p, key=f"sb_{p}", use_container_width=True):
                st.session_state.selected_part = p
                st.session_state.selected_href = None  # reset details panel

    st.markdown('<div class="sb-group">Ohjelmoinnin perusteet</div>', unsafe_allow_html=True)
    render_list(basics_parts)

    st.markdown('<div class="sb-group" style="margin-top:14px;">Ohjelmoinnin jatkokurssi</div>', unsafe_allow_html=True)
    render_list(adv_parts)

# ---------------------------------------------------------------------
# Main header
# ---------------------------------------------------------------------
part_label = st.session_state.selected_part
st.markdown(f'<div class="section-title">{part_label}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------
# Load the (cached) full index once per run
# ---------------------------------------------------------------------
items = load_index()

# ---------------------------------------------------------------------
# Filtering model:
# - part_match: accepts either Finnish ('osa-N') or English ('part-N') mirror sections.
# - query_match: simple substring match across exercise_id and url (case-insensitive).
# ---------------------------------------------------------------------
part_num = int(part_label.split()[-1])

query = st.text_input(
    "Hae/rajaa:",
    "",
    placeholder="esim. programming-exercise, quiz, ehtorakenne ...",
    label_visibility="collapsed"
)

def part_match(it: dict) -> bool:
    sec = str(it.get("section_number") or "")
    return sec.startswith(f"osa-{part_num}/") or sec.startswith(f"part-{part_num}/")

def query_match(it: dict) -> bool:
    if not query.strip():
        return True
    needle = query.strip().lower()
    return (
        needle in str(it.get("exercise_id") or "").lower()
        or needle in str(it.get("url") or "").lower()
    )

filtered = [it for it in items if part_match(it) and query_match(it)]

# ---------------------------------------------------------------------
# UI navigation:
# - Each button stores the clicked exercise URL in session state,
#   writes it to query params, and switches to the details page.
# - Keys are stable across reruns to avoid duplicate-button state issues.
# ---------------------------------------------------------------------
st.markdown('<div class="card"><h4>Tehtävät tässä osassa:</h4></div>', unsafe_allow_html=True)

if not filtered:
    st.info("Ei tuloksia valitulla osiolla ja haulla.")
else:
    cols = st.columns(3)  # keep it simple; Streamlit auto-wraps on narrow screens
    for i, it in enumerate(filtered):
        icon = {"programming": "💻", "quiz": "❓", "written": "✍️"}.get(it.get("exercise_type"), "📄")
        eid  = str(it.get("exercise_id") or "unnamed")
        label = f"{icon}  {eid}"
        # Unique, deterministic key across reruns
        if cols[i % 3].button(label, key=f"btn_{eid}_{i}"):
            # select task
            st.session_state.selected_href = str(it.get("url") or "")
            # jump to clicked task
            st.query_params.update({"page": "exercise", "href": it.get("url", "")})
            st.switch_page("pages/exercise.py")

# ---------------------------------------------------------------------
# Footer (subtle note)
# ---------------------------------------------------------------------
st.markdown('<div class="subtle">Streamlit UI</div>', unsafe_allow_html=True)