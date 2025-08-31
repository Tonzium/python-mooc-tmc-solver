# pages/Exercise.py
# -*- coding: utf-8 -*-
"""
Exercise page: shows one MOOC task, plus AI tools (analyze + solve).

For users:
- The task content is loaded from a local JSON cache when available (fast).
- You can re-fetch the task from the web via the sidebar toggle.

For developers:
- We derive a stable exercise_id from the URL anchor and store its JSON under ./data/exercises.
- 'summarize_point()' generates a short "what is the point of this exercise?" info box.
- The Analyze tab hides DeepSeek <think> chains by default; a checkbox reveals them on demand.
"""

import json
import streamlit as st
from ai.agents import solve_auto, summarize_point, build_llm
from ai.analyzer import analyze, derive_rubric_from_reference
from utils.cache import get_or_build_json, exercise_path
import re

# Scraper + AI helpers
from scraping.mooc_scraper import fetch_exercise_by_href # scrapes all tasks

st.set_page_config(
    page_title="Harjoitus • Ohjelmoinnin kurssi",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

force_detail = st.sidebar.toggle(
    "Pakota tehtävien uudelleen haku",
    value=False,
    help="Ohita paikalliset tiedostot"
)

# Minimal CSS
st.markdown(
    """
<style>
section[data-testid="stSidebar"] { min-width: 280px; width: 280px; }
div[data-testid="stMarkdownContainer"] { background: transparent !important; padding: 0 !important; box-shadow: none !important; border: none !important; }
.section-title { font-size: 42px; font-weight: 800; margin: 0 0 10px 0; }
.card { background: transparent !important; border: none !important; box-shadow: none !important; padding: 0 !important; margin-bottom: 1rem; }
.card h4 { margin: 0 0 .75rem 0; }
.btn-grid .stButton>button { width: 100%; text-align: left; }
.subtle { color: #666; }
</style>
""",
    unsafe_allow_html=True,
)

def render_question(item: dict):
    ordered = item.get("ordered_blocks") or []
    prompt_md = item.get("prompt_markdown")
    question  = item.get("question") or ""
    py_blocks = item.get("python_blocks") or []

    # Question with ordered form
    if ordered:
        for blk in ordered:
            t = blk.get("type")
            c = blk.get("content", "")
            if not c:
                continue
            if t == "code":
                st.code(c, language="python")
            else:
                st.markdown(c)
        return

    # Fallbacks if scraper didn’t provide ordered blocks yet
    if prompt_md:
        st.write("**Kysymys:**")
        st.markdown(prompt_md)
        return

    st.write("**Kysymys:**")
    st.markdown(question)
    if py_blocks:
        st.write("**Python-lohkot:**")
        for i, b in enumerate(py_blocks, 1):
            st.markdown(f"*Lohko {i}:*")
            st.code(b, language="python")

# Summarize results
def summarize_results(results):
    if not results: 
        return {"summary": "No results."}
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    fails = [r.get("name") for r in results if not r.get("passed")]
    return {"total": total, "passed": passed, "failed": total - passed, "failed_names": fails}

# Read the selected exercise from query params or session
qp = st.query_params
href = (qp.get("href") or [None])[0] or st.session_state.get("selected_href")

# Back to list
col_back, col_title = st.columns([1, 7])
with col_back:
    if st.button("← Takaisin listaan"):
        st.query_params
        try:
            st.query_params.clear()
            st.switch_page("app.py")
        except Exception:
            st.session_state.selected_href = None
            st.rerun()

if not href:
    st.error("Harjoitusta ei valittu.")
    st.page_link("app.py", label="Siirry tehtävälistaan")
    st.stop()

# derive a stable exercise_id from the URL anchor before scraping
if "#" in href:
    ex_id_from_href = href.split("#", 1)[1]
else:
    ex_id_from_href = "unknown-exercise"

json_file = exercise_path(ex_id_from_href)

# Offline-first: read the exercise JSON from disk; if missing or forced, scrape and cache.
with st.spinner("Ladataan tehtävä (välimuistista tai haetaan verkosta)…"):
    item = get_or_build_json(
        json_file,
        builder=lambda: fetch_exercise_by_href(href, headless=True),
        force=force_detail,      # set True from sidebar to refresh
        max_age_days=None        # or e.g. 3 to auto-expire in 3 days
    )

with col_title:
    st.markdown(f'<div class="section-title">{item.get("title") or "(otsikko puuttuu)"}</div>', unsafe_allow_html=True)

st.markdown(
    f"<div style='font-size:14px; font-weight:bold'>URL: "
    f"<a href='{item.get('url')}' style='font-size:18px' target='_blank'>{item.get('url')}</a>"
    f"</div>",
    unsafe_allow_html=True
)

st.markdown("#### Tehtävän anto:")

# Exercise details
render_question(item)

# When loading an exercise:
# rubrics store and retrievval
ex_id = item.get("id") or item.get("url")  # stable key
if ex_id and "rubrics" not in st.session_state:
    st.session_state["rubrics"] = {}

cached_rubric = None
if ex_id:
    cached_rubric = st.session_state["rubrics"].get(ex_id)

# I tools in this page’s sidebar
with st.sidebar:
    st.subheader("Kielimalli asetukset")
    # Keep last used value; fall back to a sensible default
    default_model = st.session_state.get("ollama_model", "deepseek-r1:latest")
    ollama_model = st.text_input(
        "Paikallinen Ollama kielimalli:",
        value=default_model,
        placeholder="e.g. llama3.2:latest, mistral:7b, deepseek-r1:32b",
        help="Kirjoita paikallisen Ollama mallisi nimi tähän. Esimerkiksi: deepseek-r1:latest",
    ).strip() or default_model
    st.session_state["ollama_model"] = ollama_model

# Main point as summarized text
key_point = f"point_{item.get('exercise_id')}"
point_text = st.session_state.get(key_point)

# Learning goal (short, student-facing):
# - Natural language model generates summary for current exercise in session_state; "Päivitä" regenerates it.
cols = st.columns([6, 1])
with cols[0]:
    st.markdown("#### Mitä tehtävässä pyritään oppimaan?")
    if not point_text:
        with st.spinner("Muodostetaan lyhyt tavoitekuvaus…"):
            point_text = summarize_point(
                item.get("question", "") or "",
                item.get("python_blocks") or [],
                model=ollama_model,
            )
            st.session_state[key_point] = point_text
    st.info(point_text or "—")

with cols[1]:
    if st.button("Päivitä", help="Luo tavoitekuvaus uudelleen"):
        with st.spinner("Päivitetään tavoitekuvaus…"):
            point_text = summarize_point(
                item.get("question", "") or "",
                item.get("python_blocks") or [],
                model=ollama_model,
            )
            st.session_state[key_point] = point_text
        st.rerun()

# Download JSON for this single exercise
one = json.dumps({"count": 1, "items": [item]}, ensure_ascii=False, indent=2)


# Analyze / Solve
tab_analyze, tab_solve = st.tabs(["Analysoi", "Ratkaise"])

# Analyze Student Work
# Analysis workflow:
# - Compact rubric mode: we prefer a short rubric over pasting a full reference solution.
# - DeepSeek <think> content is sanitized by default; the expander lets advanced users reveal it.

with tab_analyze:
    st.markdown("**Lisää koodisi analyysiä varten**")
    student_code = st.text_area("Koodisi", height=220, key="student_code")

    # Resolve the task + reference solution safely
    question = (item or {}).get("question", "")
    # Try to reuse cached AI solution as a private reference (never shown to user)
    reference_solution = st.session_state.get("ai_solution")

    test_results = st.session_state.get("results")

    # A FORM keeps the button state tidy and avoids double clicks
    with st.form("analyze_form"):
        submitted = st.form_submit_button("Analysoi", use_container_width=True)

    if submitted:
        if not student_code.strip():
            st.warning("Lisää koodi ensin ja yritä uudelleen.")
            st.stop()

        with st.spinner("Analysoidaan koodia..."):
            try:
                hints = analyze(
                    student_code=student_code,
                    question=item.get("question",""),
                    reference_solution=st.session_state.get("ai_solution"),
                    test_results=st.session_state.get("results"),
                    model=ollama_model,
                    build_llm=build_llm,
                    timeout_s=35,
                    use_compact_mode=True,
                    few_shots=None,
                    cached_rubric=cached_rubric,
                )

                # UI toggle for show/hide reasoning
                raw = st.session_state.get("hints") or ""
                # sanitize again here (belt & suspenders) — same regex as in analyzer.py
                raw_sanitized = re.sub(r"<think>[\s\S]*?</think>", "[ajattelujälki piilotettu]\n", raw, flags=re.IGNORECASE)

                with st.expander("Näytä edistynyt ajattelujälki (varoitus: voi sisältää mallin raakaa päättelyä)", expanded=False):
                    show_think = st.checkbox("Näytä <think>…</think> -sisältö")
                    if show_think:
                        st.markdown("**Palaute (raaka):**")
                        st.markdown(raw)
                    else:
                        st.markdown("**Palaute:**")
                        st.markdown(raw_sanitized)

                st.session_state["hints"] = hints

                # First-run rubric cache:
                # - If we have a reference solution and no rubric cached yet, derive a short rubric and store it.
                #   Subsequent analyses are faster and cheaper.
                if ex_id and not cached_rubric and st.session_state.get("ai_solution"):
                    rub = derive_rubric_from_reference(
                        st.session_state["ai_solution"], build_llm, ollama_model, timeout_s=12
                    )
                    st.session_state["rubrics"][ex_id] = rub
                # 👆 this caches the rubric for future runs

            except Exception as e:
                st.error(f"Analyysi epäonnistui: {e}")
                st.stop()

    # Show tests
    tests = st.session_state.get("tests", [])
    if tests:
        st.markdown("**Tests**")
        st.json(tests)

    # Show summarized results
    results = st.session_state.get("results", [])
    if results:
        st.markdown("**Results**")
        st.json(summarize_results(results))
        passed = sum(1 for r in results if r.get("passed"))
        st.success(f"Passed {passed}/{len(results)}") if passed==len(results) else st.warning(f"Passed {passed}/{len(results)}")

    # Show analysis hints
    hints = st.session_state.get("hints")
    if hints:
        st.markdown("**Palaute koodista**")
        st.markdown(hints)

# Solve Task
with tab_solve:
    # (existing) solve button and solution output below
    if st.button("Ratkaise tehtävä", use_container_width=True):
        code = solve_auto(item.get("question",""), item.get("python_blocks") or [], model=ollama_model)
        st.code(code or "# (no output)", language="python")
        st.session_state["ai_solution"] = code

    code = st.session_state.get("ai_solution", "")
    ctx = st.session_state.get("ai_context")
    if ctx:
        with st.expander("Reasoning summary (structured spec)"):
            st.json(ctx)