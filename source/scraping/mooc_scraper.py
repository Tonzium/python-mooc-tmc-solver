# -*- coding: utf-8 -*-
"""
MOOC exercise indexer & scraper (Selenium/Chrome).

Quickstart for developers:
- index_all_exercises(): fast list of all exercise anchors (no per-card scrape).
- fetch_exercise_by_href(href): scrape one exercise card (title, text, code).
- process_first_exercise(): dev helper to test the flow on the first item.

Design notes:
- We load the "All tasks" page once, scroll to bottom to trigger lazy-loading,
  and collect anchor links that contain an exercise slug (#programming-exercise).
"""

from __future__ import annotations
import json
import re
import time
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

# ---------------------------------------------------------------------------
# Constants & precompiled regexes
# ---------------------------------------------------------------------------

# Entry point that lists all exercises.
START_URL: str = "https://ohjelmointi-25.mooc.fi/kaikki-tehtavat/"

# When not logged in, a login prompt is sometimes concatenated to the end of the text.
LOGIN_PHRASE: str = "Kirjaudu sisään tehdäksesi tehtävän."
LOGIN_TAIL_RE: re.Pattern[str] = re.compile(rf"(?:\s*\n)*{re.escape(LOGIN_PHRASE)}\s*$")

# Section parsing supports both Finnish ("osa-N/X-...") and English mirror ("part-N/X-...").
SECTION_FI_RE: re.Pattern[str] = re.compile(r"/osa-(\d+)/(\d+)-")
SECTION_EN_RE: re.Pattern[str] = re.compile(r"/part-(\d+)/(\d+)-")

# Selector for the "All tasks" page: capture programming/quiz/written anchors only.
# If the site adds new card types, expand this list.
EXERCISE_LINK_SELECTOR: str = (
    "a[href*='#programming-exercise-'], "
    "a[href*='#quiz-'], "
    "a[href*='#written-exercise-']"
)

# Candidate selectors for elements inside a single exercise "card".
# We try these in order until a non-empty result is found.
TITLE_SELECTORS: Tuple[str, ...] = (
    "[class*='ProgrammingExerciseCard__Header']",  # most specific (site component)
    ".MuiCardHeader-root",                         # MUI header
    "h1, h2, h3",                                  # ultimate fallback
)

BODY_PARAGRAPH_SELECTORS: Tuple[str, ...] = (
    "[class*='MoocfiPythonEditorLoader__Wrapper'] p",  # correct prefix
    "[class*='ProgrammingExerciseCard__Body'] p",
    ".MuiCardContent-root p",
    "article p",
    "p",
)

CODE_BLOCK_SELECTORS: Tuple[str, ...] = (
    "div.gatsby-highlight pre.language-python",
    "div.gatsby-highlight pre[class*='language-']",
    "pre.language-python",
    "pre code.language-python",
    "pre",
)

# ---------------------------------------------------------------------------
# WebDriver helpers
# ---------------------------------------------------------------------------

def build_driver(headless: bool = True) -> webdriver.Chrome:
    """
    Create a Chrome WebDriver with sane defaults for scraping.

    Args:
        headless: Run without a visible browser window. Use False for debugging.

    Returns:
        A ready-to-use Selenium Chrome WebDriver instance.
    """
    opts = Options()
    if headless:
        # "--headless" is recommended with recent Chromes; renders closer to real mode.
        opts.add_argument("--headless=new")
    # Flags below improve stability inside CI/containers and reduce GPU/memory usage.
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--disable-dev-shm-usage")
    # A larger viewport to ensure elements are laid out consistently.
    opts.add_argument("--window-size=1600,2200")

    # webdriver_manager installs (and caches) a matching driver if needed.
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)


def wait_for_links(driver: webdriver.Chrome, timeout: int = 30) -> str:
    """
    Block until at least one exercise anchor is present in the DOM.

    Returns:
        The CSS selector used (handy to pass into other helpers).
    """
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, EXERCISE_LINK_SELECTOR))
    )
    return EXERCISE_LINK_SELECTOR


def scroll_to_bottom(driver: webdriver.Chrome, max_rounds: int = 18, idle_pause: float = 0.2) -> None:
    """
    Scroll to the bottom to trigger lazy-loading. Stop once height stabilizes twice.

    Args:
        max_rounds: Hard cap on scroll cycles to avoid infinite loops.
        idle_pause: Sleep between scrolls to give the page time to load.
    """
    last_height = driver.execute_script("return document.body.scrollHeight")
    stable_hits = 0

    for _ in range(max_rounds):
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(idle_pause)

        h = driver.execute_script("return document.body.scrollHeight")
        if h == last_height:
            stable_hits += 1
            if stable_hits >= 2:  # consider it stable twice in a row
                break
        else:
            stable_hits = 0
            last_height = h


# ---------------------------------------------------------------------------
# Parsing & extraction helpers
# ---------------------------------------------------------------------------

def collect_links(driver: webdriver.Chrome, selector: str) -> List[str]:
    """
    Collect exercise anchors from the "all tasks" page.

    - Removes duplicates while preserving first-seen order.
    - Sorts so that programming exercises appear first, then quizzes, then written.

    Returns:
        List of absolute or relative hrefs including the #slug.
    """
    anchors = driver.find_elements(By.CSS_SELECTOR, selector)

    # Grab hrefs that include an anchor part (ensures we have a slug).
    hrefs = []
    for a in anchors:
        href = a.get_attribute("href")
        if href and "#" in href:
            hrefs.append(href)

    # Dedupe while keeping order.
    seen, uniq = set(), []
    for u in hrefs:
        if u not in seen:
            seen.add(u)
            uniq.append(u)

    # Stable sort by "type" so buttons feel consistent in the UI.
    def score(u: str) -> int:
        if "#programming-exercise-" in u:
            return 0
        if "#quiz-" in u:
            return 1
        return 2

    uniq.sort(key=score)
    return uniq


def parse_section(url: str) -> Optional[str]:
    """
    Extract section identifier (e.g., 'osa-1/1' or 'part-1/1') from URL.

    Returns:
        A normalized 'osa-N/M' or 'part-N/M' string, or None if not matched.
    """
    m = SECTION_FI_RE.search(url)
    if m:
        return f"osa-{m.group(1)}/{m.group(2)}"
    m = SECTION_EN_RE.search(url)
    if m:
        return f"part-{m.group(1)}/{m.group(2)}"
    return None

# ------------------------------------------------------------------------------------
# DOM walk strategy:
# - We traverse headings, paragraphs, lists, tables, and code blocks in natural order.
# - Code is recognized either as <pre> or as highlighted wrappers.
# - This preserves the original text/code interleave for LLM prompts and UI rendering.
# -----------------------------------------------------------------------------------

ORDERED_XPATH = (
    # Walk in natural reading order. Include headings, paras, lists, tables.
    # Treat code as either the wrapper div OR a bare <pre> not inside the wrapper.
    ".//*[self::h1 or self::h2 or self::h3 or "
    "      self::p or self::li or self::table or "
    "      (self::div and contains(@class,'gatsby-highlight')) or "
    "      (self::pre and not(ancestor::div[contains(@class,'gatsby-highlight')]))]"
)

PY_PRE_XPATH = (
    ".//pre[contains(@class,'language-python') or "
    "      contains(@class,'gatsby-highlight') or "
    "      contains(@class,'language-')]"
)

def _is_code_node(el) -> bool:
    tag = el.tag_name.lower()
    if tag == "pre":
        return True
    if tag == "div":
        cls = (el.get_attribute("class") or "")
        if "gatsby-highlight" in cls:
            return True
    return False

def _node_text(el) -> str:
    # Selenium .text keeps newlines/indentation for <pre>
    return (el.text or "").strip()

def _iter_content_nodes(card):
    """
    Yield (node, kind) pairs for relevant descendants of 'card' in DOM order.
    kind in {'text', 'code'}.
    """
    nodes = card.find_elements(By.XPATH, ORDERED_XPATH)
    for n in nodes:
        tag = n.tag_name.lower()
        if tag == "pre":
            yield n, "code"
        elif tag in {"h1", "h2", "h3", "p", "li"}:
            yield n, "text"
        elif tag == "table":
            yield n, "text"
        else:
            yield n, "text"

def extract_question_card(driver: webdriver.Chrome, slug: str, timeout: int = 30) -> Tuple[str, str, List[str], List[Dict[str, str]], str]:
    """
    Return:
      title, question(text-only), python_blocks(code-only),
      ordered_blocks([{type:'text'|'code', content:str}]),
      prompt_markdown (LLM-ready, preserves order with ```python fences)
    """
    WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, f"[id='{slug}'], #{slug}"))
    )
    card = driver.find_element(By.CSS_SELECTOR, f"[id='{slug}'], #{slug}")

    # title (as before)
    title = ""
    for sel in TITLE_SELECTORS:
        try:
            t = card.find_element(By.CSS_SELECTOR, sel).text.strip()
            if t:
                title = t
                break
        except Exception:
            pass

    #ordered walk: text + code interleaved
    ordered_blocks: List[Dict[str, str]] = []
    text_accum: List[str] = []
    code_accum: List[str] = []

    nodes = card.find_elements(By.XPATH, ORDERED_XPATH)
    for n in nodes:
        kind = "code" if _is_code_node(n) else "text"
        content = _node_text(n)
        if not content:
            continue
        if kind == "text":
            # Strip the login footer if it shows up at the end of any text node
            content = LOGIN_TAIL_RE.sub("", content).strip()
            if not content:
                continue
            text_accum.append(content)
        else:
            code_accum.append(content)
        ordered_blocks.append({"type": kind, "content": content})

    # Fallback to old per-bucket selectors if nothing was captured (layout shift)
    if not ordered_blocks:
        # --- text fallback ---
        texts: List[str] = []
        for sel in BODY_PARAGRAPH_SELECTORS:
            for e in card.find_elements(By.CSS_SELECTOR, sel):
                t = e.text.strip()
                if t:
                    texts.append(t)
            if texts:
                break
        question = LOGIN_TAIL_RE.sub("", "\n\n".join(texts)).strip()

        # --- code fallback ---
        codes: List[str] = []
        for sel in CODE_BLOCK_SELECTORS:
            for e in card.find_elements(By.CSS_SELECTOR, sel):
                t = (e.text or "").strip()
                if t:
                    codes.append(t)
            if codes:
                break
        # de-dupe
        seen, uniq = set(), []
        for c in codes:
            if c not in seen:
                seen.add(c)
                uniq.append(c)

        ordered_blocks = [{"type": "text", "content": question}] + [
            {"type": "code", "content": c} for c in uniq
        ]
        text_accum, code_accum = [question], uniq
    else:
        # Compose question from texts only
        question = LOGIN_TAIL_RE.sub("", "\n\n".join([b["content"] for b in ordered_blocks if b["type"] == "text"])).strip()
        # De-dupe code blocks while preserving order (rare duplicate wrappers)
        seen, uniq = set(), []
        for c in code_accum:
            if c not in seen:
                seen.add(c)
                uniq.append(c)
        code_accum = uniq

    # LLM-ready prompt that preserves original order and fences code
    md_parts = []
    for b in ordered_blocks:
        if b["type"] == "text":
            md_parts.append(b["content"])
        else:
            md_parts.append("```python\n" + b["content"] + "\n```")
    prompt_markdown = "\n\n".join(md_parts).strip()

    return title, question, code_accum, ordered_blocks, prompt_markdown

#--------------------------------------------------------------------------------
# Trim noisy leading blocks:
# - We drop the very first pure-text block (often a duplicate title) and lone "/" markers.
#   This avoids showing meaningless headers in the UI or LLM prompt.
#------------------------------------------------------------------------------

# Remove noise: leading title or "/" text-only blocks
def _clean_ordered_blocks(blocks):
    cleaned = []
    for i, blk in enumerate(blocks):
        if blk["type"] == "text":
            if i == 0:
                continue
            if blk["content"].strip() in {"", "/"}:
                continue
        cleaned.append(blk)
    return cleaned

# Classify programming, quiz, written
def _classify(slug: str) -> str:
    """
    Map the slug format to a coarse exercise type.
    """
    if slug.startswith("programming-exercise-"):
        return "programming"
    if slug.startswith("quiz-"):
        return "quiz"
    if slug.startswith("written-exercise-"):
        return "written"
    return "other"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_first_exercise(headless: bool = True, save_path: Optional[str] = None) -> Dict:
    """
    Convenience function used during development:
    - Index the page,
    - Open the first exercise,
    - Extract details for that single item.

    Args:
        headless: Run Chrome headless.
        save_path: If provided, write the resulting JSON to this file (UTF-8).

    Returns:
        Dict with shape:
        {
          "count": 1,
          "generated_from": START_URL,
          "items": [ { running_problem_number, url, section_number, exercise_id,
                       exercise_type, title, question } ]
        }
    """
    driver = build_driver(headless=headless)
    try:
        driver.get(START_URL)
        selector = wait_for_links(driver)
        scroll_to_bottom(driver)
        links = collect_links(driver, selector)
        if not links:
            raise RuntimeError("No exercise links found on the page.")

        # We only take the first link by design (dev helper).
        href = urljoin(START_URL, links[0])
        base, slug = href.split("#", 1)

        driver.get(base)
        # Best-effort scroll to the anchored element (not critical if it fails).
        try:
            target = WebDriverWait(driver, 25).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, f"#{slug}, [id='{slug}']"))
            )
            driver.execute_script(
                "arguments[0].scrollIntoView({behavior:'instant', block:'start'});", target
            )
        except Exception:
            pass

        title, question, code_blocks, ordered_blocks, prompt_md = extract_question_card(driver, slug)

        # remove first two blocks because they are not giving any useful information
        ordered_blocks = _clean_ordered_blocks(ordered_blocks)

        exercise_type = _classify(slug)

        item = {
            "running_problem_number": 1,
            "url": href,
            "section_number": parse_section(href),
            "exercise_id": slug,
            "exercise_type": exercise_type,
            "title": title,
            "question": question,                 # (kept for compatibility)
            "python_blocks": code_blocks,         # (kept for compatibility)
            "ordered_blocks": ordered_blocks,     # NEW: interleaved blocks with types
            "prompt_markdown": prompt_md,         # NEW: LLM-ready, preserves order
        }

        data = {
            "count": 1,
            "generated_from": START_URL,
            "items": [item],
        }

        if save_path:
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

        return data
    finally:
        # Always clean up the browser process.
        driver.quit()


def index_all_exercises(headless: bool = True) -> Dict:
    """
    Build a fast index of *all* exercise anchors (no per-task card scraping).

    Returns:
        {
          "count": int,
          "generated_from": START_URL,
          "items": [
            { "url", "base_url", "section_number", "exercise_id", "exercise_type" }, ...
          ]
        }
    """
    driver = build_driver(headless=headless)
    try:
        driver.get(START_URL)
        selector = wait_for_links(driver)
        scroll_to_bottom(driver)
        links = collect_links(driver, selector)

        items: List[Dict[str, str]] = []
        for u in links:
            full = urljoin(START_URL, u)
            if "#" not in full:
                # Defensive: skip any weird links that lack an anchor.
                continue
            base, slug = full.split("#", 1)
            items.append(
                {
                    "url": full,
                    "base_url": base,
                    "section_number": parse_section(full),
                    "exercise_id": slug,
                    "exercise_type": _classify(slug),
                }
            )

        return {
            "count": len(items),
            "generated_from": START_URL,
            "items": items,
        }
    finally:
        driver.quit()


def fetch_exercise_by_href(href: str, headless: bool = True, timeout: int = 30) -> Dict:
    """
    Scrape one exercise (title + question) given its full href with #anchor.

    Args:
        href: Absolute or page-relative URL that includes a '#slug' anchor.
              Example: ".../osa-1/1-...#programming-exercise-hymio"
        headless: Run Chrome headless.
        timeout: Seconds to wait for the anchored card to appear.

    Returns:
        One dict shaped like items from process_first_exercise() (single element).
    """
    if "#" not in href:
        raise ValueError("Expected href with an anchor, e.g., ...#programming-exercise-xyz")

    base, slug = href.split("#", 1)

    driver = build_driver(headless=headless)
    try:
        driver.get(base)

        # Attempt to bring the target card into view to ensure it's rendered.
        try:
            target = WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, f"#{slug}, [id='{slug}']"))
            )
            driver.execute_script(
                "arguments[0].scrollIntoView({behavior:'instant', block:'start'});", target
            )
        except Exception:
            # It's okay if we can't scroll to it; extraction will still try to find it.
            pass

        title, question, code_blocks, ordered_blocks, prompt_md = extract_question_card(driver, slug)

        # remove first two blocks because they are not giving any useful information
        ordered_blocks = _clean_ordered_blocks(ordered_blocks)

        exercise_type = _classify(slug)

        return {
            "running_problem_number": 1,
            "url": href,
            "section_number": parse_section(href),
            "exercise_id": slug,
            "exercise_type": exercise_type,
            "title": title,
            "question": question,                 # (kept for compatibility)
            "python_blocks": code_blocks,         # (kept for compatibility)
            "ordered_blocks": ordered_blocks,     # NEW: interleaved blocks with types
            "prompt_markdown": prompt_md,         # NEW: LLM-ready, preserves order
        }
    finally:
        driver.quit()
