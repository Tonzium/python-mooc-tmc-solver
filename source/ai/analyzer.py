# ai/analyzer.py
"""
Analyzer: produces short, teacher-style feedback without revealing full solutions.

User-facing guarantees:
- Feedback is concise, bullet-first, and never reveals a full working answer.

Developer notes:
- Few-shots are curated and kept short to control token usage.
- Compact mode: we prefer a rubric over a full reference to lower latency.
- DeepSeek reasoning (<think>...</think>) is removed by default and can be re-enabled in the UI.
"""
from __future__ import annotations
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import inspect
import re

_SUPPORTED_HINT = {
    # common names we may want to keep if supported by your builder
    "temperature", "top_p", "max_tokens", "stop", "repeat_penalty",
    "request_timeout", "num_ctx",
}

# Find deepseek reasoning
_RE_THINK = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)

def _hide_think(text: str) -> str:
    """ Hide deepseek reasoning"""
    # replace entire reasoning block with a stub
    return _RE_THINK.sub("[ajattelujälki piilotettu]", text)

# Defensive LLM builder:
# - Strip any kwargs the concrete builder doesn't support (prevents
#   "unexpected keyword argument 'top_p'" errors when swapping backends).
# - Falls back to a minimal set if signature introspection fails.
def _llm_from_builder(build_llm, model: str, **want):
    """
    Calls your build_llm(model, **kwargs) but strips any kwargs
    that aren't in the builder's signature. Prevents 'unexpected keyword' errors.
    """
    try:
        sig = inspect.signature(build_llm)
        allowed = {k: v for k, v in want.items() if k in sig.parameters}
    except (TypeError, ValueError):
        # couldn't introspect; pass a very small, safe set
        allowed = {k: v for k, v in want.items() if k in {"temperature", "request_timeout"}}
    return build_llm(model, **allowed)

# --------------------------
# Few-shot example structure
# --------------------------
@dataclass
class FSExample:
    title: str
    question: str
    student_code: str
    tests_summary: str
    desired_feedback: str

def _truncate(s: Optional[str], max_chars: int) -> str:
    if not s:
        return ""
    s = s.strip()
    return s if len(s) <= max_chars else (s[:max_chars] + "\n... [truncated]")

# Post-filter: if the model sneaks in long code blocks, collapse them to short stubs.
# This keeps feedback focused and respects the "no full solutions" rule.
def _strip_long_code_blocks(text: str) -> str:
    import re
    def replacer(m):
        body = m.group(1)
        # allow up to 3 lines only
        lines = [ln for ln in body.splitlines() if ln.strip()]
        if len(lines) > 3:
            return "```text\n[koodiesimerkki poistettu tämän ohjeen vuoksi]\n```"
        return m.group(0)
    return re.sub(r"```(?:python|py|)\n([\s\S]*?)```", replacer, text, flags=re.IGNORECASE).strip()

# ----------------------------------
# Five concise few-shot demonstrations
# ----------------------------------
DEFAULT_FEW_SHOTS: List[FSExample] = [
    FSExample(
        title="Lauseen tulostus",
        question="Kirjoita ohjelma, joka tulostaa ruudulle hymiön: :-)",
        student_code="print(':-/')",
        tests_summary="",
        desired_feedback=(
            "Sinulla on vain pieni ongelma koodissasi \n"
            "Suosittelen tarkastamaan yksittäiset merkit"
        ),
    ),
    FSExample(
        title="Return type + naming",
        question="Toteuta `keskiarvo(nums)` joka palauttaa keskiarvon floattina.",
        student_code="def keskiarvo(nums):\n    s=0\n    for n in nums:\n        s+=n\n    return s/len(nums) if len(nums)>0 else 0",
        tests_summary="int-only input works, but expected float when len==0 is None (spec).",
        desired_feedback=(
            "- Paluuarvo tyhjälle listalle on virheellinen: ohjeen mukaan palauta None.\n"
            "- Nimeä muuttujat selkeämmin (`total`, `count`).\n"
            "- Varmista kelluva jako: Python tekee sen jo, mutta testaa myös sekadatat (int+float)."
        ),
    ),
    FSExample(
        title="Complexity + early exit",
        question="Tarkista onko merkkijono palindromi (True/False).",
        student_code="def pal(s):\n    return ''.join(reversed(s))==s",
        tests_summary="works; long strings ok; no spaces/normalization noted",
        desired_feedback=(
            "- Ratkaisu toimii, mutta huomioi normalisointi: pienet kirjaimet ja välilyönnit voivat vaikuttaa.\n"
            "- Lisää testit: \"Abba\" → True, \"ab ba\" → True jos välilyönnit ignoroidaan.\n"
            "- Hyvä O(n) aika, ei lisämuistia pakollinen."
        ),
    ),
    FSExample(
        title="Off-by-one + boundaries",
        question="Laske montako arvoa on välillä [a,b] listassa.",
        student_code="def count_range(L,a,b):\n    c=0\n    for x in L:\n        if a<x<b: c+=1\n    return c",
        tests_summary="[1,2,3], a=1, b=3 → odotettu 3, saatu 1 (FAIL).",
        desired_feedback=(
            "- Väli on suljettu [a,b], mutta käytät a<x<b (avoimet rajat). Korjaa vertailu.\n"
            "- Testaa raja-arvot: x==a, x==b."
        ),
    ),
    FSExample(
        title="Control flow + readability",
        question="Palauta ensimmäinen parillinen arvo listasta tai None.",
        student_code="def first_even(L):\n    ans=None\n    for i in range(len(L)):\n        if L[i]%2==0:\n            ans=L[i]\n    return ans",
        tests_summary="returns last even, not first; unnecessary indexing.",
        desired_feedback=(
            "- Palaa heti kun löydät parillisen (`return L[i]`). Nyt palautat viimeisen.\n"
            "- Iteroi suoraan arvoilla: `for x in L:` parantaa luettavuutta.\n"
            "- Lisää testi kun listassa ei ole parillisia."
        ),
    ),
]

# ---------------------------------------
# Cheap rubric derivation (short & fast):
# - Converts a full reference into <=6 bullets (Finnish).
# - Keeps analysis prompts small and consistent across runs.
# ---------------------------------------
def derive_rubric_from_reference(reference_solution: str, build_llm, model: str, timeout_s: int = 15) -> str:
    """
    Very short, low-cost step to turn a full reference into a compact rubric.
    This reduces tokens for the main analysis call (improves latency).
    """
    if not reference_solution:
        return ""

    system = (
        "You have just recieved student proposal code for the assingment."
        "Summarize the essential requirements of the reference solution as a short rubric. "
        "Max 6 bullet points. No code. When forming an answer use Finnish language."
    )
    user = (
        "Referenssikoodi (vain sisäiseen tarkasteluun):\n"
        "```\n{ref}\n```\n"
        "Laadi ytimekäs tarkistuslista: odotettu syöte/ulos, reunaehdot, nimikriteerit, virheenkäsittely."
    )
    prompt = ChatPromptTemplate.from_messages([("system", system), ("user", user)])

    llm = _llm_from_builder(
    build_llm,
    model,
    temperature=0.15,
    top_p=0.9,
    request_timeout=timeout_s,
    max_tokens=320,
    stop=["```", "\n\n```python", "\n```python"],
    repeat_penalty=1.05,
    )
    
    out = (prompt | llm | StrOutputParser()).invoke({"ref": _truncate(reference_solution, 3000)})
    return _truncate(out, 1000)

# ---------------------
# Main analyze function
# ---------------------
def analyze(student_code: str,
            question: str,
            reference_solution: Optional[str],
            test_results: Optional[list] = None,
            model: str = "deepseek-r1:latest",
            build_llm=None,
            timeout_s: int = 40,
            use_compact_mode: bool = True,
            few_shots: Optional[List[FSExample]] = None,
            cached_rubric: Optional[str] = None) -> str:
    """
    Returns teacher-style feedback without revealing the answer.

    Speed & quality features:
    - Few-shot prompting (5 curated examples).
    - Compact mode: send a short rubric instead of the whole reference (fewer tokens).
    - Tight model options: low temperature, small max output, stop sequences.
    """

    if build_llm is None:
        raise RuntimeError("build_llm callable must be provided.")

    if not student_code or not student_code.strip():
        return "Lisää koodi ensin kenttään 'Your code' ennen analyysiä."

    # ---------- Truncate aggressively to keep context small (faster) ----------
    student_code = _truncate(student_code, 6000)
    question = _truncate(question or "", 1500)

    # Summarize tests only if failures exist (faster & more focused)
    tests_text = ""
    if test_results:
        fails = [r for r in test_results if not r.get("passed")]
        if fails:
            tests_text = (
                f"Epäonnistuneita testejä: {len(fails)}.\n" +
                "\n".join(
                    f"- {r.get('name','(nimetön)')}: {str(r.get('message',''))[:160]}"
                    for r in fails[:8]
                )
            )
        else:
            tests_text = "Kaikki testit läpäisty (lyhyt laatuarvio riittää)."

    # ---------- Reference handling: compact rubric by default ----------
    rubric = cached_rubric or ""
    if use_compact_mode:
        if not rubric and reference_solution:
            # precompute & cache this in Streamlit to avoid paying this per click
            rubric = derive_rubric_from_reference(reference_solution, build_llm, model, timeout_s=12)
        # We never send full ref in compact mode
        ref_payload = f"(tiivis rubriikki)\n{rubric or '(ei rubriikkia)'}"
    else:
        # fall back to sending a truncated reference (still allowed but heavier)
        ref_payload = "ÄLÄ paljasta koodia opiskelijalle.\n" + _truncate(reference_solution or "", 3500)

    # ---------- Few-shot messages ----------
    shots = few_shots if few_shots is not None else DEFAULT_FEW_SHOTS
    msg_pairs = []
    for ex in shots:
        u = (
            f"Tehtävänanto (tiivistetty):\n{_truncate(ex.question, 400)}\n\n"
            f"Opiskelijan koodi:\n```python\n{_truncate(ex.student_code, 400)}\n```\n\n"
            f"Testitilanne (tiivistetty):\n{_truncate(ex.tests_summary, 220)}\n\n"
            "Laadi palaute yllä olevasta opiskelijan koodista sääntöjen mukaan."
        )
        a = _truncate(ex.desired_feedback, 500)
        msg_pairs.append(("user", u))
        msg_pairs.append(("assistant", a))

    # ---------- System + live user message ----------
    system_msg = (
        "You are a strict but helpful Python TA.\n"
        "ABSOLUTE RULES:\n"
        "1) Älä paljasta referenssiratkaisua tai valmista toimivaa koodia.\n"
        "2) Pysy ytimekkäänä: 1-6 bullet-pointia, max ~150 sanaa.\n"
        "3) Keskity: virheet, reunaehdot, refaktorointi, nimeäminen, testit.\n"
        "4) Vastaus suomeksi."
    )

    live_user = (
        "Tehtävänanto (tiivistetty):\n{question}\n\n"
        "Opiskelijan koodi:\n```python\n{student}\n```\n\n"
        "Sisäinen referenssi/arviointirubriikki (älä paljasta opiskelijalle):\n{ref}\n\n"
        "Testitilanne (tiivistetty):\n{tests}\n\n"
        "Laadi palaute yllä olevaan opiskelijan koodiin:\n"
        "- osoita virheet/puutteet\n"
        "- ehdota korjauksia sanallisesti (ilman valmista koodia)\n"
        "- mainitse reunaehdot + paremmat testit\n"
        "- lyhyesti ja selkeästi."
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_msg)] + msg_pairs + [("user", live_user)]
    )

    # ---------- FAST model options ----------
    # Most local LLM wrappers accept these kwargs; build_llm can forward to Ollama.
    llm = _llm_from_builder(
    build_llm,
    model,
    temperature=0.15,
    top_p=0.9,
    request_timeout=timeout_s,
    max_tokens=320,
    stop=["```", "\n\n```python", "\n```python"],
    repeat_penalty=1.05,
    )

    out = (prompt | llm | StrOutputParser()).invoke({
        "question": question,
        "student": student_code,
        "ref": ref_payload,
        "tests": tests_text or "(ei testiraporttia)"
    })

    clean = _strip_long_code_blocks(out)
    clean = _hide_think(clean)   # <- hide reasoning by default

    return clean or "Ei analyysiä saatavilla."
