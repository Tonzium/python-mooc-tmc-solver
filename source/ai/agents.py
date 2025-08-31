# ai/agents.py
# -*- coding: utf-8 -*-
"""
How solutions are produced:

- solve_simple(): LangChain ChatOllama → fastest path, returns fenced Python code only.
- solve_direct_ollama(): raw REST call to /api/generate as a fallback if LC path fails.
- solve_with_agents(): two-stage agentic flow (ContextSpec → final code). Heavier but robust.

Model endpoint:
- OLLAMA_HOST is normalized (scheme + localhost fix). "0.0.0.0" is mapped to "127.0.0.1"
  to avoid Windows connection errors.
"""

from __future__ import annotations
from typing import List, Dict, Any, Tuple, Optional, Literal
import json
import os
import re
import requests
from pydantic import BaseModel, Field, ValidationError

# --- LangChain core ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import Runnable

import urllib.parse

# ---------------------------------------------------------------------
# Normalize OLLAMA_HOST to a client-connectable URL:
# - Ensure scheme, fill default port 11434, replace 0.0.0.0/:: with 127.0.0.1.
# ---------------------------------------------------------------------
def _safe_ollama_base_url() -> str:
    raw = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").strip()
    # add scheme if missing
    if "://" not in raw:
        raw = "http://" + raw
    # normalize + replace 0.0.0.0 with 127.0.0.1 (client connect fix)
    u = urllib.parse.urlparse(raw)
    host = u.hostname or "127.0.0.1"
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    port = u.port or 11434
    scheme = u.scheme or "http"
    return f"{scheme}://{host}:{port}"

# Chat model: prefer langchain_ollama, fall back to community
try:
    from langchain_ollama import ChatOllama  # modern location
except Exception:
    from langchain_community.chat_models import ChatOllama  # fallback


# ---------------------------------------------------------------------
# # ChatOllama factory:
# - Only forwards a safe subset of options (top_p, num_ctx, max_tokens, etc.)
#   to keep compatibility across langchain_ollama versions.
# ---------------------------------------------------------------------
def build_llm(model: str, temperature: float = 0.2, **kwargs) -> ChatOllama:
    """
    Returns a ChatOllama instance targeting the local Ollama server.
    - Uses defaults that work well for deterministic coding tasks.
    """
    base_url = _safe_ollama_base_url()
    # Forward a safe subset if present
    forwardable = {}
    for k in ("top_p", "num_ctx", "repeat_penalty", "stop", "request_timeout", "max_tokens"):
        if k in kwargs and kwargs[k] is not None:
            forwardable[k] = kwargs[k]
    return ChatOllama(model=model, temperature=temperature, base_url=base_url, **forwardable)

def list_ollama_models(timeout: float = 1.5) -> List[str]:
    """
    Query local Ollama for installed models; graceful fallback if unreachable.
    """
    base = _safe_ollama_base_url()
    url = f"{base}/api/tags"
    try:
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        tags = [m.get("name") for m in data.get("models", []) if m.get("name")]
        # De-duplicate and keep short names first (e.g., "llama3.2" before "llama3.2:latest")
        seen, out = set(), []
        for t in sorted(tags, key=lambda x: (x.split(":")[0], len(x))):
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out or ["llama3.2", "mistral:7b", "deepseek-r1:32b"]
    except Exception:
        return ["llama3.2", "mistral:7b", "deepseek-r1:32b"]

# ---------------------------------------------------------------------
# Simplified version
# ---------------------------------------------------------------------
def solve_simple(question: str, python_blocks: List[str], model: str = "deepseek-r1:latest") -> str:
    """Smallest useful prompt to get code. Returns raw Python code."""
    llm = build_llm(model, temperature=0.1)
    blocks = "\n\n".join(f"```python\n{b}\n```" for b in (python_blocks or []))
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a concise Python tutor. Output ONLY one Python code block "
         "that solves the task. No explanations."),
        ("user",
         "Task (Finnish or English):\n{q}\n\nStarter blocks (optional):\n{b}\n\n"
         "Return ONLY the final solution in:\n```python\n# code\n```"),
    ])
    out = (prompt | llm | StrOutputParser()).invoke({"q": question, "b": blocks})
    # reuse fence extractor
    return _extract_python_code(out)

def solve_direct_ollama(question: str, python_blocks: List[str], model: str = "deepseek-r1:latest") -> str:
    base = _safe_ollama_base_url()
    blocks = "\n\n".join(f"```python\n{b}\n```" for b in (python_blocks or []))
    prompt = (
        "You are a concise Python tutor. Output ONLY one Python code block that solves the task.\n"
        "Task:\n" + question + "\n\nStarter blocks (optional):\n" + blocks +
        "\n\nReturn ONLY the final solution wrapped in a python code fence."
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1}
    }
    r = requests.post(f"{base}/api/generate", json=payload, timeout=180)
    r.raise_for_status()
    text = (r.json() or {}).get("response", "")
    return _extract_python_code(text)

# ---------------------------------------------------------------------
# "Do the simplest thing that works":
# - Try the LC path first; if it raises, fall back to a direct REST call.
# - If both fail, raise the original LC error for easier debugging.
# ---------------------------------------------------------------------

def solve_auto(question: str, python_blocks: List[str], model: str = "deepseek-r1:latest") -> str:
    try:
        return solve_simple(question, python_blocks, model=model)
    except Exception as e:
        # last-ditch: try raw REST
        try:
            return solve_direct_ollama(question, python_blocks, model=model)
        except Exception:
            # surface the original LC error to help debugging
            raise e

# ---------------------------------------------------------------------
# Context Agent schema
# ---------------------------------------------------------------------

class ContextSpec(BaseModel):
    task_type: Literal["function", "script", "io", "transform"] = "function"
    function_name: Optional[str] = None
    input_spec: str = Field(..., description="What inputs exist (stdin, parameters, file) and their types.")
    output_spec: str = Field(..., description="Exact required output/return format.")
    constraints: List[str] = Field(default_factory=list, description="Rules like no I/O, sort, rounding, etc.")
    examples: List[str] = Field(default_factory=list, description="Short I/O pairs or edge cases.")

# ---------------------------------------------------------------------
# Context Agent: condenses the natural-language prompt into a strict JSON spec
# so the Coding Agent can avoid guesswork (and reduce invalid I/O).
# ---------------------------------------------------------------------

def get_context(question: str, python_blocks: List[str], model: str) -> ContextSpec:
    """
    Runs the Context Agent and returns a structured spec.
    """
    blocks_text = "\n\n".join(f"```python\n{b}\n```" for b in (python_blocks or []))
    chain = _context_chain(model)
    try:
        return chain.invoke({"question": question, "blocks": blocks_text})
    except ValidationError as ve:
        # Fallback: minimal safe spec so CodingAgent still runs
        return ContextSpec(
            input_spec="Infer from question.",
            output_spec="Follow problem statement exactly.",
            constraints=["Write idiomatic Python 3."],
        )

def _context_chain(model: str) -> Runnable:
    llm = build_llm(model)
    parser = JsonOutputParser(pydantic_object=ContextSpec)

    EXAMPLES = [
        # 1) Small MOOC-style parsing task
        {
            "question": "Kirjoita ohjelma, joka tulostaa ruudulle hymiön: :-)",
            "blocks": "",
            "context": ContextSpec(
                task_type="python code",
                function_name="print",
                input_spec="print function",
                output_spec=":-)",
                constraints=["Only return python code"],
                examples=["expected return is print(':-)')"],
            ).dict()
        },
    ]

    example_msgs = []
    for ex in EXAMPLES:
        example_msgs += [
            ("user", f"QUESTION (Finnish):\n{ex['question']}\n\nPYTHON STARTER BLOCKS:\n{ex['blocks']}"),
            ("assistant", json.dumps(ex["context"], ensure_ascii=False)),
        ]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a precise Python problem analyst. "
             "Return ONLY JSON matching the schema.\n{format_instructions}"),
            *example_msgs,
            ("user",
             "QUESTION (Finnish):\n{question}\n\nPYTHON STARTER BLOCKS (if any):\n{blocks}\n\n"
             "If something is ambiguous, infer the most common MOOC requirement.")
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    return prompt | llm | parser


# ---------------------------------------------------------------------
# Coding Agent (writes the solution)
# ---------------------------------------------------------------------
def _coding_chain(model: str) -> Runnable:
    llm = build_llm(model)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a senior Python engineer. Write a correct, idiomatic Python 3 solution\n"
                "that satisfies the given ContextSpec. Follow ALL constraints. Do not invent new IO unless required.\n"
                "Output ONLY one Python code block using triple backticks like:\n"
                "```python\n# code here\n```"
            ),
            (
                "user",
                "ContextSpec (JSON):\n{context_json}\n\n"
                "Original question for reference:\n{question}\n\n"
                "Starter blocks (if any):\n{blocks}\n\n"
                "Write the final solution now.",
            ),
        ]
    )

    return prompt | llm | StrOutputParser()


# ---------------------------------------------------------------------
# Sanitizers:
# - _strip_think: hide DeepSeek <think> traces from user-facing text.
# - _extract_python_code: pull the first fenced python block, or raw text if unfenced.
# ---------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)

def _strip_think(x: str) -> str:
    return _THINK_RE.sub("[ajattelujälki piilotettu]", x or "")

_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)



def _extract_python_code(text: str) -> str:
    """
    Pull the first fenced code block; fall back to raw text if not fenced.
    """
    m = _CODE_FENCE_RE.search(text or "")
    return (m.group(1).strip() if m else (text or "")).strip()

# ---------------------------------------------------------------------
# Lil Chain
# ---------------------------------------------------------------------
def solve_with_agents(
    question: str,
    python_blocks: List[str],
    model: str = "deepseek-r1:latest",
) -> Tuple[str, Dict[str, Any]]:
    """
    Orchestrates Context → Coding. Returns (solution_code, context_as_dict).
    """
    context = get_context(question, python_blocks, model=model)
    context_json = json.dumps(context.dict(), ensure_ascii=False, indent=2)

    blocks_text = "\n\n".join(f"```python\n{b}\n```" for b in (python_blocks or []))
    code_gen = _coding_chain(model).invoke(
        {
            "context_json": context_json,
            "question": question,
            "blocks": blocks_text,
        }
    )

    code = _extract_python_code(code_gen)
    return code, context.dict()


# Backward-compatible helper:
def solve_problem(question: str, python_blocks: List[str], model: str = "deepseek-r1:latest") -> str:
    code, _ = solve_with_agents(question, python_blocks, model=model)
    return code

# --- Brief learning-goal summary ("what's the point") ---
def summarize_point(question: str, python_blocks: List[str], model: str = "deepseek-r1:latest") -> str:
    """
    Returns very short answer in Finnish that explain
    what concept(s) this task trains. No code, no long text.
    """
    llm = build_llm(model, temperature=0.2)
    blocks = "\n\n".join(f"```python\n{b}\n```" for b in (python_blocks or []))
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Tehtävä on annettu python ohjelmointia harjoitteleville ohjelmointialan opiskelijoille.\n"
         "Selität suomeksi mikä on oppimistavoite tässä tehtävässä.\n"
         "Kerro mikä on tehtävän ydinajatus: mitä taitoa tai käsitettä harjoitellaan juuri tässä tehtävässä.\n"
         "Älä anna python koodia. Älä jaarittele."),
        ("user",
         "Tehtävänanto:\n{q}\n\nMahdolliset python koodikielen lähtöblokit:\n{b}\n\n"
         "Kirjoita vain ytimekäs vastaus ottaen huomioon tehtävän anto ja lähtöblokit.\n"
         "Älä jaarittele vastauksessani. Pidä vastaus lyhyenä.\n"
         "Mikäli tehtävässä pyydetään 'tulostamaan' tehtävässä pyritään oppimaan print() function käyttöä.\n"
         "Älä anna python koodia.\n"
         "Typistä vastaus 1-3 ytimekkääseen bullet pointtiin.")
    ])
    text = (prompt | llm | StrOutputParser()).invoke({"q": question, "b": blocks})
    return _strip_think(text.strip().replace("```", ""))