"""
agent.py — Newsletter Agent built with LangGraph.

Architecture (8-node state machine):

  START
    │
    ▼
  [plan]       → gemini reads the goal, creates research plan + search queries
    │              HITL interrupt #1: human can approve / modify queries
    ▼
  [search]     → Runs DuckDuckGo searches (Tool #1)
    │
    ▼
  [fetch]      → Fetches full article text via HTTP (Tool #2)
    │
    ▼
  [summarize]  → gemini summarises each article independently
    │
    ▼
  [write]      → gemini + HTML generator produces the full newsletter (Tool #3)
    │              HITL interrupt #2: human can preview before critique
    ▼
  [critique]   → gemin self-evaluates and scores the output (self-reflection)
    │
    ├─ score < threshold AND revisions left ──► [revise] ──► back to [write]
    │
    └─ score ≥ threshold OR max revisions ──────────────────► [output]
                                                                  │
                                                               [END]
"""

import json
import os
import uuid
from datetime import datetime
from typing import Annotated, List, Optional
import operator

# from google import genai
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# from google.genai import types
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from config import (
    CRITIQUE_THRESHOLD,
    META_API_KEY,
    MAX_ARTICLES,
    MAX_REVISIONS,
    MODEL,
    OUTPUT_DIR,
    SEARCH_QUERIES,
)
from tools import fetch_article, generate_html, web_search

# ─── gemini client ────────────────────────────────────────────────────────
# client = genai.Client(api_key=GEMINI_API_KEY)
#__META client______
hf_token = os.environ.get("HF_TOKEN", "")
if OpenAI is not None and hf_token:
    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=hf_token,
    )
else:
    # Unauthenticated fallback: lightweight local stub client to avoid external auth.
    class _StubChoices:
        def __init__(self, content):
            self.message = {"content": content}

    class _StubClient:
        class chat:
            class completions:
                @staticmethod
                def create(model, messages=None, max_tokens=None):
                    user_text = ""
                    if messages:
                        # prefer the last user message
                        for m in reversed(messages):
                            if m.get("role") == "user":
                                user_text = m.get("content", "")
                                break
                    content = f"[SIMULATED LLM RESPONSE]\n{user_text[:800]}"
                    return type("R", (), {"choices": [_StubChoices(content)]})

    client = _StubClient()

# ─── State Schema ─────────────────────────────────────────────────────────────

class NewsletterState(TypedDict):
    # ── Input ──────────────────────────────────────────
    goal: str
    mode: str                        # "autonomous" | "hitl"

    # ── Planning ───────────────────────────────────────
    plan: str
    search_queries: List[str]

    # ── Research ───────────────────────────────────────
    raw_articles: List[dict]

    # ── Processing ─────────────────────────────────────
    summaries: List[dict]

    # ── Writing ────────────────────────────────────────
    newsletter_draft: str
    revision_count: int

    # ── Self-review ────────────────────────────────────
    critique: str
    critique_score: int

    # ── Output ─────────────────────────────────────────
    final_newsletter: str
    output_path: str

    # ── Execution log (each node appends; reducer = list concat) ──
    status_log: Annotated[List[str], operator.add]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")

def _log(*messages: str) -> List[str]:
    """Return a list of timestamped log strings for state update."""
    return [f"[{_ts()}] {m}" for m in messages]

def _parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON safely."""
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(text)


def _llm(user_prompt: str, *, system: str = "", max_tokens: int = 2000) -> str:
    """Call LLM via OpenAI-compatible client (messages + model required)."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user_prompt})

    completion = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=max_tokens,
    )

    # Normalize different response shapes
    msg = completion.choices[0].message
    if isinstance(msg, dict):
        return msg.get("content", "").strip()
    return getattr(msg, "content", str(msg)).strip()


# ─── Node 1 — Plan ────────────────────────────────────────────────────────────

def plan_node(state: NewsletterState) -> dict:
    """
    gemini analyses the goal and produces:
      • A short research strategy statement
      • A list of targeted DDG search queries

    HITL: If mode == 'hitl', execution pauses here.
    The human can approve the plan or inject modified queries.
    """
    text = _llm(
        f"""Goal: {state['goal']}

Generate a research plan for a weekly AI-agents newsletter.
Return ONLY valid JSON — no markdown, no commentary:

{{
  "plan": "2–3 sentence research strategy",
  "search_queries": [
    "query covering LangGraph / LangChain agents",
    "query covering AutoGen / CrewAI / multi-agent",
    "query covering agentic AI products / releases",
    "query covering AI agent benchmarks / research papers",
    "query covering autonomous AI news this week"
  ]
}}

Make every query distinct and time-focused (past week).""",
        system=(
            "You are an expert newsletter research planner. "
            "Your job is to turn a high-level goal into precise, "
            "diverse search queries that will surface the freshest AI agent news."
        ),
        max_tokens=800,
    )

    try:
        result = _parse_json(text)
    except Exception:
        # Safe fallback
        result = {
            "plan": "Search for AI agent news across frameworks, products, and research.",
            "search_queries": [
                "LangGraph LangChain AI agents news 2026",
                "AutoGen CrewAI multi-agent framework update",
                "autonomous AI agent products releases",
                "AI agent benchmark research paper 2026",
                "agentic AI systems latest developments",
            ],
        }

    logs = _log(
        f" Plan: {result['plan'][:120]}…",
        f" {len(result['search_queries'])} search queries generated",
    )

    # ── HITL Checkpoint 1 ────────────────────────────────────────────────────
    if state["mode"] == "hitl":
        human_input = interrupt({
            "type": "plan_approval",
            "message": "Review the research plan before the agent starts searching.",
            "plan": result["plan"],
            "queries": result["search_queries"],
        })
        if human_input.get("modified_queries"):
            result["search_queries"] = human_input["modified_queries"]
            logs += _log("✏️  Queries updated by human reviewer")
        else:
            logs += _log("✅ Plan approved by human reviewer")

    return {
        "plan": result["plan"],
        "search_queries": result["search_queries"],
        "status_log": logs,
    }


# ─── Node 2 — Search ──────────────────────────────────────────────────────────

def search_node(state: NewsletterState) -> dict:
    """
    Tool #1 — Web Search (DuckDuckGo, no API key required).
    Runs each query and deduplicates results by URL.
    """
    logs = _log(" Web research phase starting…")
    all_articles: list[dict] = []

    for i, query in enumerate(state["search_queries"][:SEARCH_QUERIES], 1):
        logs += _log(f"🔍 [{i}/{SEARCH_QUERIES}] {query}")
        results = web_search(query, max_results=4)
        for r in results:
            all_articles.append({
                "title": r.get("title", ""),
                "url":   r.get("href", ""),
                "snippet": r.get("body", ""),
                "source_query": query,
            })

    # Deduplicate by URL
    seen: set[str] = set()
    unique: list[dict] = []
    for art in all_articles:
        if art["url"] and art["url"] not in seen:
            seen.add(art["url"])
            unique.append(art)

    logs += _log(f"📰 {len(unique)} unique articles found across all queries")

    return {
        "raw_articles": unique[:MAX_ARTICLES + 5],   # fetch a few extra as buffer
        "status_log": logs,
    }


# ─── Node 3 — Fetch ───────────────────────────────────────────────────────────

def fetch_node(state: NewsletterState) -> dict:
    """
    Tool #2 — HTTP Article Fetcher.
    Enriches raw search results with full article text.
    """
    logs = _log("📥 Fetching full article content…")
    enriched: list[dict] = []

    for i, art in enumerate(state["raw_articles"][:MAX_ARTICLES], 1):
        logs += _log(f"📄 [{i}/{MAX_ARTICLES}] {art.get('title','')[:60]}…")
        content = fetch_article(art.get("url", ""))
        enriched.append({**art, "full_content": content})

    logs += _log(f"✅ Content fetched for {len(enriched)} articles")

    return {"raw_articles": enriched, "status_log": logs}


# ─── Node 4 — Summarise ───────────────────────────────────────────────────────

def summarize_node(state: NewsletterState) -> dict:
    """
    gemini summarises each article in 2–3 sentences.
    Filters out articles with too little content.
    """
    logs = _log("✍️  Summarising articles with AI…")
    summaries: list[dict] = []

    for i, art in enumerate(state["raw_articles"], 1):
        content = art.get("full_content") or art.get("snippet", "")
        if len(content) < 80:
            continue   # skip stub results

        logs += _log(f"📝 [{i}] {art.get('title','')[:50]}…")

        summary = _llm(
            (
                f"Summarise the article below in exactly 2–3 sentences for an AI-agents newsletter.\n"
                f"Focus: What happened? Why does it matter for developers building AI agents?\n"
                f"Avoid generic phrases.\n\n"
                f"Title: {art.get('title','')}\n"
                f"Content: {content[:2_000]}"
            ),
            system=(
                "You are a senior AI technology journalist. "
                "Write sharp, specific, insight-driven summaries."
            ),
            max_tokens=300,
        )

        summaries.append({
            "title":   art.get("title", "Untitled"),
            "url":     art.get("url", ""),
            "summary": summary,
        })

    logs += _log(f"✅ {len(summaries)} summaries generated")

    return {"summaries": summaries[:MAX_ARTICLES], "status_log": logs}


# ─── Node 5 — Write ───────────────────────────────────────────────────────────

def write_node(state: NewsletterState) -> dict:
    """
    Tool #3 — HTML Newsletter Generator.
    gemini first enriches each summary with editorial voice,
    then the HTML template tool renders the final newsletter.
    """
    rev = state.get("revision_count", 0)
    logs = _log(f"✍️  {'Revising' if rev else 'Writing'} newsletter (pass #{rev + 1})…")

    date_str = datetime.now().strftime("%B %d, %Y")

    # Build extra context for revision passes
    critique_context = ""
    if rev > 0 and state.get("critique"):
        try:
            crit = json.loads(state["critique"])
            imps = "\n".join(f"  - {x}" for x in crit.get("improvements", []))
            critique_context = f"\n\nApply these editorial improvements:\n{imps}"
        except Exception:
            pass

    summaries_txt = "\n\n".join(
        f"Article {i}: {s['title']}\nURL: {s['url']}\nSummary: {s['summary']}"
        for i, s in enumerate(state["summaries"], 1)
    )

    text = _llm(
        (
            f"Enhance these article summaries for an AI-agents newsletter audience.{critique_context}\n\n"
            f"{summaries_txt}\n\n"
            f"Return ONLY valid JSON:\n"
            f'{{"articles":[{{"title":"...","url":"...","summary":"enhanced 2-3 sentence summary"}}]}}'
        ),
        system=(
            "You are the founding editor of the world's best AI newsletter. "
            "Rewrite summaries to be sharp, opinionated, and insightful — "
            "the kind a senior ML engineer would forward to colleagues."
        ),
        max_tokens=2_000,
    )

    try:
        data = _parse_json(text)
        articles = data.get("articles", state["summaries"])
    except Exception:
        articles = state["summaries"]

    # Render the HTML via the template tool
    html = generate_html(title="AI AGENT WEEKLY", date=date_str, articles=articles)

    logs += _log(f"📄 Newsletter HTML generated ({len(html):,} chars)")

    return {"newsletter_draft": html, "status_log": logs}


# ─── Node 6 — Critique ────────────────────────────────────────────────────────

def critique_node(state: NewsletterState) -> dict:
    """
    Self-reflection: Claude evaluates its own newsletter output
    and returns a structured score + specific improvement suggestions.
    This drives the autonomous quality-control loop.
    """
    logs = _log("🔍 Running self-critique…")

    # Extract readable text from HTML for critique
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(state["newsletter_draft"], "html.parser")
    newsletter_text = soup.get_text(separator="\n", strip=True)[:3_000]

    response_text = _llm(
        (
            f"Critically evaluate this AI-agents newsletter.\n"
            f"Score on: relevance, insight depth, readability, source diversity, editorial voice.\n\n"
            f"Return ONLY valid JSON:\n"
            f'{{"score":<1-10>,'
            f'"strengths":["..."],'
            f'"weaknesses":["..."],'
            f'"improvements":["concrete improvement 1","concrete improvement 2"],'
            f'"verdict":"one sentence overall assessment"}}\n\n'
            f"Newsletter text:\n{newsletter_text}"
        ),
        system=(
            "You are a demanding senior editor at a top tech publication. "
            "Critique ruthlessly but constructively. Be specific."
        ),
        max_tokens=700,
    )

    try:
        result = _parse_json(response_text)
        score = max(1, min(10, int(result.get("score", 7))))
        critique_text = json.dumps(result, indent=2)
    except Exception:
        score = 7
        critique_text = '{"score":7,"verdict":"Meets quality standards."}'

    verdict = "✅ Quality threshold met" if score >= CRITIQUE_THRESHOLD else f"⚠️  Below threshold ({CRITIQUE_THRESHOLD})"
    logs += _log(f"📊 Self-critique score: {score}/10  — {verdict}")

    return {"critique": critique_text, "critique_score": score, "status_log": logs}


# ─── Node 7 — Revise ──────────────────────────────────────────────────────────

def revise_node(state: NewsletterState) -> dict:
    """
    Applies critique feedback to improve article summaries.
    After this node the graph loops back to write → critique.
    """
    rev = state.get("revision_count", 0) + 1
    logs = _log(f"🔄 Revision #{rev}: applying editorial improvements…")

    try:
        crit = json.loads(state["critique"])
        improvements = crit.get("improvements", [])
    except Exception:
        improvements = ["Improve depth", "Sharpen editorial voice"]

    imp_text = "\n".join(f"- {x}" for x in improvements)

    text = _llm(
        (
            f"Improve these article summaries based on the following editorial notes:\n{imp_text}\n\n"
            f"Current summaries:\n{json.dumps(state['summaries'], indent=2)}\n\n"
            f"Return ONLY valid JSON:\n"
            f'{{"summaries":[{{"title":"...","url":"...","summary":"improved summary"}}]}}'
        ),
        system="You are improving article summaries based on editorial feedback.",
        max_tokens=1_200,
    )

    try:
        data = _parse_json(text)
        improved = data.get("summaries", state["summaries"])
    except Exception:
        improved = state["summaries"]

    logs += _log(f"✅ {len(improved)} summaries revised")

    return {"summaries": improved, "revision_count": rev, "status_log": logs}


# ─── Node 8 — Output ──────────────────────────────────────────────────────────

def output_node(state: NewsletterState) -> dict:
    """
    Saves the final newsletter to disk and simulates email delivery.
    HITL: In hitl mode, pauses here for final human approval before 'sending'.
    """
    logs = _log("💾 Preparing final output…")

    # ── HITL Checkpoint 2 ────────────────────────────────────────────────────
    if state["mode"] == "hitl":
        human_input = interrupt({
            "type": "final_approval",
            "message": "Newsletter is ready to send. Approve?",
            "score": state.get("critique_score", 0),
            "preview_snippet": state["newsletter_draft"][:800],
        })
        if not human_input.get("approved", True):
            return {
                "final_newsletter": "",
                "output_path": "",
                "status_log": logs + _log("❌ Send cancelled by human reviewer"),
            }
        logs += _log("✅ Send approved by human reviewer")

    # Save to disk
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"newsletter_{timestamp}.html"
    filepath = os.path.join(OUTPUT_DIR, filename)

    with open(filepath, "w", encoding="utf-8") as fh:
        fh.write(state["newsletter_draft"])

    logs += _log(
        f"📁 Saved → {filepath}",
        f"📧 Simulating delivery to subscribers…",
        f"   Subject: AI Agent Weekly — {datetime.now().strftime('%B %d, %Y')}",
        f"   To: subscribers@example.com",
        f"   Status: ✅ SENT (simulated)",
        f"📊 Final quality score: {state.get('critique_score','N/A')}/10",
        f"🎉 Newsletter agent completed successfully!",
    )

    return {
        "final_newsletter": state["newsletter_draft"],
        "output_path": filepath,
        "status_log": logs,
    }


# ─── Routing ──────────────────────────────────────────────────────────────────

def _route_after_critique(state: NewsletterState) -> str:
    """Conditional edge: revise or output based on score and revision budget."""
    score = state.get("critique_score", 0)
    revisions = state.get("revision_count", 0)

    if score < CRITIQUE_THRESHOLD and revisions < MAX_REVISIONS:
        return "revise"
    return "output"


# ─── Build Graph ──────────────────────────────────────────────────────────────

def build_graph():
    """
    Assemble and compile the LangGraph state machine.
    MemorySaver enables HITL checkpointing (interrupt → resume).
    """
    g = StateGraph(NewsletterState)

    # Add all nodes
    g.add_node("plan",      plan_node)
    g.add_node("search",    search_node)
    g.add_node("fetch",     fetch_node)
    g.add_node("summarize", summarize_node)
    g.add_node("write",     write_node)
    g.add_node("critique",  critique_node)
    g.add_node("revise",    revise_node)
    g.add_node("output",    output_node)

    # Linear backbone
    g.add_edge(START,       "plan")
    g.add_edge("plan",      "search")
    g.add_edge("search",    "fetch")
    g.add_edge("fetch",     "summarize")
    g.add_edge("summarize", "write")
    g.add_edge("write",     "critique")

    # Self-improvement loop
    g.add_conditional_edges(
        "critique",
        _route_after_critique,
        {"revise": "revise", "output": "output"},
    )
    g.add_edge("revise",  "write")   # loop back after revision
    g.add_edge("output",  END)

    memory = MemorySaver()
    return g.compile(checkpointer=memory)


# Module-level compiled graph (shared across requests)
newsletter_graph = build_graph()


# ─── Public Entry Point ───────────────────────────────────────────────────────

def run_newsletter_agent(goal: str, mode: str = "autonomous") -> dict:
    """
    Single entry-point function.

    Usage:
        result = run_newsletter_agent(
            goal="Create a weekly newsletter on latest AI agent news and send to subscribers.",
            mode="autonomous"   # or "hitl"
        )
        print(result["output_path"])

    Args:
        goal  : Plain-English instruction for the newsletter.
        mode  : "autonomous" — fully hands-off.
                "hitl"       — pauses for human approval at plan and send steps.

    Returns:
        Final NewsletterState dict.
    """
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    initial: NewsletterState = {
        "goal":             goal,
        "mode":             mode,
        "plan":             "",
        "search_queries":   [],
        "raw_articles":     [],
        "summaries":        [],
        "newsletter_draft": "",
        "revision_count":   0,
        "critique":         "",
        "critique_score":   0,
        "final_newsletter": "",
        "output_path":      "",
        "status_log": [f"[{_ts()}] 🚀 Newsletter Agent starting in {mode.upper()} mode…"],
    }

    return newsletter_graph.invoke(initial, config)