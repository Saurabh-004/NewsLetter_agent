"""
tools.py — Tool library for the Newsletter Agent.

Three core tools used by the LangGraph nodes:
  1. web_search()          → DuckDuckGo search (no API key needed)
  2. fetch_article()       → HTTP fetch + BeautifulSoup text extraction
  3. generate_html()       → Professional HTML newsletter template renderer

These are plain Python functions (not LangChain @tool wrappers) because
LangGraph nodes call them directly inside node functions.
"""

import re
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS


# ─── Tool 1: Web Search ───────────────────────────────────────────────────────

def web_search(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web via DuckDuckGo — no API key required.

    Returns a list of dicts with keys: title, href, body.
    Uses timelimit='w' so results are from the past week.
    """
    try:
        with DDGS() as ddgs:
            results = list(
                ddgs.text(query, max_results=max_results, timelimit="w")
            )
        return results
    except Exception as exc:
        # Fail gracefully so the agent can continue with other queries
        return [{"title": "Search unavailable", "href": "", "body": str(exc)}]


# ─── Tool 2: Article Content Fetcher ─────────────────────────────────────────

def fetch_article(url: str, max_chars: int = 3_000) -> str:
    """
    Fetch a URL and return clean plain text up to `max_chars`.

    Strategy:
      1. Prefer <article> or <main> tags (most article sites use them)
      2. Strip scripts / styles / nav / footer noise
      3. Collapse whitespace
    """
    if not url or not url.startswith("http"):
        return "Invalid or missing URL — skipped."

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.content, "html.parser")

        # Remove noise elements
        for tag in soup(["script", "style", "nav", "footer",
                         "header", "aside", "iframe", "noscript"]):
            tag.decompose()

        # Prefer semantic containers
        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", class_=re.compile(r"content|article|post", re.I))
            or soup.body
            or soup
        )

        text = main.get_text(separator=" ", strip=True)
        text = re.sub(r"\s+", " ", text)          # collapse whitespace
        return text[:max_chars]

    except Exception as exc:
        return f"Fetch failed: {exc}"


# ─── Tool 3: HTML Newsletter Generator ───────────────────────────────────────

def generate_html(title: str, date: str, articles: list[dict]) -> str:
    """
    Render a professional, self-contained HTML newsletter.

    Each item in `articles` should have keys: title, url, summary.
    The output is a single HTML string — ready to save as .html or send as email.
    """

    def article_card(idx: int, art: dict) -> str:
        return f"""
        <div class="card">
          <div class="card-num">#{idx:02d}</div>
          <h2 class="card-title">
            <a href="{art.get('url','#')}" target="_blank" rel="noopener">
              {art.get('title','Untitled')}
            </a>
          </h2>
          <p class="card-body">{art.get('summary','')}</p>
          <a class="card-link" href="{art.get('url','#')}" target="_blank" rel="noopener">
            Read full article →
          </a>
        </div>"""

    cards_html = "\n".join(article_card(i, a) for i, a in enumerate(articles, 1))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{title}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: 'Segoe UI', Arial, sans-serif;
      background: #f0f2f5;
      color: #2d2d2d;
      line-height: 1.65;
    }}
    .wrapper {{ max-width: 680px; margin: 32px auto; background: #fff; border-radius: 12px; overflow: hidden; box-shadow: 0 8px 32px rgba(0,0,0,.12); }}

    /* Header */
    .header {{
      background: linear-gradient(135deg, #0f0c29 0%, #302b63 55%, #24243e 100%);
      padding: 44px 36px 36px;
      text-align: center;
      color: #fff;
    }}
    .header .badge {{
      display: inline-block;
      background: rgba(255,255,255,.15);
      border: 1px solid rgba(255,255,255,.25);
      border-radius: 999px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 2px;
      padding: 4px 14px;
      margin-bottom: 18px;
      text-transform: uppercase;
    }}
    .header h1 {{
      font-size: 30px;
      font-weight: 800;
      letter-spacing: 3px;
      text-transform: uppercase;
      margin-bottom: 6px;
    }}
    .header .sub {{ opacity: .65; font-size: 13px; letter-spacing: 1.5px; }}
    .header .date {{ margin-top: 14px; font-size: 13px; opacity: .5; }}

    /* Intro banner */
    .intro {{
      background: #eef4ff;
      border-left: 5px solid #302b63;
      padding: 20px 28px;
      font-size: 14.5px;
      color: #444;
    }}

    /* Content area */
    .content {{ padding: 28px 28px 8px; }}
    .section-label {{
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: #999;
      border-bottom: 1px solid #eee;
      padding-bottom: 8px;
      margin-bottom: 20px;
    }}

    /* Article cards */
    .card {{
      border: 1px solid #e8e8e8;
      border-radius: 10px;
      padding: 22px 22px 18px;
      margin-bottom: 16px;
      transition: box-shadow .2s, border-color .2s;
    }}
    .card:hover {{
      box-shadow: 0 6px 20px rgba(48,43,99,.1);
      border-color: #c5bef7;
    }}
    .card-num {{
      font-size: 10px;
      font-weight: 800;
      color: #302b63;
      letter-spacing: 1.5px;
      margin-bottom: 8px;
      text-transform: uppercase;
    }}
    .card-title {{
      font-size: 17px;
      font-weight: 700;
      margin-bottom: 10px;
      line-height: 1.3;
    }}
    .card-title a {{ color: #1a1a2e; text-decoration: none; }}
    .card-title a:hover {{ color: #302b63; text-decoration: underline; }}
    .card-body {{ font-size: 14px; color: #555; margin-bottom: 14px; }}
    .card-link {{
      font-size: 13px;
      font-weight: 600;
      color: #302b63;
      text-decoration: none;
    }}
    .card-link:hover {{ text-decoration: underline; }}

    /* Footer */
    .footer {{
      background: #1a1a2e;
      color: rgba(255,255,255,.55);
      text-align: center;
      font-size: 12px;
      padding: 24px 28px;
      margin-top: 12px;
    }}
    .footer a {{ color: rgba(255,255,255,.4); }}
  </style>
</head>
<body>
  <div class="wrapper">

    <div class="header">
      <div class="badge">Weekly Digest</div>
      <h1>{title}</h1>
      <div class="sub">Curated AI Agent Intelligence</div>
      <div class="date">{date}</div>
    </div>

    <div class="intro">
      This week's top stories, breakthroughs, and releases from the world of autonomous AI agents —
      researched, summarised, and reviewed by your AI newsletter assistant.
    </div>

    <div class="content">
      <div class="section-label">Top Stories This Week</div>
      {cards_html}
    </div>

    <div class="footer">
      <p>Generated by <strong>Newsletter Agent</strong> · Powered by Claude &amp; LangGraph</p>
      <p style="margin-top:6px;">To unsubscribe reply <em>unsubscribe</em></p>
    </div>

  </div>
</body>
</html>"""