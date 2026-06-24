"""
config.py — Central configuration for the Newsletter Agent.
All tuneable constants live here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# # ─── Anthropic ────────────────────────────────────────────────────────────────
# ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
# MODEL: str = "claude-sonnet-4-6"

#GEMINI
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
MODEL: str = "gemini-3.1-flash-lite"
# _____META ___________________________________________________________________
# META_API_KEY: str = os.getenv("HF_TOKEN", "")
# MODEL: str = "meta-llama/Llama-3.2-3B-Instruct:fastest"
# ─── Agent Behaviour ──────────────────────────────────────────────────────────
MAX_REVISIONS: int = 2          # Max self-revision loops before forcing output
CRITIQUE_THRESHOLD: int = 7     # Minimum score (out of 10) to skip revision
MAX_ARTICLES: int = 7           # Articles to include in the newsletter
SEARCH_QUERIES: int = 4         # Number of DDG queries to run

# ─── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "newsletters")