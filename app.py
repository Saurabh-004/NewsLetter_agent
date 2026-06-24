"""
app.py — FastAPI backend for the Newsletter Agent.

Endpoints:
  POST /api/run              → Start a new agent run (returns run_id)
  GET  /api/stream/{run_id}  → SSE stream of real-time execution events
  POST /api/approve/{run_id} → Submit HITL approval / rejection
  GET  /api/status/{run_id}  → Poll run status (running / waiting / complete / error)
  GET  /                     → Serve the frontend SPA
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from agent import newsletter_graph, NewsletterState
from config import OUTPUT_DIR

# ─── App Setup ────────────────────────────────────────────────────────────────

APP_DIR = Path(__file__).resolve().parent
FRONTEND_PATH = APP_DIR / "index.html"

app = FastAPI(title="Newsletter Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── In-memory Run Registry ───────────────────────────────────────────────────
# { run_id: RunContext }
runs: dict[str, dict] = {}


# ─── Request / Response Models ────────────────────────────────────────────────

class RunRequest(BaseModel):
    goal: str
    mode: str = "autonomous"   # "autonomous" | "hitl"


class ApprovalRequest(BaseModel):
    approved: bool
    modified_queries: Optional[list[str]] = None


# ─── Streaming Worker ─────────────────────────────────────────────────────────

async def _graph_worker(
    run_id: str,
    initial_state: dict,
    log_queue: asyncio.Queue,
    approval_queue: asyncio.Queue,
):
    """
    Run the LangGraph agent in the background.

    • Streams state-update chunks to log_queue for the SSE endpoint.
    • On HITL interrupt: sends an "interrupt" event and waits on approval_queue.
    • Resumes the graph with Command(resume=approval_data).
    """
    from langgraph.types import Command

    config = {"configurable": {"thread_id": run_id}}
    current_input = initial_state

    try:
        while True:
            hit_interrupt = False

            async for chunk in newsletter_graph.astream(
                current_input, config, stream_mode="updates"
            ):
                for node_name, update in chunk.items():

                    # ── HITL interrupt ───────────────────────────────────────
                    if node_name == "__interrupt__":
                        interrupt_value = update
                        # LangGraph wraps the value; unwrap if needed
                        if isinstance(interrupt_value, (list, tuple)) and interrupt_value:
                            interrupt_value = interrupt_value[0].value

                        await log_queue.put({"type": "interrupt", "data": interrupt_value})
                        runs[run_id]["status"] = "waiting_approval"

                        # Block until human submits approval
                        approval = await approval_queue.get()
                        current_input = Command(resume=approval)
                        runs[run_id]["status"] = "running"
                        hit_interrupt = True
                        break   # break inner for-loop, restart astream

                    if node_name == "__end__":
                        continue

                    # ── Regular node update ──────────────────────────────────
                    # Stream individual log lines
                    for line in update.get("status_log", []):
                        await log_queue.put({"type": "log", "message": line})

                    # Notify which node just finished
                    await log_queue.put({
                        "type": "node_complete",
                        "node": node_name,
                        "meta": {
                            k: v for k, v in update.items()
                            if k not in {"status_log", "newsletter_draft",
                                         "raw_articles", "final_newsletter"}
                        },
                    })

                    # Send newsletter HTML preview as soon as first draft exists
                    if "newsletter_draft" in update and update["newsletter_draft"]:
                        await log_queue.put({
                            "type": "newsletter_preview",
                            "html": update["newsletter_draft"],
                        })

                    # Final newsletter ready
                    if "final_newsletter" in update and update["final_newsletter"]:
                        await log_queue.put({
                            "type": "final",
                            "html":  update["final_newsletter"],
                            "path":  update.get("output_path", ""),
                            "score": update.get("critique_score", "N/A"),
                        })

            if not hit_interrupt:
                break

    except Exception as exc:
        await log_queue.put({"type": "error", "message": str(exc)})
        runs[run_id]["status"] = "error"

    else:
        runs[run_id]["status"] = "complete"

    finally:
        await log_queue.put({"type": "done"})


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/api/run")
async def start_run(req: RunRequest):
    """
    Kick off a new newsletter agent run.

    Returns { run_id, status } immediately.
    The client should then open the SSE stream at /api/stream/{run_id}.
    """
    run_id = str(uuid.uuid4())
    log_queue: asyncio.Queue = asyncio.Queue()
    approval_queue: asyncio.Queue = asyncio.Queue()

    initial_state: NewsletterState = {
        "goal":             req.goal,
        "mode":             req.mode,
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
        "status_log": [
            f"[{datetime.now().strftime('%H:%M:%S')}] "
            f"🚀 Newsletter Agent starting in {req.mode.upper()} mode…"
        ],
    }

    task = asyncio.create_task(
        _graph_worker(run_id, initial_state, log_queue, approval_queue)
    )

    runs[run_id] = {
        "task":           task,
        "log_queue":      log_queue,
        "approval_queue": approval_queue,
        "status":         "running",
        "mode":           req.mode,
    }

    return JSONResponse(
        {"run_id": run_id, "status": "started", "mode": req.mode}
    )


@app.get("/api/health")
async def health():
    """Lightweight check that the API is reachable."""
    return {"status": "ok"}


@app.get("/api/stream/{run_id}")
async def stream_run(run_id: str):
    """
    Server-Sent Events endpoint.
    The client connects here after /api/run and receives real-time updates.
    """
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")

    log_queue = runs[run_id]["log_queue"]

    async def generator():
        while True:
            try:
                event = await asyncio.wait_for(log_queue.get(), timeout=60.0)
                yield f"data: {json.dumps(event)}\n\n"
                if event["type"] == "done":
                    break
            except asyncio.TimeoutError:
                # Heartbeat to keep connection alive
                yield f"data: {json.dumps({'type': 'ping'})}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/approve/{run_id}")
async def approve(run_id: str, req: ApprovalRequest):
    """
    Submit a HITL approval or rejection for a paused run.
    The graph worker is unblocked and continues execution.
    """
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")
    if runs[run_id]["status"] != "waiting_approval":
        return {"detail": "Run is not waiting for approval"}

    payload = {"approved": req.approved}
    if req.modified_queries:
        payload["modified_queries"] = req.modified_queries

    await runs[run_id]["approval_queue"].put(payload)
    return {"status": "resumed", "approved": req.approved}


@app.get("/api/status/{run_id}")
async def get_status(run_id: str):
    """Poll-based status check (complement to SSE)."""
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")
    ctx = runs[run_id]
    return {"run_id": run_id, "status": ctx["status"], "mode": ctx["mode"]}


@app.get("/")
async def serve_frontend():
    """Serve the single-page frontend."""
    if not FRONTEND_PATH.exists():
        return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)
    html = FRONTEND_PATH.read_text(encoding="utf-8")
    # Same-origin API when UI is served by this app (empty string = relative URLs).
    injection = '<script>window.__API_BASE__="";</script>'
    html = html.replace("<head>", f"<head>\n  {injection}", 1)
    return HTMLResponse(html)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    reload_flag = os.getenv("UVICORN_RELOAD", "false").lower() in {"1", "true", "yes"}
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=reload_flag)