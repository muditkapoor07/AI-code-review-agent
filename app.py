"""FastAPI web server for the Autonomous Code Review Agent."""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import traceback
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

app = FastAPI(title="AI Code Review Agent")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


# ------------------------------------------------------------------ #
# Request models                                                       #
# ------------------------------------------------------------------ #

class PRReviewRequest(BaseModel):
    pr_url: str
    model: str = "llama-3.3-70b-versatile"
    max_passes: int = 10


class SnippetReviewRequest(BaseModel):
    code: str
    language: str = "python"
    filename: str = "snippet.py"
    model: str = "llama-3.3-70b-versatile"
    max_passes: int = 10


# ------------------------------------------------------------------ #
# Routes                                                               #
# ------------------------------------------------------------------ #

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = BASE_DIR / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/api/review/pr")
async def review_pr(req: PRReviewRequest):
    return _stream_review(mode="pr", payload=req)


@app.post("/api/review/snippet")
async def review_snippet(req: SnippetReviewRequest):
    return _stream_review(mode="snippet", payload=req)


# ------------------------------------------------------------------ #
# Streaming engine                                                     #
# ------------------------------------------------------------------ #

def _stream_review(mode: str, payload) -> StreamingResponse:
    event_queue: queue.Queue[dict | None] = queue.Queue()

    def emit(event: dict) -> None:
        event_queue.put(event)

    def run_agent() -> None:
        try:
            from openai import OpenAI
            from github.client import GitHubClient
            from tools.registry import ToolRegistry
            from agent.core import CodeReviewAgent, AgentLoopError
            from utils.logger import ReviewLogger
            from rich.console import Console

            groq_key = os.getenv("GROQ_API_KEY", "")
            github_token = os.getenv("GITHUB_TOKEN", "")

            if not groq_key:
                emit({"type": "error", "message": "GROQ_API_KEY is not set in .env"})
                return

            console = Console(stderr=True)
            logger = ReviewLogger(verbose=False, console=console)
            import httpx
            groq_client = OpenAI(
                api_key=groq_key,
                base_url="https://api.groq.com/openai/v1",
                http_client=httpx.Client(verify=False),
            )

            if mode == "pr":
                if not github_token:
                    emit({"type": "error", "message": "GITHUB_TOKEN is not set in .env"})
                    return
                with GitHubClient(token=github_token) as gh_client:
                    registry = ToolRegistry(github_client=gh_client)
                    agent = CodeReviewAgent(
                        groq_client=groq_client,
                        github_client=gh_client,
                        tool_registry=registry,
                        logger=logger,
                        model=payload.model,
                        max_passes=payload.max_passes,
                        event_callback=emit,
                    )
                    review = agent.run(payload.pr_url)
            else:
                registry = ToolRegistry(github_client=None)
                agent = CodeReviewAgent(
                    groq_client=groq_client,
                    github_client=None,
                    tool_registry=registry,
                    logger=logger,
                    model=payload.model,
                    max_passes=payload.max_passes,
                    event_callback=emit,
                )
                review = agent.run_snippet(
                    code=payload.code,
                    language=payload.language,
                    filename=payload.filename,
                )

            emit({"type": "complete", "review": review.model_dump(mode="json")})

        except Exception as exc:
            emit({"type": "error", "message": str(exc), "detail": traceback.format_exc()})
        finally:
            event_queue.put(None)

    thread = threading.Thread(target=run_agent, daemon=True)
    thread.start()

    def generate():
        while True:
            try:
                event = event_queue.get(timeout=30)
            except queue.Empty:
                # keepalive ping so the connection stays open during long Groq calls
                yield ": ping\n\n"
                continue
            if event is None:
                yield "data: " + json.dumps({"type": "done"}) + "\n\n"
                break
            yield "data: " + json.dumps(event, default=str) + "\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    print(f"\n  AI Code Review Agent running at http://localhost:{port}\n")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
