# 🤖 AI Code Review Agent

> An autonomous, multi-pass code review agent powered by Groq LLMs — reviews GitHub PRs and code snippets for bugs, security vulnerabilities, performance issues, and code quality in real time.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.135-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=flat)](https://groq.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Render](https://img.shields.io/badge/Deployed_on-Render-46E3B7?style=flat&logo=render&logoColor=white)](https://ai-code-review-agent-r5n7.onrender.com)
[![Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=flat)](https://ai-code-review-agent-r5n7.onrender.com)

---

## 📖 Overview

The **AI Code Review Agent** is a fully autonomous code review system that investigates GitHub Pull Requests and raw code snippets using a multi-pass agentic loop. Instead of a one-shot AI comment, it behaves like a senior engineer — fetching diffs, reading full file contexts, running static analysis tools, scanning for security vulnerabilities, and iterating until it is confident in its findings.

### Why it exists

Manual code reviews are time-consuming, inconsistent, and often miss subtle security flaws or logic bugs. This agent acts as a tireless first-pass reviewer that:

- Catches **OWASP Top 10** vulnerabilities before they reach production
- Runs **static analysis** (cyclomatic complexity, maintainability index) automatically
- Scans **Python dependencies** for known CVEs via `pip-audit`
- Provides **production-ready fix suggestions** — not vague comments
- Produces a **scored, structured report** downloadable as PowerPoint or Text

### Real-world use cases

| Use Case | Description |
|---|---|
| **Pre-merge gate** | Run on every PR before human review to catch obvious issues |
| **Security audit** | Scan legacy codebases for injection, hardcoded secrets, weak crypto |
| **Onboarding** | New engineers get instant, detailed feedback on their first PRs |
| **Code snippet review** | Paste any code and get a full analysis without a GitHub repo |

---

## ✨ Features

- **Autonomous multi-pass loop** — agent decides what to investigate next, iterates until confident
- **GitHub PR review** — fetches metadata, diffs, commits, and full file content via GitHub API
- **Code snippet review** — paste any code directly into the UI without needing a GitHub repo
- **OWASP Top 10 security scanning** — injection, broken auth, XSS, SSRF, hardcoded secrets, weak crypto
- **Bandit SAST integration** — runs Python-specific security scanner on changed files
- **Dependency CVE audit** — detects vulnerable packages in `requirements.txt` via `pip-audit`
- **Cyclomatic complexity analysis** — flags high-complexity functions using `radon`
- **Pattern search engine** — regex hunts for `eval()`, `shell=True`, SQL concatenation, debug artifacts
- **Redundant code detection** — unused imports, duplicate definitions, dead code after `return`/`raise`/`break`
- **Bug detection** — mutable defaults, bare excepts, silent exception swallowing, `== None`, shadowed builtins, `is` with literals; JS: loose null checks, `eval()`, `innerHTML`, `debugger`
- **Real-time streaming UI** — live activity feed shows every tool call as it happens
- **Structured report** — scores (1–10) for Code Quality, Security, Performance, Overall with rationale
- **Download report** — export as `.pptx` PowerPoint deck or formatted `.txt` file
- **Multi-model support** — switch between Groq LLaMA 3.3 70B, LLaMA 3.1 70B, LLaMA 3.1 8B, Gemma2
- **Bulletproof output pipeline** — 5-stage recovery: extract → repair → complete-truncated-JSON → nuclear regex fallback → absolute minimal report; never crashes regardless of LLM output
- **Self-tracked tool log** — agent tracks tool calls internally, never asks LLM to reproduce them (eliminates the #1 cause of token overflow and JSON truncation)
- **Render-ready** — ships with `render.yaml` for one-click cloud deployment

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser (UI)                             │
│   Left Panel: Input (PR URL / Code)   Right Panel: Live Feed   │
│                      + Report + Download                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ HTTP POST + SSE stream
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FastAPI Server (app.py)                       │
│   /api/review/pr      /api/review/snippet                      │
│   StreamingResponse (Server-Sent Events)                        │
└───────────────────────────┬─────────────────────────────────────┘
                            │ Thread + Queue
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              CodeReviewAgent (agent/core.py)                    │
│                                                                 │
│   THINK ──► ACT (tool call) ──► OBSERVE ──► REPEAT             │
│                                                                 │
│   Uses Groq API (OpenAI-compatible) with tool_use              │
└──────────┬──────────────────────────┬───────────────────────────┘
           │                          │
           ▼                          ▼
┌──────────────────┐      ┌──────────────────────────────────────┐
│  GitHub API      │      │         Tool Registry                │
│  (github/        │      │  ┌─────────────────────────────┐    │
│   client.py)     │      │  │ fetch_pr_metadata            │    │
│                  │      │  │ fetch_pr_diff                │    │
│  - PR metadata   │      │  │ fetch_file_content           │    │
│  - File diffs    │      │  │ fetch_pr_commits             │    │
│  - File content  │      │  │ analyze_complexity  (radon)  │    │
│  - Commits       │      │  │ analyze_syntax      (ast)    │    │
└──────────────────┘      │  │ run_bandit_scan     (bandit) │    │
                          │  │ run_dependency_audit(pip-aud)│    │
                          │  │ extract_functions            │    │
                          │  │ search_patterns              │    │
                          │  │ detect_redundant_code        │    │
                          │  │ detect_bugs                  │    │
                          │  └─────────────────────────────┘    │
                          └──────────────────────────────────────┘
```

### Data flow

1. User submits a PR URL or code snippet via the browser
2. FastAPI starts a background thread and opens an SSE stream to the browser
3. `CodeReviewAgent` enters the agentic loop — calls Groq with all 13 tool schemas
4. Groq decides which tools to call; agent executes them, streams events to UI, and **tracks the tool log internally**
5. Each tool result is appended to message history (truncated to stay within token limits, sliding window of 12)
6. When Groq returns `stop` or max passes is reached, agent produces `<final_review>` JSON
7. **5-stage recovery pipeline** processes the raw output:
   - Stage 1: extract JSON from tags, strip fences, unwrap `[{...}]` list wrappers
   - Stage 2: direct parse → repair trailing commas/Python booleans → complete truncated JSON
   - Stage 3: **nuclear regex extraction** — pulls individual fields from raw text if JSON is still unparseable
   - Stage 4: `_strip_problematic_fields` normalises all field types to safe defaults
   - Stage 5: absolute fallback — strips findings and returns minimal valid report rather than crashing
8. Agent's own tool log is injected into the result (overrides any LLM-generated version)
9. Validated review is streamed to browser as a `complete` event
10. UI renders the report; user can download as `.pptx` or `.txt`

---

## 🛠 Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | Groq API — LLaMA 3.3 70B Versatile (OpenAI-compatible) |
| **Backend** | FastAPI + Uvicorn |
| **LLM Client** | `openai` SDK pointed at Groq base URL |
| **Streaming** | Server-Sent Events (SSE) via `StreamingResponse` |
| **GitHub API** | Raw `httpx` REST client (no PyGitHub dependency) |
| **SAST** | `bandit` — Python security scanner |
| **Complexity** | `radon` — cyclomatic complexity + maintainability index |
| **CVE audit** | `pip-audit` — dependency vulnerability scanner |
| **Validation** | `pydantic` v2 |
| **Frontend** | Vanilla HTML/CSS/JS — zero framework dependencies |
| **Charts/UI** | Pure CSS score bars, Rich-inspired dark theme |
| **PPTX Export** | `PptxGenJS` (browser-side, CDN) |
| **Deployment** | Render (via `render.yaml`) |

---

## ⚙️ Installation

### Prerequisites

- Python 3.11+
- A [Groq API key](https://console.groq.com) (free tier available)
- A [GitHub Personal Access Token](https://github.com/settings/tokens) with `public_repo` scope (for PR reviews)

### Clone & install

```bash
git clone https://github.com/muditkapoor07/AI-code-review-agent.git
cd AI-code-review-agent
pip install -r requirements.txt
```

### Environment variables

Copy the example and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_your_groq_api_key_here
GITHUB_TOKEN=ghp_your_github_token_here
```

**Getting a Groq API key:**
1. Sign up at [console.groq.com](https://console.groq.com)
2. Navigate to **API Keys** → **Create API Key**
3. Copy the `gsk_...` key

**Getting a GitHub token:**
1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. **Generate new token (classic)**
3. Select only the `public_repo` scope
4. Copy the `ghp_...` token

### Run locally

```bash
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

---

## 🚀 Usage

### GitHub PR Review

1. Select **GitHub PR** mode (default)
2. Paste a PR URL: `https://github.com/owner/repo/pull/123`
3. Click **Start Review**
4. Watch the live activity feed as the agent investigates
5. Review the scored report when complete
6. Download as **PowerPoint** or **Text**

```
Supported URL format:
https://github.com/{owner}/{repo}/pull/{number}
```

### Code Snippet Review

1. Select **Code Snippet** mode
2. Choose language (Python, JavaScript, TypeScript, Go, Java, etc.)
3. Set a filename (e.g. `auth.py`)
4. Paste your code
5. Click **Start Review**

> No GitHub token required for snippet reviews.

### Advanced settings

Click **> Advanced settings** in the left panel to configure:

| Setting | Default | Description |
|---|---|---|
| Model | `llama-3.3-70b-versatile` | Groq model to use |
| Max passes | `10` | Max agent investigation iterations |

### Available Groq models

| Model | Quality | Token limit/min |
|---|---|---|
| `llama-3.3-70b-versatile` | Best | 12,000 TPM |
| `llama-3.1-70b-versatile` | High | 12,000 TPM |
| `llama-3.1-8b-instant` | Fast | 6,000 TPM |
| `gemma2-9b-it` | Good | 15,000 TPM |

### CLI usage

The agent also ships with a CLI entry point:

```bash
python main.py https://github.com/psf/requests/pull/6721
python main.py https://github.com/psf/requests/pull/6721 --verbose
python main.py https://github.com/psf/requests/pull/6721 --output-json report.json
python main.py https://github.com/psf/requests/pull/6721 --max-passes 5 --model llama-3.1-8b-instant
```

---

## 📂 Project Structure

```
AI-code-review-agent/
│
├── app.py                        # FastAPI server + SSE streaming engine
├── main.py                       # CLI entry point
├── requirements.txt              # Python dependencies
├── render.yaml                   # Render deployment config
├── pyproject.toml                # Project metadata
├── .env.example                  # Environment variable template
├── .gitignore
│
├── agent/
│   ├── core.py                   # Agentic loop (THINK→ACT→OBSERVE→REPEAT)
│   ├── prompts.py                # System prompt + message templates
│   └── schemas.py                # Pydantic models for structured output
│
├── github/
│   └── client.py                 # GitHub REST API client (httpx)
│
├── tools/
│   ├── registry.py               # Tool dispatcher + OpenAI/Groq JSON schemas
│   ├── github_tools.py           # fetch_pr_metadata, fetch_pr_diff, etc.
│   ├── static_analysis_tools.py  # analyze_complexity, analyze_syntax, count_code_metrics
│   ├── security_tools.py         # run_bandit_scan, run_dependency_audit
│   └── code_tools.py             # extract_functions, search_patterns, detect_redundant_code, detect_bugs
│
├── reporting/
│   ├── renderer.py               # Rich terminal renderer (CLI mode)
│   └── models.py                 # Re-exports for reporting layer
│
├── static/
│   └── index.html                # Full SPA — UI, live feed, report renderer, download
│
└── utils/
    ├── logger.py                 # Tool call logger with Rich output
    └── subprocess_runner.py      # Safe subprocess wrapper for bandit/pip-audit
```

### Key files explained

**`agent/core.py`** — The brain. Implements the `THINK → ACT → OBSERVE → REPEAT` loop using Groq's tool-use API. Key design: tracks `tool_usage_log` internally so the LLM never wastes tokens reproducing it. 5-stage output recovery pipeline: `_extract_json_str` → `_parse_with_recovery` → `_nuclear_extract` → `_strip_problematic_fields` → absolute fallback. Separate `_MAX_TOKENS` (2048, investigation) and `_MAX_TOKENS_FINAL` (4096, final JSON) limits.

**`agent/schemas.py`** — Pydantic v2 models with coercion validators on every field. Each validator is designed to accept the widest possible LLM output (fuzzy enum matching, dict→string coercion, string→int clamping, etc.) so schema validation never crashes regardless of model inconsistencies.

**`tools/registry.py`** — Maps tool names to Python callables and generates the OpenAI-format JSON schemas passed to Groq. Sanitizes oversized `source_code` arguments to prevent token overflow.

**`app.py`** — FastAPI server with SSE streaming. Runs the agent in a background thread and forwards events to the browser via a queue. Keepalive pings prevent connection drops on long-running reviews.

**`static/index.html`** — Zero-dependency SPA. Handles SSE consumption, live feed rendering, report display, and client-side PowerPoint generation via PptxGenJS.

---

## 🔐 Security Considerations

### API key handling
- Keys are loaded from `.env` at runtime via `python-dotenv`
- `.env` is in `.gitignore` — never committed to version control
- On Render, keys are set as environment variables in the dashboard — never in code
- The `.env.example` file contains only placeholder values

### SSL on corporate networks
If you're behind a corporate SSL inspection proxy (Zscaler, Cisco Umbrella, etc.), the Groq API calls use `httpx.Client(verify=False)` to bypass SSL verification. This is intentional for local dev use. For production, replace with your corporate CA bundle:

```python
httpx.Client(verify="/path/to/corporate-ca.pem")
```

### GitHub token scope
The agent only requires the `public_repo` scope — read access to public repositories. It never writes, comments, or modifies any repository data.

### Groq API key exposure
The Groq key is used server-side only — it is never sent to the browser or included in any frontend code.

### Known limitations
- No user authentication on the web UI — anyone with the URL can run reviews
- Tool results stored in memory per request — no persistence
- Large PRs (100+ files) are automatically truncated to first 50 files

---

## ⚡ Performance Considerations

### Token management
Groq free tier limits: 100K tokens/day, 12K tokens/minute (70B model). The agent manages this by:
- Capping tool results at **1,500 characters** in message history
- Keeping only the last **12 messages** in context (sliding window)
- Truncating file content to **80 lines** and diffs to **50 lines**
- Using a compact system prompt (~150 tokens)
- **Tracking tool log internally** — LLM never wastes tokens reproducing it in the final JSON
- Separate token budgets: **2,048** tokens for investigation turns, **4,096** for final JSON generation

### Streaming
Reviews stream results in real time via SSE — the user sees tool call activity immediately rather than waiting for the full review to complete. SSE keepalive pings every 30 seconds prevent connection timeouts on Render's free tier.

### Concurrency
Each review runs in a background thread with a queue — the FastAPI event loop is never blocked. Multiple simultaneous reviews are supported (limited by Groq rate limits, not the server).

### Known bottlenecks
- **Groq free tier TPM limit** — large PRs may hit the 12K token/minute wall mid-review. Workaround: use `llama-3.1-8b-instant` or upgrade to Groq Dev tier
- **`pip-audit` startup time** — dependency audit can take 10–30 seconds on first run due to PyPI metadata fetching
- **Render free tier cold starts** — first request after inactivity may take 30–60 seconds to spin up

---

## 🧪 Testing

### Smoke test — imports

```bash
python -c "
from app import app
from agent.core import CodeReviewAgent
from tools.registry import ToolRegistry
from agent.schemas import FinalReview
print('All imports OK')
"
```

### Test static analysis tools

```python
from tools.static_analysis_tools import analyze_syntax, analyze_complexity
from tools.security_tools import run_bandit_scan

code = '''
import os
def run(cmd):
    os.system(cmd)  # command injection
SECRET = "hardcoded_password_123"
'''

print(analyze_syntax(code, "test.py"))
print(run_bandit_scan(code, "test.py"))
```

### Test Groq connectivity

```bash
python -c "
import os, httpx
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()
client = OpenAI(
    api_key=os.getenv('GROQ_API_KEY'),
    base_url='https://api.groq.com/openai/v1',
    http_client=httpx.Client(verify=False)
)
resp = client.chat.completions.create(
    model='llama-3.3-70b-versatile',
    max_tokens=32,
    messages=[{'role':'user','content':'Say hello.'}]
)
print('Groq OK:', resp.choices[0].message.content)
"
```

### Test a real PR (CLI)

```bash
python main.py https://github.com/psf/requests/pull/6721 --verbose
```

### Recommended test PRs

| PR | Why it's good for testing |
|---|---|
| `https://github.com/psf/requests/pull/6721` | Python, HTTP security, mid-sized |
| `https://github.com/pallets/flask/pulls` | Python web framework, routing changes |
| `https://github.com/gohugoio/hugo/pull/14723` | Go, path validation fix |
| `https://github.com/digininja/DVWA` | Intentionally vulnerable — security findings |

---

## 🤝 Contributing

Contributions are welcome. Please follow these guidelines:

### Branch strategy

```
main          — production-ready code
feature/xxx   — new features
fix/xxx       — bug fixes
chore/xxx     — tooling, deps, docs
```

### Setup for development

```bash
git clone https://github.com/muditkapoor07/AI-code-review-agent.git
cd AI-code-review-agent
pip install -r requirements.txt
cp .env.example .env
# Fill in your keys
uvicorn app:app --reload
```

### Adding a new tool

1. Implement the function in the appropriate `tools/*.py` file
2. Add the JSON schema to `TOOL_SCHEMAS` in `tools/registry.py`
3. Register the callable in `ToolRegistry.__init__`
4. Add a summary lambda in `agent/core.py` → `_result_summary`
5. Test with a real PR

### PR guidelines

- Keep PRs focused — one feature or fix per PR
- Test locally before opening a PR
- Update `requirements.txt` if adding dependencies (`pip freeze > requirements.txt`)
- Do not commit `.env` files

### Coding standards

- Python 3.11+ type hints throughout
- Pydantic models for all structured data
- All tool functions return `{"success": bool, "result": dict, "error": str | None}`
- No bare `except:` clauses — always catch specific exceptions

---

## 🐛 Common Issues

### `GROQ_API_KEY is not set in .env`
The server is running a cached old version. Kill all Python processes on the port and restart:
```bash
# Find the PID on port 8000
netstat -ano | grep ":8000"
# Kill it via Task Manager → Details tab → End Task by PID
# Then restart
uvicorn app:app --port 8000
```

### `SSL: CERTIFICATE_VERIFY_FAILED`
You're behind a corporate SSL proxy. The app handles this automatically with `verify=False`. If it still fails, set:
```bash
export REQUESTS_CA_BUNDLE=/path/to/corporate-ca.pem
```

### `Error 429 — Rate limit reached`
Groq free tier: 100K tokens/day, 12K tokens/minute.
- Switch model to `llama-3.1-8b-instant` in Advanced settings (uses fewer tokens)
- Or wait for the daily/minute reset
- Or upgrade to [Groq Dev tier](https://console.groq.com/settings/billing)

### `Error 413 — Request too large`
Reduce **Max passes** to 3–5 in Advanced settings. The smaller 8B model has a 6K TPM limit.

### `Agent did not produce a <final_review> block`
The model timed out or hit max passes without synthesizing. Try:
- Reduce max passes to 5
- Use `llama-3.3-70b-versatile` (more instruction-following than 8B)
- Try a smaller PR with fewer changed files

### `Final review schema validation failed` or JSON parse errors
The agent now has a **5-stage recovery pipeline** that handles every known failure mode:

| Stage | What it fixes |
|---|---|
| Extract | `<final_review>` tags, markdown fences, list-wrapped `[{...}]` JSON |
| Repair | Trailing commas, Python `None/True/False` literals |
| Complete | Truncated JSON cut off mid-output due to token limit |
| Nuclear | Regex extracts `pr_title`, `verdict`, `scores`, `findings` from raw text if JSON is still broken |
| Fallback | Returns a minimal valid report with empty findings rather than crashing |

This error should **never appear** in normal operation. If it does, it means Groq returned a completely blank response — check rate limits and try again.

### `mixtral-8x7b-32768 has been decommissioned`
This model was deprecated by Groq. Use `llama-3.3-70b-versatile` or `gemma2-9b-it` instead.

### Render cold start (30–60 second delay)
Render free tier spins down after inactivity. The first request after sleep takes 30–60 seconds. This is normal — subsequent requests are fast.

---

## 🗺 Roadmap

### Near-term
- [x] **Bulletproof 5-stage output pipeline** — never crashes; nuclear regex fallback extracts fields from any output
- [x] **Self-tracked tool log** — eliminates #1 source of JSON truncation (LLM no longer wastes tokens reproducing it)
- [x] **Redundant code detection** — unused imports, duplicate definitions, dead code
- [x] **Bug detection** — mutable defaults, bare excepts, silent swallow, None comparisons, shadowed builtins
- [ ] **Multi-file PR support** — smarter file prioritization for PRs with 50+ files
- [ ] **Comment posting** — post review findings directly as GitHub PR comments
- [ ] **JavaScript/TypeScript SAST** — integrate `eslint` and `semgrep` for non-Python files
- [ ] **Webhook mode** — auto-trigger on PR open/update via GitHub webhooks
- [ ] **Review history** — persist past reviews with SQLite

### Medium-term
- [ ] **Custom rule sets** — user-defined patterns and severity thresholds
- [ ] **Team dashboard** — aggregate review scores across repos over time
- [ ] **PR diff comparison** — compare two versions of the same PR
- [ ] **Anthropic Claude support** — switchable LLM backend
- [ ] **OpenAI GPT-4o support** — switchable LLM backend

### Long-term
- [ ] **VS Code extension** — inline review comments in the editor
- [ ] **CI/CD integration** — GitHub Actions workflow that fails on CRITICAL findings
- [ ] **Fine-tuned model** — domain-specific model trained on security-focused code reviews
- [ ] **Enterprise SSO** — authentication for team deployments

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

```
MIT License

Copyright (c) 2025 Mudit Kapoor

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙌 Acknowledgements

| Library / Tool | Purpose |
|---|---|
| [Groq](https://groq.com) | Ultra-fast LLM inference API |
| [FastAPI](https://fastapi.tiangolo.com) | Modern Python web framework |
| [Bandit](https://bandit.readthedocs.io) | Python SAST security scanner |
| [Radon](https://radon.readthedocs.io) | Python code complexity metrics |
| [pip-audit](https://pypi.org/project/pip-audit) | Python dependency CVE scanner |
| [PptxGenJS](https://gitbrent.github.io/PptxGenJS) | Client-side PowerPoint generation |
| [Rich](https://rich.readthedocs.io) | Terminal formatting for CLI mode |
| [Pydantic](https://docs.pydantic.dev) | Data validation and serialisation |
| [httpx](https://www.python-httpx.org) | Async-capable HTTP client |
| [OWASP Top 10](https://owasp.org/www-project-top-ten) | Security vulnerability taxonomy |

---

<div align="center">

**Built with ❤️ by [Mudit Kapoor](https://github.com/muditkapoor07)**

[Live Demo](https://ai-code-review-agent-r5n7.onrender.com) · [Report a Bug](https://github.com/muditkapoor07/AI-code-review-agent/issues) · [Request a Feature](https://github.com/muditkapoor07/AI-code-review-agent/issues)

</div>
