# AI Code Review Agent — quick start

## 1. Install

```bash
pip install -r requirements.txt
pip install anthropic          # if LLM_PROVIDER=anthropic
# or
pip install openai             # if LLM_PROVIDER=openai or azure_openai
```

## 2. Configure

```bash
cp .env.example .env
# edit .env — fill in the block for the one provider you're using
```

## 3. Run

**Review any GitHub repo (public, or private with GITHUB_APP_TOKEN set):**
```bash
python code_review_agent.py review-repo https://github.com/owner/repo
python code_review_agent.py review-repo https://github.com/owner/repo --pr 42
python code_review_agent.py review-repo https://github.com/owner/repo --ref my-branch
```

**Review a local file directly, no GitHub involved:**
```bash
python code_review_agent.py review-files ./src/app.py
python code_review_agent.py review-files ./src/app.py ./src/utils.py
```

**Run it as a server (for GitHub webhooks, or on-demand HTTP calls):**
```bash
python code_review_agent.py serve --port 8000
```
Then either:
- point a GitHub App webhook at `http://your-host:8000/webhook/github`, or
- call it directly: `curl -X POST "http://localhost:8000/review/public-repo?repo_url=https://github.com/owner/repo"`

## Files in this project

| File | Purpose |
|---|---|
| `code_review_agent.py` | The whole agent: provider abstraction, sandbox, 6 specialist agents, orchestrator, CLI, and HTTP server |
| `master-agent-prompt.md` | The full step-by-step prompt text, same content embedded in the code above, kept as a readable reference |
| `requirements.txt` | Python dependencies |
| `.env.example` | Every environment variable the agent reads, with comments |

## Notes / known limitations

- The sandbox currently runs build/test commands as a local subprocess, not inside an isolated Docker container — do not point this at untrusted code until that's swapped in.
- Static analysis / SAST tool output (e.g. Semgrep) isn't generated automatically yet — the specialist prompts expect it as input, but nothing currently produces it.
- `review-files` mode has no build/test/profiling context by design — it's static review only, and the final report says so explicitly.
"# AI-code-review-agentVM" 
"# AI-code-review-agentVM" 
