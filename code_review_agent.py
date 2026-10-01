"""
AI Code Review Agent — reference implementation, provider-agnostic.

Single-file, runnable skeleton implementing the Orchestrator + tool loop
described in master-agent-prompt.md. Reviews any GitHub repo/PR (Mode A)
or manually submitted files (Mode B), for any language/build system,
using ANY LLM provider with tool/function calling support.

Install:
    pip install fastapi uvicorn requests python-dotenv PyGithub
    # plus whichever provider SDK(s) you'll actually use:
    pip install anthropic      # for Claude
    pip install openai         # for OpenAI, Azure OpenAI, and any
                                # OpenAI-compatible endpoint (Ollama,
                                # vLLM, LocalAI, LM Studio, etc.)

Choose your provider via environment variables — nothing in the code
below needs to change:

    LLM_PROVIDER=anthropic
        ANTHROPIC_API_KEY=...
        ANTHROPIC_MODEL=claude-sonnet-4-6            (optional, has a default)

    LLM_PROVIDER=openai
        OPENAI_API_KEY=...
        OPENAI_MODEL=gpt-4o                            (optional, has a default)
        OPENAI_BASE_URL=...                             (optional — set this to
            point at Ollama/vLLM/LocalAI/any OpenAI-compatible server instead
            of api.openai.com, e.g. http://localhost:11434/v1 for Ollama)

    LLM_PROVIDER=azure_openai
        AZURE_OPENAI_API_KEY=...
        AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com
        AZURE_OPENAI_DEPLOYMENT=...                     (your deployment name)
        AZURE_OPENAI_API_VERSION=2024-10-21             (optional, has a default)

    GITHUB_APP_TOKEN=...   (optional — only for private repos / posting comments)

Run:
    uvicorn code_review_agent:app --reload --port 8000
"""

import hashlib
import hmac
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Optional

import requests
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from github import Auth as GithubAuth
from github import Github
from pydantic import BaseModel

try:
    from dotenv import load_dotenv
    load_dotenv()  # loads .env if present; harmless no-op otherwise
except ImportError:
    pass

# Windows consoles default to cp1252, which can't print the emoji/arrows that
# appear in reports and log lines. Force UTF-8 so a final `print()` never
# crashes a review that has otherwise succeeded.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("code_review_agent")

MAX_ORCHESTRATOR_TURNS = 20  # hard cap; pairs with the 10-min wall-clock budget in the prompt
ORCHESTRATOR_MAX_TOKENS = 16000  # tool-call args (e.g. a pasted file) must fit in one response
SPECIALIST_MAX_TOKENS = 8000     # a full findings list for a large file needs well over 3k tokens
TOOL_RESULT_MAX_CHARS = 30_000   # cap on what a tool result feeds back to the orchestrator
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "8"))  # 429/5xx retries with exponential backoff

# Appended to every specialist prompt. Without it the model tries to list
# every nit in a 1000-line file and runs past max_tokens, which truncates
# the JSON and loses *all* findings — including the important ones.
SPECIALIST_OUTPUT_RULES = """

Output budget — strictly enforced:
- Report at most 12 findings. Prioritize by severity; drop the rest.
- Keep every string field to one or two sentences. No code blocks.
- Your entire reply must be under 5000 tokens. If in doubt, cut nits.
- Emit complete, valid JSON. Truncated JSON is worthless."""

# Sent back to the orchestrator when it narrates ("I will now call X...")
# instead of actually emitting a tool call. Without this the loop would
# treat that narration as the final answer and exit after one turn.
CONTINUE_NUDGE = (
    "Do not describe what you will do — do it. Call the next required tool now. "
    "Only reply with plain text after generate_report has returned, and then "
    "reply with exactly its markdown_comment_body."
)


# ═══════════════════════════════════════════════════════════════════════
# LLM provider abstraction — this is the whole point of this section.
# Every provider implements the same two operations:
#   1. complete_json(system, payload)  -> a specialist's structured reply
#   2. run_agentic_loop(system, tools, tool_impl, user_message) -> final text,
#      driving the Orchestrator's tool-call loop to completion.
# Tools are defined once, in a neutral schema (see TOOLS below), and each
# provider adapts them to its own wire format internally.
# ═══════════════════════════════════════════════════════════════════════
class LLMProvider(ABC):
    @abstractmethod
    def complete_json(self, system_prompt: str, payload: dict, max_tokens: int = SPECIALIST_MAX_TOKENS) -> dict:
        """Single-shot call, no tools — used by every specialist agent."""

    @abstractmethod
    def run_agentic_loop(
        self,
        system_prompt: str,
        tools: list[dict],
        tool_impl: dict[str, Callable],
        user_message: str,
    ) -> str:
        """Multi-turn tool-calling loop — used by the Orchestrator only."""


def _parse_json_text(text: str) -> dict:
    text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "failed_to_parse_json", "raw_response": text[:2000]}


def _run_tool(tool_impl: dict[str, Callable], name: str, args: dict) -> dict:
    fn = tool_impl.get(name)
    if not fn:
        return {"error": f"unknown tool: {name}"}
    try:
        return fn(**args)
    except Exception as exc:  # noqa: BLE001 — surfaced to the model, not swallowed
        return {"error": str(exc)}


# --------------------------------------------------------------------- #
# Anthropic (Claude)
# --------------------------------------------------------------------- #
class AnthropicProvider(LLMProvider):
    def __init__(self):
        import anthropic
        # ANTHROPIC_BASE_URL is optional: leave unset to use Anthropic's own
        # API directly, or point it at an Azure AI Foundry Claude deployment
        # (e.g. https://<your-resource>.services.ai.azure.com/anthropic) —
        # Azure serves Claude through this same Messages API format, so no
        # other code changes are needed, just this URL + your Azure API key.
        base_url = os.environ.get("ANTHROPIC_BASE_URL")  # None = api.anthropic.com
        # max_retries: shared/org-level quotas (e.g. Foundry) throw bursts of
        # 429s; the default of 2 retries gives up too quickly and loses the
        # whole review. Backoff is exponential, so 8 retries ≈ a few minutes.
        self.client = anthropic.Anthropic(
            api_key=os.environ["ANTHROPIC_API_KEY"], base_url=base_url, max_retries=LLM_MAX_RETRIES
        )
        # On Azure, "model" must be your deployment name (e.g. claude-fable-5-1),
        # not Anthropic's own model id — Azure routes by deployment name.
        self.model = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")

    def complete_json(self, system_prompt: str, payload: dict, max_tokens: int = SPECIALIST_MAX_TOKENS) -> dict:
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": json.dumps(payload)[:100_000]}],
        )
        if msg.stop_reason == "max_tokens":
            log.warning("Specialist output truncated at max_tokens=%d; JSON may be incomplete.", max_tokens)
        text = "".join(b.text for b in msg.content if b.type == "text")
        return _parse_json_text(text)

    @staticmethod
    def _to_claude_tools(tools: list[dict]) -> list[dict]:
        return [
            {"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
            for t in tools
        ]

    def run_agentic_loop(self, system_prompt, tools, tool_impl, user_message) -> str:
        claude_tools = self._to_claude_tools(tools)
        messages: list[dict] = [{"role": "user", "content": user_message}]
        report_done = False

        for _ in range(MAX_ORCHESTRATOR_TURNS):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=ORCHESTRATOR_MAX_TOKENS,
                system=system_prompt,
                tools=claude_tools,
                messages=messages,
            )
            tool_blocks = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason == "max_tokens":
                log.warning("Orchestrator hit max_tokens (%d); tool args may be truncated.", ORCHESTRATOR_MAX_TOKENS)

            if not tool_blocks:
                text = "".join(b.text for b in response.content if b.type == "text")
                if report_done:
                    return text
                # Model narrated instead of acting — push it back into the loop.
                log.info("Orchestrator replied with text before generate_report; nudging it to continue.")
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": CONTINUE_NUDGE})
                continue

            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in tool_blocks:
                log.info("Orchestrator -> %s", block.name)
                result = _run_tool(tool_impl, block.name, block.input)
                if block.name == "generate_report":
                    report_done = True
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)[:TOOL_RESULT_MAX_CHARS]}
                )
            messages.append({"role": "user", "content": tool_results})

        return "Review did not complete within the turn budget."


# --------------------------------------------------------------------- #
# OpenAI-compatible: OpenAI direct, Azure OpenAI, or any local/open
# source server exposing the OpenAI chat-completions API (Ollama, vLLM,
# LocalAI, LM Studio, etc.) — same code path, different client + base_url.
# --------------------------------------------------------------------- #
class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, client, model: str):
        self.client = client
        self.model = model

    @staticmethod
    def _to_openai_tools(tools: list[dict]) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["parameters"],
                },
            }
            for t in tools
        ]

    def complete_json(self, system_prompt: str, payload: dict, max_tokens: int = SPECIALIST_MAX_TOKENS) -> dict:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(payload)[:100_000]},
            ],
        )
        if response.choices[0].finish_reason == "length":
            log.warning("Specialist output truncated at max_tokens=%d; JSON may be incomplete.", max_tokens)
        return _parse_json_text(response.choices[0].message.content or "")

    def run_agentic_loop(self, system_prompt, tools, tool_impl, user_message) -> str:
        openai_tools = self._to_openai_tools(tools)
        messages: list[dict] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        report_done = False
        for _ in range(MAX_ORCHESTRATOR_TURNS):
            response = self.client.chat.completions.create(
                model=self.model,
                max_tokens=ORCHESTRATOR_MAX_TOKENS,
                messages=messages,
                tools=openai_tools,
            )
            choice = response.choices[0]
            message = choice.message

            if choice.finish_reason != "tool_calls" or not message.tool_calls:
                if report_done:
                    return message.content or ""
                log.info("Orchestrator replied with text before generate_report; nudging it to continue.")
                messages.append({"role": "assistant", "content": message.content or ""})
                messages.append({"role": "user", "content": CONTINUE_NUDGE})
                continue

            messages.append(
                {
                    "role": "assistant",
                    "content": message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in message.tool_calls
                    ],
                }
            )
            for tc in message.tool_calls:
                args = json.loads(tc.function.arguments or "{}")
                log.info("Orchestrator -> %s", tc.function.name)
                result = _run_tool(tool_impl, tc.function.name, args)
                if tc.function.name == "generate_report":
                    report_done = True
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)[:TOOL_RESULT_MAX_CHARS]}
                )

        return "Review did not complete within the turn budget."


def _build_openai_provider() -> LLMProvider:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ.get("OPENAI_BASE_URL"),  # None = api.openai.com; set for Ollama/vLLM/etc.
        max_retries=LLM_MAX_RETRIES,
    )
    return OpenAICompatibleProvider(client, os.environ.get("OPENAI_MODEL", "gpt-4o"))


def _build_azure_openai_provider() -> LLMProvider:
    from openai import AzureOpenAI
    client = AzureOpenAI(
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        max_retries=LLM_MAX_RETRIES,
    )
    # Azure addresses models by deployment name, passed as the "model" param
    return OpenAICompatibleProvider(client, os.environ["AZURE_OPENAI_DEPLOYMENT"])


def get_provider() -> LLMProvider:
    provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
    if provider == "anthropic":
        return AnthropicProvider()
    if provider == "openai":
        return _build_openai_provider()
    if provider == "azure_openai":
        return _build_azure_openai_provider()
    raise ValueError(
        f"Unknown LLM_PROVIDER '{provider}'. Use 'anthropic', 'openai', or 'azure_openai' "
        "— or add your own LLMProvider subclass for another backend."
    )


_llm_singleton: Optional[LLMProvider] = None


def get_llm() -> LLMProvider:
    """Lazy singleton: config is only validated the first time an LLM call
    is actually needed, so `--help` and `serve --help` work even before
    you've set any provider environment variables."""
    global _llm_singleton
    if _llm_singleton is None:
        try:
            _llm_singleton = get_provider()
        except KeyError as exc:
            raise SystemExit(
                f"Missing required environment variable: {exc}.\n"
                "Copy .env.example to .env, fill in your chosen provider's "
                "keys, and make sure it's loaded (this file auto-loads .env "
                "if python-dotenv is installed)."
            ) from exc
    return _llm_singleton


app = FastAPI(title="AI Code Review Agent")


# ═══════════════════════════════════════════════════════════════════════
# Build-system detection (language-agnostic — add a row to support a
# new stack, never touch a prompt)
# ═══════════════════════════════════════════════════════════════════════
# Directories that are never source-under-review (VCS internals, deps, builds).
SKIP_DIRS = {".git", ".hg", ".svn", "node_modules", ".venv", "venv", "__pycache__",
             "dist", "build", "target", ".idea", ".vscode", ".tox", ".mypy_cache", ".pytest_cache"}

BUILD_DETECTION = [
    {"signal": "pom.xml", "language": "Java (Maven)", "build": "mvn -B compile", "test": "mvn -B test"},
    {"signal": "build.gradle", "language": "Java/Kotlin (Gradle)", "build": "./gradlew build -x test", "test": "./gradlew test"},
    {"signal": "package.json", "language": "JavaScript/TypeScript", "build": "npm ci", "test": "npm test --if-present"},
    {"signal": "requirements.txt", "language": "Python (pip)", "build": "pip install -r requirements.txt", "test": "pytest"},
    {"signal": "pyproject.toml", "language": "Python (Poetry)", "build": "poetry install", "test": "poetry run pytest"},
    {"signal": "go.mod", "language": "Go", "build": "go build ./...", "test": "go test ./..."},
    {"signal": "Cargo.toml", "language": "Rust", "build": "cargo build", "test": "cargo test"},
    {"signal": "Gemfile", "language": "Ruby", "build": "bundle install", "test": "bundle exec rspec"},
    {"signal": "composer.json", "language": "PHP", "build": "composer install", "test": "phpunit"},
    {"signal": "CMakeLists.txt", "language": "C/C++ (CMake)", "build": "cmake --build .", "test": "ctest"},
]


def detect_build_system(repo_path: Path) -> Optional[dict]:
    """Find the build signal file closest to the repo root and return the
    matching BUILD_DETECTION entry plus `project_dir` — the directory the
    build/test commands must run from (a project may live in a subfolder
    like `repo/my-service/requirements.txt`)."""
    best: Optional[tuple[int, dict, Path]] = None
    for entry in BUILD_DETECTION:
        for hit in repo_path.glob(f"**/{entry['signal']}"):
            rel = hit.relative_to(repo_path)
            if any(part in SKIP_DIRS for part in rel.parts[:-1]):
                continue
            depth = len(rel.parts) - 1
            # Prefer shallowest; on a tie keep BUILD_DETECTION's order.
            if best is None or depth < best[0]:
                best = (depth, entry, hit.parent)
    if best is None:
        return None
    _, entry, project_dir = best
    return {**entry, "project_dir": str(project_dir),
            "project_subdir": str(project_dir.relative_to(repo_path)).replace("\\", "/") or "."}


def run_cmd(cmd: str, cwd: Path, timeout: int) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout)


# ═══════════════════════════════════════════════════════════════════════
# Tool: github_clone_and_get_context  (Mode A — any GitHub repo)
# ═══════════════════════════════════════════════════════════════════════
_GITHUB_URL_RE = re.compile(r"^https://github\.com/([\w.-]+)/([\w.-]+?)(?:\.git)?/?$")
_GIT_REF_RE = re.compile(r"^[\w./-]+$")
MAX_DIFF_CHARS = 40_000


def github_clone_and_get_context(
    repo_url: str, ref: Optional[str] = None, pr_number: Optional[int] = None
) -> dict:
    # Only accept https://github.com/<owner>/<repo> — rejects file://, SSH,
    # internal hosts, and anything else a prompt-injected model might try.
    m = _GITHUB_URL_RE.match(repo_url.strip())
    if not m:
        return {"error": f"repo_url must look like https://github.com/<owner>/<repo>, got: {repo_url[:200]}"}
    owner_repo = f"{m.group(1)}/{m.group(2)}"
    clean_url = f"https://github.com/{owner_repo}.git"

    if ref and not _GIT_REF_RE.match(ref):
        return {"error": f"invalid ref: {ref[:100]}"}

    work_dir = Path(tempfile.mkdtemp(prefix="review_"))
    clone_target = work_dir / "repo"
    # GIT_TERMINAL_PROMPT=0: fail fast on private repos instead of hanging on a
    # credential prompt that nobody is there to answer.
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}

    try:
        clone = subprocess.run(
            ["git", "clone", "--depth", "50", "--no-tags", clean_url, str(clone_target)],
            capture_output=True, text=True, timeout=180, env=env,
        )
    except FileNotFoundError:
        return {"error": "git is not installed or not on PATH"}
    except subprocess.TimeoutExpired:
        return {"error": "clone timed out after 180s"}
    if clone.returncode != 0:
        return {"error": f"clone failed: {clone.stderr[-500:]}"}

    if ref:
        # Shallow clones only have the default branch; fetch the ref explicitly.
        subprocess.run(["git", "-C", str(clone_target), "fetch", "--depth", "50", "origin", ref],
                       capture_output=True, text=True, timeout=120, env=env)
        co = subprocess.run(["git", "-C", str(clone_target), "checkout", "--", ref],
                            capture_output=True, text=True, timeout=30, env=env)
        if co.returncode != 0:
            co = subprocess.run(["git", "-C", str(clone_target), "checkout", "FETCH_HEAD"],
                                capture_output=True, text=True, timeout=30, env=env)
        if co.returncode != 0:
            return {"error": f"checkout of ref '{ref}' failed: {co.stderr[-300:]}"}

    head = subprocess.run(["git", "-C", str(clone_target), "rev-parse", "HEAD"],
                          capture_output=True, text=True, timeout=10, env=env)
    commit_sha = head.stdout.strip() if head.returncode == 0 else None

    diff, write_access, diff_error = "", False, None
    gh_token = os.environ.get("GITHUB_APP_TOKEN")

    if pr_number:
        try:
            if gh_token:
                gh = Github(auth=GithubAuth.Token(gh_token))
                gh_repo = gh.get_repo(owner_repo)
                pr = gh_repo.get_pull(pr_number)
                resp = requests.get(pr.diff_url, headers={"Authorization": f"token {gh_token}"}, timeout=30)
                write_access = bool(getattr(gh_repo.permissions, "push", False))
                commit_sha = pr.head.sha or commit_sha
            else:
                resp = requests.get(f"https://github.com/{owner_repo}/pull/{pr_number}.diff", timeout=30)
            if resp.ok:
                diff = resp.text
            else:
                diff_error = f"PR diff fetch returned HTTP {resp.status_code}"
        except Exception as exc:  # noqa: BLE001 — surfaced to the orchestrator, not swallowed
            diff_error = f"PR diff fetch failed: {type(exc).__name__}: {exc}"[:300]

    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + f"\n... [diff truncated at {MAX_DIFF_CHARS} chars]"

    detection = detect_build_system(clone_target)
    return {
        "local_path": str(clone_target),
        "owner_repo": owner_repo,
        "commit_sha": commit_sha,
        "pr_number": pr_number,
        "diff": diff,
        "diff_error": diff_error,
        "changed_files": _files_from_diff(diff) if diff else [],
        "write_access": write_access,
        "detected_build_system": detection["language"] if detection else None,
        "project_subdir": detection["project_subdir"] if detection else ".",
    }


# ═══════════════════════════════════════════════════════════════════════
# Tool: receive_manual_submission  (Mode B — no GitHub repo at all)
# ═══════════════════════════════════════════════════════════════════════
LANGUAGE_BY_EXTENSION = {
    ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript", ".java": "Java",
    ".go": "Go", ".rs": "Rust", ".rb": "Ruby", ".php": "PHP", ".cs": "C#",
    ".cpp": "C++", ".c": "C", ".kt": "Kotlin", ".swift": "Swift",
}


def stage_manual_files(files: list[dict], language_hint: Optional[str] = None) -> dict:
    """Write submitted files to a temp dir. Called directly by the CLI/API
    (not through the LLM), so file contents never round-trip through the
    model's tool-call arguments — which is what blew the token budget."""
    work_dir = Path(tempfile.mkdtemp(prefix="manual_"))
    written = []
    for f in files:
        file_path = work_dir / f["filename"]
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(f["content"], encoding="utf-8")
        written.append(f["filename"])

    guessed_language = language_hint
    if not guessed_language and written:
        guessed_language = LANGUAGE_BY_EXTENSION.get(Path(written[0]).suffix, "unknown")

    return {"local_path": str(work_dir), "files": written, "language": guessed_language, "mode": "manual"}


def receive_manual_submission(local_path: str, files: Optional[list] = None, language_hint: Optional[str] = None) -> dict:
    """Tool entry point. Files are already staged; just confirm the context."""
    path = Path(local_path)
    if not path.is_dir():
        return {"error": f"local_path does not exist: {local_path}"}
    if not files:
        files = [str(p.relative_to(path)) for p in path.rglob("*") if p.is_file()]
    guessed_language = language_hint
    if not guessed_language and files:
        guessed_language = LANGUAGE_BY_EXTENSION.get(Path(files[0]).suffix, "unknown")
    return {"local_path": str(path), "files": files, "language": guessed_language or "unknown", "mode": "manual"}


MAX_SOURCE_CHARS_PER_FILE = 60_000
MAX_SOURCE_CHARS_TOTAL = 80_000      # complete_json caps the payload at 100K chars
MAX_SOURCE_FILES = 40
SOURCE_EXTENSIONS = set(LANGUAGE_BY_EXTENSION) | {
    ".jsx", ".tsx", ".mjs", ".cjs", ".scala", ".sh", ".ps1", ".sql", ".yaml", ".yml",
    ".toml", ".json", ".md", ".txt", ".cfg", ".ini", ".html", ".css", ".dockerfile",
}


def _iter_source_files(root: Path):
    """Yield reviewable source files under root, skipping VCS/deps/binaries."""
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.is_symlink():
            continue
        rel_parts = p.relative_to(root).parts
        if any(part in SKIP_DIRS for part in rel_parts[:-1]):
            continue
        if p.suffix.lower() in SOURCE_EXTENSIONS or p.name in ("Dockerfile", "Makefile", "requirements.txt"):
            yield p


def _files_from_diff(diff: str) -> list[str]:
    """Pull the changed-file paths out of a unified diff (`+++ b/path`)."""
    files = []
    for line in diff.splitlines():
        if line.startswith("+++ b/"):
            files.append(line[6:].strip())
    return files


def _load_sources(local_path: Optional[str], files: Optional[list] = None, diff: Optional[str] = None) -> dict[str, str]:
    """Read source files so specialists get real code, not a path.

    Priority: explicit `files` → files named in `diff` → all source files in
    the tree (bounded). Always skips VCS internals, dependency folders and
    binaries, and stays under the payload cap."""
    if not local_path:
        return {}
    root = Path(local_path)
    if not root.is_dir():
        return {}

    names = list(files or [])
    if not names and diff:
        names = _files_from_diff(diff)
    if names:
        candidates = [root / n for n in names]
    else:
        candidates = list(_iter_source_files(root))

    out: dict[str, str] = {}
    total = 0
    for p in candidates:
        if not p.is_file():
            continue
        try:
            text = p.read_text(errors="replace")[:MAX_SOURCE_CHARS_PER_FILE]
        except OSError:
            continue
        if "\x00" in text[:1000]:  # binary sniff
            continue
        if total + len(text) > MAX_SOURCE_CHARS_TOTAL or len(out) >= MAX_SOURCE_FILES:
            out["__truncated__"] = f"{len(candidates) - len(out)} more file(s) omitted to fit the payload budget."
            break
        out[str(p.relative_to(root)).replace("\\", "/")] = text
        total += len(text)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Tool: run_sandbox_agent  (Mode A only — build, test, interpret)
# ═══════════════════════════════════════════════════════════════════════
SANDBOX_INTERPRETATION_PROMPT = """You are interpreting raw build and test output from an isolated sandbox
run. You are not executing anything yourself — you are given the exit
codes and raw logs from a run that already happened, and your job is
to produce a clean, structured summary.

Your job:
1. State clearly whether the build succeeded or failed, and whether
   tests succeeded, failed, or were skipped because the build failed.
2. If the build failed, extract the specific compiler/interpreter
   error (file, line, message) from the raw log.
3. If tests failed, list each failing test by name with its failure
   reason, distinguishing assertion failures from errors/exceptions
   from timeouts.
4. If the run hit a timeout or resource limit, say so explicitly.
5. Do not speculate about the cause of a failure beyond what the log
   evidence shows.

Return ONLY valid JSON:
{
  "build_status": "success" | "failure" | "timeout" | "undetected",
  "build_error": {"file": "string", "line": 0, "message": "string"} or null,
  "test_status": "success" | "failure" | "skipped_due_to_build" | "timeout",
  "test_summary": {"passed": 0, "failed": 0, "skipped": 0},
  "failing_tests": [{"name": "string", "type": "assertion|error|timeout", "reason": "string"}],
  "resource_limit_hit": false,
  "coverage_percent_changed_lines": null,
  "sandbox_summary": "1-2 sentences for a developer"
}"""


SANDBOX_MAX_TIMEOUT = 900


def _tool_available(cmd: str) -> bool:
    """Is the first word of a build/test command actually on PATH?"""
    first = cmd.split()[0]
    if first in ("./gradlew", "gradlew"):
        return True  # repo-provided wrapper, checked at run time
    return shutil.which(first) is not None


def run_sandbox_agent(local_path: str, timeout_seconds: int = 600) -> dict:
    path = Path(local_path)
    if not path.is_dir():
        return {"build_status": "failure", "sandbox_summary": f"local_path does not exist: {local_path}"}
    timeout_seconds = max(30, min(int(timeout_seconds or 600), SANDBOX_MAX_TIMEOUT))

    detection = detect_build_system(path)
    if not detection:
        return {"build_status": "undetected", "sandbox_summary": "No recognized build system signal file found."}

    # Run build/test from the folder that actually holds the build file —
    # projects often live in a subdirectory of the repo.
    project_dir = Path(detection["project_dir"])
    if project_dir != path:
        log.info("Sandbox: project root is %s (subdir '%s')", project_dir, detection["project_subdir"])
    path = project_dir

    build_cmd, test_cmd = detection["build"], detection["test"]
    # Python: use *this* interpreter so pip/pytest resolve inside our venv
    # and so we never touch the system Python.
    if detection["language"].startswith("Python (pip)"):
        py = sys.executable
        build_cmd = f'"{py}" -m pip install -q -r requirements.txt'
        test_cmd = f'"{py}" -m pytest -q --no-header -p no:cacheprovider'
    elif not _tool_available(build_cmd):
        return {
            "build_status": "undetected",
            "sandbox_summary": (
                f"Detected {detection['language']} but '{build_cmd.split()[0]}' is not installed "
                "on this host, so build/test were skipped."
            ),
        }

    log.info("Sandbox: %s → build: %s", detection["language"], build_cmd)
    raw: dict[str, Any] = {
        "language": detection["language"],
        "project_subdir": detection["project_subdir"],
        "build_command": build_cmd,
        "test_command": test_cmd,
    }
    try:
        build = run_cmd(build_cmd, cwd=path, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        raw.update(build_timed_out=True, build_exit_code=None, build_log="")
        return get_llm().complete_json(SANDBOX_INTERPRETATION_PROMPT, raw, max_tokens=1500)

    raw["build_exit_code"] = build.returncode
    raw["build_log"] = (build.stdout + build.stderr)[-6000:]

    if build.returncode == 0:
        log.info("Sandbox: build OK → test: %s", test_cmd)
        try:
            test = run_cmd(test_cmd, cwd=path, timeout=timeout_seconds)
            raw["test_exit_code"] = test.returncode
            raw["test_log"] = (test.stdout + test.stderr)[-6000:]
            # pytest exit 5 = "no tests collected"; not a failure of the code.
            if detection["language"].startswith("Python") and test.returncode == 5:
                raw["note"] = "pytest exit code 5 means no tests were collected — treat test_status as 'skipped', not 'failure'."
        except subprocess.TimeoutExpired:
            raw.update(test_timed_out=True, test_exit_code=None, test_log="")
    else:
        raw["test_skipped"] = True

    return get_llm().complete_json(SANDBOX_INTERPRETATION_PROMPT, raw, max_tokens=1500)


# ═══════════════════════════════════════════════════════════════════════
# Specialist agent prompts + tool functions
# ═══════════════════════════════════════════════════════════════════════
CODE_QUALITY_PROMPT = """You are a senior code reviewer evaluating a pull request for code quality,
maintainability, and adherence to best practices.

You will receive: the PR diff, full contents of changed files, detected
language/framework, project coding standards (if provided), and raw
static analysis tool output.

Your job:
1. Identify code quality issues NOT already flagged by the static analysis
   tool (naming clarity, function complexity, duplication across files,
   violation of stated conventions, missing error handling, unclear
   abstractions).
2. Triage static analysis findings by real-world severity in context.
3. Cite the exact file and line range for every issue.
4. Never invent an issue you cannot point to in the provided code.
5. Do not comment on formatting/whitespace if an autoformatter is already
   configured.

Return ONLY valid JSON:
{
  "findings": [
    {"file": "string", "line_start": 0, "line_end": 0,
     "severity": "critical|major|minor|nit",
     "category": "readability|maintainability|duplication|convention|error_handling|other",
     "issue": "string", "recommendation": "string",
     "source": "llm_reasoning|static_analysis_triaged"}
  ],
  "overall_quality_summary": "2-3 sentences"
}"""

SECURITY_PROMPT = """You are a security reviewer specializing in application security (OWASP
Top 10, injection classes, auth/authz flaws, secrets handling, unsafe
deserialization, SSRF, dependency vulnerabilities).

You will receive: the PR diff and changed file contents, raw SAST
findings, dependency vulnerability scan results, and a list of newly
introduced I/O operations (network calls, DB queries, file access).

Your job:
1. Validate and prioritize SAST findings — state explicitly when you
   believe a finding is a false positive and why.
2. Look for vulnerability classes tools may miss: business logic auth
   bypasses, insecure direct object references, missing input validation
   on new endpoints, hardcoded secrets, unsafe eval/exec/deserialization.
3. Flag new dependencies with known CVEs.
4. Rate every finding by exploitability and impact.
5. Never speculate about vulnerabilities in code you were not given.

Return ONLY valid JSON:
{
  "findings": [
    {"file": "string", "line_start": 0, "line_end": 0,
     "severity": "critical|high|medium|low", "cwe": "string or null",
     "vulnerability_type": "string", "description": "string",
     "exploit_scenario": "string", "recommendation": "string",
     "false_positive_of_tool_finding": false}
  ],
  "blocking": false,
  "security_summary": "2-3 sentences"
}"""

PERFORMANCE_PROMPT = """You are a performance engineer reviewing profiling data and query plans
from a test run of this pull request's code.

You will receive: profiler output (top CPU-consuming functions with
self/total time), memory allocation snapshot and leak indicators,
database query log with execution plans for queries touched by this PR,
and the relevant changed source files.

Your job:
1. Identify methods with disproportionate CPU/memory cost relative to
   what they do, connected to specific changed code.
2. Flag queries with sequential scans, missing indexes, N+1 patterns, or
   unbounded result sets, citing the specific query and plan.
3. Flag memory growth patterns consistent with a leak — only with
   evidence from the provided snapshot.
4. Do not flag performance concerns unsupported by profiler or query
   plan data.

Return ONLY valid JSON:
{
  "findings": [
    {"type": "high_cpu|high_memory|possible_leak|slow_method|expensive_query",
     "location": "string", "evidence": "string",
     "estimated_impact": "string", "recommendation": "string"}
  ],
  "performance_summary": "2-3 sentences"
}"""

TEST_GAP_PROMPT = """You are a QA engineer identifying gaps in test coverage for a pull
request, aimed at helping testers write high-value test cases.

You will receive: the PR diff and changed file contents, existing test
files covering the changed modules, and a coverage report showing which
changed lines are exercised by tests.

Your job:
1. Identify logic branches, error paths, and boundary conditions NOT
   covered by existing tests — cross-reference the coverage report.
2. Propose concrete edge-case scenarios: boundary values, empty/null
   inputs, concurrent access if relevant, malformed input, failure of
   downstream dependencies.
3. Distinguish unit-level gaps from integration-level gaps.
4. Prioritize by risk.

Return ONLY valid JSON:
{
  "uncovered_areas": [{"file": "string", "line_start": 0, "line_end": 0, "description": "string"}],
  "recommended_test_cases": [
    {"test_type": "unit|integration", "scenario": "string",
     "priority": "high|medium|low", "rationale": "string"}
  ],
  "coverage_summary": "1-2 sentences"
}"""

REPORT_PROMPT = """You are producing the final pull request review summary for developers.
You will receive the sandbox build/test result, structured JSON output
from whichever specialist agents ran (some may be absent if skipped),
and a list of skipped agents with reasons.

Your job:
1. Deduplicate overlapping findings across agents.
2. Order findings by severity, security first among equal severities.
3. If the build or tests failed, that is always the lead item,
   regardless of other findings.
4. If any agent was skipped, note this plainly near the top.
5. Write a short executive summary (3-5 sentences).
6. Produce a Markdown-formatted PR comment body — concise, scannable,
   headers and a table, not walls of prose.
7. Set recommendation: approve | approve_with_comments | request_changes
   | blocked (blocked if build/tests failed or a blocking security
   finding exists).
8. Also emit inline_comments: one entry per finding that has a concrete
   file and line range in the changed code, so it can be pinned directly
   under that line in the PR diff. Keep each body short (2-4 sentences:
   the issue, why it matters, the fix). Use the path exactly as it
   appears in the diff / file_contents keys (relative to the repo root,
   forward slashes). Skip findings with no specific line (put those only
   in the summary body). Max 15 inline comments; prioritize by severity.

Return ONLY valid JSON:
{
  "recommendation": "approve|approve_with_comments|request_changes|blocked",
  "executive_summary": "string",
  "markdown_comment_body": "string, ready to post to GitHub as-is",
  "inline_comments": [
    {"path": "relative/file.py", "line": 17, "start_line": 14,
     "severity": "critical|major|minor|nit",
     "body": "markdown, 2-4 sentences"}
  ],
  "stats": {"critical_count": 0, "major_count": 0, "minor_count": 0},
  "incomplete_review_note": "string or null"
}"""


def _with_sources(payload: dict) -> dict:
    """If the orchestrator passed a local_path, attach the real file contents
    so the specialist reviews code rather than a directory name."""
    if "file_contents" not in payload:
        sources = _load_sources(payload.get("local_path"), payload.get("files"), payload.get("diff"))
        if sources:
            payload = {**payload, "file_contents": sources}
    return payload


def run_code_quality_agent(**payload) -> dict:
    return get_llm().complete_json(CODE_QUALITY_PROMPT + SPECIALIST_OUTPUT_RULES, _with_sources(payload))


def run_security_agent(**payload) -> dict:
    return get_llm().complete_json(SECURITY_PROMPT + SPECIALIST_OUTPUT_RULES, _with_sources(payload))


def run_performance_agent(**payload) -> dict:
    return get_llm().complete_json(PERFORMANCE_PROMPT + SPECIALIST_OUTPUT_RULES, _with_sources(payload))


def run_test_gap_agent(**payload) -> dict:
    return get_llm().complete_json(TEST_GAP_PROMPT + SPECIALIST_OUTPUT_RULES, _with_sources(payload))


def generate_report(**payload) -> dict:
    report = get_llm().complete_json(REPORT_PROMPT, payload, max_tokens=6000)
    # Cache server-side so github_post_review can pick up inline_comments
    # without the orchestrator having to relay them verbatim (a model
    # copying 15 comments through a tool call is slow and error-prone).
    _CURRENT_RUN["last_report"] = report
    return report


# ═══════════════════════════════════════════════════════════════════════
# Tool: github_post_review  (only used when write_access is true)
# ═══════════════════════════════════════════════════════════════════════
def github_post_review(owner_repo: str, pr_number: int, commit_sha: str, markdown_body: str, check_status: str) -> dict:
    gh_token = os.environ.get("GITHUB_APP_TOKEN")
    if not gh_token:
        return {"posted": False, "reason": "no_github_token_configured"}
    if os.environ.get("GITHUB_POST_REVIEWS", "false").lower() not in ("1", "true", "yes"):
        return {"posted": False, "reason": "posting disabled — set GITHUB_POST_REVIEWS=true in .env to enable"}

    # Bind to the repo/PR that was actually cloned this run, not whatever the
    # model happens to pass — otherwise a prompt-injected orchestrator could
    # post comments or set statuses on arbitrary repos the token can reach.
    ctx = _CURRENT_RUN.get("clone_ctx") or {}
    if ctx.get("owner_repo") and owner_repo != ctx["owner_repo"]:
        return {"posted": False, "reason": f"owner_repo mismatch: run cloned {ctx['owner_repo']}, refusing {owner_repo}"}
    if ctx.get("pr_number") and pr_number != ctx["pr_number"]:
        return {"posted": False, "reason": f"pr_number mismatch: run is for PR #{ctx['pr_number']}, refusing #{pr_number}"}
    commit_sha = ctx.get("commit_sha") or commit_sha

    report = _CURRENT_RUN.get("last_report") or {}
    inline = report.get("inline_comments") or []
    diff = ctx.get("diff") or ""
    commentable = _commentable_lines(diff)

    result: dict[str, Any] = {"posted": False, "owner_repo": owner_repo, "pr_number": pr_number}
    try:
        gh = Github(auth=GithubAuth.Token(gh_token))
        gh_repo = gh.get_repo(owner_repo)
        pr = gh_repo.get_pull(int(pr_number))

        # Build inline review comments — only for lines GitHub will accept
        # (added/context lines inside a diff hunk on the RIGHT side).
        review_comments, skipped = [], []
        for c in inline[:15]:
            path = str(c.get("path", "")).replace("\\", "/").lstrip("./")
            line = c.get("line")
            start = c.get("start_line")
            if not path or not isinstance(line, int) or path not in commentable or line not in commentable[path]:
                skipped.append(f"{path}:{line}")
                continue
            sev = str(c.get("severity", "")).lower()
            badge = {"critical": "🔴 **Critical**", "major": "🟠 **Major**", "minor": "🟡 Minor", "nit": "⚪ Nit"}.get(sev, "")
            entry: dict[str, Any] = {"path": path, "line": line, "side": "RIGHT",
                                     "body": (f"{badge} — " if badge else "") + str(c.get("body", ""))[:4000]}
            if isinstance(start, int) and start < line and start in commentable[path]:
                entry.update(start_line=start, start_side="RIGHT")
            review_comments.append(entry)

        event = {"approve": "APPROVE", "approve_with_comments": "COMMENT",
                 "request_changes": "REQUEST_CHANGES", "blocked": "REQUEST_CHANGES"}.get(check_status, "COMMENT")
        # GitHub forbids APPROVE / REQUEST_CHANGES on your own PR (422). When the
        # token owner is the PR author, submit as COMMENT — inline comments are
        # still attached, and the commit status below still carries the verdict.
        own_pr = bool(pr.user and pr.user.login == gh.get_user().login)
        if own_pr and event != "COMMENT":
            log.info("Token owner authored this PR; submitting review as COMMENT instead of %s.", event)
            event = "COMMENT"

        body = markdown_body[:65_000]
        if skipped:
            body += f"\n\n<sub>{len(skipped)} finding(s) referenced lines outside the diff and are listed above only.</sub>"

        try:
            pr.create_review(commit=gh_repo.get_commit(commit_sha), body=body, event=event, comments=review_comments)
            result.update(posted=True, mode="review", inline_comments=len(review_comments), event=event)
        except Exception as exc:  # noqa: BLE001 — fall back to a plain comment so the report is never lost
            log.warning("create_review failed (%s); falling back to a plain PR comment.", exc)
            pr.create_issue_comment(body)
            result.update(posted=True, mode="comment", inline_comments=0, review_error=str(exc)[:200])

        gh_repo.get_commit(commit_sha).create_status(
            state="success" if check_status in ("success", "approve", "approve_with_comments") else "failure",
            description=f"AI code review: {check_status}",
            context="ai-code-review",
            target_url=pr.html_url,
        )
        result["status_set"] = True
    except Exception as exc:  # noqa: BLE001 — surfaced to the orchestrator
        result["reason"] = f"{type(exc).__name__}: {exc}"[:300]
        return result
    return result


def _commentable_lines(diff: str) -> dict[str, set[int]]:
    """Map each file in a unified diff to the set of new-side line numbers
    that appear in a hunk (added or context). GitHub rejects review comments
    on any other line."""
    out: dict[str, set[int]] = {}
    current: Optional[str] = None
    new_ln = 0
    hunk_re = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
    for raw in diff.splitlines():
        if raw.startswith("+++ b/"):
            current = raw[6:].strip()
            out.setdefault(current, set())
        elif raw.startswith("@@") and current:
            m = hunk_re.match(raw)
            if m:
                new_ln = int(m.group(1))
        elif current and not raw.startswith("---"):
            if raw.startswith("+"):
                out[current].add(new_ln); new_ln += 1
            elif raw.startswith(" ") or raw == "":
                out[current].add(new_ln); new_ln += 1
            # '-' lines don't advance the new-side counter
    return out


# Per-run context so post-review can be bound to what was actually cloned.
# Single-threaded CLI use is the common case; the server runs one review per
# worker thread, so a plain dict is adequate here.
_CURRENT_RUN: dict[str, Any] = {}


def _github_clone_and_record(repo_url: str, ref: Optional[str] = None, pr_number: Optional[int] = None) -> dict:
    _CURRENT_RUN.clear()  # fresh run: drop any previous report/clone context
    result = github_clone_and_get_context(repo_url, ref, pr_number)
    if "error" not in result:
        _CURRENT_RUN["clone_ctx"] = {
            "owner_repo": result["owner_repo"], "pr_number": pr_number,
            "commit_sha": result.get("commit_sha"), "diff": result.get("diff", ""),
        }
    return result


# ═══════════════════════════════════════════════════════════════════════
# Orchestrator: system prompt, neutral tool schemas, tool dispatch
# ═══════════════════════════════════════════════════════════════════════
ORCHESTRATOR_SYSTEM_PROMPT = """You are the orchestrator for an automated code review system that can
review any GitHub repository, any pull request, or any manually
submitted code file, in any language, using any modern build system.

You act ONLY by calling tools. Never describe what you are about to do —
just call the tool. Your only plain-text reply comes after
generate_report, and it must be exactly its markdown_comment_body.

Step 0 — Determine entry mode:
- GitHub repo URL given: call github_clone_and_get_context first.
- Manually submitted files given (the message includes a local_path
  where they are already staged): call receive_manual_submission with
  that local_path, and skip run_sandbox_agent and run_performance_agent
  entirely (no build/test/runtime context exists for a bare file).

Passing code to specialists: always pass local_path. In Mode A also pass
the diff and set files to the changed_files list from
github_clone_and_get_context so specialists focus on what changed; if
there is no PR (whole-repo review), omit files and they will sample the
tree. The specialist tools read the source from disk themselves. NEVER
paste file contents into any tool call — it is wasteful and will be
truncated.

Step 1 — If Mode A (full repo), call run_sandbox_agent. If build_status
is "failure" or "undetected", skip run_code_quality_agent and
run_performance_agent, but still call run_security_agent on the static
source, then go straight to generate_report with build failure as the
lead finding and recommendation "blocked".

Step 2 — Otherwise, call run_code_quality_agent, run_security_agent,
run_test_gap_agent, and (Mode A with a successful build only)
run_performance_agent.

Step 3 — Call generate_report with the sandbox result (or null in Mode
B), every specialist result you obtained (null for any you skipped),
and an explicit list of skipped agents with reasons.

Step 4 — If write_access is true AND a pr_number exists, call
github_post_review with the owner_repo, pr_number and commit_sha
returned by github_clone_and_get_context, the report's markdown body,
and a check_status matching its recommendation. In every case, finish
by returning the markdown report as your final answer.

Rules throughout:
- If any tool call fails, do not drop it silently — carry the failure
  into generate_report's skipped-agents list. Retry a failed specialist
  AT MOST once; never call the same specialist a third time.
- Never execute untrusted code yourself, and never through any tool
  other than run_sandbox_agent.
- Never fabricate a specialist's findings in place of calling it.
"""

# Neutral schema: {"name", "description", "parameters"} — each provider
# adapts this to its own wire format (Claude's input_schema, OpenAI's
# function.parameters) inside its run_agentic_loop implementation.
_SPECIALIST_PROPS = {
    "local_path": {"type": "string", "description": "Directory containing the code under review."},
    "files": {"type": "array", "items": {"type": "string"}, "description": "Filenames within local_path to review."},
    "language": {"type": "string"},
    "diff": {"type": "string", "description": "PR diff, if any (Mode A only)."},
    "sandbox_result": {"type": "object", "description": "Output of run_sandbox_agent, if it ran."},
}

TOOLS = [
    {
        "name": "github_clone_and_get_context",
        "description": "Clone a public (or accessible private) GitHub repo at a ref, optionally fetch a PR diff, and report write access.",
        "parameters": {
            "type": "object",
            "properties": {"repo_url": {"type": "string"}, "ref": {"type": "string"}, "pr_number": {"type": "integer"}},
            "required": ["repo_url"],
        },
    },
    {
        "name": "receive_manual_submission",
        "description": (
            "Register manually submitted code file(s) that have ALREADY been staged on disk. "
            "Pass the local_path given in the user message — never paste file contents."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "local_path": {"type": "string", "description": "Directory where the files were staged."},
                "files": {"type": "array", "items": {"type": "string"}, "description": "Filenames within local_path."},
                "language_hint": {"type": "string"},
            },
            "required": ["local_path"],
        },
    },
    {
        "name": "run_sandbox_agent",
        "description": "Clone/build/test the code in an isolated sandbox and return a structured build/test result.",
        "parameters": {
            "type": "object",
            "properties": {"local_path": {"type": "string"}, "timeout_seconds": {"type": "integer"}},
            "required": ["local_path"],
        },
    },
    {"name": "run_code_quality_agent",
     "description": "Run the code quality specialist. Pass local_path (and files); the specialist reads the source itself — do NOT paste file contents.",
     "parameters": {"type": "object", "properties": _SPECIALIST_PROPS, "required": ["local_path"], "additionalProperties": True}},
    {"name": "run_security_agent",
     "description": "Run the security specialist. Pass local_path (and files); do NOT paste file contents.",
     "parameters": {"type": "object", "properties": _SPECIALIST_PROPS, "required": ["local_path"], "additionalProperties": True}},
    {"name": "run_performance_agent",
     "description": "Run the performance specialist. Pass local_path (and files); do NOT paste file contents.",
     "parameters": {"type": "object", "properties": _SPECIALIST_PROPS, "required": ["local_path"], "additionalProperties": True}},
    {"name": "run_test_gap_agent",
     "description": "Run the test coverage/edge-case specialist. Pass local_path (and files); do NOT paste file contents.",
     "parameters": {"type": "object", "properties": _SPECIALIST_PROPS, "required": ["local_path"], "additionalProperties": True}},
    {"name": "generate_report", "description": "Aggregate all results into a final Markdown report and recommendation.",
     "parameters": {"type": "object", "properties": {}, "additionalProperties": True}},
    {
        "name": "github_post_review",
        "description": (
            "Post the final review to the PR as a GitHub Review: the markdown body as the summary, "
            "plus inline comments pinned to the exact changed lines (taken automatically from "
            "generate_report's inline_comments — you do not need to pass them). Also sets the "
            "commit status. check_status must be the report's recommendation value."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "owner_repo": {"type": "string"}, "pr_number": {"type": "integer"},
                "commit_sha": {"type": "string"}, "markdown_body": {"type": "string"},
                "check_status": {"type": "string",
                                 "enum": ["approve", "approve_with_comments", "request_changes", "blocked"]},
            },
            "required": ["owner_repo", "pr_number", "commit_sha", "markdown_body", "check_status"],
        },
    },
]

TOOL_IMPL = {
    "github_clone_and_get_context": _github_clone_and_record,
    "receive_manual_submission": receive_manual_submission,
    "run_sandbox_agent": run_sandbox_agent,
    "run_code_quality_agent": run_code_quality_agent,
    "run_security_agent": run_security_agent,
    "run_performance_agent": run_performance_agent,
    "run_test_gap_agent": run_test_gap_agent,
    "generate_report": generate_report,
    "github_post_review": github_post_review,
}


def run_orchestrator(user_message: str) -> str:
    return get_llm().run_agentic_loop(ORCHESTRATOR_SYSTEM_PROMPT, TOOLS, TOOL_IMPL, user_message)


# ═══════════════════════════════════════════════════════════════════════
# FastAPI entry points
# ═══════════════════════════════════════════════════════════════════════
class ManualReviewRequest(BaseModel):
    files: list[dict]
    language_hint: Optional[str] = None


def _review_or_502(user_message: str) -> dict:
    """Run the orchestrator and turn any failure into a JSON error the
    caller can actually read, instead of a bare 500 with the detail buried
    in the server log."""
    try:
        return {"status": "completed", "result": run_orchestrator(user_message)}
    except Exception as exc:  # noqa: BLE001 — surfaced to the HTTP caller on purpose
        log.exception("Review failed")
        raise HTTPException(
            status_code=502,
            detail={"error": type(exc).__name__, "message": str(exc)[:2000]},
        ) from exc


# NOTE: these endpoints are deliberately plain `def`, not `async def`.
# A review takes minutes of blocking LLM + subprocess work; FastAPI runs
# sync endpoints in a thread pool so the event loop (and /docs) stay
# responsive. An `async def` here would freeze the whole server per request.
def _verify_github_signature(raw_body: bytes, signature_header: Optional[str]) -> bool:
    """Validate X-Hub-Signature-256 against GITHUB_WEBHOOK_SECRET.
    If no secret is configured, accept (dev mode) but log a warning."""
    secret = os.environ.get("GITHUB_WEBHOOK_SECRET", "")
    if not secret:
        log.warning("GITHUB_WEBHOOK_SECRET not set — webhook signature NOT verified (dev mode only).")
        return True
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature_header)


def _run_pr_review_job(repo_url: str, pr_number: int, commit_sha: str) -> None:
    """Background worker for webhook-triggered reviews. GitHub gives a
    webhook 10 s to respond; a review takes minutes, so we ack first and
    do the work here. The result is posted back to the PR by the
    orchestrator via github_post_review (needs GITHUB_APP_TOKEN +
    GITHUB_POST_REVIEWS=true), and also logged."""
    log.info("Webhook review started: %s PR #%s @ %s", repo_url, pr_number, commit_sha[:8])
    try:
        result = run_orchestrator(f"Review pull request #{pr_number} in {repo_url} at commit {commit_sha}.")
        log.info("Webhook review finished for PR #%s:\n%s", pr_number, result[:4000])
    except Exception:  # noqa: BLE001
        log.exception("Webhook review FAILED for PR #%s", pr_number)


@app.post("/webhook/github")
async def github_webhook(request: Request, background_tasks: BackgroundTasks):
    raw = await request.body()
    if not _verify_github_signature(raw, request.headers.get("X-Hub-Signature-256")):
        raise HTTPException(status_code=401, detail="invalid webhook signature")

    event = request.headers.get("X-GitHub-Event", "")
    if event == "ping":
        return {"status": "pong"}
    if event != "pull_request":
        return {"status": "ignored", "reason": f"event {event!r} not handled"}

    payload = json.loads(raw or b"{}")
    if payload.get("action") not in ("opened", "synchronize", "reopened"):
        return {"status": "ignored", "reason": f"action {payload.get('action')!r} not handled"}

    repo_url = payload["repository"]["clone_url"]
    pr_number = payload["pull_request"]["number"]
    commit_sha = payload["pull_request"]["head"]["sha"]
    background_tasks.add_task(_run_pr_review_job, repo_url, pr_number, commit_sha)
    return {"status": "accepted", "pr_number": pr_number, "commit_sha": commit_sha}


def _manual_review_message(files: list[dict], language_hint: Optional[str] = None) -> str:
    staged = stage_manual_files(files, language_hint)
    return (
        "Review the following manually submitted files. There is no GitHub "
        "repo and no build/test context. The files are already staged on disk — "
        f"do not paste their contents; pass this context to the tools instead: "
        f"{json.dumps({k: staged[k] for k in ('local_path', 'files', 'language')})}"
    )


@app.post("/review/manual")
def manual_review(req: ManualReviewRequest):
    return _review_or_502(_manual_review_message(req.files, req.language_hint))


@app.post("/review/public-repo")
def public_repo_review(repo_url: str, ref: Optional[str] = None, pr_number: Optional[int] = None):
    msg = f"Review the repository {repo_url}"
    if ref:
        msg += f" at ref {ref}"
    if pr_number:
        msg += f", pull request #{pr_number}"
    return _review_or_502(msg + ".")


# ═══════════════════════════════════════════════════════════════════════
# CLI — the actual, runnable interface to this file.
#
#   python code_review_agent.py review-repo <url> [--pr N] [--ref REF]
#   python code_review_agent.py review-files <path> [<path> ...]
#   python code_review_agent.py serve [--host H] [--port P]
#
# All three ultimately call the same run_orchestrator() used by the
# FastAPI routes above — the CLI is a thin front door, not a second
# implementation.
# ═══════════════════════════════════════════════════════════════════════
def _cmd_review_repo(args) -> int:
    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    log.info("Reviewing %s (provider=%s)", args.repo_url, provider)
    message = f"Review the repository {args.repo_url}"
    if args.ref:
        message += f" at ref {args.ref}"
    if args.pr:
        message += f", pull request #{args.pr}"
    result = run_orchestrator(message + ".")
    print(result)
    return 0


def _cmd_review_files(args) -> int:
    files = []
    for raw_path in args.paths:
        p = Path(raw_path)
        if not p.is_file():
            print(f"error: not a file: {raw_path}", file=sys.stderr)
            return 1
        files.append({"filename": p.name, "content": p.read_text(errors="replace")})

    provider = os.environ.get("LLM_PROVIDER", "anthropic")
    log.info("Reviewing %d local file(s) (provider=%s): %s", len(files), provider, [f["filename"] for f in files])
    result = run_orchestrator(_manual_review_message(files))
    print(result)
    return 0


def _cmd_serve(args) -> int:
    import uvicorn
    log.info("Starting server on %s:%d (provider=%s)", args.host, args.port, os.environ.get("LLM_PROVIDER", "anthropic"))
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def build_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="code_review_agent.py",
        description="AI Code Review Agent — review any GitHub repo/PR or local file, with any LLM provider.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_repo = sub.add_parser("review-repo", help="Review a GitHub repository, branch, or pull request")
    p_repo.add_argument("repo_url", help="e.g. https://github.com/owner/repo")
    p_repo.add_argument("--pr", type=int, default=None, help="Pull request number")
    p_repo.add_argument("--ref", default=None, help="Branch, tag, or commit SHA (defaults to the repo's default branch)")
    p_repo.set_defaults(func=_cmd_review_repo)

    p_files = sub.add_parser("review-files", help="Review one or more local files directly, no GitHub repo involved")
    p_files.add_argument("paths", nargs="+", help="One or more file paths to review")
    p_files.set_defaults(func=_cmd_review_files)

    p_serve = sub.add_parser("serve", help="Start the HTTP server (GitHub webhook + on-demand review endpoints)")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=_cmd_serve)

    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
