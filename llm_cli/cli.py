# llm_cli/cli.py
"""
llm-cli: minimal interactive wrapper around a local vLLM server
Python 3.11+, Linux & macOS
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn
import psutil

import requests
import typer
from pydantic import BaseModel
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

APP = typer.Typer(add_completion=False)
SET = typer.Typer(help="Modify persistent settings.")
APP.add_typer(SET, name="set")

# ───────────────────────── configuration ───────────────────────── #

CONFIG_DIR = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / "llm-cli"
CONFIG_FILE = CONFIG_DIR / "config.json"
PID_FILE = CONFIG_DIR / "server.pid"
LOG_FILE = CONFIG_DIR / "server.log"
SERVER_URL = "http://localhost:8000"
HEALTH_URL = f"{SERVER_URL}/health"
CHAT_URL = f"{SERVER_URL}/v1/chat/completions"


class Config(BaseModel):
    checkpoint_path: str | None = None
    system_prompt: str = ""
    start_timeout: int = 90  # seconds
    backoff_initial: float = 0.25  # seconds
    backoff_max: float = 4.0  # seconds

    @classmethod
    def load(cls) -> "Config":
        if CONFIG_FILE.exists():
            return cls.model_validate_json(CONFIG_FILE.read_text())
        return cls()

    def save(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        CONFIG_FILE.write_text(self.model_dump_json(indent=2))


CFG = Config.load()

# ───────────────── helpers: exponential wait, find vllm proc ─────────────── #


def wait_for(predicate: callable[[], bool], *, timeout: float) -> bool:
    """
    Poll `predicate` with exponential back-off until it returns True or `timeout`
    seconds elapse. Returns the final truth value.
    """
    delay = CFG.backoff_initial
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(delay)
        delay = min(delay * 2, CFG.backoff_max)
    return predicate()


def find_vllm_proc() -> psutil.Process | None:
    vllm_proc = next(
        (
            p
            for p in psutil.process_iter(["pid", "cmdline"])
            if p.info["cmdline"]
            and any(
                # vllm serve <checkpoint_path>
                cmd.endswith("vllm") and p.info["cmdline"][i + 1] == "serve"
                for i, cmd in enumerate(p.info["cmdline"][:-1])
            )
        ),
        None,
    )
    return vllm_proc


# ───────────────────────────── server control ───────────────────────────── #


def server_is_up() -> bool:
    try:
        return requests.get(HEALTH_URL, timeout=1).status_code == 200
    except requests.RequestException:
        return False


def start_server(verbose: bool = False) -> None:
    if not CFG.checkpoint_path:
        typer.secho(
            "No model checkpoint set. Use `llm-cli set model …` first.", fg="red"
        )
        raise typer.Exit(1)

    typer.echo(f"Starting vLLM server with checkpoint {CFG.checkpoint_path} …")

    # Route logs either to terminal (verbose) or to ~/.config/llm-cli/server.log
    if verbose:
        log_out = None
    else:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        log_out = open(LOG_FILE, "w")

    proc = subprocess.Popen(
        ["vllm", "serve", CFG.checkpoint_path],
        stdout=log_out or None,
        stderr=log_out or None,
        start_new_session=True,
    )

    # remember the pgid so we can kill the whole group later
    PID_FILE.write_text(str(proc.pid))

    if not wait_for(
        lambda: server_is_up() or proc.poll() is not None, timeout=CFG.start_timeout
    ):
        pass  # fall-through to next check

    # Did /health succeed?
    if server_is_up():
        typer.secho("✓ server ready", fg="green")
        return

    # Otherwise the process must have ended → show short log and quit.
    rc = proc.poll()
    typer.secho(f"vLLM failed to start (exit code {rc}).", fg="red")
    if not verbose and LOG_FILE.exists():
        typer.secho("Last 20 lines of server.log:", fg="yellow")
        with LOG_FILE.open() as f:
            tail = f.readlines()[-20:]
            for line in tail:
                sys.stderr.write("│ " + line)
    typer.echo(f"\nSee full log at {LOG_FILE}")
    raise typer.Exit(1)


async def ensure_server() -> None:
    if server_is_up():
        if not PID_FILE.exists():
            vllm_proc = find_vllm_proc()
            if vllm_proc is not None:
                PID_FILE.write_text(str(vllm_proc.pid))
        return
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        start_server,
        bool(os.getenv("LLM_CLI_VERBOSE")),
    )


# ────────────────────────────── chat session ────────────────────────────── #


@dataclass
class ChatSession:
    history: list[dict[str, str]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if CFG.system_prompt:
            self.history.append({"role": "system", "content": CFG.system_prompt})

    def reset(self) -> None:
        self.history.clear()
        if CFG.system_prompt:
            self.history.append({"role": "system", "content": CFG.system_prompt})

    def append(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})


def call_llm(messages: list[dict[str, str]]) -> str:
    payload = json.dumps({"messages": messages, "stream": False})
    result = subprocess.run(
        [
            "curl",
            "-sS",
            "-X",
            "POST",
            CHAT_URL,
            "-H",
            "Content-Type: application/json",
            "-d",
            payload,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "curl failed")

    data: dict[str, Any] = json.loads(result.stdout)
    return data["choices"][0]["message"]["content"]


# ─────────────────────────── interactive prompt ─────────────────────────── #


async def chat_shell() -> None:
    await ensure_server()

    kb = KeyBindings()
    state = ChatSession()

    @kb.add("c-r")
    def _(event) -> None:  # noqa: D401
        state.reset()
        typer.secho("(history reset)\n", fg="yellow")

    session = PromptSession(key_bindings=kb)

    while True:
        try:
            user_text = await session.prompt_async("> ")
        except (EOFError, KeyboardInterrupt):
            typer.echo()
            return

        if not user_text.strip():
            continue

        state.append("user", user_text)

        loop = asyncio.get_running_loop()
        try:
            assistant = await loop.run_in_executor(None, call_llm, list(state.history))
        except Exception as exc:
            typer.secho(f"[ERROR] {exc}", fg="red")
            state.history.pop()  # revert the user line
            continue

        state.append("assistant", assistant)
        typer.echo(assistant + "\n")


# ───────────────────────────── stop command ───────────────────────────── #


@APP.command("stop")
def stop_server() -> None:
    """Gracefully terminate the background vLLM server."""

    if not PID_FILE.exists():
        vllm_proc = find_vllm_proc()
        if vllm_proc is None:
            typer.echo("No running server recorded.")
            raise typer.Exit()
        pid = vllm_proc.pid
        typer.echo(f"Found vLLM server (pid {pid}); cleaning it up.")
    else:
        try:
            pid = int(PID_FILE.read_text())
        except ValueError:
            PID_FILE.unlink(missing_ok=True)
            typer.echo("Corrupt pid file – removed.")
            raise typer.Exit(1)

    # If the process is already gone, just clean up.
    if not psutil.pid_exists(pid):
        PID_FILE.unlink(missing_ok=True)
        typer.echo("Server was not running.")
        raise typer.Exit()

    pgid = os.getpgid(pid)
    typer.echo(f"Stopping vLLM server (pgid {pgid}) …")
    os.killpg(pgid, signal.SIGTERM)

    # Wait up to 5 s
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and psutil.pid_exists(pid):
        time.sleep(0.1)

    if psutil.pid_exists(pid):
        typer.secho("Server did not shut down – still running.", fg="red")
        raise typer.Exit(1)

    PID_FILE.unlink(missing_ok=True)
    typer.secho("✓ server stopped", fg="green")


# ──────────────────────────────── commands ──────────────────────────────── #


@SET.command("model")
def set_model(path: str) -> None:
    CFG.checkpoint_path = path
    CFG.save()
    typer.echo(f"Model checkpoint saved: {path}")


@SET.command("system")
def set_system() -> None:
    editor = os.getenv("EDITOR", "vim")
    with tempfile.NamedTemporaryFile(
        prefix="llm-system-", suffix=".txt", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
        tmp.write(CFG.system_prompt.encode())

    subprocess.run([editor, str(tmp_path)])
    CFG.system_prompt = tmp_path.read_text().rstrip("\n")
    tmp_path.unlink()

    CFG.save()
    typer.echo("System prompt updated.")


@APP.command("reset")
def reset_cmd() -> None:
    ChatSession().reset()  # just to create default & save nothing
    typer.echo("Session history flushed (runtime only).")


@APP.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:  # noqa: D401
    if ctx.invoked_subcommand is None:
        try:
            asyncio.run(chat_shell())
        except RuntimeError as exc:  # e.g. event-loop closed on ^C
            typer.secho(f"\n{exc}", fg="red")


# ─────────────────────────────── entry point ────────────────────────────── #


def _die_with_ctrl_c(*_: Any) -> NoReturn:  # noqa: D401
    raise KeyboardInterrupt


signal.signal(signal.SIGINT, _die_with_ctrl_c)

if __name__ == "__main__":
    APP()
