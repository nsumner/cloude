#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path

import urllib.request
import urllib.error


# ==============================
# LOGGING (JSON STRUCTURED)
# ==============================


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in {
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            }:
                log[key] = value

        return json.dumps(log)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    log_file = "cloude.log"

    handler = logging.FileHandler(log_file)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)


logger = logging.getLogger("claude-local")


# ==============================
# CONFIG MODEL
# ==============================

LLAMA_SERVER_BIN = os.getenv("LLAMA_SERVER_BIN", "llama-server")


@dataclass(frozen=True)
class LlamaConfig:
    name: str
    gguf_path: Path
    port: int = 8080
    n_ctx: int = 100000  # Context size. Default big for claude.
    n_threads: int = 8
    n_gpu_layers: int = 999
    extra_flags: list[str] = field(default_factory=list)

    def server_cmd(self) -> list[str]:
        return [
            LLAMA_SERVER_BIN,
            "-m", str(self.gguf_path),
            "--port", str(self.port),
            "--n-gpu-layers", str(self.n_gpu_layers),
            "--ctx-size", str(self.n_ctx),
            "--threads", str(self.n_threads),
            "-t", "10",
            "-b", "1024",
            "-ub", "1024",
            "--parallel", "1",
            "-fa", "on",
            "--jinja",
            "--keep", "1024",
            "--cache-type-k", "q8_0",
            "--cache-type-v", "q8_0",
            "--swa-full",
            "--no-context-shift",
            "--chat-template-kwargs", '{"enable_thinking": false}',
            "--mlock",
            "--no-mmap",
            "--metrics",
            "--alias", self.name,
            *self.extra_flags,
        ]

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}"


# ==============================
# CONFIGURATIONS
# ==============================

CONFIGS: dict[str, LlamaConfig] = {
    "omni-small": LlamaConfig(
        name="OmniCoder-2-9B-Q3KL",
        gguf_path=Path(
            "/home/nick/Tesslate/OmniCoder-2-9B-GGUF/omnicoder-2-9b-q3_k_l.gguf"
        ),
        n_ctx=65536,
    ),
    "omni-medium": LlamaConfig(
        name="OmniCoder-2-9B-Q5KM",
        gguf_path=Path(
            "/home/nick/Tesslate/OmniCoder-2-9B-GGUF/omnicoder-2-9b-q5_k_m.gguf"
        ),
        n_ctx=65536,
    ),
    "qwen-27-desk": LlamaConfig(
        name="Qwen3.5-27B-UD-Q2",
        gguf_path=Path(
            "/home/nick/unsloth/Qwen3.5-27B-GGUF/Qwen3.5-27B-UD-Q2_K_XL.gguf"
        ),
        n_ctx=175000,
    ),
}

DEFAULT_CONFIG = "omni-small"


# ==============================
# LLAMA SERVER MANAGEMENT
# ==============================


class LlamaServer:
    def __init__(self, config: LlamaConfig):
        self.config = config
        self.proc: subprocess.Popen[str] | None = None
        self.started_by_us: bool = False

    def _validate_model(self) -> None:
        if not os.path.isfile(self.config.gguf_path):
            raise FileNotFoundError(f"GGUF model not found: {self.config.gguf_path}")

    def _is_server_running(self) -> bool:
        url = f"{self.config.base_url}/v1/models"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=1) as resp:
                return resp.status == 200
        except Exception:
            return False

    def start(self) -> None:
        self._validate_model()

        if self._is_server_running():
            logger.debug(
                "using_existing_server",
                extra={"port": self.config.port},
            )
            self.started_by_us = False
            return

        cmd = self.config.server_cmd()

        logger.debug(
            "starting_llama_server",
            extra={"cmd": shlex.join(cmd), "port": self.config.port},
        )

        self.proc = subprocess.Popen(
            cmd,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.started_by_us = True

        self._wait_until_ready()

    def _wait_until_ready(self, timeout: float = 60.0) -> None:
        url = f"{self.config.base_url}/chat/completions"
        start = time.time()

        logger.debug("waiting_for_model_ready", extra={"url": url})

        payload = json.dumps(
            {
                "model": "dummy",
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 1,
            }
        ).encode("utf-8")

        while True:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError("llama.cpp server exited early")

            try:
                req = urllib.request.Request(
                    url,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )

                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        logger.debug("model_ready")
                        return

            except Exception:
                pass

            if time.time() - start > timeout:
                raise TimeoutError("Timed out waiting for model readiness")

            time.sleep(1)

    def stop(self) -> None:
        if not self.proc or not self.started_by_us:
            return

        logger.debug("stopping_llama_server")

        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.warning("force_killing_llama_server")
            self.proc.kill()


# ==============================
# METRICS
# ==============================


def _to_visual(number: float) -> float:
    if math.isfinite(number):
        return round(number)
    return number


@dataclass
class Metrics:
    context_size: float = 0
    kv_cache_total: float = 0
    kv_cache_used: float = 0
    prompt_tps: float = 0
    generated_tps: float = 0

    def report(self):
        print("---------------------------")
        print(f"     Prompt: {_to_visual(self.prompt_tps):>5} Tok/s               ")
        print(f"   Generate: {_to_visual(self.generated_tps):>5} Tok/s               ")
        logger.info(
            "session-metrics",
            extra={"generated_tps": self.generated_tps, "prompt_tps": self.prompt_tps},
        )


def fetch_metrics(base_url: str) -> Metrics:
    url = base_url + "/metrics"

    metrics = {}
    with urllib.request.urlopen(url, timeout=2) as resp:
        text = resp.read().decode()

    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue

        key, value = line.split(" ", 1)
        metrics[key] = float(value)

    return Metrics(
        prompt_tps=metrics.get("llamacpp:prompt_tokens_seconds", 0),
        generated_tps=metrics.get("llamacpp:predicted_tokens_seconds", 0),
    )


# ==============================
# CLAUDE INVOCATION
# ==============================


def run_claude(base_url: str, model: str, passthrough_args: list[str]) -> int:
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = base_url
    env["ANTHROPIC_AUTH_TOKEN"] = "local-llama"

    cmd = ["claude", "--model", model, *passthrough_args]

    logger.debug(
        "running_claude",
        extra={"base_url": base_url, "cmd": shlex.join(cmd)},
    )

    try:
        return subprocess.call(cmd, env=env)
    except FileNotFoundError:
        logger.error("claude_not_found")
        return 1


# ==============================
# CLI
# ==============================


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run Claude Code against a local llama.cpp model",
        allow_abbrev=False
    )

    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--metrics", action="store_true")
    parser.add_argument("--port", type=int, help="Override port for the server")
    parser.add_argument("--server", action="store_true", help="Start server only")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_known_args()


def list_configs() -> None:
    max_name = max(len(name) for name in CONFIGS.keys())
    for name, cfg in CONFIGS.items():
        print(f"{name:>{max_name}}: ctx:{cfg.n_ctx:>6} model:{cfg.gguf_path.name}")
        logger.debug(
            "config",
            extra={
                "name": name,
                "model": cfg.gguf_path,
                "ctx": cfg.n_ctx,
                "threads": cfg.n_threads,
                "gpu_layers": cfg.n_gpu_layers,
                "port": cfg.port,
            },
        )


# ==============================
# MAIN
# ==============================


def main() -> int:
    args, passthrough = parse_args()

    if args.list:
        list_configs()
        return 0

    setup_logging(args.verbose)
    logger.info(f"{args} {passthrough}")

    if args.config not in CONFIGS:
        logger.error("unknown_config", extra={"config": args.config})
        return 1

    base_cfg = CONFIGS[args.config]

    # Apply port override
    if args.port:
        config = replace(base_cfg, port=args.port)
    else:
        config = base_cfg

    server = LlamaServer(config)

    def shutdown(sig, _frame):
        logger.info("received_signal", extra={"signal": sig})
        server.stop()
        sys.exit(1)

    # Note: SIGKILL cannot be caught
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        server.start()

        if args.server:
            logger.debug("server_running_only_mode")
            while True:
                time.sleep(60)

        outcome = run_claude(config.base_url, config.name, passthrough)

        if args.verbose or args.metrics:
            print("\n\n\n")
            metrics = fetch_metrics(config.base_url)
            metrics.report()
        return outcome

    finally:
        server.stop()


if __name__ == "__main__":
    sys.exit(main())
