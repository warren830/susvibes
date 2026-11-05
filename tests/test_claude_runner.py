"""Offline tests for the Claude Code runner (``evaluation_harness/claude_code``).

Docker is never called: the docker CLI is replaced by a fake at
``subprocess.run`` where a test needs one.
"""

from __future__ import annotations

import asyncio
import subprocess
from unittest import mock

import pytest

from tests.agent_import import import_agent_module

# The forwarded names, as the runner/driver contract lists them.
CONTRACT_ENV_NAMES = [
    "CLAUDE_CODE_USE_BEDROCK",
    "AWS_REGION",
    "ANTHROPIC_MODEL",
    "ANTHROPIC_SMALL_FAST_MODEL",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    "ANTHROPIC_DEFAULT_SONNET_MODEL",
    "ANTHROPIC_DEFAULT_OPUS_MODEL",
    "AWS_BEARER_TOKEN_BEDROCK",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "CLAUDE_CODE_MAX_OUTPUT_TOKENS",
]


INSTANCE = {
    "instance_id": "demo__demo-1",
    "image_name": "img:latest",
    "problem_statement": "Fix the bug in parse().",
}


class FakeDocker:
    """Stands in for the docker CLI and records each call.

    Every call succeeds unless its in-container script contains one of
    ``failing``.
    """

    def __init__(self, failing=()):
        self.failing = failing
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        script = argv[-1] if argv[:2] == ["docker", "exec"] else ""
        failed = any(marker in script for marker in self.failing)
        return mock.MagicMock(
            stdout="", stderr="boom" if failed else "", returncode=1 if failed else 0
        )

    def scripts(self):
        """The ``bash -c`` scripts run in the container, in order."""
        return [argv[-1] for argv in self.calls if argv[:2] == ["docker", "exec"] and argv[-2] == "-c"]


@pytest.fixture()
def run_docker():
    return import_agent_module("claude_code")


@pytest.fixture()
def batch(tmp_path, monkeypatch):
    # The batch runner opens docker_batch_run.log in the cwd on import.
    monkeypatch.chdir(tmp_path)
    return import_agent_module("claude_code", "batch_run_docker")


def run_instance(batch, tmp_path, docker, **options):
    setup_script = tmp_path / "setup.sh"
    setup_script.write_text("echo setup\n")
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(exist_ok=True)
    with mock.patch.object(subprocess, "run", docker):
        return asyncio.run(batch.process_instance(
            INSTANCE, 0, 1, "arm1",
            workspace_root=str(workspace_root),
            setup_script=str(setup_script),
            keep_workspace=False,
            **options,
        ))


# ── agent env ─────────────────────────────────────────────────────────


def test_agent_env_forwards_every_contract_name(run_docker):
    environ = {name: f"value-of-{name}" for name in CONTRACT_ENV_NAMES}
    environ["HOME"] = "/home/someone"

    assert run_docker.get_agent_env(environ) == {
        name: f"value-of-{name}" for name in CONTRACT_ENV_NAMES
    }


def test_agent_env_skips_unset_and_empty_vars(run_docker):
    environ = {
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "AWS_REGION": "us-west-2",
        "ANTHROPIC_MODEL": "",
        "ANTHROPIC_API_KEY": "",
    }

    assert run_docker.get_agent_env(environ) == {
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "AWS_REGION": "us-west-2",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "50000",
    }


def test_agent_env_forwards_bench_forward_env_names(run_docker):
    environ = {
        "AIDLC_SKIP_HUMAN_PRESENCE_GUARD": "1",
        "AIDLC_SESSION_OVERRIDE": "bench",
        "NOT_LISTED": "x",
        "BENCH_FORWARD_ENV": "AIDLC_SKIP_HUMAN_PRESENCE_GUARD, AIDLC_SESSION_OVERRIDE,,UNSET_NAME",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000",
    }

    assert run_docker.get_agent_env(environ) == {
        "AIDLC_SKIP_HUMAN_PRESENCE_GUARD": "1",
        "AIDLC_SESSION_OVERRIDE": "bench",
        "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "64000",
    }


# ── setup ─────────────────────────────────────────────────────────────


def test_failed_setup_fails_the_instance_without_running_claude(batch, tmp_path):
    docker = FakeDocker(failing=("setup-env.sh",))

    record = run_instance(batch, tmp_path, docker, claude_code_version="2.1.318")

    assert record["model_patch"] == ""
    assert "Environment setup failed" in record["error"]
    assert not any("claude --verbose" in script for script in docker.scripts())
