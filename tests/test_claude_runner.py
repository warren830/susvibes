"""Offline tests for the Claude Code runner (``evaluation_harness/claude_code``).

Docker is never called: the docker CLI is replaced by a fake at
``subprocess.run`` where a test needs one.
"""

from __future__ import annotations

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


@pytest.fixture()
def run_docker():
    return import_agent_module("claude_code")


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
