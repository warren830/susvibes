"""Offline tests for the Claude Code runner (``evaluation_harness/claude_code``).

Docker is never called: the docker CLI is replaced by a fake at
``subprocess.run`` where a test needs one.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from tests.agent_import import import_agent_module

REAL_RUN = subprocess.run

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
    "CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK",
]


INSTANCE = {
    "instance_id": "demo__demo-1",
    "image_name": "img:latest",
    "problem_statement": "Fix the bug in parse().",
}


PATCH = "diff --git a/parse.py b/parse.py\n"
BASE_COMMIT = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"


class FakeDocker:
    """Stands in for the docker CLI and records each call.

    Every call succeeds unless its in-container script contains one of
    ``failing``, or one of ``hanging``, which runs into its timeout.
    ``git rev-parse`` answers ``BASE_COMMIT``, and copying a file out of
    the container yields ``PATCH``.  Commands other than docker really run.
    """

    def __init__(self, failing=(), hanging=()):
        self.failing = failing
        self.hanging = hanging
        self.calls = []
        self.options = []

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        self.options.append(kwargs)
        if argv[0] != "docker":
            return REAL_RUN(argv, **kwargs)
        if any(marker in argv[-1] for marker in self.hanging):
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        if self.copies_file_out(argv):
            Path(argv[3]).write_text(PATCH)
        script = argv[-1] if argv[:2] == ["docker", "exec"] else ""
        failed = any(marker in script for marker in self.failing)
        return mock.MagicMock(
            stdout=f"{BASE_COMMIT}\n" if "rev-parse" in argv else "",
            stderr="boom" if failed else "",
            returncode=1 if failed else 0,
        )

    @staticmethod
    def copies_file_out(argv):
        return argv[:2] == ["docker", "cp"] and ":" in argv[2] and not argv[2].endswith("/.")

    def scripts(self):
        """The ``bash -c`` scripts run in the container, in order."""
        return [argv[-1] for argv in self.calls if argv[:2] == ["docker", "exec"] and argv[-2] == "-c"]

    def index(self, predicate):
        """Position of the first call whose argv satisfies ``predicate``."""
        return next(i for i, argv in enumerate(self.calls) if predicate(argv))

    def container_name(self):
        run = self.calls[self.index(lambda argv: argv[:2] == ["docker", "run"])]
        return run[run.index("--name") + 1]

    def env_file_writes(self):
        """(call index, content) of each write to the agent env file."""
        return [(i, self.options[i]["input"]) for i, argv in enumerate(self.calls) if "-i" in argv]


@pytest.fixture()
def run_docker():
    return import_agent_module("claude_code")


@pytest.fixture()
def batch(tmp_path, monkeypatch):
    # The batch runner opens docker_batch_run.log in the cwd on import.
    monkeypatch.chdir(tmp_path)
    return import_agent_module("claude_code", "batch_run_docker")


@pytest.fixture()
def parallel(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return import_agent_module("claude_code", "parallel_batch_run")


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


# ── claude command ────────────────────────────────────────────────────

UPSTREAM_TOOLS = [
    "Bash", "Edit", "Write", "Read", "Glob", "Grep", "LS", "WebFetch",
    "NotebookEdit", "NotebookRead", "TodoRead", "TodoWrite", "Agent",
]


def claude_args(docker, tmp_path):
    """The argv bash hands to ``claude``, run in a dir where an unquoted glob would expand."""
    (script,) = [s for s in docker.scripts() if "claude --verbose" in s]
    command = script[script.index("claude --verbose"):]
    shell_dir = tmp_path / "shell"
    shell_dir.mkdir()
    (shell_dir / "mcp__security-agent__decoy").touch()
    out = subprocess.run(
        ["bash", "-c", 'claude() { printf "%s\\0" "$@"; }\n' + command],
        cwd=shell_dir, capture_output=True, text=True, check=True,
    ).stdout
    return out.split("\0")[:-1]


def test_claude_gets_default_tool_lists(batch, tmp_path):
    docker = FakeDocker()

    run_instance(batch, tmp_path, docker)

    args = claude_args(docker, tmp_path)
    allowed = args[args.index("--allowedTools") + 1:args.index("--disallowedTools")]
    assert allowed == [*UPSTREAM_TOOLS, "Skill", "Task", "mcp__security-agent__*"]
    assert args[args.index("--disallowedTools") + 1:] == ["mcp__security-agent__setup"]
    assert INSTANCE["problem_statement"] in args[args.index("-p") + 1]


def test_claude_gets_overridden_tool_lists(batch, tmp_path):
    docker = FakeDocker()

    run_instance(batch, tmp_path, docker, allowed_tools=["Bash", "Read"], disallowed_tools=[])

    args = claude_args(docker, tmp_path)
    assert args[args.index("--allowedTools") + 1:] == ["Bash", "Read"]
    assert "--disallowedTools" not in args


def test_tool_options_parse_as_comma_lists(batch):
    args = batch.get_options().parse_args(["--allowed_tools", "Bash, Skill,,mcp__x__*", "--disallowed_tools", ""])

    assert batch.split_tools(args.allowed_tools) == ["Bash", "Skill", "mcp__x__*"]
    assert batch.split_tools(args.disallowed_tools) == []


# ── patch ─────────────────────────────────────────────────────────────


def test_patch_is_taken_in_the_container_after_claude_runs(batch, tmp_path):
    docker = FakeDocker()

    record = run_instance(batch, tmp_path, docker)

    assert record["model_patch"] == PATCH
    base_read = docker.index(lambda argv: "rev-parse" in argv)
    claude_run = docker.index(lambda argv: "claude --verbose" in argv[-1])
    git_add = docker.index(lambda argv: argv[:2] == ["docker", "exec"] and "add" in argv and "git" in argv)
    git_diff = docker.index(lambda argv: argv[:2] == ["docker", "exec"] and "diff" in argv and "git" in argv)
    copy_out = docker.index(FakeDocker.copies_file_out)
    assert base_read < claude_run < git_add < git_diff < copy_out
    assert docker.calls[git_add][2:4] == ["-w", "/project"]
    assert BASE_COMMIT in docker.calls[git_diff]



# ── timeout ───────────────────────────────────────────────────────────


def test_timeout_bounds_the_claude_run(batch, tmp_path):
    docker = FakeDocker()

    record = run_instance(batch, tmp_path, docker, timeout=10800)

    claude_run = docker.index(lambda argv: "claude --verbose" in argv[-1])
    assert docker.options[claude_run]["timeout"] == 10800
    assert record["timed_out"] is False


def test_timed_out_claude_run_is_recorded(batch, tmp_path):
    docker = FakeDocker(hanging=("claude --verbose",))

    record = run_instance(batch, tmp_path, docker, timeout=60)

    assert record["timed_out"] is True
    assert record["claude_success"] is False
    assert record["model_patch"] == PATCH
    # The killed docker exec leaves claude running in the container; it is
    # stopped before the patch is taken.
    claude_run = docker.index(lambda argv: "claude --verbose" in argv[-1])
    stop = docker.index(lambda argv: argv[:2] == ["docker", "exec"] and argv[-1].startswith("kill"))
    git_add = docker.index(lambda argv: argv[:2] == ["docker", "exec"] and "add" in argv and "git" in argv)
    assert claude_run < stop < git_add



# ── agent env delivery ────────────────────────────────────────────────


def test_agent_env_reaches_the_container_only_over_stdin(batch, tmp_path, monkeypatch):
    secret = "AKIA-secret value"
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", secret)
    monkeypatch.setenv("AWS_REGION", "us-west-2")
    docker = FakeDocker()

    run_instance(batch, tmp_path, docker)

    assert not any(secret in arg for argv in docker.calls for arg in argv)
    (write_at, content), = docker.env_file_writes()
    assert f"export AWS_SECRET_ACCESS_KEY='{secret}'" in content.splitlines()
    assert "export AWS_REGION=us-west-2" in content.splitlines()
    assert write_at < docker.index(lambda argv: "claude --verbose" in argv[-1])


# ── agent driver ──────────────────────────────────────────────────────

FAKE_DRIVER = """\
import argparse, json, os, pathlib, sys, time
parser = argparse.ArgumentParser()
for flag in ("--container", "--workdir", "--prompt-file", "--out-dir", "--timeout",
             "--allowed-tools", "--disallowed-tools"):
    parser.add_argument(flag, required=True)
parser.add_argument("--source", action="append", required=True)
args = parser.parse_args()
behavior = os.environ.get("FAKE_DRIVER", "completed")
if behavior == "crash":
    sys.exit("driver crashed")
if behavior == "hang":
    time.sleep(30)
out = pathlib.Path(args.out_dir)
seen = dict(vars(args), prompt=pathlib.Path(args.prompt_file).read_text(encoding="utf-8"))
(out / "seen.json").write_text(json.dumps(seen))
(out / "raw.jsonl").write_text('{"type": "result", "num_turns": 3}\\n')
(out / "summary.json").write_text(json.dumps({"status": behavior, "num_turns": 3}))
"""


@pytest.fixture()
def driver(tmp_path):
    path = tmp_path / "fake-driver"
    path.write_text(f"#!{sys.executable}\n" + FAKE_DRIVER)
    path.chmod(0o755)
    return str(path)


def run_with_driver(batch, tmp_path, docker, driver, **options):
    return run_instance(
        batch, tmp_path, docker,
        agent_driver=driver, artifacts_root=str(tmp_path / "artifacts"), **options,
    )


def test_driver_runs_the_agent_step_per_the_contract(batch, tmp_path, driver):
    docker = FakeDocker()

    record = run_with_driver(batch, tmp_path, docker, driver, timeout=10800)

    artifacts_dir = tmp_path / "artifacts" / INSTANCE["instance_id"]
    seen = json.loads((artifacts_dir / "seen.json").read_text())
    assert seen["container"] == docker.container_name()
    assert seen["workdir"] == "/project"
    assert INSTANCE["problem_statement"] in seen["prompt"]
    assert Path(seen["prompt_file"]).parent == artifacts_dir
    assert seen["out_dir"] == str(artifacts_dir)
    assert seen["timeout"] == "10800"
    assert seen["allowed_tools"] == ",".join([*UPSTREAM_TOOLS, "Skill", "Task", "mcp__security-agent__*"])
    assert seen["disallowed_tools"] == "mcp__security-agent__setup"
    assert seen["source"] == [
        "/root/.claude_env", "/root/.nvm/nvm.sh", "/root/.bashrc", "/root/.bench_agent_env",
    ]

    assert not any("claude --verbose" in script for script in docker.scripts())
    (write_at, _), = docker.env_file_writes()
    assert write_at < docker.index(lambda argv: argv[0] == driver)

    assert record["claude_stdout"] == (artifacts_dir / "raw.jsonl").read_text()
    assert record["agent_summary"] == {"status": "completed", "num_turns": 3}
    assert record["artifacts_dir"] == str(artifacts_dir)
    assert record["claude_success"] is True
    assert record["timed_out"] is False
    assert record["model_patch"] == PATCH


def test_driver_timeout_status_is_recorded(batch, tmp_path, driver, monkeypatch):
    monkeypatch.setenv("FAKE_DRIVER", "timeout")

    record = run_with_driver(batch, tmp_path, FakeDocker(), driver)

    assert record["timed_out"] is True
    assert record["claude_success"] is False
    assert record["agent_summary"]["status"] == "timeout"


def test_crashed_driver_leaves_a_failed_record_with_its_stderr(batch, tmp_path, driver, monkeypatch):
    monkeypatch.setenv("FAKE_DRIVER", "crash")

    record = run_with_driver(batch, tmp_path, FakeDocker(), driver)

    assert record["claude_success"] is False
    assert record["agent_summary"] is None
    assert record["claude_stdout"] == ""
    assert "driver crashed" in record["claude_stderr"]
    assert record["model_patch"] == PATCH


def test_hung_driver_is_stopped_after_the_grace_period(batch, tmp_path, driver, monkeypatch):
    monkeypatch.setenv("FAKE_DRIVER", "hang")
    monkeypatch.setattr(batch, "DRIVER_GRACE_SECONDS", 0)

    record = run_with_driver(batch, tmp_path, FakeDocker(), driver, timeout=1)

    assert record["timed_out"] is True
    assert record["claude_success"] is False



# ── parallel runs ─────────────────────────────────────────────────────


def test_child_runs_get_every_batch_option(parallel, tmp_path):
    options = {
        "--results_dir": "out",
        "--model": "arm2",
        "--setup_script": "../bench/arm2-setup.sh",
        "--setup_assets": "../bench/assets",
        "--claude_code_version": "2.1.318",
        "--allowed_tools": "Bash,Skill,mcp__security-agent__*",
        "--disallowed_tools": "",
        "--timeout": "10800",
        "--agent_driver": "../bench/driver",
        "--strategy": "generic",
    }
    argv = [arg for option in options.items() for arg in option] + ["--keep_workspace"]
    args = parallel.get_options().parse_args(argv)

    with mock.patch.object(subprocess, "run") as run:
        run.return_value = mock.MagicMock(returncode=0)
        parallel.run_batch_process(
            0, 0, 1, "data.jsonl", str(tmp_path / "workspace"), "python3",
            parallel.forwarded_batch_args(args),
        )

    cmd = run.call_args.args[0]
    assert cmd[:2] == ["python3", "batch_run_docker.py"]
    child = parallel.get_batch_options().parse_args(cmd[2:])
    for option in options:
        name = option.removeprefix("--")
        assert getattr(child, name) == getattr(args, name), name
    assert child.keep_workspace is True


def test_every_batch_option_is_forwarded_or_set_per_process(parallel):
    per_process = {"jsonl_file", "workspace_root", "start_idx", "num_instances", "load_from_file", "timestamp_suffix"}
    batch_options = {action.dest for action in parallel.get_batch_options()._actions} - {"help"}

    assert batch_options - per_process == {*parallel.FORWARDED_BATCH_OPTIONS, "keep_workspace"}
