"""The model patch's git commands, run against a real repository on the host.

``DockerHarnessBase.extract_patch`` runs the same commands inside the
container, so these tests pin what the patch contains without Docker.
"""

from __future__ import annotations

import os
import re
import subprocess

import pytest

from evaluation_harness.base import patch_git_commands

# Keep the user's and the system's git config out of the repository under test.
GIT_ENV = {
    "PATH": os.environ["PATH"],
    "HOME": "/nonexistent",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "agent",
    "GIT_AUTHOR_EMAIL": "agent@example.com",
    "GIT_COMMITTER_NAME": "agent",
    "GIT_COMMITTER_EMAIL": "agent@example.com",
}

AGENT_SOURCE_FILES = {
    "app.py": "app v2\n",
    "staged.py": "staged v2\n",
    "pkg/new_module.py": "new module\n",
    "pkg/aidlc_helpers.py": "a package module, not workflow state\n",
}

EXCLUDED_PATHS = [
    "aidlc-docs/inception/requirements.md",
    "aidlc/records/audit.md",
    "aidlc.settings.json",
    ".aidlc/state.json",
    ".aidlc-security.json",
    ".claude/settings.json",
    "CLAUDE.md",
    "AGENTS.md",
    ".mcp.json",
    ".gitignore",
    "setup-env.sh",
]


def git(repo, *args) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, env=GIT_ENV, check=True, capture_output=True, text=True
    ).stdout


def write(repo, path, text):
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


@pytest.fixture()
def repo(tmp_path):
    """A one-commit repository the agent then edits."""
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    for path in ("app.py", "staged.py", "AGENTS.md", ".gitignore"):
        write(repo, path, f"{path} v1\n")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "base")

    write(repo, "app.py", AGENT_SOURCE_FILES["app.py"])
    write(repo, "staged.py", AGENT_SOURCE_FILES["staged.py"])
    git(repo, "add", "staged.py")
    write(repo, "pkg/new_module.py", AGENT_SOURCE_FILES["pkg/new_module.py"])
    write(repo, "pkg/aidlc_helpers.py", AGENT_SOURCE_FILES["pkg/aidlc_helpers.py"])
    for path in EXCLUDED_PATHS:
        write(repo, path, "written by the workflow\n")
    return repo


def head(repo) -> str:
    return git(repo, "rev-parse", "HEAD").strip()


def collect_patch(repo, tmp_path, base_commit) -> str:
    patch_file = tmp_path / "model.patch"
    for command in patch_git_commands(str(patch_file), base_commit):
        subprocess.run(command, cwd=repo, env=GIT_ENV, check=True, capture_output=True)
    return patch_file.read_text()


def test_patch_has_the_agent_changes_and_none_of_the_excluded_paths(repo, tmp_path):
    patch = collect_patch(repo, tmp_path, head(repo))

    patched = re.findall(r"^diff --git a/(\S+) b/", patch, re.MULTILINE)
    assert sorted(patched) == sorted(AGENT_SOURCE_FILES)


def test_patch_applies_to_the_base_commit(repo, tmp_path):
    patch = collect_patch(repo, tmp_path, head(repo))
    base = tmp_path / "base"
    git(tmp_path, "clone", "-q", str(repo), str(base))
    (tmp_path / "patch.diff").write_text(patch)

    git(base, "apply", str(tmp_path / "patch.diff"))

    for path, text in AGENT_SOURCE_FILES.items():
        assert (base / path).read_text() == text
    for path in EXCLUDED_PATHS:
        assert (base / path).exists() == (path in ("AGENTS.md", ".gitignore"))


def test_patch_keeps_work_the_agent_committed(repo, tmp_path):
    base_commit = head(repo)
    git(repo, "commit", "-qam", "agent work")

    patch = collect_patch(repo, tmp_path, base_commit)

    patched = re.findall(r"^diff --git a/(\S+) b/", patch, re.MULTILINE)
    assert sorted(patched) == sorted(AGENT_SOURCE_FILES)
