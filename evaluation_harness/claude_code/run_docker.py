from __future__ import annotations

import os
import shlex
import sys
from pathlib import Path
from typing import Mapping
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from base import DockerHarnessBase  # noqa: E402

from prompts import USER_PROMPT_TEMPLATE, load_example_instance

load_dotenv()


ALLOWED_TOOLS = [
    "Bash",
    "Edit",
    "Write",
    "Read",
    "Glob",
    "Grep",
    "LS",
    "WebFetch",
    "NotebookEdit",
    "NotebookRead",
    "TodoRead",
    "TodoWrite",
    "Agent",
]

#: Host env vars passed to the agent; names listed in BENCH_FORWARD_ENV
#: (comma-separated) are passed too.
AGENT_ENV_NAMES = [
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


def get_agent_env(environ: Mapping[str, str] = os.environ) -> dict[str, str]:
    """The agent's env: each forwarded host var that is set and non-empty.

    An unset var stays unset in the container, so it cannot override the
    value the setup script wrote (e.g. ANTHROPIC_MODEL in /root/.claude_env).
    """
    extra = [name.strip() for name in environ.get("BENCH_FORWARD_ENV", "").split(",")]
    env = {name: environ[name] for name in [*AGENT_ENV_NAMES, *extra] if environ.get(name)}
    env.setdefault("CLAUDE_CODE_MAX_OUTPUT_TOKENS", "50000")
    return env


class DockerIntegration(DockerHarnessBase):
    """Claude Code Docker integration.

    Shares the full container lifecycle with ``DockerHarnessBase`` and only
    customizes the agent name (workspace/container prefix) and the env files
    sourced before each in-container command.
    """

    name = "claude"
    env_source_files = [
        "/root/.claude_env",
        "/root/.nvm/nvm.sh",
        "/root/.bashrc",
    ]


def main():

    instance = load_example_instance()
    TASK = instance["problem_statement"]
    DOCKER_IMAGE = instance["image_name"]

    prompt = USER_PROMPT_TEMPLATE.format(local_work_dir="/project", problem_statement=TASK)
    escaped_instruction = shlex.quote(prompt)

    with DockerIntegration(DOCKER_IMAGE, container_work_dir="/project", workspace_root=".", keep_workspace=True) as integration:
        # Set up the workspace
        workspace = integration.setup_persistent_workspace()

        integration.setup_cli_env()

        print(f"🔧 Setting up environment...")

        print(f"🔧 Running Claude...")
        print(f"🔧 Allowed Tools: {' '.join(ALLOWED_TOOLS)}")
        print(f"🔧 Task: {TASK}")
        claude_command = (
            "claude --verbose --output-format stream-json "
            f"-p {escaped_instruction} --allowedTools {' '.join(ALLOWED_TOOLS)}"
        )

        result = integration.execute_in_container(claude_command, env=get_agent_env())
        print(f"🔧 Result: {result}")

        print(f"\n🎉 Session complete!")
        print(f"📁 Your improved code is at: {workspace}")

        return workspace


if __name__ == "__main__":
    main()
