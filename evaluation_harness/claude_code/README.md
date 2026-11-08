# Claude Code Evaluation Harness

This directory contains tools for running Claude Code evaluations on code repositories using Docker containers.

## Overview

The evaluation harness executes Claude Code on code repositories packaged as Docker images, allowing for isolated and reproducible testing environments.

## Files

- **`prompts.py`** - Contains prompt templates and example task definitions for Claude Code interactions
- **`run_docker.py`** - Core Docker integration class (`DockerIntegration`) for managing containerized Claude Code execution
- **`batch_run_docker.py`** - Processes multiple evaluation instances from a JSONL file sequentially
- **`parallel_batch_run.py`** - Runs batch evaluations in parallel across multiple processes for faster processing
- **`setup-env.sh`** - Setup script that installs Claude CLI and dependencies in Docker containers
- **`convert.py`** - Converts the harness `final_results.json` into the standard format (OpenAI messages)

## Usage

### Single Instance

Run a single evaluation using `run_docker.py`:

```bash
python run_docker.py
```

### Batch Processing

Process multiple instances from a JSONL file:

```bash
python batch_run_docker.py --jsonl_file dataset.jsonl --num_instances 10
```

### Parallel Batch Processing

Run batch evaluations in parallel:

```bash
python parallel_batch_run.py --jsonl_file dataset.jsonl --num_processes 4
```

### Comparing Setups

`batch_run_docker.py` takes these options, and `parallel_batch_run.py` passes each one to every child run:

- `--setup_script <path>` is the host script that installs the agent. It can be any host path. The runner copies it to `/opt/setup-env.sh`, outside the workspace the agent sees, and runs it there. If the script fails, the instance stops with an `error` and an empty patch.
- `--setup_assets <dir>` is a host directory that the runner copies to `/opt/bench-assets` before the script runs. The script finds it through `BENCH_ASSETS`.
- `--claude_code_version <version>` reaches the setup script as `CLAUDE_CODE_VERSION`. `setup-env.sh` installs that version, or 1.0.128 when it is unset.
- `--allowed_tools` and `--disallowed_tools` take comma-separated tool names. They default to `ALLOWED_TOOLS` and `DISALLOWED_TOOLS` in `run_docker.py`. Pass `--disallowed_tools ""` to disallow nothing.
- `--timeout <seconds>` limits the agent step (default 3000). A run that reaches it is recorded with `timed_out: true`.
- `--agent_driver <path>` runs the agent step through a host executable instead of `claude -p`. See [Agent Driver](#agent-driver).

The runner takes the patch inside the container, as the diff from the commit the workspace was at when the agent started. The patch includes new files and work the agent staged or committed. It leaves out the root-level workflow files listed in `PATCH_EXCLUDES` in [`../base.py`](../base.py).

### Agent Driver

With `--agent_driver`, the runner writes the prompt to `results/<model>/<timestamp>/artifacts/<instance_id>/prompt.txt` and calls:

```bash
<agent_driver> --container <name> --workdir /project \
    --prompt-file <artifacts dir>/prompt.txt --out-dir <artifacts dir> \
    --timeout <seconds> --allowed-tools <a,b> --disallowed-tools <a,b> \
    --source /root/.claude_env --source /root/.nvm/nvm.sh \
    --source /root/.bashrc --source /root/.bench_agent_env
```

The driver sources each `--source` file in order before it starts `claude` in the container. It writes two files to the artifacts dir: `raw.jsonl`, the CLI's stream-json output, and `summary.json`, whose `status` is `completed`, `timeout`, `stuck` or `error`. It exits 0 for the first three. The runner stops a driver that is still running 300 seconds after `--timeout`. The instance record keeps `raw.jsonl` as `claude_stdout`, so `convert.py` works unchanged, and adds `agent_summary` and `artifacts_dir`.

### Convert

A batch run writes `final_results.json` under `results/<model>/<timestamp>/`; turn it into the standard format with:

```bash
python convert.py --input_dir <run dir> [--output_dir <out dir>]
```

See [`../TRAJECTORY_FORMAT.md`](../TRAJECTORY_FORMAT.md) for the record format. Because trajectories are large, the converter writes the split layout: an index `<DIR>.trials.json` whose `messages` field is a path, plus one `messages/<id>.json` per instance.

## Requirements

- Docker installed and running
- Python 3.x
- Claude API credentials (set via environment variables: `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, etc.)

## Environment Variables

Please copy a example environment file to `.env` and edit it:
```bash
cp .env.sample .env
```

Edit `.env` and fill in your API keys and model configurations:
- `ANTHROPIC_API_KEY` - Your Anthropic API key
- `ANTHROPIC_MODEL` - Model to use (default: "claude-sonnet-4-20250514")
- `ANTHROPIC_BASE_URL` - Optional custom base URL
- `CLAUDE_CODE_MAX_OUTPUT_TOKENS` - Maximum output tokens

To run Claude on Amazon Bedrock, set these instead of the API key:
- `CLAUDE_CODE_USE_BEDROCK=1`
- `AWS_REGION`, the region that serves the model
- `ANTHROPIC_MODEL`, a Bedrock inference profile ID. The default that `setup-env.sh` writes is not a Bedrock ID.
- `AWS_BEARER_TOKEN_BEDROCK`, or `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` and `AWS_SESSION_TOKEN`. The container shares the host network, so on EC2 the instance role works without them.

The runner passes every var named above to Claude, along with `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_SMALL_FAST_MODEL`, `ANTHROPIC_DEFAULT_HAIKU_MODEL`, `ANTHROPIC_DEFAULT_SONNET_MODEL`, `ANTHROPIC_DEFAULT_OPUS_MODEL`, `CLAUDE_CODE_DISABLE_REFUSAL_FALLBACK` (set it to 1 so a refused call is never re-sent to another model), and any names listed in `BENCH_FORWARD_ENV` (comma-separated). It passes only vars that are set and not empty, so an unset `ANTHROPIC_MODEL` keeps the value the setup script wrote. The values reach the container over stdin, in `/root/.bench_agent_env` (mode 0600). The runner never prints them or puts them on a command line.
