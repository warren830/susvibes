#!/usr/bin/env python3
"""
Docker Batch Runner for Multiple Tasks

This script reads instances from a JSONL file, processes each one
with Docker-based Claude execution, and saves the git diff results to JSON files.
"""

import json
import subprocess
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
import argparse
import shlex

# Import the Docker integration classes
from run_docker import DockerIntegration, USER_PROMPT_TEMPLATE, ALLOWED_TOOLS, DISALLOWED_TOOLS, claude_tool_flags, get_agent_env

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import apply_safety_hint  # noqa: E402

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('docker_batch_run.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

#: Seconds an agent driver gets past --timeout to stop the agent and write its files.
DRIVER_GRACE_SECONDS = 300

def get_options():
    parser = argparse.ArgumentParser(description='Process multiple tasks with Docker-based Claude execution')
    parser.add_argument('--jsonl_file', type=str, default='../datasets/susvibes_dataset.jsonl', help='Path to the JSONL file')
    parser.add_argument('--results_dir', type=str, default='results', help='Path to the results directory')
    parser.add_argument('--workspace_root', type=str, default='logs/workspace', help='Path to the workspace root directory')
    parser.add_argument('--start_idx', type=int, default=0, help='Start index of the instances to process')
    parser.add_argument('--num_instances', type=int, default=2, help='Number of instances to process')
    parser.add_argument('--load_from_file', type=str, default=None, help='Path to the file to load from')
    parser.add_argument('--model', type=str, default="claude", help='Model to use')
    parser.add_argument('--setup_script', type=str, default="setup-env.sh", help='Setup script to run in container')
    parser.add_argument('--setup_assets', type=str, default=None, help='Host directory copied to /opt/bench-assets in the container before the setup script runs; the script sees it as BENCH_ASSETS')
    parser.add_argument('--claude_code_version', type=str, default=None, help="Claude Code version the setup script installs, passed to it as CLAUDE_CODE_VERSION (default: the script's own)")
    parser.add_argument('--allowed_tools', type=str, default=','.join(ALLOWED_TOOLS), help='Comma-separated tools Claude may use without asking')
    parser.add_argument('--disallowed_tools', type=str, default=','.join(DISALLOWED_TOOLS), help='Comma-separated tools Claude may not use')
    parser.add_argument('--timeout', type=int, default=3000, help='Seconds the agent may run on one instance')
    parser.add_argument('--agent_driver', type=str, default=None, help='Host executable that runs the agent step instead of claude -p (see README.md)')
    parser.add_argument('--timestamp_suffix', type=str, default=None, help='Suffix to append to timestamp for unique directory names (for parallel runs)')
    parser.add_argument('--keep_workspace', action='store_true', default=False, help='Whether to keep the workspace after cleanup')
    parser.add_argument('--strategy', type=str, default='none', choices=['none', 'generic'],
                        help='Safety strategy applied to the problem statement (none = no hint)')
    return parser


def split_tools(value: str) -> List[str]:
    """Tool names from a comma-separated option value."""
    return [tool.strip() for tool in value.split(',') if tool.strip()]


def run_agent_driver(
    integration: DockerIntegration,
    agent_driver: str,
    prompt: str,
    artifacts_dir: Path,
    timeout: int,
    allowed_tools: List[str],
    disallowed_tools: List[str],
) -> Dict[str, Any]:
    """
    Run the agent step through a host-side driver instead of claude -p.

    The driver reads the prompt from a file in artifacts_dir and writes
    raw.jsonl (the CLI's stream-json output) and summary.json there.

    Returns:
        The keys execute_in_container returns, plus agent_summary
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    prompt_file = artifacts_dir / "prompt.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    cmd = [
        agent_driver,
        "--container", integration.workspace_container,
        "--workdir", integration.container_work_dir,
        "--prompt-file", str(prompt_file),
        "--out-dir", str(artifacts_dir),
        "--timeout", str(timeout),
        "--allowed-tools", ",".join(allowed_tools),
        "--disallowed-tools", ",".join(disallowed_tools),
    ]
    for env_file in integration.sourced_env_files():
        cmd += ["--source", env_file]

    start_time = time.time()
    try:
        driver = subprocess.run(cmd, stderr=subprocess.PIPE, text=True, timeout=timeout + DRIVER_GRACE_SECONDS)
        return_code, stderr, driver_timed_out = driver.returncode, driver.stderr, False
    except subprocess.TimeoutExpired:
        return_code, driver_timed_out = -1, True
        stderr = f"Agent driver timed out after {timeout + DRIVER_GRACE_SECONDS} seconds"

    raw_file = artifacts_dir / "raw.jsonl"
    summary_file = artifacts_dir / "summary.json"
    summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.exists() else None
    status = summary.get("status") if summary else None
    return {
        "stdout": raw_file.read_text(encoding="utf-8") if raw_file.exists() else "",
        "stderr": stderr,
        "return_code": return_code,
        "execution_time": time.time() - start_time,
        "success": return_code == 0 and status == "completed",
        "timed_out": driver_timed_out or status == "timeout",
        "agent_summary": summary,
    }


def load_instances(jsonl_file: str) -> List[Dict[str, Any]]:
    """
    Load instances from JSONL file.
    
    Args:
        jsonl_file: Path to the JSONL file
        
    Returns:
        List of instance dictionaries
    """
    instances = []
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        instance = json.loads(line)
                        instances.append(instance)
                    except json.JSONDecodeError as e:
                        logger.error(f"Error parsing line {line_num}: {e}")
                        continue
        logger.info(f"Loaded {len(instances)} instances from {jsonl_file}")
        return instances
    except FileNotFoundError:
        logger.error(f"File not found: {jsonl_file}")
        return []
    except Exception as e:
        logger.error(f"Error loading instances: {e}")
        return []


async def process_instance(
    instance: Dict[str, Any],
    index: int,
    total: int,
    model: str,
    workspace_root: str = ".",
    setup_script: str = "setup-env.sh",
    keep_workspace: bool = True,
    strategy: str = "none",
    setup_assets: Optional[str] = None,
    claude_code_version: Optional[str] = None,
    allowed_tools: List[str] = ALLOWED_TOOLS,
    disallowed_tools: List[str] = DISALLOWED_TOOLS,
    timeout: int = 3000,
    agent_driver: Optional[str] = None,
    artifacts_root: str = "artifacts",
) -> Dict[str, Any]:
    """
    Process a single instance with Docker-based Claude execution.

    Args:
        instance: Instance dictionary from JSONL
        index: Current instance index (0-based)
        total: Total number of instances
        model: Model name
        workspace_root: Root directory for workspaces
        setup_script: Setup script to run in container
        setup_assets: Host directory the setup script gets as BENCH_ASSETS
        claude_code_version: Claude Code version the setup script installs
        allowed_tools: Tools Claude may use without asking
        disallowed_tools: Tools Claude may not use
        timeout: Seconds the agent may run
        agent_driver: Host executable that runs the agent step instead of claude -p
        artifacts_root: Directory holding each instance's driver artifacts

    Returns:
        Result dictionary with instance_id, model, and model_patch
    """
    instance_id = instance.get('instance_id', f'unknown_{index}')
    image_name = instance.get('image_name', '')
    problem_statement = instance.get('problem_statement', '')
    workspace = None
    
    logger.info(f"Processing instance {index + 1}/{total}: {instance_id}")
    
    if not image_name:
        logger.error(f"No image_name found for instance {instance_id}")
        return {
            "instance_id": instance_id,
            "model_name_or_path": model,
            "model_patch": "",
            "error": "No image_name found"
        }
    
    if not problem_statement:
        logger.error(f"No problem_statement found for instance {instance_id}")
        return {
            "instance_id": instance_id,
            "model_name_or_path": model, 
            "model_patch": "",
            "error": "No problem_statement found"
        }
    
    try:
        # Set up Docker integration
        logger.info(f"Starting Docker integration for {instance_id}")
        
        env = get_agent_env()
        
        if strategy != "none":
            problem_statement = apply_safety_hint(problem_statement)

        # Escape the problem statement for shell execution
        prompt = USER_PROMPT_TEMPLATE.format(local_work_dir="/project", problem_statement=problem_statement)
        escaped_instruction = shlex.quote(prompt)

        with DockerIntegration(image_name, container_work_dir="/project", workspace_root=workspace_root, keep_workspace=keep_workspace) as integration:
        
            # Set up the workspace
            print(f"🔧 Setting up environment...")
            workspace = integration.setup_persistent_workspace()

            setup_env = {"CLAUDE_CODE_VERSION": claude_code_version} if claude_code_version else {}
            setup_result = integration.setup_cli_env(setup_script_path=setup_script, env=setup_env, assets_dir=setup_assets)

            # Going on would run the agent on a half-installed setup, silently measuring another configuration
            if not setup_result["success"]:
                logger.error(f"Environment setup failed for {instance_id}: {setup_result['stderr']}")
                return {
                    "instance_id": instance_id,
                    "model_name_or_path": model,
                    "model_patch": "",
                    "error": f"Environment setup failed: {setup_result['stderr']}",
                    "workspace": str(workspace)
                }

            base_commit = integration.head_commit()
            integration.write_agent_env(env)

            # Run Claude with the problem statement
            logger.info(f"Running Claude for {instance_id}")
            if agent_driver:
                artifacts_dir = Path(artifacts_root, instance_id).resolve()
                result = run_agent_driver(integration, agent_driver, prompt, artifacts_dir, timeout, allowed_tools, disallowed_tools)
            else:
                tool_flags = claude_tool_flags(allowed_tools, disallowed_tools)
                claude_command = (
                    "claude --verbose --output-format stream-json "
                    f"-p {escaped_instruction} {tool_flags}"
                )
                result = integration.execute_in_container(claude_command, timeout=timeout)
            
            if result["success"]:
                logger.info(f"Claude execution completed successfully for {instance_id}")
            else:
                logger.error(f"Claude execution failed for {instance_id}: {result['stderr']}")


            # Show final status
            print(f"\n🎉 Session complete!")
            print(f"📁 Your improved code is at: {workspace}")
            
            # Get git diff
            diff_text = integration.extract_patch(base_commit)
            
            result_dict = {
                "instance_id": instance_id,
                "model_name_or_path": model,
                "model_patch": diff_text,
                "workspace": str(workspace),
                "claude_stdout": result.get("stdout", ""),
                "claude_stderr": result.get("stderr", ""),
                "claude_success": result.get("success", False),
                "timed_out": result.get("timed_out", False)
            }
            if agent_driver:
                result_dict["agent_summary"] = result["agent_summary"]
                result_dict["artifacts_dir"] = str(artifacts_dir)
            
            # Clean up the integration
            integration.cleanup()
            
            logger.info(f"Successfully processed {instance_id}")
            return result_dict
        
    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}")
        return {
            "instance_id": instance_id,
            "model_name_or_path": model,
            "model_patch": "",
            "error": str(e),
            "workspace": str(workspace)
        }


async def main(args):
    """
    Main function to process all instances.
    """
    # Configuration
    jsonl_file = args.jsonl_file
    model = args.model
    workspace_root = args.workspace_root
    setup_script = args.setup_script

    timestamp = int(time.time())
    # Append suffix for unique directory names in parallel runs
    if args.timestamp_suffix:
        timestamp_str = f"{timestamp}_{args.timestamp_suffix}"
    else:
        timestamp_str = str(timestamp)
    workspace_root = Path(args.results_dir, "workspace")
    workspace_root.mkdir(parents=True, exist_ok=True)
    results_dir = Path(args.results_dir, model, timestamp_str)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Load instances
    instances = load_instances(jsonl_file)
    if not instances:
        logger.error("No instances to process")
        return
    
    if args.load_from_file:
        results = json.load(open(args.load_from_file))
        completed_instances = [ins["instance_id"] for ins in results]
        waiting_instances = [ins for ins in instances if ins["instance_id"] not in completed_instances]
        instances = waiting_instances
        logger.info(f"Processing {len(instances)} instances after {args.load_from_file}")
    
    if args.num_instances is None:
        instances = instances[args.start_idx:]
    else:
        instances = instances[args.start_idx:args.start_idx + args.num_instances]
    logger.info(f"Starting processing of {len(instances)} instances")
    
    # Process each instance
    results = []
    for i, instance in enumerate(instances):
        try:
            result = await process_instance(
                instance, i, len(instances), model, 
                workspace_root=workspace_root, setup_script=setup_script,
                keep_workspace=args.keep_workspace,
                strategy=args.strategy,
                setup_assets=args.setup_assets,
                claude_code_version=args.claude_code_version,
                allowed_tools=split_tools(args.allowed_tools),
                disallowed_tools=split_tools(args.disallowed_tools),
                timeout=args.timeout,
                agent_driver=args.agent_driver,
                artifacts_root=str(results_dir / "artifacts"),
            )
            
            # Save intermediate results every instance
            intermediate_file = results_dir / f"intermediate_{i + 1}.json"
            with open(intermediate_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved intermediate results to {intermediate_file}")

            results.append(result)
                
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error processing instance {i}: {e}")
            continue
    
    # Save final results
    output_file = results_dir / f"final_results.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Processing complete. Results saved to {output_file}")
    logger.info(f"Processed {len(results)} instances successfully")


if __name__ == "__main__":
    args = get_options().parse_args()
    asyncio.run(main(args))
