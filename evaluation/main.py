import argparse
import os
import subprocess
import json

def run_m3tooleval(model_path, output_dir, api_base, api_port, action_mode, task_regex_filter):
    """
    Runs the M³ToolEval benchmark.
    """
    print("Running M³ToolEval...")
    # This is a placeholder for the actual command to run the benchmark.
    # It will be replaced with the actual command in the future.
    command = f"""
    python3 ../code-act/scripts/eval/m3tooleval/main.py \
        --model {model_path} \
        --output_dir {output_dir}/m3tooleval \
        --action_mode {action_mode} \
        --task_regex_filter \"{task_regex_filter}\"
    """
    print(f"Executing command:\n{command}")
    # subprocess.run(command, shell=True, check=True)

def run_api_bank(model_path, output_dir, api_base, api_port):
    """
    Runs the API-Bank benchmark.
    """
    print("Running API-Bank...")
    # This is a placeholder for the actual command to run the benchmark.
    command = "echo 'Running API-Bank...'"
    print(f"Executing command:\n{command}")
    # subprocess.run(command, shell=True, check=True)

def run_mint_bench(model_path, output_dir):
    """
    Runs the MINT-Bench benchmark.
    """
    print("Running MINT-Bench...")
    # This is a placeholder for the actual command to run the benchmark.
    command = "echo 'Running MINT-Bench...'"
    print(f"Executing command:\n{command}")
    # subprocess.run(command, shell=True, check=True)

def run_mmlu(model_path, output_dir, ntrain, data_dir):
    """
    Runs the MMLU benchmark using the direct model loading script.
    """
    print("Running MMLU...")
    command = f"""
    python3 mmlu/evaluate_mmlu.py \
        --model {model_path} \
        --save_dir {output_dir}/mmlu \
        --data_dir {data_dir} \
        --ntrain {ntrain}
    """
    print(f"Executing command:\n{command}")
    # Note: This command is run in the 'developer2/evaluation' directory.
    # The script needs to be called from there.
    subprocess.run(command, shell=True, check=True, cwd='.')


def main():
    parser = argparse.ArgumentParser(description="Run evaluation benchmarks for a given model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Hugging Face model checkpoint.")
    parser.add_argument("--tests", nargs='+', required=True, help="List of tests to run (e.g., mmlu, m3tooleval).")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save the evaluation results.")

    # API-dependent test arguments
    parser.add_argument("--api_base", type=str, default="http://localhost", help="Base URL for the model's API (for API-dependent tests).")
    parser.add_argument("--api_port", type=int, default=8080, help="Port for the model's API (for API-dependent tests).")
    
    # M³ToolEval specific arguments
    parser.add_argument("--action_mode", type=str, default="code_as_action", choices=["text_as_action", "json_as_action", "code_as_action"], help="Action mode for M³ToolEval.")
    parser.add_argument("--task_regex_filter", type=str, default=".*", help="Regex to filter tasks for M³ToolEval.")
    
    # MMLU specific arguments
    parser.add_argument("--ntrain", "-k", type=int, default=5, help="Number of few-shot examples to use for MMLU.")
    parser.add_argument("--mmlu_data_dir", type=str, default="data/eval/mmlu", help="Directory where MMLU data is stored.")

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # --- Run evaluations ---

    if "mmlu" in args.tests:
        run_mmlu(args.model_path, args.output_dir, args.ntrain, args.mmlu_data_dir)

    # --- API-dependent tests ---
    # Setup API environment variable only if needed
    api_tests = ["m3tooleval", "api-bank", "mint-bench"]
    if any(test in args.tests for test in api_tests):
        print("Setting up API connection for relevant tests...")
        os.environ["OPENAI_API_BASE"] = f"{args.api_base}:{args.api_port}/v1"

        if "m3tooleval" in args.tests:
            run_m3tooleval(args.model_path, args.output_dir, args.api_base, args.api_port, args.action_mode, args.task_regex_filter)

        if "api-bank" in args.tests:
            run_api_bank(args.model_path, args.output_dir, args.api_base, args.api_port)

        if "mint-bench" in args.tests:
            run_mint_bench(args.model_path, args.output_dir)

    print("Evaluation finished.")

if __name__ == "__main__":
    main()
