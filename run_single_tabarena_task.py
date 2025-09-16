import argparse
import openml
import sys
from pathlib import Path

from tabrepo.benchmark.experiment import run_experiments_new
from tabrepo.models.tabpfnv2.generate import gen_tabpfnv2

def main():
    """
    Main function to run a single TabArena benchmark task.
    This script is designed to be called by a Slurm job array, where each job
    handles one task.
    """
    # --- Argument Parsing ---
    # Set up a parser to accept command-line arguments. This makes the script
    # more flexible and easier to test.
    parser = argparse.ArgumentParser(
        description="Run a single TabArena benchmark task for a given index."
    )
    parser.add_argument(
        "--task-index",
        type=int,
        required=True,
        help="The 0-based index of the task to run from the TabArena suite."
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        required=True,
        help="The base directory where experiment results will be saved."
    )
    args = parser.parse_args()

    print(f"🚀 Starting benchmark for Task Index: {args.task_index}")

    # --- Get the specific task for this job ---
    # Get the full list of 51 tasks from the TabArena suite
    try:
        all_task_ids = openml.study.get_suite("tabarena-v0.1").tasks
        print(f"Successfully fetched {len(all_task_ids)} tasks from OpenML suite 'tabarena-v0.1'.")
    except Exception as e:
        print(f"❌ Critical Error: Failed to fetch OpenML suite. {e}", file=sys.stderr)
        sys.exit(1)

    # Select the single task ID for this specific job using the provided index
    try:
        task_to_run = [all_task_ids[args.task_index]]
        print(f"✅ This job will process OpenML Task ID: {task_to_run[0]}")
    except IndexError:
        print(f"❌ Critical Error: Task index {args.task_index} is out of range. "
              f"The suite only has {len(all_task_ids)} tasks (0 to {len(all_task_ids)-1}).", file=sys.stderr)
        sys.exit(1)

    # --- Define Your Custom Model Experiment ---
    # This example uses the default TabPFN model config.
    # You can replace this with your custom model configuration.
    experiments = gen_tabpfnv2.generate_all_bag_experiments(
        num_random_configs=0,  # Using only the default config
    )
    print("🧪 Model experiments generated.")

    # --- Run the Experiment ---
    # The 'run_experiments_new' function will run your model on the single task specified.
    # It saves the output in a subdirectory within the base directory.
    output_path = Path(args.base_dir)
    output_path.mkdir(parents=True, exist_ok=True) # Ensure the base directory exists

    run_experiments_new(
        output_dir=str(output_path),
        model_experiments=experiments,
        tasks=task_to_run,
        repetitions_mode="TabArena-Lite",
    )

    print(f"🎉 Successfully completed benchmarking for Task ID: {task_to_run[0]}")
    print(f"📄 Results saved in: {args.base_dir}")

if __name__ == "__main__":
    main()
