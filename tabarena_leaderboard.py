import argparse
import os
from tabrepo.nips2025_utils.end_to_end import EndToEnd, EndToEndResults


def main():
    parser = argparse.ArgumentParser(description="Run TabPFN end-to-end evaluation.")
    parser.add_argument(
        "--path-raw",
        default=os.environ.get("PYTHONPATH") + "tabarena_benchmarking_examples/tabarena_minimal_example/custom_tabpfn_flashattn/data/",
        help="Path to the raw dataset directory",
    )
    parser.add_argument(
        "--out",
        default=os.environ.get("PYTHONPATH") + "tabarena_benchmarking_examples/tabarena_minimal_example/custom_tabpfn_flashattn/leaderboard.csv",
        help="Path to save the leaderboard results (json or csv).",
    )
    args = parser.parse_args()

    # Run EndToEnd pipeline
    end_to_end = EndToEnd.from_path_raw(path_raw=args.path_raw)
    print("Model results:")
    print(end_to_end.model_results)

    # Load cached results
    end_to_end_results = EndToEndResults.from_cache(method="CustomTabPFNv2")

    # Compare on TabArena
    leaderboard = end_to_end_results.compare_on_tabarena(args.path_raw)
    print("Leaderboard:")
    print(leaderboard)

    # Save leaderboard
    if args.out.endswith(".json"):
        leaderboard.to_json(args.out, orient="records", indent=2)
    elif args.out.endswith(".csv"):
        leaderboard.to_csv(args.out, index=False)
    else:
        leaderboard.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
