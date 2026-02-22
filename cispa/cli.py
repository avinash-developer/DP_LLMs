import argparse

from .logging_utils import configure_logging
from .training import run_training


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, choices=["sst2", "qnli", "mnli", "qqp"])
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--epsilon", type=float, default=-1, help="-1 for Infinity (Non-Private), >0 for DP")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def run_cli():
    logger = configure_logging()
    args = build_parser().parse_args()
    score, eps, params = run_training(args, logger)

    with open("temp_result.txt", "w") as result_file:
        result_file.write(f"{score},{params}")

    return score, eps, params
