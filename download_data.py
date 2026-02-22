import os
import argparse
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GLUE_TASKS = ["sst2", "qnli", "mnli", "qqp"]


def download_glue_data(cache_dir="./glue_data_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    for task in GLUE_TASKS:
        logger.info(f"Downloading: {task}")
        dataset = load_dataset("glue", task, cache_dir=cache_dir)
        logger.info(f"Train: {len(dataset['train'])} examples")
        if task == "mnli":
            logger.info(f"Val:   {len(dataset['validation_matched'])} examples")
        else:
            logger.info(f"Val:   {len(dataset['validation'])} examples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="./glue_data_cache")
    args = parser.parse_args()
    download_glue_data(args.cache_dir)
