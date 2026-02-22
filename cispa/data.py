from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from .config import BATCH_SIZE, MAX_SEQ_LENGTH, MODEL_NAME


def get_data(dataset_name, cache_dir="./glue_data_cache"):
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    glue_task = dataset_name
    validation_key = "validation_matched" if dataset_name == "mnli" else "validation"

    raw_datasets = load_dataset("glue", glue_task, cache_dir=cache_dir)

    def tokenize_function(examples):
        if glue_task == "sst2":
            args = (examples["sentence"],)
        elif glue_task == "qnli":
            args = (examples["question"], examples["sentence"])
        elif glue_task == "mnli":
            args = (examples["premise"], examples["hypothesis"])
        elif glue_task == "qqp":
            args = (examples["question1"], examples["question2"])
        else:
            args = (examples["sentence"],)

        return tokenizer(*args, padding="max_length", truncation=True, max_length=MAX_SEQ_LENGTH)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    allowed_columns = ["input_ids", "attention_mask", "token_type_ids", "label"]
    tokenized_datasets = tokenized_datasets.remove_columns(
        [column for column in tokenized_datasets["train"].column_names if column not in allowed_columns]
    )
    tokenized_datasets.set_format("torch")

    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
    )
    eval_dataloader = DataLoader(
        tokenized_datasets[validation_key],
        batch_size=BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
    )

    num_labels = len(raw_datasets["train"].features["label"].names)
    return train_dataloader, eval_dataloader, num_labels
