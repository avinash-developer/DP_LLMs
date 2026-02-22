import random

import evaluate
import numpy as np
import torch
from tqdm import tqdm

from .config import EPOCHS, MAX_GRAD_NORM
from .data import get_data
from .dp import (
    calibrate_noise_multiplier,
    custom_dp_sgd_step,
    dp_microbatch_size,
    maybe_rebuild_dp_dataloader,
    privacy_delta,
)
from .modeling import count_parameters, get_trainable_parameters, setup_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_learning_rate(method):
    if "soft-prompt" in method or "prefix" in method:
        return 1e-2
    if "ia3" in method:
        return 5e-3
    return 1e-3


def run_training(args, logger):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using Device: {device}")

    train_dl, eval_dl, num_labels = get_data(args.dataset, cache_dir="./glue_data_cache")

    model = setup_model(args.method, num_labels)
    model.to(device)

    trainable_param_count = count_parameters(model)
    logger.info(f"Trainable Parameters: {trainable_param_count}")

    learning_rate = choose_learning_rate(args.method)
    logger.info(f"Using Learning Rate: {learning_rate}")

    trainable_parameters = get_trainable_parameters(model)
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate)

    noise_multiplier = 0.0
    calibrated_epsilon = None
    sampling_probability = None

    if args.epsilon > 0:
        logger.info(f"Enabling custom DP-SGD with target epsilon: {args.epsilon}")
        train_dl = maybe_rebuild_dp_dataloader(train_dl, logger)
        sampling_probability, per_epoch_steps, (noise_multiplier, calibrated_epsilon) = calibrate_noise_multiplier(
            train_dataloader=train_dl,
            epsilon=args.epsilon,
        )
        logger.info(
            f"Custom DP-SGD active | epsilon={args.epsilon} delta={privacy_delta()} "
            f"max_grad_norm={MAX_GRAD_NORM} noise_multiplier={noise_multiplier:.4f} "
            f"sampling_probability={sampling_probability:.6f} calibrated_epsilon≈{calibrated_epsilon:.4f} "
            f"dp_batch_size={train_dl.batch_size} microbatch_size={dp_microbatch_size()} "
            f"steps_per_epoch={per_epoch_steps}"
        )

    metric = evaluate.load("glue", args.dataset if args.dataset != "mnli" else "mnli")

    logger.info("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        for batch in tqdm(train_dl, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
            batch = {key: value.to(device) for key, value in batch.items()}
            if args.epsilon > 0:
                custom_dp_sgd_step(
                    model=model,
                    batch=batch,
                    optimizer=optimizer,
                    trainable_params=trainable_parameters,
                    max_grad_norm=MAX_GRAD_NORM,
                    noise_multiplier=noise_multiplier,
                    device=device,
                )
            else:
                model_inputs = {key: value for key, value in batch.items() if key != "label"}
                model_inputs["labels"] = batch["label"]
                outputs = model(**model_inputs)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        model.eval()
        for batch in eval_dl:
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.no_grad():
                model_inputs = {key: value for key, value in batch.items() if key != "label"}
                model_inputs["labels"] = batch["label"]
                outputs = model(**model_inputs)
            predictions = torch.argmax(outputs.logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["label"])

        eval_result = metric.compute()
        logger.info(f"Epoch {epoch + 1} Result: {eval_result}")

    if "accuracy" in eval_result:
        score = eval_result["accuracy"]
    elif "f1" in eval_result:
        score = eval_result["f1"]
    elif "matthews_correlation" in eval_result:
        score = eval_result["matthews_correlation"]
    else:
        score = 0.0

    final_epsilon = "Infinity" if args.epsilon <= 0 else (calibrated_epsilon or args.epsilon)
    return score, final_epsilon, trainable_param_count
