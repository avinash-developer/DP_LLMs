import math

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import DP_BATCH_SIZE, DP_DELTA, DP_MICROBATCH_SIZE


def _log_add(logx, logy):
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:
        return b
    return math.log1p(math.exp(a - b)) + b


def _compute_log_a_int(q, sigma, alpha):
    log_a = -np.inf
    for i in range(alpha + 1):
        log_coef_i = (
            math.lgamma(alpha + 1)
            - math.lgamma(i + 1)
            - math.lgamma(alpha - i + 1)
            + i * math.log(q)
            + (alpha - i) * math.log(1 - q)
        )
        s = log_coef_i + (i * i - i) / (2 * (sigma ** 2))
        log_a = _log_add(log_a, s)
    return log_a


def compute_rdp(q, sigma, steps, orders):
    rdp_vals = []
    for order in orders:
        if q == 0:
            rdp_vals.append(0.0)
            continue
        if q == 1.0:
            rdp_vals.append(steps * order / (2 * sigma ** 2))
            continue
        log_a = _compute_log_a_int(q, sigma, int(order))
        rdp_vals.append(steps * (log_a / (order - 1)))
    return np.array(rdp_vals)


def get_epsilon_from_rdp(orders, rdp, delta):
    eps_candidates = rdp - math.log(delta) / (np.array(orders) - 1)
    return float(np.nanmin(eps_candidates))


def get_sigma_from_target_epsilon(q, steps, epsilon, delta, init_sigma=10.0, interval=0.5):
    def get_eps_for_sigma(cur_sigma):
        orders = np.arange(2, 64, 1)
        rdp = compute_rdp(q, cur_sigma, steps, orders)
        return get_epsilon_from_rdp(orders, rdp, delta)

    cur_sigma = init_sigma
    for _ in range(4):
        while True:
            cur_eps = get_eps_for_sigma(cur_sigma)
            if cur_eps < epsilon and cur_sigma > interval:
                cur_sigma -= interval
            else:
                cur_sigma += interval
                break
        interval /= 10

    final_eps = get_eps_for_sigma(cur_sigma)
    return cur_sigma, final_eps


def maybe_rebuild_dp_dataloader(train_dataloader, logger):
    if train_dataloader.batch_size == DP_BATCH_SIZE:
        return train_dataloader

    logger.warning(
        f"Using DP batch size {DP_BATCH_SIZE} instead of configured batch size {train_dataloader.batch_size} "
        "for microbatch DP computation stability."
    )
    return DataLoader(
        train_dataloader.dataset,
        shuffle=True,
        batch_size=DP_BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
    )


def calibrate_noise_multiplier(train_dataloader, epsilon):
    dataset_size = len(train_dataloader.dataset)
    sampling_probability = train_dataloader.batch_size / dataset_size
    total_steps = len(train_dataloader)
    return sampling_probability, total_steps, get_sigma_from_target_epsilon(
        q=sampling_probability,
        steps=total_steps,
        epsilon=epsilon,
        delta=DP_DELTA,
    )


def custom_dp_sgd_step(model, batch, optimizer, trainable_params, max_grad_norm, noise_multiplier, device):
    batch_size = batch["label"].shape[0]
    accumulated_grads = [torch.zeros_like(parameter, device=device) for parameter in trainable_params]

    for start_idx in range(0, batch_size, DP_MICROBATCH_SIZE):
        end_idx = min(start_idx + DP_MICROBATCH_SIZE, batch_size)
        current_mb_size = end_idx - start_idx

        optimizer.zero_grad(set_to_none=True)
        sample_inputs = {key: value[start_idx:end_idx] for key, value in batch.items() if key != "label"}
        sample_inputs["labels"] = batch["label"][start_idx:end_idx]

        outputs = model(**sample_inputs)
        loss = outputs.loss
        loss.backward()

        grad_norm_sq = torch.zeros((), device=device)
        for parameter in trainable_params:
            if parameter.grad is not None:
                grad_norm_sq += parameter.grad.detach().pow(2).sum()
        grad_norm = torch.sqrt(grad_norm_sq + 1e-12)
        clip_coef = min(1.0, max_grad_norm / (grad_norm.item() + 1e-6))

        for grad_idx, parameter in enumerate(trainable_params):
            if parameter.grad is not None:
                accumulated_grads[grad_idx].add_(parameter.grad.detach() * clip_coef * current_mb_size)

    optimizer.zero_grad(set_to_none=True)
    noise_std = noise_multiplier * max_grad_norm
    for grad_idx, parameter in enumerate(trainable_params):
        noise = torch.normal(
            mean=0.0,
            std=noise_std,
            size=parameter.shape,
            device=device,
            dtype=parameter.dtype,
        )
        parameter.grad = (accumulated_grads[grad_idx] + noise) / batch_size

    optimizer.step()


def privacy_delta():
    return DP_DELTA


def dp_microbatch_size():
    return DP_MICROBATCH_SIZE
