# =============================
# pplm_ordinal/pplm.py
# =============================
"""Core PPLM perturbation functions (adapted & simplified from Dathathri et al.)."""
from __future__ import annotations
import torch
import torch.nn.functional as F
from typing import List, Tuple


def _aggregate_hidden(
    hidden_states: torch.Tensor, mode: str = "last_hidden_mean"
) -> torch.Tensor:
    # hidden_states: (B, T, H)
    if mode == "last_hidden":
        return hidden_states[:, -1, :]
    # default: mean over sequence
    return hidden_states.mean(dim=1)


def perturb_past(
    model,
    tokenizer,
    past_key_values,
    prev_tokens: torch.Tensor,
    config,
    classifier,
    step_size: float,
    num_steps: int,
    grad_norm_threshold: float,
    kl_scale: float,
):
    device = prev_tokens.device
    if past_key_values is None:
        raise ValueError("past_key_values must not be None for perturbation.")

    # Make past tensors require grad
    past = [p.detach().clone().requires_grad_(True) for p in past_key_values]

    # Store original distribution for KL
    with torch.no_grad():
        outputs = model(
            input_ids=prev_tokens, past_key_values=past_key_values, use_cache=True
        )
        unpert_logits = outputs.logits[:, -1, :]
        unpert_probs = F.softmax(unpert_logits, dim=-1)

    for _ in range(num_steps):
        outputs = model(
            input_ids=prev_tokens,
            past_key_values=tuple(past),
            use_cache=True,
            output_hidden_states=True,
        )
        logits = outputs.logits[:, -1, :]
        hidden_states = outputs.hidden_states[-1]  # (B, T, H)
        reps = _aggregate_hidden(hidden_states, mode=config.input_rep)

        # Classifier loss
        if classifier is None:
            raise ValueError(
                "Classifier is None. Provide a classifier_path or custom classifier."
            )
        class_logits = classifier(reps)
        loss_cls = classifier.loss_for_class(class_logits, config.class_id)

        # KL loss to original LM distribution
        probs = F.softmax(logits, dim=-1)
        loss_kl = kl_scale * torch.sum(
            probs * (torch.log(probs + 1e-10) - torch.log(unpert_probs + 1e-10))
        )

        loss = loss_cls + loss_kl
        loss.backward()

        # gradient step on past
        grad_norm = 0.0
        for p in past:
            if p.grad is not None:
                grad_norm += p.grad.data.norm() ** 2
        grad_norm = grad_norm.sqrt()
        if grad_norm_threshold > 0 and grad_norm > grad_norm_threshold:
            scale = grad_norm_threshold / (grad_norm + 1e-6)
        else:
            scale = 1.0
        for p in past:
            if p.grad is not None:
                p.data = p.data - step_size * scale * p.grad.data
                p.grad.detach_()
                p.grad.zero_()

    return tuple(past)
