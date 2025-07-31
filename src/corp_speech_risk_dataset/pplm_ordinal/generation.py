# =============================
# pplm_ordinal/generation.py
# =============================
from __future__ import annotations
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from .pplm import perturb_past
from .device import choose_device
from .utils import set_seed


def pplm_generate(prompt: str, cfg, classifier=None):
    device = choose_device(cfg.device)
    set_seed(cfg.seed)

    tok_name = cfg.tokenizer_name or cfg.model_name
    tokenizer = GPT2TokenizerFast.from_pretrained(tok_name)
    model = GPT2LMHeadModel.from_pretrained(cfg.model_name).to(device)
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    past_key_values = None

    generated = input_ids.clone()

    for _ in range(cfg.length):
        outputs = model(
            input_ids=generated[:, -1:],
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
        )
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        # perturb past for next token
        if classifier is not None:
            past_key_values = perturb_past(
                model=model,
                tokenizer=tokenizer,
                past_key_values=past_key_values,
                prev_tokens=generated,
                config=cfg,
                classifier=classifier,
                step_size=cfg.step_size,
                num_steps=cfg.num_steps,
                grad_norm_threshold=cfg.grad_norm_threshold,
                kl_scale=cfg.kl_scale,
            )
            # recompute logits after perturb
            outputs = model(
                input_ids=generated[:, -1:],
                past_key_values=past_key_values,
                use_cache=True,
            )
            pert_logits = outputs.logits[:, -1, :]
            pert_probs = torch.softmax(pert_logits / cfg.temperature, dim=-1)

            # Interpolate with original (gm_scale trick)
            base_probs = torch.softmax(logits[:, -1, :] / cfg.temperature, dim=-1)
            probs = cfg.gm_scale * pert_probs + (1 - cfg.gm_scale) * base_probs
        else:
            probs = torch.softmax(logits[:, -1, :] / cfg.temperature, dim=-1)

        # top-k / top-p filtering
        next_token = _sample_from_probs(probs, top_k=cfg.top_k, top_p=cfg.top_p)
        generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

    text = tokenizer.decode(generated[0])
    return text


def _sample_from_probs(probs: torch.Tensor, top_k=0, top_p=0.0):
    logits = torch.log(probs + 1e-10)
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1].unsqueeze(1)
        logits = torch.where(
            logits < min_values, torch.full_like(logits, -1e10), logits
        )
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumprobs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        mask = cumprobs > top_p
        mask[:, 0] = False
        sorted_logits[mask] = -1e10
        logits.scatter_(1, sorted_indices, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(1)
