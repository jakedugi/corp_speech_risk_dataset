# =============================
# tests/test_pplm_step.py
# =============================
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from pplm_ordinal.pplm import perturb_past
from pplm_ordinal.classifier_api import SoftmaxOrdinalClassifier
from pplm_ordinal.config import PPLMConfig


def test_perturb_past_runs():
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.eval()
    cfg = PPLMConfig(num_classes=3)
    text = "hello world"
    ids = tok.encode(text, return_tensors="pt")
    out = model(ids, use_cache=True, output_hidden_states=True)
    past = out.past_key_values
    cls = SoftmaxOrdinalClassifier(in_dim=model.config.n_embd, num_classes=3)
    new_past = perturb_past(
        model,
        tok,
        past,
        ids,
        cfg,
        cls,
        step_size=0.01,
        num_steps=1,
        grad_norm_threshold=1.0,
        kl_scale=0.01,
    )
    assert isinstance(new_past, tuple)
    assert len(new_past) == len(past)
