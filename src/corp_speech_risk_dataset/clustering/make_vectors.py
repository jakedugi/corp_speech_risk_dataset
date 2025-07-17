#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import time
import torch
from transformers import GPT2Model, GPT2TokenizerFast
from torch.nn.utils.rnn import pad_sequence

from corp_speech_risk_dataset.clustering.reverse_utils import wl_vector
from corp_speech_risk_dataset.encoding.tokenizer import SentencePieceTokenizer


def build_concat_vectors(
    meta_path: str | Path,
    out_npy: str | Path = "concat_vectors.npy",
):
    """
    Rebuilds and saves the full (N × D) float32 array of features:
      [ text_embedding | WL_dense_vector ]
    """
    meta = json.loads(Path(meta_path).read_text())
    # smoke test
    # meta = meta[:64]
    N = len(meta)

    # ── Prepare GPT-2 for on‐the‐fly embeddings ──────────────────────────
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2Model.from_pretrained("gpt2")
    model.eval()
    # send model to MPS (Apple Silicon GPU)
    # prefer the new MPS API
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    # Embedding dimension from the model config
    E = model.config.hidden_size
    D = 2048  # WL dimension

    out = np.zeros((N, E + D), dtype=np.float32)
    BATCH_SIZE = 16
    i = 0
    # pre-filter entries with non-empty sp_ids
    entries = [m for m in meta if m.get("sp_ids")]
    total = len(entries)
    # compute 25%, 50%, 75% thresholds
    thresholds = {1: total // 4, 2: total // 2, 3: 3 * total // 4}
    printed = set()
    t0 = time.perf_counter()
    for start in range(0, total, BATCH_SIZE):
        batch = entries[start : start + BATCH_SIZE]
        # at each quarter, print once
        for q, thr in thresholds.items():
            if start >= thr and q not in printed:
                elapsed = time.perf_counter() - t0
                print(f"→ {q*25}% done at {elapsed:.1f}s elapsed", flush=True)
                printed.add(q)
        # build list of token-ID tensors, send to device
        toks = [torch.tensor(m["sp_ids"], dtype=torch.long) for m in batch]
        toks = [t.to(device) for t in toks]
        # pad to same length (batch, L_max)
        padded = pad_sequence(toks, batch_first=True)
        with torch.no_grad():
            hidden = model(input_ids=padded).last_hidden_state  # (B, L_max, E)
        # mean-pool and move back to CPU+numpy
        embs = hidden.mean(dim=1).cpu().numpy().astype(np.float32)

        # assign embeddings + WL vectors back to `out`
        for emb, m in zip(embs, batch):
            out[i, :E] = emb
            wl_dense = wl_vector(m["wl_indices"], m["wl_counts"], dim=D)
            out[i, E:] = wl_dense
            i += 1
    # save single array
    np.save(out_npy, out, allow_pickle=False)
    print(f"Wrote {out.shape} → {out_npy}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(
        description="Build concat_vectors.npy from metadata.json"
    )
    p.add_argument("--meta", required=True, help="Path to metadata.json")
    p.add_argument("--out", default="concat_vectors.npy", help="Output .npy file")
    args = p.parse_args()

    build_concat_vectors(args.meta, args.out)
