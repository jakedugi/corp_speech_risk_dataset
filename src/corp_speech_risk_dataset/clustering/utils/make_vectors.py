#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import time
import torch
from torch.nn.utils.rnn import pad_sequence
from typing import Literal

from corp_speech_risk_dataset.encoding.stembedder import get_sentence_embedder
from corp_speech_risk_dataset.clustering.utils.reverse_utils import wl_vector
from corp_speech_risk_dataset.encoding.tokenizer import SentencePieceTokenizer

# we’ll import GPT-2 classes only if needed
from transformers import GPT2Model, GPT2TokenizerFast


def build_concat_vectors(
    meta_path: str | Path,
    out_npy: str | Path = "concat_vectors.npy",
    text_model: Literal["gpt2", "st"] = "gpt2",
):
    """
    Rebuilds and saves the full (N × D) float32 array of features:
      [ text_embedding | WL_dense_vector ]
    """
    meta = json.loads(Path(meta_path).read_text())
    # smoke test
    # meta = meta[:64]
    N = len(meta)

    # ── Determine embedding dimension (from metadata or models) ─────────────
    sample = meta[0]

    # If metadata already has a dense st_emb, use its length:
    if "st_emb" in sample and sample["st_emb"] is not None:
        E = len(sample["st_emb"])
        use_metadata_embeddings = True
    else:
        use_metadata_embeddings = False
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if text_model == "gpt2":
            gpt_tok = GPT2TokenizerFast.from_pretrained("gpt2")
            gpt_mod = GPT2Model.from_pretrained("gpt2").eval().to(device)
            E = gpt_mod.config.hidden_size
        else:
            st_model = get_sentence_embedder(None)
            E = st_model.get_sentence_embedding_dimension()

    D = 2048  # WL dimension

    out = np.zeros((N, E + D), dtype=np.float32)
    BATCH_SIZE = 64  # adjust for memory vs. speed
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
        # ── Get text embeddings ─────────────────────────────────────────────
        if use_metadata_embeddings:
            # Pull precomputed vectors right out of JSON
            embs = np.vstack([m["st_emb"] for m in batch]).astype(np.float32)
        else:
            if text_model == "gpt2":
                enc = gpt_tok(
                    [m["text"] for m in batch],
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                ).to(device)
                with torch.no_grad():
                    hidden = gpt_mod(**enc).last_hidden_state
                embs = hidden.mean(dim=1).cpu().numpy().astype(np.float32)
            else:
                texts = [m["text"] for m in batch]
                embs = st_model.encode(
                    texts,
                    batch_size=BATCH_SIZE,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ).astype(np.float32)

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
    p.add_argument(
        "--text-model",
        choices=["gpt2", "st"],
        default="gpt2",
        help="`gpt2` for legacy GPT-2 mean-pool (default), or `st` for Sentence-Transformer",
    )
    args = p.parse_args()

    build_concat_vectors(
        meta_path=args.meta,
        out_npy=args.out,
        text_model=args.text_model,
    )
