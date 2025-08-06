import json
from pathlib import Path
import click
from loguru import logger
from typing import Optional, List, Dict, Any, cast, Literal
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.nn import Node2Vec, SAGEConv
import time
import numpy as np

from transformers import GPT2Model, GPT2TokenizerFast


def get_best_device() -> torch.device:
    """Get best available device with consistent priority: CUDA → MPS → CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


from src.corp_speech_risk_dataset.encoding.tokenizer import SentencePieceTokenizer
from src.corp_speech_risk_dataset.encoding.parser import to_dependency_graph
from src.corp_speech_risk_dataset.encoding.wl_features import wl_vector
from src.corp_speech_risk_dataset.encoding.stembedder import get_sentence_embedder
from src.corp_speech_risk_dataset.encoding.graphembedder import CrossModalFusion
from src.corp_speech_risk_dataset.encoding.graphembedder import (
    compute_graph_embedding,
    get_node2vec_embedder,
    get_graphsage_embedder,
    CrossModalFusion,
)
from src.corp_speech_risk_dataset.encoding.legal_bert_embedder import (
    get_legal_bert_embedder,
    LegalBertEmbedder,
)

# Will look under data/models/en.wiki.bpe.vs32000.model by default
tokenizer = SentencePieceTokenizer()


def encode_file(
    in_path: Path,
    input_root: Path,  # <-- changed from extracted_root
    tokenized_root: Path,
    st_model_name: Optional[str] = None,
    text_model: str = "none",
    batch_size: int = 64,
    graph_embed: str = "wl",
    stage: int = 4,
    overwrite: bool = False,
    text_embedder: str = "st",  # New: Legal-BERT extension
    fuse_graph: bool = True,  # New: Fusion control
    embed_batch_size: int = 32,  # New: Embedding batch size
    use_amp: bool = True,  # New: AMP for efficiency
):
    file_start = time.time()  # Start timing for file encoding
    """Helper: encode one file, writing into mirrored structure."""
    rel_path = in_path.relative_to(input_root)  # <-- use input_root
    # bump stage in filename
    old_name = rel_path.name
    new_name = old_name.replace(f"stage{stage}", f"stage{stage+1}")
    out_path = tokenized_root / rel_path.parent / new_name
    # Resume logic: truncate file to only fully completed rows and resume
    if out_path.exists():
        # read all existing lines
        with open(out_path, "r", encoding="utf-8") as f_done:
            existing = []
            for line in f_done:
                try:
                    js = json.loads(line)
                except json.JSONDecodeError:
                    break
                # stage-specific completeness checks
                if text_model.lower() == "st" and "st_emb" not in js:
                    break
                if text_model.lower() == "gpt2" and "gpt2_emb" not in js:
                    break
                if "gph_emb" not in js:
                    break
                if graph_embed == "cross" and "fused_emb" not in js:
                    break
                existing.append(js)
        skip = len(existing)
        # rewrite only the complete rows back to file
        with open(out_path, "w", encoding="utf-8") as f_trunc:
            for js in existing:
                f_trunc.write(json.dumps(js, ensure_ascii=False) + "\n")
        mode = "ab"
        print(
            f"[INFO] Resuming pipeline: preserved {skip} complete rows, will reprocess from row {skip+1}"
        )
    else:
        skip = 0
        mode = "wb"
    print(f"[INFO] Output filename updated to stage{stage+1}: {new_name}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"→ Encoding {in_path} → {out_path}")

    # -------- Text Embedders (ST, GPT-2, Legal-BERT) ------------------ #
    st_model = get_sentence_embedder(st_model_name) if st_model_name else None
    legal_bert_model = None

    # Initialize Legal-BERT if requested
    if text_embedder == "legal-bert":
        legal_bert_model = get_legal_bert_embedder(use_amp=use_amp)
        logger.info(
            f"Legal-BERT model loaded: {legal_bert_model.get_sentence_embedding_dimension()}D embeddings"
        )

    # -------- GPT-2 mean-pool (optional) ------------------------------- #
    if text_model.lower() == "gpt2":
        # Use consistent device priority: CUDA → MPS → CPU
        device = get_best_device()
        gpt_tok = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
        gpt_tok.pad_token = gpt_tok.eos_token  # <-- fix: set pad_token
        gpt_mod = GPT2Model.from_pretrained("gpt2", local_files_only=True).to(device)
        gpt_mod.eval()
    else:
        # still need a dummy device for signature
        device = get_best_device()
        gpt_mod = gpt_tok = None

    # Graph embedders & fusion
    n2v_model = get_node2vec_embedder() if graph_embed == "node2vec" else None
    # Fixed dimensions for GraphSAGE
    graph_hidden = 128
    num_layers = 2

    # Create GraphSAGE model with fixed input dimensions (16D node features)
    sage_model = (
        get_graphsage_embedder(
            in_channels=16, hidden_channels=graph_hidden, num_layers=num_layers
        )
        if graph_embed in ["graphsage", "cross"]
        else None
    )

    # now pass the real dims into fusion
    if graph_embed == "cross":
        if st_model is None:
            raise ValueError("CrossModal fusion requires sentence transformer model")
        txt_dim = st_model.get_sentence_embedding_dimension()
        if txt_dim is None:
            raise ValueError(
                "Could not determine sentence transformer embedding dimension"
            )
        # graph_hidden must match the hidden size used by GraphSAGE
        fuse_model = CrossModalFusion(text_dim=txt_dim, graph_dim=graph_hidden)
    else:
        fuse_model = None

    # Pass 1 – count rows for manual progress updates
    total = sum(1 for _ in in_path.open())
    # Confirm stages & models
    print(f"[INFO] Stage: Preparing to encode {total} entries")

    # Enhanced model info display
    if text_embedder == "legal-bert" and legal_bert_model:
        print(
            f"[INFO] Using Text Embedder: Legal-BERT ({legal_bert_model.get_sentence_embedding_dimension()}D)"
        )
        print(
            f"[INFO] Legal-BERT Config: AMP={'ON' if use_amp else 'OFF'}, batch_size={embed_batch_size}"
        )
    elif text_model == "st" and st_model:
        print(f"[INFO] Using Text Model: {text_model} ({st_model.__class__.__name__})")
    else:
        print(f"[INFO] Using Text Model: {text_model}")

    print(f"[INFO] Fusion enabled: {fuse_graph}")
    print(
        f"[INFO] Using Graph Embed: {graph_embed} "
        f"(Node2Vec={'yes' if n2v_model else 'no'}, "
        f"GraphSAGE={'yes' if sage_model else 'no'}, "
        f"Fusion={'yes' if fuse_model else 'no'})"
    )

    with open(out_path, "wb") as fout, open(in_path) as fin:
        buf_json: List[Dict[str, Any]] = []
        buf_txt: List[str] = []

        for i, line in enumerate(fin, start=1):
            # on first entry, announce loop start
            if i == 1:
                print(f"[INFO] Stage: Starting tokenization + embedding loop")
            row = json.loads(line)
            buf_json.append(row)
            buf_txt.append(row["text"])

            # Flush batch (any embedding requested) ----------------- #
            if (text_model == "none" and graph_embed == "wl") or len(
                buf_txt
            ) < batch_size:
                continue

            for enriched in _flush(
                buf_json,
                buf_txt,
                st_model,
                gpt_mod,
                gpt_tok,
                device,
                text_model,
                graph_embed,
                n2v_model,
                sage_model,
                fuse_model,
                batch_size,
                text_embedder=text_embedder,
                legal_bert_model=legal_bert_model,
                fuse_graph=fuse_graph,
                embed_batch_size=embed_batch_size,
                use_amp=use_amp,
            ):
                fout.write((json.dumps(enriched, ensure_ascii=False) + "\n").encode())
            buf_json.clear()
            buf_txt.clear()

        # final tiny batch
        if buf_json:
            for enriched in _flush(
                buf_json,
                buf_txt,
                st_model,
                gpt_mod,
                gpt_tok,
                device,
                text_model,
                graph_embed,
                n2v_model,
                sage_model,
                fuse_model,
                batch_size,
                text_embedder=text_embedder,
                legal_bert_model=legal_bert_model,
                fuse_graph=fuse_graph,
                embed_batch_size=embed_batch_size,
                use_amp=use_amp,
            ):
                fout.write((json.dumps(enriched, ensure_ascii=False) + "\n").encode())
            # capture fallback info too
            _, used_fallback, fallback_chars = tokenizer.encode_with_flag(
                buf_json[-1]["text"]
            )
            if used_fallback:
                logger.warning(f"Byte-fallback for chars: {fallback_chars}")
    # ← actually runs now, after all writing is done
    total_time = time.time() - file_start
    print(f"[INFO] Finished full encode for {in_path.name} in {total_time:.2f}s")
    logger.success(f"✔ Written tokenized output to {out_path}")
    return out_path


def _flush(
    rows: List[Dict[str, Any]],
    texts: List[str],
    st_model,
    gpt_mod,
    gpt_tok,
    device,
    text_model: str,
    graph_embed: str,
    n2v_model,
    sage_model,
    fuse_model,
    batch_size: int,
    text_embedder: str = "st",  # New: Legal-BERT extension
    legal_bert_model: Optional[LegalBertEmbedder] = None,  # New
    fuse_graph: bool = True,  # New: Fusion control
    embed_batch_size: int = 32,  # New: Embedding batch size
    use_amp: bool = True,  # New: AMP for efficiency
) -> List[Dict[str, Any]]:
    """Embed a buffered batch & yield enriched JSON rows."""
    # --- Stage: Text Embedding ---
    t0 = time.time()

    # Legal-BERT embeddings (new option)
    if text_embedder == "legal-bert" and legal_bert_model is not None:
        txt_embs = legal_bert_model.encode(
            texts,
            batch_size=embed_batch_size,
            convert_to_numpy=True,
            show_progress=False,
        )
    # Legacy text model handling (backward compatibility)
    elif text_model.lower() == "gpt2":
        enc = gpt_tok(texts, return_tensors="pt", padding=True, truncation=True).to(
            device
        )
        # Use AMP if available and requested
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                with torch.no_grad():
                    hidden = gpt_mod(**enc).last_hidden_state
        else:
            with torch.no_grad():
                hidden = gpt_mod(**enc).last_hidden_state
        txt_embs = hidden.mean(dim=1).cpu().numpy()
    elif text_model.lower() == "st" and st_model is not None:
        txt_embs = st_model.encode(
            texts, convert_to_numpy=True, batch_size=embed_batch_size
        )
    # MiniLM via text_embedder (preferred new path)
    elif text_embedder == "st" and st_model is not None:
        txt_embs = st_model.encode(
            texts, convert_to_numpy=True, batch_size=embed_batch_size
        )
    else:
        txt_embs = [None] * len(rows)

    dt_text = time.time() - t0
    # print(f"[STAGE] Text Embedding ({text_model}) for batch of {len(texts)}: {dt_text:.2f}s")

    # --- Stage: Graph Embedding ({graph_embed}) ---
    t1 = time.time()
    # if we're doing cross, compute GraphSAGE under the hood
    base_method = "graphsage" if graph_embed == "cross" else graph_embed

    # Safely cast to literal types
    if base_method in ["wl", "node2vec", "graphsage"]:
        method_literal = cast(Literal["wl", "node2vec", "graphsage"], base_method)
        gph_list = [
            compute_graph_embedding(
                t, method_literal, n2v_model, sage_model, fuse_model
            )
            for t in texts
        ]
        gph_embs = torch.stack(gph_list) if gph_list else None
    else:
        gph_embs = None
    dt_graph = time.time() - t1
    # print(f"[STAGE] Graph Embedding ({base_method}) for batch of {len(texts)}: {dt_graph:.2f}s")

    # --- Stage: Fusion ({graph_embed}) ---
    # Enhanced fusion logic: respect fuse_graph flag and handle Legal-BERT
    if (
        graph_embed == "cross"
        and fuse_graph
        and not isinstance(txt_embs, list)
        and gph_embs is not None
    ):
        t2 = time.time()
        # Use AMP for fusion if available and requested
        if use_amp and torch.cuda.is_available():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                fused_embs = fuse_model(torch.tensor(txt_embs), gph_embs).cpu().numpy()
        else:
            fused_embs = fuse_model(torch.tensor(txt_embs), gph_embs).cpu().numpy()
        dt_fuse = time.time() - t2
        # print(f"[STAGE] CrossModalFusion for batch of {len(texts)}: {dt_fuse:.2f}s")
    else:
        fused_embs = None

    # ensure iterable
    if txt_embs is None:
        txt_embs = [None] * len(rows)

    enriched_rows: List[Dict[str, Any]] = []
    for i, (row, txt_vec) in enumerate(zip(rows, txt_embs)):
        wl_vec = wl_vector(row["text"])
        sp_ids, used_fallback, fallback_chars = tokenizer.encode_with_flag(row["text"])

        # Handle different text embedding types (enhanced for Legal-BERT)
        gpt2_emb = None
        st_emb = None
        legal_bert_emb = None

        if text_embedder == "legal-bert" and txt_vec is not None:
            legal_bert_emb = txt_vec.tolist()
        elif text_model.lower() == "gpt2" and txt_vec is not None:
            gpt2_emb = txt_vec.tolist()
        elif (
            text_model.lower() == "st" or text_embedder == "st"
        ) and txt_vec is not None:
            st_emb = txt_vec.tolist()

        enriched_rows.append(
            {
                **row,
                "sp_ids": sp_ids,
                "byte_fallback": used_fallback,
                "fallback_chars": fallback_chars,
                "deps": [
                    (h, t, l)
                    for h, t, l in to_dependency_graph(row["text"]).edges.data("dep")
                ],
                "wl_indices": wl_vec.indices.tolist(),
                "wl_counts": wl_vec.data.tolist(),
                "st_model": (
                    st_model._first_module().__class__.__name__ if st_model else None
                ),
                "text_embedder": text_embedder,  # Track which embedder was used
                "gpt2_emb": gpt2_emb,
                "st_emb": st_emb,
                "legal_bert_emb": legal_bert_emb,  # New: Legal-BERT embeddings
                "gph_method": graph_embed,
                "gph_emb": gph_embs[i].tolist() if gph_embs is not None else None,
                **(
                    {"fused_emb": fused_embs[i].tolist()}
                    if graph_embed == "cross" and fused_embs is not None and fuse_graph
                    else {}
                ),
            }
        )
    return enriched_rows


@click.group(chain=True, invoke_without_command=True)
@click.argument("in_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--out-root",
    "out_root",
    required=True,
    type=click.Path(file_okay=False, dir_okay=True, path_type=Path),
    help="Directory where outputs will be mirrored (for recursive mode) or single file if not).",
)
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    help="Recursively process all *_stage{stage}.jsonl under in_path.",
)
@click.option(
    "--stage",
    type=int,
    default=5,
    show_default=True,
    help="Current stage number; used to match and bump filenames.",
)
@click.option(
    "--text-embedder",
    type=click.Choice(["st", "gpt2", "legal-bert"]),
    default="st",
    show_default=True,
    help="Text embedder: st (SentenceTransformer), gpt2 (GPT-2), or legal-bert (Legal-BERT for legal documents).",
)
@click.option(
    "--fuse-graph/--no-fuse-graph",
    default=True,
    show_default=True,
    help="Enable/disable fusion of text embeddings with graph embeddings.",
)
@click.option(
    "--embed-batch-size",
    type=int,
    default=32,
    show_default=True,
    help="Batch size for embedding generation (32 optimized for T4 GPU).",
)
@click.option(
    "--use-amp/--no-amp",
    default=True,
    show_default=True,
    help="Enable/disable Automatic Mixed Precision for GPU efficiency.",
)
@click.option(
    "--fusion-epochs",
    type=int,
    default=15,
    show_default=True,
    help="Number of CrossModal fusion training epochs.",
)
@click.option(
    "--fusion-batch-size",
    type=int,
    default=256,
    show_default=True,
    help="Batch size for CrossModal fusion training.",
)
@click.option(
    "--fusion-temperature",
    type=float,
    default=0.07,
    show_default=True,
    help="Temperature for InfoNCE loss in CrossModal fusion.",
)
@click.option(
    "--fusion-patience",
    type=int,
    default=3,
    show_default=True,
    help="Early stopping patience for CrossModal fusion.",
)
@click.option(
    "--fusion-dropout",
    type=float,
    default=0.1,
    show_default=True,
    help="Dropout rate for CrossModal fusion stability.",
)
@click.option(
    "--fusion-heads",
    type=int,
    default=4,
    show_default=True,
    help="Number of attention heads for CrossModal fusion.",
)
@click.option(
    "--adaptive-temperature",
    is_flag=True,
    default=True,
    show_default=True,
    help="Use adaptive temperature in InfoNCE loss for legal text.",
)
@click.option(
    "--hard-negative-weight",
    type=float,
    default=1.2,
    show_default=True,
    help="Weight for hard negative mining in contrastive loss.",
)
@click.option(
    "--hidden-dim",
    type=int,
    default=256,
    show_default=True,
    help="Hidden dimensions for GraphSAGE.",
)
@click.option(
    "--epochs",
    type=int,
    default=40,
    show_default=True,
    help="Number of training epochs for GraphSAGE.",
)
@click.pass_context
def cli(
    ctx,
    in_path,
    out_root,
    recursive,
    stage,
    text_embedder,
    fuse_graph,
    embed_batch_size,
    use_amp,
    fusion_epochs,
    fusion_batch_size,
    fusion_temperature,
    fusion_patience,
    fusion_dropout,
    fusion_heads,
    adaptive_temperature,
    hard_negative_weight,
    hidden_dim,
    epochs,
):
    """
    Chainable pipeline: tokenize → embed → graph → fuse.
    If no subcommands given, runs full encode (all).
    """
    # prepare file list
    in_path = Path(in_path)
    if recursive and in_path.is_dir():
        pattern = f"*_stage{stage}.jsonl"
        files = list(in_path.rglob(pattern))
        input_root = in_path  # <-- set input_root to the directory
    elif in_path.is_file():
        files = [in_path]
        input_root = in_path.parent  # <-- set input_root to parent of file
    else:
        raise click.ClickException(
            "in_path must be a file or a directory with --recursive"
        )
    ctx.obj = {
        "files": files,
        "out_root": Path(out_root),
        "processors": [],
        "stage": stage,
        "input_root": input_root,  # <-- store input_root
        "text_embedder": text_embedder,  # New: Legal-BERT extension
        "fuse_graph": fuse_graph,  # New: Fusion control
        "embed_batch_size": embed_batch_size,  # New: Embedding batch size
        "use_amp": use_amp,  # New: AMP for efficiency
        "fusion_epochs": fusion_epochs,  # New: Fusion training epochs
        "fusion_batch_size": fusion_batch_size,  # New: Fusion batch size
        "fusion_temperature": fusion_temperature,  # New: Fusion temperature
        "fusion_patience": fusion_patience,  # New: Fusion patience
        "fusion_dropout": fusion_dropout,  # New: Fusion dropout
        "fusion_heads": fusion_heads,  # New: Fusion attention heads
        "adaptive_temperature": adaptive_temperature,  # New: Adaptive temperature
        "hard_negative_weight": hard_negative_weight,  # New: Hard negative weight
        "hidden_dim": hidden_dim,  # New: GraphSAGE hidden dimensions
        "epochs": epochs,  # New: GraphSAGE training epochs
    }
    # schedule full encode if no subcommands
    if ctx.invoked_subcommand is None:
        ctx.invoke(all)


@cli.command()
@click.option("--text-model", type=click.Choice(["none", "gpt2", "st"]), default="none")
@click.option("--st-model", default=None)
@click.option(
    "--graph-embed", type=click.Choice(["wl", "node2vec", "graphsage"]), default="wl"
)
@click.pass_context
def all(ctx, text_model, st_model, graph_embed):
    # fallback: run full encode_file for each file in ctx.obj['files']
    for in_file in ctx.obj["files"]:
        out_dir = ctx.obj["out_root"]
        out_dir.mkdir(parents=True, exist_ok=True)
        base = Path(in_file).stem
        stage = ctx.obj["stage"]
        input_root = ctx.obj["input_root"]  # <-- get input_root
        # derive output file name by bumping stage in filename
        out_file = out_dir / f"{base.replace(f'stage{stage}', f'stage{stage+1}')}.jsonl"
        encode_file(
            Path(in_file),
            input_root=input_root,  # <-- pass input_root
            tokenized_root=out_dir,
            st_model_name=st_model,
            text_model=text_model,
            graph_embed=graph_embed,
            stage=stage,
            overwrite=False,
            text_embedder=ctx.obj["text_embedder"],  # New: Legal-BERT extension
            fuse_graph=ctx.obj["fuse_graph"],  # New: Fusion control
            embed_batch_size=ctx.obj["embed_batch_size"],  # New: Embedding batch size
            use_amp=ctx.obj["use_amp"],  # New: AMP for efficiency
        )


@cli.command("tokenize")
@click.option("--text-model", type=click.Choice(["none", "gpt2", "st"]), default="none")
@click.option("--st-model", default=None)
@click.pass_context
def cmd_tokenize(ctx, text_model, st_model):
    """Tokenize raw JSONL (add sp_ids, byte_fallback)."""

    def processor(lines):
        import time

        start_time = time.time()
        print(f"[TOKENIZATION] Using tokenizer: {text_model.upper()}")
        from src.corp_speech_risk_dataset.encoding.tokenizer import (
            SentencePieceTokenizer,
        )
        from transformers import GPT2TokenizerFast

        # initialize only the requested tokenizer
        tok_sp = None
        tok_gpt = None
        if text_model == "gpt2":
            print(f"[TOKENIZATION] Loading GPT-2 tokenizer (50,257 vocab)")
            tok_gpt = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
            tok_gpt.pad_token = tok_gpt.eos_token  # <-- fix: set pad_token
        else:
            print(f"[TOKENIZATION] Loading SentencePiece tokenizer (32,000 vocab)")
            tok_sp = SentencePieceTokenizer()
        try:
            for line in lines:
                js = json.loads(line)
                # clear previous tokenization fields
                js.pop("sp_ids", None)
                js.pop("byte_fallback", None)
                js.pop("fallback_chars", None)
                if text_model == "gpt2" and tok_gpt is not None:
                    # GPT-2 fast-tokenization
                    enc = tok_gpt(
                        js["text"], return_tensors="pt", padding=True, truncation=True
                    )
                    js["sp_ids"] = enc["input_ids"][0].tolist()
                    js["byte_fallback"] = False
                    js["fallback_chars"] = []
                else:
                    # SentencePiece fallback for 'none' or 'st'
                    if tok_sp is not None:
                        ids, uf, chars = tok_sp.encode_with_flag(js["text"])
                        js["sp_ids"], js["byte_fallback"], js["fallback_chars"] = (
                            ids,
                            uf,
                            chars,
                        )
                yield json.dumps(js, ensure_ascii=False)
        finally:
            dt = time.time() - start_time
            print(f"[TOKENIZATION] Completed in {dt:.2f}s")

    ctx.obj["processors"].append(processor)


@cli.command("embed")
@click.option("--st-model", default=None, help="SentenceTransformer model name")
@click.pass_context
def cmd_embed(ctx, st_model):
    """Embed text via SentenceTransformer, GPT2, or Legal-BERT."""

    # LOAD MODEL ONCE OUTSIDE THE PROCESSOR (FIXED!)
    text_embedder = ctx.obj["text_embedder"]
    embed_batch_size = ctx.obj["embed_batch_size"]
    use_amp = ctx.obj["use_amp"]

    # Initialize the appropriate model ONCE
    model = None
    embedding_dim = None
    emb_field = None

    if text_embedder == "legal-bert":
        print(f"[TEXT EMBEDDING] Loading Legal-BERT model (ONCE)")
        from src.corp_speech_risk_dataset.encoding.legal_bert_embedder import (
            get_legal_bert_embedder,
        )

        model = get_legal_bert_embedder(use_amp=use_amp)
        embedding_dim = model.get_sentence_embedding_dimension()
        emb_field = "legal_bert_emb"
    elif text_embedder == "st" and st_model:
        print(f"[TEXT EMBEDDING] Loading SentenceTransformer model (ONCE): {st_model}")
        from src.corp_speech_risk_dataset.encoding.stembedder import (
            get_sentence_embedder,
        )

        model = get_sentence_embedder(st_model)
        embedding_dim = model.get_sentence_embedding_dimension()
        emb_field = "st_emb"
    else:
        raise ValueError(
            f"Embed subcommand requires either --st-model with --text-embedder st, or --text-embedder legal-bert"
        )

    print(f"[TEXT EMBEDDING] Model output dimension: {embedding_dim}D")
    print(f"[TEXT EMBEDDING] Batch size: {embed_batch_size}")

    def processor(lines):
        import time

        start_time = time.time()

        batch, metas = [], []
        try:
            for line in lines:
                js = json.loads(line)
                # clear previous text embeddings
                js.pop("st_emb", None)
                js.pop("gpt2_emb", None)
                js.pop("legal_bert_emb", None)
                batch.append(js["text"])
                metas.append(js)
                if len(batch) >= embed_batch_size:
                    embs = model.encode(batch, convert_to_numpy=True)
                    for mj, emb in zip(metas, embs):
                        mj[emb_field] = emb.tolist()
                        mj["text_embedder"] = (
                            text_embedder  # Track which embedder was used
                        )
                        yield json.dumps(mj, ensure_ascii=False)
                    batch, metas = [], []
            # flush leftover
            if batch:
                embs = model.encode(batch, convert_to_numpy=True)
                for mj, emb in zip(metas, embs):
                    mj[emb_field] = emb.tolist()
                    mj["text_embedder"] = text_embedder  # Track which embedder was used
                    yield json.dumps(mj, ensure_ascii=False)
        finally:
            dt = time.time() - start_time
            print(f"[TEXT EMBEDDING] Completed in {dt:.2f}s")

    ctx.obj["processors"].append(processor)


@cli.command("graph")
@click.option(
    "--graph-embed",
    type=click.Choice(["wl", "node2vec", "graphsage", "cross"]),
    default="wl",
)
@click.option(
    "--eval-graph",
    is_flag=True,
    help="Run comprehensive evaluation on GraphSAGE model (reconstruction + classification + clustering)",
)
@click.option(
    "--epochs",
    type=int,
    default=40,  # Increased from 15 to 40 for 10^-4 target
    help="Number of training epochs for GraphSAGE (default: 40)",
)
@click.option(
    "--max-samples",
    type=int,
    default=15000,  # Increased for better legal text coverage
    help="Maximum training graphs to sample for GraphSAGE (default: 15,000)",
)
@click.option(
    "--max-files",
    type=int,
    default=1000,  # Increased for better diversity
    help="Maximum files to sample from for GraphSAGE training (default: 1,000)",
)
@click.option(
    "--batch-size",
    type=int,
    default=512,
    help="Batch size for GraphSAGE training (default: 512)",
)
@click.option(
    "--hidden-dim",
    type=int,
    default=256,  # Increased default to 256 for legal text
    help="Hidden dimensions for GraphSAGE (default: 256, optimized for legal text)",
)
@click.option(
    "--loss-type",
    type=click.Choice(["mse", "cosine", "hybrid"]),
    default="hybrid",
    help="Reconstruction loss type: MSE, scaled-cosine, or hybrid (default: hybrid)",
)
@click.option(
    "--dgi-weight",
    type=float,
    default=0.1,
    help="Weight for DGI contrastive loss (default: 0.1)",
)
@click.option(
    "--num-negative",
    type=int,
    default=30,
    help="Number of negative samples for DGI loss (default: 30)",
)
@click.option(
    "--fusion-epochs",
    type=int,
    default=10,
    help="Number of CrossModal fusion training epochs (default: 10)",
)
@click.option(
    "--fusion-samples",
    type=int,
    default=5000,
    help="Number of text-graph pairs for CrossModal fusion training (default: 5000)",
)
@click.option(
    "--fusion-batch-size",
    type=int,
    default=256,
    help="Batch size for CrossModal fusion training (default: 256)",
)
@click.option(
    "--fusion-temperature",
    type=float,
    default=0.07,
    help="Temperature for InfoNCE loss in CrossModal fusion (default: 0.07)",
)
@click.option(
    "--fusion-patience",
    type=int,
    default=2,
    help="Early stopping patience for CrossModal fusion (default: 2)",
)
@click.option(
    "--use-compile",
    is_flag=True,
    help="Use torch.compile for CrossModal fusion speedup",
)
@click.option(
    "--use-amp",
    is_flag=True,
    help="Use Automatic Mixed Precision (AMP) for speedup",
)
@click.option(
    "--fusion-dropout",
    type=float,
    default=0.1,
    help="Dropout rate for CrossModal fusion stability (default: 0.1)",
)
@click.option(
    "--fusion-heads",
    type=int,
    default=4,
    help="Number of attention heads for CrossModal fusion (default: 4)",
)
@click.option(
    "--adaptive-temperature",
    is_flag=True,
    default=True,
    help="Use adaptive temperature in InfoNCE loss for legal text",
)
@click.option(
    "--hard-negative-weight",
    type=float,
    default=1.2,
    help="Weight for hard negative mining in contrastive loss (default: 1.2)",
)
@click.pass_context
def cmd_graph(
    ctx,
    graph_embed,
    eval_graph,
    epochs,
    max_samples,
    max_files,
    batch_size,
    hidden_dim,
    loss_type,
    dgi_weight,
    num_negative,
    fusion_epochs,
    fusion_samples,
    fusion_batch_size,
    fusion_temperature,
    fusion_patience,
    use_compile,
    use_amp,
    fusion_dropout,
    fusion_heads,
    adaptive_temperature,
    hard_negative_weight,
):
    """Embed graphs via WL, Node2Vec, GraphSAGE, or Cross-Modal Fusion targeting 10^-4 MSE."""
    # Print setup information once
    print(f"\n[STEP] GRAPH EMBEDDING")
    print(f"Method: {graph_embed.upper()}")

    if graph_embed in ["graphsage", "cross"]:
        print(f"GraphSAGE Hyperparameters (10^-4 MSE Target):")
        print(f"  ├─ Epochs: {epochs} (extended for convergence)")
        print(f"  ├─ Max training samples: {max_samples:,}")
        print(f"  ├─ Max files to sample: {max_files}")
        print(f"  ├─ Batch size: {batch_size}")
        print(f"  ├─ Hidden dimensions: {hidden_dim} (3 layers)")
        print(f"  ├─ Neighbor sampling: [50, 25] (increased fan-out for legal text)")
        print(f"  ├─ Loss type: {loss_type.upper()}")
        print(f"  ├─ DGI weight: {dgi_weight} (contrastive)")
        print(f"  ├─ Negative samples: {num_negative}")
        print(f"  ├─ Dropout: 0.3 (legal domain regularization)")
        print(f"  ├─ Validation split: 20%")
        print(f"  ├─ Early stopping: 8 epochs patience")
        print(f"  └─ Evaluation mode: {'ON' if eval_graph else 'OFF'}")
        if graph_embed == "cross":
            print(f"CrossModal Fusion (InfoNCE Optimized):")
            print(f"  ├─ Training samples: 2,000")
            print(f"  ├─ Batch size: 256")
            print(f"  ├─ Temperature: 0.07")
            print(f"  ├─ Epochs: 12")
            print(f"  └─ Early stopping: 3 epochs patience")

    # imports for dynamic graph processing
    from src.corp_speech_risk_dataset.encoding.graphembedder import (
        compute_graph_embedding,
        get_node2vec_embedder,
        get_graphsage_embedder,
        train_graphsage_model,
        _nx_to_pyg,
    )
    from src.corp_speech_risk_dataset.encoding.parser import to_dependency_graph

    node2vec_model = get_node2vec_embedder() if graph_embed == "node2vec" else None
    # Create GraphSAGE model with configurable dimensions
    sage_model = (
        get_graphsage_embedder(in_channels=16, hidden_channels=hidden_dim, num_layers=2)
        if graph_embed in ["graphsage", "cross"]
        else None
    )

    # Train GraphSAGE model if needed
    training_metrics = {}
    fusion_model = None  # Initialize fusion model
    if graph_embed == "graphsage" and sage_model is not None:
        print(f"Preparing to train GraphSAGE model...")

        # Collect sample graphs for training - improved sampling strategy
        training_graphs = []
        training_labels = []  # For evaluation if available
        training_texts = []  # For CrossModalFusion training
        sample_count = 0
        min_nodes = 3  # Lower threshold - even simple structures are useful
        files_to_sample = min(max_files, len(ctx.obj["files"]))

        print(
            f"Sampling from {files_to_sample} files to collect up to {max_samples:,} training graphs..."
        )
        print(f"Minimum nodes per graph: {min_nodes}")

        for i, file_path in enumerate(ctx.obj["files"][:files_to_sample]):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    file_samples = 0
                    for line in f:
                        if sample_count >= max_samples:
                            break
                        try:
                            js = json.loads(line)
                            dep_graph = to_dependency_graph(js["text"])
                            node_count = len(list(dep_graph.nodes()))
                            if node_count >= min_nodes:
                                pyg_data = _nx_to_pyg(dep_graph)
                                training_graphs.append(pyg_data)
                                training_texts.append(
                                    js["text"]
                                )  # Store text for fusion training

                                # Extract labels for evaluation if available
                                label = 0  # Default label
                                if "label" in js:
                                    label = int(js["label"])
                                elif "false_advertising" in js:
                                    label = int(js["false_advertising"])
                                elif "target" in js:
                                    label = int(js["target"])
                                training_labels.append(label)

                                sample_count += 1
                                file_samples += 1
                        except Exception as e:
                            continue

                # Progress update every 25 files for better visibility
                if (i + 1) % 25 == 0:
                    print(
                        f"  Processed {i+1}/{files_to_sample} files, collected {sample_count:,} graphs"
                    )

                if sample_count >= max_samples:
                    break
            except Exception as e:
                continue

        if training_graphs:
            print(
                f"\n[GRAPHSAGE TRAINING] Collected {len(training_graphs):,} training graphs"
            )
            print(
                f"[GRAPHSAGE TRAINING] Label distribution: {np.bincount(training_labels) if training_labels else 'No labels'}"
            )

            # Use labels for evaluation if available, otherwise None
            eval_labels = (
                training_labels
                if eval_graph and len(set(training_labels)) > 1
                else None
            )

            sage_model, training_metrics = train_graphsage_model(
                sage_model,
                training_graphs,
                epochs=epochs,
                neighbor_samples=[
                    50,
                    25,
                ],  # Increased fan-out for legal text (was [25, 10])
                val_split=0.2,
                batch_size=batch_size,
                eval_mode=eval_graph,
                val_labels=eval_labels,
                loss_type=loss_type,
                dgi_weight=dgi_weight,
                num_negative=num_negative,
            )

            print(f"[GRAPHSAGE TRAINING] Training completed!")

            # Train CrossModalFusion if needed
            if graph_embed == "cross":
                from src.corp_speech_risk_dataset.encoding.stembedder import (
                    get_sentence_embedder,
                )
                from src.corp_speech_risk_dataset.encoding.graphembedder import (
                    train_crossmodal_fusion,
                )

                print(f"\n[CROSSMODAL FUSION] Starting fusion training...")

                # Get text embeddings for training - respect text_embedder setting
                text_embedder = ctx.obj.get("text_embedder", "st")
                embed_batch_size = ctx.obj.get("embed_batch_size", 64)
                use_amp = ctx.obj.get("use_amp", True)

                if text_embedder == "legal-bert":
                    from src.corp_speech_risk_dataset.encoding.legal_bert_embedder import (
                        get_legal_bert_embedder,
                    )

                    model = get_legal_bert_embedder(use_amp=use_amp)
                    text_embs_np = model.encode(
                        training_texts[:fusion_samples],
                        convert_to_numpy=True,
                        batch_size=embed_batch_size,
                    )
                    text_embs = torch.tensor(text_embs_np)
                else:
                    st_model = get_sentence_embedder("all-MiniLM-L6-v2")
                    text_embs = st_model.encode(
                        training_texts[:fusion_samples],
                        convert_to_tensor=True,
                        batch_size=embed_batch_size,
                    )

                # Get graph embeddings for same texts
                graph_embs = []
                sage_model.eval()
                with torch.no_grad():
                    for i, text in enumerate(
                        training_texts[:fusion_samples]
                    ):  # Match text embeddings count
                        graph_emb = compute_graph_embedding(
                            text, "graphsage", None, sage_model
                        )
                        graph_embs.append(graph_emb)

                # Create and train fusion model - get dimension from appropriate model
                if text_embedder == "legal-bert":
                    txt_dim = model.get_sentence_embedding_dimension()
                else:
                    txt_dim = st_model.get_sentence_embedding_dimension()
                if txt_dim is None:
                    raise ValueError("Could not determine text embedding dimension")
                fusion_model = CrossModalFusion(
                    text_dim=txt_dim,
                    graph_dim=hidden_dim,
                    dropout=fusion_dropout,
                    num_heads=fusion_heads,
                )

                # Apply torch.compile optimization if requested
                if use_compile:
                    try:
                        compiled_model = torch.compile(fusion_model)
                        print(f"[CROSSMODAL FUSION] torch.compile enabled for speedup")
                        # Use compiled model for training
                        fusion_model = compiled_model  # type: ignore
                    except Exception as e:
                        print(f"[CROSSMODAL FUSION] torch.compile failed: {e}")

                fusion_model = train_crossmodal_fusion(
                    fusion_model,
                    [text_embs[i] for i in range(len(text_embs))],
                    graph_embs,
                    epochs=fusion_epochs,
                    batch_size=fusion_batch_size,
                    temperature=fusion_temperature,
                    patience=fusion_patience,
                    use_amp=use_amp,
                    adaptive_temperature=adaptive_temperature,
                    hard_negative_weight=hard_negative_weight,
                )

                print(f"[CROSSMODAL FUSION] Fusion training completed!")

            # Print evaluation results if available
            if eval_graph and training_metrics:
                print(f"\n[GRAPHSAGE EVALUATION RESULTS]")
                print(
                    f"├─ Final train loss: {training_metrics.get('final_train_loss', 0):.4e}"
                )
                print(
                    f"├─ Final val loss: {training_metrics.get('final_val_loss', 0):.4e}"
                )
                print(
                    f"├─ Best val loss: {training_metrics.get('best_val_loss', 0):.4e}"
                )
                print(f"├─ Epochs trained: {training_metrics.get('epochs_trained', 0)}")

                if "classification_f1" in training_metrics:
                    print(
                        f"├─ Classification F1: {training_metrics['classification_f1']:.3f}"
                    )
                    print(
                        f"├─ Classification AUC: {training_metrics['classification_auc']:.3f}"
                    )

                if "silhouette_score" in training_metrics:
                    print(
                        f"└─ Silhouette Score: {training_metrics['silhouette_score']:.3f}"
                    )

                # Recommendations based on results
                val_loss = training_metrics.get("final_val_loss", 0)
                train_loss = training_metrics.get("final_train_loss", 0)
                f1_score = training_metrics.get("classification_f1", 0)

                print(f"\n[GRAPHSAGE RECOMMENDATIONS]")
                if val_loss > train_loss * 2:
                    print(
                        " High validation loss suggests overfitting - consider reducing capacity or adding dropout"
                    )
                if f1_score < 0.6:
                    print(
                        "  Low F1 score - consider enriching node features or increasing neighbor sampling"
                    )
                if f1_score > 0.8:
                    print(
                        " Good F1 score - embeddings are learning task-relevant patterns"
                    )
                if val_loss < 1e-3:
                    print(
                        " Low validation loss - model is learning graph structure well"
                    )

                    # Handle cross-modal fusion setup for inference
    if graph_embed == "cross" and fusion_model is None:
        # If we didn't train fusion above, create untrained one (fallback) - respect text_embedder
        text_embedder = ctx.obj.get("text_embedder", "st")
        use_amp = ctx.obj.get("use_amp", True)

        if text_embedder == "legal-bert":
            from src.corp_speech_risk_dataset.encoding.legal_bert_embedder import (
                get_legal_bert_embedder,
            )

            model = get_legal_bert_embedder(use_amp=use_amp)
            txt_dim = model.get_sentence_embedding_dimension()
        else:
            from src.corp_speech_risk_dataset.encoding.stembedder import (
                get_sentence_embedder,
            )

            st_model = get_sentence_embedder("all-MiniLM-L6-v2")
            txt_dim = st_model.get_sentence_embedding_dimension()

        if txt_dim is None:
            raise ValueError("Could not determine text embedding dimension")
        fusion_model = CrossModalFusion(
            text_dim=txt_dim,
            graph_dim=hidden_dim,
            dropout=fusion_dropout,
            num_heads=fusion_heads,
        )
        print("  Using untrained CrossModalFusion - will train on existing embeddings")

    if graph_embed == "node2vec":
        print(f"Node2Vec model loaded")
    elif graph_embed == "wl":
        print(f"Weisfeiler-Lehman kernel features")
    print("=" * 60)

    total_processed = 0
    step_start_time = time.time()

    # Initialize fusion_model for processor scope - will be set if cross-modal training occurs
    if "fusion_model" not in locals():
        fusion_model = None

    def processor(lines):
        nonlocal total_processed, fusion_model
        batch_count = 0
        for line in lines:
            js = json.loads(line)
            # clear any stale graph embedding
            js.pop("gph_emb", None)
            js.pop("fused_emb", None)  # Clear any stale fusion embeddings

            # IMPORTANT: Extract graph features based on the method
            from src.corp_speech_risk_dataset.encoding.parser import to_dependency_graph

            if graph_embed == "wl":
                # Only compute WL features for WL method
                from src.corp_speech_risk_dataset.encoding.wl_features import wl_vector

                wl_vec = wl_vector(js["text"])
                js["wl_indices"] = wl_vec.indices.tolist()
                js["wl_counts"] = wl_vec.data.tolist()
                # No dependency edges needed for WL
                js["deps"] = []

                # Use the utility function with the pre-created model
                gph_tensor = compute_graph_embedding(
                    js["text"], "wl", node2vec_model, sage_model, fusion_model
                )
                js["gph_emb"] = gph_tensor.tolist()

            elif graph_embed == "cross":
                # For cross-modal: use existing embeddings from data if available
                text_embedder = ctx.obj.get("text_embedder", "st")

                # Check for appropriate text embedding based on text_embedder setting
                has_text_emb = (
                    "legal_bert_emb" in js
                    if text_embedder == "legal-bert"
                    else "st_emb" in js
                )

                # Debug: Print what embeddings are available
                if total_processed == 0:  # Only print for first entry
                    available_fields = [k for k in js.keys() if "emb" in k]
                    print(f"[DEBUG] text_embedder setting: {text_embedder}")
                    print(f"[DEBUG] Available embedding fields: {available_fields}")
                    print(
                        f"[DEBUG] has_text_emb: {has_text_emb}, has gph_emb: {'gph_emb' in js}"
                    )

                if has_text_emb and "gph_emb" in js:
                    # Use existing embeddings from previous stages
                    if text_embedder == "legal-bert":
                        text_emb = torch.tensor(js["legal_bert_emb"])
                    else:
                        text_emb = torch.tensor(js["st_emb"])
                    graph_emb = torch.tensor(js["gph_emb"])

                    # Fuse embeddings using fusion model
                    if fusion_model is not None:
                        fusion_model.eval()
                        # Ensure tensors and model are on CPU for inference
                        fusion_model = fusion_model.cpu()
                        with torch.no_grad():
                            fused_emb = fusion_model(
                                text_emb.unsqueeze(0), graph_emb.unsqueeze(0)
                            )[0]
                        js["fused_emb"] = fused_emb.tolist()
                    else:
                        # Fallback: just concatenate embeddings
                        fused_emb = torch.cat([text_emb, graph_emb])
                        js["fused_emb"] = fused_emb.tolist()
                else:
                    # Fallback: compute embeddings if not available - respect text_embedder
                    text_embedder = ctx.obj.get("text_embedder", "st")
                    use_amp = ctx.obj.get("use_amp", True)

                    if text_embedder == "legal-bert":
                        from src.corp_speech_risk_dataset.encoding.legal_bert_embedder import (
                            get_legal_bert_embedder,
                        )

                        model = get_legal_bert_embedder(use_amp=use_amp)
                        text_emb_np = model.encode([js["text"]], convert_to_numpy=True)
                        text_emb = torch.tensor(text_emb_np[0])
                    else:
                        from src.corp_speech_risk_dataset.encoding.stembedder import (
                            get_sentence_embedder,
                        )

                        st_model = get_sentence_embedder("all-MiniLM-L6-v2")
                        text_emb = st_model.encode(
                            [js["text"]], convert_to_tensor=True
                        )[0]

                    # Get graph embedding via GraphSAGE
                    dep_graph = to_dependency_graph(js["text"])
                    js["deps"] = [(h, t, l) for h, t, l in dep_graph.edges.data("dep")]
                    graph_emb = compute_graph_embedding(
                        js["text"],
                        "graphsage",
                        node2vec_model,
                        sage_model,
                        fusion_model,
                    )

                    # Fuse embeddings using trained fusion model
                    if fusion_model is not None:
                        fusion_model.eval()
                        fusion_model = fusion_model.cpu()
                        with torch.no_grad():
                            fused_emb = fusion_model(
                                text_emb.unsqueeze(0), graph_emb.unsqueeze(0)
                            )[0]
                        js["fused_emb"] = fused_emb.tolist()
                        js["gph_emb"] = graph_emb.tolist()
                        # Save text embedding to appropriate field
                        if text_embedder == "legal-bert":
                            js["legal_bert_emb"] = text_emb.tolist()
                        else:
                            js["st_emb"] = text_emb.tolist()
                    else:
                        # Fallback if no trained fusion model
                        js["gph_emb"] = graph_emb.tolist()

            else:
                # For GraphSAGE and Node2Vec: extract dependency parse information
                dep_graph = to_dependency_graph(js["text"])
                js["deps"] = [(h, t, l) for h, t, l in dep_graph.edges.data("dep")]
                # No WL features needed for GraphSAGE/Node2Vec
                js.pop("wl_indices", None)
                js.pop("wl_counts", None)

                # Use the utility function with the pre-created model
                gph_tensor = compute_graph_embedding(
                    js["text"], graph_embed, node2vec_model, sage_model, fusion_model
                )
                js["gph_emb"] = gph_tensor.tolist()

            js["gph_method"] = graph_embed

            # Store training metrics in metadata for first entry if available
            if (
                total_processed == 0
                and graph_embed in ["graphsage", "cross"]
                and training_metrics
            ):
                js["graphsage_training_metrics"] = training_metrics

            total_processed += 1

            # Progress logging every 1000 entries
            if total_processed % 1000 == 0:
                print(f"\n[GRAPH EMBEDDING] Processed {total_processed:,} entries")
                if graph_embed == "wl":
                    print(
                        f"  └─ WL features: {len(js.get('wl_indices', []))} dimensions"
                    )
                elif graph_embed == "cross":
                    print(
                        f"  └─ CrossModal: {len(js.get('fused_emb', []))}D fused embeddings"
                    )
                    print(
                        f"  └─ Text: {len(js.get('st_emb', []))}D, Graph: {len(js.get('gph_emb', []))}D"
                    )
                elif graph_embed == "graphsage":
                    print(f"  └─ GraphSAGE: {len(js.get('gph_emb', []))}D embeddings")
                    print(
                        f"  └─ Last entry: {len(js.get('deps', []))} dependency edges"
                    )
                else:
                    print(
                        f"  └─ Last entry: {len(js.get('deps', []))} dependency edges"
                    )
                print(f"  └─ {graph_embed.upper()} embedding computed successfully")

            yield json.dumps(js, ensure_ascii=False)

    def cleanup():
        step_time = time.time() - step_start_time
        print(f"\n[STEP COMPLETE] GRAPH EMBEDDING")
        print(f"Processed: {total_processed:,} entries")
        print(f"Time: {step_time:.2f}s")
        if graph_embed in ["graphsage", "cross"] and training_metrics:
            print(f"Training metrics saved in first entry metadata")
        print("=" * 60)

    ctx.obj["processors"].append(processor)
    if "cleanup_funcs" not in ctx.obj:
        ctx.obj["cleanup_funcs"] = []
    ctx.obj["cleanup_funcs"].append(cleanup)


@cli.command("fuse")
@click.option(
    "--train-fusion",
    is_flag=True,
    default=True,
    help="Train CrossModal fusion with InfoNCE loss (default: True)",
)
@click.option(
    "--max-training-samples",
    type=int,
    default=5000,
    help="Maximum samples for fusion training (default: 5000)",
)
@click.pass_context
def cmd_fuse(ctx, train_fusion, max_training_samples):
    """Fuse text+graph embeddings via CrossModalFusion with advanced InfoNCE training."""

    # Get fusion hyperparameters from context
    fusion_epochs = ctx.obj.get("fusion_epochs", 15)
    fusion_batch_size = ctx.obj.get("fusion_batch_size", 256)
    fusion_temperature = ctx.obj.get("fusion_temperature", 0.1)
    fusion_dropout = ctx.obj.get("fusion_dropout", 0.1)
    fusion_heads = ctx.obj.get("fusion_heads", 8)
    fusion_patience = ctx.obj.get("fusion_patience", 3)
    adaptive_temperature = ctx.obj.get("adaptive_temperature", True)
    hard_negative_weight = ctx.obj.get("hard_negative_weight", 1.3)
    use_amp = ctx.obj.get("use_amp", True)
    text_embedder = ctx.obj.get("text_embedder", "st")

    print(f"\n[STEP] ADVANCED CROSS-MODAL FUSION")
    print(f"Model: CrossModalFusion with InfoNCE loss")
    if train_fusion:
        print(f"Training: ENABLED with your hyperparameters")
        print(f"  ├─ Epochs: {fusion_epochs}")
        print(f"  ├─ Batch size: {fusion_batch_size}")
        print(f"  ├─ Temperature: {fusion_temperature}")
        print(f"  ├─ Dropout: {fusion_dropout}")
        print(f"  ├─ Attention heads: {fusion_heads}")
        print(f"  ├─ Patience: {fusion_patience}")
        print(f"  ├─ Adaptive temperature: {adaptive_temperature}")
        print(f"  ├─ Hard negative weight: {hard_negative_weight}")
        print(f"  ├─ AMP: {use_amp}")
        print(f"  └─ Text embedder: {text_embedder}")
    else:
        print(f"Training: DISABLED (using untrained fusion)")
    print(f"Text embedder: {text_embedder}")
    print("=" * 60)

    from src.corp_speech_risk_dataset.encoding.graphembedder import (
        CrossModalFusion,
        train_crossmodal_fusion,
    )
    import torch

    total_processed = 0
    step_start_time = time.time()
    fusion_model = None
    text_dim = None
    graph_dim = None

    def processor(lines):
        nonlocal total_processed, fusion_model, text_dim, graph_dim

        # collect all incoming lines; nothing to do if empty
        batch = list(lines)
        if not batch:
            return []  # empty generator

        # gather embeddings
        emb_text = []
        emb_graph = []
        batch_data = []

        for l in batch:
            js = json.loads(l)
            # clear previous fusion embeddings
            js.pop("fused_emb", None)

            # Use appropriate text embedding field based on text_embedder setting
            if text_embedder == "legal-bert" and "legal_bert_emb" in js:
                emb_text.append(js["legal_bert_emb"])
            elif "st_emb" in js:
                emb_text.append(js["st_emb"])
            else:
                raise ValueError(
                    f"No compatible text embeddings found. Expected {'legal_bert_emb' if text_embedder == 'legal-bert' else 'st_emb'}"
                )

            emb_graph.append(js["gph_emb"])
            batch_data.append(js)

        # Initialize and train fusion model on first batch
        if fusion_model is None:
            text_dim = len(emb_text[0])
            graph_dim = len(emb_graph[0])
            print(
                f"\nFusion config: {text_dim}D (text) + {graph_dim}D (graph) → {max(text_dim, graph_dim)}D"
            )

            fusion_model = CrossModalFusion(
                text_dim=text_dim,
                graph_dim=graph_dim,
                dropout=fusion_dropout,
                num_heads=fusion_heads,
            )

            if train_fusion:
                print(
                    f"\n[FUSION TRAINING] Collecting training samples from all files..."
                )

                # Collect training data from all files
                training_text_embs = []
                training_graph_embs = []
                sample_count = 0

                # First, collect from current batch
                training_text_embs.extend(emb_text)
                training_graph_embs.extend(emb_graph)
                sample_count += len(emb_text)

                # Collect from remaining files if needed
                if sample_count < max_training_samples:
                    for file_path in ctx.obj["files"]:
                        if sample_count >= max_training_samples:
                            break
                        try:
                            with open(file_path, "r", encoding="utf-8") as f:
                                for line in f:
                                    if sample_count >= max_training_samples:
                                        break
                                    try:
                                        js = json.loads(line)
                                        # Check for required embeddings
                                        has_text = (
                                            "legal_bert_emb" in js
                                            if text_embedder == "legal-bert"
                                            else "st_emb" in js
                                        )
                                        if has_text and "gph_emb" in js:
                                            if text_embedder == "legal-bert":
                                                training_text_embs.append(
                                                    js["legal_bert_emb"]
                                                )
                                            else:
                                                training_text_embs.append(js["st_emb"])
                                            training_graph_embs.append(js["gph_emb"])
                                            sample_count += 1
                                    except Exception:
                                        continue
                        except Exception:
                            continue

                print(
                    f"[FUSION TRAINING] Collected {len(training_text_embs):,} training pairs"
                )

                # Train the fusion model
                fusion_model = train_crossmodal_fusion(
                    fusion_model,
                    [torch.tensor(emb) for emb in training_text_embs],
                    [torch.tensor(emb) for emb in training_graph_embs],
                    epochs=fusion_epochs,
                    batch_size=fusion_batch_size,
                    temperature=fusion_temperature,
                    patience=fusion_patience,
                    use_amp=use_amp,
                    adaptive_temperature=adaptive_temperature,
                    hard_negative_weight=hard_negative_weight,
                )

                print(f"[FUSION TRAINING] Training completed!")

        # Apply fusion to current batch
        fusion_model.eval()
        with torch.no_grad():
            fused = fusion_model(
                torch.tensor(emb_text), torch.tensor(emb_graph)
            ).tolist()

        # emit fused rows
        for js, f in zip(batch_data, fused):
            js["fused_emb"] = f
            total_processed += 1

            # Progress logging every 1000 entries
            if total_processed % 1000 == 0:
                print(f"\n[CROSS-MODAL FUSION] Processed {total_processed:,} entries")
                print(
                    f"  └─ Fused {text_dim}D text + {graph_dim}D graph → {len(f)}D combined"
                )

            yield json.dumps(js, ensure_ascii=False)

    def cleanup():
        step_time = time.time() - step_start_time
        print(f"\n[STEP COMPLETE] CROSS-MODAL FUSION")
        print(f"Processed: {total_processed:,} entries")
        if text_dim and graph_dim:
            print(f"Output dimensions: {text_dim + graph_dim}D")
        print(f"Time: {step_time:.2f}s")
        print("=" * 60)

    ctx.obj["processors"].append(processor)
    if "cleanup_funcs" not in ctx.obj:
        ctx.obj["cleanup_funcs"] = []
    ctx.obj["cleanup_funcs"].append(cleanup)


@cli.result_callback()
@click.pass_context
def run_pipeline(ctx, processors, **kwargs):
    files = ctx.obj["files"]
    out_root = ctx.obj["out_root"]
    stage = ctx.obj["stage"]
    from tqdm import tqdm

    # Print pipeline configuration
    print(f"\n{'='*80}")
    print(f"PIPELINE CONFIGURATION")
    print(f"{'='*80}")
    print(f"Input stage: {stage}")
    print(f"Output stage: {stage + 1}")
    print(f"Processing steps: {len(ctx.obj['processors'])} processors configured")

    # Determine what steps are being run based on processor count and context
    if len(ctx.obj["processors"]) == 0:
        print("Steps: ALL (tokenize + embed + graph + fuse)")
    else:
        print("Steps: Custom pipeline with the following processors:")
        for i, proc in enumerate(ctx.obj["processors"], 1):
            proc_name = proc.__name__ if hasattr(proc, "__name__") else str(proc)
            print(f"  {i}. {proc_name}")
    print(f"{'='*80}\n")

    # First pass: count total entries across all files
    print("[PHASE 1] Counting total entries across all files...")
    total_entries = 0
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                file_entries = sum(1 for _ in f)
                total_entries += file_entries
        except Exception as e:
            logger.warning(f"Could not count entries in {file_path}: {e}")

    print(f"[PHASE 1] Found {total_entries:,} total entries to process\n")

    # Progress bar over total entries
    print(
        f"[PHASE 2] Processing {len(files)} files through {len(ctx.obj['processors'])}-step pipeline...\n"
    )

    with tqdm(total=total_entries, desc="Processing entries", unit="entry") as pbar:
        for in_file in files:
            # Open input stream and apply processors
            with open(in_file, "r", encoding="utf-8") as stream:
                proc_stream = stream
                for proc in ctx.obj["processors"]:
                    proc_stream = proc(proc_stream)
                # Prepare output
                base = in_file.stem
                out_root.mkdir(parents=True, exist_ok=True)
                out_file = (
                    out_root
                    / f"{base.replace(f'stage{stage}', f'stage{stage+1}')}.jsonl"
                )
                with open(out_file, "w", encoding="utf-8") as fout:
                    for line in proc_stream:
                        fout.write(line + "\n")
                        pbar.update(1)  # Update progress for each entry processed

    # Run cleanup functions
    if "cleanup_funcs" in ctx.obj:
        for cleanup_func in ctx.obj["cleanup_funcs"]:
            cleanup_func()

    print(f"\n[PIPELINE COMPLETE] All {len(files)} files processed successfully!")
    print(f"Output directory: {out_root}")
    print("=" * 80)


if __name__ == "__main__":
    cli()
