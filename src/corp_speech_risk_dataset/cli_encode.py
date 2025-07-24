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

from transformers import GPT2Model, GPT2TokenizerFast
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

    # -------- Sentence-Transformer (optional) --------------------------- #
    st_model = get_sentence_embedder(st_model_name) if st_model_name else None

    # -------- GPT-2 mean-pool (optional) ------------------------------- #
    if text_model.lower() == "gpt2":
        # Prefer CUDA GPU, then Apple MPS, then CPU
        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        gpt_tok = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
        gpt_tok.pad_token = gpt_tok.eos_token  # <-- fix: set pad_token
        gpt_mod = GPT2Model.from_pretrained("gpt2", local_files_only=True)
        gpt_mod.eval()
        gpt_mod = gpt_mod.to(device)
    else:
        # still need a dummy device for signature
        device = torch.device("cpu")
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
    print(
        f"[INFO] Using Text Model: {text_model} "
        f"({st_model.__class__.__name__ if st_model else 'none'})"
    )
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
) -> List[Dict[str, Any]]:
    """Embed a buffered batch & yield enriched JSON rows."""
    # --- Stage: Text Embedding ---
    t0 = time.time()
    if text_model.lower() == "gpt2":
        enc = gpt_tok(texts, return_tensors="pt", padding=True, truncation=True).to(
            device
        )
        with torch.no_grad():
            hidden = gpt_mod(**enc).last_hidden_state
        txt_embs = hidden.mean(dim=1).cpu().numpy()
    elif text_model.lower() == "st":
        txt_embs = st_model.encode(texts, convert_to_numpy=True, batch_size=batch_size)
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
    if graph_embed == "cross" and not isinstance(txt_embs, list):
        t2 = time.time()
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

        # Handle different text embedding types
        gpt2_emb = None
        st_emb = None
        if text_model.lower() == "gpt2" and txt_vec is not None:
            gpt2_emb = txt_vec.tolist()
        elif text_model.lower() == "st" and txt_vec is not None:
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
                "gpt2_emb": gpt2_emb,
                "st_emb": st_emb,
                "gph_method": graph_embed,
                "gph_emb": gph_embs[i].tolist() if gph_embs is not None else None,
                **(
                    {"fused_emb": fused_embs[i].tolist()}
                    if graph_embed == "cross" and fused_embs is not None
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
@click.pass_context
def cli(ctx, in_path, out_root, recursive, stage):
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
        # print(f"[STAGE] Tokenization (model={text_model}) start")
        from src.corp_speech_risk_dataset.encoding.tokenizer import (
            SentencePieceTokenizer,
        )
        from transformers import GPT2TokenizerFast

        # initialize only the requested tokenizer
        tok_sp = None
        tok_gpt = None
        if text_model == "gpt2":
            tok_gpt = GPT2TokenizerFast.from_pretrained("gpt2", local_files_only=True)
            tok_gpt.pad_token = tok_gpt.eos_token  # <-- fix: set pad_token
        else:
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
            # print(f"[STAGE] Tokenization done: {dt:.2f}s")

    ctx.obj["processors"].append(processor)


@cli.command("embed")
@click.option("--st-model", required=True)
@click.pass_context
def cmd_embed(ctx, st_model):
    """Embed text via SentenceTransformer or GPT2."""

    def processor(lines):
        import time

        start_time = time.time()
        # print(f"[STAGE] Text Embedding (model={st_model}) start")
        from src.corp_speech_risk_dataset.encoding.stembedder import (
            get_sentence_embedder,
        )

        model = get_sentence_embedder(st_model)
        batch, metas = [], []
        try:
            for line in lines:
                js = json.loads(line)
                # clear previous text embeddings
                js.pop("st_emb", None)
                js.pop("gpt2_emb", None)
                batch.append(js["text"])
                metas.append(js)
                if len(batch) >= 64:
                    embs = model.encode(batch, convert_to_numpy=True)
                    for mj, emb in zip(metas, embs):
                        mj["st_emb"] = emb.tolist()
                        yield json.dumps(mj, ensure_ascii=False)
                    batch, metas = [], []
            # flush leftover
            if batch:
                embs = model.encode(batch, convert_to_numpy=True)
                for mj, emb in zip(metas, embs):
                    mj["st_emb"] = emb.tolist()
                    yield json.dumps(mj, ensure_ascii=False)
        finally:
            dt = time.time() - start_time
            # print(f"[STAGE] Text Embedding done: {dt:.2f}s")

    ctx.obj["processors"].append(processor)


@cli.command("graph")
@click.option(
    "--graph-embed", type=click.Choice(["wl", "node2vec", "graphsage"]), default="wl"
)
@click.pass_context
def cmd_graph(ctx, graph_embed):
    """Embed graphs via WL, Node2Vec, or GraphSAGE."""
    # imports for dynamic graph processing
    from src.corp_speech_risk_dataset.encoding.graphembedder import (
        compute_graph_embedding,
        get_node2vec_embedder,
        get_graphsage_embedder,
    )

    node2vec_model = get_node2vec_embedder() if graph_embed == "node2vec" else None
    # Create GraphSAGE model with fixed dimensions
    sage_model = (
        get_graphsage_embedder(in_channels=16, hidden_channels=128, num_layers=2)
        if graph_embed == "graphsage"
        else None
    )

    def processor(lines):
        import time

        start_time = time.time()
        # print(f"[STAGE] Graph Embedding (method={graph_embed}) start")
        try:
            for line in lines:
                js = json.loads(line)
                # clear any stale graph embedding
                js.pop("gph_emb", None)

                # Use the utility function with the pre-created model
                gph_tensor = compute_graph_embedding(
                    js["text"], graph_embed, node2vec_model, sage_model
                )

                js["gph_emb"] = gph_tensor.tolist()
                yield json.dumps(js, ensure_ascii=False)
        finally:
            dt = time.time() - start_time
            # print(f"[STAGE] Graph Embedding done: {dt:.2f}s")

    ctx.obj["processors"].append(processor)


@cli.command("fuse")
@click.pass_context
def cmd_fuse(ctx):
    """Fuse text+graph embeddings via CrossModalFusion."""

    def processor(lines):
        import time

        start_time = time.time()
        # print("[STAGE] CrossModalFusion start")
        from src.corp_speech_risk_dataset.encoding.graphembedder import CrossModalFusion

        try:
            # collect all incoming lines; nothing to do if empty
            batch = list(lines)
            if not batch:
                return []  # empty generator
            texts = [json.loads(l)["text"] for l in batch]
            # gather embeddings
            emb_text = []
            emb_graph = []
            for l in batch:
                js = json.loads(l)
                # clear previous fusion embeddings
                js.pop("fused_emb", None)
                emb_text.append(js["st_emb"])
                emb_graph.append(js["gph_emb"])
            # perform fusion - use the actual dimensions from the embeddings
            text_dim = len(emb_text[0])
            graph_dim = len(emb_graph[0])
            fuse = CrossModalFusion(text_dim, graph_dim)
            import torch

            fused = fuse(torch.tensor(emb_text), torch.tensor(emb_graph)).tolist()
            # emit fused rows
            for l, f in zip(batch, fused):
                js = json.loads(l)
                js["fused_emb"] = f
                yield json.dumps(js, ensure_ascii=False)
        finally:
            dt = time.time() - start_time
            # print(f"[STAGE] CrossModalFusion done: {dt:.2f}s")

    ctx.obj["processors"].append(processor)


@cli.result_callback()
@click.pass_context
def run_pipeline(ctx, processors, in_path, out_root, recursive, stage):
    files = ctx.obj["files"]
    out_root = ctx.obj["out_root"]
    stage = ctx.obj["stage"]
    # Group input files by their immediate parent directory (case folders)
    case_dirs = sorted({f.parent for f in files})
    # Progress bar over case folders
    from tqdm import tqdm

    for case_dir in tqdm(case_dirs, desc="Encoding case folders", unit="folder"):
        # Process all files in this case folder
        case_files = [f for f in files if f.parent == case_dir]
        for in_file in case_files:
            # Count total entries in the file
            total = sum(1 for _ in open(in_file, "r", encoding="utf-8"))
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


if __name__ == "__main__":
    cli()
