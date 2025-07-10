import json
from pathlib import Path
import click
from loguru import logger
from src.corp_speech_risk_dataset.encoding.tokenizer import SentencePieceTokenizer
from src.corp_speech_risk_dataset.encoding.parser import to_dependency_graph
from src.corp_speech_risk_dataset.encoding.wl_features import wl_vector

# Will look under data/models/en.wiki.bpe.vs32000.model by default
tokenizer = SentencePieceTokenizer()

def encode_file(in_path: Path, extracted_root: Path, tokenized_root: Path):
    """Helper: encode one file, writing into mirrored structure."""
    rel_path = in_path.relative_to(extracted_root)
    out_path = tokenized_root / rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"→ Encoding {in_path} → {out_path}")
    with open(out_path, "wb") as fout:
        for line in in_path.open():
            row = json.loads(line)
            vec = wl_vector(row["text"])
            # capture fallback info too
            sp_ids, used_fallback, fallback_chars = tokenizer.encode_with_flag(row["text"])
            if used_fallback:
                logger.warning(f"Byte-fallback for chars: {fallback_chars}")
            enriched = {
                **row,
                "sp_ids": sp_ids,
                "byte_fallback": used_fallback,
                "fallback_chars": fallback_chars,
                "deps": [
                    (h, t, l)
                    for h, t, l in to_dependency_graph(row["text"]).edges.data("dep")
                ],
                "wl_indices": vec.indices.tolist(),
                "wl_counts": vec.data.tolist(),
            }
            fout.write((json.dumps(enriched, ensure_ascii=False) + "\n").encode())
    logger.success(f"✔ Written tokenized output to {out_path}")

@click.command(context_settings={"show_default": True})
@click.argument(
    "in_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True, path_type=Path),
)
@click.option(
    "-r",
    "--recursive",
    is_flag=True,
    help="When IN_PATH is a directory, process all *_stage4.jsonl files under it.",
)
def main(in_path: Path, recursive: bool):
    """
    Encode extracted quotes, either for a single .stage4.jsonl file
    or all such files under a directory (with -r/--recursive).
    """
    extracted_root = Path("data/extracted").resolve()
    tokenized_root = Path("data/tokenized").resolve()

    # Normalize
    in_path = in_path.resolve()

    if in_path.is_dir():
        if not recursive:
            raise click.ClickException(
                "IN_PATH is a directory; pass -r/--recursive to process all files."
            )
        # Process every *_stage4.jsonl under the tree:
        pattern = "*_stage4.jsonl"
        files = list(in_path.rglob(pattern))
        if not files:
            logger.warning(f"No files matching '{pattern}' under {in_path}")
            return
        for file in files:
            # ensure it’s under data/extracted/
            try:
                file.relative_to(extracted_root)
            except ValueError:
                logger.warning(f"Skipping {file}: not under {extracted_root}")
                continue
            encode_file(file, extracted_root, tokenized_root)
    else:
        # Single‐file mode
        try:
            in_path.relative_to(extracted_root)
        except ValueError as e:
            raise click.ClickException(f"Input must be under {extracted_root}: {e}")
        encode_file(in_path, extracted_root, tokenized_root)

if __name__ == "__main__":
    main() 