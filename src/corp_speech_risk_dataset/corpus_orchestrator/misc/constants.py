import polars as pl

# TARGET_SCHEMA: preserves all original and new fields for lossless, auditable processing
TARGET_SCHEMA = {
    # Original metadata
    "doc_id": pl.Utf8,
    "stage": pl.Int64,
    "text": pl.Utf8,
    "context": pl.Utf8,
    "speaker": pl.Utf8,
    "score": pl.Float64,
    "urls": pl.List(pl.Utf8),
    "_src": pl.Utf8,
    # New encoding columns
    "sp_ids": pl.List(pl.Int64),
    "deps": pl.List(
        pl.Struct(
            [
                pl.Field("head", pl.Int64),
                pl.Field("child", pl.Int64),
                pl.Field("label", pl.Utf8),
            ]
        )
    ),
    "wl_indices": pl.List(pl.Int64),
    "wl_counts": pl.List(pl.Int64),
}
