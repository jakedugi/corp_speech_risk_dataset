"""Plotly helper to scatter 2‑D coords with hover‑tooltips."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import plotly.express as px
import pandas as pd


class Visualizer:
    """Write an interactive HTML scatter‑plot."""

    def __init__(self):
        pass

    def scatter(
        self,
        df: pd.DataFrame,
        *,
        out_html: str | Path = "clusters.html",
        title: str = "Vector clusters (UMAP)",
    ) -> Path:
        # ←— NEW: color by bucket, annotate missingness & real y
        fig = px.scatter(
            df,
            x="x",
            y="y",
            color="bucket",  # color by low/med/high/missing  [plotly.express.scatter docs turn0search17 ]
            hover_data=[
                "sentence",
                "idx",
                "doc_id",
                "final_judgement_real",
                "is_missing",
            ],
            opacity=0.7,
            title=title,
        )
        out_html = Path(out_html)
        fig.write_html(out_html, include_plotlyjs="cdn")
        return out_html
