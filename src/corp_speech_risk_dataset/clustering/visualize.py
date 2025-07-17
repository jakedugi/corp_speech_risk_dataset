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
        coords: pd.DataFrame,
        *,
        out_html: str | Path = "clusters.html",
        title: str = "Vector clusters (UMAP)",
    ) -> Path:
        fig = px.scatter(
            coords,
            x="x",
            y="y",
            color="cluster",
            hover_data=["sentence", "idx", "doc_id"],
            opacity=0.7,
            title=title,
        )
        out_html = Path(out_html)
        fig.write_html(out_html, include_plotlyjs="cdn")
        return out_html
