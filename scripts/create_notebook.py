import nbformat as nbf
from pathlib import Path
import polars as pl
import duckdb
import holoviews as hv, hvplot.pandas
import datashader as ds
from datashader import transfer_functions as tf
hv.extension('bokeh')

STAGE_MAP = {
    0: ["../../data/stage0_raw.jsonl"],
    1: ["../../data/stage1_cleaner.jsonl"],
    2: ["../../data/stage2_extractor.jsonl"],
    3: ["../../data/stage3_attributor.jsonl"],
    4: ["../../data/stage4_reranker.jsonl"]
}

def create_notebook(data_root: Path, notebook_path: Path):
    """
    Programmatically generates the pipeline visualization Jupyter notebook.
    This approach is more robust than trying to write the complex JSON manually.
    """
    data_root = data_root.expanduser().resolve()  # Always use absolute path
    nb = nbf.v4.new_notebook()

    # --- Papermill parameters cell (always first) ---
    param_cell = nbf.v4.new_code_cell(
        "# --- Papermill parameters ---\n"
        "from pathlib import Path\n"
        "from corp_speech_risk_dataset.utils.discovery import find_stage_files\n\n"
        f"DATA_ROOT = Path(r\"{data_root}\")  # absolute path\n"
        "STAGE_MAP  = find_stage_files(DATA_ROOT)\n"
        "print('→ DATA_ROOT:', DATA_ROOT)\n"
        "print({k: len(v) for k, v in STAGE_MAP.items()})\n"
        "assert STAGE_MAP, f'STAGE_MAP empty – does {DATA_ROOT} contain *_stage*.jsonl ?'\n",
        metadata={"tags": ["parameters"]}
    )
    nb.cells.insert(0, param_cell)
    # --- Print notebook CWD for debugging ---
    nb.cells.insert(1, nbf.v4.new_code_cell(
        "from pathlib import Path\nprint('Notebook CWD:', Path.cwd())"
    ))

    # --- CELL 1: Title and Intro (Markdown) ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "# Quote Extraction Pipeline Visualization\n\n"
        "This notebook loads the JSONL outputs from each stage of the quote extraction pipeline, "
        "combines them into a single DataFrame, and provides visualizations to analyze the process."
    ))

    # --- CELL 2: Environment Setup (Code) ---
    nb.cells.append(nbf.v4.new_code_cell(
        "import os\n"
        "import sys\n"
        "import pandas as pd\n"
        "import matplotlib\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "# Add project root to sys.path to allow for local module imports\n"
        "project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))\n"
        "if project_root not in sys.path:\n"
        "    sys.path.insert(0, project_root)\n\n"
        "print(f\"pandas version: {pd.__version__}\")\n"
        "print(f\"matplotlib version: {matplotlib.__version__}\")\n\n"
        "sns.set_theme(style=\"whitegrid\")"
    ))

    # --- CELL 3: Configuration (Markdown) ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 1. Configuration & Parameters\n\n"
        "Define paths to the data files and set visualization parameters."
    ))

    # --- CELL 4: Configuration (Code) ---
    nb.cells.append(nbf.v4.new_code_cell(
        "import polars as pl\n"
        "import duckdb\n"
        "import holoviews as hv, hvplot.pandas\n"
        "import datashader as ds\n"
        "from datashader import transfer_functions as tf\n"
        "hv.extension('bokeh')\n"
        "# Visualization settings\n"
        "SCORE_NULL_PLACEHOLDER = -1.0\n"
        "MAX_SPEAKERS = 10\n"
        "MAX_QUOTES = 3\n"
        "# pd.set_option('display.max_rows', 20)\n"
        "# pd.set_option('display.min_rows', 10)\n"
    ))

    # --- CELL 5: Helpers (Markdown) ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 2. Utility Functions\n\n"
        "Helper functions to load and prepare the data for analysis."
    ))

    # --- CELL 6: Helpers (Code) ---
    nb.cells.append(nbf.v4.new_code_cell(
        "def load_stage(stage: int) -> pl.DataFrame:\n"
        "    paths = STAGE_MAP.get(stage, [])\n"
        "    if not paths:\n"
        "        return pl.DataFrame()\n"
        "    # Properly quote each path as a plain string\n"
        "    joined = ', '.join(f\"'{str(p)}'\" for p in paths)\n"
        "    q = f\"SELECT *, {stage} AS stage FROM read_json_auto([{joined}])\"\n"
        "    return duckdb.query(q).pl()   # ~2-3× faster than pandas.read_json\n\n"
        "def prepare_dataframe(dfs: list[pl.DataFrame]) -> pl.DataFrame:\n"
        "    if not any(df.height > 0 for df in dfs):\n"
        "        return pl.DataFrame()\n"
        "    full = pl.concat(dfs)\n"
        "    full = full.with_columns([\n"
        "        pl.col('speaker').fill_null('<none>'),\n"
        "        pl.col('score').fill_null(SCORE_NULL_PLACEHOLDER)\n"
        "    ])\n"
        "    return full.sort(['stage', 'speaker', 'score'], descending=[False, False, True])"
    ))

    # --- CELL 7: Data Loading (Markdown) ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 3. Data Loading & Preparation\n\n"
        "Load data from all four stages into a single pandas DataFrame."
    ))

    # --- CELL 8: Data Loading (Code) ---
    nb.cells.append(nbf.v4.new_code_cell(
        "import polars as pl\n"
        "import duckdb\n"
        "# 1. Define the target schema\n"
        "TARGET_SCHEMA = {\n"
        "    'doc_id':  pl.Utf8,\n"
        "    'text':    pl.Utf8,\n"
        "    'context': pl.Utf8,\n"
        "    'speaker': pl.Utf8,\n"
        "    'score':   pl.Float64,\n"
        "    'urls':    pl.List(pl.Utf8),\n"
        "    'stage':   pl.Int64,\n"
        "    '_src':    pl.Utf8,\n"
        "}\n\n"
        "# 2. Guaranteed Polars loader\n"
        "def load_stage(stage: int) -> pl.DataFrame:\n"
        "    paths = STAGE_MAP.get(stage, [])\n"
        "    if not paths:\n"
        "        return pl.DataFrame()\n"
        "    joined = ', '.join(f\"'{p}'\" for p in paths)\n"
        "    arrow_tbl = duckdb.query(f\"SELECT * FROM read_json_auto([{joined}])\").arrow()\n"
        "    df = pl.from_arrow(arrow_tbl)\n"
        "    # drop any existing 'stage' and inject our literal\n"
        "    return df.drop('stage').with_columns(pl.lit(stage, pl.Int64).alias('stage'))\n\n"
        "# 3. Align helper (using with_columns)\n"
        "def align_schema(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:\n"
        "    for col, dtype in schema.items():\n"
        "        if col not in df.columns:\n"
        "            df = df.with_columns(pl.lit(None, dtype=dtype).alias(col))\n"
        "        else:\n"
        "            df = df.with_columns(df[col].cast(dtype).alias(col))\n"
        "    return df.select(list(schema.keys()))\n\n"
        "# 4. Load, align, concat\n"
        "dfs = [load_stage(s) for s in sorted(STAGE_MAP)]\n"
        "dfs = [df for df in dfs if df.height > 0]\n"
        "aligned = [align_schema(df, TARGET_SCHEMA) for df in dfs]\n"
        "full_df = pl.concat(aligned, how='vertical') if aligned else pl.DataFrame()\n"
        "print('Combined DataFrame shape:', full_df.shape)\n"
        "display(full_df.head(10)) if full_df.height > 0 else print('DataFrame is empty. Please run `make run` first to generate the data.')"
    ))

    # --- CELL 9: Stage Inspection (Markdown) ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 4. Stage-By-Stage Inspection\n\n"
        "Let's examine the output of each pipeline stage. We'll print a formatted summary of the quotes found at each step."
    ))

    # --- CELL 10: Stage Inspection (Code) ---
    nb.cells.append(nbf.v4.new_code_cell(
        "def print_stage_overview(df: pl.DataFrame):\n"
        "    \"\"\"Iterates through the dataframe and prints a formatted summary.\"\"\"\n"
        "    if df.is_empty():\n"
        "        print('No data to display.')\n"
        "        return\n"
        "    for stage in df['stage'].unique().to_list():\n"
        "        stage_df = df.filter(pl.col('stage') == stage)\n"
        "        print(f\"\\n\\n{'='*10} Stage {stage} {'='*10}\")\n"
        "        for speaker in stage_df['speaker'].unique().to_list():\n"
        "            grp = stage_df.filter(pl.col('speaker') == speaker)\n"
        "            count = grp.height\n"
        "            print(f\"\\n-- Speaker: {speaker!r} ({count} rows) --\")\n"
        "            top5 = grp.sort('score', reverse=True).head(5)\n"
        "            for row in top5.iter_rows(named=True):\n"
        "                score = row['score']\n"
        "                score_str = f\"[{score:.2f}]\" if score >= 0 else \"[---]\"\n"
        "                text = row['text'].replace('\\n', ' ')\n"
        "                if len(text) > 120:\n"
        "                    text = text[:117] + '...'\n"
        "                print(f\"  • {score_str} {text}\")\n"
    ))

    # --- CELL 11: Visualizations (Markdown) ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 5. Cross-Stage Comparison & Visualizations\n\n"
        "Now let's visualize the filtering process across stages."
    ))
    
    # --- CELL 12: Chart 1 (Markdown) ---
    nb.cells.append(nbf.v4.new_markdown_cell("### 5.1 Quote Survival Rate by Stage"))

    # --- CELL 13: Chart 1 (Code) ---
    nb.cells.append(nbf.v4.new_code_cell(
        "import polars as pl\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "# Polars-native check for emptiness\n"
        "if not full_df.is_empty():\n"
        "    stage_counts = (\n"
        "        full_df\n"
        "        .group_by('stage')\n"
        "        .agg(pl.col('doc_id').n_unique().alias('count'))\n"
        "        .sort('stage')\n"
        "    )\n"
        "    labels = [f'stage{s}' for s in stage_counts['stage'].to_list()]\n"
        "    counts = stage_counts['count'].to_list()\n"
        "    plt.figure(figsize=(10, 6))\n"
        "    ax = sns.barplot(x=labels, y=counts)\n"
        "    ax.set_title('Number of Items Passing Each Stage')\n"
        "    ax.set_ylabel('Count (log scale)')\n"
        "    ax.set_yscale('log')\n"
        "    ax.bar_label(ax.containers[0])\n"
        "    plt.show()\n"
        "else:\n"
        "    print('No data to plot.')"
    ))

    # --- CELL 14: Chart 2 (Markdown) ---
    nb.cells.append(nbf.v4.new_markdown_cell("### 5.2 Speaker Distribution in Final Stage"))
    
    # --- CELL 15: Chart 2 (Code) ---
    nb.cells.append(nbf.v4.new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "# 5.2: Final-stage speaker counts\n"
        "if not full_df.is_empty():\n"
        "    final = full_df.filter(pl.col('stage') == 4)\n"
        "    if not final.is_empty():\n"
        "        speaker_counts = (\n"
        "            final\n"
        "            .group_by('speaker')\n"
        "            .agg(pl.len().alias('n_quotes'))\n"
        "            .sort('n_quotes', descending=True)\n"
        "            .head(20)\n"
        "        )\n"
        "        df_sp = speaker_counts.to_pandas()\n"
        "        plt.figure(figsize=(8, 6))\n"
        "        sns.barplot(data=df_sp, y='speaker', x='n_quotes', orient='h')\n"
        "        plt.title('stage4_reranker: Top 20 Speakers by Quote Count')\n"
        "        plt.xlabel('Number of Quotes')\n"
        "        plt.ylabel('Speaker')\n"
        "        plt.tight_layout()\n"
        "        plt.show()\n"
        "    else:\n"
        "        print('No quotes survived to the final stage.')\n"
        "else:\n"
        "    print('No data to plot.')"
    ))

    # --- CELL 16: Chart 3 (Markdown) ---
    nb.cells.append(nbf.v4.new_markdown_cell("### 5.3 Score Distribution in Final Stage"))

    # --- CELL 17: Chart 3 (Code) ---
    nb.cells.append(nbf.v4.new_code_cell(
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "# 5.3: Score histogram for final-stage quotes\n"
        "if not full_df.is_empty():\n"
        "    with_scores = (\n"
        "       full_df\n"
        "       .filter((pl.col('stage') == 4) & (pl.col('score') >= 0))\n"
        "       .select('score')\n"
        "    )\n"
        "    if not with_scores.is_empty():\n"
        "        scores = with_scores.to_series().to_list()\n"
        "        plt.figure(figsize=(8, 5))\n"
        "        sns.histplot(scores, bins=20, kde=True)\n"
        "        plt.title('stage4_reranker: Semantic Score Distribution')\n"
        "        plt.xlabel('Similarity Score')\n"
        "        plt.ylabel('Frequency')\n"
        "        plt.tight_layout()\n"
        "        plt.show()\n"
        "    else:\n"
        "        print('No scored quotes in the final stage to plot.')\n"
        "else:\n"
        "    print('No data to plot.')"
    ))

    # --- CELL 18: Conclusion (Markdown) ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 6. Conclusion\n\n"
        "This notebook provides a detailed, stage-by-stage analysis of the quote extraction pipeline. "
        "The visualizations highlight the filtering effectiveness at each step, from raw text to final, semantically-scored quotes."
    ))

    # --- Write the notebook to a file ---
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)
    
    print(f"Successfully created notebook at: {notebook_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    create_notebook(Path(args.data_root), Path(args.out)) 