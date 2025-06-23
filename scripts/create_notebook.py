import nbformat as nbf
from pathlib import Path

def create_notebook():
    """
    Programmatically generates the pipeline visualization Jupyter notebook.
    This approach is more robust than trying to write the complex JSON manually.
    """
    nb = nbf.v4.new_notebook()

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
        "from pathlib import Path\n\n"
        "# The `run_extraction.py` script saves the data in a `data` directory at the project root.\n"
        "DATA_DIR = Path(\"../../data\")\n"
        "FILES = {\n"
        "    0: DATA_DIR / \"stage0_raw.jsonl\",\n"
        "    1: DATA_DIR / \"stage1_cleaner.jsonl\",\n"
        "    2: DATA_DIR / \"stage2_extractor.jsonl\",\n"
        "    3: DATA_DIR / \"stage3_attributor.jsonl\",\n"
        "    4: DATA_DIR / \"stage4_reranker.jsonl\"\n"
        "}\n\n"
        "# Visualization settings\n"
        "SCORE_NULL_PLACEHOLDER = -1.0"
    ))

    # --- CELL 5: Helpers (Markdown) ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 2. Utility Functions\n\n"
        "Helper functions to load and prepare the data for analysis."
    ))

    # --- CELL 6: Helpers (Code) ---
    nb.cells.append(nbf.v4.new_code_cell(
        "def load_stage(stage: int) -> pd.DataFrame:\n"
        "    \"\"\"Loads a single stage's JSONL file into a DataFrame.\"\"\"\n"
        "    file_path = FILES[stage]\n"
        "    if not file_path.exists():\n"
        "        print(f\"Warning: File not found for stage {stage} at {file_path}. Please run `make run` first.\")\n"
        "        return pd.DataFrame()\n"
        "    return pd.read_json(file_path, lines=True)\n\n"
        "def prepare_dataframe(dfs: list[pd.DataFrame]) -> pd.DataFrame:\n"
        "    \"\"\"Concatenates, cleans, and sorts the DataFrames from all stages.\"\"\"\n"
        "    if not any(not df.empty for df in dfs):\n"
        "        return pd.DataFrame()\n"
        "    full = pd.concat(dfs, ignore_index=True)\n"
        "    full[\"speaker\"] = full[\"speaker\"].fillna(\"<none>\")\n"
        "    full[\"score\"] = full[\"score\"].fillna(SCORE_NULL_PLACEHOLDER)\n"
        "    return full.sort_values([\"stage\", \"speaker\", \"score\"], ascending=[True, True, False])"
    ))

    # --- CELL 7: Data Loading (Markdown) ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 3. Data Loading & Preparation\n\n"
        "Load data from all four stages into a single pandas DataFrame."
    ))

    # --- CELL 8: Data Loading (Code) ---
    nb.cells.append(nbf.v4.new_code_cell(
        "dfs = [load_stage(s) for s in FILES]\n"
        "full_df = prepare_dataframe(dfs)\n\n"
        "if not full_df.empty:\n"
        "    print(\"Combined DataFrame shape:\", full_df.shape)\n"
        "    display(full_df.head(10))\n"
        "else:\n"
        "    print(\"DataFrame is empty. Please run `make run` first to generate the data.\")"
    ))

    # --- CELL 9: Stage Inspection (Markdown) ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 4. Stage-By-Stage Inspection\n\n"
        "Let's examine the output of each pipeline stage. We'll print a formatted summary of the quotes found at each step."
    ))

    # --- CELL 10: Stage Inspection (Code) ---
    nb.cells.append(nbf.v4.new_code_cell(
        "def print_stage_overview(df):\n"
        "    \"\"\"Iterates through the dataframe and prints a formatted summary.\"\"\"\n"
        "    if df.empty:\n"
        "        print(\"No data to display.\")\n"
        "        return\n"
        "    \n"
        "    for stage, stage_df in df.groupby(\"stage\"):\n"
        "        print(f\"\\n\\n{'='*10} Stage {stage} {'='*10}\")\n"
        "        for speaker, grp in stage_df.groupby(\"speaker\"):\n"
        "            print(f\"\\n-- Speaker: {speaker!r} ({len(grp)} rows) --\")\n"
        "            # Sort by score for display purposes and show top 5\n"
        "            for _, row in grp.sort_values('score', ascending=False).head(5).iterrows():\n"
        "                score_str = f\"[{row.score:.2f}]\" if row.score >= 0 else \"[---]\"\n"
        "                # For raw text, show a snippet\n"
        "                display_text = str(row.text).replace('\\n', ' ')\n"
        "                if len(display_text) > 120:\n"
        "                    display_text = display_text[:117] + '...'\n"
        "                print(f\"  • {score_str} {display_text}\")\n\n"
        "print_stage_overview(full_df)"
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
        "if not full_df.empty:\n"
        "    stage_counts = full_df.groupby('stage')['doc_id'].nunique()\n"
        "    stage_labels = {0: 'stage0_raw', 1: 'stage1_cleaner', 2: 'stage2_extractor', 3: 'stage3_attributor', 4: 'stage4_reranker'}\n"
        "    labels = [stage_labels.get(s, str(s)) for s in stage_counts.index]\n"
        "    plt.figure(figsize=(10, 6))\n"
        "    ax = sns.barplot(x=labels, y=stage_counts.values)\n"
        "    ax.set_title('Number of Items Passing Each Stage')\n"
        "    ax.set_ylabel('Count (Log Scale)')\n"
        "    ax.set_yscale('log')\n"
        "    ax.bar_label(ax.containers[0])\n"
        "    plt.show()\n"
        "else:\n"
        "    print(\"No data to plot.\")"
    ))

    # --- CELL 14: Chart 2 (Markdown) ---
    nb.cells.append(nbf.v4.new_markdown_cell("### 5.2 Speaker Distribution in Final Stage"))
    
    # --- CELL 15: Chart 2 (Code) ---
    nb.cells.append(nbf.v4.new_code_cell(
        "if not full_df.empty:\n"
        "    final_df = full_df[full_df.stage == 4]\n"
        "    if not final_df.empty:\n"
        "        plt.figure(figsize=(12, 8))\n"
        "        top_speakers = final_df['speaker'].value_counts().nlargest(20).index\n"
        "        speaker_df = final_df[final_df['speaker'].isin(top_speakers)]\n"
        "        ax = sns.countplot(y=speaker_df['speaker'], order=top_speakers, hue=speaker_df['speaker'], legend=False)\n"
        "        ax.set_title('stage4_reranker: Final Quote Count per Speaker (Top 20)')\n"
        "        ax.set_xlabel('Number of Quotes')\n"
        "        ax.set_ylabel('Speaker')\n"
        "        plt.tight_layout()\n"
        "        plt.show()\n"
        "    else:\n"
        "        print(\"No quotes survived to the final stage.\")\n"
        "else:\n"
        "    print(\"No data to plot.\")"
    ))

    # --- CELL 16: Chart 3 (Markdown) ---
    nb.cells.append(nbf.v4.new_markdown_cell("### 5.3 Score Distribution in Final Stage"))

    # --- CELL 17: Chart 3 (Code) ---
    nb.cells.append(nbf.v4.new_code_cell(
        "if not full_df.empty:\n"
        "    final_df_with_scores = full_df[(full_df.stage == 4) & (full_df.score >= 0)]\n"
        "    if not final_df_with_scores.empty:\n"
        "        plt.figure(figsize=(10, 6))\n"
        "        ax = sns.histplot(final_df_with_scores['score'], bins=20, kde=True)\n"
        "        ax.set_title('stage4_reranker: Distribution of Semantic Scores for Final Quotes')\n"
        "        ax.set_xlabel('Similarity Score')\n"
        "        ax.set_ylabel('Frequency')\n"
        "        plt.show()\n"
        "    else:\n"
        "        print(\"No scored quotes in the final stage to plot.\")\n"
        "else:\n"
        "    print(\"No data to plot.\")"
    ))

    # --- CELL 18: Conclusion (Markdown) ---
    nb.cells.append(nbf.v4.new_markdown_cell(
        "## 6. Conclusion\n\n"
        "This notebook provides a detailed, stage-by-stage analysis of the quote extraction pipeline. "
        "The visualizations highlight the filtering effectiveness at each step, from raw text to final, semantically-scored quotes."
    ))

    # --- Write the notebook to a file ---
    notebook_path = Path("notebooks/reports/pipeline_visualization.ipynb")
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    with open(notebook_path, 'w') as f:
        nbf.write(nb, f)
    
    print(f"Successfully created notebook at: {notebook_path}")

if __name__ == "__main__":
    create_notebook() 