# Improved JSON Handling for disso_prompt_cli.py

This document explains the improvements made to `disso_prompt_cli.py` to automatically handle malformed JSON inputs from LLM outputs.

## Problem

LLM outputs often contain:
- Smart quotes (" " ' ') instead of ASCII quotes
- Code fences (```json ... ```)
- Trailing commas
- Comments (JSON5 features)

These cause JSON parsing failures when trying to hydrate dissertation sections.

## Solution

1. **Added json5 dependency** - Robust JSON parser that handles:
   - Trailing commas
   - Comments (// and /* */)
   - Unquoted keys
   - More flexible syntax

2. **Implemented _sanitize_json_text()** - Preprocessing that:
   - Removes code fences
   - Converts smart quotes to ASCII quotes
   - Handles various Unicode quote characters

3. **Updated paste-final command** - Now:
   - Detects structured JSON outputs
   - Extracts section_markdown correctly
   - Hydrates all fields automatically
   - Generates clean LaTeX

## Installation

```bash
# The json5 package is already added to pyproject.toml
uv sync
```

## Usage

### Basic Workflow

```bash
# 1. Save your LLM output to a file (even with smart quotes!)
cat > output_3_1_1.json << 'EOF'
{
"section_markdown": "### 3.1.1 Data sources...",
"figure_placeholders": ["Fig. X: Source and provenance flow..."],
"todo_list": ["Fill the exact collection window dates..."],
"evidence_map": [...],
...
}
EOF

# 2. Paste it with ALL fields hydrated automatically + LaTeX generation
uv run python src/corp_speech_risk_dataset/disso_prompt/disso_prompt_cli.py \
  paste-final \
  --csv src/corp_speech_risk_dataset/disso_prompt/disso_sections.csv \
  --id 3.1.1 \
  --from-file output_3_1_1.json \
  --store-latex \
  --mark-done

# 3. Export the LaTeX version
uv run python src/corp_speech_risk_dataset/disso_prompt/disso_prompt_cli.py \
  export \
  --csv src/corp_speech_risk_dataset/disso_prompt/disso_sections.csv \
  --id 3.1.1 \
  --format latex \
  --out section_3_1_1.tex
```

### What Gets Hydrated

From a structured JSON output with these fields:
- `section_markdown` → stored as final_output_markdown
- `figure_placeholders` → converted to figures_plan_json
- `table_placeholders` → converted to tables_plan_json
- `todo_list` → stored as references_needed_queries_json
- `transition_to_next` → stored as transition_to_next_summary
- `evidence_map` → appended to notes
- `bib_entries` → appended to notes
- Full JSON → stored as final_output_json

## Technical Details

### JSON Sanitization

The `_sanitize_json_text()` function handles:

```python
# Remove code fences
text = re.sub(r'^\s*```(?:json)?\s*|\s*```\s*$', '', text, flags=re.DOTALL)

# Normalize smart quotes
text = text.replace("\u201c", '"').replace("\u201d", '"')  # " and "
text = text.replace("\u2018", "'").replace("\u2019", "'")  # ' and '
```

### Robust Parsing

```python
# Clean and parse with json5
cleaned_content = _sanitize_json_text(raw_content)
structured_data = json5.loads(cleaned_content)

# Extract markdown for conversion
if isinstance(structured_data, dict) and "section_markdown" in structured_data:
    md_text = str(structured_data["section_markdown"])
    _hydrate_structured_output_fields(row, structured_data)
```

## Benefits

1. **No manual cleanup needed** - Paste LLM outputs directly
2. **Automatic field mapping** - All structured fields are preserved
3. **Clean LaTeX output** - Only markdown content is converted
4. **Error resilience** - Graceful fallbacks for parsing failures
5. **Unicode handling** - Proper escape sequences avoid encoding issues

## Example

Input (with smart quotes and code fences):
```
```json
{
"section_markdown": "### 3.1.1 Data sources...",
"figure_placeholders": ["Fig. X: "Smart quotes" example"],
"todo_list": ["Task with 'single quotes'"],
}
```
```

Output:
- Markdown correctly extracted
- LaTeX properly generated
- All fields hydrated in CSV
- No manual JSON fixing required!
