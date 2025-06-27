import pandas as pd
import re

# A helper to pull out every Name (Title) pair
def extract_officers(raw: str) -> list[tuple[str,str]]:
    """Return list of (name, title) found in a messy raw string."""
    if not isinstance(raw, str):
        return []
    # 1. drop any [n] footnotes
    s = re.sub(r"\[\s*\d+\s*\]", "", raw)
    # 2. collapse multi-spaces
    s = re.sub(r"\s+", " ", s).strip()
    # 3. find all Name (Title) patterns
    #    - Name: starts with uppercase letter, then any chars until '('
    #    - Title: anything not ')'
    pattern = r"([A-Z][^()]{2,}?)\s*\(\s*([^)]+?)\s*\)"
    matches = re.findall(pattern, s)
    # 4. clean up trailing commas & ampersands
    cleaned = []
    for name, title in matches:
        name = name.rstrip(" ,&").strip()

        # Handle titles appended with a hyphen, e.g., "John Doe - Chairman"
        if " - " in name:
            name = name.split(" - ", 1)[0]

        # Remove nicknames in quotes, e.g., 'Leonard "Lenny" Debs' -> 'Leonard Debs'
        name = re.sub(r'\s*".*?"\s*', " ", name)

        # Collapse multiple spaces into one and strip
        name = re.sub(r"\s+", " ", name).strip()

        title = title.rstrip(" ,&").strip()
        if name and title: # ensure we don't have empty names or titles
            cleaned.append((name, title))
    return cleaned

# Apply the helper across the DataFrame
# 1. load
df = pd.read_csv("data/sp500_aliases_enriched.csv", dtype=str)

# 2. pick up all the exec columns
exec_cols = [col for col in df.columns if col.startswith("exec")]

# 3. Fill NaNs in exec-cols with empty strings
df[exec_cols] = df[exec_cols].fillna("")

# 4. for each row, stitch together every execN, then extract officers in one shot
def gather_officers(row):
    # now every row[col] is a str (possibly empty), so join will never see a float
    blob = " ".join(row[col] for col in exec_cols)
    return extract_officers(blob)

# 5. build a new column of lists of tuples
df["officers"] = df.apply(gather_officers, axis=1)

# 6. If you want each officer on its own row, explode:
rows = (df
        .explode("officers")                       # one tuple per row
        .dropna(subset=["officers"])
        .assign(
            name  = lambda d: d["officers"].str[0],
            title = lambda d: d["officers"].str[1]
        )
        .drop(columns=exec_cols + ["officers"])
       )

# 7. write out
rows.to_csv("data/sp500_officers_cleaned.csv", index=False) 