"""
Script to clean and normalize S&P 500 official names for legal risk dataset.
- Keeps only 'official_name' column
- Drops duplicates
- For names containing (Class X), (The), any quotes, or commas, appends a cleaned version
- Normalizes whitespace
- Alphabetizes and saves to new CSV
"""

import re
import pandas as pd

# 1. Load the data
df = pd.read_csv("data/sp500_officers_cleaned.csv")

# 2. Keep only the 'official_name' column
df = df[["official_name"]]

# 3. Drop duplicate names
df = df.drop_duplicates()

# 4. Define regex for patterns to clean:
#    - (Class A), (Class B), ...
#    - (The) (with capital T)
#    - any straight or curly quotes
#    - commas
pattern = re.compile(
    r"(\(\s*Class\s+[A-Z]\s*\)"  # (Class A), (Class B), etc.
    r"|\(The\)"  # (The) with capital T
    r'|["""]'  # any straight or curly quotes
    r"|,)"  # commas
)

# 5. For each name containing one of these, append a cleaned duplicate
clean_rows = []
for name in df["official_name"]:
    if pattern.search(name):
        cleaned = pattern.sub("", name)  # strip only those patterns
        cleaned = " ".join(cleaned.split())  # collapse any extra spaces
        clean_rows.append(cleaned)

# 6. Append the cleaned entries into the same column
df_final = pd.concat(
    [df, pd.DataFrame({"official_name": clean_rows})], ignore_index=True
)

# 7. Sort the entire column alphabetically
df_final = df_final.sort_values("official_name").reset_index(drop=True)

# 8. Save to a new CSV
df_final.to_csv("data/sp500_official_names_cleaned.csv", index=False)
