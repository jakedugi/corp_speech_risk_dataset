#!/usr/bin/env python3
"""
Manual year mapping for cases without extractable years.
Filled in based on case ID pattern analysis.
"""

case_year_mapping = {
    "24-60040_ca5": 2024,  # 396 records - 5th Circuit Court of Appeals case from 2024
    "24-10951_ca5": 2024,  # 59 records - 5th Circuit Court of Appeals case from 2024
    # Removed bankruptcy cases: 09-11435_nysb, 17-00276_paeb, 15-10116_ksb
}

# Court code meanings:
# nysb = New York Southern District Bankruptcy Court
# ca5 = 5th Circuit Court of Appeals
# paeb = Pennsylvania Eastern District Bankruptcy Court
# ksb = Kansas Bankruptcy Court
