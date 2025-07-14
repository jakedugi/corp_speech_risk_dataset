"""
Query templates and query builder for CourtListener API.

- STATUTE_QUERIES: All supported statute search templates.
- build_queries: Expands a statute and optional company CSV into ready-to-use search strings.
"""
from __future__ import annotations
from pathlib import Path
import csv

# ---- 1. Raw templates (exactly what you had) -------------------------------
STATUTE_QUERIES: dict[str, str] = {
    "FTC Section 5": '''(
            "FTC Act"
            OR "Section 5"
            OR "15 U.S.C. § 45"
            OR "15 U.S.C. § 45(a)"
            OR "15 U.S.C. § 45(b)"
            OR "unfair methods of competition"
        )
        AND
        (
            tweet
            OR "Twitter post"
            OR "X post"
            OR "Facebook post"
            OR "Instagram post"
            OR website
            OR blog
            OR TikTok
            OR YouTube
            OR LinkedIn
            OR "press release"
            OR "press statement"
            OR "press conference"
            OR podcast
            OR webinar
            OR "earnings call"
            OR "SEC filing"
            OR advertisement
            OR promotion
            OR marketing
            OR "marketing materials"
            OR "company statement"
            OR "corporate speech"
            OR "internal memo"
            OR "executive statement"
            OR claim
        )
        AND
        (
            deceptive
            OR misleading
            OR fraudulent
            OR "false claim"
            OR "unfair practice"
            OR "unfair methods"
            OR "false advertising"
    )''',
    "FTC Section 12": '("Section 12" OR "15 USC 52" OR "15 U.S.C. § 52") AND (Instagram OR TikTok OR YouTube OR tweet OR website OR "landing page") AND (efficacy OR health OR "performance claim")',
    "Lanham Act § 43(a)": '("Lanham Act" OR "Section 43(a)" OR "15 USC 1125(a)" OR "15 U.S.C. § 1125(a)") AND (Twitter OR "X post" OR influencer OR hashtag OR TikTok OR "sponsored post" OR website) AND ("false advertising" OR "false endorsement" OR misrepresentation)',
    "SEC Rule 10b-5": '("10b-5" OR "Rule 10b-5" OR "17 CFR 240.10b-5") AND ("press release" OR tweet OR blog OR "CEO post" OR Reddit OR Discord) AND ("material misstatement" OR fraud OR "stock price" OR "market manipulation")',
    "SEC Regulation FD": '("Regulation FD" OR "Reg FD" OR "17 CFR 243.100" OR "Rule FD") AND (CEO OR CFO OR executive) AND ("Facebook post" OR tweet OR "LinkedIn post" OR webcast OR blog) AND ("material information" OR disclosure OR "selective disclosure")',
    "NLRA § 8(a)(1)": '("Section 8(a)(1)" OR "29 USC 158(a)(1)" OR "29 U.S.C. § 158(a)(1)") AND (tweet OR "X post" OR Facebook OR website OR memo OR blog) AND (union OR unionize OR collective-bargaining OR organizing) AND (threat OR promise OR coercive)',
    "CFPA UDAAP": '(UDAAP OR "12 USC 5531" OR "12 U.S.C. § 5531" OR "Consumer Financial Protection Act") AND (loan OR fintech OR credit OR "BNPL") AND (website OR app OR tweet OR "Instagram ad" OR webinar) AND (deceptive OR misleading OR unfair)',
    "California § 17200 / 17500": '("Business and Professions Code § 17200" OR "Bus & Prof Code 17200" OR "§ 17500") AND (site OR newsletter OR email OR Instagram OR TikTok OR tweet) AND (claim OR representation OR advertisement) AND (misleading OR deceptive OR untrue)',
    "NY GBL §§ 349–350": '("GBL § 349" OR "General Business Law § 349" OR "GBL § 350") AND (webinar OR "landing page" OR infomercial OR website OR tweet OR Facebook) AND (misleading OR deceptive OR fraud OR "false advertising")',
    "FD&C Act § 331": '("21 USC 331" OR "21 U.S.C. § 331" OR "FD&C Act") AND (marketing OR promo OR blog OR website OR Facebook OR tweet OR "YouTube video") AND (misbranding OR "risk disclosure" OR "omitted risk")'
}

# ---- 2. Public helper ------------------------------------------------------

def build_queries(
    statute: str,
    company_file: Path | None = None,
    chunk_size: int = 10,
) -> list[str]:
    """
    Expand a single statute into one or many *final* CourtListener
    search strings (one per company-name chunk).
    Args:
        statute: Statute name
        company_file: Optional CSV file with company names
        chunk_size: Number of companies per query chunk (default 50)
    """
    base_q = STATUTE_QUERIES[statute].strip()
    if not company_file:
        return [base_q]  # no chunking needed
    with open(company_file, newline="") as fh:
        names = [r["official_name"].strip() for r in csv.DictReader(fh)]
    out: list[str] = []
    for i in range(0, len(names), chunk_size):
        chunk = names[i : i + chunk_size]
        filter_str = "(" + " OR ".join(f'"{n}"' for n in sorted(chunk)) + ")"
        out.append(f"{base_q}\nAND\n{filter_str}")
    return out
