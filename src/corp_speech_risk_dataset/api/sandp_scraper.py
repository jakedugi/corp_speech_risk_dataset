#!/usr/bin/env python3
import logging
from pathlib import Path
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
import time
from requests.adapters import HTTPAdapter
from requests.exceptions import HTTPError
import re
from collections import defaultdict, deque
from functools import wraps, lru_cache
from urllib3.util.retry import Retry
import pandas as pd
import polars as pl
import requests
from bs4 import BeautifulSoup
import warnings
from bs4 import XMLParsedAsHTMLWarning

# Token bucket: max 10 calls per 1 second window
_calls = deque(maxlen=10)

def sec_rate_limited(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        now = time.monotonic()
        if len(_calls) == 10 and now - _calls[0] < 1:
            time.sleep(1 - (now - _calls[0]))
        result = fn(*args, **kwargs)
        _calls.append(time.monotonic())
        return result
    return wrapper

# ────────── Config ──────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

DATA_DIR      = Path(__file__).parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)
BASE_CSV      = DATA_DIR / "sp500_aliases.csv"
OUT_CSV       = DATA_DIR / "sp500_aliases_enriched.csv"

WIKI_LIST     = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
PAGE_BASE     = "https://en.wikipedia.org" # base for urljoin
EDGAR_INDEX   = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
EDGAR_BROWSE_JSON = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=DEF+14A&count=1&owner=exclude&output=json"
EDGAR_BROWSE_ATOM = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=DEF+14A&count=1&owner=exclude&output=atom"
EDGAR_ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/def14a.htm"

HEADERS       = {"User-Agent": "jake@jakedugan.com"}
MAX_PEOPLE    = 100

# any of these words in the <th> of an infobox row will trigger a people scrape
ROLE_KEYWORDS = {
    "people", "key people", "leadership", "management", "executive", 
    "governing", "board", "director", "officer", "team"
}

# proxy filter for historical DEF 14A table rows
SPX_TITLES    = {
    # C-Suite
    r"\\bchief executive officer\\b",      # CEO
    r"\\bceo\\b",
    r"\\bchief financial officer\\b",      # CFO
    r"\\bcfo\\b",
    r"\\bchief operating officer\\b",      # COO
    r"\\bcoo\\b",
    r"\\bchief legal officer\\b",          # CLO / General Counsel
    r"\\bclo\\b",
    r"\\bgeneral counsel\\b",
    r"\\btreasurer\\b",

    # Board leadership
    r"\\bchair(man|person)?\\b",           # Chair / Chairman / Chairperson
    r"\\bboard director\\b",               # Director
    r"\\bdirector\\b",
}

# Robust HTTP Session with Retries & Backoff
session = requests.Session()
session.headers.update(HEADERS)
retry_strategy = Retry(
    total=5,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "HEAD"]
)
adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
session.mount("https://data.sec.gov", adapter)
session.mount("https://www.sec.gov", adapter)

# increase pool size for wikipedia
wiki_adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
session.mount("https://en.wikipedia.org", wiki_adapter)

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

HEADING_REGEX = re.compile(
    r"""(?xi)
    (
      executive\s+officers
      | directors\s+and\s+executive\s+officers
      | board\s+of\s+directors
      | key\s+people
      | leadership\s+team
      | management\s+team
      | corporate\s+officers?
      | section\s+16\s+officers?
      | named\s+executive\s+officers?
      | [""]?officers[""]?
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE
)

@sec_rate_limited
def safe_get_json(url):
    resp = session.get(url, timeout=10)
    if resp.status_code != 200 or "application/json" not in resp.headers.get("Content-Type", ""):
        logging.warning(f"Non-JSON or bad status for {url}: {resp.status_code}")
        return None
    try:
        return resp.json()
    except ValueError:
        logging.warning(f"Invalid JSON body for {url}")
        return None

@sec_rate_limited
def safe_head_html(url):
    try:
        head = session.head(url, timeout=5, allow_redirects=True)
        if head.status_code == 200 and "html" in head.headers.get("Content-Type", ""):
            return True
        logging.warning(f"HEAD check failed for {url}: status {head.status_code}, content-type {head.headers.get('Content-Type')}")
        return False
    except requests.exceptions.RequestException as e:
        logging.warning(f"HEAD request failed for {url}: {e}")
        return False

# ────────── Step 1: fetch base S&P table ──────────
def fetch_sp500_with_links():
    """Fetches S&P 500 list with direct Wikipedia links and CIKs."""
    logging.info(f"Fetching S&P 500 list from {WIKI_LIST}")
    r = session.get(WIKI_LIST, timeout=10)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "lxml")
    table = soup.find("table", {"id": "constituents"})
    rows = table.find_all("tr")[1:]
    records = []
    for row in rows:
        cells = row.find_all("td")
        if len(cells) > 6:
            ticker = cells[0].text.strip()
            link = cells[1].find("a")["href"]
            name = cells[1].get_text(" ", strip=True)
            cik = cells[6].text.strip()
            url = urllib.parse.urljoin(PAGE_BASE, link)
            records.append({"ticker": ticker, "official_name": name, "wiki_url": url, "cik": cik})
    df = pl.DataFrame(records)
    df.write_csv(BASE_CSV)
    logging.info(f"→ wrote base CSV with direct wiki links: {BASE_CSV}")

fetch_sp500_with_links()


# ────────── Helper: scrape Wikipedia infobox ──────────
def fetch_wiki_people(wiki_url: str) -> list[str]:
    """Fetches 'Key people' from a company's Wikipedia page URL."""
    logging.debug(f"[WIKI] GET {wiki_url}")
    try:
        time.sleep(1.0)
        r = session.get(wiki_url, timeout=5)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")
        inf = soup.select_one("table.infobox.vcard")
        if not inf:
            logging.warning(f"[WIKI] no infobox for {wiki_url}")
            return []

        for tr in inf.find_all("tr"):
            th = tr.find("th")
            td = tr.find("td")
            if not th or not td:
                continue
            heading = th.get_text(" ", strip=True).lower()
            if any(keyword in heading for keyword in ROLE_KEYWORDS):
                # Try list items first
                items = [li.get_text(" ", strip=True) for li in td.find_all("li")]
                if not items:
                    # fallback on pipes
                    raw = td.get_text(separator="|")
                    items = [x.strip() for x in raw.split("|") if x.strip()]
                logging.info(f"[WIKI] {wiki_url}: found {len(items)} under '{heading}'")
                return items[:MAX_PEOPLE]

        logging.debug(f"[WIKI] {wiki_url}: no matching people row")
        return []
    except Exception as e:
        logging.warning(f"[WIKI] Could not fetch/parse {wiki_url}: {e}")
        return []


# ────────── Helper: fetch latest DEF 14A accession ──────────
@sec_rate_limited
def fetch_submissions(cik: str) -> dict | None:
    """Return the 'recent' filings dict (including primaryDocument)."""
    url = EDGAR_INDEX.format(cik=cik)
    resp = session.get(url, timeout=10)
    if resp.status_code != 200 or "application/json" not in resp.headers.get("Content-Type",""):
        logging.warning(f"Bad submissions JSON for CIK={cik}: {resp.status_code}")
        return None
    try:
        return resp.json().get("filings", {}).get("recent", {})
    except ValueError:
        logging.warning(f"Invalid JSON at {url}")
        return None

@lru_cache(maxsize=None)
def fetch_def14a_filename(cik: str, acc: str) -> str | None:
    acc_clean = acc.replace("-", "")
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_clean}/{acc}-index.htm"
    try:
        resp = session.get(index_url, timeout=10)
        if resp.status_code != 200:
            logging.warning(f"Index page missing for CIK={cik}, ACC={acc}")
            return None
        soup = BeautifulSoup(resp.text, "lxml")
        tbl = soup.find("table", class_="tableFile")
        if not tbl:
            logging.warning(f"No tableFile on index page for CIK={cik}")
            return None
        for row in tbl.find_all("tr")[1:]:
            cols = [td.get_text(strip=True) for td in row.find_all("td")]
            if len(cols) >= 4 and cols[3] == "DEF 14A":
                href = row.find("a", href=True)["href"]
                return href.split("/")[-1]
    except Exception as e:
        logging.warning(f"Failed to parse index page for CIK={cik}, ACC={acc}: {e}")
    return None

def get_latest_def14a_url(cik: str) -> str | None:
    recent = fetch_submissions(cik)
    if not recent:
        return None
    for form, acc, doc in zip(
        recent.get("form", []),
        recent.get("accessionNumber", []),
        recent.get("primaryDocument", [])
    ):
        if form.upper() == "DEF 14A":
            acc_no_dash = acc.replace("-", "")
            if doc:
                return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_dash}/{doc}"
            # fallback: try to parse the index page for the filename
            filename = fetch_def14a_filename(cik, acc)
            if filename:
                return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_dash}/{filename}"
            else:
                logging.warning(f"No DEF 14A filename found for CIK={cik}, ACC={acc}")
                return None
    return None

@sec_rate_limited
def fetch_browse_json(cik):
    url = EDGAR_BROWSE_JSON.format(cik=int(cik))
    data = safe_get_json(url)
    if not data: return None
    for form, acc in zip(data["filings"]["recent"]["form"], data["filings"]["recent"]["accessionNumber"]):
        if form.upper() == "DEF 14A":
            return acc # Return with dashes
    return None

@sec_rate_limited
def fetch_browse_atom(cik: str) -> str | None:
    url = EDGAR_BROWSE_ATOM.format(cik=int(cik))
    try:
        resp = session.get(url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "xml")
        entry = soup.find("entry")
        if entry:
            href_tag = entry.find("link", {"rel":"alternate"})
            if href_tag and href_tag.has_attr('href'):
                href = href_tag["href"]
                return href.split("/")[-2]  # Returns accession with dashes
    except Exception as e:
        logging.warning(f"[EDGAR] ATOM call failed for {cik}: {e}")
    return None

@lru_cache(maxsize=None)
def get_accession(cik: str) -> str | None:
    # Priority: submissions -> browse JSON -> ATOM
    for fn in (fetch_submissions, fetch_browse_json, fetch_browse_atom):
        acc = fn(cik)
        if acc:
            logging.info(f"Accession {acc} for CIK={cik} via {fn.__name__}")
            return acc
    logging.error(f"No DEF 14A accession for CIK={cik}")
    return None

@sec_rate_limited
def fetch_filing_via_index(cik, acc):
    acc_no_dash = acc.replace("-", "")
    idx_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_no_dash}/{acc}-index.htm"
    try:
        resp = session.get(idx_url, timeout=10)
        if resp.status_code != 200:
            logging.warning(f"No index page at {idx_url}: {resp.status_code}")
            return None
        soup = BeautifulSoup(resp.text, "lxml")
        row = soup.find("tr", lambda t: t and t.find("td", string=re.compile(r"DEF\s*14A", re.I)))
        if row:
            doc_link = row.find("a", href=True)["href"]
            return urllib.parse.urljoin("https://www.sec.gov", doc_link)
    except Exception as e:
        logging.warning(f"Failed to parse index page {idx_url}: {e}")
    return None


# ────────── Helper: scrape proxy table for names & titles ──────────
def fallback_text_search(soup):
    text = soup.get_text(separator="\n")
    matches = HEADING_REGEX.split(text)
    if len(matches) >= 3:
        block = matches[2]
        # Try to extract lines like "Name — Title" or bullet points
        found = re.findall(r"[•\-\*]?\s*([A-Z][a-zA-Z\s\.'-]+)\s+[\u2014\-]+\s+([A-Za-z ,]+)", block)
        return [f"{name} — {title}" for name, title in found]
    return []

def fetch_edgar_people(cik: str) -> list[str]:
    doc_url = get_latest_def14a_url(cik)
    if not doc_url:
        logging.warning(f"No DEF 14A found for CIK={cik}")
        return []
    try:
        resp = session.get(doc_url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        tbl = soup.find(lambda t: t.name=="table" and "Name and Principal Occupation" in t.get_text(" ", True))
        if not tbl:
            logging.warning(f"No principal-occupation table at {doc_url}, trying fallback text search.")
            # fallback: try to extract officer blocks from text
            fallback_people = fallback_text_search(soup)
            if fallback_people:
                logging.info(f"Fallback text search found {len(fallback_people)} officers for CIK={cik}")
                return fallback_people[:MAX_PEOPLE]
            else:
                logging.warning(f"Fallback text search found no officers for CIK={cik}")
                return []
        title_map = defaultdict(list)
        MAX_PER_TITLE = 5
        for row in tbl.find_all("tr")[1:]:
            cols = [td.get_text(" ", strip=True) for td in row.find_all(["th","td"])]
            if not cols: continue
            name, title = cols[0], cols[-1].lower()
            for pat in SPX_TITLES:
                if re.search(pat, title):
                    if len(title_map[pat]) < MAX_PER_TITLE and name not in title_map[pat]:
                        title_map[pat].append(name)
                    break
        out = []
        for names in title_map.values():
            out.extend(names)
            if len(out) >= MAX_PEOPLE:
                return out[:MAX_PEOPLE]
        logging.info(f"CIK={cik}: scraped {len(out)} officers")
        return out
    except Exception as e:
        logging.error(f"Failed to scrape EDGAR people for CIK={cik}, URL={doc_url}: {e}")
        return []

def is_binding_officer(title: str) -> bool:
    """Check if a title corresponds to a legally binding officer/director."""
    t = title.lower()
    for pat in SPX_TITLES:
        if re.search(pat, t):
            return True
    return False


# ────────── Step 3: parallel enrich & write ──────────
if __name__ == "__main__":
    df = pl.read_csv(BASE_CSV).to_pandas()  # small, okay
    with ThreadPoolExecutor(max_workers=10) as exe:
        wiki_futs  = {row['wiki_url']: exe.submit(fetch_wiki_people, row['wiki_url']) for _, row in df.iterrows()}
        edgar_futs = {c: exe.submit(fetch_edgar_people, c) for c in df.cik}

    # assemble records
    records = []
    for _, row in df.iterrows():
        wiki_list  = wiki_futs[row['wiki_url']].result()
        edgar_list = edgar_futs[row['cik']].result()
        # merge & dedupe, wiki first
        merged = wiki_list + [x for x in edgar_list if x not in wiki_list]

        rec = {
            "ticker":        row['ticker'],
            "official_name": row['official_name'],
            "cik":           row['cik'],
        }
        for i, person in enumerate(merged[:MAX_PEOPLE], start=1):
            rec[f"exec{i}"] = person.strip()
        records.append(rec)

    out_df = pl.DataFrame(records)
    out_df.write_csv(OUT_CSV)
    logging.info(f"→ wrote enriched CSV: {OUT_CSV}")