import time
import requests
import pandas as pd

BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

# Keep the query focused enough to avoid pulling "everything",
# but broad enough to get a few hundred relevant abstracts.
DEFAULT_QUERY = """
(BBB OR "blood brain barrier" OR LRP1 OR "receptor mediated transcytosis"
OR "brain delivery" OR "central nervous system delivery")
AND PUB_TYPE:"Review"
"""


def fetch_page(query: str, page: int = 1, page_size: int = 100) -> dict:
    """
    Fetch one page of results from Europe PMC.

    Key point:
    - Use a stable sort (FIRST_PDATE_D) so page=1,2,3,... returns consistent, non-random pages.
    """
    params = {
        "query": query,
        "format": "json",
        "pageSize": page_size,
        "page": page,
        "resultType": "core",
    }
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def parse_results(payload: dict) -> list:
    """
    Convert the Europe PMC JSON payload into a list of row dicts.
    We keep only papers with:
    - title
    - abstract text
    """
    results = payload.get("resultList", {}).get("result", [])
    rows = []

    for r in results:
        title = r.get("title")
        abstract = r.get("abstractText") or r.get("abstract")

        if not title or not abstract:
            continue

        if r.get("pubType") and "review" not in str(r.get("pubType")).lower():
            continue

        rows.append({
            "id": r.get("pmid") or r.get("id"),
            "pmcid": r.get("pmcid"),
            "title": title.strip(),
            "abstract": abstract.strip(),
            "year": r.get("pubYear"),
            "journal": r.get("journalTitle"),
            "source": r.get("source"),
            "is_open_access": r.get("isOpenAccess"),
            "has_full_text": r.get("hasFullText"),
            "full_text_url": r.get("fullTextUrlList"),
        })

    return rows


def build_corpus_by_year_slices(
    base_query: str,
    slices: list,
    page_size: int = 200,
    sleep_s: float = 0.3,
    min_abstract_len: int = 200
) -> pd.DataFrame:
    all_rows = []

    for (y0, y1) in slices:
        q = year_query(base_query, y0, y1)

        payload = fetch_page(query=q, page=1, page_size=page_size)
        results = payload.get("resultList", {}).get("result", [])
        rows = parse_results(payload)
        all_rows.extend(rows)

        print(f"years {y0}-{y1}: returned={len(results)} kept={len(rows)} total_kept={len(all_rows)}")
        time.sleep(sleep_s)

    df = pd.DataFrame(all_rows)
    if len(df) == 0:
        return df

    df = df.drop_duplicates(subset=["id"])
    df = df.drop_duplicates(subset=["title"])
    df = df[df["abstract"].str.len() >= min_abstract_len].copy()
    df["text"] = df["title"] + "\n\n" + df["abstract"]
    df = df.reset_index(drop=True)
    return df


def year_query(base_query: str, year_from: int, year_to: int) -> str:
    return f"({base_query}) AND (PUB_YEAR:[{year_from} TO {year_to}])"


def summarize_full_text_availability(df: pd.DataFrame) -> None:
    print("\n=== FULL TEXT AVAILABILITY ===")
    print("Total documents:", len(df))

    if "has_full_text" in df.columns:
        print("has_full_text counts:")
        print(df["has_full_text"].value_counts(dropna=False))

    if "is_open_access" in df.columns:
        print("\nis_open_access counts:")
        print(df["is_open_access"].value_counts(dropna=False))

    if {"has_full_text", "is_open_access"}.issubset(df.columns):
        mask = (df["has_full_text"] == "Y") | (df["is_open_access"] == "Y")
        print("\nPotential full-text candidates:", int(mask.sum()))


def main():

    base = DEFAULT_QUERY

    slices = [
        (2020, 2026),
        (2013, 2019),
    ]

    df = build_corpus_by_year_slices(
        base,
        slices,
        page_size=200,
        sleep_s=0.3,
    )

    out_path = "data/abstracts.csv"
    df.to_csv(out_path, index=False)

    summarize_full_text_availability(df)

    print(f"\nsaved: {out_path} ({len(df)} docs)")
    if len(df) > 0:
        print(df.head(3)[["id", "year", "title", "has_full_text", "is_open_access"]])
    else:
        print("No documents fetched. Try adjusting the query or year range.")


if __name__ == "__main__":
    main()