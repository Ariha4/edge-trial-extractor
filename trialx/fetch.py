"""Download neurotech / BCI-adjacent trials from the ClinicalTrials.gov v2 API.

Usage:
    python -m trialx.fetch --out data/raw_studies.jsonl --max-studies 4000

Each line of the output file is one raw study record as returned by the API.
Only the standard library is used so this runs anywhere.
"""
import argparse
import json
import time
import urllib.parse
import urllib.request

API_URL = "https://clinicaltrials.gov/api/v2/studies"

# Broad neurotech search terms. Several queries are used because one query
# alone returns too few BCI-specific studies to train on.
SEARCH_TERMS = [
    "brain-computer interface",
    "brain machine interface",
    "neural implant",
    "deep brain stimulation",
    "spinal cord stimulation",
    "transcranial magnetic stimulation",
    "vagus nerve stimulation",
    "electroencephalography",
    "neurofeedback",
    "neuroprosthesis",
]


def fetch_page(search_term, page_token, page_size):
    query_params = {"query.term": search_term, "pageSize": str(page_size), "format": "json"}
    if page_token:
        query_params["pageToken"] = page_token
    request_url = API_URL + "?" + urllib.parse.urlencode(query_params)
    request = urllib.request.Request(request_url, headers={"User-Agent": "edge-trial-extractor/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_studies(max_studies, page_size=500, pause_seconds=0.5):
    studies_by_id = {}
    for search_term in SEARCH_TERMS:
        page_token = None
        while len(studies_by_id) < max_studies:
            page = fetch_page(search_term, page_token, page_size)
            for study in page.get("studies", []):
                nct_id = study.get("protocolSection", {}).get("identificationModule", {}).get("nctId")
                if nct_id and nct_id not in studies_by_id:
                    studies_by_id[nct_id] = study
            page_token = page.get("nextPageToken")
            print(f"{search_term!r}: {len(studies_by_id)} unique studies so far")
            if not page_token:
                break
            time.sleep(pause_seconds)
        if len(studies_by_id) >= max_studies:
            break
    return list(studies_by_id.values())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/raw_studies.jsonl")
    parser.add_argument("--max-studies", type=int, default=4000)
    args = parser.parse_args()

    studies = fetch_studies(args.max_studies)
    with open(args.out, "w", encoding="utf-8") as out_file:
        for study in studies:
            out_file.write(json.dumps(study) + "\n")
    print(f"Wrote {len(studies)} studies to {args.out}")


if __name__ == "__main__":
    main()
