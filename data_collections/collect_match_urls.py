# collect_all_vct2024_matches_json.py

import requests
from bs4 import BeautifulSoup
import re
import json

BASE_URL    = "https://www.vlr.gg"
VCT_URL     = f"{BASE_URL}/vct-2023"
HEADERS     = {"User-Agent": "Mozilla/5.0"}
OUTPUT_JSON = "vct23_match_urls.json"

def fetch_region_events():
    """Return a list of every region-specific event URL under /vct-2024."""
    resp = requests.get(VCT_URL, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    events = set()
    for a in soup.find_all("a", href=re.compile(r"^/event/\d+/")):
        href = a["href"]
        if "/matches/" not in href:
            events.add(BASE_URL + href)
    return sorted(events)

def fetch_all_matches(all_matches_url):
    """Return all '-vs-' match URLs from the ?series_id=all page."""
    resp = requests.get(all_matches_url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    return {
        BASE_URL + a["href"]
        for a in soup.find_all("a", href=re.compile(r"^/\d+/.+-vs-"))
    }

def main():
    region_events = fetch_region_events()
    print(f"Found {len(region_events)} region events, scraping matches…")

    all_match_urls = set()

    for event_url in region_events:
        # extract ID and slug so we can build the “all” endpoint
        m = re.search(r"/event/(\d+)/(.*)$", event_url)
        if not m:
            print(f"⚠️  Skipping unrecognized URL: {event_url}")
            continue

        event_id, slug = m.group(1), m.group(2)
        all_matches_url = (
            f"{BASE_URL}/event/matches/{event_id}/{slug}/?series_id=all"
        )

        print(f" • {slug} → ", end="", flush=True)
        matches = fetch_all_matches(all_matches_url)
        print(f"{len(matches)} matches")
        all_match_urls.update(matches)

    # write to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(sorted(all_match_urls), f, indent=2)

    print(f"\n✅  Saved {len(all_match_urls)} unique match URLs to {OUTPUT_JSON}")

if __name__ == "__main__":
    main()
