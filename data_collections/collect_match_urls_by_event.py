# collect_match_urls_by_event.py

import requests
from bs4 import BeautifulSoup
import re
import json

BASE_URL = "https://www.vlr.gg"
VCT_URL = f"{BASE_URL}/vct-2025"
HEADERS = {"User-Agent": "Mozilla/5.0"}
OUTPUT_JSON = "vct25_match_urls.json"


def fetch_region_events():
    """
    Return a list of every region-specific event URL under the
    "completed events" section of /vct-2025.
    """
    resp = requests.get(VCT_URL, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")

    events = set()

    # Find all the main column containers for events on the page
    event_columns = soup.find_all("div", class_="events-container-col")

    for column in event_columns:
        # Check if this column is the "completed events" column.
        # We use a CSS selector because it's more robust.
        # It finds a div that has BOTH 'wf-label' and 'mod-completed' classes,
        # ignoring any other classes like 'mod-large' that might be present.
        header = column.select_one("div.wf-label.mod-completed")

        # If the header exists and contains the correct text, process this column
        if header and "completed events" in header.text.lower():
            # Find all event links within this specific column
            for a in column.find_all("a", href=re.compile(r"^/event/\d+/")):
                href = a["href"]
                # We only want the main event link, not the /matches/ sub-link
                if "/matches/" not in href:
                    events.add(BASE_URL + href)
            # Since we found the completed events column, we can stop searching
            break

    if not events:
        print("⚠️  Warning: Could not find any completed events. The page structure may have changed.")

    return sorted(list(events))


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
    if not region_events:
        print("No completed events found to scrape. Exiting.")
        return

    print(f"Found {len(region_events)} completed region events, scraping matches…")

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
        try:
            matches = fetch_all_matches(all_matches_url)
            print(f"{len(matches)} matches")
            all_match_urls.update(matches)
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error {e.response.status_code} for {all_matches_url}. Skipping.")

    # write to JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(sorted(list(all_match_urls)), f, indent=2)

    print(f"\n✅  Saved {len(all_match_urls)} unique match URLs to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()