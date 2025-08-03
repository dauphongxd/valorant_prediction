# collect_all_match_urls.py

import requests
from bs4 import BeautifulSoup
import re
import json
import os

# --- Configuration ---
BASE_URL = "https://www.vlr.gg"
RESULTS_URL = f"{BASE_URL}/matches/results"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}
OUTPUT_JSON = "vlr_all_match_urls.json"


# --- End Configuration ---


def load_previous_matches(filename):
    """
    Loads a set of previously scraped match URLs from a JSON file.
    Returns an empty set if the file doesn't exist or is invalid.
    """
    if not os.path.exists(filename):
        print("No previous match file found. Starting a fresh scrape.")
        return set()

    try:
        with open(filename, "r", encoding="utf-8") as f:
            # Ensure we handle empty files gracefully
            content = f.read()
            if not content:
                return set()
            urls = json.loads(content)
            print(f"✅ Loaded {len(urls)} previously scraped match URLs.")
            return set(urls)
    except (json.JSONDecodeError, IOError) as e:
        print(f"⚠️  Warning: Could not read or parse {filename}. Starting fresh. Error: {e}")
        return set()


def scrape_page(page_url):
    """
    Scrapes a single page of match results and returns a set of full match URLs.
    """
    try:
        resp = requests.get(page_url, headers=HEADERS, timeout=10)
        resp.raise_for_status()  # Will raise an exception for 4xx/5xx status codes
        soup = BeautifulSoup(resp.content, "html.parser")

        # This regex is key to finding only match URLs, e.g., /23456/team-a-vs-team-b
        match_links = soup.find_all("a", href=re.compile(r"^/\d+/.+-vs-"))

        if not match_links:
            return set()

        return {BASE_URL + a["href"] for a in match_links}

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Error fetching {page_url}: {e}")
        return None  # Return None to indicate a network-level error


def main():
    """
    Main function to drive the scraping process.
    """
    previous_matches = load_previous_matches(OUTPUT_JSON)
    all_found_urls = set()

    page = 1
    stale_page_count = 0

    print("\nScraping pages from vlr.gg/matches/results...")

    while True:
        # Construct the URL for the current page
        url_to_scrape = f"{RESULTS_URL}?page={page}"
        print(f" • Scraping page {page}... ", end="", flush=True)

        found_on_page = scrape_page(url_to_scrape)

        # Handle network errors
        if found_on_page is None:
            print("Stopping due to network error.")
            break

        # If a page is empty, we've reached the end
        if not found_on_page:
            print("No matches found. Reached the last page.")
            break

        # Check if all matches on this page are already in our saved list
        # This is the key optimization for re-runs.
        new_matches_on_page = found_on_page - previous_matches - all_found_urls

        print(f"Found {len(found_on_page)} matches ({len(new_matches_on_page)} new).")

        if not new_matches_on_page and len(found_on_page) > 0:
            stale_page_count += 1
            # If we hit 2 consecutive pages with no new matches, we can be confident
            # that we don't need to go further back in time.
            if stale_page_count >= 2:
                print("\nStopping scrape: Found 2 consecutive pages with no new matches.")
                break
        else:
            # Reset the stale counter if we find a new match
            stale_page_count = 0

        all_found_urls.update(found_on_page)
        page += 1

    # --- Finalizing and Saving ---
    if not all_found_urls and not previous_matches:
        print("\nNo matches were found at all. Exiting.")
        return

    # Combine newly found URLs with the previously saved ones
    combined_urls = previous_matches.union(all_found_urls)

    newly_added_count = len(combined_urls) - len(previous_matches)

    print("\n--- Scrape Summary ---")
    print(f"Previously saved matches: {len(previous_matches)}")
    print(f"Newly found matches: {newly_added_count}")
    print(f"Total unique matches: {len(combined_urls)}")

    # Write the sorted list of unique URLs to the JSON file
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(sorted(list(combined_urls)), f, indent=2, ensure_ascii=False)

    print(f"\n✅  Saved {len(combined_urls)} unique match URLs to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()