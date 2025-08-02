import requests
from bs4 import BeautifulSoup
import json
import time


def scrape_vlr_teams_to_file(file_path='vlr_team_short_names.json'):
    """
    Scrapes team names and their shortened versions from all regional VLR.gg
    ranking pages and saves them to a JSON file.

    Args:
        file_path (str): The name of the output JSON file.

    Returns:
        bool: True if scraping and file writing were successful, False otherwise.
    """
    base_url = "https://www.vlr.gg"
    regions = [
        "europe", "north-america", "brazil", "asia-pacific", "korea", "china",
        "japan", "la-s", "la-n", "oceania", "mena", "gc", "collegiate"
    ]

    team_data = {}
    processed_team_urls = set()  # Use a set to track processed team URLs to avoid duplicates

    # Use headers to mimic a real browser visit
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    print("Starting the scraping process...")

    # Loop through each regional ranking page
    for region in regions:
        rankings_url = f"{base_url}/rankings/{region}"
        print(f"\n--- Scraping region: {region} ---")

        try:
            response = requests.get(rankings_url, headers=headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Find all team links on the current regional ranking page
            # The link is in an <a> tag with class 'rank-item-team'
            team_links = soup.select('a.rank-item-team')

            if not team_links:
                print(f"No teams found for region: {region}")
                continue

            for link in team_links:
                team_url_path = link.get('href')
                if not team_url_path:
                    continue

                team_url = base_url + team_url_path

                # If we've already processed this team, skip it
                if team_url in processed_team_urls:
                    continue

                processed_team_urls.add(team_url)

                try:
                    # Visit the individual team page
                    team_response = requests.get(team_url, headers=headers)
                    team_response.raise_for_status()
                    team_soup = BeautifulSoup(team_response.text, 'html.parser')

                    # Extract the full team name from <h1 class="wf-title">
                    full_name_element = team_soup.find('h1', class_='wf-title')
                    full_name = full_name_element.text.strip() if full_name_element else None

                    # Extract the shortened team name from <h2 class="wf-title team-header-tag">
                    short_name_element = team_soup.find('h2', class_='wf-title team-header-tag')
                    short_name = short_name_element.text.strip() if short_name_element else "N/A"

                    if full_name:
                        team_data[full_name] = short_name
                        print(f"Scraped: {full_name}: {short_name}")
                    else:
                        print(f"Could not find full name for URL: {team_url}")

                    # Be respectful to the server, wait a bit between requests
                    time.sleep(0.5)

                except requests.exceptions.RequestException as e:
                    print(f"Could not process team URL {team_url}: {e}")
                    continue

        except requests.exceptions.RequestException as e:
            print(f"Error fetching rankings page for {region}: {e}")
            continue

    # Write the final dictionary to a JSON file
    try:
        with open(file_path, 'w', encoding='utf-8') as json_file:
            json.dump(team_data, json_file, indent=4, ensure_ascii=False)

        print(f"\nScraping complete. Found {len(team_data)} unique teams.")
        print(f"Data successfully saved to {file_path}")
        return True
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")
        return False


if __name__ == '__main__':
    scrape_vlr_teams_to_file()