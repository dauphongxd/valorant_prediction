import requests
from bs4 import BeautifulSoup
import pandas as pd

# ----------------- CONFIGURATION -----------------
# This is the base URL for the global stats page
BASE_URL = 'https://www.vlr.gg/stats/'

# --- FILTERS ---
# You can change these parameters to filter the data.
# Leave a value as None to use the site's default.
#
# timespan: '90d' (default), '60d', '30d', or 'all' for all time.
# min_rounds: Minimum rounds played to be included. Default is 200.
# region: 'na', 'eu', 'br', 'lan', 'las', 'oce', 'apac', 'kr', 'mn' (MENA), 'cn'
#
PARAMS = {
    'timespan': '90d',  # Options: '30d', '60d', '90d', 'all'
    'min_rounds': '200',  # Set the minimum number of rounds played
    'region': None  # Options: 'na', 'eu', 'br', 'lan', etc. or None for all regions
}


# -------------------------------------------------


def scrape_vlr_stats_page(url, params):
    """
    Scrapes the main player statistics table from a VLR.gg stats page.

    Args:
        url (str): The base URL for the stats page.
        params (dict): A dictionary of query parameters to filter the stats.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the player stats,
                      or None if the table cannot be found.
    """
    headers = {'User-Agent': 'Mozilla/5.0'}

    # Filter out None values from params before making the request
    active_params = {k: v for k, v in params.items() if v is not None}

    try:
        resp = requests.get(url, headers=headers, params=active_params)
        resp.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        print(f"Successfully fetched data from: {resp.url}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the URL: {e}")
        return None

    soup = BeautifulSoup(resp.content, 'html.parser')

    stats_table = soup.select_one('table.wf-table.mod-stats')
    if not stats_table:
        print("Could not find the player stats table on the page.")
        return None

    all_players_data = []
    player_rows = stats_table.select('tbody tr')

    if not player_rows:
        print("No players found with the specified filters.")
        return None

    print(f"Found {len(player_rows)} players to scrape...")

    for row in player_rows:
        cells = row.find_all('td')
        if not cells:
            continue

        # Helper to safely extract text from a cell
        def get_cell_text(cell, selector='span'):
            element = cell.select_one(selector) if selector else cell
            return element.get_text(strip=True) if element else 'N/A'

        agent_imgs = cells[1].select('img')
        agents = ', '.join(sorted([img['title'] for img in agent_imgs if 'title' in img.attrs]))

        player_data = {
            'Player': get_cell_text(cells[0], '.text-of'),
            'Team': get_cell_text(cells[0], '.stats-player-country'),
            'Agents': agents,
            'RND': get_cell_text(cells[2], selector=None),
            'Rating': get_cell_text(cells[3], 'span'),
            'ACS': get_cell_text(cells[4], 'span'),
            'K:D': get_cell_text(cells[5], 'span'),
            'KAST': get_cell_text(cells[6], 'span'),
            'ADR': get_cell_text(cells[7], 'span'),
            'KPR': get_cell_text(cells[8], 'span'),
            'APR': get_cell_text(cells[9], 'span'),
            'FKPR': get_cell_text(cells[10], 'span'),
            'FDPR': get_cell_text(cells[11], 'span'),
            'HS%': get_cell_text(cells[12], 'span'),
            'CL%': get_cell_text(cells[13], 'span'),
            'Clutches': get_cell_text(cells[14], selector=None),
            'KMax': get_cell_text(cells[15], 'a'),
            'K': get_cell_text(cells[16], selector=None),
            'D': get_cell_text(cells[17], selector=None),
            'A': get_cell_text(cells[18], selector=None),
            'FK': get_cell_text(cells[19], selector=None),
            'FD': get_cell_text(cells[20], selector=None),
        }
        all_players_data.append(player_data)

    if not all_players_data:
        print("No player data was extracted.")
        return None

    df = pd.DataFrame(all_players_data)
    return df


if __name__ == "__main__":
    player_stats_df = scrape_vlr_stats_page(BASE_URL, PARAMS)

    if player_stats_df is not None:
        print("\n--- Global Player Statistics ---")
        # Use .to_string() to ensure all columns and rows are printed
        print(player_stats_df.to_string())

        # Optionally, save the data to a CSV file
        try:
            filename = 'vlr_global_player_stats.csv'
            player_stats_df.to_csv(filename, index=False)
            print(f"\nSuccessfully saved data to {filename}")
        except Exception as e:
            print(f"\nCould not save to CSV file. Error: {e}")