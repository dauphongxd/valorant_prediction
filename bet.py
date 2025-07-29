# bet.py (Refactored to be a reusable and robust module)

import logging
import unicodedata
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import argparse
import requests  # Moved import to top

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('selenium').setLevel(logging.WARNING)


def normalize_name(name):
    """Converts a name to lowercase and removes diacritics."""
    nfkd_form = unicodedata.normalize('NFD', name)
    ascii_name = ''.join([c for c in nfkd_form if not unicodedata.combining(c)])
    return ascii_name.lower()


def get_vlr_odds(team1_name, team2_name):
    """
    Scrapes VLR.gg for match odds and returns the data as a dictionary.
    This version is robust against missing image/alt tags.
    """
    service = Service(executable_path='./chromedriver.exe')
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--log-level=3')
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36')

    driver = None
    try:
        driver = webdriver.Chrome(service=service, options=options)
        driver.get("https://www.vlr.gg/matches")
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, "a.match-item")))
        page_source = driver.page_source
    except Exception as e:
        logging.error(f"Selenium failed to load VLR matches page: {e}")
        return {'error': 'selenium_error', 'message': str(e)}
    finally:
        if driver:
            driver.quit()

    soup = BeautifulSoup(page_source, 'html.parser')
    all_match_cards = soup.select('a.match-item')

    norm_user_team1 = normalize_name(team1_name)
    norm_user_team2 = normalize_name(team2_name)
    found_match_info = None

    for link in all_match_cards:
        team_divs = link.find_all('div', class_='match-item-vs-team-name')
        if len(team_divs) >= 2:
            scraped_team1_full = team_divs[0].text.strip()
            scraped_team2_full = team_divs[1].text.strip()
            if (norm_user_team1 in normalize_name(scraped_team1_full) and norm_user_team2 in normalize_name(
                    scraped_team2_full)) or \
                    (norm_user_team1 in normalize_name(scraped_team2_full) and norm_user_team2 in normalize_name(
                        scraped_team1_full)):
                found_match_info = {'href': "https://www.vlr.gg" + link['href'], 'team1_vlr': scraped_team1_full,
                                    'team2_vlr': scraped_team2_full}
                break

    if not found_match_info:
        return {'error': 'match_not_found'}

    try:
        response = requests.get(found_match_info['href'], headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        betting_cards = soup.find_all('a', class_='match-bet-item')

        if not betting_cards:
            return {'error': 'no_odds_listed', **found_match_info}

        odds_data = []
        for card in betting_cards:
            try:
                teams = [team.text.strip() for team in card.find_all('span', class_='match-bet-item-team')]
                odds = [float(odd.text.strip()) for odd in card.find_all('span', class_='match-bet-item-odds')]

                # --- THIS IS THE FIX ---
                # Safely get the bookmaker name without causing a crash
                img_tag = card.find('img')
                bookmaker_name = img_tag.get('alt', 'Unknown Bookmaker') if img_tag else 'Unknown Bookmaker'
                # -----------------------

                if len(teams) == 2 and len(odds) == 2:
                    if normalize_name(teams[0]) in normalize_name(found_match_info['team1_vlr']):
                        odds_data.append({'bookmaker': bookmaker_name, 'team1_odds': odds[0], 'team2_odds': odds[1]})
                    else:
                        odds_data.append({'bookmaker': bookmaker_name, 'team1_odds': odds[1], 'team2_odds': odds[0]})
            # Added KeyError to the list of caught exceptions for safety
            except (AttributeError, IndexError, TypeError, ValueError, KeyError) as e:
                logging.warning(f"Skipping a malformed betting card. Error: {e}")
                continue

        if not odds_data:
            return {'error': 'no_odds_listed', **found_match_info}

        return {'status': 'success', 'team1_vlr': found_match_info['team1_vlr'],
                'team2_vlr': found_match_info['team2_vlr'], 'odds': odds_data}

    # The exception will now print the actual error from requests or parsing
    except Exception as e:
        logging.error(f"Failed to scrape specific match page: {e}", exc_info=True)  # exc_info gives more detail
        return {'error': 'page_scrape_failed', **found_match_info}


# This part allows the script to still be run from the command line for testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valorant Betting Data Tool - VLR.gg Scraper")
    parser.add_argument('team1', type=str, help='The first team name.')
    parser.add_argument('team2', type=str, help='The second team name.')
    args = parser.parse_args()
    results = get_vlr_odds(args.team1, args.team2)
    print(results)