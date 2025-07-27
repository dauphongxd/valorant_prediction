"""
Valorant Betting Data Tool - VLR.gg Scraper Version (Final, Robust)

Handles special characters and partial team names.
"""
import requests
from bs4 import BeautifulSoup
import argparse
import logging
import unicodedata  # Import the library for handling special characters
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('selenium').setLevel(logging.WARNING)

# ==============================================================================
# HELPER FUNCTION FOR NORMALIZING NAMES
# ==============================================================================

def normalize_name(name):
    """
    Converts a name to lowercase and removes diacritics (special characters).
    e.g., 'LEVIATÁN' -> 'leviatan'
    """
    # NFD form separates base characters from their accents
    nfkd_form = unicodedata.normalize('NFD', name)
    # Filter out non-spacing marks (the accents)
    ascii_name = ''.join([c for c in nfkd_form if not unicodedata.combining(c)])
    return ascii_name.lower()

# ==============================================================================
# VLR.GG SCRAPING FUNCTIONS
# ==============================================================================

def get_vlr_odds(team1_name, team2_name):
    """
    Uses Selenium to load the page, then scrapes VLR.gg for match odds.
    """
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'}

    logging.info("Starting browser to load JavaScript content from VLR.gg...")
    service = Service(executable_path='./chromedriver.exe')
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--log-level=3')
    options.add_argument(f'user-agent={headers["User-Agent"]}')

    driver = None
    page_source = None
    try:
        driver = webdriver.Chrome(service=service, options=options)
        matches_url = "https://www.vlr.gg/matches"
        driver.get(matches_url)
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, "a.match-item")))
        logging.info("JavaScript content loaded successfully.")
        page_source = driver.page_source
    except Exception as e:
        print(f"\n❌ An error occurred with Selenium: {e}")
        return
    finally:
        if driver:
            driver.quit()
            logging.info("Browser closed.")

    soup = BeautifulSoup(page_source, 'html.parser')

    # --- Part 2: Parse HTML with improved matching logic ---
    all_match_cards = soup.select('a.match-item')
    target_href = None

    # Normalize the user's input once
    norm_user_team1 = normalize_name(team1_name)
    norm_user_team2 = normalize_name(team2_name)

    for link in all_match_cards:
        team_divs = link.find_all('div', class_='match-item-vs-team-name')
        if len(team_divs) >= 2:
            scraped_team1 = team_divs[0].text.strip()
            scraped_team2 = team_divs[1].text.strip()

            # Normalize the scraped names for comparison
            norm_scraped_team1 = normalize_name(scraped_team1)
            norm_scraped_team2 = normalize_name(scraped_team2)

            # *** NEW ROBUST MATCHING LOGIC ***
            # Check if user input is contained within the scraped names, in either order.
            match1 = norm_user_team1 in norm_scraped_team1 and norm_user_team2 in norm_scraped_team2
            match2 = norm_user_team1 in norm_scraped_team2 and norm_user_team2 in norm_scraped_team1

            if match1 or match2:
                target_href = link['href']
                break

    if not target_href:
        print(f"\n❌ Could not find an upcoming match between '{team1_name}' and '{team2_name}'.")
        print("   Please double-check the spelling as it appears on VLR.gg.")
        return

    match_page_url = "https://www.vlr.gg" + target_href
    logging.info(f"Match URL found! Scraping odds from: {match_page_url}")

    # --- Part 3: Scrape the odds (no changes needed here) ---
    try:
        response = requests.get(match_page_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        betting_cards = soup.find_all('a', class_='match-bet-item')

        if not betting_cards:
            print(f"\n🟡 Match page found, but no betting odds are listed on VLR.gg for this match yet.")
            return

        print(f"\n--- ✅ Odds Found for {scraped_team1} vs {scraped_team2} ---") # Use original scraped names for display
        for card in betting_cards:
            try:
                img_src = card.find('img')['src']
                bookmaker = img_src.split('/')[-1].split('.')[0].replace('-', ' ').capitalize()
                teams = [team.text.strip() for team in card.find_all('span', class_='match-bet-item-team')]
                odds = [odd.text.strip() for odd in card.find_all('span', class_='match-bet-item-odds')]

                if len(teams) == 2 and len(odds) == 2:
                    print(f"\nBookmaker: {bookmaker}")
                    print(f"  - {teams[0]}: {odds[0]}")
                    print(f"  - {teams[1]}: {odds[1]}")
            except (AttributeError, IndexError, TypeError) as e:
                logging.warning(f"Skipping a malformed betting card. Error: {e}")
                continue
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to load the specific match page: {e}")

# ==============================================================================
# MAIN EXECUTION & COMMAND-LINE INTERFACE
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valorant Betting Data Tool - VLR.gg Scraper (Final, Robust Version).")
    parser.add_argument('command', choices=['get-odds'])
    parser.add_argument('team1', type=str, help='The first team name (can be partial/simplified).')
    parser.add_argument('team2', type=str, help='The second team name (can be partial/simplified).')
    args = parser.parse_args()

    if args.command == 'get-odds':
        get_vlr_odds(args.team1, args.team2)