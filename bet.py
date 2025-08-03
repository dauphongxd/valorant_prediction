import logging
import unicodedata
from selenium import webdriver
# --- NEW IMPORTS ---
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
# --- END NEW IMPORTS ---
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import requests


def normalize_name(name):
    """Converts a name to lowercase and removes diacritics for matching."""
    nfkd_form = unicodedata.normalize('NFD', name)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)]).lower()


def scrape_upcoming_matches_list():
    """Scrapes the main vlr.gg/matches page to get a list of all upcoming matches."""
    service = ChromeService(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--log-level=3')
    options.add_argument('--disable-gpu')
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
        return None
    finally:
        if driver:
            driver.quit()

    soup = BeautifulSoup(page_source, 'html.parser')
    all_match_cards = soup.select('a.match-item')
    scraped_matches = []

    for link in all_match_cards:
        if len(link.find_all('div', class_='match-item-vs-team-name')) < 2:
            continue
        team1 = link.find_all('div', class_='match-item-vs-team-name')[0].text.strip()
        team2 = link.find_all('div', class_='match-item-vs-team-name')[1].text.strip()
        match_time = link.find('div', class_='match-item-time').text.strip()
        status = link.find('div', class_='ml-status').text.strip()
        url = "https://www.vlr.gg" + link['href']
        scraped_matches.append({
            'team1_name': team1,
            'team2_name': team2,
            'match_time': match_time,
            'status': status,
            'vlr_url': url
        })
    return scraped_matches


def scrape_results_page():
    """Scrapes the main vlr.gg/matches page to get a list of all upcoming matches."""
    service = ChromeService(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--log-level=3')
    options.add_argument('--disable-gpu')
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36')

    driver = None
    try:
        driver = webdriver.Chrome(service=service, options=options)
        driver.get("https://www.vlr.gg/matches/results")
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.CSS_SELECTOR, "a.match-item")))
        page_source = driver.page_source
    except Exception as e:
        logging.error(f"Selenium failed to load VLR results page: {e}")
        return None
    finally:
        if driver:
            driver.quit()

    soup = BeautifulSoup(page_source, 'html.parser')
    all_match_cards = soup.select('a.match-item')
    completed_match_urls = []

    for link in all_match_cards:
        if 'href' in link.attrs:
            completed_match_urls.append("https://www.vlr.gg" + link['href'])

    return completed_match_urls


def scrape_match_page_odds(match_url: str):
    """
    Scrapes odds using a simple, robust positional method. It assumes the
    left-side odds on a card belong to the left-side team in the header.
    This eliminates all fragile name-matching logic.
    """
    try:
        response = requests.get(match_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        betting_cards = soup.find_all('a', class_='match-bet-item')
        if not betting_cards:
            return []

        all_odds_data = []
        for card in betting_cards:
            try:
                # --- THIS IS THE NEW, SIMPLIFIED LOGIC ---
                odds_elements = card.select('.match-bet-item-odds')
                if len(odds_elements) != 2:
                    continue

                # Trust the visual order: odds[0] is for team1, odds[1] is for team2.
                team1_odds = float(odds_elements[0].text.strip())
                team2_odds = float(odds_elements[1].text.strip())

                img_tag = card.select_one('img')
                bookmaker = img_tag.get('alt', 'Unknown Bookmaker') if img_tag else 'Unknown Bookmaker'

                all_odds_data.append({
                    'bookmaker': bookmaker,
                    'team1_odds': team1_odds,
                    'team2_odds': team2_odds
                })
                # ---------------------------------------------
            except Exception as e:
                logging.warning(f"Skipping a card on {match_url} due to parsing error: {e}")
                continue

        return all_odds_data

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logging.warning(f"Match page not found (404) for {match_url}. It may have been canceled.")
            return "404_NOT_FOUND"  # Return our special signal
        else:
            logging.error(f"HTTP error when scraping {match_url}: {e}")
            return None  # Return None for other HTTP errors (e.g., 503)

    except Exception as e:
        logging.error(f"Failed to scrape specific match page {match_url}: {e}")
        return None


def scrape_match_winner(match_url: str):
    """
    Scrapes a completed match page on VLR.gg to find the winning team.
    This version is resilient and uses two methods to find the winner.
    """
    try:
        response = requests.get(match_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Method 1: The most reliable way (if it exists)
        winner_element = soup.select_one('div.match-header-vs-team.mod-win .wf-title-med')
        if winner_element:
            winner_name = winner_element.text.strip()
            return winner_name

        # Method 2: Fallback for pages that don't use 'mod-win'
        team_name_elements = soup.select('div.match-header-link-name .wf-title-med')
        score_elements = soup.select(
            'div.match-header-vs-score .js-spoiler span.match-header-vs-score-winner, div.match-header-vs-score .js-spoiler span.match-header-vs-score-loser')

        if len(team_name_elements) < 2 or len(score_elements) < 2:
            return None

        team1_name = team_name_elements[0].text.strip()
        team2_name = team_name_elements[1].text.strip()

        score1_classes = score_elements[0].get('class', [])

        if 'match-header-vs-score-winner' in score1_classes:
            return team1_name
        else:  # If the first team isn't the winner, the second must be.
            return team2_name

    except Exception as e:
        logging.error(f"An unexpected error occurred while scraping winner from {match_url}: {e}")
        return None