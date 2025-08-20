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

        # --- FINAL, MORE PRECISE SCRAPING LOGIC ---
        best_of_format = 'N/A'
        # This selector specifically targets the divs that contain the time AND the format.
        note_elements = link.select('.match-item-details-note')

        for note in note_elements:
            note_text = note.text.strip().lower()
            if 'bo3' in note_text:
                best_of_format = 'Bo3'
                break
            elif 'bo5' in note_text:
                best_of_format = 'Bo5'
                break
            elif 'bo1' in note_text:
                best_of_format = 'Bo1'
                break
        # --- END OF FINAL LOGIC ---

        team1 = link.find_all('div', class_='match-item-vs-team-name')[0].text.strip()
        team2 = link.find_all('div', class_='match-item-vs-team-name')[1].text.strip()

        # --- NEW CHECK FOR TBD ---
        if team1.lower() == 'tbd' or team2.lower() == 'tbd':
            logging.info(f"Skipping match with TBD opponent: {team1} vs {team2}")
            continue
        # --- END NEW CHECK ---

        match_time = link.find('div', class_='match-item-time').text.strip()
        status = link.find('div', class_='ml-status').text.strip()
        url = "https://www.vlr.gg" + link['href']

        scraped_matches.append({
            'team1_name': team1,
            'team2_name': team2,
            'match_time': match_time,
            'status': status,
            'vlr_url': url,
            'best_of_format': best_of_format
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
    Scrapes odds from a match page. Returns an empty list if the match is
    already completed or if no valid odds are found.
    This version is robust against finished-match pages and bet history cards.
    """
    try:
        response = requests.get(match_url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # --- FIX #1: HIGH-LEVEL CHECK ---
        # If a winner element exists on the page, the match is over.
        # There are no valid odds to scrape, so we return an empty list immediately.
        if soup.select_one('div.match-header-vs-team.mod-win'):
            logging.info(f"Match at {match_url} has already finished. Skipping odds check.")
            return []
        # --- END FIX #1 ---

        betting_cards = soup.find_all('a', class_='match-bet-item')
        if not betting_cards:
            return []

        all_odds_data = []
        for card in betting_cards:
            try:
                # --- FIX #2: DEFENSIVE CHECK IN LOOP ---
                # Bet history cards contain the word "returned". We must skip them.
                if 'returned' in card.get_text(separator=" ", strip=True):
                    continue
                # --- END FIX #2 ---

                odds_elements = card.select('.match-bet-item-odds')
                if len(odds_elements) != 2:
                    continue

                # This handles any other weird text, though the checks above should prevent it.
                odds_text_t1 = odds_elements[0].text.strip().replace('$', '')
                odds_text_t2 = odds_elements[1].text.strip().replace('$', '')

                team1_odds = float(odds_text_t1)
                team2_odds = float(odds_text_t2)

                img_tag = card.select_one('img')
                bookmaker = img_tag.get('alt', 'Unknown Bookmaker') if img_tag else 'Unknown Bookmaker'

                all_odds_data.append({
                    'bookmaker': bookmaker,
                    'team1_odds': team1_odds,
                    'team2_odds': team2_odds
                })
            except (ValueError, IndexError) as e:
                logging.warning(f"Skipping an invalid odds/bet card on {match_url}: {e}")
                continue

        return all_odds_data

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logging.warning(f"Match page not found (404) for {match_url}. It may have been canceled.")
            return "404_NOT_FOUND"
        else:
            logging.error(f"HTTP error when scraping {match_url}: {e}")
            return None

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

        # --- Method 1: The most reliable way (if it exists) ---
        winner_element = soup.select_one('div.match-header-vs-team.mod-win .wf-title-med')
        if winner_element:
            winner_name = winner_element.text.strip()
            # This method is the gold standard, so we can trust it and return immediately.
            return winner_name

        # --- Method 2: The new, more robust fallback logic ---
        team_name_elements = soup.select('div.match-header-link-name .wf-title-med')

        # This selector now specifically finds the container for the scores.
        score_container = soup.select_one('div.match-header-vs-score .js-spoiler')

        # Basic validation to ensure we have the elements we need
        if not score_container or len(team_name_elements) < 2:
            logging.warning(f"Could not find all necessary winner elements on {match_url}")
            return None

        # Get the team names based on their visual position (left and right)
        team1_name = team_name_elements[0].text.strip()  # Left team
        team2_name = team_name_elements[1].text.strip()  # Right team

        # Get all spans directly inside the score container to preserve their order
        all_spans_in_order = score_container.find_all('span', recursive=False)

        # Filter out the middle ':' span to get only the two score spans
        actual_score_spans = [s for s in all_spans_in_order if 'colon' not in s.get('class', [])]

        if len(actual_score_spans) < 2:
            logging.warning(f"Could not find two score spans on {match_url}")
            return None

        score1_span = actual_score_spans[0]  # Left score
        score2_span = actual_score_spans[1]  # Right score

        # Now we perform the unambiguous check
        if 'match-header-vs-score-winner' in score1_span.get('class', []):
            # If the left score has the 'winner' class, the left team won.
            return team1_name

        if 'match-header-vs-score-winner' in score2_span.get('class', []):
            # If the right score has the 'winner' class, the right team won.
            return team2_name

        # As a final check, if we can't find a winner, maybe we can find a loser
        if 'match-header-vs-score-loser' in score2_span.get('class', []):
            # If the right team lost, the left team must have won.
            return team1_name

        if 'match-header-vs-score-loser' in score1_span.get('class', []):
            # If the left team lost, the right team must have won.
            return team2_name

        # If we reach this point, we could not determine a winner.
        logging.error(f"Could not determine a winner on {match_url} using fallback logic.")
        return None

    except Exception as e:
        logging.error(f"An unexpected error occurred while scraping winner from {match_url}: {e}")
        return None