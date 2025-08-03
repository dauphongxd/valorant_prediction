import os
import re
import json
import requests
import pandas as pd
from bs4 import BeautifulSoup


def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


def scrape_match(url: str):
    """Return a list of per-player rows with exactly your 17 fields."""
    headers = {'User-Agent': 'Mozilla/5.0'}
    resp = requests.get(url, headers=headers)
    soup = BeautifulSoup(resp.content, 'html.parser')

    match_id = re.findall(r'/(\d+)', url)[0]

    # --- ROBUSTNESS CHECK ---
    # Check for the main score elements first. If they don't exist, this is
    # likely a forfeited or unusual match. Skip it to prevent crashing.
    winner_score_el = soup.select_one('.match-header-vs-score-winner')
    loser_score_el = soup.select_one('.match-header-vs-score-loser')

    if not winner_score_el or not loser_score_el:
        print(f"    -> Skipping match {match_id}: Page is missing score elements (likely a forfeit).")
        return []  # Return an empty list, no data will be processed for this match

    # If the checks pass, continue with the scraping logic
    # --- END OF CHECK ---

    event_name = soup.select_one('.match-header-event > div > div').get_text(strip=True)
    event_stage = clean(soup.select_one('.match-header-event-series').get_text(strip=True))
    date_el = soup.select_one('.match-header-date .moment-tz-convert')
    if date_el and date_el.get('data-utc-ts'):
        match_date = date_el['data-utc-ts'].split(' ')[0]
    else:
        match_date = date_el.get_text(strip=True) if date_el else 'N/A'

    team_a = soup.select_one('.match-header-link.mod-1 .wf-title-med').get_text(strip=True)
    team_b = soup.select_one('.match-header-link.mod-2 .wf-title-med').get_text(strip=True)

    # best-of
    best_of = ''
    for note in soup.select('.match-header-vs-note'):
        if 'Bo' in note.text:
            best_of = note.get_text(strip=True)
            break

    # final match score
    w = winner_score_el.get_text(strip=True)
    l = loser_score_el.get_text(strip=True)
    team_a_score, team_b_score = w, l

    rows = []
    for game in soup.select('.vm-stats-game'):
        if game.get('data-game-id') == 'all':
            continue

        map_name_el = game.select_one('.map > div > span')
        # Handle maps that might be missing a name
        if not map_name_el or not map_name_el.find(text=True, recursive=False):
            continue
        map_name = map_name_el.find(text=True, recursive=False).strip()

        scores = game.select('.score')
        map_score_a = scores[0].get_text(strip=True)
        map_score_b = scores[1].get_text(strip=True)
        winner = team_a if int(map_score_a) > int(map_score_b) else team_b
        team_a_win = 1 if winner == team_a else 0

        tables = game.select('table.wf-table-inset.mod-overview')
        for tbl_idx, tbl in enumerate(tables):
            player_team = team_a if tbl_idx == 0 else team_b

            for prow in tbl.select('tbody tr'):
                p_el = prow.select_one('td.mod-player a')
                player_name = p_el.get_text(strip=True) if p_el else 'N/A'

                stats = prow.select('.mod-stat')

                def get_stat(i):
                    if i < len(stats):
                        e = stats[i].select_one('.side.mod-both')
                        return e.get_text(strip=True) if e else 'N/A'
                    return 'N/A'

                rows.append({
                    'match_id': match_id,
                    'event_name': event_name,
                    'event_stage': event_stage,
                    'match_date': match_date,
                    'team_a': team_a,
                    'team_b': team_b,
                    'best_of': best_of,
                    'team_a_score': team_a_score,
                    'team_b_score': team_b_score,
                    'map_name': map_name,
                    'map_score_a': map_score_a,
                    'map_score_b': map_score_b,
                    'winner': winner,
                    'player_team': player_team,
                    'player_name': player_name,
                    'acs': get_stat(1),
                    'kills': get_stat(2),
                    'deaths': get_stat(3),
                    'assists': get_stat(4),
                    'team_a_win': team_a_win
                })

    return rows


if __name__ == "__main__":
    # 1) Load URLs
    with open("vlr_all_match_urls.json", "r") as f:
        urls = json.load(f)

    # 2) See which matches we've already scraped
    scraped = set()
    if os.path.exists("data.csv"):
        existing = pd.read_csv("data.csv", usecols=["match_id"])
        scraped = set(existing["match_id"].astype(str))

    all_rows = []
    for url in urls:
        mid = re.findall(r'/(\d+)', url)[0]
        if mid in scraped:
            # Silencing this print statement to reduce clutter, but you can re-enable it
            print(f"• Skipping {mid} (already in data.csv)")
            continue
        print(f"• Scraping {mid} …")

        try:
            match_rows = scrape_match(url)
            if match_rows:
                all_rows.extend(match_rows)
        except Exception as e:
            # Add a general catch-all to prevent one bad match from stopping the whole scrape
            print(f"    -> CRITICAL ERROR scraping {mid}: {e}. Skipping.")

    # 3) Append new rows to data.csv
    if all_rows:
        df = pd.DataFrame(all_rows, columns=[
            'match_id', 'event_name', 'event_stage', 'match_date',
            'team_a', 'team_b', 'best_of', 'team_a_score', 'team_b_score',
            'map_name', 'map_score_a', 'map_score_b', 'winner',
            'player_team', 'player_name', 'acs', 'kills', 'deaths', 'assists', 'team_a_win'
        ])
        df.to_csv("data.csv",
                  mode='a',
                  index=False,
                  header=not os.path.exists("data.csv"))
        print(f"\n✅ Appended {len(df)} new player rows to data.csv")
    else:
        print("\nℹ️  No new matches found or all new matches were skipped.")