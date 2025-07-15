import requests
from bs4 import BeautifulSoup
import re
import pandas as pd

def clean(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()

url = 'https://www.vlr.gg/298687/trace-esports-vs-tyloo-champions-tour-2024-china-kickoff-opening-a'
headers = {'User-Agent': 'Mozilla/5.0'}
resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.content, 'html.parser')

# --- Match-level fields ---
match_id    = re.findall(r'/(\d+)', url)[0]
event_name  = soup.select_one('.match-header-event > div > div').get_text(strip=True)
event_stage = clean(soup.select_one('.match-header-event-series').get_text(strip=True))
date_elem   = soup.select_one('.match-header-date .moment-tz-convert')
match_date  = date_elem.get('data-utc-ts', '').split(' ')[0] if date_elem and date_elem.get('data-utc-ts') else date_elem.get_text(strip=True)
team_a      = soup.select_one('.match-header-link.mod-1 .wf-title-med').get_text(strip=True)
team_b      = soup.select_one('.match-header-link.mod-2 .wf-title-med').get_text(strip=True)
best_of     = next((n.get_text(strip=True) for n in soup.select('.match-header-vs-note') if 'Bo' in n.text), '')
score_w     = soup.select_one('.match-header-vs-score-winner').get_text(strip=True)
score_l     = soup.select_one('.match-header-vs-score-loser').get_text(strip=True)
team_a_score, team_b_score = score_w, score_l

rows = []
# iterate maps
for game in soup.select('.vm-stats-game'):
    if game.get('data-game-id') == 'all':
        continue

    map_name    = game.select_one('.map > div > span').find(text=True, recursive=False).strip()
    scores      = game.select('.score')
    map_score_a = scores[0].get_text(strip=True)
    map_score_b = scores[1].get_text(strip=True)
    winner      = team_a if int(map_score_a) > int(map_score_b) else team_b
    team_a_win  = 1 if winner == team_a else 0

    # two tables: first is team A, second is team B
    for tbl_idx, tbl in enumerate(game.select('table.wf-table-inset.mod-overview')):
        player_team = team_a if tbl_idx == 0 else team_b

        for row in tbl.select('tbody tr'):
            # extract player name
            player_el = row.select_one('td.mod-player a')
            player_name = player_el.get_text(strip=True) if player_el else 'N/A'

            stats = row.select('.mod-stat')
            def get_stat(i):
                el = stats[i].select_one('.side.mod-both')
                return el.get_text(strip=True) if el else 'N/A'

            rows.append({
                'match_id':      match_id,
                'event_name':    event_name,
                'event_stage':   event_stage,
                'match_date':    match_date,
                'team_a':        team_a,
                'team_b':        team_b,
                'best_of':       best_of,
                'team_a_score':  team_a_score,
                'team_b_score':  team_b_score,
                'map_name':      map_name,
                'map_score_a':   map_score_a,
                'map_score_b':   map_score_b,
                'winner':        winner,
                'player_team':   player_team,
                'player_name':   player_name,
                'acs':           get_stat(1),
                'kills':         get_stat(2),
                'deaths':        get_stat(3),
                'assists':       get_stat(4),
                'team_a_win':    team_a_win
            })

df = pd.DataFrame(rows, columns=[
    'match_id','event_name','event_stage','match_date',
    'team_a','team_b','best_of','team_a_score','team_b_score',
    'map_name','map_score_a','map_score_b','winner',
    'player_team','player_name','acs','kills','deaths','assists','team_a_win'
])
df.to_csv('data.csv', index=False)
