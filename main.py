import requests
from bs4 import BeautifulSoup
import re

# Sample match URL
url = 'https://www.vlr.gg/498628/paper-rex-vs-fnatic-valorant-masters-toronto-2025-gf'  # Replace with actual match link
headers = {'User-Agent': 'Mozilla/5.0'}

resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.content, 'html.parser')

# === Match ID ===
match_id = re.findall(r'/(\d+)', url)[0]

# === Event Name ===
event_elem = soup.select_one('.match-header-event > div > div')
event_name = event_elem.get_text(strip=True) if event_elem else 'N/A'

# === Event Stage ===
stage_elem = soup.select_one('.match-header-event-series')
event_stage = stage_elem.get_text(strip=True) if stage_elem else 'N/A'

# === Match Date ===
date_elem = soup.select_one('.match-header-date .moment-tz-convert[data-moment-format="dddd, MMMM Do"]')
match_date = date_elem.get_text(strip=True) if date_elem else 'N/A'

# === Patch ===
patch_elem = soup.select_one('.match-header-date div[style*="italic"]')
patch = patch_elem.get_text(strip=True) if patch_elem else 'N/A'

# === Teams ===
team_a_elem = soup.select_one('.match-header-link.mod-1 .wf-title-med')
team_b_elem = soup.select_one('.match-header-link.mod-2 .wf-title-med')
team_a = team_a_elem.get_text(strip=True) if team_a_elem else 'N/A'
team_b = team_b_elem.get_text(strip=True) if team_b_elem else 'N/A'

# === Score ===
score_winner = soup.select_one('.match-header-vs-score-winner')
score_loser = soup.select_one('.match-header-vs-score-loser')
team_a_score = score_winner.get_text(strip=True) if score_winner else 'N/A'
team_b_score = score_loser.get_text(strip=True) if score_loser else 'N/A'

# === BO Type (e.g. Bo3) ===
bo_elem = soup.select('.match-header-vs-note')
bo_type = ''
for note in bo_elem:
    if "Bo" in note.text:
        bo_type = note.get_text(strip=True)
        break

# === Map Bans / Picks ===
note_elem = soup.select_one('.match-header-note')
ban_pick_notes = note_elem.get_text(strip=True) if note_elem else 'N/A'

# === Print All ===
print("Match ID:", match_id)
print("Event:", event_name)
print("Stage:", event_stage)
print("Date:", match_date)
print("Patch:", patch)
print("Team A:", team_a)
print("Team B:", team_b)
print("Score A:", team_a_score)
print("Score B:", team_b_score)
print("BO Type:", bo_type)
print("Ban/Pick Notes:", ban_pick_notes)

print("\n" + "=" * 20)
print("PER-MAP STATISTICS")
print("=" * 20)

maps_data = []
# Find all the containers for each game's stats (Map 1, Map 2, etc.)
map_stat_containers = soup.select('.vm-stats-game')

for game in map_stat_containers:
    game_id = game.get('data-game-id')
    # Skip the "all" tab which is for overall match stats
    if game_id == 'all':
        continue

    # --- Map Info ---
    # The map name is the first text node within the complex span
    map_name_elem = game.select_one('.map > div > span')
    map_name = map_name_elem.find(text=True, recursive=False).strip() if map_name_elem else 'N/A'

    map_duration_elem = game.select_one('.map-duration')
    map_duration = map_duration_elem.get_text(strip=True) if map_duration_elem else 'N/A'

    scores_elems = game.select('.score')
    map_score_a = scores_elems[0].get_text(strip=True) if len(scores_elems) > 0 else 'N/A'
    map_score_b = scores_elems[1].get_text(strip=True) if len(scores_elems) > 1 else 'N/A'

    # --- Player Stats for this Map ---
    players_stats_list = []
    # There are two tables per map, one for each team
    stat_tables = game.select('table.wf-table-inset.mod-overview')

    for table in stat_tables:
        player_rows = table.select('tbody tr')
        for row in player_rows:
            player_name_elem = row.select_one('.mod-player .text-of')
            player_name = player_name_elem.get_text(strip=True) if player_name_elem else 'N/A'

            # A player might use multiple agents in a match (on different maps), so we grab all for this map
            agent_imgs = row.select('.mod-agents img')
            agents = [img['title'] for img in agent_imgs] if agent_imgs else ['N/A']

            # Get all stat columns for the player
            stats = row.select('.mod-stat')


            # Helper function to safely extract stat text from the '.mod-both' span
            def get_stat(stat_index):
                if len(stats) > stat_index:
                    stat_elem = stats[stat_index].select_one('.side.mod-both')
                    if stat_elem:
                        return stat_elem.get_text(strip=True)
                return 'N/A'


            player_data = {
                'name': player_name,
                'agents': ', '.join(agents),
                'rating': get_stat(0),
                'acs': get_stat(1),
                'kills': get_stat(2),
                'deaths': get_stat(3),
                'assists': get_stat(4),
                'kd_diff': get_stat(5),
                'kast': get_stat(6),
                'adr': get_stat(7),
                'hs_percent': get_stat(8),
                'first_kills': get_stat(9),
                'first_deaths': get_stat(10),
                'fk_fd_diff': get_stat(11),
            }
            players_stats_list.append(player_data)

    # Store all collected data for this map
    map_info = {
        'name': map_name,
        'duration': map_duration,
        'score_a': map_score_a,
        'score_b': map_score_b,
        'players': players_stats_list
    }
    maps_data.append(map_info)

# === Print All Per-Map Data ===
for i, map_data in enumerate(maps_data):
    print(f"\n--- Map {i + 1}: {map_data['name']} ---")
    print(f"Score: {team_a} {map_data['score_a']} - {map_data['score_b']} {team_b}")
    print(f"Duration: {map_data['duration']}")
    print("\nPlayer Scoreboard:")

    # Print header
    print(
        f"{'Player':<15} {'Agent':<10} {'R':>5} {'ACS':>5} {'K':>3} {'D':>3} {'A':>3} {'KAST':>6} {'ADR':>5} {'HS%':>5}")
    print(
        f"{'-' * 15:<15} {'-' * 10:<10} {'-' * 5:>5} {'-' * 5:>5} {'-' * 3:>3} {'-' * 3:>3} {'-' * 3:>3} {'-' * 6:>6} {'-' * 5:>5} {'-' * 5:>5}")

    for player in map_data['players']:
        # Print each player's stats
        print(
            f"{player['name']:<15} {player['agents']:<10} {player['rating']:>5} {player['acs']:>5} {player['kills']:>3} {player['deaths']:>3} {player['assists']:>3} {player['kast']:>6} {player['adr']:>5} {player['hs_percent']:>5}")

        # Split teams with a blank line for readability
        if len(map_data['players']) > 5 and map_data['players'].index(player) == 4:
            print("")

print("\n" + "=" * 20)
print("HEAD-TO-HEAD HISTORY")
print("=" * 20)

h2h_history = []
# Find the main container for the H2H section
h2h_section = soup.select_one('.wf-card.match-h2h')

if h2h_section:
    # Find all the individual past match items within the H2H section
    past_matches = h2h_section.select('a.wf-module-item.mod-h2h')

    for match in past_matches:
        # Extract event name and series
        event_name_elem = match.select_one('.match-h2h-matches-event-name')
        event_series_elem = match.select_one('.match-h2h-matches-event-series')
        event_name = event_name_elem.get_text(strip=True) if event_name_elem else 'N/A'
        event_series = event_series_elem.get_text(strip=True) if event_series_elem else ''
        full_event = f"{event_name} - {event_series}" if event_series else event_name

        # Extract date
        date_elem = match.select_one('.match-h2h-matches-date')
        h2h_date = date_elem.get_text(strip=True) if date_elem else 'N/A'

        # Extract scores
        score_elems = match.select('.match-h2h-matches-score span')
        if len(score_elems) >= 2:
            # The HTML structure consistently places the first team's score in the first span
            # and the second team's score in the second span.
            score_a = score_elems[0].get_text(strip=True)
            score_b = score_elems[1].get_text(strip=True)

            # Determine the winner by checking which score span has the 'mod-win' class
            # The second team's score span has the class 'ra', the first has 'rf'
            winner_elem = match.select_one('.match-h2h-matches-score .mod-win')
            if winner_elem and 'ra' in winner_elem.get('class', []):
                winner_name = team_b  # Team B from the main match header won
            else:
                winner_name = team_a  # Team A from the main match header won
        else:
            score_a, score_b, winner_name = 'N/A', 'N/A', 'N/A'

        # Extract the full link to the match
        match_link = f"https://www.vlr.gg{match['href']}" if match.has_attr('href') else 'N/A'

        h2h_history.append({
            'date': h2h_date,
            'event': full_event,
            'score_a': score_a,
            'score_b': score_b,
            'winner': winner_name,
            'link': match_link
        })
else:
    print("No Head-to-Head history found on the page.")

# === Print All H2H Data ===
if h2h_history:
    # Print header for the H2H results
    print(f"{'Date':<12} | {'Event':<45} | {'Score':<10} | {'Winner'}")
    print(f"{'-' * 12} | {'-' * 45} | {'-' * 10} | {'-' * 15}")

    for match in h2h_history:
        # Format the score string based on the teams from the main match page
        score_string = f"{match['score_a']} - {match['score_b']}"
        print(f"{match['date']:<12} | {match['event']:<45} | {score_string:<10} | {match['winner']}")
else:
    # This message will print if the H2H section was found but was empty
    print("No past matches found in the Head-to-Head section.")

# === Recent Matches History for Each Team ===
print("\n" + "=" * 20)
print("RECENT MATCH HISTORY")
print("=" * 20)


# A reusable function to parse a team's history section
def scrape_team_history(history_container, team_name):
    """Parses the recent matches container for a single team."""
    history_list = []
    matches = history_container.select('a.match-histories-item')

    for match in matches:
        opponent_name_elem = match.select_one('.match-histories-item-opponent-name')
        opponent_name = opponent_name_elem.get_text(strip=True) if opponent_name_elem else 'N/A'

        date_elem = match.select_one('.match-histories-item-date')
        match_date = date_elem.get_text(strip=True) if date_elem else 'N/A'

        # Determine the result from the class of the parent link
        result = 'Win' if 'mod-win' in match.get('class', []) else 'Loss'

        # Get the scores
        score_elems = match.select('.match-histories-item-result span')
        if len(score_elems) >= 2:
            # .rf is always the team's score, .ra is the opponent's
            team_score = score_elems[0].get_text(strip=True)
            opponent_score = score_elems[1].get_text(strip=True)
        else:
            team_score, opponent_score = 'N/A', 'N/A'

        match_link = f"https://www.vlr.gg{match['href']}" if match.has_attr('href') else 'N/A'

        history_list.append({
            'opponent': opponent_name,
            'date': match_date,
            'result': result,
            'team_score': team_score,
            'opponent_score': opponent_score,
            'link': match_link
        })
    return history_list


# Select the two history containers from the page
history_containers = soup.select('.wf-card.mod-dark.match-histories')

if len(history_containers) >= 2:
    # The first container is for team_a, the second is for team_b
    team_a_history = scrape_team_history(history_containers[0], team_a)
    team_b_history = scrape_team_history(history_containers[1], team_b)

    # --- Print Team A History ---
    print(f"\n--- {team_a}'s Last 5 Matches ---")
    print(f"{'Result':<8} | {'Score':<8} | {'vs':<4} | {'Opponent':<30} | {'Date'}")
    print(f"{'-' * 8} | {'-' * 8} | {'-' * 4} | {'-' * 30} | {'-' * 10}")
    for past_match in team_a_history:
        score_str = f"{past_match['team_score']}-{past_match['opponent_score']}"
        print(
            f"{past_match['result']:<8} | {score_str:<8} | {'vs.':<4} | {past_match['opponent']:<30} | {past_match['date']}")

    # --- Print Team B History ---
    print(f"\n--- {team_b}'s Last 5 Matches ---")
    print(f"{'Result':<8} | {'Score':<8} | {'vs':<4} | {'Opponent':<30} | {'Date'}")
    print(f"{'-' * 8} | {'-' * 8} | {'-' * 4} | {'-' * 30} | {'-' * 10}")
    for past_match in team_b_history:
        score_str = f"{past_match['team_score']}-{past_match['opponent_score']}"
        print(
            f"{past_match['result']:<8} | {score_str:<8} | {'vs.':<4} | {past_match['opponent']:<30} | {past_match['date']}")

else:
    print("Could not find recent match history sections.")
