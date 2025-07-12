import pandas as pd

# === Load matches and players ===
matches = pd.read_csv('matches.csv')
players = pd.read_csv('players.csv')

# === Clean percentage columns ===
def clean_percent(col):
    return col.str.replace('%', '', regex=False).astype(float)

players['kast'] = clean_percent(players['kast'])
players['hs_percent'] = clean_percent(players['hs_percent'])

# === Convert numeric stats ===
numeric_cols = ['rating', 'acs', 'kills', 'deaths', 'assists', 'adr']
players[numeric_cols] = players[numeric_cols].apply(pd.to_numeric, errors='coerce')

# === Drop rows with missing data
players.dropna(subset=numeric_cols + ['kast', 'hs_percent'], inplace=True)

# === Aggregate player stats PER team PER match/map ===
agg = players.groupby(['match_id', 'map_name', 'team']).agg({
    'rating': 'mean',
    'acs': 'mean',
    'kills': 'sum',
    'deaths': 'sum',
    'assists': 'sum',
    'kast': 'mean',
    'adr': 'mean',
    'hs_percent': 'mean'
}).reset_index()

# === Pivot team stats: rows = match_id + map_name, columns = stat_teamname ===
pivot = agg.pivot(index=['match_id', 'map_name'], columns='team')
pivot.columns = ['_'.join(col).strip() for col in pivot.columns.values]
pivot.reset_index(inplace=True)

# === Merge with matches ===
merged = matches.merge(pivot, on=['match_id', 'map_name'], how='left')

# === Label: 1 if team_a wins map, 0 otherwise ===
merged['team_a_win'] = (merged['winner'] == merged['team_a']).astype(int)

# === Save final dataset ===
merged.to_csv('ml_dataset.csv', index=False)
print("✅ Saved ML-ready dataset to ml_dataset.csv")
