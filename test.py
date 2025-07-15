import pandas as pd

# 0) Load & clean EXACTLY like in your aggregate script
players = pd.read_csv('players.csv')

# strip “%” and convert
players['kast']       = players['kast'].str.rstrip('%').astype(float)
players['hs_percent'] = players['hs_percent'].str.rstrip('%').astype(float)

# coerce the rest to numeric
for c in ['rating','acs','kills','deaths','assists','adr']:
    players[c] = pd.to_numeric(players[c], errors='coerce')

# drop any rows that still have NAs
players.dropna(subset=['rating','acs','kills','deaths','assists','adr','kast','hs_percent'], inplace=True)

# 1) Now group and build the fallback dict
stat_cols = ['rating','acs','kills','deaths','assists','kast','adr','hs_percent']
grouped = players.groupby(['team','map_name'])[stat_cols].mean()

# turn into a lookup: (team, map) → { stat: value, … }
team_map_stats = {
    (team,mp): row.to_dict()
    for (team,mp), row in grouped.iterrows()
}
