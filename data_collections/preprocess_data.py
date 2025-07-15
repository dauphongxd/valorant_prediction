#!/usr/bin/env python3
import pandas as pd

def main():
    # 1. Load raw scraped data
    df = pd.read_csv('data.csv')

    # 2. Ensure numeric types & drop any bad rows
    for col in ['acs','kills','deaths','assists','team_a_win']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(subset=['acs','kills','deaths','assists'], inplace=True)

    # 3. Aggregate player stats to team-level per map
    agg = (
        df
        .groupby(['match_id','map_name','player_team'])
        .agg(team_acs    = ('acs',   'mean'),
             team_kills  = ('kills', 'sum'),
             team_deaths = ('deaths','sum'),
             team_assists= ('assists','sum'))
        .reset_index()
    )

    # 4. Pivot so each row is one (match,map) with both teams side-by-side
    pivot = agg.pivot(index=['match_id','map_name'],
                      columns='player_team',
                      values=['team_acs','team_kills','team_deaths','team_assists'])
    # flatten the MultiIndex columns
    pivot.columns = [f"{stat}_{team}" for stat,team in pivot.columns]
    pivot = pivot.reset_index()

    # 5. Pull in the label + metadata (just one copy per map)
    meta = (
        df
        .drop_duplicates(['match_id','map_name'])
        [['match_id','map_name','team_a','team_b','team_a_win','best_of','event_stage','match_date']]
    )

    # 6. Merge metadata + pivoted stats
    ready = meta.merge(pivot, on=['match_id','map_name'])

    # 7. Compute difference features (A − B)
    for stat in ['acs','kills','deaths','assists']:
        ready[f'{stat}_diff'] = ready.apply(
            lambda r: r[f'team_{stat}_{r.team_a}'] - r[f'team_{stat}_{r.team_b}'],
            axis=1
        )

    # 8. (Optional) Drop the raw per-team blocks now that you have diffs
    drop_cols = [c for c in ready.columns
                 if any(c.startswith(f'team_{s}_') for s in ['acs','kills','deaths','assists'])]
    ready = ready.drop(columns=drop_cols)

    # 9. Write out a new CSV that your trainer will consume
    output_path = 'ready.csv'
    ready.to_csv(output_path, index=False)
    print(f"✅ Preprocessed data written to {output_path}")

if __name__ == "__main__":
    main()
