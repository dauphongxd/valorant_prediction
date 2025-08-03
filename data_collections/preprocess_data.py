#!/usr/bin/env python3
import pandas as pd


def main():
    # 1. Load raw scraped data
    print("Loading raw scraped data from data.csv...")
    try:
        df = pd.read_csv('data.csv')
    except MemoryError:
        print("\n--- FATAL ERROR ---")
        print("Failed to load data.csv into memory.")
        print("The file is too large to be processed at once on this machine.")
        print("Consider using a machine with more RAM or a more advanced chunking script.")
        return

    print(f"Data loaded successfully. Shape: {df.shape}")

    # 2. Ensure numeric types & drop any bad rows
    print("Cleaning data and enforcing numeric types...")
    numeric_cols = ['acs', 'kills', 'deaths', 'assists', 'team_a_win']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['acs', 'kills', 'deaths', 'assists'], inplace=True)
    df['team_a_win'] = df['team_a_win'].astype(int)
    print("Data cleaning complete.")

    # 3. Aggregate player stats to team-level per map
    print("Aggregating player stats to the team level...")
    agg = (
        df
        .groupby(['match_id', 'map_name', 'player_team'])
        .agg(
            team_acs=('acs', 'mean'),
            team_kills=('kills', 'sum'),
            team_deaths=('deaths', 'sum'),
            team_assists=('assists', 'sum')
        )
        .reset_index()
    )
    print("Aggregation complete.")

    # 4. Pull in the label + metadata (just one copy per map)
    print("Extracting unique match metadata...")
    meta = (
        df
        .drop_duplicates(['match_id', 'map_name'])
        [['match_id', 'map_name', 'team_a', 'team_b', 'team_a_win', 'best_of', 'event_stage', 'match_date']]
    )

    # 5. Merge stats for Team A and Team B into a single row per map
    # This replaces the memory-intensive pivot and apply operations.
    print("Merging team stats...")

    # Merge to get Team A's stats alongside metadata
    map_data_a = meta.merge(
        agg,
        left_on=['match_id', 'map_name', 'team_a'],
        right_on=['match_id', 'map_name', 'player_team'],
        how='left'
    )

    # Merge again on the same result to bring in Team B's stats
    map_data_full = map_data_a.merge(
        agg,
        left_on=['match_id', 'map_name', 'team_b'],
        right_on=['match_id', 'map_name', 'player_team'],
        how='left',
        suffixes=('_a', '_b')  # Suffixes handle columns with the same name (e.g., 'player_team')
    )
    print("Merge complete.")

    # Clean up column names from the merges
    map_data_full.rename(columns={
        'team_acs_a': 'team_a_acs', 'team_kills_a': 'team_a_kills', 'team_deaths_a': 'team_a_deaths',
        'team_assists_a': 'team_a_assists',
        'team_acs_b': 'team_b_acs', 'team_kills_b': 'team_b_kills', 'team_deaths_b': 'team_b_deaths',
        'team_assists_b': 'team_b_assists'
    }, inplace=True)
    map_data_full.drop(columns=['player_team_a', 'player_team_b'], inplace=True, errors='ignore')

    # 6. Compute difference features (vectorized and memory-efficient)
    print("Computing difference features...")
    stats = ['acs', 'kills', 'deaths', 'assists']
    for stat in stats:
        col_a = f'team_a_{stat}'
        col_b = f'team_b_{stat}'
        # Ensure columns exist before subtracting to prevent errors
        if col_a in map_data_full.columns and col_b in map_data_full.columns:
            map_data_full[f'{stat}_diff'] = map_data_full[col_a] - map_data_full[col_b]

    # 7. (Optional) Drop the raw per-team blocks now that you have diffs
    cols_to_drop = []
    for stat in stats:
        cols_to_drop.extend([f'team_a_{stat}', f'team_b_{stat}'])

    ready = map_data_full.drop(columns=cols_to_drop, errors='ignore')

    # Drop any rows with NaN diffs, which can result from incomplete match data
    diff_cols = [f'{s}_diff' for s in stats]
    ready.dropna(subset=diff_cols, inplace=True)

    # 8. Write out a new CSV that your trainer will consume
    output_path = 'ready.csv'
    ready.to_csv(output_path, index=False)
    print(f"\n✅ Preprocessed data written to {output_path}. Final shape: {ready.shape}")


if __name__ == "__main__":
    main()