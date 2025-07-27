import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
MODEL_FILE = 'model.pkl'
STATS_FILE = 'team_stats.pkl'
COLUMNS_FILE = 'model_columns.pkl'
DATA_FILE = 'ready.csv'
TIME_WEIGHT_HALF_LIFE_DAYS = 90


# --- ADVANCED FEATURE ENGINEERING (FROM YOUR BEST SCRIPTS) ---

def prepare_data(df):
    """Transforms raw match data into a "team-centric" format."""
    df['match_date'] = pd.to_datetime(df['match_date'])
    df['kdr_diff'] = (df['kills_diff'] / df['deaths_diff'].abs()).fillna(0)
    stat_cols = ['acs_diff', 'kdr_diff', 'assists_diff']
    df_a = df[['match_date', 'team_a', 'team_b'] + stat_cols].copy()
    df_a.rename(columns={'team_a': 'team', 'team_b': 'opponent'}, inplace=True)
    df_b = df[['match_date', 'team_b', 'team_a'] + stat_cols].copy()
    for col in stat_cols: df_b[col] = -df_b[col]
    df_b.rename(columns={'team_b': 'team', 'team_a': 'opponent'}, inplace=True)
    match_stats_df = pd.concat([df_a, df_b], ignore_index=True)
    match_stats_df.sort_values(by=['team', 'match_date'], inplace=True)
    return match_stats_df


def calculate_time_weighted_stats(match_stats_df):
    """Calculates the exponentially weighted moving average (EWM) for team stats."""
    stat_cols = {'acs_diff': 'acs', 'kdr_diff': 'kdr', 'assists_diff': 'assists'}
    match_stats_df.rename(columns=stat_cols, inplace=True)
    ewm_stats = {}
    for stat in stat_cols.values():
        ewm_stats[f'ewm_{stat}'] = match_stats_df.groupby('team').apply(
            lambda x: x.set_index('match_date')[stat].ewm(halflife=f'{TIME_WEIGHT_HALF_LIFE_DAYS}D',
                                                          times=x['match_date']).mean()
        ).reset_index(name=f'ewm_{stat}')[f'ewm_{stat}']
    ewm_df = pd.DataFrame(ewm_stats)
    full_stats_df = pd.concat([match_stats_df.reset_index(drop=True), ewm_df], axis=1)
    return full_stats_df


def create_feature_dataset(df, full_stats_df):
    """Merges time-weighted stats back to create the final model features."""
    ewm_cols_to_shift = [col for col in full_stats_df.columns if col.startswith('ewm_')]
    shifted_stats = full_stats_df.groupby('team')[ewm_cols_to_shift].transform(lambda x: x.shift(1))
    shifted_stats.fillna(0, inplace=True)
    full_stats_df_shifted = pd.concat([full_stats_df[['match_date', 'team', 'opponent']], shifted_stats], axis=1)
    df = pd.merge(df, full_stats_df_shifted, how='left', left_on=['match_date', 'team_a', 'team_b'],
                  right_on=['match_date', 'team', 'opponent']).rename(
        columns={col: col + '_a' for col in shifted_stats.columns}).drop(columns=['team', 'opponent'])
    df = pd.merge(df, full_stats_df_shifted, how='left', left_on=['match_date', 'team_b', 'team_a'],
                  right_on=['match_date', 'team', 'opponent']).rename(
        columns={col: col + '_b' for col in shifted_stats.columns}).drop(columns=['team', 'opponent'])
    for stat in ['ewm_acs', 'ewm_kdr', 'ewm_assists']:
        df[f'feature_{stat}_diff'] = df[f'{stat}_a'] - df[f'{stat}_b']
    df = pd.get_dummies(df, columns=['map_name', 'best_of'], prefix=['map', 'bo'], dtype=float)
    return df


# --- MAIN TRAINING PIPELINE ---

def main():
    print("--- Upgraded Decision Tree Training Pipeline ---")
    df = pd.read_csv(DATA_FILE)
    df.drop(columns=['match_id', 'event_stage'], inplace=True, errors='ignore')

    print("Step 1: Running advanced feature engineering...")
    match_stats_df = prepare_data(df)
    full_stats_df = calculate_time_weighted_stats(match_stats_df)
    feature_df = create_feature_dataset(df.copy(), full_stats_df)

    print("Step 2: Defining features and training model...")
    feature_cols = [col for col in feature_df.columns if
                    col.startswith('feature_') or col.startswith('map_') or col.startswith('bo_')]
    X = feature_df[feature_cols]
    y = feature_df['team_a_win']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # --- THIS IS THE FIX ---
    # We use aggressive hyperparameters to force the tree to generalize.
    # It cannot create a "leaf" for a prediction unless it has at least 25 historical examples.
    # This prevents the 0% and 100% problem.
    model = DecisionTreeClassifier(
        max_depth=8,  # A slightly shallower tree
        min_samples_leaf=25,  # The key parameter: forces generalization
        random_state=42
    )

    model.fit(X_train, y_train)
    print("\n--- Model Evaluation ---")

    print("\nStep 3: Saving model and prediction artifacts...")
    joblib.dump(model, MODEL_FILE)

    # Save the final "form" of each team for predictions
    final_team_stats = full_stats_df.groupby('team').last()
    final_team_stats = final_team_stats[['ewm_acs', 'ewm_kdr', 'ewm_assists']].to_dict('index')
    joblib.dump(final_team_stats, STATS_FILE)

    # Save the column order
    joblib.dump(X_train.columns.tolist(), COLUMNS_FILE)

    print(f"\nTraining complete. Artifacts saved: {MODEL_FILE}, {STATS_FILE}, {COLUMNS_FILE}")


if __name__ == '__main__':
    main()