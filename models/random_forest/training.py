import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
from datetime import datetime

# --- CONFIGURATION ---
DATA_FILE = 'ready.csv'
MODEL_FILE = 'model.joblib'
STATS_FILE = 'team_stats.joblib'
COLUMNS_FILE = 'model_columns.joblib'
# --- Time Weighting Configuration ---
TIME_WEIGHT_HALF_LIFE_DAYS = 90


def calculate_weighted_team_stats(df):
    """
    Calculates historical performance stats for each team, giving more weight
    to more recent matches. (Includes assists_diff)
    """
    print("Calculating time-weighted historical team stats...")
    current_date = df['match_date'].max()
    df['match_age_days'] = (current_date - df['match_date']).dt.days
    decay_rate = np.log(2) / TIME_WEIGHT_HALF_LIFE_DAYS
    df['weight'] = np.exp(-decay_rate * df['match_age_days'])

    team_stats = {}
    all_teams = pd.concat([df['team_a'], df['team_b']]).unique()

    for team in all_teams:
        team_a_matches = df[df['team_a'] == team]
        team_b_matches = df[df['team_b'] == team]

        total_weight = team_a_matches['weight'].sum() + team_b_matches['weight'].sum()
        if total_weight == 0: continue

        wins_as_a = (team_a_matches['team_a_win'] * team_a_matches['weight']).sum()
        wins_as_b = ((1 - team_b_matches['team_a_win']) * team_b_matches['weight']).sum()
        weighted_win_rate = (wins_as_a + wins_as_b) / total_weight

        acs_adv = ((team_a_matches['acs_diff'] * team_a_matches['weight']).sum() - (
                    team_b_matches['acs_diff'] * team_b_matches['weight']).sum()) / total_weight
        kdr_adv = ((team_a_matches['kdr_diff'] * team_a_matches['weight']).sum() - (
                    team_b_matches['kdr_diff'] * team_b_matches['weight']).sum()) / total_weight
        assists_adv = ((team_a_matches['assists_diff'] * team_a_matches['weight']).sum() - (
                    team_b_matches['assists_diff'] * team_b_matches['weight']).sum()) / total_weight

        team_stats[team] = {
            'win_rate': weighted_win_rate,
            'avg_acs_advantage': acs_adv,
            'avg_kdr_advantage': kdr_adv,
            'avg_assists_advantage': assists_adv
        }
    print("Team stats calculation complete.")
    return team_stats


def create_feature_dataframe(df, team_stats):
    """Builds the final DataFrame for model training."""
    print("Engineering features for the model...")

    def create_features_for_row(row):
        team_a_stats = team_stats.get(row['team_a'], {})
        team_b_stats = team_stats.get(row['team_b'], {})

        features = {
            'win_rate_diff': team_a_stats.get('win_rate', 0.5) - team_b_stats.get('win_rate', 0.5),
            'acs_adv_diff': team_a_stats.get('avg_acs_advantage', 0) - team_b_stats.get('avg_acs_advantage', 0),
            'kdr_adv_diff': team_a_stats.get('avg_kdr_advantage', 0) - team_b_stats.get('avg_kdr_advantage', 0),
            'assists_adv_diff': team_a_stats.get('avg_assists_advantage', 0) - team_b_stats.get('avg_assists_advantage',
                                                                                                0),
            'best_of': row['best_of'],
            'map_name': row['map_name'],
            # --- REMOVED 'event_stage' ---
        }
        return pd.Series(features)

    model_df = df.apply(create_features_for_row, axis=1)
    model_df['team_a_win'] = df['team_a_win']
    print("Feature engineering complete.")
    return model_df


def main():
    """Main function to run the focused training pipeline."""
    print("Loading and cleaning data...")
    df = pd.read_csv(DATA_FILE, parse_dates=['match_date'])
    # --- REMOVED 'event_stage' from the dataframe ---
    df = df.drop(columns=['match_id', 'event_stage'])
    df['kdr_diff'] = (df['kills_diff'] / df['deaths_diff'].abs()).replace([np.inf, -np.inf], 0).fillna(0)

    team_stats = calculate_weighted_team_stats(df)
    model_df = create_feature_dataframe(df, team_stats)

    X = model_df.drop('team_a_win', axis=1)
    y = model_df['team_a_win']

    # --- REMOVED 'event_stage' from one-hot encoding ---
    categorical_features = ['best_of', 'map_name']
    X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    model_columns = X_encoded.columns.tolist()

    print("Training the Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=150, random_state=42, oob_score=True, min_samples_leaf=5)
    rf_model.fit(X_encoded, y)
    print(f"Model training complete. Out-of-Bag Accuracy: {rf_model.oob_score_:.2%}")

    print("Saving model and supporting files...")
    joblib.dump(rf_model, MODEL_FILE)
    joblib.dump(team_stats, STATS_FILE)
    joblib.dump(model_columns, COLUMNS_FILE)
    print(f"Files saved: {MODEL_FILE}, {STATS_FILE}, {COLUMNS_FILE}")


if __name__ == "__main__":
    main()