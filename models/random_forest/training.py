# training_v2.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Define file paths for clarity
DATA_FILE = 'ready.csv'
MODEL_FILE = 'model.joblib'
STATS_FILE = 'team_stats.joblib'
COLUMNS_FILE = 'model_columns.joblib'


def calculate_team_stats(df):
    """Calculates historical performance stats for each team from the dataset."""
    print("Calculating historical team stats...")
    team_stats = {}
    all_teams = pd.concat([df['team_a'], df['team_b']]).unique()

    for team in all_teams:
        team_a_matches = df[df['team_a'] == team]
        team_b_matches = df[df['team_b'] == team]
        total_games = len(team_a_matches) + len(team_b_matches)
        if total_games == 0: continue

        wins = team_a_matches['team_a_win'].sum() + (1 - team_b_matches['team_a_win']).sum()
        win_rate = wins / total_games
        acs_advantage = (team_a_matches['acs_diff'].sum() - team_b_matches['acs_diff'].sum()) / total_games
        kdr_advantage = (team_a_matches['kdr_diff'].sum() - team_b_matches['kdr_diff'].sum()) / total_games

        team_stats[team] = {
            'win_rate': win_rate,
            'avg_acs_advantage': acs_advantage,
            'avg_kdr_advantage': kdr_advantage
        }
    print("Team stats calculation complete.")
    return team_stats


def create_feature_dataframe(df, team_stats):
    """Builds the final DataFrame for model training."""
    print("Engineering features for the model...")

    def create_features_for_row(row):
        team_a_stats = team_stats.get(row['team_a'], {})
        team_b_stats = team_stats.get(row['team_b'], {})
        default_stats = {'win_rate': 0.5, 'avg_acs_advantage': 0, 'avg_kdr_advantage': 0}

        features = {
            'win_rate_diff': team_a_stats.get('win_rate', 0.5) - team_b_stats.get('win_rate', 0.5),
            'acs_adv_diff': team_a_stats.get('avg_acs_advantage', 0) - team_b_stats.get('avg_acs_advantage', 0),
            'kdr_adv_diff': team_a_stats.get('avg_kdr_advantage', 0) - team_b_stats.get('avg_kdr_advantage', 0),
            # REMOVED map_name from features
            'best_of': row['best_of']
        }
        return pd.Series(features)

    model_df = df.apply(create_features_for_row, axis=1)
    model_df['team_a_win'] = df['team_a_win']
    print("Feature engineering complete.")
    return model_df


def main():
    """Main function to run the training pipeline."""
    print("Loading and cleaning data...")
    df = pd.read_csv(DATA_FILE)
    df = df.drop(columns=['match_id', 'event_stage', 'map_name'])  # Also drop map_name here
    df['kdr_diff'] = (df['kills_diff'] / df['deaths_diff'].abs()).replace([np.inf, -np.inf], 0).fillna(0)

    team_stats = calculate_team_stats(df)
    model_df = create_feature_dataframe(df, team_stats)

    X = model_df.drop('team_a_win', axis=1)
    y = model_df['team_a_win']

    # One-Hot Encode only 'best_of'
    X_encoded = pd.get_dummies(X, columns=['best_of'], drop_first=True)
    model_columns = X_encoded.columns.tolist()

    print("Training the Random Forest model...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, oob_score=True)
    rf_model.fit(X_encoded, y)
    print(f"Model training complete. Out-of-Bag Accuracy: {rf_model.oob_score_:.2%}")

    print("Saving model and supporting files...")
    joblib.dump(rf_model, MODEL_FILE)
    joblib.dump(team_stats, STATS_FILE)
    joblib.dump(model_columns, COLUMNS_FILE)
    print(f"Files saved: {MODEL_FILE}, {STATS_FILE}, {COLUMNS_FILE}")


if __name__ == "__main__":
    main()