# predict.py

import pandas as pd
import numpy as np
import joblib
import torch
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
ARTIFACTS_FILE = 'tabnet_optimized_artifacts.pkl'


def load_artifacts(path):
    try:
        artifacts = joblib.load(path)
        print("Optimized TabNet model and all artifacts loaded successfully.")
        return artifacts
    except FileNotFoundError:
        print(f"Error: Artifacts file not found at '{path}'. Please run the updated training.py first.")
        return None

def _create_feature_vector(artifacts, team_a, team_b, best_of, map_name):
    """
    Helper function to create a single feature vector for prediction.
    This is called by the main predict_match function.
    """
    global_stats = artifacts['global_stats']
    map_stats = artifacts['map_stats']
    h2h_stats = artifacts['h2h_stats']
    label_encoders = artifacts['label_encoders']
    num_features = artifacts['numerical_features']
    cat_features = artifacts['categorical_features']

    default_stats = {'ewm_acs': 0.0, 'ewm_kdr': 0.0, 'ewm_assists': 0.0}

    # 1. Determine stat source: map-specific or global
    use_map_stats = map_name in map_stats and team_a in map_stats[map_name].index and team_b in map_stats[map_name].index
    stats_source = map_stats.get(map_name, global_stats) if use_map_stats else global_stats

    stats_a = stats_source.loc[team_a].to_dict() if team_a in stats_source.index else default_stats
    stats_b = stats_source.loc[team_b].to_dict() if team_b in stats_source.index else default_stats

    # 2. Look up H2H stats (advantage for team_a)
    h2h_key = tuple(sorted((team_a, team_b)))
    h2h_win_rate = h2h_stats.get(h2h_key, 0.5)
    h2h_advantage = (h2h_win_rate if team_a == h2h_key[0] else 1 - h2h_win_rate) - 0.5

    # 3. Build feature dictionary with relative differences
    feature_dict = {
        'ewm_acs_diff': stats_a['ewm_acs'] - stats_b['ewm_acs'],
        'ewm_kdr_diff': stats_a['ewm_kdr'] - stats_b['ewm_kdr'],
        'ewm_assists_diff': stats_a['ewm_assists'] - stats_b['ewm_assists'],
        'h2h_advantage': h2h_advantage,
        'map_name': map_name if map_name else '__UNKNOWN__',
        'best_of': best_of
    }
    input_df = pd.DataFrame([feature_dict])

    # 4. Preprocess features
    for col in cat_features:
        le = label_encoders[col]
        # Handle unseen categories during prediction
        input_df[col] = input_df[col].astype(str).apply(lambda x: x if x in le.classes_ else '__UNKNOWN__')
        input_df[col] = le.transform(input_df[col])

    # Ensure the column order matches the trained model
    return input_df[num_features + cat_features].values, use_map_stats


def predict_match(artifacts, team_a, team_b, best_of, map_name):
    """
    Predicts the match outcome by making two predictions (A vs B and B vs A)
    and averaging them to ensure perfect symmetry.
    """
    model = artifacts['model']

    # --- Prediction 1: Perspective of Team A vs Team B ---
    X_ab, use_map_stats = _create_feature_vector(artifacts, team_a, team_b, best_of, map_name)
    prob_a_wins_perspective_a = model.predict_proba(X_ab)[0][1]

    # --- Prediction 2: Perspective of Team B vs Team A ---
    X_ba, _ = _create_feature_vector(artifacts, team_b, team_a, best_of, map_name)
    prob_b_wins_perspective_b = model.predict_proba(X_ba)[0][1]
    # The implied win probability for team A from B's perspective
    prob_a_wins_perspective_b = 1.0 - prob_b_wins_perspective_b

    # --- Final Averaged Prediction ---
    final_prob_a_wins = (prob_a_wins_perspective_a + prob_a_wins_perspective_b) / 2.0

    # Display Results
    print("\n--- Prediction Results (Symmetrically Averaged) ---")
    fallback_msg = "(using map-specific stats)" if use_map_stats else "(using global stats as fallback)"
    print(f"Matchup: {team_a} vs. {team_b} on '{map_name or 'N/A'}' ({best_of}) {fallback_msg}")
    print(f"  > {team_a} Win Probability: {final_prob_a_wins:.2%}")
    print(f"  > {team_b} Win Probability: {(1 - final_prob_a_wins):.2%}")
    print("-" * 50)


def main():
    artifacts = load_artifacts(ARTIFACTS_FILE)
    if not artifacts: return

    print("\n--- Optimized TabNet Esports Match Predictor ---")
    while True:
        print("\nEnter match details (or type 'exit' to quit).")
        team_a = input("Enter Team A: ").strip()
        if team_a.lower() == 'exit': break

        team_b = input("Enter Team B: ").strip()
        if team_b.lower() == 'exit': break

        map_name = input("Enter Map Name (or leave blank): ").strip()

        while True:
            bo_num = input("Enter Best Of (1, 3, or 5): ").strip()
            if bo_num in ['1', '3', '5']:
                best_of = f'Bo{bo_num}'
                break
            else:
                print("Invalid input.")

        predict_match(artifacts, team_a, team_b, best_of, map_name)


if __name__ == '__main__':
    main()