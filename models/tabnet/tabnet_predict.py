# catboost_predict.py

import pandas as pd
import numpy as np
import joblib
import torch
import warnings
import os

warnings.filterwarnings('ignore')

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


def get_prediction(team_a, team_b, map_name, best_of, artifacts_dir='.'):
    """
    NEW UNIFIED ENTRY POINT
    """
    try:
        artifact_path = os.path.join(artifacts_dir, 'tabnet_optimized_artifacts.pkl')
        artifacts = joblib.load(artifact_path)
    except FileNotFoundError:
        print(f"Error: Artifacts file not found at '{artifact_path}'.")
        return None

    model = artifacts['model']

    X_ab, _ = _create_feature_vector(artifacts, team_a, team_b, best_of, map_name)
    prob_a_wins_perspective_a = model.predict_proba(X_ab)[0][1]

    X_ba, _ = _create_feature_vector(artifacts, team_b, team_a, best_of, map_name)
    prob_b_wins_perspective_b = model.predict_proba(X_ba)[0][1]
    prob_a_wins_perspective_b = 1.0 - prob_b_wins_perspective_b

    final_prob_a_wins = (prob_a_wins_perspective_a + prob_a_wins_perspective_b) / 2.0
    return final_prob_a_wins


def main():
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

        final_prob_a_wins = get_prediction(team_a, team_b, map_name or 'N/A', best_of, artifacts_dir='.')

        if final_prob_a_wins is not None:
            print("\n--- Prediction Results (Symmetrically Averaged) ---")
            print(f"Matchup: {team_a} vs. {team_b} on '{map_name or 'N/A'}' ({best_of})")
            print(f"  > {team_a} Win Probability: {final_prob_a_wins:.2%}")
            print(f"  > {team_b} Win Probability: {(1 - final_prob_a_wins):.2%}")
            print("-" * 50)
        else:
            print("Prediction Failed.")


if __name__ == '__main__':
    main()