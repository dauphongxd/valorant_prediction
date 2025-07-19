import pandas as pd
import numpy as np
import joblib
import datetime
import os


def load_model_data(path):
    """Loads the dictionary containing the pipeline and all helper data."""
    try:
        model_data = joblib.load(path)
        return model_data
    except FileNotFoundError:
        print(f"Error: Model file not found at '{path}'")
        print("Please run catboost_training.py first to create the model file.")
        return None


def build_prediction_input(team_a, team_b, map_name, best_of, model_data):
    """
    Constructs the feature DataFrame for a new prediction.
    This function's interface does not change.
    """
    map_stats = model_data['map_stats']
    global_stats = model_data['global_stats']
    all_teams = model_data['all_teams']

    if team_a not in all_teams or team_b not in all_teams:
        raise ValueError("One or both teams were not found in the training data.")

    stats_source_df = None
    is_fallback = False
    if not map_name or map_name not in map_stats:
        is_fallback = True
        stats_source_df = global_stats
    else:
        stats_source_df = map_stats[map_name]
        if team_a not in stats_source_df.index or team_b not in stats_source_df.index:
            is_fallback = True
            stats_source_df = global_stats

    stat_diff = {
        f'{stat}_diff': stats_source_df.at[team_a, stat] - stats_source_df.at[team_b, stat]
        for stat in ['acs', 'kills', 'deaths', 'assists']
    }
    team_diff = {
        f'team_{t}': (1 if t == team_a else -1 if t == team_b else 0)
        for t in all_teams
    }
    context_features = {'best_of': best_of, 'recency_score': 1.0}
    features = {**stat_diff, **team_diff, **context_features}
    X_new = pd.DataFrame([features])

    return X_new, is_fallback


def get_prediction(team_a, team_b, map_name, best_of, artifacts_dir='.'):
    """
    NEW UNIFIED ENTRY POINT
    """
    model_path = os.path.join(artifacts_dir, 'model.pkl')
    model_data = load_model_data(model_path)
    if not model_data:
        return None

    try:
        # Predict P(A beats B)
        X_a_vs_b, is_fallback1 = build_prediction_input(
            team_a, team_b, map_name, best_of, model_data
        )
        pipeline = model_data['pipeline']
        prob_a_wins_raw = pipeline.predict_proba(X_a_vs_b)[0, 1]

        # Predict P(B beats A)
        X_b_vs_a, is_fallback2 = build_prediction_input(
            team_b, team_a, map_name, best_of, model_data
        )
        prob_b_wins_raw = pipeline.predict_proba(X_b_vs_a)[0, 1]

        # Normalize for symmetry
        total_prob = prob_a_wins_raw + prob_b_wins_raw
        prob_a_wins_final = 0.5 if total_prob == 0 else prob_a_wins_raw / total_prob
        return prob_a_wins_final

    except (ValueError, KeyError) as e:
        print(f"\nError making prediction: {e}")
        return None


if __name__ == '__main__':
    print("\n--- Esports Match Predictor ---")
    team_a_input = input("Enter Team A: ").strip()
    team_b_input = input("Enter Team B: ").strip()
    map_name_input = input("Enter Map Name (or leave blank for general): ").strip()

    best_of_input_str = input("Best of (1, 3, or 5): ").strip()
    best_of_map = {'1': 'Bo1', '3': 'Bo3', '5': 'Bo5'}
    best_of_final = best_of_map.get(best_of_input_str)

    if not best_of_final:
        print(f"[!] Invalid format '{best_of_input_str}'. Defaulting to Bo3.")
        best_of_final = 'Bo3'

    prob_a_wins_final = get_prediction(team_a_input, team_b_input, map_name_input or 'N/A', best_of_final,
                                       artifacts_dir='.')

    if prob_a_wins_final is not None:
        winner = team_a_input if prob_a_wins_final > 0.5 else team_b_input
        print("\n--- Prediction ---")
        print(f"P({team_a_input} wins) = {prob_a_wins_final:.3f}")
        print(f"P({team_b_input} wins) = {1 - prob_a_wins_final:.3f}")
        print(f"Predicted Winner: {winner}")
    else:
        print("Prediction failed.")