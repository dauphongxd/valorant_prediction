# predict.py (Updated with simplified 'Best Of' input)

import pandas as pd
import numpy as np
import joblib
import datetime

MODEL_PATH = 'model.pkl'


def load_model_data(path=MODEL_PATH):
    """Loads the dictionary containing the pipeline and all helper data."""
    try:
        model_data = joblib.load(path)
        return model_data
    except FileNotFoundError:
        print(f"Error: Model file not found at '{path}'")
        print("Please run training.py first to create the model file.")
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


if __name__ == '__main__':
    model_data = load_model_data()

    if model_data:
        print("\n--- Esports Match Predictor ---")
        team_a_input = input("Enter Team A: ").strip()
        team_b_input = input("Enter Team B: ").strip()
        map_name_input = input("Enter Map Name (or leave blank for general): ").strip()

        # --- MODIFIED SECTION: Simplified 'Best Of' input ---
        best_of_input_str = input("Best of (1, 3, or 5): ").strip()

        # Map the numeric input to the string format the model expects
        best_of_map = {
            '1': 'Bo1',
            '3': 'Bo3',
            '5': 'Bo5'
        }
        # Use .get() to safely retrieve the value. If not found, it returns None.
        best_of_final = best_of_map.get(best_of_input_str)

        # If the user entered an invalid format, default to Bo3 and inform them.
        if not best_of_final:
            print(f"[!] Invalid format '{best_of_input_str}'. Defaulting to Bo3.")
            best_of_final = 'Bo3'
        # --- END OF MODIFIED SECTION ---

        try:
            # Predict P(A beats B)
            X_a_vs_b, is_fallback1 = build_prediction_input(
                team_a_input, team_b_input, map_name_input, best_of_final, model_data
            )
            pipeline = model_data['pipeline']
            prob_a_wins_raw = pipeline.predict_proba(X_a_vs_b)[0, 1]

            # Predict P(B beats A)
            X_b_vs_a, is_fallback2 = build_prediction_input(
                team_b_input, team_a_input, map_name_input, best_of_final, model_data
            )
            prob_b_wins_raw = pipeline.predict_proba(X_b_vs_a)[0, 1]

            # Normalize for symmetry
            total_prob = prob_a_wins_raw + prob_b_wins_raw
            prob_a_wins_final = 0.5 if total_prob == 0 else prob_a_wins_raw / total_prob

            if is_fallback1 or is_fallback2:
                display_map = map_name_input if map_name_input else 'N/A'
                print(
                    f"\n[!] NOTE: Map '{display_map}' not in training data or missing stats. Using general team stats as a fallback.")

            winner = team_a_input if prob_a_wins_final > 0.5 else team_b_input

            print("\n--- Prediction ---")
            print(f"P({team_a_input} wins) = {prob_a_wins_final:.3f}")
            print(f"P({team_b_input} wins) = {1 - prob_a_wins_final:.3f}")
            print(f"Predicted Winner: {winner}")

        except ValueError as e:
            print(f"\nError making prediction: {e}")
        except KeyError as e:
            print(
                f"\nError: A team ({e}) might be missing from the stats. Ensure they have played at least one match in the dataset.")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")