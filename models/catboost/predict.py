# predict_catboost.py

import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- Global Variables ---
try:
    # NEW: Load the CatBoost artifacts
    MODEL = joblib.load('catboost_model.pkl')
    TEAM_STATS = joblib.load('catboost_team_stats.pkl')
    MODEL_COLUMNS = joblib.load('catboost_model_columns.pkl')
    print("CatBoost model and artifacts loaded successfully.")
except FileNotFoundError:
    print("Error: Model artifacts not found. Please run training_catboost.py first.")
    MODEL, TEAM_STATS, MODEL_COLUMNS = None, None, None


def _get_single_prediction(team_a, team_b, map_name, best_of):
    """
    Internal helper to get a single, one-sided prediction from the model.
    Returns the raw probability of team_a winning.
    *** MODIFIED FOR CATBOOST: Simplified DataFrame creation. ***
    """
    default_stats = {'ewm_acs': 0, 'ewm_kdr': 0, 'ewm_assists': 0}

    stats_a = TEAM_STATS.get(team_a, default_stats)
    stats_b = TEAM_STATS.get(team_b, default_stats)

    # --- CATBOOST CHANGE: Create the input DataFrame directly ---
    # The structure is much simpler now.
    feature_dict = {
        'feature_ewm_acs_diff': stats_a['ewm_acs'] - stats_b['ewm_acs'],
        'feature_ewm_kdr_diff': stats_a['ewm_kdr'] - stats_b['ewm_kdr'],
        'feature_ewm_assists_diff': stats_a['ewm_assists'] - stats_b['ewm_assists'],
        'map_name': map_name,
        'best_of': best_of
    }

    # Create a DataFrame from the dictionary, ensuring column order matches training
    input_df = pd.DataFrame([feature_dict], columns=MODEL_COLUMNS)
    # --- END OF CHANGE ---

    # Predict probability -> returns [[P(loss), P(win)]]
    return MODEL.predict_proba(input_df)[0][1]


def predict_winner_normalized(team_a, team_b, map_name, best_of):
    """
    Predicts win probability using Symmetrical Normalization.
    This function's logic was correct and remains unchanged.
    """
    if not all([MODEL, TEAM_STATS, MODEL_COLUMNS]):
        return 0.5

    prob_a_raw = _get_single_prediction(team_a, team_b, map_name, best_of)
    prob_b_raw = _get_single_prediction(team_b, team_a, map_name, best_of)
    total_prob = prob_a_raw + prob_b_raw

    if total_prob == 0:
        return 0.5

    final_prob_a = prob_a_raw / total_prob
    return final_prob_a


def main():
    """Main function to run the prediction CLI."""
    if not all([MODEL, TEAM_STATS, MODEL_COLUMNS]):
        return

    print("\n--- CatBoost Esports Match Winner Prediction ---")
    print("Enter 'exit' at any prompt to quit.")

    while True:
        team_a = input("\nEnter name for Team A: ").strip()
        if team_a.lower() == 'exit': break

        team_b = input("Enter name for Team B: ").strip()
        if team_b.lower() == 'exit': break

        map_name = input("Enter Map Name (e.g., Icebox, Haven): ").strip()
        if not map_name:  # Handle empty input for map
            map_name = 'N/A'
        if map_name.lower() == 'exit': break

        best_of_num = input("Enter Best Of (1, 3, or 5): ").strip()
        if best_of_num.lower() == 'exit': break

        if best_of_num in ['1', '3', '5']:
            best_of = f"Bo{best_of_num}"
        else:
            print(f"[!] Invalid format '{best_of_num}'. Defaulting to Bo3.")
            best_of = 'Bo3'

        print("\nCalculating...")

        prob_a_wins = predict_winner_normalized(team_a, team_b, map_name, best_of)
        prob_b_wins = 1 - prob_a_wins

        print("\n--- Prediction Results ---")
        print(f"Matchup: {team_a} vs. {team_b} on '{map_name}' ({best_of})")
        print(f"  > {team_a} Win Probability: {prob_a_wins:.2%}")
        print(f"  > {team_b} Win Probability: {prob_b_wins:.2%}")

        if team_a not in TEAM_STATS:
            print(f"  (Note: '{team_a}' is a new team with no historical data.)")
        if team_b not in TEAM_STATS:
            print(f"  (Note: '{team_b}' is a new team with no historical data.)")

        print("-" * 30)


if __name__ == '__main__':
    main()