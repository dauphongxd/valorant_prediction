import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- Global Variables ---
try:
    MODEL = joblib.load('xgb_model.pkl')
    TEAM_STATS = joblib.load('team_stats.pkl')
    MODEL_COLUMNS = joblib.load('model_columns.pkl')
    print("XGBoost model and artifacts loaded successfully.")
except FileNotFoundError:
    print("Error: Model artifacts not found. Please run training.py first.")
    MODEL, TEAM_STATS, MODEL_COLUMNS = None, None, None


def _get_single_prediction(team_a, team_b, map_name, best_of):
    """
    Internal helper to get a single, one-sided prediction from the model.
    Returns the raw probability of team_a winning.
    """
    default_stats = {'ewm_acs': 0, 'ewm_kdr': 0, 'ewm_assists': 0}

    stats_a = TEAM_STATS.get(team_a, default_stats)
    stats_b = TEAM_STATS.get(team_b, default_stats)

    # --- REFACTORED AND CORRECTED SECTION ---
    # Build a dictionary of features first for clarity and robustness.
    # The feature difference is correctly calculated based on the team_a and team_b
    # passed into this function call.
    feature_dict = {
        'feature_ewm_acs_diff': stats_a['ewm_acs'] - stats_b['ewm_acs'],
        'feature_ewm_kdr_diff': stats_a['ewm_kdr'] - stats_b['ewm_kdr'],
        'feature_ewm_assists_diff': stats_a['ewm_assists'] - stats_b['ewm_assists']
    }

    # Start with a zero-filled DataFrame with the correct columns
    input_df = pd.DataFrame(0, index=[0], columns=MODEL_COLUMNS)

    # Populate the feature values from the dictionary
    for key, value in feature_dict.items():
        if key in input_df.columns:
            input_df.loc[0, key] = value

    # Handle one-hot encoded categorical features
    map_col = f'map_{map_name}'
    if map_col in input_df.columns:
        input_df.loc[0, map_col] = 1.0

    bo_col = f'bo_{best_of}'
    if bo_col in input_df.columns:
        input_df.loc[0, bo_col] = 1.0
    # --- END OF CORRECTION ---

    # Predict probability -> returns [[P(loss), P(win)]]
    return MODEL.predict_proba(input_df)[0][1]


def predict_winner_normalized(team_a, team_b, map_name, best_of):
    """
    Predicts win probability using Symmetrical Normalization.
    This function's logic was correct and remains unchanged.
    """
    if not all([MODEL, TEAM_STATS, MODEL_COLUMNS]):
        return 0.5  # Default to neutral if artifacts are not loaded

    # 1. Get raw probability from Team A's perspective
    prob_a_raw = _get_single_prediction(team_a, team_b, map_name, best_of)

    # 2. Get raw probability from Team B's perspective
    prob_b_raw = _get_single_prediction(team_b, team_a, map_name, best_of)

    # 3. Apply Symmetrical Normalization
    total_prob = prob_a_raw + prob_b_raw

    # Handle the edge case where both scores are 0 to avoid DivisionByZeroError
    if total_prob == 0:
        return 0.5

    final_prob_a = prob_a_raw / total_prob
    return final_prob_a


def main():
    """Main function to run the prediction CLI."""
    if not all([MODEL, TEAM_STATS, MODEL_COLUMNS]):
        return

    print("\n--- XGBoost Esports Match Winner Prediction ---")
    print("Enter 'exit' at any prompt to quit.")

    while True:
        team_a = input("\nEnter name for Team A: ").strip()
        if team_a.lower() == 'exit': break

        team_b = input("Enter name for Team B: ").strip()
        if team_b.lower() == 'exit': break

        map_name = input("Enter Map Name (e.g., Icebox, Haven): ").strip()
        if map_name.lower() == 'exit': break

        best_of_num = input("Enter Best Of (1, 3, or 5): ").strip()
        if best_of_num.lower() == 'exit': break

        # 2. Format the number into the string the model needs (e.g., '3' -> 'Bo3')
        #    This safely handles any invalid input by defaulting to Bo3.
        if best_of_num in ['1', '3', '5']:
            best_of = f"Bo{best_of_num}"
        else:
            print(f"[!] Invalid format '{best_of_num}'. Defaulting to Bo3.")
            best_of = 'Bo3'

        print("\nCalculating...")

        # --- Use the Symmetrical Normalization function ---
        prob_a_wins = predict_winner_normalized(team_a, team_b, map_name, best_of)
        prob_b_wins = 1 - prob_a_wins

        print("\n--- Prediction Results ---")
        print(f"Matchup: {team_a} vs. {team_b} on '{map_name or 'N/A'}' ({best_of})")
        print(f"  > {team_a} Win Probability: {prob_a_wins:.2%}")
        print(f"  > {team_b} Win Probability: {prob_b_wins:.2%}")

        if team_a not in TEAM_STATS:
            print(f"  (Note: '{team_a}' is a new team with no historical data.)")
        if team_b not in TEAM_STATS:
            print(f"  (Note: '{team_b}' is a new team with no historical data.)")

        print("-" * 30)


if __name__ == '__main__':
    main()