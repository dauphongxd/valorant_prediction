# predict.py

import pandas as pd
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- Global Variables ---
try:
    MODEL = joblib.load('lgbm_model.pkl')
    TEAM_STATS = joblib.load('team_stats.pkl')
    MODEL_COLUMNS = joblib.load('model_columns.pkl')
    print("Model and artifacts loaded successfully.")
except FileNotFoundError:
    print("Error: Model artifacts not found. Please run training.py first.")
    MODEL, TEAM_STATS, MODEL_COLUMNS = None, None, None


def _get_single_prediction(team_a: str, team_b: str, map_name: str, best_of: str):
    """
    Internal helper function to get a single, one-sided prediction.
    This is not guaranteed to be symmetrical.
    """
    if not all([MODEL, TEAM_STATS, MODEL_COLUMNS]):
        return 0.5  # Default to neutral if artifacts are not loaded

    # Default stats for teams not in our historical data
    default_stats = {'ewm_acs': 0, 'ewm_kills': 0, 'ewm_assists': 0}

    # Get stats for each team, falling back to default if team is new
    stats_a = TEAM_STATS.get(team_a, default_stats)
    stats_b = TEAM_STATS.get(team_b, default_stats)

    # --- Create Input DataFrame for the Model ---
    input_df = pd.DataFrame(0, index=[0], columns=MODEL_COLUMNS)

    # 1. Calculate and fill in the comparative 'form' features
    for stat in ['ewm_acs', 'ewm_kills', 'ewm_assists']:
        feature_name = f'feature_{stat}_diff'
        if feature_name in input_df.columns:
            input_df.loc[0, feature_name] = stats_a[stat] - stats_b[stat]

    # 2. Fill in the one-hot encoded categorical features
    map_col = f'map_{map_name}'
    if map_col in input_df.columns:
        input_df.loc[0, map_col] = 1.0

    bo_col = f'bo_{best_of}'
    if bo_col in input_df.columns:
        input_df.loc[0, bo_col] = 1.0

    input_df = input_df[MODEL_COLUMNS]
    win_proba = MODEL.predict_proba(input_df)[0][1]
    return win_proba


def predict_winner_symmetrical(team_a: str, team_b: str, map_name: str, best_of: str):
    """
    Predicts the win probability for team_a against team_b by averaging
    the model's prediction from both perspectives to ensure symmetry.

    Args:
        team_a (str): Name of the first team.
        team_b (str): Name of the second team.
        map_name (str): The map the match is played on.
        best_of (str): The match format (e.g., 'Bo1', 'Bo3').

    Returns:
        float: The symmetrical probability of team_a winning.
    """
    # 1. Prediction from Team A's perspective -> P(A wins)
    prob_a_wins_from_a = _get_single_prediction(team_a, team_b, map_name, best_of)

    # 2. Prediction from Team B's perspective -> P(B wins)
    prob_b_wins_from_b = _get_single_prediction(team_b, team_a, map_name, best_of)

    # 3. Infer P(A wins) from Team B's perspective
    prob_a_wins_from_b = 1 - prob_b_wins_from_b

    # 4. Average the two perspectives for a robust, symmetrical result
    symmetrical_prob = (prob_a_wins_from_a + prob_a_wins_from_b) / 2

    return symmetrical_prob


def main():
    """Main function to run the prediction CLI."""
    if not all([MODEL, TEAM_STATS, MODEL_COLUMNS]):
        return

    print("\n--- Esports Match Winner Prediction ---")
    print("Enter 'exit' at any prompt to quit.")

    while True:
        team_a = input("\nEnter name for Team A: ")
        if team_a.lower() == 'exit': break

        team_b = input("Enter name for Team B: ")
        if team_b.lower() == 'exit': break

        # Handle blank map input gracefully
        map_name = input("Enter Map Name (or leave blank if unknown): ")
        if map_name.lower() == 'exit': break

        # **UX IMPROVEMENT HERE**
        best_of = input("Enter Match Format (e.g., 1, 3, 5): ")
        if best_of.lower() == 'exit': break

        # Format input '3' into 'Bo3' for the model
        if best_of.isdigit():
            best_of = f"Bo{best_of}"

        print("\nCalculating...")

        # --- Use the new symmetrical prediction function ---
        prob_a_wins = predict_winner_symmetrical(team_a, team_b, map_name, best_of)
        prob_b_wins = 1 - prob_a_wins

        print("\n--- Prediction Results ---")
        print(f"Matchup: {team_a} vs. {team_b} on '{map_name or 'Unknown Map'}' ({best_of})")
        print(f"  > {team_a} Win Probability: {prob_a_wins:.2%}")
        print(f"  > {team_b} Win Probability: {prob_b_wins:.2%}")

        if team_a not in TEAM_STATS:
            print(f"  (Note: '{team_a}' is a new team with no historical data.)")
        if team_b not in TEAM_STATS:
            print(f"  (Note: '{team_b}' is a new team with no historical data.)")

        print("-" * 30)


if __name__ == '__main__':
    main()