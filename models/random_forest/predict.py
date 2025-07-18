import pandas as pd
import joblib

# Define file paths for the model
MODEL_FILE = 'model.joblib'
STATS_FILE = 'team_stats.joblib'
COLUMNS_FILE = 'model_columns.joblib'


def load_model_and_data():
    """Loads the trained model, team stats, and model columns from disk."""
    try:
        model = joblib.load(MODEL_FILE)
        team_stats = joblib.load(STATS_FILE)
        model_columns = joblib.load(COLUMNS_FILE)
        print("Model and data loaded successfully.")
        return model, team_stats, model_columns
    except FileNotFoundError:
        print(f"Error: Model files not found. Please run the training script first.")
        return None, None, None


# --- NEW HELPER FUNCTION ---
# This function performs a single, one-sided prediction. It's the core
# logic that will be called twice.
def _get_single_prediction(model, team_stats, model_columns, team_a, team_b, best_of_str, map_name):
    """
    Builds a feature vector for A vs B and returns the model's raw probability of A winning.
    """
    default_stats = {'win_rate': 0.5, 'avg_acs_advantage': 0, 'avg_kdr_advantage': 0, 'avg_assists_advantage': 0}
    team_a_stats = team_stats.get(team_a, default_stats)
    team_b_stats = team_stats.get(team_b, default_stats)

    # Feature differences are calculated based on the specific order of team_a vs team_b
    feature_dict = {
        'win_rate_diff': team_a_stats['win_rate'] - team_b_stats['win_rate'],
        'acs_adv_diff': team_a_stats['avg_acs_advantage'] - team_b_stats['avg_acs_advantage'],
        'kdr_adv_diff': team_a_stats['avg_kdr_advantage'] - team_b_stats['avg_kdr_advantage'],
        'assists_adv_diff': team_a_stats['avg_assists_advantage'] - team_b_stats['avg_assists_advantage'],
        'best_of': best_of_str,
        'map_name': map_name,
    }

    input_df = pd.DataFrame([feature_dict])
    input_df_encoded = pd.get_dummies(input_df)

    # Align columns with the training data to handle new maps/BoX formats
    input_df_aligned = input_df_encoded.reindex(columns=model_columns, fill_value=0)

    # Return the raw probability of the first team (team_a) winning
    return model.predict_proba(input_df_aligned)[0][1]


# --- REFACTORED MAIN PREDICTION FUNCTION ---
# This function now orchestrates the symmetrical prediction and prints the result.
def predict_match(model, team_stats, model_columns, user_team_a, user_team_b, best_of_str, map_name):
    """
    Predicts the outcome by averaging predictions from both perspectives to ensure symmetry.
    """
    # Issue warnings for new teams (only needs to be done once)
    if user_team_a not in team_stats:
        print(f"Warning: Team '{user_team_a}' not in historical data. Using default stats.")
    if user_team_b not in team_stats:
        print(f"Warning: Team '{user_team_b}' not in historical data. Using default stats.")

    # 1. Get prediction from the first perspective: P(A wins | A vs B)
    prob_a_wins_raw = _get_single_prediction(model, team_stats, model_columns, user_team_a, user_team_b, best_of_str,
                                             map_name)

    # 2. Get prediction from the second perspective: P(B wins | B vs A)
    prob_b_wins_raw = _get_single_prediction(model, team_stats, model_columns, user_team_b, user_team_a, best_of_str,
                                             map_name)

    # 3. Apply the Symmetrical Averaging formula
    #    (A's chance from A's view + A's chance from B's view) / 2
    final_prob_a = (prob_a_wins_raw + (1 - prob_b_wins_raw)) / 2
    final_prob_b = 1 - final_prob_a

    # 4. Display the final, symmetrical results
    print("\n--- Prediction (Symmetrical Average) ---")
    print(f"Match: {user_team_a} vs. {user_team_b} ({best_of_str})")
    print(f"Map: {map_name or 'Not specified'}")
    print("--------------------")
    print(f"Win Probability for {user_team_a}: {final_prob_a:.2%}")
    print(f"Win Probability for {user_team_b}: {final_prob_b:.2%}")
    print("--------------------\n")


def main():
    """Main function to run the prediction loop."""
    model, team_stats, model_columns = load_model_and_data()
    if not model: return
    print("\nWelcome to the Focused Esports Match Predictor!")
    print("This model considers team form (recency) and map.")

    while True:
        print("\nEnter match details (or type 'exit' to quit).")
        team_a = input("Enter Team A: ").strip()
        if team_a.lower() == 'exit': break

        team_b = input("Enter Team B: ").strip()
        if team_b.lower() == 'exit': break

        while True:
            best_of_num = input("Enter Best Of (1, 3, or 5): ").strip()
            if best_of_num in ['1', '3', '5']:
                best_of_str = f"Bo{best_of_num}"
                break
            else:
                print("Invalid input. Please enter 1, 3, or 5.")

        map_name = input("Enter Map Name (e.g., Ascent, Haven) or press Enter to skip: ").strip()

        # This function call now points to our new orchestrator
        predict_match(model, team_stats, model_columns, team_a, team_b, best_of_str, map_name)


if __name__ == "__main__":
    main()