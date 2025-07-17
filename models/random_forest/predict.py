import pandas as pd
import joblib

# Define file paths for the model created by training_v2.py
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
        print(f"Error: Model files not found. Please run 'training_v2.py' first.")
        return None, None, None


def predict_match(model, team_stats, model_columns, user_team_a, user_team_b, best_of_str):
    """
    Predicts the outcome of a single match using a canonical ordering
    to ensure symmetrical results.
    """
    # --- THE CORE FIX: CANONICAL ORDERING ---
    # Sort the teams alphabetically to create a consistent matchup order.
    canonical_order = sorted([user_team_a, user_team_b])
    team_1 = canonical_order[0]
    team_2 = canonical_order[1]

    # Use a default dictionary for teams not in our historical data
    default_stats = {'win_rate': 0.5, 'avg_acs_advantage': 0, 'avg_kdr_advantage': 0}
    team_1_stats = team_stats.get(team_1, default_stats)
    team_2_stats = team_stats.get(team_2, default_stats)

    if user_team_a not in team_stats:
        print(f"Warning: Team '{user_team_a}' not in historical data. Using default stats.")
    if user_team_b not in team_stats:
        print(f"Warning: Team '{user_team_b}' not in historical data. Using default stats.")

    # Create feature differences based on the canonical order
    feature_dict = {
        'win_rate_diff': team_1_stats['win_rate'] - team_2_stats['win_rate'],
        'acs_adv_diff': team_1_stats['avg_acs_advantage'] - team_2_stats['avg_acs_advantage'],
        'kdr_adv_diff': team_1_stats['avg_kdr_advantage'] - team_2_stats['avg_kdr_advantage'],
        'best_of': best_of_str
    }

    # Standard prediction steps
    input_df = pd.DataFrame([feature_dict])
    input_df_encoded = pd.get_dummies(input_df)
    input_df_aligned = input_df_encoded.reindex(columns=model_columns, fill_value=0)

    # The model predicts the probability for the canonical team_1 winning
    prediction_proba = model.predict_proba(input_df_aligned)[0]

    # P(team_1 wins) is at index 1, P(team_1 loses) is at index 0
    team_1_win_prob = prediction_proba[1]
    team_2_win_prob = prediction_proba[0]

    # Map the results back to the user's original input order
    if user_team_a == team_1:
        user_team_a_prob = team_1_win_prob
        user_team_b_prob = team_2_win_prob
    else:  # user_team_a must be team_2
        user_team_a_prob = team_2_win_prob
        user_team_b_prob = team_1_win_prob

    # Display the final, consistent result
    print("\n--- Prediction ---")
    print(f"Match: {user_team_a} vs. {user_team_b} ({best_of_str})")
    print("--------------------")
    print(f"Win Probability for {user_team_a}: {user_team_a_prob:.2%}")
    print(f"Win Probability for {user_team_b}: {user_team_b_prob:.2%}")
    print("--------------------\n")


def main():
    """Main function to run the prediction loop."""
    model, team_stats, model_columns = load_model_and_data()
    if not model: return
    print("\nWelcome to the Esports Match Predictor!")

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
            elif best_of_num.lower() == 'exit':
                return
            else:
                print("Invalid input. Please enter 1, 3, or 5.")

        predict_match(model, team_stats, model_columns, team_a, team_b, best_of_str)


if __name__ == "__main__":
    main()