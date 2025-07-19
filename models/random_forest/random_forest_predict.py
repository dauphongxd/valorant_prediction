import pandas as pd
import joblib
import os


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


def get_prediction(team_a, team_b, map_name, best_of, artifacts_dir='.'):
    """
    NEW UNIFIED ENTRY POINT
    Predicts the outcome by averaging predictions from both perspectives.
    """
    try:
        model = joblib.load(os.path.join(artifacts_dir, 'model.joblib'))
        team_stats = joblib.load(os.path.join(artifacts_dir, 'team_stats.joblib'))
        model_columns = joblib.load(os.path.join(artifacts_dir, 'model_columns.joblib'))
    except FileNotFoundError:
        print("Error: Model files not found.")
        return None

    prob_a_wins_raw = _get_single_prediction(model, team_stats, model_columns, team_a, team_b, best_of, map_name)
    prob_b_wins_raw = _get_single_prediction(model, team_stats, model_columns, team_b, team_a, best_of, map_name)

    final_prob_a = (prob_a_wins_raw + (1 - prob_b_wins_raw)) / 2
    return final_prob_a


def main():
    """Main function to run the prediction loop."""
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
        final_prob_a = get_prediction(team_a, team_b, map_name or 'N/A', best_of_str, artifacts_dir='.')

        if final_prob_a is not None:
            final_prob_b = 1 - final_prob_a
            print("\n--- Prediction (Symmetrical Average) ---")
            print(f"Match: {team_a} vs. {team_b} ({best_of_str})")
            print(f"Map: {map_name or 'Not specified'}")
            print("--------------------")
            print(f"Win Probability for {team_a}: {final_prob_a:.2%}")
            print(f"Win Probability for {team_b}: {final_prob_b:.2%}")
            print("--------------------\n")
        else:
            print("Prediction failed.")


if __name__ == "__main__":
    main()