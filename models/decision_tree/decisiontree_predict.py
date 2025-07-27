import pandas as pd
import joblib
import warnings
import os

warnings.filterwarnings('ignore')


# --- STANDARD PREDICTION LOGIC (FROM YOUR BEST SCRIPTS) ---

def _get_single_prediction(team_a, team_b, map_name, best_of, MODEL, TEAM_STATS, MODEL_COLUMNS):
    """Internal helper to get a single, one-sided prediction from the model."""
    default_stats = {'ewm_acs': 0, 'ewm_kdr': 0, 'ewm_assists': 0}
    stats_a = TEAM_STATS.get(team_a, default_stats)
    stats_b = TEAM_STATS.get(team_b, default_stats)

    feature_dict = {
        'feature_ewm_acs_diff': stats_a['ewm_acs'] - stats_b['ewm_acs'],
        'feature_ewm_kdr_diff': stats_a['ewm_kdr'] - stats_b['ewm_kdr'],
        'feature_ewm_assists_diff': stats_a['ewm_assists'] - stats_b['ewm_assists']
    }
    input_df = pd.DataFrame(0, index=[0], columns=MODEL_COLUMNS)
    for key, value in feature_dict.items():
        if key in input_df.columns:
            input_df.loc[0, key] = value
    map_col = f'map_{map_name}'
    if map_col in input_df.columns:
        input_df.loc[0, map_col] = 1.0
    bo_col = f'bo_{best_of}'
    if bo_col in input_df.columns:
        input_df.loc[0, bo_col] = 1.0

    return MODEL.predict_proba(input_df)[0][1]


def get_prediction(team_a, team_b, map_name, best_of, artifacts_dir='.'):
    """UNIFIED ENTRY POINT - Predicts win probability using Symmetrical Normalization."""
    try:
        model = joblib.load(os.path.join(artifacts_dir, 'model.pkl'))
        team_stats = joblib.load(os.path.join(artifacts_dir, 'team_stats.pkl'))
        model_columns = joblib.load(os.path.join(artifacts_dir, 'model_columns.pkl'))
    except FileNotFoundError:
        print("Error: Model artifacts not found. Please run the training script first.")
        return None

    prob_a_raw = _get_single_prediction(team_a, team_b, map_name, best_of, model, team_stats, model_columns)
    prob_b_raw = _get_single_prediction(team_b, team_a, map_name, best_of, model, team_stats, model_columns)
    total_prob = prob_a_raw + prob_b_raw

    if total_prob == 0: return 0.5
    return prob_a_raw / total_prob


def main():
    """Main function to run the prediction CLI."""
    print("\n--- Upgraded Decision Tree Match Predictor ---")
    print("Enter 'exit' at any prompt to quit.")

    # Pre-load stats just to check for new teams later
    try:
        team_stats = joblib.load(os.path.join('.', 'team_stats.pkl'))
    except FileNotFoundError:
        print("Warning: team_stats.pkl not found. Cannot check for new teams.")
        team_stats = {}

    while True:
        team_a = input("\nEnter name for Team A: ").strip()
        if team_a.lower() == 'exit': break
        team_b = input("Enter name for Team B: ").strip()
        if team_b.lower() == 'exit': break
        map_name = input("Enter Map Name (e.g., Ascent): ").strip()
        if map_name.lower() == 'exit': break
        best_of_num = input("Enter Best Of (1, 3, or 5): ").strip()
        if best_of_num.lower() == 'exit': break

        if best_of_num in ['1', '3', '5']:
            best_of = f"Bo{best_of_num}"
        else:
            print(f"[!] Invalid format '{best_of_num}'. Defaulting to Bo3.")
            best_of = 'Bo3'

        print("\nCalculating...")
        prob_a_wins = get_prediction(team_a, team_b, map_name, best_of, artifacts_dir='.')

        if prob_a_wins is not None:
            prob_b_wins = 1 - prob_a_wins
            print("\n--- Prediction Results ---")
            print(f"Matchup: {team_a} vs. {team_b} on '{map_name or 'N/A'}' ({best_of})")
            print(f"  > {team_a} Win Probability: {prob_a_wins:.2%}")
            print(f"  > {team_b} Win Probability: {prob_b_wins:.2%}")
            if team_a not in team_stats: print(f"  (Note: '{team_a}' is a new team with no historical data.)")
            if team_b not in team_stats: print(f"  (Note: '{team_b}' is a new team with no historical data.)")
            print("-" * 30)
        else:
            print("Prediction failed.")


if __name__ == '__main__':
    main()