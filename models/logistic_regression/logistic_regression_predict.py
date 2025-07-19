import pandas as pd
import joblib
import os



def build_input(a, b, stats_df):
    # stat‐diff
    sd = {
        f'{stat}_diff': stats_df.at[a, stat] - stats_df.at[b, stat]
        for stat in ['acs','kills','deaths','assists']
    }
    # team‐diff
    td = {
        f'team_{t}': (1 if t==a else -1 if t==b else 0)
        for t in stats_df.index
    }
    return pd.DataFrame([ {**sd, **td} ])


def get_prediction(team_a, team_b, map_name, best_of, artifacts_dir='.'):
    """
    NEW UNIFIED ENTRY POINT.
    Note: map_name and best_of are ignored by this model.
    """
    try:
        model_path = os.path.join(artifacts_dir, 'model.pkl')
        pipeline, stats_df = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Could not load model from {model_path}")
        return None

    try:
        if team_a not in stats_df.index or team_b not in stats_df.index:
            print(f"Error: One or both teams not found in stats data.")
            return None

        X_new = build_input(team_a, team_b, stats_df)
        p = pipeline.predict_proba(X_new)[0, 1]
        return p
    except KeyError as e:
        print(f"Error: A team might be missing from the stats. Missing key: {e}")
        return None


if __name__ == '__main__':
    team_a = input("Team A: ").strip()
    team_b = input("Team B: ").strip()

    # UPDATED to use new entry point
    p = get_prediction(team_a, team_b, None, None, artifacts_dir='.')

    if p is not None:
        winner = team_a if p > 0.5 else team_b
        print(f"\nP({team_a} beats {team_b}) = {p:.3f}")
        print("Predicted winner:", winner)
    else:
        print("Prediction failed.")
