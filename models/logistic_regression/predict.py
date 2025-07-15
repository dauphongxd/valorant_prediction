import pandas as pd
import joblib

def load_model(path='model.pkl'):
    pipeline, stats_df = joblib.load(path)
    return pipeline, stats_df

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

if __name__ == '__main__':
    team_a = input("Team A: ").strip()
    team_b = input("Team B: ").strip()

    pipe, stats_df = load_model()

    X_new = build_input(team_a, team_b, stats_df)
    p     = pipe.predict_proba(X_new)[0,1]
    winner = team_a if p > 0.5 else team_b

    print(f"\nP({team_a} beats {team_b}) = {p:.3f}")
    print("Predicted winner:", winner)
