# predict_match.py

import pandas as pd
import joblib

STAT_COLS = ['rating', 'acs', 'kills', 'deaths', 'assists']

def load_data(csv_path):
    # 1) Read with low_memory=False
    df = pd.read_csv(csv_path, low_memory=False)

    # 2) Coerce stat columns to numeric
    for col in STAT_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3) Drop rows where any stat is missing
    df.dropna(subset=STAT_COLS + ['player_team'], inplace=True)
    return df

def get_team_avg_stats(df, team_name):
    team_df = df[df['player_team'].str.lower() == team_name.lower()]
    if team_df.empty:
        raise ValueError(f"❌ Team '{team_name}' not found in the dataset.")
    # Now mean() only sees floats
    return team_df[STAT_COLS].mean().tolist()

def predict_match(team_a, team_b, model, scaler, df):
    try:
        a_stats = get_team_avg_stats(df, team_a)
        b_stats = get_team_avg_stats(df, team_b)
    except ValueError as e:
        print(e)
        return

    features = a_stats + b_stats
    features_scaled = scaler.transform([features])
    pred = model.predict(features_scaled)[0]
    prob = model.predict_proba(features_scaled)[0]

    winner = team_a if pred == 1 else team_b
    print(f"\n🏆 Predicted Winner: {winner}")
    print(f"✅ Probabilities — {team_a}: {prob[1]:.2f}, {team_b}: {prob[0]:.2f}")

if __name__ == "__main__":
    # Load model, scaler, and cleaned data
    model  = joblib.load("logistic_model.pkl")
    scaler = joblib.load("scaler.pkl")
    df     = load_data("your_file_cleaned.csv")

    # Ask the user
    team_a = input("Enter Team A name: ").strip()
    team_b = input("Enter Team B name: ").strip()

    predict_match(team_a, team_b, model, scaler, df)
