# catboost_training.py

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import warnings

warnings.filterwarnings('ignore')


def prepare_data(df):
    """
    Transforms raw match data into a "team-centric" format suitable for
    calculating time-weighted stats.
    """
    df['match_date'] = pd.to_datetime(df['match_date'])

    # Calculate KDR_diff per match, handling division by zero.
    # We use abs() on deaths_diff as it's always the negative of kills_diff.
    df['kdr_diff'] = (df['kills_diff'] / df['deaths_diff'].abs()).fillna(0)

    stat_cols = ['acs_diff', 'kdr_diff', 'assists_diff']

    # Create two dataframes: one for team_a's perspective, one for team_b's
    df_a = df[['match_date', 'team_a', 'team_b'] + stat_cols].copy()
    df_a.rename(columns={'team_a': 'team', 'team_b': 'opponent'}, inplace=True)

    df_b = df[['match_date', 'team_b', 'team_a'] + stat_cols].copy()
    # For team_b's perspective, the diffs must be inverted
    for col in stat_cols:
        df_b[col] = -df_b[col]
    df_b.rename(columns={'team_b': 'team', 'team_a': 'opponent'}, inplace=True)

    match_stats_df = pd.concat([df_a, df_b], ignore_index=True)
    match_stats_df.sort_values(by=['team', 'match_date'], inplace=True)

    return match_stats_df


def calculate_time_weighted_stats(match_stats_df, half_life_days=90):
    """
    Calculates the exponentially weighted moving average (EWM) for team stats.
    This gives more weight to recent matches.
    """
    # Rename diff columns to base stat names for clarity
    stat_cols = {'acs_diff': 'acs', 'kdr_diff': 'kdr', 'assists_diff': 'assists'}
    match_stats_df.rename(columns=stat_cols, inplace=True)

    ewm_stats = {}
    for stat in stat_cols.values():
        # Use .ewm with times for correct handling of irregular time intervals
        ewm_stats[f'ewm_{stat}'] = match_stats_df.groupby('team').apply(
            lambda x: x.set_index('match_date')[stat].ewm(halflife=f'{half_life_days}D', times=x['match_date']).mean()
        ).reset_index(name=f'ewm_{stat}')[f'ewm_{stat}']

    ewm_df = pd.DataFrame(ewm_stats)
    full_stats_df = pd.concat([match_stats_df.reset_index(drop=True), ewm_df], axis=1)

    return full_stats_df


def create_feature_dataset(df, full_stats_df):
    """
    Merges calculated time-weighted stats back into the original match dataframe
    to create the final features for the model.
    """
    ewm_cols_to_shift = [col for col in full_stats_df.columns if col.startswith('ewm_')]

    # Shift stats within each team's group to prevent data leakage from the current match
    shifted_stats = full_stats_df.groupby('team')[ewm_cols_to_shift].transform(lambda x: x.shift(1))
    shifted_stats.fillna(0, inplace=True)  # Fill NaNs for a team's first match

    full_stats_df_shifted = pd.concat([full_stats_df[['match_date', 'team', 'opponent']], shifted_stats], axis=1)

    # Merge stats for team_a
    df = pd.merge(
        df, full_stats_df_shifted, how='left',
        left_on=['match_date', 'team_a', 'team_b'],
        right_on=['match_date', 'team', 'opponent']
    ).rename(columns={col: col + '_a' for col in shifted_stats.columns}).drop(columns=['team', 'opponent'])

    # Merge stats for team_b
    df = pd.merge(
        df, full_stats_df_shifted, how='left',
        left_on=['match_date', 'team_b', 'team_a'],
        right_on=['match_date', 'team', 'opponent']
    ).rename(columns={col: col + '_b' for col in shifted_stats.columns}).drop(columns=['team', 'opponent'])

    # Create the final comparative features (difference in historical form)
    for stat in ['ewm_acs', 'ewm_kdr', 'ewm_assists']:
        df[f'feature_{stat}_diff'] = df[f'{stat}_a'] - df[f'{stat}_b']

    # One-hot encode categorical features
    df = pd.get_dummies(df, columns=['map_name', 'best_of'], prefix=['map', 'bo'], dtype=float)

    return df


def main():
    print("--- XGBoost Model Training Pipeline ---")

    # 1. Load Data
    try:
        df = pd.read_csv('ready.csv')
    except FileNotFoundError:
        print("Error: ready.csv not found. Please place it in the same directory.")
        return

    df.drop(columns=['match_id', 'event_stage'], inplace=True)

    # 2. Feature Engineering
    print("Step 1: Preparing data and calculating historical stats...")
    match_stats_df = prepare_data(df)
    full_stats_df = calculate_time_weighted_stats(match_stats_df)

    print("Step 2: Creating final feature set for the model...")
    feature_df = create_feature_dataset(df.copy(), full_stats_df)

    # 3. Model Training
    feature_cols = [col for col in feature_df.columns if
                    col.startswith('feature_') or col.startswith('map_') or col.startswith('bo_')]
    X = feature_df[feature_cols]
    y = feature_df['team_a_win']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Step 3: Training XGBoost model on {len(X_train)} samples...")

    # *** FIX IS HERE ***
    # Define the model with early stopping rounds as an init parameter
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        n_estimators=2000,
        learning_rate=0.02,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        seed=42,
        early_stopping_rounds=50  # Pass it here
    )

    # Fit the model, passing the validation set to eval_set for early stopping
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
    # *** END OF FIX ***

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    accuracy = accuracy_score(y_val, y_pred_proba > 0.5)
    auc = roc_auc_score(y_val, y_pred_proba)

    print("\n--- Model Evaluation (on validation set) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc:.4f}")
    # The best_iteration attribute is still available
    print(f"Optimal number of trees: {model.best_iteration}")

    # 4. Save Artifacts for Prediction
    print("\nStep 4: Saving model and prediction artifacts...")
    joblib.dump(model, 'xgb_model.pkl')

    final_team_stats = full_stats_df.groupby('team').last()
    final_team_stats = final_team_stats[['ewm_acs', 'ewm_kdr', 'ewm_assists']].to_dict('index')
    joblib.dump(final_team_stats, 'team_stats.pkl')

    joblib.dump(X_train.columns.tolist(), 'model_columns.pkl')

    print("\nTraining complete. Artifacts saved: xgb_model.pkl, team_stats.pkl, model_columns.pkl")


if __name__ == '__main__':
    main()