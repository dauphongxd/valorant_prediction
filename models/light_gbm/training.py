# training.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import warnings

warnings.filterwarnings('ignore')


def prepare_data(df):
    """
    Transforms the raw match data into a format suitable for calculating
    time-weighted team statistics.
    """
    df['match_date'] = pd.to_datetime(df['match_date'])
    stat_cols = ['acs_diff', 'kills_diff', 'assists_diff']

    df_a = df[['match_date', 'team_a', 'team_b'] + stat_cols].copy()
    df_a.rename(columns={'team_a': 'team', 'team_b': 'opponent'}, inplace=True)

    df_b = df[['match_date', 'team_b', 'team_a'] + stat_cols].copy()
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
    stat_cols = {'acs_diff': 'acs', 'kills_diff': 'kills', 'assists_diff': 'assists'}
    match_stats_df.rename(columns=stat_cols, inplace=True)

    ewm_stats = {}
    for stat in stat_cols.values():
        ewm_stats[f'ewm_{stat}'] = match_stats_df.groupby('team').apply(
            lambda x: x.set_index('match_date')[stat].ewm(halflife=f'{half_life_days}D', times=x['match_date']).mean()
        ).reset_index(name=f'ewm_{stat}')[f'ewm_{stat}']

    ewm_df = pd.DataFrame(ewm_stats)
    full_stats_df = pd.concat([match_stats_df.reset_index(drop=True), ewm_df], axis=1)

    return full_stats_df


def create_feature_dataset(df, full_stats_df):
    """
    Merges the calculated time-weighted stats back into the original match dataframe
    to create the final features for the model.
    """
    # *** FIX IS HERE ***
    # To prevent data leakage, we use the stats *before* the current match.
    # We only shift the EWM columns, not the key columns.
    ewm_cols_to_shift = [col for col in full_stats_df.columns if col.startswith('ewm_')]

    # Apply the shift only to the desired statistical columns
    shifted_stats = full_stats_df.groupby('team')[ewm_cols_to_shift].transform(lambda x: x.shift(1))

    # Fill NaN values for a team's first match with 0 (no prior form)
    shifted_stats.fillna(0, inplace=True)

    # Concatenate the original, un-shifted keys with the shifted stats. This avoids duplicate columns.
    full_stats_df_shifted = pd.concat([full_stats_df[['match_date', 'team', 'opponent']], shifted_stats], axis=1)

    # *** END OF FIX ***

    # Merge team_a's historical stats
    df = pd.merge(
        df,
        full_stats_df_shifted,
        how='left',
        left_on=['match_date', 'team_a', 'team_b'],
        right_on=['match_date', 'team', 'opponent']
    )
    df.rename(columns={col: col + '_a' for col in shifted_stats.columns}, inplace=True)
    df.drop(columns=['team', 'opponent'], inplace=True)

    # Merge team_b's historical stats
    df = pd.merge(
        df,
        full_stats_df_shifted,
        how='left',
        left_on=['match_date', 'team_b', 'team_a'],
        right_on=['match_date', 'team', 'opponent']
    )
    df.rename(columns={col: col + '_b' for col in shifted_stats.columns}, inplace=True)
    df.drop(columns=['team', 'opponent'], inplace=True)

    # Create the final comparative features (difference in form)
    for stat in ['ewm_acs', 'ewm_kills', 'ewm_assists']:
        df[f'feature_{stat}_diff'] = df[f'{stat}_a'] - df[f'{stat}_b']

    # Handle categorical features
    df = pd.get_dummies(df, columns=['map_name', 'best_of'], prefix=['map', 'bo'], dtype=float)

    return df


def main():
    print("Starting model training process...")

    # 1. Load Data
    try:
        df = pd.read_csv('ready.csv')
    except FileNotFoundError:
        print("Error: ready.csv not found. Please place it in the same directory.")
        return

    # Ignore specified columns
    df.drop(columns=['match_id', 'event_stage'], inplace=True)
    df['match_date'] = pd.to_datetime(df['match_date'])

    # 2. Feature Engineering Pipeline
    print("Step 1: Preparing data and calculating historical stats...")
    match_stats_df = prepare_data(df)
    full_stats_df = calculate_time_weighted_stats(match_stats_df)

    print("Step 2: Creating final feature set...")
    feature_df = create_feature_dataset(df.copy(), full_stats_df)

    # 3. Model Training
    # Define features (X) and target (y)
    feature_cols = [col for col in feature_df.columns if
                    col.startswith('feature_') or col.startswith('map_') or col.startswith('bo_')]
    X = feature_df[feature_cols]
    y = feature_df['team_a_win']

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Step 3: Training LightGBM model on {len(X_train)} samples...")

    # Define LightGBM parameters (tuned for good general performance)
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': 2000,
        'learning_rate': 0.02,
        'num_leaves': 20,
        'max_depth': 5,
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1,
        'colsample_bytree': 0.7,
        'subsample': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
    }

    model = lgb.LGBMClassifier(**params)

    # Train with early stopping
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='auc',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    # Evaluate model
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    accuracy = accuracy_score(y_val, y_pred_proba > 0.5)
    auc = roc_auc_score(y_val, y_pred_proba)

    print("\n--- Model Evaluation (on validation set) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc:.4f}")

    # 4. Save Artifacts for Prediction
    print("\nStep 4: Saving model and prediction artifacts...")

    # Save the trained model
    joblib.dump(model, 'lgbm_model.pkl')

    # Save the final state of team stats for new predictions
    final_team_stats = full_stats_df.groupby('team').last()
    final_team_stats = final_team_stats[['ewm_acs', 'ewm_kills', 'ewm_assists']].to_dict('index')
    joblib.dump(final_team_stats, 'team_stats.pkl')

    # Save the column order for the prediction script
    joblib.dump(X_train.columns.tolist(), 'model_columns.pkl')

    print("\nTraining complete. Artifacts saved: lgbm_model.pkl, team_stats.pkl, model_columns.pkl")


if __name__ == '__main__':
    main()