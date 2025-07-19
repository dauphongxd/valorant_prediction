# training_catboost.py

import pandas as pd
import numpy as np
# NEW: Import CatBoost
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import warnings

warnings.filterwarnings('ignore')


# The first two functions are identical to your XGBoost script
def prepare_data(df):
    """
    Transforms raw match data into a "team-centric" format suitable for
    calculating time-weighted stats.
    """
    df['match_date'] = pd.to_datetime(df['match_date'])
    df['kdr_diff'] = (df['kills_diff'] / df['deaths_diff'].abs()).fillna(0)
    stat_cols = ['acs_diff', 'kdr_diff', 'assists_diff']
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
    """
    stat_cols = {'acs_diff': 'acs', 'kdr_diff': 'kdr', 'assists_diff': 'assists'}
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
    Merges calculated time-weighted stats back into the original match dataframe
    to create the final features for the model.
    *** MODIFIED FOR CATBOOST: No more pd.get_dummies! ***
    """
    ewm_cols_to_shift = [col for col in full_stats_df.columns if col.startswith('ewm_')]
    shifted_stats = full_stats_df.groupby('team')[ewm_cols_to_shift].transform(lambda x: x.shift(1))
    shifted_stats.fillna(0, inplace=True)
    full_stats_df_shifted = pd.concat([full_stats_df[['match_date', 'team', 'opponent']], shifted_stats], axis=1)

    df = pd.merge(
        df, full_stats_df_shifted, how='left',
        left_on=['match_date', 'team_a', 'team_b'],
        right_on=['match_date', 'team', 'opponent']
    ).rename(columns={col: col + '_a' for col in shifted_stats.columns}).drop(columns=['team', 'opponent'])

    df = pd.merge(
        df, full_stats_df_shifted, how='left',
        left_on=['match_date', 'team_b', 'team_a'],
        right_on=['match_date', 'team', 'opponent']
    ).rename(columns={col: col + '_b' for col in shifted_stats.columns}).drop(columns=['team', 'opponent'])

    for stat in ['ewm_acs', 'ewm_kdr', 'ewm_assists']:
        df[f'feature_{stat}_diff'] = df[f'{stat}_a'] - df[f'{stat}_b']

    # --- CATBOOST CHANGE ---
    # We DO NOT one-hot encode. CatBoost will handle the raw columns.
    # We just need to make sure the categorical columns are of a type CatBoost understands (like string or category).
    df['map_name'] = df['map_name'].astype(str)
    df['best_of'] = df['best_of'].astype(str)

    return df


def main():
    print("--- CatBoost Model Training Pipeline ---")

    try:
        df = pd.read_csv('ready.csv')
    except FileNotFoundError:
        print("Error: ready.csv not found. Please place it in the same directory.")
        return
    df.drop(columns=['match_id', 'event_stage'], inplace=True)

    print("Step 1: Preparing data and calculating historical stats...")
    match_stats_df = prepare_data(df)
    full_stats_df = calculate_time_weighted_stats(match_stats_df)

    print("Step 2: Creating final feature set for the model (CatBoost optimized)...")
    feature_df = create_feature_dataset(df.copy(), full_stats_df)

    # --- CATBOOST CHANGE: Define categorical and numerical features ---
    categorical_features = ['map_name', 'best_of']
    numerical_features = [col for col in feature_df.columns if col.startswith('feature_')]
    feature_cols = numerical_features + categorical_features

    X = feature_df[feature_cols]
    y = feature_df['team_a_win']

    # Fill any potential NaN in categorical features for safety
    X[categorical_features] = X[categorical_features].fillna('N/A')

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Step 3: Training CatBoost model on {len(X_train)} samples...")

    # --- CATBOOST CHANGE: Instantiate and train CatBoostClassifier ---
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.02,
        depth=6,
        eval_metric='AUC',
        cat_features=categorical_features,  # Tell CatBoost which columns are categorical
        random_seed=42,
        verbose=0,  # Suppress iteration-by-iteration output
        early_stopping_rounds=50
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        plot=False  # Set to True if you want to see a learning curve plot
    )
    # --- END OF CHANGE ---

    y_pred_proba = model.predict_proba(X_val)[:, 1]
    accuracy = accuracy_score(y_val, y_pred_proba > 0.5)
    auc = roc_auc_score(y_val, y_pred_proba)

    print("\n--- Model Evaluation (on validation set) ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC Score: {auc:.4f}")
    print(f"Optimal number of trees: {model.get_best_iteration()}")

    print("\nStep 4: Saving model and prediction artifacts...")
    joblib.dump(model, 'catboost_model.pkl')

    # The team_stats artifact remains the same
    final_team_stats = full_stats_df.groupby('team').last()
    final_team_stats = final_team_stats[['ewm_acs', 'ewm_kdr', 'ewm_assists']].to_dict('index')
    joblib.dump(final_team_stats, 'catboost_team_stats.pkl')

    # The model_columns artifact now saves the simplified feature list
    joblib.dump(feature_cols, 'catboost_model_columns.pkl')

    print(
        "\nTraining complete. Artifacts saved: catboost_model.pkl, catboost_team_stats.pkl, catboost_model_columns.pkl")


if __name__ == '__main__':
    main()