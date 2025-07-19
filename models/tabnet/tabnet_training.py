# catboost_training.py

import pandas as pd
import numpy as np
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import torch
import joblib
import optuna
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_FILE = 'ready.csv'
ARTIFACTS_FILE = 'tabnet_optimized_artifacts.pkl'
TIME_WEIGHT_HALF_LIFE_DAYS = 90
N_SPLITS_K_FOLD = 5
N_OPTUNA_TRIALS = 30


def calculate_advanced_stats(df):
    """
    Calculates stats based on the TRUE historical timeline.
    This function should only ever see the original, unduplicated data.
    """
    print("Step 1: Calculating advanced time-weighted statistics from original data...")
    df['match_date'] = pd.to_datetime(df['match_date'])
    # This check is a safeguard to ensure we're not using contaminated data
    if df['match_date'].duplicated().any():
        print("WARNING: Duplicate dates found, EWM stats may be incorrect if data was pre-augmented.")

    df['kdr_diff'] = (df['kills_diff'] / df['deaths_diff'].abs()).fillna(0)

    def get_ewm_stats(sub_df):
        stat_cols = ['acs_diff', 'kdr_diff', 'assists_diff']
        df_a = sub_df[['match_date', 'team_a', 'team_b'] + stat_cols].copy()
        df_a.rename(columns={'team_a': 'team', 'team_b': 'opponent'}, inplace=True)
        df_b = sub_df[['match_date', 'team_b', 'team_a'] + stat_cols].copy()
        for col in stat_cols: df_b[col] = -df_b[col]
        df_b.rename(columns={'team_b': 'team', 'team_a': 'opponent'}, inplace=True)
        team_df = pd.concat([df_a, df_b], ignore_index=True).sort_values(by=['team', 'match_date'])
        team_df.rename(columns={'acs_diff': 'acs', 'kdr_diff': 'kdr', 'assists_diff': 'assists'}, inplace=True)

        final_stats = {}
        for stat in ['acs', 'kdr', 'assists']:
            ewm = team_df.groupby('team').apply(
                lambda x: x[stat].ewm(halflife=f'{TIME_WEIGHT_HALF_LIFE_DAYS}D',
                                      times=x['match_date']).mean() if not x.empty else pd.Series()
            )
            final_stats[f'ewm_{stat}'] = ewm.groupby('team').last()
        return pd.DataFrame(final_stats)

    global_stats = get_ewm_stats(df)
    map_stats = {map_name: get_ewm_stats(map_df) for map_name, map_df in df.groupby('map_name')}

    df['h2h_key'] = df.apply(lambda row: tuple(sorted((row['team_a'], row['team_b']))), axis=1)
    h2h_stats = {}
    for key, group in df.groupby('h2h_key'):
        group = group.sort_values('match_date')
        perspective_team = key[0]
        group['perspective_win'] = (group['team_a'] == perspective_team) == group['team_a_win']
        ewm_win_rate = \
            group['perspective_win'].ewm(halflife=f'{TIME_WEIGHT_HALF_LIFE_DAYS}D',
                                         times=group['match_date']).mean().iloc[
                -1]
        h2h_stats[key] = ewm_win_rate

    print("Advanced stats calculation complete.")
    return global_stats, map_stats, h2h_stats


def create_feature_dataset(df, global_stats, map_stats, h2h_stats):
    """ Creates the feature dataset with relative differences. """
    print("Step 2: Engineering features from advanced stats...")
    rows = []
    default_stats = {'ewm_acs': 0.0, 'ewm_kdr': 0.0, 'ewm_assists': 0.0}

    for _, row in df.iterrows():
        team_a, team_b = row['team_a'], row['team_b']
        map_name = row['map_name']

        use_map_stats = map_name in map_stats and team_a in map_stats[map_name].index and team_b in map_stats[
            map_name].index
        stats_source = map_stats[map_name] if use_map_stats else global_stats

        stats_a = stats_source.loc[team_a].to_dict() if team_a in stats_source.index else default_stats
        stats_b = stats_source.loc[team_b].to_dict() if team_b in stats_source.index else default_stats

        h2h_key = tuple(sorted((team_a, team_b)))
        h2h_win_rate = h2h_stats.get(h2h_key, 0.5)
        h2h_advantage = (h2h_win_rate if team_a == h2h_key[0] else 1 - h2h_win_rate) - 0.5

        features = {
            'ewm_acs_diff': stats_a['ewm_acs'] - stats_b['ewm_acs'],
            'ewm_kdr_diff': stats_a['ewm_kdr'] - stats_b['ewm_kdr'],
            'ewm_assists_diff': stats_a['ewm_assists'] - stats_b['ewm_assists'],
            'h2h_advantage': h2h_advantage,
            'map_name': map_name, 'best_of': row['best_of']
        }
        rows.append(features)

    feature_df = pd.DataFrame(rows)
    feature_df['target'] = df['team_a_win'].values
    return feature_df


def main():
    print("--- Optimized TabNet Training Pipeline ---")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device.upper()}")

    # --- CORRECTED LOGIC: STAGE 1 - Use ORIGINAL data for stats and features ---
    df_original = pd.read_csv(DATA_FILE)
    df_original.drop(columns=['match_id', 'event_stage'], inplace=True, errors='ignore')

    # Calculate stats on the pristine, original data
    global_stats, map_stats, h2h_stats = calculate_advanced_stats(df_original)

    # Create the base feature set from the original data
    model_df = create_feature_dataset(df_original, global_stats, map_stats, h2h_stats)

    # --- CORRECTED LOGIC: STAGE 2 - Augment the FINAL FEATURE set ---
    print("\nStep 3: Augmenting final feature set for perfect symmetry...")
    categorical_features = ['map_name', 'best_of']
    numerical_features = [col for col in model_df.columns if col not in categorical_features + ['target']]

    # Create the flipped version
    model_df_flipped = model_df.copy()
    model_df_flipped[numerical_features] = -model_df_flipped[numerical_features]
    model_df_flipped['target'] = 1 - model_df_flipped['target']

    # Combine original and flipped feature sets
    final_model_df = pd.concat([model_df, model_df_flipped], ignore_index=True).sample(frac=1, random_state=42)
    print(f"Symmetrical training data created. Total rows: {len(final_model_df)}")

    X = final_model_df[numerical_features + categorical_features]
    y = final_model_df['target']

    # --- The rest of the pipeline now uses the correctly augmented data ---
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        # Use original df to find all possible categories to avoid data leakage
        all_categories = list(model_df[col].astype(str).unique()) + ['__UNKNOWN__']
        le.fit(all_categories)
        X[col] = le.transform(X[col].astype(str))
        label_encoders[col] = le

    cat_idxs = [X.columns.get_loc(col) for col in categorical_features]
    cat_dims = [len(le.classes_) for le in label_encoders.values()]

    print(f"\nStep 4: Starting Optuna hyperparameter search ({N_OPTUNA_TRIALS} trials with PRUNING)...")

    def objective(trial):
        mask_type = trial.suggest_categorical("mask_type", ["entmax", "sparsemax"])
        n_da = trial.suggest_int("n_da", 8, 32, step=4)
        n_steps = trial.suggest_int("n_steps", 3, 7)
        gamma = trial.suggest_float("gamma", 1.0, 2.0)
        n_shared = trial.suggest_int("n_shared", 1, 3)
        cat_emb_dim = trial.suggest_int("cat_emb_dim", 2, 8)

        tabnet_params = dict(
            n_d=n_da, n_a=n_da, n_steps=n_steps, gamma=gamma,
            cat_idxs=cat_idxs, cat_dims=cat_dims, cat_emb_dim=cat_emb_dim,
            n_independent=2, n_shared=n_shared, mask_type=mask_type,
            device_name=device, optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2), scheduler_params={"step_size": 10, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR, verbose=0
        )

        kf = KFold(n_splits=N_SPLITS_K_FOLD, shuffle=True, random_state=42)
        cv_scores = []
        for i, (train_idx, val_idx) in enumerate(kf.split(X.values, y.values)):
            X_train, y_train = X.values[train_idx], y.values[train_idx]
            X_val, y_val = X.values[val_idx], y.values[val_idx]

            model = TabNetClassifier(**tabnet_params)
            model.fit(
                X_train, y_train, eval_set=[(X_val, y_val)],
                patience=15, max_epochs=100, eval_metric=['auc']
            )
            preds = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, preds)
            cv_scores.append(score)

            trial.report(score, i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(cv_scores)

    study = optuna.create_study(direction="maximize", study_name="TabNet Optimization",
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    print(f"Search complete. Best AUC: {study.best_value:.4f}")
    print("Best hyperparameters:", study.best_params)

    print("\nStep 5: Training final model with best hyperparameters on all data...")
    best_params = study.best_params
    final_params = dict(
        n_d=best_params['n_da'], n_a=best_params['n_da'], n_steps=best_params['n_steps'],
        gamma=best_params['gamma'], cat_idxs=cat_idxs, cat_dims=cat_dims,
        cat_emb_dim=best_params['cat_emb_dim'], n_independent=2, n_shared=best_params['n_shared'],
        mask_type=best_params['mask_type'], device_name=device,
        optimizer_fn=torch.optim.Adam, optimizer_params=dict(lr=2e-2),
        scheduler_params={"step_size": 10, "gamma": 0.9}, scheduler_fn=torch.optim.lr_scheduler.StepLR, verbose=10
    )

    final_model = TabNetClassifier(**final_params)
    final_model.fit(X.values, y.values, max_epochs=100, patience=15, batch_size=1024)

    print("\nStep 6: Saving final model and all artifacts...")
    artifacts = {
        'model': final_model, 'global_stats': global_stats, 'map_stats': map_stats,
        'h2h_stats': h2h_stats, 'label_encoders': label_encoders,
        'numerical_features': numerical_features, 'categorical_features': categorical_features
    }
    joblib.dump(artifacts, ARTIFACTS_FILE)
    print(f"Pipeline complete. Artifacts saved to {ARTIFACTS_FILE}")


if __name__ == '__main__':
    main()