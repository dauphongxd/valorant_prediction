# training.py (Corrected for TabTransformer)

import pandas as pd
import numpy as np
from tab_transformer_pytorch import TabTransformer
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib
import optuna
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_FILE = 'ready.csv'
ARTIFACTS_FILE = 'tabtransformer_optimized_artifacts.pkl'
TIME_WEIGHT_HALF_LIFE_DAYS = 90
N_SPLITS_K_FOLD = 5
N_OPTUNA_TRIALS = 30


# --- HELPER WRAPPER CLASS FOR TRAINING ---
class TabTransformerWrapper:
    def __init__(self, **kwargs):
        # We'll extract training-specific params and model-specific params
        self.patience = kwargs.pop('patience', 20)
        self.max_epochs = kwargs.pop('max_epochs', 150)
        self.batch_size = kwargs.pop('batch_size', 1024)
        self.lr = kwargs.pop('lr', 2e-2)
        self.verbose = kwargs.pop('verbose', True)
        self.device = kwargs.pop('device_name', 'cuda' if torch.cuda.is_available() else 'cpu')

        # --- FIX #1: Store num_continuous ---
        # It's a model parameter, but we need it for slicing, so we store it separately.
        self.num_continuous = kwargs.get('num_continuous', 0)

        # The rest are model hyperparameters
        self.model = TabTransformer(**kwargs).to(self.device)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def fit(self, X_train, y_train, eval_set=None):
        # --- FIX #2: Use self.num_continuous to slice the data correctly ---
        # Continuous features are the first 'self.num_continuous' columns
        # Categorical features are the rest
        X_train_cont = torch.tensor(X_train[:, :self.num_continuous], dtype=torch.float).to(self.device)
        X_train_cat = torch.tensor(X_train[:, self.num_continuous:], dtype=torch.long).to(self.device)
        y_train = torch.tensor(y_train, dtype=torch.float).unsqueeze(1).to(self.device)

        train_dataset = TensorDataset(X_train_cat, X_train_cont, y_train)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        if eval_set:
            X_val, y_val = eval_set[0]
            X_val_cont = torch.tensor(X_val[:, :self.num_continuous], dtype=torch.float).to(self.device)
            X_val_cat = torch.tensor(X_val[:, self.num_continuous:], dtype=torch.long).to(self.device)
            y_val = torch.tensor(y_val, dtype=torch.float).unsqueeze(1).to(self.device)
            val_dataset = TensorDataset(X_val_cat, X_val_cont, y_val)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCEWithLogitsLoss()

        for epoch in range(self.max_epochs):
            self.model.train()
            for x_cat, x_cont, y in train_loader:
                optimizer.zero_grad()
                pred = self.model(x_cat, x_cont)
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()

            if eval_set:
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for x_cat, x_cont, y in val_loader:
                        pred = self.model(x_cat, x_cont)
                        val_loss += criterion(pred, y).item()
                val_loss /= len(val_loader)

                if self.verbose and (epoch % 10 == 0):
                    print(f"Epoch {epoch + 1}/{self.max_epochs}, Val Loss: {val_loss:.4f}")

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch + 1}")
                        break

    def predict_proba(self, X_pred):
        self.model.eval()
        # --- FIX #3: Use self.num_continuous to slice the data correctly here as well ---
        X_pred_cont = torch.tensor(X_pred[:, :self.num_continuous], dtype=torch.float).to(self.device)
        X_pred_cat = torch.tensor(X_pred[:, self.num_continuous:], dtype=torch.long).to(self.device)

        with torch.no_grad():
            preds = self.model(X_pred_cat, X_pred_cont)
            probs = torch.sigmoid(preds).cpu().numpy()

        return np.hstack([1 - probs, probs])


# --- The rest of the script is largely fine, but we must ensure the data order ---

def calculate_advanced_stats(df):
    # This function is unchanged...
    print("Step 1: Calculating advanced time-weighted statistics from original data...")
    df['match_date'] = pd.to_datetime(df['match_date'])
    if df['match_date'].duplicated().any():
        print("WARNING: Duplicate dates found, EWM stats may be incorrect if data was pre-augmented.")
    df['kdr_diff'] = (df['kills_diff'] / df['deaths_diff'].abs()).fillna(0)

    def get_ewm_stats(sub_df):
        stat_cols = ['acs_diff', 'kdr_diff', 'assists_diff']
        df_a = sub_df[['match_date', 'team_a', 'team_b'] + stat_cols].copy();
        df_a.rename(columns={'team_a': 'team', 'team_b': 'opponent'}, inplace=True)
        df_b = sub_df[['match_date', 'team_b', 'team_a'] + stat_cols].copy()
        for col in stat_cols: df_b[col] = -df_b[col]
        df_b.rename(columns={'team_b': 'team', 'team_a': 'opponent'}, inplace=True)
        team_df = pd.concat([df_a, df_b], ignore_index=True).sort_values(by=['team', 'match_date'])
        team_df.rename(columns={'acs_diff': 'acs', 'kdr_diff': 'kdr', 'assists_diff': 'assists'}, inplace=True)
        final_stats = {}
        for stat in ['acs', 'kdr', 'assists']:
            ewm = team_df.groupby('team').apply(lambda x: x[stat].ewm(halflife=f'{TIME_WEIGHT_HALF_LIFE_DAYS}D',
                                                                      times=x[
                                                                          'match_date']).mean() if not x.empty else pd.Series())
            final_stats[f'ewm_{stat}'] = ewm.groupby('team').last()
        return pd.DataFrame(final_stats)

    global_stats = get_ewm_stats(df)
    map_stats = {map_name: get_ewm_stats(map_df) for map_name, map_df in df.groupby('map_name')}
    df['h2h_key'] = df.apply(lambda row: tuple(sorted((row['team_a'], row['team_b']))), axis=1)
    h2h_stats = {}
    for key, group in df.groupby('h2h_key'):
        group = group.sort_values('match_date')
        perspective_team = key[0];
        group['perspective_win'] = (group['team_a'] == perspective_team) == group['team_a_win']
        ewm_win_rate = \
        group['perspective_win'].ewm(halflife=f'{TIME_WEIGHT_HALF_LIFE_DAYS}D', times=group['match_date']).mean().iloc[
            -1]
        h2h_stats[key] = ewm_win_rate
    print("Advanced stats calculation complete.")
    return global_stats, map_stats, h2h_stats


def create_feature_dataset(df, global_stats, map_stats, h2h_stats):
    # This function is unchanged...
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
        features = {'ewm_acs_diff': stats_a['ewm_acs'] - stats_b['ewm_acs'],
                    'ewm_kdr_diff': stats_a['ewm_kdr'] - stats_b['ewm_kdr'],
                    'ewm_assists_diff': stats_a['ewm_assists'] - stats_b['ewm_assists'], 'h2h_advantage': h2h_advantage,
                    'map_name': map_name, 'best_of': row['best_of']}
        rows.append(features)
    feature_df = pd.DataFrame(rows)
    feature_df['target'] = df['team_a_win'].values
    return feature_df


def main():
    print("--- TabTransformer Training Pipeline ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device.upper()}")

    # Stages 1-3 are unchanged
    df_original = pd.read_csv(DATA_FILE);
    df_original.drop(columns=['match_id', 'event_stage'], inplace=True, errors='ignore')
    global_stats, map_stats, h2h_stats = calculate_advanced_stats(df_original)
    model_df = create_feature_dataset(df_original, global_stats, map_stats, h2h_stats)
    print("\nStep 3: Augmenting final feature set for perfect symmetry...")
    categorical_features = ['map_name', 'best_of']
    numerical_features = [col for col in model_df.columns if col not in categorical_features + ['target']]
    model_df_flipped = model_df.copy();
    model_df_flipped[numerical_features] = -model_df_flipped[numerical_features];
    model_df_flipped['target'] = 1 - model_df_flipped['target']
    final_model_df = pd.concat([model_df, model_df_flipped], ignore_index=True).sample(frac=1, random_state=42)
    print(f"Symmetrical training data created. Total rows: {len(final_model_df)}")

    X = final_model_df[numerical_features + categorical_features]
    y = final_model_df['target']

    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder();
        all_categories = list(model_df[col].astype(str).unique()) + ['__UNKNOWN__'];
        le.fit(all_categories)
        X[col] = le.transform(X[col].astype(str));
        label_encoders[col] = le

    cat_dims = [len(le.classes_) for le in label_encoders.values()]
    num_continuous = len(numerical_features)

    # --- FIX #4: ENSURE COLUMN ORDER ---
    # The wrapper expects continuous features first, then categorical. We must enforce this order.
    X = X[numerical_features + categorical_features]

    print(f"\nStep 4: Starting Optuna hyperparameter search for TabTransformer ({N_OPTUNA_TRIALS} trials)...")

    def objective(trial):
        dim = trial.suggest_int("dim", 16, 64, step=8)
        depth = trial.suggest_int("depth", 2, 6)
        heads = trial.suggest_categorical("heads", [2, 4, 8])
        attn_dropout = trial.suggest_float("attn_dropout", 0.1, 0.5)
        ff_dropout = trial.suggest_float("ff_dropout", 0.1, 0.5)
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        transformer_params = dict(
            categories=tuple(cat_dims),
            num_continuous=num_continuous,
            dim=dim,
            depth=depth,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            lr=lr,
            device_name=device,
            verbose=False,
            patience=15,
            max_epochs=100
        )

        kf = KFold(n_splits=N_SPLITS_K_FOLD, shuffle=True, random_state=42)
        cv_scores = []
        for i, (train_idx, val_idx) in enumerate(kf.split(X.values, y.values)):
            X_train, y_train = X.values[train_idx], y.values[train_idx]
            X_val, y_val = X.values[val_idx], y.values[val_idx]

            model = TabTransformerWrapper(**transformer_params)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

            preds = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, preds)
            cv_scores.append(score)

            trial.report(score, i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return np.mean(cv_scores)

    study = optuna.create_study(direction="maximize", study_name="TabTransformer Optimization",
                                pruner=optuna.pruners.MedianPruner())
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS)

    print(f"Search complete. Best AUC: {study.best_value:.4f}")
    print("Best hyperparameters:", study.best_params)

    print("\nStep 5: Training final model with best hyperparameters on all data...")
    best_params = study.best_params
    final_params = dict(
        categories=tuple(cat_dims),
        num_continuous=num_continuous,
        dim=best_params['dim'],
        depth=best_params['depth'],
        heads=best_params['heads'],
        attn_dropout=best_params['attn_dropout'],
        ff_dropout=best_params['ff_dropout'],
        lr=best_params['lr'],
        device_name=device,
        verbose=True,
        patience=20,
        max_epochs=150
    )

    final_model = TabTransformerWrapper(**final_params)
    final_model.fit(X.values, y.values)

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