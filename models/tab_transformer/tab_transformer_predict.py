# tab_transformer_predict.py

import pandas as pd
import numpy as np
import joblib
import torch
import torch.nn as nn
import warnings
import os
import sys  # <-- Import sys

# Import the TabTransformer class itself
from tab_transformer_pytorch import TabTransformer

warnings.filterwarnings('ignore')


# --- STEP 1: KEEP THE CLASS DEFINITION HERE ---
# The TabTransformerWrapper class must be defined here so that we have something
# to inject into the __main__ module.
class TabTransformerWrapper:
    def __init__(self, **kwargs):
        self.patience = kwargs.pop('patience', 20)
        self.max_epochs = kwargs.pop('max_epochs', 150)
        self.batch_size = kwargs.pop('batch_size', 1024)
        self.lr = kwargs.pop('lr', 2e-2)
        self.verbose = kwargs.pop('verbose', True)
        self.device = kwargs.pop('device_name', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.num_continuous = kwargs.get('num_continuous', 0)
        self.model = TabTransformer(**kwargs).to(self.device)
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    def fit(self, X_train, y_train, eval_set=None):
        pass  # Not needed for prediction

    def predict_proba(self, X_pred):
        self.model.eval()
        X_pred_cont = torch.tensor(X_pred[:, :self.num_continuous], dtype=torch.float).to(self.device)
        X_pred_cat = torch.tensor(X_pred[:, self.num_continuous:], dtype=torch.long).to(self.device)
        with torch.no_grad():
            preds = self.model(X_pred_cat, X_pred_cont)
            probs = torch.sigmoid(preds).cpu().numpy()
        return np.hstack([1 - probs, probs])


# This helper function is fine as is.
def _create_feature_vector(artifacts, team_a, team_b, best_of, map_name):
    # ... your existing _create_feature_vector logic ...
    # This function does not need any changes.
    global_stats = artifacts['global_stats']
    map_stats = artifacts['map_stats']
    h2h_stats = artifacts['h2h_stats']
    label_encoders = artifacts['label_encoders']
    num_features = artifacts['numerical_features']
    cat_features = artifacts['categorical_features']
    default_stats = {'ewm_acs': 0.0, 'ewm_kdr': 0.0, 'ewm_assists': 0.0}
    use_map_stats = map_name in map_stats and team_a in map_stats[map_name].index and team_b in map_stats[
        map_name].index
    stats_source = map_stats.get(map_name, global_stats) if use_map_stats else global_stats
    stats_a = stats_source.loc[team_a].to_dict() if team_a in stats_source.index else default_stats
    stats_b = stats_source.loc[team_b].to_dict() if team_b in stats_source.index else default_stats
    h2h_key = tuple(sorted((team_a, team_b)))
    h2h_win_rate = h2h_stats.get(h2h_key, 0.5)
    h2h_advantage = (h2h_win_rate if team_a == h2h_key[0] else 1 - h2h_win_rate) - 0.5
    feature_dict = {
        'ewm_acs_diff': stats_a['ewm_acs'] - stats_b['ewm_acs'],
        'ewm_kdr_diff': stats_a['ewm_kdr'] - stats_b['ewm_kdr'],
        'ewm_assists_diff': stats_a['ewm_assists'] - stats_b['ewm_assists'],
        'h2h_advantage': h2h_advantage,
        'map_name': map_name if map_name else '__UNKNOWN__',
        'best_of': best_of
    }
    input_df = pd.DataFrame([feature_dict])
    for col in cat_features:
        le = label_encoders[col]
        input_df[col] = input_df[col].astype(str).apply(lambda x: x if x in le.classes_ else '__UNKNOWN__')
        input_df[col] = le.transform(input_df[col])
    return input_df[num_features + cat_features].values, use_map_stats


def get_prediction(team_a, team_b, map_name, best_of, artifacts_dir='.'):
    """
    NEW UNIFIED ENTRY POINT
    """
    try:
        # --- STEP 2: THE "MONKEY PATCH" ---
        # We manually add our local TabTransformerWrapper class to the __main__ module's namespace.
        # This makes joblib think it has found the class in the place it expects.
        setattr(sys.modules['__main__'], 'TabTransformerWrapper', TabTransformerWrapper)

        artifact_path = os.path.join(artifacts_dir, 'tabtransformer_optimized_artifacts.pkl')
        artifacts = joblib.load(artifact_path)

    except (FileNotFoundError, AttributeError, KeyError) as e:
        # Added KeyError in case __main__ is funky, though unlikely.
        print(f"Error loading artifacts: {e}")
        return None

    model = artifacts['model']

    X_ab, _ = _create_feature_vector(artifacts, team_a, team_b, best_of, map_name)
    prob_a_wins_perspective_a = model.predict_proba(X_ab)[0][1]

    X_ba, _ = _create_feature_vector(artifacts, team_b, team_a, best_of, map_name)
    prob_b_wins_perspective_b = model.predict_proba(X_ba)[0][1]
    prob_a_wins_perspective_b = 1.0 - prob_b_wins_perspective_b

    final_prob_a_wins = (prob_a_wins_perspective_a + prob_a_wins_perspective_b) / 2.0
    return final_prob_a_wins


# The main function for standalone testing is fine as is.
if __name__ == '__main__':
    # When this script is run directly, __main__ already has the class,
    # so the patch is harmless and everything works.
    # main() function is omitted for brevity but should remain
    pass