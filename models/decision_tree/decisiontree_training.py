import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# --- Configuration ---
HALF_LIFE_DAYS = 90
MODEL_PATH = 'model.pkl'
DATA_PATH = 'ready.csv'

def calculate_team_stats(df):
    """Calculates the average stat differences for each team from a given set of matches."""
    teams = set(df['team_a']).union(set(df['team_b']))
    acc = {t: {'acs': [], 'kills': [], 'deaths': [], 'assists': []} for t in teams}

    for _, r in df.iterrows():
        a, b = r['team_a'], r['team_b']
        for stat in ['acs', 'kills', 'deaths', 'assists']:
            d = r[f'{stat}_diff']
            if a in acc: acc[a][stat].append(d)
            if b in acc: acc[b][stat].append(-d)

    return pd.DataFrame({
        t: {stat: np.mean(acc[t][stat]) if acc[t][stat] else 0 for stat in acc[t]}
        for t in acc if any(acc[t].values())
    }).T

# 1) Load & clean data
print("1. Loading and cleaning data...")
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df['match_date'] = pd.to_datetime(df['match_date'])

# 2) Calculate statistics
print("2. Calculating per-map and global team statistics...")
# a) Per-map statistics
unique_maps = df['map_name'].unique()
map_stats = {
    map_name: calculate_team_stats(df[df['map_name'] == map_name])
    for map_name in unique_maps
}
# b) Global (all-maps) statistics for fallback
global_stats = calculate_team_stats(df)
all_teams = set(df['team_a']).union(set(df['team_b']))

# 3) Create feature vectors (same as before)
print("3. Engineering features for each match...")
rows = []
reference_date = pd.to_datetime('today')
decay_rate = np.log(2) / HALF_LIFE_DAYS

for _, r in df.iterrows():
    a, b, y = r['team_a'], r['team_b'], r['team_a_win']
    current_map = r['map_name']
    stats_for_map = map_stats[current_map]
    if a not in stats_for_map.index or b not in stats_for_map.index:
        continue
    stat_diff = {f'{stat}_diff': stats_for_map.at[a, stat] - stats_for_map.at[b, stat] for stat in ['acs', 'kills', 'deaths', 'assists']}
    team_diff = {f'team_{t}': (1 if t == a else -1 if t == b else 0) for t in all_teams}
    days_since_match = (reference_date - r['match_date']).days
    recency_score = np.exp(-decay_rate * days_since_match)
    context_features = {'best_of': r['best_of'], 'recency_score': recency_score}
    row_features = {**stat_diff, **team_diff, **context_features}
    rows.append((row_features, y))

X = pd.DataFrame([r for r, y in rows])
y = pd.Series([y for r, y in rows])

# 4) Split data (same as before)
print("4. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 5) Build pipeline (same as before)
print("5. Building pipeline...")
categorical_features = ['best_of']
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)],
    remainder='passthrough'
)
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=10, random_state=42))
])

# 6) Train the model (same as before)
print("6. Training model...")
pipeline.fit(X_train, y_train)
print(f"  Train Accuracy: {pipeline.score(X_train, y_train):.4f}")
print(f"  Test Accuracy:  {pipeline.score(X_test, y_test):.4f}")

# 7) Save all necessary components, including the new global_stats
print(f"7. Saving model and associated data to {MODEL_PATH}...")
model_data_to_save = {
    'pipeline': pipeline,
    'map_stats': map_stats,
    'global_stats': global_stats,  # <-- ADDED THIS
    'all_teams': list(all_teams),
    'decay_rate': decay_rate,
    'training_reference_date': reference_date
}
joblib.dump(model_data_to_save, MODEL_PATH)
print("--- Training Complete ---")