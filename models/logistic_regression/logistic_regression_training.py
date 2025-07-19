import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# 1) Load & clean
df = pd.read_csv('ready.csv')
df.columns = df.columns.str.strip()

# 2) Build a per‐team average of each _diff
teams = set(df['team_a']).union(df['team_b'])
acc = {t: {'acs': [], 'kills': [], 'deaths': [], 'assists': []} for t in teams}

for _, r in df.iterrows():
    a, b = r['team_a'], r['team_b']
    for stat in ['acs','kills','deaths','assists']:
        d = r[f'{stat}_diff']
        acc[a][stat].append(d)     # team_a sees +d
        acc[b][stat].append(-d)    # team_b sees −d

stats_df = pd.DataFrame({
    t: {stat: np.mean(acc[t][stat]) for stat in acc[t]}
    for t in acc
}).T

# 3) Create one feature‐vector per match: everything flips under A↔B
rows = []
for _, r in df.iterrows():
    a, b, y = r['team_a'], r['team_b'], r['team_a_win']
    # stat‐diff features
    stat_diff = {
        f'{stat}_diff': stats_df.at[a,stat] - stats_df.at[b,stat]
        for stat in ['acs','kills','deaths','assists']
    }
    # team‐diff one‐hots
    team_diff = {
        f'team_{t}': (1 if t==a else -1 if t==b else 0)
        for t in stats_df.index
    }
    row = {**stat_diff, **team_diff}
    rows.append((row, y))

X = pd.DataFrame([r for r,y in rows])
y = pd.Series([y for r,y in rows])

# 4) Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42
)

# 5) Build a pipeline that
#    - (Optionally) rescales the _diffs without centering (mean=0 remains)
#    - No intercept on the logistic reg so σ(−z)=1−σ(z)
numeric_cols = [c for c in X.columns if c.endswith('_diff')]

preprocessor = ColumnTransformer([
    ('scale_diffs', StandardScaler(with_mean=False), numeric_cols)
], remainder='passthrough')

pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(
        fit_intercept=False,
        max_iter=500,
        solver='lbfgs'
    )),
])

# 6) Train & evaluate
pipeline.fit(X_train, y_train)
print("Train acc:", pipeline.score(X_train, y_train))
print("Test  acc:", pipeline.score(X_test,  y_test))

# 7) Save model + stats
joblib.dump((pipeline, stats_df), 'model.pkl')
print("Saved invariant model to model.pkl")
