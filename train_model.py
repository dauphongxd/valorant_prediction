import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# === Load the ML dataset ===
df = pd.read_csv('ml_dataset.csv')

# === Drop non-numeric and irrelevant columns ===
drop_cols = ['match_id', 'event_name', 'event_stage', 'match_date', 'patch',
             'team_a', 'team_b', 'winner', 'map_name']
df.drop(columns=drop_cols, inplace=True)

# === Separate features and label ===
X = df.drop(columns=['team_a_win'])
y = df['team_a_win']

# === Handle missing values if any ===
X.fillna(X.mean(), inplace=True)

# === Scale features ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Split dataset ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === Train Logistic Regression model ===
model = LogisticRegression()
model.fit(X_train, y_train)

# === Evaluate model ===
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.2%}")

# === Save model and scaler ===
joblib.dump(model, 'model_logistic.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("✅ Model saved to 'model_logistic.pkl'")
print("✅ Scaler saved to 'scaler.pkl'")
