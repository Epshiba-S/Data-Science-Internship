import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

print("📖 Starting backend process: Reading OlympicAthletes.csv...")
df = pd.read_csv("OlympicAthletes.csv")

# 1. Clean data & address missing numeric values
df['Age'] = df['Age'].fillna(df['Age'].median())

# Fill missing categorical text values so they don't break the encoder
df['Country'] = df['Country'].fillna('Unknown')
df['Sport'] = df['Sport'].fillna('Unknown')

# 2. Establish Supervised Target Goal (1 = Won Gold, 0 = No Gold)
df['Won_Gold'] = df['Gold Medals'].apply(lambda x: 1 if x > 0 else 0)

# 3. Feature Selection
features = ['Age', 'Country', 'Sport', 'Silver Medals', 'Bronze Medals']
X = df[features].copy()
y = df['Won_Gold']

# 4. Fit Label Encoders and handle string structures safely
encoders = {}
for col in ['Country', 'Sport']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

# 5. Train-Test Split (FIXED: altered 'test_test_size' to 'test_size')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Initialize Ensemble Algorithm
print("🤖 Fitting Supervised Random Forest Classifier Model...")
model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
model.fit(X_train, y_train)

# 7. Model Verification Output Check
y_pred = model.predict(X_test)
print("\n================ MODEL EVALUATION ================")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print("\nClassification Matrix Report:")
print(classification_report(y_test, y_pred))

# 8. Store binary pickle tracking structures locally
print("\n📦 Serialization phase: Saving artifacts to folder directory...")
with open('olympic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('encoders.pkl', 'wb') as f:
    pickle.dump(encoders, f)

print("🎉 Complete! 'olympic_model.pkl' and 'encoders.pkl' built successfully.")