import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# 1. Load Data
df = pd.read_csv('employee_data.csv')

# 2. Preprocessing & Encoding
# We save the encoders so we can use them later in the app
encoders = {}
for col in ['Gender', 'Education', 'Department']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 3. Define Features and Target
X = df.drop(['Employee_ID', 'Salary'], axis=1)
y = df['Salary']

# 4. Train the Model
# We use Random Forest as it usually gives the best results
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. Save the Model and Encoders to a file
# This 'picks' up the brain of your AI so the app can use it instantly
model_data = {
    "model": model,
    "encoders": encoders,
    "features": list(X.columns)
}

with open('salary_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("Success: Model trained and saved as 'salary_model.pkl'!")