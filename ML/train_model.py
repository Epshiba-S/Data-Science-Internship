import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# 1. Load the data
df = pd.read_csv('titanic.csv')

# 2. Simple Preprocessing
# We pick the most important features for a basic UI
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = df[features].copy()
y = df['Survived']

# Handle missing values (Age is often missing)
X['Age'] = X['Age'].fillna(X['Age'].median())

# Convert 'Sex' to numbers: male = 0, female = 1
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# 3. Train the Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# 4. Save the Model to a file
# This creates a file called 'titanic_model.pkl' in your folder
with open('titanic_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model trained and saved successfully as 'titanic_model.pkl'!")

