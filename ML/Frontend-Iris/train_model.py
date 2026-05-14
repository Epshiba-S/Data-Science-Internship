# save_model.py
import pickle
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load and train
iris = datasets.load_iris()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(iris.data)
model = SVC(kernel='rbf', probability=True)
model.fit(X_scaled, iris.target)

# Save files using pickle
with open('svc_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)