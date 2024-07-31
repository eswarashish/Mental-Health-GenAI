import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

df = pd.read_csv('diabetes.csv')

df.isnull().sum()


x = df.drop(columns=['Outcome'])
y = df['Outcome']


normalizer = MinMaxScaler()
x_normalized = normalizer.fit_transform(x)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x_normalized)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.25, random_state=42)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

filename = 'rf_model.sav'
pickle.dump(rf, open(filename, 'wb'))
with open('normalizer.pkl', 'wb') as f:
    pickle.dump(normalizer, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
