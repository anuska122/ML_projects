#Early Detection of Parkinson's Disease
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

df = pd.read_csv('/Users/anushkahadkhale/Desktop/Early Detection of Parkinson’s Disease/parkinsons/parkinsons.data')
print(df.head())
print(df.info())
#dropping the non-feature column
df =  df.drop(['name'],axis=1)

#data visualization and EDA
sns.countplot(x='status',data=df)
plt.title('Parkinson’s vs. Healthy')
plt.show()
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlations')
plt.show()

X= df.drop('status',axis=1)
y = df['status']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
#spliting data into train and test data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


#Using Logistic regression from scratch(using gd)
class logisticRegression:
    def __init__(self,lr=0.01,epochs=1000):
        self.lr = lr
        self.epochs = epochs
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def fit(self,X,y):
        self.m,self.n = X.shape
        self.w = np.zeros(self.n)
        self.b = 0
        for _ in range(self.epochs):
            linear_model = np.dot(X,self.w) + self.b
            predictions = self.sigmoid(linear_model)

            dw = (1/self.m) * np.dot(X.T,(predictions-y))
            db = (1/self.m) * np.sum(predictions-y)
            self.w -= self.lr * dw
            self.b -= self.lr * db
    def predict(self, X):
        linear_model = np.dot(X, self.w) + self.b
        probs = self.sigmoid(linear_model)
        return [1 if i > 0.5 else 0 for i in probs]
    
# 6. Train Custom Model
lr = logisticRegression(lr=0.01, epochs=2000)
lr.fit(X_train, y_train)
# 7. Predict
y_pred = lr.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

#here i have implemented how to analaysis if the person has parklinson or not using the custom made Logistic Regression model
new_patient = {
    'MDVP:Fo(Hz)': 119.992,
    'MDVP:Fhi(Hz)': 157.302,
    'MDVP:Flo(Hz)': 74.997,
    'MDVP:Jitter(%)': 0.00784,
    'MDVP:Jitter(Abs)': 0.00007,
    'MDVP:RAP': 0.00370,
    'MDVP:PPQ': 0.00554,
    'Jitter:DDP': 0.01109,
    'MDVP:Shimmer': 0.04374,
    'MDVP:Shimmer(dB)': 0.426,
    'Shimmer:APQ3': 0.02182,
    'Shimmer:APQ5': 0.03130,
    'MDVP:APQ': 0.02971,
    'Shimmer:DDA': 0.06545,
    'NHR': 0.02211,
    'HNR': 21.033,
    'RPDE': 0.414783,
    'DFA': 0.815285,
    'spread1': -4.813031,
    'spread2': 0.266482,
    'D2': 2.301442,
    'PPE': 0.284654
}
#converting into dataframe
X_new = pd.DataFrame([new_patient])
X_new_scaled = scaler.transform(X_new)

model = logisticRegression()
model.fit(X_train,y_train)
y_pred1 = model.predict(X_new_scaled)
print("Prediction:", "Parkinson's likely" if y_pred1[0] == 1 else "Healthy")

# Optional: Get probability (risk score)
prob = model.sigmoid(np.dot(X_new_scaled, model.w) + model.b)
print(f"Risk Probability: {prob[0]*100:.2f}%")

#using RandomForest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train,y_train)
y_prediction = rf.predict(X_test)
print(accuracy_score(y_prediction,y_test))

