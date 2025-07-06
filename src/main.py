from data_preprocessing import load_data, preprocess_data, train_test_split_data
from logistic_model import LogisticRegressionGD
from rf_model import train_rf, evaluate_rf
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load and preprocess data
df = load_data("/Users/anushkahadkhale/Desktop/Ml_project/Early_detection_of_parkinsons/data/parkinsons/parkinsons.data")
X_scaled, y, scaler = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split_data(X_scaled, y)

# Train logistic model
model = LogisticRegressionGD(lr=0.01, epochs=2000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Custom Logistic Regression:")
print(classification_report(y_test, y_pred))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d')
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

# Train Random Forest
rf = train_rf(X_train, y_train)
acc, rf_preds = evaluate_rf(rf, X_test, y_test)
print("Random Forest Accuracy:", acc)
