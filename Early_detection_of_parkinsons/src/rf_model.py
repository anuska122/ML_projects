from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_rf(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model
def evaluate_rf(model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    return acc, preds
