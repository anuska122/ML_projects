import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    df = pd.read_csv(path)
    df = df.drop(['name'], axis=1)
    return df

def preprocess_data(df):
    X = df.drop('status', axis=1)
    y = df['status']
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler

def train_test_split_data(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42)
