import pandas as pd
import mlflow
import mlflow.sklearn
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "heartDisease_preprocessing")

    X_train_path = os.path.join(data_dir, "X_train.csv")
    X_test_path  = os.path.join(data_dir, "X_test.csv")
    y_train_path = os.path.join(data_dir, "y_train.csv")
    y_test_path  = os.path.join(data_dir, "y_test.csv")

    for p in [X_train_path, X_test_path, y_train_path, y_test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"[ERROR] File tidak ditemukan: {p}")

    X_train = pd.read_csv(X_train_path)
    X_test  = pd.read_csv(X_test_path)
    y_train = pd.read_csv(y_train_path).values.ravel()
    y_test  = pd.read_csv(y_test_path).values.ravel()

    return X_train, y_train, X_test, y_test


def train_basic_model(X_train, y_train, X_test, y_test):
    mlflow.autolog()

    with mlflow.start_run():
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    train_basic_model(X_train, y_train, X_test, y_test)