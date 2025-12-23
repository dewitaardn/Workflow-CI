import pandas as pd
import mlflow
import mlflow.sklearn
import os
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "heartDisease_preprocessing")
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    return X_train, y_train, X_test, y_test

def train_basic():
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.autolog()
    X_train, y_train, X_test, y_test = load_data()

    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    time.sleep(5)

    run_id = os.environ.get("MLFLOW_RUN_ID")

    if not run_id:
        for _ in range(3):
            active_run = mlflow.active_run()
            if active_run:
                run_id = active_run.info.run_id
                break
            time.sleep(2)

    if run_id:
        print(f"[INFO] Run ID detected: {run_id}")
        with open("run_id.txt", "w") as f:
            f.write(run_id)
    else:
        raise RuntimeError("Gagal mendapatkan Run ID untuk workflow CI.")

if __name__ == "__main__":
    train_basic()