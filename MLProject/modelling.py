import pandas as pd
import mlflow
import mlflow.sklearn
import os
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
    mlflow.autolog()
    X_train, y_train, X_test, y_test = load_data()

    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Accuracy: {acc:.4f}")

    # Mengambil ID dari run yang dibuat oleh mlflow run
    run = mlflow.active_run()
    if not run:
        run = mlflow.last_active_run()
        
    run_id = run.info.run_id
    print(f"[INFO] Run ID detected: {run_id}")
    
    # Menulis run_id.txt (akan muncul di folder MLProject/)
    with open("run_id.txt", "w") as f:
        f.write(run_id)

if __name__ == "__main__":
    train_basic()