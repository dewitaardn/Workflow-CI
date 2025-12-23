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
    
    # Memuat dataset
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test  = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).values.ravel()
    y_test  = pd.read_csv(os.path.join(data_dir, "y_test.csv")).values.ravel()
    return X_train, y_train, X_test, y_test

def train_basic():
    # Menyamakan lokasi database dengan workflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.autolog()
    
    X_train, y_train, X_test, y_test = load_data()

    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"[INFO] Accuracy: {acc:.4f}")

    # Jeda singkat agar database SQLite selesai menulis (lock release)
    time.sleep(2)

    # Mencari Run ID dengan Retry Logic
    run_id = None
    for i in range(5): # Coba 5 kali
        # Coba ambil ID dari environment variable bawaan mlflow run
        run_id = os.environ.get("MLFLOW_RUN_ID")
        
        if not run_id:
            active_run = mlflow.active_run()
            if active_run:
                run_id = active_run.info.run_id
        
        if not run_id:
            last_run = mlflow.last_active_run()
            if last_run:
                run_id = last_run.info.run_id
        
        if run_id:
            break
            
        print(f"[RETRY {i+1}] Mencari Run ID di database...")
        time.sleep(2)

    if run_id:
        print(f"[INFO] Run ID detected: {run_id}")
        # Menyimpan Run ID ke dalam file txt di dalam folder MLProject
        with open("run_id.txt", "w") as f:
            f.write(run_id)
    else:
        raise RuntimeError("Gagal mendapatkan Run ID setelah beberapa kali mencoba.")

if __name__ == "__main__":
    train_basic()