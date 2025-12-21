import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load dataset 
DATA_DIR = "heartDisease_preprocessing"

X_train = pd.read_csv(f"{DATA_DIR}/X_train.csv")
X_test  = pd.read_csv(f"{DATA_DIR}/X_test.csv")
y_train = pd.read_csv(f"{DATA_DIR}/Y_train.csv").values.ravel()
y_test  = pd.read_csv(f"{DATA_DIR}/Y_test.csv").values.ravel()

# baseline model
with mlflow.start_run():
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, "model")

    print("Accuracy:", acc)