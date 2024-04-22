import mlflow
from mlflow.models import infer_signature

import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("http://localhost:5000")

X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

mlflow.set_tracking_uri("http://127.0.0.1:5000")

mlflow.set_experiment("MLflow Quickstart")

with mlflow.start_run():

    mlflow.log_params(params)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1 score", f1)

    mlflow.set_tag("Training Info", "Basic LR model for iris data")

    signature = infer_signature(X_train, lr.predict(X_train))

    model_info = mlflow.sklearn.log_model(sk_model=lr,
