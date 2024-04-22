# Import des librairies
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# On charge les données
df = pd.read_csv("ds_salaries.csv")

# Prétraitment des données
# Définition des variables catégorielles et numérique
numerical_features = ["work_year", "salary", "remote_ratio"]
categorial_features = ["experience_level", "employment_type", "job_title", "salary_currency", 
                       "employee_residence", "company_location", "company_size"]

# Création du préprocesseur
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorial_features)
    ])

# Création du pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', XGBRegressor())])

# On définie les données explicative et la target
X = df.drop('salary_in_usd', axis=1)
y = df['salary_in_usd']

# La grille de paramètres que notre modèle utiliseras
params = {
    "regressor__n_estimators": [200, 300, 500],
    "regressor__max_depth": [6, 8, 10],
    "regressor__learning_rate": [0.01, 0.05, 0.1],
    "regressor__random_state": [8888]
}

grid_search = GridSearchCV(pipeline, param_grid=params, cv=5, scoring='neg_mean_squared_error', verbose=0)

# On divise les données en jeu d'entrainement et de validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# On entraine le modèle
grid_search.fit(X_train, y_train)

# Meilleur modèle
best_model = grid_search.best_estimator_

# On fait les prédictions sur l'ensemble de test
y_pred = best_model.predict(X_test)

# Métriques d'évaluation pour notre modèle
r2 = r2_score(y_test, y_pred)
MSE = mean_squared_error(y_test, y_pred)

# Utilisation de MLFlow
mlflow.set_tracking_uri("http://localhost:5000")

mlflow.set_experiment("Salaries Prediction")

with mlflow.start_run():

    mlflow.log_params(params)

    mlflow.log_metric("r2", r2)
    mlflow.log_metric("MSE", MSE)

    mlflow.set_tag("Training Info", "XGBoost Regressor model for ds_salaries data")

    signature = infer_signature(X_train, best_model.predict(X_train))

    model_info = mlflow.sklearn.log_model(
    sk_model=best_model,
    artifact_path="XGBRegressor_model",
    signature=signature,
    input_example=X_train,
    registered_model_name="tracking-XGBoost",
)

loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)

ohe_categories = best_model.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(input_features=categorial_features)
feature_names = list(ohe_categories) + numerical_features

result = pd.DataFrame(X_test, columns=feature_names)
result["actual class"] = y_test
result["predicted class"] = predictions

print(result)