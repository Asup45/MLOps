# %%
import pandas as pd
import numpy as np
import missingno as msno
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV

# %%
# The initial dataset has been downloaded on the data folder.
# We open it with pandas

df = pd.read_csv('data/ds_salaries.csv')

# %%
df.info()

# %%
df.sample(10)

# %%
# list the different values of the column "salary_currency"
df['salary_currency'].value_counts()

# %%
usd_to_eur = 0.94  # 1 USD = 0.94 EUR
gbp_to_eur = 1.14  # 1 GBP = 1.14 EUR 
inr_to_eur = 0.011 # 1 INR = 0.011 EUR
cad_to_eur = 0.70  # 1 CAD = 0.70 EUR
aud_to_eur = 0.64  # 1 AUD = 0.64 EUR
sgd_to_eur = 0.71  # 1 SGD = 0.71 EUR [[10]]
brl_to_eur = 0.18  # 1 BRL = 0.18 EUR [[1]]
pln_to_eur = 0.22  # 1 PLN = 0.22 EUR [[2]] 
chf_to_eur = 1.01  # 1 CHF = 1.01 EUR
huf_to_eur = 0.0027 # 1 HUF = 0.0027 EUR [[2]]
dkk_to_eur = 0.13  # 1 DKK = 0.13 EUR [[2]]
jpy_to_eur = 0.0073 # 1 JPY = 0.0073 EUR [[2]]
try_to_eur = 0.051 # 1 TRY = 0.051 EUR [[2]]
thb_to_eur = 0.027 # 1 THB = 0.027 EUR 
ils_to_eur = 0.26  # 1 ILS = 0.26 EUR
hkd_to_eur = 0.12  # 1 HKD = 0.12 EUR
czk_to_eur = 0.041 # 1 CZK = 0.041 EUR [[2]]
mxn_to_eur = 0.052 # 1 MXN = 0.052 EUR
clp_to_eur = 0.0011 # 1 CLP = 0.0011 EUR


# %%
# create a new column "salary_eur" that contains the salary in EUR
df['salary_eur'] = df['salary']

# set type of df['salary_eur'] to float
df['salary_eur'] = df['salary_eur'].astype(float)

df.loc[df['salary_currency'] == 'USD', 'salary_eur'] = df['salary'] * usd_to_eur
df.loc[df['salary_currency'] == 'GBP', 'salary_eur'] = df['salary'] * gbp_to_eur
df.loc[df['salary_currency'] == 'INR', 'salary_eur'] = df['salary'] * inr_to_eur
df.loc[df['salary_currency'] == 'CAD', 'salary_eur'] = df['salary'] * cad_to_eur
df.loc[df['salary_currency'] == 'AUD', 'salary_eur'] = df['salary'] * aud_to_eur
df.loc[df['salary_currency'] == 'SGD', 'salary_eur'] = df['salary'] * sgd_to_eur
df.loc[df['salary_currency'] == 'BRL', 'salary_eur'] = df['salary'] * brl_to_eur
df.loc[df['salary_currency'] == 'PLN', 'salary_eur'] = df['salary'] * pln_to_eur
df.loc[df['salary_currency'] == 'CHF', 'salary_eur'] = df['salary'] * chf_to_eur
df.loc[df['salary_currency'] == 'HUF', 'salary_eur'] = df['salary'] * huf_to_eur
df.loc[df['salary_currency'] == 'DKK', 'salary_eur'] = df['salary'] * dkk_to_eur
df.loc[df['salary_currency'] == 'JPY', 'salary_eur'] = df['salary'] * jpy_to_eur
df.loc[df['salary_currency'] == 'TRY', 'salary_eur'] = df['salary'] * try_to_eur
df.loc[df['salary_currency'] == 'THB', 'salary_eur'] = df['salary'] * thb_to_eur
df.loc[df['salary_currency'] == 'ILS', 'salary_eur'] = df['salary'] * ils_to_eur
df.loc[df['salary_currency'] == 'HKD', 'salary_eur'] = df['salary'] * hkd_to_eur
df.loc[df['salary_currency'] == 'CZK', 'salary_eur'] = df['salary'] * czk_to_eur
df.loc[df['salary_currency'] == 'MXN', 'salary_eur'] = df['salary'] * mxn_to_eur
df.loc[df['salary_currency'] == 'CLP', 'salary_eur'] = df['salary'] * clp_to_eur


# %%
df.sample(10)

# %%
# drop columns "salary" and "salary_currency" and "salary_in_usd"
df = df.drop(columns=['salary', 'salary_currency', 'salary_in_usd'])

# %%
df.sample(10)

# %%
msno.matrix(df)

# %%
# Split the df dataframe into two dataframes : df_train and df_added (1000 lines in df_added, the rest in df_train. Chose lines randomly)
df_added = df.sample(1000)
df_train = df.drop(df_added.index)

df_added.to_csv('data/ds_salaries_added.csv', index=False)
df_train.to_csv('data/ds_salaries_train.csv', index=False)

# %%
# We now focus on the train dataset
df = pd.read_csv('data/ds_salaries_train.csv')

# %%
# Remove variables df_added and df_train
del df_added, df_train

# %%
df.info()

# %%
df.sample(10)

# %%
# Séparer les features (X) et la target (y)
X = df.drop('salary_eur', axis=1)
y = df['salary_eur']


# %%
# Créer un objet LabelEncoder
le = LabelEncoder()

for col in ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']:
    X[col] = le.fit_transform(X[col])

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)


# %%
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# %%
# print model performance
print('RMSE: {:.2f}'.format(rmse))
print('MAE: {:.2f}'.format(mae))
print('R^2: {:.2f}'.format(r2))


# %% [markdown]
# ### Avec une version "naive" de xgboost, le score n'est pas bon. Il va falloir optimiser les hyperparamètres.

# %%
param_grid = {
    'max_depth': [2, 3,4, 5,6, 7, 9],
    'learning_rate': [0.5,0.3,0.2,0.15, 0.1, 0.05],
    'n_estimators': [50, 100, 150, 200, 300, 400],
}


# %%
grid_search = GridSearchCV(estimator=XGBRegressor(objective='reg:squarederror'), 
                           param_grid=param_grid,
                           cv=5,  # 5-fold cross-validation
                           n_jobs=-1,  # utiliser tous les CPU
                           verbose=2)  # afficher les logs


# %%
grid_search.fit(X_train, y_train)


# %%
print(f"Meilleurs hyperparamètres : {grid_search.best_params_}")
print(f"Meilleur score de validation croisée : {grid_search.best_score_:.4f}")

# %%
y_pred = grid_search.predict(X_test)
print(f"RMSE sur l'ensemble de test : {mean_squared_error(y_test, y_pred, squared=False):.2f}")

# %%



