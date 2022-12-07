import pandas as pd
from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv("Loan_Data.csv")

# print(df.isna().sum())
# print(df.head)

# df = df.drop(columns="Loan_ID")
# print(df.columns)

# PREPROCESSING
# 1 Split data
X = df.drop(columns="Loan_Status", axis=1)
y = df["Loan_Status"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# change value N/Y in y_train into 0/1
y_train = y_train.apply(lambda x: 1 if x == "Y" else 0)
y_test = y_test.apply(lambda x: 1 if x == "Y" else 0)

# print(X_train)
# print(X.dtypes)

num_cols = []
cat_cols = []

for i in range(len(X.dtypes)):
    if(X.dtypes[i] == "object"):
        cat_cols.append(X.dtypes.index[i])
    else:
        num_cols.append(X.dtypes.index[i])

# print(num_cols)
# print(cat_cols)

# MODELING
# Transformers Steps
# syntax dibawah menggantikan nilai missing value pada suatu data frame yg bertipe categorical
cat_transformer = Pipeline([
    ("c_i", SimpleImputer(strategy="most_frequent")),
    ("e", OneHotEncoder())
])

# syntax dibawah menggantikan nilai missing value pada suatu data frame yg bertipe numerical
num_transformer = Pipeline([
    ("n_i", SimpleImputer(strategy="mean"))
])

transformer = [
    ("n_t", num_transformer, num_cols),
    ("c_t", cat_transformer, cat_cols)
]

# print(transformer)
# Logistic Regression

model_lr = Pipeline([
        ("pre", ColumnTransformer(transformers=transformer)),
        ("model", LogisticRegression())
    ])

# print(model_lr)

model_lr.fit(X_train, y_test)
# model_lr.fit(X_test, y_test)
# print(classification_report(y_test, model_lr.predict(X_test)))

# joblib.dump(model_lr, "model_lr.joblib")


