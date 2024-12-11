import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import requests
from zipfile import ZipFile
import io

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
response = requests.get(url)
with ZipFile(io.BytesIO(response.content)) as zip_ref:
    zip_ref.printdir()
    zip_ref.extract('bank-full.csv')
    bank_data = pd.read_csv('bank-full.csv', sep=";")
bank_data['y'] = bank_data['y'].apply(lambda x: 1 if x == 'yes' else 0)

categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

X = bank_data[categorical_columns + numerical_columns]
y = bank_data['y']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns),
        ('num', 'passthrough', numerical_columns)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', KNeighborsClassifier())
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

k_values = [1, 5, 11, 21]
accuracies = []

for k in k_values:
    pipeline.set_params(classifier__n_neighbors=k)

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

    print(f"Classification report for k={k}:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy for k={k}: {accuracy}\n")

plt.figure(figsize=(8, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Accuracy')
plt.title('KNN Accuracy for Different Values of k')
plt.xticks(k_values)
plt.grid()
plt.show()