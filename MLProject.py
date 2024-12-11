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



# adeel 

# Step 2: Load dataset
bank_data = pd.read_csv('bank-full.csv', sep=";")

# Step 3: Preprocess target column
bank_data['y'] = bank_data['y'].apply(lambda x: 1 if x == 'yes' else 0)

# Step 4: Define categorical and numerical columns
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

X = bank_data[categorical_columns + numerical_columns]
y = bank_data['y']

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
        ('num', 'passthrough', numerical_columns)
    ]
)

# Step 7: Define pipeline
pipeline = imPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Step 8: Define hyperparameter search space
param_distributions = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

# Step 9: RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions,
    n_iter=20,  # Number of parameter settings to sample
    cv=3,       # 3-fold cross-validation
    scoring='accuracy',
    n_jobs=-1,  # Use all CPU cores
    random_state=42
)

# Step 10: Fit the model
print("Fitting the model...")
random_search.fit(X_train, y_train)

# Step 11: Evaluate the model
print("Best hyperparameters:", random_search.best_params_)
print("Best cross-validation score:", random_search.best_score_)

y_pred = random_search.predict(X_test)
print("Classification report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Step 12: Plot performance for number of estimators
results = random_search.cv_results_

plt.figure(figsize=(8, 6))
mean_test_scores = []
n_estimators_values = param_distributions['classifier__n_estimators']

for n_estimators in n_estimators_values:
    mask = [param['classifier__n_estimators'] == n_estimators for param in results['params']]
    scores = np.array(results['mean_test_score'])[mask]
    mean_test_scores.append(np.mean(scores))

plt.plot(n_estimators_values, mean_test_scores, marker='o', linestyle='-', color='g')
plt.xlabel('Number of Estimators')
plt.ylabel('Mean Accuracy')
plt.title('Random Forest Accuracy for Different Number of Estimators')
plt.grid()
plt.show()










import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as imPipeline
from sklearn.model_selection import GridSearchCV
from zipfile import ZipFile
import requests
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), categorical_columns),
                  ('num', 'passthrough', numerical_columns)])

pipeline = imPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)), 
    ('classifier', XGBClassifier())
])

param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [3, 5, 7],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

grid_search.fit(X_train, y_train)

print("Best hyperparameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

y_pred = grid_search.predict(X_test)

print("Classification report:")
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

results = grid_search.cv_results_

plt.figure(figsize=(8, 6))

mean_test_scores = []
n_estimators_values = param_grid['classifier__n_estimators']

for n_estimators in n_estimators_values:
    scores = results['mean_test_score'][results['param_classifier__n_estimators'] == n_estimators]
    mean_test_scores.append(np.mean(scores)) 

plt.plot(n_estimators_values, mean_test_scores, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Estimators')
plt.ylabel('Mean Accuracy')
plt.title('XGBoost Accuracy for Different Number of Estimators')
plt.grid()
plt.show()