from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd

import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

df = pd.read_csv('https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv')

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)


param_grid = {
    'n_estimators': [10, 50, 100, 200, 300],
    'max_depth': [None,5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
}

mlflow.set_experiment('HyperParameter Tuning_3_params')

with mlflow.start_run():
  model = RandomForestClassifier()

  grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
  
  grid_search.fit(X_train, y_train)
  
  print(grid_search.best_params_)
  print(grid_search.best_score_)

  # Parameters logging
  mlflow.log_params(grid_search.best_params_)
  mlflow.log_metric('best_score', grid_search.best_score_)

  # data logging

  train_df = X_train
  train_df['Outcome'] = y_train

  test_df = X_test
  test_df['Outcome'] = y_test

  # Convert DataFrames to MLflow datasets
  train_dataset = mlflow.data.from_pandas(train_df)
  test_dataset = mlflow.data.from_pandas(test_df)

  # Log the datasets
  mlflow.log_input(train_dataset, 'train_data')  
  mlflow.log_input(test_dataset, 'test_data')    

  # Code logging
  mlflow.log_artifact(__file__)

  # Model logging
  mlflow.sklearn.log_model(grid_search.best_estimator_, 'model')






