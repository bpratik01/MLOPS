from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import pandas as pd
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment('HyperParameter Tuning_3_params')

df = pd.read_csv('https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/master/diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

param_grid = {
    'n_estimators': [10, 50, 100, 200, 300],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
}

with mlflow.start_run(run_name="Best Experiment") as parent_run:
    model = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    
    grid_search.fit(X_train, y_train)
    
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    mlflow.log_params(best_params)
    mlflow.log_metric('best_score', best_score)
    
    train_df = X_train.copy()
    train_df['Outcome'] = y_train
    test_df = X_test.copy()
    test_df['Outcome'] = y_test
    train_csv = "train_data.csv"
    test_csv = "test_data.csv"
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    mlflow.log_artifact(train_csv, artifact_path="data")
    mlflow.log_artifact(test_csv, artifact_path="data")
    
    mlflow.log_artifact(__file__)
    
    cv_results = grid_search.cv_results_
    for i, params in enumerate(cv_results["params"]):
        with mlflow.start_run(run_name=f"Candidate Experiment {i}", nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score", cv_results["mean_test_score"][i])
            mlflow.log_metric("std_test_score", cv_results["std_test_score"][i])
    
    mlflow.sklearn.log_model(grid_search.best_estimator_, "model")
