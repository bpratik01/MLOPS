import mlflow
import numpy
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



data = load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y)

max_depth = 20
n_estimators = 100

with mlflow.start_run():

  model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)

  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)

  mlflow.log_metric('accuracy', accuracy)
  mlflow.log_param('max_depth', max_depth)
  mlflow.log_param('n_estimatior', n_estimators)

  print(f'Accuracy: {accuracy}')

