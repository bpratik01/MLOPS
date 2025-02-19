import mlflow
import mlflow.sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = load_iris()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

max_depth = 5
num_estimators = 100

mlflow.set_experiment('iris-rf_AL')

with mlflow.start_run(run_name='rf_iris_md=5_ne=100'):
  model = DecisionTreeClassifier(max_depth=max_depth)

  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)

  mlflow.log_metric('accuracy', accuracy)
  mlflow.log_param('max_depth', max_depth)
  mlflow.log_param('num_estimators', num_estimators)
  
  #logging the data

  train_data = pd.DataFrame(X_train, columns=data.feature_names)
  train_data['target'] = y_train

  test_data = pd.DataFrame(X_test, columns=data.feature_names)
  test_data['target'] = y_test

  train_data = mlflow.data.from_pandas(train_data)
  test_data = mlflow.data.from_pandas(test_data)

  mlflow.log_input(train_data, 'train_data')
  mlflow.log_input(test_data, 'test_data')


  cm = confusion_matrix(y_test, y_pred)
  plt.figure(figsize=(6, 7))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
              xticklabels=data.target_names, 
              yticklabels=data.target_names, cbar=False)
  plt.xlabel('Predicted')
  plt.ylabel('Truth')
  plt.title('Confusion Matrix')
  plt.savefig('confusion_matrix.png')
  mlflow.log_artifact('confusion_matrix.png', 'images')

  mlflow.log_artifact(__file__, 'code')

  mlflow.sklearn.log_model(model, 'model')

  print(f'Accuracy: {accuracy}')

