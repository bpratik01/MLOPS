from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

data = load_iris()

X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

max_depth = 10

mlflow.set_experiment('iris-dt')  #if this experiment does not exist, it will be created

# with mlflow.start_run(experiment_id='0'):
# we can mention the exp id here by creating it via gui and then using it here

with mlflow.start_run(run_name='decision_tree_fixed_fig_size'):
    model = DecisionTreeClassifier(max_depth=max_depth)

    model.fit(X_train, y_train)
     
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_param('max_depth', max_depth)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,7))
    sns.heatmap(cm, annot= True, fmt= 'd', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names, cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Truth')

    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png', 'images')


    # to track code using mlflow we can use log_artifact
    mlflow.log_artifact('src/iris-dt.py')

    print(f'Accuracy: {accuracy}')

