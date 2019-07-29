from pandas import DataFrame
# Create empty table
# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn
import mlflow.models
#from mlflow.models import FlavorBackend


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2
warnings.filterwarnings("ignore")
np.random.seed(40)
#mlflow.set_tracking_uri("http://10.43.12.78:5000")
# Read the wine-quality csv file (make sure you're running this from the root of MLflow!)
wine_path = flow_variables['path']
data = pd.read_csv(wine_path)    
print(type(data))
# Split the data into training and test sets. (0.75, 0.25) split.
train, test = train_test_split(data)
# The predicted column is "quality" which is a scalar from [3, 9]
train_x = train.drop(["quality"], axis=1)
test_x = test.drop(["quality"], axis=1)
train_y = train[["quality"]]
test_y = test[["quality"]]
alpha = 0.5
l1_ratio =  0.5

output_table = DataFrame(columns = {'Run Id'})
os.environ['HADOOP_HOME'] = "/home/ubuntu/hadoop"
#mlflow.create_experiment("Server1","hdfs://spark-master:8020/mlflow")
mlflow.set_experiment("Server1")
with mlflow.start_run():
	print(mlflow.get_artifact_uri())
	lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
	lr.fit(train_x, train_y)
	predicted_qualities = lr.predict(test_x)
	
	(rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
	mlflow.log_param("alpha", alpha)
	mlflow.log_param("l1_ratio", l1_ratio)
	mlflow.log_metric("rmse", rmse)
	mlflow.log_metric("r2", r2)
	mlflow.log_metric("mae", mae)
	for epoch in range(0, 3):
		mlflow.log_metric(key="quality", value=2*epoch, step=epoch)
	mlflow.log_artifact(wine_path,".")
	mlflow.sklearn.log_model(lr,".")
	r = mlflow.active_run().info.run_uuid
	print("Model saved in run %s" % r)
	output_table = output_table.append({'Run Id':r},ignore_index=True)