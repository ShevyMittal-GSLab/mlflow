

import matplotlib.pyplot as plt



import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
import mlflow.sklearn
import mlflow.models

def eval_metrics(actual, pred):
	rmse = np.sqrt(mean_squared_error(actual, pred))
	mae = mean_absolute_error(actual, pred)
	r2 = r2_score(actual, pred)
	return rmse, mae, r2 
df = spark.sql('select * from queens_bronx').toPandas() 
target = "Property_Type"
train, test = train_test_split(df)
train_x = train.drop(["Property_Type"], axis=1)
test_x = test.drop(["Property_Type"], axis=1)
train_y = train[["Property_Type"]]
test_y = test[["Property_Type"]]
height = [846, 3, 5, 3, 4, 6, 177, 11, 2, 4, 1]
bars = ['Apartment', 'Bed & Breakfast', 'Boat', 'Bungalow', 'Camper/RV', 'Condominium', 'House', 'Loft', 'Other', 'Townhouse', 'Villa']
y_pos = np.arange(len(bars))
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.ylabel("count")
plt.xlabel(target)
plt.show()
plt.savefig("plot.png")

alpha = 0.5
l1_ratio = 0.5
random_state = 42
max_iter = None
mlflow_run_name = 'ElasticNet' + '_0'

mlflow.set_tracking_uri("http://10.43.13.1:5000")
experiment_name = "ElasticNet_WineQuality"
mlflow.set_experiment(experiment_name)
with mlflow.start_run():
	lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
	lr.fit(train_x, train_y)
	
	predicted_qualities = lr.predict(test_x)
	
	(rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
	print("Elasticnet model (alpha=%f, l1_ratio=%f):" % (alpha, l1_ratio))
	print("  RMSE: %s" % rmse)
	print("  MAE: %s" % mae)
	print("  R2: %s" % r2)
	
	mlflow.log_param("alpha", alpha)
	mlflow.log_param("l1_ratio", l1_ratio)
	mlflow.log_metric("rmse", rmse)
	mlflow.log_metric("r2", r2)
	mlflow.log_metric("mae", mae)
	mlflow.log_artifact("plot.png")
	mlflow.sklearn.log_model(lr,".")

	runId = mlflow.active_run().info.run_id
	expId = mlflow.active_run().info.experiment_id
	artifact_uri = mlflow.active_run().info.artifact_uri
	

