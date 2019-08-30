

import matplotlib.pyplot as plt



import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import xgboost as xgb 
import mlflow
import mlflow.sklearn
import mlflow.models

def eval_metrics(actual, pred):
	rmse = np.sqrt(mean_squared_error(actual, pred))
	mae = mean_absolute_error(actual, pred)
	r2 = r2_score(actual, pred)
	return rmse, mae, r2 
df = spark.sql('select * from knime_datasets.tbl11').toPandas() 
target = "Price"
df['Price'] = pd.to_numeric(df['Price'],errors='coerce')
df['Review_Scores_Rating5'] = pd.to_numeric(df['Review_Scores_Rating5'],errors='coerce')
df['Review_Scores_Rating12'] = pd.to_numeric(df['Review_Scores_Rating12'],errors='coerce')
df = df.dropna(how='any',axis=0)
train, test = train_test_split(df)
train_x = train[['Review_Scores_Rating5', 'Review_Scores_Rating12']]
test_x = test[['Review_Scores_Rating5', 'Review_Scores_Rating12']]
train_y = train[["Price"]]
test_y = test[["Price"]]
height = [15, 2, 4, 2, 9, 1, 1, 9, 1, 2, 11, 1, 18, 1, 2, 9, 1, 1, 8, 2, 12, 1, 5, 3, 1, 23, 3, 1, 1, 2, 7, 5, 1, 3, 12, 1, 5, 1, 1, 1, 1, 1, 1, 2, 15, 2, 2, 1, 2, 1, 6, 2, 2, 3, 3, 1, 1, 1, 2, 1, 1, 1, 9, 1, 1, 2, 7, 3, 6, 1, 1, 1, 4, 2, 5, 1, 8]
bars = ['100', '101', '105', '109', '110', '111', '112', '115', '118', '119', '120', '122', '125', '128', '129', '130', '131', '133', '135', '139', '140', '143', '145', '148', '149', '150', '155', '156', '157', '159', '160', '165', '169', '170', '175', '179', '180', '185', '186', '188', '190', '195', '198', '199', '200', '205', '210', '215', '220', '222', '225', '235', '240', '250', '275', '285', '289', '299', '300', '325', '385', '480', '75', '76', '78', '79', '80', '83', '85', '86', '88', '89', '90', '92', '95', '98', '99']
y_pos = np.arange(len(bars))
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.ylabel("count")
plt.xlabel(target)
plt.show()
plt.savefig("target_count_plot.png")

alpha = 0
learning_rate = 0.3
colsample_bytree = 0.5
max_depth = 6
objective = 'reg:linear'
n_estimators = 1
subsample = 1
gamma = 0
reg_lambda = 1

mlflow.set_tracking_uri("http://10.43.13.1:5000")
experiment_name = "Airbnb_Regression"
mlflow.set_experiment(experiment_name)
with mlflow.start_run():
	xg_reg = xgb.XGBRegressor(objective =objective, colsample_bytree = colsample_bytree, learning_rate = learning_rate,max_depth = max_depth, alpha = alpha, n_estimators = n_estimators,gamma = gamma, reg_lambda=reg_lambda,subsample=subsample)
	xg_reg.fit(train_x, train_y)
	predicted_qualities = xg_reg.predict(test_x)
	(rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)
	
	print("XGBoost model")
	
	mlflow.log_param("objective", objective)
	mlflow.log_param("colsample_bytree", colsample_bytree)
	mlflow.log_param("learning_rate", learning_rate)
	mlflow.log_param("max_depth", max_depth)
	mlflow.log_param("alpha", alpha)
	mlflow.log_param("n_estimators", n_estimators)
	mlflow.log_param("gamma", gamma)
	mlflow.log_param("lambda", reg_lambda)
	mlflow.log_param("subsample", subsample)
	mlflow.log_param("Model","XGBoost")
	
	mlflow.log_metric("rmse", rmse)
	mlflow.log_metric("r2", r2)
	mlflow.log_metric("mae", mae)
	
	mlflow.log_artifact("target_count_plot.png")
	mlflow.sklearn.log_model(xg_reg,".")

	runId = mlflow.active_run().info.run_id
	expId = mlflow.active_run().info.experiment_id
	artifact_uri = mlflow.active_run().info.artifact_uri
	

