

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
df = spark.sql('select * from knime_datasets.queens_dataset').toPandas() 
target = "Price"
df = df[df.Price.notnull()]
df['Price'] = pd.to_numeric(df['Price'],errors='coerce')
df = df[df.Review_Scores_Rating5.notnull()]
df['Review_Scores_Rating5'] = pd.to_numeric(df['Review_Scores_Rating5'],errors='coerce')
df = df[df.Beds.notnull()]
df['Beds'] = pd.to_numeric(df['Beds'],errors='coerce')
df = df[df.Number_Of_Reviews.notnull()]
df['Number_Of_Reviews'] = pd.to_numeric(df['Number_Of_Reviews'],errors='coerce')
df = df[df.Review_Scores_Rating12.notnull()]
df['Review_Scores_Rating12'] = pd.to_numeric(df['Review_Scores_Rating12'],errors='coerce')
train, test = train_test_split(df)
train_x = train[['Review_Scores_Rating5', 'Beds', 'Number_Of_Reviews', 'Review_Scores_Rating12']]
test_x = test[['Review_Scores_Rating5', 'Beds', 'Number_Of_Reviews', 'Review_Scores_Rating12']]
train_y = train[["Price"]]
test_y = test[["Price"]]
height = [1, 53, 2, 1, 9, 1, 1, 8, 1, 1, 5, 17, 16, 1, 2, 3, 1, 4, 1, 1, 1, 2, 1, 25, 1, 1, 5, 1, 2, 3, 1, 1, 1, 5, 1, 1, 1, 1, 1, 4, 2, 2, 2, 3, 15, 1, 2, 5, 6, 36, 2, 3, 8, 15, 53, 3, 14, 11, 8, 101, 7, 8, 6, 40, 98, 4, 9, 3, 10, 98, 5, 15, 17, 40, 135, 2, 6, 3, 6, 108, 5, 9, 10, 1, 39, 86, 2, 6, 4, 3, 126, 2, 3, 8, 32, 75, 1, 3, 2, 10, 45, 1, 2, 10, 22, 44, 1, 1, 3, 27, 3, 3, 1, 18]
bars = ['10', '100', '101', '103', '105', '108', '109', '110', '113', '114', '115', '120', '125', '127', '129', '130', '133', '135', '136', '137', '138', '140', '142', '150', '155', '158', '160', '174', '175', '180', '198', '199', '2,500', '200', '21', '212', '213', '22', '225', '25', '250', '26', '27', '29', '30', '300', '31', '32', '33', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '680', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '92', '93', '94', '95', '96', '97', '98', '99']
y_pos = np.arange(len(bars))
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.ylabel("count")
plt.xlabel(target)
plt.show()
plt.savefig("target_count_plot.png")

alpha = 10
learning_rate = 0.1
colsample_bytree = 0.3
max_depth = 3
objective = 'reg:linear'
n_estimators = 5
subsample = 0.1
gamma = 5
reg_lambda = 3

mlflow.set_tracking_uri("http://10.43.13.1:5000")
experiment_name = "XGBoost Regression"
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
	

