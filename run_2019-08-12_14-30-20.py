

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
df = spark.sql('select * from knime_datasets.queens_bronx').toPandas() 
target = "Price"
df[df.Price.notnull()]
df[df.Review_Scores_Rating5.notnull()]
df[df.Number_of_Records.notnull()]
df[df.Number_Of_Reviews.notnull()]
df[df.Review_Scores_Rating12.notnull()]
train, test = train_test_split(df)
train_x = train[["Review_Scores_Rating5","Number_of_Records","Number_Of_Reviews","Review_Scores_Rating12"]]
test_x = test[["Review_Scores_Rating5","Number_of_Records","Number_Of_Reviews","Review_Scores_Rating12"]]
train_y = train[["Price"]]
test_y = test[["Price"]]
height = [1, 2, 71, 1, 1, 1, 2, 16, 2, 4, 9, 41, 1, 1, 2, 17, 2, 11, 51, 1, 1, 43, 2, 1, 7, 23, 1, 1, 23, 2, 1, 3, 22, 2, 1, 2, 9, 1, 7, 71, 3, 1, 1, 1, 4, 25, 1, 8, 1, 4, 9, 1, 23, 1, 1, 2, 6, 5, 1, 1, 7, 7, 1, 10, 1, 1, 5, 30, 3, 1, 4, 1, 1, 1, 6, 1, 9, 1, 3, 1, 2, 1, 1, 1, 21, 1, 2, 1, 5, 1, 3, 2, 1, 2, 1, 2, 11, 1, 1, 2, 1, 11, 3, 1, 1, 2, 1, 1, 1, 2, 1, 1, 4, 1, 4, 1, 1, 5, 1, 2, 1, 7, 1, 1, 1, 3, 13, 1, 1, 1, 5, 16, 2, 2, 1, 1, 19, 1, 1, 1, 6, 11, 20, 3, 1, 1, 29, 1, 1, 3, 2, 17, 35, 4, 3, 2, 30, 2, 4, 6, 42]
bars = ['1,111', '1,500', '100', '101', '102', '103', '104', '105', '107', '108', '109', '110', '112', '113', '114', '115', '118', '119', '120', '123', '124', '125', '126', '127', '129', '130', '131', '132', '135', '137', '138', '139', '140', '142', '143', '144', '145', '148', '149', '150', '155', '156', '157', '158', '159', '160', '162', '165', '168', '169', '170', '172', '175', '176', '178', '179', '180', '185', '186', '188', '189', '190', '192', '195', '196', '198', '199', '200', '205', '209', '210', '212', '215', '219', '220', '222', '225', '229', '230', '232', '235', '238', '240', '244', '250', '255', '260', '270', '275', '277', '279', '280', '288', '290', '295', '299', '300', '305', '315', '325', '330', '350', '375', '380', '395', '4,000', '400', '420', '45', '450', '460', '475', '49', '5,000', '50', '500', '52', '55', '569', '59', '590', '60', '600', '625', '63', '64', '65', '66', '675', '68', '69', '70', '700', '71', '72', '725', '75', '750', '76', '77', '78', '79', '80', '81', '83', '84', '85', '850', '86', '87', '88', '89', '90', '92', '93', '94', '95', '950', '97', '98', '99']
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
	

