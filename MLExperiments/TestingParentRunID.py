

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
df = spark.sql('select * from knime_datasets.bronx_statenisland').toPandas() 
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

N = 60
g1 = (0.6 + 0.6 * np.random.rand(N), np.random.rand(N))
g2 = (0.4+0.3 * np.random.rand(N), 0.5*np.random.rand(N))
g3 = (0.3*np.random.rand(N),0.3*np.random.rand(N))

data = (g1, g2, g3)
colors = ("red", "green", "blue")
groups = ("coffee", "tea", "water")

# Create plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

for data, color, group in zip(data, colors, groups):
	x,y = data
	ax.scatter(x, y, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

plt.title('Matplot scatter plot')
plt.legend(loc=2)
plt.show()
plt.savefig("target_count_plot.png")

alpha = 0.5
l1_ratio = 0.5
random_state = 42
max_iter = None
mlflow_run_name = 'ElasticNet' + '_0'

mlflow.set_tracking_uri("http://10.43.13.1:5000")
experiment_name = "TestingParentRunID"
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
	mlflow.log_artifact("target_count_plot.png")
	mlflow.sklearn.log_model(lr,".")

	runId = mlflow.active_run().info.run_id
	expId = mlflow.active_run().info.experiment_id
	artifact_uri = mlflow.active_run().info.artifact_uri
	

