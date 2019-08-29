

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
df = spark.sql('select * from knime_datasets.queens').toPandas() 
target = "Price"
df = df[df.Price.notnull()]
df['Price'] = pd.to_numeric(df['Price'],errors='coerce')
df = df[df.Review_Scores_Rating5.notnull()]
df['Review_Scores_Rating5'] = pd.to_numeric(df['Review_Scores_Rating5'],errors='coerce')
df = df[df.Number_Of_Reviews.notnull()]
df['Number_Of_Reviews'] = pd.to_numeric(df['Number_Of_Reviews'],errors='coerce')
df = df[df.Review_Scores_Rating12.notnull()]
df['Review_Scores_Rating12'] = pd.to_numeric(df['Review_Scores_Rating12'],errors='coerce')
train, test = train_test_split(df)
train_x = train[['Review_Scores_Rating5', 'Number_Of_Reviews', 'Review_Scores_Rating12']]
test_x = test[['Review_Scores_Rating5', 'Number_Of_Reviews', 'Review_Scores_Rating12']]
train_y = train[["Price"]]
test_y = test[["Price"]]
height = [45, 3, 1, 4, 1, 1, 1, 10, 1, 4, 3, 2, 1, 1, 2, 1, 1, 26, 1, 3, 1, 2, 3, 2, 1, 1, 1, 1, 2, 1, 1053, 11, 24, 30, 21, 186, 9, 16, 43, 95, 485, 16, 18, 11, 16, 307, 11, 11, 27, 126, 673, 12, 16, 8, 25, 828, 9, 16, 28, 131, 428, 7, 23, 16, 9, 343, 10, 12, 9, 116, 431, 6, 23, 11, 23, 281, 12, 15, 24, 210, 1381, 9, 8, 11, 12, 159, 11, 16, 11, 109, 416, 5, 12, 11, 10, 248, 8, 10, 23, 109, 231, 4, 10, 5, 20, 679, 3, 11, 14, 108, 390, 3, 5, 3, 8, 246, 6, 13, 10, 99, 261, 4, 16, 3, 5, 287, 8, 13, 40, 346, 12, 1, 1, 2, 1, 8, 1, 1, 1, 2, 1, 1018, 9, 4, 4, 8, 63, 8, 10, 10, 36, 105, 8, 6, 1, 7, 106, 7, 5, 7, 32, 226, 2, 8, 4, 8, 371, 3, 2, 8, 48, 85, 5, 7, 5, 84, 5, 4, 5, 43, 138, 4, 2, 5, 1, 72, 9, 15, 114, 2, 722, 2, 3, 2, 38, 2, 3, 2, 18, 69, 1, 1, 10, 45, 1, 5, 5, 16, 2, 47, 2, 2, 1, 3, 197, 3, 3, 3, 20, 4, 70, 3, 4, 53, 2, 1, 7, 25, 4, 35, 1, 2, 3, 90, 1, 7, 112, 5, 1, 1, 1, 1, 1, 28, 384, 1, 8, 1, 2, 4, 7, 3, 18, 1, 1, 1, 12, 1, 6, 2, 40, 1, 2, 81, 1, 1, 1, 11, 3, 13, 4, 6, 1, 2, 3, 17, 1, 1, 1, 19, 1, 2, 1, 31, 65, 228, 3, 1, 1, 8, 7, 23, 1, 8, 1, 1, 8, 4, 3, 3, 70, 1, 4, 13, 18, 1, 1, 16, 2, 2, 5, 37, 18, 3, 41, 1, 5, 1, 56, 1, 2, 1, 181, 156, 1, 3, 2, 12, 3, 2, 8, 1, 19, 10, 1, 37, 2, 3, 7, 6, 2, 4, 1, 2, 9, 2, 1, 3, 1, 11, 187, 111, 1, 4, 3, 17, 3, 2, 4, 1, 24, 3, 1, 24, 2, 52, 6, 1, 96, 8, 1, 22, 29, 4, 1, 488, 111, 1, 1, 7, 3, 28, 6, 1, 6, 1, 20, 41, 4, 5, 9, 316, 50, 2, 17, 2, 22, 3, 10, 2, 42, 1, 3, 1, 123, 1, 12, 1, 18, 1, 522, 51, 13, 43, 1, 6, 1, 26, 1, 1, 40, 3, 1, 1, 4, 552, 29, 1, 27, 34, 3, 64, 1, 2, 191, 2, 5, 2, 12, 585, 29, 14, 1, 52, 1, 6, 18, 1, 39, 1, 2, 701, 33, 16, 1, 30, 2, 2, 70, 1, 1, 1, 173, 2, 3, 1, 684, 35, 18, 35, 2, 25, 1, 41, 1, 532, 12, 1, 41, 26, 3, 57, 1, 1, 220, 3, 3, 637, 18, 13, 31, 19, 25, 1, 492, 6, 32, 30, 2, 76, 1, 492, 3, 1, 5]
bars = ['1,000', '1,050', '1,065', '1,100', '1,170', '1,174', '1,195', '1,200', '1,239', '1,250', '1,300', '1,350', '1,356', '1,368', '1,400', '1,495', '1,499', '1,500', '1,550', '1,600', '1,650', '1,700', '1,750', '1,800', '1,850', '1,900', '1,990', '1,999', '10', '10,000', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '2,000', '2,250', '2,295', '2,486', '2,499', '2,500', '2,520', '2,599', '2,695', '2,750', '20', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '232', '233', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '247', '248', '249', '25', '250', '252', '253', '254', '255', '256', '257', '258', '259', '260', '262', '263', '264', '265', '266', '267', '268', '269', '27', '270', '271', '272', '273', '274', '275', '276', '277', '278', '279', '28', '280', '281', '282', '285', '286', '287', '288', '289', '29', '290', '291', '292', '294', '295', '296', '298', '299', '3,000', '3,100', '3,200', '3,390', '3,500', '3,750', '30', '300', '302', '305', '306', '307', '308', '309', '31', '310', '311', '312', '313', '315', '318', '319', '32', '320', '321', '323', '325', '326', '327', '328', '329', '33', '330', '333', '335', '336', '339', '34', '340', '342', '343', '344', '345', '346', '347', '348', '349', '35', '350', '355', '356', '357', '359', '36', '360', '361', '365', '366', '369', '37', '370', '372', '374', '375', '378', '379', '38', '380', '383', '384', '385', '386', '388', '389', '39', '390', '393', '395', '396', '397', '398', '399', '4,000', '4,500', '4,750', '40', '400', '401', '405', '409', '41', '410', '412', '415', '419', '42', '420', '422', '425', '428', '429', '43', '430', '432', '435', '437', '439', '44', '440', '444', '445', '446', '449', '45', '450', '454', '455', '459', '46', '460', '461', '465', '469', '47', '470', '472', '475', '479', '48', '480', '485', '49', '490', '492', '495', '499', '5,000', '5,999', '50', '500', '505', '509', '51', '510', '52', '520', '524', '525', '529', '53', '54', '540', '545', '549', '55', '550', '559', '56', '560', '57', '570', '575', '579', '58', '580', '585', '589', '59', '590', '595', '597', '599', '6,500', '60', '600', '61', '62', '620', '625', '626', '63', '630', '635', '64', '640', '645', '647', '649', '65', '650', '656', '66', '67', '675', '68', '680', '685', '69', '690', '695', '698', '699', '70', '700', '71', '715', '72', '723', '725', '73', '735', '74', '740', '749', '75', '750', '76', '765', '77', '770', '775', '78', '780', '785', '789', '79', '795', '799', '8,000', '80', '800', '81', '82', '825', '83', '830', '84', '840', '85', '850', '855', '86', '87', '875', '88', '880', '888', '89', '895', '899', '90', '900', '91', '92', '93', '94', '945', '95', '950', '96', '97', '975', '98', '985', '99', '995', '997', '999']
y_pos = np.arange(len(bars))
plt.bar(y_pos, height)
plt.xticks(y_pos, bars)
plt.ylabel("count")
plt.xlabel(target)
plt.show()
plt.savefig("target_count_plot.png")

alpha = 0
learning_rate = 0.3
colsample_bytree = 1
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
	

