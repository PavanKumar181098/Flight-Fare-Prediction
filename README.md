# Flight-Fare-Prediction
# dataset: https://www.kaggle.com/datasets/nikhilmittal/flight-fare-prediction-mh
# Key features: 
datetime datatype, label encoding and one hot encoding, randomized search cv and grid search cv, train-test model, matpotlib plots, feature elimination. extra trees regressor for feature importances. Random Forest model, XGBoost and ridge regression models and heroku 

# Business Objective: 
This is a supervised machine learning project, the business objective was to predict the fare of an airplane ticket depending on 10 different factors: airline, date of journey, source, destination and route being some of those factors. The 11th column was price or fare of the airline ticket. 

# Data Visualizations: 
Firstly, all the necessary libraries were imported, initial study of the data like its size ( number of rows and columns), data type of the features, any null values. Secondly, we start with the EDA - as the size of the dataset is huge and there is also a lot of diversity of the input features, most of the eda was about mainpulating the data like changing the date of departure and date of arrival into date-time datatype and splitting into month and date. Splitting the arrival time into hours and minutes and dropping the arrival_date column, arrival_time, departure_time columns  that is dropping unnecessary features or dropping those features after they have been manipulated like converted into other features.  Handling categorical data by converting them into integers using function or label_encoder(). A lot of plots of were plotted like catplots( of price against city, to check how is the distribution of price against cities and aircraft companies).  Dropping columns like route and additional_info. One hot encoding is also done.  The training and testing dataset are separate and hence no splitting is required. The EDA that was done on training dataset is also done on the testing dataset.  

# Feature selection: 
heatmap is constructed with the input data to be plotted being the correaltion between the various features of the dataset and also the price. The extra trees regressor ensemble method was used to fined the feature importances and the top 20 features are plotted with total_stops being the most important feature.  

# Data Modelling: 
Random Forest algorithm is used, the train-test splitting is done on the training data set with test_size parmeter equal to 0.20. After the model is fit on the x_train and y_train, the model is tested on x_test and the predicted values are obtained and are plotted against the y_test values as a scatter plot. sklearn.metrics library is used to calculate the errors like mean absolute error(MAE), mean squared error(MSE) and root mean squared error(RMSE) and the r squared score is used to obtain the testing accuracy. R2 score is 0.79. Hyper parameter training is done using RandomizedSearchCV and GridSearchCV. The best parameters are loaded and the errors and r2 value is calculated again.  The next model used is XGBM Regressor. the r2 score is 0.84. After hyperparameter tuning the r2 value is the same. The next model is Ridge Regression model gives a r2 score of 0.62. Hence the finalized model is xgboost with an r2 score of 0.84. This model is then used to predict all the prices of the testing dataset. 

# This model is deployed using Heroku API.
