





import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn.datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score

data=pd.read_csv(r'C:\Users\Jose\Desktop\insurance.csv')

print (data)

data.head()

data.info()

data.isnull().sum()

sns.countplot(x='smoker',data=data)

sns.countplot(x='smoker',hue='region',data=data)

sns.countplot(x='smoker',hue='sex',data=data)

sns.countplot(x='smoker',hue='children',data=data)

data['age'].plot.hist()

sns.countplot(x='children',hue='smoker',data=data)

#DATAWRANGLIN

data.isnull()

sns.boxplot(x='sex',y='age',data=data)

sns.boxplot(x='smoker',y='age',data=data)

data.head(20)

data.drop(['bmi'],axis=1,inplace=True)

data.head()

sex=pd.get_dummies(data['sex'],drop_first=True)
sex.head()

smoker=pd.get_dummies(['smoker'],drop_first=True)
smoker.head(10)

region=pd.get_dummies(data['region'],drop_first=False)
region.head(10)

data.dtypes

data.head()

data=pd.concat([data,region,sex,smoker],axis=1)

data.head(10)

data.drop(['smoker','sex','region'],axis=1,inplace=True)

data.head()

X = data.drop('charges', axis=1)
y = data['charges']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
print("Mean Squared Error:", mse)
mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)
print("Root Mean Squared Error:", rmse)

****Explanation of Outcome****
-From the dataset analysed above,we found the following insights;
	*we find that there is only few people who smoke from the regions recorded.
	*Also,when comparing people who smoke with respect to sex,we find that most people who smoke are men compared to female.
	*When ploting a histogram for the age that makes up the population,we fing age goup in the population ranges from 10-65 years
	*We also find that,most smokers are from the regionof southeast while southwest and Northwest people have fewer smokers recorded.
	*Also from the population data, its found that, most people smokers are adults.

  
**PROCESS OF PREDICTION**
Splits the data into features (stored in X) and target (stored in y).
Splits the data into training and testing sets (80% for training and 20% for testing).
Fits a linear regression model to the training data using the LinearRegression class from scikit-learn.
Makes predictions on the test data using the predict method.
Calculates the mean squared error (MSE) between the actual target values and the predicted values. The MSE is a measure of the quality of the model's predictions.

The mean squared error (MSE) is a common metric for evaluating the performance of a regression model. It measures the average of the squared differences between the actual target values and the predicted values. A lower MSE indicates a better-performing model.

The MSE is calculated by taking the difference between the actual and predicted values, squaring them, and then taking the mean of the squared differences. This gives a measure of the average magnitude of the error in the model's predictions, regardless of the direction of the error. The MSE penalizes large errors more than small errors