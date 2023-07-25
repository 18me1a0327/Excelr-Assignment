"""
Created on Tue Jul 25 15:36:27 2023

@author: Lenovo

2) Salary_hike -> Build a prediction model for Salary_hike
"""
import pandas as pd
import numpy as np

df = pd.read_csv("D:/Assignment and data set/Salary_Data.csv")
df

df.shape
#here renamed the Salary --> salary_hike
df=df.rename({'Salary':'salary_hike'},axis=1)
df
#Spliting of X and Y
x=df[['YearsExperience']]
y=df[['salary_hike']]


#Exploratory Data Analysis
df.info()
"""<class 'pandas.core.frame.DataFrame'>
RangeIndex: 30 entries, 0 to 29
Data columns (total 2 columns):
 #   Column           Non-Null Count  Dtype  
---  ------           --------------  -----  
 0   YearsExperience  30 non-null     float64
 1   salary_hike      30 non-null     float64
dtypes: float64(2)
memory usage: 608.0 bytes"""
df.describe()
"""      YearsExperience    salary_hike
count        30.000000      30.000000
mean          5.313333   76003.000000
std           2.837888   27414.429785
min           1.100000   37731.000000
25%           3.200000   56720.750000
50%           4.700000   65237.000000
75%           7.700000  100544.750000
max          10.500000  122391.000000"""
df.head()
df.tail()
df.dtypes
"""YearsExperience    float64
salary_hike        float64
dtype: object"""
df.isnull().sum()
"""
YearsExperience    0
salary_hike        0
dtype: int64"""
#To find outliers
df.boxplot(column='YearsExperience',vert=False)
df.boxplot(column='salary_hike')
#The Delivery Time has outliers 
#Data visulation
import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.xlabel("YearsExperience")
plt.ylabel("salary_hike")
plt.show()
#colinear 
df.corr()
"""                YearsExperience  salary_hike
YearsExperience         1.000000     0.978242
salary_hike             0.978242     1.000000"""
#Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x,y)
LR.intercept_   #array([25792.20019867])
LR.coef_        #array([[9449.96232146]])

#Predectoin
df[["salary_hike"]]
salary_pred = LR.predict(x)
y
#constructing regrassion line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(x,y,color='red')
plt.scatter(x,y=salary_pred,color='blue')
plt.plot(x,salary_pred,color='black')
plt.xlabel('years of experience')
plt.ylabel('Salary_hike')
plt.show()


#Finding Errors by using Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y,salary_pred)
R2= r2_score(y, salary_pred)
print("Mean_squared_error:", mse.round(3))
print("Root Mean Sqaure error:",np.sqrt(mse).round(3))
print("R square:", R2.round(3))

"""Mean_squared_error: 31270951.722
Root Mean Sqaure error: 5592.044
R square: 0.957"""


#==================================================================================
#Applying log transfromation to x

x_log = np.log(df[["YearsExperience"]])

#Data visulation
import matplotlib.pyplot as plt
plt.scatter(x_log,y)
plt.xlabel("YearsExperience")
plt.ylabel("salary_hike")
plt.show()


#Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_log,y)
LR.intercept_   #array([14927.97177006])
LR.coef_        #array([[40581.98795978]])

#Predectoin
df[["salary_hike"]]
salary_pred = LR.predict(x_log)
y

#constructing regrassion line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(x_log,y,color='red')
plt.scatter(x_log,salary_pred,color='blue')
plt.plot(x_log,salary_pred,color='black')
plt.xlabel('YearsExperience')
plt.ylabel('salary_hike')
plt.show()


#Finding Errors by using Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y,salary_pred)
R2= r2_score(y, salary_pred)
print("Mean_squared_error:", mse.round(3))
print("Root Mean Sqaure error:",np.sqrt(mse).round(3))
print("R square:", R2.round(3))
"""Mean_squared_error: 106149618.722
Root Mean Sqaure error: 10302.894
R square: 0.854"""


#==================================================================================
#Applying log transfromation to y

y_log = np.log(df[["salary_hike"]])

#Data visulation
import matplotlib.pyplot as plt
plt.scatter(x,y_log)
plt.xlabel("YearsExperience")
plt.ylabel("salary_hike")
plt.show()

#Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x,y_log)
LR.intercept_   #array([10.5074019])
LR.coef_        #array([[0.1055516]])

#Predectoin
df[["YearsExperience"]]
deli_pred = LR.predict(x)
y_log

#constructing regrassion line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(x,y_log,color='red')
plt.scatter(x,deli_pred,color='blue')
plt.plot(x,deli_pred,color='black')
plt.xlabel('YearsExperience')
plt.ylabel('salary_hike')
plt.show()


#Finding Errors by using Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_log,salary_pred)
R2= r2_score(y_log,salary_pred)
print("Mean_squared_error:", mse.round(3))
print("Root Mean Sqaure error:",np.sqrt(mse).round(3))
print("R square:", R2.round(3))
"""Mean_squared_error: 6395090081.219
Root Mean Sqaure error: 79969.307
R square: -48642806750.685"""

#==================================================================================
#Applying log transfromation to x and y
x_log = np.log(df[["YearsExperience"]])
y_log = np.log(df[["salary_hike"]])

#Data visulation
import matplotlib.pyplot as plt
plt.scatter(x_log,y_log)
plt.xlabel("YearsExperience")
plt.ylabel("salary_hike")
plt.show()

#Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_log,y_log)
LR.intercept_   #array([10.32804318])
LR.coef_        #array([[0.56208883]])

#Predectoin
df[["YearsExperience"]]
salary_pred = LR.predict(x_log)
y_log

#constructing regrassion line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(x_log,y_log,color='red')
plt.scatter(x_log,salary_pred,color='blue')
plt.plot(x_log,salary_pred,color='black')
plt.xlabel('YearsExperience')
plt.ylabel('salary_hike')
plt.show()


#Finding Errors by using Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_log,salary_pred)
R2= r2_score(y_log, salary_pred)
print("Mean_squared_error:", mse.round(3))
print("Root Mean Sqaure error:",np.sqrt(mse).round(3))
print("R square:", R2.round(3))
"""Mean_squared_error: 0.012
Root Mean Sqaure error: 0.112
R square: 0.905"""

#==================================================================================
#Applying Sq Root Transformation of X
x_sq = np.sqrt(df[["YearsExperience"]])


#Data visulation
import matplotlib.pyplot as plt
plt.scatter(x_sq,y)
plt.xlabel("YearsExperience")
plt.ylabel("salary_hike")
plt.show()

#Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_sq,y)
LR.intercept_   #array([-16055.76911696])
LR.coef_        #array([[41500.68058303]])

#Predectoin
df[["salary_hike"]]
salary_pred = LR.predict(x_sq)
y

#constructing regrassion line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(x_sq,y,color='red')
plt.scatter(x_sq,salary_pred,color='blue')
plt.plot(x_sq,salary_pred,color='black')
plt.xlabel('YearsExperience')
plt.ylabel('salary_hike')
plt.show()


#Finding Errors by using Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y,salary_pred)
R2= r2_score(y, salary_pred)
print("Mean_squared_error:", mse.round(3))
print("Root Mean Sqaure error:",np.sqrt(mse).round(3))
print("R square:", R2.round(3))
"""Mean_squared_error: 50127755.617
Root Mean Sqaure error: 7080.096
R square: 0.931"""

#==================================================================================
#Applying Sq Root Transformation of X x and y
x_sq = np.sqrt(df[["YearsExperience"]])
y_sq = np.sqrt(df[["salary_hike"]])

#Data visulation
import matplotlib.pyplot as plt
plt.scatter(x_sq,y_sq)
plt.xlabel("YearsExperience")
plt.ylabel("salary_hike")
plt.show()

#Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_sq,y_sq)
LR.intercept_   #array([103.56803065])
LR.coef_        #array([[75.6269319]])

#Predectoin
df[["YearsExperience"]]
salry_pred = LR.predict(x_sq)
y_sq

#constructing regrassion line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(x_sq,y_sq,color='red')
plt.scatter(x_sq,deli_pred,color='blue')
plt.plot(x_sq,deli_pred,color='black')
plt.xlabel('YearsExperience')
plt.ylabel('salary_hike')
plt.show()


#Finding Errors by using Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_sq,salary_pred)
R2= r2_score(y_sq, salary_pred)
print("Mean_squared_error:", mse.round(3))
print("Root Mean Sqaure error:",np.sqrt(mse).round(3))
print("R square:", R2.round(3))
"""Mean_squared_error: 6409195034.906
Root Mean Sqaure error: 80057.448
R square: -2687836.779"""






































