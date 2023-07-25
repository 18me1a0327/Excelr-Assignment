"""
1) Delivery_time -> Predict delivery time using sorting time 
"""
import pandas as pd
import numpy as np

df = pd.read_csv("D:/Assignment and data set/delivery_time.csv")
df

df.shape
#Spliting of X and Y
x = df[["Sorting Time"]]
y = df[["Delivery Time"]]


#Exploratory Data Analysis
df.info()
"""Data columns (total 2 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Delivery Time  21 non-null     float64
 1   Sorting Time   21 non-null     int64  
dtypes: float64(1), int64(1)
memory usage: 464.0 bytes"""
df.describe()
"""     Delivery Time  Sorting Time
count      21.000000     21.000000
mean       16.790952      6.190476
std         5.074901      2.542028
min         8.000000      2.000000
25%        13.500000      4.000000
50%        17.830000      6.000000
75%        19.750000      8.000000
max        29.000000     10.000000"""
df.head()
df.tail()
df.dtypes
"""Delivery Time    float64
Sorting Time       int64
dtype: object"""
df.isnull().sum()
"""
Delivery Time    0
Sorting Time     0
dtype: int64"""
#To find outliers
df.boxplot(column='Sorting Time',vert=False)
df.boxplot(column='Delivery Time')
#The Delivery Time has outliers 
#Data visulation
import matplotlib.pyplot as plt
plt.scatter(x,y)
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()
#colinear 
df.corr()
"""              Delivery Time  Sorting Time
Delivery Time       1.000000      0.825997
Sorting Time        0.825997      1.000000"""
#Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x,y)
LR.intercept_   #array([6.58273397])
LR.coef_        #array([[1.6490199]])

#Predectoin
df[["Sorting Time"]]
deli_pred = LR.predict(x)
y
#constructing regrassion line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(x,y,color='red')
plt.scatter(x,y=deli_pred,color='blue')
plt.plot(df[['Sorting Time']],deli_pred,color='black')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()


#Finding Errors by using Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y,deli_pred)
R2= r2_score(y, deli_pred)
print("Mean_squared_error:", mse.round(3))
print("Root Mean Sqaure error:",np.sqrt(mse).round(3))
print("R square:", R2.round(3))


np.min(x)#2
np.max(x)#10
"""Mean_squared_error: 7.793
Root Mean Sqaure error: 2.792
R square: 0.682"""
#These value are without applying any Transfromation


#==================================================================================
#Applying log transfromation to x

x_log = np.log(df[["Sorting Time"]])

#Data visulation
import matplotlib.pyplot as plt
plt.scatter(x_log,y)
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

#Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_log,y)
LR.intercept_   #array([1.15968351])
LR.coef_        #array([[9.04341346]])

#Predectoin
df[["Sorting Time"]]
deli_pred = LR.predict(x_log)
y

#constructing regrassion line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(x_log,y,color='red')
plt.scatter(x_log,deli_pred,color='blue')
plt.plot(x_log,deli_pred,color='black')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()


#Finding Errors by using Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y,deli_pred)
R2= r2_score(y, deli_pred)
print("Mean_squared_error:", mse.round(3))
print("Root Mean Sqaure error:",np.sqrt(mse).round(3))
print("R square:", R2.round(3))
"""Mean_squared_error: 7.47
Root Mean Sqaure error: 2.733
R square: 0.695"""


#==================================================================================
#Applying log transfromation to y

y_log = np.log(df[["Delivery Time"]])

#Data visulation
import matplotlib.pyplot as plt
plt.scatter(x,y_log)
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

#Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x,y_log)
LR.intercept_   #array([2.12137185])
LR.coef_        #array([[0.1055516]])

#Predectoin
df[["Sorting Time"]]
deli_pred = LR.predict(x)
y_log

#constructing regrassion line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(x,y_log,color='red')
plt.scatter(x,deli_pred,color='blue')
plt.plot(x,deli_pred,color='black')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()


#Finding Errors by using Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_log,deli_pred)
R2= r2_score(y_log, deli_pred)
print("Mean_squared_error:", mse.round(3))
print("Root Mean Sqaure error:",np.sqrt(mse).round(3))
print("R square:", R2.round(3))
"""Mean_squared_error: 0.028
Root Mean Sqaure error: 0.167
R square: 0.711"""

#==================================================================================
#Applying log transfromation to x and y
x_log = np.log(df[["Sorting Time"]])
y_log = np.log(df[["Delivery Time"]])

#Data visulation
import matplotlib.pyplot as plt
plt.scatter(x_log,y_log)
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

#Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_log,y_log)
LR.intercept_   #array([1.74198709])
LR.coef_        #array([[0.59752233]])

#Predectoin
df[["Sorting Time"]]
deli_pred = LR.predict(x_log)
y_log

#constructing regrassion line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(x_log,y_log,color='red')
plt.scatter(x_log,deli_pred,color='blue')
plt.plot(x_log,deli_pred,color='black')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()


#Finding Errors by using Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_log,deli_pred)
R2= r2_score(y_log, deli_pred)
print("Mean_squared_error:", mse.round(3))
print("Root Mean Sqaure error:",np.sqrt(mse).round(3))
print("R square:", R2.round(3))
"""Mean_squared_error: 0.022
Root Mean Sqaure error: 0.148
R square: 0.772"""

#==================================================================================
#Applying Sq Root Transformation of X
x_sq = np.sqrt(df[["Sorting Time"]])


#Data visulation
import matplotlib.pyplot as plt
plt.scatter(x_sq,y)
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

#Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_sq,y)
LR.intercept_   #array([-2.51883662])
LR.coef_        #array([[7.93659075]])

#Predectoin
df[["Sorting Time"]]
deli_pred = LR.predict(x_sq)
y

#constructing regrassion line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(x_sq,y,color='red')
plt.scatter(x_sq,deli_pred,color='blue')
plt.plot(x_sq,deli_pred,color='black')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()


#Finding Errors by using Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y,deli_pred)
R2= r2_score(y, deli_pred)
print("Mean_squared_error:", mse.round(3))
print("Root Mean Sqaure error:",np.sqrt(mse).round(3))
print("R square:", R2.round(3))
"""Mean_squared_error: 7.461
Root Mean Sqaure error: 2.732
R square: 0.696"""

#==================================================================================
#Applying Sq Root Transformation of X x and y
x_sq = np.sqrt(df[["Sorting Time"]])
y_sq = np.sqrt(df[["Delivery Time"]])

#Data visulation
import matplotlib.pyplot as plt
plt.scatter(x_sq,y_sq)
plt.xlabel("Sorting Time")
plt.ylabel("Delivery Time")
plt.show()

#Model fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_sq,y_sq)
LR.intercept_   #array([1.61347867])
LR.coef_        #array([[1.00221688]])

#Predectoin
df[["Sorting Time"]]
deli_pred = LR.predict(x_sq)
y_sq

#constructing regrassion line between model predicted values and original values
import matplotlib.pyplot as plt
plt.scatter(x_sq,y_sq,color='red')
plt.scatter(x_sq,deli_pred,color='blue')
plt.plot(x_sq,deli_pred,color='black')
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()


#Finding Errors by using Metrics
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_sq,deli_pred)
R2= r2_score(y_sq, deli_pred)
print("Mean_squared_error:", mse.round(3))
print("Root Mean Sqaure error:",np.sqrt(mse).round(3))
print("R square:", R2.round(3))
"""Mean_squared_error: 0.101
Root Mean Sqaure error: 0.318
R square: 0.729"""

































































