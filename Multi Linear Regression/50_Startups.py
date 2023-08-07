"""Problem Statement
Prepare a prediction model for profit of 50_startups data.
Do transformations for getting better predictions of profit and make a table containing R^2 value for each prepared model.

Features
R&D Spend -- Research and devolop spend in the past few years
Administration -- spend on administration in the past few years
Marketing Spend -- spend on Marketing in the past few years
State -- states from which data is collected
Profit -- profit of each state in the past few years"""
# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

#Importing Libraries
import pandas as pd
import numpy as np

#Importing Dataset
df = pd.read_csv("D:/Assignment and data set/50_Startups.csv")
df
df.head()
df.shape

#Descriptive Analysis
df.describe()
#checking missing values
df.isnull().sum()
#checking data type
df.info()

list(df)
#Checking for Duplicated Values
df.duplicated()
#checking corelation 
df.corr()

#Data visulation
## scatterplot between R&D Spend and profit
import matplotlib.pyplot as plt
plt.scatter(x=df[['R&D Spend']],y=df['Profit'],color='red')
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.show()
df['R&D Spend'].hist() 
df.boxplot(column='R&D Spend',vert=False) 

## scatterplot between Administration and profit
plt.scatter(x=df[['Administration']],y=df['Profit'],color='red')
plt.xlabel('Administration')
plt.ylabel('Profit')
plt.show()
df['Administration'].hist() 
df.boxplot(column='Administration',vert=False)

## scatterplot between Marketing Spend and profit
plt.scatter(x=df[['Marketing Spend']],y=df['Profit'],color='red')
plt.xlabel('Marketing Spend')
plt.ylabel('Profit')
plt.show()
df['Marketing Spend'].hist() 
df.boxplot(column='Marketing Spend',vert=False)

import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)

#Model Fitting and Model selecting

#Model 1
Y=df[['Profit']]
X=df[['R&D Spend']] ### M1
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y
Y_pred=LR.predict(X)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(Y,Y_pred)
R2_1=r2_score(Y,Y_pred)
print('Mean squared error:',mse.round(3))
print('Root mean sqared error:',np.sqrt(mse).round(3))
print('R square:',R2_1.round(3))

### Model-2
Y=df[['Profit']]
X=df[['R&D Spend','Administration']] ### M2
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y
Y_pred=LR.predict(X)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(Y,Y_pred)
R2_2=r2_score(Y,Y_pred)
print('Mean squared error:',mse.round(3))
print('Root mean sqared error:',np.sqrt(mse).round(3))
print('R square:',R2_2.round(3))

### Model-3
Y=df[['Profit']]
X=df[['Marketing Spend ']]  ### M3
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y
Y_pred=LR.predict(X)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(Y,Y_pred)
R2_3=r2_score(Y,Y_pred)
print('Mean squared error:',mse.round(3))
print('Root mean sqared error:',np.sqrt(mse).round(3))
print('R square:',R2_3.round(3))

### Model-4
Y=df[['Profit']]
X=df[['Marketing Spend ','Administration']] ### M4
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y
Y_pred=LR.predict(X)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(Y,Y_pred)
R2_4=r2_score(Y,Y_pred)
print('Mean squared error:',mse.round(3))
print('Root mean sqared error:',np.sqrt(mse).round(3))
print('R square:',R2_4.round(2))

### Model-5
Y=df[['Profit']]
X=df[['R&D Spend','Marketing Spend','Administration']] ### M5
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)
LR.intercept_
LR.coef_
Y
Y_pred=LR.predict(X)
from sklearn.metrics import mean_squared_error,r2_score
mse=mean_squared_error(Y,Y_pred)
R2_5=r2_score(Y,Y_pred)
print('Mean squared error:',mse.round(3))
print('Root mean sqared error:',np.sqrt(mse).round(3))
print('R square:',R2_5.round(3))

# Putting all Models RMSE and R2 in Dataframe format
d={'Models':['Model-1','Model-2','Model-3','Model-4','Model-5'],'R-Squared':[R2_1,R2_2,R2_3,R2_4,R2_5]}
R2_df=pd.DataFrame(d)
R2_df


#By considering R2 Value of Model-4 is giving best results
#Model 4 is having Multicoliniarity
import statsmodels.formula.api as smf
model=smf.ols('Profit~Administration+Marketing Spend',data=df).fit()
model.summary()

import statsmodels.formula.api as smf
model = smf.ols('Administration~Marketing Spend',data=df).fit()
R2 = model.rsquared
VIF = 1/(1-R2)
print('Variance influence factor:',VIF)

# Residual Analysis
model.resid
model.resid.hist()

## Test for Normality of Residuals (Q-Q Plot)
import matplotlib.pyplot as plt
import statsmodels.api as sm
qqplot=sm.qqplot(model.resid,line='q')
plt.title('Q-Q plot of residuals')
plt.show()

model.fittedvalues # predicted values
model.resid # error values

# cheking pattern.no pattern no issues
import matplotlib.pyplot as plt
plt.scatter(model.fittedvalues,model.resid)
plt.title('Residual Plot')
plt.xlabel('Fitted values')
plt.ylabel('residual values')
plt.show()
# Model Deletion Diagnostics
## Detecting Influencers/Outliers

## Cooks Distance
model_influence = model.get_influence()
(cooks, pvalue) = model_influence.cooks_distance

cooks = pd.DataFrame(cooks)

#Plot the influencers values using stem plot
import matplotlib.pyplot as plt
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(df)), np.round(cooks[0],5))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()

# index and value of influencer where c is more than .5
cooks[0][cooks[0]>0.5]
df.tail()

## High Influence points
from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()
### Leverage Cutoff
k = df.shape[1]
n = df.shape[0]
leverage_cutoff = (3*(k + 1)/n)
leverage_cutoff
cooks[0][cooks[0]>leverage_cutoff]
### No valuse are under leverage cuttoff 0.36

#==============================================================================================
# data transformation
from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df["State"] = LE.fit_transform(df["State"])
y = df["State"]
#standard scaler
from sklearn.preprocessing import StandardScaler
df_cont = df[['Administration','Marketing Spend']]

ss = StandardScaler()

ss_cont    = ss.fit_transform(df_cont)
ss_cont  = pd.DataFrame(ss_cont)
ss_cont.columns= ['R&D Spend','Administration','Marketing Spend']
x=ss_cont
df = pd.concat([ss_cont,df['State'],df['Profit']],axis=1)
df

#Data Partining into train and test
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(x,y,train_size=0.75, random_state=42)
x_train.shape
x_test.shape
y_train
y_test

#selecting model
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_train,y_train)

#model predictions
Y_pred_train = LR.predict(x_train)
Y_pred_test = LR.predict(x_test)

#Metrics
from sklearn.metrics import mean_squared_error , r2_score
mse = mean_squared_error(y_train,Y_pred_train)
r2 = r2_score(y_train,Y_pred_train)
print("Mean Square root is:",mse.round(3))
print("Root Mean Square root is:",np.sqrt(mse))
print("R-Square is:",r2.round(3))

#Apply validation set method and calc average training error and average test error.

#cross vaidation set approach

Training_err=[]
Test_err=[]

for i in range(1,1001):
    x_train,x_test,y_train,y_test = train_test_split(x,y,train_size=0.75, random_state=i)
    LR.fit(x_train,y_train)
    Y_pred_train = LR.predict(x_train)
    Y_pred_test = LR.predict(x_test)
    Training_err.append(np.sqrt(mean_squared_error(y_train, Y_pred_train)))
    Test_err.append(np.sqrt(mean_squared_error(y_test, Y_pred_test)))
print("Average Training Error",np.mean(Training_err).round(3))
print("Average Test Error",np.mean(Test_err).round(3))





























