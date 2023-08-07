#Importing Libraries
# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

#Importing Dataset
df=pd.read_csv("D:/Assignment and data set/ToyotaCorolla.csv",encoding='latin1')
df

#dorpping 
df.drop(['Id', 'Model', 'Mfg_Month', 'Mfg_Year',
       'Fuel_Type',  'Met_Color', 'Color', 'Automatic',
       'Cylinders',   'Mfr_Guarantee',
       'BOVAG_Guarantee', 'Guarantee_Period', 'ABS', 'Airbag_1', 'Airbag_2',
       'Airco', 'Automatic_airco', 'Boardcomputer', 'CD_Player',
       'Central_Lock', 'Powered_Windows', 'Power_Steering', 'Radio',
       'Mistlamps', 'Sport_Model', 'Backseat_Divider', 'Metallic_Rim',
       'Radio_cassette', 'Tow_Bar'],axis=1,inplace=True)
list(df)
df.head()
df.tail()

#Descriptive Analysis
df.describe()
#checking data type
df.info()
#checking missing values
df.isnull().sum()
#checking corelation 
#Visulization- Heatmap correlation
sns.heatmap(df.corr(), cmap='Blues' ,annot=True)

#Splitting of x and y
#Here "price" is y variable
y = df["Price"]

x = df.iloc[:,1:]

#Data visulation
import matplotlib.pyplot as plt
import seaborn as sns
sns.pairplot(df,height=1.5)
#scatterplot between Age_08_04 and Price
import matplotlib.pyplot as plt
plt.scatter(x=df[['Age_08_04']],y=df['Price'],color='red')
plt.xlabel('Age_08_04')
plt.ylabel('Price')
plt.show()
df['Age_08_04'].hist() 
df.boxplot(column='Age_08_04',vert=False) 

#scatterplot between KM and price
plt.scatter(x=df[['KM']],y=df['Price'],color='red')
plt.xlabel('KM')
plt.ylabel('Price')
plt.show()
#Histgraph for KM
df['KM'].hist() 
#Boxplot for KM
df.boxplot(column='KM',vert=False)

## scatterplot between HP and price
plt.scatter(x=df[['HP']],y=df['Price'],color='red')
plt.xlabel('HP')
plt.ylabel('Price')
plt.show()
df['HP'].hist() 
df.boxplot(column='HP',vert=False)

#scatterplot between CC and price
plt.scatter(x=df[['CC']],y=df['Price'],color='red')
plt.xlabel('CC')
plt.ylabel('Price')
plt.show()
#Histgraph for CC
df['CC'].hist() 
#Boxplot for CC
df.boxplot(column='CC',vert=False)

#scatterplot between Doors and price
plt.scatter(x=df[['Doors']],y=df['Price'],color='red')
plt.xlabel('Doors')
plt.ylabel('Price')
plt.show()
#Histgraph for Doors
df['Doors'].hist() 
#Boxplot for Doors
df.boxplot(column='Doors',vert=False)

#scatterplot between Gears and price
plt.scatter(x=df[['Gears']],y=df['Price'],color='red')
plt.xlabel('Gears')
plt.ylabel('Price')
plt.show()
#Histgraph for Gears
df['Gears'].hist() 
#Boxplot for Gears
df.boxplot(column='Gears',vert=False)

#scatterplot between Quarterly_Tax and price
plt.scatter(x=df[['Quarterly_Tax']],y=df['Price'],color='red')
plt.xlabel('Quarterly_Tax')
plt.ylabel('Price')
plt.show()
#Histgraph for Quarterly_Tax
df['Quarterly_Tax'].hist() 
#Boxplot for Quarterly_Tax
df.boxplot(column='Quarterly_Tax',vert=False)

#scatterplot between Weight and price
plt.scatter(x=df[['Weight']],y=df['Price'],color='red')
plt.xlabel('Weight')
plt.ylabel('Price')
plt.show()
#Histgraph for Weight
df['Weight'].hist() 
#Boxplot for Weight
df.boxplot(column='Weight',vert=False)

#Multicollinearity:
#1.Scatterplot matrix 
#2.Correlation
#3.VIF
import statsmodels.formula.api as smf
Model=smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit()
Model.summary()
#### variance influence factror
import statsmodels.formula.api as smf
Model=smf.ols('HP~KM+Age_08_04+cc+Doors+Gears+Quarterly_Tax+Weight',data=df).fit()
R2=Model.rsquared
VIF=1/1-R2
print('variance influencing factor',VIF.round(3))

######## Resididual analysis
Model.resid
Model.resid.hist()

#### Test for normality of residuals 
import matplotlib.pyplot as plot 
import statsmodels.api as sm
qqplot=sm.qqplot(Model.resid,line='q')
plt.title('Q-Q plot of residuals')
plt.show()
#predicted values of model
Model.fittedvalues  
#error values of model
Model.resid 
#Pattern checking
import matplotlib.pyplot as plt
plt.scatter(Model.fittedvalues,Model.resid)
plt.title('Residual plot')
plt.xlabel('Model fittedvalues')
plt.ylabel('Model residual')
plt.show()

# Model Deletion Diagnostics
#Detecting Influencers/Outliers

### cooks distance
Model_influence = Model.get_influence()
(cooks, pvalue) = Model_influence.cooks_distance

cooks = pd.DataFrame(cooks)

#Plot the influencers values using stem plot
import matplotlib.pyplot as plt
fig = plt.subplots(figsize=(20, 7))
plt.stem(np.arange(len(df)), np.round(cooks[0],5))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()

#index and value of influencer where c is more than .5
cooks[0][cooks[0]>0.5]

## High Influence points
from statsmodels.graphics.regressionplots import influence_plot
influence_plot(Model)
plt.show()
### Leverage Cutoff
k = df.shape[1]
n = df.shape[0]
leverage_cutoff = (3*(k + 1)/n)
leverage_cutoff
cooks[0][cooks[0]>leverage_cutoff]
df.shape
#droping the rows which are under levarage cutoff
df.drop([8,10,11,12,13,14,15,16,49,53,80,141,221,601,654,956,960,991,1044],inplace=True)
df.shape

#================================================================================
#data Transformations 
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss_x=ss.fit_transform(X)
ss_x=pd.DataFrame(ss_x)
ss_x.columns=['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']
ss_x

#data partation of Testing and Training 
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.70)
X_train.shape
X_test.shape
Y_train.shape
X_test.shape

#Selecting few models
#model fitting for Linear regression
from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X_train,Y_train)

#model predictions
Y_train_pred=LR.predict(X_train)
Y_test_pred=LR.predict(X_test)

#matrics
from sklearn.metrics import mean_squared_error
mse_train=np.sqrt(mean_squared_error(Y_train_pred,Y_train))
mse_test=np.sqrt(mean_squared_error(Y_test_pred,Y_test))
print('Training mean sqared error',mse_train.round(2))
print('Test mean sqared error',mse_test.round(2))


#Cross validation for all chosen models
#K-Fold validation
from sklearn.model_selection import KFold
Kf=KFold(5)
Training_mse=[]
Test_mse=[]
for train_index,test_index in Kf.split(X):
    X_train,X_test=X.iloc[train_index],X.iloc[test_index]
    Y_train,Y_test=Y.iloc[train_index],Y.iloc[test_index]
    LR.fit(X_train,Y_train)
    Y_train_pred=LR.predict(X_train)
    Y_test_pred=LR.predict(X_test)

Training_mse.append(np.sqrt(mean_squared_error(Y_train,Y_train_pred)))
Test_mse.append(np.sqrt(mean_squared_error(Y_test,Y_test_pred)))
print('training mean squared error:',np.mean(Training_mse).round(3))    
print('test mean squared error:',np.mean(Test_mse).round(3))
 
#Shrinking Methods
#Ridge Regression
from sklearn.linear_model import Lasso
LS=Lasso(alpha=8)

LS.fit(X,Y)
d1=pd.DataFrame(list(X))
d2=pd.DataFrame(LR.coef_)
#a1=pd.DataFrame(LS.coef_)
#a3=pd.DataFrame(LS.coef_)
#a5=pd.DataFrame(LS.coef_)
a8=pd.DataFrame(LS.coef_)
df_lasso=pd.concat([d1,d2,a1,a3,a5,a8],axis=1)
df_lasso.columns=['names','LR','alpha1','alpha','alpha5','alpha8']
df_lasso

#droping 5th column
df
X=df.iloc[:,1:]
X
X_new=X.drop(X.columns[[5]],axis=1)
X_new


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss_x=ss.fit_transform(X_new)

from sklearn.model_selection import KFold
Kf=KFold(5)
Training_mse=[]
Test_mse=[]
for train_index,test_index in Kf.split(ss_x):
    X_train,X_test=ss_x[train_index],ss_x[test_index]
    Y_train,Y_test=Y.iloc[train_index],Y.iloc[test_index]
    LR.fit(X_train,Y_train)
    Y_train_pred=LR.predict(X_train)
    Y_test_pred=LR.predict(X_test)

Training_mse.append(np.sqrt(mean_squared_error(Y_train,Y_train_pred)))
Test_mse.append(np.sqrt(mean_squared_error(Y_test,Y_test_pred)))
print('training mean squared error:',np.mean(Training_mse).round(3))    
print('test mean squared error:',np.mean(Test_mse).round(3)) 

#Final Model Fitting
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
ss_x    = ss.fit_transform(X_new)
LR.fit(X_new,Y)
LR.intercept_
LR.coef_
#Final Model Fitted Values
final_model=pd.DataFrame({'Age_08_04':25,'KM':50000,'HP':100,'cc':2500,'Doors':4,'Quarterly_Tax':250,'Weight':1200},index=[0])
model=LR.predict(final_model)
model








                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            q`1111111111111111111111111111111111111111111111111111111111q



















































































































