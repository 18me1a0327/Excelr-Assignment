#Importing Libraries
# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

#Importing Dataset
Fraud_check=pd.read_csv("D:/Assignment and data set/Fraud_check.csv")
Fraud_check

Fraud_check.head()
Fraud_check.tail()

# Feature Engineering
# Converting taxable_income <= 30000 as "Risky" and others are "Good"
df=Fraud_check.copy()
df['taxable_category'] = pd.cut(x = df['Taxable.Income'], bins = [10002,30000,99620], labels = ['Risky', 'Good'])
df.head()
#Data Exploration
#Descriptive Statistics
df.describe()
df.info()
#Missing Values
df.isnull().sum()
#Duplicated Values
df.duplicated().sum()
#columns
df.columns

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

sns.countplot(x='Marital.Status',data=df)

sns.pairplot(Fraud_check)

#Boxplot
ot=df.copy() 
fig, axes=plt.subplots(3,1,figsize=(14,6),sharex=False,sharey=False)
sns.boxplot(x='Taxable.Income',data=ot,palette='crest',ax=axes[0])
sns.boxplot(x='City.Population',data=ot,palette='crest',ax=axes[1])
sns.boxplot(x='Work.Experience',data=ot,palette='crest',ax=axes[2])
plt.tight_layout(pad=2.0)

# Having a look at the correlation matrix
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(df.corr(), annot=True, fmt='.1g', cmap="viridis", cbar=False, linewidths=0.5, linecolor='black')

#piechart
plt.figure(figsize = (12,8))
plt.pie(df['taxable_category'].value_counts(),
       labels=df.taxable_category.unique(),
       explode = [0.07,0.0],
       autopct= '%.2f%%',
       shadow= True,
       startangle= 190,
       textprops = {'size':'large',
                   'fontweight':'bold',
                    'rotation':'0',
                   'color':'black'})
plt.legend(loc= 'upper right')
plt.title("Class Type Distribution Pie Chart", fontsize = 18, fontweight = 'bold')
plt.show()

#Data Pre-Processing

data = df.copy()
data.rename(columns={'Marital.Status':'Marital_Status', 'Taxable.Income':'Taxable_Income','Work.Experience':'Work_Experience','City.Population':'City_Population'}, inplace = True)
data.drop('Taxable_Income', axis=1, inplace = True)
categorical_features = data.describe(include=["object",'category']).columns
categorical_features

#Creating dummy vairables of the categorical features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in categorical_features:
        le.fit(data[col])
        data[col] = le.transform(data[col])
data.head()

#Data Pre-processing for feature Selection
data_ = df.copy()
data_.drop('Taxable.Income',axis=1, inplace =True)
data_ = pd.get_dummies(data_.iloc[:,:-1])
data_.head()

data_['Taxable_Income'] = df.taxable_category
data_.head()

le = LabelEncoder()
le.fit(data_["Taxable_Income"])
data_["Taxable_Income"]=le.transform(data_["Taxable_Income"])
data_.head()

# split into input (X) and output (y) variables

x = data_.iloc[:, :-1]

y=  data_.Taxable_Income

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0,stratify=y)
print("Shape of X_train: ",x_train.shape)
print("Shape of X_test: ", x_test.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_test",y_test.shape)


# Random Forest 
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(
                  n_estimators=150,
                  max_samples=0.9,
                  max_features=0.3,
                  random_state=6)


# Model Fitting
RFC.fit(x_train,y_train)
Y_pred_train = RFC.predict(x_train)
Y_pred_test = RFC.predict(x_test)

#Finding test and train accuracy score 
from sklearn.metrics import accuracy_score
acc1 = accuracy_score(y_train, Y_pred_train)
print("Training Accuracy:", (acc1).round(2))
acc2 = accuracy_score(y_test, Y_pred_test)
print("Test Accuracy:", (acc2).round(2))


#Bagging
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
bag = BaggingClassifier(base_estimator=DecisionTreeClassifier(max_depth=5),
                  n_estimators=200,
                  max_samples=0.7,
                  max_features=0.4,
                  random_state=10)
bag.fit(x_train,y_train)

# Model Fitting
Y_pred_train = bag.predict(x_train)
Y_pred_test = bag.predict(x_test)

#Finding test and train accuracy score
from sklearn.metrics import accuracy_score
acc1 = accuracy_score(y_train, Y_pred_train)
print("Training Accuracy:", (acc1).round(2))
acc2 = accuracy_score(y_test, Y_pred_test)
print("Test Accuracy:", (acc2).round(2))

# Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(n_estimators=100,
                  learning_rate=0.1,
                  max_features=0.7,
                  random_state=10)
GBR.fit(x_train,y_train)

# Model Fitting
Y_pred_train = GBR.predict(x_train)
Y_pred_test = GBR.predict(x_test)

#Finding test and train accuracy score
from sklearn.metrics import mean_squared_error
err1 = np.sqrt(mean_squared_error(y_train, Y_pred_train))
print("Gradient Boosting-Training Error:", (err1).round(2))
err2 = np.sqrt(mean_squared_error(y_test, Y_pred_test))
print("Gradient Boosting-Test Error:", (err2).round(2))
print("Variance Between Train and Test:",(err2-err1).round(2))

 
#ada boost regressor
from sklearn.ensemble import AdaBoostRegressor
ABR = AdaBoostRegressor(n_estimators=100,base_estimator=None,
                  learning_rate=0.)
GBR.fit(x_train,y_train)
Y_pred_train = ABR.predict(x_train)
Y_pred_test = ABR.predict(x_test)

from sklearn.metrics import mean_squared_error
err1 = np.sqrt(mean_squared_error(y_train, Y_pred_train))
print("Gradient Boosting-Training Error:", (err1).round(2))
err2 = np.sqrt(mean_squared_error(y_test, Y_pred_test))
print("Gradient Boosting-Test Error:", (err2).round(2))
print("Variance Between Train and Test:",(err2-err1).round(2))



                
#Grid Search CV
d1={'n_estimators':[50,150,200,250],
    'max_samples':[0.1,0.5,0.7,0.9],
    'max_features':[0.3,0.5,0.7,0.9],
    'random_state':[2,6,8,10]}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(estimator=RFC, param_grid=d1)
grid.fit(x_train,y_train)
print(grid.best_score_)
print(grid.best_params_)


























