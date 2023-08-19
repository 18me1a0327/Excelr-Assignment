#Importing Libraries
# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

#Importing Dataset
Company_Data=pd.read_csv("D:/Assignment and data set/Company_Data.csv")
Company_Data

Company_Data.head()
Company_Data.tail()

# Converting taxable_income <= 30000 as "Risky" and others are "Good"
df=Company_Data.copy()
df['Sales_cat'] = pd.cut(x = df['Sales'], bins = [0,5.39,9.32,17], labels=['Low','Medium','High'], right = False)
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

sns.pairplot(df)

#Boxplot
ot=df.copy() 
fig, axes=plt.subplots(8,1,figsize=(14,12),sharex=False,sharey=False)
sns.boxplot(x='Sales',data=ot,palette='crest',ax=axes[0])
sns.boxplot(x='CompPrice',data=ot,palette='crest',ax=axes[1])
sns.boxplot(x='Income',data=ot,palette='crest',ax=axes[2])
sns.boxplot(x='Advertising',data=ot,palette='crest',ax=axes[3])
sns.boxplot(x='Population',data=ot,palette='crest',ax=axes[4])
sns.boxplot(x='Price',data=ot,palette='crest',ax=axes[5])
sns.boxplot(x='Age',data=ot,palette='crest',ax=axes[6])
sns.boxplot(x='Education',data=ot,palette='crest',ax=axes[7])
plt.tight_layout(pad=2.0)

# Having a look at the correlation matrix
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(df.corr(), annot=True, fmt='.1g', cmap="viridis", cbar=False, linewidths=0.5, linecolor='black')

#piechart
plt.figure(figsize = (12,8))
plt.pie(df['Sales_cat'].value_counts(),
       labels=df.Sales_cat.unique(),
       explode = [0.04,0.03,0.03],
       autopct= '%.2f%%',
       shadow= True,
       startangle= 181,
       textprops = {'size':'large',
                   'fontweight':'bold',
                    'rotation':'0',
                   'color':'black'})
plt.legend(loc= 'upper right')
plt.title("Class Type Distribution Pie Chart", fontsize = 18, fontweight = 'bold')
plt.show()

#Data Pre-Processing
data_ = df.copy()
data_.drop('Sales',axis=1, inplace =True)
data_ = pd.get_dummies(data_.iloc[:,:-1])
data_.head()

data_['Sales'] = df.Sales_cat
data_.head()

#Creating dummy vairables of the categorical features
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(data_["Sales"])
data_["Sales"]=le.transform(data_["Sales"])
data_.head()

# split into input (X) and output (y) variables
X = data_.iloc[:, :-1]

y=  data_.Sales

model_data = data_[['Price', 'Advertising','Population', 'Income', 'Age', 'ShelveLoc_Good', 'ShelveLoc_Bad', 'ShelveLoc_Medium','Sales']]
model_data.head()

x = model_data.drop('Sales',axis=1)
y = model_data['Sales']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0,stratify=y)
print("Shape of X_train: ",x_train.shape)
print("Shape of X_test: ", x_test.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_test",y_test.shape)

# Random Forest 
from sklearn.ensemble import RandomForestClassifier
RFC = RandomForestClassifier(
                  n_estimators=400,
                  max_samples=0.4,
                  max_features=0.1,
                  random_state=9)


### Model Fitting
RFC.fit(x_train,y_train)
Y_pred_train = RFC.predict(x_train)
Y_pred_test = RFC.predict(x_test)

### Metrics
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
