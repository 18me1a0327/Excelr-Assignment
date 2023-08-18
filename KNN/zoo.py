#Importing Libraries
# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

#Importing Dataset
zoo=pd.read_csv("D:/Assignment and data set/zoo.csv")
zoo

zoo.head()
zoo.tail()

#Data Exploration
#Descriptive Statistics
zoo.describe()
zoo.info()
#Missing Values
zoo.isnull().sum()
#Duplicated Values
zoo.duplicated().sum()
#columns
zoo.columns

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns


sns.pairplot(zoo)

# Having a look at the correlation matrix
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(zoo.corr(), annot=True, fmt='.1g', cmap="viridis", cbar=False, linewidths=0.5, linecolor='black')

# Set default plot grid
sns.set_style('whitegrid')

# Plot histogram of classes
plt.rcParams['figure.figsize'] = (7,7)
sns.countplot(zoo['type'], palette='YlGnBu')
ax = plt.gca()
ax.set_title("Histogram of Classes")

#plot bar of classes
plt.figure(figsize = (16,9))
ax = sns.barplot(x = zoo['type'].value_counts().index.tolist(), y = zoo['type'].value_counts().tolist())
plt.yticks(rotation = 0, fontsize = 14)
plt.xticks(rotation = 45, fontsize = 12)
plt.title("Animal Class Type Distribution",  fontsize = 18, fontweight = 'bold')
plt.xlabel('Animal Types')
plt.ylabel('Counts')
for i in ax.containers:
    ax.bar_label(i,)

#plot piechart of classes
plt.figure(figsize = (12,8))
plt.pie(zoo['type'].value_counts(),
       labels=zoo.type.unique(),
       explode = [0.05,0.0,0.0,0.0,0.0,0.0,0.0],
       autopct= '%.2f%%',
       shadow= True,
       startangle= 190,
       textprops = {'size':'large',
                   'fontweight':'bold',
                    'rotation':'0',
                   'color':'black'})
plt.legend(loc= 'upper right')
plt.title("Animal Class Type Distribution Pie Chart", fontsize = 18, fontweight = 'bold')
plt.show()

#Data Pre-Processing
zoo.drop(['animal name'], axis=1, inplace=True)
zoo.head(1)

#spltting of x and y
y=zoo['type']

x=zoo.iloc[:,1:17]
x
#importing train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
print("Shape of X_train: ",x_train.shape)
print("Shape of X_test: ", x_test.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_test",y_test.shape)



#Selecting and fitting of  KNeighborsClassifer
from sklearn.neighbors import KNeighborsClassifier
KNN = KNeighborsClassifier()
KNN.fit(x_train,y_train)


#Using gridsearchCV for finding best n_neighbors
n_neighbors=list(range(1,50))
parameters={'n_neighbors':n_neighbors}

from sklearn.model_selection import GridSearchCV
grid=GridSearchCV(estimator=KNN, param_grid=parameters)
grid.fit(x_train,y_train)
print(grid.best_score_)
print(grid.best_params_)

#predictions of x_train and x_test
Y_pred_train = KNN.predict(x_train)
Y_pred_test = KNN.predict(x_test)

#Finding test and train accuracy score 
from sklearn.metrics import accuracy_score
acc1 = accuracy_score(y_train,Y_pred_train)
print("Training Accuracy: ",acc1.round(2))
acc2 = accuracy_score(y_test,Y_pred_test)
print("Test Accuracy: ",acc2.round(2))

#K-Fold validation cross validation 
from sklearn.model_selection import KFold
Kf=KFold(5)
Training_mse=[]
Test_mse=[]
for train_index,test_index in Kf.split(x):
    x_train,x_test=x.loc[train_index],x.iloc[test_index]
    y_train,y_test=y.iloc[train_index],y.iloc[test_index]
    KNN.fit(x_train,y_train)
    Y_train_pred=KNN.predict(x_train)
    Y_test_pred=KNN.predict(x_test)

Training_mse.append(accuracy_score(y_train,Y_train_pred))
Test_mse.append(accuracy_score(y_test,Y_test_pred))
print('trining accuracy score:',np.mean(Training_mse).round(3))    
print('test accuracy score:',np.mean(Test_mse).round(3)) 

























