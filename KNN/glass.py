#Importing Libraries
# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

#Importing Dataset
glass=pd.read_csv("D:/Assignment and data set/glass.csv")
glass

glass.head()
glass.tail()

#Data Exploration
#Descriptive Statistics
glass.describe()
glass.info()
#Missing Values
glass.isnull().sum()
#Duplicated Values
glass.duplicated().sum()
#columns
glass.columns

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

ot=glass.copy() 
fig, axes=plt.subplots(9,1,figsize=(14,12),sharex=False,sharey=False)
sns.boxplot(x='RI',data=ot,palette='crest',ax=axes[0])
sns.boxplot(x='Na',data=ot,palette='crest',ax=axes[1])
sns.boxplot(x='Mg',data=ot,palette='crest',ax=axes[2])
sns.boxplot(x='Al',data=ot,palette='crest',ax=axes[3])
sns.boxplot(x='Si',data=ot,palette='crest',ax=axes[4])
sns.boxplot(x='K',data=ot,palette='crest',ax=axes[5])
sns.boxplot(x='Ca',data=ot,palette='crest',ax=axes[6])
sns.boxplot(x='Ba',data=ot,palette='crest',ax=axes[7])
sns.boxplot(x='Fe',data=ot,palette='crest',ax=axes[8])
plt.tight_layout(pad=2.0)

sns.pairplot(glass)

# Having a look at the correlation matrix
fig, ax = plt.subplots(figsize=(15,10))
sns.heatmap(glass.corr(), annot=True, fmt='.1g', cmap="viridis", cbar=False, linewidths=0.5, linecolor='black')

# Set default plot grid
sns.set_style('whitegrid')
# Plot histogram of classes
plt.rcParams['figure.figsize'] = (7,7)
sns.countplot(glass['Type'], palette='YlGnBu')
ax = plt.gca()
ax.set_title("Histogram of Classes")

#plot bar of classes
plt.figure(figsize = (16,9))
ax = sns.barplot(x = glass['Type'].value_counts().index.tolist(), y = glass['Type'].value_counts().tolist())
plt.yticks(rotation = 0, fontsize = 14)
plt.xticks(rotation = 45, fontsize = 12)
plt.title("Class Type Distribution",  fontsize = 18, fontweight = 'bold')
plt.xlabel('Types')
plt.ylabel('Counts')
for i in ax.containers:
    ax.bar_label(i,)

#plot piechart of classes
plt.figure(figsize = (12,8))
plt.pie(glass['Type'].value_counts(),
       labels=glass.Type.unique(),
       explode = [0.05,0.0,0.0,0.0,0.0,0.0],
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


#importing train and test
from sklearn.model_selection import train_test_split

#Test Train Split 
x = glass.drop('Type',axis=1)
y = glass[['Type']]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
print("Shape of X_train: ",x_train.shape)
print("Shape of X_test: ", x_test.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_test",y_test.shape)

# data Transformations 
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
ss_x=ss.fit_transform(x)
ss_x=pd.DataFrame(ss_x)
ss_x.columns=['RI','Na','Mg','Al','Si','K','Ca','Ba','Fe']
ss_x



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















































