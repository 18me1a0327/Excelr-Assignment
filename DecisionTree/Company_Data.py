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

#importing train and test
#spltting of x and y
x = data.drop('taxable_category',axis=1)
y = data['taxable_category']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0,stratify=y)
print("Shape of X_train: ",x_train.shape)
print("Shape of X_test: ", x_test.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_test",y_test.shape)

#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='entropy')
DT.fit(x_train,y_train)

#predictions of x_train and x_tests
Y_pred_train = DT.predict(x_train)
Y_pred_test = DT.predict(x_test)

#Finding test and train accuracy scor
from sklearn.metrics import accuracy_score
acc1 = accuracy_score(y_train, Y_pred_train)
print("Training Accuracy:", (acc1).round(2))
acc2 = accuracy_score(y_test, Y_pred_test)
print("Test Accuracy:", (acc2).round(2))

#criterion='entropy'
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini')
DT.fit(x_train,y_train)

#predictions of x_train and x_tests
Y_pred_train = DT.predict(x_train)
Y_pred_test = DT.predict(x_test)

#Finding test and train accuracy score
from sklearn.metrics import accuracy_score
acc1 = accuracy_score(y_train, Y_pred_train)
print("Training Accuracy:", (acc1).round(2))
acc2 = accuracy_score(y_test, Y_pred_test)
print("Test Accuracy:", (acc2).round(2))

#Building Decision Tree Classifier
from sklearn import tree
plt.figure(figsize=(15,15))
tree.plot_tree(DT,rounded=True,filled=True)
plt.show()

#Building Decision Tree Classifier using Gini Criteria
model_gini = DecisionTreeClassifier(criterion='gini', random_state=0)
model_gini.fit(x_train,y_train)
DecisionTreeClassifier(random_state=0)
plt.figure(figsize=(15,10),dpi=500)
tree.plot_tree(model_gini,filled=True)
plt.show()
