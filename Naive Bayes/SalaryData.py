#Importing Libraries
# Supress Warnings
# Supressing Warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np

# Importing Training Dataset
salary_train = pd.read_csv('D:/Assignment and data set/SalaryData_Train.csv')
salary_train

# Importing Testing Dataset
salary_test = pd.read_csv('D:/Assignment and data set/SalaryData_Test.csv')
salary_test

# Merging Train and Test Data
raw_data = salary_train.append(salary_test)
raw_data.reset_index(inplace=True)
raw_data

raw_data.head()
raw_data.tail()


#Data Exploration
#Descriptive Statistics
raw_data.describe()
raw_data.info()
#Missing Values
raw_data.isnull().sum()
#Duplicated Values
raw_data.duplicated().sum()
#columns
raw_data.columns

import matplotlib.pyplot as plt
import seaborn as sns

#Exploratory Data Analysis
#Data Visualization
fig= plt.figure(figsize=(18, 6))
sns.heatmap(raw_data.corr(), annot=True);
plt.xticks(rotation=45)

#boxplot
ot=raw_data.copy() 
fig, axes=plt.subplots(4,1,figsize=(14,8),sharex=False,sharey=False)
sns.boxplot(x='age',data=ot,palette='crest',ax=axes[0])
sns.boxplot(x='capitalgain',data=ot,palette='crest',ax=axes[1])
sns.boxplot(x='capitalloss',data=ot,palette='crest',ax=axes[2])
sns.boxplot(x='hoursperweek',data=ot,palette='crest',ax=axes[3])
plt.tight_layout(pad=2.0)

#pie chart
plt.figure(figsize = (12,8))
plt.pie(raw_data['Salary'].value_counts(),
       labels=raw_data.Salary.unique(),
       explode = [0.07,0.0],
       autopct= '%.2f%%',
        colors = ["#F96167", "#FCE77D"],
       shadow= True,
       startangle= 190,
       textprops = {'size':'large',
                   'fontweight':'bold',
                    'rotation':'0',
                   'color':'black'})
plt.legend(loc= 'upper right')
plt.title("Class Type Distribution Pie Chart", fontsize = 18, fontweight = 'bold')
plt.show()

j_df = pd.DataFrame()

j_df[' <=50K'] = raw_data[raw_data['Salary'] == ' <=50K']['workclass'].value_counts()
j_df[' >50K'] = raw_data[raw_data['Salary'] == ' >50K']['workclass'].value_counts()
j_df.plot.bar(title = 'Workclass and Salary',figsize=(14,6))
j_df.plot(figsize=(14,6))

#Data Transfomation:
# Label Encoding of categrical variables
from sklearn import preprocessing
 
# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()

df= raw_data.copy()
# Encode labels in column 'species'.
df["education"]=label_encoder.fit_transform(df["education"])
df["workclass"]=label_encoder.fit_transform(df["workclass"])
df["maritalstatus"]=label_encoder.fit_transform(df["maritalstatus"])
df["sex"]=label_encoder.fit_transform(df["sex"])
df["race"]=label_encoder.fit_transform(df["race"])
df["occupation"]=label_encoder.fit_transform(df["occupation"])
df["relationship"]=label_encoder.fit_transform(df["relationship"])
df["native"]=label_encoder.fit_transform(df["native"])
df['Salary'] = np.where(df['Salary'].str.contains(" >50K"), 1, 0)
#df["Salary"]=label_encoder.fit_transform(df["Salary"])

df.head(10)

# Test Train Split With Imbalanced Dataset
x = df.drop('Salary',axis=1)
y = df['Salary']


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1,stratify=y)

#Types of Naive Bayes algorithm

#Naive bayes
from sklearn.naive_bayes import MultinomialNB
MNB=MultinomialNB()
MNB.fit(x_train,y_train)

#Model Predictions
Y_pred_train=MNB.predict(x_train)
Y_pred_test=MNB.predict(x_test)

#metrics and accuracy score
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train, Y_pred_train)
print("Training Accuracy score:", (ac1*100).round(2))
ac2 = accuracy_score(y_test, Y_pred_test)
print("Test Accuracy score:", (ac2*100).round(2))

#Gaussian Naïve Bayes
from sklearn.naive_bayes import GaussianNB
GNB = GaussianNB()
GNB.fit(x_train,y_train)

#Model Predictions
Y_pred_train=MNB.predict(x_train)
Y_pred_test=MNB.predict(x_test)

#metrics and accuracy score
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train, Y_pred_train)
print("Training Accuracy score:", (ac1*100).round(2))
ac2 = accuracy_score(y_test, Y_pred_test)
print("Test Accuracy score:", (ac2*100).round(2))

#Bernoulli Naïve Bayes
from sklearn.naive_bayes import BernoulliNB
BNB = BernoulliNB()
BNB.fit(x_train,y_train)

#Model Predictions
Y_pred_train=MNB.predict(x_train)
Y_pred_test=MNB.predict(x_test)

#metrics and accuracy score
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train, Y_pred_train)
ac2 = accuracy_score(y_test, Y_pred_test)
print("Training Accuracy score:", (ac1*100).round(2))
print("Test Accuracy score:", (ac2*100).round(2))




























