# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

#Importing Dataset

# Importing Training and Testing Dataset
salary_train = pd.read_csv('D:/Assignment and data set/SalaryData_Train(1).csv')
salary_train

salary_test = pd.read_csv('D:/Assignment and data set/SalaryData_Test(1).csv')
salary_test

# Merging Train and Test Data
raw_data = salary_train.append(salary_test)
raw_data.reset_index(inplace=True,drop=True)
raw_data

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

#Numerical Variables
# List of Numerical Variables
numerical_features=[feature for feature in raw_data.columns if raw_data[feature].dtypes != 'O']

print('Number of numerical variables:', len(numerical_features))

# Visualize the numerical variables
raw_data[numerical_features].head()

#Discrete Feature
discrete_feature=[feature for feature in numerical_features if len(raw_data[feature].unique())<25]
print('Discrete Variables Count: {}'.format(len(discrete_feature)))

#Continuous Variable
continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature]
print('Continuous Feature Count {}'.format(len(continuous_feature)))

# Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

fig= plt.figure(figsize=(18, 8))
sns.heatmap(raw_data.corr(), annot=True);
plt.xticks(rotation=45)


#Outliers Detection
ot=raw_data.copy() 
fig, axes=plt.subplots(4,1,figsize=(14,8),sharex=False,sharey=False)
sns.boxplot(x='age',data=ot,palette='crest',ax=axes[0])
sns.boxplot(x='capitalgain',data=ot,palette='crest',ax=axes[1])
sns.boxplot(x='capitalloss',data=ot,palette='crest',ax=axes[2])
sns.boxplot(x='hoursperweek',data=ot,palette='crest',ax=axes[3])
plt.tight_layout(pad=2.0)

#After Log-Transformation
for feature in continuous_feature:
    data=raw_data.copy()
    data[feature]=np.log(data[feature])
    data.boxplot(column=feature)
    plt.ylabel(feature)
    plt.title(feature)
    plt.show()

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

#Data Pre-Processing
#Label Encoding Technique
from sklearn import preprocessing
 
# label_encoder object knows how to understand word labels.
df= raw_data.copy()
label_encoder = preprocessing.LabelEncoder()
df["education"]=label_encoder.fit_transform(df["education"])
df["workclass"]=label_encoder.fit_transform(df["workclass"])
df["maritalstatus"]=label_encoder.fit_transform(df["maritalstatus"])
df["sex"]=label_encoder.fit_transform(df["sex"])
df["race"]=label_encoder.fit_transform(df["race"])
df["occupation"]=label_encoder.fit_transform(df["occupation"])
df["relationship"]=label_encoder.fit_transform(df["relationship"])
df["native"]=label_encoder.fit_transform(df["native"])
df.head(10)

df['Salary'] = raw_data.Salary
df['Salary'] = np.where(df['Salary'].str.contains(" >50K"), 1, 0)
df.head()

#Applying Standard Scaler
df[continuous_feature]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features = df[continuous_feature]
df[continuous_feature] = scaler.fit_transform(features.values)
df.head()

#Test Train Split With Imbalanced Dataset
x = df.drop('Salary',axis=1)
y = df['Salary']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1,stratify=y)

print("Shape of X_train: ",x_train.shape)
print("Shape of X_test: ", x_test.shape)
print("Shape of y_train: ",y_train.shape)
print("Shape of y_test",y_test.shape)

# Sklearn Support Vector Classifier Using Linear, Polynomial and RBF Kernel, 
''' For support vector classifier we have three types of kernals which are 
linear,poly,rbf .we will fit 3 svm model based on their kernals'''

#### Support Vector classifier 
### Kernal='linear'
from sklearn.svm import SVC
clf = SVC(kernel='linear',C=5.0)
clf.fit(x_train, y_train)

### Y predictions
Y_pred_train = clf.predict(x_train)
Y_pred_test  = clf.predict(x_test)

### accuracy score
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train, Y_pred_train)
print("Training Accuracy score:", (ac1*100).round(2))
ac2 = accuracy_score(y_test, Y_pred_test)
print("Test Accuracy score:", (ac2*100).round(2))
print('Variaance between test and train accuracy',(ac1-ac2).round(2))

from sklearn.metrics import confusion_matrix as cm, accuracy_score as ac, classification_report as report,\
roc_curve, roc_auc_score , recall_score , precision_score, f1_score

# plot confusion matrix to describe the performance of classifier.
cm_df=cm(y_test, Y_pred_test)
class_label = ["No", "Yes"]
df_cm = pd.DataFrame(cm_df, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

#ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, Y_pred_test)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, linewidth=2, color='red')
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for SVM Classifier using Polynomial Kernel for Predicting Size_category')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()
ROC_AUC = roc_auc_score(y_test,Y_pred_test)

print('ROC AUC : {:.4f}'.format(ROC_AUC))


#Support Vector classifier
#Kernal='Poly
from sklearn.svm import SVC
clf = SVC(kernel='poly',degree=5.0)
clf.fit(x_train, y_train)

### Y prediictions
Y_pred_train = clf.predict(x_train)
Y_pred_test  = clf.predict(x_test)

### accuracy score
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train, Y_pred_train)
print("Training Accuracy score:", (ac1*100).round(2))
ac2 = accuracy_score(y_test, Y_pred_test)
print("Test Accuracy score:", (ac2*100).round(2))
print('Variaance between test and train accuracy',(ac1-ac2).round(2))

# plot confusion matrix to describe the performance of classifier.
cm_df=cm(y_test, Y_pred_test)
class_label = ["No", "Yes"]
df_cm = pd.DataFrame(cm_df, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

#ROC Curve
from sklearn.metrics import confusion_matrix as cm, accuracy_score as ac, classification_report as report,\
roc_curve, roc_auc_score , recall_score , precision_score, f1_score
fpr, tpr, thresholds = roc_curve(y_test, Y_pred_test)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, linewidth=2, color='red')
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for SVM Classifier using Polynomial Kernel for Predicting Size_category')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()
ROC_AUC = roc_auc_score(y_test,Y_pred_test)

print('ROC AUC : {:.4f}'.format(ROC_AUC))

#### Support Vector classifier
### Kernal=rbf
from sklearn.svm import SVC
clf = SVC(kernel='rbf',gamma='scale')
clf.fit(x_train, y_train)

### Y predictions
Y_pred_train = clf.predict(x_train)
Y_pred_test  = clf.predict(x_test)

# accuracy score
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train, Y_pred_train)
print("Training Accuracy score:", (ac1*100).round(2))
ac2 = accuracy_score(y_test, Y_pred_test)
print("Test Accuracy score:", (ac2*100).round(2))
print('Variaance between test and train accuracy',(ac1-ac2).round(2))

# plot confusion matrix to describe the performance of classifier.
cm_df=cm(y_test, Y_pred_test)
class_label = ["No", "Yes"]
df_cm = pd.DataFrame(cm_df, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

#ROC Curve
from sklearn.metrics import confusion_matrix as cm, accuracy_score as ac, classification_report as report,\
roc_curve, roc_auc_score , recall_score , precision_score, f1_score
fpr, tpr, thresholds = roc_curve(y_test, Y_pred_test)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, linewidth=2, color='red')
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for SVM Classifier using Polynomial Kernel for Predicting Size_category')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()
ROC_AUC = roc_auc_score(y_test,Y_pred_test)

print('ROC AUC : {:.4f}'.format(ROC_AUC))

#### Support Vector classifier
### Kernal= sigmoid
from sklearn.svm import SVC
clf = SVC(kernel='sigmoid',gamma='scale')
clf.fit(x_train, y_train)

### Y predictions
Y_pred_train = clf.predict(x_train)
Y_pred_test  = clf.predict(x_test)

# accuracy score
from sklearn.metrics import accuracy_score
ac1 = accuracy_score(y_train, Y_pred_train)
print("Training Accuracy score:", (ac1*100).round(2))
ac2 = accuracy_score(y_test, Y_pred_test)
print("Test Accuracy score:", (ac2*100).round(2))
print('Variaance between test and train accuracy',(ac1-ac2).round(2))

# plot confusion matrix to describe the performance of classifier.
cm_df=cm(y_test, Y_pred_test)
class_label = ["No", "Yes"]
df_cm = pd.DataFrame(cm_df, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

#ROC Curve
from sklearn.metrics import confusion_matrix as cm, accuracy_score as ac, classification_report as report,\
roc_curve, roc_auc_score , recall_score , precision_score, f1_score
fpr, tpr, thresholds = roc_curve(y_test, Y_pred_test)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, linewidth=2, color='red')
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for SVM Classifier using Polynomial Kernel for Predicting Size_category')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()
ROC_AUC = roc_auc_score(y_test,Y_pred_test)

print('ROC AUC : {:.4f}'.format(ROC_AUC))








































































