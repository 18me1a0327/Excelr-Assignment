# -*- coding: utf-8 -*-
"""Forest_Fires.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1uKDWOADOI8mg4jPPoxmBGepo7k6Nxh8J
"""

from google.colab import files
uploaded = files.upload()

# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import tensorflow as tf

forestfire = pd.read_csv('/content/forestfires.csv')
forestfire

"""Data Exploration"""

# print shape of dataset with rows and columns
print(forestfire.shape)
# print top 5 records
forestfire.head()

"""Data Exploration"""

# print shape of dataset with rows and columns
print(forestfire.shape)
# print top 5 records
forestfire.head()

forestfire.describe()

forestfire.info()

forestfire.isnull().sum()

forestfire.duplicated()

forestfire.columns

forestfire.drop(columns=['dayfri', 'daymon', 'daysat', 'daysun', 'daythu',
       'daytue', 'daywed', 'monthapr', 'monthaug', 'monthdec', 'monthfeb',
       'monthjan', 'monthjul', 'monthjun', 'monthmar', 'monthmay', 'monthnov',
       'monthoct', 'monthsep'],inplace=True)
forestfire

# List of Numerical Variables
numerical_features=[feature for feature in forestfire.columns if forestfire[feature].dtypes != 'O']

print('Number of numerical variables:', len(numerical_features))

# Visualize the numerical variables
forestfire[numerical_features].head()

discrete_feature=[feature for feature in numerical_features if len(forestfire[feature].unique())<25]
print('Discrete Variables Count: {}'.format(len(discrete_feature)))

continuous_feature=[feature for feature in numerical_features if feature not in discrete_feature]
print('Continuous Feature Count {}'.format(len(continuous_feature)))

# find categorical variables

categorical = [var for var in forestfire.columns if forestfire[var].dtype=='O']

print('There are {} categorical variables\n'.format(len(categorical)))

print('The categorical variables are :\n\n', categorical)

"""Data Visualization"""

#Importing Libraries seaborn and matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

fig= plt.figure(figsize=(18, 8))
sns.heatmap(forestfire.corr(), annot=True);
plt.xticks(rotation=45)

sns.set_style('darkgrid')
sns.pairplot(forestfire, hue='size_category')
plt.show()

ot=forestfire.copy()
fig, axes=plt.subplots(7,1,figsize=(14,12),sharex=False,sharey=False)
sns.boxplot(x='FFMC',data=ot,palette='crest',ax=axes[0])
sns.boxplot(x='DMC',data=ot,palette='crest',ax=axes[1])
sns.boxplot(x='DC',data=ot,palette='crest',ax=axes[2])
sns.boxplot(x='ISI',data=ot,palette='crest',ax=axes[3])
sns.boxplot(x='temp',data=ot,palette='crest',ax=axes[4])
sns.boxplot(x='RH',data=ot,palette='crest',ax=axes[5])
sns.boxplot(x='area',data=ot,palette='crest',ax=axes[6])
plt.tight_layout(pad=2.0)

#forestfireAfter Log-Transformation
for feature in continuous_feature:
    data=forestfire.copy()
    data[feature]=np.log(data[feature])
    data.boxplot(column=feature)
    plt.ylabel(feature)
    plt.title(feature)
    plt.show()

"""Data Pre-Processing"""

continuous_feature=[feature for feature in forestfire.columns if forestfire[feature].dtype!='O']
print('Continuous Feature Count {}'.format(len(continuous_feature)))

forestfire[continuous_feature]

from sklearn.preprocessing import StandardScaler
df_standard_scaled = forestfire.copy()
features = df_standard_scaled[continuous_feature]

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

df_standard_scaled[continuous_feature] = scaler.fit_transform(features.values)
df_standard_scaled.head()

"""Label Encoding"""

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
forestfire["day"] = le.fit_transform(forestfire["day"])
forestfire["month"] = le.fit_transform(forestfire["month"])
forestfire["size_category"] = le.fit_transform(forestfire["size_category"])

forestfire

"""Test Train Split With Imbalanced Dataset"""

x = forestfire.drop('size_category',axis=1)
y = forestfire[['size_category']]

# Splitting data into test data and train data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=3)

import keras
from keras.models import Sequential
from keras.layers import Dense
import keras
keras. __version__

forestfire
#assigning predictor variables to x and response variable to y
x = forestfire.drop('size_category',axis=1)
y = forestfire[['size_category']]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state=42)

scaler_train = StandardScaler()
scaler_test = StandardScaler()

x_train_scaled = scaler_train.fit_transform(x_train) # scaling train data -- predictor
x_test_scaled  = scaler_test.fit_transform(x_test) # scaling test data -- predictor

print(x_train_scaled.shape)
print(x_test_scaled.shape)
print(y_train.shape)
print(y_test.shape)

def toFindBestParams(x_train, y_train, x_test, y_test):
    #print(y_test.shape)
    #sys.exit()

    #defining list of hyperparameters
    batch_size_list = [5 , 10 , 15 , 20]
    epoch_list      = [5 , 10 , 50 , 100]

    # initializing the trials
    for batch_trial in batch_size_list:
        for epochs_trial in epoch_list:

            # create ANN model
            model = Sequential()
            # Defining the first layer of the model
            model.add(Dense(units=50, input_dim=x_train.shape[1], kernel_initializer='normal', activation='tanh'))

            # Defining the Second layer of the model
            model.add(Dense(units=6, kernel_initializer='normal', activation='tanh'))

            # The output neuron is a single fully connected node
            # Since we will be predicting a single number
            model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

            # Compiling the model
            model.compile(loss='binary_crossentropy', optimizer='adam',  metrics=['accuracy'])

            # Fitting the ANN to the Training set
            model_trained = model.fit(x_train, y_train ,batch_size = batch_trial, epochs = epochs_trial, verbose=0)

             # Fetching the accuracy of the training
            Accuracy_train = model_trained.history['accuracy'][-1]

            # printing the results of the current iteration
            print('batch_size:', batch_trial,'-', 'epochs:',epochs_trial, 'Accuracy:',Accuracy_train)

# Calling the function
toFindBestParams(x_train, y_train, x_test, y_test)

ann = Sequential()

ann.add(Dense(units=15,activation='relu'))
ann.add(Dense(units=10,activation='relu'))
ann.add(Dense(units=1,activation='sigmoid'))
ann.compile(optimizer='Adam',loss='binary_crossentropy', metrics=['accuracy'])
history = ann.fit(x_train, y_train, validation_split=0.33, batch_size = 10, epochs = 100)

history.history.keys()

# summarize history for accuracy
plt.figure(figsize=(12,8))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.figure(figsize=(12,8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# generating predictions for test data
y_predict_test = ann.predict(x_test)

# creating table with test price & predicted price for test
test_prediction = pd.DataFrame()
test_prediction['Test_Actual'] = y_test.size_category
test_prediction['Test_Probability'] = y_predict_test
def probToBinary(varProb):
    if varProb >= 0.5:
        return 1
    else:
        return 0

# converting the probability of target variable to binary class of test data
test_prediction['Test_Predicted'] = test_prediction['Test_Probability'].apply(probToBinary)
print(test_prediction.shape)
test_prediction.head(10)

# generating predictions for train data
y_predict_train = ann.predict(x_train)

# creating table with test price & predicted price for test
train_prediction = pd.DataFrame()
train_prediction['Train_Actual'] = y_train.size_category
train_prediction['Train_Probability'] = y_predict_train
train_prediction['Train_Predicted'] = train_prediction['Train_Probability'].apply(probToBinary)
print(train_prediction.shape)
train_prediction.head(10)

# plot confusion matrix to describe the performance of classifier.
from sklearn.metrics import confusion_matrix
cm_df=confusion_matrix(test_prediction['Test_Actual'], test_prediction['Test_Predicted'])
class_label = ["No", "Yes"]
df_cm = pd.DataFrame(cm_df, index = class_label, columns = class_label)
sns.heatmap(df_cm, annot = True, fmt = "d")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.show()

from sklearn.metrics import roc_curve,auc,roc_auc_score
fpr, tpr, thresholds = roc_curve(test_prediction['Test_Actual'], test_prediction['Test_Predicted'])
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, linewidth=2, color='red')
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for SVM Classifier using Linear Kernel for Predicting Size_category')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()
ROC_AUC = roc_auc_score(test_prediction['Test_Actual'], test_prediction['Test_Predicted'])
print('ROC AUC : {:.4f}'.format(ROC_AUC))

# plot histogram of predicted probabilities
# adjust the font size
plt.rcParams['font.size'] = 12
# plot histogram with 10 bins
plt.hist(test_prediction['Test_Probability'], bins = 10)
# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of Forest Burned Area')
# set the x-axis limit
plt.xlim(0,1)
# set the title
plt.xlabel('Predicted probabilities of Forest Burned Area')
plt.ylabel('Frequency')