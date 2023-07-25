"""
Created on Thu Jul 20 09:51:15 2023

@author: Lenovo
"""
import pandas as pd
import numpy as np
df = pd.read_csv("D:/Assignment and data set/BuyerRatio.csv")
df

df.shape
df.head()
df.tail()

#sperating the values
df_updated = df.iloc[:,1:]

#creating a array with the update data set
np.array(df_updated)

#importing scipy package
import scipy.stats as stats

#here using chi-contingency 
stats.chi2_contingency(df_updated)

#chisqure_value:1.595945538661058,
pvalue=0.6603094907091882
#DregeeofFreedom:3

if pvalue < 0.05:
    print("ho is rejected and h1 is accepted")
else:
    print("h1 is rejected and ho is accepeted ")

#Here pvalue is less then alpha. so, h1 is rejected and ho is accepeted 

 






















