"""
Created on Wed Jul 19 11:06:35 2023

@author: Lenovo
"""
#importing libraries
import pandas as pd
import numpy as np

#importing Dataset
df = pd.read_csv("D:/Assignment and data set/LabTAT.csv")
df
df.shape
df.head()
df.tail()

#spiling of x and y
lab1=df.iloc[:,0]
lab2=df.iloc[:,1]
lab3=df.iloc[:,2]
lab4=df.iloc[:,3]

"By using Anova F_oneway test"
#H0 :There is no difference in average TAT among the different laboratories.
#H1 :There is no difference in average TAT among the different laboratories.

from scipy import stats

statics,pval = stats.f_oneway(lab1,lab2,lab3,lab4)


if pval<0.05:
    print("reject the null hypothesis and accepting the alternate hypothesis ")
else:
    print("reject the alternate hypothesis and accepting the null hypothesis ")

#pvalue:2.1156708949992414e-57
#Here pvalue is less 0.05. so, null hypothesis is rejected and accepted alternative hypothesis
#So,there is a difference in average of TAT value among the different laboratories
























