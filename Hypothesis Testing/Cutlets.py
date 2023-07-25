"""
Created on Wed Jul 19 09:16:46 2023

@author: Lenovo
"""
#importing packages
import pandas as pd
import numpy as np

#importing Dataset
df = pd.read_csv("D:/Assignment and data set/Cutlets.csv")
df
df.shape
df.head()
df.tail()

#spiling of x and y
unitA=df.iloc[:,0]
unitB=df.iloc[:,1]
#mean of both units A&B
unitA.mean()
unitB.mean()

"By using two sided z sample test"
#H0 : unitA = unitB
#H1 : unitA != unitB

#import scipy package
import scipy.stats as stats

#for any one side alpha value
stats.norm.ppf(.95).round(2)
#one side alpha value:1.64

from scipy import stats
zcal, pval = stats.ttest_ind(unitA,unitB)
print("zcalculted pvalue is ",zcal.round(4))
print("p-value is ",pval.round(4)) 

#zcalculted pvalue is  0.7229
#p-value is  0.4722

#alpha is 5%=>0.05
if pval<0.05:
    print ("Ho is reject and H1 accepted")
else:
    print ("H1 is reject and H0 accepted")
    

#Here p-value is geater than 0.05. so,here H1 is rejected and Ho is accepted
#There is no significant difference in the diamaters of unit A and unit B (diameteres of the cutlets are same)


