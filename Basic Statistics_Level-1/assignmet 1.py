
'''Q7) Calculate Mean, Median, Mode, Variance, Standard Deviation, Range & comment about the values / draw inferences, for the given dataset
-	For Points, Score, Weigh>

Find Mean, Median, Mode, Variance, Standard Deviation, and Range and also Comment about the values/ Draw some inferences.

Use Q7.csv file '''
import pandas as pd 
import numpy as np
df = pd.read_csv("D:/Excelr Data science/Q7.csv")
df

df.head()
df.tail()
#mean
df.mean()
#ouput:
'''
Points     3.596563
Score      3.217250
Weigh     17.848750
dtype: float64
'''
#median
df.median()
#output:
'''
Points     3.695
Score      3.325
Weigh     17.710
dtype: float64
'''
#mode
#Weigh

df.Weigh.mode()
#output:
'''0    17.02
1    18.90
Name: Weigh, dtype: float64'''
df.Score.mode()
#outpu:
'''0    3.44
Name: Score, dtype: float64'''
#mode
df.Points.mode()
#outpu:
'''
0    3.07
1    3.92
Name: Points, dtype: float64'''
#standard variance
df.std()
#outpu:
'''
Points    0.534679
Score     0.978457
Weigh     1.786943
dtype: float64'''
#variance:
df.var()
#output:
'''
Points    0.285881
Score     0.957379
Weigh     3.193166
dtype: float64'''
#Range:
df.describe()
df_point = df.Points.max() - df.Points.min()
df_weigh = df.Weigh.max() - df.Weigh.min()
df_score = df.Score.max() - df.Score.min()
#output:
'''
points: 2.17
weigh: 8.399999999999999
score: 3.9110000000000005'''
#==============================================================================
'''
Q8) Calculate Expected Value for the problem below
a)	The weights (X) of patients at a clinic (in pounds), are
108, 110, 123, 134, 135, 145, 167, 187, 199
Assume one of the patients is chosen at random. What is the Expected Value of the Weight of that patient?
'''
x = np.array([108, 110, 123, 134, 135, 145, 167, 187, 199])

weights = x.mean() 
weights      
#Output: 145.33333333333334
#==============================================================================

'''
Q9) Calculate Skewness, Kurtosis & draw inferences on the following data
      Car’s speed and distance 
Use Q9_a.csv

'''
import scipy.stats as skew
import scipy.stats as kurtosis
df =pd.read_csv("D:/Excelr Data science/Q9_a.csv")   
df    

df['speed'].skew()
df['speed'].kurtosis()
#output:
#skew:-0.11750986144663393
#kurtosis:-0.5089944204057617

df['dist'].skew()
df['dist'].kurtosis()   
'''#output:
#skew:0.8068949601674215
#kurtosis: 0.4050525816795765  '''
#drawing the  
df_new = df.loc[ :, ['speed','dist']]
df_new.boxplot()
df_new.hist()

#==============================================================================
"""9bSP and Weight(WT)
Use Q9_b.csv"""
df = pd.read_csv("D:/Excelr Data science/Q9_b.csv")
df  
df.skew()
df.kurtosis()
#output:
"""
sp skew:1.6114501961773586      
wt skew:-0.6147533255357768
sp #kurtosis: 2.9773289437871835
wt kurtosis:0.9502914910300326"""
df_new = df.loc[ :, ['SP','WT']]
df_new.boxplot()
df_new.hist()
#==============================================================================
"""Q11) Suppose we want to estimate the average weight of an adult male in    Mexico.
    We draw a random sample of 2,000 men from a population of 3,000,000 men and weigh them.
    We find that the average person in our sample weighs 200 pounds, and the standard deviation of the sample is 30 pounds. 
    Calculate 94%,98%,96% confidence interval?"""
import pandas as pd
import numpy as np
from scipy import stats

#Average weight person in Mexico with 94%=0.96 confidence interval
#alpha =6%
df_ci = stats.norm.interval(0.94,200,30/np.sqrt(2000))
print("Average weight person in Mexico with 94%=0.94 confidence interval:",df_ci)
#output:Average weight person in Mexico with 94%=0.94 confidence interval(198.738325292158, 201.261674707842)

#Average weight person in Mexico with 98%=0.98 confidence interval
#alpha =2%
df_ci = stats.norm.interval(0.98,200,30/np.sqrt(2000))
print("Average weight person in Mexico with 98%=0.98 confidence interval:",df_ci)
#output:Average weight person in Mexico with 98%=0.98 confidence interval(198.43943840429978, 201.56056159570022)

#Average weight person in Mexico with 96%=0.96 confidence interval
#alpha =4%
df_ci = stats.norm.interval(0.96,200,30/np.sqrt(2000))
print("Average weight person in Mexico with 96%=0.96 confidence interval:",df_ci)
#output:Average weight person in Mexico with 96%=0.96 confidence interval:(198.62230334813333, 201.37769665186667)
#==============================================================================
'''
Q12) Below are the scores obtained by a student in tests 
34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56
1)	Find mean, median, variance, standard deviation.
2)	What can we say about the student marks? 
'''
import pandas as pd 
import numpy as np
x = np.array([34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56])


x.mean()
np.median(x)
x.std() 
x.var()
#Output:
'''
Mean = 41.0
Median = 40.5
Std = 4.910306620885412
Var = 24.11111111111111
'''

#==============================================================================
'''Q 20) Calculate probability from the given dataset for the below cases

Data _set: Cars.csv
Calculate the probability of MPG of Cars for the below cases.
       MPG <- Cars$MPG
a.	P(MPG>38)
b.	P(MPG<40)
c.  P (20<MPG<50)'''
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv("D:/Excelr Data science/Cars.csv")
df
mpg =df["MPG"]
mpg
#a.	P(MPG>38)
p=38
n=mpg.mean()
d=mpg.std()
p_38=1-stats.norm.cdf(p,n,d)
print("the probability MPG>38%:",p_38)
#b.P(MPG<40)
p=40
p_40=stats.norm.cdf(p,n,d)
print("the probability MPG>40%:",p_40)
#c.P (20<MPG<50)
p_20_50=stats.norm.cdf(50,n,d) - stats.norm.cdf(20,n,d)
print("The probabilty 20<MPG<50:",p_20_50)
#==============================================================================
'''
Q 21) Check whether the data follows normal distribution
a)	Check whether the MPG of Cars follows Normal Distribution 
        Dataset: Cars.csv
'''  
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
df = pd.read_csv("D:/Excelr Data science/Cars.csv")
df

df['MPG'].hist()

sns.distplot(df['MPG'])
plt.grid(True)
plt.show()
#==============================================================================
'''b)Check Whether the Adipose Tissue (AT) and Waist Circumference(Waist)  from 
wc-at data set  follows Normal Distribution 
       Dataset: wc-at.csv'''
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

df = pd.read_csv("D:/Excelr Data science/wc-at.csv")
df.shape
df
df.mean()
df.median()
df.mode()

sns.distplot(df['Waist'])
plt.show()


sns.distplot(df['AT'])
plt.show()

sns.boxplot(df['AT'])
plt.show()
# mean> median, right whisker is larger than left whisker, data is positively skewed.
sns.boxplot(df['Waist'])
plt.show()
# mean> median, both the whisker are of same lenght, median is slightly shifted towards left. Data is fairly symetrical
#==============================================================================

'''Q 22) Calculate the Z scores of 90% confidence interval,
94% confidence interval, 60% confidence interval 
'''
import pandas as pd 
import numpy as np
from scipy import stats
from scipy.stats import norm 

''' 90% 
#0.90 = c
#alpha = 1 - c= 1- 0.90 = 0.10 
# alpha/2 = 0.5
# 1-0.5 = 0.95'''
# Z-score of 90% confidence interval 
z_score = stats.norm.ppf(0.95)
print("Z-score of 90% confidence interval :",z_score)
# Z-score of 94% confidence interval
stats.norm.ppf(0.97)
print("Z-score of 94% confidence interval :",z_score)
# Z-score of 60% confidence interval
stats.norm.ppf(0.8)
print("Z-score of 60% confidence interval :",z_score)

#==============================================================================
'''
Q 23) Calculate the t scores of 95% confidence interval, 96% confidence interval, 
99% confidence interval for sample size of 25
'''
import numpy as np
from scipy import stats
from scipy.stats import norm

'''
1+0.95/2 
0.97,0.98,0.995
n=25'''
df = 24

t_score=stats.t.ppf(0.97,df) #here df is same
print("t-score of 95% confidence interval:",t_score)


t_score=stats.t.ppf(0.98,df) #here df is same
print("t-score of 96% confidence interval:",t_score)

t_score=stats.t.ppf(0.995,df) #here df is same
print("t-score of 99% confidence interval:",t_score)
"""
t-score of 95% confidence interval: 1.973994288847133
t-score of 96% confidence interval: 2.1715446760080677
t-score of 99% confidence interval: 2.796939504772804
"""
#==============================================================================
''' Q 24) A Government company claims that an average light bulb lasts 270 days.
 A researcher randomly selects 18 bulbs for testing. 
 The sampled bulbs last an average of 260 days, with a standard deviation of 90 days. 
 If the CEO's claim were true, what is the probability that 18 randomly selected bulbs
 would have an average life of no more than 260 days
Hint:  
   rcode   pt(tscore,df)  
 df  degrees of freedom

µ=270, x ̅=260, SD=90, n=18, df=n-1=18-1= 17

 T-score =   (X-µ)/(s/√n)  =   (260-270)/(90/√18)  = -10/21.23
                           = -0.47
'''                           
import pandas as pd
import numpy as np

from scipy import stats
from scipy.stats import norm
stats.t.cdf(-0.47,17)














