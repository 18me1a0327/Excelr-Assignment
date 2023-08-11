import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import statsmodels.api as smf
import warnings
warnings.filterwarnings('ignore')


measure_x = [24.23,25.53,25.41,24.14,29.62,28.25,25.81,24.39,40.26,32.95,91.36,25.99,39.42,26.71,35.00]
name_of_company = ['Allied Signal','Bankers Trust','General Mills','ITT Industries','J.P.Morgan & Co.','Lehman Brothers',
      'Marriott','MCI','Merrill Lynch','Microsoft','Morgan Stanley','Sun Microsystems','Travelers','US Airways',
      'Warner-Lambert']
plt.figure(figsize = (8,8))
plt.pie(measure_x, labels = name_of_company, autopct= '%.2f%%',shadow = True,
        explode = [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.1,0.0,0.0,0.0,0.0],
        textprops = {'size':'medium',
                   'fontweight':'bold',
                   'color':'black'})
plt.title('Name of Companies with respect to X', fontsize = 18, fontweight = 'bold', color='b')
plt.savefig('Question1piechart.png')
plt.show()
plt.figure(figsize = (10,9))
ax = sns.barplot(y = measure_x, x = name_of_company, edgecolor='black')
plt.xticks(rotation = 90, fontsize = 12, fontweight = 'bold')
for i in ax.containers:
    ax.bar_label(i,)
plt.show()
sns.boxplot(measure_x)
plt.savefig('question1boxplot.png')
plt.show()
measure_x=pd.Series(measure_x)
Q1 = np.quantile(measure_x,0.25)
Q3 = np.quantile(measure_x,0.75)
med = np.median(measure_x)
IQR = Q3 - Q1
upper_bound = Q3+(1.5*IQR)
lower_bound = Q1-(1.5*IQR)
print('First Quantile=', Q1, 'Second Quantile=', med, 'Third Quantile=', Q3,
      'Inter-Quartile Range=', IQR, 'Upper Whisker=', upper_bound, 'Lower Whisker=', lower_bound)
Outliers = measure_x[(measure_x <= lower_bound) | (measure_x >= upper_bound)]
print('The outlier in the boxplot:',Outliers)
Outliers = measure_x[(measure_x >= upper_bound)]
print('The outlier in the boxplot:',Outliers)
print(measure_x.describe())
print("Variance=",measure_x.var())


