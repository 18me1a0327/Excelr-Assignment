#Importing Libraries
# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

#Importing Dataset
df=pd.read_csv("D:/Assignment and data set/crime_data.csv")
df

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
#Exploratory Data Analysis
#imported matplot for visualization
import matplotlib.pyplot as plt
import seaborn as sns
# correlation heatmap
f,ax = plt.subplots(figsize=(18,12))
sns.heatmap(df.corr(), annot=True, linewidths =.5, fmt ='.1f',ax=ax)
plt.show()

#plot histgraph
for i in df.columns:
    data=df.copy()
    data[i].hist(bins=10)
    plt.ylabel('Count')
    plt.title(i)
    plt.show()

#boxplot
ot=df.copy() 
fig, axes=plt.subplots(4,1,figsize=(12,8),sharex=False,sharey=False)
sns.boxplot(x='Murder',data=ot,palette='crest',ax=axes[0])
sns.boxplot(x='Assault',data=ot,palette='crest',ax=axes[1])
sns.boxplot(x='UrbanPop',data=ot,palette='crest',ax=axes[2])
sns.boxplot(x='Rape',data=ot,palette='crest',ax=axes[3])
plt.tight_layout(pad=2.0)

#finding outlier
def outlier_function(df,col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3-q1    
    upper = q3+(1.5*iqr)
    lower = q1-(1.5*iqr)
    return lower,upper
outlier_function(df,'Rape')
outlier_function(df,'UrbanPop')
outlier_function(df,'Assault')
outlier_function(df,'Murder')

#
sns.set_style(style='darkgrid')
sns.pairplot(df)

#Splitting of x and y
x = df.iloc[:,1:]


#Data Transformations
#improting  StandardScaler
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()

for column in df.columns:
    if df[column].dtype=='object':
        continue
    df[column]=ss.fit_transform(df[[column]])
df

#Hierarchical clustering
#Merthod=single
#Dendograms
import scipy.cluster.hierarchy as shc
plt.figure(figsize=(10, 7))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(x, method='single'))

#AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='single')
Y = cluster.fit_predict(x)
Y

#creating Y dataframe
Y_new = pd.DataFrame(Y) 
#Y value counts
Y_new.value_counts() 

#Method = complete
#Dendograms
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(30,15))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(x, method='complete'))

#AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='complete')
Y = cluster.fit_predict(x)
Y
#creating Y dataframe
Y_new = pd.DataFrame(Y)  
# Y value counts
Y_new.value_counts()  


#Method = averagee
#Dendograms
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt
plt.figure(figsize=(30,15))  
plt.title("Customer Dendograms")  
dend = shc.dendrogram(shc.linkage(x, method='average'))

#AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='average')
Y = cluster.fit_predict(x)
Y
#creating Y dataframe
Y_new = pd.DataFrame(Y)  
# Y value counts
Y_new.value_counts()  

''' For AgglomerativeClustering i tried with all the three different methods 
    such as single , complete and average. Amoung these methods complete linkage 
    is best clusers '''



#Initializing KMeans clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5,n_init=20)
#Fitting with inputs
kmeans = kmeans.fit(x) 
# Predicting the clusters
Y = kmeans.predict(x)
#creating Y dataframe
Y_new = pd.DataFrame(Y) 
#Y value counts
Y_new.value_counts()  

#Total with in centroid sum of squares 
kmeans.inertia_

clust = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(x)
    clust.append(kmeans.inertia_)

#Elbow method 
plt.scatter(x=range(1,11), y=clust,color='red')
plt.plot(range(1,11), clust,color='black')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertial values')
plt.show()

# DBSCAN
X = df.iloc[:,1:].values 
X

from sklearn.cluster import DBSCAN
dbscan = DBSCAN(eps=3, min_samples=5)
#fitting dbscan
dbscan.fit(X) 

Y = dbscan.labels_
pd.DataFrame(Y).value_counts()

### creating cluster id with dataframe
df["Cluster id"] = pd.DataFrame(Y)
df.head()

#Checking the noise points
noise_points = df[df["Cluster id"] == -1]
noise_points

#final data
Finaldata = df[(df["Cluster id"] == 0)| (df["Cluster id"] == 1)
               |(df["Cluster id"] == 2)].reset_index(drop=True)
Finaldata

"""
For DBSCAN i have taken eps=3 because below 3 more noise points are occuring which 
i do not want that much of outliers after that i kept the noise points out and prepared
 a new final data which has other cluster ids 0's,1's and 2's and kept them in a correct indexing"""





























