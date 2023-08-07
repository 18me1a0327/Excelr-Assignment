"""
Created on Fri Jul 28 23:30:48 2023
"""
# improting pandas and numpy 
import pandas as pd
import numpy as np

#importing dataset
wine_data = pd.read_csv("D:/Assignment and data set/wine.csv")
wine_data


#Exploratory Data Analysis
wine_data.head()

wine_data.tail()

wine_data.isnull().sum()

wine_data.shape


wine_data.columns

list(wine_data)


#Descriptive Statistics
wine_data.describe()
#Looking for some statistical information about each feature, we can see that the features have very diferrent scales
wine_data.info()

#Exploratory Data Analysis
#Checking the skewness of our dataset.
wine_data.skew()
"""
1.A normally distribuited data has a skewness close to zero.
2.Skewness greather than zero means that there is more weight in the left side of the data.
3.In another hand, skewness smaller than 0 means that there is more weight in the right side of the data"""

#Data visualization
#importing matplotlib and seabron
import matplotlib.pyplot as plt
import seaborn as sn
sn.pairplot(wine_data)
plt.show()
# Outliers Detection
wine_data.plot( kind = 'box', subplots = True, layout = (4,4), sharex = False, sharey = False,color='black')
plt.show()

sn.pairplot(wine_data,palette="dark")
#correlation heatmap
f,ax = plt.subplots(figsize=(18,12))
sn.heatmap(wine_data.corr(), annot=True, linewidths =.5, fmt ='.1f',ax=ax)
plt.show()
"""
The is some Unique points in this correlation matrix:
1.Phenols and Flavanoids is positively correlated, Dilution and Proanthocyanins
2.Flavanoids is positively correlated with Proanthocyanins and Dilution
3.Dilution is positively correlated with Hue
4.Alcohol is positively correlated with Proline"""
#plotting scattter plot Between Phenols and Flavanoids.
plt.scatter(x=wine_data['Phenols'], y=wine_data['Flavanoids'], color='blue',lw=0.1)
plt.xlabel('Phenols')
plt.ylabel('Flavanoids')
plt.title('Data represented by the 2 strongest positively Correlated features',fontweight='bold')
plt.show()

#Data Preprocessing
#Applying Standard Scaler on the Data
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
std = ss.fit_transform(wine_data)
std.shape


# Principal Compound Analysis
from sklearn.decomposition import PCA
pca = PCA(n_components=3)
PC = pca.fit_transform(std)

### Creating a data frame to observe the variance
df = pd.DataFrame(pca.explained_variance_ratio_)
df

### Taking first 3 principal components and creaating a dataframe
pca_df = pd.DataFrame(data=PC, columns=['PC1', 'PC2', 'PC3'])
pca_df


#PCA plot in 2D
# Figure size
plt.figure(figsize=(8,6))

# Scatterplot
plt.scatter(pca_df.iloc[:,0], pca_df.iloc[:,1], s=40)
plt.title('PCA plot in 2D using Strongest Principle Components')
plt.xlabel('PC1')
plt.ylabel('PC2')



#AgglomerativeClustering
from sklearn.cluster import AgglomerativeClustering
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean')
Y = cluster.fit_predict(pca_df)
Y
#creating Y dataframe
Y_new = pd.DataFrame(Y)  
# Y value counts
Y_new.value_counts()  # Y value counts

# Initializing KMeans clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3,n_init=20)
kmeans = kmeans.fit(pca_df) # Fitting with inputs
# Predicting the clusters
Y = kmeans.predict(pca_df)
Y_new = pd.DataFrame(Y)  ### creating Y dataframe
Y_new.value_counts()  ### Y value counts

#### Total with in centroid sum of squares 
kmeans.inertia_
clust = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i,random_state=0)
    kmeans.fit(pca_df)
    clust.append(kmeans.inertia_)

##### Elbow method 

plt.scatter(x=range(1,11), y=clust,color='red')
plt.plot(range(1,11), clust,color='black')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('inertial values')
plt.show()


#Comparing between PCA-based clusters with original "Type" column
#Using AgglomerativeClustering
wine_data['PCA_Cluster'] = Y_new 

#### Using groupby function for Type and PCA_Cluster for comparing average 
cluster_comparison = wine_data.groupby(['Type', 'PCA_Cluster']).mean()
cluster_comparison

#Heirarchical cluster
from scipy.cluster import hierarchy

lk = hierarchy.linkage(PC, method='complete')
dendro = hierarchy.dendrogram(lk)



