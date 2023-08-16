#Importing Libraries
# Supress Warnings
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

#Importing Dataset
df=pd.read_csv("D:/Assignment and data set/book (1).csv",encoding='Latin1')
df

df.head()
df.tail()

# Droping the index Column
df.drop(columns='Unnamed: 0',inplace=True)
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

#Data Visualization
import matplotlib.pyplot as plt
import seaborn as sns

#histgram 
plt.figure(figsize=(10,4))
df['Book.Rating'].hist(bins=70)

#bar  
plt.figure(figsize=(10,6))
df['Book.Rating'].value_counts().plot(kind='bar')
plt.title('Ratings Frequency',  fontsize = 18, fontweight = 'bold')

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

plt.rcParams['figure.figsize'] = (15, 15)
wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121).generate(str(df.sum()))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Items',fontsize = 20)
plt.show()

# pivot Table
books = df.pivot_table(index='User.ID',
                                 columns='Book.Title',
                                 values='Book.Rating')

# Impute those NaNs with 0 values
books.fillna(0,inplace=True)
books

# Calculating Cosine Similarity between Users
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

# Calculating Cosine Similarity between Users on array data
user_sim=1-pairwise_distances(books.values,metric='cosine')
user_sim

# Store the results in a dataframe
user_sim_df = pd.DataFrame(user_sim)
user_sim_df

# Set the index and column names to user ids 
user_sim_df.index=df['User.ID'].unique()
user_sim_df.columns=df['User.ID'].unique()
user_sim_df

# Nullifying diagonal values
np.fill_diagonal(user_sim,0)
user_sim

# Most Similar Users
user_sim.idxmax(axis=1)

# extract the books which userId 162107 & 276726 have watched
df[(df['User.ID']==162107) | (df['User.ID']==276726)]

# extract the books which userId 276729 & 276726 have watched
df[(df['User.ID']==276729) | (df['User.ID']==276726)]
user_1=df[(df['User.ID']==276729)]
user_2=df[(df['User.ID']==276726)]
user_1['Book.Title']
user_2['Book.Title']
pd.merge(user_1,user_2,on='Book.Title',how='outer')





























