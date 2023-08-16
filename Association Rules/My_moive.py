#Importing Libraries
# Supress Warnings
# Supressing Warnings
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np

#Importing Dataset
df=pd.read_csv("D:/Assignment and data set/my_movies.csv")
df

df.head()
df.tail()
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

#plot histgraph
for i in df.columns:
    data=df.copy()
    data[i].hist(bins=10)
    plt.ylabel('Count')
    plt.title(i)
    plt.show()


#Data Visualization
from wordcloud import WordCloud

plt.rcParams['figure.figsize'] = (15, 15)
wordcloud = WordCloud(background_color = 'white', width = 1200,  height = 1200, max_words = 121).generate(str(data.sum()))
plt.imshow(wordcloud)
plt.axis('off')
plt.title('Items',fontsize = 20)
plt.show()

    
#Association rules with 15% Support and 40% confidence
from mlxtend.frequent_patterns import apriori,association_rules
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets
frequent_itemsets.shape

#with 40% Confidence
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
rules
rules.shape
list(rules)

#shorting the rules in ascending 
rules.sort_values('lift',ascending = False)

#shorting the rules in ascending with 10 
rules.sort_values('lift',ascending = False)[0:10]

# Lift Ratio > 1 is a good influential rule in selecting the associated transactions
lift=rules[rules.lift>1]
lift

# visualization of obtained rule
plt.figure(figsize=(16,9))
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()

#Histogram for support , cinfidence and lift
rules[['support','confidence','lift']].hist()

#Scatter plot between support and confidence 
import matplotlib.pyplot as plt
plt.scatter(rules['support'], rules['confidence'])
plt.show()

#scatter plot between support and confidence using seborn
import seaborn as sns
sns.scatterplot('support', 'confidence', data=rules, hue='antecedents')
plt.show()

#Association rules with 20% Support and 60% confidence
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frequent_itemsets
frequent_itemsets.shape

#with 50% Confidence
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.5)
rules
rules.shape
list(rules)

#shorting the rules in ascending 
rules.sort_values('lift',ascending = False)

#shorting the rules in ascending with 20 
rules.sort_values('lift',ascending = False)[0:20]

# Lift Ratio <2 is a good influential rule in selecting the associated transactions
lift= rules[rules.lift<2]
lift

# visualization of obtained rule
plt.figure(figsize=(16,9))
plt.scatter(rules['support'],rules['confidence'])
plt.xlabel('support')
plt.ylabel('confidence') 
plt.show()

#Histogram for support , cinfidence and lift
rules[['support','confidence','lift']].hist()

#Scatter plot between support and confidence 
import matplotlib.pyplot as plt
plt.scatter(rules['support'], rules['confidence'])
plt.show()

#scatter plot between support and confidence using seborn
import seaborn as sns
sns.scatterplot('support', 'confidence', data=rules, hue='antecedents')
plt.show()






















































