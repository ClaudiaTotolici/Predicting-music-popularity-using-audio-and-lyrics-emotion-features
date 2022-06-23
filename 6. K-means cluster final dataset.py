#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns


# In[52]:


df = pd.read_csv('final.csv')


# In[53]:


df.info()


# In[54]:


df=df.drop(columns =['Unnamed: 0']) 


# In[55]:


df.head()


# **optimal number of clusters**
# https://gist.github.com/sejaldua/54a8aeb0bf9a9b4a66dcbf92535a6ebc#file-optimal-number-of-clusters-py

# **elbow method** https://towardsdatascience.com/k-means-clustering-and-pca-to-categorize-music-by-similar-audio-features-df09c93e8b64

# In[24]:


#! pip install kneed


# In[34]:


from kneed import KneeLocator

# get within cluster sum of squares for each value of k
wcss = []
max_clusters = 21
for i in range(1, max_clusters):
    kmeans= KMeans(i, init='k-means++', random_state=42)
    kmeans.fit(df[['valence_divided','arousal_divided']])
    wcss.append(kmeans.inertia_)
      
# programmatically locate the elbow
n_clusters = KneeLocator([i for i in range(1, max_clusters)], wcss, curve='convex', direction='decreasing').knee
print("Optimal number of clusters", n_clusters)
    
# visualize the curve in order to locate the elbow
fig = plt.figure(figsize=(10,8))
plt.plot(range(1, 21), wcss, marker='o', linestyle='--')
plt.vlines(n_clusters, ymin=0, ymax=max(wcss), linestyles='dashed')
plt.xlabel('Number of Clusters', fontsize=18)
plt.ylabel('Within Cluster Sum of Squares (WCSS)', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()


# In[56]:


kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
kmeans.fit(df[['valence_divided','arousal_divided']])
df.loc[:,'mood_k'] = kmeans.labels_
centroids = kmeans.cluster_centers_


# In[27]:


fig = plt.figure(figsize=(10,8))
plt.scatter(df['valence_divided'], df['arousal_divided'], c= df['mood_k'], s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.xlabel('Average Valence', fontsize=18)
plt.ylabel('Average Arousal', fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()


# In[57]:


df.head()


# In[58]:


sns.scatterplot(x="valence_divided", y="arousal_divided", hue="mood_k", 
                data=df, palette='Paired', s=50);
plt.legend(loc='lower right');
plt.show()
#['0.80','0.63','0.63','0.65','0.94']


# In[65]:


sns.scatterplot(x="valence_divided", y="arousal_divided", hue="mood_k", 
                data=df,palette='Paired', s=50)
plt.legend(title = "Accuracy",
      labels=['0.80','0.63','0.63','0.65','0.94'] )
plt.show()
#palette='Paired'
#, labels=['0.80','0.63','0.63','0.65','0.94']


# In[12]:


#df.to_csv("final+k.csv")


# In[ ]:


ax.legend(handles, ['Yes', 'No'], loc='lower right')

