#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data_lyrics=pd.read_csv('only_msd_lyrics.csv')


# In[4]:


data_lyrics.head()


# In[5]:


data_lyrics.info()


# In[3]:


billboard=pd.read_csv("billboard_unique.csv")


# In[6]:


billboard.head()


# In[7]:


billboard.info()


# In[8]:


df_all = billboard.merge(data_lyrics.drop_duplicates(), on=['track','artist'], 
                   how='left', indicator=True)
df_all


# In[11]:


df_new=df_all.loc[df_all['_merge'] == 'left_only']


# In[12]:


df_new.info()


# In[13]:


nolyrics=df_new[['track', 'artist','date']]


# In[14]:


nolyrics.head()


# In[15]:


#nolyrics.to_csv('nolyrics_billboard.csv')


# In[ ]:




