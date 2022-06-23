#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd


# In[3]:


df1=pd.read_csv('msd_spotifyid_18000(14485)_unique_audiof.csv')


# In[4]:


df2=pd.read_csv('msd_spotifyid_18000-40000(17608)_unique_audiof.csv')


# In[5]:


df3=pd.read_csv('msd_spotifyid_40000-47000(5611)_unique_audiof.csv')


# In[6]:


df4=pd.read_csv('msd_spotifyid_47000-60000(10420)_unique_audiof.csv')


# In[7]:


df5=pd.read_csv('msd_spotifyid_60000-80000(15931)_unique_audiof.csv')


# In[8]:


df6=pd.read_csv('msd_spotifyid_80000-100000(15953)_unique_audiof.csv')


# In[9]:


df7=pd.read_csv('msd_spotifyid_100000-120000(15930)_unique_audiof.csv')


# In[10]:


df8=pd.read_csv('msd_spotifyid_120000 - finally(6219)_unique_audiof.csv')


# In[11]:


features=df1.append([df2, df3,df4,df5,df6,df7,df8])


# In[12]:


del features['Unnamed: 0']


# In[13]:


features.head()


# In[14]:


features.info()


# In[15]:


features.to_csv("msd_audio_spotify_complete.csv")


# In[ ]:




