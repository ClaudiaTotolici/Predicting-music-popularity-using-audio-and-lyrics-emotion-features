#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd

data=pd.read_csv('billboard.csv',header=None)


# In[38]:


data.columns = ['track', 'artist', 'date']


# In[39]:


data[:10]


# In[26]:


data.info()


# In[33]:


no_duplicates = data.drop_duplicates(
  subset = ['track', 'artist']).reset_index(drop = True)


# In[34]:


no_duplicates.info()


# In[35]:


no_duplicates[:10]


# In[36]:


no_duplicates.to_csv("billboard_unique.csv")


# In[ ]:




