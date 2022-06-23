#!/usr/bin/env python
# coding: utf-8

# In[155]:


import pandas as pd

msd_all=pd.read_csv('msd_summary.csv')


# In[156]:


msd = msd_all[msd_all['year'] >= 1990] 


# In[157]:


msd = msd[msd['year'] <= 2010]


# In[158]:


#msd_all.info()


# In[159]:


msd.info()


# In[160]:


msd_important=msd[['title', 'artist_name','year','duration','track_id']]


# In[161]:


msd_important.head()


# In[162]:


billboard=pd.read_csv("billboard_unique.csv")


# In[165]:


billboard = billboard[['track', 'artist']]


# In[166]:


billboard.head()


# In[142]:


#msd_important['billboard'] = np.where(msd_important['artist_name'].isin(billboard['artist']) & msd_important['title'].isin(billboard['track']) , 1, 0)


# for i in range(len(msd_important)):
#     for j in range(len(billboard)):
#         if msd_important['artist_name'][i]==billboard['artist'][j]:
#             if msd_important['title'][i]==billboard['track'][j]:
#                 msd_important['billboard']=1
#             else:
#                 msd_important['billboard']=0
#         else:
#             msd_important['billboard']=0

# In[168]:


billboard['billboard']=1


# In[169]:


msd_important.rename(columns = {'title':'track', 'artist_name':'artist'}, inplace = True)


# In[170]:


msd_important[:15]


# In[171]:


msd_new = msd_important.merge(billboard,left_on=['track','artist'],right_on=['track','artist'],how='left')


# In[172]:


msd_new['billboard'] = msd_new['billboard'].fillna(0).astype(int)


# In[173]:


msd_new[:15]


# In[176]:


msd_new['billboard'].value_counts()


# In[175]:


#msd_new.to_csv("msd_new.csv")


# In[ ]:




