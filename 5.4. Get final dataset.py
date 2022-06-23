#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


billboard_all=pd.read_csv('billboard_spotifyid_audiof_arousalval.csv')


# In[3]:


billboard_all=billboard_all.drop(columns =['Unnamed: 0'])


# In[4]:


billboard_all.info()


# In[5]:


billboard_all.head()


# In[ ]:





# In[6]:


msd_audio=pd.read_csv('msd_audio_spotify_complete.csv')


# In[7]:


msd_audio=msd_audio.drop(columns =['Unnamed: 0','duration'])


# In[8]:


msd_audio.head()


# In[9]:


msd_lyrics=pd.read_csv('valence_arousal_musixmatch.csv')


# In[10]:


msd_lyrics.info()


# In[11]:


msd_lyrics=msd_lyrics[['title','artist', 'MSDID','valence_divided','arousal_divided']]


# In[12]:


msd_lyrics.rename(columns = {'title':'track'}, inplace = True)


# In[13]:


msd_lyrics.head()


# In[14]:


msd_lyrics.info()


# In[15]:


msd_all = msd_lyrics.merge(msd_audio,left_on=['track','artist','MSDID'],right_on=['track','artist','MSDID'],how='left')


# In[16]:


msd_all.head()


# In[17]:


msd_all.info()


# In[18]:


msd_all = msd_all[msd_all['Spotify_ID'].notna()]


# In[19]:


msd_all.info()


# In[20]:


billboard_all=billboard_all.drop(columns =['date'])


# In[21]:


billboard_all.info()


# In[22]:


msd_all=msd_all.drop(columns =['MSDID','year'])


# In[23]:


msd_all.info()


# In[24]:


final=billboard_all.append([msd_all])


# In[25]:


final.info()


# In[26]:


final=final[final['danceability']!=-999]


# In[27]:


final=final[final['speechiness']!=-999]


# In[28]:


final.info()


# In[29]:


#final.to_csv("final.csv")


# In[30]:


final['billboard'].value_counts()


# In[31]:


final.isnull().values.any()


# In[32]:


import numpy as np


# In[35]:


cols = ['mode', 'acousticness', 'danceability', 'key', 'energy', 'valence', 'speechiness', 'loudness', 'instrumentalness','tempo','liveness', 'duration_ms']
np.round(final[cols].describe(), 2).T[['count','mean', 'std', 'min', 'max']].to_csv('summary stats (partial).csv', sep=',')


# In[38]:


cols2 = ['valence_divided','arousal_divided']
np.round(final[cols2].describe(), 2).transpose().to_csv('summary dimansional.csv', sep=',')


# In[ ]:




