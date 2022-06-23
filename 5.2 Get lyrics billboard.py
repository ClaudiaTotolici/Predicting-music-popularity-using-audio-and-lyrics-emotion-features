#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
from bs4 import BeautifulSoup
from lyricsgenius import Genius
from requests.exceptions import ReadTimeout


# In[24]:


df = pd.read_csv('nolyrics_billboard.csv')


# In[25]:


del df['Unnamed: 0']


# In[26]:


df.head()


# In[27]:


df.info()


# In[28]:


df = df.astype({'track':'string', 'artist':'string'})


# CLIENT ID
# ebUKObXHBe8HIbTNI8mCZnsNvXiwL3nRyKiq-vt_VkHaH9eFek6wcib8DOhkBSKR
# CLIENT SECRET
# 5Zx30xHoyPRYXQdj5BGC5YjPWMmhB4pll1DEEWqTwQ1TLfTeG9kp6Z4dtRxzHv_u1d_i-Pch_jHRqc74RQgweg

# In[29]:


access_token = 'isPIAjp-nOQFYUPTOed_1cpHE0a6B-fIlyvCRr86tELyFfCjbspdfeWRtQ9nFJ2k'


# In[30]:


df1=df[0:500]


# In[34]:


genius = Genius(access_token, timeout=40, retries=5)
genius.verbose = False # Turn off status messages
genius.remove_section_headers = True # Remove section headers (e.g. [Chorus]) from lyrics when searching
for i in range(len(df1)):
    song = genius.search_song(df1["artist"][i], df1["track"][i])
    try:
        lyrics = song.lyrics
        df1.loc[i, 'lyrics'] = lyrics
        print(i)
    except AttributeError:
        df1.loc[i, 'lyrics'] = 'nolyrics'
        print("no")
        
df1.head()


# In[35]:


#df1.to_csv('df1lyrics.csv')


# In[50]:


df2=df[500:1000]


# In[51]:


df2.reset_index(inplace=True)


# In[52]:


del df2['index']


# In[53]:


df2.head()


# In[54]:


genius = Genius(access_token, timeout=60, retries=5)
genius.verbose = False # Turn off status messages
genius.remove_section_headers = True # Remove section headers (e.g. [Chorus]) from lyrics when searching
for i in range(len(df2)):
    song = genius.search_song(df2["artist"][i], df2["track"][i])
    try:
        lyrics = song.lyrics
        df2.loc[i, 'lyrics'] = lyrics
        print(i)
    except AttributeError:
        df2.loc[i, 'lyrics'] = 'nolyrics'
        print("no")
df2.head()


# In[55]:


#df2.to_csv('df2lyrics.csv')


# In[56]:


df3=df[1000:1500]


# In[57]:


df3.reset_index(inplace=True)


# In[58]:


del df3['index']


# In[59]:


genius = Genius(access_token, timeout=60, retries=5)
genius.verbose = False # Turn off status messages
genius.remove_section_headers = True # Remove section headers (e.g. [Chorus]) from lyrics when searching
for i in range(len(df3)):
    song = genius.search_song(df3["artist"][i], df3["track"][i])
    try:
        lyrics = song.lyrics
        df3.loc[i, 'lyrics'] = lyrics
        print(i)
    except AttributeError:
        df3.loc[i, 'lyrics'] = 'nolyrics'
        print("no")
df3.head()


# In[60]:


#df3.to_csv('df3lyrics.csv')


# In[61]:


df4=df[1500:2000]


# In[62]:


df4.reset_index(inplace=True)


# In[63]:


del df4['index']


# In[64]:


genius = Genius(access_token, timeout=60, retries=5)
genius.verbose = False # Turn off status messages
genius.remove_section_headers = True # Remove section headers (e.g. [Chorus]) from lyrics when searching
for i in range(len(df4)):
    song = genius.search_song(df4["artist"][i], df4["track"][i])
    try:
        lyrics = song.lyrics
        df4.loc[i, 'lyrics'] = lyrics
        print(i)
    except AttributeError:
        df4.loc[i, 'lyrics'] = 'nolyrics'
        print("no")
df4.head()


# In[65]:


#df4.to_csv('df4lyrics.csv')


# In[70]:


df5=df[2000:2500]


# In[71]:


df5.reset_index(inplace=True)


# In[72]:


del df5['index']


# In[73]:


genius = Genius(access_token, timeout=60, retries=5)
genius.verbose = False # Turn off status messages
genius.remove_section_headers = True # Remove section headers (e.g. [Chorus]) from lyrics when searching
for i in range(len(df5)):
    song = genius.search_song(df5["artist"][i], df5["track"][i])
    try:
        lyrics = song.lyrics
        df5.loc[i, 'lyrics'] = lyrics
        print(i)
    except AttributeError:
        df5.loc[i, 'lyrics'] = 'nolyrics'
        print("no")
df5.head()


# In[74]:


#df5.to_csv('df5lyrics.csv')


# In[75]:


df6=df[2500:3500]


# In[76]:


df6.reset_index(inplace=True)


# In[77]:


del df6['index']


# In[78]:


genius = Genius(access_token, timeout=60, retries=5)
genius.verbose = False # Turn off status messages
genius.remove_section_headers = True # Remove section headers (e.g. [Chorus]) from lyrics when searching
for i in range(len(df6)):
    song = genius.search_song(df6["artist"][i], df6["track"][i])
    try:
        lyrics = song.lyrics
        df6.loc[i, 'lyrics'] = lyrics
        print(i)
    except AttributeError:
        df6.loc[i, 'lyrics'] = 'nolyrics'
        print("no")
df6.head()


# In[79]:


#df6.to_csv('df6lyrics.csv')


# In[81]:


df7=df[3500:5963]


# In[82]:


df7.reset_index(inplace=True)


# In[83]:


del df7['index']


# In[84]:


genius = Genius(access_token, timeout=60, retries=5)
genius.verbose = False # Turn off status messages
genius.remove_section_headers = True # Remove section headers (e.g. [Chorus]) from lyrics when searching
for i in range(len(df7)):
    song = genius.search_song(df7["artist"][i], df7["track"][i])
    try:
        lyrics = song.lyrics
        df7.loc[i, 'lyrics'] = lyrics
        print(i)
    except AttributeError:
        df7.loc[i, 'lyrics'] = 'nolyrics'
        print("no")
df7.head()


# In[85]:


df7.to_csv('df7lyrics.csv')


# In[86]:


df_new=df1.append([df2,df3,df4,df5,df6,df7])


# In[87]:


df_new.info()


# In[88]:


df_new.head()


# In[90]:


df_new.to_csv("billboard-nolyrics-lyrics.csv")


# In[ ]:




