#!/usr/bin/env python
# coding: utf-8

# In[1]:


all_song_meta_dict = dict()
with open('mxm_779k_matches.txt','r',encoding="utf8") as f:
    lines = f.readlines()
    for i in range(18, len(lines)):
        line = lines[i].split('<SEP>')
        MSDID = line[0]
        artist = line[1]
        title = line[2]
        all_song_meta_dict[str(MSDID)] = {'artist': artist, 'title': title}


# **CODE FROM:** https://www.kaggle.com/code/taciturno/lyric-based-recommender

# In[2]:


import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
from tqdm import tqdm

remove_these = set(stopwords.words('english'))


# In[3]:


REMOVE_STOPWORDS = True


# In[4]:


with open('mxm_dataset_train.txt','r') as f:
    lines = f.readlines()
    words = lines[17].replace('%','').split(',')
    all_songs_dict = dict()
    for i,l in list(enumerate(lines))[18:]:
        song_info = l.split(',')
        MSDID = song_info[0]
        song_bow = [x.split(':') for x in song_info[2:]]
        song_dict = {}
        for word, word_count in song_bow:
            song_dict[int(word)] = int(word_count.replace('\n',''))
        word_lists = [[words[word-1]]*song_dict[word] for word in song_dict.keys()]
        song = [word for word_list in word_lists for word in word_list]
        if REMOVE_STOPWORDS:
            song = [w for w in song if w not in remove_these]
        all_songs_dict[str(MSDID)] = ' '.join(song).replace('\n','')


# In[5]:


print(len(all_songs_dict.keys()))
song_msd_ids = list(all_songs_dict.keys())


# In[6]:


len(all_song_meta_dict.keys())


# In[7]:


d = {
    'MSDID': list(all_songs_dict.keys()),
    'cleaned_text': [all_songs_dict[x] for x in all_songs_dict.keys()]
    }
msdid_df = pd.DataFrame.from_dict(d)
print(msdid_df.shape)
msdid_df.head()


# In[8]:


d = {
    'MSDID': msdid_df['MSDID'],
    'artist': [all_song_meta_dict[x]['artist'] for x in msdid_df['MSDID']],
    'title': [all_song_meta_dict[x]['title'] for x in msdid_df['MSDID']]
    }
meta_df = pd.DataFrame.from_dict(d)
print(meta_df.shape)
meta_df.head()


# In[9]:


full_df = pd.merge(msdid_df, meta_df, on='MSDID', how='left')
print(full_df.shape)
full_df.head()


# In[10]:


#full_df.to_csv('musixmatch.csv')


# In[11]:


df1=pd.read_csv('msd_new.csv')


# In[13]:


df1.rename(columns = {'track_id':'MSDID'}, inplace = True)


# In[14]:


df1.head()


# In[15]:


df1.info()


# In[15]:


def pair_columns(df, col1, col2):
    return df[col1] + df[col2]

def paired_mask(df1, df2, col1, col2):
    return pair_columns(df1, col1, col2).isin(pair_columns(df2, col1, col2))

identical = df1.loc[paired_mask(df1, full_df, "MSDID", "artist")]


# In[16]:


identical.info()


# In[17]:


def pair_columns(df, col1):
    return df[col1] 

def paired_mask(df1, df2, col1):
    return pair_columns(df1, col1).isin(pair_columns(df2, col1))

identical2 = df1.loc[paired_mask(df1, full_df, "MSDID")]


# In[18]:


identical2.info()


# In[19]:


lyrics_years = full_df.loc[paired_mask(full_df, df1, "MSDID")]


# In[20]:


lyrics_years.info()


# In[21]:


#identical.to_csv('only_msd_lyrics.csv')


# In[25]:


identical2['billboard'].value_counts()


# **BACK TO LYRICS**

# In[26]:


anew=pd.read_csv('Ratings_Warriner_et_al.csv')


# In[27]:


anew.head()


# In[28]:


anew_important=anew[['Word', 'V.Mean.Sum','A.Mean.Sum','D.Mean.Sum']]


# In[29]:


anew_important.rename(columns = {'Word':'word', 'V.Mean.Sum':'valence','A.Mean.Sum':'arousal','D.Mean.Sum':'dominance'}, inplace = True)


# In[30]:


anew_important.head()


# In[31]:


anew_important.to_csv('anew_important.csv')


# In[32]:


test_code=lyrics_years


# In[33]:


test_code = test_code.astype({'cleaned_text':'string'})


# In[34]:


test_code.dtypes


# In[35]:


test_code.info()


# In[36]:


test_code.reset_index(inplace=True)


# In[37]:


test_code.head()


# In[38]:


test_code['cleaned_text'][45]


# In[39]:


anew_important.info()


# In[40]:


anew_important = anew_important.astype({'word':'string'})


# In[41]:


anew_dict=anew_important.set_index('word').T.to_dict('list')
anew_dict


# In[42]:


test_code['valence']=0.0
test_code['arousal']=0.0
test_code['words_count']=0.0
test_code['valence_divided']=0.0
test_code['arousal_divided']=0.0


# In[43]:


for i in range(len(test_code)):
    valence=0
    arousal=0
    words_count=0
    for word in test_code["cleaned_text"][i].split():
        if word in anew_dict:
            valence += anew_dict[word][0]
            arousal += anew_dict[word][1]
            words_count += 1
    test_code["valence"][i]=valence
    test_code["arousal"][i]=arousal
    test_code["words_count"][i]=words_count
    if words_count != 0:
        test_code['valence_divided'][i]=valence/words_count
        test_code['arousal_divided'][i]=arousal/words_count

test_code.head()


# for i in range(len(test_code)):
#     valence=0
#     arousal=0
#     for word in test_code["cleaned_text"][i].split():
#         if word in anew_dict:
#             valence += anew_dict[word][0]
#             arousal += anew_dict[word][1]
#     test_code["valence"][i]=valence
#     test_code["arousal"][i]=arousal
# test_code.head()

# for i in range(len(test_code)):
#     valence=0
#     arousal=0
#     words_count=0
#     for word in test_code["cleaned_text"][i].split():
#         if word in anew_dict:
#             valence += anew_dict[word][0]
#             arousal += anew_dict[word][1]
#             words_count += 1
#     test_code["valence"][i]=valence
#     test_code["arousal"][i]=arousal
#     test_code["words_count"][i]=words_count
# test_code.head()

# In[44]:


test_code = test_code[test_code.words_count > 14]


# In[45]:


test_code.reset_index(inplace=True)


# In[46]:


test_code.info()


# https://github.com/stepthom/lexicon-sentiment-analysis/blob/master/doAnalysis.py
#     
# *use lexicon as dictionary*

# In[47]:


try_test=test_code


# In[48]:


try_test.head()


# In[49]:


#try_test.to_csv("valence_arousal_musixmatch.csv")


# To normalize in [−1,1] you can use:
# 
# x′′=2*(x−minx)/(maxx−minx)−1
# 
# In general, you can always get a new variable x′′′ in [a,b]:
# 
# x′′′=(b−a)x−minxmaxx−minx+a

# In[46]:


try_test["valence_n2"] = 2*((try_test["valence_divided"]-try_test["valence_divided"].min()) / (try_test["valence_divided"].max()-try_test["valence_divided"].min()))-1


# In[47]:


try_test["arousal_n2"] = 2*((try_test["arousal_divided"]-try_test["arousal_divided"].min()) / (try_test["arousal_divided"].max()-try_test["arousal_divided"].min()))-1


# In[48]:


try_test["valence_n1"] = 2*((try_test["valence"]-try_test["valence"].min()) / (try_test["valence"].max()-try_test["valence"].min()))-1


# In[49]:


try_test["arousal_n1"] = 2*((try_test["arousal"]-try_test["arousal"].min()) / (try_test["arousal"].max()-try_test["arousal"].min()))-1


# In[50]:


display(try_test)


# In[51]:


try_test['mood']=""


# In[52]:


for i in range(len(try_test)):
    if try_test["valence_n2"][i] > 0.34:
        if try_test["arousal_n2"][i]>0.34:
            try_test['mood'][i]='happy'
        elif try_test["arousal_n2"][i] < -0.34:
            try_test['mood'][i]='relaxed'
    if try_test["valence_n2"][i] < -0.34:
        if try_test["arousal_n2"][i] > 0.34:
            try_test['mood'][i]='angry'
        elif try_test["arousal_n2"][i] < -0.34:
            try_test['mood'][i]='sad'
        


# for i in range(len(try_test)):
#     if try_test["valence_n2"][i] > 0.1:
#         if try_test["arousal_n2"][i]>0.1:
#             try_test['mood'][i]='happy'
#         elif try_test["arousal_n2"][i] < -0.1:
#             try_test['mood'][i]='relaxed'
#     if try_test["valence_n2"][i] < -0.1:
#         if try_test["arousal_n2"][i] > 0.1:
#             try_test['mood'][i]='angry'
#         elif try_test["arousal_n2"][i] < -0.1:
#             try_test['mood'][i]='sad'

# In[53]:


display(try_test)


# In[54]:


del try_test['level_0']


# In[55]:


try_test.head()


# In[56]:


#try_test.to_csv('musixmatch_years.csv')


# In[57]:


try_test['mood'].value_counts()


# In[58]:


try_test['valence'].idxmax()


# In[59]:


try_test['cleaned_text'][19110]


# In[60]:


try_test['title'][19110]


# In[61]:


try_test['artist'][19110]


# In[62]:


try_test['MSDID'][19110]


# In[63]:


import matplotlib


# In[64]:


try_test['valence'].plot(kind="hist")


# In[65]:


try_test['arousal'].plot(kind="hist")


# In[66]:


try_test['valence_n2'].plot(kind="hist")


# In[67]:


try_test['arousal_n2'].plot(kind="hist")


# In[ ]:




