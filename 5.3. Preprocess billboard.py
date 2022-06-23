#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from stemming.porter2 import stem


# In[ ]:





# In[181]:


df = pd.read_csv('billboard-nolyrics-lyrics1.csv')


# In[182]:


df = df[df.lyrics != "nolyrics"]
df=df.drop(5107)
df.reset_index(inplace=True)


# In[183]:


df[5105:5115]


# In[184]:


df=df.drop(columns =['Unnamed: 0', 'index', 'Unnamed: 0.1'])


# In[185]:


df.info()


# In[186]:


#! pip install stemming==1.0


# In[187]:


with open('mxm_dataset_train.txt','r') as f:
    lines = f.readlines()
    words_5000 = lines[17].replace('%','').replace('\n','').split(',')
    print(words_5000)


# **code from** https://github.com/tbertinmahieux/MSongsDB/blob/master/Tasks_Demos/Lyrics/lyrics_to_bow.py *the code to recreate it is this python code*

# In[188]:


def lyrics_to_bow(lyrics):
    """
    Main function to stem and create bag of words.
    It is what we used for the musiXmatch dataset.
    It is heavily oriented towards English lyrics, we apologize for that.
    INPUT
        lyrics as a string
    RETURN
        dictionary word -> count
        or None if something was wrong (e.g. not enough words)
    """
    # remove end of lines
    lyrics_flat = lyrics.replace('\r', '\n').replace('\n', ' ').lower()
    lyrics_flat = ' ' + lyrics_flat + ' '
    # special cases (English...)
    lyrics_flat = lyrics_flat.replace("'m ", " am ")
    lyrics_flat = lyrics_flat.replace("'re ", " are ")
    lyrics_flat = lyrics_flat.replace("'ve ", " have ")
    lyrics_flat = lyrics_flat.replace("'d ", " would ")
    lyrics_flat = lyrics_flat.replace("'ll ", " will ")
    lyrics_flat = lyrics_flat.replace(" he's ", " he is ")
    lyrics_flat = lyrics_flat.replace(" she's ", " she is ")
    lyrics_flat = lyrics_flat.replace(" it's ", " it is ")
    lyrics_flat = lyrics_flat.replace(" ain't ", " is not ")
    lyrics_flat = lyrics_flat.replace("n't ", " not ")
    lyrics_flat = lyrics_flat.replace("'s ", " ")
    # remove boring punctuation and weird signs
    punctuation = (',', "'", '"', ",", ';', ':', '.', '?', '!', '(', ')',
                   '{', '}', '/', '\\', '_', '|', '-', '@', '#', '*')
    for p in punctuation:
        lyrics_flat = lyrics_flat.replace(p, '')
    words = filter(lambda x: x.strip() != '', lyrics_flat.split(' '))
    # stem words
    words = map(lambda x: stem(x), words)
    bow = {}
    for w in words:
        if not w in bow.keys():
            bow[w] = 1
        else:
            bow[w] += 1
    # not big enough? remove instrumental ones among others
    if len(bow) <= 3:
        return None
    # done
    return bow


# In[189]:


lyrics_to_bow(df["lyrics"][5109])


# In[190]:


for i in range(5638):
    strr=""
    lyrics1=lyrics_to_bow(df["lyrics"][i])
    lyrics2 = lyrics1.copy()
    for key in lyrics1:
        if key not in words_5000:
            del lyrics2[key]
    for key in lyrics2:
        strr += (key+" ")*lyrics2[key]
    print(i)
    df.loc[i, 'bow'] = strr


# In[191]:


df.head()


# In[192]:


anew=pd.read_csv('anew_important.csv')
anew=anew[['word', 'valence','arousal','dominance']]
anew.head()


# In[193]:


anew = anew.astype({'word':'string'})


# In[194]:


anew_dict=anew.set_index('word').T.to_dict('list')
anew_dict


# In[195]:


df['valence']=0.0
df['arousal']=0.0
df['words_count']=0.0
df['valence_divided']=0.0
df['arousal_divided']=0.0
df["total_words"]=0.0


# In[196]:


for i in range(len(df)):
    valence=0
    arousal=0
    words_count=0
    total_words=0
    for word in df["bow"][i].split():
        total_words +=1
        if word in anew_dict:
            valence += anew_dict[word][0]
            arousal += anew_dict[word][1]
            words_count += 1
    df["total_words"][i]=total_words
    df["valence"][i]=valence
    df["arousal"][i]=arousal
    df["words_count"][i]=words_count
    if words_count != 0:
        df['valence_divided'][i]=valence/words_count
        df['arousal_divided'][i]=arousal/words_count

df.head()


# In[197]:


df["bow"][0]


# In[198]:


words_count=0
total_words=0
for word in df["bow"][0].split():
        total_words +=1
        if word in anew_dict:
            print(word)
            words_count += 1


# In[199]:


df.reset_index(inplace=True)


# In[200]:


for i in range(len(df)):
    if df["total_words"][i]<100:
        df=df.drop(i)


# In[201]:


(df["total_words"]<100).value_counts()


# In[202]:


df = df[df.words_count > 14]


# In[203]:


df.reset_index(inplace=True)


# In[204]:


#df.to_csv("checklyrics.csv")


# In[205]:


df=df.drop(columns =['index', 'level_0'])


# In[206]:


df[:20]


# In[207]:


df.info()


# In[208]:


#df.to_csv("billboard-arousal-valence.csv")


# *delete rows with less than 100 words - not yet?*
# **delete rows with less than 14 words in the lexicon**

# In[216]:


df=df[['track','artist', 'date','valence_divided','arousal_divided']]


# **add column with 1 for billboard**

# In[217]:


df['billboard']=1


# In[218]:


df.head()


# In[219]:


#df.to_csv("billboard-arousal-valence-important.csv")


# In[220]:


df.info()


# In[ ]:




