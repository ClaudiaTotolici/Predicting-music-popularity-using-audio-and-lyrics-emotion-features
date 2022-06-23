#!/usr/bin/env python
# coding: utf-8

# import spotipy
# from spotipy.oauth2 import SpotifyClientCredentials
# import pandas as pd
# from tqdm import tqdm
# from requests.exceptions import ReadTimeout
# 
# client_credentials = SpotifyClientCredentials(client_id = '07acea8986b545a5a6ec460f7a24ec5d', client_secret = '2721665094fe453cad5855163f6d9516')
# spotify = spotipy.Spotify(client_credentials_manager = client_credentials,requests_timeout=30, retries=10)
# 
# data=pd.read_csv('only_msd_lyrics.csv')
# 

# import spotipy
# from spotipy.oauth2 import SpotifyClientCredentials
# import pandas as pd
# from tqdm import tqdm
# from requests.exceptions import ReadTimeout
# 
# client_credentials = SpotifyClientCredentials(client_id = 'fa2da88a898e43ccac604af954377de8', client_secret = '7a7cb9fe71884f7d9a81c3c6210e5f81')
# spotify = spotipy.Spotify(client_credentials_manager = client_credentials,requests_timeout=90, retries=10)
# 
# data=pd.read_csv('only_msd_lyrics.csv')

# In[1]:


import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
from tqdm import tqdm
from requests.exceptions import ReadTimeout

client_credentials = SpotifyClientCredentials(client_id = '374866cf80604075ab048f553e2ee5f7', client_secret = '7dc5fae57fdf4d75b871fafb7062779b')
spotify = spotipy.Spotify(client_credentials_manager = client_credentials,requests_timeout=90, retries=10)

data=pd.read_csv('only_msd_lyrics.csv')


# **CODE FROM**: https://github.com/sahilmishra0012/PREDICTING-HIT-SONGS-USING-SPOTIFY-DATA/blob/master/Extract%20Spotify%20ID/MSD/extract_Spotify_ID_MSD.py & https://github.com/sahilmishra0012/PREDICTING-HIT-SONGS-USING-SPOTIFY-DATA/blob/master/Extract%20Features/MSD/extract_MSD_features.py 

# In[2]:


#!pip install spotipy


# In[2]:


data.info()


# In[3]:


del data["Unnamed: 0"]
del data["Unnamed: 0.1"]


# In[4]:


#data=data[:1000]
data.head()


# In[5]:


data=data[120000:]


# In[6]:


data.info()


# In[9]:


ll=[]
for i in tqdm(data.iterrows()):
    results = spotify.search(q='artist:' + str(i[1]['artist']) + ' track:' + str(i[1]['track']), type='track')
    if len(results['tracks']['items'])!=0:
        ll.append(results['tracks']['items'][0]['id'])
    else:
        ll.append(-999)

data['Spotify_ID']=ll


# In[28]:


data.to_csv("msd_spotifyid_120000-finally.csv")


# In[29]:


data.head()


# In[30]:


data['Spotify_ID'].value_counts()


# In[31]:


data=data[data['Spotify_ID']!=-999]


# In[32]:


data = data.drop_duplicates(subset = ["Spotify_ID"])


# In[33]:


data_short=data


# In[34]:


danceability=[]
energy=[]
key=[]
loudness=[]
mode=[]
speechiness=[]
acousticness=[]
instrumentalness=[]
liveness=[]
valence=[]
tempo=[]
duration_ms=[]


# In[35]:


for i in tqdm(data_short.iterrows()):
    results = spotify.audio_features(i[1][6])
    if results[0]!=None:
        danceability.append(results[0]['danceability'])
        energy.append(results[0]['energy'])
        key.append(results[0]['key'])
        loudness.append(results[0]['loudness'])
        mode.append(results[0]['mode'])
        speechiness.append(results[0]['speechiness'])
        acousticness.append(results[0]['acousticness'])
        instrumentalness.append(results[0]['instrumentalness'])
        liveness.append(results[0]['liveness'])
        valence.append(results[0]['valence'])
        tempo.append(results[0]['tempo'])
        duration_ms.append(results[0]['duration_ms'])
    if results[0]==None:
        danceability.append(-999)
        energy.append(-999)
        key.append(-999)
        loudness.append(-999)
        mode.append(-999)
        speechiness.append(-999)
        acousticness.append(-999)
        instrumentalness.append(-999)
        liveness.append(-999)
        valence.append(-999)
        tempo.append(-999)
        duration_ms.append(-999)


# In[36]:


len(danceability)


# In[37]:


data_short['danceability']=danceability
data_short['energy']=energy
data_short['key']=key
data_short['loudness']=loudness
data_short['mode']=mode
data_short['speechiness']=speechiness
data_short['acousticness']=acousticness
data_short['instrumentalness']=instrumentalness
data_short['liveness']=liveness
data_short['valence']=valence
data_short['tempo']=tempo
data_short['duration_ms']=duration_ms


# In[38]:


data_short[:15]


# birdy_uri = 'spotify:artist:2WX2uTcsvV5OnS0inACecP'
# spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id = '07acea8986b545a5a6ec460f7a24ec5d', client_secret = '2721665094fe453cad5855163f6d9516'))
# 
# results = spotify.artist_albums(birdy_uri, album_type='album')
# albums = results['items']
# while results['next']:
#     results = spotify.next(results)
#     albums.extend(results['items'])
# 
# for album in albums:
#     print(album['name'])

# In[39]:


data_short.info()


# In[40]:


data_short.to_csv("msd_spotifyid_120000 - finally(6219)_unique_audiof.csv")


# In[ ]:




