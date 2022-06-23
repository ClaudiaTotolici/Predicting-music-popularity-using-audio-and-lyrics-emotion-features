#!/usr/bin/env python
# coding: utf-8

# **Split data into training_validation_test**

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV, GridSearchCV
from sklearn.metrics import classification_report


# In[21]:


df = pd.read_csv('final+k.csv')


# In[22]:


df=df.drop(columns =['Unnamed: 0','track','artist','Spotify_ID','valence_divided','arousal_divided']) 


# In[23]:


df.info()


# In[24]:


df.describe()


# In[25]:


import plotly.express as px
fig = px.imshow(df.corr())
fig.show()


# In[26]:


y=df['billboard']
x=df.drop(['billboard'], axis=1)
x_train,x_test,y_train,y_test = train_test_split(x, y,
                                               test_size = 0.20,stratify=y,                                          
                                               random_state=2767)
#use stratify to keep the same distribution between hits and nonhits
#y is the target
#stratify=y, 


# **SMOTE_TOMEK**

# In[27]:


#!pip install imblearn


# #### apply it only on training data

# In[28]:


from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

stomek = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'),sampling_strategy = 0.8,random_state=2767)


# from imblearn.combine import SMOTETomek
# from imblearn.under_sampling import TomekLinks
# 
# stomek = SMOTETomek(sampling_strategy = 1.,random_state=2767)
# x_stomek, y_stomek = stomek.fit_resample(x_train, y_train)

# In[29]:


important_features = ['mode', 'acousticness', 'danceability', 'key', 'energy', 'valence', 'speechiness', 'loudness', 'instrumentalness','tempo','liveness', 'mood_k']

x_train = x_train[important_features]
x_test = x_test[important_features]


# In[30]:


print(x_test)


# **RANDOM FOREST RENDOMIZED SEARCH**

# In[31]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import RandomizedSearchCV
import time


# In[32]:


def eval_model_val(model): 
    y_pred=model.predict(x_test)
    f1=f1_score(y_test, y_pred, average='macro')
    prec=precision_score(y_test, y_pred, average='binary')
    rec=recall_score(y_test, y_pred, average='binary')
    report=classification_report(y_test, y_pred)
    return f1, prec, rec, report


# In[33]:


def eval_model_train(model): 
    pred = model.predict(x_train)
    f1 = f1_score(y_train,pred,average='macro')
    prec = precision_score(y_train,pred)
    rec = recall_score(y_train,pred)
    report=classification_report(y_train, pred)
    return f1, prec, rec, report


# In[34]:


def exec_time(start, end):
    diff_time = end - start
    m, s = divmod(diff_time, 60)
    h, m = divmod(m, 60)
    s,m,h = int(round(s, 0)), int(round(m, 0)), int(round(h, 0))
    return f"{h}:{m}:{s}"


# In[35]:


Rf_model = RandomForestClassifier()
print(Rf_model.get_params())


# param = {'n_estimators': [500,750,1000],
#                'max_features': ['auto', 'sqrt'],
#                'max_depth': [10,45,80],
#                'min_samples_split': [2, 10, 15],
#                'min_samples_leaf': [1, 4, 9],
#                'bootstrap': [True, False]}

# In[38]:


param = {'classification__n_estimators': [100,200,300],
               'classification__max_depth': [5,10,20,40],
               #'classification__max_depth': [10],
               'classification__min_samples_split': [2, 7, 10, 15],
               'classification__min_samples_leaf': [1, 5, 15, 30],
               'classification__max_features':['sqrt', 0.2, 0.35],
               #'classification__criterion': ['gini','entropy']
               }


# In[39]:


from imblearn.pipeline import Pipeline
model = Pipeline(steps=[('sampling', stomek),('classification', Rf_model)])


# In[40]:


start_time = time.time()
random_search = GridSearchCV( estimator = model, param_grid=param, cv = 5, verbose=1, 
                                   scoring = ["f1", 'recall'], refit = 'f1', n_jobs = -1)
# Fit the random search model
random_search.fit(x_train, y_train)
end_time = time.time()


# In[41]:


randommodel_time = exec_time(start_time,end_time)
randommodel_time


# In[42]:


random_search.best_params_


# In[43]:


f1_randtest, precision_randtest, recall_randtest, report_randtest= eval_model_val(random_search)
print("Precision = {} \n Recall = {} \n f1 = {} \n {}".format(precision_randtest, recall_randtest, f1_randtest, report_randtest))


# In[44]:


f1_randtrain, precision_randtrain, recall_randtrain, report_randtrain= eval_model_train(random_search)
print("Precision_tr = {} \n Recall_tr = {} \n f1_tr = {} \n {}".format(f1_randtrain, precision_randtrain, recall_randtrain, report_randtrain))


# In[45]:


y_pred=random_search.predict(x_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)


# In[101]:


#!pip install mlxtend


# In[110]:


from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

class_names = ['Not Hit', 'Hit']

fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=False,
                                class_names=class_names
                               )
plt.show()


# In[94]:


import matplotlib.pyplot as plot
from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(random_search, x_test, y_test,cmap='Blues')  


# In[46]:


pred=random_search.predict(x_train)
cm_tr=confusion_matrix(y_train, pred)
print(cm_tr)


# **Try one more random forest**

# In[201]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score


# In[192]:


rf = RandomForestClassifier(n_estimators = 200, random_state = 2767, min_samples_split= 5, min_samples_leaf = 2, max_features = 'sqrt',
                           max_depth = 100, criterion = 'gini', bootstrap = False )


# In[193]:


rf.fit(x_stomek, y_stomek)


# In[194]:


y_pred=rf.predict(x_test)


# In[195]:


f1_score(y_test, y_pred, average='macro')


# In[202]:


precision_score(y_test, y_pred, pos_label=1, average='binary')
#precision_score(y_test, y_pred, average='macro')


# In[203]:


recall_score(y_test, y_pred, average='binary')


# In[205]:


accuracy_score(y_test, y_pred)


# In[204]:


cm=confusion_matrix(y_test, y_pred)
print(cm)


# In[55]:


feature_list=list(x.columns)
# Get numerical feature importances
importances = list(rf.feature_importances_)
# List of tuples with variable and importance
feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
# Sort the feature importances by most important first
feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];


# In[50]:


sns.heatmap(cm, annot=True, cmap='Blues',
                   xticklabels=['No Hit', 'Hit'],
                   yticklabels=['No Hit', 'Hit'])


# In[207]:


print(classification_report(y_test, y_pred))


# In[ ]:




