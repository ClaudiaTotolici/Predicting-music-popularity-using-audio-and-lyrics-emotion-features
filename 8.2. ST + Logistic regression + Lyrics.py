#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import time
from sklearn.metrics import classification_report


# In[2]:


df = pd.read_csv('final+k.csv')


# In[3]:


df=df.drop(columns =['Unnamed: 0','track','artist','Spotify_ID','valence_divided','arousal_divided']) 


# In[4]:


df.info()


# In[5]:


df.groupby('billboard').mean()


# In[ ]:





# In[6]:


from sklearn.preprocessing import StandardScaler


# In[7]:


sc = StandardScaler()


# In[8]:


y=df['billboard']
x=df.drop(['billboard'], axis=1)
x_train,x_test,y_train,y_test = train_test_split(x, y,
                                               test_size = 0.20,stratify=y,                                          
                                               random_state=2767)


# In[9]:


important_features = ['mode', 'acousticness', 'danceability', 'key', 'energy', 'valence', 'speechiness', 'loudness', 'instrumentalness','tempo','liveness',
                     'mood_k' ]

x_train = x_train[important_features]
x_test = x_test[important_features]


# In[10]:


#x_train = sc.fit_transform(x_test)
#x_test = sc.fit_transform(x_test)


# In[ ]:





# from imblearn.combine import SMOTETomek
# from imblearn.under_sampling import TomekLinks
# 
# stomek = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'),sampling_strategy = 0.8,random_state=2767)
# x_stomek, y_stomek = stomek.fit_resample(x_train, y_train)

# In[11]:


from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

stomek = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'),sampling_strategy = 0.8,random_state=2767)


# from collections import Counter
# #check new samples
# counter_train = Counter(y_train)
# counter_trains = Counter(y_stomek)
# counter_test = Counter(y_test)
# print(counter_train, counter_trains, counter_test)

# **LOGISTIC REGRESSION**

# import statsmodels.api as sm
# logit_model=sm.Logit(y_stomek,x_stomek)
# result=logit_model.fit()
# print(result.summary2())

# *We need to perform Feature Scaling when we are dealing with Gradient Descent Based algorithms (Linear and Logistic Regression, Neural Network) and Distance-based algorithms (KNN, K-means, SVM) as these are very sensitive to the range of the data points*

# In[ ]:





# In[12]:


lr_model = LogisticRegression()
print(lr_model.get_params())


# In[13]:


def eval_model(model): 
    y_pred=model.predict(x_test)
    f1=f1_score(y_test, y_pred, average='macro')
    prec=precision_score(y_test, y_pred, average='binary')
    rec=recall_score(y_test, y_pred, average='binary')
    report=classification_report(y_test, y_pred)
    return f1, prec, rec, report


# In[14]:


def eval_model_train(model): 
    pred = model.predict(x_train)
    f1 = f1_score(y_train,pred,average='macro')
    prec = precision_score(y_train,pred)
    rec = recall_score(y_train,pred)
    report=classification_report(y_train, pred)
    return f1, prec, rec, report


# In[15]:


def exec_time(start, end):
   diff_time = end - start
   m, s = divmod(diff_time, 60)
   h, m = divmod(m, 60)
   s,m,h = int(round(s, 0)), int(round(m, 0)), int(round(h, 0))
   return f"{h}:{m}:{s}"


# start_time = time.time()
# random_search = RandomizedSearchCV(estimator = lr_model, param_distributions=param, cv = 5, verbose=1, random_state=2767, 
#                                    scoring = ["f1", 'precision', 'recall'], refit = 'precision', n_jobs = -1)
# #### Fit the random search model
# random_search.fit(x_train_std, y_stomek)
# end_time = time.time()

# In[16]:


from imblearn.pipeline import Pipeline
model = Pipeline(steps=[('standard',sc),('sampling', stomek),('classification', lr_model)])


# #### We can now use the pipeline created as a normal classifier where resampling will happen when calling fit and disabled when calling decision_function, predict_proba, or predict.

# In[17]:


param = {'classification__penalty' : ['l1', 'l2','none'],
                                      #, 'elasticnet'],
    'classification__C' : np.logspace(-4, 4, 20),
    'classification__solver' : ['lbfgs','sag','saga'],
         #'newton-cg',
    'classification__max_iter' : [100, 1000,2500, 5000] }


# In[18]:


start_time = time.time()
random_search = GridSearchCV(estimator = model, param_grid=param, cv = 5, verbose=1, 
                                   scoring = ["f1", 'recall'], refit = 'f1', n_jobs = -1)
random_search.fit(x_train, y_train)
end_time = time.time()


# In[ ]:





# In[19]:


randommodel_time = exec_time(start_time,end_time)
randommodel_time


# In[20]:


random_search.best_params_


# In[21]:


f1_randtest, precision_randtest, recall_randtest, report_randtest= eval_model(random_search)
print("Precision = {} \n Recall = {} \n f1 = {} \n {}".format(precision_randtest, recall_randtest, f1_randtest,report_randtest))


# In[22]:


y_pred=random_search.predict(x_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)


# In[23]:


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


# In[24]:


f1_randtrain, precision_randtrain, recall_randtrain, report_randtrain= eval_model_train(random_search)
print("Precision_tr = {} \n Recall_tr = {} \n f1_tr = {} \n {}".format(f1_randtrain, precision_randtrain, recall_randtrain, report_randtrain))


# In[25]:


pred=random_search.predict(x_train)
cm_tr=confusion_matrix(y_train, pred)
print(cm_tr)


# In[ ]:




