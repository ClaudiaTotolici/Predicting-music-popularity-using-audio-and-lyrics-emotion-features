#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
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


y=df['billboard']
x=df.drop(['billboard'], axis=1)
x_train,x_test,y_train,y_test = train_test_split(x, y,
                                               test_size = 0.20,stratify=y,                                          
                                               random_state=2767)


# In[7]:


important_features = ['mode', 'acousticness', 'danceability', 'key', 'energy', 'valence', 'speechiness', 'loudness', 'instrumentalness','tempo','liveness',
                     'mood_k']

x_train = x_train[important_features]
x_test = x_test[important_features]


# In[8]:


from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

stomek = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'),sampling_strategy = 0.8,random_state=2767)


# **SVM**

# *We need to perform Feature Scaling when we are dealing with Gradient Descent Based algorithms (Linear and Logistic Regression, Neural Network) and Distance-based algorithms (KNN, K-means, SVM) as these are very sensitive to the range of the data points*

# In[9]:


from sklearn.preprocessing import StandardScaler


# In[10]:


sc = StandardScaler()


# In[ ]:





# In[11]:


svm_model = SVC()
print(svm_model.get_params())


# In[12]:


def eval_model(model): 
    y_pred=model.predict(x_test)
    f1=f1_score(y_test, y_pred, average='macro')
    prec=precision_score(y_test, y_pred, average='binary')
    rec=recall_score(y_test, y_pred, average='binary')
    report=classification_report(y_test, y_pred)
    return f1, prec, rec, report


# In[13]:


def eval_model_train(model): 
    pred = model.predict(x_train)
    f1 = f1_score(y_train,pred,average='macro')
    prec = precision_score(y_train,pred)
    rec = recall_score(y_train,pred)
    report=classification_report(y_train, pred)
    return f1, prec, rec, report


# In[14]:


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

# In[15]:


from imblearn.pipeline import Pipeline
model = Pipeline(steps=[('standard',sc),('sampling', stomek),('classification', svm_model)])


# #### We can now use the pipeline created as a normal classifier where resampling will happen when calling fit and disabled when calling decision_function, predict_proba, or predict.

# In[16]:


param = {'classification__C': [0.1, 1, 10, 100, 1000],
              'classification__gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale'],
              #'classification__class_weight':[{0:1,1:1}, 'balanced']
              'classification__kernel': ['rbf']}


# In[17]:


start_time = time.time()
random_search = HalvingGridSearchCV(estimator = model, param_grid=param, cv = 5, verbose=1,random_state=2767,
                                   scoring = 'f1', n_jobs = -1)
#param_grid=param
#### Fit the random search model
random_search.fit(x_train, y_train)
end_time = time.time()


# In[ ]:





# In[18]:


randommodel_time = exec_time(start_time,end_time)
randommodel_time


# In[19]:


random_search.best_params_


# In[20]:


f1_randtest, precision_randtest, recall_randtest, report_randtest= eval_model(random_search)
print("Precision = {} \n Recall = {} \n f1 = {} \n {}".format(precision_randtest, recall_randtest, f1_randtest,report_randtest))


# In[21]:


y_pred=random_search.predict(x_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)


# In[25]:


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


# In[26]:


from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt

class_names = ['Not Hit', 'Hit']

fig, ax = plot_confusion_matrix(conf_mat=cm,
                                show_absolute=True,
                                colorbar=False,
                                class_names=class_names
                               )
plt.show()


# In[22]:


f1_randtrain, precision_randtrain, recall_randtrain, report_randtrain= eval_model_train(random_search)
print("Precision_tr = {} \n Recall_tr = {} \n f1_tr = {} \n {}".format(f1_randtrain, precision_randtrain, recall_randtrain, report_randtrain))


# In[23]:


pred=random_search.predict(x_train)
cm_tr=confusion_matrix(y_train, pred)
print(cm_tr)


# In[ ]:




