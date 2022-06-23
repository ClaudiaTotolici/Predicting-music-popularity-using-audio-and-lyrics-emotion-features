#!/usr/bin/env python
# coding: utf-8

# In[30]:


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


# In[31]:


df = pd.read_csv('final+k.csv')


# In[32]:


df=df.drop(columns =['Unnamed: 0','track','artist','Spotify_ID','valence_divided','arousal_divided']) 


# In[33]:


df.info()


# In[34]:


df.groupby('billboard').mean()


# In[ ]:





# In[35]:


y=df['billboard']
x=df.drop(['billboard'], axis=1)
x_train,x_test,y_train,y_test = train_test_split(x, y,
                                               test_size = 0.20,stratify=y,                                          
                                               random_state=2767)


# In[36]:


important_features = ['mode', 'acousticness', 'danceability', 'key', 'energy', 'valence', 'speechiness', 'loudness', 'instrumentalness','tempo','liveness']

x_train = x_train[important_features]
x_test = x_test[important_features]


# In[37]:


from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

stomek = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'),sampling_strategy = 0.8,random_state=2767)


# **SVM**

# *We need to perform Feature Scaling when we are dealing with Gradient Descent Based algorithms (Linear and Logistic Regression, Neural Network) and Distance-based algorithms (KNN, K-means, SVM) as these are very sensitive to the range of the data points*

# In[38]:


from sklearn.preprocessing import StandardScaler


# In[39]:


sc = StandardScaler()


# In[ ]:





# In[40]:


svm_model = SVC()
print(svm_model.get_params())


# In[41]:


def eval_model(model): 
    y_pred=model.predict(x_test)
    f1=f1_score(y_test, y_pred, average='macro')
    prec=precision_score(y_test, y_pred, average='binary')
    rec=recall_score(y_test, y_pred, average='binary')
    report=classification_report(y_test, y_pred)
    return f1, prec, rec, report


# In[42]:


def eval_model_train(model): 
    pred = model.predict(x_train)
    f1 = f1_score(y_train,pred,average='macro')
    prec = precision_score(y_train,pred)
    rec = recall_score(y_train,pred)
    report=classification_report(y_train, pred)
    return f1, prec, rec, report


# In[43]:


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

# In[44]:


from imblearn.pipeline import Pipeline
model = Pipeline(steps=[('standard',sc),('sampling', stomek),('classification', svm_model)])


# #### We can now use the pipeline created as a normal classifier where resampling will happen when calling fit and disabled when calling decision_function, predict_proba, or predict.

# In[45]:


param = {'classification__C': [0.1, 1, 10, 100, 1000],
              'classification__gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale'],
              #'classification__class_weight':[{0:1,1:1}, 'balanced']
              'classification__kernel': ['rbf']}


# In[48]:


start_time = time.time()
random_search = HalvingGridSearchCV(estimator = model, param_grid=param, cv = 5, verbose=1,random_state=2767,
                                   scoring = 'f1', n_jobs = -1)
#param_grid=param
#### Fit the random search model
random_search.fit(x_train, y_train)
end_time = time.time()


# In[ ]:





# In[49]:


randommodel_time = exec_time(start_time,end_time)
randommodel_time


# In[50]:


random_search.best_params_


# In[51]:


f1_randtest, precision_randtest, recall_randtest, report_randtest= eval_model(random_search)
print("Precision = {} \n Recall = {} \n f1 = {} \n {}".format(precision_randtest, recall_randtest, f1_randtest,report_randtest))


# In[52]:


y_pred=random_search.predict(x_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)


# In[57]:


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


# In[60]:


def error_analysis(guessed, missed):
    error_analysis_df = pd.concat([guessed, missed], axis=1)
    error_analysis_df.columns=['guessed', 'missed']
    error_analysis_df['diff'] = error_analysis_df.missed - error_analysis_df.guessed
    error_analysis_df['perc_diff'] = 100*error_analysis_df['diff']/error_analysis_df.guessed
    error_analysis_df['abs_perc_diff'] = error_analysis_df['perc_diff'].abs()
    return error_analysis_df.sort_values('abs_perc_diff', ascending=False)


# In[70]:


#pverall missclassification diff
X_test_guessed = x_test[y_pred == y_test]
X_test_missed = x_test[y_pred != y_test]
error_analysis(guessed=X_test_guessed.mean(), missed=X_test_missed.mean())


# In[68]:


error_analysis(guessed=x_train.mean(), missed=x_test.mean()).head(10)


# In[64]:


#for false negatives
false_negatives_mask = (y_pred == 0) & (y_test == 1) 
true_negatives_mask = (y_pred == 0) & (y_test == 0)

X_test_guessed = x_test[false_negatives_mask].mean()
X_test_missed = x_test[true_negatives_mask].mean()
error_analysis(guessed=X_test_guessed, missed=X_test_missed).head(10)


# In[66]:


#for false positives
false_positives_mask = (y_pred == 1) & (y_test == 0)
true_positives_mask = (y_pred == 1) & (y_test == 1)

X_test_guessed = x_test[false_positives_mask].mean()
X_test_missed = x_test[true_positives_mask].mean()
error_analysis(guessed=X_test_guessed, missed=X_test_missed).head(10)


# In[ ]:





# In[53]:


f1_randtrain, precision_randtrain, recall_randtrain, report_randtrain= eval_model_train(random_search)
print("Precision_tr = {} \n Recall_tr = {} \n f1_tr = {} \n {}".format(f1_randtrain, precision_randtrain, recall_randtrain, report_randtrain))


# In[54]:


pred=random_search.predict(x_train)
cm_tr=confusion_matrix(y_train, pred)
print(cm_tr)

