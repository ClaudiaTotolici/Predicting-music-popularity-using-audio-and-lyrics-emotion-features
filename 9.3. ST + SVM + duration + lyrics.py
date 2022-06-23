#!/usr/bin/env python
# coding: utf-8

# In[5]:


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


# In[6]:


df = pd.read_csv('final+k.csv')


# In[7]:


df=df.drop(columns =['Unnamed: 0','track','artist','Spotify_ID','valence_divided','arousal_divided']) 


# In[8]:


df.info()


# use log loss to plot outliers with feature on y -> you can do table for mood k, but account for the number for each class, if 10 then it might not work -> accuracy -> ranges for continuous 

# In[9]:


df.groupby('billboard').mean()


# In[30]:


x_test.shape


# In[10]:


y=df['billboard']
x=df.drop(['billboard'], axis=1)
x_train,x_test,y_train,y_test = train_test_split(x, y,
                                               test_size = 0.20,stratify=y,                                          
                                               random_state=2767)


# In[11]:


from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks

stomek = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'),sampling_strategy = 0.8,random_state=2767)


# **SVM**

# *We need to perform Feature Scaling when we are dealing with Gradient Descent Based algorithms (Linear and Logistic Regression, Neural Network) and Distance-based algorithms (KNN, K-means, SVM) as these are very sensitive to the range of the data points*

# In[12]:


from sklearn.preprocessing import StandardScaler


# In[13]:


sc = StandardScaler()


# In[ ]:





# In[14]:


svm_model = SVC()
print(svm_model.get_params())


# In[15]:


def eval_model(model): 
    y_pred=model.predict(x_test)
    f1=f1_score(y_test, y_pred, average='macro')
    prec=precision_score(y_test, y_pred, average='binary')
    rec=recall_score(y_test, y_pred, average='binary')
    report=classification_report(y_test, y_pred)
    return f1, prec, rec, report


# In[16]:


def eval_model_train(model): 
    pred = model.predict(x_train)
    f1 = f1_score(y_train,pred,average='macro')
    prec = precision_score(y_train,pred)
    rec = recall_score(y_train,pred)
    report=classification_report(y_train, pred)
    return f1, prec, rec, report


# In[17]:


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

# In[32]:


from imblearn.pipeline import Pipeline
model = Pipeline(steps=[('standard',sc),('sampling', stomek),('classification', svm_model)])


# #### We can now use the pipeline created as a normal classifier where resampling will happen when calling fit and disabled when calling decision_function, predict_proba, or predict.

# In[19]:


param = {'classification__C': [0.1, 1, 10, 100, 1000],
              'classification__gamma': [1, 0.1, 0.01, 0.001, 0.0001, 'scale'],
              #'classification__class_weight':[{0:1,1:1}, 'balanced']
              'classification__kernel': ['rbf']}


# In[20]:


start_time = time.time()
random_search = HalvingGridSearchCV(estimator = model, param_grid=param, cv = 5, verbose=1,random_state=2767,
                                   scoring = 'f1', n_jobs = -1)
#param_grid=param
#### Fit the random search model
random_search.fit(x_train, y_train)
end_time = time.time()


# In[ ]:





# In[ ]:





# In[21]:


randommodel_time = exec_time(start_time,end_time)
randommodel_time


# In[22]:


random_search.best_params_


# In[23]:


f1_randtest, precision_randtest, recall_randtest, report_randtest= eval_model(random_search)
print("Precision = {} \n Recall = {} \n f1 = {} \n {}".format(precision_randtest, recall_randtest, f1_randtest,report_randtest))


# In[24]:


y_pred=random_search.predict(x_test)
cm=confusion_matrix(y_test, y_pred)
print(cm)


# In[49]:


y_pred.shape


# In[46]:


#!pip install interpret-community
#!pip install raiwidgets


# In[27]:


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


# **Error analysis**

# In[50]:


def error_analysis(guessed, missed):
    error_analysis_df = pd.concat([guessed, missed], axis=1)
    error_analysis_df.columns=['guessed', 'missed']
    error_analysis_df['diff'] = error_analysis_df.missed - error_analysis_df.guessed
    error_analysis_df['perc_diff'] = 100*error_analysis_df['diff']/error_analysis_df.guessed
    error_analysis_df['abs_perc_diff'] = error_analysis_df['perc_diff'].abs()
    return error_analysis_df.sort_values('abs_perc_diff', ascending=False)


# In[66]:


false_negatives_mask = (y_pred == 0) & (y_test == 1) 
true_negatives_mask = (y_pred == 0) & (y_test == 0)
guessed = x_test['mood_k'][false_negatives_mask]
missed = x_test['mood_k'][true_negatives_mask]
error_analysis_df = pd.concat([guessed, missed], axis=1)
error_analysis_df.columns=['FN', 'TN']
error_analysis_df[600:620]
error_analysis_df.FN.value_counts()


# In[87]:


#error_analysis_df.TN.value_counts()


# In[86]:


false_positives_mask = (y_pred == 1) & (y_test == 0)
true_positives_mask = (y_pred == 1) & (y_test == 1)
guessed = x_test['mood_k'][false_positives_mask]
missed = x_test['mood_k'][true_positives_mask]
initial = x_test['mood_k']
error_analysis_df = pd.concat([guessed, missed, initial], axis=1)
error_analysis_df.columns=['FP', 'TP', 'all']
error_analysis_df[600:620]
error_analysis_df.FP.value_counts()


# In[85]:


error_analysis_df.info()


# In[81]:


error_analysis_df['all'].value_counts()


# In[105]:


#for false positives - FP/TN
false_positives_mask = (y_pred == 1) & (y_test == 0) 
true_negatives_mask = (y_pred == 0) & (y_test == 0)

X_test_guessed = x_test[true_negatives_mask].mean()
X_test_missed = x_test[false_positives_mask].mean()
error_analysis(guessed=X_test_guessed, missed=X_test_missed)
#.to_csv("falsepositives - FP-TN.csv")

# the model is more likely to missclassify the non-hit songs with a lower instrumentalness. 
# the non-hits that are wrongly classified as hits have higher danceability and valence scores.


# In[108]:


#for false negative - FN/TP
false_neagtives_mask = (y_pred == 0) & (y_test == 1)
true_positives_mask = (y_pred == 1) & (y_test == 1)

X_test_guessed = x_test[true_positives_mask].mean()
X_test_missed = x_test[false_negatives_mask].mean()
error_analysis(guessed=X_test_guessed, missed=X_test_missed)
#.to_csv("falsenegative - FN-TP.csv")


# In[98]:


#for false negatives - FN/TN
false_negatives_mask = (y_pred == 0) & (y_test == 1) 
true_negatives_mask = (y_pred == 0) & (y_test == 0)

X_test_guessed = x_test[true_negatives_mask].mean()
X_test_missed = x_test[false_negatives_mask].mean()
error_analysis(guessed=X_test_guessed, missed=X_test_missed)
#.to_csv("falsenegatives - FN/TN.csv")

#the model is more likely to wrongly predict as non-hit songs the tracks with a lower instrumentalness
#the songs(hits) that are wrongly predicted as non-hit songs have lower instrumentalness levels compared to the actual non-hit songs


# In[97]:


#for false positives
false_positives_mask = (y_pred == 1) & (y_test == 0)
true_positives_mask = (y_pred == 1) & (y_test == 1)

X_test_guessed = x_test[true_positives_mask].mean()
X_test_missed = x_test[false_positives_mask].mean()
error_analysis(guessed=X_test_guessed, missed=X_test_missed)
#.to_csv("falsepositives.csv")


# import shap
# #Fits the explainer
# explainer = shap.Explainer(random_search)
# #Calculates the SHAP values - It takes some time
# shap_values = explainer(x_test)

# In[ ]:


f1_randtrain, precision_randtrain, recall_randtrain, report_randtrain= eval_model_train(random_search)
print("Precision_tr = {} \n Recall_tr = {} \n f1_tr = {} \n {}".format(f1_randtrain, precision_randtrain, recall_randtrain, report_randtrain))


# In[26]:


pred=random_search.predict(x_train)
cm_tr=confusion_matrix(y_train, pred)
print(cm_tr)


# In[ ]:




