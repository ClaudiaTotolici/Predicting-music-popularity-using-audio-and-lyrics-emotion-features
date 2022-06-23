#!/usr/bin/env python
# coding: utf-8

# In[1]:


import billboard
import time
import csv
import datetime


# In[13]:


previousDate=datetime.date(2010, 12, 25) 
print(previousDate)
chart = billboard.ChartData('hot-100',previousDate)
ranks=[]
for k in range(1094):
    chart = billboard.ChartData('hot-100',previousDate)
    for i in chart:
        ranks.append(tuple((i.title,i.artist,chart.date)))
    previousDate=previousDate-datetime.timedelta(days=7)
    print(previousDate)
print (ranks)
with open("billboard.csv", "a", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(ranks)


# In[ ]:




