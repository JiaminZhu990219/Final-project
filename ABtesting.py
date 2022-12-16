#!/usr/bin/env python
# coding: utf-8

# In[24]:


import pandas as pd 
import numpy as np


# In[2]:


#read data
users = pd.read_pickle(r'/Users/zdwh/Documents/python/数据/subscribers/subscribers')


# In[46]:


data = users.loc[users['account_creation_date'].astype('string').str[:7] == '2019-07',['subid','plan_type','revenue_net','join_fee']]
data['revenue_all'] = data['revenue_net'] + data['join_fee']
data = data.loc[data['revenue_all'] > 0,:]
x1 = data.loc[data['plan_type'] == 'base_uae_14_day_trial',:]
x2 = data.loc[data['plan_type'] == 'low_uae_no_trial',:]


# In[47]:


# z test
import statsmodels.stats.weightstats as zw
zw.ztest(x1['revenue_all'],x2['revenue_all'],alternative = 'larger')


# In[55]:


data2 = users.loc[users['account_creation_date'].astype('string').str[:7] == '2019-11',['subid','plan_type','revenue_net','join_fee']]
data2['revenue_all'] = data2['revenue_net'] + data2['join_fee']
data2 = data2.loc[data2['revenue_all'] > 0,:]
y1 = data2.loc[data2['plan_type'] == 'base_uae_14_day_trial',:]
y2 = data2.loc[data2['plan_type'] == 'high_uae_14_day_trial',:]


# In[57]:


# z test
import statsmodels.stats.weightstats as zw
zw.ztest(y1['revenue_all'],y2['revenue_all'],alternative = 'smaller')

