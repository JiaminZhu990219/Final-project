#!/usr/bin/env python
# coding: utf-8

# In[252]:


import pandas as pd 
import numpy as np


# In[273]:


#read data
users = pd.read_pickle(r'/Users/zdwh/Documents/python/数据/subscribers/subscribers')
service_reps = pd.read_pickle(r'/Users/zdwh/Documents/python/数据/subscribers/customer_service_reps')
engagement = pd.read_pickle(r'/Users/zdwh/Documents/python/数据/subscribers/engagement')


# In[274]:


#Data processing
#Determine if there is churn by current_sub_TF
is_churn = service_reps.loc[:,['subid','current_sub_TF']].drop_duplicates()


# In[275]:


#user behavior data
user_behavior = engagement.groupby('subid').sum()
user_behavior = user_behavior.iloc[:,:-1]


# In[276]:


#Aggregate data
data1 = users.loc[:,
          ['subid','package_type','num_weekly_services_utilized','preferred_genre','intended_use'
           ,'weekly_consumption_hour','num_ideal_streaming_services','age','male_TF','country',
           'attribution_technical','op_sys','plan_type','monthly_price',
           'creation_until_cancel_days','cancel_before_trial_end','initial_credit_card_declined',
           'revenue_net','join_fee','language','paid_TF','refund_after_trial_TF','payment_type']]
#left join 
data2 = data1.merge(user_behavior, on = 'subid',how = 'inner')
data_result = data2.merge(is_churn,on = 'subid',how = 'inner')
data_result.info()


# In[277]:


#fill missing value
#for numerical variables,fill missing value with mean()
data_result['age'] = data_result['age'].fillna(data_result['age'].mean())

#for some numerical variables,missing means not using this kind of service or haven't done it yet,like num_weekly_services_utillzed,so null means 0
data_result['num_weekly_services_utilized'] = data_result['num_weekly_services_utilized'].fillna(0)
data_result['weekly_consumption_hour'] = data_result['weekly_consumption_hour'].fillna(0)
data_result['num_ideal_streaming_services'] = data_result['num_ideal_streaming_services'].fillna(0)
data_result['creation_until_cancel_days'] = data_result['creation_until_cancel_days'].fillna(0)
data_result['revenue_net'] = data_result['revenue_net'].fillna(0)
data_result['join_fee'] = data_result['join_fee'].fillna(0)

#for character variables,using 'unknown' fill missing value
data_result['package_type'] = data_result['package_type'].fillna('unknown')
data_result['preferred_genre'] = data_result['preferred_genre'].fillna('unknown')
data_result['intended_use'] = data_result['intended_use'].fillna('unknown')
data_result['op_sys'] = data_result['op_sys'].fillna('unknown')
data_result['payment_type'] = data_result['payment_type'].fillna('unknown')

print(data_result.info())


# In[278]:


#convert True/False to 0/1
data_result[['cancel_before_trial_end','current_sub_TF','male_TF','initial_credit_card_declined','paid_TF','refund_after_trial_TF']]  = data_result[['cancel_before_trial_end','current_sub_TF','male_TF','initial_credit_card_declined','paid_TF','refund_after_trial_TF']].astype(int)


# In[261]:


#corr
print(data_result.corr()['current_sub_TF'])
import seaborn as sns 
sns.pairplot(data_result,x_vars = ['num_weekly_services_utilized','weekly_consumption_hour','num_ideal_streaming_services','age','male_TF'],y_vars = 'current_sub_TF',height = 3,kind = 'reg')
sns.pairplot(data_result,x_vars = ['monthly_price','creation_until_cancel_days','cancel_before_trial_end','initial_credit_card_declined','revenue_net'],y_vars = 'current_sub_TF',height = 3,kind = 'reg')
sns.pairplot(data_result,x_vars = ['join_fee','paid_TF','refund_after_trial_TF','app_opens','cust_service_mssgs'],y_vars = 'current_sub_TF',height = 3,kind = 'reg')
sns.pairplot(data_result,x_vars = ['num_videos_completed','num_videos_more_than_30_seconds','num_videos_rated','num_series_started'],y_vars = 'current_sub_TF',height = 3,kind = 'reg')


# In[279]:


#choose feature variables
data_result = data_result[[
    'num_weekly_services_utilized','weekly_consumption_hour','num_ideal_streaming_services','male_TF'
    ,'creation_until_cancel_days','initial_credit_card_declined','revenue_net','paid_TF','refund_after_trial_TF'
    ,'app_opens','cust_service_mssgs','num_videos_completed','num_videos_more_than_30_seconds','num_videos_rated',
    'num_series_started','package_type','intended_use','op_sys','plan_type','payment_type','current_sub_TF'
]]
data_result.info()


# In[281]:


## convert character variables to numeric variables
from sklearn.preprocessing import OneHotEncoder
data_result = pd.get_dummies(data_result, columns=[
    "package_type","intended_use"
    ,"op_sys","plan_type","payment_type"])
data_result


# In[286]:


#4.Data Scaling
from sklearn.preprocessing import StandardScaler
# x = StandardScaler().fit_transform(x)
for i in range(data_result.shape[1]):
    if data_result[[data_result.columns[i]]].dtypes[0] == 'float64':
        data_result[data_result.columns[i]] = StandardScaler().fit_transform(data_result[[data_result.columns[i]]])


# In[288]:


#5.Training set and testing set division
from sklearn.model_selection import train_test_split #Invoke the function that randomly divides the training and test sets in cross-validation
x_train,x_test,y_train,y_test = train_test_split(x,y,random_state = 2020221127,test_size = 0.3) #Randomly selected, the testing set accounts for 30%.
#Check the training set and testing set
print(x_train.shape) #Training set x sample size 4773, total 88 feature variables
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[290]:


#6.Logistic regression model fitting
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000) #The default maximum iteration count of 100 will indicate no convergence, so increase the maximum iteration count.
result = model.fit(x_train,y_train)
#Print Model Accuracy
print("train_score:{:.3f}".format(result.score(x_train,y_train)))
print("test_score:{:.3f}".format(result.score(x_test,y_test)))
print("coef_:")
print(result.coef_)


# In[295]:


#prediction
y_pred = model.predict(x_test)

y_pred_proba = model.predict_proba(x_test)
y_pred_proba


# In[299]:


#Model Evaluation
from sklearn.metrics import confusion_matrix
m = confusion_matrix(y_test,y_pred)
#ROC curve 
from sklearn.metrics import roc_curve
fpr,tpr,thres = roc_curve(y_test,y_pred_proba[:,1])
import matplotlib.pyplot as plt 
plt.plot(fpr,tpr)
plt.title('ROC')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.show()


# In[332]:


x_test['Renewal_prob'] = y_pred_proba[:,1]
x_test['current_sub_TF'] = y_test
warning = x_test.loc[x_test['current_sub_TF'] == 1,:]
warning[warning['Renewal_prob']<0.2]

