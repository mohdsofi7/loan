#!/usr/bin/env python
# coding: utf-8

# In[24]:


import numpy as np
import pandas as pd
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC


# In[25]:


df=pd.read_excel("Bank_Personal_Loan_Modelling.xlsx")


# In[26]:


X=df.drop('Personal Loan',axis=1)


# In[27]:


y=df['Personal Loan']


# In[28]:


X


# In[29]:


y


# In[ ]:





# In[ ]:





# In[30]:


from imblearn.under_sampling import RandomUnderSampler


# In[31]:


rus = RandomUnderSampler(sampling_strategy=1) # Numerical value
# rus = RandomUnderSampler(sampling_strategy="not minority") # String


# In[32]:


X_res,y_res=rus.fit_resample(X, y)


# In[33]:


y_res.value_counts()


# In[34]:


X_train,X_test,y_train,y_test=train_test_split(X_res,y_res,test_size=0.3,random_state=121)


# In[35]:


rfc= RandomForestClassifier(n_jobs=2, max_features='sqrt')


# In[36]:


param_grid = {
    'n_estimators':[50,100,150,200,250], #no of trees
    'max_depth':[5,10,15,20],            #depth of tree
    'min_samples_split':[4,6,8]}         #no.of sample splits
CV_rfc=RandomizedSearchCV(estimator=rfc,param_distributions=param_grid,cv=10)


# In[37]:


CV_rfc.fit(X_train,y_train)
print(CV_rfc.best_score_,CV_rfc.best_params_)


# In[38]:


y_pred_rfc=CV_rfc.predict(X_test)


# In[39]:


y_pred_rfc


# In[40]:


print('saving model as pkl file.......')
pickle.dump(CV_rfc, open('loan2.pkl','wb'))


# In[41]:


model = pickle.load(open('loan2.pkl','rb'))


# In[ ]:





# In[ ]:





# In[ ]:




