#!/usr/bin/env python
# coding: utf-8

# In[21]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import pickle


# In[4]:


dataframe = pd.read_csv("Data/train.csv")


# In[5]:


dataframe.isna().sum()


# In[15]:


dataframe= dataframe.dropna().reset_index()


# In[18]:


dataframe = dataframe.drop(['index'],axis=1)


# In[19]:


dataframe.sample()


# In[20]:


regressor = LinearRegression()


# In[27]:


X = np.array(dataframe['x']).reshape(-1,1)
Y = np.array(dataframe['y']).reshape(-1,1)


# In[28]:


regressor.fit(X,Y)


# In[29]:


test = pd.read_csv("Data/test.csv")


# In[31]:


test.shape


# In[32]:


cross_val = test.head(298)
test_data = test.tail(2)


# In[34]:


test_data.to_csv('Data/live_test.csv')


# In[38]:


X_cross = np.array(test['x']).reshape(-1,1)
Y_cross = np.array(test['y']).reshape(-1,1)


# In[39]:


y_pred = regressor.predict(X_cross)


# In[41]:


r2_score(y_pred,Y_cross)


# In[43]:


file = open('pickle/linearreg.pkl','wb')
pickle.dump(regressor,file)
file.close()


# In[ ]:




