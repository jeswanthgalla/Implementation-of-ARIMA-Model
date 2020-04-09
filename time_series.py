#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


# In[2]:


def parser(x):
    return pd.datetime.strptime('190'+x, '%Y-%m')


# In[5]:


df = pd.read_csv('shampoo.csv',header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)


# In[6]:


df.head()


# In[7]:


df.plot(figsize =(10,10))


# In[12]:


pd.plotting.autocorrelation_plot(df)
plt.show()


# In[8]:


X = df.values


# In[11]:


size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()


# In[17]:


for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print("Predicted = %f, Actual = %f" %(yhat,obs))


# In[20]:


error = mean_squared_error(test,predictions)
print(error)


# In[21]:


plt.plot(test)
plt.plot(predictions,color = 'red')
plt.show()


# In[ ]:




