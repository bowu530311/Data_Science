#!/usr/bin/env python
# coding: utf-8

# In[1]:


API_KEY = "191caf34002f35b2b3782947f96e96e4-5b18df0a86f5037b652524e7315bb477"
ACCOUNT_ID = "101-011-20224689-001"
OANDA_URL = 'https://api-fxpractice.oanda.com/v3'
#OANDA_URL = 'https://stream-fxpractice.oanda.com/v3'
SECURE_HEADER = {
    'Authorization': f'Bearer {API_KEY}',
    'Content-type': 'application/json'
}

BUY = 1
SELL = -1
NONE = 0

# In[ ]:




