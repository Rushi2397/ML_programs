#!/usr/bin/env python
# coding: utf-8

# In[37]:


import pandas as pd
import pickle


# In[2]:


data=pd.read_csv("Sales_2021.csv")


# In[3]:


data


# In[4]:


data.isnull().sum()


# In[5]:


data.info()


# In[6]:


data.shape


# In[7]:


data.columns


# In[8]:


data.skew()


# In[9]:


data.kurtosis()


# In[10]:


data.cov()


# In[11]:


data.corr()


# In[12]:


data.describe()


# In[13]:


import statistics as s

a=s.harmonic_mean(data['Sales'])
print(a)
b=s.geometric_mean(data['Sales'])
print(b)


# In[14]:


data.plot(kind='box')


# In[15]:


data.plot(kind='hist')


# In[16]:


import statsmodels.api as sm
from statsmodels.formula.api import ols

model=ols('Sales~Advt',data=data).fit()
model1=sm.stats.anova_lm(model)
print(model1)
print(model.summary())


# In[17]:


data['pre1']=model.predict()


# In[18]:


data


# In[19]:


data['res1']=data.Sales-data.pre1


# In[20]:


data


# In[21]:


from scipy.stats import zscore
data['zscore']=zscore(data['res1'])


# In[22]:


data


# In[23]:


data[data['zscore']>1.96]


# # Dummy

# In[24]:


data['Dummy']=data['res1']


# In[25]:


a=data.copy()
for i in range(0,len(a)):
    if (np.any(a['zscore'].values[i]>1.96)):
        data['Dummy'].values[i]=0
    else:
        data['Dummy'].values[i]=1
data


# In[26]:


x=data[['Advt','Dummy']]
y=data['Sales']
x.head()


# In[27]:


import matplotlib.pyplot as plt
plt.scatter(data['res1'],y)


# In[28]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[29]:


x_train.head()


# In[30]:


import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.linear_model import LinearRegression

x_train1=sm.add_constant(x_train)
model=sm.OLS(y_train,x_train1).fit()
print(model.summary())


# In[31]:


from sklearn.linear_model import LinearRegression

regr=LinearRegression()
regr.fit(x_train,y_train)

print('intercept',regr.intercept_)
print('coef',regr.coef_)


# In[32]:


y_pred=regr.predict(x_test)
y_pred


# In[33]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

print("mean absolute error",metrics.mean_absolute_error(y_test,y_pred))
print("mean squared error",metrics.mean_squared_error(y_test,y_pred))
print("root mean squared error",np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# # Forward regression

# In[34]:


from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression

lg=LinearRegression()
sfs1=sfs(lg,k_features=2,forward=True,verbose=2,scoring='neg_mean_squared_error')
sfs1=sfs1.fit(x,y)


# In[35]:


from sklearn.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression

lg=LinearRegression()
sfs1=sfs(lg, k_feature=2,forward=False,verbose=2,scoring='neg_mean_squared_error')
sfs1=sfs1.fit(x,y)


# In[ ]:


from statsmodels.stats.diagnostic import het_breuschpagan

model=ols('y~x',data=data).fit()
_,pvalue,_,_=het_breuschpagan(model.resid,model.model.exog)
print(pvalue)

if pvalue>0.05:
    print("This is Heteroscadisticity")
else:
    print("This is Homoscadisticity")


# In[36]:





# In[ ]:




