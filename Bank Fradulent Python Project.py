#!/usr/bin/env python
# coding: utf-8

# In[ ]:


###### Step1. I have bank data from 2009 to 2014. First we will do EDA and data validation.First of all we will identify which are numeric variables and how many are categorical variables.


# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data=pd.read_csv("C:\\Users\\Dell 5370\\Desktop\\bank.csv")


# In[5]:


data.head()


# ######  Here the first two columns obs and account_id should not have any impact on the output. So we can drop the first two columns.

# In[6]:


data.drop(['Obs','account_id'],axis=1, inplace=True)


print(data.info())
print(data.shape)


# In[7]:


data.columns


# ###### 5 are categorical variables and 33 are numeric variables. There are no missing values. 682 observations are there and 38 columns out of which 37 are features and 1 is target variable.

# In[8]:


#print the categorical variables if any one of them contains too many unique values. 
#In such cases we have to do something to reduce the unique values by clubing some of them together.
print('Sex=',data['sex'].unique())
print('card=',data['card'].unique())
print('Second=',data['second'].unique())
print('Frequency=',data['frequency'].unique())
print('Region=',data['region'].unique())
print('good=',data['good'].unique())


# ###### We have confirmed the basic sanity of the data.Now we will do some visualisations.

# In[9]:


data.plot(kind='box', subplots=True, layout=(16,2), sharex=False, sharey=False,figsize=(20, 40))
plt.show()


# In[10]:


data.hist(layout=(16,2),figsize=(20, 40))
plt.show()


# In[11]:


data.describe()


# In[12]:


data.cov()


# In[13]:


data.kurtosis()


# In[14]:


data.skew()


# ###### From the boxplots the following features seems to have outliars: 1)cardwdln 2)cardwdlt 3)bankcolt 4) bankrn 5)cardwdlnd 6)othcrnd 7)acardwdl 8)cashwdt 9) cardwdltd . These are the main columns with outliars. We will calculate the z-score of each elements of these columns and replace them with the corresponding median.

# In[15]:


from scipy import stats
z1=stats.zscore(data['cardwdln'])
z2=stats.zscore(data['cardwdlt'])
z3=stats.zscore(data['bankcolt'])
z4=stats.zscore(data['bankrn'])
z5=stats.zscore(data['cardwdlnd'])
z6=stats.zscore(data['othcrnd'])
z7=stats.zscore(data['acardwdl'])
z8=stats.zscore(data['cashwdt'])
z9=stats.zscore(data['cardwdltd'])


# In[16]:


#insert the calculated z-Score into the dataframe
data.insert(0,"Z-Score_cardwdln", list(z1), True)
data.insert(0,"Z-Score_cardwdlt", list(z2), True) 
data.insert(0,"Z-Score_bankcolt", list(z3), True) 
data.insert(0,"Z-Score_bankrn", list(z4), True) 
data.insert(0,"Z-Score_cardwdlnd", list(z5), True) 
data.insert(0,"Z-Score_othcrnd", list(z6), True) 
data.insert(0,"Z-Score_acardwdl", list(z7), True) 
data.insert(0,"Z-Score_cashwdt", list(z8), True) 
data.insert(0,"Z-Score_cardwdltd", list(z9), True) 


# In[17]:


data.head()


# In[18]:


#testing How I can filter out the high z-scores from a single column
data[data['Z-Score_cardwdltd']>1.96]['cardwdltd']


# In[19]:


# Filtering out the extreme z-scores from the required columns 
# and imputing NaN values in the corresponding columns

data.loc[data['Z-Score_cardwdln']>1.96,'cardwdln']=np.nan
data.loc[data['Z-Score_cardwdln']<-1.96,'cardwdln']=np.nan

data.loc[data['Z-Score_cardwdlt']>1.96,'cardwdlt']=np.nan
data.loc[data['Z-Score_cardwdlt']<-1.96,'cardwdlt']=np.nan

data.loc[data['Z-Score_bankcolt']>1.96,'bankcolt']=np.nan
data.loc[data['Z-Score_bankcolt']<-1.96,'bankcolt']=np.nan

data.loc[data['Z-Score_bankrn']>1.96,'bankrn']=np.nan
data.loc[data['Z-Score_bankrn']<-1.96,'bankrn']=np.nan

data.loc[data['Z-Score_cardwdlnd']>1.96,'cardwdlnd']=np.nan
data.loc[data['Z-Score_cardwdlnd']<-1.96,'cardwdlnd']=np.nan

data.loc[data['Z-Score_othcrnd']>1.96,'othcrnd']=np.nan
data.loc[data['Z-Score_othcrnd']<-1.96,'othcrnd']=np.nan

data.loc[data['Z-Score_acardwdl']>1.96,'acardwdl']=np.nan
data.loc[data['Z-Score_acardwdl']<-1.96,'acardwdl']=np.nan

data.loc[data['Z-Score_cashwdt']>1.96,'cashwdt']=np.nan
data.loc[data['Z-Score_cashwdt']<-1.96,'cashwdt']=np.nan

data.loc[data['Z-Score_cardwdltd']>1.96,'cardwdltd']=np.nan
data.loc[data['Z-Score_cardwdltd']<-1.96,'cardwdltd']=np.nan


# In[20]:


data.info()


# ###### We See that those columns has some blank values in them now.

# In[21]:


# save a copy of the data to disc.
data.to_csv('see.csv')


# In[22]:


data['bankrn'].median()


# In[23]:


# imputing the median values in place of the NaN values
data['cardwdln']=data['cardwdln'].fillna(data['cardwdln'].median())
data['cardwdlt']=data['cardwdlt'].fillna(data['cardwdlt'].median())
data['bankcolt']=data['bankcolt'].fillna(data['bankcolt'].median())
data['bankrn']=data['bankrn'].fillna(data['bankrn'].median())
data['cardwdlnd']=data['cardwdlnd'].fillna(data['cardwdlnd'].median())
data['othcrnd']=data['othcrnd'].fillna(data['othcrnd'].median())
data['acardwdl']=data['acardwdl'].fillna(data['acardwdl'].median())
data['cashwdt']=data['cashwdt'].fillna(data['cashwdt'].median())
data['cardwdltd']=data['cardwdltd'].fillna(data['cardwdltd'].median())


# In[24]:


# getting rid of the z-Score columns
data=data.drop(['Z-Score_cardwdltd', 'Z-Score_cashwdt','Z-Score_acardwdl','Z-Score_othcrnd','Z-Score_cardwdlnd','Z-Score_bankrn','Z-Score_bankcolt','Z-Score_cardwdlt','Z-Score_cardwdln'], axis=1)
data.shape


# ###### Check again whether the data quality has improved or not after having removed the outliars.

# In[25]:


data.plot(kind='box', subplots=True, layout=(16,2), sharex=False, sharey=False,figsize=(20, 40))
plt.show()


# In[27]:


data.hist(layout=(16,2),figsize=(20, 40))
plt.show()


# ######  After removing the outliars data doesnot seem to be improved much. However we will start modelling now.

# In[28]:


# Split the features and the target variables first

X=data.iloc[:,0:37]
y=data.iloc[:,-1]


# In[29]:


X.head()


# ###### But we have categorical variables in the training data set. and those variables are 1)sex 2)card 3)second 4) frequency 5) region .  So we have to do do label encoding here. We also have to do label encoding for the target variable.

# ###### The following code will create dummy variables and also give appropriate column names for them. You just input the list of columns to be encoded.

# In[30]:


cat_vars=['sex','card','second','frequency','region']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(X[var], prefix=var)
    data1=X.join(cat_list)
    X=data1
cat_vars=['sex','card','second','frequency','region']
data_vars=X.columns.values.tolist()
to_keep=[i for i in data_vars if i not in cat_vars]


# In[31]:


#Our final data columns will be:

X_data_final=X[to_keep]
X_data_final.columns.values


# In[32]:


#Encode the y variable as well
from sklearn.preprocessing import LabelEncoder
labelencoder_y=LabelEncoder()
y_data_final=labelencoder_y.fit_transform(y)
y_data_final


# ###### Now we are good to do the modelling

# In[35]:



from sklearn.metrics import confusion_matrix , classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[36]:


# Creating test- Train splits
X_train,X_test,y_train,y_test=train_test_split(X_data_final,y_data_final,test_size=0.4, random_state=42)

# create the claassifier
logreg=LogisticRegression(max_iter=500)

#fit the classifier to the training data
logreg.fit(X_train,y_train)

#predict th lebels of the test set
y_pred=logreg.predict(X_test)

# compute and print the confision martis and the classification report
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# ######  So in the first attempt the precision of the model is 0.81 for 0 and 0.93 for 1.  For 0 the precision is 0.81 and for 1 the precision is 0.93. But the f1 score of 0 is poor only 0.54%. That means it is not clasifying the 0s correctly. With default number of iterations (100 iterations) the algo was failing to converge. When I increased the iterations to 500, it got converged.

# In[38]:


logreg.score(X_test,y_test)


# In[39]:


# ROC Curve -----

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# ######  The score is good because it is classifying most of the yes correctly. But actually the model performance is not so good. Let us see if we can discard any non significant variables:
# 

# In[40]:


import statsmodels.api as sm
logit_model=sm.Logit(y_train,X_train)
result=logit_model.fit(maxiter=100)
print(result.summary2())


# ###### The statsmodels classifiar is failing to converge even though I have increased the number of iterations. So, I cannot find which variables are significant and which are not.

# #### Decision tree classifiar

# In[42]:


# Decision tree
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)

predict = clf.predict(X_test)
print(classification_report(y_test,predict))
print("confusion matrix")
print(confusion_matrix(y_test,predict))


# ####  Naive Bisen Classifiar

# In[43]:


from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
Naive_Gender = GaussianNB()
# multiple variables
multiNaive = MultinomialNB()
Naive_Gender.fit(X_train, y_train)


# In[44]:


Npredict = Naive_Gender.predict(X_test)
# printing predictions
print(classification_report(y_test,Npredict))
print("confuison matrix")
print(confusion_matrix(y_test,Npredict))


# #### KNN model classifiar

# In[45]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)


# In[46]:


pred = knn.predict(X_test)
# printing predictions
print(classification_report(y_test,pred))
print("confuison matrix")
print(confusion_matrix(y_test,pred))


# ######  So, For all the classifiars, the Logistic regression model classifiar is giving the best results as the precision values of both 0s and 1s are the highest of all the models.

# In[ ]:





# In[ ]:




