#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, plot_confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
get_ipython().system('pip install xgboost')
from xgboost import XGBClassifier 
import xgboost as xgb
from matplotlib.colors import ListedColormap


# In[17]:


#load data
data_test=pd.read_csv("fraudTest.csv")
data_train=pd.read_csv("fraudTrain.csv")


# In[18]:


data_train.head()


# In[19]:


data_train.shape


# In[20]:


data_train.info()


# In[21]:


data_train.describe()


# In[22]:


data_train.columns


# In[23]:


sns.countplot(x='is_fraud', data=data_train)


# ### so the data is imbalanced ,will deal with this later!

# In[24]:


#Drop Columns that are not relevant to predicy fraud transaction
drop_columns = ['Unnamed: 0','cc_num','merchant','trans_num','unix_time','first','last','street','zip']
data_train.drop(columns=drop_columns,inplace=True)
data_test.drop(columns=drop_columns,inplace=True)


# In[25]:


#handle date time
data_train['trans_date_trans_time']=pd.to_datetime(data_train['trans_date_trans_time'])
data_train['trans_date']=data_train['trans_date_trans_time'].dt.strftime('%Y-%m-%d')
data_train['trans_date']=pd.to_datetime(data_train['trans_date'])
data_train['dob']=pd.to_datetime(data_train['dob'])

data_test['trans_date_trans_time']=pd.to_datetime(data_test['trans_date_trans_time'])
data_test['trans_date']=data_test['trans_date_trans_time'].dt.strftime('%Y-%m-%d')
data_test['trans_date']=pd.to_datetime(data_test['trans_date'])
data_test['dob']=pd.to_datetime(data_test['dob'])


# In[26]:


data_train.info()


# In[27]:


data_train['trans_month'] = pd.DatetimeIndex(data_train['trans_date']).month
data_train['trans_year'] = pd.DatetimeIndex(data_train['trans_date']).year


# In[28]:


data_train.head()


# In[29]:


#Calculate Age
data_train["age"] = data_train["trans_date"]-data_train["dob"]
data_train["age"]=data_train["age"].astype('timedelta64[Y]')

data_test["age"] = data_test["trans_date"]-data_test["dob"]
data_test["age"]=data_test["age"].astype('timedelta64[Y]')


# In[30]:


#Calculate distance between merchant and home location
data_train['latitudinal_distance'] = abs(round(data_train['merch_lat']-data_train['lat'],3))
data_train['longitudinal_distance'] = abs(round(data_train['merch_long']-data_train['long'],3))

data_test['latitudinal_distance'] = abs(round(data_test['merch_lat']-data_test['lat'],3))
data_test['longitudinal_distance'] = abs(round(data_test['merch_long']-data_test['long'],3))


# In[31]:


#Drop Columns that are not relevant to predicy fraud transaction
drop_columns = ['trans_date_trans_time','city','lat','long','dob','merch_lat','merch_long','trans_date','state']
data_train.drop(columns=drop_columns,inplace=True)
data_test.drop(columns=drop_columns,inplace=True)


# In[32]:


data_train.head(),data_test.head()


# ### deal with Categorical Features!

# In[33]:


encoder = LabelEncoder()
data_train["category"] = encoder.fit_transform(data_train["category"])
data_train["gender"] = encoder.fit_transform(data_train["gender"])
data_train["job"] = encoder.fit_transform(data_train["job"])


# In[34]:


get_ipython().system('pip install imbalanced-learn ')
from imblearn.under_sampling import RandomUnderSampler 

undersample = RandomUnderSampler(sampling_strategy=0.5)


# In[35]:


x=data_train.drop("is_fraud",axis=1)
y=data_train["is_fraud"]


# In[36]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_oversampled, y_oversampled = smote.fit_resample(x, y)


# In[37]:


test = pd.DataFrame(y_oversampled, columns = ['is_fraud'])


# In[38]:


sns.countplot(x="is_fraud",data=test)


# ### now the data is balanced!

# In[24]:


#models traning
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
classifiers = [ 
               RandomForestClassifier(),
               KNeighborsClassifier(), 
              LogisticRegression(),
              XGBClassifier()]
for cls in classifiers:
    cls.fit(X_train, y_train)

# Dictionary of pipelines and model types for ease of reference
pipe_dict = { 0: "RandomForest", 1: "KNeighbours",2:"LogisticRegression",3:"xgboost"}


# In[25]:


for i, model in enumerate(classifiers):
    cv_score = cross_val_score(model, X_train,y_train,scoring="accuracy", cv=10)
    print("%s: %f " % (pipe_dict[i], cv_score.mean()))


# In[26]:


from sklearn import tree, metrics
dtree=tree.DecisionTreeClassifier(criterion='gini')
dtree.fit(X_train,y_train)


# In[27]:


y_pred = dtree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Decision Tree Model:")
print("Accuracy:", accuracy)


# In[28]:


# Model Evaluation
# creating lists of varios scores
precision =[]
recall =[]
f1_score = []
trainset_accuracy = []
testset_accuracy = []

for i in classifiers:
    pred_train = i.predict(X_train)
    pred_test = i.predict(X_test)
    prec = metrics.precision_score(y_test, pred_test)
    recal = metrics.recall_score(y_test, pred_test)
    f1_s = metrics.f1_score(y_test, pred_test)
    train_accuracy = model.score(X_train,y_train)
    test_accuracy = model.score(X_test,y_test)
  
    #Appending scores
    precision.append(prec)
    recall.append(recal)
    f1_score.append(f1_s)
    trainset_accuracy.append(train_accuracy)
    testset_accuracy.append(test_accuracy)
# initialise data of lists.
data = {'Precision':precision,
'Recall':recall,
'F1score':f1_score,
'Accuracy on Testset':testset_accuracy,
'Accuracy on Trainset':trainset_accuracy}
# Creates pandas DataFrame.
Results = pd.DataFrame(data, index =["RandomForest", "KNeighbours","LogisticRegression","xgboost"])
cmap2 = ListedColormap(["#F5F5DC","#808080"])
Results.style.background_gradient(cmap=cmap2)


# In[29]:


cmap = ListedColormap(["#F5F5DC", "#808080"])
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15,10))

for cls, ax in zip(classifiers, axes.flatten()):
    plot_confusion_matrix(cls, 
                          X_test, 
                          y_test, 
                          ax=ax, 
                          cmap= cmap,   
                          )
    ax.title.set_text(type(cls).__name__)
plt.tight_layout()  
plt.show()


# ### test the model

# In[54]:


encoder = LabelEncoder()
data_test["category"] = encoder.fit_transform(data_test["category"])
data_test["gender"] = encoder.fit_transform(data_test["gender"])
data_test["job"] = encoder.fit_transform(data_test["job"])


# In[55]:


#deal with test data
x=data_test.drop("is_fraud",axis=1)
y=data_test["is_fraud"]


# In[56]:


#xgb model because it give me a good accuracy in train_data 
model=xgb.XGBClassifier()
model.fit(x,y)


# In[57]:


y_hat=model.predict(x)


# In[58]:


accuracy = accuracy_score(y, y_hat)
print("XGBOOT model:")
print("acurracy",accuracy)


# In[64]:


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y,y_hat)


# In[65]:


cm_df = pd.DataFrame(cm, index=['Actual 0', 'Actual 1'], columns=['Predicted 0', 'Predicted 1'])
print(cm_df)


# In[ ]:




