#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.simplefilter(action='ignore')
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
from matplotlib.colors import ListedColormap


# In[2]:


data=pd.read_csv("Churn_Modelling.csv")


# In[3]:


data.head()


# In[4]:


data=data.drop(["RowNumber", "CustomerId", "Surname"], axis = 1)


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.describe()


# In[8]:


data.isnull().sum()


# In[9]:


data.duplicated().sum()


# # now we do visualization to know which feature most affect on predict

# In[10]:


sns.countplot(x='Exited', data=data)


# # so the data is imbalanced ,will solve this later!

# lets divide features

# In[11]:


continuous = ['Age', 'CreditScore', 'Balance', 'EstimatedSalary']
categorical = ['Geography', 'Gender', 'Tenure', 'NumOfProducts', 'HasCrCard', 'IsActiveMember']

print('Continuous: ', ', '.join(continuous))
print('Categorical: ', ', '.join(categorical))


# In[12]:


data[continuous].hist(figsize=(12, 10),
                          bins=20,
                          layout=(2, 2),
                          color='steelblue',
                          edgecolor='firebrick',
                          linewidth=1.5);


# In[13]:


#lats see corelatuion
fig, ax = plt.subplots(figsize=(7, 6))

sns.heatmap(data[continuous].corr(),
            annot=True,
            annot_kws={'fontsize': 16},
            cmap='Blues',
            ax=ax)

ax.tick_params(axis='x', rotation=45)
ax.tick_params(axis='y', rotation=360);


# # There is no significant intercorrelation between our features, so we do not have to worry about multicollinearity.
# 
# Let's look at these features in greater detail.

# In[14]:


sns.boxplot(x ="Exited", y = 'Age', data = data)


# ### here is a clear difference between age groups since older customers are more likely to churn.
# 

# In[15]:


sns.boxplot(x ="Exited", y = 'CreditScore', data = data)


# ### There is no significant difference between retained and churned customers in terms of their credit scores.
# 

# In[16]:


sns.boxplot(x ="Exited", y = 'Balance', data = data)


# ### the two distributions are quite similar. There is a big percentage of non-churned customers with a low account balance.

# In[17]:


sns.boxplot(x ="Exited", y = 'EstimatedSalary', data = data)


# ### the two distributions are quite similar

# # Categorical Variables

# In[18]:


sns.countplot(x = 'Geography',hue="Exited", data = data)


# In[19]:


sns.countplot(x = 'Gender',hue="Exited", data = data)


# ### Female customers are more likely to churn.
# 
# 

# In[20]:


sns.countplot(x = 'Tenure',hue="Exited", data = data)


# ### The number of years (tenure) does not seem to affect the churn rate.
# 
# 

# In[21]:


sns.countplot(x = 'NumOfProducts',hue="Exited", data = data)


# In[22]:


sns.countplot(x = 'HasCrCard',hue="Exited", data = data)


# In[23]:


sns.countplot(x = 'IsActiveMember',hue="Exited", data = data)


# ### now we can drop coulmns that no affect

# In[24]:


data.drop(['Tenure', 'HasCrCard', 'EstimatedSalary'],axis=1,inplace=True)
data.head()


# ### deal with Categorical Features!

# In[25]:


data['Gender'] = LabelEncoder().fit_transform(data['Gender'])

data['Geography'] = data['Geography'].map({
    'Germany': 1,
    'Spain': 0,
    'France': 0
})


# In[26]:


data.head(15)


# ### deal with imbalanced data!

# In[27]:


get_ipython().system('pip install imbalanced-learn ')
from imblearn.under_sampling import RandomUnderSampler 

undersample = RandomUnderSampler(sampling_strategy=0.5)


# In[28]:


x=data.drop("Exited",axis=1)
y=data["Exited"]


# In[29]:


from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_oversampled, y_oversampled = smote.fit_resample(x, y)


# In[30]:


test = pd.DataFrame(y_oversampled, columns = ['Exited'])


# In[31]:


sns.countplot(x="Exited",data=test)


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
classifiers = [MultinomialNB(), 
               RandomForestClassifier(),
               KNeighborsClassifier(), 
               SVC(),
              LogisticRegression(),
              XGBClassifier()]
for cls in classifiers:
    cls.fit(X_train, y_train)

# Dictionary of pipelines and model types for ease of reference
pipe_dict = {0: "NaiveBayes", 1: "RandomForest", 2: "KNeighbours",3: "SVC",4:"LogisticRegression",5:"xgboost"}


# In[34]:


for i, model in enumerate(classifiers):
    cv_score = cross_val_score(model, X_train,y_train,scoring="accuracy", cv=10)
    print("%s: %f " % (pipe_dict[i], cv_score.mean()))


# In[35]:


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
Results = pd.DataFrame(data, index =["NaiveBayes", "RandomForest", "KNeighbours","SVC","LogisticRegression","xgboost"])
cmap2 = ListedColormap(["#F5F5DC","#808080"])
Results.style.background_gradient(cmap=cmap2)


# In[36]:


cmap = ListedColormap(["#F5F5DC", "#808080"])
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(15,10))

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


# In[ ]:




