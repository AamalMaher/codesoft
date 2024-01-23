#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, plot_confusion_matrix, classification_report, accuracy_score, f1_score
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap


# In[2]:


data=pd.read_csv("spam.csv",encoding_errors= 'replace')
data.head()
    


# # Exploring and cleaning the data set

# In[3]:


data.columns


# In[4]:


data=data.drop("Unnamed: 2",axis=1)


# In[5]:


data=data.drop("Unnamed: 3",axis=1)


# In[6]:


data=data.drop("Unnamed: 4",axis=1)


# In[7]:


data.head()


# In[8]:


data.info()


# In[9]:


data.describe(include="object")


# In[10]:


data.columns=["type","message"]


# In[11]:


data.head()


# In[12]:


data.isnull().sum()


# In[13]:


data["message"].unique()


# In[14]:


data["message"].value_counts(normalize=1)


# # data visualization

# In[15]:


sns.countplot(x="type",data=data)


# ## so tha data is imbalanced

# # DATA PREPREPROCESSING
# 1.cleaning text
# 2.tokanization
# 3.removing stop words
# 4.lemmatization
# 

# In[16]:


def Clean(message):
    sms = re.sub('[^a-zA-Z]', ' ', message) #Replacing all non-alphabetic characters with a space
    sms = sms.lower() #converting to lowercase
    sms = sms.split()
    sms = ' '.join(sms)
    return sms

data["Clean_message"] = data["message"].apply(Clean)


# In[17]:


data.head()


# In[18]:


# breaking complex data into smaller units called token

data["Tokenize_message"]=data.apply(lambda row: nltk.word_tokenize(row["Clean_message"]), axis=1)


# In[19]:


#removing the stop words
def remove_stopwords(message):
    stop_words = set(stopwords.words("english"))
    filtered_message= [word for word in message if word not in stop_words]
    return filtered_message

data["Nostopword_message"] = data["Tokenize_message"].apply(remove_stopwords)


# In[20]:


data.head()


# In[21]:


lemmatizer = WordNetLemmatizer()
# lemmatize string
def lemmatize_word(message):
    # provide context i.e. part-of-speech
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in message]
    return lemmas
data["Lemmatized_message"] = data["Nostopword_message"].apply(lemmatize_word)


# In[22]:


data.head()


# In[23]:


corpus= []
for i in data["Lemmatized_message"]:
    msg = ' '.join([row for row in i])
    corpus.append(msg)
    
corpus[:5]


# In[24]:


label_encoder = LabelEncoder()
data["type"] = label_encoder.fit_transform(data["type"])


# In[25]:


#Changing text data in to numbers. 
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus).toarray()
#Let's have a look at our feature 
X.dtype


# In[26]:


y = data["type"] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifiers = MultinomialNB()
classifiers.fit(X_train, y_train)
y_pred = classifiers.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
metrics.confusion_matrix(y_test, y_pred)


# In[27]:


classifiers = [MultinomialNB(), 
               RandomForestClassifier(),
               KNeighborsClassifier(), 
               SVC()]
for cls in classifiers:
    cls.fit(X_train, y_train)

# Dictionary of pipelines and model types for ease of reference
pipe_dict = {0: "NaiveBayes", 1: "RandomForest", 2: "KNeighbours",3: "SVC"}


# In[28]:


for i, model in enumerate(classifiers):
    cv_score = cross_val_score(model, X_train,y_train,scoring="accuracy", cv=10)
    print("%s: %f " % (pipe_dict[i], cv_score.mean()))


# In[29]:


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
Results = pd.DataFrame(data, index =["NaiveBayes", "RandomForest", "KNeighbours","SVC"])
cmap2 = ListedColormap(["#808080","#FFC0CB"])
Results.style.background_gradient(cmap=cmap2)


# In[31]:


cmap = ListedColormap(["#808080", "#FFC0CB"])
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


# In[ ]:




