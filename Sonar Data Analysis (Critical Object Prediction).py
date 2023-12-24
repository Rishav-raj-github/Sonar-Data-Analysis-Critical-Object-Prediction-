#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Aanlysing the sonat data to differentiate between Rock and mines
# This process include (Logistic Regression approach)
# IT include supervised Learning moduel

# Logistic regression include that the 'Best fit line should be crossing between the data'


# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[3]:


Sonar_Data = pd.read_csv('Copy of sonar data.csv', header = None)


# In[4]:


Sonar_Data.head


# In[5]:


Sonar_Data.describe()


# In[6]:


Sonar_Data.shape


# In[7]:


Sonar_Data[60].value_counts()


# In[8]:


## Grouping the data based on (M) and (R) --> M Stand for mine, R for Rocks


# In[9]:


Sonar_Data.groupby(60).mean()


# In[10]:


# Supervised Machine Learning model consist of Labels
# Unsupervised Machine Learning Does not consist of Label's


# In[11]:


X = Sonar_Data.drop(columns = 60, axis = 1)
Y = Sonar_Data[60]


# In[12]:


# Splitting of the Training and Test Data


# In[13]:


X_train, X_test, Y_train, Y_test = train_test_split(X , Y, test_size= 0.1, stratify= Y , random_state=1)


# In[14]:


Y_train.shape


# In[15]:


# Training our Model using the Logistic regression supervised Learning Method


# In[16]:


model = LogisticRegression()


# In[17]:


# Now training the model with Trainee Data


# In[18]:


model.fit(X_train,Y_train)


# In[19]:


# Checking the accuracy of our Model --> Model Evaluation


# In[20]:


# Accuracy on Training Data


# In[21]:


X_train_prediction = model.predict(X_train)


# In[22]:


training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[23]:


print('Accuracy on training data : ', training_data_accuracy)


# In[24]:


# Accuracy test on test Data


# In[25]:


X_test_prediction = model.predict(X_test)


# In[26]:


test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[27]:


print('Accuracy on test data : ' , test_data_accuracy)


# In[28]:


# MArking the predectiv system


# In[29]:


input_Sonar_data = (0.0124,0.0433,0.0604,0.0449,0.0597,0.0355,0.0531,0.0343,0.1052,0.2120,0.1640,0.1901,0.3026,0.2019,0.0592,0.2390,0.3657,0.3809,0.5929,0.6299,0.5801,0.4574,0.4449,0.3691,0.6446,0.8940,0.8978,0.4980,0.3333,0.2350,0.1553,0.3666,0.4340,0.3082,0.3024,0.4109,0.5501,0.4129,0.5499,0.5018,0.3132,0.2802,0.2351,0.2298,0.1155,0.0724,0.0621,0.0318,0.0450,0.0167,0.0078,0.0083,0.0057,0.0174,0.0188,0.0054,0.0114,0.0196,0.0147,0.0062)


# In[30]:


#Changing the Data Type to Numpy Array

input_data_as_numpy_array = np.asarray(input_Sonar_data)


# In[31]:


# Reshapinng the numpy array as we are predicting for one instance


# In[32]:


input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)


# In[33]:


prediction = model.predict(input_data_reshaped)


# In[34]:


print(prediction)


# In[35]:


if (prediction[0] == 'R'):
    print ('The object is Rock')
    
else: 
    print('The Object is Mine')


# In[ ]:





# In[ ]:




