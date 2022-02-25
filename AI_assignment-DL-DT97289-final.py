#!/usr/bin/env python
# coding: utf-8

#   # Name:Saad,Mohammad
# 
#   # Student no :300267006                                                  
#  
#  #  Assignment (1)
# 
#   # ELG7186-AI for CS

# In[1]:


import numpy as np 
import pandas as pd
import csv
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,classification_report,ConfusionMatrixDisplay
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from keras.regularizers import l1,l2
tf.random.set_seed(seed=1500)
np.random.seed(seed=150)


# Reading training data CSV file and removing some columns<br>
# The columns were elinmated based on manual visual inspection to the CSV file and some trials and errors<br>

# In[2]:


df=pd.read_csv("traindata.csv")

y=df['Class']
y=np.array(y)
df_train=df.drop(['ID','land','land','count','dst_host_count','dst_host_srv_count','dst_host_rerror_rate','duration','srv_diff_host_rate','dst_host_srv_rerror_rate','wrong_fragment','urgent','root_shell','su_attempted','num_shells','num_outbound_cmds','is_host_login','Class'], axis=1)


df_test=pd.read_csv("testdata.csv")
IDs=df_test['ID']
df_test=df_test.drop(['ID','land','land','wrong_fragment','count','dst_host_count','duration','dst_host_srv_count','dst_host_srv_rerror_rate','dst_host_rerror_rate','srv_diff_host_rate','urgent','root_shell','su_attempted','num_shells','num_outbound_cmds','is_host_login'], axis=1)


# The cloumns with nominal data was encoded using label encoding:

# In[3]:


le = LabelEncoder()
df_train['protocol_type'] = le.fit_transform(df_train['protocol_type'])
df_train['service'] = le.fit_transform(df_train['service'])
df_train['flag'] = le.fit_transform(df_train['flag'])

df_test['protocol_type'] = le.fit_transform(df_test['protocol_type'])
df_test['service'] = le.fit_transform(df_test['service'])
df_test['flag'] = le.fit_transform(df_test['flag'])


# The proposed solution is based on combining the results from a decision tree model with a deep learning model.

# ### Decision Tree 
# The parameter that have increased the score for the decision tree is max_leaf_nodes equals to 7 

# In[4]:



clf = DecisionTreeClassifier(random_state=0,max_leaf_nodes=7).fit(df_train,y)#96


# ### Deep learning 
#  

# In[5]:



model = keras.models.Sequential([layers.BatchNormalization(input_shape=[27]),
layers.Dense(64, activation="relu"),
layers.BatchNormalization(),                                                                   
layers.Dense(32, activation="relu"),
layers.BatchNormalization(),
layers.Dense(16, activation="relu"),
layers.BatchNormalization(), 
layers.Dense(16, activation="relu"),
layers.BatchNormalization(),                                 
layers.Dense(8, activation="relu"),
layers.BatchNormalization(),                                  
layers.Dense(1, activation="sigmoid")])
optimizer = keras.optimizers.Adam(epsilon=0.0000001)
model.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["binary_accuracy"])
history = model.fit(df_train,y,batch_size=256, epochs=75)


# The optimum threshhold found is to be more or equal to 0.001

# In[6]:


y_prdict2=clf.predict(df_test)#Decision tree
print(y_prdict2)
y_prdict=model.predict(df_test)#DL
print(y_prdict)
y_prdict1=(model.predict(df_test) >= 0.001).astype("int")#DL
print(y_prdict1)


# ### Model combination
# The 2-models were combined by appling a code that performs on the predictions like an OR gate to give the final results 

# In[7]:


t=y_prdict1.reshape(y_prdict1.shape[0])+y_prdict2
k=[]
for i in range(t.shape[0]):
    if t[i]==2:#if both said attack
        k.append(1)
    elif t[i]==1:#if one of them said attack
        k.append(1)
    else:
        k.append(0)


k=np.array(k)
print(k)


# Writing the preditions to a CSV file to be submitted

# In[8]:


header = ['ID','Class']
with open('Mohammad Saad.csv', 'w', encoding='UTF8', newline='') as f:
    # create the csv writer
    writer = csv.writer(f)

    # write a row to the csv file
    writer.writerow(header)
    
    for w in range(len(IDs)):
        writer.writerow([IDs[w], k[w]])
        

