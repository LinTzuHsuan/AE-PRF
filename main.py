# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 11:52:13 2021

@author: Tzu-Hsuan Lin
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.utils import plot_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, matthews_corrcoef


#%% Data preprocessing
raw_data = pd.read_csv("./creditcard.csv")

X_train, test = train_test_split(raw_data, train_size=0.8, random_state=0)
X_train, X_test = train_test_split(X_train, train_size=0.8, random_state=0)
X_train.loc[:,"Time"] = X_train["Time"].apply(lambda x : x / 3600 % 24) 
X_train.loc[:,'Amount'] = np.log(X_train['Amount']+1)
X_test.loc[:,"Time"] = X_test["Time"].apply(lambda x : x / 3600 % 24) 
X_test.loc[:,'Amount'] = np.log(X_test['Amount']+1)

test.loc[:,"Time"] = test["Time"].apply(lambda x : x / 3600 % 24) 
test.loc[:,'Amount'] = np.log(test['Amount']+1)

y_train = X_train['Class'].values
X_train = X_train.drop(['Class'], axis=1).values

#%% Autoencoder
activation = 'relu'
input_dim = X_train.shape[1]
encoding_dim = 26
nb_epoch = 3 
batch_size = 64

input_layer = Input(shape=(input_dim, ), name='Input')
encoder = Dense(encoding_dim, activation='tanh', 
                activity_regularizer=regularizers.l1(10e-5), name='encoder1')(input_layer)
encoder = Dense(22, activation=activation, name='encoder2')(encoder)
encoder = Dense(18, activation=activation, name='encoder3')(encoder)
decoder = Dense(22, activation=activation, name='decoder1')(encoder)
decoder = Dense(encoding_dim, activation=activation, name='decoder2')(decoder)
decoder = Dense(input_dim, activation=activation, name='decoder3')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
plot_model(autoencoder, to_file='./summary.png', show_shapes=True)

autoencoder.compile(optimizer='adam', 
                    loss='mean_squared_error', 
                    metrics=['accuracy'])
checkpointer = ModelCheckpoint(filepath="./best_model.h5",
                              verbose=0,
                              save_best_only=True)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0)

history = autoencoder.fit(X_train, X_train,
                    epochs=nb_epoch,
                    batch_size=batch_size,
                    verbose=1,
                    shuffle=True,
                    validation_split=0.1,
                    callbacks=[checkpointer,early_stopping]).history

encoder_all = Model(input_layer,encoder)
encoder_all.save("./encoder.h5")
encoder_all = tf.keras.models.load_model('encoder.h5', custom_objects={'leaky_relu': tf.nn.leaky_relu})
enc_all = encoder_all.predict(X_train)

#%% Random forest
y_test = X_test['Class'].values
X_test = X_test.drop(['Class'], axis=1).values

forest = RandomForestClassifier(criterion='gini', n_estimators=100,random_state=0, oob_score=True)
forest.fit(enc_all,y_train)

test_x = encoder_all.predict(X_test)
predicted_proba = forest.predict_proba(test_x)

threshold=0.25 # You may set another threshold to get the best result

mse = (predicted_proba[:,1] >= threshold).astype('int')

#%% Performance evaluation (Validation data)
print("Below is the result of validation data")
TP=0
FN=0
FP=0
TN=0

for i in range(0, int(y_test.shape[0])):
  if (y_test[i]) :
    if (mse[i] > 0) :
      TP = TP + 1
    else :
      FN = FN + 1
  else:
    if (mse[i] > 0) :
      FP = FP + 1
    else :
      TN = TN + 1
      
# Accuracy
Accuracy = accuracy_score(y_test,mse)

# TPR
TPR = TP/(TP+FN)

# TNR
TNR = TN/(TN+FP)

# MCC
MCC = matthews_corrcoef(y_test,mse)

# Confusion matrix
LABELS = ["Normal", "Fraud"]
y_pred = [1 if e > threshold else 0 for e in mse]
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(12, 12))
akws = {"size": 22, "color":'r'}
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, annot_kws=akws, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
plt.close()

# Results
print('Accuracy:', Accuracy)
print('TPR:', TPR)
print('TNR:', TNR)
print('MCC:', MCC)



#%% To the test data (test data)
yy_test = test['Class'].values
test = test.drop(['Class'], axis=1).values

test_xx = encoder_all.predict(test)
test_predicted_proba = forest.predict_proba(test_xx)

test_mse = (test_predicted_proba[:,1] >= threshold).astype('int')

print("Below is the result of test data")
TP=0
FN=0
FP=0
TN=0

for i in range(0, int(yy_test.shape[0])):
  if (yy_test[i]) :
    if (test_mse[i] > 0) :
      TP = TP + 1
    else :
      FN = FN + 1
  else:
    if (test_mse[i] > 0) :
      FP = FP + 1
    else :
      TN = TN + 1
      
# Accuracy
Accuracy = accuracy_score(yy_test,test_mse)

# TPR
TPR = TP/(TP+FN)

# TNR
TNR = TN/(TN+FP)

# MCC
MCC = matthews_corrcoef(yy_test,test_mse)

# Confusion matrix
LABELS = ["Normal", "Fraud"]
y_pred = [1 if e > threshold else 0 for e in test_mse]
conf_matrix = confusion_matrix(yy_test, y_pred)
plt.figure(figsize=(12, 12))
akws = {"size": 22, "color":'r'}
sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, annot_kws=akws, fmt="d")
plt.title("Confusion matrix")
plt.ylabel('True class')
plt.xlabel('Predicted class')
plt.show()
plt.close()

# Results
print('Accuracy:', Accuracy)
print('TPR:', TPR)
print('TNR:', TNR)
print('MCC:', MCC)