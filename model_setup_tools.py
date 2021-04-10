## for data wrangling
from util_tools import *
from datetime import datetime

import pandas as pd
import numpy as np
import os

## for plotting
import seaborn as sns
import matplotlib.pyplot as plt

## for reproducibility
from numpy.random import seed

## for machine learning
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr,pearsonr,zscore

import xgboost as xgb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from collections import Counter
from tensorflow.keras import backend as K




def build_model(encoder_type = "blosum62", conv_layers = True, rnn= False):
    ## note: for now the RNN model only works with the embedding encoder.
    if encoder_type == "onehot":
        inputs = Input(shape=(20,9,1))
    elif encoder_type == "blosum62":
        inputs = Input(shape=(24,9,1))
    elif encoder_type == "reducedProp":
        inputs = Input(shape=(6,9,1))
    elif encoder_type == "embedding":
        inputs = Input(shape=(None,))
    
    if conv_layers:
        hidden1 = Conv2D(16,7,activation="elu",padding="same")(inputs)
        hidden2 = Conv2D(16,5,activation="elu",padding="same")(hidden1)
        hidden2  = BatchNormalization()(hidden2)
        hidden3 = Conv2D(8,5,activation="elu",padding="same")(hidden2)
        hidden4 = Conv2D(8,3,activation="elu",padding="same")(hidden3)
        hidden4  = BatchNormalization()(hidden4)
        flatten = Flatten()(hidden4)
    else:
        flatten = Flatten()(inputs)
        
    if rnn:
#         vocabulary_size, embedding_dim, input_length=max_review_length
        embeddinglayer = Embedding(input_dim=20, output_dim=64)(inputs)
        gru1 = Bidirectional(GRU(64,return_sequences=True))(embeddinglayer)
        gru2 = GRU(64)(gru1)
        flatten = Flatten()(gru2)
    
    dense1  = Dense(200,activation="relu")(flatten)
    dense1  = BatchNormalization()(dense1)
    dense1  = Dropout(0.7)(dense1)

    dense2  = Dense(200,activation="relu")(dense1)
    dense2  = BatchNormalization()(dense2)
    dense2  = Dropout(0.8)(dense2)

    dense3  = Dense(100,activation="relu")(dense2)
    dense3  = BatchNormalization()(dense3)
    dense3  = Dropout(0.8)(dense3)
    
    dense4  = Dense(50,activation="relu")(dense3)
    dense4  = BatchNormalization()(dense4)
    dense4  = Dropout(0.8)(dense4)
    
    dense5  = Dense(25,activation="relu")(dense4)
    dense5  = BatchNormalization()(dense5)
    dense5  = Dropout(0.8)(dense5)

    outputs = Dense(1,activation="linear")(dense5)

    # dense3
    model = keras.Model(inputs=inputs, outputs=outputs)
    # model.summary()

    return model

def SPRCC(y_true, y_pred):
    ## thi 

    ## was going to use the winsorization, but commented out for now, more thoughts needed

    ##silencing this for now by commenting it out.
     
    # y_true = K.flatten(y_true)
    # y_pred = K.flatten(y_pred)

    # # Handle (=) inequalities
    # diff1 =  y_true
    # diff1 *= K.cast(y_true >= 0.0, "float32")
    # diff1 *= K.cast(y_true <= 1.0, "float32")

    # # Handle (>) inequalities
    # diff2 = y_true - 2.0
    # diff2 *= K.cast(y_true >= 2.0, "float32")
    # diff2 *= K.cast(y_true <= 3.0, "float32")
    # diff2 *= K.cast(diff2 > 0.0, "float32")

    # # Handle (<) inequalities
    # diff3 = y_true - 4.0
    # diff3 *= K.cast(y_true >= 4.0, "float32")
    # diff3 *= K.cast(diff3 > 0.0, "float32")

    # y_true = diff1 + diff2 + diff3
    
    return ( tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32),tf.cast(y_true, tf.float32)], Tout = tf.float32) )

def inequaility_loss(y_true, y_pred):
    ## adapted from MHCflurry
    from tensorflow.keras import backend as K
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    # Handle (=) inequalities
    diff1 = y_pred - y_true
    diff1 *= K.cast(y_true >= 0.0, "float32")
    diff1 *= K.cast(y_true <= 1.0, "float32")

    # Handle (>) inequalities
    diff2 = y_pred - (y_true - 2.0)
    diff2 *= K.cast(y_true >= 2.0, "float32")
    diff2 *= K.cast(y_true <= 3.0, "float32")
    diff2 *= K.cast(diff2 > 0.0, "float32")

    # Handle (<) inequalities
    diff3 = y_pred - (y_true - 4.0)
    diff3 *= K.cast(y_true >= 4.0, "float32")
    diff3 *= K.cast(diff3 > 0.0, "float32")

    denominator = K.maximum(
        K.sum(K.cast(K.not_equal(y_true, 2.0), "float32"), 0),
        1.0)

    result = (
            K.sum(K.square(diff1)) +
            K.sum(K.square(diff2)) +
            K.sum(K.square(diff3))) / denominator

    return result


def seeding_for_reproducibility():
    ## that the name of the function said...
    seed(123454566)
    tensorflow.random.set_seed(123454566)

######### model parameters ##############
epochs = 5000
batch_size = 8
patience = 100
optimizer = "nadam"
########################################

def train_test_model(model_type = "blosum62",conv=True, rnn=False,epochs = 100, batch_size = 16, patience=2, optimizer= "nadam",data_dict=None):

    ## setting up the train val data set for now, not touching the testing data just yet. Going to call "Val data" as "Test data" for now.
    if (X == None) or (Y == None):
        print("training data mission!")
    if model_type == "blosum62":
        ## a little clunky now, will rewrite this part to pass in the correct data with the correct shape.
        X = np.array(data_dict[model_type]).reshape(-1,24,9,1)
    elif model_type == "onehot":
        X = np.array(data_dict[model_type]).reshape(-1,20,9,1)
    elif model_type == "reducedProp":
        X = np.array(data_dict[model_type]).reshape(-1,6,9,1)
    elif model_type == "embedding":
        X = np.array(data_dict[model_type]).reshape(-1,9)
#     Y = Peptide_9mers.loc[:,"meas"].apply(np.log10)
    Y = data_dict["meas"]
    
    model = build_model(model_type,conv, rnn)
    model.compile(
#         loss=inequaility_loss,
        loss = "mse",
        optimizer=optimizer,
        metrics=[SPRCC]
    )
    es = EarlyStopping(monitor='val_loss',mode="min",patience=patience) ## stop early call back
    now = datetime.now()
    experiment_ID = now.strftime("%y%m%d.%Hh%Mm%Ss") ## using date string as exp_ID
    os.makedir(experiment_ID)
    mc = ModelCheckpoint('./experiments/{}/best_{}_conv={}_model.h5'.format(experiment_ID,model_type,conv),monitor='val_loss', mode='min', save_best_only=True) ## sving best model callback
    [X_train, X_test, y_train, y_test]=train_test_split(X,Y,test_size=0.2,random_state=10)
    history = model.fit(X_train,y_train,batch_size=batch_size,epochs = epochs, validation_split=0.1,callbacks=[es,mc])
    return (history,model,X_test,y_test,experiment_ID)

def plot_model(history=False,model=None,X_test=None,y_test=None,experiment_ID=None):
    if history:
        plt.figure()
        sns.lineplot(data=pd.DataFrame(history.history).reset_index(),x="index",y="loss")
        sns.lineplot(data=pd.DataFrame(history.history).reset_index(),x="index",y="val_loss")
        plt.legend(["loss","val_loss"])
    y_pred = model.predict(X_test)
    ## transform y_test for inquality loss, but for now, we silence it
#     bat1 = y_test*(y_test<=1)
#     bat2 = (y_test-2)*(y_test>=2)*(y_test<=3)
#     bat3 = (y_test-4)*(y_test>=4)
#     y_test = bat1+bat2+bat3
    plt.figure()
    sns.scatterplot(x=y_test,y=y_pred.squeeze())
    R, pv = spearmanr(y_test,y_pred.squeeze())
    plt.title("SRCC: {0:0.4f}".format(R))
    plt.savefig("'./experiments/{}/SPRCC.png".format(experiment_ID))
