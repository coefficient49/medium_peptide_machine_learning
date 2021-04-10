## for data wrangling
from util_tools import *
from model_setup_tools import *

import pandas as pd
import numpy as np

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


## copy pasted here to run after prototyping in the jupyter notebook.
## WIP with


## examining anothe data set, which is not used. Will test to see if the data is able to be used with current model in the future.
if False:
    
    # Quick plot to visualize the distribution of the data value
    # there is a pretty large skew, like with most biological data, so we are gping to take the log transform
    # to make mental math easier, I am going to use log10 instead of log
    Counter(pd.read_csv("curated_training_data.affinity.csv").allele)
    df = pd.read_csv("curated_training_data.affinity.csv").query("allele == 'HLA-A*02:01'")
    seqs = df.peptide
    meas = df.measurement_value
    sns.distplot(meas.apply(np.log10).replace(-np.inf,0)) ## setting the flow of the data as 0 (log of 1), there is no fractional value, but there are a few '0' values which would return -inf
    seqspep = seqs[[x==9 for x in map(len,seqs)]] ## ensuring we are only getting the 9mers for this model
    meas = meas[[x==9 for x in map(len,seqs)]] ## extracting the affinity data for the 9mer peptides.



## retrieve the 9mer data
training,testing=get_HLA_A_02_01()



## set up the different encoders from the utility script
onehot = encoder("onehot")
blosum = encoder("blosum62")
reducedProp =encoder("reducedProp")
embedding =encoder("embedding")



## setting up the training and testing data.
Peptide_9mers = training.query("peptide_length == 9")
seqs = Peptide_9mers.loc[:,"sequence"]
off_set = Peptide_9mers["inequality"].map({"=":0,">":2,"<":4})
meas = Peptide_9mers.loc[:,"meas"].apply(np.log10)
## from anothe idea to use winsorize the data between 0 and 1 using quantiles...
## then "hack" the system to pass in different equalities using different y value ranges
## didn't appear to work better than without the winsorization... maybe a larger data set would work better? More thinking required.

# meas = from_ic50(Peptide_9mers.loc[:,"meas"]) +off_set 
# 
# seqs = seqspep
# Peptide_9mers

blosum_seqs = [blosum.encode(x) for x in seqs]
onehot_seqs = [onehot.encode(x) for x in seqs]
reducedProp_seqs = [reducedProp.encode(x) for x in seqs]
embedding_seqs = [embedding.encode(x) for x in seqs]

data_dict = {"blosum62"       : blosum_seqs,
             "onehot"         : onehot_seqs,
             "reducedProp"    : reducedProp_seqs,
             "embedding_seqs" : embedding_seqs,
             "meas"           : meas}

print(data_dict)




epochs = 5000
batch_size = 8
patience = 100
optimizer = "nadam"

seeding_for_reproducibility()
history,model,X_test,y_test,experiment_ID = train_test_model(model_type = "embedding",
                                               conv=False,
                                               rnn = True,
                                               epochs = epochs, 
                                               batch_size = batch_size, 
                                               patience=patience, 
                                               optimizer= optimizer,
                                               data_dict=data_dict)

plot_model(history,model,X_test,y_test,experiment_ID)