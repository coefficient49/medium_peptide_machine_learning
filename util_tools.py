###### data sources ######

test_set = "https://raw.githubusercontent.com/pcpLiu/DeepSeqPan/master/dataset/weekly_data_all_rm_duplicate.txt"
train_set = "https://raw.githubusercontent.com/pcpLiu/DeepSeqPan/master/dataset/bdata.20130222.mhci.txt"
protein_sequence = ""
## referenced paper: https://www.nature.com/articles/s41598-018-37214-1 ##


import pandas as pd
import numpy as np
from glob import glob
from collections import Counter

def get_training_data():
    ## check directory for data set file, if not present, download a new copy.
    if glob("training.csv"):
        print("local training file found... loading...")
        df = pd.read_csv("./training.csv")
    else:
        print("local training file not found... downloading...")
        df = pd.read_csv(train_set,sep="\t")
        df.to_csv("training.csv")
    return df

def get_testing_data():
    ## check directory for file, if not present, download a new copy.
    if glob("testing.csv"):
        print("local testing file found... loading...")
        df = pd.read_csv("testing.csv")
    else:
        print("local testing file not found... downloading...")
        df = pd.read_csv(test_set,sep="\t")
        df.to_csv("testing.csv")
    return df

def get_data():
    ## quick function to load data.
    dfTraining = get_training_data()
    dfTesting = get_testing_data()
    return (dfTraining, dfTesting)

def get_HLA_A_02_01():
    ###### focus on getting HLA-A *02:01 first ######## simplify this example using only 1 MHC class.
    dfTraining = get_training_data()
    dfTraining = dfTraining.query("mhc == 'HLA-A*02:01'")
    dfTesting = get_testing_data()
    dfTesting = dfTesting.query("Allele == 'HLA-A*02:01'")
    return (dfTraining, dfTesting)

class encoder():
    #### class object for encoding peptide sequence into something the machine can understand ####
    """
    input is just how you would like to encode the peptides currently only allowed for :
    how = "onehot" or "blosum62" or "reducedProp" or ""embedding"

    For the future: ML-based learned embeddings like BERTs and UNIREPs

    """
    def __init__(self, how="onehot"):
        self.encoder_method = how
        self.how = how
        
        if how not in ["onehot","blosum62","reducedProp","embedding"]:
            print("Currently, this script only take \"onehot\" or \"blosum62\" or \"reducedProp\" or  \"embedding\" ")
        #### define the translator "vocabulary" to use for encoding simple encoding
        if how == "onehot":
        ##### simple one-hot encoding
            self.alphabet = "ACDEFGHIKLMNPQRSTVWY"
            self._translator = {x:xi for xi,x in enumerate(self.alphabet)}
        elif how == "blosum62":
        ##### using blosum62 similarity matrix
            with open("blosum62.txt","r") as handle:
                for xi,x in enumerate(handle):
                    if xi == 0:
                        alphabet = {}
                    else:
                        vector = x.split()
                        alphabet[vector[0]] = [int(x) for x in vector[1::]]
            self._translator = alphabet
            self.alphabet = vector
        elif how == "reducedProp":
            df = pd.read_csv("aa_properties.csv",index_col=0)
            index = df.index
            alphabet = {x:list(df.loc[x,:]) for x in index}
            self._translator = alphabet
            # self.alphabet = vector
        elif how == "embedding":
            ## th is is useful for RNNs
            alphabet = 'ACDEFGHIKLMNPQRSTVWY' 
            self._translator = dict(zip(alphabet, range(len(alphabet))))

        elif how == "bert":
            ## WIP
            None
        elif how == "unirep":
            ## WIP
            None
            
        
    def onehot(self,sequenceIn):            
        mat = np.zeros([20,len(sequenceIn)])
        for xi,x in enumerate(sequenceIn):
            rowNum = self._translator[x]
            mat[rowNum,xi]=1
        return mat
    
    def blosum(self,sequenceIn):
        mat = np.zeros([24,len(sequenceIn)])
        for xi,x in enumerate(sequenceIn):
            
            mat[:,xi]=self._translator[x]
        return mat
    
    def reducedProp(self,sequenceIn):
        mat = np.zeros([6,len(sequenceIn)])
        for xi,x in enumerate(sequenceIn):
            mat[:,xi]=self._translator[x]
        return mat
    
    def embedding(self,sequenceIn):
        mat = np.zeros([1,len(sequenceIn)])
        for xi,x in enumerate(sequenceIn):
            mat[0,xi]=self._translator[x]
        return mat
    
    def BERTembeding(self, sequenceIn):
    	# to be done
        None
    
    def UNIREPembedding(self, sequenceIn):
    	# to be done
        None
    
    def encode(self,sequence):
        if self.how == "onehot":
            return self.onehot(sequence)
        elif self.how == "blosum62":
            return self.blosum(sequence)
        elif self.how == "reducedProp":
            return self.reducedProp(sequence)
        elif self.how == "embedding":
            return self.embedding(sequence)


if __name__ == "__main__":
    #### test the function before moving on to the notebook ###
    blosum = encoder("embedding")
    sequences = ["AGCSTHCTHSTHCY"]    
    [print(blosum.encode(x)) for x in sequences]
# training, testing = get_HLA_A_02_01()

# print(Counter(testing.loc[:,"Measurement type"]))
