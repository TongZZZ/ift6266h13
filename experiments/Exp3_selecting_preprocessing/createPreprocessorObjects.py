import csv
import numpy as np
import cPickle

from pylearn2.datasets.dense_design_matrix import DefaultViewConverter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import pylearn2.datasets.preprocessing
from contest_dataset import ContestDataset


def fitAndPickle(preprocessor, filename):
    dataset = ContestDataset(which_set='train')
    preprocessor.apply(dataset, can_fit = True)
    
    cPickle.dump(preprocessor, open(filename,"wb"), 2)
    

# Create bunch of processor objects fitted on the whole train+valid dataset and pickle them

# Create a ZCA preprocessor
fitAndPickle(preprocessor = pylearn2.datasets.preprocessing.ZCA(),
             filename = "./ZCA_preprocess.pkl")

# Create a preprocessor performing GCN
fitAndPickle(preprocessor = pylearn2.datasets.preprocessing.GlobalContrastNormalization(subtract_mean=False),
             filename = "./GCN_preprocess.pkl")

# Create a preprocessor performing DC Centering followed by GCN
fitAndPickle(preprocessor = pylearn2.datasets.preprocessing.GlobalContrastNormalization(subtract_mean=True),
             filename = "./DC_GCN_preprocess.pkl") 
             
# Create a preprocessor performing DC Centering, GCN and then ZCA
pipeline = pylearn2.datasets.preprocessing.Pipeline()
pipeline.items = [pylearn2.datasets.preprocessing.GlobalContrastNormalization(subtract_mean=True),
                  pylearn2.datasets.preprocessing.ZCA()]
fitAndPickle(preprocessor = pipeline, filename = "./DC_GCN_ZCA_preprocess.pkl")
 
# Create a preprocessor performing DC Centering, GCN and then standardization
pipeline = pylearn2.datasets.preprocessing.Pipeline()
pipeline.items = [pylearn2.datasets.preprocessing.GlobalContrastNormalization(subtract_mean=True),
                  pylearn2.datasets.preprocessing.Standardize()]
fitAndPickle(preprocessor = pipeline, filename = "./DC_GCN_STD_preprocess.pkl") 
 
# Create a preprocessor performing pixel standardization (for each pixel,
# remove the mean across the dataset for this pixel and divice by the
# standard deviation across the dataset for this pixel.
fitAndPickle(preprocessor = pylearn2.datasets.preprocessing.Standardize(),
             filename = "./STD_preprocess.pkl") 
             

                         

             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             
             





