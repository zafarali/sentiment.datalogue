"""
    Contains utilities for data management
"""

from glob import glob
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from itertools import product
import os

# a oneliner to create a new folder
create_folder = lambda f: [ os.makedirs('./'+f) if not os.path.exists('./'+f) else False ]

def load_data(dataset='train', lim=12500, sentiments='both', folder_location='./aclImdb'):
    """
        Loads the dataset into a pandas dataframe.
        @params:
            dataset[=train]: the dataset to load (must be test or train)
            sentiment
            lim[=12500]: the number of examples to load from the sentiment category
            sentiment[=both]: the sentiment category to load.
        @returns:
            pandas.DataFrame instance with the information requested
    """
    assert dataset in ['train', 'test'], 'dataset must be training or testing'
    assert sentiments in ['both', 'pos', 'neg'], 'Sentiment must be both, positive or negative'
    
    sentiments = ['pos', 'neg'] if sentiments == 'both' else [ sentiments ]
    
    # load all the folders for data parsing
    data_locations = [ folder_location + '/' + dataset + '/' + sentiment + '/*.txt' for sentiment in sentiments ]
    
    data = []
    
    for data_folder in data_locations:
        for filepath in glob(data_folder)[:lim]: # use the lim parameter to load only a few rows
            
            # load the data
            text = next(open(filepath, 'r'))
            sentiment, rating = filepath.split('/')[-2:]
            is_positive = sentiment == 'pos'
            record_id, rating = rating.split('_')
            rating = rating.split('.txt')[0]
            record_id, rating = int(record_id), int(rating)
            
            data.append([record_id, text, filepath, rating, is_positive])
    
    if len(data) == 0:
        return IOError('No data was found')

    return pd.DataFrame(data, columns=['record_id', 'text', 'filepath', 'rating', 'is_positive'])

# code from:
# http://stackoverflow.com/questions/8955448/save-load-scipy-sparse-csr-matrix-in-portable-data-format
def save_sparse_csr(filename, array):
    """
        Save a CSR
        @params:
            filename
            array
    """
    np.savez( filename , data = array.data , indices=array.indices, 
        indptr = array.indptr , shape = array.shape )

def load_sparse_csr(filename):
    """
        Load a CSR
            @params:   
                filename
    """ 
    loader = np.load(filename)
    return csr_matrix( ( loader['data'] , loader['indices'] , loader['indptr'] ) ,
        shape = loader['shape'])


def generate_folder(root_folder, embeddings=False, folder_titles=True):
    """
        Yields the folders that represent all the precomputed features 
    """
    if not embeddings:
        vectorizers = [ 'tfidf' , 'hashing' ]
        n_features = [ 5000 , 10000 ]
        n_grams = [ (1,1) , (1,2) , (2,2) ]

        for v, f, n in product(vectorizers, n_features, n_grams):
            yield root_folder + '/processed/'+v+'_'+str(f)+'_'+str(n[0])+'-'+str(n[1])+'grams/', v + 'n='+str(f) + 'g='+str(n)+')'
