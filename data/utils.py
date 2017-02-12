"""
    Contains utilities for data management
"""

from glob import glob
import pandas as pd

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
            

