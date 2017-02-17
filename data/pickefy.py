from utils import load_data, create_folder
import os

create_folder('./processed')

for dataset in ['train', 'test']:
	for sentiment in ['both', 'pos', 'neg']:
		print('Saving data for sentiment=' + sentiment + ' in dataset=' + dataset )
		load_data(dataset=dataset, sentiments=sentiment).to_pickle('./processed/pd.DF.' + dataset + '_' + sentiment + '.pickle')

print('Saved all data to pickle format.')

