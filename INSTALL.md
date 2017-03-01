# Step 1: Install the dependencies

Make sure you are using Python 3.4. Now ensure that you have the following packages installed using `pip` or `pip3`:

1. Numpy 1.12.0
2. Matplotlib 1.4.3
3. Pandas 0.16.2
5. nltk 3.0.3
4. Scikit-learn 0.16.1
5. Theano 0.8.2/Tensorflow 1.0.0
6. Keras 1.2.2
7. seaborn


# Step 2: Download the data

Go to [Stanford AI / Sentiment](http://ai.stanford.edu/~amaas/data/sentiment/) to download the data. Unzip the data into the `./data` directory. Next download [GloVe](http://nlp.stanford.edu/data/glove.6B.zip) embeddings (Warning: direct link) and unzip it into the `./data/` directory. Alternative instructions can be found in `./data/README.md` (using command line)

# Step 3: Preprocess the data

`cd` into `./data` and run the following:

1. Create TFIDF representations of the data:
```python feature_extract.py```

2. Create word embedding representations of the data:
```python create_emeddings.py```


# Step 4: Jump into the *deep* end!

The parameters for each model can be found in `./params/`. This will define the network architecture you want to use. All models can be launched from `./run`. Here is the help information:

```
usage: run [-h] -e | [-o |] -n | -a | -p | [-t |] [-lr |] [-ut |] [-v |]
           [-es |] [-nmi |] [-nmx |] [-et |] [-ss |]

optional arguments:
  -h, --help            show this help message and exit

named arguments:
  -e |, --embedding |   Representation of the data one of: 'glove', 'learned',
                        'tfidf', 'counts'
  -o |, --outfile |     File to store trained model
  -n |, --epochs |      Number of epochs to train network
  -a |, --architecture |
                        Network Architecture to use (FNN, CNN, LSTM)
  -p |, --params |      Parameter file to use to build the architecture
  -t |, --optimizer |   Optimizer to use
  -lr |, --learningrate |
                        Optimizer to use
  -ut |, --use-test |   Use the test set to evaluate the model

embedding arguments:
  -v |, --vocabsize |   Number of words in vocabulary
  -es |, --embeddingsize |
                        Size of embedding dimension
  -nmi |, --ngram-min |
                        Minmum n-gram size
  -nmx |, --ngram-max |
                        Maximum ngram size
  -et |, --embedding-trainable |
                        Defines if the embedding is trainable
  -ss |, --subset |     Uses a subset of data to do training
```

Most of these arguments are optional. Some arguments have certain accepted ranges.

These are the arguments you MUST specify: `-e -n -a -p`:

1. `-a` the architecture, currently supported: `[FNN, CNN, LSTM, biLSTM, CNNLSTM]`. Note that if you want to run a CNNbiLSTM, you must do this using the `params.json` file.
2. `-e` the embedding to use. It must be one of `[tfidf, glove, learned]`. Currently FNN can only use `tfidf`. Glove and Learned Embeddings are available. If you want to make them fixed (not trainable), use `-et false`
3. `-n` the number of epochs to run the model.
4. `-p` a `json` file that specifies how to set up the deep learning network. See the folder `./params` for examples that correspond to each architecture.

By default, loss and accuracy curves as well as model are saved at the end of each run. To use all training data and use the `testing` set to obtain these estimates and to save this into a `./results.csv` use `--use-test true`. This must be used 

# FNN

## FNN - tfidf Embeddings
```
./run -n 10 -e glove -a FNN -o './FNNglove' -p './params/FNN-vanilla.json' -v 5000 -es 50 --use-test true
```

## CNN

### CNN - Glove Embeddings
```
./run -n 10 -e glove -a CNN -o './CNNglove' -p './params/CNN-vanilla.json' -v 5000 -es 50 --use-test true
```

### CNN - Learned Embeddings
```
./run -n 10 -e learned -a CNN -o './CNNlearned' -p './params/CNN-vanilla.json' -v 5000 -es 50 --use-test true
```

## LSTM

### LSTM - Glove Embeddings
```
./run -n 10 -e glove -a LSTM -o './LSTMglove' -p './params/LSTM-vanilla.json' -v 5000 -es 50 --use-test true
```

### LSTM - Learned Embeddings
```
./run -n 10 -e glove -a LSTM -o './LSTMlearned' -p './params/LSTM-vanilla.json' -v 5000 -es 50 --use-test true
```

## BiLSTMs
Note that the params file for a BiLSTM is the same as a LSTM.

### BiLSTM - Glove Embeddings
```
./run -n 10 -e glove -a biLSTM -o './biLSTMglove' -p './params/LSTM-vanilla.json' -v 5000 -es 50 --use-test true
```

### BiLSTM - Learned Embeddings

```
./run -n 10 -e learned -a biLSTM -o './biLSTMlearned' -p './params/LSTM-vanilla.json' -v 5000 -es 50 --use-test true
```

## CNN-LSTM

### CNN-LSTM Glove Embeddings

```
./run -n 10 -e glove -a LSTMCNN -o './CNNLSTM' -p './params/LSTMCNN-vanilla.json' -v 5000 -es 50 --use-test true
```

### CNN-LSTM Learned Embeddings

```
./run -n 10 -e learned -a LSTMCNN -o './CNNLSTM' -p './params/LSTMCNN-vanilla.json' -v 5000 -es 50 --use-test true
```

## CNN-BiLSTM
Note that there is no special `biLSTMCNN` option, it must be specified in the params file.

### CNN-BiLSTM Glove Embeddings

```
./run -n 10 -e glove -a LSTMCNN -o './CNNbiLSTM' -p './params/biLSTMCNN-vanilla.json' -v 5000 -es 50 --use-test true
```

### CNN-BiLSM Learned Embeddings

```
./run -n 10 -e learned -a LSTMCNN -o './CNNbiLSTM' -p './params/biLSTMCNN-vanilla.json' -v 5000 -es 50 --use-test true


## Logisitic Regression
To obtain results for the logistic regression model with tfidf model run:

```
python best_LR.py
```