# Sentiment Analysis

Motivation: This Sentiment analysis challenge was for Datalogue recruiting

Goal: Write a neural network that can classify sentiments using a corpus in `./data/`

HOWTO: Details on how to deploy these models and run them see [INSTALL.md](https://github.com/zafarali/sentiment.datalogue/blob/master/INSTALL.md).

You can track the progress and completed tasks of this project here: [Progress](https://github.com/zafarali/sentiment.datalogue/projects/1)

I will be keeping track of my explorations and observations in this README for anyone else who wants to explore.


## Requirements

0. Python 3.4.3
1. Numpy 1.12.0
2. Matplotlib 1.4.3
3. Pandas 0.16.2
5. nltk 3.0.3
4. Scikit-learn 0.16.1
5. Theano 0.8.2/Tensorflow 1.0.0
6. Keras 1.2.2

## Progress

### 11/02/2017

Doing some EDA using a IPython notebook. I want to learn what kind of words are in each corpus. The goal of doing this would be that we can use this knowledge to remove words which have low predictive power. We will also use this time to preprocess some of the data and save the datasets in an easy form for loading later.

I first looked at some global statistics, how many instances do we have?

#### Distribution of Ratings

![Ratings Distribution](https://cloud.githubusercontent.com/assets/6295292/22858576/6ca32df0-f08f-11e6-8a88-048db20a3f34.png)

As explained in `./data/aclImdb/README` the middle ground ratings were removed and this is reflected on our distribution. 

#### Analysis of Most common words

An interesting further analysis might be to see the most common words in each rating category. This could lead us to find words that are shared amongst all the ratings and remove them as "non-predictive". I first used `Counter` and `nltk` to iteratively find the stop words to remove from the corpus. I ended up removing `nltk.corpus.stopwords.words('english') + list(string.punctuation) + ['``', '...', '--', ',', ':', 'br', '\'s', '\'', 'n\'t']` from the corpus.

(!) Before we go further. One must stop and ask the following question: Isn't more than just the words in each corpus that makes a difference? What about the order of the words? This is a fine observation, but no analysis is without some merit and knowing this might pay off later.

Since each dataset contained the same number of instances, we don't need to normalize and therefore let us look at the frequencies of the 15 most common words in ech of the corpos(es?, copri?) as well as in the joint dataset:


![Word Distribution](https://cloud.githubusercontent.com/assets/6295292/22858892/c27a6c58-f098-11e6-9114-31a2b58eb000.png)

(Disclaimer: I have cut the barchart at 25000 to ensure we get a good look at the most important stuff. I have also supressed the counts for words that appear in one or the other, not out of laziness (...) but because it also gives us some information)

This is incredibly interesting! As expected "film" and "movie" are the dominant words used in movie revies. Wierdly, people used "good" when talking about positive and negative sentiments almost equally (~`that movie was good for nothing! :@`). The word "like" was used slightly more often in the negative context. Even the next few words after that in my mind can only be put in the positive form ~`even I would really recommend seeing this movie`, but the words "even", "really", "would", "like" are used more in the negative...

After this we get more discriminative words: 

1. People who were speaking in the positive seemed to talk about the "story" and "great", "well", "first". (~`Well that story was great!`)
2. People talking in the negative seemd to talk about "get", "much", "could", and ofcourse "bad". To me these words are not particularly negative(ish).

There are two things to take away from this:

1. Our wordly misconceptions about which words would be used most often is wrong.
2. Using common words / frequencies do not make for good features.

#### Future: Dunnings Loglikelihood

In a [previous competition (twitfem)](https://github.com/zafarali/twitfem) we had used the Dunnings Loglikelihood to tell us which words appeared in each corpus not by chance. This might be a useful metric here to remove non-predictive words. Let us explore it. I looked into the literature to try to understand further what this"likelihood" test was doing. 

Dunning, Ted. ["Accurate methods for the statistics of surprise and coincidence."](http://www.aclweb.org/website/old_anthology/J/J93/J93-1003.pdf) Computational linguistics 19.1 (1993): 61-74.

### 16/02/2017: (preliminary) Feature Extraction

I used `HashingVectorizer` and `TfidfTransformer` to obtain features for positive and negative reviews. I capped the number of features at 5000 and 10000 to allow for quick prototyping. For each of these I also extract three datasets: bigrams, unigrams and bigrams + unigrams. 

### 17/02/2017: Baseline methods

The `./baseline_analysis.py` script generates train/test results for the different features that were extracted the day before. It's interesting to note that bigram and unigram+bigram features do not seem to capture the sentiments well (as seen from `./RESULTS.md`). This could be due to the cap introduced of 5000 and 10000 features. We can see this difference by comparing the classifiers for 5000 and 10000 features respectively, to see that they perform marginally better in terms of accuracy. There is however, evidence of overfitting to the training set when using bigrams and unigrams+bigrams as the testing accuracies are ≈>10% lower than training. The NaiveBayes classifier was dissapointing and the best performing model is LogisticRegression using TFIDF using a cap of 10000 features.

#### Future:

1. Obtain the GloVe Embeddings for the dataset (this will probably be done when I start creating neural networks) 
2. Try the LogisticRegression using TFIDF and a larger number of features.


### 19/02/2017: Neural Network Models

Firstly I created a script in `./data/create_embeddings.py` which will convert the sentences into word indices and a corresponding word embedding matrix. This will allow us to use word embeddings like GloVe and word2vec. It also has a character encoder to allow us to do [Text Understanding from Scratch](https://arxiv.org/pdf/1502.01710.pdf).

Now that we have our baseline model, we modify the experiments slightly here. We will not use `X_test.npy` until we have decided what our model parameters will be. Then for each model we will evaluate `X_test` allowing us to get a fair comparison amongst them.

![image](https://cloud.githubusercontent.com/assets/6295292/23109507/d25df13a-f6e7-11e6-967c-808b9c39b018.png)

It seems from the distribution of text lengths that a text length of about 1000 should be good to capture the diversity in the data. 

*Edit*: Turns out that sequence length here just corresponds to the length of the characters. Considering that the average english word contains 5 characters, I predict that 800/5 = 160 is the average length of the review in words. Thus in my future experiments I capped to 300 words which would ensure that we capture as much information as necessary.

### 25/02/2017

#### Basic Model: Fully connected neural network

So I spent the day writing a `run` script as well visualizations for the metrics that the challenge requires: accuracy, cost and auc curves. Finally got embeddings to work!

I proceeded to do the following experiments where I vary only one of the "hyperparameters" at a time:

1. Compare "relu" vs "sigmoid"
2. Comapre dropouts of `[0, 0.25, 0.5, 0.75]`
3. batch size of 15 vs 32
4. hidden configurations `[500, 250, 125]`, `[500, 250]`, `[500, 500, 125]`.

Note that I did not perform k-fold cross validation to quantitatively compare each of these due to the computational overload and tight time-frame of the project. 

*Findings:* I found that `relu`'s are a brilliant way to train networks quickly compared to the good ol' `sigmoid`. Also noise doesn't help train the network with tfidf. This could be because the tfidf values are already quite small and noising them will result in very blurry data that doesn't really make sense. A larger batch size of 32 outperformed a batch size of 15. Dropout was necessary to prevent overfitting too quickly.

Turns out that fully connected neural network is able to use the tfidf preprocess and do a really good job when compared to logistic regression on the dataset with 1-grams and a vocabulary size of 10000. The network parameters are:

```
{
	"hiddens": [500, 500, 125], 
	"dropout": 0.50, 
	"activations": ["relu", "relu", "relu"],
	"input_noise": false
}
```

This served as a good first glimpse into how well the model can perform and what our benchmarks to beat are. In particular we seem to have found good batch sizes and good activation functions.

### 26/02/2017

#### Convolution Neural Networks

Using an embedding layer to extract features from a sequence and padding to 300 words. Based on the discussion in [Kim, Yoon. "Convolutional neural networks for sentence classification." arXiv preprint arXiv:1408.5882 (2014).](https://arxiv.org/abs/1408.5882). To create the dataset, I converted sequences of word vectors which are present in the location of words which represent the 5000 most common words. I also used the Glove embedding size of 50. I experimented to look for different number of filters and filter sizes that was able to learn fastest. I found that the following parameters performed the best:

```
{
	"activation": "relu",
	"filters": [250],
	"filter_sizes": [3], 
	"dropout": false,
	"hiddens": [],
	"batch_norm": false,
	"maxpool_size":false
}
```

Unsurprisingly these are the same parameters used in the out of the box [CNN example from keras](https://github.com/fchollet/keras/blob/master/examples/imdb_cnn.py). I also compared using Glove vs Learned embeddings. When comparing GloVe vs Learned Embeddings:

| | GloVe | Learned |
|----|----|----|
| Accuracy | 0.854 | 0.831 |
| AUC-ROC | 0.932 | 0.91 |

It seems that Glove Embeddings (with fine-turning) to slightly better and converge faster (2 epochs vs 5 epochs). 

#### LSTMs

I tried LSTMs with output sizes of 50 (due to computational constraints on my side). I set the dropout for the hidden gates to be `0.2` as in the keras implementation. Here the LSTM learns the sequential dependencies between the words

```
{
	"dropW":0.2,
	"dropU":0.2,
	"n_dims":50
}
```

#### BiLSTMs

I used the same configuration as above for LSTMs for a bidirectional LSTM which incorporates information from the other direction of the sequence. This however, doesn't seem to affect how quickly the model learns but the model is (slightly) more accurate:

![image](https://cloud.githubusercontent.com/assets/6295292/23369931/f30ba130-fce0-11e6-9980-c5629400025f.png)
![image](https://cloud.githubusercontent.com/assets/6295292/23370096/95774b5e-fce1-11e6-9066-62b7fb4e81ce.png)

It also has a slightly better ROC curve:
![image](https://cloud.githubusercontent.com/assets/6295292/23371743/21cb181a-fce7-11e6-9d8a-3902e172af3f.png)

| | Accuracy | AUC-ROC |
|----|----|----|
| LSTM - Glove | 0.850 | 0.922 |
| LSTM - Learned | 0.849 | 0.920 |
| BiLSTM - Glove | 0.859| 0.930 |
| BiLSTM - Learned | 0.852 | 0.923 |

It also seems using GloVe embeddings are slightly better than using learned embeddings. However, I believe that learning custom *sentiment embeddings* like in [Tang, Duyu, et al. "Sentiment embeddings with applications to sentiment analysis." IEEE Transactions on Knowledge and Data Engineering 28.2 (2016): 496-509.](http://ieeexplore.ieee.org.proxy3.library.mcgill.ca/stamp/stamp.jsp?tp=&arnumber=7296633) might be an even better option.

Given that both LSTMs and CNNs reach an almost equivalent accuracy, maybe combining them can do even better?

#### CNN-LSTMS

Inspired by [Wang, Jin, et al. "Dimensional sentiment analysis using a regional cnn-lstm model." The 54th Annual Meeting of the Association for Computational Linguistics. Vol. 225. 2016.](http://www.aclweb.org/anthology/P/P16/P16-2.pdf#page=259) where they break up word sequences into regions, apply convolutions and then use an LSTM layer to integrate the final information to feed into a classifier, I used `Convolution1D` or `LocallyConnected1D` to extract local information (i.e bigrams, short sequences of words) to feed into the BiLSTM layer from above to see if we can integrate all this information to get a better predictive model.

The final model combined the CNN base model combined with an LSTM layer on top:
```
Embedding
CONV1D
RELU
AVGPOOL
LSTM/BILSTM
DENSE
```

<!-- Considering that fine-tuning GloVe embeddings seem to create better performing models, I used that for training.  -->
I found that I had to reduce the number of filters in the convolution layer to ensure that the model began learning. However, I found that increasing the filter size did not make learning better. I also found that adding `Dense` layers on top of the architecture did not result in better performance, and in fact made it slightly worse.

I found the following parameters to perform the best:

```
{
	"n_filters": 70,
	"filter_size": 2,
	"pool_length": 5,
	"dropout": 0.2,
	"dropW": 0.2,
	"dropU": 0.2,
	"lstm_dims": 50,
	"bidirectional": true,
	"hiddens":[]
}
```

A regular LSTM model performed only slightly worse. 


#### 28/02/2017

Now that we have identified all the "good" parameters, our empirical observations need to be made quantitative by repeating the experiments and evaluating on the test dataset. To do this I have implemented in `run` the `--use-test true` flag that can be used to do this which saves the results into an `results.csv` file. To save time, I take an average of the accuracy and AUC from 3-4 repeats. The ROC-AUC and loss/accuracy curves are from one run only.


#### Graphs!

![image](https://cloud.githubusercontent.com/assets/6295292/23449818/07b04a46-fe26-11e6-8496-f0294820415a.png)




