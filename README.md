# Sentiment Analysis

Motivation: This Sentiment analysis challenge was for Datalogue recruiting

Goal: Write a neural network that can classify sentiments using a corpus in `./data/`

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

It seems from the distribution of text lengths that a text length of about 1000 should be good to capture the diversity in the data. (? need to investigate)

#### Basic Model: Fully connected neural network

We start with the obvious basic model: a fully connected neural network.




