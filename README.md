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

#### Dunnings Loglikelihood

In a [previous competition (twitfem)](https://github.com/zafarali/twitfem) we had used the Dunnings Loglikelihood to tell us which words appeared in each corpus not by chance. This might be a useful metric here to remove non-predictive words. Let us explore it. I looked into the literature to try to understand further what this"likelihood" test was doing. 

Dunning, Ted. ["Accurate methods for the statistics of surprise and coincidence."](http://www.aclweb.org/website/old_anthology/J/J93/J93-1003.pdf) Computational linguistics 19.1 (1993): 61-74.

