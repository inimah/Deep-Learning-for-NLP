# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "09.11.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"
from __future__ import print_function
import numpy as np
import pandas as pd
import os
import sys
import re
from itertools import groupby
from string import punctuation
import _pickle as cPickle
import nltk
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.decomposition import PCA


DATAPATH = 'data/airlines'

# saving file into pickle format
def savePickle(dataToWrite,pickleFilename):
	f = open(pickleFilename, 'wb')
	cPickle.dump(dataToWrite, f)
	f.close()

# reading file in pickle format
def readPickle(pickleFilename):
	f = open(pickleFilename, 'rb')
	obj = cPickle.load(f)
	f.close()
	return obj

# clean text /line from repeated consecutive punctuation (removes duplicates to single punctuation)
# before tokenization
def removeDuplicatePunctuations(text):

	punc = set(punctuation) 
	newtext = []
	for k, g in groupby(text):
		if k in punc:
			newtext.append(k)
		else:
			newtext.extend(g)

	textClean = ''.join(newtext)

	return textClean

# replace '.' with '. ' 
# before tokenization
# this is to ensure tokenization able to split between 'w1.w2' if this exists in text
# commonly found in twitter text since user will minimize the use of space ' '
def replaceEOF(text):

	new_text = ''
	for w in text:
		new_w = w.replace('.','. ')
		new_text += ' '.join(new_w)
	return new_text

def tokenizeWords(tweet):

	#tokens = nltk.word_tokenize(tweet)
	tokens = tweet.split()
	return tokens

# change negation form "n't" to its original form "not"
# this is done after tokenization
def changeNegation(array_of_words):

	neg = "n't"
	new_tokens = []
	for w in array_of_words:
		if w.endswith(neg):
			 w1 = w[:-len(neg)]
			 # for original token = "can't"
			 if w1=='ca':
			 	w1 = w1 + 'n'
			 new_tokens.append(w1)
			 new_tokens.append('not')
		else:
			new_tokens.append(w)

	return new_tokens

# discard forms of 've, 's in tokens 
def changeVerbForm(array_of_words):

	new_tokens = []
	for w in array_of_words:
		if "'" in w:
			new_w = w[:w.index("'")]
		else:
			new_w = w
		new_tokens.append(new_w)

	return new_tokens



# this function will detect number in string
# e.g '3-4' will be recognized as '34' instead
def numregex(array_of_words, myregex=re.compile(r'\d')):

	return "".join([s for s in array_of_words if myregex.search(s)])

# change token "@" as 'twitteraccount'
# change token "#" as 'twittermention'
# change token 'RT' as 'retweet'
# change token 'http' as 'urllink'
# change number to 'number'
def changeTweet(array_of_words):

	new_tokens = []
	for w in array_of_words:
		if w.startswith('@'):
			new_w = 'twitteraccount'
		elif w.startswith('#'):
			new_w = 'twittermention'
		elif w == 'RT':
			new_w = 'retweet'
		elif 'http' in w:
			new_w = 'urllink'
		elif numregex(w) != '':
			new_w = 'number'
		else:
			new_w = w

		new_tokens.append(new_w.lower())
	return new_tokens

# check if token endswith punctuation
# this is for training sets without splitting into sentences 
def endswithPunctuation(array_of_words):

	eof_punc = ['.','?','!',',',';']
	new_tokens = []
	for i,w in enumerate(array_of_words):
		if len(w) != 0:
			if w[-1] in eof_punc:
				if len(w) > 1:
					new_tokens.append(w[:-1])
					new_tokens.append('EOF')
				elif len(w) == 1:
					new_tokens.append('EOF')
			else:
				new_tokens.append(w)

	return new_tokens

# final clean: lowercase and remove nonalpha string
def checkIsAlpha(array_of_words):

	return [w for w in array_of_words if w.isalpha()]


# splitting into sentences
# done after checking with previous function endswithPunctuation()

def endOfSentence(i, array_of_words):
	
	return array_of_words[i] in ('EOF') and (i == len(array_of_words) - 1 or not array_of_words[i+1] in ('EOF'))

def splittingSentences(array_of_words):

	sent_tokens = []
	begin = 0

	for i, word in enumerate(array_of_words):
		if(endOfSentence(i,array_of_words)):
			sent_tokens.append(array_of_words[begin:i+1])
			begin = i+1

	if begin < len(array_of_words):
		sent_tokens.append(array_of_words[begin:])

	return sent_tokens


def indexingVocabulary(array_of_words):

	
	# frequency of word across document corpus
	tf = nltk.FreqDist(array_of_words) 
	wordIndex = list(tf.keys())
	# add 'zero' as the first vocab and 'UNK' as unknown words
	wordIndex.insert(0,'SOF')
	if 'EOF' not in array_of_words:	
		wordIndex.append('EOF')
	wordIndex.append('UNK')
	# indexing word vocabulary : pairs of (index,word)
	vocab=dict([(i,wordIndex[i]) for i in range(len(wordIndex))])

	return vocab, tf  


if __name__ == '__main__':

	

	data = pd.read_csv("data/airlines_tweets.csv")
	data.columns = [col.replace(":", "_") for col in data.columns]

	'''
	In [5]: data.columns
	Out[5]: 
	Index(['tweet_id', 'airline_sentiment', 'airline_sentiment_confidence',
	   'negativereason', 'negativereason_confidence', 'airline',
	   'airline_sentiment_gold', 'name', 'negativereason_gold',
	   'retweet_count', 'text', 'tweet_coord', 'tweet_created',
	   'tweet_location', 'user_timezone'],
	  dtype='object')

	'''


	datatweets =  data[['tweet_id','airline_sentiment','airline_sentiment_confidence', 'negativereason', 'airline','text','user_timezone']]

	raw_tweets = []
	for i in range(len(datatweets)):
		txt = datatweets['text'][i]
		raw_tweets.append(txt)

	clean_tweets = []

	for i, text in enumerate(raw_tweets):
		remove_duplicate_punc = removeDuplicatePunctuations(text)
		replace_eof = replaceEOF(remove_duplicate_punc)
		clean_tweets.append(replace_eof)


	tokenized_tweets = []
	sent_tweets = []
	for i, text in enumerate(clean_tweets):
		tokens = tokenizeWords(text)
		change_negation = changeNegation(tokens)
		change_verb_form = changeVerbForm(change_negation)
		change_tweet_form = changeTweet(change_verb_form)
		add_eof = endswithPunctuation(change_tweet_form)
		check_isalpha = checkIsAlpha(add_eof)
		
		# with splitting into sentences
		sent_tokens = splittingSentences(check_isalpha)
		sent_tweets.append(sent_tokens)

		# without splitting into sentences 
		tokenized_tweets.append(check_isalpha)

	# build vocabulary index
	# first, merge all array of words in document corpus
	all_tokenized_tweets = []
	for i, array_of_words in enumerate(tokenized_tweets):
		all_tokenized_tweets.extend(array_of_words)

	vocab, vocabTF = indexingVocabulary(all_tokenized_tweets)

	savePickle(vocab, os.path.join(DATAPATH,'vocab'))
	savePickle(vocabTF, os.path.join(DATAPATH,'vocabTF'))

	revertVocab = dict((v,k) for (k,v) in vocab.items())
	eof_token = revertVocab['EOF']	

	# encode document into integer format as input of RNN model
	encodedint_tweets = []
	for i, text in enumerate(tokenized_tweets):
		numTokens = [revertVocab[i] for i in text]
		encodedint_tweets.append(numTokens)

	# for splitted sentences
	encodedsent_tweets = []
	for i, sentences in enumerate(sent_tweets):
		numSentence = []
		for j, text in enumerate(sentences):
			sent_tokens = [revertVocab[i] for i in text]
			numSentence.append(sent_tokens)
		encodedsent_tweets.append(numSentence)


	labels = []
	for i in range(len(datatweets)):
		txt = datatweets['airline_sentiment'][i]
		labels.append(txt)

	sent_reason = []
	for i in range(len(datatweets)):
		txt = datatweets['negativereason'][i]
		sent_reason.append(txt)


	# clean 'nan' values to empty string instead
	cont = np.nan_to_num(sent_reason)
	context = [(x.replace('nan', '')) for x in cont]


	# zip string format of training sets to be stored as pickle
	# likewise, do the same for encoded version (integer format)
	labelled_strtweets = list(zip(tokenized_tweets,context,labels))
	labelled_numtweets = list(zip(encodedint_tweets,context,labels))
	savePickle(labelled_strtweets, os.path.join(DATAPATH,'labelled_strtweets'))
	savePickle(labelled_numtweets, os.path.join(DATAPATH,'labelled_numtweets'))

	# for splitted sentences tweets
	labelled_sent_strtweets = list(zip(sent_tweets,context,labels))
	labelled_sent_numtweets = list(zip(encodedsent_tweets,context,labels))
	savePickle(labelled_sent_strtweets, os.path.join(DATAPATH,'labelled_sent_strtweets'))
	savePickle(labelled_sent_numtweets, os.path.join(DATAPATH,'labelled_sent_numtweets'))

	
