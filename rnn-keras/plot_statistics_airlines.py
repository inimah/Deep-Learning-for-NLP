# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__date__ = "20.05.2017"
#__update__ = "09.09.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import os
import sys
import numpy as np
import pandas as pd
#import _pickle as cPickle
import cPickle

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

DATAPATH = 'data/airlines'
RESULTS = 'plots'

# reading file in pickle format
def readPickle(pickleFilename):
	f = open(pickleFilename, 'rb')
	obj = cPickle.load(f)
	f.close()
	return obj

# saving file into pickle format
def savePickle(dataToWrite,pickleFilename):
	f = open(pickleFilename, 'wb')
	cPickle.dump(dataToWrite, f)
	f.close()

# from labelled tokenized documents
def _countWord(tokenized_docs):
	count_words = []
	for i,data in enumerate(tokenized_docs):
		count_ = len(data[0])
		count_words.append(count_)
	return count_words

# count words for splitted sentences 
def _countWordSent(tokenized_docs):
	sent_words = []
	for i,data in enumerate(tokenized_docs):
		count_words = []
		for j, tokens in enumerate(data[0]):
			count_ = len(tokens)
			count_words.append(count_)
		sent_words.append(count_words)
	return sent_words

def plotFreq(counts, hist_name, title, xaxis, yaxis):

	data = [go.Histogram(x=counts)]

	layout = go.Layout(
		title=title,
		xaxis = dict(
				title = xaxis
			),
		yaxis = dict(
				title = yaxis
			)
	)

	fig = go.Figure(data=data, layout=layout)

	plot(fig, filename= os.path.join(RESULTS,hist_name))


	return 0

if __name__ == '__main__':


	vocab = readPickle(os.path.join(DATAPATH,'vocab'))
	labelled_strtweets = readPickle(os.path.join(DATAPATH,'labelled_strtweets'))
	labelled_numtweets = readPickle(os.path.join(DATAPATH,'labelled_numtweets'))

	labelled_sent_strtweets = readPickle(os.path.join(DATAPATH,'labelled_sent_strtweets'))
	labelled_sent_numtweets = readPickle(os.path.join(DATAPATH,'labelled_sent_numtweets'))
	
	# this will result frequency-based histogram plot 
	# from here, we can decide in which threshold (number of words per training sets that we want to keep to train the model on) 
	count_words = _countWord(labelled_numtweets)
	plotFreq(count_words,'word_length.html','Sequence length (number of words) per document (single tweet)','number of words','number of tweets')

	count_sent = _countWord(labelled_sent_numtweets)
	plotFreq(count_sent,'num_sentence.html','Number of sentences per document (single tweet)','number of sentences','number of tweets')

	count_words_sent = _countWordSent(labelled_sent_numtweets)

	avg_words_sent = []
	for i, w in enumerate(count_words_sent):
		sum_sent = sum(w)
		avg_sent = sum_sent/len(w)
		avg_words_sent.append(avg_sent)
	plotFreq(avg_words_sent,'sentence_length.html','Average sentence length (number of words)','number of words','number of tweets')

	'''

	

	# e.g. from plot, we will discard text sequence that has n-words < 6 and n-words > 35
	# store the index
	tmp_index = []
	for i, num in enumerate(count_words):
		if (num < 6) or (num > 35):
			tmp_index.append(i)

	savePickle(tmp_index, os.path.join(DATAPATH,'discard_docindex'))

	# re-storing original data after being reduced by this index (sequence length < 5)
	data = pd.read_csv("data/airlines_tweets.csv")
	data.columns = [col.replace(":", "_") for col in data.columns]
	datatweets =  data[['tweet_id','airline_sentiment','airline_sentiment_confidence', 'negativereason', 'airline','text','user_timezone']]

	new_df = pd.DataFrame(data=None,columns=datatweets.columns)
	j = 0
	for i in range(len(datatweets)):
		if i not in tmp_index:
			new_df.loc[j]=datatweets.loc[i]
			j += 1

	savePickle(new_df, os.path.join(DATAPATH,'fin_dftweets'))

	# re-storing labelled data after being discarded by number of sequences
	new_labelled_strtweets = []
	for i, data in enumerate(labelled_strtweets):
		if i not in tmp_index:
			new_labelled_strtweets.append(data)

	new_labelled_numtweets = []
	for i, data in enumerate(labelled_numtweets):
		if i not in tmp_index:
			new_labelled_numtweets.append(data)

	savePickle(new_labelled_strtweets, os.path.join(DATAPATH,'fin_labelled_strtweets'))
	savePickle(new_labelled_numtweets, os.path.join(DATAPATH,'fin_labelled_numtweets'))


	new_labelled_sent_strtweets = []
	for i, data in enumerate(labelled_sent_strtweets):
		if i not in tmp_index:
			new_labelled_sent_strtweets.append(data)

	new_labelled_sent_numtweets = []
	for i, data in enumerate(labelled_sent_numtweets):
		if i not in tmp_index:
			new_labelled_sent_numtweets.append(data)

	savePickle(new_labelled_sent_strtweets, os.path.join(DATAPATH,'fin_labelled_sent_strtweets'))
	savePickle(new_labelled_sent_numtweets, os.path.join(DATAPATH,'fin_labelled_sent_numtweets'))

	'''

	

	

	