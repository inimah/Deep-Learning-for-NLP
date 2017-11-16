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
matplotlib.use('agg')
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

def plotPieSentiment(x,y):

	labels = x
	values = y

	trace = go.Pie(labels=labels, values=values)

	#iplot([trace], filename='plot_pie_sentiment.html')

	plot([trace], filename=os.path.join(RESULTS,'plot_pie_sentiment.html'))

	return 0

def plotBarSentiment(x,y):

	trace = go.Bar(
		x=x,
		y=y,
		marker=dict(
		color=['rgba(55, 128, 191, 0.7)', 'rgba(219, 64, 82, 0.7)',
			   'rgba(50, 171, 96, 0.7)']),
	)

	data = [trace]
	layout = go.Layout(
		title='Sentiment Classification of Tweets',
		xaxis = dict(
				title = 'Sentiment'
			),
		yaxis = dict(
				title = 'Number of Tweets'
			)
	)

	fig = go.Figure(data=data, layout=layout)
	plot(fig, filename=os.path.join(RESULTS,'plot_bar_sentiment.html'))

	return 0



def plotPieReasons(x,y):

	

	fig = {
		  "data": [
			{
			  "values": y,
			  "labels": x,
			  "domain": {"x": [0, .48]},
			  "name": "Negative sentiment",
			  "hoverinfo":"label+percent+name",
			  "hole": .4,
			  "type": "pie"
			}],
		  "layout": {
				"title":"Reasons for Negative Sentiment",
				"annotations": [
					{
						"font": {
							"size": 20
						},
						"showarrow": False,
						"text": "Negative",
						"x": 0.20,
						"y": 0.5
					}
				]
			}
		}
	plot(fig, filename=os.path.join(RESULTS,'plot_pie_negreasons.html'))

	return 0

if __name__ == '__main__':


	vocab = readPickle(os.path.join(DATAPATH,'vocab'))
	labelled_strtweets = readPickle(os.path.join(DATAPATH,'fin_labelled_strtweets'))
	labelled_numtweets = readPickle(os.path.join(DATAPATH,'fin_labelled_numtweets'))

	df_tweets = pd.DataFrame(labelled_strtweets, columns=['tweets', 'context', 'sentiment'])
	df_pos = df_tweets[df_tweets['sentiment']=='positive']
	df_neg = df_tweets[df_tweets['sentiment']=='negative']
	df_neutral = df_tweets[df_tweets['sentiment']=='neutral']

	sentiment_x = ['neutral','positive','negative']
	sentiment_y = [len(df_neutral),len(df_pos),len(df_neg)]

	plotPieSentiment(sentiment_x,sentiment_y)

	plotBarSentiment(sentiment_x,sentiment_y)


	'''
	In [12]: len(df_tweets)
	Out[12]: 13993

	In [13]: len(df_pos)
	Out[13]: 2105

	In [15]: len(df_neg)
	Out[15]: 9041

	In [17]: len(df_neutral)
	Out[17]: 2847

	'''

	# context or reasons for negative sentiment
	reason_neg = list(set(df_neg['context']))
	'''
	Out[21]: 
	['Lost Luggage',
	 'Late Flight',
	 'Flight Booking Problems',
	 'Bad Flight',
	 'Flight Attendant Complaints',
	 'Damaged Luggage',
	 "Can't Tell",
	 'longlines',
	 'Customer Service Issue',
	 'Cancelled Flight']

	'''

	n_tweets = []
	for i, reason in enumerate(reason_neg):
		df_tmp = df_neg[df_neg['context']==reason]
		num_df = len(df_tmp)
		n_tweets.append(num_df)

	plotPieReasons(reason_neg, n_tweets)


	tokenized_tweets = []
	labels_tweets = []
	context_tweets = []
	for i, data in enumerate(labelled_strtweets):
		text = data[0]
		context = data[1]
		label = data[2]
		tokenized_tweets.append(text)
		context_tweets.append(context)
		labels_tweets.append(label)
