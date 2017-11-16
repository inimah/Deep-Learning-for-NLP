from __future__ import print_function
import os
import sys
import time
import math

from plotly import __version__
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.plotly as py
import plotly.graph_objs as go
from plotly import tools
#import _pickle as cPickle
import cPickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


DATAPATH = 'data/airlines'
LM_PATH = 'embeddings/rnn/run1'


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

if __name__ == '__main__':

	# number of documents in batch training
	batch_enc = 1000
	# processing gradient loss per mini batch document 
	mini_batch = 100

	# nsamples (in this case is number of sample subsets for each epoch) = total_documents / batch_enc
	labelled_strtweets = readPickle(os.path.join(DATAPATH,'fin_labelled_strtweets'))
	total_documents = len(labelled_strtweets)

	nsubsets = math.ceil(total_documents/float(batch_enc))
	# nbatch_samples (in this case number of subsets for each mini batch when computing gradient loss function) = batch_enc / mini_batch
	# e.g. for each 1000 subset of document, the model trains gradient loss every 100 documents
	nsubsets_minibatch = math.ceil(batch_enc / mini_batch)

	lm_loss = readPickle(os.path.join(LM_PATH,'lm_bilstm_loss'))
	

	lm_accuracy  = readPickle(os.path.join(LM_PATH,'lm_bilstm_accuracy'))
	

	#arr_loss_v4_50 = np.zeros((nb_epoch, nsubsets, nsubsets_minibatch))
	all_loss = []
	avg_loss = []
	avg_ep_loss = []
	for i, arr_batch in enumerate(lm_loss):
		tmp_batch = []
		for j, arr_minibatch in enumerate(arr_batch):
			all_loss.extend(arr_minibatch)
			tmp_batch.extend(arr_minibatch)
			avg_minibatch = sum(arr_minibatch)/len(arr_minibatch)
			avg_loss.extend([avg_minibatch])
		avg_batch = sum(tmp_batch)/len(tmp_batch)
		avg_ep_loss.extend([avg_batch])

	

	all_accuracy = []
	avg_accuracy = []
	avg_ep_accuracy = []
	for i, arr_batch in enumerate(lm_accuracy):
		tmp_batch = []
		for j, arr_minibatch in enumerate(arr_batch):
			all_accuracy.extend(arr_minibatch)
			tmp_batch.extend(arr_minibatch)
			avg_minibatch = sum(arr_minibatch)/len(arr_minibatch)
			avg_accuracy.extend([avg_minibatch])
		avg_batch = sum(tmp_batch)/len(tmp_batch)
		avg_ep_accuracy.extend([avg_batch])

	
			

	X = [ i for i in range(len(all_loss))]
	

	X_avg = [ i for i in range(len(avg_loss))]


	X_ep_avg = [ i for i in range(len(avg_ep_loss))]
	


	Y = all_loss

	Y_avg = avg_loss
	
	Y_ep_avg = avg_ep_loss
	

	Y_acc = all_accuracy


	Y_avg_acc = avg_accuracy


	Y_ep_avg_acc = avg_ep_accuracy


	loss = go.Scatter(
	x = X,
	y = Y,
	mode = 'lines'
	)

	

	avg_loss = go.Scatter(
	x = X_avg,
	y = Y_avg,
	mode = 'lines'
	)

	

	avg_ep_loss = go.Scatter(
	x = X_ep_avg,
	y = Y_ep_avg,
	mode = 'lines'
	)

	


	acc = go.Scatter(
	x = X,
	y = Y_acc,
	mode = 'lines'
	)

	
	avg_acc = go.Scatter(
	x = X_avg,
	y = Y_avg_acc,
	mode = 'lines'
	)



	avg_ep_acc = go.Scatter(
	x = X_ep_avg,
	y = Y_ep_avg_acc,
	mode = 'lines'
	)

	
	data_loss = [loss]
	data_avg_loss = [avg_loss]
	data_ep_avg_loss = [avg_ep_loss]

	data_acc = [acc]
	data_avg_acc = [avg_acc]
	data_ep_avg_acc = [avg_ep_acc]

	layout_loss = go.Layout(
		title = 'Training error loss',
		xaxis = dict(
			 	title ='Batch-epoch'
			),
		yaxis = dict(
				title = 'Error'
			)
		)

	layout_acc = go.Layout(
		title = 'Training accuracy',
		xaxis = dict(
			 	title ='Batch-epoch'
			),
		yaxis = dict(
				title = 'Accuracy'
			)
		)

	layout_ep_loss = go.Layout(
		title = 'Training error loss',
		xaxis = dict(
			 	title ='Epoch'
			),
		yaxis = dict(
				title = 'Error'
			)
		)

	layout_ep_acc = go.Layout(
		title = 'Training accuracy',
		xaxis = dict(
			 	title ='Epoch'
			),
		yaxis = dict(
				title = 'Accuracy'
			)
		)

	fig_loss = go.Figure(data=data_loss, layout=layout_loss)
	fig_acc = go.Figure(data=data_acc, layout=layout_acc)

	plot(fig_loss, filename= os.path.join(LM_PATH,'training_error.html'), image='png')
	plot(fig_acc, filename= os.path.join(LM_PATH,'training_accuracy.html'), image='png')

	fig_avg_loss = go.Figure(data=data_avg_loss, layout=layout_loss)
	fig_avg_acc = go.Figure(data=data_avg_acc, layout=layout_acc)

	plot(fig_avg_loss, filename= os.path.join(LM_PATH,'training_avg_error.html'), image='png')
	plot(fig_avg_acc, filename= os.path.join(LM_PATH,'training_avg_accuracy.html'), image='png')

	fig_ep_avg_loss = go.Figure(data=data_ep_avg_loss, layout=layout_ep_loss)
	fig_ep_avg_acc = go.Figure(data=data_ep_avg_acc, layout=layout_ep_acc)

	plot(fig_ep_avg_loss, filename= os.path.join(LM_PATH,'training_ep_avg_error.html'), image='png')
	plot(fig_ep_avg_acc, filename= os.path.join(LM_PATH,'training_ep_avg_accuracy.html'), image='png')