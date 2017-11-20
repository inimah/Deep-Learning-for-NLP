# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "12.11.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import os
import sys
import numpy as np
#import _pickle as cPickle
import cPickle
import h5py

from language_models import *
from keras.callbacks import Callback
from keras.models import Sequential, Model
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical

from keras.models import load_model


'''
DATAPATH = 'data/airlines/python2'
W2V_PATH = 'embeddings/w2v/python2'
RESULTS = 'embeddings/rnn/run2'
'''
DATAPATH = 'data/airlines'
W2V_PATH = 'embeddings/w2v'
RESULTS = 'embeddings/sent_rnn/run3'


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

# saving file into hdf5 format
# only works for array with same length
################################################
def saveH5File(h5Filename,datasetName,dataToWrite):
		# h5Filename should be in format 'name-of-file.h5'
		# datasetName in String "" format
		with h5py.File(h5Filename, 'w') as hf:
				hf.create_dataset(datasetName,  data=dataToWrite)

# reading file in hdf5 format
# only works for array with same length
################################################
def readH5File(h5Filename,datasetName):
		# h5Filename should be in format 'name-of-file.h5'
		# datasetName in String "" format
		with h5py.File(h5Filename, 'r') as hf:
				data = hf[datasetName][:]
		return data


# return a numeric sequence of words in a sentence  into one-hot vector
# this is for last layer with time distributed (3D dimension)
# the text input is whole text of documents (without splitting into sentences)
def matrixVectorization3D(docs,timesteps,vocab_length):
	# create zero matrix
	sequences = np.zeros((len(docs), timesteps, vocab_length))
	# the encoding here is also the index of word in vocabulary list as such 
	# it is the index of one hot vector with '1' value
	for i, text in enumerate(docs):
		for j, word_index in enumerate(text):
			sequences[i, j, word_index] = 1
	return sequences

if __name__ == '__main__':

	# reading stored pre-processed 

	vocab = readPickle(os.path.join(DATAPATH,'vocab'))
	
	revertVocab = dict((v,k) for (k,v) in vocab.items())
	eof_token = revertVocab['EOF']

	# use word embedding from previously trained skipgram model
	w2v_embed1 = readPickle(os.path.join(W2V_PATH,'w2v_embed1'))
	

	VOCAB_LENGTH = len(vocab)
	EMBEDDING_DIM = w2v_embed1.shape[1]
	MAX_SEQUENCE_LENGTH = 35

	X = readPickle(os.path.join(RESULTS,'lm_paddedx_input'))
	Y = readPickle(os.path.join(RESULTS,'lm_paddedy_output'))

	model = load_model(os.path.join(RESULTS,'lm_bilstm.h5'))
	model_weights = h5py.File(os.path.join(RESULTS,'weights_lm_bilstm.hdf5_100.hdf5'),'r')

	# project input to the last prediction layer
	# this is also similar with directly use model to predict sequence of input --> model.predict(X)
	prediction = Model(inputs=model.input, outputs=model.get_layer('td_prediction').output)
	
	# sampling from preprocessed (padded) input sequence
	# at the moment we are using fixed and closed style of training and testing
	# (using fixed vocabulary, using same training and testing sets)
	test_X = X[100:200]
	pred_test = prediction.predict(test_X)

	# input text to be predicted (in its tokenized string format)
	str_X = []
	for i, seq in enumerate(test_X):
		tmp = ""
		for num in seq:
			w = vocab[num]
			w= w.replace('EOF','.')
			if w != 'SOF':
				tmp += w + " "
		str_X.append(tmp)

	# decoding the prediction values to its integer format
	# taking maximum weight
	dec_X = [] 
	for i, seqpred in enumerate(pred_test):
		dec_seq=[]
		for seq in seqpred:
			tmp = np.argmax(seq)   
			dec_seq.append(tmp)
		dec_X.append(dec_seq)


	# output text in its string / textual form
	str_dec_X = []
	for i, seq in enumerate(dec_X):
		tmp = ""
		for num in seq:
			w = vocab[num]
			w = w.replace('EOF','.')
			if w != 'SOF':
				tmp += w + " "
		str_dec_X.append(tmp)

	# this will be needed later to check original tweets
	df_tweets = readPickle(os.path.join(DATAPATH,'fin_dftweets'))
	tweets = df_tweets['text']

	labelled_tweets = readPickle(os.path.join(DATAPATH,'fin_labelled_sent_strtweets'))
	tokenized_tweets = []
	for i, data in enumerate(labelled_tweets):
		text = data[0]
		tokenized_tweets.append(text)

