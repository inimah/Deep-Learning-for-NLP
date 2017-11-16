# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "20.09.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"


from __future__ import print_function
import os
import sys
import numpy as np
import _pickle as cPickle
from language_models import *
from sklearn.preprocessing import LabelEncoder

DATAPATH = 'data/airlines'
RESULTS = 'embeddings/w2v'


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

	vocab = readPickle(os.path.join(DATAPATH,'vocab'))
	labelled_docs  = readPickle(os.path.join(DATAPATH,'fin_labelled_strtweets'))

	tokenized_docs = []
	for i, data in enumerate(labelled_docs):
		text = data[0]
		tokenized_docs.append(text)

	# word2vec model of mail subjects
	# word dimension = 50
	w2v_1, w2v_2, w2v_embed1, w2v_embed2 = wordEmbedding(tokenized_docs, vocab, 50, 50)


	# Skipgram
	w2v_1.save(os.path.join(RESULTS,'w2v_1'))
	# CBOW
	w2v_2.save(os.path.join(RESULTS,'w2v_2'))

	savePickle(w2v_embed1, os.path.join(RESULTS,'w2v_embed1'))
	savePickle(w2v_embed2, os.path.join(RESULTS,'w2v_embed2'))

	

	# create document representation of word vectors

	# By averaging word vectors
	avg_embed1 = averageWE(w2v_embed1, vocab, tokenized_docs)
	avg_embed2 = averageWE(w2v_embed2, vocab, tokenized_docs)

	savePickle(avg_embed1, os.path.join(RESULTS,'avg_embed1'))
	savePickle(avg_embed2, os.path.join(RESULTS,'avg_embed2'))


	# By averaging and idf weights of word vectors
	avgIDF_embed1 = averageIdfWE(w2v_embed1, vocab, tokenized_docs)
	avgIDF_embed2 = averageIdfWE(w2v_embed2, vocab, tokenized_docs)

	savePickle(avgIDF_embed1, os.path.join(RESULTS,'avgIDF_embed1'))
	savePickle(avgIDF_embed2, os.path.join(RESULTS,'avgIDF_embed2'))

	

