# -*- coding: utf-8 -*-
#__author__ = "@tita"
#__update__ = "11.10.2017"
#__maintainer__ = "@tita"
#__email__ = "i.nimah@tue.nl"

from __future__ import print_function
import os
import sys
import time
import numpy as np
import _pickle as cPickle
import lda
import matplotlib
import matplotlib.pyplot as plt


DATAPATH = 'data/airlines'
RESULTS = 'embeddings/word_lda'


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

		

	############################
	# LDA on TFIDF matrix

	td_tfidf_arr = readPickle(os.path.join(DATAPATH,'arr_td'))
	tfidf_arr = np.array(td_tfidf_arr, dtype='int32')

	# default iteration = 1000
	model_tfidf = lda.LDA(n_topics=10, n_iter=1500, random_state=1)
	model_tfidf.fit(tfidf_arr)

	plt.figure()
	plt.plot(model_tfidf.loglikelihoods_[5:])
	plt.savefig(os.path.join(RESULTS,'lda_loglikelihood.png'))
	plt.clf()


	topic_word_tfidf = model_tfidf.topic_word_
	topic_doc_tfidf = model_tfidf.doc_topic_

	savePickle(model_tfidf, os.path.join(RESULTS,'lda_tfidf'))
	savePickle(topic_word_tfidf, os.path.join(RESULTS,'topic_word_tfidf'))
	savePickle(topic_doc_tfidf, os.path.join(RESULTS,'topic_doc_tfidf'))




