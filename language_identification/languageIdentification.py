# Author: Chien-Wei Lin

import sys
import numpy as np
from sklearn import preprocessing
import json

experiment = ''
unk = ''

hyperparameter = {
	'lr' : 0.1,
	'retain_rate': 0.5,
	'batch_size': 100, 
	'epoch': 5
}

structure = [50, 20]

def sgm(v, derivative = False):
	y = 1 / (1 + np.exp(-x))
	if not der:
		return y
	else:
		return y * (1 - y)

class Feature:
	def __init__(self, dataset):
		self.formDictionary(dataset)

	def formDictionary(self, file_name):
		f = open(file_name, 'r')
		self.dic = set()
		for line in f:
			idx = line.find(' ') + 1
			for i in range(idx, len(line)):
				self.dic.add(line[i])
		self.dic.add(unk)
		self.dic = list(self.dic)
		f.close()

	def initEncoder(self):
		self.encoder = preprocessing.LabelEncoder()
		self.encoder.fit(self.dict)

	def initOneHot(self):
		self.onehot = preprocessing.OneHotEncoder()
		encode_words = self.encoder.transform(self.dict)
		self.onehot.fit([[w] for w in encode_words])

	def chars2Vec(self, chars):
		encode_words =[[c] for c in self.encoder.transform(chars)]
		return self.onehot.transform(encode_words).toarray().tolist()

	def extractFeature(self, file_name, isTest=False):
		f = open(file_name, 'r')
		for line in f:
			idx = line.find(' ') + 1
			for i in range(idx, len(line)):


class Networks:
	def __init__(self, act=sgm, recover=False):
		self.initPara()
		self.initActivation(act)
		if not recover:
			self.initWeights()
		else:
			self.loadNetwork()

	def initPara(self):
		self.lr = hyperparameter['lr']
		self.retain_rate = hyperparameter['retain_rate']
		self.batch_size = hyperparameter['batch_size']
		self.epoch = hyperparameter['epoch']

	def initActivation(self, act):
		self.activation = act

	def initWeights(self):


	def feedForward(self, features):
		return out

	def backPropagate(self):

	def adaptLearnRate(self, dev_features, dev_targets):
		predict_values = self.predict(dev_features)
		loss = self.loss(predict_values, dev_targets)
		# TODO
		accuracy = loss / len(predict_values)
		if pre_accuracy >= accuracy:
			self.lr = self.lr / 2

	def trainBatch(self, train_features, train_targets):
		predict_values = self.feedForward(train_features)
		self.backPropagate(train_targets)

	def train(self, train_features, train_targets, dev_features, dev_targets):
		for i in range(self.epoch):
			for batch in self.batches:
				self.trainBatch(batch)
			self.adaptLearnRate(dev_features, dev_targets)

	def predict(self, features):
		return self.feedForward(features)

	def loss(self, predict_values, targets):
		dif = sum([1 if predict_values[i] != targets[i] else 0 for i in range(len(targets))])
		return dif


if __name__ == '__main__':
	train_file = sys.argv[1]
	dev_file = sys.argv[2]
	test_file = sys.argv[3]

	train_features, train_targets = loadData(train_file)
	dev_features, dev_targets = loadData(dev_file)

	networks = Networks()
	networks.train(train_features, train_targets, dev_features, dev_targets)

