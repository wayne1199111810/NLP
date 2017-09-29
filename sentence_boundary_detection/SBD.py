# Author: Chien-Wei Lin
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import numpy as np
from numpy import argmax
import os, sys, string

unk = 'unk'
out_file = 'SBD.test.out'

class Feature:
	def __init__(self, dataset):
		self.formDictionary(dataset)

	def formDictionary(self, dataset):
		self.dict = set()
		for data in dataset:
			if data[2] != 'TOK':
				left = data[1][0:-1] if data[2] == 'EOS' else data[1]
				right_idx = data[0] - dataset[0][0] + 1
				right = Feature.getRightfromTrain(dataset, right_idx)
				self.dict.add(left)
				self.dict.add(right)
		self.dict.add(unk)
		self.dict = list(self.dict)

		self.encodeWords()
		self.createOheHot()

	def encodeWords(self):
		self.encoder = preprocessing.LabelEncoder()
		self.encoder.fit(self.dict)

	def createOheHot(self):
		self.onehot = preprocessing.OneHotEncoder()
		encode_words = self.encoder.transform(self.dict)
		self.onehot.fit([[w] for w in encode_words])

	def extractFeature(self, dataset):
		x, y, idx = [], [], []
		for data in dataset:
			if data[1][-1] == '.':
				vector = []
				left = self.isInDict(data[1])
				right_idx = data[0] - dataset[0][0] + 1
				right = self.isInDict(dataset[right_idx][1]) if len(dataset) > right_idx else ''
				vector.extend(self.word2Vec(left))
				vector.extend(self.word2Vec(right))
				vector.append(Feature.leftIsLongerThan3(left))
				vector.append(Feature.isCapitalized(left))
				vector.append(Feature.isCapitalized(right))
				vector.append(Feature.isMorePeriod(left))
				vector.append(Feature.numberOfUpper(left))
				vector.append(Feature.isPunctuation(right))
				x.append(vector)
				y.append(data[2])
				idx.append(data[0] - dataset[0][0])
		return x, y, idx

	@staticmethod
	def getRightfromTrain(dataset, right_idx):
		if len(dataset) > right_idx:
			line = dataset[right_idx]
			right = line[1][0:-1] if line[2] == 'EOS' else line[1]
		else:
			right = ''
		return right

	@staticmethod
	def leftIsLongerThan3(left):
		return 1.0 if len(left) > 3 else 0.0

	@staticmethod
	def isCapitalized(s):
		return 1.0 if len(s) > 1 and s[0].isupper() else 0.0

	@staticmethod
	def isMorePeriod(left):
		count = 0
		for i in left:
			if i == '.':
				count += 1
		return 1.0 if count > 0 else 0.0

	@staticmethod
	def numberOfUpper(left):
		count = 0.0
		for i in left:
			if i.isupper():
				count += 1
		return count

	@staticmethod
	def isPunctuation(s):
		return 1.0 if s in string.punctuation else 0.0

	def word2Vec(self, w):
		if w not in self.dict:
			w = unk
		encode_word = self.encoder.transform([w])
		return self.onehot.transform([encode_word]).toarray().tolist()[0]

	def isInDict(self, w):
		if w[-1] == '.' and w[0:-1] in self.dict:
			return w[0:-1]
		elif w in self.dict:
			return w
		else:
			return unk

	def words2Vec(self, words):
		encode_words =[[w] for w in self.encoder.transform(words)]
		return self.onehot.transform(encode_words).toarray().tolist()

	def vec2Word(self, array):
		return self.encoder.inverse_transform([np.argmax(array)])[0]

class Trainer():
	def __init__(self):
		self.clf = DecisionTreeClassifier()

	def train(self, x, y):
		self.clf.fit(x, y)
		predict_values = self.clf.predict(x)

	def predict(self, x, y):
		predict_values = self.clf.predict(x)
		Trainer.evaluate(y, predict_values)
		return predict_values

	@staticmethod
	def evaluate(y, predict_values):
		error = sum([1 if y[i] != predict_values[i] else 0 for i in range(len(y))])
		print('Loss: ' + str(error))
		accuracy = (1 - float(error)/len(y)) * 100
		print('Accuracy: ' + str(accuracy) + '%')

def loadData(path):
	file = open(path ,'r')
	dataset = [[int(line.split()[0]), line.split()[1], line.split()[2]] for line in file]
	file.close()
	file = open('out.txt', 'w')
	for data in dataset:
		file.write(str(data) + '\n')
	file.close()
	return dataset

def writeFile(out_file, test_dataset, predict_values, test_idx):
	f = open(out_file, 'w')
	for i in range(len(test_idx)):
		idx = test_dataset[test_idx[i]][0]
		w = test_dataset[test_idx[i]][1]
		label = predict_values[i]
		s = str(idx) + ' ' + w + ' ' + label + '\n'
		f.write(s)
	f.close()

def dataPreprocessing(train_file, test_file):
	train_dataset = loadData(train_file)
	test_dataset = loadData(test_file)
	f = Feature(train_dataset)
	train_x, train_y, train_idx = f.extractFeature(train_dataset)
	test_x, test_y, test_idx = f.extractFeature(test_dataset)
	del train_idx
	return train_x, train_y, test_x, test_y, test_idx, test_dataset

if __name__ == '__main__':
	train_file = sys.argv[1]
	test_file = sys.argv[2]
	train_x, train_y, test_x, test_y, test_idx, test_dataset = dataPreprocessing(train_file, test_file)
	trainer = Trainer()
	trainer.train(train_x, train_y)
	predict_values = trainer.predict(test_x, test_y)
	writeFile(out_file, test_dataset, predict_values, test_idx)
	