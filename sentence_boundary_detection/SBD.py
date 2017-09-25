from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import numpy as np
from numpy import argmax
import os, sys

unk = 'unk'
out_file = 'SBD.test.out'

class Feature:
	def __init__(self, dataset):
		self.formDictionary(dataset)

	def formDictionary(self, dataset):
		print('==== Forming Dictionary ====')
		self.dict = set()
		for data in dataset:
			if data[2] != 'TOK':
				left = data[1][0:-1] if data[2] == 'EOS' else data [1]
				right_idx = data[0] - dataset[0][0] + 1
				right = dataset[right_idx][1] if len(dataset) > right_idx else ''
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
		print('==== Extracting Feature ====')
		x, y = [], []
		for data in dataset:
			if data[1][-1] == '.':
				vector = []
				left = self.containingPeriod(data[1])
				right_idx = data[0] - dataset[0][0] + 1
				right = dataset[right_idx][1] if len(dataset) > right_idx else ''
				vector.extend(self.word2Vec(left))
				vector.extend(self.word2Vec(right))
				vector.append(Feature.leftIsLongerThan3(left))
				vector.append(Feature.isCapitalized(left))
				vector.append(Feature.isCapitalized(right))
				x.append(vector)
				y.append(data[2])
		return x, y

	@staticmethod
	def leftIsLongerThan3(left):
		return 1.0 if len(left) else 0.0

	@staticmethod
	def isCapitalized(s):
		return 1.0 if len(s) > 1 and s[0].isupper() else 0.0

	def word2Vec(self, w):
		if w not in self.dict:
			w = unk
		encode_word = self.encoder.transform([w])
		return self.onehot.transform([encode_word]).toarray().tolist()[0]

	def containingPeriod(self, w):
		if w[0:-1] in self.dict:
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
		print('==== Training Start ====')
		self.clf.fit(x, y)
		predict_values = self.clf.predict(x)
		error = 0
		error = sum([1 if y[i] != predict_values[i] else 0 for i in range(len(y))])
		print('Loss: ' + str(error))
		print('Accuracy: ' + str(1 - float(error)/len(y)))

	def predict(self, x):
		print('==== Decoding ====')
		y = self.clf.predict(x)
		return y

def loadData(path):
	file = open(path ,'r')
	dataset = [[int(line.split()[0]), line.split()[1], line.split()[2]] for line in file]
	file.close()
	file = open('out.txt', 'w')
	for data in dataset:
		file.write(str(data) + '\n')
	file.close()
	return dataset

def writeFile(out_file, test_dataset, predict_values):
	f = open(out_file, 'w')
	f.close()

if __name__ == '__main__':
	path = os.getcwd() + '/SBD.train'
	train_file = sys.argv[1]
	test_file = sys.argv[2]
	train_dataset = loadData(train_file)
	test_dataset = loadData(test_file)

	f = Feature(train_dataset)
	train_x, train_y = f.extractFeature(train_dataset)
	trainer = Trainer()
	trainer.train(train_x, train_y)

	test_x, test_y = f.extractFeature(test_dataset)
	predict_values = trainer.predict(test_x)
	error = 0
	error = sum([1 if test_y[i] != predict_values[i] else 0 for i in range(len(test_y))])
	print('Test Loss: ' + str(error))
	print('Test Accuracy: ' + str(1 - float(error)/len(test_y)))