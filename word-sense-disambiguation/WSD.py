# Author: Chien-Wei Lin
# 

import sys
import string
import math

OUT_FILE_TYPE = '.wsd.out'
FOLDS = 5
SMOOTHING_PARA = 2

class Instance:
	def __init__(self, word, lines):
		context_body = False
		self.word = word
		for line in lines:
			if Instance.isAnswer(line):
				self.getSense(line)
				self.initID(line)
			elif Instance.isContext(line):
				context_body = True
			elif Instance.isEndContext(line):
				context_body = False
			elif context_body:
				self.initContext(line, word)

	def getSense(self, line):
		words = line.strip().split('%')
		idx_of_quote = words[1].rfind('\"')
		self.sense = words[1][:idx_of_quote]

	def initID(self, line):
		words = line.strip().split()
		self.id = words[1].split('\"')[1]

	def initContext(self, line, word):
		head = '<head>' + word + '</head>'
		start_idx = line.rfind(head)
		end_idx = start_idx + len(head)
		line = line[0:start_idx] + line[end_idx:]
		self.context = list(line.split())
		Instance.filterPunctuation(self.context)
		self.lowerContextLetter()
		self.context = set(self.context)

	def lowerContextLetter(self):
		for i in range(len(self.context)):
			self.context[i] = self.context[i].lower()

	@staticmethod
	def isAnswer(line):
		words = line.strip().split()
		return False if len(words) == 0 or words[0] != '<answer' else True

	@staticmethod
	def isContext(line):
		words = line.strip().split()
		return False if len(words) == 0 or words[0] != '<context>' else True

	@staticmethod
	def isEndContext(line):
		words = line.strip().split()
		return False if len(words) == 0 or words[0] != '</context>' else True

	@staticmethod
	def isInstance(line):
		words = line.strip().split()
		return False if len(words) == 0 or words[0] != '<instance' else True

	@staticmethod
	def isEndInstance(line):
		words = line.strip().split()
		return False if len(words) == 0 or words[0] != '</instance>' else True

	@staticmethod
	def filterPunctuation(words):
		for i in range(len(words)):
			if len(words[i]) > 0 and words[i][-1] in string.punctuation:
				words[i] = words[i][:-1]

class NB:
	def __init__(self):
		self.initCounts()

	def initCounts(self):
		self.sense_count = dict()
		self.word_present_given_sense = dict()

	def fit(self, instances):
		self.initCounts()
		dictionary = set()
		for ins in instances:
			self.plusSenseCount(ins.sense)
			for w in ins.context:
				self.plusWordPresent(w, ins.sense)
				dictionary.add(w)
		self.setWordCountForSense(dictionary)

	def plusSenseCount(self, sense):
		if sense not in self.sense_count:
			self.sense_count[sense] = 1
		else:
			self.sense_count[sense] += 1

	def plusWordPresent(self, w, sense):
		if sense not in self.word_present_given_sense:
			self.word_present_given_sense[sense] = dict()
		if w not in self.word_present_given_sense[sense]:
			self.word_present_given_sense[sense][w] = 1
		else:
			self.word_present_given_sense[sense][w] += 1

	def setWordCountForSense(self, dictionary):
		num_of_words_in_sense = list([len(self.word_present_given_sense[sense]) for sense in self.word_present_given_sense])
		self.word_count = dict()
		for sense in self.word_present_given_sense:
			if max(num_of_words_in_sense) > SMOOTHING_PARA * min(num_of_words_in_sense):
				self.word_count[sense] = len(dictionary)
			else:
				self.word_count[sense] = len(self.word_present_given_sense[sense])

	def predict(self, instances):
		predict_senses = []
		for ins in instances:
			predict_sense = self.argMaxSense(ins)
			predict_senses.append(predict_sense)
		return predict_senses

	def argMaxSense(self, ins):
		argmax = dict()
		for sense in self.sense_count:
			log_prob = self.computeLogProb(ins, sense)
			argmax[sense] = log_prob
		predict_sense = max(argmax, key=lambda key: argmax[key])
		return predict_sense

	def computeLogProb(self, ins, sense):
		log_p_s = math.log(float(self.sense_count[sense]) / sum([self.sense_count[i] for i in self.sense_count]))
		log_p_f_given_s = 0
		denominator = self.sense_count[sense] + self.word_count[sense]
		for w in ins.context:
			if w in self.word_present_given_sense[sense]:
				numerator = float(self.word_present_given_sense[sense][w]) + 1
			else:
				numerator =  float(1)
			log_p_f_given_s += math.log(numerator / denominator)
		return log_p_s + log_p_f_given_s

	def evaluate(self, predict_senses, test_instances, k):
		length = len(test_instances)
		true_senses = list([test_instances[i].sense for i in range(length)])
		loss = sum([1 if true_senses[i] != predict_senses[i] else 0 for i in range(length)])
		acc = (1 - float(loss) / length) * 100
		acc = float("{0:.2f}".format(acc))
		# self.printWrong(predict_senses, test_instances)
		print('Loss: ' + str(loss))
		print('Accuracy: ' + str(acc) + '%')
		return acc

	def printWrong(self, predict_senses, test_instances):
		length = len(test_instances)
		true_senses = list([test_instances[i].sense for i in range(length)])
		for i in range(length):
			if true_senses[i] != predict_senses[i]:
				print(test_instances[i].id)

def run(word, in_filename, out_filename):
	instances = getData(word, in_filename)
	out_file = open(out_filename, 'w')
	result = list()
	for i in range(FOLDS):
		acc = cv(instances, i, out_file)
		result.append(acc)
	out_file.close()
	print('==== Average Testing Accuracy ====')
	mean = float(sum(result)) / len(result)
	print(str(mean) + '%')

def getData(word, filename):
	with open(filename, 'r') as f:
		lineInOneInstance = False
		lines = list()
		instances = []
		for line in f:
			if not lineInOneInstance and Instance.isInstance(line):
				lineInOneInstance = True
				lines.append(line)
			elif Instance.isEndInstance(line):
				lineInOneInstance = False
				instances.append(Instance(word, lines))
				lines = list()
			elif lineInOneInstance:
				lines.append(line)
	return instances

def cv(instances, k, out_file):
	print('==== Folds: ' + str(k + 1) + ' ====')
	train_idx, test_idx = list(), list()
	getIdx(len(instances), k, train_idx, test_idx)
	nb = NB()
	train_instances = list([instances[i] for i in train_idx])
	test_instances = list([instances[i] for i in test_idx])
	nb.fit(train_instances)
	predict_senses = nb.predict(train_instances)
	predict_senses = nb.predict(test_instances)
	acc = nb.evaluate(predict_senses, test_instances, k)
	writeFile(predict_senses, test_instances, k, out_file)
	return acc

def getIdx(length, k, train_idx, test_idx):
	i = int(math.ceil(float(length) / FOLDS))
	idx = list(range(length))
	train_idx.extend(idx[0: i * k])
	if k < FOLDS - 1:
		train_idx = train_idx.extend(idx[i * (k + 1):])
		test_idx.extend(idx[i * k: i * (k + 1)])
	else:
		test_idx.extend(idx[i * k:])

def writeFile(predict_senses, instances, k, out_file):
	out_file.write('Fold ' + str(k) + '\n')
	for ins, sense in zip(instances, predict_senses):
		s = ins.id + ' ' + ins.word + '%' + sense + '\n'
		out_file.write(s)

if __name__ == '__main__':
	in_filename = sys.argv[1]
	word = in_filename.split('.')[0]
	out_filename = word + OUT_FILE_TYPE
	run(word, in_filename, out_filename)