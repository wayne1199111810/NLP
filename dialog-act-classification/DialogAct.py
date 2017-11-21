import sys
import string
import math

OUT_FILE_TYPE = '.out'
ADVISOR_HEAD = 'Advisor:'
enc = 'utf-8'

class Instance:
	def __init__(self, line):
		line = line.strip()
		self.getAct(line)
		self.getContext(line)

	def getAct(self, line):
		act = line.split()[1]
		self.act = act[1:-1].strip()

	def getContext(self, line):
		body = Instance.getBody(line)
		self.body = body
		context = Instance.addSpace2Punctuation(body)
		self.context = set(context)

	@staticmethod
	def addSpace2Punctuation(body):
		modified_body = ''
		for i in range(len(body)):
			if body[i] in string.punctuation and i + 1 < len(body) and body[i + 1].isspace():
				modified_body += ' ' + body[i]
			elif body[i] in string.punctuation and i + 1 == len(body):
				modified_body += ' ' + body[i]
			else:
				modified_body += body[i].lower()
		return modified_body.strip().split()

	@staticmethod
	def getBody(line):
		idx = line.find(']')
		return line[idx + 1:].strip()

class NB:
	def __init__(self):
		self.initCounts()

	def initCounts(self):
		self.word_count = dict()
		self.act_count = dict()
		self.word_present_given_act = dict()
		self.dictionary = set()

	def fit(self, instances):
		self.initCounts()
		for ins in instances:
			self.plusActCount(ins.act)
			for w in ins.context:
				self.dictionary.add(w)
				self.plusWordPresent(w, ins.act)
		self.setWordCountForAct()

	def plusActCount(self, act):
		if act not in self.act_count:
			self.act_count[act] = 1
		else:
			self.act_count[act] += 1

	def plusWordPresent(self, w, act):
		if act not in self.word_present_given_act:
			self.word_present_given_act[act] = dict()
		if w not in self.word_present_given_act[act]:
			self.word_present_given_act[act][w] = 1
		else:
			self.word_present_given_act[act][w] += 1

	def setWordCountForAct(self):
		for act in self.word_present_given_act:
			self.word_count[act] = len(self.word_present_given_act[act])

	def predict(self, instances):
		predict_acts = []
		for ins in instances:
			predict_act = self.argMaxAct(ins)
			predict_acts.append(predict_act)
		return predict_acts

	def argMaxAct(self, ins):
		argmax = dict()
		for act in self.act_count:
			log_prob = self.computeLogProb(ins, act)
			argmax[act] = log_prob
		predict_act = max(argmax, key=lambda key: argmax[key])
		return predict_act

	def computeLogProb(self, ins, act):
		log_p_act = math.log(float(self.act_count[act]) / sum([self.act_count[i] for i in self.act_count]))
		log_p_f_given_act = 0
		denominator = self.act_count[act] + len(self.dictionary)
		for w in ins.context:
			if w in self.word_present_given_act[act]:
				numerator = float(self.word_present_given_act[act][w]) + 1
			else:
				numerator =  float(1)
			log_p_f_given_act += math.log(numerator / denominator)
		return log_p_act + log_p_f_given_act

	def evaluate(self, predict_acts, test_instances):
		length = len(test_instances)
		true_acts = list([test_instances[i].act for i in range(length)])
		loss = sum([1 if true_acts[i] != predict_acts[i] else 0 for i in range(length)])
		acc = (1 - float(loss) / length) * 100
		acc = float("{0:.2f}".format(acc))
		print('Loss: ' + str(loss))
		print('Accuracy: ' + str(acc) + '%')
		# self.printWrong(predict_acts, test_instances)

	def printWrong(self, predict_acts, test_instances):
		length = len(test_instances)
		true_acts = list([test_instances[i].act for i in range(length)])
		for i in range(length):
			if true_acts[i] != predict_acts[i]:
				print('===============')
				print('Should be [' + true_acts[i] + '] , but predict as [' + predict_acts[i] + ']')
				print(test_instances[i].body)
				print(test_instances[i].context)


def run(train_file, test_file, out_filename):
	train_ins = getData(train_file)
	test_ins = getData(test_file)
	nb = NB()
	nb.fit(train_ins)

	print('==== test ====')
	predict_acts = nb.predict(test_ins)
	nb.evaluate(predict_acts, test_ins)
	writeFile(test_file, predict_acts, out_filename)

def getData(filename):
	f = open(filename, 'r')
	instances = []

	for line in f:
		if lineIsAdvisor(line):
			instances.append(Instance(line))
	return instances

def lineIsAdvisor(line):
	line = line.strip()
	words = line.split(':')
	advisor = 'Advisor'
	return False if len(words) == 0 or words[0] != advisor else True

def writeFile(in_filename, predict_acts, out_filename):
	out_file = open(out_filename, 'w')
	in_file = open(in_filename, 'r')
	advisor_count = 0
	for line in in_file:
		if lineIsAdvisor(line):
			body = Instance.getBody(line)
			s = 'Advisor: [' + predict_acts[advisor_count] + ']' + ' ' + body + '\n'
			advisor_count += 1
		else:
			s = line
		out_file.write(s)
	out_file.close()
	in_file.close()

if __name__ == '__main__':
	train_file = sys.argv[1]
	test_file = sys.argv[2]
	out_filename = test_file + OUT_FILE_TYPE
	run(train_file, test_file, out_filename)