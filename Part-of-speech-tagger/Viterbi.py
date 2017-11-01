# Auther: Chien-Wei Lin

START_OF_SENTENCE = '<s>'
START_OF_SENTENCE_INSTANCE = START_OF_SENTENCE + '/' + START_OF_SENTENCE
OUTPUT_FILE = 'POS.test.out'

def printEvaluation(word_count, loss)
	print str(word_count) + ' words'
	print 'Loss: ' + str(loss)
	print 'Accuracy: ' + str(1 - float(loss) / word_count)

class Dictionary:
	def __init__(self, filename):
		self.createDictionary(filename)
		self.createFromWord2idx()
		self.createFromTag2idx()
		self.createFromIdx2Tag()

	def createDictionary(self, filename):
		self.dic = set()
		self.pos_tags = set()
		with open(filename, 'r') as f:
			for line in f:
				line = line.strip()
				words = list([instance.split('/')[0] for instance in line.split()])
				tags = list([instance.split('/')[1] for instance in line.split()])
				self.dic.update(words)
				self.pos_tags.update(tags)
		self.addStartOfSentenceToDic()
		self.dic = list(self.dic)
		self.pos_tags = list(self.pos_tags)

	def createFromWord2idx(self):
		self.from_word2idx = dict()
		idx = 0
		for word in self.dic:
			self.from_word2idx[word] = idx
			idx += 1

	def createFromTag2idx(self):
		self.from_tag2idx = dict()
		idx = 0
		for tag in self.pos_tags:
			self.from_tag2idx[tag] = idx
			idx += 1

	def addStartOfSentenceToDic(self):
		self.dic.add(START_OF_SENTENCE)
		self.pos_tags.add(START_OF_SENTENCE)

	def createFromIdx2Tag(self):
		self.from_idx2tag = dict()
		for tag in self.pos_tags:
			idx = self.from_tag2idx[tag]
			self.from_idx2tag[idx] = tag

	def createFromIdx2Word(self):
		self.from_idx2word = dict()
		for word in self.dic:
			idx = self.from_word2idx[word]
			self.from_idx2word[idx] = word

	def getTags(self):
		return self.pos_tags

	def getWords(self):
		return self.dic

	def getTagIdx(self, tag):
		return self.from_tag2idx[tag]

	def getWordIdx(self, word):
		return self.from_word2idx[word]

	def getWordFromIdx(self, idx):
		return self.from_idx2word[idx]

	def getTagFromIdx(self, idx):
		return self.from_idx2tag[idx]

class Bigram:
	def __init__(self, filename):
		self.my_dic = Dictionary(filename)
		self.initBigram()
		self.constructBigram(filename)

	def initBigram(self):
		num_of_tags = len(self.my_dic.getTags())
		num_of_words = len(self.my_dic.getWords())
		self.tag_bigram = [[0 for j in range(num_of_tags)] for i in range(num_of_tags)]
		self.word_given_tag_bigram = [[0 for j in range(num_of_words)] for i in range(num_of_tags)]

	def constructBigram(self, filename):
		with open(filename, 'r') as f:
			for line in f:
				instances = list([START_OF_SENTENCE_INSTANCE]) + line.strip().split()
				for pre_ins, next_ins in zip(instances[:-1], instances[1:]):
					pre_w, pre_t = pre_ins.split('/')
					next_w, next_t = next_ins.split('/')
					self.tagsBigramPlus1(pre_t, next_t)
					self.wordGivenTagBigramPlus1(next_w, next_t)
					self.normalizeBigramToProb()

	def tagsBigramPlus1(self, pre_tag, next_tag):
		pre_tag_idx = self.my_dic.getTagIdx(pre_tag)
		next_tag_idx = self.my_dic.getTagIdx(next_tag)
		self.tag_bigram[pre_tag_idx][next_tag_idx] += 1

	def wordGivenTagBigramPlus1(self, word, tag):
		word_idx = self.my_dic.getWordIdx(word)
		tag_idx = self.my_dic.getTagIdx(tag)
		self.word_given_tag_bigram[tag_idx][word_idx] += 1

	def normalizeBigramToProb(self):
		number_of_tags = len(self.my_dic.getTags())
		for pre_idx in range(len(self.number_of_tags)):
			tag_count = sum(self.tag_bigram[pre_idx])
			for next_idx in range(number_of_tags):
				self.tag_bigram[pre_idx][next_idx] = self.tag_bigram[pre_idx][next_idx] / tag_count

class Frequency(Trainer):
	def __init__(self, filename):
		self.my_dic = Dictionary(filename)
		self.initWordTagTable()

	def initWordTagTable(self):
		num_of_tags = len(self.my_dic.getTags())
		num_of_words = len(self.my_dic.getWords())
		self.word_tag_table = [[0 for t in range(num_of_tags)] for w in range(num_of_words)]

	def wordTagTablePlus1By(self, word, tag):
		word_idx = self.my_dic.from_word2idx[word]
		tag_idx = self.my_dic.from_tag2idx[tag]
		self.word_tag_table[word_idx][tag_idx] += 1

	def createFromWord2MostFreqTag(self):
		self.from_word_2_most_freq_Tag = dict()
		for word_idx in range(len(self.my_dic.getWords())):
			max_value = max(self.word_tag_table[word_idx])
			tag_idx = self.word_tag_table[word_idx].index(max_value)
			word = self.my_dic.getWordFromIdx(word_idx)
			tag = self.my_dic.getTagFromIdx(tag_idx)
			self.from_word_2_most_freq_Tag[word] = tag

	def train(self, filename):
		with open(filename, 'r') as f:
			for line in f:
				instances = line.strip().split()
				for instance in instances:
					word, tag = instance.split('/')
					self.wordTagTablePlus1By(word, tag)
		self.createFromWord2MostFreqTag()

	def predict(self, filename):
		print '==== Predict from Most frequent tag ===='
		word_count = 0
		loss = 0
		with open(filename, 'r') as f:
			for line in f:
				instances = line.strip().split()
				for instance in instances:
					word, tag = instance.split('/')
					word_count += 1
					if self.from_word_2_most_freq_Tag[word] != tag:
						loss += 1
		printEvaluation(word_count, loss)

class Viterbi(Trainer):
	def __init__(self):
		self.my_dic = Dictionary(filename)

	def train(self, filename):
		self.bigram = Bigram(filename)

	def predict(self, filename):
		print '==== Predict from Viterbi Algorithm ===='
		num_of_total_tags = len(self.my_dic.getTags())
		out_file = open(OUTPUT_FILE, 'w')
		word_count, loss = 0, 0
		with open(filename, 'r') as f:
			for line in f:
				instances = line.strip().split()
				num_of_words = len(instances)
				words = [instance.split('/')[0] for instance in instances]
				tags = [instance.split('/')[1] for instance in instances]
				score, back_ptr = self.iteration(words, num_of_total_tags)
				sequence = self.sequenceIdentification(words, score, back_ptr)
				word_count, loss = self.evaluate(word_count, loss, sequence, tags)
				self.writeFile(out_file, words, sequence)
				self.deleteList(words, tags, score, back_ptr, sequence)
		out_file.close()
		printEvaluation(word_count, loss)

	def evaluate(self, word_count, loss, sequence, tags):
		for i in range(len(tags)):
			word_count += 1
			predict_tag = self.my_dic.getTagFromIdx(sequence[i])
			loss = loss + 1 if tags[i] != predict_tag else loss
		return word_count, loss

	def writeFile(self, out_file, words, tags):
		for i in range(len(words)):
			out_file.write(words[i] + '/' + tags[i] + ' ')
		out_file.write('\n')

	def deleteList(self, words, tags, score, back_ptr, sequence);
		del words, tags, sequence
		for i in score:
			del i
		for i in back_ptr:
			del i
		del score, back_ptr

	def initialization(self, word, number_of_tags, num_of_words):
		score = [[0 for j in range(num_of_words)]for i in range(num_of_tags)]
		back_ptr = [[0 for j in range(num_of_words)]for i in range(num_of_tags)]
		word_idx = self.my_dic.getWordIdx(word)
		for tag_idx in range(num_of_tags):
			p_w_given_t =  self.bigram.word_given_tag_bigram[tag_idx][word_idx]
			start_of_sentece_idx = self.my_dic.getTagIdx(START_OF_SENTENCE)
			p_t_given_start_of_sentence = self.tag_bigram[start_of_sentece_idx][tag_idx]
			score[tag_idx][0] = p_w_given_t * p_t_given_start_of_sentence
		return score, back_ptr

	def iteration(self, words, num_of_tags):
		num_of_words = len(words)
		if num_of_words == 0:
			return
		score, back_ptr = self.initialization(words[0], num_of_tags, num_of_words)
		for i_th_word in range(1, num_of_words):
			word_idx = self.my_dic.getWordIdx(words[i_th_word])
			for tag_idx in range(num_of_tags):
				p_w_given_t =  self.bigram.word_given_tag_bigram[tag_idx][word_idx]
				max_idx, max_value = self.getMaxScoreBigram()
				back_ptr[tag_idx][i_th_word] = max_idx

	def getMaxScoreBigram(self, i_th_word, tag_idx, score, num_of_tags):
		max_idx, max_value = 0, -1
		for j in range(num_of_tags):
			p_Tag_idx_given_j = self.tag_bigram[j][tag_idx]
			value = score[j][i_th_word-1] * p_Tag_idx_given_j
			if max_value <= value:
				max_value = value
				max_idx = j
		return max_idx, max_value

	def sequenceIdentification(self, words, score, back_ptr):
		sequence = [0 for i in range(len(words))]
		sequence[-1] = self.getMaxTagFromScore(score)
		for w in range(len(words)-1):
			idx = len(words) - 1 - w
			sequence[idx] = back_ptr[sequence[idx+1]][idx+1]
		return sequence

	def getMaxTagFromScore(self, score):
		opt_t = 0
		max_value = -1
		for i in range(len(score)):
			if score[i][-1] >= max_value:
				max_value = score[i][-1]
				opt_t = i
		return opt_t

class Trainer:
	def __init__(self):

	def train(self, filename):
		pass

	def predict(self):
		pass

	@staticmethod
	def trainAndPredict(algo, train_file, test_file):
		algo.train(train_file)
		algo.predict(test_file)

if __name__ == '__main__':
	train_file = sys.argv[1]
	test_file = sys.argv[2]
	Trainer.trainAndPredict(Frequency(train_file), train_file, test_file)
	Trainer.trainAndPredict(Viterbi(train_file), train_file, test_file)