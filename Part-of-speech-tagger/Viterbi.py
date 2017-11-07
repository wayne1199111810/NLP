# Author: Chien-Wei Lin (chienwli)
import sys
import numpy as np

START_OF_SENTENCE = '<s>'
START_OF_SENTENCE_INSTANCE = START_OF_SENTENCE + '/' + START_OF_SENTENCE
OUTPUT_FILE = 'POS.test.out'
unk = ''

def printEvaluation(word_count, loss):
    print str(word_count) + ' words'
    print 'Loss: ' + str(loss)
    print 'Accuracy: ' + str(1 - float(loss) / word_count)

class Dictionary:
    def __init__(self, filename):
        self.createDictionary(filename)
        self.createFromWord2idx()
        self.createFromTag2idx()
        self.createFromIdx2Tag()
        self.createFromIdx2Word()

    def createDictionary(self, filename):
        self.dic = set()
        self.pos_tags = set()
        with open(filename, 'r') as f:
            for line in f:
                line = line.strip()
                words = list([instance[0:instance.rfind('/')] for instance in line.split()])
                tags = list([instance[instance.rfind('/')+1:] for instance in line.split()])
                self.dic.update(words)
                self.pos_tags.update(tags)
        self.addStartOfSentenceToDic()
        self.dic.add(unk)
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
        if word not in self.dic:
            return self.from_word2idx[unk]
        return self.from_word2idx[word]

    def getWordFromIdx(self, idx):
        return self.from_idx2word[idx]

    def getTagFromIdx(self, idx):
        return self.from_idx2tag[idx]

class Trainer:
    def __init__(self):
        pass

    def train(self, filename):
        pass

    def predict(self):
        pass

    @staticmethod
    def trainAndPredict(algo, train_file, test_file):
        algo.train(train_file)
        algo.predict(test_file)

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
        self.most_freq_tag = self.getMostAppearTag()

    def getMostAppearTag(self):
        num_of_tags = len(self.my_dic.getTags())
        num_of_words = len(self.my_dic.getWords())
        tags_count = list([sum([self.word_tag_table[j][i] for j in range(num_of_words)]) for i in range(num_of_tags)])
        max_value = max(tags_count)
        tag_idx = tags_count.index(max_value)
        return self.my_dic.getTagFromIdx(tag_idx)

    def train(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                instances = line.strip().split()
                for instance in instances:
                    word, tag = instance[0:instance.rfind('/')], instance[instance.rfind('/')+1:]
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
                    word, tag = instance[0:instance.rfind('/')], instance[instance.rfind('/')+1:]
                    word_count += 1
                    if word not in self.from_word_2_most_freq_Tag:
                        if tag != self.most_freq_tag:
                            loss += 1
                    elif self.from_word_2_most_freq_Tag[word] != tag:
                        loss += 1
        printEvaluation(word_count, loss)

class Bigram:
    def __init__(self, dic, filename):
        self.my_dic = dic
        self.initBigram()
        print 'create bigram'
        self.constructBigram(filename)

    def initBigram(self):
        num_of_tags = len(self.my_dic.getTags())
        num_of_words = len(self.my_dic.getWords())
        self.tag_bigram = np.zeros((num_of_tags, num_of_tags))
        self.word_given_tag_bigram = np.zeros((num_of_tags, num_of_words))

    def constructBigram(self, filename):
        with open(filename, 'r') as f:
            for line in f:
                instances = list([START_OF_SENTENCE_INSTANCE]) + line.strip().split()
                for pre_ins, next_ins in zip(instances[:-1], instances[1:]):
                    pre_t = pre_ins[pre_ins.rfind('/')+1:]
                    next_w, next_t = next_ins[0:next_ins.rfind('/')], next_ins[next_ins.rfind('/')+1:]
                    self.tagsBigramPlus1(pre_t, next_t)
                    self.wordGivenTagBigramPlus1(next_w, next_t)
        self.laplaceSmooth()

    def tagsBigramPlus1(self, pre_tag, next_tag):
        pre_tag_idx = self.my_dic.getTagIdx(pre_tag)
        next_tag_idx = self.my_dic.getTagIdx(next_tag)
        self.tag_bigram[pre_tag_idx][next_tag_idx] += 1

    def wordGivenTagBigramPlus1(self, word, tag):
        word_idx = self.my_dic.getWordIdx(word)
        tag_idx = self.my_dic.getTagIdx(tag)
        self.word_given_tag_bigram[tag_idx][word_idx] += 1

    def laplaceSmooth(self):
        num_of_tags = len(self.my_dic.getTags())
        num_of_words = len(self.my_dic.getWords())
        count = np.sum(self.tag_bigram, axis=1) + num_of_tags
        self.tag_bigram = (self.tag_bigram + 1) / count[:,None]
        count = np.sum(self.word_given_tag_bigram, axis=1) + num_of_words
        self.word_given_tag_bigram = (self.word_given_tag_bigram + 1) / count[:,None]
        print 'size of tag_bigram:' + str(self.tag_bigram.shape)
        print 'size of word given tag bigram:' + str(self.word_given_tag_bigram.shape)

class Viterbi(Trainer):
    def __init__(self, filename):
        self.my_dic = Dictionary(filename)

    def train(self, filename):
        self.bigram = Bigram(self.my_dic, filename)

    def predict(self, filename):
        print '==== Predict from Viterbi Algorithm ===='
        num_of_tags = len(self.my_dic.getTags())
        out_file = open(OUTPUT_FILE, 'w')
        word_count, loss, unk_count = 0, 0, 0
        line_num = 0
        with open(filename, 'r') as f:
            for line in f:
                line_num += 1
                if line_num % 50 == 0:
                    print str(line_num / 5) + '%'
                instances = line.strip().split()
                num_of_words = len(instances)
                filter_words = list([instance[0:instance.rfind('/')] for instance in instances])
                unk_count += self.replaceUnknownWithUnk(filter_words)
                tags = list([instance[instance.rfind('/')+1:] for instance in instances])
                score = np.zeros((num_of_tags, num_of_words))
                back_ptr = np.zeros((num_of_tags, num_of_words))
                sequence = np.zeros((num_of_words))
                self.iteration(filter_words, num_of_tags, score, back_ptr)
                self.sequenceIdentification(filter_words, score, back_ptr, sequence)
                word_count, loss = self.evaluate(word_count, loss, sequence, tags)
                origin_words = list([instance[0:instance.rfind('/')] for instance in instances])
                self.writeFile(out_file, origin_words, sequence)
        print 'number of unk words:' + str(unk_count)
        out_file.close()
        printEvaluation(word_count, loss)

    def replaceUnknownWithUnk(self, words):
        count = 0
        for i in range(len(words)):
            if words[i] not in self.my_dic.dic:
                count += 1
                words[i] = unk
        return count

    def evaluate(self, word_count, loss, sequence, tags):
        for i in range(len(tags)):
            word_count += 1
            predict_tag = self.my_dic.getTagFromIdx(sequence[i])
            loss = loss + 1 if tags[i] != predict_tag else loss
        return word_count, loss

    def writeFile(self, out_file, words, sequence):
        for i in range(len(words)):
            tag = self.my_dic.getTagFromIdx(sequence[i])
            out_file.write(words[i] + '/' + tag + ' ')
        out_file.write('\n')

    def initialization(self, word, num_of_tags, score):
        word_idx = self.my_dic.getWordIdx(word)
        for tag_idx in range(num_of_tags):
            p_w_given_t =  self.bigram.word_given_tag_bigram[tag_idx][word_idx]
            start_of_sentece_idx = self.my_dic.getTagIdx(START_OF_SENTENCE)
            p_t_given_start_of_sentence = self.bigram.tag_bigram[start_of_sentece_idx][tag_idx]
            score[tag_idx][0] = p_w_given_t * p_t_given_start_of_sentence

    def iteration(self, words, num_of_tags, score, back_ptr):
        num_of_words = len(words)
        self.initialization(words[0], num_of_tags, score)
        for i_th_word in range(1, num_of_words):
            word_idx = self.my_dic.getWordIdx(words[i_th_word])
            for tag_idx in range(num_of_tags):
                p_w_given_t =  self.bigram.word_given_tag_bigram[tag_idx][word_idx]
                max_idx, max_value = self.getMaxScoreBigram(i_th_word, tag_idx, score, num_of_tags)
                score[tag_idx][i_th_word] = p_w_given_t * max_value
                back_ptr[tag_idx][i_th_word] = max_idx
        return score, back_ptr

    def getMaxScoreBigram(self, i_th_word, tag_idx, score, num_of_tags):
        max_idx, max_value = 0, -1
        for j in range(num_of_tags):
            p_Tag_idx_given_j = self.bigram.tag_bigram[j][tag_idx]
            value = score[j][i_th_word-1] * p_Tag_idx_given_j
            if max_value <= value:
                max_value = value
                max_idx = j
        return max_idx, max_value

    def sequenceIdentification(self, words, score, back_ptr, sequence):
        sequence[-1] = np.argmax(score, axis = 0)[-1]
        for w in range(len(words)-1):
            idx = len(words) - 2 - w
            sequence[idx] = back_ptr[int(sequence[idx+1])][idx+1]
        return sequence

if __name__ == '__main__':
    train_file = sys.argv[1]
    test_file = sys.argv[2]
    Trainer.trainAndPredict(Frequency(train_file), train_file, test_file)
    Trainer.trainAndPredict(Viterbi(train_file), train_file, test_file)