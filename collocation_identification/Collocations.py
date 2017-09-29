# Author: Chien-Wei Lin
import sys
import math
import string
import operator

K = 20
discard = 5

def buildTable(input_file):
	f = open(input_file, 'r')
	dictionary = set()
	for line in f:
		filter_line = list(filter(lambda w: w not in string.punctuation, line.split()))
		for w in filter_line:
			dictionary.add(w)
	count = 0
	word_idx = {}
	for w in dictionary:
		word_idx[w] = count
		count += 1
	bigram = [[0 for j in range(count)] for i in range(count)]
	f.close()
	return bigram, word_idx, dictionary

def filterBigramlessThan(bigram, discard):
	for w1 in range(len(bigram)):
		for w2 in range(len(bigram[w1])):
			if bigram[w1][w2] < discard:
				bigram[w1][w2] = 0

def biGram(input_file):
	f = open(input_file, 'r')
	bigram, word_idx, dictionary = buildTable(input_file)
	for line in f:
		filter_line = list(filter(lambda w: w not in string.punctuation, line.split()))
		for w1, w2 in zip(filter_line[:-1], filter_line[1:]):
			bigram[word_idx[w1]][word_idx[w2]] += 1
	f.close()
	filterBigramlessThan(bigram, discard)
	bi_wc = sum([sum(w) for w in bigram])
	return bigram, bi_wc, word_idx, dictionary

def uniGram(input_file):
	f = open(input_file, 'r')
	unigram = {}
	wc = 0
	for line in f:
		for word in line.split():
			if word not in string.punctuation:
				unigram[word] = unigram[word] + 1 if word in unigram else 1
				wc += 1
	f.close()
	return unigram, wc

def getExpectValue(bi_wc, o):
	e = [0 for i in range(4)]
	e[0] = (o[0] + o[1]) * (o[0] + o[2]) / bi_wc
	e[1] = (o[0] + o[1]) * (o[1] + o[3]) / bi_wc
	e[2] = (o[2] + o[3]) * (o[0] + o[2])  / bi_wc
	e[3] = (o[2] + o[3]) * (o[1] + o[3]) / bi_wc
	return e

def getObserveValue(w1, w2, bi_wc, bigram):
	o = [0 for i in range(4)]
	o[0] = float(bigram[w1][w2])
	o[1] = float(sum(bigram[w1]) - o[0])
	o[2] = float(sum([bigram[w][w2] for w in range(len(bigram))]) - o[0])
	o[3] = float(bi_wc - sum(o[0:3]))
	return o

def getXSquare(w1, w2, bi_wc, bigram):
	o = getObserveValue(w1, w2, bi_wc, bigram)
	e = getExpectValue(bi_wc, o)
	x = 0
	for i in range(len(e)):
		x += (o[i] - e[i])**2 / e[i]
	del e
	del o
	return x

def insertTopK(result, pair, value, k):
	if len(result) > k:
		sorted_dict = sorted(result.items(), key=operator.itemgetter(1))
		if result[sorted_dict[0][0]] < value:
			del result[sorted_dict[0][0]]
			result[pair] = value
	else:
		result[pair] = value

def chi_square(bigram, bi_wc, word_idx, dictionary):
	result = {}
	for w1 in dictionary:
		for w2 in dictionary:
			if w1 != w2 and bigram[word_idx[w1]][word_idx[w2]] >= 5:
				x = getXSquare(word_idx[w1], word_idx[w2], bi_wc, bigram)
				insertTopK(result, (w1, w2), x, K)
	print('==== chi_square ====')
	printResult(result)

def getMutualInfo(w1, w2, bigram, bi_wc, word_idx, unigram, wc):
	p_w1 = float(unigram[w1]) / wc 
	p_w2 = float(unigram[w2]) / wc
	p_w1w2 = float(bigram[word_idx[w1]][word_idx[w2]])
	p_w1w2 = p_w1w2 / bi_wc
	pmi = math.log(p_w1w2 / (p_w1*p_w2), 2)
	return pmi

def PMI(bigram, bi_wc, word_idx, dictionary, unigram, wc):
	result = {}
	for w1 in dictionary:
		for w2 in dictionary:
			if w1 != w2 and bigram[word_idx[w1]][word_idx[w2]] >= 5:
				pmi = getMutualInfo(w1, w2, bigram, bi_wc, word_idx, unigram, wc)
				if pmi is not None:
					insertTopK(result, (w1, w2), pmi, K)
	print('==== PMI ====')
	printResult(result)

def printResult(result):
	sorted_result = sorted(result.items(), key=operator.itemgetter(1))
	for (w1, w2), value in sorted_result[::-1]:
		print(w1 + ' ' + w2 + ' ' + str(value))

if __name__ == '__main__':
	input_file = sys.argv[1]
	measure = sys.argv[2]
	bigram, bi_wc, word_idx, dictionary = biGram(input_file)
	if measure == 'chi-square':
		chi_square(bigram, bi_wc, word_idx, dictionary)
	elif measure == 'PMI':
		unigram, wc = uniGram(input_file)
		PMI(bigram, bi_wc, word_idx, dictionary, unigram, wc)
	else:
		print('No such measures')
		raise 