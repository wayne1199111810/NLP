# Author: Chien-Wei Lin

import sys
import numpy as np
from sklearn import preprocessing
import json
import random
import io

unk = ''
CHARLENGTH = 5
SEED = 546123

hyperparameter = {
    'lr' : 0.1,
    'retain_rate': 0.5,
    'batch_size': 1, 
    'epoch': 20,
    'width': 100
}

def sgm(v, derivative = False):
    y = 1 / (1 + np.exp(-v))
    if not derivative:
        return y
    else:
        return v * (1 - v)

def Kronecker(i, j):
    return 1 if i == j else 0

def softmax(y, derivative = False, e=None):
    if not derivative:
        ey = np.exp(y)
        out = ey / np.array([np.sum(ey, axis=0)])
    else:
        out = np.zeros(y.shape)
        for k in range(y.shape[1]):
            for j in range(y.shape[0]):
                for i in range(y.shape[0]):
                    out[j][k] += e[i][k] * y[i][k] * (Kronecker(i, j) - y[j][k])
    return out

def filterCharsUnkonw(s, dic, unk):
    for i in range(len(s)):
        if s[i] not in dic:
            s[i] = unk

def filterLanguageUnkonw(l, dic, unk):
    for i in range(len(l)):
        if l[i] not in dic:
            l[i] = unk

def dataPreprocessing(train_file, dev_file):
    print('dataPreprocessing')
    feature_obj = Feature(train_file)

    # train_features, train_targets = feature_obj.extractTrainFeature(train_file)
    train_features, train_targets = feature_obj.extractTrainLines(train_file)
    dev_features, dev_targets = None, None
    dev_features, dev_targets = feature_obj.extractTrainLines(dev_file)
    return feature_obj, train_features, train_targets, dev_features, dev_targets

def retrieveData():
    with open('feature_obj', 'r') as f:
        feature_obj = json.load(f)
    with open('train_data', 'r') as f:
        data = json.load(f)
        train_features, train_targets = data['features'], data['targets']
    with open('dev_data', 'r') as f:
        data = json.load(f)
        dev_features, dev_targets = data['features'], data['targets']
    with open('test_data', 'r') as f:
        data = json.load(f)
        test_features = data['features']

    hyperparameter['shape'] = [CHARLENGTH*len(self.dic), hyperparameter['width'], len(self.languages)]
    return feature_obj, train_features, train_targets, dev_features, dev_targets, test_features

class OneHotEncoder:
    def __init__(self, dic, filterUnkonwn, unk=unk):
        self.dic = dic
        self.unk = unk
        self.initEncoder()
        self.initOneHot()
        self.filterUnkonwn = filterUnkonwn

    def initEncoder(self):
        self.encoder = preprocessing.LabelEncoder()
        self.encoder.fit(self.dic)

    def initOneHot(self):
        self.onehot = preprocessing.OneHotEncoder()
        encode_labels = self.encoder.transform(self.dic)
        self.onehot.fit([[w] for w in encode_labels])

    def transform(self, s):
        self.filterUnkonwn(s, self.dic, self.unk)
        encode_labels = [[c] for c in self.encoder.transform(s)]
        return self.onehot.transform(encode_labels).toarray()

class Feature:
    def __init__(self, dataset):
        self.createDictionary(dataset)
        self.initCharOneHotEncoder()
        self.initLanguageOneHotEncoder()
        hyperparameter['shape'] = [CHARLENGTH*len(self.dic), hyperparameter['width'], len(self.languages)]

    def createDictionary(self, file_name):
        f = io.open(file_name, mode="r")
        self.dic = set()
        self.languages = set()
        for line in f:
            # line = line.strip()
            idx = line.find(' ')
            for i in range(idx, len(line)):
                self.dic.add(line[i])
            self.languages.add(line[0:idx].strip())
        self.dic.add(unk)
        self.dic = list(self.dic)
        self.languages = list(self.languages)
        print('dictionary:' + str(len(self.dic)))
        print('languages: ' + str(len(self.languages)))
        f.close()

    def initCharOneHotEncoder(self):
        self.chars_encoder = OneHotEncoder(self.dic, filterCharsUnkonw)

    def initLanguageOneHotEncoder(self):
        self.language_encoder = OneHotEncoder(self.languages, filterLanguageUnkonw)

    def chars2Vec(self, string):
        vecs = self.chars_encoder.transform([c for c in string])
        out = []
        for vec in vecs:
            out.extend(vec)
        return out

    def language2Vec(self, language):
        return self.language_encoder.transform([language]).tolist()[0]

    def array2Language(self, array):
        return self.language_encoder.encoder.inverse_transform([np.argmax(array)])[0]

    @staticmethod
    def getCharsFromIdx(s, idx):
        if len(s) > CHARLENGTH + idx:
            return [s[i] for i in range(idx, idx+CHARLENGTH)]
        else:
            tmp = [s[i] for i in range(idx, len(s))]
            tmp.extend([unk for i in range(idx+CHARLENGTH-len(s))])
            return tmp

    # def extractTrainFeature(self, file_name):
    #     features = []
    #     targets = []
    #     f = io.open(file_name, mode="r")
    #     for line in f:
    #         idx = line.find(' ')
    #         language = line[0:idx].strip()
    #         for i in range(idx, len(line)):
    #             chars = Feature.getCharsFromIdx(line, i)
    #             features.append(self.chars2Vec(chars))
    #             targets.append(self.language2Vec(language))
    #     self.randomize(features, targets)
    #     f.close()
    #     features, targets = np.array(features), np.array(targets)
    #     return features, targets

    def extractTrainLines(self, file_name):
        raw_features = []
        raw_targets = []
        f = io.open(file_name, mode="r")
        for line in f:
            idx = line.find(' ')
            raw_features.append(line[idx:].strip())
            raw_targets.append(line[0:idx].strip())
        f.close()
        self.randomize(raw_features, raw_targets)
        return raw_features, raw_targets

    def extractTestFeature(self, test_feature_file):
        features = []
        f_feature = open(test_feature_file, mode="r")
        for line, in f_feature:
            # line = line.strip()
            for i in range(len(line)-CHARLENGTH):
                chars = Feature.getCharsFromIdx(line, i)
                features.append(self.chars2Vec(chars))
        f_feature.close()
        return features

    def randomize(self, features, targets):
        random.seed(SEED)
        random.shuffle(features)
        random.shuffle(targets)

class Networks:
    def __init__(self, outlayer=softmax, act=sgm, recover=False):
        self.initActivation(act)
        self.softmax = outlayer
        if not recover:
            self.initPara()
            self.initWeights()
        else:
            self.loadNetwork()

    def initPara(self):
        self.lr = hyperparameter['lr']
        self.retain_rate = hyperparameter['retain_rate']
        self.batch_size = hyperparameter['batch_size']
        self.epoch = hyperparameter['epoch']
        self.width = hyperparameter['width']
        self.shape = hyperparameter['shape']
        self.y, self.deltas_w, self.deltas_b = None, None, None

    def initActivation(self, act):
        self.activation = act

    def initWeights(self):
        self.weights = [np.random.randn(j, i) for i, j in zip(self.shape[:-1], self.shape[1:])]
        self.bias = [np.random.randn(i, 1) for i in self.shape[1:]]

    def initY(self, size=None):
        del self.y
        self.y = [np.zeros((i, size)) for i in self.shape[1:]] if size else []

    def initDelta(self):
        del self.deltas_w
        del self.deltas_b
        self.deltas_w = []
        self.deltas_b = []

    def feedForward(self, features):
        Y = features.T
        self.initY()
        self.y.append(Y)
        v = np.dot(self.weights[0], Y) + self.bias[0]
        Y = self.activation(v)
        self.y.append(Y)
        Y = np.dot(self.weights[1], Y) + self.bias[1]
        self.y.append(Y)
        self.out = self.softmax(self.y[-1])
        return self.out

    def backPropagate(self, train_features, train_targets):
        # hidden -> output layer
        self.initDelta()
        e = train_targets.T - self.out
        if e.shape[1] == 0:
            return
        delta = -1 * np.array([np.sum(self.softmax(self.out, True, e), axis=1)]).T
        delta_w = np.outer(delta, np.array([np.sum(self.y[-2], axis=1)]).T)

        self.deltas_w.append(delta_w)
        self.deltas_b.append(delta)

        # input -> hidden layer
        dEdy1 = np.dot(delta.T, self.weights[-1])
        delta = dEdy1 * np.array([np.sum(self.activation(self.y[-2], True), axis=1)])
        delta_w = np.outer(delta, np.array([np.sum(self.y[-3], axis=1)]).T)
        
        self.deltas_w.append(delta_w)
        self.deltas_w.reverse()
        self.deltas_b.append(delta.T)
        self.deltas_b.reverse()

        for idx in range(len(self.weights)):
            self.weights[idx] -= self.lr*self.deltas_w[idx]/self.batch_size
            self.bias[idx] -= self.lr*self.deltas_b[idx]/self.batch_size

    def adaptLearnRate(self, dev_features, dev_targets):
        predict_values = self.predict(dev_features)
        loss = self.loss(predict_values, dev_targets)
        # TODO
        accuracy = loss / len(predict_values)
        if self.pre_accuracy >= accuracy:
            self.lr = self.lr / 2

    def trainBatch(self, train_features, train_targets):
        predict_values = self.feedForward(train_features)
        self.backPropagate(train_features, train_targets)

    def train(self, train_features, train_targets, dev_features, dev_targets):
        for i in range(self.epoch):
            print('==== epoch ' + str(i) + ' ====')
            for batch in range(int(len(train_features) / self.batch_size)):
                start = self.batch_size * batch
                end = self.batch_size * (batch + 1)
                self.trainBatch(train_features[start:end], train_targets[start:end])
            start = self.batch_size * (int(len(train_features) / self.batch_size) + 1)
            self.trainBatch(train_features[start:], train_targets[start:])
            # self.adaptLearnRate(dev_features, dev_targets)
            self.evaluate(train_features, train_targets)

    def trainLineByLine(self, train_raw_features, train_raw_targets, dev_raw_features, dev_raw_targets, feature_obj):
        for i in range(self.epoch):
            print('==== epoch ' + str(i) + ' ====')
            for line, language in zip(train_raw_features, train_raw_targets):
                train_features = list()
                train_targets = list()
                for i in range(len(line)-CHARLENGTH):
                    chars = Feature.getCharsFromIdx(line, i)
                    train_features.append(feature_obj.chars2Vec(chars))
                    train_targets.append(feature_obj.language2Vec(language))
                train_features = np.array(train_features)
                train_targets = np.array(train_targets)
                self.trainBatch(train_features, train_targets)
            self.evaluateByLine(train_raw_features, train_raw_targets, feature_obj)

            for line, language in zip(dev_raw_features, dev_raw_targets):
                dev_features = list()
                dev_targets = list()
                for i in range(len(line)-CHARLENGTH):
                    chars = Feature.getCharsFromIdx(line, i)
                    dev_features.append(feature_obj.chars2Vec(chars))
                    dev_targets.append(feature_obj.language2Vec(language))
                dev_features = np.array(dev_features)
                dev_targets = np.array(dev_targets)
            self.evaluateByLine(dev_raw_features, dev_raw_targets, feature_obj)

    def predict(self, features):
        predict_values = self.feedForward(features)
        indices = np.argmax(predict_values, axis=0)
        predict_values = np.zeros(predict_values.shape)
        for i in range(len(indices)):
            predict_values[indices[i]][i] = 1
        return predict_values

    def loss(self, predict_values, targets):
        indices1 = np.array([np.argmax(predict_values, axis=0)])
        indices2 = np.array([np.argmax(targets.T, axis=0)])
        dif = sum([1 if indices1[0][i] != indices2[0][i] else 0 for i in range(indices1.shape[1])])
        return dif

    def evaluate(self, features, targets):
        predict_values = self.predict(features)
        loss = self.loss(predict_values, targets)
        accuracy = 1 - float(loss) / predict_values.shape[1]
        print('==== Evaluation ====')
        print('Loss: ' + str(loss))
        print('Accuracy: ' + str(accuracy))

    def evaluateByLine(self, raw_features, raw_targets, feature_obj):
        loss = 0
        count = 0
        for line, language in zip(raw_features, raw_targets):
            features = list()
            targets = list()
            for i in range(len(line)-CHARLENGTH):
                chars = Feature.getCharsFromIdx(line, i)
                features.append(feature_obj.chars2Vec(chars))
                targets.append(feature_obj.language2Vec(language))
            if len(features) == 0:
                continue
            features = np.array(features)
            targets = np.array(targets) 
            predict_values = self.predict(features)
            count += predict_values.shape[1]
            loss += self.loss(predict_values, targets)
        accuracy = 1 - float(loss) / count
        print('==== Evaluation ====')
        print('Loss: ' + str(loss))
        print('Accuracy: ' + str(accuracy))        

    @staticmethod
    def saveNetwork(file_name, networks):
        net = {
            'weights': networks.weights,
            'lr': networks.lr,
            'retain_rate': networks.retain_rate,
            'batch_size': networks.batch_size,
            'epoch': networks.epoch
        }
        with open(file_name, 'w') as f:
            json.dump(net, f)

    def loadNetwork(self, file_name):
        with open(file_name, 'r') as f:
            net = json.load(f)
        self.lr = net['lr']
        self.retain_rate = net['retain_rate']
        self.batch_size = net['batch_size']
        self.epoch = net['epoch']
        self.weights = net['weights']

def voting(predict_values, feature_obj):
    indices = np.argmax(predict_values, axis=0)
    votes = dict()
    for i in range(indices.shape[0]):
        idx = indices[i]
        if idx not in votes:
            votes[idx] = 1
        else:
            votes[idx] += 1
    max_count = -1
    num = 0
    for i in votes:
        if votes[i] > max_count:
            num = i
    array = np.zeros((1, len(feature_obj.languages)))
    array[0][num] = 1
    return feature_obj.array2Language(array)

def outputTest(networks, test_file, output_file, feature_obj):
    f = open(test_file, 'r')
    out_flie = open(output_file, 'w')
    count = 0
    for line in f:
        count += 1
        features = list()
        for i in range(len(line)-CHARLENGTH):
            chars = Feature.getCharsFromIdx(line, i)
            features.append(feature_obj.chars2Vec(chars))
        features = np.array(features)
        predict_values = networks.predict(features)
        language = voting(predict_values, feature_obj)
        out_flie.write(str(count) + ' ' + language + '\n')
    out_flie.close()
    f.close()

if __name__ == '__main__':
    train_file = sys.argv[1]
    dev_file = sys.argv[2]
    test_file = sys.argv[3]
    output_file = 'languageIdentificationPart1.output'

    feature_obj, train_features, train_targets, dev_features, dev_targets = dataPreprocessing(train_file,
        dev_file)

    print('initialize networks')
    networks = Networks()
    print('start training')
    # networks.train(train_features, train_targets, dev_features, dev_targets)
    networks.trainLineByLine(train_features, train_targets, dev_features, dev_targets, feature_obj)
    outputTest(networks, test_file, output_file, feature_obj)
