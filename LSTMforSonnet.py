# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 04:01:28 2020

@author: HyeongChan Jo
"""
import Dictionary
import Utility
from Sonnet import Sonnets

import re
import numpy as np
from pickle import dump
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Lambda
from keras import utils
from pickle import load
from keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
from keras.layers import LSTM
from itertools import chain
from keras.layers.embeddings import Embedding
from keras import backend as K

class LSTM_char:
    def __init__(self, seqLen = 40, step = 4, LSTM_numUnits = 200):
        self.seqLen = seqLen
        self.step = step          # gap between one sample and the next sample
        self.sonnets = []
        self.trainSeq = []
        self.numSpace = 10       # number of spaces at the end of each sonnet that can signal the end of sonnet
        self.mapping = []
        self.txt = []
        self.trainSeq_encoded = []
        self.voca_size = 0
        self.X = []
        self.y = []
        self.model = Sequential()
        self.LSTM_numUnits = LSTM_numUnits
    
    def SonnetLoader(self, path):
        if path[0]!='.':
            path = './data/' + path + '.txt'
        with open(path) as f:
            self.txt = f.read()
            
            numIdx = [x.isdigit() for x in self.txt]
            numIdx = [i for i, x in enumerate(numIdx) if x == True]   # the location of each number (shows where each sonnet starts)
            
            for i in range(len(numIdx)):
                if i == len(numIdx)-1:
                    self.sonnets.append(self.txt[numIdx[i]+2:])
                elif numIdx[i+1]-numIdx[i] != 1:
                    self.sonnets.append(self.txt[numIdx[i]+2:numIdx[i+1]-1])
        f.close()
    
    def getTrainSeq(self, includeSpaces = True):
        for sonnet in self.sonnets:
            for i in range(self.seqLen, len(sonnet), self.step):
                self.trainSeq.append(sonnet[i-self.seqLen:i+1])
                if includeSpaces and len([x for x in sonnet[i:i+1+self.numSpace] if x == ' '])==self.numSpace:
                    break
                
    def getMapping(self):
        chars = sorted(list(set(self.txt)))
        self.mapping = dict((c, i) for i, c in enumerate(chars))
        for seq in self.trainSeq:
            self.trainSeq_encoded.append([self.mapping[char] for char in seq])
        self.trainSeq_encoded = np.array(self.trainSeq_encoded)
        self.voca_size = len(self.mapping)
        self.X, self.y= self.trainSeq_encoded[:, :-1], self.trainSeq_encoded[:, -1]
        self.X = np.array([utils.to_categorical(x, num_classes=self.voca_size) for x in self.X])
        self.y = utils.to_categorical(self.y, num_classes=self.voca_size)
        
    def Train(self, patience = 10, numEpoch = 100, fileName='model'):
        # patientce: how many epochs can we wait to see a decrese in loss function
        self.model.add(LSTM(self.LSTM_numUnits, input_shape = (self.X.shape[1], self.X.shape[2])))
        self.model.add(Dense(self.voca_size, activation = 'softmax'))
        print(self.model.summary())
        
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # early stopping
        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=patience)    # patience was 5 originally
        self.model.fit(self.X, self.y, epochs=numEpoch, verbose=2, callbacks=[es])
        
        # save the model and mapping to file
        name = "%s_char.h5" % fileName
        self.model.save(name)
        name = "%s_char.pkl" % fileName
        dump(self.mapping, open(name, 'wb'))
        
    def LoadModel(self, modelName = 'model_0314_earlyStop.h5', mappingName = 'mapping_0314_earlyStop.pkl'):
        self.model = load_model(modelName)
        self.mapping = load(open(mappingName, 'rb'))
        
    def Predict(self, inputText, outputText_len=100, temperature = 1):
        # Take car of temperature by adding lambda layer to the model        
        model_predict = Sequential()
        model_predict.add(self.model.layers[0])
        model_predict.add(Lambda(lambda x: x/temperature))
        model_predict.add(self.model.layers[-1])
        model_predict.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_predict.set_weights(self.model.get_weights())
        
        # make predictions
        for _ in range(outputText_len):
            inputText_encoded = [self.mapping[char] for char in inputText]
            inputText_encoded = pad_sequences([inputText_encoded], maxlen = self.seqLen, truncating = 'pre')
            inputText_encoded = utils.to_categorical(inputText_encoded, num_classes=len(self.mapping))
            outputChar = model_predict.predict_classes(inputText_encoded, verbose=0)
            for char, index in self.mapping.items():
            	if index == outputChar:
            		outputChar = char
            		break
            inputText += outputChar
        return inputText
    
    def perplexity_train(self, temperature=1):
        if len(self.model.layers)==0:
            print('No trained model found')
            pass
        elif self.X==[] or self.y==[]:
            print('No intput/output data found')
            pass
        
        model_predict = Sequential()
        model_predict.add(self.model.layers[0])
        model_predict.add(Lambda(lambda x: x/temperature))
        model_predict.add(self.model.layers[-1])
        model_predict.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_predict.set_weights(self.model.get_weights())
        
        model_predict.summary()
        result = model_predict.evaluate(self.X, self.y)
        print(result)
        
        perplexity = np.exp(result[0])
        accuracy = result[1]
        print('perplexity: ', perplexity)
        print('accuracy: ', accuracy)
        
        return perplexity, accuracy
            

class LSTM_word(LSTM_char):
    def __init__(self, seqLen = 40, step = 1, LSTM_numUnits = 200):
        self.seqLen = seqLen
        self.step = step          # gap between one sample and the next sample
        self.sonnets = []
        self.trainSeq = []
        self.numSpace = 10       # number of spaces at the end of each sonnet that can signal the end of sonnet
        self.mapping = []
        self.txt = []
        self.trainSeq_encoded = []
        self.voca_size = 0
        self.X_orig = []
        self.X = []
        self.y = []
        self.model = Sequential()
        self.LSTM_numUnits = LSTM_numUnits
    
    def SonnetLoader(self, path='shakespeare'):
        if path=='shakespeare':
            syl_dict = Dictionary.syl_predef()  # load predefined syllable dictionary
            a = Utility.SonnetLoader(path, syl_dict)
        elif isinstance(path, list) and len(path)>1:
            df = Utility.DictLoader('dict_both_syl')
            df2 = Utility.DictLoader('dict_both_stress')
            a = [Utility.SonnetLoader(x) for x in path]
            a = list(chain.from_iterable(a))
            for temp in a:
                temp.SetDict(df)
                temp.SetDict_stress(df2)
        elif path == 'spenser':
            df = Utility.DictLoader('dict_spenser_syl') # load predefined syllable dictionary
            df2 = Utility.DictLoader('dict_spenser_stress')
            a = Utility.SonnetLoader(path)
            for temp in a:
                temp.SetDict(df)
                temp.SetDict_stress(df2)
        else:
            a = Utility.SonnetLoader(path)
#        if len([x for x in path if x=='shakespeare'])!=0:
#            syl_dict = Dictionary.syl_predef()  # load predefined syllable dictionary
#            self.sonnets = Sonnets(a, syl_dict)
#        else: 
#            self.sonnets = Sonnets(a)
        self.sonnets = Sonnets(a)
        
    
    def getTrainSeq(self):
        self.mapping = self.sonnets.sonnetList[0].index_map
        self.mapping['\n'] = len(self.mapping)
        for sonnet in self.sonnets.sonnetList:
            sonnet_concat = []
            for line in sonnet.indexform:
                sonnet_concat = sonnet_concat+line
                sonnet_concat = sonnet_concat+[len(self.mapping)-1]    # line change is also predicted
            for i in range(self.seqLen, len(sonnet_concat), self.step):
                self.trainSeq.append(sonnet_concat[i-self.seqLen:i+1])
        
    def getMapping(self):
#        for seq in self.trainSeq:
#            self.trainSeq_encoded.append([self.mapping[word] for word in seq])
#        self.trainSeq_encoded = np.array(self.trainSeq_encoded)
        self.trainSeq_encoded = np.array(self.trainSeq)
        self.voca_size = len(self.mapping)
        self.X_orig, self.y= self.trainSeq_encoded[:, :-1], self.trainSeq_encoded[:, -1]
        self.X = np.array([utils.to_categorical(x, num_classes=self.voca_size) for x in self.X_orig])
        self.y = utils.to_categorical(self.y, num_classes=self.voca_size)
#        print(self.X[0].size)
#        print(self.X[0][0].size)
#        print(self.y.size)
    
    def Train(self, useWordEmbedding = False, embeddingSize = 100, patience = 10, numEpoch = 100, fileName = 'model'):
        # patientce: how many epochs can we wait to see a decrese in loss function
        if useWordEmbedding:
            self.model.add(Embedding(self.voca_size, embeddingSize, input_length=self.seqLen))
            self.model.add(LSTM(self.LSTM_numUnits))
        else:
            self.model.add(LSTM(self.LSTM_numUnits, input_shape = (self.X.shape[1], self.X.shape[2])))
        self.model.add(Dense(self.voca_size, activation = 'softmax'))
        print(self.model.summary())
        
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # early stopping
        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
        
        if useWordEmbedding:
            self.model.fit(self.X_orig, self.y, epochs=numEpoch, verbose=2, callbacks=[es])
        else: 
            self.model.fit(self.X, self.y, epochs=numEpoch, verbose=2, callbacks=[es])
        
        # save the model & mapping to file
        if useWordEmbedding:
            name = "%s_withWordEmbedding %d.h5" % (fileName, embeddingSize)
            self.model.save(name)
            name = "%s_mapping_withWordEmbedding %d.pk1" % (fileName, embeddingSize)
            dump(self.mapping, open(name, 'wb'))
        else:
            name = "%s_withWords_noEmbedding.h5" % fileName
            self.model.save(name)
            name = "%s_mapping_withWords_noEmbedding.pkl" % fileName
            dump(self.mapping, open(name, 'wb'))
        
    def LoadModel(self, modelName = 'model.h5', mappingName = 'mapping.pkl'):
        self.model = load_model(modelName)
        self.mapping = load(open(mappingName, 'rb'))
                
    def Predict(self, inputText, outputText_len=100, temperature = 1, checkPentameter = False, std=0.5, useWordEmbedding=False):
        stddev_orig = 0.1
        stddev = stddev_orig
        stedev_step = 0.025
        numWrong_thres = 50
        numWrong=0
        
        # Take car of temperature by adding lambda layer to the model        
        model_predict = Sequential()
        for i in range(len(self.model.layers)-1):
            model_predict.add(self.model.layers[i])
        model_predict.add(Lambda(lambda x: x/temperature))
        if checkPentameter:
            model_predict.add(Lambda(lambda x: K.random_normal(shape = [1], mean = 0, stddev = 0.1)*x))
            #model_predict.add(Lambda(lambda x: self.temp(x, shape = [1], mean = 0, stddev = 0.1)))
        #import pdb; pdb.set_trace()
        model_predict.add(self.model.layers[-1])
        model_predict.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_predict.set_weights(self.model.get_weights())
        
        model_predict.summary()
        
        # parse the input text
        idx_lineChng = []
        inputText_parsed = re.sub(r"[^-'\w\s]", '', inputText).split()
        if inputText[-1:]=='\n':
            inputText_parsed = inputText_parsed+['\n']    # line change is also predicted
            idx_lineChng.append(len(inputText_parsed)-1)
            
        # make predictions
        i=0
        while i<outputText_len:
            inputText_encoded = [self.mapping[word] for word in inputText_parsed]
            inputText_encoded = pad_sequences([inputText_encoded], maxlen = self.seqLen, truncating = 'pre')
            if not useWordEmbedding:
                inputText_encoded = utils.to_categorical(inputText_encoded, num_classes=len(self.mapping))
            outputWord = model_predict.predict_classes(inputText_encoded, verbose=0)
            for word, index in self.mapping.items():
            	if index == outputWord:
            		outputWord = word
            		break
            
            if checkPentameter and (outputWord=='\n' or len(inputText_parsed)-idx_lineChng[-1]>10):
                line = inputText_parsed[idx_lineChng[-1]+1:]
                if self.sonnets.IsRegular_line(line) and self.sonnets.IsRegular_stress_line(line) and outputWord=='\n':
                    inputText_parsed.append(outputWord)
                    i+=1
                    idx_lineChng.append(len(inputText_parsed)-1)
                    print('Completed sentence: ', inputText_parsed[idx_lineChng[-2]+1:])
                    
                    # use the original model again, without random noise
                    model_predict = Sequential()
                    for layerIdx in range(len(self.model.layers)-1):
                        model_predict.add(self.model.layers[layerIdx])
                    model_predict.add(Lambda(lambda x: x/temperature))
                    model_predict.add(self.model.layers[-1])
                    model_predict.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                    model_predict.set_weights(self.model.get_weights())
                    
                    numWrong = 0
                    stddev = stddev_orig
                else:
                    #print('Wrong sentence: ', inputText_parsed[idx_lineChng[-1]+1:])
                    
                    model_predict = Sequential()
                    for layerIdx in range(len(self.model.layers)-1):
                        model_predict.add(self.model.layers[layerIdx])
                    model_predict.add(Lambda(lambda x: K.random_normal(shape = [1], mean = 0, stddev = stddev)+x))
                    model_predict.add(Lambda(lambda x: x/temperature))
                    model_predict.add(self.model.layers[-1])
                    model_predict.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
                    model_predict.set_weights(self.model.get_weights())
                    
                    i = i - len(inputText_parsed[idx_lineChng[-1]+1:])
                    inputText_parsed = inputText_parsed[0:idx_lineChng[-1]+1]
                    
                    numWrong+=1
                    if numWrong > numWrong_thres:
                        stddev = stddev + stedev_step
            else:
                inputText_parsed.append(outputWord)
                i+=1
            
        # join the parsed words
        outputText = " ".join(inputText_parsed)
        return outputText

    def perplexity_train(self, useWordEmbedding = False, temperature=1):
        if len(self.model.layers)==0:
            print('No trained model found')
            pass
        elif self.X==[] or self.y==[]:
            print('No intput/output data found')
            pass
        
        model_predict = Sequential()
#        model_predict.add(self.model.layers[0])
#        model_predict.add(self.model.layers[1])
        model_predict = Sequential()
        for i in range(len(self.model.layers)-1):
            model_predict.add(self.model.layers[i])
        model_predict.add(Lambda(lambda x: x/temperature))
            
        model_predict.add(self.model.layers[-1])
        model_predict.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_predict.set_weights(self.model.get_weights())
        
        model_predict.summary()
        
        #import pdb; pdb.set_trace()
        if useWordEmbedding:
            result = model_predict.evaluate(self.X_orig, self.y)
        else:
            result = model_predict.evaluate(self.X, self.y)
        print(result)
        
        perplexity = np.exp(result[0])
        accuracy = result[1]
        print('perplexity: ', perplexity)
        print('accuracy: ', accuracy)
        
        return perplexity, accuracy