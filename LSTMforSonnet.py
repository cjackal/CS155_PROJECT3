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
                    break
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
        
    def Train(self, patience = 10, numEpoch = 100):
        # patientce: how many epochs can we wait to see a decrese in loss function
        self.model.add(LSTM(self.LSTM_numUnits, input_shape = (self.X.shape[1], self.X.shape[2])))
        self.model.add(Dense(self.voca_size, activation = 'softmax'))
        print(self.model.summary())
        
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        # early stopping
        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)
        self.model.fit(self.X, self.y, epochs=numEpoch, verbose=2, callbacks=[es])
        
        # save the model to file
        self.model.save('model.h5')
        # save the mapping
        dump(self.mapping, open('mapping.pkl', 'wb'))
        
    def LoadModel(self, modelName = 'model.h5', mappingName = 'mapping.pkl'):
        self.model = load_model('model.h5')
        self.mapping = load(open('mapping.pkl', 'rb'))
        
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
    
    def SonnetLoader(self, path):
        syl_dict = Dictionary.syl_predef()  # load predefined syllable dictionary
        a = Utility.SonnetLoader('shakespeare', syl_dict)
        self.sonnets = Sonnets(a)
    
    def getTrainSeq(self):
        for sonnet in self.sonnets.sonnetList:
            sonnet_concat = []
            for line in sonnet.stringform:
                sonnet_concat = sonnet_concat+line
                sonnet_concat = sonnet_concat+['\n']    # line change is also predicted
            for i in range(self.seqLen, len(sonnet_concat), self.step):
                self.trainSeq.append(sonnet_concat[i-self.seqLen:i+1])
                
    def getMapping(self):
        self.mapping = self.sonnets.sonnetList[0].index_map
        self.mapping['\n'] = len(self.mapping)
        for seq in self.trainSeq:
            self.trainSeq_encoded.append([self.mapping[word] for word in seq])
        self.trainSeq_encoded = np.array(self.trainSeq_encoded)
        self.voca_size = len(self.mapping)
        self.X_orig, self.y= self.trainSeq_encoded[:, :-1], self.trainSeq_encoded[:, -1]
        self.X = np.array([utils.to_categorical(x, num_classes=self.voca_size) for x in self.X_orig])
        self.y = utils.to_categorical(self.y, num_classes=self.voca_size)
        print(self.X[0].size)
        print(self.X[0][0].size)
        print(self.y.size)
    
    def Train(self, useWordEmbedding = False, embeddingSize = 100, patience = 10, numEpoch = 100):
        # patientce: how many epochs can we wait to see a decrese in loss function
        if useWordEmbedding:
            from keras.layers.embeddings import Embedding
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
            name = "model_withWordEmbedding %d.h5" % embeddingSize
            self.model.save(name)
            name = "mapping_withWordEmbedding %d.pk1" % embeddingSize
            dump(self.mapping, open(name, 'wb'))
        else:
            self.model.save('model_withWords_noEmbedding.h5')
            dump(self.mapping, open('mapping_withWords_noEmbedding.pkl', 'wb'))
        
    def LoadModel(self, modelName = 'model.h5', mappingName = 'mapping.pkl'):
        self.model = load_model('model_withWords.h5')
        self.mapping = load(open('mapping_withWords.pkl', 'rb'))
        
    def Predict(self, inputText, outputText_len=100, temperature = 1):
        # Take car of temperature by adding lambda layer to the model        
        model_predict = Sequential()
        model_predict.add(self.model.layers[0])
        model_predict.add(Lambda(lambda x: x/temperature))
        model_predict.add(self.model.layers[-1])
        model_predict.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model_predict.set_weights(self.model.get_weights())
        
        # parse the input text
        inputText_parsed = re.sub(r"[^-'\w\s]", '', inputText).split()
        if inputText[-1:]=='\n':
            inputText_parsed = inputText_parsed+['\n']    # line change is also predicted
    
        # make predictions
        for _ in range(outputText_len):
            inputText_encoded = [self.mapping[word] for word in inputText_parsed]
            inputText_encoded = pad_sequences([inputText_encoded], maxlen = self.seqLen, truncating = 'pre')
            inputText_encoded = utils.to_categorical(inputText_encoded, num_classes=len(self.mapping))
            outputWord = model_predict.predict_classes(inputText_encoded, verbose=0)
            for word, index in self.mapping.items():
            	if index == outputWord:
            		outputWord = word
            		break
            inputText_parsed.append(outputWord)
            
        # join the parsed words
        outputText = " ".join(inputText_parsed)
        return outputText