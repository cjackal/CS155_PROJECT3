# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 04:01:28 2020

@author: HyeongChan Jo
"""
import numpy as np
from pickle import dump
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Lambda
from keras import utils
from pickle import load
from keras.models import load_model
from keras.models import clone_model
from keras.models import Model
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
    
    def getTrainSeq(self):
        for sonnet in self.sonnets:
            for i in range(self.seqLen, len(sonnet), self.step):
                self.trainSeq.append(sonnet[i-self.seqLen:i+1])
                if len([x for x in sonnet[i:i+1+self.numSpace] if x == ' '])==self.numSpace:
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
        
    def Train(self):
        self.model.add(LSTM(self.LSTM_numUnits, input_shape = (self.X.shape[1], self.X.shape[2])))
        self.model.add(Dense(self.voca_size, activation = 'softmax'))
        print(self.model.summary())
        
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        self.model.fit(self.X, self.y, epochs=100, verbose=2)
        
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
            
