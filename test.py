# -*- coding: utf-8 -*-
"""
Created on Sun Mar 15 18:07:11 2020

@author: HyeongChan Jo
"""


# LSTM with word embedding - dimension: 50, 100, 200
#from LSTMforSonnet import LSTM_char
from LSTMforSonnet import LSTM_word
#from keras.models import Sequential
#from keras.layers import Dense
#
#embedSize = [50, 100, 200]
#for dim in embedSize:
#    test2 = LSTM_word()
#    test2.SonnetLoader('shakespeare')
#    test2.getTrainSeq()
#    test2.getMapping()
#    test2.Train(useWordEmbedding = True, embeddingSize = dim)


#from Sonnet import Sonnet, Sonnets
#import Utility
#import Dictionary
#syl_dict = Dictionary.syl_predef()  # load predefined syllable dictionary
#a = Utility.SonnetLoader('shakespeare', syl_dict)
#sonnet_all_sh = Sonnets(a)
#
#print(len(sonnet_all_sh.dict))


# LSTM with word embedding - dimension: 50, 100, 200
#from LSTMforSonnet import LSTM_char, LSTM_word
#from keras.models import Sequential
#from keras.layers import Dense
#
#embedSize = [5, 10, 25]
#for dim in embedSize:
#    test2 = LSTM_word()
#    test2.SonnetLoader(['shakespeare', 'Spenser_v2'])
#    test2.getTrainSeq()
#    test2.getMapping()
#    test2.Train(useWordEmbedding = True, embeddingSize = dim, numEpoch = 200)




#from Sonnet import Sonnet, Sonnets
#import Utility
#import Dictionary
#syl_dict = Dictionary.syl_predef()  # load predefined syllable dictionary
#a = Utility.SonnetLoader('Spenser_v2')
##a = Utility.SonnetLoader('shakespeare')
#sonnet_all_sh = Sonnets(a)
#
##print(len(sonnet_all_sh.dict))



#from LSTMforSonnet import LSTM_char, LSTM_word
#from keras.models import Sequential
#from keras.layers import Dense
#
#embedSize = [5, 10, 25]
#for dim in embedSize:
#    modelName = "model_Spenser_withWordEmbedding %d.h5" % dim
#    mappingName = "model_Spenser_mapping_withWordEmbedding %d.pk1" % dim
#    test2 = LSTM_word()
#    test2.LoadModel(modelName = modelName, mappingName = mappingName)
#    test2.SonnetLoader('shakespeare')



## with checking petmeter
#from LSTMforSonnet import LSTM_char, LSTM_word
#from keras.models import Sequential
#from keras.layers import Dense
#
#embedSize = [25]
#for dim in embedSize:
##    modelName = "model_Spenser_withWordEmbedding %d.h5" % dim
##    mappingName = "model_Spenser_mapping_withWordEmbedding %d.pk1" % dim
##    modelName = "model_withWordEmbedding %d.h5" % dim
##    mappingName = "mapping_withWordEmbedding %d.pk1" % dim
#    modelName = "model_unit400_withWordEmbedding %d.h5" % dim
#    mappingName = "model_unit400_mapping_withWordEmbedding %d.pk1" % dim
#    
#    test2 = LSTM_word()
#    test2.SonnetLoader('shakespeare')
#    test2.getTrainSeq()
#    test2.getMapping()
#
#    test2.LoadModel(modelName = modelName, mappingName = mappingName)
#    test2.model.summary()
#    
#    print(test2.perplexity_train(useWordEmbedding=True))





#from LSTMforSonnet import LSTM_char, LSTM_word
#from keras.models import Sequential
#from keras.layers import Dense
#
#numUnit = [100, 200, 300, 400]
#for un in numUnit:
#    test2 = LSTM_word(LSTM_numUnits = un)
#    test2.SonnetLoader(['shakespeare', 'Spenser_v2'])
#    test2.getTrainSeq()
#    test2.getMapping()
#    test2.Train(useWordEmbedding = True, numEpoch = 200, fileName = "model_Both_unit%d" % un)







# with checking pentameter
#from LSTMforSonnet import LSTM_char, LSTM_word
#from keras.models import Sequential
#from keras.layers import Dense
#
### model with minimum perplexity
#useWordEmbedding = True
#embedSize = [100]
#numUnit = [200]
#data = 'shakespeare'
#
#if data == 'shakespeare':
#    fileName_data = ''
#elif data == 'spenser':
#    fileName_data = 'Spenser_'
#elif data == 'both':
#    fileName_data = 'Both_'
#
#for i, dim in enumerate(embedSize):
#    for j, un in enumerate(numUnit):
#        if useWordEmbedding:
#            modelName = "model_%sunit%d_withWordEmbedding %d.h5" % (fileName_data, un, dim)
#            mappingName = "model_%sunit%d_mapping_withWordEmbedding %d.pk1" % (fileName_data, un, dim)
#        test2 = LSTM_word()
#        test2.LoadModel(modelName = modelName, mappingName = mappingName)
#        if data == 'shakespeare':
#            test2.SonnetLoader('shakespeare')
#        elif data == 'spenser':
#            test2.SonnetLoader('Spenser_v2')
#        elif data == 'both':
#            test2.SonnetLoader(['shakespeare', 'Spenser_v2'])
#        
#        tempList = [1.5, 1, 0.75, 0.25]
#        predicted = [test2.Predict("shall i compare thee to a summer's day?\n", outputText_len=150, temperature = x, checkPentameter = True, useWordEmbedding = True) for x in tempList]
#        for i, x in enumerate(predicted):
#            print('temperature: ', tempList[i], ', embedding size: ', dim, '\n', x, '\n')



# Look into perplexity - with characters
from LSTMforSonnet import LSTM_char, LSTM_word
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

stepList = [1, 2, 3, 4, 5, 6, 8]
stepList = [1]
patienceList = [5]
perplexity = np.zeros((len(stepList), len(patienceList)))
accuracy = np.zeros((len(stepList), len(patienceList)))

for i, step in enumerate(stepList):
    for j, patience in enumerate(patienceList):
        maxEpoch = '_maxEpoch400'   
            # when step==1, maxEpoch was 200, because it stopped before reaching 200 due to early stopping;
            # otherwise, it finished between 200 and 400 epochs, so maxEpoch was set to 400
        modelName = "model_step%d_patience%d%s_char.h5" % (step, patience, maxEpoch)
        print(modelName)
        mappingName = "model_step%d_patience%d%s_char.pkl" % (step, patience, maxEpoch)
        test2 = LSTM_char(step = 4) # for the purpose of testing; therefore, it is different from the settings used in each model
        test2.SonnetLoader('shakespeare')
        test2.getTrainSeq()
        test2.getMapping()
        test2.LoadModel(modelName = modelName, mappingName = mappingName)
        test2.model.summary()
        
        test2.Predict("shall i compare thee to a summer's day?\n", outputText_len=300, temperature = 1)
        
        [perplexity[j, i], accuracy[j, i]] = test2.perplexity_train()
        print("step ", step, "patience", patience)
        print(perplexity[j, i])
        print(accuracy[j, i])
