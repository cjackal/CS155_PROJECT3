# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:13:31 2020

@author: HyeongChan Jo
"""

import pandas as pd
import nltk
from nltk.corpus import cmudict
from itertools import chain
from nltk.tokenize import SyllableTokenizer # use syllable tokenizer only when syllabel&stress info is not found in cmudict
import re
nltk.download('cmudict')

def syl_predef(filePath = './data/Syllable_dictionary.txt'):
    syl_dict = pd.read_csv(filePath, sep=' ', names=["word", "length1", "length2"])
    syl_dict.fillna(0, inplace=True)
    # Syllable length of weak ending is negated, NaNs are replaced with 0
    for i in syl_dict.index:
        if syl_dict["length2"][i]==0:                             ### Caveat: There are words with 0 syllables, but they all consists of fixed syllable length.
            syl_dict["length1"][i] = int(syl_dict["length1"][i])
        else:
            if syl_dict["length1"][i][0]=='E':
                syl_dict["length1"][i] = -int(syl_dict["length1"][i][1:])
                syl_dict["length2"][i] = int(syl_dict["length2"][i])
            elif syl_dict["length2"][i][0]=='E':
                syl_dict["length1"][i] = int(syl_dict["length1"][i])
                syl_dict["length2"][i] = -int(syl_dict["length2"][i][1:])
            else:
                syl_dict["length1"][i] = int(syl_dict["length1"][i])
                syl_dict["length2"][i] = int(syl_dict["length2"][i])
    syl_dict.set_index("word", inplace=True)
    """
    length2==0 iff fixed syllable length
    |length1|<|length2| if variable syllable length
    """
    syl_dict.head()
    return syl_dict
    
def sylAndStr_nltk(wordList, dict_syl=[]):
    # returns dataFrame object that contains number of syllables and stress information from the given wordList
    pro = cmudict.dict()
    stressList = []
    unmatched = []  # list of unmatched words
    wordList_list = []
    syl_num = []
    
    for word in wordList: # get number of syllables and stress data
        temp = pro.get(word)
        if temp==None:
            unmatched.append(word)
        else:
            stressList_temp = [[ [phoneme[-1] if phoneme[-1].isdigit() else None] for phoneme in phonemeList] for phonemeList in temp]
            for i in range(len(stressList_temp)):
                stressList_temp[i] = [int(s[0]) for s in stressList_temp[i] if s!=[None]]
                idx = [x for x in range(len(stressList_temp[i])) if stressList_temp[i][x]==2]
                if len(idx)!=0:
                    for idx_each in idx:
                        stressList_temp[i][idx_each] = 0.5 # for convenience, put 0.5 instead of 2 for secondary stress
            stressList.append(stressList_temp)
            wordList_list.append(word)
            
            if len(dict_syl)!=0 and len([x for x in dict_syl.index if x==word])!=0:
                syl_num.append( set([dict_syl.loc[word][0], dict_syl.loc[word][1]]) )
            else:
                syl_num.append( set([len(stress) for stress in stressList_temp]) )
    #syl_num = [ set([len(stress) for stress in chosenWord]) for chosenWord in stressList]  
    
    if len(unmatched)!=0:   # Take care of the words that were not found in the nltk package
        
        SSP = SyllableTokenizer()
        
        for word in unmatched:  # for each of the words not found in the dictionary, put the number of syllables based on SSP, and add an empty entry in stressList
            syl = set()
            if len(dict_syl)!=0 and len([x for x in dict_syl.index if x==word])!=0:
                for x in [dict_syl.loc[word][0], dict_syl.loc[word][1]]:
                    syl.add(x)
                if len(syl)>1:
                    syl.discard(0)
                syl_num.append(syl)
            else:
                word_temp = re.sub(r"'", "", word)
                syl.add(len(SSP.tokenize(word_temp)))
                syl_num.append(syl)
            stress_temp = []
            
            for x in syl:
                temp = [[0]*x, [1]*x]
                for i, _ in enumerate(temp[0]):
                    if i%2==1:  
                        temp[0][i] = 1
                        temp[1][i] = 0
                stress_temp.append(temp)
            stress_temp = list(chain.from_iterable(stress_temp))
            stressList.append(stress_temp)   # if the stress data is unknown, assume that it follows the iambic pentameter; 
                                                # hence, put two hypothetical stress list - [0, 1, 0, 1, ...], [1, 0, 1, 0, ...] - which follows iambic pentameter
            wordList_list.append(word)
                                                
    for i, stress_eachWord in enumerate(stressList):    # leave only unique stresses in each word
        if len(set(map(tuple, x) for x in stress_eachWord))==len(stress_eachWord):
            continue
        stress_temp = []
        for j, stress_eachStress in enumerate(stress_eachWord):
            duplicate_stress = any([stress_eachStress==x for x in stressList[i][j+1:]])
            if duplicate_stress == False:
                stress_temp.append(stress_eachStress)
        stressList[i] = stress_temp
      
    # split the data in syl_num into length1 and length2, to make a consistent structure in syllable dictionary
    length1 = []
    length2 = []
    for syl_eachWord in syl_num:
        i = 0
        for x in syl_eachWord:
            if i == 0:
                length1.append(x)
            elif i == 1:
                length2.append(x)
            i += 1
        if i==1:
            length2.append(0)               
            
    #df = pd.DataFrame(list(zip(wordList_list, syl_num, stressList)), columns =['word', 'syl_num', 'stress']) 
    
    df_syl = pd.DataFrame(list(zip(wordList_list, length1, length2)), columns =['word', 'length1', 'length2']) 
    df_syl.set_index("word", inplace=True)
    
    df_stress = pd.DataFrame(list(zip(wordList_list, stressList)), columns =['word', 'stress']) 
    df_stress.set_index("word", inplace=True)
    
    return df_syl, df_stress
