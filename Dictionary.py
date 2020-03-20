# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:13:31 2020

@author: HyeongChan Jo
"""

'''
Functions for defining dictionaries
'''


import pandas as pd
import nltk
from nltk.corpus import cmudict
# nltk.download('cmudict')

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
    # syl_dict.head()
    return syl_dict
    
def sylAndStr_nltk(wordList):
    # returns dataFrame object that contains number of syllables and stress information from the given wordList
    pro = cmudict.dict()
    stressList = []
    
    for word in wordList:
        temp = pro[word]
        stressList_temp = [[[phoneme[-1] if phoneme[-1].isdigit() else None] for phoneme in phonemeList] for phonemeList in temp]
        for i in range(len(stressList_temp)):
            stressList_temp[i] = [int(s[0]) for s in stressList_temp[i] if s!=[None]]
        stressList.append(stressList_temp)
    
    syl_num = [ [len(stress) for stress in chosenWord] for chosenWord in stressList]        
    df = pd.DataFrame(list(zip(syl_num, stressList)), columns =['syl_num', 'stress']) 
    
    return df