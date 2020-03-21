# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:13:31 2020

@author: HyeongChan Jo
"""

import pandas as pd
import nltk
from nltk.corpus import cmudict
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
    
def sylAndStr_nltk(wordList):
    # returns dataFrame object that contains number of syllables and stress information from the given wordList
    pro = cmudict.dict()
    stressList = []
    unmatched = []  # list of unmatched words
    wordList_list = []
    
    for word in wordList:
        #temp = pro[word]
        temp = pro.get(word)
        if temp==None:
            unmatched.append(word)
        else:
            stressList_temp = [[[phoneme[-1] if phoneme[-1].isdigit() else None] for phoneme in phonemeList] for phonemeList in temp]
            for i in range(len(stressList_temp)):
                stressList_temp[i] = [int(s[0]) for s in stressList_temp[i] if s!=[None]]
            stressList.append(stressList_temp)
            wordList_list.append(word)
    syl_num = [ [len(stress) for stress in chosenWord] for chosenWord in stressList]  
    
    if len(unmatched)!=0:
        from nltk.tokenize import SyllableTokenizer # use syllable tokenizer only when syllabel&stress info is not found in cmudict
        SSP = SyllableTokenizer()
        
        for word in unmatched:  # for each of the words not found in the dictionary, put the number of syllables based on SSP, and add an empty entry in stressList
            temp = SSP.tokenize(word)
            syl_num.append([len(temp)])
            stressList.append([])
            
    for i, syl_num_list in enumerate(syl_num):
        if len(syl_num_list)==1 or len(set(tuple(x) for x in stressList[i]))==len(stressList[i]):
            continue
        syl_num_temp = []
        stress_temp = []
        for j, syl_num_each in enumerate(syl_num_list):
            duplicate = any([syl_num_each==x for x in syl_num_list[j+1:]])
            duplicate_stress = any([stressList[i][j]==x for x in stressList[i][j+1:]])
            if duplicate == False and duplicate_stress == False:
                syl_num_temp.append(syl_num_each)
                stress_temp.append(stressList[i][j])
        syl_num[i] = syl_num_temp
        stressList[i] = stress_temp
      
    df = pd.DataFrame(list(zip(wordList_list, syl_num, stressList)), columns =['word', 'syl_num', 'stress']) 
    df.set_index("word", inplace=True)
    
    return df
