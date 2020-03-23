# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:13:31 2020

@author: HyeongChan Jo
"""

'''
Functions for defining dictionaries
'''

import re
import pandas as pd
from itertools import chain
import nltk
from nltk.corpus import cmudict
from nltk.tokenize import SyllableTokenizer 
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
    
def sylAndStr_nltk(wordList, dict_syl=[]):
    # returns dataFrame object that contains number of syllables and stress information from the given wordList
    pro = cmudict.dict()
    stressList = []         # List of possible stress per word, in the form of list of lists
    syl_num = []            # List of syllable lengths per word, in the form of list of sets
    wordList_list = []      # Listed wordList
    exceptional = ["t'", "th'"]

    SSP = SyllableTokenizer()
    
    for unprocessed_word in wordList: # get number of syllables and stress data
        if pro.get(unprocessed_word)==None:
            if unprocessed_word in exceptional:     # Exception handler. Re-searching word in CMU dictionary after removing apostrophes.
                word = unprocessed_word
            else:
                if pro.get(re.sub(r"'$", "", unprocessed_word))!=None:
                    word = re.sub(r"'$", "", unprocessed_word)
                elif pro.get(re.sub(r"^'", "", unprocessed_word))!=None:
                    word = re.sub(r"^'", "", unprocessed_word)
                else:
                    word = re.sub(r"'$", "", re.sub(r"^'", "", unprocessed_word))
        else:
            word = unprocessed_word
        temp = pro.get(word)
        
        if temp==None: # Take care of the words that were not found in the nltk package
            syl = set() # for each of the words not found in the dictionary, put the number of syllables based on SSP, and add an empty entry in stressList
            if len(dict_syl)!=0 and (word in dict_syl.index):
                syl = {dict_syl.loc[word][0], dict_syl.loc[word][1]}
                if len(syl)>1:
                    syl.discard(0)
                syl_num.append(syl)
            else:
                word_temp = re.sub(r"'", "", word)          ### Now <word> have no apos at the start or end, do we still need to remove interior apos?
                syl.add(len(SSP.tokenize(word_temp)))
                syl_num.append(syl)
            
            stress_temp = []
            for x in syl:
                temp = [[0]*abs(x), [1]*abs(x)]
                for i, _ in enumerate(temp[0]):
                    if i%2==1:  
                        temp[0][i] = 1
                        temp[1][i] = 0
                stress_temp.append(temp)
            stress_temp = list(chain.from_iterable(stress_temp))
            stressList.append(stress_temp)   # if the stress data is unknown, assume that it follows the iambic pentameter; 
                                                # hence, put two hypothetical stress list - [0, 1, 0, 1, ...], [1, 0, 1, 0, ...] - which follows iambic pentameter
        else:
            stressList_temp = [[(phoneme[-1] if phoneme[-1].isdigit() else None) for phoneme in phonemeList] for phonemeList in temp]
            for i in range(len(stressList_temp)):
                stressList_temp_temp = []
                for s in stressList_temp[i]:
                    if s!=None:
                        if int(s)==2:
                            stressList_temp_temp.append(0.5)   # for convenience, put 0.5 instead of 2 for secondary stress
                        else:
                            stressList_temp_temp.append(int(s))
                stressList_temp[i] = stressList_temp_temp
            stressList.append(stressList_temp)
            
            if len(dict_syl)!=0 and (word in dict_syl.index):
                syl = {dict_syl.loc[word][0], dict_syl.loc[word][1]}
                if len(syl)>1:
                    syl.discard(0)
                syl_num.append(syl)
            else:
                syl_num.append( set(len(stress) for stress in stressList_temp) )
        wordList_list.append(word)
                                                
    for i, stress_eachWord in enumerate(stressList):    # leave only unique stresses in each word
        stress_temp = []
        for stress_eachStress in stress_eachWord:
            if stress_eachStress not in stress_temp:
                stress_temp.append(stress_eachStress)
        stressList[i] = stress_temp
      
    # split the data in syl_num into length1 and length2, to make a consistent structure in syllable dictionary
    length1 = []
    length2 = []
    for syl_eachWord in syl_num:
        if len(syl_eachWord)>=2:
            lenmin, lenmax = 10, 0
            for x in syl_eachWord:
                if abs(lenmin)>abs(x):
                    lenmin = x
                if abs(lenmax)<abs(x):
                    lenmax = x
            length1.append(lenmin)
            length2.append(lenmax)
        else:
            for x in syl_eachWord:
                length1.append(x)
            length2.append(0)               
            
    #df = pd.DataFrame(list(zip(wordList_list, syl_num, stressList)), columns =['word', 'syl_num', 'stress']) 
    
    df_syl = pd.DataFrame(list(zip(wordList_list, length1, length2)), columns=['word', 'length1', 'length2']) 
    df_syl.set_index("word", inplace=True)
    
    df_stress = pd.DataFrame(list(zip(wordList_list, stressList)), columns=['word', 'stress']) 
    df_stress.set_index("word", inplace=True)
    
    return df_syl, df_stress