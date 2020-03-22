# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:08:52 2020

@author: HyeongChan Jo, Juhyun Kim
"""
import re

## class 'sonnet' for saving data about a single sonnet
class Sonnet:
#    Parameters:
#            stringform: list of strings. sonnet as a list of words
#            
#            is_ending:  list of logical vluases. Whether the sonnet is ending. False by default, and True only at the end of each line
#            
#            dict:       dictionary for syllable
#            
#            index_map:  map for converting word to index. self.index_map[word] corresponds to index
#            
#            word_to_index:    sonnet with words replaced with the corresponding idx
#
#            WordList:   list of unique words in the sonnet
#   
#            RhymePair:  Pair of words that rhymes in the sonnet
#
#            dict_stress:  dictionary for stress
    
    def __init__(self, sonnet, predefinedDict = []):
        self.stringform = sonnet        ### sonnet as a list of words itself
        
        is_ending = [[False for _ in range(len(line))] for line in sonnet]
        for line in is_ending:
            line[-1] = True
        self.is_ending = is_ending      ### Encoding the location of the end of each lines (having the same shape as stringform)
        
        if len(predefinedDict) != 0:
            self.SetDict(predefinedDict)
        else:
            self.dict = []
            
        self.WordList = self.returnWordList()
        self.RhymePair = self.returnRhymePair()
        self.dict_stress = []

    def __repr__(self):
        s = ''
        for line in self.stringform:
            for word in line:
                s += word+' '
            s += '\n'
        return s
    
    def SetDict(self, df, removeApo=True):  ### Set the syllable dictionary.
        self.dict = df                  ### Temporary format: rows indexed by the words, with two columns of possible syllables
        idxmap = {}
        for i, s in enumerate(self.dict.index.to_numpy()):
            idxmap[s] = i
        self.index_map = idxmap         ### {key:value}={word:idx}
        word_to_idx = []
        if removeApo:
            for i, line in enumerate(self.stringform):
                word_to_idx_line = []
                for j, word in enumerate(line):
                    idx = self.index_map.get(word)
                    if isinstance(idx, int):
                        word_to_idx_line.append(idx)
                    else:
                        self.stringform[i][j] = re.sub(r"'$", "", self.stringform[i][j])
                        self.stringform[i][j] = re.sub(r"^'", "", self.stringform[i][j])
                        word_to_idx_line.append(self.index_map[self.stringform[i][j]])
                word_to_idx.append(word_to_idx_line)
        else:
            unmatched = []
            for line in self.stringform:
                word_to_idx_line = []
                for word in line:
                    idx = self.index_map.get(word)
                    if isinstance(idx, int):
                        word_to_idx_line.append(idx)
                    else:
                        unmatched.append(word)
                word_to_idx.append(word_to_idx_line)
                
                if len(unmatched)!=0:
                    print(self.unmatched)
                    raise KeyError
                    
        self.word_to_index = word_to_idx    ### sonnet with words replaced with the corresponding idx
        self.WordList = self.returnWordList() # reassign word list, as some of it got changed (e.g. removing apostrophes, etc)

    def IsRegular(self):
        """
        Check if the given sonnet is in regular (pentameter) form.
        Must set the syllable dictionary beforehand.
        With a little modification, can assign a valid syllable length for the words.
        """
        try:
            df = self.dict
            isregular = False
            regularity = 0
            if len(self.stringform)==14:
                for line in self.stringform:
                    syllable_counter_min = 0
                    syllable_counter_max = 0
                    for i in range(len(line)):
                        if i<len(line)-1:
                            if df.loc[line[i]][1]==0:
                                syllable_counter_max += df.loc[line[i]][0]
                                syllable_counter_min += df.loc[line[i]][0]
                            else:
                                if df.loc[line[i]][0]<0:
                                    syllable_counter_max += df.loc[line[i]][1]
                                    syllable_counter_min += df.loc[line[i]][1]
                                elif df.loc[line[i]][1]<0:
                                    syllable_counter_max += df.loc[line[i]][0]
                                    syllable_counter_min += df.loc[line[i]][0]
                                else:
                                    syllable_counter_max += df.loc[line[i]][1]
                                    syllable_counter_min += df.loc[line[i]][0]
                        else:
                            if df.loc[line[i]][1]==0:
                                syllable_counter_max += df.loc[line[i]][0]
                                syllable_counter_min += df.loc[line[i]][0]
                            else:
                                syllable_counter_max += abs(df.loc[line[i]][1])
                                syllable_counter_min += abs(df.loc[line[i]][0])
                                
                                
                    if syllable_counter_min <= 10 <= syllable_counter_max:
                        regularity += 1
            if regularity==14:
                isregular = True
            return isregular

        except AttributeError:
            print("Set the syllable dictionary to use.")
            
    def IsRegular_line(self, line):
        """
        Check if the given line is in regular (pentameter) form: 
        Must set the syllable dictionary beforehand.
        """
        df = self.dict
        syllable_counter_min = 0
        syllable_counter_max = 0
        isregular = False
        for i in range(len(line)):
            if i<len(line)-1:
                if df.loc[line[i]][1]==0:
                    syllable_counter_max += df.loc[line[i]][0]
                    syllable_counter_min += df.loc[line[i]][0]
                else:
                    if df.loc[line[i]][0]<0:
                        syllable_counter_max += df.loc[line[i]][1]
                        syllable_counter_min += df.loc[line[i]][1]
                    elif df.loc[line[i]][1]<0:
                        syllable_counter_max += df.loc[line[i]][0]
                        syllable_counter_min += df.loc[line[i]][0]
                    else:
                        syllable_counter_max += df.loc[line[i]][1]
                        syllable_counter_min += df.loc[line[i]][0]
            else:
                if df.loc[line[i]][1]==0:
                    syllable_counter_max += df.loc[line[i]][0]
                    syllable_counter_min += df.loc[line[i]][0]
                else:
                    syllable_counter_max += abs(df.loc[line[i]][1])
                    syllable_counter_min += abs(df.loc[line[i]][0])
        if syllable_counter_min <= 10 <= syllable_counter_max:
            isregular = True
            
        return isregular
    
    def IsRegular_stress_line(self, line, strict = False):
        """
        Check if the given line is in regular (pentameter) form in terms of stress: 
        Must set the syllable dictionary beforehand.
        The input "strict" decides whether stress should strictly follow iambic pentameter (i.e. stress from nltk should strictly follow 0, 1, 0, 1, ...)
        or it can have some syllables with same stress in a row (i.e. 1, 1, 1, ... for a few words. c.f. Shall I compare thee ... also falls into this category, because "shall" has a primary stress)
        """
        from itertools import product
        from itertools import chain
        import numpy as np
        maxLen = 10  # assume that maximum number of words in a single line is 10
        df = self.dict_stress
        print(df)
        print(line)
        stress = [df.loc[x][0] for x in line]
        for i in range(maxLen-len(stress)):
            stress.append([[-1]]) # fix the length of the list to 10
        
        comb = list(product(stress[0], stress[1], stress[2], stress[3], stress[4], stress[5], stress[6], stress[7], stress[8], stress[9]))
        isregular = False
        for x in comb:
            stressList = list(chain.from_iterable(x))
            stressList = np.array([x for x in stressList if x!=-1])
            print(stressList)
            
            stressChng = [stressList[i]-stressList[i-1] for i in range(1, len(stressList), 1)]
            isregular_temp = True
            for i, y in enumerate(stressChng):
                if not strict and ((i%2 == 0 and y<0) or (i%2 == 1 and y>0)):
                    isregular_temp = False
                    break
                elif strict and ((i%2 == 0 and y<=0) or (i%2 == 1 and y>=0)):
                    isregular_temp = False
                    break
                
            if isregular_temp==True:
                isregular = True
                print('True: ', stressList)
                break
            
        return isregular


    def returnWordList(self):
        s = set()
        for line in self.stringform:
            s |= set(line)
        return s
    
    def returnRhymePair(self):
        pair = []
        paring = [[0,2],[1,3],[4,6],[5,7],[8,10],[9,11],[12,13]]
        for couple in paring:
            i, j = couple
            if j<len(self.stringform):
                pair.append({self.stringform[i][-1], self.stringform[j][-1]})
        return pair



## class 'sonnets' for saving data about multiple sonnets
class Sonnets(Sonnet):
    def __init__(self, sonnetList, predefinedDict = []):
        self.sonnetList = sonnetList
        self.is_ending = [eachSonnet.is_ending for eachSonnet in sonnetList]
        self.dict = sonnetList[0].dict
        self.dict_stress = sonnetList[0].dict_stress
        
        self.WordList = set()
        for sonnet in self.sonnetList:
            self.WordList |= set(sonnet.WordList)
        
        self.RhymePair = []
        for sonnet in self.sonnetList:
            self.RhymePair.append(sonnet.RhymePair)
            
        if len(self.dict)!=0:
            self.word_to_indexList = [eachSonnet.word_to_index for eachSonnet in sonnetList]
        elif len(predefinedDict)!=0:
            self.SetDict(predefinedDict)
            for sonnet in self.sonnetList:
                sonnet.SetDict(predefinedDict)
        else:
            self.SetDict_new()
            
        if len(self.dict)!=0 and len(self.dict_stress)==0:
            self.SetDict_stress()
            
    def SetDict_new(self):
        # define new dictionary, based on nltk
        import Dictionary
        self.dict, self.dict_stress = Dictionary.sylAndStr_nltk(self.WordList)

    def SetDict_stress(self):   # set dictionary for stress based on nltk
        if len(self.dict)!=0 and len(self.dict_stress)==0:
            import Dictionary
            dict_temp, self.dict_stress = Dictionary.sylAndStr_nltk(self.WordList, self.dict)
            for i, x in enumerate(self.dict_stress["stress"]):
                word = self.dict_stress.index[i]
                sylNum_end = None               # number of syllables if a word is at the end of the line
                if self.dict.loc[word][0]<0: 
                    sylNum_end = abs(self.dict["length1"][i])
                elif self.dict.loc[word][1]<0:
                    sylNum_end = abs(self.dict["length2"][i])
                    
                if sylNum_end == None:
                    continue
                sylNum_temp = [dict_temp.loc[word][0], dict_temp.loc[word][1]]
                if sylNum_end != sylNum_temp[0] and sylNum_end != sylNum_temp[1]: # add additional pronunciation for the words which have less syllables at the end of the line
                    #print(word)
                    #print('stress list: ', self.dict_stress["stress"][i], 'syllable list: ', self.dict.loc[word][0], self.dict.loc[word][1])
                    stress_temp = [xx[0:-1] for xx in x]
                    self.dict_stress["stress"][i] = self.dict_stress["stress"][i]+stress_temp
                    
        print(self.dict_stress)
                
                    
                    
            