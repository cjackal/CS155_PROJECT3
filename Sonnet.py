# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:08:52 2020

@author: HyeongChan Jo, Juhyun Kim
"""

import re
from itertools import product, chain

## Master class 'sonnet'

class Sonnet:
#    Attributes:
#            stringform: list of strings. sonnet as a list of words
#            
#            is_ending:  list of logical values. Whether the sonnet is ending. False by default, True only at the end of each line
#            
#            dict_syl:   predefined dictionary for syllable
#        
#            dict_stress:predefined dictionary for stress
#            
#            index_map:  map for converting word to index. self.index_map[word] corresponds to index
#            
#            indexform:  sonnet with words replaced with the corresponding idx
#       
#    Methods:
#            WordList:   list of unique words in the sonnet
#    
#            RhymePair:  Pair of words that rhymes in the sonnet

    def __init__(self, sonnet, Dict_syl=[], Dict_stress=[]):
        self.stringform = sonnet        ### sonnet as a list of words itself
        is_ending = [[False for _ in range(len(line))] for line in sonnet]
        for line in is_ending:
            line[-1] = True
        self.is_ending = is_ending      ### Encoding the location of the end of each lines (having the same shape as stringform)
        if len(Dict_syl)!=0:
            self.SetDict(Dict_syl)    ### Set the syllable dictionary. Rows indexed by the words, with two columns of possible syllables
        if len(Dict_stress)!=0:
            self.SetDict_stress(Dict_stress)  ### Set the stress dictionary. Rows indexed by the words, with one column of list of possible stresses

    def __repr__(self):
        s = ''
        for line in self.stringform:
            for i, word in enumerate(line):
                if i==0:
                    s += word.capitalize()
                else:
                    s += ' ' + word
            s += '\n'
        return s

    def SetDict(self, df):
        self.dict_syl = df
        self.Word_to_Index()

    def SetDict_stress(self, df):
        try:
            df_syl = self.dict_syl
            if not df.index.equals(df_syl.index):
                print("Indices of syllable and stress dictionaries do not match.")
            else:
                self.dict_stress = df
        except AttributeError:
            print("Set the syllable dictionary to use.")
    
    def Word_to_Index(self, removeApo=True):                               
        try:
            df = self.dict_syl

            idxmap = {}
            for i, s in enumerate(df.index.to_numpy()):
                idxmap[s] = i
            self.index_map = idxmap         ### {key:value}={word:idx}

            word_to_idx = []
            unmatched = []
            for i, line in enumerate(self.stringform):
                word_to_idx_line = []
                for j, word in enumerate(line):
                    idx = self.index_map.get(word)
                    if isinstance(idx, int):
                        word_to_idx_line.append(idx)
                    else:
                        if removeApo:
                            if isinstance(self.index_map.get(re.sub(r"'$", "", word)), int):
                                word_to_idx_line.append(self.index_map.get(re.sub(r"'$", "", word)))
                            elif isinstance(self.index_map.get(re.sub(r"^'", "", word)), int):
                                word_to_idx_line.append(self.index_map.get(re.sub(r"^'", "", word)))
                            elif isinstance(self.index_map.get(re.sub(r"'$", "", re.sub(r"^'", "", word))), int):
                                word_to_idx_line.append(self.index_map.get(re.sub(r"'$", "", re.sub(r"^'", "", word))))
                            else:
                                unmatched.append(word)
                        else:
                            unmatched.append(word)
                word_to_idx.append(word_to_idx_line)
            if len(unmatched)!=0:
                print(unmatched)
                raise KeyError
            self.indexform = word_to_idx    ### sonnet with words replaced with the corresponding idx

        except AttributeError:
            print("Set the syllable dictionary to use.")

    def IsRegular(self, strict=False):  ### Do all possible regularity check.
        sonnetlen = len(self.stringform)
        try:
            isregular_syl = self.IsRegular_syl()
        except:
            isregular_syl = True
            print("No syllable dictionary assigned.")
        try:
            isregular_stress = self.IsRegular_stress(strict=strict)
        except:
            isregular_stress = True
            print("No stress dictionary assigned.")
        return (sonnetlen==14 and isregular_syl and isregular_stress)

    def IsRegular_syl(self, verbose=False):
        """
        Check if the given sonnet is in regular (pentameter) form.
        Must set the syllable dictionary beforehand.
        With a little modification, can assign a valid syllable length for the words.
        """
        try:
            df = self.dict_syl
            regularity = 0
            for _, line in enumerate(self.indexform):
                syllable_counter_min = 0
                syllable_counter_max = 0
                for i, word in enumerate(line):
                    if i<len(line)-1:
                        if df.iloc[word, 1]==0:
                            syllable_counter_max += df.iloc[word, 0]
                            syllable_counter_min += df.iloc[word, 0]
                        else:
                            if df.iloc[word, 0]<0:
                                syllable_counter_max += df.iloc[word, 1]
                                syllable_counter_min += df.iloc[word, 1]
                            elif df.iloc[word, 1]<0:
                                syllable_counter_max += df.iloc[word, 0]
                                syllable_counter_min += df.iloc[word, 0]
                            else:
                                syllable_counter_max += df.iloc[word, 1]
                                syllable_counter_min += df.iloc[word, 0]
                    else:
                        if df.iloc[word, 1]==0:
                            syllable_counter_max += df.iloc[word, 0]
                            syllable_counter_min += df.iloc[word, 0]
                        else:
                            syllable_counter_max += abs(df.iloc[word, 1])
                            syllable_counter_min += abs(df.iloc[word, 0])
                if syllable_counter_min <= 10 <= syllable_counter_max:
                    regularity += 1
                elif verbose:
                    print(f"Line {_} is not regular:")
                    print("Minimal possible syllables:", syllable_counter_min, ", Maximal possible syllables:", syllable_counter_max)
            return (regularity==len(self.indexform))

        except AttributeError:
            print("Set the syllable dictionary to use.")

    def IsRegular_stress(self, strict=False, verbose=False):
        """
        Check if the given sonnet is in regular (pentameter) form.
        Must set the syllable dictionary beforehand.
        The input "strict" decides whether stress should strictly follow iambic pentameter (i.e. stress from nltk should strictly follow 0, 1, 0, 1, ...)
        or it can have some syllables with same stress in a row
        (i.e. 1, 1, 1, ... for a few words. c.f. Shall I compare thee ... also falls into this category, because "shall" has a primary stress)
        """
        try:
            df = self.dict_stress
            regularity = 0
            for _, line in enumerate(self.indexform):
                stress = [df.iloc[word, 0] for word in line]
                comb = list(product(*stress))

                isregular = False
                for x in comb:
                    stressList = list(chain.from_iterable(x))
                    stressChng = [stressList[i]-stressList[i-1] for i in range(1, len(stressList))]
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
                        break
                if isregular:
                    regularity += 1
                elif verbose:
                    print(f"Line {_} is not regular:")
                    print("Possible stress type:", *(list(chain.from_iterable(x)) for x in comb))
            return (regularity==len(self.indexform))

        except AttributeError:
            print("Set the stress dictionary to use.")

    def WordList(self):
        s = set()
        for line in self.stringform:
            s |= set(line)
        return s
    
    def RhymePair(self):
        pair = []
        paring = [[0,2],[1,3],[4,6],[5,7],[8,10],[9,11],[12,13]]
        for couple in paring:
            i, j = couple
            pair.append({self.stringform[i][-1], self.stringform[j][-1]})
        return pair
