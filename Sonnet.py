# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:08:52 2020

@author: HyeongChan Jo, Juhyun Kim
"""

import re

## Master class 'sonnet'

class Sonnet:
#    Parameters:
#            stringform: list of strings. sonnet as a list of words
#            
#            is_ending:  list of logical vluases. Whether the sonnet is ending. False by default, and True only at the end of each line
#            
#            dict:       predefined dictionary for syllable
#            
#            index_map:  map for converting word to index. self.index_map[word] corresponds to index
#            
#            word_to_index:    sonnet with words replaced with the corresponding idx
#
#            WordList:   list of unique words in the sonnet
#   
#            RhymePair:  Pair of words that rhymes in the sonnet

    def __init__(self, sonnet, predefinedDict=[]):
        self.stringform = sonnet        ### sonnet as a list of words itself
        is_ending = [[False for _ in range(len(line))] for line in sonnet]
        for line in is_ending:
            line[-1] = True
        self.is_ending = is_ending      ### Encoding the location of the end of each lines (having the same shape as stringform)
        if len(predefinedDict)!=0:
            self.SetDict(predefinedDict)

    def __repr__(self):
        s = ''
        for line in self.stringform:
            for word in line:
                s += word+' '
            s += '\n'
        return s
    
    def SetDict(self, df, removeApo=True):              ### Set the syllable dictionary.
        self.dict = df                  ### Temporary format: rows indexed by the words, with two columns of possible syllables
        idxmap = {}
        for i, s in enumerate(self.dict.index.to_numpy()):
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
                            word_to_idx_line.append(self.index_map.get(re.sub(r"'$", "", word)))
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

        self.word_to_index = word_to_idx    ### sonnet with words replaced with the corresponding idx

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
