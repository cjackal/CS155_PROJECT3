# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:15:31 2020

@author: HyeongChan Jo
"""

'''
Utility for Sonnet analysis
'''

import re
import pandas as pd
from Sonnet import Sonnet

def parse_observations(text):
    # Convert text to dataset.
    lines = [line.split() for line in text.split('\n') if line.split()]

    obs_counter = 0
    obs = []
    obs_map = {}

    for line in lines:
        obs_elem = []
        
        for word in line:
            word = re.sub(r'[^\w]', '', word).lower()
            if word not in obs_map:
                # Add unique words to the observations map.
                obs_map[word] = obs_counter
                obs_counter += 1
            
            # Add the encoded word.
            obs_elem.append(obs_map[word])
        
        # Add the encoded sequence.
        obs.append(obs_elem)

    return obs, obs_map

def SonnetLoader(path, syl_dict=[]):
    """
    Load sonnets from txt, return a list consisting of one sonnet per element.
    Each sonnet consists of a list of lines in the sonnet, which is again a list of words in the line.
    TODO: There are apostrophes for possessive form of nouns or quotation marks, which must be distinguished from apostrophe for omission.
    Probably the easiest cure is deleting them manually before loading the sonnets?
    I have checked that sonnet 2, 8 and 145 have a quotation and 14 has a possesive noun (not that no more though).
    Why this a problem? Run "IsRegular" function on the second sonnet and you will see.
    """
    sonnets = []
    if path[0]!='.':
        path = './data/' + path + '.txt'
    with open(path) as f:
        txt = f.read()
        lines = txt.split('\n')
        for i in range(len(lines)):
            lines[i] = re.sub(r"^\s+", '', lines[i]).lower()
        beginning = 0
        sonnet_is_read = False
        i = 0
        while i<len(lines):
            if lines[i].isdigit():
                beginning = i
                sonnet_is_read = True
            elif len(lines[i])==0:
                if sonnet_is_read:
                    sonnets.append(lines[beginning+1:i])
                    sonnet_is_read = False
            i+=1
        if sonnet_is_read:
            sonnets.append(lines[beginning+1:])
    f.close()
    for sonnet in sonnets:
        for i in range(len(sonnet)):
            sonnet[i] = re.sub(r"[^-'\w\s]", '', sonnet[i]).split()
            # line = []
            # for word in sonnet[i]:
            #     line.append(re.sub(r"s'$", "s", word))
            # sonnet[i] = line
    if len(syl_dict)==0:
        return [Sonnet(sonnet) for sonnet in sonnets]
    else:
        sonnetList = [Sonnet(sonnet) for sonnet in sonnets]
        for sonnet in sonnetList:
            sonnet.SetDict(syl_dict)
        return sonnetList

def DictLoader(path, sep='@'):
    if path[0]!='.':
        path = './models/' + path + '.csv'
    df = pd.read_csv(path, sep=sep, index_col=0)
    if df.columns[0]=='stress':
        for i in range(len(df)):
            df.iloc[i, 0] = eval(df.iloc[i, 0])
    return df

def Roman_Decimal(s):
    roman = 0
    i = 0

    value = {'I':1, 'V':5, 'X':10, 'L':50, 'C':100}
    
    while i < len(s):     
        s1 = value[s[i]] 
        if i+1 < len(s): 
            s2 = value[s[i+1]] 
            if s1 >= s2: 
                roman += s1 
                i += 1
            else: 
                roman += s2 - s1 
                i += 2
        else: 
            roman += s1 
            i += 1
    return roman