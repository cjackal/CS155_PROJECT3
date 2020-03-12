# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 15:15:31 2020

@author: HyeongChan Jo
"""

'''
Utility for Sonnet analysis
'''

import re
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

def SonnetLoader(path):
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
            lines[i] = re.sub("^\s+", '', lines[i]).lower()
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

    #print(sonnets)
    return [Sonnet(sonnet) for sonnet in sonnets]