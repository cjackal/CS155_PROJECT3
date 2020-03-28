# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 16:06:45 2020

@author: HyeongChan Jo
"""

import pandas as pd
import nltk
from nltk.corpus import cmudict
import Dictionary

#nltk.test.unit.test_tokenize.
from nltk.tokenize import SyllableTokenizer 

SSP = SyllableTokenizer()
test = SSP.tokenize(("graceth"))
print(test)




#test = ["th'utmost", "'twixt", 'budded', 'eye-glances', 'scath', 'comfortless', 'arion', 'craftily', 'thereunto', 'ravished', "t'accuse", 'embaseth', 'encage', "janus'", 'outworn', "soul's", 'didst', "t'abide", 'tuneless', 'unwarily', 'whylest', 'soom', 'footstool', 'thralls', "th'only", "hearts'", 'lothsome', 'lillies', "t'adorn", 'unquiet', 'paps', 'freewill', 'reascend', 'foully', 'honour', 'languishment', 'love-pined', 'unreave', 'lusts', 'aread', "flesh's", "phoebus'", 'wouds', "virtue's", 'plenteous', 'eek', 'captiving', 'misintended', 'misseth', 'pleasauns', 'erst', 'priefe', 'vouchsafe', 'remembreth', 'lioness', 'resounded', 'thyself', 'drawen', 'woodbine', 'glooming', 'guilefull', 'tormenteth', 'heares', 'aright', 'cheerless', 'immortally', "renew'th", 'love-afamished', 'lordeth', 'requite', 'therewith', 'unpityed', 'drizling', 'betokening', 'flits', 'storm-beaten', 'queene', 'disconsolate', 'cunningly', 'dints', 'disdaineth', 'faery', 'plast', 'whenas', 'humbless', "tempests'", "th'anvil", 'enlumined', 'wonts', 'wooers', 'succour', 'sufficeth', 'aught', "treason's", 'piteous', 'whilest', 'longwhile', 'youself', "th'importune", "th'image", 'outwent', 'shinedst', 'antick', 'depraves', 'plaints', 'greediness', 'portliness', 'thereto', 'nought', 'dispise', 'recure', "angel's", "bath'd", 'eternize', 'tempests', 'gladsome', 'descry', 'snaky', 'half-trembling', 'covetize', "pleasure's", 'amearst', "the'accomplishment", 'needeth', 'abhored', 'compassed', 'extermities', 'whereto', 'valorous', 'surcease', 'pitty', 'enbrew', 'godess', "t'increase", 'whereof', 'three-score', 'unvalued', 'persever', "abondon'd", 'cockatrices', 'languor', 'rues', "poets'", 'archers', 'belay', "thought's", "craftsman's", 'extremest', 'so-hot', 'woxen', 'embrew', 'pricketh', 'thralled', 'felly', "viper's", 'firbloom', "adders'", "fram'd", "'scaped", 'frieseth', 'twixt', 'upbrought', 'unspotted', 'disparagement', 'likest', 'revengeful', 'fordone', 'forelock', 'eyen', "ulysses'", 'lustfull', 'assayed', 'handmaid', 'self-same', 'emprize', 'jove', "danger's", 'long-while', 'noyous', 'carefull', 'wretches', "spider's", 'visnomy', 'dead-doing', 'heretics', 'disobeys', 'beseen', 'misdeem', 'forhead', 'broom-flower', 'meekness', 'defiled', "sorrow's", 'saphires', 'aswagement', 'diversely', 'pityless', 'makest', "th'author", 'lowliness', 'doest', 'harrowed', 'blissing', 'captived', 'flys', 'furies', 'excellect', 'embase', 'jasmines', 'gladness', 'matchable', 'quod', 'assoyle', 'intreat', "look's", 'thm', 'boldened', 'bellamores', 'seemeth', 'guileful', 'endite', 'faedry', 'unwares', 'self-pleasing', 'shewed', 'comptroll', 'knowen', 'indias', 'ensample', 'sithens', 'sdeigne', 'coat-armor', 'graceth', 'aslake', 'sprites', 'false-forged', 'gazers', "fancy's", 'reighneth', 'augmenteth', 'despight', 'runneth', "starv'd", "blam'd", 'tyraness', 'colours', 'lurkest', 'baseness', 'thessalian', 'prefixed', 'unrighteous', 'tradefull', 'mayst', "'gainst", 'dumpish', 'rich-laden', 'worthily', 'accursed', "t'achieve", 'forepast', 'dight', 'sith', 'long-lacked', 'damsels', 'venemous', 'contrained', 'loosly', 'drossy', "'mongst", "adorn'd", 'plights', 'rend', "'stonished", 'pleasance', 'maketh', 'plaint', 'cruelness', 'ruinate', 'durefull', 'enchased', 'langor', "th'", 'quoth', 'meeds', "lov'd", 'mought', 'allurement', 'meede', "deriv'd", "abode's", 'heart-thrilling', 'stoures', "enur'd", 'despoiled', 'elizabeths', 'aswage', 'pensiveness', "thund'ring", "beauty's", "wretch's", 'mazed', 'persueth', 'doubtfully', 'close-bleeding', 'meed', 'weene', 'juncats', 'gillyflowers', "foes'", 'love-learned', 'scorning', 'troublous', 'entreat', 'warried', 'rashly', "lookers'", 'langour', 'mantleth', 'wrongest']
#test_syllable = SSP.tokenize(test)
#print(test_syllable)


test= Dictionary.sylAndStr_nltk(["apple", "graceth", "fire"])
print(test)