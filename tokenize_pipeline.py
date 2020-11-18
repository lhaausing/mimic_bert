import re
import calendar
from collections import Counter
from tqdm import tqdm
import scispacy #https://allenai.github.io/scispacy/
import en_core_sci_md
import spacy
import pandas as pd


email_map = 'mail'
org_map = ['hospital', 'company', 'university']
location_map = ['address', 'location' 'ward', 'state', 'country']
name_map = 'name'
number_map = ['number', 'telephone']
date_map = ['month', 'year', 'day'] + [calendar.month_name[i].lower() for i in range(1,13)]

def filter_brackets(text):

    unk_list = set(re.findall(r'\[\*.+?\*\]', text))
    unk_map = {}
    for elem in unk_list:
        find = False
        elem = elem.lower()
        if email_map in elem:
            unk_map[elem] = 'email'
            continue
        for w in org_map:
            if w in elem:
                unk_map[elem] = 'unknown {}'.format(w)
                find = True
                continue
        if find: continue
        for w in location_map:
            if w in elem:
                unk_map[elem] = 'unknown {}'.format(w)
                find = True
                continue
        if find: continue
        if name_map in elem:
            unk_map[elem] = 'unknown person'
            continue
        for w in date_map:
            if w in elem or re.search('[a-zA-Z]', elem) is None:
                unk_map[elem] = 'unknown date'
                find = True
                continue
        if find: continue
        unk_map[elem] = '[UNK]'
    for elem in unk_list:
        text = text.replace(elem, unk_map[elem])
    text = text.replace("unknown person unknown person", "unknown person")
    return text



replace_LIST = [['dr\.',''] ,['DR\.','']
                ,['m\.d\.',''] ,['M\.D\.','']
                ,['yo', 'years old. ']
                ,['p\.o', 'orally. ']
                ,['P\.O', 'orally. ']
                ,['po', 'orally. ']
                ,['PO', 'orally. ']
                ,['q\.d\.', 'once a day. ']
                ,['Q\.D\.', 'once a day. ']
                ,['qd', 'once a day. ']
                ,['QD', 'once a day. ']
                ,['I\.M\.', 'intramuscularly. ']
                ,['i\.m\.', 'intramuscularly. ']
                ,['b\.i\.d\.', 'twice a day. ']
                ,['B\.I\.D\.', 'twice a day. ']
                ,['bid', 'twice a day. ']
                ,['BID', 'twice a day. ']
                ,['Subq\.', 'subcutaneous. ']
                ,['SUBQ\.', 'subcutaneous. ']
                ,['t\.i\.d\.', 'three times a day. ']
                ,['tid', 'three times a day. ']
                ,['T\.I\.D\.', 'three times a day. ']
                ,['TID', 'three times a day. ']
                ,['q\.i\.d\.', 'four times a day. ']
                ,['Q\.I\.D\.', 'four times a day. ']
                ,['qid', 'four times a day. ']
                ,['QID', 'four times a day. ']
                ,['I\.V\.', 'intravenous. ']
                ,['i\.v\.', 'intravenous. ']
                ,['q\.h\.s\.', 'before bed. ']
                ,['Q\.H\.S\.', 'before bed. ']
                ,['qhs', 'before bed. ']
                ,['Qhs', 'before bed. ']
                ,['QHS', 'before bed. ']
                ,[' hr ', ' hours ']
                ,[' hrs ', ' hours ']
                ,['hr(s)', 'hours']
                ,['O\.D\.', 'in the right eye. ']
                ,['o\.d\.', 'in the right eye. ']
                ,['OD', 'in the right eye. ']
                ,['od', 'in the right eye. ']
                ,['5X', 'a day five times a day. ']
                ,['5x', 'a day five times a day. ']
                ,['OS', 'in the left eye. ']
                ,['os', 'in the left eye. ']
                ,['q\.4h', 'every four hours. ']
                ,['Q\.4H', 'every four hours. ']
                ,['q24h', 'every 24 hours. ']
                ,['Q24H', 'every 24 hours. ']
                ,['q4h', 'every four hours. ']
                ,['Q4H', 'every four hours. ']
                ,['O\.U\.', 'in both eyes. ']
                ,['o\.u\.', 'in both eyes. ']
                ,['OU', 'in both eyes. ']
                ,['ou', 'in both eyes. ']
                ,['q\.6h', 'every six hours. ']
                ,['Q\.6H', 'every six hours. ']
                ,['q6h', 'every six hours. ']
                ,['Q6H', 'every six hours. ']
                ,['q\.8h', 'every eight hours. ']
                ,['Q\.8H', 'every eight hours. ']
                ,['q8h', 'every eight hours. ']
                ,['Q8H', 'every eight hours. ']
                ,['q8hr', 'every eight hours. ']
                ,['Q8hr', 'every eight hours. ']
                ,['Q8HR', 'every eight hours. ']
                ,['q\.12h', 'every 12 hours. ']
                ,['Q\.12H', 'every 12 hours. ']
                ,['q12h', 'every 12 hours. ']
                ,['Q12H', 'every 12 hours. ']
                ,['q12hr', 'every 12 hours. ']
                ,['Q12HR', 'every 12 hours. ']
                ,['Q12hr', 'every 12 hours. ']
                ,['q\.o\.d\.', 'every other day. ']
                ,['Q\.O\.D\.', 'every other day. ']
                ,['qod', 'every other day. ']
                ,['QOD', 'every other day. ']
                ,['prn', 'as needed.']
                ,['PRN', 'as needed.']
                ,['[0-9]+\.','']]

def preprocess_re_sub(x):
    processed_text = x
    for find,replace in replace_LIST:
        processed_text=re.sub(find,replace,processed_text)
    return processed_text

def preprocess_replace(text):
    text = preprocess_re_sub(text).strip()
    text = re.sub('\t|\,|\?|admission date:|discharge date:|date of birth:|addendum:|--|__|==', '',text.lower())
    text = text.replace('\n\n', '<parah>')
    text = text.replace('\n', ' ')
    text = text.replace(":", ' : ')
    for i in range(3):
        text = text.replace('  ', ' ')
    text = text.split('<parah>')
    text = [elem.strip() for elem in text if elem != '']
    return '\n'.join(text)

nlp = spacy.load('en_core_sci_md', disable=['tagger','ner'] )

pattern = re.compile('[^A-Za-z0-9 \n\t\r\v\f]')


#process given sentence
line = filter_brackets(line)
line = preprocess_replace(line)

#use scispacy to tokenize sentence
line = str(nlp(line))

#add space after and before all the punctuation
pattern_match = set(pattern.findall(line))
for i in pattern_match:
    line = line.replace(i, ' '+i+' ')
line = line.replace('[ unk ]', '[unk]')

#replace number with [num] token
line = re.sub(r"\d+\.?\d*", ' [num] ', line)

#add cls
line = '[cls] ' + line + ' \n'


