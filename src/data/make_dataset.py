# -*- coding: utf-8 -*-
import click
import logging
import pandas as pd
import re
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
from fuzzywuzzy import fuzz
from nltk.stem import SnowballStemmer
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

snow= SnowballStemmer('english')

stop_words = set(stopwords.words('english'))
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

from pathlib import Path


def get_unigram(question):
    return [word for word in word_tokenize(question.lower()) if word 
            not in stop_words]
def get_common_unigram(row):
    return len(set(row['unigram_ques1']).intersection(
        set(row['unigram_ques2'])))

def get_common_word_ratio(row):
    return row['word_share'] / max(len(list(
        set((row['unigram_ques1'] + row['unigram_ques2'])))),1)

def normalized_word_Total(row):
    w1 = set(map(lambda word: str(word).lower().strip(),
                 row['question1'].split(" ")))
    w2 = set(map(lambda word: str(word).lower().strip(),
                 row['question2'].split(" ")))    
    return 1.0 * (len(w1) + len(w2))

def cleaning(s):
    s = str(s)
    s = s.lower()
    s = re.sub('\s\W',' ',s)
    s = re.sub('\W,\s',' ',s)
    s = re.sub(r'[^\w]', ' ', s)
    s = re.sub("\d+", "", s)
    s = re.sub('\s+',' ',s)
    s = re.sub('[!@#$_\?]', '', s)
    s = s.replace("co","")
    s = s.replace("https","")
    s = s.replace(",","")
    s = s.replace("[\w*"," ")
    s = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", s)
    s = re.sub(r"what's", "what is ", s)
    s = re.sub(r"\'s", " ", s)
    s = re.sub(r"\'ve", " have ", s)
    s = re.sub(r"can't", "cannot ", s)
    s = re.sub(r"n't", " not ", s)
    s = re.sub(r"i'm", "i am ", s)
    s = re.sub(r"\'re", " are ", s)
    s = re.sub(r"\'d", " would ", s)
    s = re.sub(r"\'ll", " will ", s)
    s = re.sub(r",", " ", s)
    s = re.sub(r"\.", " ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\/", " ", s)
    s = re.sub(r"\^", " ^ ", s)
    s = re.sub(r"\+", " + ", s)
    s = re.sub(r"\-", " - ", s)
    s = re.sub(r"\=", " = ", s)
    s = re.sub(r"'", " ", s)
    s = re.sub(r"(\d+)(k)", r"\g<1>000", s)
    s = re.sub(r":", " : ", s)
    s = re.sub(r" e g ", " eg ", s)
    s = re.sub(r" b g ", " bg ", s)
    s = re.sub(r" u s ", " american ", s)
    s = re.sub(r"\0s", "0", s)
    s = re.sub(r" 9 11 ", "911", s)
    s = re.sub(r"e - mail", "email", s)
    s = re.sub(r"j k", "jk", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"what's", "", s)
    s = re.sub(r"\'s", " ", s)
    s = re.sub(r"\'ve", " have ", s)
    s = re.sub(r"can't", "cannot ", s)
    s = re.sub(r"n't", " not ", s)
    s = re.sub(r" m ", " am ", s)
    s = re.sub(r"\'re", " are ", s)
    s = re.sub(r"\'d", " would ", s)
    s = re.sub(r"60k", " 60000 ", s)
    s = re.sub(r"\0s", "0", s)
    s = re.sub(r"e-mail", "email", s)
    s = re.sub(r"\s{2,}", " ", s)
    s = re.sub(r"quikly", "quickly", s)
    s = re.sub(r" usa ", " america ", s)
    s = re.sub(r" uk ", " england ", s)
    s = re.sub(r"imrovement", "improvement", s)
    s = re.sub(r"intially", "initially", s)
    s = re.sub(r" dms ", "direct messages ", s)  
    s = re.sub(r"demonitization", "demonetization", s) 
    s = re.sub(r"actived", "active", s)
    s = re.sub(r"kms", " kilometers ", s)
    s = re.sub(r" cs ", " computer science ", s) 
    s = re.sub(r" upvotes ", " up votes ", s)
    s = re.sub(r" iPhone ", " phone ", s)
    s = re.sub(r"\0rs ", " rs ", s) 
    s = re.sub(r"calender", "calendar", s)
    s = re.sub(r"ios", "operating system", s)
    s = re.sub(r"gps", "GPS", s)
    s = re.sub(r"gst", "GST", s)
    s = re.sub(r"programing", "programming", s)
    s = re.sub(r"bestfriend", "best friend", s)
    s = re.sub(r"III", "3", s) 
    s = re.sub(r"the us", "america", s)
    return s





# =============================================================================
# @click.command()
# @click.argument('input_filepath', type=click.Path(exists=True))
# @click.argument('output_filepath', type=click.Path())
# =============================================================================
def main(input_filepath, output_filepath):
    df = pd.read_csv(input_filepath)
    df = df.sample(frac=0.4).reset_index(drop=True)
    df_clean = df[(df.question1.str.len() >9) & (df.question2.str.len() >9)]
    df_clean['question1'] = [cleaning(s) for s in df_clean['question1']]
    df_clean['question2'] = [cleaning(s) for s in df_clean['question2']]
    df_clean['unigram_ques1'] = df_clean['question1'].apply(
        lambda x : list(set(get_unigram(str(x)))))
    df_clean['unigram_ques2'] = df_clean['question2'].apply(
        lambda x : list(set(get_unigram(str(x)))))
    df_clean['word_share'] = df_clean.apply(lambda x : get_common_unigram(x),
                                            axis=1)
    df_clean2 = df_clean.drop(df_clean[(df_clean['unigram_ques1'].apply(
        lambda x: len(x)) ==0) | (df_clean['unigram_ques2'].apply(
            lambda x: len(x)) ==0) ].index, axis=0)
    
    df_clean2['cleanQ1'] = df_clean2['unigram_ques1'].apply(
        lambda x: " ".join(x))
    df_clean2['cleanQ2'] = df_clean2['unigram_ques2'].apply(
        lambda x: " ".join(x))
    
    df_clean2['common_word_ratio'] = df_clean2.apply(
        lambda x: get_common_word_ratio(x), axis =1)
    
    df_clean2['fuzz_ratio'] = df_clean2.apply(
        lambda x : fuzz.ratio(x['cleanQ1'],x['cleanQ2']), axis=1)
    df_clean2['Partial_Ratio']=df_clean2.apply(
        lambda x: fuzz.partial_ratio(x['cleanQ1'],x['cleanQ2']) ,axis=1)
    df_clean2['Token_Sort_Ratio']=df_clean2.apply(
        lambda x: fuzz.token_sort_ratio(x['cleanQ1'],x['cleanQ2']) ,axis=1)

    df_clean2['Token_Set_Ratio']=df_clean2.apply(
        lambda x: fuzz.token_set_ratio(x['cleanQ1'],x['cleanQ2']) ,axis=1)
    
    
    
    df_data = df_clean2[['id', 'is_duplicate', 'common_word_ratio', 
             'fuzz_ratio', 'Partial_Ratio', 'Token_Sort_Ratio', 
         'Token_Set_Ratio', 'cleanQ1', 'cleanQ2']]
    
    df_data.to_csv(output_filepath)
    
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """



if __name__ == '__main__':
    
    p = Path(__file__).parents[2]
    
    print(p)
    '''log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
   # load_dotenv(find_dotenv())'''

    main(str(p)+"/data/raw/train.csv", 
         str(p)+"/data/processed/clean_data_code.csv")
