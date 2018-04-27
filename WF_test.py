# -*- coding: UTF-8 -*-

import sys
import numpy as np
import pandas as pd
import os
import nltk
import string

from pandas import Series, DataFrame
from nltk.corpus import stopwords
from rake_test import Metric, Rake
# from rake_nltk import Metric, Rake


##  get file path
dir_path = r'E:\Learning Resources\CASIA\word frequency\test_file' # direct path
file_name = os.listdir(dir_path)    # return all the files and filenames
file_list = [os.path.join(dir_path, x) for x in file_name]  # return all the file path

##  get column name
col_name = '内容'


def foo():
    file_i = []
    for file in file_list:
        file_i.append(pd.read_excel(file))
    return file_i

def find_key_words_from_text(content, rank):
    r = Rake(ranking_metric=Metric.WORD_FREQUENCY)
    r.extract_keywords_from_list(content)
    return r.get_ranked_phrases_with_scores()[:rank]

def entity_reco():
    pass

if __name__ == "__main__":
    file_i = foo()
    for file in file_i:
        head_data = file.head(5)
        content = DataFrame(head_data, columns=[col_name])
        key_words_list = find_key_words_from_text(content, 20)
        key_words_dict = {'热词': [key_word[1] for key_word in key_words_list],'TF-IDF':[key_word[0] for key_word in key_words_list]}
        data = DataFrame(key_words_dict)
        data.to_csv("test.csv", index=False, sep=',')





