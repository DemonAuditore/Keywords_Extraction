# -*- coding: UTF-8 -*-

import sys
import numpy as np
import pandas as pd
import os
import nltk
import string
import xlrd

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
        df = pd.read_excel(file)
        # book = xlrd.open_workbook(file)
        # # xlrd用于获取每个sheet的sheetname
        # for sheet in book.sheets():
        #     df = pd.read_excel(file, sheet.name)
        #     if df.empty:
        #         continue
        file_i.append(df)
    return file_i

def find_key_words_from_text(content, Compare):
    r = Rake(ranking_metric=Metric.TF_IDF, COMP_FLAG=Compare)
    r.extract_keywords_from_list(content)
    return r.get_ranked_phrases_with_scores()

def self_excel_writer(COMP, fileName, content):
    s = 'keywords_{a}.xlsx'
    writer = pd.ExcelWriter(s.format(a=fileName.split('.')[0]))
    if COMP:
        entities_list, words_list, whole_list = find_key_words_from_text(content, COMP)
        ##  unzip data
        entities_list = list(zip(*entities_list))
        words_list = list(zip(*words_list))
        whole_list = list(zip(*whole_list))
        ##  write data into excel files
        if whole_list:
            # whole keywords list
            whole_dict = {'热词': list(whole_list[0]),
                          'TF-IDF': list(whole_list[1]),
                          '词频': list(whole_list[2])
                          }
            data_whole = DataFrame(whole_dict)
            data_whole.to_excel(writer, 'keywords', index=False)
        if entities_list:
            # entities keywords list
            entities_dict = {'实体热词': list(entities_list[0]),
                             'TF-IDF-实体': list(entities_list[1]),
                             '词频-实体': list(entities_list[2])
                             }
            data_entities = DataFrame(entities_dict)
            data_entities.to_excel(writer, 'entities', index=False)
        if words_list:
            # non-entities keywords list
            words_dict = {'非实体热词': list(words_list[0]),
                          'TF-IDF-非实体': list(words_list[1]),
                          '词频-非实体': list(words_list[2])
                          }
            data_words = DataFrame(words_dict)
            data_words.to_excel(writer, 'non-entities', index=False)
    else:
        if key_words_list:
            key_words_list = find_key_words_from_text(content, COMP)
            key_words_list = list(zip(*key_words_list))
            key_words_dict = {'热词': list(key_words_list[0]),
                              'TF-IDF': list(key_words_list[1]),
                              '词频': list(key_words_list[2])
                              }
            data_keywords = DataFrame(key_words_dict)
            data_keywords.to_excel(writer, 'keywords', index=False)
    writer.save()


if __name__ == "__main__":
    COMP = True
    file_i = foo()
    for i,file in enumerate(file_i):
        head_data = file.head(200)
        content = DataFrame(head_data, columns=[col_name])
        fileName = file_name[i]
        self_excel_writer(COMP, fileName, content)





