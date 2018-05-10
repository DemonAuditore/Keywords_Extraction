# -*- coding: utf-8 -*-
"""Implementation of Rapid Automatic Keyword Extraction algorithm.
As described in the paper `Automatic keyword extraction from individual
documents` by Stuart Rose, Dave Engel, Nick Cramer and Wendy Cowley.
"""

import string
from collections import Counter, defaultdict
from itertools import chain, groupby, product

import nltk
from enum import Enum
from nltk.tokenize import wordpunct_tokenize

from pandas import Series, DataFrame
from collections import defaultdict

from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

from langid.langid import LanguageIdentifier, model

import numpy as np

import sys
import re

from nltk.tag.stanford import StanfordNERTagger
from collections import Counter, defaultdict
import os
from itertools import chain, groupby, product


JAVA_PATH = "C:/Program Files/Java/jdk1.8.0_92/bin/java.exe"
os.environ['JAVAHOME'] = JAVA_PATH



class Metric(Enum):
    """Different metrics that can be used for ranking."""

    WORD_FREQUENCY = 0  # Uses f(w) as the metric
    TF_IDF = 1  # Uses TF-IDF as the metric
    TERM_FREQUENCY = 2  # Uses TF-IDF as the metric


class Rake(object):
    """Rapid Automatic Keyword Extraction Algorithm."""

    def __init__(
            self,
            stopwords=None,
            punctuations=None,
            language="english",
            ranking_metric=Metric.WORD_FREQUENCY,
            col_name='内容',
            rank = 1,

    ):
        """Constructor.
        :param stopwords: List of Words to be ignored for keyword extraction.
        :param punctuations: Punctuations to be ignored for keyword extraction.
        :param language: Language to be used for stopwords
        """
        # By default use degree to frequency ratio as the metric.
        if isinstance(ranking_metric, Metric):
            self.metric = ranking_metric
        else:
            self.metric = Metric.WORD_FREQUENCY

        # If stopwords not provided we use language stopwords by default.
        self.stopwords = stopwords
        if self.stopwords is None:
            self.stopwords = nltk.corpus.stopwords.words(language)

        # If punctuations are not provided we ignore all punctuation symbols.
        self.punctuations = punctuations
        if self.punctuations is None:
            self.punctuations = string.punctuation

        # All things which act as sentence breaks during keyword extraction.
        self.to_ignore = set(chain(self.stopwords, self.punctuations,[',"']))

        # Stuff to be extracted from the provided text.
        self.frequency_dist = None
        self.tfidf_dict = None
        # self.frequency_dict = None
        self.NE_dict = defaultdict(lambda : 'O')
        self.tf_dict = None
        self.degree = None
        self.rank_list = None
        self.ranked_phrases = None

        # Stuff for pandas
        self.col_name = col_name
        self.rank = rank

    def extract_keywords_from_list(self, content):
        """Method to extract keywords from the text list.
        :param text: Text to extract keywords from, provided as a string.
        """
        text_list = self.get_textlist_from_dataframe(content)
        phrase_list = []
        for text in text_list:
            if not self.language_identification(text):
                continue
            phrase_list_text = self.extract_keywords_from_text(text)
            phrase_list.append('`'.join(list(phrase_list_text)))

        if phrase_list:
            self._build_tfidf_dic(phrase_list)
            phrase_list = [val for sublist in phrase_list for val in sublist.split('`')]
            self._build_frequency_dist_list(phrase_list)
            self._build_ranklist(phrase_list)

    def language_identification(self, text):
        identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
        result = identifier.classify(text)[0]
        if result == 'en':
            return True
        return False

    def get_textlist_from_dataframe(self, content):
        con_list = []
        for index, row in content.iterrows():
            con = row[self.col_name]
            if con is np.nan:
                continue

            con_list.append(con)
        # print(len(con_list))
        # sys.exit(0)
        return con_list

    def extract_keywords_from_text(self, text):
        """Method to extract keywords from the text provided.
        :param text: Text to extract keywords from, provided as a string.
        """
        sentences = nltk.tokenize.sent_tokenize(text)
        phrase_list = self.extract_keywords_from_sentences(sentences)
        return phrase_list

    def extract_keywords_from_sentences(self, sentences):
        """Method to extract keywords from the list of sentences provided.
        :param sentences: Text to extraxt keywords from, provided as a list
                          of strings, where each string is a sentence.
        """
        phrase_list = self._generate_phrases(sentences)
        return phrase_list

    def get_ranked_phrases(self):
        """Method to fetch ranked keyword strings.
        :return: List of strings where each string represents an extracted
                 keyword string.
        """
        return self.ranked_phrases

    def get_ranked_phrases_with_scores(self, NE_FLAG = False):
        """Method to fetch ranked keyword strings along with their scores.
        :return: List of tuples where each tuple is formed of an extracted
                 keyword string and its score. Ex: (5.68, 'Four Scoures')
        """
        if not self.rank_list or len(self.rank_list) < self.rank:
            return []
        return self.rank_list[:self.rank]

    def get_word_frequency_distribution(self):
        """Method to fetch the word frequency distribution in the given text.
        :return: Dictionary (defaultdict) of the format `word -> frequency`.
        """
        return self.frequency_dist

    def get_tfidf_dictionary(self):
        """Method to fetch the word frequency distribution in the given text.
        :return: Dictionary (defaultdict) of the format `word -> frequency`.
        """
        return self.tfidf_dict

    def get_named_entities_list(self):
        """Method to fetch the word frequency distribution in the given text.
        :return: Dictionary (defaultdict) of the format `word -> frequency`.
        """

        return [self.NE_dict[ph[1]] for ph in self.rank_list]

    def _build_tfidf_dic(self, phrase_list):
        """Builds tfidf dictionary of the phrases in the given text list.
           :param phrase_list: List of strings where each string is a group
           of phrases that are separated by '`', for example: 'red apple`green
            apple`yellow apple'.
           :result: Dictionary
        """
        tfidf_dict = defaultdict(lambda : 0)
        # tf_dict = defaultdict(lambda: 0)

        vectorizer = CountVectorizer(token_pattern='(?u)\w+[\s\`]\w+')
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform(phrase_list))

        words = vectorizer.get_feature_names()
        weight = tfidf.toarray()
        # print(len(words))
        # print(len(tfidf.toarray().sum(axis=0)))

        for i in range(len(weight)):
            for j in range(len(words)):
                tfidf_dict[words[j]] = weight[i][j]
        self.tfidf_dict = tfidf_dict


    def _build_frequency_dist_list(self, phrase_list):
        """Builds frequency distribution of the words in the given body of text.
        :param phrase_list: List of List of strings where each sublist is a
                            collection of words which form a contender phrase.
        """
        #  wf[phrase] = sum(wf[word]) for word in phrase
        phrase_list = [tuple(x.split()) for x in phrase_list]
        phrase_list = set(phrase_list)
        self.frequency_dist = Counter(chain.from_iterable(phrase_list))




    def _build_ranklist(self, phrase_list):
        """Method to rank each contender phrase using the formula
              phrase_score = sum of scores of words in the phrase.
              word_score = d(w)/f(w) where d is degree and f is frequency.
        :param phrase_list: List of List of strings where each sublist is a
                            collection of words which form a contender phrase.
        """
        self.rank_list = []
        # phrase_list = [val for sublist in phrase_list for val in sublist.split('`')]
        # phrase_set = set(phrase_list)
        phrase_list = list(set(phrase_list))
        for phrase in phrase_list:
            rank = 0.0
            if self.metric == Metric.TF_IDF:
                rank += 1.0 * self.tfidf_dict[phrase]
            if self.metric == Metric.WORD_FREQUENCY:
                if self.NE_dict[phrase] == 'O':
                    coe = 1.0
                else:
                    coe = 0.5
                for word in phrase.split():
                    rank += 1.0 * self.frequency_dist[word]
                rank = rank * coe
            self.rank_list.append((rank, phrase))
        self.rank_list.sort(reverse=True)
        self.ranked_phrases = [ph[1] for ph in self.rank_list]


    def _generate_phrases(self, sentences):
        """Method to generate contender phrases given the sentences of the text
        document.
        :param sentences: List of strings where each string represents a
                          sentence which forms the text.
        :return: Set of string tuples where each tuple is a collection
                 of words forming a contender phrase.
        """
        phrase_list = set()
        # phrase_list = []
        # Create contender phrases from sentences.
        for sentence in sentences:
            word_list = [word for word in wordpunct_tokenize(sentence)]
            # print(self._get_phrase_list_from_words(word_list))
            # sys.exit(0)
            phrase_list.update(self._get_phrase_list_from_words(word_list))
        return phrase_list

    def _get_phrase_list_from_words(self, word_list):
        """Method to create contender phrases from the list of words that form
        a sentence by dropping stopwords and punctuations and grouping the left
        words into phrases. Ex:
        Sentence: Red apples, are good in flavour.
        List of words: ['red', 'apples', ",", 'are', 'good', 'in', 'flavour']
        List after dropping punctuations and stopwords.
        List of words: ['red', 'apples', *, *, good, *, 'flavour']
        List of phrases: ['red apples', 'good', 'flavour']
        :param word_list: List of words which form a sentence when joined in
                          the same order.
        :return: List of contender phrases that are formed after dropping
                 stopwords and punctuations.
        """
        tagger = StanfordNERTagger(
            'E:\Learning Resources\TOOLS\stanford-ner-2015-12-09\classifiers\english.all.3class.distsim.crf.ser.gz',
            'E:\Learning Resources\TOOLS\stanford-ner-2015-12-09\stanford-ner.jar',
            encoding='utf-8')

        phrase_list = []

        classified_paragraphs_list = tagger.tag_sents([word_list])
        groups = groupby(classified_paragraphs_list[0], lambda x: x[1])
        for key, group in groups:
            group_list = [g[0].lower() for g in group]
            if key is 'O':
                rs_groups = groupby(group_list, lambda x: x not in self.to_ignore and re.match(r'[a-zA-Z0-9]+',x) != None)
                rs_group_list = [' '.join(list(rs_group[1])) for rs_group in rs_groups if rs_group[0]]

                phrase_list.extend(rs_group_list)
                # phrase_list.extend(group_list)
            else:
                phrase = ' '.join(group_list)
                phrase_list.append(phrase)
                self.NE_dict[phrase] = key

        return phrase_list
        # return phrase_list

        # groups = groupby(word_list, lambda x: x not in self.to_ignore)
        # return [' '.join(list(group[1])) for group in groups if group[0]]
