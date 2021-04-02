import math
import re

import nltk
from nltk.stem.porter import *


def process_doc(content):
    """
    Processes the content of a document
    :param content: content string of a document (or query)
    :return: dictionary with term frequency in this document, the positions of terms in the document,
     and the lengths of vector of the document
    """
    doc_dict = {}
    doc_positions = {}
    stemmer = PorterStemmer()
    words = nltk.word_tokenize(content)
    size = len(words)
    alnum_regex = re.compile('[^a-zA-Z0-9]')
    skip_counter = 0
    positional_index = 0
    for index, word in enumerate(words):
        if skip_counter > 0:
            skip_counter -= 1
            continue
        if index < size - 2 and word == '(' and words[index + 2] == ')' and is_enum(words[index + 1]):
            skip_counter = 2
            continue
        word = alnum_regex.sub('', stemmer.stem(word.lower()))
        ws = re.match(r"([0-9]+)([a-z]+)", word, re.I)
        if ws:
            ws = ws.groups()
            for w in ws:
                if w != '':
                    if w not in doc_dict:
                        doc_dict[w] = 0
                        doc_positions[w] = []
                    doc_dict[w] += 1
                    doc_positions[w].append(positional_index)
                    positional_index += 1
        elif word != '':
            if word not in doc_dict:
                doc_dict[word] = 0
                doc_positions[word] = []
            doc_dict[word] += 1
            doc_positions[word].append(positional_index)
            positional_index += 1
    acc = 0
    for word in doc_dict:
        tf = 1 + math.log(doc_dict[word], 10)
        acc += tf * tf
    return doc_dict, doc_positions, math.sqrt(acc)


def is_enum(token):
    """
    Checks whether the token can together with brackets at the sides form a enumeration index
    :param token: the token to check
    :return: true if it is a number, or a single letter, or a Roman numeral
    """
    if token.isdigit():
        return True
    if token.isalpha():
        if len(token) == 1:
            return True
        token = token.upper()
        regex = re.compile('^(?=[MDCLXVI])M*(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[XV]|V?I{0,3})$')
        if regex.match(token):
            return True
        return False
