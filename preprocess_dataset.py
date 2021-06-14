"""
Module Docstring
"""

__author__ = "Your Name"
__version__ = "0.1.0"
__license__ = "MIT"

import argparse

import pandas as pd
import numpy as np

import string
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

stop_words = set(stopwords.words('english'))
punctuation_table = str.maketrans(
    {key: None for key in string.punctuation})
sno = SnowballStemmer('english')


def main(args):
    descriptions_list = list(args.df['description'])
    descriptions_list = [str(r) for r in descriptions_list]
    full_corpus = ' '.join(descriptions_list)
    sentences_tokenized = sent_tokenize(full_corpus)


def normalize_text(text):
    try:
    word_list = word_tokenize(raw_text)
    normalized_sentence = []
    for w in word_list:
        try:
            w = str(w)
            lower_case_word = str.lower(w)
            stemmed_word = sno.stem(lower_case_word)
            no_punctuation = stemmed_word.translate(punctuation_table)
            if len(no_punctuation) > 1 and no_punctuation not in stop_words:
                normalized_sentence.append(no_punctuation)
        except:
            continue
    return normalized_sentence
    except:
        return ''


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument("arg", help="Required positional argument")

    # Optional argument flag which defaults to False
    parser.add_argument("-f", "--flag", action="store_true", default=False)

    # Optional argument which requires a parameter (eg. -d test)
    parser.add_argument("-n", "--name", action="store", dest="name")

    # Optional verbosity counter (eg. -v, -vv, -vvv, etc.)
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbosity (-v, -vv, etc)")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()
    main(args)
