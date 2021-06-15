"""
Module Docstring
"""

__author__ = "Team Baard"
__version__ = "0.1.0"
__license__ = "MIT"

import argparse
import os
import requests
import shutil
import pickle
import string
import itertools

import pandas as pd
import numpy as np

from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.metrics.pairwise import cosine_similarity


def load_tf_idf_weights(pkl='models/vectorizer.pickle'):
    tf_idf = pickle.load(open(pkl, "rb"))
    return dict(zip(tf_idf.get_feature_names(), tf_idf.idf_))


def load_ngrams(ngram='models/ngrams'):
    return Phrases.load(ngram)


def load_descriptor_map(dmap='models/descriptor_mapping.csv'):
    return pd.read_csv(dmap).set_index('raw descriptor')


def preprocess_description(description, ngram, descriptor_map, level=3):
    tokens = tokenize_description(description)
    phrase = ngram[tokens]
    descriptors = [map_descriptor(word, descriptor_map, level)
                   for word in phrase]
    descripters_cleaned = [str(desc)
                           for desc in descriptors if desc is not None]

    return descripters_cleaned


def tokenize_description(description):
    stop_words = set(stopwords.words('english'))
    punctuation_table = str.maketrans(
        {key: None for key in string.punctuation})
    stemmer = SnowballStemmer('english')

    normalized_description = []
    word_tokens = word_tokenize(description)
    for word in word_tokens:
        lower_case = str.lower(str(word))
        stemmed_word = stemmer.stem(lower_case)
        no_punctuation = stemmed_word.translate(punctuation_table)
        if len(no_punctuation) > 1 and no_punctuation not in stop_words:
            normalized_description.append(no_punctuation)

    return normalized_description


def map_descriptor(word, mapping, level=3):
    if word in list(mapping.index):
        return mapping[f'level_{level}'][word]


def main(args):
    # TEST CODE START
    df = pd.read_csv(
        'data/processed/wine_dataset_all.csv', dtype=str).dropna(subset=["description"]).drop_duplicates(subset=['description'])
    wine_descriptions = df.sample(12)
    # TEST CODE END

    # load models
    embeddings = Word2Vec.load('models/wine_word2vec_model.bin')
    tfidf_weightings = load_tf_idf_weights()
    ngrams = load_ngrams()
    descriptor_map = load_descriptor_map()

    # process descriptions
    wine_descriptions['normalized_descriptors'] = wine_descriptions['description'].apply(
        lambda x: preprocess_description(x, ngrams, descriptor_map, level=3))

    # remove duplicates
    descriptors = wine_descriptions['normalized_descriptors'].tolist()
    descriptor_list_all = list(itertools.chain.from_iterable(descriptors))
    descriptor_list = list(set(descriptor_list_all))

    # get embeddings from descriptors
    descriptor_vectors = []
    for term in set(descriptor_list):
        word_vector = embeddings.wv.get_vector(term).reshape(1, 300)
        descriptor_vectors.append(word_vector)

    # get 8 descriptors
    input_vectors_listed = [a.tolist() for a in descriptor_vectors]
    input_vectors_listed = [a[0] for a in input_vectors_listed]
    kmeans = KMeans(n_clusters=8, random_state=0).fit(input_vectors_listed)

    closest, _ = pairwise_distances_argmin_min(
        kmeans.cluster_centers_, input_vectors_listed)
    sampled_descriptors = list(np.array(descriptor_list)[closest])

    # users picks 2 descriptors
    choices = []
    i = 0
    while i != 2:
        print('The wine descriptors are: ', sampled_descriptors)
        choice = input('Please choose first descriptor from list: ')
        if choice in sampled_descriptors:
            key = sampled_descriptors.index(choice)
            choices.append(sampled_descriptors.pop(key))
            i += 1
        else:
            print('Please choose a descriptor from the list.')

    # create user wine vector
    user_vector = get_wine_vector(choices, tfidf_weightings, embeddings)

    # vectorize wine descriptors
    wine_descriptions['wine_vectors'] = wine_descriptions['normalized_descriptors'].apply(
        lambda x: get_wine_vector(x, tfidf_weightings, embeddings))

    # calculate cosine similarity between user and wines
    wine_descriptions['cosine_similarity'] = wine_descriptions['wine_vectors'].apply(
        lambda x: cosine_similarity(user_vector, x))

    # take first 6 wines based on similarity
    wine_descriptions = wine_descriptions.sort_values(
        by=['cosine_similarity'], ascending=False).head(6)

    wine_descriptions.to_csv('test3.csv')

    print('Based on your preference we recommend these wines:')
    for idx, wine in enumerate(wine_descriptions['wine_name']):
        print(f'{idx + 1}: {wine} \n')

    return wine_descriptions


def get_wine_vector(descriptors, tf_idf, embeddings):
    wine_vector = []
    for term in descriptors:
        tfidf_weighting = tf_idf[term]
        word_vector = embeddings.wv.get_vector(term).reshape(1, 300)
        weighted_vector = tfidf_weighting * word_vector
        wine_vector.append(weighted_vector)

    return sum(wine_vector) / len(wine_vector)


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    # parser.add_argument("arg", help="Required positional argument")

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
