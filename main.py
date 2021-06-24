"""
Main running point of application.
"""

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from gensim.models.phrases import Phrases
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from random import shuffle
import itertools
import string
import pickle
import argparse

__author__ = "Team Baard"
__version__ = "1.0.0"


def load_tf_idf_weights(pkl='models/tfidf_vectorizer.pickle'):
    """Return TF-IDF weights dictionary using saved pickle file."""
    tf_idf = pickle.load(open(pkl, "rb"))
    return dict(zip(tf_idf.get_feature_names(), tf_idf.idf_))


def load_ngrams(ngram='models/ngrams'):
    """Return N-gram language model."""
    return Phrases.load(ngram)


def load_descriptor_map(dmap='models/descriptor_mapping.csv'):
    """Return disciptor map as Pandas dataframe."""
    return pd.read_csv(dmap).set_index('raw descriptor')


def preprocess_description(description, ngram, descriptor_map, level=3):
    """
    Tokenizes text, created n-grams and maps words in wine
    description to descriptor map.

    Keyword arguments:
    description -- the wine description (str)
    ngram -- pretrained n-gram model
    descriptor_map -- descriptor map (dataframe)

    returns: preprocessed descriptor from wine description
    """
    tokens = tokenize_description(description)
    phrase = ngram[tokens]
    descriptors = [map_descriptor(word, descriptor_map, level)
                   for word in phrase]
    descripters_cleaned = [str(desc)
                           for desc in descriptors if desc is not None]

    return descripters_cleaned


def tokenize_description(description):
    """Tokenizes, stems, and removes punctuation from wine description"""
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
    """Maps word in wine description to descriptor mapping"""
    if word in list(mapping.index):
        return mapping[f'level_{level}'][word]


def get_wine_vector(descriptors, tf_idf, embeddings):
    """
    Creates wine vector from wine description.

    Keyword arguments:
    descriptors -- list of wine descriptors (list)
    tf_idf -- pretrained TF-IDF model (dict)
    embeddings -- Word2Vec embeddings

    returns: weighted wine vector
    """
    wine_vector = []
    for term in descriptors:
        if term in tf_idf.keys():
            tfidf_weighting = tf_idf[term]
            word_vector = embeddings.wv.get_vector(term).reshape(1, 300)
            weighted_vector = tfidf_weighting * word_vector
            wine_vector.append(weighted_vector)

    return sum(wine_vector) / len(wine_vector)


def main(args):
    wine_descriptions = pd.read_csv(args.df)

    # load models
    embeddings = Word2Vec.load('models/word2vec_model.bin')
    tfidf_weightings = load_tf_idf_weights()
    ngrams = load_ngrams()
    descriptor_map = load_descriptor_map()

    # process descriptions
    wine_descriptions['NORMALIZED_DESCRIPTION'] = wine_descriptions['DESCRIPTION'].apply(
        lambda x: preprocess_description(x, ngrams, descriptor_map, level=3))

    # remove duplicates
    descriptors = wine_descriptions['NORMALIZED_DESCRIPTION'].tolist()
    descriptor_list_all = list(itertools.chain.from_iterable(descriptors))
    descriptor_list = list(set(descriptor_list_all))
    print('All descriptors:', descriptor_list, '\n')

    # remove descriptors that are not easy to understand
    easy_descriptors = pd.read_csv(
        'models/user_descriptors.csv')['descriptors'].tolist()
    easy_descriptors_list = []
    for descriptor in descriptor_list:
        if descriptor in easy_descriptors:
            easy_descriptors_list.append(descriptor)

    print('All user descriptors:', easy_descriptors_list, '\n')

    # get embeddings from descriptors
    descriptor_vectors = []
    for term in set(easy_descriptors_list):
        word_vector = embeddings.wv.get_vector(term).reshape(1, 300)
        descriptor_vectors.append(word_vector)

    # get 8 descriptors
    input_vectors_listed = [a.tolist() for a in descriptor_vectors]
    input_vectors_listed = [a[0] for a in input_vectors_listed]
    kmeans = KMeans(n_clusters=8, random_state=0).fit(input_vectors_listed)

    closest, _ = pairwise_distances_argmin_min(
        kmeans.cluster_centers_, input_vectors_listed)
    sampled_descriptors = list(np.array(easy_descriptors_list)[closest])

    # split 8 descriptors into 2 sets of 4
    shuffle(sampled_descriptors)
    q1 = sampled_descriptors[:4]
    q2 = sampled_descriptors[4:]

    # user picks 2 descriptors
    choices = []
    i = 0
    while i < 2:
        print('The wine descriptors are: ', q1)
        choice = input('Please choose descriptor from list: ')
        if choice in q1:
            key = q1.pop(q1.index(choice))
            choices.append(key)
            i += 1
        else:
            print('That descriptor is not in the list, please choose another one.')

    print('\n')

    i = 0
    while i < 2:
        print('The wine descriptors are: ', q2)
        choice = input('Please choose descriptor from list: ')
        if choice in q2:
            key = q2.pop(q2.index(choice))
            choices.append(key)
            i += 1
        else:
            print('That descriptor is not in the list, please choose another one.')

    print('\n')
    print('User choose:', choices)

    # create user wine vector
    user_vector = get_wine_vector(choices, tfidf_weightings, embeddings)

    # vectorize wine descriptors
    wine_descriptions['WINE_VECTORS'] = wine_descriptions['NORMALIZED_DESCRIPTION'].apply(
        lambda x: get_wine_vector(x, tfidf_weightings, embeddings))

    # calculate cosine similarity between user and wines
    wine_descriptions['COSINE_SIMILARITY'] = wine_descriptions['WINE_VECTORS'].apply(
        lambda x: cosine_similarity(user_vector, x))

    # take first 6 wines based on similarity
    wine_descriptions = wine_descriptions.sort_values(
        by=['COSINE_SIMILARITY'], ascending=False).head(6)

    print('\n')

    print('Based on your preference we recommend these wines: \n')
    for idx, wine in enumerate(wine_descriptions['NAME']):
        print(f'{idx + 1}: {wine} \n')

    return wine_descriptions


if __name__ == "__main__":
    """ This is executed when run from the command line """
    parser = argparse.ArgumentParser()

    # Required positional argument
    parser.add_argument(
        "df", help="Required: path to CSV file containing wine desscriptions.")

    # Specify output of "--version"
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s (version {version})".format(version=__version__))

    args = parser.parse_args()
    main(args)
