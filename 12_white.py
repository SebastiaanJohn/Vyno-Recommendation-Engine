"""
Module Docstring
"""

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
import numpy as np
import pandas as pd
from random import shuffle
import itertools
import string
import pickle
import argparse
import random

np.random.seed(42)

__author__ = "Team Baard"
__version__ = "1.0.0"
__license__ = "MIT"


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
        else:
            continue

    return sum(wine_vector) / len(wine_vector)

# ['lemon_blossom', 'lime', 'blackcurrant', 'apple']

# All user descriptors: ['passion_fruit', 'lemon_blossom', 'oak', 'nectarine', 'wood', 'apple', 'vanilla', 'citrus', 'peach', 'lime', 'pear', 'tangerine', 'spice', 'melon', 'blackcurrant'] 
# 1: Celler del Roure, Cullerot, DO Valencia, Spain 

# 2: Stella Bella, Suckfizzle Sauvignon Blanc, Semillon, Margaret River, Australia 

# 3: Paul Hobbs, Crossbarn Chardonnay, Sonoma Coast, California, USA 

# 4: Domaine Klur, Katz, Pinot Gris, Alsace, France 

# 5: ILatium Morini, Soave, DOC, Veneto, Italy 

# 6: Iona, Elgin Highlands Wild Ferment Sauvignon Blanc, Elgin, South Africa 
# ['vanilla', 'citrus', 'passion_fruit', 'melon']
# ['peach', 'tangerine', 'lemon_blossom', 'apple']

def main(args):
    # TEST CODE START
    # df = pd.read_csv(
    #     'darley_merchant_wines.csv', dtype=str).dropna(subset=["DESCRIPTION"]).drop_duplicates(subset=['DESCRIPTION'])
    # grouped = df.groupby(df.TYPE)
    # white_df = grouped.get_group('White Wine')
    # wine_descriptions = white_df.iloc
    # wine_descriptions = white_df.sample(n = 12, random_state = np.random.RandomState())
    # TEST CODE END

    wine_descriptions = pd.read_csv('josephbarneswines_test.csv')
    # wine_descriptions.to_csv('josephbarneswines_test.csv')

    # load models
    embeddings = Word2Vec.load('models/word2vec_model.bin')
    tfidf_weightings = load_tf_idf_weights()
    ngrams = load_ngrams()
    descriptor_map = load_descriptor_map()

    # process descriptions
    wine_descriptions['normalized_descriptors'] = wine_descriptions['DESCRIPTION'].apply(
        lambda x: preprocess_description(x, ngrams, descriptor_map, level=3))

    # remove duplicates
    descriptors = wine_descriptions['normalized_descriptors'].tolist()
    descriptor_list_all = list(itertools.chain.from_iterable(descriptors))
    descriptor_list = list(set(descriptor_list_all))
    names = wine_descriptions['NAME'].tolist()
    descriptions = wine_descriptions['DESCRIPTION'].tolist()
    for i in range(len(names)):
        print('Wine: ' + names[i])
        # print('description: ' + descriptions[i])
        # print('descriptors: ' + str(descriptors[i]) + '\n')

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
    # shuffle(sampled_descriptors)
    q1 = sampled_descriptors[:4]
    q2 = sampled_descriptors[4:]

    # user picks 2 descriptors
    choices = []
    while True:
        print('The wine descriptors are: ', q1)
        choice = input('Please choose first descriptor from list: ')
        if choice in q1:
            choices.append(choice)
            break

    print('\n')

    while True:
        print('The wine descriptors are: ', q2)
        choice = input('Please choose second descriptor from list: ')
        if choice in q2:
            choices.append(choice)
            break

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

    wine_csv = wine_descriptions.sort_values(
        by=['cosine_similarity'], ascending=False)

    wine_csv.to_csv(path_or_buf = 'example_output_white.csv')

    print('\n')

    print('Based on your preference we recommend these wines: \n')
    for idx, wine in enumerate(wine_descriptions['NAME']):
        print(f'{idx + 1}: {wine} \n')

    return wine_descriptions


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
