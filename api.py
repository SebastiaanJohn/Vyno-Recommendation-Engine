from flask import *
from flask import jsonify
__author__ = "Team Baard"
__version__ = "0.1.0"
__license__ = "MIT"

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
import base64
import json

np.random.seed(42)


def load_tf_idf_weights(pkl='models/tfidf_vectorizer.pickle'):
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

def get_wine_vector(descriptors, tf_idf, embeddings):
    wine_vector = []
    for term in descriptors:
        tfidf_weighting = tf_idf[term]
        word_vector = embeddings.wv.get_vector(term).reshape(1, 300)
        weighted_vector = tfidf_weighting * word_vector
        wine_vector.append(weighted_vector)

    return sum(wine_vector) / len(wine_vector)

df = pd.read_csv(
    'josephbarneswines_test.csv', dtype=str).dropna(subset=["description"]).drop_duplicates(subset=['description'])

# print(len(df))
wine_descriptions = df

# print(len(wine_descriptions))

app = Flask(__name__)

@app.route('/12wines', methods=["GET", "POST"])
def get_descriptors():
    # get 12 wines as input   
    data = request.get_json()

    # TEST CODE END
    # load models
    embeddings = Word2Vec.load('models/word2vec_model.bin')
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

    # remove descriptors that are not easy to understand
    easy_descriptors = pd.read_csv(
        'models/user_descriptors.csv')['descriptors'].tolist()
    easy_descriptors_list = []
    for descriptor in descriptor_list:
        if descriptor in easy_descriptors:
            easy_descriptors_list.append(descriptor)


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

    data = {}
    data['descriptors'] = sampled_descriptors
    a = wine_descriptions.to_dict(orient = 'index')
    data['json_frame'] = a

    return jsonify(data)


@app.route('/2words', methods=["GET", "POST"])
def get_wines():
    data = request.get_json()
    a = data["json_frame"]
    # a_json = json.loads(a)
    wine_descriptions = pd.DataFrame.from_dict(a, orient="index")
    
    choices = data['descriptors']
    
    embeddings = Word2Vec.load('models/word2vec_model.bin')
    tfidf_weightings = load_tf_idf_weights()

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



    data = {}
    print(wine_descriptions['title'])
    a = wine_descriptions['title'].to_dict()
    data['wine_descriptions'] = a                                                      
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True, port="8000")