"""
Module Docstring
"""

__author__ = "Your Name"
__version__ = "0.1.0"
__license__ = "MIT"

import argparse
from gensim.models import Word2Vec


def main(wines, df, min_count=5):
    embeddings = Word2Vec.load("models/wine_word2vec_model.bin")

    wine_description_mincount = df.loc[df['descriptor_count'] > min_count]
    wine_description_mincount.reset_index(inplace=True)

    input_vectors = list(wine_description_mincount['description_vector'])
    input_vectors_list = [a.tolist() for a in input_vectors]
    input_vectors_list = [a[0] for a in input_vectors_list]

    descriptors = []
    for wine in wines:
        descriptors.append(wine_description_mincount['normalized_descriptors'][])

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
