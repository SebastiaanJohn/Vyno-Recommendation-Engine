"""
Preprocess raw CSV dataset
"""

__author__ = "Sebastiaan Dijkstra"
__version__ = "0.1.0"
__license__ = "MIT"

import pandas as pd
import numpy as np


def main():
    DATA = 'dataset/'

    # preprocess wine dataset 1
    cols = ['country', 'description', 'Wine name', 'province',
            'Region', 'Sub region', 'Grape', 'Winery/ Vineyard']
    wine_df1 = pd.read_csv(
        DATA + 'Vyno - Wine dataset 1.csv', dtype=str, usecols=cols)
    wine_df1.columns = [title.lower() for title in wine_df1.columns]
    wine_df1 = wine_df1.rename(columns={
        'wine name': 'wine_name',
        'sub region': 'sub_region',
        'winery/ vineyard': 'vineyard'
    })

    # preprocess wine dataset 2
    wine_df2 = pd.read_csv(
        DATA + 'Vyno - Wine dataset 2.csv', dtype=str).iloc[:, 1:]
    wine_df2.columns = [title.lower() for title in wine_df2.columns]
    wine_df2 = wine_df2.rename(columns={
        'wine name': 'wine_name',
        'sub region': 'sub_region',
        'winery/ vineyard': 'vineyard'
    })
    cols2 = ['country', 'description', 'wine_name', 'province',
             'region', 'sub_region', 'grape', 'vineyard', 'title']
    wine_df2 = wine_df2[cols2]

    # combine datasets into 1 dataframe
    wine_df = pd.concat([wine_df1, wine_df2], axis=0)

    # save to csv
    wine_df.to_csv('Wine dataset combined.csv')

    return wine_df


if __name__ == "__main__":
    main()
