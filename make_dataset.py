"""
Download processed pickled dataset.
"""

__author__ = "Sebastiaan Dijkstra"
__version__ = "0.1.0"
__license__ = "MIT"

import requests


def main():
    url = 'https://www.dropbox.com/s/q0lmcpq2l3h09dr/wine_df.pkl?dl=0'
    r = requests.get(url, allow_redirects=True)

    with open('wine_df.pkl', 'wb') as f:
        f.write(r.content)


if __name__ == "__main__":
    main()
