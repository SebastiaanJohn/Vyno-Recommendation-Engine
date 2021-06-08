import requests

url = 'https://www.dropbox.com/s/q0lmcpq2l3h09dr/wine_df.pkl?dl=0'
r = requests.get(url, allow_redirects=True)

with open('wine_df.pkl', 'wb') as f:
    f.write(r.content)
