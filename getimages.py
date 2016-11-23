import os
import urllib.request
from urllib.request import urlopen
import json
from bs4 import BeautifulSoup as BS

jsons = os.listdir('json')

# Google Bigquery-ből exportált .json fileok alapján tölt le képeket és teszi mappákba
for category in jsons:
    with open('json/' + category) as json_data:
        d = json.load(json_data)
        test_count = int(len(d) * 0.5)
        valid_count = int(len(d) * 0.75)
        for idx, line in enumerate(d):
            url = line['original_landing_url']
            out = 'train/'
            cnt = idx + 1
            if idx > test_count:
                out = 'test/'
                cnt = idx-test_count
            if idx > valid_count:
                out = 'validation/'
                cnt = (idx - valid_count)


            folder = category.split('.')[0]
            print(url)

            try:
                response = urlopen(url + "/sizes/z")
                html_src = response.read()
                soup = BS(html_src)
                for img in soup.find_all("img"):
                    if img['src'][8] == 'c':
                        img_url = img['src']

                urllib.request.urlretrieve(img_url, 'data/' + out + folder + '/' + folder + '{0}'.format(
                    str(cnt).zfill(3)) + '.jpeg')

            except Exception: # van kép amit már leszedtek :(
                pass
