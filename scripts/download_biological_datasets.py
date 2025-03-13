import csv
from os import makedirs
from tqdm import tqdm
import urllib.request

with open("data/biological_datasets.csv", "r") as f:
    reader = csv.reader(f, delimiter=";", skipinitialspace=True)
    for row in tqdm(list(reader)[1:]):
        title, url, file_name = row
        
        with urllib.request.urlopen(f"https://datadryad.org{url}") as f:
            data = f.read().decode('utf-8')

        makedirs(f"data/biological_datasets/{title}", exist_ok=True)

        with open(f"data/biological_datasets/{title}/{file_name}", "w") as f:
            f.write(data)
