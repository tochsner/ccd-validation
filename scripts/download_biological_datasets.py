import csv
from pathlib import Path
from tqdm import tqdm
import urllib.request

with open("data/biological_datasets.csv", "r") as f:
    reader = csv.reader(f, delimiter=";", skipinitialspace=True)
    for row in tqdm(list(reader)[1:]):
        title, url, file_name = row

        output_file_name = Path(f"data/biological_datasets/{title}/{file_name}")
        if output_file_name.exists():
            continue
        
        with urllib.request.urlopen(f"https://datadryad.org{url}") as f:
            data = f.read().decode('utf-8')

        output_file_name.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file_name, "w") as f:
            f.write(data)
