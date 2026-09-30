'''
usage: python3 helper_scripts/list_movies.py
creates: A txt file with all the existing movies inside the csv.
'''

import pandas as pd

PATH_TO_CSV = "data-formation/dataset_films_clean.csv"

file = open("helper_scripts/list_of_movies.txt","w")
movies = pd.read_csv(PATH_TO_CSV)
for i,title in enumerate(movies["title"]):
  file.write(f"{i+1}.) {title}\n")

