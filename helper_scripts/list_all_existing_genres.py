'''
usage: python3 helper_scripts/list_all_existing_genres.py
creates: A txt file with all unique in existence genres.
'''

import pandas as pd
import ast

PATH_TO_CSV = "data-formation/dataset.csv"
#PATH_TO_CSV = "data-formation/dataset_films_clean.csv"

file = open("helper_scripts/output/list_of_unique_genres.txt", "w")
movies = pd.read_csv(PATH_TO_CSV)

genres = set() #No multiple

for i,movie in movies.iterrows():
  movie_genres = ast.literal_eval(movie["genres"])
  for genre in movie_genres:
    genres.add(genre)

i = 0
for genre in genres:
  i+=1
  file.write(f"{i}.) {genre}\n")

