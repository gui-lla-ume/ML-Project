'''
usage: python3 helper_scripts/list_movies_with_genres.py
creates: A txt file with all movies and their genres.
'''

import pandas as pd
import ast

PATH_TO_CSV = "data-formation/dataset_films_clean.csv"

file = open("helper_scripts/output/list_of_all_movies_with_their_genre(s).txt", "w")
movies = pd.read_csv(PATH_TO_CSV)

for i,movie in movies.iterrows():
  genres = ast.literal_eval(movie["genres"])
  genre_names = [genre["name"] for genre in genres]
  file.write(f"{i+1}.) {movie['title']} {genre_names}\n")
  
  
  