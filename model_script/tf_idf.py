#python3 tf_idf.py
import math
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
PATH_TO_CSV = "../data-formation/dataset.csv"

def load_dataset():
  dataset = pd.read_csv(PATH_TO_CSV)
  return dataset

def split_dataset(dataset):
  #make 70 training_dataset and 30 remaining_30
  training_dataset, remaining_30 = train_test_split(dataset,test_size=0.30,random_state=10)
  #make 15% for validation_dataset and 15% test_dataset out of remaining_dataset (30*0.50 =15)!
  validation_dataset, test_dataset = train_test_split(remaining_30,test_size=0.50,random_state=10)
  
  return training_dataset,validation_dataset,test_dataset

def split_subtitles_into_words(subtitles):
  all_movies_words = []
  for subtitle in subtitles:
    current_movie_words = subtitle.split()
    all_movies_words.append(current_movie_words)
  return all_movies_words

#Basically different movies containing this word.
def calculate_document_frequency(all_movies_words):
  document_frequency={}
  for current_movie_words in all_movies_words:
    current_movie_unique_words = set(current_movie_words) #remove multiples from collection
    for word in current_movie_unique_words:
      if word in document_frequency:
        document_frequency[word] += 1
      else:
        document_frequency[word] = 1

  #print(max(document_frequency, key=document_frequency.get))
  return document_frequency

def calculate_inverse_document_frequency(document_frequency,number_of_movies):
  inverse_document_frequency = {}
  for word in document_frequency:
    df = document_frequency[word]
    idf = math.log(number_of_movies/df)
    inverse_document_frequency[word] = idf
  return inverse_document_frequency

#Basically how often occurs one word in specific movie.
def calculate_term_frequency(current_movie_words):
  term_frequency = {}
  for word in current_movie_words:
    if word in term_frequency:
      term_frequency[word] += 1
    else:
      term_frequency[word] = 1
  return term_frequency

def calculate_tfidf(current_movie_words, inverse_document_frequency):
  term_frequency = calculate_term_frequency(current_movie_words)
  tfidf = {}
  for word in term_frequency:
    if word in inverse_document_frequency:
      tf = term_frequency[word]
      idf = inverse_document_frequency[word]
      tfidf[word] = tf * idf
      
  return tfidf

def calculate_tfidf_for_all_movies(all_movies_words,inverse_document_frequency):
  movies_tfidf = []
  for current_movie_words in all_movies_words:
    tfidf = calculate_tfidf(current_movie_words, inverse_document_frequency)
    movies_tfidf.append(tfidf)
  return movies_tfidf


def process_training_dataset(training_dataset):
  training_subtitles = training_dataset["full_subtitles"]
  all_training_movies_words = split_subtitles_into_words(training_subtitles)
  number_of_training_movies = len(all_training_movies_words)
  document_frequency = calculate_document_frequency(all_training_movies_words)
  inverse_document_frequency = calculate_inverse_document_frequency(document_frequency,number_of_training_movies)
  training_movies_tfidf = calculate_tfidf_for_all_movies(all_training_movies_words,inverse_document_frequency)
  
  return training_movies_tfidf,inverse_document_frequency

def process_validation_dataset(validation_dataset, inverse_document_frequency):
  validation_subtitles = validation_dataset["full_subtitles"]
  all_validation_movies_words = split_subtitles_into_words(validation_subtitles)
  validation_movies_tfidf = calculate_tfidf_for_all_movies(all_validation_movies_words,inverse_document_frequency)

  return validation_movies_tfidf

def process_test_dataset(test_dataset, inverse_document_frequency):
  test_subtitles = test_dataset["full_subtitles"]
  all_test_movies_words = split_subtitles_into_words(test_subtitles)
  test_movies_tfidf = calculate_tfidf_for_all_movies(all_test_movies_words,inverse_document_frequency)

  return test_movies_tfidf

def create_feature_matrix(movies_tfidf,feature_words):
  number_of_movies = len(movies_tfidf)
  number_of_words = len(feature_words)
  feature_matrix = np.zeros((number_of_movies,number_of_words))
  
  word_positions = {}
  for position in range(len(feature_words)):
    word = feature_words[position]
    word_positions[word] = position
  for movie_position in range(len(movies_tfidf)):
    current_movie_tfidf = movies_tfidf[movie_position]
    for word in current_movie_tfidf:
      word_position = word_positions[word]
      feature_matrix[movie_position][word_position] = current_movie_tfidf[word]

  return feature_matrix

def main():
  dataset = load_dataset()
  training_dataset, validation_dataset, test_dataset = split_dataset(dataset)
  
  training_movies_tfidf, inverse_document_frequency = process_training_dataset(training_dataset)
  validation_movies_tfidf = process_validation_dataset(validation_dataset,inverse_document_frequency)
  test_movies_tfidf = process_test_dataset(test_dataset,inverse_document_frequency)

  feature_words = list(inverse_document_frequency.keys())
  X_training = create_feature_matrix(training_movies_tfidf,feature_words)
  X_validation = create_feature_matrix(validation_movies_tfidf,feature_words)
  X_test = create_feature_matrix(test_movies_tfidf,feature_words)

if __name__ == "__main__":
  main()

