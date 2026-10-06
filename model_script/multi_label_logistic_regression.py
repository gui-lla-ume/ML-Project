import ast
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from tf_idf import (load_dataset, split_dataset, process_training_dataset,
                    process_validation_dataset, process_test_dataset, create_feature_matrix)
from sklearn.metrics import precision_score, recall_score, f1_score, log_loss

GENRES_FILE_PATH = "../helper_scripts/output/list_of_unique_genres.txt"


def load_genres(filepath):
  genres = []
  with open(filepath, "r", encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if line:
        if ".)" in line:
          genre = line.split(".)", 1)[1].strip()
        else:
          genre = line
        genres.append(genre)
  return genres


def parse_all_movie_genres(genre_str, target_genres_set):
  if pd.isna(genre_str):
    return []
  try:
    movie_genres = ast.literal_eval(genre_str)
  except (ValueError, SyntaxError):
    cleaned = str(genre_str).strip("[]'\"").replace("'", "").replace('"', "")
    movie_genres = [genre.strip() for genre in cleaned.split(",") if genre.strip()]
  valid_genres = []
  for genre in movie_genres:
    if genre in target_genres_set:
      valid_genres.append(genre)
  return valid_genres


def prepare_data(dataset, target_genres):
  target_genres_set = set(target_genres)
  valid_indices = []
  genre_lists = []

  for index, row in dataset.iterrows():
    raw_genres = row["genres"]
    raw_subtitles = row["full_subtitles"]
    genres = parse_all_movie_genres(raw_genres, target_genres_set)
    if genres and pd.notna(raw_subtitles) and str(raw_subtitles).strip():
      valid_indices.append(index)
      genre_lists.append(genres)
  prepared_dataset = dataset.loc[valid_indices].copy()

  return prepared_dataset, genre_lists


def create_label_matrix(genre_lists, target_genres):
  number_of_movies = len(genre_lists)
  number_of_genres = len(target_genres)

  label_matrix = np.zeros((number_of_movies, number_of_genres), dtype=int)

  genre_positions = {}

  for position in range(number_of_genres):
    genre = target_genres[position]
    genre_positions[genre] = position

  for movie_position in range(number_of_movies):

    current_movie_genres = genre_lists[movie_position]

    for genre in current_movie_genres:
      genre_position = genre_positions[genre]
      label_matrix[movie_position][genre_position] = 1

  return label_matrix


def train_models(X_train, y_training, target_genres):
  models = []

  for genre_position in range(len(target_genres)):
    genre = target_genres[genre_position]
    print(f"Training model for {genre}...")
    current_genre_labels = y_training[:, genre_position]
    model = LogisticRegression(solver="liblinear", max_iter=300, random_state=10)
    model.fit(X_train, current_genre_labels)

    models.append(model)

  return models


def predict_labels(models, X):
  number_of_movies = X.shape[0]
  number_of_genres = len(models)

  predictions = np.zeros((number_of_movies, number_of_genres), dtype=int)

  for genre_position in range(number_of_genres):
    model = models[genre_position]
    probabilities = model.predict_proba(X)[:, 1]
    predictions[:, genre_position] = (probabilities >= 0.5).astype(int)

  return predictions


def evaluate_model(y_true, y_pred, target_genres):

  f1_scores = []
  for genre_position in range(len(target_genres)):
    genre = target_genres[genre_position]
    true_labels = y_true[:, genre_position]
    predicted_labels = y_pred[:, genre_position]
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)
    f1_scores.append(f1)

    print(genre)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1: {f1:.4f}")
    print()
  average_f1 = sum(f1_scores) / len(f1_scores)
  print(f"Average F1 Score: {average_f1:.4f}")


def predict_probabilities(models, X):
  probabilities = np.zeros((X.shape[0], len(models)))
  for genre_position in range(len(models)):
    model = models[genre_position]
    probabilities[:, genre_position] = model.predict_proba(X)[:, 1]

  return probabilities


def main():

  target_genres = load_genres(GENRES_FILE_PATH)
  print(f"Loaded {len(target_genres)} genres")

  dataset = load_dataset()
  training_dataset, validation_dataset, test_dataset = split_dataset(dataset)
  training_dataset, train_genre_lists = prepare_data(training_dataset, target_genres)
  validation_dataset, val_genre_lists = prepare_data(validation_dataset, target_genres)
  test_dataset, test_genre_lists = prepare_data(test_dataset, target_genres)

  print(f"Dataset split size - "
        f"Train: {len(training_dataset)} | "
        f"Val: {len(validation_dataset)} | "
        f"Test: {len(test_dataset)}")

  y_training = create_label_matrix(train_genre_lists, target_genres)
  y_validation = create_label_matrix(val_genre_lists, target_genres)
  y_test = create_label_matrix(test_genre_lists, target_genres)

  print(f"Label matrix ready: "
        f"{y_training.shape[0]} movies x "
        f"{y_training.shape[1]} genres")

  print("\nCalculating TF-IDF")
  training_movies_tfidf, inverse_document_frequency = (process_training_dataset(training_dataset))
  validation_movies_tfidf = process_validation_dataset(validation_dataset,
                                                       inverse_document_frequency)
  test_movies_tfidf = process_test_dataset(test_dataset, inverse_document_frequency)

  feature_words = list(inverse_document_frequency.keys())
  X_train = create_feature_matrix(training_movies_tfidf, feature_words)
  X_val = create_feature_matrix(validation_movies_tfidf, feature_words)
  X_test = create_feature_matrix(test_movies_tfidf, feature_words)

  print(f"Vocabulary size: {len(feature_words)}")

  print("\nTraining Logistic Regression models\n")
  models = train_models(X_train, y_training, target_genres)

  training_probabilities = predict_probabilities(models, X_train)
  training_losses = []

  for genre_position in range(len(target_genres)):
    loss = log_loss(y_training[:, genre_position],
                    training_probabilities[:, genre_position],
                    labels=[0, 1])
    training_losses.append(loss)

  training_loss = sum(training_losses) / len(training_losses)
  y_val_pred = predict_labels(models, X_val)
  validation_f1_micro = f1_score(y_validation, y_val_pred, average="micro", zero_division=0)
  validation_f1_macro = f1_score(y_validation, y_val_pred, average="macro", zero_division=0)

  print(f"Training Loss: {training_loss:.4f}")
  print(f"Validation F1 Micro: {validation_f1_micro:.4f}")
  print(f"Validation F1 Macro: {validation_f1_macro:.4f}")
  print("\nValidation Set Performance\n")
  y_val_pred = predict_labels(models, X_val)
  evaluate_model(y_validation, y_val_pred, target_genres)

  print("\nTest Set Performance\n")
  y_test_pred = predict_labels(models, X_test)
  evaluate_model(y_test, y_test_pred, target_genres)


if __name__ == "__main__":
  main()




