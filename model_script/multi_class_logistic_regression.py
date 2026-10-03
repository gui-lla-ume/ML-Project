import ast
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from tf_idf import load_dataset, split_dataset

GENRES_FILE_PATH = "../helper_scripts/output/list_of_unique_genres.txt"

def load_genres(filepath):
    """Loads target genres list from file."""
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
    """
    Parses the movie's genre string and returns a LIST of all genres
    that overlap with our target 20 genres.
    """
    if pd.isna(genre_str):
        return []
    try:
        movie_genres = ast.literal_eval(genre_str)
    except (ValueError, SyntaxError):
        cleaned = str(genre_str).strip("[]'\"").replace("'", "").replace('"', "")
        movie_genres = [g.strip() for g in cleaned.split(",") if g.strip()]
    
    # Filter to only include genres present in our 20 target genres
    return [g for g in movie_genres if g in target_genres_set]

def prepare_data(df, target_genres):
    """
    Prepares training/testing data directly from raw subtitle strings:
    - Retains movies with a non-empty subtitle and at least one target genre.
    - ground_truth_lists: list of lists containing all valid genres for each movie.
    - primary_labels: single label per movie (first valid genre) used for training the Softmax model.
    """
    target_genres_set = set(target_genres)
    subtitles = []
    ground_truth_lists = []
    primary_labels = []

    for _, row in df.iterrows():
        raw_genre = row["genres"]
        raw_sub = row["full_subtitles"]
        
        genres = parse_all_movie_genres(raw_genre, target_genres_set)
        
        # Ensure movie has at least one target genre and non-empty subtitle string
        if genres and pd.notna(raw_sub) and str(raw_sub).strip():
            subtitles.append(str(raw_sub))
            ground_truth_lists.append(genres)
            primary_labels.append(genres[0])  # Used as representative single-class target during training

    return subtitles, ground_truth_lists, np.array(primary_labels)

def evaluate_in_genre_list(model, X, ground_truth_lists, k=1):
    """
    Evaluates accuracy by checking if ANY of the model's top-k predicted genres
    is contained within the movie's list of ground-truth genres.
    """
    probs = model.predict_proba(X)
    classes = model.classes_
    
    hits = 0
    total = len(ground_truth_lists)
    
    for i in range(total):
        movie_true_genres = set(ground_truth_lists[i])
        
        # Get indices of top-k predicted probability classes
        top_k_indices = np.argsort(probs[i])[-k:][::-1]
        top_k_predicted_genres = [classes[idx] for idx in top_k_indices]
        
        # Check if there is any intersection between predicted top-k and true genres
        if any(pred in movie_true_genres for pred in top_k_predicted_genres):
            hits += 1

    accuracy = (hits / total) * 100
    return accuracy

def main():
    # 1. Load target genre list
    target_genres = load_genres(GENRES_FILE_PATH)
    print(f"Loaded {len(target_genres)} target genres.")

    # 2. Load dataset and split using tf_idf split helper
    print("Loading dataset...")
    dataset = load_dataset()
    train_df, val_df, test_df = split_dataset(dataset)

    # 3. Extract subtitle texts, evaluation genre lists, and primary training labels
    train_subs, train_genre_lists, y_train = prepare_data(train_df, target_genres)
    val_subs, val_genre_lists, _ = prepare_data(val_df, target_genres)
    test_subs, test_genre_lists, _ = prepare_data(test_df, target_genres)

    print(f"Dataset split size - Train: {len(y_train)} | Val: {len(val_subs)} | Test: {len(test_subs)}")

    # 4. Scikit-Learn TfidfVectorizer (Fast, Memory-Efficient)
    print("\nExtracting TF-IDF features with TfidfVectorizer...")
    vectorizer = TfidfVectorizer(
        stop_words="english",  # Removes common stop words (e.g., 'the', 'and')
        max_features=25000,    # Limits vocab to top 25,000 words (dramatically speeds up model training)
        sublinear_tf=True      # Uses 1 + log(tf) scaling to handle long script texts
    )

    # Fit ONCE on training data, transform validation and test sets
    X_train = vectorizer.fit_transform(train_subs)
    X_val = vectorizer.transform(val_subs)
    X_test = vectorizer.transform(test_subs)

    print(f"Feature matrix ready. Vocabulary size: {X_train.shape[1]} features.")

    # 5. Fast Multinomial / Softmax Logistic Regression Training
    print("\nTraining Multinomial Logistic Regression model...")
    model = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=300,        # Sufficient for convergence with 25k features
        tol=1e-3,            # Early stopping threshold to save training time
        n_jobs=-1,           # Uses all available CPU cores to speed up fitting
        random_state=10
    )
    model.fit(X_train, y_train)
    print("Model training complete!")

    # 6. Evaluate on Validation Set
    val_acc = evaluate_in_genre_list(model, X_val, val_genre_lists, k=1)
    print(f"\nValidation Accuracy (Top-1 prediction in genre list): {val_acc:.2f}%")

    # 7. Final Evaluation on Test Set
    print("\n--- Test Set Performance ---")
    test_top1_acc = evaluate_in_genre_list(model, X_test, test_genre_lists, k=1)
    test_top2_acc = evaluate_in_genre_list(model, X_test, test_genre_lists, k=2)
    test_top3_acc = evaluate_in_genre_list(model, X_test, test_genre_lists, k=3)

    print(f"Top-1 Accuracy (Most plausible genre is in true list):  {test_top1_acc:.2f}%")
    print(f"Top-2 Accuracy (Top-2 predictions overlap with list):  {test_top2_acc:.2f}%")
    print(f"Top-3 Accuracy (Top-3 predictions overlap with list):  {test_top3_acc:.2f}%")

if __name__ == "__main__":
    main()