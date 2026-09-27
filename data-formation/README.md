# Data Source & Preprocessing Steps

This folder contains the data cleaning and preparation pipeline used to transform raw movie subtitles and metadata into a clean dataset ready for Machine Learning.

## Dataset Overview (Kaggle)

The raw data was collected from [Kaggle](https://www.kaggle.com/datasets/adiamaan/movie-subtitle-dataset) and consists of two primary files:

1. **`movies_meta.csv`**: Contains IMDb metadata for **4,693 movies**, including movie IDs (`imdb_id`), titles (`title`), and assigned genres (`genres`).
2. **`movies_subtitles.csv`**: Contains subtitle lines broken down sentence by sentence, linked to each movie via `imdb_id`.

## Data Preprocessing Workflow (`data_cleaner.py`)

Individual subtitle sentences (often numbering over 1,000 per movie) cannot be classified effectively in isolation. The script `data_cleaner.py` performs the following steps:

1. **Missing Value Handling:** Replaces `NaN` or empty text values with empty strings to prevent execution errors.
2. **Text Aggregation:** Groups all subtitle lines belonging to the same `imdb_id` and concatenates them into a single continuous text string (`full_subtitles`).
3. **Metadata Merging:** Merges the grouped subtitles with `movies_meta.csv` based on `imdb_id`.
4. **Export:** Saves the final structured dataset as `dataset_films_clean.csv` (**4,693 rows**, one per movie) with columns: `[imdb_id, title, genres, full_subtitles]`.

## How to Reproduce

1. Download `movies_meta.csv` and `movies_subtitles.csv` from the Kaggle source.
2. Place both files in the root or data directory.
3. Run the processing script:

```bash
python data-formation/data_cleaner.py