import ast
import json
import pandas as pd

def extract_genre_names(val):
    # Handle NaN or None
    if pd.isna(val) or val is None:
        return []

    # If it's a string, try converting to a Python object
    if isinstance(val, str):
        val = val.strip()
        if not val or val == '[]':
            return []
        try:
            # Try standard JSON parsing
            val = json.loads(val)
        except Exception:
            try:
                # Fall back to literal_eval (for single-quoted string representations)
                val = ast.literal_eval(val)
            except Exception:
                return []

    # Extract names if it's a list of dictionaries
    if isinstance(val, list):
        return [
            item.get('name')
            for item in val
            if isinstance(item, dict) and 'name' in item
        ]

    return []


subtitles_df = pd.read_csv('movies_subtitles.csv')
movies_df = pd.read_csv('movies_meta.csv')

subtitles_df['text'] = subtitles_df['text'].fillna('').astype(str)

subtitles_grouped = (
    subtitles_df.groupby('imdb_id')['text']
    .agg(' '.join)
    .reset_index()
)
subtitles_grouped.rename(columns={'text': 'full_subtitles'}, inplace=True)

movies_df['genres'] = movies_df['genres'].apply(extract_genre_names)

final_dataset = pd.merge(movies_df[['imdb_id', 'title', 'genres']], subtitles_grouped, on='imdb_id')

final_dataset.to_csv('dataset.csv', index=False)
print("Dataset cleaned !")