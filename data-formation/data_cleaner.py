import pandas as pd

subtitles_df = pd.read_csv('movies_subtitles.csv')
movies_df = pd.read_csv('movies_meta.csv')

subtitles_df['text'] = subtitles_df['text'].fillna('').astype(str)

subtitles_grouped = (
    subtitles_df.groupby('imdb_id')['text']
    .agg(' '.join)
    .reset_index()
)
subtitles_grouped.rename(columns={'text': 'full_subtitles'}, inplace=True)

# 3. Fusion avec les métadonnées
final_dataset = pd.merge(movies_df[['imdb_id', 'title', 'genres']], subtitles_grouped, on='imdb_id')

# 4. Sauvegarder le CSV propre
final_dataset.to_csv('dataset_films_clean.csv', index=False)
print("Traitement terminé avec succès !")