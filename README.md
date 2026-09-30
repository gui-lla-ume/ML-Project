# Movie Genre Classification from Subtitles

This Machine Learning project aims to predict movie genres based on their full subtitle texts using Natural Language Processing (NLP) techniques.

## Project Structure

```text
├── .gitignore                                 # Excludes heavy CSV files and environment directories
├── README.md                                  # Project overview and layout
├── data-formation/
│   ├── data_cleaner.py                        # Script to merge raw subtitles with IMDb metadata
│   └── README.md                              # Documentation on data sources and preprocessing
└── model_script/
    ├── tf_idf.py                              # Creates TF-IDF features from movie subtitles
    ├── multi_class_logistic_regression.py     # Predicts one genre per movie
    ├── multi_label_logistic_regression.py     # Predicts multiple genres per movie
    └── README.md                              # Documentation for feature extraction and modeling
```

## Tech Stack
- Language: Python 3.11+
- Data Processing: Pandas
- Machine Learning & NLP: Scikit-Learn
- Version Control: Git & GitHub

## Use of AI
Within the scope of this project, we used AI for : 
- Generating markdown files fastly
- ...