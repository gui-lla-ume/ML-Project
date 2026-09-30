import re
import pandas as pd
PATH_TO_CSV = "dataset.csv"

dataset = pd.read_csv(PATH_TO_CSV)

def convert_to_lowercase(text):
  return str(text).lower()

'''
(function) def sub(
 pattern: str | Pattern[str],
 repl: str | ((Match[str]) -> str),
 string: str,
 count: int = 0,
 flags: _FlagsType = 0
) -> str
'''

def remove_html(text):
  #<u><b>
  return re.sub(pattern=r"<[^>]+>", repl=" ", string=text)

def remove_special_characters(text):
  #non a,b,c character
  return re.sub(r"[^a-z\s]", repl=" ", string=text)

''' 
#This causes that each film is one line in csv. This looks a bit ugly.
def remove_extra_spaces(text):
  cleaned_text = ""
  previous_character_was_space = False
  for character in text:
    if character.isspace():
      if previous_character_was_space is False:
        cleaned_text += " "
      previous_character_was_space = True
    else:
      cleaned_text += character
      previous_character_was_space = False
  return cleaned_text.strip()
'''

def preprocess_text(text):
  text = convert_to_lowercase(text)
  text = remove_html(text)
  text = remove_special_characters(text)
  #text = remove_extra_spaces(text)
  return text

preprocessed_subtitles = []
for subtitle in dataset["full_subtitles"]:
    preprocessed_subtitle = preprocess_text(subtitle)
    preprocessed_subtitles.append(preprocessed_subtitle)

dataset["full_subtitles"] = preprocessed_subtitles
dataset.to_csv(PATH_TO_CSV, index=False)

