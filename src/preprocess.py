import re
from unidecode import unidecode

def clean_text(text):
  if not isinstance(text, str):
    return ''
  text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
  text = re.sub(r'\s+', ' ', text).strip()
  text = unidecode(text)
  text = text.lower()
  
  return text
  