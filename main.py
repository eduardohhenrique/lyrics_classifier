import pandas as pd 
from src.preprocess import clean_text
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report


df1 = pd.read_excel('./data/dataset_genero_musical_1.xlsx')
df2 = pd.read_excel('./data/dataset_genero_musical_2.xlsx')
df = pd.concat([df1, df2], ignore_index=True)

df['musica'] = df['musica'].apply(clean_text)

print(df.head())