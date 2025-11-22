import pandas as pd
import nltk
from nltk.corpus import stopwords
from src.preprocess import clean_text
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline


# Stopwords
nltk.download('stopwords')
stop_words = stopwords.words('portuguese')

# Limpeza de dados
df1 = pd.read_excel('./data/dataset_genero_musical_1.xlsx')
df2 = pd.read_excel('./data/dataset_genero_musical_2.xlsx')
df = pd.concat([df1, df2], ignore_index=True)

df['musica'] = df['musica'].apply(clean_text)

print(df.head())

# Treino/teste
x = df['musica']
y = df['genero']

x_train, x_test, y_train, y_test = train_test_split(
  x, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words=stop_words)),
    ('clf', LinearSVC())
])

# Hiperparâmetros
param_grid = {
  "tfidf__ngram_range": [(1, 2), (1, 3)],
  "tfidf__min_df": [1, 2, 3, 5],
  "tfidf__max_features": [20000, 50000, None],
  "clf__C": [0.05, 0.5, 1, 2, 3]
}

grid = GridSearchCV(
    pipeline,
    param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid.fit(x_train, y_train)

best_model = grid.best_estimator_
print("Melhores Hiperparâmetros:", grid.best_params_)