import pandas as pd
import joblib
import os
import nltk
from nltk.corpus import stopwords
from src.preprocess import clean_text
from src.evaluate import avaliar_modelo
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
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

# Confusion Matrix e Classificação
labels = ['BOSSA NOVA', 'FUNK', 'GOSPEL', 'SERTANEJO']

y_pred, cm = avaliar_modelo(best_model, x_test, y_test, labels)


# Teste final
letra = '''Quem não sabe o que eu passei
Vai ter que me escutar
Tô tendo a vida de rei
Tive sem te atrapalhar
Pele de ouro burgues
Muita grana pra gastar
E uma cama e as estrelas

Tá brilhando aqui na minha frente
Lançou da polo ice no pingente
Toma Lacoste, tome glockada
Que eu parto pro lar de parapente

Mais uma mulher pra mim é frequente
Foda que a branca pensa diferente
Já usou bolsa, da mais barata
Hoje venceu só quer bolsa da Fendi

Se tá devendo algo a si mesmo
Se cobrando com medo de amar
No meu passado não foi desse jeito
Morro de medo de me apaixonar

Mas você não manda em você mesmo
Deus que manda ele que vai mandar
Energias são jogadas no vento
Tô colhendo o que eu pude plantar

Se tá devendo algo a si mesmo
Se cobrando com medo de amar
No meu passado não foi desse jeito
Morro de medo de me apaixonar

Mas você não manda em você mesmo
Deus que manda ele que vai mandar
Energias são jogadas no vento
Tô colhendo o que eu pude plantar

Te quero, tá difícil
Amor pro meu compromisso
Sucesso, meu filho fez da água virar vinho
E eu peço menino
Deus guia nossos caminho e é desde novinho
Diferente o meu brilho

Te quero, tá difícil
Sucesso, meu filho fez da água virar vinho
E eu peço menino
Deus guia nossos caminho e é desde novinho
Diferente o meu brilho

Te quero, tá difícil
Amor pro meu compromisso
Sucesso, meu filho fez da água virar vinho
E eu peço menino
Deus guia nossos caminho e é desde novinho
Diferente o meu brilho

Se tá devendo algo a si mesmo
Se cobrando com medo de amar
No meu passado não foi desse jeito
Morro de medo de me apaixonar

Mas você não manda em você mesmo
Deus que manda ele que vai mandar
Energias são jogadas no vento
Tô colhendo o que eu pude plantar

Te quero, tá difícil
Amor pro meu compromisso
Sucesso, meu filho fez da água virar vinho
E eu peço menino
Deus guia nossos caminho e é desde novinho
Diferente o meu brilho

Te quero tá difícil
Amor com o meu compromisso
Sucesso meu filho fez da água virar vinho
E eu peço menino pra guiar nossos
Caminhos e é desde novinho
Diferente o meu brilho'''

letra_limpa = clean_text(letra)

# Predição
predicao = best_model.predict([letra_limpa])
print("Eu acho que é: ", predicao[0])

# Salvar modelo
joblib.dump(best_model, 'models/lyrics_classifier_svm.pkl')