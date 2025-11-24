Projeto de ML para classificar gêneros de letra de musica [gospel, sertanejo, funk e bossa nova].
Utiliza TF-IDF + LinearSVM.

Estrutura:
lyrics-classifier/
│
├── data/
│   ├── dataset_genero_musical_1.xlsx
│   └── dataset_genero_musical_2.xlsx
│
├── models/
│   └── svm_tfidf_model.pkl
│
├── src/
│   └── preprocess.py
│
├── main.py          # Treino do modelo
├── predict.py       # Arquivo para fazer previsões
├── requirements.txt
└── README.md