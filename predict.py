import joblib
from src.preprocess import clean_text


best_model = joblib.load('models/lyrics_classifier_svm.pkl')

letra = '''Cole aqui sua letra: [Gospel, sertanejo, funk, bossa nova]'''
letra_limpa = clean_text(letra)
predicao = best_model.predict([letra_limpa])
print("Eu acho que é:", predicao[0])


# Para rodar, digite no terminal: python predict.py