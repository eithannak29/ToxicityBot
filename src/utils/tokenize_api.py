import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from minbpe import BasicTokenizer, RegexTokenizer
import tiktoken
from typing import List
import pandas as pd

# Télécharger les stopwords et le lemmatizer de WordNet
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Initialiser le lemmatizer et les stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.update(["'ll", "'t", "'ve", "'d"])

def normalize_text(phrase: str, lowercase: bool = True, remove_stopwords: bool = True, lemmatization: bool = True) -> str:
    """
    Normalise le texte en supprimant les stopwords, la mise en minuscule et la lemmatisation.
    
    Args:
    phrase (str): Le texte à normaliser.
    lowercase (bool, optional): Mettre le texte en minuscule (par défaut True).
    remove_stopwords (bool, optional): Supprimer les stopwords (par défaut True).
    lemmatization (bool, optional): Lemmatiser les mots (par défaut True).
    
    Returns:
    str: Texte normalisé.
    """
    tokens = word_tokenize(phrase)
    if lowercase:
        tokens = [word.lower() for word in tokens]
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]
    if lemmatization:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def gpt_tokenize(df_test: pd.DataFrame, lowercase: bool = True, remove_stopwords: bool = True, lemmatization: bool = True) -> List[str]:
    """
    Tokenise le texte en utilisant le modèle GPT-4 de OpenAI.
    
    Args:
    df_test (DataFrame): DataFrame contenant le texte à tokeniser.
    lowercase (bool, optional): Mettre le texte en minuscule (par défaut True).
    remove_stopwords (bool, optional): Supprimer les stopwords (par défaut True).
    lemmatization (bool, optional): Lemmatiser les mots (par défaut True).
    
    Returns:
    list: Liste des tokens.
    """
    viet_list = list(df_test["comment_text"])
    viet_strings = " ".join(viet_list)
    normalized_text = normalize_text(viet_strings, lowercase, remove_stopwords, lemmatization)
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(normalized_text)
    return [str(token) for token in tokens]

def byte_tokenize(df_test: pd.DataFrame, lowercase: bool = True, remove_stopwords: bool = True, lemmatization: bool = True) -> List[str]:
    """
    Tokenise le texte en utilisant l'encodage de bytes.
    
    Args:
    df_test (DataFrame): DataFrame contenant le texte à tokeniser.
    lowercase (bool, optional): Mettre le texte en minuscule (par défaut True).
    remove_stopwords (bool, optional): Supprimer les stopwords (par défaut True).
    lemmatization (bool, optional): Lemmatiser les mots (par défaut True).
    
    Returns:
    list: Liste des tokens.
    """
    viet_list = list(df_test["comment_text"])
    viet_strings = " ".join(viet_list)
    normalized_text = normalize_text(viet_strings, lowercase, remove_stopwords, lemmatization)
    tokens = normalized_text.encode("utf-8")
    tokens = list(map(int, tokens))
    return [str(token) for token in tokens]

def basicTokenize(df_test: pd.DataFrame, vocab_size: int = 1024, lowercase: bool = True, remove_stopwords: bool = True, lemmatization: bool = True) -> List[str]:
    """
    Tokenise le texte en utilisant le BasicTokenizer.
    
    Args:
    df_test (DataFrame): DataFrame contenant le texte à tokeniser.
    vocab_size (int, optional): Taille du vocabulaire (par défaut 1024).
    lowercase (bool, optional): Mettre le texte en minuscule (par défaut True).
    remove_stopwords (bool, optional): Supprimer les stopwords (par défaut True).
    lemmatization (bool, optional): Lemmatiser les mots (par défaut True).
    
    Returns:
    list: Liste des tokens.
    """
    viet_list = list(df_test["comment_text"])
    viet_strings = " ".join(viet_list)
    normalized_text = normalize_text(viet_strings, lowercase, remove_stopwords, lemmatization)
    tokenizer = BasicTokenizer()
    tokenizer.train(normalized_text, vocab_size=vocab_size)
    viet_tokens = tokenizer.encode(normalized_text)
    tokenizer.save('BasicTokenizer')
    return [str(token) for token in viet_tokens]

def regexTokenize(df_test: pd.DataFrame, vocab_size: int = 1024, lowercase: bool = True, remove_stopwords: bool = True, lemmatization: bool = True) -> List[str]:
    """
    Tokenise le texte en utilisant le RegexTokenizer.
    
    Args:
    df_test (DataFrame): DataFrame contenant le texte à tokeniser.
    vocab_size (int, optional): Taille du vocabulaire (par défaut 1024).
    lowercase (bool, optional): Mettre le texte en minuscule (par défaut True).
    remove_stopwords (bool, optional): Supprimer les stopwords (par défaut True).
    lemmatization (bool, optional): Lemmatiser les mots (par défaut True).
    
    Returns:
    list: Liste des tokens.
    """
    viet_list = list(df_test["comment_text"])
    viet_strings = " ".join(viet_list)
    normalized_text = normalize_text(viet_strings, lowercase, remove_stopwords, lemmatization)
    tokenizer = RegexTokenizer()
    tokenizer.train(normalized_text, vocab_size=vocab_size)
    viet_tokens = tokenizer.encode(normalized_text)
    tokenizer.save('RegexTokenizer')
    return [str(token) for token in viet_tokens]

if __name__ == "__main__":
    import preprocessing
    
    # Charger les données
    (df_train, df_val, df_test) = preprocessing.load_dataframes()
    
    # Exemple d'utilisation de la fonction gpt_tokenize
    tokens = gpt_tokenize(df_test)
    print(tokens[:10])
