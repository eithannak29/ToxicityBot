import sys
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from minbpe import BasicTokenizer, RegexTokenizer
import tiktoken
from typing import List
import pandas as pd

# Télécharger les stopwords et le lemmatizer de WordNet Si besoin
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')

# Initialiser le lemmatizer et les stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
stop_words.update(["'ll", "'t", "'ve", "'d"])


def lemmatize_normalize_text(text):
    tokens = word_tokenize(text)  # Tokenisation des mots
    pos_tags = pos_tag(tokens)  # Étiquetage POS des mots
    lemmatized_tokens = []
    for token, pos in pos_tags:
        if token.isupper():  # Si le mot est en majuscule (nom propre), conservez-le tel quel
            lemmatized_tokens.append(token)
        elif token.lower() not in stop_words:  # Si le mot n'est pas dans les stop words, lemmatisez-le
            if pos.startswith('V'):  # Verbe
                lemmatized_token = lemmatizer.lemmatize(token.lower(), pos='v')
            elif pos.startswith('J'):  # Adjectif
                lemmatized_token = lemmatizer.lemmatize(token.lower(), pos='a')
            elif pos.startswith('R'):  # Adverbe
                lemmatized_token = lemmatizer.lemmatize(token.lower(), pos='r')
            elif pos.startswith('N'):  # Nom
                lemmatized_token = lemmatizer.lemmatize(token.lower(), pos='n')
            else:
                lemmatized_token = lemmatizer.lemmatize(token.lower())  # Autres cas
            lemmatized_tokens.append(lemmatized_token)
    return ' '.join(lemmatized_tokens)  # Rejoignez les mots lemmatisés en une seule chaîne de caractères


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
    if lemmatization:
        phrase = lemmatize_normalize_text(phrase)
    tokens = word_tokenize(phrase)
    if lowercase:
        tokens = [word.lower() for word in tokens]
    if remove_stopwords:
        tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

def gpt_tokenize(df_test: pd.DataFrame, normalize: bool = False, lowercase: bool = True, remove_stopwords: bool = True, lemmatization: bool = True) -> List[str]:
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
    if (normalize):
        normalized_text = normalize_text(viet_strings, lowercase, remove_stopwords, lemmatization)
    else:
        normalized_text = viet_strings
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(normalized_text)
    decoded_tokens = enc.decode(tokens)
    decoded_words = decoded_tokens.split()
    return decoded_words

def byte_pair_tokenize(df_test: pd.DataFrame, tokenizer=None, normalize: bool = False, lowercase: bool = True, remove_stopwords: bool = True, lemmatization: bool = True, vocab_size: int = 1024) -> List[str]:
    """
    Tokenise le texte en utilisant le Byte Pair Encoding (BPE).
    
    Args:
    df_test (DataFrame): DataFrame contenant le texte à tokeniser.
    tokenizer (BasicTokenizer, optional): Tokenizer pré-entraîné (par défaut None).
    lowercase (bool, optional): Mettre le texte en minuscule (par défaut True).
    remove_stopwords (bool, optional): Supprimer les stopwords (par défaut True).
    lemmatization (bool, optional): Lemmatiser les mots (par défaut True).
    vocab_size (int, optional): Taille du vocabulaire (par défaut 1024).
    
    Returns:
    list: Liste des tokens.
    """
    viet_list = list(df_test["comment_text"])
    viet_strings = " ".join(viet_list)

    if (normalize):
        normalized_text = normalize_text(viet_strings, lowercase, remove_stopwords, lemmatization)
    else:
        normalized_text = viet_strings
    
    if tokenizer is None:
        tokenizer = BasicTokenizer()
        tokenizer.train(normalized_text, vocab_size=vocab_size)
        tokenizer.save('byte_pair_tokenizer')
        
    viet_tokens = tokenizer.encode(normalized_text)
    decoded_tokens = tokenizer.decode(viet_tokens)
    decoded_words = decoded_tokens.split()
    
    return decoded_words

def regex_tokenize(df_test: pd.DataFrame, tokenizer=None, normalize: bool = False, lowercase: bool = True, remove_stopwords: bool = True, lemmatization: bool = True, vocab_size: int = 1024) -> List[str]:
    """
    Tokenise le texte en utilisant le RegexTokenizer.
    
    Args:
    df_test (DataFrame): DataFrame contenant le texte à tokeniser.
    tokenizer (RegexTokenizer, optional): Tokenizer pré-entraîné (par défaut None).
    lowercase (bool, optional): Mettre le texte en minuscule (par défaut True).
    remove_stopwords (bool, optional): Supprimer les stopwords (par défaut True).
    lemmatization (bool, optional): Lemmatiser les mots (par défaut True).
    vocab_size (int, optional): Taille du vocabulaire (par défaut 1024).
    
    Returns:
    list: Liste des tokens.
    """
    viet_list = list(df_test["comment_text"])
    viet_strings = " ".join(viet_list)

    
    if (normalize):
        normalized_text = normalize_text(viet_strings, lowercase, remove_stopwords, lemmatization)
    else:
        normalized_text = viet_strings
    
    if tokenizer is None:
        tokenizer = RegexTokenizer()
        tokenizer.train(normalized_text, vocab_size=vocab_size)
        tokenizer.save('regex_tokenizer')
    
    viet_tokens = tokenizer.encode(normalized_text)
    decoded_tokens = tokenizer.decode(viet_tokens)
    decoded_words = decoded_tokens.split()
    
    return decoded_words

if __name__ == "__main__":
    import preprocessing
    (df_train, df_val, df_test) = preprocessing.load_dataframes()
    print(df_test["comment_text"][0])
    # new_sentence = lemmatize_normalize_text(df_test["comment_text"][0])

    tokens = gpt_tokenize(df_test)
    print(tokens[:10])
