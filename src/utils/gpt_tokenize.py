import sys
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from gensim.models import Word2Vec
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from preprocessing import load_dataframes
from minbpe import BasicTokenizer
from minbpe import RegexTokenizer
import tiktoken


lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words('english'))

def normalize_text(viet_list):

    tokens = viet_list

    tokens = [word.lower() for word in tokens]

    tokens = [word for word in tokens if word not in stop_words]

    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return tokens

def gpt_tokenize(df_test):  
    viet_list = list(df_test["user_input"])
    viet_strings = " ".join(viet_list)
    normalized_tokens = normalize_text(viet_strings)
    viet_strings = " ".join(normalized_tokens)
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(viet_strings, allowed_special={'<|endofprompt|>'})
    return [str(token) for token in tokens]

def normalize2_text(text):
    # Tokenisation et normalisation d'un texte
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]  # Garde seulement les mots alphabétiques
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens