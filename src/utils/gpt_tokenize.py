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
stop_words.add("'ll")
stop_words.add("'t")
stop_words.add("'ve")
stop_words.add("'d")

def normalize_text(phrase):
 # Ici j'initialise les stopwords 
    lemmatizer = WordNetLemmatizer()

    stop_words = set(stopwords.words('english'))
    stop_words.add("'ll")
    stop_words.add("'t")
    stop_words.add("'ve")
    stop_words.add("'d")
#
    tokens = word_tokenize(phrase)
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    text = " ".join(tokens)
    return text
# gpt_tokenize refait de facon a ce qu'il tokenize des phrases et non tout le dataframe
def gpt_tokenize(text):  
    newtokens = normalize_text(text)
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(newtokens)
    tokens_list = [(enc.decode([word])) for word in tokens]
    return tokens_list


def normalize2_text(text):
    # Tokenisation et normalisation d'un texte
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]  # Garde seulement les mots alphabétiques
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens