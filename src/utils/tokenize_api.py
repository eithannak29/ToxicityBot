import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from minbpe import BasicTokenizer, RegexTokenizer
import tiktoken
from typing import List
import pandas as pd
import re
# from textblob import TextBlob
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.pre_tokenizers import PreTokenizer

from typing import Optional

# Télécharger les stopwords et le lemmatizer de WordNet Si besoin
# nltk.download("punkt")
# nltk.download("stopwords")
# nltk.download("wordnet")
# nltk.download("averaged_perceptron_tagger")

# Initialiser le lemmatizer et les stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
stop_words.update(["'ll", "'t", "'ve", "'d"])

def lemmatize_normalize_text(tokens):
    pos_tags = pos_tag(tokens)

    # Create an output list using list comprehension for optimal performance
    lemmatized_tokens = [
        (
            lemmatizer.lemmatize(token.lower(), pos="v")
            if pos.startswith("V")
            else (
                lemmatizer.lemmatize(token.lower(), pos="a")
                if pos.startswith("J")
                else (
                    lemmatizer.lemmatize(token.lower(), pos="r")
                    if pos.startswith("R")
                    else (
                        lemmatizer.lemmatize(token.lower(), pos="n")
                        if pos.startswith("N")
                        else lemmatizer.lemmatize(token.lower())
                    )
                )
            )
        )
        for token, pos in pos_tags
    ]

    return lemmatized_tokens


# def lemmatize_normalize_text(text):
#     tokens = word_tokenize(text)  # Tokenisation des mots
#     pos_tags = pos_tag(tokens)  # Étiquetage POS des mots
#     lemmatized_tokens = []
#     for token, pos in pos_tags:
#         if token.isupper():  # Si le mot est en majuscule (nom propre), conservez-le tel quel
#             lemmatized_tokens.append(token)
#         elif token.lower() not in stop_words:  # Si le mot n'est pas dans les stop words, lemmatisez-le
#             if pos.startswith('V'):  # Verbe
#                 lemmatized_token = lemmatizer.lemmatize(token.lower(), pos='v')
#             elif pos.startswith('J'):  # Adjectif
#                 lemmatized_token = lemmatizer.lemmatize(token.lower(), pos='a')
#             elif pos.startswith('R'):  # Adverbe
#                 lemmatized_token = lemmatizer.lemmatize(token.lower(), pos='r')
#             elif pos.startswith('N'):  # Nom
#                 lemmatized_token = lemmatizer.lemmatize(token.lower(), pos='n')
#             else:
#                 lemmatized_token = lemmatizer.lemmatize(token.lower())  # Autres cas
#             lemmatized_tokens.append(lemmatized_token)
#     return ' '.join(lemmatized_tokens)  # Rejoignez les mots lemmatisés en une seule chaîne de caractères


# def correct_spelling(text: str) -> str:
#     """
#     Corrige l'orthographe du texte en utilisant la bibliothèque TextBlob.

#     Args:
#     text (str): Le texte à corriger.

#     Returns:
#     str: Texte corrigé.
#     """
#     blob = TextBlob(text)
#     corrected_text = str(blob.correct())
#     return corrected_text

def replace_emojis(text: str) -> str:
    """
    Remplace les emojis du texte par leur équivalent en mots.

    Args:
    text (str): Le texte à traiter.

    Returns:
    str: Texte avec les emojis remplacés.
    """
    # Dictionnaire contenant les mappings d'emojis à remplacer
    emoji_mappings = {
        ":)": "happy",
        ":(": "sad",
        ":D": "laughing",
        ":P": "tongue_out",
        ":O": "surprised",
        ":|": "neutral",
    }
    # Expressions régulières pour capturer des motifs d'emojis supplémentaires
    additional_emoji_patterns = [
        r"\bwatching\b",  # Exemple de motif d'émoticône supplémentaire
    ]

    # Combine les motifs d'emojis avec ceux du dictionnaire
    all_emoji_patterns = "|".join(re.escape(key) for key in emoji_mappings.keys())
    all_emoji_patterns += "|" + "|".join(additional_emoji_patterns)
    # Génère une regex qui capture tous les emojis dans le dictionnaire et les motifs supplémentaires
    emoji_pattern = re.compile(all_emoji_patterns)

    # Remplace les emojis par les mots correspondants du dictionnaire
    cleaned_text = emoji_pattern.sub(
        lambda match: emoji_mappings.get(match.group(0), match.group(0)), text
    )

    return cleaned_text


def remove_special_characters(
    text: str, remove_punctuations=False, remove_duplications=False, space_after_punctuations=True
) -> str:
    """
    Supprime les caractères spéciaux, les adresses e-mail, les URLs, etc. du texte.

    Args:
    text (str): Le texte à traiter.

    Returns:
    str: Texte sans les caractères spéciaux, les adresses e-mail, les URLs, etc.
    """
    # Suppression des caractères spécifiés '<', '>', '\n', les adresses e-mail, les URLs et les mots contenant '@'
    # cleaned_text = re.sub(r'[<>\n@=]', '', text)
    # Suppression des adresses e-mail
    cleaned_text = re.sub(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text
    )
    # Suppression des URLs
    cleaned_text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        " ",
        cleaned_text,
    )
    # Suppression des mots contenant '@'
    # cleaned_text = re.sub(r'\b\w*@\w*\b', '', cleaned_text)
    cleaned_text = re.sub(
        r"wikipedia[\w%\'\.]*", " ", cleaned_text, flags=re.IGNORECASE
    )
    # Remplacement des motifs ressemblant à un pénis par le mot 'penis'
    cleaned_text = re.sub(r"\b8=+[=d]+", "penis", cleaned_text)

    # Supppresion des chiffres
    cleaned_text = re.sub(r"\d|\\n|\\t|\\d", " ", cleaned_text)

    if remove_punctuations:
        # Suppresion des ponctuations
        if space_after_punctuations:
            cleaned_text = re.sub(r"[^\w\s]", " ", cleaned_text)
        else:
            cleaned_text = re.sub(r"[^\w\s]", "", cleaned_text)

    cleaned_text = re.sub(r"[^\x00-\x7F]+", " ", cleaned_text)  # Keep ascii
    if remove_duplications:
        cleaned_text = re.sub(
            r"(.)\1{2,}", r"\1", cleaned_text
        )  # Replace youuuuuuuu by you
    cleaned_text = re.sub(r"\s+", " ", cleaned_text)  # Delete all big spaces
    return cleaned_text


def preprocess_text(
    text: str,
    is_replace_emojis: bool = True,
    is_remove_special_characters: bool = True,
    is_lowercase: bool = True,
    remove_stopwords: bool = True,
    is_lemmatization: bool = True,
    remove_punctuations=True,
    space_after_punctuations = True,
    remove_duplications=True,
    tokenize=word_tokenize,
    is_pretokenizer = False
) -> str:
    """
    Prétraitement complet du texte : correction de l'orthographe, remplacement des emojis,
    suppression des caractères spéciaux, des adresses e-mail, des URLs, etc., et normalisation du texte.

    Args:
    text (str): Le texte à prétraiter.

    Returns:
    str: Texte prétraité.
    """
    print("Texte initial:", text)
    
    if is_replace_emojis:
        text = replace_emojis(text)
        print("Après remplacement des emojis:", text)
    
    if is_remove_special_characters:
        text = remove_special_characters(text, remove_punctuations, remove_duplications, space_after_punctuations)
        print("Après suppression des caractères spéciaux:", text)
    
    if is_lemmatization:
        tokens = word_tokenize(text) if not is_pretokenizer else tokenize(text)
        print("Après tokenisation:", tokens)
        
        if is_lowercase:
            tokens = [word.lower() for word in tokens]
            print("Après conversion en minuscules:", tokens)
        
        tokens = lemmatize_normalize_text(tokens)
        print("Après lemmatisation:", tokens)
        
        if remove_stopwords:
            tokens = [word for word in tokens if word not in stop_words]
            print("Après suppression des stopwords:", tokens)
        
        processed_text = (
            " ".join(tokenize(" ".join(tokens)))
            if tokenize != word_tokenize
            else " ".join(tokens)
        )
        print("Texte final après re-tokenisation:", processed_text)
        return processed_text
    
    elif is_lowercase:
        text = text.lower()
        print("Après conversion en minuscules:", text)
        
        if remove_stopwords:
            tokens = word_tokenize(text) if not is_pretokenizer else tokenize(text)
            tokens = [word for word in tokens if word not in stop_words]
            print("Après suppression des stopwords:", tokens)
            
            processed_text = (
                " ".join(tokenize(" ".join(tokens)))
                if tokenize != word_tokenize
                else " ".join(tokens)
            )
            print("Texte final après re-tokenisation:", processed_text)
            return processed_text
    
    processed_text = " ".join(tokenize(text))
    print("Texte final:", processed_text)
    return processed_text


def gpt_tokenize(
    viet_strings: str,
    normalize: bool = False,
    lowercase: bool = True,
    remove_stopwords: bool = True,
    lemmatization: bool = True,
) -> List[str]:
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
    if normalize:
        normalized_text = preprocess_text(viet_strings)
    else:
        normalized_text = viet_strings
    enc = tiktoken.encoding_for_model("gpt-4")
    tokens = enc.encode(normalized_text)
    # decoded_tokens = enc.decode(tokens)
    # decoded_words = decoded_tokens.split()
    return [str(token) for token in tokens]


def byte_pair_tokenize(
    viet_strings: str,
    tokenizer=None,
    normalize: bool = False,
    lowercase: bool = True,
    remove_stopwords: bool = True,
    lemmatization: bool = True,
    vocab_size: int = 1024,
) -> List[str]:
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
    if normalize:
        normalized_text = preprocess_text(viet_strings)
    else:
        normalized_text = viet_strings

    if tokenizer is None:
        tokenizer = BasicTokenizer()
        tokenizer.train(normalized_text, vocab_size=vocab_size)
        tokenizer.save("byte_pair_tokenizer")

    tokens = tokenizer.encode(normalized_text)
    decoded_tokens = tokenizer.decode(tokens)

    return [str(token) for token in tokens]



def byte_pair_tokenize_transformer(
    sentence,tokenizer=None, comment_text_train: Optional[pd.DataFrame] = None, vocab_size: int = 65536, save_path = "data/trained_tokenizer.json" 
):
    # Sample text data (in practice, use a large corpus)

    # Initialize a BPE tokenizer
    if tokenizer is None:
        tokenizer = Tokenizer(BPE())

        # Set the normalizer and pre-tokenizer
        tokenizer.pre_tokenizer = Whitespace() # type: ignore

        # Initialize the BPE trainer with a specified vocabulary size
        vocab_size = vocab_size  # Set your desired vocabulary size
        trainer = BpeTrainer(vocab_size=vocab_size) # type: ignore

        # Train the tokenizer on your text data
        tokenizer.train_from_iterator(comment_text_train, trainer=trainer)

        # Save the trained tokenizer to a file
        tokenizer.save(save_path)

    # Load the trained tokenizer (for future use
    # Tokenize a new sentence using the trained tokenizer
    tokens = tokenizer.encode(sentence) # type: ignore
    return tokens.tokens


def regex_tokenize(
    viet_strings: str,
    tokenizer=None,
    normalize: bool = False,
    lowercase: bool = True,
    remove_stopwords: bool = True,
    lemmatization: bool = True,
    vocab_size: int = 1024,
) -> List[str]:
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
    if normalize:
        normalized_text = preprocess_text(viet_strings)
    else:
        normalized_text = viet_strings

    if tokenizer is None:
        tokenizer = RegexTokenizer()
        tokenizer.train(normalized_text, vocab_size=vocab_size)
        tokenizer.save("regex_tokenizer")

    viet_tokens = tokenizer.encode(normalized_text)
    decoded_tokens = tokenizer.decode(viet_tokens)
    decoded_words = decoded_tokens.split()

    return decoded_words


if __name__ == "__main__":
    print(
        byte_pair_tokenize(
            """What is Lorem Ipsum?
Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.

Why do we use it?
It is a long established fact that a reader will be distracted by the readable content of a page when looking at its layout. The point of using Lorem Ipsum is that it has a more-or-less normal distribution of letters, as opposed to using 'Content here, content here', making it look like readable English. Many desktop publishing packages and web page editors now use Lorem Ipsum as their default model text, and a search for 'lorem ipsum' will uncover many web sites still in their infancy. Various versions have evolved over the years, sometimes by accident, sometimes on purpose (injected humour and the like).


Where does it come from?
Contrary to popular belief, Lorem Ipsum is not simply random text. It has roots in a piece of classical Latin literature from 45 BC, making it over 2000 years old. Richard McClintock, a Latin professor at Hampden-Sydney College in Virginia, looked up one of the more obscure Latin words, consectetur, from a Lorem Ipsum passage, and going through the cites of the word in classical literature, discovered the undoubtable source. Lorem Ipsum comes from sections 1.10.32 and 1.10.33 of "de Finibus Bonorum et Malorum" (The Extremes of Good and Evil) by Cicero, written in 45 BC. This book is a treatise on the theory of ethics, very popular during the Renaissance. The first line of Lorem Ipsum, "Lorem ipsum dolor sit amet..", comes from a line in section 1.10.32.

The standard chunk of Lorem Ipsum used since the 1500s is reproduced below for those interested. Sections 1.10.32 and 1.10.33 from "de Finibus Bonorum et Malorum" by Cicero are also reproduced in their exact original form, accompanied by English versions from the 1914 translation by H. Rackham.

Where can I get some?
There are many variations of passages of Lorem Ipsum available, but the majority have suffered alteration in some form, by injected humour, or randomised words which don't look even slightly believable. If you are going to use a passage of Lorem Ipsum, you need to be sure there isn't anything embarrassing hidden in the middle of text. All the Lorem Ipsum generators on the Internet tend to repeat predefined chunks as necessary, making this the first true generator on the Internet. It uses a dictionary of over 200 Latin words, combined with a handful of model sentence structures, to generate L""",
            normalize=True,
            vocab_size=65536,
        )
    )
    # import preprocessing

    # (df_train, df_val, df_test) = preprocessing.load_dataframes()

    # # Texte initial
    # # original_text = df_test["comment_text"][0] #met ta phrase pour testt
    # original_text = "Thankz yuu for understanding. I thinkk very highly of you and wouldd not revert without this discussion. :) 8=="
    # print("Texte initial:")
    # print(original_text)
    # print()

    # # Texte après correction de l'orthographe
    # corrected_text = correct_spelling(original_text)
    # print("Texte après correction de l'orthographe:")
    # print(corrected_text)
    # print()

    # # Texte après remplacement des emojis
    # emoji_replaced_text = replace_emojis(corrected_text)
    # print("Texte après remplacement des emojis:")
    # print(emoji_replaced_text)
    # print()

    # # Texte après suppression des caractères spéciaux
    # special_chars_removed_text = remove_special_characters(emoji_replaced_text)
    # print("Texte après suppression des caractères spéciaux:")
    # print(special_chars_removed_text)
    # print()

    # # Texte après normalisation
    # normalized_text = preprocess_text(original_text)
    # print("Texte après normalisation complète:")
    # print(normalized_text)
    # print()

    # # Test avec GPT Tokenizer sans normalisation
    # tokens = gpt_tokenize(df_test)
    # print("Tokens obtenus avec GPT Tokenizer sans normalisation:")
    # print(tokens[:10])
    # print()

    # # Test avec GPT Tokenizer avec normalisation
    # tokens = gpt_tokenize(df_test, normalize=True)
    # print("Tokens obtenus avec GPT Tokenizer avec normalisation:")
    # print(tokens[:10])
    # print()