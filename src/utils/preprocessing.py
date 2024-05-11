import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.model_selection import train_test_split
from constants import TEST_LABEL_PATH, TEST_PATH, TRAIN_PATH
from utils.tokenize_api import preprocess_text, gpt_tokenize
from nltk.tokenize import word_tokenize
import os


def load_dataframe_test(path_label: str, path_text: str) -> pd.DataFrame:
    texts = pd.read_csv(path_text)
    labels = pd.read_csv(path_label)
    labels = labels[labels["toxic"] != -1]
    df_test = pd.merge(labels, texts, on="id")
    return df_test


def remove_empty_lines(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["comment_text"] != ""]


def set_overall_toxic(df: pd.DataFrame) -> pd.DataFrame:
    df["overall_toxic"] = 0
    df.loc[
        (df["toxic"] == 1)
        | (df["severe_toxic"] == 1)
        | (df["obscene"] == 1)
        | (df["threat"] == 1)
        | (df["insult"] == 1)
        | (df["identity_hate"] == 1),
        "overall_toxic",
    ] = 1
    return df


def load_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_test = load_dataframe_test(TEST_LABEL_PATH, TEST_PATH)
    df_train = pd.read_csv(TRAIN_PATH)
    df_train = set_overall_toxic(df_train) 
    df_test = set_overall_toxic(df_test)
    df_test = remove_empty_lines(df_test)
    df_train = remove_empty_lines(df_train)
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
    return df_train, df_val, df_test

def preprocess_text_default(text: str, tokenize_func=word_tokenize, is_remove_special_characters: bool = True,
                            remove_stopwords: bool = True, is_replace_emojis: bool = True, is_lowercase: bool = True,
                            is_lemmatization: bool = True, remove_punctuations: bool = True) -> str:
    return preprocess_text(text, tokenize_func=tokenize_func,
                           is_remove_special_characters=is_remove_special_characters,
                           remove_stopwords=remove_stopwords, is_replace_emojis=is_replace_emojis,
                           is_lowercase=is_lowercase, is_lemmatization=is_lemmatization,
                           remove_punctuations=remove_punctuations)

def preprocess_dataframe(df_train: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame,
                         params: Dict[str, Dict[str, Any]] = None, output_dir: str = 'data') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if params is None:
        params = {
            "word_tokenize_no_normalization": {
                "tokenize": word_tokenize,
                "is_remove_special_characters": False,
                "remove_stopwords": False,
                "is_replace_emojis": False,
                "is_lowercase": False,
                "is_lemmatization": False,
                "remove_punctuations": False,
            },
            "gpt_tokenize_no_normalization": {
                "tokenize": gpt_tokenize,
                "is_remove_special_characters": False,
                "remove_stopwords": False,
                "is_replace_emojis": False,
                "is_lowercase": False,
                "is_lemmatization": False,
                "remove_punctuations": False,
            },
            "word_tokenize_normalization": {
                "tokenize": word_tokenize,
                "is_remove_special_characters": True,
                "remove_stopwords": True,
                "is_replace_emojis": True,
                "is_lowercase": True,
                "is_lemmatization": True,
                "remove_punctuations": False,
            },
            "gpt_tokenize_normalization": {
                "tokenize": gpt_tokenize,
                "is_remove_special_characters": True,
                "remove_stopwords": True,
                "is_replace_emojis": True,
                "is_lowercase": True,
                "is_lemmatization": True,
                "remove_punctuations": False,
            },
            "word_tokenize_full_normalization": {
                "tokenize": word_tokenize,
                "is_remove_special_characters": True,
                "remove_stopwords": True,
                "is_replace_emojis": True,
                "is_lowercase": True,
                "is_lemmatization": True,
                "remove_punctuations": True,
            },
            "gpt_tokenize_full_normalization": {
                "tokenize": gpt_tokenize,
                "is_remove_special_characters": True,
                "remove_stopwords": True,
                "is_replace_emojis": True,
                "is_lowercase": True,
                "is_lemmatization": True,
                "remove_punctuations": True,
            },
            "word_tokenize_simple_normalization": {
                "tokenize": word_tokenize,
                "is_remove_special_characters": False,
                "remove_stopwords": True,
                "is_replace_emojis": False,
                "is_lowercase": True,
                "is_lemmatization": False,
                "remove_punctuations": False,
            },
            "gpt_tokenize_simple_normalization": {
                "tokenize": gpt_tokenize,
                "is_remove_special_characters": False,
                "remove_stopwords": True,
                "is_replace_emojis": False,
                "is_lowercase": True,
                "is_lemmatization": False,
                "remove_punctuations": False,
            },
        }

    if not os.path.exists(os.path.join(output_dir, 'df_train_preprocessed.parquet')) or \
       not os.path.exists(os.path.join(output_dir, 'df_val_preprocessed.parquet')) or \
       not os.path.exists(os.path.join(output_dir, 'df_test_preprocessed.parquet')):
        for tn, p in params.items():
            print(f"Processing {tn}")
            df_train[f'comment_text_{tn}'] = df_train['comment_text'].apply(
                lambda x: preprocess_text_default(x, **p))
            df_val[f'comment_text_{tn}'] = df_val['comment_text'].apply(
                lambda x: preprocess_text_default(x, **p))
            df_test[f'comment_text_{tn}'] = df_test['comment_text'].apply(
                lambda x: preprocess_text_default(x, **p))

        df_train.rename(columns={'comment_text': 'comment_text_baseline'}, inplace=True)
        df_val.rename(columns={'comment_text': 'comment_text_baseline'}, inplace=True)
        df_test.rename(columns={'comment_text': 'comment_text_baseline'}, inplace=True)

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        df_train.to_parquet(os.path.join(output_dir, 'df_train_preprocessed.parquet'), index=False)
        df_val.to_parquet(os.path.join(output_dir, 'df_val_preprocessed.parquet'), index=False)
        df_test.to_parquet(os.path.join(output_dir, 'df_test_preprocessed.parquet'), index=False)
    else:
        df_train = pd.read_parquet(os.path.join(output_dir, 'df_train_preprocessed.parquet'))
        df_val = pd.read_parquet(os.path.join(output_dir, 'df_val_preprocessed.parquet'))
        df_test = pd.read_parquet(os.path.join(output_dir, 'df_test_preprocessed.parquet'))

    return df_train, df_val, df_test

# if __name__ == "__main__":
#     output_dir = 'data'

#     df_train, df_val, df_test = load_dataframes()
#     df_train, df_val, df_test = preprocess_dataframe(df_train, df_val, df_test, output_dir=output_dir)
