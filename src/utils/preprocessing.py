import ast
import pandas as pd
from datasets import load_dataset
from typing import Tuple
from sklearn.model_selection import train_test_split
from constant import TEST_LABEL_PATH, TEST_PATH, TRAIN_PATH

def load_dataframe_test(path_label: str, path_text:str ) -> pd.DataFrame:
    texts = pd.read_csv(path_text)
    labels = pd.read_csv(path_label)
    labels = labels[labels['toxic'] != -1]
    df_test = pd.merge(labels, texts, on='id')
    return df_test

def remove_empty_lines(df: pd.DataFrame ) -> pd.DataFrame:
    return df[df['comment_text'] != ""]

def load_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_test  = load_dataframe_test(TEST_LABEL_PATH, TEST_PATH)
    df_train  = pd.read_csv(TRAIN_PATH)
    df_test = remove_empty_lines(df_test)
    df_train = remove_empty_lines(df_train)
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
    return df_train, df_val, df_test