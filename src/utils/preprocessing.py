import ast
import pandas as pd
from datasets import load_dataset
from typing import Tuple
from sklearn.model_selection import train_test_split
from constants import TEST_LABEL_PATH, TEST_PATH, TRAIN_PATH

def load_dataframe_test(path_label: str, path_text:str ) -> pd.DataFrame:
    texts = pd.read_csv(path_text)
    labels = pd.read_csv(path_label)
    labels = labels[labels['toxic'] != -1]
    df_test = pd.merge(labels, texts, on='id')
    return df_test

def remove_empty_lines(df: pd.DataFrame ) -> pd.DataFrame:
    return df[df['comment_text'] != ""]

def load_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_test  = load_dataframe_test(TEST_LABEL_PATH, TEST_PATH)
    df_train  = pd.read_csv(TRAIN_PATH)
    df_test = remove_empty_lines(df_test)
    df_train = remove_empty_lines(df_train)
    df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=42)
    return df_train, df_val, df_test


def load_dataframes(test_size:int =0.2,seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    
    df_train = dataset['train'].to_pandas()
    df_test = dataset['test'].to_pandas()
    
    df_train = preprocess_df(df_train)
    df_test = preprocess_df(df_test)
    
    df_train, df_val = train_test_split(df_train, test_size=test_size, random_state=seed)
    
    return (df_train, df_val, df_test)


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    cols_to_keep = ['user_input', 'human_annotation', 'toxicity', 'jailbreaking', 'openai_moderation']
    for col in df.columns:
        if col not in cols_to_keep:
            df.drop(col, axis=1, inplace=True)
            
    df["openai_moderation"] = df["openai_moderation"].apply(ast.literal_eval)
    
    categories = CATEGORIES
    for content_type in categories:
        df[content_type] = df["openai_moderation"].apply(lambda x: next((item[1] for item in x if item[0] == content_type), 0))
    
    df.drop('openai_moderation', axis=1, inplace=True)
    
    return df


def binarize_categories(df: pd.DataFrame) -> pd.DataFrame:
    df_res = pd.DataFrame()
    df_res['max'] = df[CATEGORIES].idxmax(axis=1)

    for category in CATEGORIES:
        df_res[category] = df_res.apply(lambda x: 1 if x['max'] == category else 0, axis=1)
        
    df_res.loc[df['toxicity'] == 0, CATEGORIES] = 0

    return df_res

if __name__ == "__main__":
    print("Preprocessing done")