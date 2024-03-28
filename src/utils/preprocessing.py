import ast
import pandas as pd
from datasets import load_dataset
from typing import Tuple
from sklearn.model_selection import train_test_split

from constants import CATEGORIES


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
    df['max'] = df[CATEGORIES].idxmax(axis=1)

    for category in CATEGORIES:
        df[category] = df.apply(lambda x: 1 if x['max'] == category else 0, axis=1)
        
    df.loc[df['toxicity'] == 0, CATEGORIES] = 0

    return df

if __name__ == "__main__":
    print("Preprocessing done")