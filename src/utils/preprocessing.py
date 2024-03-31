import ast
import pandas as pd
from datasets import load_dataset
from typing import Tuple
from sklearn.model_selection import train_test_split

from constants import CATEGORIES



def load_dataframes2(sampling_strategy: str = 'undersample', test_size: float = 0.2, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset = load_dataset("lmsys/toxic-chat", "toxicchat0124")
    
    # Prétraitement des données
    df = preprocess_df(dataset['train'].to_pandas())
    
    # Séparation des classes
    df_toxic = df[df['toxicity'] == 1]
    df_non_toxic = df[df['toxicity'] == 0]
    
    # Application de la stratégie d'équilibrage
    if sampling_strategy == 'undersample':
        df_non_toxic_sampled = df_non_toxic.sample(n=len(df_toxic), random_state=seed)
        df_balanced = pd.concat([df_toxic, df_non_toxic_sampled])
    elif sampling_strategy == 'oversample':
        df_toxic_sampled = df_toxic.sample(n=len(df_non_toxic), replace=True, random_state=seed)
        df_balanced = pd.concat([df_toxic_sampled, df_non_toxic])
    else:
        raise ValueError("sampling_strategy doit être 'undersample' ou 'oversample'")
    
    # Séparation des données en jeux d'entraînement, de validation et de test
    df_train, df_temp = train_test_split(df_balanced, test_size=test_size, random_state=seed)
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=seed)
    
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