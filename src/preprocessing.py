import pandas as pd
import spacy
from tqdm.auto import tqdm
from typing import List

# Data preparation function
def prepare_data(df: pd.DataFrame, use_cols=None) -> pd.DataFrame:
    """
    Select relevant columns and convert UNIX timestamp to datetime.
    """
    if use_cols is None:
        use_cols = ['Score', 'Summary', 'Text', 'Time']
    df2 = df[use_cols].copy()
    df2['Time'] = pd.to_datetime(df2['Time'], unit='s')
    return df2

# Load spaCy model with fallback to download if missing
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
nlp.max_length = 10_000_000

# Single text cleaning

def clean_text(text: str) -> str:
    """
    Lemmatize, keep only alphabetic tokens, drop stop words.
    """
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

# Batch cleaning

def clean_corpus(texts: List[str], batch_size: int = 200, n_process: int = 1) -> List[str]:
    """
    Apply clean_text in batches using spaCy pipe for speed.
    """
    cleaned = []
    for doc in tqdm(nlp.pipe(texts, batch_size=batch_size, n_process=n_process), total=len(texts)):
        tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
        cleaned.append(" ".join(tokens))
    return cleaned
