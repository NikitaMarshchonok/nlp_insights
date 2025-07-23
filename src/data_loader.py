# src/data_loader.py

import pandas as pd


def load_data(path: str, nrows: int = None) -> pd.DataFrame:
    """
    Load the Amazon Fine Food Reviews dataset from CSV.

    :param path: путь к файлу Reviews.csv
    :param nrows: сколько строк читать (None — весь файл)
    :return: DataFrame с загруженными данными
    """
    df = pd.read_csv(path, nrows=nrows)
    return df
