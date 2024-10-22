import pandas as pd


def to_dict(keys: pd.Series, values: pd.Series) -> dict:
    return dict(zip(keys, values))
