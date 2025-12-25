"""Helper functions"""
import pandas as pd
from datetime import datetime

def format_currency(value: float) -> str:
    return f"${value:,.2f}"

def format_percentage(value: float) -> str:
    return f"{value:+.2f}%"

def create_timestamp() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def save_results(data: pd.DataFrame, filename: str, directory: str = './data'):
    import os
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, filename)
    data.to_csv(filepath, index=False)
    return filepath