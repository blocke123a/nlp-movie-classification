'''
__init__.py
'''

# Define the __all__ variable
__all__ = ["load_config", "get_data", "process_data", "train_bayes"]

from .ingest import get_data
from .utils import load_config
from .process import process_data
from .naive_bayes import train_bayes
