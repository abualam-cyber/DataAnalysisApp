import pandas as pd
import numpy as np
from datetime import datetime
import polars as pl
from typing import Optional, Union, List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataHandler:
    def __init__(self):
        self.data = None
        self.original_data = None
        self.date_columns = []
        self.numeric_columns = []
        self.categorical_columns = []
        self._setup_logging()

    def _setup_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def load_data(self, file_path: str) -> bool:
        """Load data from various file formats."""
        try:
            file_extension = file_path.split('.')[-1].lower()
            
            if file_extension == 'csv':
                self.data = pd.read_csv(file_path, parse_dates=True)
            elif file_extension in ['xls', 'xlsx']:
                self.data = pd.read_excel(file_path)
            elif file_extension == 'parquet':
                self.data = pd.read_parquet(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
            
            self.original_data = self.data.copy()
            self._identify_column_types()
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            return False

    def _identify_column_types(self):
        """Identify column types in the dataset."""
        self.date_columns = []
        self.numeric_columns = []
        self.categorical_columns = []

        for column in self.data.columns:
            if pd.api.types.is_datetime64_any_dtype(self.data[column]):
                self.date_columns.append(column)
            elif pd.api.types.is_numeric_dtype(self.data[column]):
                self.numeric_columns.append(column)
            else:
                # Try to convert to datetime
                try:
                    pd.to_datetime(self.data[column])
                    self.date_columns.append(column)
                except:
                    self.categorical_columns.append(column)

    def clean_data(self) -> None:
        """Clean the dataset by handling missing values and outliers."""
        if self.data is None:
            return

        # Handle missing values
        for column in self.numeric_columns:
            self.data[column].fillna(self.data[column].mean(), inplace=True)
        
        for column in self.categorical_columns:
            self.data[column].fillna(self.data[column].mode()[0], inplace=True)
        
        for column in self.date_columns:
            self.data[column].fillna(method='ffill', inplace=True)

        # Remove duplicate rows
        self.data.drop_duplicates(inplace=True)

    def get_column_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for each column."""
        stats = {}
        
        for column in self.data.columns:
            column_stats = {
                'type': str(self.data[column].dtype),
                'missing': self.data[column].isnull().sum(),
                'unique_values': len(self.data[column].unique())
            }
            
            if column in self.numeric_columns:
                column_stats.update({
                    'mean': float(self.data[column].mean()),
                    'std': float(self.data[column].std()),
                    'min': float(self.data[column].min()),
                    'max': float(self.data[column].max())
                })
            
            stats[column] = column_stats
        
        return stats

    def get_columns_by_type(self) -> Dict[str, List[str]]:
        """Get columns grouped by their types."""
        return {
            'date': self.date_columns,
            'numeric': self.numeric_columns,
            'categorical': self.categorical_columns
        }

    def get_data_sample(self, n: int = 5) -> pd.DataFrame:
        """Get a sample of the data."""
        return self.data.head(n) if self.data is not None else None

    def get_correlation_matrix(self) -> Optional[pd.DataFrame]:
        """Get correlation matrix for numeric columns."""
        if not self.numeric_columns:
            return None
        return self.data[self.numeric_columns].corr()

    def filter_data(self, conditions: Dict[str, Any]) -> None:
        """Filter data based on conditions."""
        if self.data is None:
            return

        for column, condition in conditions.items():
            if column not in self.data.columns:
                continue

            if isinstance(condition, (list, tuple)):
                self.data = self.data[self.data[column].isin(condition)]
            elif isinstance(condition, dict):
                if 'min' in condition:
                    self.data = self.data[self.data[column] >= condition['min']]
                if 'max' in condition:
                    self.data = self.data[self.data[column] <= condition['max']]
            else:
                self.data = self.data[self.data[column] == condition]

    def reset_data(self) -> None:
        """Reset data to original state."""
        if self.original_data is not None:
            self.data = self.original_data.copy()
            self._identify_column_types()

    def get_groupby_stats(self, group_by: str, agg_columns: List[str]) -> pd.DataFrame:
        """Get aggregated statistics for specified columns grouped by a column."""
        if not all(col in self.data.columns for col in [group_by] + agg_columns):
            return None

        agg_dict = {}
        for col in agg_columns:
            if col in self.numeric_columns:
                agg_dict[col] = ['mean', 'sum', 'count', 'std']
            else:
                agg_dict[col] = ['count', 'nunique']

        return self.data.groupby(group_by).agg(agg_dict)
