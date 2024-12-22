# Author: Abu S. Alam <Gray Duck Systems>
#
# Description:
# This file implements the `Forecaster` class, which is responsible for generating forecasts
# using various models such as Linear Regression, Random Forest, Holt-Winters, and SARIMA.
# It preprocesses data, handles missing values, and creates visualizations for the forecasts.
# This module is integral to the application, enabling predictive analytics and supporting
# decision-making by providing accurate and interactive forecast results.


import numpy as np
import pandas as pd
import logging
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class Forecaster:
    def __init__(self):
        """Initialize the forecaster with various models."""
        self.data = None
        self.target_column = None
        self.date_column = None
        self.forecast_steps = 30
        self.forecast_cache = {}
        self.logger = logging.getLogger(__name__)
        self.scaler = MinMaxScaler()
        self.models = {
            'linear': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100),
            'holt_winters': None,  # Will be initialized when needed
            'sarima': None  # Will be initialized when needed
        }
        self.fitted_models = {}

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess data by handling missing values and ensuring proper formatting."""
        processed_data = data.copy()
        
        try:
            # Handle missing values in target column
            if processed_data[self.target_column].isnull().any():
                # For time series data, forward fill then backward fill
                processed_data[self.target_column] = processed_data[self.target_column].fillna(method='ffill')
                processed_data[self.target_column] = processed_data[self.target_column].fillna(method='bfill')
                
                # If still have missing values (e.g., at the start), fill with mean
                if processed_data[self.target_column].isnull().any():
                    mean_value = processed_data[self.target_column].mean()
                    processed_data[self.target_column] = processed_data[self.target_column].fillna(mean_value)
            
            # Handle missing dates by reindexing with full date range
            processed_data[self.date_column] = pd.to_datetime(processed_data[self.date_column])
            date_range = pd.date_range(
                start=processed_data[self.date_column].min(),
                end=processed_data[self.date_column].max(),
                freq='D'  # Daily frequency
            )
            processed_data = processed_data.set_index(self.date_column)
            processed_data = processed_data.reindex(date_range)
            processed_data.index.name = self.date_column
            processed_data = processed_data.reset_index()
            
            # Handle any new missing values from reindexing
            processed_data[self.target_column] = processed_data[self.target_column].interpolate(method='time')
            
            # Remove any remaining missing values
            processed_data = processed_data.dropna()
            
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error preprocessing data: {str(e)}")
            raise ValueError(f"Error preprocessing data: {str(e)}")

    def set_data(self, data: pd.DataFrame, target_column: str, date_column: str):
        """Set the data for forecasting."""
        if data is None or target_column is None or date_column is None:
            raise ValueError("Data, target column, and date column must be provided")
            
        try:
            self.data = data.copy()
            self.target_column = target_column
            self.date_column = date_column
            
            # Preprocess the data
            self.data = self._preprocess_data(self.data)
            
            # Sort by date
            self.data = self.data.sort_values(self.date_column)
            
            self.logger.info(f"Data set successfully with {len(self.data)} rows")
            
        except Exception as e:
            self.logger.error(f"Error setting data: {str(e)}")
            raise ValueError(f"Error setting data: {str(e)}")

    def create_forecast(self, model_type: str, periods: int = 30) -> Dict[str, Any]:
        """Create a forecast using the specified model type."""
        if self.data is None:
            raise ValueError("Data not set. Call set_data first.")

        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")

        try:
            # Store forecast periods
            self.forecast_steps = periods
            
            # Ensure we have enough data for training
            min_train_size = 2 * periods  # At least 2x the forecast period
            if len(self.data) < min_train_size:
                raise ValueError(f"Not enough data for forecasting. Need at least {min_train_size} points.")
            
            # Split data into train and test
            train_size = int(len(self.data) * 0.8)
            train_data = self.data[:train_size]
            test_data = self.data[train_size:]

            # Create forecast based on model type
            if model_type == 'linear':
                forecast_result = self._linear_forecast(train_data, test_data)
            elif model_type == 'random_forest':
                forecast_result = self._random_forest_forecast(train_data, test_data)
            elif model_type == 'holt_winters':
                forecast_result = self._holt_winters_forecast(train_data, test_data)
            elif model_type == 'sarima':
                forecast_result = self._sarima_forecast(train_data, test_data)
            else:
                raise ValueError(f"Unknown model type: {model_type}")

            self.forecast_cache[model_type] = forecast_result
            return forecast_result

        except Exception as e:
            self.logger.error(f"Error creating forecast with {model_type}: {str(e)}")
            raise

    def _calculate_metrics(self, actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
        """Calculate forecast accuracy metrics."""
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predicted))
        return {'MSE': mse, 'RMSE': rmse, 'MAE': mae}

    def _create_forecast_plot(self, train_data: pd.DataFrame, test_data: pd.DataFrame, 
                            forecast: np.ndarray, title: str) -> go.Figure:
        """Create an interactive plot of the forecast."""
        fig = go.Figure()

        # Add training data
        fig.add_trace(go.Scatter(
            x=train_data[self.date_column],
            y=train_data[self.target_column],
            name='Training Data',
            mode='lines'
        ))

        # Add test data
        fig.add_trace(go.Scatter(
            x=test_data[self.date_column],
            y=test_data[self.target_column],
            name='Test Data',
            mode='lines'
        ))

        # Add forecast
        fig.add_trace(go.Scatter(
            x=test_data[self.date_column],
            y=forecast[:len(test_data)],
            name='Forecast',
            mode='lines'
        ))

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Value',
            showlegend=True
        )

        return fig

    def _create_features(self, data: pd.DataFrame) -> np.ndarray:
        """Create features for ML models."""
        features = []
        dates = pd.to_datetime(data[self.date_column])
        
        features.append(dates.dt.year)
        features.append(dates.dt.month)
        features.append(dates.dt.day)
        features.append(dates.dt.dayofweek)
        features.append(dates.dt.quarter)
        
        return np.column_stack(features)

    def _linear_forecast(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Create forecast using Linear Regression."""
        # Prepare features
        train_features = self._create_features(train_data)
        test_features = self._create_features(test_data)
        
        # Train model
        model = self.models['linear']
        model.fit(train_features, train_data[self.target_column])
        
        # Make predictions
        forecast = model.predict(test_features)
        
        # Calculate metrics
        metrics = self._calculate_metrics(test_data[self.target_column], forecast)
        
        # Create visualization
        fig = self._create_forecast_plot(
            train_data, test_data, forecast,
            'Linear Regression Forecast'
        )
        
        return {
            'forecast': forecast,
            'metrics': metrics,
            'figure': fig
        }

    def _random_forest_forecast(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Create forecast using Random Forest."""
        # Prepare features
        train_features = self._create_features(train_data)
        test_features = self._create_features(test_data)
        
        # Train model
        model = self.models['random_forest']
        model.fit(train_features, train_data[self.target_column])
        
        # Make predictions
        forecast = model.predict(test_features)
        
        # Calculate metrics
        metrics = self._calculate_metrics(test_data[self.target_column], forecast)
        
        # Create visualization
        fig = self._create_forecast_plot(
            train_data, test_data, forecast,
            'Random Forest Forecast'
        )
        
        return {
            'forecast': forecast,
            'metrics': metrics,
            'figure': fig
        }

    def _holt_winters_forecast(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Create forecast using Holt-Winters method."""
        # Fit model
        model = ExponentialSmoothing(
            train_data[self.target_column],
            seasonal_periods=12,
            trend='add',
            seasonal='add'
        )
        fitted_model = model.fit()
        
        # Make predictions
        forecast = fitted_model.forecast(len(test_data))
        
        # Calculate metrics
        metrics = self._calculate_metrics(test_data[self.target_column], forecast)
        
        # Create visualization
        fig = self._create_forecast_plot(
            train_data, test_data, forecast,
            'Holt-Winters Forecast'
        )
        
        return {
            'forecast': forecast,
            'metrics': metrics,
            'figure': fig
        }

    def _sarima_forecast(self, train_data: pd.DataFrame, test_data: pd.DataFrame) -> Dict[str, Any]:
        """Create forecast using SARIMA."""
        # Fit model
        model = SARIMAX(
            train_data[self.target_column],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12)
        )
        fitted_model = model.fit(disp=False)
        
        # Make predictions
        forecast = fitted_model.forecast(len(test_data))
        
        # Calculate metrics
        metrics = self._calculate_metrics(test_data[self.target_column], forecast)
        
        # Create visualization
        fig = self._create_forecast_plot(
            train_data, test_data, forecast,
            'SARIMA Forecast'
        )
        
        return {
            'forecast': forecast,
            'metrics': metrics,
            'figure': fig
        }

    def get_available_forecasts(self) -> List[str]:
        """Get list of available forecast models."""
        return list(self.forecast_cache.keys())

    def get_available_models(self) -> List[str]:
        """Return list of available forecasting models."""
        return ['linear', 'random_forest', 'holt_winters', 'sarima']
