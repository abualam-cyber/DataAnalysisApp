import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List
import logging

class Visualizer:
    def __init__(self):
        self.data = None
        self.logger = logging.getLogger(__name__)
        self.color_schemes = {
            'default': None,
            'viridis': px.colors.sequential.Viridis,
            'plasma': px.colors.sequential.Plasma,
            'inferno': px.colors.sequential.Inferno,
            'magma': px.colors.sequential.Magma,
            'cividis': px.colors.sequential.Cividis,
            'rainbow': px.colors.sequential.Rainbow,
            'blues': px.colors.sequential.Blues,
            'reds': px.colors.sequential.Reds,
            'greens': px.colors.sequential.Greens
        }

    def set_data(self, data: pd.DataFrame) -> None:
        """Set the data for visualization."""
        try:
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")
            self.data = data.copy()
            self.logger.info(f"Data set successfully with {len(data)} rows and {len(data.columns)} columns")
        except Exception as e:
            self.logger.error(f"Error setting data: {str(e)}")
            raise

    def _get_colors(self, color_scheme: str) -> Optional[List[str]]:
        """Get color sequence for the specified scheme."""
        return self.color_schemes.get(color_scheme)

    def create_bar_chart(self, x_column: str, y_column: str, 
                        color_scheme: str = 'default', title: str = '') -> go.Figure:
        """Create a bar chart."""
        try:
            if self.data is None:
                raise ValueError("No data available. Please set data first.")
                
            if x_column not in self.data.columns:
                raise ValueError(f"Column '{x_column}' not found in data")
                
            if y_column not in self.data.columns:
                raise ValueError(f"Column '{y_column}' not found in data")
                
            colors = self._get_colors(color_scheme)
            fig = px.bar(self.data, x=x_column, y=y_column,
                        color_discrete_sequence=colors,
                        title=title or f"Bar Chart: {y_column} by {x_column}")
            
            fig.update_layout(
                xaxis_title=x_column,
                yaxis_title=y_column
            )
            return fig
        except Exception as e:
            self.logger.error(f"Error creating bar chart: {str(e)}")
            raise

    def create_line_chart(self, x_column: str, y_column: str,
                         color_scheme: str = 'default', title: str = '') -> go.Figure:
        """Create a line chart."""
        try:
            if self.data is None:
                raise ValueError("No data available. Please set data first.")
                
            if x_column not in self.data.columns:
                raise ValueError(f"Column '{x_column}' not found in data")
                
            if y_column not in self.data.columns:
                raise ValueError(f"Column '{y_column}' not found in data")
                
            colors = self._get_colors(color_scheme)
            fig = px.line(self.data, x=x_column, y=y_column,
                         color_discrete_sequence=colors,
                         title=title or f"Line Chart: {y_column} over {x_column}")
            
            fig.update_layout(
                xaxis_title=x_column,
                yaxis_title=y_column
            )
            return fig
        except Exception as e:
            self.logger.error(f"Error creating line chart: {str(e)}")
            raise

    def create_scatter_plot(self, x_column: str, y_column: str,
                          color_column: Optional[str] = None,
                          size_column: Optional[str] = None,
                          color_scheme: str = 'default',
                          title: str = '') -> go.Figure:
        """Create a scatter plot."""
        try:
            if self.data is None:
                raise ValueError("No data available. Please set data first.")
                
            if x_column not in self.data.columns:
                raise ValueError(f"Column '{x_column}' not found in data")
                
            if y_column not in self.data.columns:
                raise ValueError(f"Column '{y_column}' not found in data")
                
            if color_column and color_column not in self.data.columns:
                raise ValueError(f"Color column '{color_column}' not found in data")
                
            if size_column and size_column not in self.data.columns:
                raise ValueError(f"Size column '{size_column}' not found in data")
                
            colors = self._get_colors(color_scheme)
            fig = px.scatter(self.data, x=x_column, y=y_column,
                           color=color_column,
                           size=size_column,
                           color_discrete_sequence=colors,
                           title=title or f"Scatter Plot: {y_column} vs {x_column}")
            
            fig.update_layout(
                xaxis_title=x_column,
                yaxis_title=y_column
            )
            return fig
        except Exception as e:
            self.logger.error(f"Error creating scatter plot: {str(e)}")
            raise

    def create_histogram(self, column: str,
                        color_scheme: str = 'default',
                        title: str = '') -> go.Figure:
        """Create a histogram."""
        try:
            if self.data is None:
                raise ValueError("No data available. Please set data first.")
                
            if column not in self.data.columns:
                raise ValueError(f"Column '{column}' not found in data")
                
            colors = self._get_colors(color_scheme)
            fig = px.histogram(self.data, x=column,
                             color_discrete_sequence=colors,
                             title=title or f"Histogram of {column}")
            
            fig.update_layout(
                xaxis_title=column,
                yaxis_title="Count"
            )
            return fig
        except Exception as e:
            self.logger.error(f"Error creating histogram: {str(e)}")
            raise

    def create_box_plot(self, x_column: str, y_column: str,
                       color_scheme: str = 'default',
                       title: str = '') -> go.Figure:
        """Create a box plot."""
        try:
            if self.data is None:
                raise ValueError("No data available. Please set data first.")
                
            if x_column not in self.data.columns:
                raise ValueError(f"Column '{x_column}' not found in data")
                
            if y_column not in self.data.columns:
                raise ValueError(f"Column '{y_column}' not found in data")
                
            colors = self._get_colors(color_scheme)
            fig = px.box(self.data, x=x_column, y=y_column,
                        color_discrete_sequence=colors,
                        title=title or f"Box Plot: {y_column} by {x_column}")
            
            fig.update_layout(
                xaxis_title=x_column,
                yaxis_title=y_column
            )
            return fig
        except Exception as e:
            self.logger.error(f"Error creating box plot: {str(e)}")
            raise

    def create_heatmap(self, title: str = '') -> go.Figure:
        """Create a correlation heatmap."""
        try:
            if self.data is None:
                raise ValueError("No data available. Please set data first.")
                
            numeric_data = self.data.select_dtypes(include=[np.number])
            if numeric_data.empty:
                raise ValueError("No numeric columns available for correlation")
                
            corr_matrix = numeric_data.corr()
            fig = px.imshow(corr_matrix,
                          color_continuous_scale=px.colors.sequential.RdBu,
                          title=title or "Correlation Heatmap")
            
            fig.update_layout(
                xaxis_title="Features",
                yaxis_title="Features"
            )
            return fig
        except Exception as e:
            self.logger.error(f"Error creating heatmap: {str(e)}")
            raise

    def create_bubble_chart(self, x_column: str, y_column: str,
                          size_column: str,
                          color_column: Optional[str] = None,
                          color_scheme: str = 'default',
                          title: str = '') -> go.Figure:
        """Create a bubble chart."""
        try:
            if self.data is None:
                raise ValueError("No data available. Please set data first.")
                
            if not size_column:
                raise ValueError("Bubble chart requires a size parameter")
                
            if x_column not in self.data.columns:
                raise ValueError(f"Column '{x_column}' not found in data")
                
            if y_column not in self.data.columns:
                raise ValueError(f"Column '{y_column}' not found in data")
                
            if size_column not in self.data.columns:
                raise ValueError(f"Column '{size_column}' not found in data")
                
            if color_column and color_column not in self.data.columns:
                raise ValueError(f"Color column '{color_column}' not found in data")
                
            colors = self._get_colors(color_scheme)
            fig = px.scatter(self.data, x=x_column, y=y_column,
                           size=size_column,
                           color=color_column,
                           color_discrete_sequence=colors,
                           title=title or f"Bubble Chart: {y_column} vs {x_column}")
            
            fig.update_layout(
                xaxis_title=x_column,
                yaxis_title=y_column
            )
            return fig
        except Exception as e:
            self.logger.error(f"Error creating bubble chart: {str(e)}")
            raise

    def create_3d_scatter(self, x_column: str, y_column: str, z_column: str,
                         size_column: Optional[str] = None,
                         color_scheme: str = 'default',
                         title: str = '') -> go.Figure:
        """Create a 3D scatter plot."""
        try:
            if self.data is None:
                raise ValueError("No data available. Please set data first.")
                
            if x_column not in self.data.columns:
                raise ValueError(f"Column '{x_column}' not found in data")
                
            if y_column not in self.data.columns:
                raise ValueError(f"Column '{y_column}' not found in data")
                
            if z_column not in self.data.columns:
                raise ValueError(f"Column '{z_column}' not found in data")
                
            if size_column and size_column not in self.data.columns:
                raise ValueError(f"Size column '{size_column}' not found in data")
                
            colors = self._get_colors(color_scheme)
            
            fig = go.Figure()
            
            # Add scatter3d trace
            scatter_kwargs = {
                'x': self.data[x_column],
                'y': self.data[y_column],
                'z': self.data[z_column],
                'mode': 'markers',
            }
            
            if size_column:
                scatter_kwargs['marker'] = {
                    'size': self.data[size_column],
                    'sizeref': self.data[size_column].max() / 20,
                }
            
            if colors:
                scatter_kwargs['marker'] = scatter_kwargs.get('marker', {})
                scatter_kwargs['marker']['color'] = colors[0]
                
            fig.add_trace(go.Scatter3d(**scatter_kwargs))
            
            # Update layout
            fig.update_layout(
                title=title or f"3D Scatter Plot: {x_column} vs {y_column} vs {z_column}",
                scene=dict(
                    xaxis_title=x_column,
                    yaxis_title=y_column,
                    zaxis_title=z_column
                )
            )
            
            return fig
            
        except Exception as e:
            self.logger.error(f"Error creating 3D scatter plot: {str(e)}")
            raise

    def create_3d_surface(self, x_column: str, y_column: str,
                         z_column: str, title: str = '') -> go.Figure:
        """Create a 3D surface plot."""
        try:
            if self.data is None:
                raise ValueError("No data available. Please set data first.")
                
            if x_column not in self.data.columns:
                raise ValueError(f"Column '{x_column}' not found in data")
                
            if y_column not in self.data.columns:
                raise ValueError(f"Column '{y_column}' not found in data")
                
            if z_column not in self.data.columns:
                raise ValueError(f"Column '{z_column}' not found in data")
                
            fig = go.Figure(data=[go.Surface(z=self.data.pivot(x_column, y_column, z_column).values)])
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis_title=x_column,
                    yaxis_title=y_column,
                    zaxis_title=z_column
                )
            )
            return fig
        except Exception as e:
            self.logger.error(f"Error creating 3D surface plot: {str(e)}")
            raise

    def create_visualization(self, data: pd.DataFrame, chart_type: str,
                           x_column: str, y_column: str = None,
                           color_column: str = None, size_column: str = None,
                           color_scheme: str = 'default') -> go.Figure:
        """Create visualization based on the specified chart type."""
        try:
            # Set the data
            self.set_data(data)
            
            # Create visualization based on chart type
            if chart_type == 'bar':
                return self.create_bar_chart(x_column, y_column, color_scheme)
            elif chart_type == 'line':
                return self.create_line_chart(x_column, y_column, color_scheme)
            elif chart_type == 'scatter':
                return self.create_scatter_plot(x_column, y_column, color_column, size_column, color_scheme)
            elif chart_type == 'histogram':
                return self.create_histogram(x_column, color_scheme)
            elif chart_type == 'box':
                return self.create_box_plot(x_column, y_column, color_scheme)
            elif chart_type == 'heatmap':
                return self.create_heatmap()
            elif chart_type == 'bubble':
                if not size_column:
                    raise ValueError("Bubble chart requires a size parameter")
                return self.create_bubble_chart(x_column, y_column, size_column, color_column, color_scheme)
            elif chart_type == '3d_scatter':
                if not y_column:
                    raise ValueError("3D scatter plot requires a Z-axis parameter")
                return self.create_3d_scatter(x_column, y_column, color_column, size_column, color_scheme)
            else:
                raise ValueError(f"Unknown chart type: {chart_type}")
                
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            raise

    def get_available_color_schemes(self) -> List[str]:
        """Return list of available color schemes."""
        return list(self.color_schemes.keys())
