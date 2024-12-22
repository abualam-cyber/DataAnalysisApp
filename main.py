"""
Advanced Data Analysis Dashboard

This script creates a comprehensive dashboard application using the Dash framework to facilitate advanced data analysis. 
It integrates functionality for:
- Data upload and preprocessing.
- Interactive data visualization.
- Time-series forecasting with multiple models.
- Automated report generation.

Modules and Classes:
- DataHandler: Handles data ingestion, preprocessing, and type identification.
- Visualizer: Generates different types of visualizations.
- Forecaster: Provides forecasting capabilities with various statistical and ML models.
- ReportGenerator: Creates comprehensive reports based on user-selected content.

Author: Abu S. Alam <Gray Duck Systems>
"""


import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import base64
import io
import json
import logging
from datetime import datetime
import os

from data_handler import DataHandler
from visualizer import Visualizer
from forecaster import Forecaster
from report_generator import ReportGenerator

# Initialize components
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
data_handler = DataHandler()
visualizer = Visualizer()
forecaster = Forecaster()
report_generator = ReportGenerator()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for storing current state
current_visualization = None
current_forecast = None
current_forecast_metrics = None

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Advanced Data Analysis Dashboard", className="text-center mb-4"),
            
            # File Upload Section
            dbc.Card([
                dbc.CardHeader("Data Upload"),
                dbc.CardBody([
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '60px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'margin': '10px'
                        },
                        multiple=False
                    ),
                    html.Div(id='output-data-upload')
                ])
            ], className="mb-4"),
            
            # Visualization Section
            dbc.Card([
                dbc.CardHeader("Data Visualization"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Chart Type"),
                            dcc.Dropdown(
                                id='chart-type',
                                options=[
                                    {'label': 'Line Chart', 'value': 'line'},
                                    {'label': 'Bar Chart', 'value': 'bar'},
                                    {'label': 'Scatter Plot', 'value': 'scatter'},
                                    {'label': 'Histogram', 'value': 'histogram'},
                                    {'label': 'Box Plot', 'value': 'box'},
                                    {'label': 'Heatmap', 'value': 'heatmap'},
                                    {'label': 'Bubble Chart', 'value': 'bubble'},
                                    {'label': '3D Scatter', 'value': '3d_scatter'}
                                ],
                                placeholder="Select chart type"
                            ),
                            
                            html.Label("X-Axis", className="mt-3"),
                            dcc.Dropdown(id='x-axis', placeholder="Select X-axis column"),
                            
                            html.Label("Y-Axis", className="mt-3"),
                            dcc.Dropdown(id='y-axis', placeholder="Select Y-axis column"),
                            
                            html.Label("Color By", className="mt-3"),
                            dcc.Dropdown(id='color-by', placeholder="Select color column"),
                            
                            html.Label("Size By", className="mt-3"),
                            dcc.Dropdown(id='size-by', placeholder="Select size column"),
                            
                            html.Label("Color Scheme", className="mt-3"),
                            dcc.Dropdown(
                                id='color-scheme',
                                options=[
                                    {'label': 'Default', 'value': 'default'},
                                    {'label': 'Viridis', 'value': 'viridis'},
                                    {'label': 'Plasma', 'value': 'plasma'},
                                    {'label': 'Inferno', 'value': 'inferno'},
                                    {'label': 'Magma', 'value': 'magma'},
                                    {'label': 'Cividis', 'value': 'cividis'}
                                ],
                                placeholder="Select color scheme"
                            ),
                            
                            dbc.Button("Create Chart", id="create-viz-button", 
                                     color="primary", className="mt-4")
                        ], width=4)
                    ], className="mt-3"),
                    
                    html.Div(id='viz-error', className="mt-3 text-danger"),
                    dcc.Graph(id='visualization-output', className="mt-3")
                ])
            ], className="mb-4"),
            
            # Forecasting Section
            dbc.Card([
                dbc.CardHeader("Forecasting"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Target Column"),
                            dcc.Dropdown(id='forecast-target', placeholder="Select target column"),
                            
                            html.Label("Date Column", className="mt-3"),
                            dcc.Dropdown(id='forecast-date', placeholder="Select date column"),
                            
                            html.Label("Model Type", className="mt-3"),
                            dcc.Dropdown(
                                id='forecast-model',
                                options=[
                                    {'label': 'Linear Regression', 'value': 'linear'},
                                    {'label': 'SARIMA', 'value': 'sarima'},
                                    {'label': 'Holt-Winters', 'value': 'holt_winters'},
                                    {'label': 'Random Forest', 'value': 'random_forest'}
                                ],
                                placeholder="Select model type"
                            ),
                            
                            html.Label("Forecast Periods", className="mt-3"),
                            dcc.Input(
                                id='forecast-periods',
                                type='number',
                                min=1,
                                value=30,
                                className="form-control"
                            ),
                            
                            dbc.Button("Create Forecast", id="create-forecast-button", 
                                     color="primary", className="mt-4")
                        ], width=4)
                    ]),
                    
                    html.Div(id='forecast-error', className="mt-3 text-danger"),
                    dcc.Graph(id='forecast-output', className="mt-3"),
                    html.Div(id='forecast-metrics', className="mt-3")
                ])
            ], className="mb-4"),
            
            # Report Generation Section
            dbc.Card([
                dbc.CardHeader("Report Generation"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Report Title"),
                            dcc.Input(
                                id='report-title',
                                type='text',
                                placeholder='Enter report title',
                                className="form-control"
                            ),
                            
                            html.Label("Report Contents", className="mt-3"),
                            dcc.Checklist(
                                id='report-contents',
                                options=[
                                    {'label': ' Raw Data Preview', 'value': 'raw_data'},
                                    {'label': ' Summary Statistics', 'value': 'summary_stats'},
                                    {'label': ' Data Quality Analysis', 'value': 'data_quality'},
                                    {'label': ' Correlation Analysis', 'value': 'correlation'},
                                    {'label': ' Visualizations', 'value': 'visualizations'},
                                    {'label': ' Forecasting', 'value': 'forecasting'}
                                ],
                                value=[],
                                className="mt-2"
                            ),
                            
                            html.Label("Report Format", className="mt-3"),
                            dcc.RadioItems(
                                id='report-format',
                                options=[
                                    {'label': ' HTML', 'value': 'html'},
                                    {'label': ' PDF', 'value': 'pdf'}
                                ],
                                value='html',
                                className="mt-2"
                            ),
                            
                            dbc.Button("Generate Report", id="generate-report-button", 
                                     color="primary", className="mt-4")
                        ], width=4)
                    ]),
                    html.Div(id='report-status', className="mt-3")
                ])
            ])
        ], width=12)
    ])
], fluid=True)

def parse_contents(contents, filename):
    """Parse uploaded file contents."""
    try:
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        
        try:
            if 'csv' in filename:
                df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
            elif 'xls' in filename:
                df = pd.read_excel(io.BytesIO(decoded))
            elif 'parquet' in filename:
                df = pd.read_parquet(io.BytesIO(decoded))
            else:
                return html.Div([
                    'Unsupported file type. Please upload a CSV, Excel, or Parquet file.'
                ])
        except Exception as e:
            print(e)
            return html.Div([
                'There was an error processing this file.'
            ])

        data_handler.data = df
        data_handler._identify_column_types()
        
        return html.Div([
            html.H5(f'Uploaded: {filename}'),
            
            html.H6("Data Preview:"),
            dash.dash_table.DataTable(
                data=df.head().to_dict('records'),
                columns=[{'name': i, 'id': i} for i in df.columns],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
                style_header={
                    'backgroundColor': 'rgb(230, 230, 230)',
                    'fontWeight': 'bold'
                }
            ),
            
            html.Hr(),
            html.H6("Column Types:"),
            html.Div([
                html.P(f"Numeric Columns: {', '.join(data_handler.numeric_columns)}"),
                html.P(f"Categorical Columns: {', '.join(data_handler.categorical_columns)}"),
                html.P(f"Date Columns: {', '.join(data_handler.date_columns)}")
            ])
        ])
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

@app.callback(
    [Output('output-data-upload', 'children'),
     Output('x-axis', 'options'),
     Output('y-axis', 'options'),
     Output('color-by', 'options'),
     Output('size-by', 'options'),
     Output('forecast-target', 'options'),
     Output('forecast-date', 'options')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    """Update the app when data is uploaded."""
    if contents is None:
        return html.Div("Upload a file."), [], [], [], [], [], []

    children = parse_contents(contents, filename)
    
    if data_handler.data is not None:
        columns = [{'label': col, 'value': col} for col in data_handler.data.columns]
        numeric_cols = [{'label': col, 'value': col} for col in data_handler.numeric_columns]
        date_cols = [{'label': col, 'value': col} for col in data_handler.date_columns]
        return children, columns, columns, columns, numeric_cols, numeric_cols, date_cols
    
    return children, [], [], [], [], [], []

@app.callback(
    [Output('visualization-output', 'figure'),
     Output('viz-error', 'children')],
    Input('create-viz-button', 'n_clicks'),
    [State('chart-type', 'value'),
     State('x-axis', 'value'),
     State('y-axis', 'value'),
     State('color-by', 'value'),
     State('size-by', 'value'),
     State('color-scheme', 'value')]
)
def update_visualization(n_clicks, chart_type, x, y, color_by, size_by, color_scheme):
    """Create and update visualizations."""
    global current_visualization
    
    if n_clicks is None:
        return go.Figure(), ""
        
    if not chart_type:
        return go.Figure(), "Please select a chart type"
        
    if not x:
        return go.Figure(), "Please select an X-axis column"
        
    if chart_type != 'histogram' and not y:
        return go.Figure(), "Please select a Y-axis column"
        
    if chart_type == 'bubble' and not size_by:
        return go.Figure(), "Bubble chart requires a size parameter"
        
    if chart_type == '3d_scatter' and not color_by:
        return go.Figure(), "3D scatter plot requires a Z-axis parameter"

    try:
        logger.info(f"Creating visualization with: chart_type={chart_type}, x={x}, y={y}, color_by={color_by}, size_by={size_by}, color_scheme={color_scheme}")
        logger.info(f"Data shape: {data_handler.data.shape}")
        logger.info(f"Data columns: {data_handler.data.columns.tolist()}")
        
        # Set data in visualizer
        visualizer.set_data(data_handler.data)
        
        # Create visualization based on chart type
        if chart_type == 'bar':
            fig = visualizer.create_bar_chart(x, y, color_scheme)
        elif chart_type == 'line':
            fig = visualizer.create_line_chart(x, y, color_scheme)
        elif chart_type == 'scatter':
            fig = visualizer.create_scatter_plot(x, y, color_by, size_by, color_scheme)
        elif chart_type == 'histogram':
            fig = visualizer.create_histogram(x, color_scheme)
        elif chart_type == 'box':
            fig = visualizer.create_box_plot(x, y, color_scheme)
        elif chart_type == 'heatmap':
            fig = visualizer.create_heatmap()
        elif chart_type == 'bubble':
            fig = visualizer.create_bubble_chart(x, y, size_by, color_by, color_scheme)
        elif chart_type == '3d_scatter':
            fig = visualizer.create_3d_scatter(x, y, color_by, size_by, color_scheme)
        else:
            return go.Figure(), f"Unknown chart type: {chart_type}"
        
        # Store current visualization
        current_visualization = fig
        
        if fig == go.Figure():
            logger.error("Empty figure returned from visualizer")
            return fig, "Error creating visualization. Please check the logs."
            
        return fig, ""
        
    except Exception as e:
        logger.error(f"Error in visualization callback: {str(e)}")
        return go.Figure(), str(e)

@app.callback(
    [Output('forecast-output', 'figure'),
     Output('forecast-metrics', 'children')],
    Input('create-forecast-button', 'n_clicks'),
    [State('forecast-target', 'value'),
     State('forecast-date', 'value'),
     State('forecast-model', 'value'),
     State('forecast-periods', 'value')]
)
def update_forecast(n_clicks, target, date, model, periods):
    """Create and update forecasts."""
    global current_forecast, current_forecast_metrics
    
    if n_clicks is None:
        return go.Figure(), ""
        
    if not target or not date or not model:
        return go.Figure(), "Please fill in all required fields"

    try:
        # Convert periods to int with default value
        try:
            periods = int(periods) if periods else 30
        except (ValueError, TypeError):
            periods = 30

        # Set data for forecasting
        forecaster.set_data(data_handler.data, target, date)
        
        # Create forecast
        forecast_result = forecaster.create_forecast(model, periods)
        
        if not forecast_result:
            raise ValueError("Failed to create forecast")
            
        fig = forecast_result.get('figure')
        metrics = forecast_result.get('metrics', {})
        
        if not fig:
            raise ValueError("No visualization available")
            
        # Store current forecast and metrics
        current_forecast = fig
        current_forecast_metrics = metrics
            
        metrics_div = html.Div([
            html.H6("Forecast Metrics:", className="mt-3"),
            html.Table([
                html.Tr([html.Td("RMSE:"), html.Td(f"{metrics.get('RMSE', 0):.3f}")]),
                html.Tr([html.Td("MAE:"), html.Td(f"{metrics.get('MAE', 0):.3f}")]),
                html.Tr([html.Td("MSE:"), html.Td(f"{metrics.get('MSE', 0):.3f}")])
            ], className="table table-sm")
        ])
        
        return fig, metrics_div
        
    except Exception as e:
        logger.error(f"Error creating forecast: {str(e)}")
        return go.Figure(), html.Div([
            html.H6("Error:", style={'color': 'red'}),
            html.P(str(e))
        ])

@app.callback(
    Output('report-status', 'children'),
    [Input('generate-report-button', 'n_clicks')],
    [State('report-title', 'value'),
     State('report-contents', 'value'),
     State('report-format', 'value')]
)
def generate_report(n_clicks, title, contents, report_format):
    """Generate the report with selected contents."""
    if n_clicks is None:
        return ""
        
    if not title:
        return html.Div("Please enter a report title", style={'color': 'red'})
        
    if not contents:
        return html.Div("Please select report contents", style={'color': 'red'})

    try:
        # Create reports directory if it doesn't exist
        output_dir = os.path.join(os.getcwd(), "reports")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"report_{timestamp}.{report_format}"
        output_path = os.path.join(output_dir, filename)
        
        # Set data and clear previous report
        report_generator.set_data(data_handler.data)
        report_generator.clear()
        
        # Add selected contents
        if contents:
            if 'raw_data' in contents:
                report_generator.add_raw_data_preview()
            if 'summary_stats' in contents:
                report_generator.add_summary_statistics()
            if 'data_quality' in contents:
                report_generator.add_data_quality_analysis()
            if 'correlation' in contents:
                report_generator.add_correlation_analysis()
                
        # Add current visualization if available
        if 'visualizations' in contents and current_visualization:
            report_generator.add_visualization(
                current_visualization,
                "Current Visualization"
            )
                    
        # Add current forecast if available
        if 'forecasting' in contents and current_forecast:
            report_generator.add_forecast(
                current_forecast,
                "Current Forecast",
                current_forecast_metrics
            )
                    
        # Generate the report
        success = report_generator.generate_report(title, output_path, report_format)
        
        if success:
            return html.Div([
                html.P(f"Report generated successfully!", style={'color': 'green'}),
                html.P(f"Saved to: {output_path}", style={'font-style': 'italic'})
            ])
        else:
            return html.Div("Error generating report. Please check the logs.", style={'color': 'red'})
            
    except Exception as e:
        logger.error(f"Error in report generation: {str(e)}")
        return html.Div([
            html.P("Error generating report:", style={'color': 'red'}),
            html.P(str(e), style={'font-style': 'italic'})
        ])

if __name__ == '__main__':
    import sys
    import webbrowser
    from threading import Timer
    import socket
    
    def find_free_port():
        """Find a free port on the system."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    def open_browser(port):
        """Open the browser after the server starts."""
        webbrowser.open_new(f'http://127.0.0.1:{port}/')
    
    try:
        # Find an available port
        port = find_free_port()
        
        # Open browser after 1.5 seconds
        Timer(1.5, open_browser, args=(port,)).start()
        
        # Start the server
        app.run_server(
            debug=False,
            port=port,
            host='127.0.0.1',
            dev_tools_hot_reload=False
        )
    except Exception as e:
        import tkinter as tk
        from tkinter import messagebox
        
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "Error",
            f"An error occurred while starting the application:\n{str(e)}\n\nPlease check the logs for more details."
        )
        sys.exit(1)
