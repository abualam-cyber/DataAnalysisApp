import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from typing import Optional, Dict, Any, List
import os
import jinja2
import pdfkit
import logging
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

class ReportGenerator:
    def __init__(self):
        self.data = None
        self.sections = []
        self.logger = logging.getLogger(__name__)
        self.template_dir = os.path.join(os.path.dirname(__file__), 'templates')
        self._setup_templates()

    def _setup_templates(self):
        """Setup Jinja2 templates."""
        if not os.path.exists(self.template_dir):
            os.makedirs(self.template_dir)
            
        # Create base template if it doesn't exist
        base_template_path = os.path.join(self.template_dir, 'base.html')
        if not os.path.exists(base_template_path):
            with open(base_template_path, 'w') as f:
                f.write("""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>{{ title }}</title>
                    <style>
                        body { font-family: Arial, sans-serif; margin: 40px; }
                        .section { margin-bottom: 30px; }
                        .visualization { margin: 20px 0; }
                        table { border-collapse: collapse; width: 100%; }
                        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                        th { background-color: #f2f2f2; }
                        .metrics { margin: 20px 0; }
                        .metric-item { margin: 5px 0; }
                    </style>
                </head>
                <body>
                    <h1>{{ title }}</h1>
                    <div class="timestamp">Generated on: {{ timestamp }}</div>
                    {% for section in sections %}
                        <div class="section">
                            <h2>{{ section.title }}</h2>
                            {{ section.content | safe }}
                        </div>
                    {% endfor %}
                </body>
                </html>
                """)

    def set_data(self, data: pd.DataFrame):
        """Set the data for report generation."""
        self.data = data

    def clear(self):
        """Clear all sections."""
        self.sections = []

    def add_raw_data_preview(self, rows: int = 5):
        """Add raw data preview to the report."""
        if self.data is None:
            return

        preview = self.data.head(rows).to_html()
        self.sections.append({
            'title': 'Raw Data Preview',
            'content': f"""
                <p>First {rows} rows of the dataset:</p>
                {preview}
                <p>Total rows: {len(self.data)}</p>
                <p>Total columns: {len(self.data.columns)}</p>
            """
        })

    def add_summary_statistics(self):
        """Add summary statistics to the report."""
        if self.data is None:
            return

        summary = self.data.describe().to_html()
        
        # Create a cleaner data types display
        dtypes_df = pd.DataFrame(index=self.data.columns)
        dtypes_df.index.name = 'Column'
        dtypes_df = dtypes_df.reset_index()
        
        self.sections.append({
            'title': 'Summary Statistics',
            'content': f"""
                <h3>Numerical Statistics</h3>
                {summary}
                <h3>Data Types</h3>
                <table>
                    <tr><th>Column</th></tr>
                    {''.join(f'<tr><td>{col}</td></tr>' for col in self.data.columns)}
                </table>
            """
        })

    def add_data_quality_analysis(self):
        """Add data quality analysis to the report."""
        if self.data is None:
            return

        # Calculate missing values
        missing = self.data.isnull().sum()
        missing_pct = (missing / len(self.data) * 100).round(2)
        missing_df = pd.DataFrame({
            'Missing Values': missing,
            'Percentage': missing_pct
        }).to_html()

        # Calculate duplicates
        duplicates = len(self.data) - len(self.data.drop_duplicates())
        
        # Create data quality metrics
        content = f"""
            <h3>Missing Values Analysis</h3>
            {missing_df}
            <h3>Duplicate Rows</h3>
            <p>Number of duplicate rows: {duplicates}</p>
        """

        self.sections.append({
            'title': 'Data Quality Analysis',
            'content': content
        })

    def add_correlation_analysis(self):
        """Add correlation analysis to the report."""
        if self.data is None:
            return

        # Calculate correlation matrix for numerical columns
        numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
        if len(numerical_cols) > 1:
            correlation = self.data[numerical_cols].corr()
            
            # Create heatmap
            plt.figure(figsize=(10, 8))
            sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
            plt.title('Correlation Matrix')
            
            # Save plot to base64
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            img_str = base64.b64encode(buf.read()).decode()
            plt.close()

            content = f"""
                <h3>Correlation Matrix</h3>
                <img src="data:image/png;base64,{img_str}" alt="Correlation Matrix">
            """

            self.sections.append({
                'title': 'Correlation Analysis',
                'content': content
            })

    def _figure_to_html(self, fig, title: str = "") -> str:
        """Convert a plotly figure to HTML with base64 encoded image."""
        try:
            # Save figure as PNG image
            img_bytes = fig.to_image(format="png", width=1200, height=800)
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            
            return f"""
                <div class="visualization">
                    <h3>{title}</h3>
                    <img src="data:image/png;base64,{img_base64}" 
                         style="width: 100%; max-width: 1200px;">
                </div>
            """
        except Exception as e:
            self.logger.error(f"Error converting figure to HTML: {str(e)}")
            return f"<p>Error displaying visualization: {str(e)}</p>"

    def add_visualization(self, fig: go.Figure, title: str = "Visualization"):
        """Add a visualization to the report."""
        if fig is None:
            return
            
        try:
            viz_html = self._figure_to_html(fig, title)
            self.sections.append({
                'title': 'Visualization',
                'content': viz_html
            })
        except Exception as e:
            self.logger.error(f"Error adding visualization: {str(e)}")
            self.sections.append({
                'title': 'Visualization',
                'content': f"<p>Error adding visualization: {str(e)}</p>"
            })

    def add_forecast(self, fig: go.Figure, title: str = "Forecast", metrics: Optional[Dict] = None):
        """Add a forecast visualization and metrics to the report."""
        if fig is None:
            return
            
        try:
            # Convert figure to HTML
            viz_html = self._figure_to_html(fig, title)
            
            # Add metrics if available
            metrics_html = ""
            if metrics:
                metrics_html = """
                    <div class="metrics">
                        <h4>Forecast Metrics</h4>
                        <table>
                            <tr><th>Metric</th><th>Value</th></tr>
                """
                for metric, value in metrics.items():
                    metrics_html += f"<tr><td>{metric}</td><td>{value:.3f}</td></tr>"
                metrics_html += "</table></div>"
            
            self.sections.append({
                'title': 'Forecast Results',
                'content': f"{viz_html}{metrics_html}"
            })
        except Exception as e:
            self.logger.error(f"Error adding forecast: {str(e)}")
            self.sections.append({
                'title': 'Forecast Results',
                'content': f"<p>Error adding forecast: {str(e)}</p>"
            })

    def add_feature_importance(self):
        """Add feature importance analysis to the report."""
        if self.data is None:
            return

        try:
            # Calculate basic feature importance using correlation with target
            numerical_cols = self.data.select_dtypes(include=['int64', 'float64']).columns
            if len(numerical_cols) > 1:
                correlations = self.data[numerical_cols].corr().iloc[0].abs().sort_values(ascending=False)
                
                # Create bar plot
                fig = px.bar(
                    x=correlations.index,
                    y=correlations.values,
                    title='Feature Importance (Based on Correlation)',
                    labels={'x': 'Features', 'y': 'Absolute Correlation'}
                )
                
                plot_html = fig.to_html(full_html=False)
                
                self.sections.append({
                    'title': 'Feature Importance Analysis',
                    'content': f"""
                        <div class="visualization">
                            {plot_html}
                        </div>
                    """
                })
        except Exception as e:
            self.logger.error(f"Error in feature importance analysis: {str(e)}")

    def generate_report(self, title: str, output_path: str, format: str = 'html') -> bool:
        """Generate the report in the specified format."""
        try:
            if not self.sections:
                raise ValueError("No content to generate report")

            # Setup Jinja environment
            env = jinja2.Environment(
                loader=jinja2.FileSystemLoader(self.template_dir)
            )
            template = env.get_template('base.html')

            # Render HTML
            html_content = template.render(
                title=title,
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                sections=self.sections
            )

            # Save as HTML first
            html_path = output_path.replace('.pdf', '.html') if format == 'pdf' else output_path
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

            # Convert to PDF if requested
            if format == 'pdf':
                try:
                    options = {
                        'page-size': 'A4',
                        'margin-top': '0.75in',
                        'margin-right': '0.75in',
                        'margin-bottom': '0.75in',
                        'margin-left': '0.75in',
                        'encoding': "UTF-8",
                        'no-outline': None
                    }
                    
                    # Use wkhtmltopdf path for Windows
                    config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')
                    pdfkit.from_file(html_path, output_path, options=options, configuration=config)
                    
                    # Clean up HTML file
                    os.remove(html_path)
                except Exception as e:
                    self.logger.error(f"PDF conversion failed: {str(e)}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            return False
