from setuptools import setup, find_packages

setup(
    name="data_analysis_app",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        'dash',
        'dash-bootstrap-components',
        'plotly',
        'pandas',
        'numpy',
        'statsmodels',
        'scikit-learn',
        'jinja2',
        'pdfkit',
        'kaleido'
    ],
    entry_points={
        'console_scripts': [
            'data_analysis_app=main:app.run_server'
        ]
    }
)
