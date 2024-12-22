import os
import sys
import subprocess
import shutil
from pathlib import Path

def install_dependencies():
    """Install required packages."""
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

def create_directories():
    """Create necessary directories."""
    Path("assets").mkdir(exist_ok=True)
    Path("templates").mkdir(exist_ok=True)

def clean_build():
    """Clean previous builds."""
    for path in ["build", "dist"]:
        if os.path.exists(path):
            shutil.rmtree(path)

def build_app():
    """Build the application."""
    subprocess.check_call([
        sys.executable,
        "-m",
        "PyInstaller",
        "--name=DataAnalysisApp",
        "--onefile",
        "--windowed",
        "--add-data=templates;templates",
        "--add-data=assets;assets",
        "--hidden-import=dash",
        "--hidden-import=dash_bootstrap_components",
        "--hidden-import=plotly",
        "--hidden-import=pandas",
        "--hidden-import=numpy",
        "--hidden-import=statsmodels",
        "--hidden-import=scikit-learn",
        "--hidden-import=jinja2",
        "--hidden-import=pdfkit",
        "--hidden-import=kaleido",
        "main.py"
    ])

if __name__ == "__main__":
    try:
        print("Installing dependencies...")
        install_dependencies()
        
        print("Creating directories...")
        create_directories()
        
        print("Cleaning previous builds...")
        clean_build()
        
        print("Building application...")
        build_app()
        
        print("\nBuild completed! The executable is in the 'dist' directory.")
        
    except Exception as e:
        print(f"Error during build: {str(e)}", file=sys.stderr)
        sys.exit(1)
