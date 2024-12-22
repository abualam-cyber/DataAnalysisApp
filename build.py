import os
import sys
import subprocess
import shutil

def install_dependencies():
    """Install required packages."""
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

def build_application():
    """Build the standalone application."""
    print("Building application...")
    
    # Clean previous builds
    if os.path.exists("build"):
        shutil.rmtree("build")
    if os.path.exists("dist"):
        shutil.rmtree("dist")
    
    # Create assets directory if it doesn't exist
    if not os.path.exists("assets"):
        os.makedirs("assets")
    
    # Create templates directory if it doesn't exist
    if not os.path.exists("templates"):
        os.makedirs("templates")
    
    # Run PyInstaller
    subprocess.check_call([
        "pyinstaller",
        "--clean",
        "--noconfirm",
        "data_analysis_app.spec"
    ])
    
    print("\nBuild completed! You can find the executable in the 'dist' directory.")

if __name__ == "__main__":
    try:
        install_dependencies()
        build_application()
    except Exception as e:
        print(f"Error during build: {str(e)}")
        sys.exit(1)
