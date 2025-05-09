#!/usr/bin/env python3
"""
Setup script to install missing dependencies for the CodeChat backend.
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a Python package using pip."""
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print(f"Successfully installed {package}")

def download_nltk_data():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    print("Successfully downloaded NLTK data")

def main():
    """Main function to install all dependencies."""
    # Check if redis is installed
    try:
        import redis
        print("Redis package is already installed")
    except ImportError:
        install_package("redis>=4.3.4")
    
    # Check if nltk is installed
    try:
        import nltk
        print("NLTK package is already installed")
        # Download NLTK data
        download_nltk_data()
    except ImportError:
        install_package("nltk")
        download_nltk_data()
    
    # Check if psutil is installed
    try:
        import psutil
        print("psutil package is already installed")
    except ImportError:
        install_package("psutil>=5.9.0")
    
    print("\nAll dependencies have been installed successfully!")
    print("You can now run the backend server.")

if __name__ == "__main__":
    main()