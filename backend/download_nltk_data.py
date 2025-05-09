#!/usr/bin/env python3
"""
Script to download required NLTK data.
"""

import nltk
import sys

def main():
    """Download required NLTK data."""
    print("Downloading NLTK data...")
    
    # Download punkt tokenizer
    try:
        nltk.download('punkt')
        print("Successfully downloaded 'punkt' tokenizer")
    except Exception as e:
        print(f"Error downloading 'punkt': {e}")
    
    # Instead of creating punkt_tab, let's modify the code to not use it
    print("Note: We'll update the code to avoid using punkt_tab")
        
    # Download stopwords
    try:
        nltk.download('stopwords')
        print("Successfully downloaded 'stopwords'")
    except Exception as e:
        print(f"Error downloading 'stopwords': {e}")
    
    print("\nNLTK data download complete!")

if __name__ == "__main__":
    main()