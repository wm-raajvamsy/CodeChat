#!/usr/bin/env python3
"""
Script to clear the embedding cache for a knowledge base.
"""

import os
import sys
import shutil
import glob

def clear_cache(kb_id=None):
    """Clear the embedding cache for a knowledge base or all knowledge bases."""
    data_dir = os.path.join(os.getcwd(), 'data')
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist")
        return
    
    if kb_id:
        # Clear cache for a specific knowledge base
        cache_dir = os.path.join(data_dir, kb_id, 'cache')
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
            print(f"Cache cleared for knowledge base {kb_id}")
        else:
            print(f"No cache directory found for knowledge base {kb_id}")
    else:
        # Clear cache for all knowledge bases
        cache_dirs = glob.glob(os.path.join(data_dir, '*', 'cache'))
        if not cache_dirs:
            print("No cache directories found")
            return
            
        for cache_dir in cache_dirs:
            kb_id = os.path.basename(os.path.dirname(cache_dir))
            shutil.rmtree(cache_dir)
            print(f"Cache cleared for knowledge base {kb_id}")

def main():
    """Main function to clear the embedding cache."""
    if len(sys.argv) > 1:
        kb_id = sys.argv[1]
        clear_cache(kb_id)
    else:
        # Ask for confirmation before clearing all caches
        confirm = input("Are you sure you want to clear all embedding caches? (y/n): ")
        if confirm.lower() == 'y':
            clear_cache()
        else:
            print("Operation cancelled")

if __name__ == "__main__":
    main()