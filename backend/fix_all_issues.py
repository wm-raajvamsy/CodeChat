#!/usr/bin/env python3
"""
Comprehensive script to fix all issues with the CodeChat backend.
"""

import os
import sys
import subprocess
import json
import glob
from pathlib import Path

def install_dependencies():
    """Install required dependencies."""
    print("Installing required dependencies...")
    
    # Install Redis
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "redis>=4.3.4"])
        print("Successfully installed Redis")
    except Exception as e:
        print(f"Error installing Redis: {e}")
    
    # Install NLTK
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "nltk"])
        print("Successfully installed NLTK")
    except Exception as e:
        print(f"Error installing NLTK: {e}")
    
    # Download NLTK data
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        print("Successfully downloaded NLTK data")
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")

def fix_code_graph(graph_path: str) -> None:
    """Fix the structure of a CodeGraph file."""
    print(f"Fixing CodeGraph at {graph_path}")
    
    try:
        # Load the graph
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)
        
        # Initialize new graph structure
        new_graph = {
            'nodes': [],
            'edges': {},
            'file_contents': graph_data.get('file_contents', {})
        }
        
        # Handle nodes
        if 'nodes' in graph_data:
            if isinstance(graph_data['nodes'], dict):
                # Convert dict to list format
                for node_id, node_data in graph_data['nodes'].items():
                    node_dict = {'id': node_id}
                    if isinstance(node_data, dict):
                        node_dict.update(node_data)
                    new_graph['nodes'].append(node_dict)
            elif isinstance(graph_data['nodes'], list):
                # Ensure each node has an id
                for i, node in enumerate(graph_data['nodes']):
                    if not isinstance(node, dict):
                        new_graph['nodes'].append({'id': str(i), 'content': str(node)})
                    elif 'id' not in node:
                        node_copy = node.copy()
                        node_copy['id'] = str(i)
                        new_graph['nodes'].append(node_copy)
                    else:
                        new_graph['nodes'].append(node)
        
        # Handle edges
        if 'edges' in graph_data:
            if isinstance(graph_data['edges'], list):
                # Convert list to dict format
                for edge in graph_data['edges']:
                    if 'source' in edge and 'target' in edge:
                        source = edge['source']
                        if source not in new_graph['edges']:
                            new_graph['edges'][source] = []
                        new_graph['edges'][source].append({
                            'to': edge['target'],
                            'type': edge.get('type', 'unknown')
                        })
            elif isinstance(graph_data['edges'], dict):
                new_graph['edges'] = graph_data['edges']
        
        # Save the fixed graph
        with open(graph_path, 'w') as f:
            json.dump(new_graph, f)
        
        print(f"Graph fixed and saved successfully: {len(new_graph['nodes'])} nodes, {len(new_graph['edges'])} edge sources")
        
    except Exception as e:
        print(f"Error fixing graph: {str(e)}")

def fix_all_graphs():
    """Fix all CodeGraph files in the data directory."""
    # Get the data directory
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist, creating it")
        os.makedirs(data_dir)
        return
    
    # Find all graph.json files
    graph_files = glob.glob(os.path.join(data_dir, '*', 'graph.json'))
    
    if not graph_files:
        print("No graph files found")
        return
    
    print(f"Found {len(graph_files)} graph files")
    
    # Fix each graph file
    for graph_path in graph_files:
        fix_code_graph(graph_path)

def main():
    """Main function to fix all issues."""
    print("Starting comprehensive fix for CodeChat backend...")
    
    # Install dependencies
    install_dependencies()
    
    # Fix all code graphs
    fix_all_graphs()
    
    print("\nAll fixes have been applied. The system should now work correctly.")
    print("If you still encounter issues, please restart the backend server.")

if __name__ == "__main__":
    main()