#!/usr/bin/env python3
"""
Script to fix CodeGraph structure issues.
"""

import os
import json
import sys
from typing import Dict, List, Any

def fix_code_graph(graph_path: str) -> None:
    """Fix the structure of a CodeGraph file."""
    print(f"Fixing CodeGraph at {graph_path}")
    
    try:
        # Load the graph
        with open(graph_path, 'r') as f:
            graph_data = json.load(f)
        
        # Check if it's already in the right format
        if isinstance(graph_data, dict) and 'nodes' in graph_data and 'edges' in graph_data:
            print("Graph already has the correct structure")
            
            # Ensure edges is a dictionary
            if not isinstance(graph_data['edges'], dict):
                print("Converting edges to dictionary format")
                edges_dict = {}
                
                # Convert list of edges to dictionary
                for edge in graph_data['edges']:
                    if 'source' in edge and 'target' in edge:
                        source = edge['source']
                        if source not in edges_dict:
                            edges_dict[source] = []
                        edges_dict[source].append({
                            'to': edge['target'],
                            'type': edge.get('type', 'unknown')
                        })
                
                graph_data['edges'] = edges_dict
                print(f"Converted {len(graph_data['edges'])} edge sources")
            
            # Ensure nodes are properly formatted
            for i, node in enumerate(graph_data['nodes']):
                if not isinstance(node, dict):
                    graph_data['nodes'][i] = {'id': str(i), 'content': str(node)}
            
            # Save the fixed graph
            with open(graph_path, 'w') as f:
                json.dump(graph_data, f)
            
            print("Graph fixed and saved successfully")
            return
        
        # If it's not in the right format, convert it
        print("Converting graph to proper format")
        
        # Initialize new graph structure
        new_graph = {
            'nodes': [],
            'edges': {},
            'file_contents': {}
        }
        
        # Handle different possible formats
        if isinstance(graph_data, dict):
            if 'nodes' in graph_data:
                new_graph['nodes'] = graph_data['nodes']
            
            if 'file_contents' in graph_data:
                new_graph['file_contents'] = graph_data['file_contents']
            
            # Convert edges to dictionary format
            if 'edges' in graph_data and isinstance(graph_data['edges'], list):
                edges_dict = {}
                for edge in graph_data['edges']:
                    if 'source' in edge and 'target' in edge:
                        source = edge['source']
                        if source not in edges_dict:
                            edges_dict[source] = []
                        edges_dict[source].append({
                            'to': edge['target'],
                            'type': edge.get('type', 'unknown')
                        })
                new_graph['edges'] = edges_dict
        
        # Save the fixed graph
        with open(graph_path, 'w') as f:
            json.dump(new_graph, f)
        
        print("Graph fixed and saved successfully")
        
    except Exception as e:
        print(f"Error fixing graph: {str(e)}")

def main():
    """Main function to fix CodeGraph files."""
    if len(sys.argv) < 2:
        print("Usage: python fix_code_graph.py <graph_path>")
        return
    
    graph_path = sys.argv[1]
    if not os.path.exists(graph_path):
        print(f"Error: Graph file {graph_path} does not exist")
        return
    
    fix_code_graph(graph_path)

if __name__ == "__main__":
    main()