#!/usr/bin/env python3
"""
Script to fix all CodeGraph files in the knowledge bases.
"""

import os
import json
import sys
from typing import Dict, List, Any
import glob

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
            if isinstance(graph_data['nodes'], dict):
                print("Converting nodes from dict to list format")
                nodes_list = []
                for node_id, node_data in graph_data['nodes'].items():
                    node_dict = {'id': node_id}
                    if isinstance(node_data, dict):
                        node_dict.update(node_data)
                    nodes_list.append(node_dict)
                graph_data['nodes'] = nodes_list
                print(f"Converted {len(nodes_list)} nodes")
            elif isinstance(graph_data['nodes'], list):
                # Ensure each node has an id
                for i, node in enumerate(graph_data['nodes']):
                    if not isinstance(node, dict):
                        graph_data['nodes'][i] = {'id': str(i), 'content': str(node)}
                    elif 'id' not in node:
                        node['id'] = str(i)
            
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
    """Main function to fix all CodeGraph files."""
    # Get the data directory
    data_dir = os.path.join(os.getcwd(), 'data')
    if not os.path.exists(data_dir):
        print(f"Error: Data directory {data_dir} does not exist")
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

if __name__ == "__main__":
    main()