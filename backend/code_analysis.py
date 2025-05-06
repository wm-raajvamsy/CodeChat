import ast
import re
from typing import Dict, List, Set, Any, Optional
from pathlib import Path
import json
import os

class CodeGraph:
    def __init__(self):
        self.nodes = {}  # Files, functions, classes
        self.edges = {}  # Dependencies, calls, imports
        self.file_contents = {}  # Store file contents for reference
        
    def add_node(self, node_id: str, node_type: str, metadata: Dict):
        """Add a node to the graph with its metadata"""
        self.nodes[node_id] = {
            'type': node_type,
            'metadata': metadata
        }
        
    def add_edge(self, from_id: str, to_id: str, edge_type: str):
        """Add an edge between nodes with a specific type"""
        if from_id not in self.edges:
            self.edges[from_id] = []
        self.edges[from_id].append({
            'to': to_id,
            'type': edge_type
        })
        
    def get_related_nodes(self, node_id: str, max_depth: int = 2) -> List[Dict]:
        """Get nodes related to a given node up to a certain depth"""
        related = []
        visited = set()
        
        def traverse(current_id: str, depth: int):
            if depth > max_depth or current_id in visited:
                return
                
            visited.add(current_id)
            if current_id in self.edges:
                for edge in self.edges[current_id]:
                    related.append({
                        'node': self.nodes[edge['to']],
                        'relationship': edge['type']
                    })
                    traverse(edge['to'], depth + 1)
        
        traverse(node_id, 0)
        return related
    
    def save(self, path: str):
        """Save the graph to disk"""
        data = {
            'nodes': self.nodes,
            'edges': self.edges,
            'file_contents': self.file_contents  # Include file contents in saved data
        }
        with open(path, 'w') as f:
            json.dump(data, f)
    
    @classmethod
    def load(cls, path: str) -> 'CodeGraph':
        """Load the graph from disk"""
        graph = cls()
        with open(path, 'r') as f:
            data = json.load(f)
            graph.nodes = data['nodes']
            graph.edges = data['edges']
            graph.file_contents = data.get('file_contents', {})  # Load file contents if available
        return graph

def analyze_python_file(file_path: str, content: str) -> Dict:
    """Analyze a Python file and extract its structure"""
    try:
        tree = ast.parse(content)
        structure = {
            'imports': [],
            'functions': {},
            'classes': {},
            'dependencies': set(),
            'usages': {}
        }
        
        # Extract imports
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    structure['imports'].append(name.name)
                    structure['dependencies'].add(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    structure['imports'].append(f"{node.module}.{node.names[0].name}")
                    structure['dependencies'].add(node.module)
        
        # Extract functions and classes
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                structure['functions'][node.name] = {
                    'lineno': node.lineno,
                    'end_lineno': node.end_lineno,
                    'calls': extract_function_calls(node)
                }
            elif isinstance(node, ast.ClassDef):
                structure['classes'][node.name] = {
                    'lineno': node.lineno,
                    'end_lineno': node.end_lineno,
                    'methods': {n.name: {'lineno': n.lineno, 'end_lineno': n.end_lineno} 
                              for n in node.body if isinstance(n, ast.FunctionDef)}
                }
        
        return structure
    except Exception as e:
        print(f"Error analyzing Python file {file_path}: {e}")
        return {
            'imports': [],
            'functions': {},
            'classes': {},
            'dependencies': set(),
            'usages': {}
        }

def analyze_js_file(file_path: str, content: str) -> Dict:
    """Analyze a JavaScript file and extract its structure"""
    structure = {
        'imports': [],
        'functions': {},
        'classes': {},
        'dependencies': set(),
        'usages': {}
    }
    
    # Extract imports
    import_pattern = r'(?:import\s+(?:{[^}]*}|\S+)\s+from\s+["\']([^"\']+)["\']|require\(["\']([^"\']+)["\']\))'
    for match in re.finditer(import_pattern, content):
        imp = match.group(1) or match.group(2)
        structure['imports'].append(imp)
        structure['dependencies'].add(imp)
    
    # Extract functions
    function_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>)'
    for match in re.finditer(function_pattern, content):
        func_name = match.group(1) or match.group(2)
        structure['functions'][func_name] = {
            'calls': extract_js_function_calls(content, func_name)
        }
    
    # Extract classes
    class_pattern = r'class\s+(\w+)'
    for match in re.finditer(class_pattern, content):
        class_name = match.group(1)
        structure['classes'][class_name] = {
            'methods': extract_js_class_methods(content, class_name)
        }
    
    return structure

def extract_function_calls(node: ast.FunctionDef) -> List[str]:
    """Extract function calls from a Python function definition"""
    calls = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                calls.append(child.func.id)
            elif isinstance(child.func, ast.Attribute):
                calls.append(child.func.attr)
    return calls

def extract_js_function_calls(content: str, func_name: str) -> List[str]:
    """Extract function calls from a JavaScript function"""
    calls = []
    # This is a simplified version - you might want to use a proper JS parser
    call_pattern = r'(\w+)\([^)]*\)'
    for match in re.finditer(call_pattern, content):
        calls.append(match.group(1))
    return calls

def extract_js_class_methods(content: str, class_name: str) -> Dict:
    """Extract methods from a JavaScript class"""
    methods = {}
    # This is a simplified version - you might want to use a proper JS parser
    method_pattern = r'(\w+)\s*\([^)]*\)\s*{'
    for match in re.finditer(method_pattern, content):
        methods[match.group(1)] = {}
    return methods

def build_code_graph(directory: str) -> CodeGraph:
    """Build a code graph from a directory"""
    graph = CodeGraph()
    
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(('.py', '.js', '.jsx', '.ts', '.tsx')):
                file_path = os.path.join(root, file)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    graph.file_contents[file_path] = content
                    
                    # Add file node
                    graph.add_node(file_path, 'file', {
                        'name': file,
                        'path': file_path
                    })
                    
                    # Analyze file content
                    if file.endswith('.py'):
                        structure = analyze_python_file(file_path, content)
                    else:
                        structure = analyze_js_file(file_path, content)
                    
                    # Add function nodes and edges
                    for func_name, func_data in structure['functions'].items():
                        func_id = f"{file_path}::{func_name}"
                        graph.add_node(func_id, 'function', {
                            'name': func_name,
                            'file': file_path,
                            'lineno': func_data.get('lineno'),
                            'end_lineno': func_data.get('end_lineno')
                        })
                        graph.add_edge(file_path, func_id, 'contains')
                        
                        # Add call edges
                        for call in func_data.get('calls', []):
                            graph.add_edge(func_id, call, 'calls')
                    
                    # Add class nodes and edges
                    for class_name, class_data in structure['classes'].items():
                        class_id = f"{file_path}::{class_name}"
                        graph.add_node(class_id, 'class', {
                            'name': class_name,
                            'file': file_path,
                            'lineno': class_data.get('lineno'),
                            'end_lineno': class_data.get('end_lineno')
                        })
                        graph.add_edge(file_path, class_id, 'contains')
                        
                        # Add method edges
                        for method_name, method_data in class_data.get('methods', {}).items():
                            method_id = f"{class_id}::{method_name}"
                            graph.add_node(method_id, 'method', {
                                'name': method_name,
                                'class': class_name,
                                'file': file_path,
                                'lineno': method_data.get('lineno'),
                                'end_lineno': method_data.get('end_lineno')
                            })
                            graph.add_edge(class_id, method_id, 'contains')
                    
                    # Add import edges
                    for imp in structure['imports']:
                        graph.add_edge(file_path, imp, 'imports')
    
    return graph 