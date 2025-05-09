import ast
import re
from typing import Dict, List, Set, Any, Optional
from pathlib import Path
import json
import os

class CodeGraph:
    def __init__(self):
        self.nodes = []  # Files, functions, classes (list of dicts)
        self.edges = {}  # Dependencies, calls, imports (dict of lists)
        self.file_contents = {}  # Store file contents for reference
        
    def add_node(self, node_id: str, node_type: str, metadata: Dict):
        """Add a node to the graph with its metadata"""
        # Add node as a dictionary in the list
        self.nodes.append({
            'id': node_id,
            'type': node_type,
            'metadata': metadata
        })
        
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
        
        # Create a node lookup dictionary for faster access
        node_lookup = {}
        for node in self.nodes:
            if isinstance(node, dict) and 'id' in node:
                node_lookup[node['id']] = node
        
        def traverse(current_id: str, depth: int):
            if depth > max_depth or current_id in visited:
                return
                
            visited.add(current_id)
            if current_id in self.edges:
                for edge in self.edges[current_id]:
                    to_id = edge.get('to', '')
                    if to_id and to_id in node_lookup:
                        related.append({
                            'node': node_lookup[to_id],
                            'relationship': edge.get('type', 'unknown')
                        })
                        traverse(to_id, depth + 1)
        
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
            
            # Handle nodes - ensure it's a list
            if 'nodes' in data:
                if isinstance(data['nodes'], dict):
                    # Convert dict to list format
                    nodes_list = []
                    for node_id, node_data in data['nodes'].items():
                        node_dict = {'id': node_id}
                        node_dict.update(node_data)
                        nodes_list.append(node_dict)
                    graph.nodes = nodes_list
                else:
                    graph.nodes = data['nodes']
            
            # Handle edges - ensure it's a dict
            if 'edges' in data:
                if isinstance(data['edges'], list):
                    # Convert list to dict format
                    edges_dict = {}
                    for edge in data['edges']:
                        if 'source' in edge and 'target' in edge:
                            source = edge['source']
                            if source not in edges_dict:
                                edges_dict[source] = []
                            edges_dict[source].append({
                                'to': edge['target'],
                                'type': edge.get('type', 'unknown')
                            })
                    graph.edges = edges_dict
                else:
                    graph.edges = data['edges']
            
            # Load file contents if available
            graph.file_contents = data.get('file_contents', {})
            
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
    """Analyze a JavaScript/TypeScript file and extract its structure"""
    structure = {
        'imports': [],
        'functions': {},
        'classes': {},
        'dependencies': set(),
        'usages': {}
    }
    
    # Extract imports - enhanced pattern to catch more import variations
    import_patterns = [
        # ES6 imports
        r'import\s+(?:{[^}]*}|\*\s+as\s+\w+|\w+)\s+from\s+["\']([^"\']+)["\']',
        # CommonJS require
        r'(?:const|let|var)\s+(?:{[^}]*}|\w+)\s*=\s*require\(["\']([^"\']+)["\']\)',
        # Direct require
        r'require\(["\']([^"\']+)["\']\)',
        # Dynamic import
        r'import\(["\']([^"\']+)["\']\)',
        # TypeScript imports
        r'import\s+type\s+(?:{[^}]*}|\w+)\s+from\s+["\']([^"\']+)["\']'
    ]
    
    for pattern in import_patterns:
        for match in re.finditer(pattern, content):
            imp = match.group(1)
            if imp:
                structure['imports'].append(imp)
                structure['dependencies'].add(imp)
    
    # Extract functions - enhanced patterns
    function_patterns = [
        # Named function declarations
        r'function\s+(\w+)\s*\([^)]*\)\s*{',
        # Arrow functions assigned to variables
        r'(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*=>',
        # Object method definitions
        r'(\w+)\s*:\s*function\s*\([^)]*\)',
        # ES6 method definitions
        r'(\w+)\s*\([^)]*\)\s*{',
        # TypeScript function with type annotations
        r'function\s+(\w+)\s*<[^>]*>\s*\([^)]*\)',
        # Async functions
        r'async\s+function\s+(\w+)\s*\([^)]*\)'
    ]
    
    for pattern in function_patterns:
        for match in re.finditer(pattern, content):
            func_name = match.group(1)
            if func_name and not func_name in ['if', 'for', 'while', 'switch', 'catch']:
                # Skip if the match is actually a control structure
                structure['functions'][func_name] = {
                    'calls': extract_js_function_calls(content, func_name)
                }
    
    # Extract classes - enhanced patterns
    class_patterns = [
        # Standard class declaration
        r'class\s+(\w+)(?:\s+extends\s+\w+)?\s*{',
        # TypeScript interface
        r'interface\s+(\w+)(?:\s+extends\s+\w+)?\s*{',
        # TypeScript type
        r'type\s+(\w+)\s*=\s*{',
        # React component as class
        r'class\s+(\w+)\s+extends\s+(?:React\.)?Component'
    ]
    
    for pattern in class_patterns:
        for match in re.finditer(pattern, content):
            class_name = match.group(1)
            if class_name:
                methods = extract_js_class_methods(content, class_name)
                structure['classes'][class_name] = {
                    'methods': methods
                }
    
    # Extract React functional components
    react_component_pattern = r'(?:const|function)\s+(\w+)\s*=?\s*(?:\([^)]*\)|props)\s*=>\s*{'
    for match in re.finditer(react_component_pattern, content):
        component_name = match.group(1)
        if component_name and component_name[0].toUpperCase() == component_name[0]:  # Check if starts with uppercase
            structure['functions'][component_name] = {
                'is_component': True,
                'calls': extract_js_function_calls(content, component_name)
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
    
    # Find the function definition
    function_patterns = [
        # Named function declarations
        rf'function\s+{re.escape(func_name)}\s*\([^)]*\)\s*{{',
        # Arrow functions assigned to variables
        rf'(?:const|let|var)\s+{re.escape(func_name)}\s*=\s*(?:async\s*)?\([^)]*\)\s*=>',
        # Object method definitions
        rf'{re.escape(func_name)}\s*:\s*function\s*\([^)]*\)',
        # ES6 method definitions
        rf'{re.escape(func_name)}\s*\([^)]*\)\s*{{',
        # Async functions
        rf'async\s+function\s+{re.escape(func_name)}\s*\([^)]*\)'
    ]
    
    function_start = -1
    function_end = -1
    
    for pattern in function_patterns:
        match = re.search(pattern, content)
        if match:
            function_start = match.start()
            # Find the closing brace
            brace_count = 0
            in_string = False
            string_char = None
            
            for i in range(function_start, len(content)):
                char = content[i]
                
                # Handle strings to avoid counting braces inside strings
                if char in ['"', "'"]:
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                
                # Skip if we're inside a string
                if in_string:
                    continue
                
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        function_end = i + 1
                        break
            
            if function_end > function_start:
                break
    
    # If we found the function, extract calls from its body
    if function_start >= 0 and function_end > function_start:
        function_body = content[function_start:function_end]
        
        # More comprehensive pattern for function calls
        call_patterns = [
            r'(\w+)\s*\([^)]*\)',  # Basic function call
            r'(\w+)\s*`[^`]*`',    # Template literal call
            r'new\s+(\w+)\s*\(',   # Constructor call
            r'(\w+)\s*\.\s*(\w+)\s*\(',  # Method call
            r'(\w+)\s*\[\s*[^\]]+\s*\]\s*\('  # Computed property call
        ]
        
        for pattern in call_patterns:
            for match in re.finditer(pattern, function_body):
                call_name = match.group(1)
                # Filter out common keywords that aren't function calls
                if call_name not in ['if', 'for', 'while', 'switch', 'catch', 'function', 'return']:
                    calls.append(call_name)
    else:
        # Fallback to simple pattern if function body not found
        call_pattern = r'(\w+)\([^)]*\)'
        for match in re.finditer(call_pattern, content):
            call_name = match.group(1)
            if call_name not in ['if', 'for', 'while', 'switch', 'catch']:
                calls.append(call_name)
    
    return list(set(calls))  # Remove duplicates

def extract_js_class_methods(content: str, class_name: str) -> Dict:
    """Extract methods from a JavaScript class"""
    methods = {}
    
    # Find the class definition
    class_pattern = rf'class\s+{re.escape(class_name)}(?:\s+extends\s+\w+)?\s*{{'
    class_match = re.search(class_pattern, content)
    
    if class_match:
        class_start = class_match.start()
        # Find the closing brace of the class
        brace_count = 0
        in_string = False
        string_char = None
        class_end = -1
        
        for i in range(class_start, len(content)):
            char = content[i]
            
            # Handle strings to avoid counting braces inside strings
            if char in ['"', "'"]:
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
            
            # Skip if we're inside a string
            if in_string:
                continue
            
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    class_end = i + 1
                    break
        
        if class_end > class_start:
            class_body = content[class_start:class_end]
            
            # Enhanced method patterns
            method_patterns = [
                r'(\w+)\s*\([^)]*\)\s*{',  # Regular method
                r'async\s+(\w+)\s*\([^)]*\)\s*{',  # Async method
                r'get\s+(\w+)\s*\(\)\s*{',  # Getter
                r'set\s+(\w+)\s*\([^)]*\)\s*{',  # Setter
                r'static\s+(\w+)\s*\([^)]*\)\s*{',  # Static method
                r'(\w+)\s*=\s*\([^)]*\)\s*=>',  # Arrow function property
                r'(\w+)\s*=\s*function\s*\([^)]*\)'  # Function property
            ]
            
            for pattern in method_patterns:
                for match in re.finditer(pattern, class_body):
                    method_name = match.group(1)
                    if method_name and method_name not in ['constructor']:
                        methods[method_name] = {
                            'calls': extract_js_function_calls(class_body, method_name)
                        }
            
            # Always add constructor if present
            constructor_match = re.search(r'constructor\s*\([^)]*\)\s*{', class_body)
            if constructor_match:
                methods['constructor'] = {
                    'calls': extract_js_function_calls(class_body, 'constructor')
                }
    else:
        # Fallback for interfaces or types
        interface_pattern = rf'(?:interface|type)\s+{re.escape(class_name)}(?:\s+extends\s+\w+)?\s*{{'
        interface_match = re.search(interface_pattern, content)
        
        if interface_match:
            interface_start = interface_match.start()
            # Find the closing brace
            brace_count = 0
            interface_end = -1
            
            for i in range(interface_start, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        interface_end = i + 1
                        break
            
            if interface_end > interface_start:
                interface_body = content[interface_start:interface_end]
                
                # Extract method signatures from interface
                method_sig_pattern = r'(\w+)\s*(?:\([^)]*\))?\s*:'
                for match in re.finditer(method_sig_pattern, interface_body):
                    method_name = match.group(1)
                    if method_name:
                        methods[method_name] = {}
    
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