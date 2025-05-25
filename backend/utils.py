"""
Utility functions for the CodeChat backend
"""

import re
from typing import List, Dict, Any

# Import language extensions from app.py
SUPPORTED_EXTENSIONS = {
    'python': ['.py'],
    'javascript': ['.js', '.jsx', '.ts', '.tsx'],
    'generic': ['.md', '.txt', '.json', '.yaml', '.yml', '.html', '.css', '.scss', '.less']
}

def extract_imports(content: str, file_ext: str) -> List[str]:
    """
    Extract import statements based on file type
    
    Args:
        content: The file content
        file_ext: The file extension
        
    Returns:
        A list of imported modules
    """
    imports = []
    
    if file_ext in SUPPORTED_EXTENSIONS['python']:
        # Python imports
        import_pattern = r'^(?:from\s+(\S+)\s+import\s+|import\s+(\S+))'
        for line in content.split('\n'):
            match = re.match(import_pattern, line)
            if match:
                imp = match.group(1) or match.group(2)
                imports.append(imp)
                
    elif file_ext in SUPPORTED_EXTENSIONS['javascript']:
        # JavaScript imports
        import_pattern = r'(?:import\s+(?:{[^}]*}|\S+)\s+from\s+["\']([^"\']+)["\']|require\(["\']([^"\']+)["\']\))'
        for match in re.finditer(import_pattern, content):
            imp = match.group(1) or match.group(2)
            imports.append(imp)
            
    return imports

def should_exclude_path(path: str) -> bool:
    """
    Check if a path should be excluded from processing
    
    Args:
        path: The path to check
        
    Returns:
        True if the path should be excluded, False otherwise
    """
    # Common directories to exclude
    exclude_dirs = [
        'node_modules', '__pycache__', '.git', '.github', '.vscode', 
        'venv', 'env', '.env', 'dist', 'build', 'target', 'out',
        '.idea', '.cache', '.pytest_cache', '.next', '.nuxt'
    ]
    
    # Common file patterns to exclude
    exclude_patterns = [
        '.DS_Store', '*.pyc', '*.pyo', '*.pyd', '*.so', '*.dll', '*.class',
        '*.log', '*.sqlite', '*.sqlite3', '*.db', '*.min.js', '*.min.css'
    ]
    
    # Check if path contains any excluded directory
    for exclude_dir in exclude_dirs:
        if exclude_dir in path.split('/'):
            return True
    
    # Check if path matches any excluded pattern
    for pattern in exclude_patterns:
        if '*' in pattern:
            # Convert glob pattern to regex
            regex_pattern = pattern.replace('.', '\\.').replace('*', '.*')
            if re.search(regex_pattern, path):
                return True
        elif pattern in path:
            return True
            
    # Check for hidden files/directories (starting with .)
    path_parts = path.split('/')
    for part in path_parts:
        if part.startswith('.') and part not in ['.', '..']:
            return True
            
    return False

def standardize_error_handling(func):
    """
    Decorator to standardize error handling across functions
    
    Args:
        func: The function to decorate
        
    Returns:
        The decorated function with standardized error handling
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            import traceback
            import logging
            
            # Log the error
            logging.error(f"Error in {func.__name__}: {str(e)}")
            logging.error(traceback.format_exc())
            
            # Return a standardized error response if the function returns a value
            if func.__annotations__.get('return'):
                return {
                    'error': str(e),
                    'function': func.__name__,
                    'traceback': traceback.format_exc()
                }
            
            # Re-raise the exception if the function doesn't return a value
            raise
            
    return wrapper