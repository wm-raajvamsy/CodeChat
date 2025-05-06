from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import faiss
import numpy as np
import pickle
import traceback
import time
import json
import re
import ast
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
from InstructorEmbedding import INSTRUCTOR
from tqdm import tqdm
from code_analysis import CodeGraph, build_code_graph
from progressive_search import ProgressiveSearch
import logging
from logging.handlers import RotatingFileHandler
import threading
from threading import Lock, Timer
import signal

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Remove default Flask logger
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Custom logging filter to ignore status check logs
class StatusCheckFilter(logging.Filter):
    def filter(self, record):
        return not (
            'OPTIONS /api/knowledge-bases' in record.getMessage() or
            'GET /api/knowledge-bases' in record.getMessage() or
            'GET /api/status' in record.getMessage()
        )

# Apply filter to werkzeug logger
log.addFilter(StatusCheckFilter())

# Configuration
INDEX_PATH = 'code_index.faiss'  # File with FAISS index
CHUNKS_PATH = 'code_chunks.pkl'  # File with code chunks
METADATA_PATH = 'code_metadata.pkl'  # File with metadata
KB_PATH = 'knowledge_bases.json'  # Path to store knowledge base info
UPLOADS_DIR = 'uploads'  # Directory to store uploaded files
DATA_DIR = 'data'  # Directory to store indexed data per knowledge base
GRAPH_PATH = 'code_graph.json'  # Path to store code graph

QUERY_INSTRUCTION = "Represent the question for code retrieval:"
INSTRUCTION = "Represent the code snippet for retrieval:"  # Base embedding instruction

# Supported file extensions - add more as needed
SUPPORTED_EXTENSIONS = {
    'python': ['.py'],
    'javascript': ['.js', '.jsx', '.ts', '.tsx'],
    'web': ['.html', '.css'],
    'config': ['.json', '.yaml', '.yml', '.toml'],
    'docs': ['.md', '.txt']
}

# Files and directories to exclude
EXCLUDED_PATTERNS = [
    'node_modules', 'venv', 'env', '__pycache__', 
    'dist', 'build', '.git', '.idea', '.vscode',
    'package-lock.json', 'packageLock.json', 'yarn.lock', '*.min.js',
    '*.min.css', '*.map', '*.log', '*.bak', '*.tmp', '.yalc', 'Pods', 'lib', 
    '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.svg', '.ico', '.mov'
]

# Global variables to hold the model and data
instructor_model = None
knowledge_bases = {}  # Dictionary to hold multiple knowledge bases
code_graphs = {}  # Dictionary to hold code graphs for each knowledge base

# Dictionary to store indexing threads
indexing_threads = {}

# Add after other global variables
search_progress = {}
search_progress_lock = Lock()

# Add at the top with other imports
SEARCH_TIMEOUT = 1200  # 2 minutes timeout for search operations

def create_directory_if_not_exists(path):
    """Create directory if it doesn't exist"""
    try:
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
            # Set permissions to ensure writability
            os.chmod(path, 0o755)
    except Exception as e:
        print(f"Error creating directory {path}: {e}")
        raise

def get_kb_data_path(kb_id):
    """Get the data path for a specific knowledge base"""
    kb_dir = os.path.join(DATA_DIR, kb_id)
    try:
        create_directory_if_not_exists(kb_dir)
        # Ensure directory is writable
        test_file = os.path.join(kb_dir, '.test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        return kb_dir
    except Exception as e:
        print(f"Error creating/accessing data directory for {kb_id}: {e}")
        raise

def load_model():
    """Load the Instructor embedding model"""
    global instructor_model
    print("Loading Instructor model...")
    instructor_model = INSTRUCTOR('hkunlp/instructor-base')
    print("Model loaded successfully!")
    return instructor_model

def load_knowledge_bases():
    """Load the knowledge bases information from disk"""
    global knowledge_bases
    create_directory_if_not_exists(DATA_DIR)
    
    if os.path.exists(KB_PATH):
        try:
            with open(KB_PATH, 'r') as f:
                knowledge_bases = json.load(f)
                print(f"Loaded {len(knowledge_bases)} knowledge bases")
        except Exception as e:
            print(f"Error loading knowledge bases: {e}")
            knowledge_bases = {}
    else:
        # Create empty knowledge bases file
        with open(KB_PATH, 'w') as f:
            json.dump({}, f)
        knowledge_bases = {}
    
    return knowledge_bases

# Initialize server
def init_server():
    """Initialize server by loading model and knowledge bases"""
    # Create necessary directories
    create_directory_if_not_exists(UPLOADS_DIR)
    create_directory_if_not_exists(DATA_DIR)
    
    # Load knowledge bases
    load_knowledge_bases()
    
    # Load model
    load_model()

# Initialize on startup
init_server()

# ========== CODE CHUNK CLASS ==========
class CodeChunk:
    """Represents a meaningful chunk of code with metadata"""
    def __init__(self, 
                content: str, 
                file_path: str, 
                chunk_type: str,
                name: str = "",
                line_start: int = 0,
                line_end: int = 0,
                dependencies: List[str] = None,
                git_info: Dict = None):
        self.content = content
        self.file_path = file_path
        self.chunk_type = chunk_type  # function, class, module, etc.
        self.name = name
        self.line_start = line_start
        self.line_end = line_end
        self.dependencies = dependencies or []
        self.git_info = git_info or {}
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'content': self.content,
            'file_path': self.file_path,
            'chunk_type': self.chunk_type,
            'name': self.name,
            'line_start': self.line_start,
            'line_end': self.line_end,
            'dependencies': self.dependencies,
            'git_info': self.git_info
        }
        
    @classmethod
    def from_dict(cls, data: Dict) -> 'CodeChunk':
        """Create CodeChunk from dictionary"""
        return cls(
            content=data['content'],
            file_path=data['file_path'],
            chunk_type=data['chunk_type'],
            name=data['name'],
            line_start=data['line_start'],
            line_end=data['line_end'],
            dependencies=data['dependencies'],
            git_info=data['git_info']
        )
    
    def get_embedding_text(self) -> str:
        """Format chunk with metadata for embedding"""
        header = f"FILE: {self.file_path}\nTYPE: {self.chunk_type}\nNAME: {self.name}\n"
        if self.dependencies:
            header += f"DEPENDENCIES: {', '.join(self.dependencies)}\n"
        if self.git_info.get('last_modified'):
            header += f"LAST_MODIFIED: {self.git_info.get('last_modified')}\n"
        if self.git_info.get('commit_msg'):
            header += f"COMMIT_MSG: {self.git_info.get('commit_msg')}\n"
        header += "\nCODE:\n"
        return header + self.content

# ========== UTILITY FUNCTIONS ==========
def get_ollama_models_via_cli():
    # Run `ollama list` and capture stdout
    proc = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        check=True
    )
    # The CLI prints a table; each line has "<model-name>    <tag>"
    lines = proc.stdout.strip().splitlines()
    models = []
    for line in lines:
        parts = line.split()
        if parts:
            models.append(parts[0])
    return models

def save_knowledge_bases():
    """Save the knowledge bases information to disk"""
    try:
        with open(KB_PATH, 'w') as f:
            json.dump(knowledge_bases, f)
    except Exception as e:
        print(f"Error saving knowledge bases: {e}")
        raise

def get_git_info(file_path: str) -> Dict:
    """Get git information for a file"""
    try:
        # Check if we're in a git repository
        cmd = ["git", "rev-parse", "--is-inside-work-tree"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            return {}
        
        # Get last commit info for file
        cmd = ["git", "log", "-1", "--pretty=format:%h|%an|%ad|%s", "--", file_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout:
            commit_hash, author, date, message = result.stdout.split("|", 3)
            return {
                'commit_hash': commit_hash,
                'author': author,
                'last_modified': date,
                'commit_msg': message
            }
    except (subprocess.SubprocessError, ValueError) as e:
        print(f"Git info error for {file_path}: {e}")
    return {}

def clone_git_repository(git_url: str, kb_id: str) -> str:
    """Clone a git repository to the uploads directory"""
    import subprocess
    
    repo_dir = os.path.join(UPLOADS_DIR, kb_id)
    
    # Remove existing directory if it exists
    if os.path.exists(repo_dir):
        import shutil
        shutil.rmtree(repo_dir)
    
    create_directory_if_not_exists(repo_dir)
    
    try:
        # Check if the path is a local directory
        if os.path.isdir(git_url):
            # If it's a local directory, copy it
            subprocess.run(["cp", "-rf", git_url, repo_dir], check=True)
        else:
            # If it's a git URL, clone it
            subprocess.run(["git", "clone", git_url, repo_dir], check=True)
        return repo_dir
    except subprocess.SubprocessError as e:
        print(f"Error cloning/copying repository: {e}")
        raise

def extract_imports(content: str, file_ext: str) -> List[str]:
    """Extract import statements based on file type"""
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
    """Check if a path should be excluded based on patterns"""
    path_parts = path.split(os.sep)
    
    for pattern in EXCLUDED_PATTERNS:
        # Handle glob patterns
        if pattern.startswith('*'):
            if path.endswith(pattern[1:]):
                return True
        elif pattern.endswith('*'):
            if path.startswith(pattern[:-1]):
                return True
        else:
            # Check if pattern matches any part of the path exactly
            if any(part == pattern for part in path_parts):
                return True
            # Check if pattern is a file extension and path ends with it
            if pattern.startswith('.') and path.endswith(pattern):
                return True
    return False

# ========== CODE PARSING FUNCTIONS ==========
def parse_python_file(file_path: str, content: str) -> List[CodeChunk]:
    """Parse Python file into semantic chunks"""
    chunks = []
    git_info = get_git_info(file_path)
    imports = extract_imports(content, '.py')
    
    try:
        tree = ast.parse(content)
        
        # Module-level docstring
        if ast.get_docstring(tree):
            module_doc = ast.get_docstring(tree)
            chunks.append(CodeChunk(
                content=module_doc,
                file_path=file_path,
                chunk_type='module_docstring',
                name=os.path.basename(file_path),
                dependencies=imports,
                git_info=git_info
            ))
        
        # Module-level chunk (imports, constants, etc.)
        module_content = content.split('\n')
        
        # Extract classes and functions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                try:
                    func_lines = module_content[node.lineno-1:node.end_lineno]
                    func_content = '\n'.join(func_lines)
                    chunks.append(CodeChunk(
                        content=func_content,
                        file_path=file_path,
                        chunk_type='function',
                        name=node.name,
                        line_start=node.lineno,
                        line_end=node.end_lineno,
                        dependencies=imports,
                        git_info=git_info
                    ))
                except (AttributeError, IndexError):
                    # Fallback for older Python versions without end_lineno
                    pass
            elif isinstance(node, ast.ClassDef):
                try:
                    class_lines = module_content[node.lineno-1:node.end_lineno]
                    class_content = '\n'.join(class_lines)
                    chunks.append(CodeChunk(
                        content=class_content,
                        file_path=file_path,
                        chunk_type='class',
                        name=node.name,
                        line_start=node.lineno,
                        line_end=node.end_lineno,
                        dependencies=imports,
                        git_info=git_info
                    ))
                except (AttributeError, IndexError):
                    # Fallback for older Python versions without end_lineno
                    pass
        
        # Add a whole file chunk for context
        chunks.append(CodeChunk(
            content=content,
            file_path=file_path,
            chunk_type='file',
            name=os.path.basename(file_path),
            dependencies=imports,
            git_info=git_info
        ))
        
    except SyntaxError:
        # Fall back to basic chunking if parsing fails
        chunks.append(CodeChunk(
            content=content,
            file_path=file_path,
            chunk_type='file',
            name=os.path.basename(file_path),
            dependencies=imports,
            git_info=git_info
        ))
    
    return chunks

def parse_js_file(file_path: str, content: str) -> List[CodeChunk]:
    """Parse JavaScript/TypeScript file into semantic chunks using regex patterns"""
    chunks = []
    git_info = get_git_info(file_path)
    imports = extract_imports(content, '.js')
    
    # Extract functions - both normal and arrow functions
    function_pattern = r'(?:function\s+(\w+)|const\s+(\w+)\s*=\s*function|const\s+(\w+)\s*=\s*\([^)]*\)\s*=>)'
    function_matches = list(re.finditer(function_pattern, content))
    
    # Extract classes 
    class_pattern = r'class\s+(\w+)'
    class_matches = list(re.finditer(class_pattern, content))
    
    # Extract component definitions for React
    component_pattern = r'(?:const|let|var)\s+(\w+)\s*=\s*\((?:[^)]*)\)\s*=>\s*(?:\{|\()'
    component_matches = list(re.finditer(component_pattern, content))
    
    # Combine all matches and sort by position
    all_matches = function_matches + class_matches + component_matches
    all_matches.sort(key=lambda x: x.start())
    
    # Process each match to create chunks
    for i, match in enumerate(all_matches):
        start_pos = match.start()
        end_pos = content.find('\n}', start_pos)
        if end_pos == -1:  # If no closing bracket found, go to next match or end
            end_pos = all_matches[i+1].start() if i+1 < len(all_matches) else len(content)
        
        name = match.group(1) or match.group(2) or match.group(3) or "unnamed"
        chunk_type = 'component' if match in component_matches else 'class' if match in class_matches else 'function'
        
        # Extract code chunk
        chunk_content = content[start_pos:end_pos+2]  # +2 to include the closing bracket
        
        # Estimate line numbers
        line_start = content[:start_pos].count('\n') + 1
        line_end = line_start + chunk_content.count('\n')
        
        chunks.append(CodeChunk(
            content=chunk_content,
            file_path=file_path,
            chunk_type=chunk_type,
            name=name,
            line_start=line_start,
            line_end=line_end,
            dependencies=imports,
            git_info=git_info
        ))
    
    # Add a whole file chunk for context
    chunks.append(CodeChunk(
        content=content,
        file_path=file_path,
        chunk_type='file',
        name=os.path.basename(file_path),
        dependencies=imports,
        git_info=git_info
    ))
    
    return chunks

def parse_generic_file(file_path: str, content: str) -> List[CodeChunk]:
    """Basic parsing for other file types"""
    chunks = []
    git_info = get_git_info(file_path)
    ext = os.path.splitext(file_path)[1]
    
    # For generic files, create a whole-file chunk
    chunks.append(CodeChunk(
        content=content,
        file_path=file_path,
        chunk_type='file',
        name=os.path.basename(file_path),
        dependencies=extract_imports(content, ext),
        git_info=git_info
    ))
    
    # For large files, also create smaller chunks
    if len(content) > 1000:
        lines = content.split('\n')
        chunk_size = 50  # lines per chunk
        for i in range(0, len(lines), chunk_size):
            chunk_content = '\n'.join(lines[i:i+chunk_size])
            chunks.append(CodeChunk(
                content=chunk_content,
                file_path=file_path,
                chunk_type='file_segment',
                name=f"{os.path.basename(file_path)}:{i//chunk_size}",
                line_start=i+1,
                line_end=min(i+chunk_size, len(lines)),
                git_info=git_info
            ))
    
    return chunks

def extract_repository_structure(repo_path: str) -> Dict:
    """Extract repository structure information"""
    structure = {
        'files': [],
        'dirs': [],
        'readme': None,
        'git_info': {}
    }
    
    # Get repository-level git info
    try:
        cmd = ["git", "-C", repo_path, "log", "-1", "--pretty=format:%h|%an|%ad|%s"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        if result.stdout:
            commit_hash, author, date, message = result.stdout.split("|", 3)
            structure['git_info'] = {
                'commit_hash': commit_hash,
                'author': author,
                'last_modified': date,
                'commit_msg': message
            }
    except (subprocess.SubprocessError, ValueError):
        pass
    
    # Get repository structure
    for root, dirs, files in os.walk(repo_path):
        rel_path = os.path.relpath(root, repo_path)
        if rel_path == '.':
            rel_path = ''
        
        # Skip hidden directories and files
        if any(part.startswith('.') for part in rel_path.split(os.sep) if part):
            continue
            
        # Skip excluded directories
        if should_exclude_path(rel_path):
            # Modify dirs in-place to prevent os.walk from traversing excluded dirs
            dirs[:] = [d for d in dirs if not should_exclude_path(os.path.join(rel_path, d))]
            continue
        
        # Add directory
        if rel_path:
            structure['dirs'].append(rel_path)
        
        # Add files
        for file in files:
            if file.startswith('.'):
                continue
                
            file_path = os.path.join(rel_path, file)
            
            # Skip excluded files
            if should_exclude_path(file) or should_exclude_path(file_path):
                continue
                
            structure['files'].append(file_path)
            
            # Check for README
            if file.lower() == 'readme.md':
                try:
                    with open(os.path.join(repo_path, file_path), 'r', encoding='utf-8', errors='ignore') as f:
                        structure['readme'] = f.read()
                except Exception as e:
                    print(f"Error reading README: {e}")
    
    return structure

# ========== EMBEDDING & INDEXING FUNCTIONS ==========
def embed_codebase(repo_path: str, kb_id: str) -> Tuple[List[CodeChunk], np.ndarray]:
    """
    Walks through the repo and embeds code chunks using semantic parsing.
    Returns a list of CodeChunks and their embeddings.
    """
    if instructor_model is None:
        load_model()
    
    all_chunks = []
    supported_exts = [ext for exts in SUPPORTED_EXTENSIONS.values() for ext in exts]
    
    # Update progress - Starting repository analysis
    knowledge_bases[kb_id]['progress'] = 5
    knowledge_bases[kb_id]['current_operation'] = 'Analyzing repository structure'
    save_knowledge_bases()
    
    # Extract repository structure
    repo_structure = extract_repository_structure(repo_path)
    
    # Update progress - Repository structure analyzed
    knowledge_bases[kb_id]['progress'] = 10
    knowledge_bases[kb_id]['current_operation'] = 'Creating repository overview'
    save_knowledge_bases()
    
    # Create a repository overview chunk
    repo_name = os.path.basename(os.path.abspath(repo_path))
    overview_content = f"# Repository: {repo_name}\n\n"
    
    if repo_structure['readme']:
        overview_content += f"## README\n{repo_structure['readme']}\n\n"
    
    overview_content += "## Structure\n"
    overview_content += f"- Directories: {len(repo_structure['dirs'])}\n"
    overview_content += f"- Files: {len(repo_structure['files'])}\n"
    
    # Create an overview chunk
    overview_chunk = CodeChunk(
        content=overview_content,
        file_path="repository_overview",
        chunk_type="repository_overview",
        name=repo_name,
        git_info=repo_structure['git_info']
    )
    all_chunks.append(overview_chunk)
    
    # Update progress - Starting file processing
    knowledge_bases[kb_id]['progress'] = 15
    knowledge_bases[kb_id]['current_operation'] = 'Processing files'
    save_knowledge_bases()
    
    # Get total number of files for progress calculation (excluding excluded paths)
    total_files = 0
    for root, _, files in os.walk(repo_path):
        rel_root = os.path.relpath(root, repo_path)
        if should_exclude_path(rel_root):
            continue
        for f in files:
            if any(f.endswith(ext) for ext in supported_exts):
                rel_path = os.path.join(rel_root, f)
                if not should_exclude_path(rel_path):
                    total_files += 1
    
    processed_files = 0
    
    # Process each file
    for root, dirs, files in os.walk(repo_path):
        # Skip excluded directories
        rel_root = os.path.relpath(root, repo_path)
        if should_exclude_path(rel_root):
            dirs.clear()  # Clear dirs list to prevent walking into excluded directories
            continue
            
        # Filter out excluded directories
        dirs[:] = [d for d in dirs if not should_exclude_path(os.path.join(rel_root, d))]
            
        for fname in files:
            # Skip excluded files
            rel_path = os.path.join(rel_root, fname)
            if should_exclude_path(rel_path):
                continue
                
            if any(fname.endswith(ext) for ext in supported_exts):
                fpath = os.path.join(root, fname)
                
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                except Exception as e:
                    print(f"Error reading {fpath}: {e}")
                    continue
                
                # Parse file based on its type
                ext = os.path.splitext(fname)[1]
                file_chunks = []
                
                if ext in SUPPORTED_EXTENSIONS['python']:
                    file_chunks = parse_python_file(rel_path, content)
                elif ext in SUPPORTED_EXTENSIONS['javascript']:
                    file_chunks = parse_js_file(rel_path, content)
                else:
                    file_chunks = parse_generic_file(rel_path, content)
                
                all_chunks.extend(file_chunks)
                
                # Update progress for file processing
                processed_files += 1
                file_progress = 15 + (processed_files / total_files * 40)  # 15-55%
                knowledge_bases[kb_id]['progress'] = int(file_progress)
                knowledge_bases[kb_id]['current_operation'] = f'Processing files ({processed_files}/{total_files})'
                save_knowledge_bases()
    
    # Update progress - Starting embedding
    knowledge_bases[kb_id]['progress'] = 55
    knowledge_bases[kb_id]['current_operation'] = 'Generating embeddings'
    save_knowledge_bases()
    
    # Prepare for embedding
    embedding_texts = [chunk.get_embedding_text() for chunk in all_chunks]
    
    # Prepare instruction-text pairs
    inputs = [[INSTRUCTION, text] for text in embedding_texts]
    
    # Embed in batches to avoid memory issues
    batch_size = 32
    all_embeddings = []
    total_batches = (len(inputs) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch = inputs[i:i+batch_size]
        batch_embeddings = instructor_model.encode(batch, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)
        
        # Update progress for embedding
        batch_progress = 55 + ((i // batch_size + 1) / total_batches * 40)  # 55-95%
        knowledge_bases[kb_id]['progress'] = int(batch_progress)
        knowledge_bases[kb_id]['current_operation'] = f'Generating embeddings ({i + len(batch)}/{len(inputs)})'
        save_knowledge_bases()
    
    # Combine all embeddings
    embeddings = np.vstack(all_embeddings).astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Update progress - Completed
    knowledge_bases[kb_id]['progress'] = 95
    knowledge_bases[kb_id]['current_operation'] = 'Finalizing'
    save_knowledge_bases()
    
    return all_chunks, embeddings

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    """
    Builds a FAISS index (inner product on normalized vectors).
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # Using inner product (cosine if normalized)
    index.add(embeddings)
    return index

def save_embeddings(kb_id: str, chunks: List[CodeChunk], index: faiss.Index) -> None:
    """Save embeddings, index and metadata to disk for a specific knowledge base"""
    kb_dir = get_kb_data_path(kb_id)
    chunks_path = os.path.join(kb_dir, 'chunks.pkl')
    index_path = os.path.join(kb_dir, 'index.faiss')
    metadata_path = os.path.join(kb_dir, 'metadata.pkl')
    
    # Save chunks and metadata
    chunks_data = [chunk.to_dict() for chunk in chunks]
    with open(chunks_path, 'wb') as f:
        pickle.dump(chunks_data, f)
    
    # Save FAISS index
    faiss.write_index(index, index_path)
    
    # Extract repository structure metadata
    repo_metadata = {
        'chunk_count': len(chunks),
        'chunk_types': {}
    }
    
    # Count chunk types
    for chunk in chunks:
        if chunk.chunk_type not in repo_metadata['chunk_types']:
            repo_metadata['chunk_types'][chunk.chunk_type] = 0
        repo_metadata['chunk_types'][chunk.chunk_type] += 1
    
    # Save metadata
    with open(metadata_path, 'wb') as f:
        pickle.dump(repo_metadata, f)
    
    # Update knowledge base information
    if kb_id in knowledge_bases:
        knowledge_bases[kb_id].update({
            'chunkCount': len(chunks),
            'lastModified': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            'status': 'ready'
        })
        save_knowledge_bases()

def load_kb_data(kb_id: str) -> Tuple[List[CodeChunk], faiss.Index, Dict]:
    """Load embeddings, index and metadata for a specific knowledge base"""
    kb_dir = get_kb_data_path(kb_id)
    chunks_path = os.path.join(kb_dir, 'chunks.pkl')
    index_path = os.path.join(kb_dir, 'index.faiss')
    metadata_path = os.path.join(kb_dir, 'metadata.pkl')
    
    if not os.path.exists(chunks_path) or not os.path.exists(index_path):
        return None, None, None
    
    try:
        # Load chunks
        with open(chunks_path, 'rb') as f:
            chunks_data = pickle.load(f)
        
        chunks = [CodeChunk.from_dict(chunk_data) for chunk_data in chunks_data]
        
        # Load FAISS index
        index = faiss.read_index(index_path)
        
        # Load metadata if available
        metadata = {}
        if os.path.exists(metadata_path):
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        
        return chunks, index, metadata
    
    except Exception as e:
        print(f"Error loading knowledge base data for {kb_id}: {e}")
        traceback.print_exc()
        return None, None, None

def search_kbs(kb: [], query: str, top_k: int = 5) -> List[Dict]:
    """
    Search a specific knowledge base for relevant code snippets
    """
    if instructor_model is None:
        load_model()
    
    all_chunks = []   # will be a list of lists: [[chunks_kb1], [chunks_kb2], …]
    all_indices = []  # parallel list of FAISS Index objects

    for kb_id in kb:
        chunks, idx, _ = load_kb_data(kb_id)
        if chunks is None or idx is None:
            print(f"Knowledge base {kb_id} not found or data is corrupt")
            continue

        all_chunks.append(chunks)
        all_indices.append(idx)
        
    # Embed the query
    q_emb = instructor_model.encode([[QUERY_INSTRUCTION, query]])
    q_emb = np.array(q_emb).astype('float32')
    faiss.normalize_L2(q_emb)

    # 1) Search each index and collect (score, chunk)
    all_hits = []
    for idx_obj, chunks in zip(all_indices, all_chunks):
        D, I = idx_obj.search(q_emb, top_k)
        for score, local_i in zip(D[0], I[0]):
            if 0 <= local_i < len(chunks):
                all_hits.append((score, chunks[local_i]))

    # 2) Sort all hits by descending score and pick top_k overall
    all_hits.sort(key=lambda x: x[0], reverse=True)
    top_hits = all_hits[:top_k]

    # 3) Format results exactly as before
    results = []
    for score, chunk in top_hits:
        results.append({
            'content':    chunk.content,
            'file_path':  chunk.file_path,
            'chunk_type': chunk.chunk_type,
            'name':       chunk.name,
            'score':      float(score),
            'metadata': {
                'line_start':   chunk.line_start,
                'line_end':     chunk.line_end,
                'dependencies': chunk.dependencies,
                'git_info':     chunk.git_info
            }
        })
    
    return results

def search_kb(kb_id: str, query: str, top_k: int = 5) -> List[Dict]:
    """
    Search a specific knowledge base for relevant code snippets
    """
    if instructor_model is None:
        load_model()
    
    chunks, index, _ = load_kb_data(kb_id)
    if chunks is None or index is None:
        print(f"Knowledge base {kb_id} not found or data is corrupt")
        return []
    
    # Embed the query
    q_emb = instructor_model.encode([[QUERY_INSTRUCTION, query]])
    q_emb = np.array(q_emb).astype('float32')
    faiss.normalize_L2(q_emb)
    
    # Search index
    D, I = index.search(q_emb, top_k)
    
    # Format results
    results = []
    for i, score in zip(I[0], D[0]):
        if i < len(chunks):
            chunk = chunks[i]
            results.append({
                'content': chunk.content,
                'file_path': chunk.file_path,
                'chunk_type': chunk.chunk_type,
                'name': chunk.name,
                'score': float(score),
                'metadata': {
                    'line_start': chunk.line_start,
                    'line_end': chunk.line_end,
                    'dependencies': chunk.dependencies,
                    'git_info': chunk.git_info
                }
            })
    
    return results

# ========== API ROUTES ==========
@app.route('/api/status', methods=['GET'])
def status():
    """Check if the API is running and ready"""
    if instructor_model is None:
        return jsonify({"status": "warming_up", "message": "Model is still loading"}), 503
    
    # Get status of all knowledge bases
    kb_statuses = {}
    for kb_id, kb_info in knowledge_bases.items():
        kb_statuses[kb_id] = {
            "status": kb_info.get("status", "unknown"),
            "name": kb_info.get("name", ""),
            "chunkCount": kb_info.get("chunkCount", 0),
            "lastModified": kb_info.get("lastModified", ""),
            "progress": kb_info.get("progress", 0),
            "current_operation": kb_info.get("current_operation", "")
        }
    
    return jsonify({
        "status": "ok",
        "model": "instructor-base",
        "knowledge_bases": kb_statuses
    })

def update_search_progress(kb_id: str, progress: int, operation: str):
    """Update search progress for a knowledge base"""
    with search_progress_lock:
        search_progress[kb_id] = {
            'progress': progress,
            'current_operation': operation,
            'status': 'searching'
        }

@app.route('/api/search-progress/<kb_id>', methods=['GET'])
def get_search_progress(kb_id: str):
    """Get current search progress for a knowledge base"""
    with search_progress_lock:
        return jsonify(search_progress.get(kb_id, {'status': 'not_found'}))

@app.route('/api/search', methods=['POST'])
def search():
    """Enhanced search endpoint using progressive search"""
    try:
        data = request.json
        query = data.get('query')
        kb_id = data.get('kb_id')
        
        if not query or not kb_id:
            return jsonify({'error': 'Missing query or kb_id'}), 400
        
        if kb_id not in knowledge_bases:
            return jsonify({'error': 'Knowledge base not found'}), 404
        
        # Initialize progress
        update_search_progress(kb_id, 0, "Starting search")
        
        # Load code graph if not in memory
        if kb_id not in code_graphs:
            update_search_progress(kb_id, 20, "Loading code graph")
            graph_path = os.path.join(get_kb_data_path(kb_id), GRAPH_PATH)
            if os.path.exists(graph_path):
                code_graphs[kb_id] = CodeGraph.load(graph_path)
            else:
                update_search_progress(kb_id, 0, "Error: Code graph not found")
                return jsonify({'error': 'Code graph not found. Please index the knowledge base first.'}), 404
        
        # Initialize progressive search
        update_search_progress(kb_id, 40, "Initializing search engine")
        search_engine = ProgressiveSearch(code_graphs[kb_id], instructor_model)
        
        # Set up progress callback
        def progress_callback(progress: int, operation: str):
            update_search_progress(kb_id, progress, operation)
        
        search_engine.set_progress_callback(progress_callback)
        
        # Set up timeout
        search_completed = threading.Event()
        search_error = [None]  # Use list to store error message
        
        def timeout_handler():
            if not search_completed.is_set():
                search_error[0] = "Search operation timed out"
                search_completed.set()
        
        timer = Timer(SEARCH_TIMEOUT, timeout_handler)
        timer.start()
        
        try:
            # Perform search
            update_search_progress(kb_id, 60, "Performing search")
            results = search_engine.search(query)
            search_completed.set()
            
            if search_error[0]:
                raise Exception(search_error[0])
            
            return jsonify({
                'results': results,
                'query': query,
                'kb_id': kb_id
            })
            
        finally:
            timer.cancel()
        
    except Exception as e:
        print(f"Error in search: {str(e)}")
        traceback.print_exc()
        update_search_progress(kb_id, 0, f"Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload a file to be indexed"""
    create_directory_if_not_exists(UPLOADS_DIR)
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    kb_id = request.form.get('kb_id')
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not kb_id:
        return jsonify({"error": "Knowledge base ID is required"}), 400
    
    if kb_id not in knowledge_bases:
        return jsonify({"error": f"Knowledge base {kb_id} not found"}), 404
    
    filename = file.filename
    ext = os.path.splitext(filename)[1].lower()
    
    # Check if we support this file type
    supported = False
    for exts in SUPPORTED_EXTENSIONS.values():
        if ext in exts:
            supported = True
            break
    
    if not supported:
        return jsonify({"error": f"Unsupported file extension: {ext}"}), 400
    
    # Save the file to uploads directory
    upload_dir = os.path.join(UPLOADS_DIR, kb_id)
    create_directory_if_not_exists(upload_dir)
    
    filepath = os.path.join(upload_dir, filename)
    file.save(filepath)
    
    try:
        # Parse file based on its type
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Add to knowledge base (will be indexed on next update)
        if ext in SUPPORTED_EXTENSIONS['python']:
            chunks = parse_python_file(filename, content)
        elif ext in SUPPORTED_EXTENSIONS['javascript']:
            chunks = parse_js_file(filename, content)
        else:
            chunks = parse_generic_file(filename, content)
        
        # Update knowledge base status
        knowledge_bases[kb_id]['status'] = 'pending_update'
        knowledge_bases[kb_id]['lastModified'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        save_knowledge_bases()
        
        return jsonify({
            "message": f"File '{filename}' uploaded successfully",
            "fileInfo": {
                "name": filename,
                "extension": ext,
                "size": len(content),
                "chunks": len(chunks)
            }
        })
        
    except Exception as e:
        print(f"Error processing uploaded file: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Error processing file: {str(e)}"}), 500

@app.route('/api/knowledge-bases', methods=['GET'])
def list_knowledge_bases():
    """List all knowledge bases with their current status"""
    # Cache the response for 1 second to reduce load
    current_time = time.time()
    if not hasattr(list_knowledge_bases, 'last_response') or \
       current_time - list_knowledge_bases.last_time > 1:
        kb_list = {}
        for kb_id, kb_info in knowledge_bases.items():
            # Check if thread is still alive for indexing knowledge bases
            if kb_id in indexing_threads and kb_info['status'] == 'indexing':
                if not indexing_threads[kb_id].is_alive():
                    kb_info['status'] = 'error'
                    kb_info['error_message'] = 'Indexing process failed'
                    kb_info['current_operation'] = 'Error: Indexing process failed'
                    save_knowledge_bases()
            
            kb_list[kb_id] = {
                "id": kb_id,
                "name": kb_info.get("name", ""),
                "description": kb_info.get("description", ""),
                "status": kb_info.get("status", "unknown"),
                "chunkCount": kb_info.get("chunkCount", 0),
                "lastModified": kb_info.get("lastModified", ""),
                "progress": kb_info.get("progress", 0),
                "current_operation": kb_info.get("current_operation", "")
            }
        list_knowledge_bases.last_response = {"knowledge_bases": kb_list}
        list_knowledge_bases.last_time = current_time
    
    return jsonify(list_knowledge_bases.last_response)

@app.route('/api/knowledge-bases', methods=['POST'])
def create_knowledge_base():
    """Create a new knowledge base from a Git repository"""
    logger = logging.getLogger('api')
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    name = data.get('name')
    git_url = data.get('git_url')
    
    if not name:
        return jsonify({"error": "Knowledge base name is required"}), 400
    
    if not git_url:
        return jsonify({"error": "Git repository URL is required"}), 400
    
    # Generate a unique ID
    kb_id = f"kb_{int(time.time())}"
    logger.info(f"Creating new knowledge base {kb_id} with name {name}")
    
    try:
        # Create knowledge base entry
        knowledge_bases[kb_id] = {
            "id": kb_id,
            "name": name,
            "description": data.get('description', ''),
            "git_url": git_url,
            "created": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "lastModified": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            "chunkCount": 0,
            "status": "pending",
            "progress": 0,
            "current_operation": "Initializing"
        }
        
        # Create directory for knowledge base
        kb_dir = get_kb_data_path(kb_id)
        create_directory_if_not_exists(kb_dir)
        logger.info(f"Created directory structure for {kb_id}")
        
        # Save knowledge bases info
        save_knowledge_bases()
        
        # Copy/clone the repository
        logger.info(f"Copying/cloning repository from {git_url} to {kb_id}")
        knowledge_bases[kb_id]['current_operation'] = 'Copying repository'
        knowledge_bases[kb_id]['progress'] = 10
        save_knowledge_bases()
        
        repo_path = clone_git_repository(git_url, kb_id)
        logger.info(f"Repository copied/cloned successfully to {repo_path}")
        
        # Start indexing in background thread
        logger.info(f"Starting indexing thread for {kb_id}")
        indexing_thread = threading.Thread(
            target=index_knowledge_base_async,
            args=(kb_id,),
            daemon=True
        )
        indexing_threads[kb_id] = indexing_thread
        indexing_thread.start()
        
        response = {
            "message": "Knowledge base creation started",
            "kb_id": kb_id,
            "kb_info": knowledge_bases[kb_id]
        }
        logger.info(f"Knowledge base creation initiated for {kb_id}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error creating knowledge base {kb_id}: {str(e)}")
        logger.error(traceback.format_exc())
        # Clean up if something went wrong
        if kb_id in knowledge_bases:
            del knowledge_bases[kb_id]
            save_knowledge_bases()
        return jsonify({"error": f"Error creating knowledge base: {str(e)}"}), 500

@app.route('/api/knowledge-bases/<kb_id>', methods=['GET'])
def get_knowledge_base(kb_id):
    """Get information about a specific knowledge base"""
    if kb_id not in knowledge_bases:
        return jsonify({"error": f"Knowledge base {kb_id} not found"}), 404
    
    # Try to load metadata if available
    kb_dir = get_kb_data_path(kb_id)
    metadata_path = os.path.join(kb_dir, 'metadata.pkl')
    metadata = {}
    
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
        except Exception as e:
            print(f"Error loading metadata for {kb_id}: {e}")
    
    # Return knowledge base info with metadata
    return jsonify({
        "kb_info": knowledge_bases[kb_id],
        "metadata": metadata
    })

@app.route('/api/knowledge-bases/<kb_id>', methods=['DELETE'])
def delete_knowledge_base(kb_id):
    """Delete a knowledge base"""
    if kb_id not in knowledge_bases:
        return jsonify({"error": f"Knowledge base {kb_id} not found"}), 404
    
    # Delete knowledge base data
    kb_dir = get_kb_data_path(kb_id)
    upload_dir = os.path.join(UPLOADS_DIR, kb_id)
    
    try:
        # Remove from knowledge bases dict
        del knowledge_bases[kb_id]
        save_knowledge_bases()
        
        # Remove data directories
        import shutil
        if os.path.exists(kb_dir):
            shutil.rmtree(kb_dir)
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        
        return jsonify({"message": f"Knowledge base {kb_id} deleted successfully"})
    
    except Exception as e:
        print(f"Error deleting knowledge base {kb_id}: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Error deleting knowledge base: {str(e)}"}), 500

@app.route('/api/knowledge-bases/<kb_id>/index', methods=['POST'])
def index_knowledge_base(kb_id):
    """Enhanced index endpoint that builds code graph"""
    try:
        if kb_id not in knowledge_bases:
            return jsonify({'error': 'Knowledge base not found'}), 404
        
        repo_path = os.path.join(UPLOADS_DIR, kb_id)
        if not os.path.exists(repo_path):
            return jsonify({'error': 'Repository not found'}), 404
        
        # Check if already indexing
        if kb_id in indexing_threads and indexing_threads[kb_id].is_alive():
            return jsonify({'error': 'Knowledge base is already being indexed'}), 409
        
        # Update status to indexing
        knowledge_bases[kb_id]['status'] = 'indexing'
        knowledge_bases[kb_id]['progress'] = 0
        knowledge_bases[kb_id]['current_operation'] = 'Starting indexing process'
        knowledge_bases[kb_id]['indexing_state'] = {
            'current_file': None,
            'processed_files': 0,
            'total_files': 0,
            'current_phase': 'initialization',
            'last_successful_phase': None,
            'processed_files_list': []
        }
        save_knowledge_bases()
        
        # Start indexing in background thread
        indexing_thread = threading.Thread(
            target=index_knowledge_base_async,
            args=(kb_id,),
            daemon=True
        )
        indexing_threads[kb_id] = indexing_thread
        indexing_thread.start()
        
        return jsonify({
            'message': 'Indexing started successfully',
            'kb_id': kb_id,
            'status': knowledge_bases[kb_id]
        })
        
    except Exception as e:
        print(f"Error starting indexing for {kb_id}: {str(e)}")
        traceback.print_exc()
        
        # Update knowledge base status
        if kb_id in knowledge_bases:
            knowledge_bases[kb_id]['status'] = 'error'
            knowledge_bases[kb_id]['error_message'] = str(e)
            save_knowledge_bases()
            
        return jsonify({
            'error': f'Error starting indexing: {str(e)}',
            'kb_id': kb_id,
            'status': knowledge_bases.get(kb_id, {})
        }), 500

@app.route('/api/ollama/<kb_id>', methods=['POST'])
def query_ollama(kb_id):
    """Query Ollama with search results for enhanced answers"""
    if instructor_model is None:
        return jsonify({"error": "Server is still initializing"}), 503
    
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    query = data.get('query')
    top_k = data.get('top_k', 5)
    model = data.get('model', 'qwen2:14b-instruct')  # Default to qwen2
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    if kb_id not in knowledge_bases:
        return jsonify({"error": f"Knowledge base {kb_id} not found"}), 404
    
    try:
        # First, search the knowledge base
        snippets = search_kb(kb_id, query, top_k)
        if not snippets:
            return jsonify({"error": "No relevant code snippets found"}), 404
        
        # Format snippets for Ollama with better context
        formatted_snippets = []
        for idx, snippet in enumerate(snippets, 1):
            # Get file extension and type
            file_ext = os.path.splitext(snippet['file_path'])[1]
            file_type = "JavaScript" if file_ext in ['.js', '.jsx'] else "TypeScript" if file_ext in ['.ts', '.tsx'] else "Other"
            
            # Format snippet with enhanced metadata
            formatted = f"[SNIPPET {idx}] {snippet['file_path']}\n"
            formatted += f"Type: {file_type}\n"
            formatted += f"Chunk Type: {snippet['chunk_type']}\n"
            if snippet.get('name'):
                formatted += f"Name: {snippet['name']}\n"
            if snippet['metadata'].get('dependencies'):
                formatted += f"Dependencies: {', '.join(snippet['metadata']['dependencies'])}\n"
            if snippet['metadata'].get('git_info'):
                git_info = snippet['metadata']['git_info']
                if git_info.get('last_modified'):
                    formatted += f"Last Modified: {git_info['last_modified']}\n"
                if git_info.get('commit_msg'):
                    formatted += f"Commit Message: {git_info['commit_msg']}\n"
            formatted += f"\nCode:\n```{file_ext[1:] if file_ext else ''}\n{snippet['content']}\n```\n"
            formatted_snippets.append(formatted)
        
        context = "\n\n".join(formatted_snippets)
        
        # Create structured prompt for Ollama
        prompt = f"""You are a helpful programming assistant that answers code questions based on repository snippets.

CODE CONTEXT:
{context}

Based on the above code snippets from the repository, please answer the following question:
{query}

Provide a clear and concise answer, referencing specific parts of the code when needed.
If you're unsure about something, acknowledge the uncertainty rather than making assumptions.
If the answer requires code examples, always format code blocks properly using markdown syntax.
"""

        # Configure Ollama endpoint
        OLLAMA_URL = "http://localhost:11434/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,  # Reduced for more focused answers
                "top_p": 0.9,
                "top_k": 40,
                "max_tokens": 1000
            }
        }
        
        # Call Ollama API
        import requests
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        
        # Return both the LLM response and the snippets used
        return jsonify({
            "response": data.get("response", "").strip(),
            "snippets": snippets,
            "model": model
        })
        
    except requests.RequestException as e:
        return jsonify({
            "error": f"Error connecting to Ollama: {str(e)}",
            "details": "Make sure Ollama is running on http://localhost:11434"
        }), 503
    except Exception as e:
        print(f"Error querying Ollama: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Error querying Ollama: {str(e)}"}), 500

@app.route('/api/knowledge-bases/<kb_id>/stats', methods=['GET'])
def knowledge_base_stats(kb_id):
    """Get detailed statistics about a knowledge base"""
    if kb_id not in knowledge_bases:
        return jsonify({"error": f"Knowledge base {kb_id} not found"}), 404
    
    try:
        # Load metadata
        _, _, metadata = load_kb_data(kb_id)
        if metadata is None:
            return jsonify({"error": "Metadata not available"}), 404
        
        # Add knowledge base info
        stats = {
            "kb_info": knowledge_bases[kb_id],
            "metadata": metadata
        }
        
        return jsonify(stats)
    
    except Exception as e:
        print(f"Error getting stats for {kb_id}: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Error getting stats: {str(e)}"}), 500

@app.route('/api/knowledge-bases/<kb_id>/indexing-state', methods=['GET'])
def get_indexing_state(kb_id):
    """Get detailed information about the current indexing state"""
    if kb_id not in knowledge_bases:
        return jsonify({"error": "Knowledge base not found"}), 404
    
    kb = knowledge_bases[kb_id]
    
    # Get basic knowledge base info
    state = {
        "id": kb_id,
        "name": kb.get("name", ""),
        "status": kb.get("status", "unknown"),
        "progress": kb.get("progress", 0),
        "current_operation": kb.get("current_operation", ""),
        "lastModified": kb.get("lastModified", ""),
        "error_message": kb.get("error_message", None)
    }
    
    # Add detailed indexing state if available
    if "indexing_state" in kb:
        indexing_state = kb["indexing_state"]
        state["indexing_details"] = {
            "current_phase": indexing_state.get("current_phase", "unknown"),
            "current_file": indexing_state.get("current_file", None),
            "processed_files": indexing_state.get("processed_files", 0),
            "total_files": indexing_state.get("total_files", 0),
            "last_successful_phase": indexing_state.get("last_successful_phase", None),
            "last_error": indexing_state.get("last_error", None),
            "processed_files_list": indexing_state.get("processed_files_list", [])[-10:]  # Last 10 files
        }
        
        # Calculate additional metrics
        if indexing_state.get("total_files", 0) > 0:
            state["indexing_details"]["completion_percentage"] = round(
                (indexing_state.get("processed_files", 0) / indexing_state.get("total_files", 0)) * 100, 2
            )
        
        # Add phase descriptions
        state["indexing_details"]["phase_descriptions"] = {
            "initialization": "Setting up indexing process",
            "building_graph": "Building code dependency graph",
            "processing_files": "Processing and analyzing files",
            "saving_graph": "Saving code graph to disk",
            "completed": "Indexing completed successfully"
        }
    
    # Add thread information if indexing is in progress
    if kb_id in indexing_threads:
        thread = indexing_threads[kb_id]
        state["thread_info"] = {
            "is_alive": thread.is_alive(),
            "is_daemon": thread.daemon,
            "name": thread.name
        }
    
    # Add file system information
    kb_path = os.path.join(UPLOADS_DIR, kb_id)
    if os.path.exists(kb_path):
        state["file_system"] = {
            "exists": True,
            "size": sum(os.path.getsize(os.path.join(dirpath,filename)) 
                       for dirpath, dirnames, filenames in os.walk(kb_path) 
                       for filename in filenames),
            "file_count": sum(len(files) for _, _, files in os.walk(kb_path)),
            "has_progress_file": os.path.exists(os.path.join(kb_path, 'indexing_progress.json')),
            "has_graph_file": os.path.exists(os.path.join(kb_path, GRAPH_PATH))
        }
    else:
        state["file_system"] = {
            "exists": False
        }
    
    return jsonify(state)

# Serve static files from the frontend directory
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_static(path):
    """Serve static files from the frontend directory"""
    if path != "" and os.path.exists(os.path.join('frontend', path)):
        return send_from_directory('frontend', path)
    else:
        return send_from_directory('frontend', 'index.html')

@app.route('/api/ollama/models', methods=['GET'])
def list_ollama_models():
    """
    Return the names of all models currently available to the Ollama CLI.
    """
    try:
        models = get_ollama_models_via_cli()
        return jsonify({"models": models})
    except subprocess.CalledProcessError as e:
        # Ollama CLI returned an error (e.g. daemon not running)
        return jsonify({
            "error": "Failed to list Ollama models",
            "details": e.stderr or str(e)
        }), 500

@app.route('/api/combineSearch', methods=['POST'])
def combinedSearch():
    """Search the combined code index for relevant snippets"""
    if instructor_model is None:
        return jsonify({"error": "Server is still initializing"}), 503
    
    # Get request data
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    query = data.get('query')
    knowledgeBases = data.get('kb_id')
    top_k = data.get('top_k', 5)
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        results = search_kbs(knowledgeBases, query, top_k)
        return jsonify({"results": results})
    except Exception as e:
        print(f"Error during search: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

@app.route('/api/search-all', methods=['POST'])
def search_all():
    """Search across all active knowledge bases"""
    if instructor_model is None:
        return jsonify({"error": "Server is still initializing"}), 503
    
    # Get request data
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    query = data.get('query')
    top_k = data.get('top_k', 5)
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # Search all knowledge bases with 'ready' status
        all_results = []
        active_kbs = [kb_id for kb_id, info in knowledge_bases.items() if info.get('status') == 'ready']
        
        if not active_kbs:
            return jsonify({"message": "No active knowledge bases found", "results": []}), 200
        
        # Search each knowledge base
        for kb_id in active_kbs:
            kb_results = search_kb(kb_id, query, top_k)
            for result in kb_results:
                result['kb_id'] = kb_id
                result['kb_name'] = knowledge_bases[kb_id]['name']
            all_results.extend(kb_results)
        
        # Sort by score
        all_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:top_k]
        
        return jsonify({
            "results": all_results,
            "knowledge_bases_searched": len(active_kbs)
        })
    
    except Exception as e:
        print(f"Error during search: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

def index_knowledge_base_async(kb_id: str, resume_from: str = None):
    """Background thread function to handle indexing with resume capability"""
    try:
        print(f"Starting indexing process for knowledge base {kb_id}")
        kb = knowledge_bases.get(kb_id)
        if not kb:
            print(f"Knowledge base {kb_id} not found")
            return

        # Update status to indexing
        kb['status'] = 'indexing'
        kb['progress'] = 0
        kb['current_operation'] = 'Starting repository analysis'
        kb['indexing_state'] = {
            'current_file': None,
            'processed_files': 0,
            'total_files': 0,
            'current_phase': 'initialization',
            'last_successful_phase': None
        }
        save_knowledge_bases()

        # Set a longer timeout for code graph building (15 minutes)
        timeout_seconds = 900  # 15 minutes
        timeout_occurred = [False]

        def timeout_handler():
            timeout_occurred[0] = True
            print(f"Timeout while building code graph for {kb_id}")
            kb['status'] = 'error'
            kb['error_message'] = 'Code graph building timed out after 15 minutes. The repository might be too large or complex.'
            save_knowledge_bases()

        # Start the timeout timer
        timer = Timer(timeout_seconds, timeout_handler)
        timer.start()

        try:
            kb_path = os.path.join(UPLOADS_DIR, kb_id)
            if not os.path.exists(kb_path):
                raise Exception(f"Repository path {kb_path} does not exist")

            # Initialize or load progress tracking
            if resume_from and os.path.exists(os.path.join(kb_path, 'indexing_progress.json')):
                with open(os.path.join(kb_path, 'indexing_progress.json'), 'r') as f:
                    progress_data = json.load(f)
                    kb['indexing_state'].update(progress_data)
                    print(f"Resuming indexing from phase: {progress_data['current_phase']}")
            else:
                # Calculate total files for progress tracking (excluding node_modules and other excluded paths)
                total_files = 0
                for root, _, files in os.walk(kb_path):
                    rel_root = os.path.relpath(root, kb_path)
                    if should_exclude_path(rel_root):
                        continue
                    for file in files:
                        if file.endswith(('.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.h', '.hpp')):
                            rel_path = os.path.join(rel_root, file)
                            if not should_exclude_path(rel_path):
                                total_files += 1
                kb['indexing_state']['total_files'] = total_files

            # Build code graph with progress updates
            kb['current_operation'] = 'Building code graph'
            kb['progress'] = 10
            kb['indexing_state']['current_phase'] = 'building_graph'
            save_knowledge_bases()

            graph = {
                'nodes': [],  # Files and dependencies
                'edges': [],   # Import relationships
                'file_contents': {}  # Store file contents
            }

            # Process files with resume capability
            for root, dirs, files in os.walk(kb_path):
                # Skip excluded directories
                rel_root = os.path.relpath(root, kb_path)
                if should_exclude_path(rel_root):
                    dirs.clear()  # Clear dirs list to prevent walking into excluded directories
                    continue

                # Filter out excluded directories
                dirs[:] = [d for d in dirs if not should_exclude_path(os.path.join(rel_root, d))]

                for file in files:
                    if file.endswith(('.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cpp', '.h', '.hpp')):
                        file_path = os.path.join(root, file)
                        relative_path = os.path.relpath(file_path, kb_path)
                        
                        # Skip excluded files
                        if should_exclude_path(relative_path):
                            continue
                        
                        # Skip already processed files if resuming
                        if resume_from and relative_path in kb['indexing_state'].get('processed_files_list', []):
                            continue

                        # Update current file being processed
                        kb['indexing_state']['current_file'] = relative_path
                        kb['current_operation'] = f'Processing {relative_path}'
                        
                        # Add file node
                        graph['nodes'].append({
                            'id': relative_path,
                            'type': 'file',
                            'name': file
                        })

                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read()
                                
                            # Store file content
                            graph['file_contents'][relative_path] = content

                            # Extract imports based on file type
                            if file.endswith(('.py')):
                                # Python imports
                                import_pattern = r'^(?:from\s+(\S+)\s+import\s+|import\s+(\S+))'
                                for line in content.split('\n'):
                                    match = re.match(import_pattern, line.strip())
                                    if match:
                                        imported = match.group(1) or match.group(2)
                                        # Handle relative imports
                                        if imported.startswith('.'):
                                            imported = os.path.normpath(os.path.join(os.path.dirname(relative_path), imported))
                                        graph['edges'].append({
                                            'source': relative_path,
                                            'target': imported,
                                            'type': 'imports'
                                        })
                            elif file.endswith(('.js', '.jsx', '.ts', '.tsx')):
                                # JavaScript/TypeScript imports
                                import_patterns = [
                                    r'^import\s+(?:\*\s+as\s+\w+\s+from\s+)?[\'"]([^\'"]+)[\'"]',
                                    r'^import\s+{[^}]+}\s+from\s+[\'"]([^\'"]+)[\'"]',
                                    r'^import\s+\w+\s+from\s+[\'"]([^\'"]+)[\'"]',
                                    r'^require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)'
                                ]
                                
                                for line in content.split('\n'):
                                    for pattern in import_patterns:
                                        match = re.match(pattern, line.strip())
                                        if match:
                                            imported = match.group(1)
                                            # Handle relative imports
                                            if imported.startswith('.'):
                                                imported = os.path.normpath(os.path.join(os.path.dirname(relative_path), imported))
                                            graph['edges'].append({
                                                'source': relative_path,
                                                'target': imported,
                                                'type': 'imports'
                                            })
                                            break
                                    
                                    # Look for class inheritance
                                    class_match = re.match(r'^class\s+\w+\s+extends\s+(\w+)', line.strip())
                                    if class_match:
                                        parent_class = class_match.group(1)
                                        graph['edges'].append({
                                            'source': relative_path,
                                            'target': parent_class,
                                            'type': 'extends'
                                        })
                                    
                                    # Look for interface implementations
                                    impl_match = re.match(r'^class\s+\w+\s+implements\s+(\w+)', line.strip())
                                    if impl_match:
                                        interface = impl_match.group(1)
                                        graph['edges'].append({
                                            'source': relative_path,
                                            'target': interface,
                                            'type': 'implements'
                                        })
                                    
                                    # Look for component composition
                                    comp_match = re.search(r'<(\w+)[\s>]', line.strip())
                                    if comp_match:
                                        component = comp_match.group(1)
                                        graph['edges'].append({
                                            'source': relative_path,
                                            'target': component,
                                            'type': 'uses_component'
                                        })
                            elif file.endswith(('.java', '.cpp', '.h', '.hpp')):
                                # Java/C++ imports
                                import_pattern = r'^(?:import|#include)\s+[\"<]([^\">]+)[\">]'
                                for line in content.split('\n'):
                                    match = re.match(import_pattern, line.strip())
                                    if match:
                                        imported = match.group(1)
                                        # Handle relative imports
                                        if imported.startswith('.'):
                                            imported = os.path.normpath(os.path.join(os.path.dirname(relative_path), imported))
                                        graph['edges'].append({
                                            'source': relative_path,
                                            'target': imported,
                                            'type': 'imports'
                                        })
                        except Exception as e:
                            print(f"Error processing file {file_path}: {str(e)}")
                            continue

                        # Update progress tracking
                        kb['indexing_state']['processed_files'] += 1
                        if 'processed_files_list' not in kb['indexing_state']:
                            kb['indexing_state']['processed_files_list'] = []
                        kb['indexing_state']['processed_files_list'].append(relative_path)
                        
                        # Calculate progress
                        progress = min(60, int((kb['indexing_state']['processed_files'] / kb['indexing_state']['total_files']) * 50) + 10)
                        kb['progress'] = progress
                        
                        # Save progress every 10 files
                        if kb['indexing_state']['processed_files'] % 10 == 0:
                            # Save progress to file
                            with open(os.path.join(kb_path, 'indexing_progress.json'), 'w') as f:
                                json.dump(kb['indexing_state'], f)
                            save_knowledge_bases()

            # Save the graph
            kb['current_operation'] = 'Saving code graph'
            kb['progress'] = 70
            kb['indexing_state']['current_phase'] = 'saving_graph'
            save_knowledge_bases()

            graph_path = os.path.join(get_kb_data_path(kb_id), GRAPH_PATH)
            with open(graph_path, 'w') as f:
                json.dump(graph, f)

            if timeout_occurred[0]:
                return

            # Create chunks and embeddings
            kb['current_operation'] = 'Creating code chunks'
            kb['progress'] = 80
            kb['indexing_state']['current_phase'] = 'creating_chunks'
            save_knowledge_bases()

            chunks, embeddings = embed_codebase(kb_path, kb_id)
            
            # Create FAISS index
            kb['current_operation'] = 'Building search index'
            kb['progress'] = 90
            kb['indexing_state']['current_phase'] = 'building_index'
            save_knowledge_bases()

            dimension = embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)

            # Save embeddings and chunks
            save_embeddings(kb_id, chunks, index)

            # Update status to ready
            kb['status'] = 'ready'
            kb['progress'] = 100
            kb['current_operation'] = 'Completed'
            kb['indexing_state']['current_phase'] = 'completed'
            save_knowledge_bases()

            # Clean up progress file after successful completion
            progress_file = os.path.join(kb_path, 'indexing_progress.json')
            if os.path.exists(progress_file):
                os.remove(progress_file)

        except Exception as e:
            print(f"Error during indexing: {str(e)}")
            traceback.print_exc()
            kb['status'] = 'error'
            kb['error_message'] = str(e)
            save_knowledge_bases()
        finally:
            timer.cancel()

    except Exception as e:
        print(f"Unexpected error in indexing thread for {kb_id}: {str(e)}")
        kb = knowledge_bases.get(kb_id)
        if kb:
            kb['status'] = 'error'
            kb['error_message'] = f'Unexpected error: {str(e)}'
            save_knowledge_bases()

@app.route('/api/knowledge-bases/<kb_id>/resume', methods=['POST'])
def resume_indexing(kb_id):
    """Resume the indexing process for a knowledge base"""
    if kb_id not in knowledge_bases:
        return jsonify({"error": "Knowledge base not found"}), 404
    
    kb = knowledge_bases[kb_id]
    if kb['status'] != 'error':
        return jsonify({"error": "Can only resume from error state"}), 400
    
    try:
        # Start new indexing thread with resume flag
        indexing_thread = threading.Thread(
            target=index_knowledge_base_async,
            args=(kb_id, True),
            daemon=True
        )
        indexing_threads[kb_id] = indexing_thread
        indexing_thread.start()
        
        return jsonify({
            "message": "Indexing resumed",
            "kb_id": kb_id,
            "status": "indexing"
        })
    except Exception as e:
        return jsonify({"error": f"Error resuming indexing: {str(e)}"}), 500

if __name__ == '__main__':
    # Load model and knowledge bases
    load_model()
    load_knowledge_bases()
    
    # Create necessary directories
    create_directory_if_not_exists(UPLOADS_DIR)
    create_directory_if_not_exists(DATA_DIR)
    
    # Start the server
    app.run(host='0.0.0.0', port=6146, debug=True)