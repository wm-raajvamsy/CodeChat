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

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
INDEX_PATH = 'code_index.faiss'  # File with FAISS index
CHUNKS_PATH = 'code_chunks.pkl'  # File with code chunks
METADATA_PATH = 'code_metadata.pkl'  # File with metadata
KB_PATH = 'knowledge_bases.json'  # Path to store knowledge base info
UPLOADS_DIR = 'uploads'  # Directory to store uploaded files
DATA_DIR = 'data'  # Directory to store indexed data per knowledge base

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

def create_directory_if_not_exists(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)

def get_kb_data_path(kb_id):
    """Get the data path for a specific knowledge base"""
    kb_dir = os.path.join(DATA_DIR, kb_id)
    create_directory_if_not_exists(kb_dir)
    return kb_dir

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

def save_knowledge_bases():
    """Save the knowledge bases information to disk"""
    with open(KB_PATH, 'w') as f:
        json.dump(knowledge_bases, f)

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
        # Clone the repository
        subprocess.run(["cp", "-rf", git_url, repo_dir], check=True)
        return repo_dir
    except subprocess.SubprocessError as e:
        print(f"Error cloning repository: {e}")
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
    for pattern in EXCLUDED_PATTERNS:
        # Handle glob patterns
        if pattern.startswith('*') and path.endswith(pattern[1:]):
            return True
        if pattern.endswith('*') and path.startswith(pattern[:-1]):
            return True
        # Direct match
        elif pattern in path:
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
    
    # Extract repository structure
    repo_structure = extract_repository_structure(repo_path)
    
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
    
    # Process each file
    for root, dirs, files in os.walk(repo_path):
        # Skip hidden directories and files
        if any(part.startswith('.') for part in Path(root).parts):
            continue
            
        # Skip excluded directories
        rel_root = os.path.relpath(root, repo_path)
        if should_exclude_path(rel_root):
            dirs[:] = []  # Skip all subdirectories
            continue
            
        # Modify dirs in-place to skip excluded directories
        dirs[:] = [d for d in dirs if not should_exclude_path(d) and not should_exclude_path(os.path.join(rel_root, d))]
            
        for fname in files:
            # Skip excluded files
            if should_exclude_path(fname):
                continue
                
            if any(fname.endswith(ext) for ext in supported_exts):
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, repo_path)
                
                # Skip excluded paths
                if should_exclude_path(rel_path):
                    continue
                
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
    
    print(f"Extracted {len(all_chunks)} code chunks for knowledge base {kb_id}")
    
    # Prepare for embedding
    embedding_texts = [chunk.get_embedding_text() for chunk in all_chunks]
    
    # Prepare instruction-text pairs
    inputs = [[INSTRUCTION, text] for text in embedding_texts]
    print(f"Embedding {len(inputs)} chunks with Instructor model...")
    
    # Embed in batches to avoid memory issues
    batch_size = 32
    all_embeddings = []
    
    for i in tqdm(range(0, len(inputs), batch_size)):
        batch = inputs[i:i+batch_size]
        batch_embeddings = instructor_model.encode(batch, show_progress_bar=False)
        all_embeddings.append(batch_embeddings)
    
    # Combine all embeddings
    embeddings = np.vstack(all_embeddings).astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
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
    
    return jsonify({
        "status": "ok",
        "model": "instructor-base",
        "knowledge_bases": len(knowledge_bases),
        "supported_extensions": SUPPORTED_EXTENSIONS
    })

@app.route('/api/search', methods=['POST'])
def search():
    """Search the code index for relevant snippets"""
    if instructor_model is None:
        return jsonify({"error": "Server is still initializing"}), 503
    
    # Get request data
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    query = data.get('query')
    kb_id = data.get('kb_id')
    top_k = data.get('top_k', 5)
    
    if not query:
        return jsonify({"error": "No query provided"}), 400
    
    try:
        # If kb_id is provided, search only that knowledge base
        if kb_id:
            if kb_id not in knowledge_bases:
                return jsonify({"error": f"Knowledge base {kb_id} not found"}), 404
            
            results = search_kb(kb_id, query, top_k)
            return jsonify({"results": results})
        
        # If no kb_id, search all knowledge bases
        all_results = []
        for kb_id in knowledge_bases:
            kb_results = search_kb(kb_id, query, top_k)
            for result in kb_results:
                result['kb_id'] = kb_id
            all_results.extend(kb_results)
        
        # Sort by score
        all_results = sorted(all_results, key=lambda x: x['score'], reverse=True)[:top_k]
        
        return jsonify({"results": all_results})
    
    except Exception as e:
        print(f"Error during search: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Search failed: {str(e)}"}), 500

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
    """List all knowledge bases"""
    return jsonify({"knowledge_bases": knowledge_bases})

@app.route('/api/knowledge-bases', methods=['POST'])
def create_knowledge_base():
    """Create a new knowledge base from a Git repository"""
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400
    
    name = data.get('name')
    git_url = data.get('git_url')  # New field for Git repository URL
    
    if not name:
        return jsonify({"error": "Knowledge base name is required"}), 400
    
    if not git_url:
        return jsonify({"error": "Git repository URL is required"}), 400
    
    # Generate a unique ID
    kb_id = f"kb_{int(time.time())}"
    
    # Create knowledge base entry
    knowledge_bases[kb_id] = {
        "id": kb_id,
        "name": name,
        "description": data.get('description', ''),
        "git_url": git_url,  # Store the Git URL
        "created": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "lastModified": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "chunkCount": 0,
        "status": "pending"  # pending, indexing, ready, error
    }
    
    # Create directory for knowledge base
    kb_dir = get_kb_data_path(kb_id)
    create_directory_if_not_exists(kb_dir)
    
    # Save knowledge bases info
    save_knowledge_bases()
    index_knowledge_base(kb_id)
    return jsonify({
        "message": "Knowledge base created successfully",
        "kb_id": kb_id,
        "kb_info": knowledge_bases[kb_id]
    })
    
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
    """Index a knowledge base from its Git repository"""
    if kb_id not in knowledge_bases:
        return jsonify({"error": f"Knowledge base {kb_id} not found"}), 404
    
    # Get Git URL from knowledge base info
    git_url = knowledge_bases[kb_id].get('git_url')
    if not git_url:
        return jsonify({"error": "No Git URL found for this knowledge base"}), 400
    
    # Update status to indexing
    knowledge_bases[kb_id]['status'] = 'indexing'
    save_knowledge_bases()
    
    try:
        # Start indexing in a background thread to avoid blocking
        import threading
        def index_background():
            try:
                if instructor_model is None:
                    load_model()
                
                # Clone the Git repository
                try:
                    repo_path = clone_git_repository(git_url, kb_id)
                except Exception as e:
                    print(f"Error cloning repository: {e}")
                    knowledge_bases[kb_id]['status'] = 'error'
                    knowledge_bases[kb_id]['error'] = f"Git clone failed: {str(e)}"
                    save_knowledge_bases()
                    return
                
                # Embed the codebase
                chunks, embeddings = embed_codebase(repo_path, kb_id)
                
                # Build FAISS index
                index = build_faiss_index(embeddings)
                
                # Save to disk
                save_embeddings(kb_id, chunks, index)
                
                # Update knowledge base status
                knowledge_bases[kb_id]['status'] = 'ready'
                knowledge_bases[kb_id]['chunkCount'] = len(chunks)
                knowledge_bases[kb_id]['lastModified'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
                save_knowledge_bases()
                
                print(f"Indexing completed for {kb_id} with {len(chunks)} chunks")
                
            except Exception as e:
                print(f"Error indexing knowledge base {kb_id}: {e}")
                traceback.print_exc()
                knowledge_bases[kb_id]['status'] = 'error'
                knowledge_bases[kb_id]['error'] = str(e)
                save_knowledge_bases()
        
        # Start background thread
        thread = threading.Thread(target=index_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "message": f"Indexing started for knowledge base {kb_id}",
            "status": "indexing"
        })
        
    except Exception as e:
        print(f"Error starting indexing for {kb_id}: {e}")
        traceback.print_exc()
        knowledge_bases[kb_id]['status'] = 'error'
        knowledge_bases[kb_id]['error'] = str(e)
        save_knowledge_bases()
        return jsonify({"error": f"Error starting indexing: {str(e)}"}), 500

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
        
        # Format snippets for Ollama
        formatted_snippets = []
        for idx, snippet in enumerate(snippets, 1):
            # Format snippet with metadata
            formatted = f"[SNIPPET {idx}] {snippet['file_path']} ({snippet['chunk_type']} {snippet['name']})\n"
            if snippet['metadata']['dependencies']:
                formatted += f"Dependencies: {', '.join(snippet['metadata']['dependencies'])}\n"
            formatted += f"\n{snippet['content']}\n"
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
"""

        # Configure Ollama endpoint
        OLLAMA_URL = "http://localhost:11434/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9,
                "top_k": 40
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

if __name__ == '__main__':
    # Load model and knowledge bases
    load_model()
    load_knowledge_bases()
    
    # Create necessary directories
    create_directory_if_not_exists(UPLOADS_DIR)
    create_directory_if_not_exists(DATA_DIR)
    
    # Start the server
    app.run(host='0.0.0.0', port=6146, debug=True)