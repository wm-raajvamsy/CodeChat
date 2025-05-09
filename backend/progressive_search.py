from typing import Dict, List, Any, Optional
from code_analysis import CodeGraph
import numpy as np
from InstructorEmbedding import INSTRUCTOR
import faiss
import json
import re
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
from threading import Lock
import multiprocessing
import os
from functools import partial
import psutil

class ProgressiveSearch:
    def __init__(self, code_graph: CodeGraph, instructor_model: INSTRUCTOR, kb_id: str = None):
        self.code_graph = code_graph
        self.instructor_model = instructor_model
        self.query_instruction = "Represent the question for code retrieval:"
        self.code_instruction = "Represent the code snippet for retrieval:"
        self.function_instruction = "Represent the function for retrieval:"
        self.class_instruction = "Represent the class for retrieval:"
        self.progress_callback = None
        self.kb_id = kb_id
        
        # Enhanced caching with LRU-like behavior
        self._cached_embeddings = {}
        self._cache_lock = Lock()
        self._cache_size_limit = 2000  # Increased cache size for better performance
        self._cache_access_count = defaultdict(int)
        
        # CPU optimization
        self._num_cores = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
        self._max_workers = self._num_cores * 2  # Threads per core
        self._batch_size = 32  # Increased batch size for better throughput
        
        # Process pool for CPU-intensive tasks
        self._process_pool = ProcessPoolExecutor(max_workers=self._num_cores)
        
        # FAISS index for fast vector search
        self.faiss_index = None
        self.faiss_ids = []
        self.faiss_id_to_content = {}
        self.faiss_initialized = False
        
        # Try to load cached embeddings if available
        if kb_id:
            self.load_cache()
        
    def set_progress_callback(self, callback):
        """Set a callback function to report progress"""
        self.progress_callback = callback
        
    def _update_progress(self, progress: int, operation: str):
        """Update progress if callback is set"""
        if self.progress_callback:
            self.progress_callback(progress, operation)
        
    def _get_cached_embedding(self, text: str, instruction: str) -> np.ndarray:
        """Enhanced thread-safe caching with LRU-like behavior"""
        cache_key = f"{instruction}:{text}"
        
        with self._cache_lock:
            # Check cache
            if cache_key in self._cached_embeddings:
                self._cache_access_count[cache_key] += 1
                return self._cached_embeddings[cache_key]
            
            # Compute embedding
            embedding = self.instructor_model.encode(
                [[instruction, text]],
                show_progress_bar=False,
                batch_size=1  # Ensure we don't overload memory
            )[0]
            
            # Manage cache size - more efficient approach
            if len(self._cached_embeddings) >= self._cache_size_limit:
                # Remove 20% least accessed items for better memory management
                items_to_remove = int(self._cache_size_limit * 0.2)
                sorted_items = sorted(
                    self._cache_access_count.items(),
                    key=lambda x: x[1]
                )
                for key, _ in sorted_items[:items_to_remove]:
                    del self._cached_embeddings[key]
                    del self._cache_access_count[key]
            
            # Add to cache
            self._cached_embeddings[cache_key] = embedding
            self._cache_access_count[cache_key] = 1
            
            return embedding
            
    def batch_encode(self, texts, instruction):
        """Encode multiple texts in a single batch for efficiency"""
        if not texts:
            return []
            
        # Create instruction pairs for each text
        instruction_pairs = [[instruction, text] for text in texts]
        
        # Encode in batches for better performance
        return self.instructor_model.encode(
            instruction_pairs,
            show_progress_bar=False,
            batch_size=32  # Adjust based on available memory
        )
        
    def save_cache(self):
        """Save the embedding cache to disk"""
        if not self.kb_id:
            print("Cannot save cache: No knowledge base ID provided")
            return
            
        try:
            import pickle
            import os
            
            # Create cache directory if it doesn't exist
            cache_dir = os.path.join('data', self.kb_id, 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            
            # Save embeddings cache
            cache_path = os.path.join(cache_dir, 'embeddings_cache.pkl')
            with open(cache_path, 'wb') as f:
                # Only save the embeddings, not the access counts
                pickle.dump(self._cached_embeddings, f)
                
            # Save FAISS index if initialized
            if self.faiss_initialized:
                faiss_path = os.path.join(cache_dir, 'faiss_index.bin')
                faiss.write_index(self.faiss_index, faiss_path)
                
                # Save FAISS metadata
                faiss_meta_path = os.path.join(cache_dir, 'faiss_metadata.pkl')
                with open(faiss_meta_path, 'wb') as f:
                    pickle.dump({
                        'faiss_ids': self.faiss_ids,
                        'faiss_id_to_content': self.faiss_id_to_content
                    }, f)
                
            print(f"Cache saved to {cache_dir} ({len(self._cached_embeddings)} embeddings)")
        except Exception as e:
            print(f"Error saving cache: {str(e)}")
    
    def load_cache(self):
        """Load the embedding cache from disk"""
        if not self.kb_id:
            print("Cannot load cache: No knowledge base ID provided")
            return False
            
        try:
            import pickle
            import os
            
            cache_dir = os.path.join('data', self.kb_id, 'cache')
            cache_path = os.path.join(cache_dir, 'embeddings_cache.pkl')
            
            if not os.path.exists(cache_path):
                print(f"No cache file found at {cache_path}")
                return False
                
            # Load embeddings cache
            with open(cache_path, 'rb') as f:
                self._cached_embeddings = pickle.load(f)
                # Initialize access counts
                self._cache_access_count = defaultdict(int)
                for key in self._cached_embeddings:
                    self._cache_access_count[key] = 1
                    
            # Load FAISS index if available
            faiss_path = os.path.join(cache_dir, 'faiss_index.bin')
            faiss_meta_path = os.path.join(cache_dir, 'faiss_metadata.pkl')
            
            if os.path.exists(faiss_path) and os.path.exists(faiss_meta_path):
                self.faiss_index = faiss.read_index(faiss_path)
                
                # Load FAISS metadata
                with open(faiss_meta_path, 'rb') as f:
                    faiss_meta = pickle.load(f)
                    self.faiss_ids = faiss_meta['faiss_ids']
                    self.faiss_id_to_content = faiss_meta['faiss_id_to_content']
                    
                self.faiss_initialized = True
                print(f"FAISS index loaded with {self.faiss_index.ntotal} vectors")
                
            print(f"Cache loaded from {cache_dir} ({len(self._cached_embeddings)} embeddings)")
            return True
        except Exception as e:
            print(f"Error loading cache: {str(e)}")
            return False
    
    def precompute_embeddings(self):
        """Pre-compute and store embeddings for all code chunks"""
        # Check if we already have cached embeddings
        if self.kb_id and len(self._cached_embeddings) > 0:
            print(f"Using {len(self._cached_embeddings)} cached embeddings")
            
            # If FAISS index is not initialized, build it
            if not self.faiss_initialized:
                self.build_faiss_index()
                
            return
            
        print("Pre-computing embeddings for all code chunks...")
        total_files = len(self.code_graph.file_contents)
        
        # Process files in batches
        batch_size = 10
        file_items = list(self.code_graph.file_contents.items())
        
        for i in range(0, len(file_items), batch_size):
            batch = file_items[i:i+batch_size]
            for file_path, content in batch:
                chunks = self._get_semantic_chunks(content)
                
                # Process chunks in batches
                chunk_batch_size = 32
                for j in range(0, len(chunks), chunk_batch_size):
                    chunk_batch = chunks[j:j+chunk_batch_size]
                    
                    # Skip empty batches
                    if not chunk_batch:
                        continue
                        
                    # Batch encode chunks
                    embeddings = self.batch_encode(chunk_batch, self.code_instruction)
                    
                    # Cache embeddings
                    with self._cache_lock:
                        for chunk, embedding in zip(chunk_batch, embeddings):
                            cache_key = f"{self.code_instruction}:{chunk}"
                            self._cached_embeddings[cache_key] = embedding
                            self._cache_access_count[cache_key] = 1
                
                print(f"Processed file {file_path} ({i+1}/{total_files})")
                
        print(f"Embeddings pre-computed and cached for {total_files} files.")
        print(f"Total cached embeddings: {len(self._cached_embeddings)}")
        
        # Save the cache to disk
        if self.kb_id:
            self.save_cache()
        
        # Build FAISS index after precomputing embeddings
        self.build_faiss_index()
        
    def build_faiss_index(self):
        """Build a FAISS index for fast similarity search"""
        import faiss
        
        print("Building FAISS index for fast vector search...")
        
        # Get all embeddings
        embeddings = []
        self.faiss_ids = []
        self.faiss_id_to_content = {}
        
        # Process files in batches to avoid memory issues
        file_items = list(self.code_graph.file_contents.items())
        total_files = len(file_items)
        
        for file_idx, (file_path, content) in enumerate(file_items):
            chunks = self._get_semantic_chunks(content)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Get embedding from cache or compute it
                cache_key = f"{self.code_instruction}:{chunk}"
                
                with self._cache_lock:
                    if cache_key in self._cached_embeddings:
                        embedding = self._cached_embeddings[cache_key]
                    else:
                        # Compute and cache embedding
                        embedding = self._get_cached_embedding(chunk, self.code_instruction)
                
                # Add to FAISS data
                embeddings.append(embedding)
                chunk_id = f"{file_path}:{chunk_idx}"
                self.faiss_ids.append(chunk_id)
                self.faiss_id_to_content[chunk_id] = {
                    'content': chunk,
                    'file_path': file_path
                }
            
            if (file_idx + 1) % 10 == 0 or file_idx + 1 == total_files:
                print(f"Processed {file_idx + 1}/{total_files} files for FAISS index")
        
        if not embeddings:
            print("No embeddings to index. FAISS index not created.")
            return
            
        # Convert to numpy array
        embeddings_array = np.array(embeddings).astype('float32')
        
        # Build index
        dimension = embeddings_array.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.faiss_index.add(embeddings_array)
        
        self.faiss_initialized = True
        print(f"FAISS index built with {len(embeddings)} vectors of dimension {dimension}")
        
        # Save the FAISS index to disk
        if self.kb_id:
            self.save_cache()
        
    def search_faiss(self, query_embedding, top_k=20):
        """Search the FAISS index with a query embedding"""
        if not self.faiss_initialized or self.faiss_index is None:
            print("FAISS index not initialized. Building index...")
            self.build_faiss_index()
            
            if not self.faiss_initialized:
                print("Failed to build FAISS index. Falling back to regular search.")
                return []
        
        # Ensure query embedding is in the right format
        query_embedding_np = np.array([query_embedding]).astype('float32')
        
        # Search the index
        scores, indices = self.faiss_index.search(query_embedding_np, top_k)
        
        # Convert to results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(self.faiss_ids):
                continue  # Skip invalid indices
                
            chunk_id = self.faiss_ids[idx]
            chunk_data = self.faiss_id_to_content[chunk_id]
            
            # Find node metadata
            file_path = chunk_data['file_path']
            node_metadata = {}
            for node in self.code_graph.nodes:
                if node.get('id') == file_path:
                    node_metadata = node.get('metadata', {})
                    break
            
            results.append({
                'type': 'semantic',
                'content': chunk_data['content'],
                'file_path': file_path,
                'similarity': float(scores[0][i]),
                'metadata': node_metadata
            })
        
        return results

    def understand_context(self, query: str) -> Dict:
        """Enhanced context understanding with semantic analysis"""
        # Update progress for query embedding
        self._update_progress(60, "Embedding query")
        
        # Analyze query intent and extract key concepts
        query_embedding = self._get_cached_embedding(query, self.query_instruction)
        
        # Update progress for concept extraction
        self._update_progress(65, "Extracting concepts")
        
        # Extract key concepts and technical terms
        concepts = self._extract_concepts(query)
        
        # Update progress for semantic matching
        self._update_progress(70, "Finding relevant files")
        
        # Find most relevant files with semantic matching
        file_scores = []
        total_files = len(self.code_graph.file_contents)
        processed_files = 0
        
        # Process files in batches to avoid memory issues
        batch_size = 10
        file_items = list(self.code_graph.file_contents.items())
        
        for i in range(0, len(file_items), batch_size):
            batch = file_items[i:i+batch_size]
            for file_path, content in batch:
                # Get semantic chunks for better matching
                chunks = self._get_semantic_chunks(content)
                max_similarity = 0
                
                # Process chunks in smaller batches
                chunk_batch_size = 5
                for j in range(0, len(chunks), chunk_batch_size):
                    chunk_batch = chunks[j:j+chunk_batch_size]
                    chunk_embeddings = [
                        self._get_cached_embedding(chunk, self.code_instruction)
                        for chunk in chunk_batch
                    ]
                    
                    for chunk, chunk_embedding in zip(chunk_batch, chunk_embeddings):
                        similarity = np.dot(query_embedding, chunk_embedding)
                        max_similarity = max(max_similarity, similarity)
                
                file_scores.append((file_path, max_similarity))
                processed_files += 1
                
                # Update progress for file processing
                file_progress = 70 + (processed_files / total_files * 10)  # 70-80%
                self._update_progress(int(file_progress), f"Processing files ({processed_files}/{total_files})")
        
        # Sort by relevance
        file_scores.sort(key=lambda x: x[1], reverse=True)
        relevant_files = [f[0] for f in file_scores[:5]]
        
        # Update progress for intent analysis
        self._update_progress(80, "Analyzing query intent")
        
        # Determine query type and intent
        query_type, query_intent = self._analyze_query_intent(query)
        
        return {
            'query_type': query_type,
            'query_intent': query_intent,
            'concepts': self._extract_concepts(query),
            'relevant_files': relevant_files,
            'query_embedding': query_embedding.tolist()
        }
    
    def _extract_concepts(self, query: str) -> List[str]:
        """Extract key technical concepts from the query"""
        # Common technical terms and patterns
        patterns = [
            r'\b(?:function|method|class|module|component|hook|state|props|config|option)\b',
            r'\b(?:true|false|null|undefined|void|return|export|import)\b',
            r'\b(?:if|else|for|while|switch|case|try|catch|finally)\b',
            r'\b(?:const|let|var|function|class|interface|type|enum)\b',
            r'\b(?:async|await|Promise|then|catch)\b'
        ]
        
        concepts = set()
        for pattern in patterns:
            matches = re.finditer(pattern, query, re.IGNORECASE)
            concepts.update(match.group() for match in matches)
        
        return list(concepts)
    
    def _analyze_query_intent(self, query: str) -> tuple:
        """Analyze query intent and type"""
        query_lower = query.lower()
        
        # Query types
        query_types = {
            'how_to': ['how to', 'how do i', 'how can i', 'steps to'],
            'bug_fix': ['error', 'bug', 'issue', 'problem', 'fix', 'resolve'],
            'feature_request': ['add', 'implement', 'create', 'new', 'feature'],
            'location': ['where', 'find', 'locate', 'search'],
            'explanation': ['what is', 'explain', 'describe', 'tell me about'],
            'comparison': ['difference', 'compare', 'versus', 'vs'],
            'configuration': ['configure', 'setup', 'settings', 'options']
        }
        
        # Determine query type
        query_type = 'general'
        for type_name, keywords in query_types.items():
            if any(keyword in query_lower for keyword in keywords):
                query_type = type_name
                break
        
        # Determine intent
        intent = {
            'is_technical': any(term in query_lower for term in ['code', 'function', 'class', 'method', 'api']),
            'is_configuration': any(term in query_lower for term in ['config', 'setting', 'option', 'parameter']),
            'is_usage': any(term in query_lower for term in ['use', 'usage', 'example', 'sample']),
            'is_implementation': any(term in query_lower for term in ['implement', 'create', 'build', 'develop'])
        }
        
        return query_type, intent
    
    def _get_semantic_chunks(self, content: str) -> List[str]:
        """Split content into semantic chunks for better matching"""
        # Split by function/class definitions
        chunks = []
        
        # Function/class patterns
        patterns = [
            r'(?:function|class|const|let|var)\s+\w+\s*[=\(]',  # Function/class declarations
            r'export\s+(?:function|class|const|let|var)\s+\w+',  # Exported declarations
            r'interface\s+\w+\s*{',  # Interface definitions
            r'type\s+\w+\s*=',  # Type definitions
        ]
        
        # Find all matches
        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, content))
        
        # Sort matches by position
        matches.sort(key=lambda m: m.start())
        
        # Create chunks
        last_end = 0
        for match in matches:
            if match.start() > last_end:
                chunks.append(content[last_end:match.start()].strip())
            chunks.append(content[match.start():match.end()].strip())
            last_end = match.end()
        
        # Add remaining content
        if last_end < len(content):
            chunks.append(content[last_end:].strip())
        
        return [chunk for chunk in chunks if chunk]
    
    def _process_chunk_batch(self, chunks: List[str], file_path: str, query_embedding: np.ndarray) -> List[Dict]:
        """Process a batch of chunks in parallel"""
        results = []
        
        # Create a partial function with fixed arguments
        process_chunk = partial(
            self._process_single_chunk,
            file_path=file_path,
            query_embedding=query_embedding
        )
        
        # Process chunks in parallel using process pool
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)
        
        return results

    def _process_single_chunk(self, chunk: str, file_path: str, query_embedding: np.ndarray) -> Optional[Dict]:
        """Process a single chunk with optimized embedding computation"""
        try:
            chunk_embedding = self._get_cached_embedding(chunk, self.code_instruction)
            similarity = np.dot(query_embedding, chunk_embedding)
            
            if similarity > 0.5:  # Threshold for relevance
                # Find node metadata efficiently
                node_metadata = next(
                    (node.get('metadata', {}) for node in self.code_graph.nodes 
                     if node.get('id') == file_path),
                    {}
                )
                
                return {
                    'type': 'semantic',
                    'content': chunk,
                    'file_path': file_path,
                    'similarity': float(similarity),
                    'metadata': node_metadata
                }
        except Exception as e:
            print(f"Error processing chunk: {str(e)}")
        return None

    def _process_file_batch(self, batch: List[tuple], query_embedding: np.ndarray) -> List[Dict]:
        """Process a batch of files with optimized parallel processing"""
        results = []
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            futures = []
            for file_path, content in batch:
                # Get semantic chunks
                chunks = self._get_semantic_chunks(content)
                
                # Split chunks into smaller batches for better memory management
                chunk_batches = [
                    chunks[i:i + self._batch_size]
                    for i in range(0, len(chunks), self._batch_size)
                ]
                
                # Submit chunk batch processing tasks
                for chunk_batch in chunk_batches:
                    future = executor.submit(
                        self._process_chunk_batch,
                        chunk_batch,
                        file_path,
                        query_embedding
                    )
                    futures.append(future)
            
            # Collect results as they complete
            for future in as_completed(futures):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
        
        return results

    def hybrid_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Combine semantic search with BM25 lexical search for better results"""
        from rank_bm25 import BM25Okapi
        
        try:
            # Step 1: Semantic search using FAISS
            self._update_progress(60, "Performing semantic search")
            query_embedding = self._get_cached_embedding(query, self.query_instruction)
            
            # Use FAISS for fast semantic search
            if self.faiss_initialized:
                semantic_results = self.search_faiss(query_embedding, top_k * 2)
            else:
                # Fallback to regular semantic search
                semantic_results = self._quick_semantic_search(query_embedding, top_k * 2)
            
            # Step 2: BM25 lexical search
            self._update_progress(70, "Performing lexical search")
            
            # Simple tokenization function that doesn't rely on NLTK
            def simple_tokenize(text):
                # Remove punctuation and split by whitespace
                import re
                text = re.sub(r'[^\w\s]', ' ', text.lower())
                return [token for token in text.split() if token.strip()]
            
            # Tokenize query using our simple tokenizer
            tokenized_query = simple_tokenize(query)
            
            # Prepare corpus for BM25
            corpus = []
            doc_ids = []
            
            # Use a subset of files for efficiency
            relevant_files = set([r['file_path'] for r in semantic_results[:10]])
            if not relevant_files:
                # If no semantic results, use all files
                relevant_files = set(self.code_graph.file_contents.keys())
            
            # Limit to 100 files maximum for performance
            if len(relevant_files) > 100:
                relevant_files = list(relevant_files)[:100]
            
            # Build corpus from relevant files
            for file_path in relevant_files:
                if file_path in self.code_graph.file_contents:
                    content = self.code_graph.file_contents[file_path]
                    chunks = self._get_semantic_chunks(content)
                    
                    for i, chunk in enumerate(chunks):
                        # Tokenize chunk using our simple tokenizer
                        tokenized_chunk = simple_tokenize(chunk)
                        
                        corpus.append(tokenized_chunk)
                        doc_ids.append(f"{file_path}:{i}")
            
            # Skip BM25 if corpus is empty
            lexical_results = []
            if corpus:
                # Create BM25 model
                bm25 = BM25Okapi(corpus)
                
                # Get BM25 scores
                bm25_scores = bm25.get_scores(tokenized_query)
                
                # Get top BM25 results
                top_n = min(top_k * 2, len(bm25_scores))
                if top_n > 0:  # Ensure we have results
                    top_indices = np.argsort(bm25_scores)[-top_n:][::-1]
                    
                    # Convert to results format
                    for idx in top_indices:
                        if bm25_scores[idx] > 0:  # Only include relevant results
                            file_path, chunk_idx = doc_ids[idx].split(':', 1)
                            chunk_idx = int(chunk_idx)
                            chunks = self._get_semantic_chunks(self.code_graph.file_contents[file_path])
                            
                            # Find node metadata
                            node_metadata = {}
                            for node in self.code_graph.nodes:
                                if node.get('id') == file_path:
                                    node_metadata = node.get('metadata', {})
                                    break
                            
                            # Normalize score
                            normalized_score = float(bm25_scores[idx] / max(bm25_scores)) if max(bm25_scores) > 0 else 0
                            
                            lexical_results.append({
                                'type': 'lexical',
                                'content': chunks[chunk_idx] if chunk_idx < len(chunks) else "",
                                'file_path': file_path,
                                'similarity': normalized_score,
                                'metadata': node_metadata
                            })
            
            # Step 3: Combine and rank results
            self._update_progress(90, "Ranking results")
            combined_results = self._combine_and_rank_results(
                semantic_results,
                lexical_results,
                [],
                [],
                {'query_embedding': query_embedding}
            )
            
            self._update_progress(100, "Search complete")
            return combined_results[:top_k]
            
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def cached_search(self, query: str, top_k: int = 10):
        """Cache search results for common queries"""
        import hashlib
        
        try:
            # Try to use Redis for caching if available
            redis_available = False
            try:
                import redis
                import pickle
                redis_available = True
            except ImportError:
                print("Redis module not installed, using in-memory cache")
                
            if redis_available:
                try:
                    # Connect to Redis
                    r = redis.Redis(host='localhost', port=6379, db=0, socket_connect_timeout=1)
                    
                    # Test connection
                    r.ping()
                    
                    # Create cache key
                    query_hash = hashlib.md5(query.encode()).hexdigest()
                    cache_key = f"search:{query_hash}:{top_k}"
                    
                    # Check cache
                    cached = r.get(cache_key)
                    if cached:
                        print(f"Cache hit for query: {query}")
                        self._update_progress(100, "Retrieved from cache")
                        return pickle.loads(cached)
                    
                    # Perform search
                    results = self._perform_search(query, top_k)
                    
                    # Cache results (expire after 1 hour)
                    r.setex(cache_key, 3600, pickle.dumps(results))
                    
                    return results
                    
                except Exception as e:
                    print(f"Redis connection error, using in-memory cache: {str(e)}")
            
            # Fallback to in-memory cache if Redis is not available or connection failed
            query_hash = hashlib.md5(query.encode()).hexdigest()
            cache_key = f"search:{query_hash}:{top_k}"
            
            # Check in-memory cache
            if hasattr(self, '_memory_cache') and cache_key in self._memory_cache:
                print(f"Memory cache hit for query: {query}")
                self._update_progress(100, "Retrieved from memory cache")
                return self._memory_cache[cache_key]
            
            # Perform search
            results = self._perform_search(query, top_k)
            
            # Cache in memory
            if not hasattr(self, '_memory_cache'):
                self._memory_cache = {}
                
            # Limit memory cache size
            if len(self._memory_cache) > 100:
                # Remove a random item to avoid growing too large
                self._memory_cache.pop(next(iter(self._memory_cache)))
                
            self._memory_cache[cache_key] = results
            
            return results
                
        except Exception as e:
            print(f"Error in cached search: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Fallback to direct search
            return self._perform_search(query, top_k)
    
    def progressive_search(self, query: str, top_k: int = 10, callback=None):
        """Perform search in stages, returning results progressively"""
        try:
            # Get query embedding once for all stages
            query_embedding = self._get_cached_embedding(query, self.query_instruction)
            context = {'query': query, 'query_embedding': query_embedding}
            
            # Stage 1: Quick search using FAISS (return in ~100ms)
            self._update_progress(60, "Performing quick search")
            if self.faiss_initialized:
                quick_results = self.search_faiss(query_embedding, top_k)
            else:
                # Fallback to quick semantic search
                quick_results = self._quick_semantic_search(query_embedding, top_k)
                
            # Return initial results immediately
            if callback:
                callback(quick_results[:top_k], "initial")
            
            # Stage 2: Hybrid search (return in ~500ms)
            self._update_progress(70, "Performing hybrid search")
            hybrid_results = self.hybrid_search(query, top_k * 2)
            
            # Combine quick and hybrid results
            combined_results = self._combine_and_rank_results(
                quick_results,
                hybrid_results,
                [],
                [],
                context
            )
            
            # Return intermediate results
            if callback:
                callback(combined_results[:top_k], "intermediate")
            
            # Stage 3: Full analysis with graph traversal (return in ~1-2s)
            self._update_progress(80, "Performing detailed analysis")
            
            # Get pattern-based results with error handling
            try:
                pattern_results = self._pattern_based_search(query, context)
            except Exception as e:
                print(f"Error in pattern-based search: {str(e)}")
                pattern_results = []
            
            # Get context-aware expansion with error handling
            try:
                expanded_results = self._context_aware_expansion(
                    combined_results, 
                    pattern_results, 
                    context
                )
            except Exception as e:
                print(f"Error in context-aware expansion: {str(e)}")
                expanded_results = []
            
            # Get usage analysis with error handling
            try:
                usage_results = self._analyze_usage(combined_results, context)
            except Exception as e:
                print(f"Error in usage analysis: {str(e)}")
                usage_results = []
            
            # Final combination and ranking
            self._update_progress(90, "Finalizing results")
            final_results = self._combine_and_rank_results(
                combined_results,
                pattern_results,
                expanded_results,
                usage_results,
                context
            )
            
            # Return final results
            self._update_progress(100, "Search complete")
            if callback:
                callback(final_results[:top_k], "final")
                
            return final_results[:top_k]
            
        except Exception as e:
            print(f"Error in progressive search: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _perform_search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Actual search implementation with fallback mechanisms"""
        try:
            # Use progressive search with a collector callback
            results_collector = {'final': []}
            
            def collector_callback(results, stage):
                if stage == 'final':
                    results_collector['final'] = results
            
            # Run progressive search
            self.progressive_search(query, top_k, collector_callback)
            
            # Return the final results
            if results_collector['final']:
                return results_collector['final']
            
            # Fallback to traditional search if progressive search fails
            print("Progressive search returned no results, falling back to traditional search")
            
            # Step 1: Quick semantic search
            self._update_progress(60, "Performing fallback search")
            query_embedding = self._get_cached_embedding(query, self.query_instruction)
            
            # Get initial results using parallel processing
            initial_results = self._quick_semantic_search(query_embedding, top_k)
            
            # Step 2: Detailed search if needed
            self._update_progress(70, "Performing detailed search")
            detailed_results = self._semantic_search(query, {'query_embedding': query_embedding})
            
            # Step 3: Combine and rank results
            self._update_progress(90, "Ranking results")
            results = self._combine_and_rank_results(
                initial_results,
                detailed_results,
                [],
                [],
                {'query_embedding': query_embedding}
            )
            
            self._update_progress(100, "Search complete")
            return results[:top_k]
            
        except Exception as e:
            print(f"Error in search: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
            
    def search(self, query: str, top_k: int = 10) -> List[Dict]:
        """Main search method with caching"""
        return self.cached_search(query, top_k)

    def _quick_semantic_search(self, query_embedding: np.ndarray, top_k: int) -> List[Dict]:
        """Optimized quick semantic search with parallel processing"""
        results = []
        
        # Get file items and create optimal batch size
        file_items = list(self.code_graph.file_contents.items())
        optimal_batch_size = max(1, len(file_items) // (self._num_cores * 2))
        
        # Process files in parallel batches
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # Create batches
            batches = [
                file_items[i:i + optimal_batch_size]
                for i in range(0, len(file_items), optimal_batch_size)
            ]
            
            # Submit batch processing tasks
            future_to_batch = {
                executor.submit(self._process_file_batch, batch, query_embedding): batch
                for batch in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    print(f"Error in batch processing: {str(e)}")
        
        # Sort and return top results
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]

    def _semantic_search(self, query: str, context: Dict) -> List[Dict]:
        """Enhanced semantic search with parallel processing"""
        results = []
        query_embedding = context['query_embedding']
        
        # Get files to process
        files_to_process = context.get('relevant_files', self.code_graph.file_contents.keys())
        file_items = [(f, self.code_graph.file_contents[f]) for f in files_to_process]
        
        # Calculate optimal batch size based on available memory
        available_memory = psutil.virtual_memory().available
        memory_per_file = sum(len(content) for _, content in file_items) / len(file_items)
        optimal_batch_size = max(1, int(available_memory / (memory_per_file * 2)))
        
        # Process files in parallel
        with ThreadPoolExecutor(max_workers=self._max_workers) as executor:
            # Create batches
            batches = [
                file_items[i:i + optimal_batch_size]
                for i in range(0, len(file_items), optimal_batch_size)
            ]
            
            # Submit batch processing tasks
            future_to_batch = {
                executor.submit(self._process_file_batch, batch, query_embedding): batch
                for batch in batches
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_batch):
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    print(f"Error in batch processing: {str(e)}")
        
        return results
    
    def _pattern_based_search(self, query: str, context: Dict) -> List[Dict]:
        """Search based on code patterns and concepts"""
        results = []
        
        # Extract concepts from query if not provided in context
        if 'concepts' not in context:
            # Simple concept extraction - split query into words and filter out common words
            stop_words = set(['the', 'a', 'an', 'in', 'on', 'at', 'for', 'to', 'of', 'with', 'by'])
            concepts = [word.lower() for word in query.split() if word.lower() not in stop_words and len(word) > 2]
            context['concepts'] = concepts
        
        concepts = context['concepts']
        
        for file_path, content in self.code_graph.file_contents.items():
            # Search for concept matches
            for concept in concepts:
                pattern = re.compile(rf'\b{re.escape(concept)}\b', re.IGNORECASE)
                matches = pattern.finditer(content)
                
                for match in matches:
                    # Get surrounding context
                    start = max(0, match.start() - 200)
                    end = min(len(content), match.end() + 200)
                    context = content[start:end]
                    
                    # Find the node in the graph that matches this file path
                    node_metadata = {}
                    for node in self.code_graph.nodes:
                        if node.get('id') == file_path:
                            node_metadata = node.get('metadata', {})
                            break
                    
                    results.append({
                        'type': 'pattern',
                        'content': context,
                        'file_path': file_path,
                        'concept': concept,
                        'metadata': node_metadata
                    })
        
        return results
    
    def _context_aware_expansion(self, initial_results: List[Dict], pattern_results: List[Dict], context: Dict) -> List[Dict]:
        """Expand search results based on context and relationships"""
        expanded = []
        seen = set()
        
        # Combine initial and pattern results
        all_results = initial_results + pattern_results
        
        for result in all_results:
            if result['file_path'] in seen:
                continue
                
            seen.add(result['file_path'])
            
            # Get related nodes
            related = self.code_graph.get_related_nodes(result['file_path'])
            for rel in related:
                if rel['node']['type'] in ['function', 'class', 'method']:
                    # Find the node in the graph that matches this file path
                    node_metadata = {}
                    for node in self.code_graph.nodes:
                        if node.get('id') == rel['node'].get('metadata', {}).get('file_path', ''):
                            node_metadata = node.get('metadata', {})
                            break
                    
                    expanded.append({
                        'type': 'related',
                        'content': rel['node'].get('content', ''),
                        'file_path': rel['node'].get('metadata', {}).get('file_path', ''),
                        'relationship': rel['relationship'],
                        'metadata': node_metadata
                    })
        
        return expanded
    
    def _analyze_usage(self, results: List[Dict], context: Dict) -> List[Dict]:
        """Analyze how code is used in the codebase"""
        usage_results = []
        
        # Check if code_graph has the right structure
        if not hasattr(self.code_graph, 'edges') or not isinstance(self.code_graph.edges, dict):
            print("Warning: Code graph edges not properly initialized, skipping usage analysis")
            return usage_results
            
        if not hasattr(self.code_graph, 'nodes') or not isinstance(self.code_graph.nodes, list):
            print("Warning: Code graph nodes not properly initialized, skipping usage analysis")
            return usage_results
        
        for result in results:
            if 'file_path' not in result:
                continue
                
            # Find where this code is used
            for node in self.code_graph.nodes:
                if not isinstance(node, dict):
                    continue
                    
                node_id = node.get('id', '')
                if not node_id or node_id not in self.code_graph.edges:
                    continue
                    
                for edge in self.code_graph.edges.get(node_id, []):
                    if not isinstance(edge, dict):
                        continue
                        
                    if edge.get('to') == result['file_path']:
                        # Find the node metadata
                        node_metadata = {}
                        for n in self.code_graph.nodes:
                            if isinstance(n, dict) and n.get('id') == node_id:
                                node_metadata = n.get('metadata', {})
                                break
                        
                        usage_results.append({
                            'type': 'usage',
                            'content': node.get('content', ''),
                            'file_path': node.get('id', ''),
                            'usage_type': edge['type'],
                            'metadata': node_metadata
                        })
        
        return usage_results
    
    def _combine_and_rank_results(
        self,
        semantic_results: List[Dict],
        pattern_results: List[Dict],
        expanded_results: List[Dict],
        usage_results: List[Dict],
        context: Dict
    ) -> List[Dict]:
        """Combine and rank all results based on relevance and context"""
        # Combine all results
        all_results = []
        
        # Add semantic results with highest weight
        for result in semantic_results:
            all_results.append({
                **result,
                'weight': result.get('similarity', 0.8)
            })
        
        # Add pattern results with medium-high weight
        for result in pattern_results:
            all_results.append({
                **result,
                'weight': 0.7
            })
        
        # Add expanded results with medium weight
        for result in expanded_results:
            all_results.append({
                **result,
                'weight': 0.6
            })
        
        # Add usage results with lower weight
        for result in usage_results:
            all_results.append({
                **result,
                'weight': 0.5
            })
        
        # Apply advanced ranking algorithm
        self._apply_advanced_ranking(all_results, context)
        
        # Remove duplicates and sort by weight
        seen = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x['weight'], reverse=True):
            result_id = f"{result.get('file_path', '')}:{result.get('content', '')[:100]}"
            if result_id not in seen:
                seen.add(result_id)
                unique_results.append(result)
        
        return unique_results
        
    def _apply_advanced_ranking(self, results: List[Dict], context: Dict):
        """Apply advanced ranking algorithm to improve result relevance"""
        query_embedding = context.get('query_embedding')
        
        # Extract query terms if available
        query_terms = set()
        if 'query' in context:
            query_terms = set(context['query'].lower().split())
        
        for result in results:
            # Base weight from similarity
            base_weight = result.get('weight', 0.5)
            
            # 1. Boost based on result type
            type_boost = 0.0
            if result['type'] == 'semantic':
                type_boost = 0.2
            elif result['type'] == 'lexical':
                type_boost = 0.15
            elif result['type'] == 'pattern':
                type_boost = 0.1
            
            # 2. Boost based on file importance
            file_boost = 0.0
            file_path = result.get('file_path', '').lower()
            
            # Prioritize important files
            if 'readme' in file_path or 'documentation' in file_path:
                file_boost += 0.15
            elif 'test' in file_path:
                file_boost += 0.05  # Tests are useful for understanding usage
            elif 'example' in file_path:
                file_boost += 0.1
            
            # Deprioritize less relevant files
            if 'node_modules' in file_path or 'vendor' in file_path:
                file_boost -= 0.2
            elif 'dist' in file_path or 'build' in file_path:
                file_boost -= 0.15
                
            # 3. Boost based on content quality
            content_boost = 0.0
            content = result.get('content', '').lower()
            
            # Prioritize code with comments
            if '/**' in content or '*/' in content or '#' in content:
                content_boost += 0.1
            
            # Prioritize complete functions/classes
            if ('function' in content and 'return' in content) or ('class' in content and 'constructor' in content):
                content_boost += 0.1
                
            # 4. Boost based on query intent match
            intent_boost = 0.0
            if context.get('query_intent', {}).get('is_technical', False) and result['type'] in ['semantic', 'pattern']:
                intent_boost += 0.1
            
            if context.get('query_intent', {}).get('is_usage', False) and result['type'] == 'usage':
                intent_boost += 0.15
            
            if context.get('query_intent', {}).get('is_implementation', False) and result['type'] in ['semantic', 'related']:
                intent_boost += 0.15
                
            # 5. Term frequency boost
            term_boost = 0.0
            if query_terms:
                content_words = set(content.split())
                matching_terms = query_terms.intersection(content_words)
                if matching_terms:
                    term_boost = len(matching_terms) / len(query_terms) * 0.2
            
            # Combine all boosts
            total_boost = type_boost + file_boost + content_boost + intent_boost + term_boost
            
            # Apply boost with diminishing returns to avoid extreme values
            result['weight'] = base_weight * (1 + min(total_boost, 0.5))
            
            # Store individual boost factors for debugging
            result['boost_factors'] = {
                'type_boost': type_boost,
                'file_boost': file_boost,
                'content_boost': content_boost,
                'intent_boost': intent_boost,
                'term_boost': term_boost,
                'total_boost': total_boost
            }