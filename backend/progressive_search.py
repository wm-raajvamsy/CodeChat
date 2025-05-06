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
    def __init__(self, code_graph: CodeGraph, instructor_model: INSTRUCTOR):
        self.code_graph = code_graph
        self.instructor_model = instructor_model
        self.query_instruction = "Represent the question for code retrieval:"
        self.code_instruction = "Represent the code snippet for retrieval:"
        self.function_instruction = "Represent the function for retrieval:"
        self.class_instruction = "Represent the class for retrieval:"
        self.progress_callback = None
        
        # Enhanced caching with LRU-like behavior
        self._cached_embeddings = {}
        self._cache_lock = Lock()
        self._cache_size_limit = 1000  # Maximum number of cached embeddings
        self._cache_access_count = defaultdict(int)
        
        # CPU optimization
        self._num_cores = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
        self._max_workers = self._num_cores * 2  # Threads per core
        self._batch_size = 10  # Optimal batch size for memory usage
        
        # Process pool for CPU-intensive tasks
        self._process_pool = ProcessPoolExecutor(max_workers=self._num_cores)
        
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
                show_progress_bar=False
            )[0]
            
            # Manage cache size
            if len(self._cached_embeddings) >= self._cache_size_limit:
                # Remove least accessed items
                sorted_items = sorted(
                    self._cache_access_count.items(),
                    key=lambda x: x[1]
                )
                for key, _ in sorted_items[:100]:  # Remove 100 least accessed items
                    del self._cached_embeddings[key]
                    del self._cache_access_count[key]
            
            # Add to cache
            self._cached_embeddings[cache_key] = embedding
            self._cache_access_count[cache_key] = 1
            
            return embedding

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

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Enhanced parallel search with optimized resource usage"""
        try:
            # Step 1: Quick semantic search
            self._update_progress(60, "Performing initial search")
            query_embedding = self._get_cached_embedding(query, self.query_instruction)
            
            # Get initial results using parallel processing
            initial_results = self._quick_semantic_search(query_embedding, top_k)
            
            # Return early if results are good enough
            if initial_results and all(r['similarity'] > 0.7 for r in initial_results):
                self._update_progress(100, "Search complete")
                return initial_results[:top_k]
            
            # Step 2: Detailed search if needed
            self._update_progress(70, "Performing detailed search")
            detailed_results = self._semantic_search(query, {'query_embedding': query_embedding})
            
            # Step 3: Combine and rank results
            self._update_progress(90, "Ranking results")
            final_results = self._combine_and_rank_results(
                initial_results,
                detailed_results,
                [],
                [],
                {'query_embedding': query_embedding}
            )
            
            self._update_progress(100, "Search complete")
            return final_results[:top_k]
            
        except Exception as e:
            print(f"Error in search: {str(e)}")
            return []

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
        
        for result in results:
            if 'file_path' not in result:
                continue
                
            # Find where this code is used
            for node in self.code_graph.nodes:
                for edge in self.code_graph.edges.get(node.get('id', ''), []):
                    if edge['to'] == result['file_path']:
                        # Find the node metadata
                        node_metadata = {}
                        for n in self.code_graph.nodes:
                            if n.get('id') == node.get('id'):
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
        
        # Apply context-based boosting
        for result in all_results:
            # Boost results that match query intent
            if context['query_intent']['is_technical'] and result['type'] in ['semantic', 'pattern']:
                result['weight'] *= 1.2
            
            if context['query_intent']['is_usage'] and result['type'] == 'usage':
                result['weight'] *= 1.2
            
            if context['query_intent']['is_implementation'] and result['type'] in ['semantic', 'related']:
                result['weight'] *= 1.2
        
        # Remove duplicates and sort by weight
        seen = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x['weight'], reverse=True):
            result_id = f"{result.get('file_path', '')}:{result.get('content', '')[:100]}"
            if result_id not in seen:
                seen.add(result_id)
                unique_results.append(result)
        
        return unique_results 