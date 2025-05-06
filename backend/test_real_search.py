import sys
import os
from progressive_search import ProgressiveSearch
from code_analysis import CodeGraph
from InstructorEmbedding import INSTRUCTOR
import json

def test_real_search():
    # Get the absolute path to the backend directory
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the real code graph from the knowledge base
    kb_id = "kb_1746507143"
    kb_path = os.path.join(backend_dir, "data", kb_id, "code_graph.json")
    
    print(f"Looking for code graph at: {kb_path}")
    
    with open(kb_path, 'r') as f:
        code_graph_data = json.load(f)
    
    # Create CodeGraph instance and populate it
    code_graph = CodeGraph()
    code_graph.nodes = code_graph_data.get('nodes', {})
    code_graph.edges = code_graph_data.get('edges', {})
    code_graph.file_contents = code_graph_data.get('file_contents', {})
    
    # Initialize the instructor model
    instructor_model = INSTRUCTOR('hkunlp/instructor-large')
    
    # Initialize search engine with real data
    search_engine = ProgressiveSearch(code_graph, instructor_model)
    
    # Test payload
    query = "what happens if make localresource true in expo-web-preview? which files will be added?"
    
    # Print debug information
    print("\nQuery Understanding:")
    context = search_engine.understand_context(query)
    print(f"Query Type: {context['query_type']}")
    print(f"Query Intent: {context['query_intent']}")
    print(f"Extracted Concepts: {context['concepts']}")
    print(f"Relevant Files: {context['relevant_files']}")
    
    # Perform the search
    results = search_engine.search(query, top_k=5)
    
    # Print results
    print("\nSearch Results:")
    for i, result in enumerate(results):
        print(f"\nResult {i+1}:")
        print(f"File: {result['file_path']}")
        print(f"Type: {result['type']}")
        print(f"Content:\n{result['content']}")
        print(f"Weight: {result['weight']}")
        if 'similarity' in result:
            print(f"Similarity: {result['similarity']}")
        if 'concept' in result:
            print(f"Matching Concept: {result['concept']}")

if __name__ == '__main__':
    test_real_search() 