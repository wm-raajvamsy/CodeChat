import unittest
from progressive_search import ProgressiveSearch
from code_analysis import CodeGraph
from InstructorEmbedding import INSTRUCTOR
import os
import json

class TestProgressiveSearch(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize the instructor model
        cls.instructor_model = INSTRUCTOR('hkunlp/instructor-large')
        
        # Create a sample code graph for testing
        cls.code_graph = CodeGraph()
        
        # Add test files to the code graph
        test_files = {
            'expo-web-preview/config.js': '''
                export const config = {
                    localResource: true,
                    // When localResource is true, the following files will be added:
                    // - assets/
                    // - fonts/
                    // - images/
                    // - styles/
                    // - components/
                };
            ''',
            'expo-web-preview/README.md': '''
                # Expo Web Preview
                
                ## Configuration
                
                ### localResource
                When set to true, the following local resources will be included:
                - assets/ - Contains all static assets
                - fonts/ - Custom font files
                - images/ - Image resources
                - styles/ - CSS and style files
                - components/ - Reusable component files
                
                This ensures all necessary files are bundled with the preview.
            ''',
            'expo-web-preview/components/Preview.js': '''
                import { config } from '../config';
                
                export function Preview() {
                    const { localResource } = config;
                    
                    // Handle local resource loading
                    if (localResource) {
                        // Load local resources
                        loadLocalResources();
                    }
                    
                    return (
                        <div>
                            {localResource ? 'Using local resources' : 'Using remote resources'}
                        </div>
                    );
                }
            '''
        }
        
        # Add files to the code graph
        for file_path, content in test_files.items():
            # Add file node
            cls.code_graph.add_node(file_path, 'file', {
                'name': os.path.basename(file_path),
                'path': file_path
            })
            
            # Store file content
            cls.code_graph.file_contents[file_path] = content
            
            # Add edges for imports
            if 'config.js' in file_path:
                cls.code_graph.add_edge('expo-web-preview/components/Preview.js', file_path, 'imports')
        
        # Initialize the search engine
        cls.search_engine = ProgressiveSearch(cls.code_graph, cls.instructor_model)
    
    def test_search_localresource(self):
        """Test searching for information about localResource in expo-web-preview"""
        query = "what happens if make localresource true in expo-web-preview? which files will be added?"
        
        # Print debug information about the query understanding
        print("\nQuery Understanding:")
        context = self.search_engine.understand_context(query)
        print(f"Query Type: {context['query_type']}")
        print(f"Query Intent: {context['query_intent']}")
        print(f"Extracted Concepts: {context['concepts']}")
        print(f"Relevant Files: {context['relevant_files']}")
        
        # Perform the search
        results = self.search_engine.search(query, top_k=5)
        
        # Print all results with full details
        print("\nDetailed Search Results:")
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
        
        # Verify we got results
        self.assertGreater(len(results), 0, "Search should return results")
        
        # Check if we found the relevant information
        found_config = False
        found_readme = False
        found_preview = False
        
        for result in results:
            content = result['content'].lower()
            file_path = result['file_path'].lower()
            
            if 'config.js' in file_path and ('localresource' in content or 'local resource' in content):
                found_config = True
                print("\nFound config.js with localResource")
            elif 'readme.md' in file_path and ('localresource' in content or 'local resource' in content):
                found_readme = True
                print("\nFound README.md with local resource info")
            elif 'preview.js' in file_path and ('localresource' in content or 'local resource' in content):
                found_preview = True
                print("\nFound Preview.js with localResource usage")
        
        # Print what was found and what wasn't
        print("\nSearch Coverage:")
        print(f"Found config.js: {found_config}")
        print(f"Found README.md: {found_readme}")
        print(f"Found Preview.js: {found_preview}")
        
        # Verify we found the key information
        self.assertTrue(found_config, "Should find config file with localResource setting")
        self.assertTrue(found_readme, "Should find README with local resource information")
        self.assertTrue(found_preview, "Should find Preview component using localResource")
        
        # Verify result ranking
        self.assertGreater(
            results[0]['weight'],
            results[-1]['weight'],
            "Results should be ranked by relevance"
        )

if __name__ == '__main__':
    unittest.main() 