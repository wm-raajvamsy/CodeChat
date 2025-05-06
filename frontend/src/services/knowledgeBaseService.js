import axios from 'axios';
import { handleApiError } from './errorHandlingService';

// Base API URL (ending in /api)
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:6146/api';

/**
 * Check if the API is running and ready
 * @returns {Promise<Object>} - { status, model, knowledge_bases, supported_extensions }
 */
export const getApiStatus = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/status`);
    return response.data;
  } catch (error) {
    throw handleApiError(error, 'Failed to fetch API status');
  }
};

/**
 * Fetch all available knowledge bases
 * @returns {Promise<Array>} - Array of knowledge base objects
 */
export const getAllKnowledgeBases = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/knowledge-bases`, {
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('authToken')}`
      }
    });
    return response.data.knowledge_bases;
  } catch (error) {
    throw handleApiError(error, 'Failed to fetch knowledge bases');
  }
};

/**
 * Create a new knowledge base from a Git repository
 *
 * @param {string} name - Name of the knowledge base
 * @param {string} gitUrl - Git repository URL
 * @param {string} [description] - Optional description
 * @returns {Promise<Object>} - The created knowledge base information
 */
export const createKnowledgeBase = async (name, gitUrl, description = '') => {
  try {
    const response = await axios.post(
      `${API_BASE_URL}/knowledge-bases`,
      { name, git_url: gitUrl, description },
      {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`
        }
      }
    );
    return response.data.kb_info;
  } catch (error) {
    throw handleApiError(error, 'Failed to create knowledge base');
  }
};

/**
 * Remove a knowledge base by ID
 *
 * @param {string} id - The knowledge base ID to remove
 * @returns {Promise<Object>} - Response with removal status
 */
export const removeKnowledgeBase = async (id) => {
  try {
    const response = await axios.delete(
      `${API_BASE_URL}/knowledge-bases/${id}`,
      {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`
        }
      }
    );
    return response.data;
  } catch (error) {
    throw handleApiError(error, 'Failed to remove knowledge base');
  }
};

/**
 * Get detailed information about a specific knowledge base
 *
 * @param {string} id - The knowledge base ID
 * @returns {Promise<Object>} - Detailed knowledge base information and metadata
 */
export const getKnowledgeBaseDetails = async (id) => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/knowledge-bases/${id}`,
      {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`
        }
      }
    );
    // { kb_info, metadata }
    return response.data;
  } catch (error) {
    throw handleApiError(error, 'Failed to fetch knowledge base details');
  }
};

/**
 * Start or trigger indexing of a knowledge base
 *
 * @param {string} id - The knowledge base ID to index
 * @returns {Promise<Object>} - Status of indexing operation
 */
export const rebuildKnowledgeBase = async (id) => {
  try {
    const response = await axios.post(
      `${API_BASE_URL}/knowledge-bases/${id}/index`,
      {},
      {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`
        }
      }
    );
    return response.data;
  } catch (error) {
    throw handleApiError(error, 'Failed to rebuild knowledge base');
  }
};

/**
 * Get search progress for a knowledge base
 * 
 * @param {string} id - Knowledge base ID
 * @returns {Promise<Object>} - Search progress information
 */
export const getSearchProgress = async (id) => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/search-progress/${id}`,
      {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`
        }
      }
    );
    return response.data;
  } catch (error) {
    throw handleApiError(error, 'Failed to get search progress');
  }
};

/**
 * Search within a specific knowledge base
 *
 * @param {string} id - Knowledge base ID
 * @param {string} query - Search query
 * @param {number} [topK=5] - Number of results to return
 * @param {Function} [onProgress] - Callback for progress updates
 * @returns {Promise<Array>} - Array of search results
 */
export const searchKnowledgeBase = async (id, query, topK = 5, onProgress) => {
  try {
    // Start progress polling
    let progressInterval;
    if (onProgress) {
      progressInterval = setInterval(async () => {
        try {
          const progress = await getSearchProgress(id);
          onProgress(progress);
          if (progress.status === 'complete' || progress.status === 'error') {
            clearInterval(progressInterval);
          }
        } catch (error) {
          console.error('Error polling search progress:', error);
        }
      }, 5000); // Poll every 5 seconds instead of 500ms
    }

    const response = await axios.post(
      `${API_BASE_URL}/search`,
      { kb_id: id, query, top_k: topK },
      {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`
        }
      }
    );

    // Clear progress polling
    if (progressInterval) {
      clearInterval(progressInterval);
    }

    return response.data.results;
  } catch (error) {
    throw handleApiError(error, 'Failed to search knowledge base');
  }
};

/**
 * Search across multiple knowledge bases
 *
 * @param {string[]} ids - Array of knowledge base IDs
 * @param {string} query - Search query
 * @param {number} [topK=5] - Number of results to return
 * @returns {Promise<Array>} - Array of search results
 */
export const searchCombineKnowledgeBase = async (ids, query, topK = 5) => {
  try {
    const response = await axios.post(
      `${API_BASE_URL}/combineSearch`,
      { kb_id: ids, query, top_k: topK },
      {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`
        }
      }
    );
    return response.data.results;
  } catch (error) {
    throw handleApiError(error, 'Failed to search knowledge bases');
  }
};

/**
 * Search across all active knowledge bases
 *
 * @param {string} query - Search query
 * @param {number} [topK=5] - Number of results to return
 * @returns {Promise<Object>} - { results: Array, knowledge_bases_searched: number }
 */
export const searchAllKnowledgeBases = async (query, topK = 5) => {
  try {
    const response = await axios.post(
      `${API_BASE_URL}/search-all`,
      { query, top_k: topK },
      {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`
        }
      }
    );
    return response.data;
  } catch (error) {
    throw handleApiError(error, 'Failed to search across all knowledge bases');
  }
};

/**
 * Upload a file to a knowledge base for indexing
 *
 * @param {File} file - File object to upload
 * @param {string} kbId - Knowledge base ID
 * @returns {Promise<Object>} - Information about the uploaded file
 */
export const uploadFileToKnowledgeBase = async (file, kbId) => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('kb_id', kbId);

    const response = await axios.post(
      `${API_BASE_URL}/upload`,
      formData,
      {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`,
          'Content-Type': 'multipart/form-data'
        }
      }
    );
    return response.data;
  } catch (error) {
    throw handleApiError(error, 'Failed to upload file to knowledge base');
  }
};

/**
 * Get detailed statistics about a knowledge base
 *
 * @param {string} id - Knowledge base ID
 * @returns {Promise<Object>} - Stats including kb_info and metadata
 */
export const getKnowledgeBaseStats = async (id) => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/knowledge-bases/${id}/stats`,
      {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`
        }
      }
    );
    return response.data;
  } catch (error) {
    throw handleApiError(error, 'Failed to fetch knowledge base stats');
  }
};

/**
 * Query the Ollama LLM for enhanced answers using repository snippets
 *
 * @param {string} id - Knowledge base ID
 * @param {string} query - Question to ask
 * @param {number} [topK=5] - Number of context snippets
 * @param {string} [model='qwen2:14b-instruct'] - Ollama model to use
 * @returns {Promise<Object>} - { response, snippets, model }
 */
export const queryOllama = async (id, query, topK = 5, model = 'qwen2:14b-instruct') => {
  try {
    const response = await axios.post(
      `${API_BASE_URL}/ollama/${id}`,
      { query, top_k: topK, model },
      {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`
        }
      }
    );
    return response.data;
  } catch (error) {
    throw handleApiError(error, 'Failed to query Ollama');
  }
};

/**
 * Get detailed indexing state for a knowledge base
 * @param {string} id - Knowledge base ID
 * @returns {Promise<Object>} - Detailed indexing state information
 */
export const getIndexingState = async (id) => {
  try {
    const response = await axios.get(
      `${API_BASE_URL}/knowledge-bases/${id}/indexing-state`,
      {
        headers: {
          'Authorization': `Bearer ${localStorage.getItem('authToken')}`
        }
      }
    );
    return response.data;
  } catch (error) {
    throw handleApiError(error, 'Failed to fetch indexing state');
  }
};
