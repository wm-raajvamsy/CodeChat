// services/errorHandlingService.js

/**
 * Centralized error handling for API calls
 * 
 * @param {Error} error - The error object from axios
 * @param {string} defaultMessage - Default message if error details are unavailable
 * @returns {Error} Enhanced error object with user-friendly message
 */
export const handleApiError = (error, defaultMessage = 'An error occurred') => {
    // Create enhanced error object
    const enhancedError = new Error();
    
    // Set default properties
    enhancedError.originalError = error;
    enhancedError.isApiError = true;
    
    // Handle Axios specific errors
    if (error.response) {
      // Server responded with error status
      const status = error.response.status;
      const responseData = error.response.data;
      
      enhancedError.status = status;
      enhancedError.statusText = error.response.statusText;
  
      // Extract error message from response if available
      if (responseData && responseData.message) {
        enhancedError.message = responseData.message;
      } else if (responseData && responseData.error) {
        enhancedError.message = responseData.error;
      } else {
        enhancedError.message = `${defaultMessage} (${status})`;
      }
  
      // Handle specific status codes
      if (status === 401) {
        enhancedError.type = 'AuthError';
        enhancedError.message = 'Authentication required. Please log in again.';
        
        // Clear auth token and redirect to login
        localStorage.removeItem('authToken');
        localStorage.removeItem('user');
        
        // Only redirect if we're in a browser context
        if (typeof window !== 'undefined') {
          // Slight delay to allow the error to be shown
          setTimeout(() => {
            window.location.reload();
          }, 2000);
        }
      } else if (status === 403) {
        enhancedError.type = 'ForbiddenError';
        enhancedError.message = 'You don\'t have permission to access this resource.';
      } else if (status === 404) {
        enhancedError.type = 'NotFoundError';
        enhancedError.message = 'The requested resource was not found.';
      } else if (status === 429) {
        enhancedError.type = 'RateLimitError';
        enhancedError.message = 'Too many requests. Please try again later.';
      } else if (status >= 500) {
        enhancedError.type = 'ServerError';
        enhancedError.message = 'Server error. Please try again later.';
      }
    } else if (error.request) {
      // Request was made but no response received
      enhancedError.type = 'NetworkError';
      enhancedError.message = 'Network error. Please check your connection or try again later.';
      enhancedError.request = error.request;
    } else {
      // Other errors
      enhancedError.type = 'UnknownError';
      enhancedError.message = error.message || defaultMessage;
    }
  
    // Log the error for debugging (consider removing in production)
    console.error('API Error:', {
      message: enhancedError.message,
      type: enhancedError.type,
      status: enhancedError.status,
      originalError: error
    });
  
    return enhancedError;
  };
  
  /**
   * Unified error display component for rendering error states
   * 
   * @param {Error} error - Error object to display
   * @returns {Object} Object with error type, message, and suggested action
   */
  export const getErrorDisplayInfo = (error) => {
    // Default error information
    let errorInfo = {
      title: 'Error',
      message: 'An unexpected error occurred',
      action: 'Please try again later',
      severity: 'error', // 'warning', 'error', 'critical'
      retryable: true
    };
  
    // Customize based on error type
    if (error.isApiError) {
      switch (error.type) {
        case 'AuthError':
          errorInfo = {
            title: 'Authentication Error',
            message: error.message,
            action: 'Please log in again',
            severity: 'warning',
            retryable: false
          };
          break;
        case 'ForbiddenError':
          errorInfo = {
            title: 'Access Denied',
            message: error.message,
            action: 'Contact support if you believe this is an error',
            severity: 'warning',
            retryable: false
          };
          break;
        case 'NotFoundError':
          errorInfo = {
            title: 'Not Found',
            message: error.message,
            action: 'Check the resource ID or path',
            severity: 'warning',
            retryable: false
          };
          break;
        case 'RateLimitError':
          errorInfo = {
            title: 'Rate Limit Exceeded',
            message: error.message,
            action: 'Please wait before trying again',
            severity: 'warning',
            retryable: true
          };
          break;
        case 'NetworkError':
          errorInfo = {
            title: 'Network Error',
            message: 'Unable to connect to the server',
            action: 'Check your internet connection and try again',
            severity: 'error',
            retryable: true
          };
          break;
        case 'ServerError':
          errorInfo = {
            title: 'Server Error',
            message: 'The server encountered an error',
            action: 'Please try again later',
            severity: 'error',
            retryable: true
          };
          break;
        default:
          errorInfo = {
            title: 'Error',
            message: error.message,
            action: 'Please try again',
            severity: 'error',
            retryable: true
          };
      }
    } else if (error.name === 'SyntaxError') {
      errorInfo = {
        title: 'Invalid Response',
        message: 'The server returned an invalid response',
        action: 'Please try again or contact support',
        severity: 'error',
        retryable: true
      };
    }
  
    return errorInfo;
  };
  
  /**
   * Reports errors to a monitoring service (placeholder)
   * 
   * @param {Error} error - The error to report
   * @param {Object} context - Additional context about the error
   */
  export const reportError = (error, context = {}) => {
    // In a real application, this would send the error to a service like Sentry
    // For now, just log to console with context
    console.error('Error Report:', {
      error,
      context,
      timestamp: new Date().toISOString(),
      // Add user info if available
      user: localStorage.getItem('user') ? JSON.parse(localStorage.getItem('user')).email : 'anonymous'
    });
  };