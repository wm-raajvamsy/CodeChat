"""
Error handling utilities for the CodeChat backend
"""

import traceback
import logging
import functools
import json
from typing import Callable, Any, Dict, Optional
from flask import jsonify, Response

def api_error_handler(func: Callable) -> Callable:
    """
    Decorator for API endpoints to standardize error handling
    
    Args:
        func: The API endpoint function to decorate
        
    Returns:
        Decorated function with standardized error handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Response:
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the error
            logging.error(f"API Error in {func.__name__}: {str(e)}")
            logging.error(traceback.format_exc())
            
            # Return standardized error response
            return jsonify({
                'error': str(e),
                'status': 'error',
                'endpoint': func.__name__
            }), 500
    
    return wrapper

def background_task_error_handler(update_status_func: Optional[Callable] = None) -> Callable:
    """
    Decorator for background tasks to standardize error handling
    
    Args:
        update_status_func: Optional function to call to update status on error
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Log the error
                logging.error(f"Background task error in {func.__name__}: {str(e)}")
                logging.error(traceback.format_exc())
                
                # Update status if provided
                if update_status_func and len(args) > 0:
                    # Assume first argument is the entity ID
                    entity_id = args[0]
                    update_status_func(entity_id, 'error', f"Error: {str(e)}")
                
                # Return None to indicate failure
                return None
        
        return wrapper
    
    return decorator

def log_exception(e: Exception, context: str = "") -> Dict[str, str]:
    """
    Log an exception and return a standardized error dict
    
    Args:
        e: The exception to log
        context: Additional context information
        
    Returns:
        Standardized error dictionary
    """
    error_msg = f"{context}: {str(e)}" if context else str(e)
    logging.error(error_msg)
    logging.error(traceback.format_exc())
    
    return {
        'error': str(e),
        'traceback': traceback.format_exc(),
        'context': context
    }