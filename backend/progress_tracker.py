"""
Progress tracking utilities for long-running operations
"""

import time
import threading
from typing import Dict, Any, Callable

class ProgressTracker:
    """
    A class to track progress of long-running operations with optimized persistence
    """
    
    def __init__(self, save_callback: Callable, min_save_interval: float = 2.0):
        """
        Initialize the progress tracker
        
        Args:
            save_callback: Function to call to save progress
            min_save_interval: Minimum time between saves in seconds
        """
        self.save_callback = save_callback
        self.min_save_interval = min_save_interval
        self.last_save_time = 0
        self.progress_data = {}
        self.lock = threading.Lock()
        self.pending_save = False
        self.save_timer = None
    
    def update_progress(self, entity_id: str, progress: int, operation: str, 
                        additional_data: Dict[str, Any] = None, force_save: bool = False):
        """
        Update progress for an entity
        
        Args:
            entity_id: ID of the entity being processed
            progress: Progress percentage (0-100)
            operation: Current operation description
            additional_data: Additional data to store
            force_save: Force saving even if min_save_interval hasn't elapsed
        """
        with self.lock:
            # Update progress data
            if entity_id not in self.progress_data:
                self.progress_data[entity_id] = {}
                
            self.progress_data[entity_id]['progress'] = progress
            self.progress_data[entity_id]['current_operation'] = operation
            
            if additional_data:
                for key, value in additional_data.items():
                    self.progress_data[entity_id][key] = value
            
            # Check if we should save
            current_time = time.time()
            time_since_last_save = current_time - self.last_save_time
            
            if force_save or time_since_last_save >= self.min_save_interval:
                self._save_progress_now()
            else:
                # Schedule a save if one isn't already pending
                if not self.pending_save:
                    self.pending_save = True
                    delay = self.min_save_interval - time_since_last_save
                    
                    # Cancel any existing timer
                    if self.save_timer:
                        self.save_timer.cancel()
                        
                    # Schedule a new save
                    self.save_timer = threading.Timer(delay, self._save_progress_now)
                    self.save_timer.daemon = True
                    self.save_timer.start()
    
    def _save_progress_now(self):
        """Save progress data immediately"""
        with self.lock:
            self.save_callback(self.progress_data)
            self.last_save_time = time.time()
            self.pending_save = False
    
    def get_progress(self, entity_id: str) -> Dict[str, Any]:
        """
        Get progress for an entity
        
        Args:
            entity_id: ID of the entity
            
        Returns:
            Progress data for the entity or empty dict if not found
        """
        with self.lock:
            return self.progress_data.get(entity_id, {})
    
    def clear_progress(self, entity_id: str):
        """
        Clear progress for an entity
        
        Args:
            entity_id: ID of the entity
        """
        with self.lock:
            if entity_id in self.progress_data:
                del self.progress_data[entity_id]
                self._save_progress_now()
    
    def cleanup(self):
        """Clean up resources"""
        if self.save_timer:
            self.save_timer.cancel()
            
        # Final save
        self._save_progress_now()