"""
Output logging utility to capture terminal output for PDF export.
Supports checkpoint/resume to combine logs from multiple runs.
"""

import os
import sys
from datetime import datetime
from typing import Optional
from pathlib import Path


class OutputLogger:
    """
    Logger that captures stdout and stderr to a file.
    Can append to existing log files for checkpoint resume.
    """
    
    def __init__(self, log_file: str, append: bool = False):
        """
        Initialize output logger.
        
        Parameters:
        -----------
        log_file : str
            Path to log file
        append : bool
            If True, append to existing file. If False, overwrite.
        """
        self.log_file = log_file
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        self.log_file_handle = None
        self.append = append
        
        # Create directory if needed
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
    
    def start(self):
        """Start logging - redirect stdout and stderr to file."""
        mode = 'a' if self.append else 'w'
        self.log_file_handle = open(self.log_file, mode, encoding='utf-8')
        
        # Create a class that writes to both file and original stdout
        class TeeOutput:
            def __init__(self, file_handle, original_stream, log_file_path):
                self.file = file_handle
                self.original = original_stream
                self.buffer = getattr(original_stream, 'buffer', None)
                self.name = log_file_path  # For checkpoint detection
                self.log_file = log_file_path  # Alternative attribute name
            
            def write(self, text):
                # Write to file
                self.file.write(text)
                self.file.flush()
                # Also write to original stream (terminal)
                self.original.write(text)
                self.original.flush()
            
            def flush(self):
                self.file.flush()
                self.original.flush()
        
        # Redirect stdout and stderr
        sys.stdout = TeeOutput(self.log_file_handle, self.original_stdout, self.log_file)
        sys.stderr = TeeOutput(self.log_file_handle, self.original_stderr, self.log_file)
        
        if self.append:
            print("\n" + "="*80)
            print(f"RESUMING EXECUTION - Logging continues from previous run")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80 + "\n")
        else:
            print("="*80)
            print(f"OUTPUT LOGGING STARTED")
            print(f"Log file: {self.log_file}")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80 + "\n")
    
    def stop(self):
        """Stop logging - restore original stdout and stderr."""
        if self.log_file_handle:
            print("\n" + "="*80)
            print(f"OUTPUT LOGGING ENDED")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("="*80 + "\n")
            
            self.log_file_handle.close()
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def get_log_file_path(checkpoint_dir: str, model_type: str, problem_type: str = 'regression') -> str:
    """
    Get log file path based on checkpoint directory.
    
    Parameters:
    -----------
    checkpoint_dir : str
        Checkpoint directory path
    model_type : str
        Model type (e.g., 'LSTM', 'CNN', 'ANN')
    problem_type : str
        Problem type ('regression' or 'classification')
        
    Returns:
    --------
    log_file : str
        Path to log file
    """
    # Extract base directory from checkpoint_dir
    # checkpoint_dir is like: checkpoints/regression/lstm or checkpoints/classification/lstm
    log_dir = os.path.join(os.path.dirname(checkpoint_dir), 'logs', problem_type, model_type.lower())
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{model_type.lower()}_output.log")
    return log_file


def check_existing_log(log_file: str) -> bool:
    """
    Check if log file exists and has content.
    
    Parameters:
    -----------
    log_file : str
        Path to log file
        
    Returns:
    --------
    exists : bool
        True if log file exists and has content
    """
    return os.path.exists(log_file) and os.path.getsize(log_file) > 0

