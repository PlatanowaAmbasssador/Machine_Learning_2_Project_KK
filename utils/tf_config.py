"""
TensorFlow configuration and optimization utilities.
Helps reduce first epoch overhead by pre-initializing TensorFlow.
"""

import os
import tensorflow as tf
from tensorflow import keras


def configure_tensorflow(minimal=True):
    """
    Configure TensorFlow for optimal performance.
    Reduces first epoch overhead by pre-initializing GPU/CPU.
    
    Parameters:
    -----------
    minimal : bool
        If True, only do essential configuration (GPU memory growth, warmup).
        If False, enable additional optimizations (mixed precision, threading).
    """
    # Set memory growth to avoid allocating all GPU memory at once
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Configured {len(gpus)} GPU(s) with memory growth")
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    
    if not minimal:
        # Set mixed precision for faster training (if supported)
        # NOTE: Can cause slowdowns on some systems, disabled by default
        try:
            policy = keras.mixed_precision.Policy('mixed_float16')
            keras.mixed_precision.set_global_policy(policy)
            print("✓ Enabled mixed precision training")
        except:
            # Mixed precision not available, continue without it
            pass
        
        # Set thread count for CPU
        if not gpus:
            # Use all available CPU cores
            tf.config.threading.set_inter_op_parallelism_threads(0)  # 0 = use all
            tf.config.threading.set_intra_op_parallelism_threads(0)  # 0 = use all
            print("✓ Configured CPU threading")
    
    # Optimize TensorFlow for performance
    # NOTE: XLA JIT compilation disabled - causes issues with bidirectional LSTM models
    # (creates cycles in computation graph that XLA cannot handle)
    
    # Minimal warmup - just force TensorFlow initialization
    # This is fast and helps reduce first epoch overhead
    try:
        _ = tf.constant(1.0)
    except:
        pass
    
    return len(gpus) > 0




def clear_tf_session():
    """
    Clear TensorFlow session and free memory.
    Useful between folds to prevent memory buildup.
    """
    keras.backend.clear_session()
    import gc
    gc.collect()

