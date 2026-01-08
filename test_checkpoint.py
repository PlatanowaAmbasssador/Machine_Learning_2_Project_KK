#!/usr/bin/env python3
"""
Quick test script to verify checkpoint functionality.
"""

import os
import sys

project_root = os.getcwd()
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from walk_forward.checkpoint import load_checkpoint, save_checkpoint

# Test checkpoint directory
checkpoint_dir = "checkpoints/test"

# Check if checkpoint exists
print(f"Checking checkpoint directory: {checkpoint_dir}")
checkpoint = load_checkpoint(checkpoint_dir)

if checkpoint:
    print(f"\n✓ Checkpoint found!")
    print(f"  Last completed fold: {checkpoint['last_completed_fold']}")
    print(f"  Number of results: {len(checkpoint['results'])}")
    print(f"  Number of fold results: {len(checkpoint['fold_results'])}")
    print(f"  Test signals: {len(checkpoint['all_test_signals'])}")
    print(f"  Test predictions: {len(checkpoint['all_test_predictions'])}")
else:
    print(f"\n✗ No checkpoint found at {checkpoint_dir}")
    print(f"  Make sure you've run at least one fold of training")

# List all checkpoint directories
print(f"\n{'='*60}")
print("All checkpoint directories:")
print(f"{'='*60}")
if os.path.exists("checkpoints"):
    for item in os.listdir("checkpoints"):
        item_path = os.path.join("checkpoints", item)
        if os.path.isdir(item_path):
            metadata_path = os.path.join(item_path, "metadata.json")
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
                print(f"  {item}: Fold {metadata.get('last_completed_fold', 'unknown')} completed")
            else:
                print(f"  {item}: (no metadata)")
else:
    print("  No checkpoints directory found")

