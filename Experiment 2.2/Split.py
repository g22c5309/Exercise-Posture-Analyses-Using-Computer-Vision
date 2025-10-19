import os
import random
import shutil

def simple_split(data_dir, output_dir, train_size=0.7, val_size=0.2, test_size=0.1):
    """Simple split without sklearn dependencies - train:val:test"""
    
    # Validate that ratios sum to 1.0
    assert abs(train_size + val_size + test_size - 1.0) < 0.001, "Ratios must sum to 1.0"
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    for category in os.listdir(data_dir):
        category_path = os.path.join(data_dir, category)
        if not os.path.isdir(category_path):
            continue
            
        # Get all files
        all_files = [f for f in os.listdir(category_path) if f.endswith('.npz')]
        
        if not all_files:
            continue
            
        # Shuffle and split manually
        random.shuffle(all_files)
        
        # Calculate split indices
        train_idx = int(len(all_files) * train_size)
        val_idx = train_idx + int(len(all_files) * val_size)
        
        # Split files
        train_files = all_files[:train_idx]
        val_files = all_files[train_idx:val_idx]
        test_files = all_files[val_idx:]
        
        # Create category directories
        train_category_dir = os.path.join(train_dir, category)
        val_category_dir = os.path.join(val_dir, category)
        test_category_dir = os.path.join(test_dir, category)
        os.makedirs(train_category_dir, exist_ok=True)
        os.makedirs(val_category_dir, exist_ok=True)
        os.makedirs(test_category_dir, exist_ok=True)
        
        # Copy files
        for file in train_files:
            shutil.copy2(os.path.join(category_path, file), 
                        os.path.join(train_category_dir, file))
        
        for file in val_files:
            shutil.copy2(os.path.join(category_path, file), 
                        os.path.join(val_category_dir, file))
        
        for file in test_files:
            shutil.copy2(os.path.join(category_path, file), 
                        os.path.join(test_category_dir, file))
        
        print(f"{category}: {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")

# Usage
simple_split("Workout_npz", "AR val Split", train_size=0.7, val_size=0.15, test_size=0.15)