import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def split_npz_dataset(raw_data_folder, output_folder, test_ratio=0.2):
    classes = [c for c in os.listdir(raw_data_folder) if not c.startswith('.')] 

    for class_name in classes:
        class_path = os.path.join(raw_data_folder, class_name)
        files = [f for f in os.listdir(class_path) if f.endswith('.npz') and not f.startswith('.')]  # Skip hidden files

        if len(files) == 0:
            print(f" Skipping '{class_name}' (no .npz files found)")
            continue

        # Safe split even with 1 file
        if len(files) < 2:
            train_files = files
            test_files = []
        else:
            train_files, test_files = train_test_split(files, test_size=test_ratio, random_state=42)

        # Copy train files
        train_dir = os.path.join(output_folder, 'train', class_name)
        os.makedirs(train_dir, exist_ok=True)
        for file in train_files:
            shutil.copy2(os.path.join(class_path, file), os.path.join(train_dir, file))

        # Copy test files
        test_dir = os.path.join(output_folder, 'test', class_name)
        os.makedirs(test_dir, exist_ok=True)
        for file in test_files:
            shutil.copy2(os.path.join(class_path, file), os.path.join(test_dir, file))

        print(f"{class_name}: {len(train_files)} train, {len(test_files)} test")

    print("\n Dataset split complete!")

split_npz_dataset(raw_data_folder='Experiment 3.1/Dataset', output_folder='Dataset Split', test_ratio=0.2)