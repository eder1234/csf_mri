import os
import random
import numpy as np

# Directories to search
base_dirs = ['data/test', 'data/train']

# Get all subfolders in both directories
all_folders = []
for base in base_dirs:
    if os.path.exists(base):
        for entry in os.listdir(base):
            folder_path = os.path.join(base, entry)
            if os.path.isdir(folder_path):
                all_folders.append(folder_path)

if not all_folders:
    print('No folders found in data/test or data/train.')
    exit(1)

# Randomly select a folder
selected_folder = random.choice(all_folders)
print(f'Analyzed folder: {selected_folder}')

# List of numpy files to inspect
npy_files = ['mag.npy', 'mask.npy', 'phase.npy']

for npy_file in npy_files:
    file_path = os.path.join(selected_folder, npy_file)
    if os.path.exists(file_path):
        arr = np.load(file_path)
        print(f'{npy_file}: shape {arr.shape}')
    else:
        print(f'{npy_file}: file not found')
