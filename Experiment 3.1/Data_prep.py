'''This file takes a pickle file and converts it into a numpy array file'''
#######################################################################################
import pickle 
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

with open('Experiment 3.1/data_3D.pickle', 'rb') as file :
    data = pickle.load(file)


labels = data['labels']
poses = data['poses']
print(labels)


ERROR_LABELS = {
    1: "correct",  
    2: "Feet too wide",
    3: "Knees inward",
    4: "Not low enough",
    5: 'Front bent', 
    6: 'Knee passes toe',
    7: 'Arched back',
    8: 'Hunch back',
    10: 'correct'
}


# Create DataFrame
df = pd.DataFrame({
    'act': labels[:, 0],    # Exercise name
    'sub': labels[:, 1],    # Subject ID
    'lab': labels[:, 2],    # Error code (1-8)
    'rep': labels[:, 3],    # Repetition number
    'frame': labels[:, 4],  # Frame number
    'pose': list(poses)     # Pose data
})


# Convert lab to integers
try:
    df['lab'] = df['lab'].astype(int)
    df['error_name'] = df['lab'].map(ERROR_LABELS)
except ValueError as e:
    print(f"Conversion error: {e}\nUnique lab values:", df['lab'].unique())

# Add error descriptions
df['error_name'] = df['lab'].map(ERROR_LABELS)
df

#-------------------------------------------------------------------------------------------#

# Assuming you have a DataFrame 'df' with the necessary columns
output_dir = "Dataset"
os.makedirs(output_dir, exist_ok=True)

grouped = df.groupby(['act', 'error_name', 'sub', 'rep'])

for (exercise, error_name, subject, rep), group in tqdm(grouped, desc = "Processing trials"):
    # Create directory name by combining exercise and error type
    if error_name.lower() == 'correct':
        class_name = f"{exercise}_correct"
    else:
        # Remove any spaces and make it lowercase for consistency
        error_clean = error_name.replace(' ', '_').lower()
        class_name = f"{exercise}_{error_clean}"
    
    # Create the class directory
    class_dir = os.path.join(output_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # Sort frames
    group = group.sort_values('frame')
    
    poses = np.stack(group['pose'].values)  # (T, C, V)

    # Padding - St-gcn expects the number of frames in each set to be the same
    T, V, C = poses.shape
    desired_T = 30  # Desired number of frames
    if T > desired_T:
        start_idx = (T - desired_T) // 2
        poses = poses[start_idx:start_idx + desired_T]
    elif T < desired_T:
        pad_before = (desired_T - T) // 2
        pad_after = desired_T - T - pad_before
        poses = np.concatenate([
            np.repeat(poses[:1], pad_before, axis=0),  # Repeat first frame
            poses,
            np.repeat(poses[-1:], pad_after, axis=0)   # Repeat last frame
        ], axis=0)
    
    # Transpose to (C, T, V)
    poses = poses.transpose(1, 0, 2)  # (C, T, V)
    
    # Save with descriptive filename including error type
    trial_id = f"{exercise}_{error_name.replace(' ', '_').lower()}_{subject}_rep{rep}"
    np.savez(
        os.path.join(class_dir, f"{trial_id}.npz"),
        data=poses,
        label=class_name,  # Use the combined class name as label
    )
################################################################################################