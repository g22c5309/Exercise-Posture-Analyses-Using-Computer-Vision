'''This file takes a pickle file and converts it into a numpy array file'''
#######################################################################################
import pickle 
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

with open('data_3D.pickle', 'rb')as file :
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
    10: 'correct' # As there is a miss labeling in the dataset
}

# Creating DataFrame
df = pd.DataFrame({
    'act': labels[:, 0],    # Exercise name
    'sub': labels[:, 1],    # Subject ID
    'lab': labels[:, 2],    # Error code
    'rep': labels[:, 3],    # Repetition number
    'frame': labels[:, 4],  # Frame number
    'pose': list(poses)     # Pose data
})

#-------------------------------------------------------------------------------------------#

# Convert lab to integers
try:
    df['lab'] = df['lab'].astype(int)
    df['error_name'] = df['lab'].map(ERROR_LABELS)
except ValueError as e:
    print(f"Conversion error: {e}\nUnique lab values:", df['lab'].unique())

# Adding error (incorrect posture) descriptions
df['error_name'] = df['lab'].map(ERROR_LABELS)

# Mapping all exercises to three categories
def categorize_exercise(exercise_name):
    exercise_name = exercise_name.lower()
    if 'lunge' in exercise_name:
        return 'Lunges'
    elif 'squat' in exercise_name:
        return 'Squat'
    elif 'plank' in exercise_name:
        return 'Plank'
    else:
        return exercise_name  # or handle other exercises as needed

# Apply categorization
df['exercise_category'] = df['act'].apply(categorize_exercise)

output_dir = "Dataset" # Name of output file
os.makedirs(output_dir, exist_ok=True)

# Group by the categorized exercise AND original act name to handle repeating names
grouped = df.groupby(['exercise_category', 'act', 'error_name', 'sub', 'rep'])

for (exercise_category, original_act, error_name, subject, rep), group in tqdm(grouped, desc="Processing trials"):
    exercise_dir = os.path.join(output_dir, exercise_category) # Create directory for the exercise category only
    os.makedirs(exercise_dir, exist_ok=True)
    
    # Sort frames
    group = group.sort_values('frame')
    
    poses = np.stack(group['pose'].values)  # (T, C, V)

    #------------------------------- Padding and Uniformly Sampling ----------------------------------#
    # Padding - St-gcn expects the number of frames in each set to be the same
    T, V, C = poses.shape

    desired_T = 32  # Desired number of frames
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
    poses = poses.transpose(1, 0, 2) # Changing the Shape
    #-------------------------------------------------------------------------------------------------#
    
    # Save directly in the exercise directory
    trial_id = f"{subject}_rep{rep}"
    
    # Create a filename that includes the original exercise name and error type
    safe_original_act = original_act.replace(' ', '_').replace('/', '_').replace('\\', '_')
    safe_error_name = error_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    filename = f"{safe_original_act}_{safe_error_name}_{trial_id}.npz"
    
    np.savez(
        os.path.join(exercise_dir, filename),
        data=poses,
        label=exercise_category,  # Use the category as label
        original_exercise=original_act,  # Store original name as metadata
        error_type=error_name  # Store error type as metadata
    )