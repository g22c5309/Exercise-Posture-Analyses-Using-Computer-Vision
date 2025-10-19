import cv2
import numpy as np
import mediapipe as mp
import os
COCO_JOINTS = [
    0,   # Nose
    11, 12,  # Left/Right Shoulders
    13, 14,  # Left/Right Elbows
    15, 16,  # Left/Right Wrists
    23, 24,  # Left/Right Hips
    25, 26,  # Left/Right Knees
    27, 28,  # Left/Right Ankles
    29, 30,  # Left/Right Heels
    31, 32   # Left/Right Toes
]

def extract_pose_tensor(video_path, num_frames=16):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False)
    
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    pose_sequence = []

    for i in range(total_frames):
        success, frame = cap.read()
        if not success:
            break

        if i in frame_indices:
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image_rgb)
            h, w = frame.shape[:2]

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark
                frame_data = [
                    [landmarks[j].x, landmarks[j].y, landmarks[j].z, landmarks[j].visibility]
                    for j in COCO_JOINTS
                ]
            else:
                frame_data = [[0, 0, 0, 0]] * len(COCO_JOINTS)

            pose_sequence.append(frame_data)

    cap.release()
    pose.close()
    
    return np.array(pose_sequence).transpose(2, 0, 1)  # [C, T, V]


def process_videos(input_folder, output_folder, num_frames=16):
    """Processes all videos in class subfolders"""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    for class_name in os.listdir(input_folder):
        class_path = os.path.join(input_folder, class_name)
        if not os.path.isdir(class_path):
            continue
            
        output_class_path = os.path.join(output_folder, class_name)
        os.makedirs(output_class_path, exist_ok=True)
        
        print(f"Processing class: {class_name}")
        for video_file in os.listdir(class_path):
            if video_file.lower().endswith(video_extensions):
                video_path = os.path.join(class_path, video_file)
                video_name = os.path.splitext(video_file)[0]
                pose_tensor = extract_pose_tensor(video_path, num_frames)
                
                np.savez(
                    os.path.join(output_class_path, f"{video_name}.npz"),
                    data=pose_tensor,
                    label=class_name
                )
                print(f"  Saved: {video_name}.npz")

# Example usage
process_videos(input_folder="Exercises", output_folder="Dataset", num_frames=16)