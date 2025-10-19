from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import os
from ST_GCN_1 import ST_GCN
import torch.nn as nn
from sklearn.metrics import classification_report
from torch.utils.tensorboard import SummaryWriter  
from datetime import datetime  
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import io
import torchvision.transforms as transforms
from PIL import Image
import random

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set seed before anything else
set_seed(42)

class PoseDataset(Dataset):
    def __init__(self, npz_folder, class_to_idx, target_frames=30):
        self.data = []
        self.labels = []
        self.target_frames = target_frames
        self.class_to_idx = class_to_idx
        
        # Verify the folder exists
        if not os.path.exists(npz_folder):
            raise FileNotFoundError(f"Directory {npz_folder} not found")
            
        # Load and process all data
        for class_name in os.listdir(npz_folder):
            if class_name not in class_to_idx:
                continue
                
            class_path = os.path.join(npz_folder, class_name)
            if not os.path.isdir(class_path):
                continue
                
            for npz_file in os.listdir(class_path):
                if not npz_file.endswith('.npz'):
                    continue
                    
                npz_path = os.path.join(class_path, npz_file)
                try:
                    with np.load(npz_path) as f:
                        data = f['data']
                        # Resample to target_frames if needed
                        if data.shape[1] != target_frames:
                            data = self._resample_data(data)
                        self.data.append(data)
                        self.labels.append(class_to_idx[class_name])
                except Exception as e:
                    print(f"Error loading {npz_path}: {str(e)}")
    
    def _resample_data(self, data):
        """Resample data to target_frames using linear interpolation"""
        from scipy.interpolate import interp1d
        original_frames = data.shape[1]
        x_original = np.linspace(0, 1, original_frames)
        x_new = np.linspace(0, 1, self.target_frames)
        
        resampled_data = np.zeros((data.shape[0], self.target_frames, data.shape[2]))
        for c in range(data.shape[0]):
            for v in range(data.shape[2]):
                interp_fn = interp1d(x_original, data[c, :, v], kind='linear')
                resampled_data[c, :, v] = interp_fn(x_new)
        return resampled_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.data[idx])  # [C, T, V]
        x[:3] = (x[:3] - x[:3].mean(dim=(1,2), keepdim=True)) / (x[:3].std(dim=(1,2), keepdim=True) + 1e-8)
        y = torch.LongTensor([self.labels[idx]])
        return x, y    

# Initialize TensorBoard writer
log_dir = os.path.join("runs", datetime.now().strftime("%Y%m%d-%H%M%S"))
writer = SummaryWriter(log_dir)

class_to_idx = {
                'Shoulder Press': 0,
                'Squat': 1
                }

# loaders - ADDED VALIDATION SET
train_dataset = PoseDataset('AR val Split/Train', class_to_idx)
train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)

val_dataset = PoseDataset('AR val Split/Val', class_to_idx)  # NEW: Validation dataset
val_loader = DataLoader(val_dataset, batch_size=12, shuffle=False)

test_dataset = PoseDataset('AR val Split/Test', class_to_idx)
test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ST_GCN(num_classes=len(class_to_idx)).to(device)

#################################################################################
CrossEntropy = nn.CrossEntropyLoss() # Loss
optimiser = torch.optim.AdamW(  # AdamW is better than Adam
    model.parameters(), 
    lr=0.001, 
    weight_decay=0.001,  # regularization increase
    betas=(0.9, 0.999)
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser, 
    mode='min',  # Monitoring loss 
    patience=5,
    factor=0.5,
    verbose=True
)
#####################################################################################
num_epochs = 60

best_accuracy = 0.0
patience = 3
no_improvement = 0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).squeeze(1) 

        optimiser.zero_grad()
        outputs = model(inputs)  # [N, num_classes]
        loss = CrossEntropy(outputs, labels)
        loss.backward()
        optimiser.step()
        
        running_loss += loss.item()
        
        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
    
    # Calculate training metrics
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct_train / total_train
    
    # Write training metrics to TensorBoard
    writer.add_scalar('Loss/train', train_loss, epoch)
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    
    # NEW: Validation phase
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).squeeze(1)

            outputs = model(inputs)
            loss = CrossEntropy(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    
    # Calculate validation metrics
    val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * correct_val / total_val
    
    # Write validation metrics to TensorBoard
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_accuracy, epoch)
    
    # Evaluation on test set
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).squeeze(1)

            outputs = model(inputs)
            loss = CrossEntropy(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate test metrics
    test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct_test / total_test
    
    # Write test metrics to TensorBoard
    writer.add_scalar('Loss/test', test_loss, epoch)
    writer.add_scalar('Accuracy/test', test_accuracy, epoch)
    
    # Save best model based on VALIDATION accuracy (not test)
    if val_accuracy > best_accuracy:  # CHANGED: Using validation accuracy for model selection
        best_accuracy = val_accuracy
        no_improvement = 0
        torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pth'))
        print(f"New best model saved with validation accuracy: {val_accuracy:.2f}%")
    else:
        no_improvement += 1
        if no_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Print epoch summary - ADDED VALIDATION METRICS
    print(f"Epoch {epoch+1}/{num_epochs}: "
          f"\nTrain Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"\nVal Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, "  # NEW: Validation metrics
          f"\nTest Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
    print("-------------------------------------------------------------")
    
    # Update learning rate scheduler based on validation loss
    scheduler.step(val_loss)  # CHANGED: Using validation loss for scheduling

# Close TensorBoard writer
writer.close()

# Final evaluation on test set
print("\nFinal Classification Report:\n")
print(classification_report(all_labels, all_preds, target_names=class_to_idx.keys(), zero_division=0))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_to_idx.keys(), 
            yticklabels=class_to_idx.keys())
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

# Save confusion matrix to TensorBoard
buf = io.BytesIO()
plt.savefig(buf, format='jpeg')
buf.seek(0)
image = Image.open(buf)
image = transforms.ToTensor()(image)
writer.add_image('Confusion Matrix', image)
plt.close()

print(f"\nTensorBoard logs saved at: {log_dir}")
print(f"Best model saved with validation accuracy: {best_accuracy:.2f}%")


##############################################################################################
print("\n=== Evaluating Best Model ===")

# Load the best model
best_model_path = os.path.join(log_dir, 'best_model.pth')
model.load_state_dict(torch.load(best_model_path))
model.eval()

# Evaluate best model on test set
all_preds = []
all_labels = []
all_probs = []
test_loss = 0.0
correct_test = 0
total_test = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device).squeeze(1)

        outputs = model(inputs)
        loss = CrossEntropy(outputs, labels)
        test_loss += loss.item()
        
        probs = torch.nn.functional.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# Calculate metrics for best model
test_loss = test_loss / len(test_loader)
test_accuracy = 100 * correct_test / total_test

print(f"Best Model Test Loss: {test_loss:.4f}")
print(f"Best Model Test Accuracy: {test_accuracy:.2f}%")

# Detailed classification report
print("\n=== Best Model Classification Report ===")
print(classification_report(all_labels, all_preds, 
                           target_names=class_to_idx.keys(), 
                           zero_division=0))

# Per-class accuracy
print("\n=== Per-Class Accuracy ===")
cm = confusion_matrix(all_labels, all_preds)
class_accuracy = cm.diagonal() / cm.sum(axis=1)
for i, class_name in enumerate(class_to_idx.keys()):
    print(f"{class_name}: {class_accuracy[i]:.3f}")

# Confusion matrix for best model
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_to_idx.keys(), 
            yticklabels=class_to_idx.keys())
plt.title(f'Best Model Confusion Matrix (Test Accuracy: {test_accuracy:.2f}%)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.yticks(rotation=0)

# Save confusion matrix
confusion_matrix_path = os.path.join(log_dir, 'best_model_confusion_matrix.png')
plt.savefig(confusion_matrix_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"\nBest model confusion matrix saved at: {confusion_matrix_path}")