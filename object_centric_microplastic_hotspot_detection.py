import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

class MicroplasticHotspotDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Initialize the MicroplasticHotspotDataset.

        Args:
            data_dir (str): Directory containing the microplastic hotspot images and labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.images = []
        self.labels = []
        
        # Load image paths and labels from the data directory
        for filename in os.listdir(data_dir):
            if filename.endswith(".jpg") or filename.endswith(".png"):
                image_path = os.path.join(data_dir, filename)
                label_path = os.path.join(data_dir, filename.split(".")[0] + ".txt")
                
                with open(label_path, "r") as f:
                    label = f.read().strip().split(",")
                    label = [float(coord) for coord in label]
                
                self.images.append(image_path)
                self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image, label) where image is the input image and label is the corresponding bounding box coordinates.
        """
        image_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        
        return image, torch.tensor(label)

class SlotAttentionModel(nn.Module):
    def __init__(self, num_slots, slot_dim, hidden_dim):
        """
        Initialize the Slot Attention model.

        Args:
            num_slots (int): Number of slots (objects) to detect.
            slot_dim (int): Dimension of each slot.
            hidden_dim (int): Dimension of the hidden layers.
        """
        super(SlotAttentionModel, self).__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.hidden_dim = hidden_dim
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, slot_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )
        
        self.slot_attention = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_slots),
            nn.Softmax(dim=1)
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(slot_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)  # Output bounding box coordinates (x, y, width, height)
        )

    def forward(self, x):
        """
        Forward pass of the Slot Attention model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Detected microplastic hotspot bounding boxes.
        """
        x = self.encoder(x)
        x = x.view(x.size(0), self.slot_dim, -1).permute(0, 2, 1)
        slots = self.slot_attention(x)
        slots = slots.permute(0, 2, 1)
        slots = slots.view(slots.size(0), self.num_slots, self.slot_dim)
        outputs = self.decoder(slots)
        return outputs

def train_model(model, dataloader, criterion, optimizer, device, epochs):
    """
    Train the Slot Attention model.

    Args:
        model (SlotAttentionModel): Slot Attention model to train.
        dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function for object detection.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to use for training (e.g., 'cuda' or 'cpu').
        epochs (int): Number of training epochs.

    Possible Errors:
    - RuntimeError: If there is a mismatch between the model's output size and the label size.
    - ValueError: If the data size is not a multiple of the batch size.

    Solutions:
    - Ensure that the model's output size matches the expected bounding box format (x, y, width, height).
    - Make sure that the batch size is set correctly and the data size is divisible by the batch size.
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

def evaluate_model(model, dataloader, device, iou_threshold=0.5):
    """
    Evaluate the trained Slot Attention model on the test data.

    Args:
        model (SlotAttentionModel): Trained Slot Attention model.
        dataloader (DataLoader): DataLoader for test data.
        device (torch.device): Device to use for evaluation (e.g., 'cuda' or 'cpu').
        iou_threshold (float): Intersection over Union (IoU) threshold for determining true positives.

    Returns:
        tuple: (precision, recall, f1_score) metrics for the model's performance.

    Possible Errors:
    - RuntimeError: If there is a mismatch between the model's output size and the label size.
    - ValueError: If the data size is not a multiple of the batch size.

    Solutions:
    - Ensure that the model's output size matches the expected bounding box format (x, y, width, height).
    - Make sure that the batch size is set correctly and the data size is divisible by the batch size.
    """
    model.eval()
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Convert bounding box coordinates to pixel values
            outputs[:, :, 0] *= images.size(2)  # x-coordinate
            outputs[:, :, 1] *= images.size(3)  # y-coordinate
            outputs[:, :, 2] *= images.size(2)  # width
            outputs[:, :, 3] *= images.size(3)  # height
            
            # Compute IoU between predicted and ground truth bounding boxes
            for i in range(outputs.size(0)):
                pred_boxes = outputs[i]
                gt_boxes = labels[i]
                
                for pred_box in pred_boxes:
                    max_iou = 0.0
                    max_gt_box = None
                    
                    for gt_box in gt_boxes:
                        iou = compute_iou(pred_box, gt_box)
                        if iou > max_iou:
                            max_iou = iou
                            max_gt_box = gt_box
                    
                    if max_iou >= iou_threshold:
                        true_positives += 1
                    else:
                        false_positives += 1
                
                false_negatives += len(gt_boxes) - torch.sum(torch.any(outputs[i] >= iou_threshold, dim=0))
    
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1_score = 2 * (precision * recall) / (precision + recall)
    
    return precision, recall, f1_score

def compute_iou(box1, box2):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.

    Args:
        box1 (torch.Tensor): First bounding box (x, y, width, height).
        box2 (torch.Tensor): Second bounding box (x, y, width, height).

    Returns:
        float: IoU value between the two bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    area1 = w1 * h1
    area2 = w2 * h2
    
    intersection_left = max(x1, x2)
    intersection_top = max(y1, y2)
    intersection_right = min(x1 + w1, x2 + w2)
    intersection_bottom = min(y1 + h1, y2 + h2)
    
    if intersection_right < intersection_left or intersection_bottom < intersection_top:
        return 0.0
    
    intersection_area = (intersection_right - intersection_left) * (intersection_bottom - intersection_top)
    union_area = area1 + area2 - intersection_area
    
    iou = intersection_area / union_area
    return iou

def main():
    # Set the device to use for training and evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set the paths to the data directories
    train_data_dir = 'path/to/training/data'
    test_data_dir = 'path/to/testing/data'
    
    # Set the hyperparameters
    num_slots = 5
    slot_dim = 64
    hidden_dim = 128
    batch_size = 32
    num_workers = 4
    epochs = 50
    learning_rate = 0.001
    
    # Define the data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create the datasets and data loaders
    train_dataset = MicroplasticHotspotDataset(train_data_dir, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    test_dataset = MicroplasticHotspotDataset(test_data_dir, transform=transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    # Initialize the Slot Attention model
    model = SlotAttentionModel(num_slots, slot_dim, hidden_dim).to(device)
    
    # Initialize the loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model
    train_model(model, train_dataloader, criterion, optimizer, device, epochs)
    
    # Evaluate the trained model
    precision, recall, f1_score = evaluate_model(model, test_dataloader, device)
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

if __name__ == "__main__":
    main()
