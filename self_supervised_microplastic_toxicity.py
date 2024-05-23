import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import random
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class MicroplasticDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Initialize the MicroplasticDataset.

        Args:
            data_dir (str): Directory containing the microplastic images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_dir = data_dir
        self.transform = transform
        self.images = os.listdir(data_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image, augmented_image) where image is the original image and augmented_image is the randomly augmented version of the image.
        """
        img_path = os.path.join(self.data_dir, self.images[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
            augmented_image = self.transform(image)
        return image, augmented_image

class SimCLRModel(nn.Module):
    def __init__(self, base_model, projection_dim=128):
        """
        Initialize the SimCLR model.

        Args:
            base_model (nn.Module): Base encoder model (e.g., ResNet).
            projection_dim (int): Dimension of the projection head output.
        """
        super(SimCLRModel, self).__init__()
        self.encoder = base_model
        self.projection_head = nn.Sequential(
            nn.Linear(base_model.fc.in_features, base_model.fc.in_features),
            nn.ReLU(),
            nn.Linear(base_model.fc.in_features, projection_dim)
        )

    def forward(self, x):
        """
        Forward pass of the SimCLR model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output of the projection head.
        """
        features = self.encoder(x)
        return self.projection_head(features)

def contrastive_loss(features1, features2, temperature=0.5):
    """
    Compute the contrastive loss for SimCLR.

    Args:
        features1 (torch.Tensor): Features of the first set of augmented images.
        features2 (torch.Tensor): Features of the second set of augmented images.
        temperature (float): Temperature parameter for scaling the similarity scores.

    Returns:
        torch.Tensor: Contrastive loss value.
    """
    batch_size = features1.shape[0]
    features = torch.cat([features1, features2], dim=0)
    similarity_matrix = torch.matmul(features, features.T)
    
    # Exclude self-similarities from the similarity matrix
    mask = ~torch.eye(2 * batch_size, dtype=torch.bool)
    similarity_matrix = similarity_matrix[mask].view(2 * batch_size, -1)
    
    # Compute the contrastive loss
    similarity_matrix = similarity_matrix / temperature
    labels = torch.arange(batch_size, dtype=torch.long)
    loss = nn.CrossEntropyLoss()(similarity_matrix, torch.cat([labels, labels], dim=0))
    return loss

def train_simclr(model, dataloader, optimizer, device, epochs):
    """
    Train the SimCLR model.

    Args:
        model (SimCLRModel): SimCLR model to train.
        dataloader (DataLoader): DataLoader for training data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to use for training (e.g., 'cuda' or 'cpu').
        epochs (int): Number of training epochs.

    Possible Errors:
    - RuntimeError: If there is a mismatch between the model's input size and the data size.
    - ValueError: If the data size is not a multiple of the batch size.

    Solutions:
    - Ensure that the input images are of the correct size and match the model's expected input size.
    - Make sure that the batch size is set correctly and the data size is divisible by the batch size.
    """
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, augmented_images in dataloader:
            images = images.to(device)
            augmented_images = augmented_images.to(device)
            
            # Forward pass
            features1 = model(images)
            features2 = model(augmented_images)
            
            # Compute contrastive loss
            loss = contrastive_loss(features1, features2)
            total_loss += loss.item()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

def train_classifier(encoder, classifier, dataloader, criterion, optimizer, device, epochs):
    """
    Train the classifier on top of the pre-trained encoder.

    Args:
        encoder (nn.Module): Pre-trained encoder model.
        classifier (nn.Module): Classifier model to train.
        dataloader (DataLoader): DataLoader for training data.
        criterion (nn.Module): Loss function for classification.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        device (torch.device): Device to use for training (e.g., 'cuda' or 'cpu').
        epochs (int): Number of training epochs.

    Possible Errors:
    - RuntimeError: If there is a mismatch between the encoder's output size and the classifier's input size.
    - ValueError: If the data size is not a multiple of the batch size.

    Solutions:
    - Ensure that the output size of the encoder matches the input size of the classifier.
    - Make sure that the batch size is set correctly and the data size is divisible by the batch size.
    """
    encoder.eval()
    classifier.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            with torch.no_grad():
                features = encoder(images)
            outputs = classifier(features)
            
            # Compute classification loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")

def evaluate_model(encoder, classifier, dataloader, device):
    """
    Evaluate the trained model on the test data.

    Args:
        encoder (nn.Module): Pre-trained encoder model.
        classifier (nn.Module): Trained classifier model.
        dataloader (DataLoader): DataLoader for test data.
        device (torch.device): Device to use for evaluation (e.g., 'cuda' or 'cpu').

    Returns:
        tuple: (accuracy, f1) where accuracy is the classification accuracy and f1 is the F1 score.

    Possible Errors:
    - RuntimeError: If there is a mismatch between the encoder's output size and the classifier's input size.
    - ValueError: If the data size is not a multiple of the batch size.

    Solutions:
    - Ensure that the output size of the encoder matches the input size of the classifier.
    - Make sure that the batch size is set correctly and the data size is divisible by the batch size.
    """
    encoder.eval()
    classifier.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            features = encoder(images)
            outputs = classifier(features)
            
            # Get predictions
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions, average='weighted')
    return accuracy, f1

def main():
    # Set the device to use for training and evaluation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set the paths to the data directories
    unlabeled_data_dir = 'path/to/unlabeled/microplastic/images'
    labeled_data_dir = 'path/to/labeled/microplastic/images'
    
    # Set the hyperparameters
    batch_size = 32
    num_workers = 4
    epochs_simclr = 100
    epochs_classifier = 50
    learning_rate = 0.001
    projection_dim = 128
    
    # Define the data transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create the datasets and data loaders
    unlabeled_dataset = MicroplasticDataset(unlabeled_data_dir, transform=train_transform)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    labeled_dataset = MicroplasticDataset(labeled_data_dir, transform=test_transform)
    labeled_dataloader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    
    # Initialize the base encoder model (e.g., ResNet)
    base_model = ...  # Choose a suitable base encoder model
    
    # Initialize the SimCLR model
    simclr_model = SimCLRModel(base_model, projection_dim=projection_dim).to(device)
    
    # Initialize the optimizer for SimCLR training
    optimizer_simclr = optim.Adam(simclr_model.parameters(), lr=learning_rate)
    
    # Train the SimCLR model
    train_simclr(simclr_model, unlabeled_dataloader, optimizer_simclr, device, epochs_simclr)
    
    # Initialize the classifier model
    classifier = nn.Linear(base_model.fc.in_features, num_classes).to(device)
    
    # Initialize the optimizer and loss function for classifier training
    optimizer_classifier = optim.Adam(classifier.parameters(), lr=learning_rate)
    criterion_classifier = nn.CrossEntropyLoss()
    
    # Train the classifier
    train_classifier(simclr_model.encoder, classifier, labeled_dataloader, criterion_classifier, optimizer_classifier, device, epochs_classifier)
    
    # Evaluate the trained model
    test_dataloader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    accuracy, f1 = evaluate_model(simclr_model.encoder, classifier, test_dataloader, device)
    print(f"Test Accuracy: {accuracy:.4f}, Test F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
