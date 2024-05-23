import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from yolov5.models.yolo import Model
from yolov5.utils.datasets import LoadImages
from yolov5.utils.general import non_max_suppression

class MicroplasticDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Initialize the MicroplasticDataset.

        Args:
            image_dir (str): Directory containing the microplastic images.
            mask_dir (str): Directory containing the corresponding segmentation masks.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (image, mask) where image is the input image and mask is the corresponding segmentation mask.
        """
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
        return image, mask

class UNet(nn.Module):
    def __init__(self, num_classes):
        """
        Initialize the U-Net model.

        Args:
            num_classes (int): Number of output classes (including background).
        """
        super(UNet, self).__init__()
        self.num_classes = num_classes
        # Define the U-Net architecture
        # ...

    def forward(self, x):
        """
        Forward pass of the U-Net model.

        Args:
            x (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Output segmentation mask tensor.
        """
        # Implement the forward pass of the U-Net model
        # ...

def train_unet(model, dataloader, criterion, optimizer, device, num_epochs):
    """
    Train the U-Net model.

    Args:
        model (UNet): U-Net model to be trained.
        dataloader (DataLoader): DataLoader for the training dataset.
        criterion (nn.Module): Loss function.
        optimizer (optim.Optimizer): Optimizer for updating the model parameters.
        device (torch.device): Device to run the training on (GPU or CPU).
        num_epochs (int): Number of training epochs.

    Possible Errors:
    - GPU out of memory: If the model and data are too large for the available GPU memory.
    - Vanishing or exploding gradients: If the model architecture or hyperparameters are not properly tuned.

    Solutions:
    - Reduce the batch size to fit the model and data into the available GPU memory.
    - Adjust the learning rate, use gradient clipping, or try a different optimizer to stabilize the training.
    """
    model.to(device)
    for epoch in range(num_epochs):
        model.train()
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

def predict_microplastics_yolov5(model_path, image_path, conf_thresh=0.5, iou_thresh=0.5):
    """
    Predict microplastics in an image using YOLOv5.

    Args:
        model_path (str): Path to the trained YOLOv5 model.
        image_path (str): Path to the input image.
        conf_thresh (float): Confidence threshold for object detection.
        iou_thresh (float): IoU threshold for non-maximum suppression.

    Returns:
        list: List of detected microplastics, each represented as a dictionary containing 'bbox' and 'conf'.

    Possible Errors:
    - Model not found: If the specified model path is incorrect or the model file doesn't exist.
    - Invalid image: If the image file is corrupted or not in a supported format.

    Solutions:
    - Ensure that the model path is correct and the model file exists.
    - Verify that the image file is valid and in a supported format (e.g., JPEG, PNG).
    """
    model = Model(model_path)
    dataset = LoadImages(image_path)
    detections = []
    for _, img, _ in dataset:
        pred = model(img, augment=False, visualize=False)
        pred = non_max_suppression(pred, conf_thresh, iou_thresh)
        for det in pred:
            if len(det):
                for *xyxy, conf, cls in reversed(det):
                    bbox = [int(x) for x in xyxy]
                    detections.append({'bbox': bbox, 'conf': float(conf)})
    return detections

def segment_microplastics_unet(model, image_path, device):
    """
    Segment microplastics in an image using U-Net.

    Args:
        model (UNet): Trained U-Net model.
        image_path (str): Path to the input image.
        device (torch.device): Device to run the inference on (GPU or CPU).

    Returns:
        numpy.ndarray: Segmentation mask as a 2D numpy array.

    Possible Errors:
    - Invalid image: If the image file is corrupted or not in a supported format.
    - Model not trained: If the provided U-Net model is not trained.

    Solutions:
    - Verify that the image file is valid and in a supported format (e.g., JPEG, PNG).
    - Ensure that the U-Net model is properly trained before performing segmentation.
    """
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.squeeze().cpu().numpy()
    return predicted

def visualize_detections(image_path, detections, output_path):
    """
    Visualize the detected microplastics on the input image.

    Args:
        image_path (str): Path to the input image.
        detections (list): List of detected microplastics.
        output_path (str): Path to save the output image with visualized detections.

    Possible Errors:
    - Invalid image: If the image file is corrupted or not in a supported format.
    - Directory not found: If the output directory does not exist.

    Solutions:
    - Verify that the image file is valid and in a supported format (e.g., JPEG, PNG).
    - Create the output directory if it does not exist.
    """
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    for detection in detections:
        bbox = detection['bbox']
        draw.rectangle(bbox, outline="red", width=2)
    image.save(output_path)

def main():
    # Set up paths and parameters
    image_dir = "path/to/microplastic/images"
    mask_dir = "path/to/segmentation/masks"
    model_path = "path/to/trained/yolov5/model.pt"
    output_dir = "path/to/output/directory"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create the dataset and dataloader
    dataset = MicroplasticDataset(image_dir, mask_dir, transform=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Initialize the U-Net model
    unet_model = UNet(num_classes=2)

    # Set up the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(unet_model.parameters(), lr=0.001)

    # Train the U-Net model
    train_unet(unet_model, dataloader, criterion, optimizer, device, num_epochs=10)

    # Perform microplastic detection using YOLOv5
    image_path = "path/to/input/image.jpg"
    detections = predict_microplastics_yolov5(model_path, image_path)

    # Perform microplastic segmentation using U-Net
    segmentation_mask = segment_microplastics_unet(unet_model, image_path, device)

    # Visualize the detections
    output_path = os.path.join(output_dir, "detections.jpg")
    visualize_detections(image_path, detections, output_path)

    # Visualize the segmentation mask
    output_path = os.path.join(output_dir, "segmentation.jpg")
    plt.imsave(output_path, segmentation_mask, cmap="gray")

if __name__ == "__main__":
    main()
