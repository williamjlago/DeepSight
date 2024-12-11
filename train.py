import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from deepsight_model import DeepSightModel
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

class FacesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # Labeling: sad=1, happy=0
        happy_dir = os.path.join(root_dir, "Happy-Joyful Faces")
        sad_dir = os.path.join(root_dir, "Sad-Downcast Faces")

        happy_images = [os.path.join(happy_dir, f) for f in os.listdir(happy_dir) if
                        f.lower().endswith(('.jpg', '.jpeg'))]
        sad_images = [os.path.join(sad_dir, f) for f in os.listdir(sad_dir) if
                      f.lower().endswith(('.jpg', '.jpeg'))]

        for img_path in happy_images:
            self.samples.append((img_path, 0.0))
        for img_path in sad_images:
            self.samples.append((img_path, 1.0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")  # Grayscale

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor([label], dtype=torch.float32)

if __name__ == "__main__":
    data_root = ""

    # Data augmentations for training
    transform_train = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Validation/test: no augmentation, just normalization
    transform_val_test = transforms.Compose([
        transforms.Resize((250, 250)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    full_dataset = FacesDataset(root_dir=data_root, transform=None)
    total_len = len(full_dataset)
    train_len = int(total_len * 0.8)
    val_len = int(total_len * 0.1)
    test_len = total_len - train_len - val_len
    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_len, val_len, test_len])

    # Assign transforms after splitting
    train_dataset.dataset.transform = transform_train
    val_dataset.dataset.transform = transform_val_test
    test_dataset.dataset.transform = transform_val_test

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    if torch.cuda.is_available():
        print("CUDA found, using GPU to train...")
    else:
        print("CUDA not available, using CPU to train...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = DeepSightModel().to(device)

    # Compute pos_weight
    # sad=1: 516 images
    # happy=0: 1041 images
    # pos_weight = #neg/#pos = 1041/516 ~ 2.0
    pos_weight = torch.tensor([1041 / 516], dtype=torch.float32).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 20
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # Outputs are logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * images.size(0)

        train_loss = total_train_loss / len(train_loader.dataset)

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item() * images.size(0)

                preds = (torch.sigmoid(outputs) >= 0.5).float()
                correct_val += (preds == labels).sum().item()

        val_loss = total_val_loss / len(val_loader.dataset)
        val_accuracy = correct_val / (len(val_loader.dataset) * 1.0)

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")

    # Testing phase
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    total_test_loss = 0.0
    correct_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_test_loss += loss.item() * images.size(0)

            preds = (torch.sigmoid(outputs) >= 0.5).float()
            correct_test += (preds == labels).sum().item()

    test_loss = total_test_loss / len(test_loader.dataset)
    test_accuracy = correct_test / (len(test_loader.dataset) * 1.0)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

    # Visualization of predictions on test set
    # Show a subset of images and their predictions
    max_images_to_show = 10
    count = 0
    test_loader_vis = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=1)

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader_vis:
            if count >= max_images_to_show:
                break
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) >= 0.5).float()

            # Get the single image and label
            img = images[0].cpu()
            label = labels[0].item()
            pred = preds[0].item()

            # Undo normalization: img was normalized with mean=0.5, std=0.5
            # img = (img * std) + mean => img = img * 0.5 + 0.5
            img = img * 0.5 + 0.5
            pil_img = transforms.ToPILImage()(img)

            pred_emotion = "Sad" if pred == 1.0 else "Happy"
            true_emotion = "Sad" if label == 1.0 else "Happy"
            correctness = "Correct" if pred == label else "Incorrect"

            plt.figure(figsize=(4,4))
            plt.imshow(pil_img, cmap='gray')
            plt.title(f"Pred: {pred_emotion} | True: {true_emotion} | {correctness}")
            plt.axis('off')
            plt.show()
            count += 1
