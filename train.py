import pathlib

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt


# configuration ------------------------------------------------------------------------

# the path to the directory containing the training data
TRAIN_DIR = pathlib.Path(__file__).parent / "data"
# print("TRAIN_DIR:", TRAIN_DIR.resolve())
# print("Exists?", TRAIN_DIR.exists())
# print("Number of jpg files:", len(list(TRAIN_DIR.glob("*.jpg"))))
# print("All items in TRAIN_DIR:")
# for p in TRAIN_DIR.iterdir():
#     print("-", repr(p.name))

# print("glob *.jpg:", list(TRAIN_DIR.glob("*.jpg"))[:5])
# print("glob *.JPG:", list(TRAIN_DIR.glob("*.JPG"))[:5])
# print("rglob *.jpg:", list(TRAIN_DIR.rglob("*.jpg"))[:5])

# the six classes of cities in our dataset
CLASSES = sorted(
    [
        "Anaheim",
        "Bakersfield",
        "Los_Angeles",
        "Riverside",
        "SLO",
        "San_Diego",
    ]
)

# we will later use this to convert a string class (like "Los_Angeles") to a numerical
# label (like 2), since PyTorch models work with numerical labels rather than strings.
CLASS_TO_NUMBER = {name: i for i, name in enumerate(CLASSES)}

# we'll resize all images to this size before feeding them into the model smaller image
# sizes means fewer parameters and therefore faster training, but also less information
# for the model to learn from
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128

BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 10
PATIENCE = 5 # stop if val accuracy doesn't improve for 7 epochs

# the percentage of the training data to set aside as a validation set.
VALIDATION_FRACTION = 0.2

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu") 
print(f"Using device: {device}")


# dataset ------------------------------------------------------------------------------

# this is a custom dataset loader that loads the images from disk and applies any
# transformations we specify.

class SoCalDataset(Dataset):
    """Loads images from the training set."""

    def __init__(self, root, transform=None):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.samples = []  # list of (path, label_index)

        # this determines the class label of an image based on its filename. For
        # example, the image "Los_Angeles-123.jpg" would be labeled as "Los_Angeles". We
        # then convert that string to a numerical label using the CLASS_TO_INDEX
        # dictionary defined above.
        for path in sorted(self.root.glob("*.jpg")):
            label = path.name.rsplit("-", 1)[0]
            self.samples.append((path, CLASS_TO_NUMBER[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# model --------------------------------------------------------------------------------

# this is a simple logistic regression model. It consists of a single linear layer that
# takes in the flattened pixel values and outputs a vector of class scores.

# when you're defining your own model, this is the part of the code that you'll likely
# change the most

# class CNN(nn.Module):
#     def __init__(self, num_classes):
#         super().__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(3, 16, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),

#             nn.Conv2d(16, 32, 3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )

#         self.fc = nn.Linear(32 * 8 * 16, num_classes)

#     def forward(self, x):
#         x = self.conv(x)
#         x = torch.flatten(x, 1)
#         return self.fc(x)

class FineTunedResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # load resnet18 with pretrained ImageNet weights
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # replace the final layer: 512 -> num_classes (6 cities)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)


# training -----------------------------------------------------------------------------


def main():
    # Step 1) define transformations on the images. Here we resize the images and
    # convert them to tensors.
    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
            transforms.ToTensor(),
        ]
    )

    # Step 2) split the data into a train set and a validation set, and create data
    # loaders for each. See DSC 80 for why doing this is important!
    full_dataset = SoCalDataset(TRAIN_DIR, transform=transform)
    val_size = int(len(full_dataset) * VALIDATION_FRACTION)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Step 3) define the model, loss function, and optimizer. The model is a simple
    # logistic regression model defined above. The loss function is cross-entropy loss,
    # which is commonly used for multi-class classification problems. The optimizer is
    # Adam.

    model = FineTunedResNet(len(CLASSES)).to(device)

    # freeze everything except layer4 and fc
    for name, param in model.model.named_parameters():
        if "layer4" not in name and "fc" not in name:
            param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Step 4) the training loop.

    best_val_accuracy = 0.0
    epochs_without_improvement = 0
    train_losses = []
    val_accuracies = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += images.size(0)

        avg_loss = total_loss / total
        accuracy = correct / total

        # evaluate on the validation set
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images) 
                val_correct += (outputs.argmax(dim=1) == labels).sum().item()
                val_total += images.size(0)
        val_accuracy = val_correct / val_total

        train_losses.append(avg_loss)
        val_accuracies.append(val_accuracy)

        # save best model + early stopping
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), "model.pt")
            print(f"  → Saved new best model (val_accuracy: {val_accuracy:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  → No improvement ({epochs_without_improvement}/{PATIENCE})")
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        


    print(f"Best val accuracy: {best_val_accuracy:.4f}")
    epochs_ran = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 4))

    # training loss (empirical risk)
    plt.subplot(1, 2, 1)
    plt.plot(epochs_ran, train_losses, marker='o')
    plt.title("Training Loss (Empirical Risk)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # val accuracy (bonus, useful to see)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_ran, val_accuracies, marker='o', color='orange')
    plt.title("Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.savefig("training_curve.png")
    print("Saved training_curve.png")

if __name__ == "__main__":
    main()
