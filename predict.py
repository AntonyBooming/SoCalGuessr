import pathlib

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image


# these must match the values used during training in train.py
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

# when we trained the model, we resized all images to this size before feeding them into
# the model. We need to do the same thing here, since the model's weights were trained
# on images of this size.
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128


def load_and_transform_image(path):
    """Load an image from disk and apply the same transforms used during training.

    Parameters
    ----------
    path : str or pathlib.Path
        Path to a .jpg image file.

    Returns
    -------
    torch.Tensor
        A tensor of shape (1, 3, IMAGE_WIDTH, IMAGE_HEIGHT) ready to be fed into
        the model.

    """
    image = Image.open(path).convert("RGB")
    pipeline = transforms.Compose(
        [
            transforms.Resize((IMAGE_WIDTH, IMAGE_HEIGHT)),
            transforms.ToTensor(),
        ]
    )
    return pipeline(image).unsqueeze(0)  # add batch dimension


# we must re-define the model architecture here in order to load the saved weights.


# must match train.py exactly
class FineTunedResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=None)  # no pretrained weights needed here
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
 
    def forward(self, x):
        return self.model(x)


def predict(test_dir):
    """Load the saved model and predict a label for every image in `test_dir`.

    Parameters
    ----------
    test_dir : pathlib.Path
        Path to a directory containing .jpg test images.

    Returns
    -------
    dict[str, str]
        A dictionary mapping each image filename (e.g. "00001.jpg") to a predicted
        class label (e.g. "Los_Angeles").

    """
    test_dir = pathlib.Path(test_dir)

    # Step 1) load the trained model.
    model = FineTunedResNet(len(CLASSES))
    model.load_state_dict(torch.load("model.pt", weights_only=True, map_location=torch.device('cpu')))
    model.eval()

    # Step 2) run prediction on every test image.
    # Step 2) run prediction on every test image.
    image_paths = sorted(test_dir.glob("*.jpg"))
    images = torch.cat([load_and_transform_image(p) for p in image_paths], dim=0)

    predictions = {}
    with torch.no_grad():
        batch_size = 64
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            outputs = model(batch)
            predicted_indices = outputs.argmax(dim=1).tolist()
            for path, idx in zip(image_paths[i:i+batch_size], predicted_indices):
                predictions[path.name] = CLASSES[idx]

    return predictions


# if __name__ == "__main__":
#     preds = predict("./data")
#     print("Predictions:")
#     for filename, label in sorted(preds.items()):
#         print(f"{filename}: {label}")
