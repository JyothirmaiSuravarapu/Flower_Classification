# Import necessary libraries
import argparse
import torch
from torch import nn, optim
from torchvision import transforms, datasets, models
from collections import OrderedDict
import time
from contextlib import contextmanager
import os

# Define a context manager to keep the workspace active during training
@contextmanager
def active_session():
    try:
        yield
    except Exception as e:
        print(e)
    finally:
        time.sleep(5)

# Parse command line arguments
parser = argparse.ArgumentParser(description="Train a deep learning model on a dataset")
parser.add_argument("data_dir", help="Path to the dataset")
parser.add_argument("--save_dir", default=".", help="Directory to save the checkpoint")
parser.add_argument("--arch", default="vgg", choices=['vgg', 'densenet'], help="Pre-trained model architecture (vgg or densenet)")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--hidden_units", type=int, default=512, help="Number of hidden units in the classifier")
parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

args = parser.parse_args()

# Define data transformations
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(root=f'{args.data_dir}/{x}', transform=data_transforms[x])
                  for x in ['train', 'valid', 'test']}

# Define dataloaders
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64, shuffle=True)
               for x in ['train', 'valid', 'test']}

# Load the pre-trained model
if args.arch == 'vgg':
    model = models.vgg16(pretrained=True)
    input_size = 25088  # VGG16 input size
elif args.arch == 'densenet':
    model = models.densenet121(pretrained=True)
    input_size = 1024  # DenseNet121 input size
else:
    raise ValueError("Invalid architecture. Choose between 'vgg' and 'densenet'.")

# Freeze parameters of the pre-trained model
for param in model.parameters():
    param.requires_grad = False

# Replace the classifier with a new one
classifier = nn.Sequential(
    nn.Linear(input_size, args.hidden_units),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(args.hidden_units, 102),
    nn.LogSoftmax(dim=1)
)
model.classifier = classifier

# Define the criterion and optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
model.to(device)

# Training loop
with active_session():  # Keep the workspace active during training
    for epoch in range(args.epochs):
        training_loss = 0
        model.train()
        for inputs, labels in dataloaders['train']:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()

        # Validate the model
        validation_loss = 0
        accuracy = 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in dataloaders['valid']:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model.forward(inputs)
                loss = criterion(outputs, labels)
                validation_loss += loss.item()

                ps = torch.exp(outputs)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

        # Print training and validation statistics
        print(f"Epoch {epoch + 1}/{args.epochs}.. "
              f"Training Loss: {training_loss / len(dataloaders['train']):.3f}.. "
              f"Validation Loss: {validation_loss / len(dataloaders['valid']):.3f}.. "
              f"Validation Accuracy: {accuracy / len(dataloaders['valid']):.3f}")

# Save the checkpoint
os.makedirs(args.save_dir, exist_ok=True)

# Define the checkpoint path
checkpoint_path = os.path.join(args.save_dir, 'model.pth')

model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint = {'architecture': args.arch,
              'classifier': model.classifier,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict()}
torch.save(checkpoint, checkpoint_path)
print(f"Checkpoint saved to: {checkpoint_path}")
