# Import necessary libraries
import argparse
import torch
from torchvision import models, transforms
from PIL import Image
import json

# Define a function to load the checkpoint and rebuild the model
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = getattr(models, checkpoint['architecture'])(pretrained=True)
    model.classifier = checkpoint['classifier']
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

# Define a function to process the image for inference
def process_image(image_path):
    # Open the image file
    img = Image.open(image_path)
    
    # Define the image transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Apply the transformation
    img_tensor = transform(img)
    
    # Convert to a 1D tensor
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

# Define the predict function
def predict(image_path, model, topk=5, category_names=None, gpu=False):
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
    model.to(device)
    
    # Set model to evaluation mode
    model.eval()

    # Process the image
    img_tensor = process_image(image_path)
    
    # Move the input tensor to the device
    img_tensor = img_tensor.to(device)

    # Perform inference
    with torch.no_grad():
        output = model(img_tensor)

    # Calculate probabilities and classes
    probabilities = torch.exp(output)
    top_probabilities, top_classes = probabilities.topk(topk)
    
    # Convert indices to class labels
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    top_classes = [idx_to_class[idx.item()] for idx in top_classes[0]]

    # Map class labels to flower names if category_names is provided
    if category_names is not None:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_classes = [cat_to_name[class_label] for class_label in top_classes]

    return top_probabilities[0].tolist(), top_classes

# Parse command line arguments
parser = argparse.ArgumentParser(description="Make predictions using a trained deep learning model")
parser.add_argument("image_path", help="Path to the input image")
parser.add_argument("checkpoint", help="Path to the trained model checkpoint")
parser.add_argument("--top_k", type=int, default=5, help="Number of top most likely classes to display")
parser.add_argument("--category_names", help="Path to a JSON file mapping class labels to category names")
parser.add_argument("--gpu", action="store_true", help="Use GPU for inference")

args = parser.parse_args()

# Load the model from the checkpoint
model = load_checkpoint(args.checkpoint)

# Make predictions
probs, classes = predict(args.image_path, model, topk=args.top_k, category_names=args.category_names, gpu=args.gpu)

# Print the results
for i, (prob, class_name) in enumerate(zip(probs, classes), 1):
    print(f"{i}. Class: {class_name}, Probability: {prob:.4f}")
