import torchvision.transforms.functional as F
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision.models as models

num_classes = 10

# Define the model architecture (should match the one used during training)
loaded_model = models.vgg16(pretrained=False)  # Load a new instance of VGG model
num_features = loaded_model.classifier[6].in_features
loaded_model.classifier[6] = nn.Linear(num_features, num_classes)  # Modify the final fully connected layer

# Load the saved model
loaded_model.load_state_dict(torch.load('vgg_model.pth'))  # Load the saved state dictionary

# Set the model to evaluation mode
loaded_model.eval()

# Load the image and process it (same as before)
image_path = '/home/liam/git/Ai_Fungi_Finder/Data/Testing/beefsteak_fungus/9.png'
image = Image.open(image_path).convert('RGB')

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
image_tensor = transform(image)
image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

# Forward pass through the model (same as before)
with torch.no_grad():
    outputs = loaded_model(image_tensor)



# Use a threshold for filtering the predictions
score_threshold = 0.8
boxes = outputs[0]['boxes'][outputs[0]['scores'] > score_threshold]
# Convert normalized coordinates to pixel coordinates
image_width, image_height = image.size
boxes = boxes * torch.tensor([image_width, image_height, image_width, image_height], dtype=torch.float32)
boxes = boxes.int()

# Draw bounding boxes on the image
from torchvision.utils import draw_bounding_boxes
drawn_image = draw_bounding_boxes(F.to_tensor(image), boxes, width=4)

# Convert the drawn image to numpy array for visualization
drawn_image_np = np.transpose(drawn_image.numpy(), (1, 2, 0))

# Visualize the image with bounding boxes
import matplotlib.pyplot as plt
plt.imshow(drawn_image_np)
plt.axis('off')
plt.show()
