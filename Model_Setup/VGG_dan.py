import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms, models

# Prepare the dataset
# Define data transforms (e.g., resizing, normalization)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_root = '/home/liam/git/Ai_Fungi_Finder/Data'

# Load the dataset
train_dataset = datasets.ImageFolder(root=dataset_root, transform=data_transforms)
val_dataset = datasets.ImageFolder(root=dataset_root, transform=data_transforms)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)

# Load pre-trained VGG model
vgg_model = models.vgg16(pretrained=True)

# Disable preexisting training from being modified
for param in vgg_model.parameters():
    param.requires_grad = False

# Replace Classifier layer
num_features = vgg_model.classifier[6].in_features
vgg_model.classifier[6] = nn.Linear(num_features, len(train_dataset))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg_model.parameters(), lr=0.001)

# Define learning rate scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)  # Adjust parameters as needed

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_model.to(device)
num_epochs = 10
for epoch in range(num_epochs):
    print("Beginning training")
    vgg_model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = vgg_model(inputs.to(device))  # Move inputs to device
        loss = criterion(outputs, labels.to(device))  # Move labels to device
        loss.backward()
        optimizer.step()
        print("train")
    
    # Step the scheduler
    scheduler.step()

    # Validation
    vgg_model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for inputs, labels in val_loader:
            outputs = vgg_model(inputs.to(device))  # Move inputs to device
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels.to(device)).sum().item()  # Move labels to device
        val_accuracy = total_correct / total_samples
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy}')


# Save the trained model
torch.save(vgg_model.state_dict(), 'vgg_model_dan.pth')

# Test the final model on the test set if available
# Evaluate the model using a similar procedure as validation






## Load the model and use it to predict the mushroom type

# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# from PIL import Image

# # Define the model architecture (VGG16 with modified classifier)
# model = models.vgg16(pretrained=False)  # Assuming you're not using pre-trained weights
# num_features = model.classifier[6].in_features
# model.classifier[6] = nn.Linear(num_features, num_classes)  # Assuming num_classes is known
# model.load_state_dict(torch.load('vgg_model.pth'))
# model.eval()

# # Define data transforms (the same as used during training)
# data_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# # Load and preprocess the image
# image_path = 'path_to_your_image.jpg'
# image = Image.open(image_path)
# input_tensor = data_transforms(image)
# input_batch = input_tensor.unsqueeze(0)  # Add batch dimension

# # Perform inference
# with torch.no_grad():
#     output = model(input_batch)

# # Convert output probabilities to predicted class
# probabilities = torch.nn.functional.softmax(output[0], dim=0)
# predicted_class = torch.argmax(probabilities).item()

# # Display the predicted class
# print("Predicted Mushroom Type:", predicted_class)