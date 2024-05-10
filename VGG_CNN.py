import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision.models as models



# Prepare the dataset
# Define data transforms (e.g., resizing, normalization)
data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset_root = '/home/liam/git/Ai_Fungi_Finder/Data'
test_dataset_root = '/home/liam/git/Ai_Fungi_Finder/Data'

# Load the dataset
train_dataset = datasets.ImageFolder(root=dataset_root, transform=data_transforms)
val_dataset = datasets.ImageFolder(root=dataset_root, transform=data_transforms)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)


# Load pre-trained VGG model
vgg_model = models.vgg16(pretrained=True)



#Disable preexisting training from being modifed
#//////////////////////////////////////////////////////////////////////////////////////////////////////////
for param in vgg_model.parameters():
    param.requires_grad = False


# Modify the final fully connected layer
# num_classes = 215  # Number of mushroom species
# vgg_model.classifier[6] = nn.Linear(4096, num_classes)  # Modify the last FC layer


#Replaace Classifer layer
#//////////////////////////////////////////////////////////////////////////////////////////////////////////
num_features = vgg_model.classifier[6].in_features
vgg_model.classifier[6] = nn.Linear(num_features, len(train_dataset))




# Define loss function and optimizer
#////////////////////////////////////////////////////////////////////////////////////////////////////////////
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vgg_model.parameters(), lr=0.001)

# Training loop
#/////////////////////////////////////////////////////////////////////////////////////////////////////
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_model.to(device)
num_epochs = 10
for epoch in range(num_epochs):
    print("begining training")
    vgg_model.train()
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = vgg_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation
    vgg_model.eval()
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for inputs, labels in val_loader:
            outputs = vgg_model(inputs)
            _, predicted = torch.max(outputs, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
        val_accuracy = total_correct / total_samples
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_accuracy}')
 
# Test the final model on the test set if available
# Evaluate the model using a similar procedure as validation
