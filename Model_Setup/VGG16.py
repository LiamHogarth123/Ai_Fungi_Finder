from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns  # For visualizing the confusion matrix


def plot_model_history(model_history, acc='accuracy', val_acc='val_accuracy'):
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    num_epochs = len(model_history.history[acc])
    tick_positions = np.arange(1, num_epochs + 1, max(1, num_epochs // 10))
    
    axs[0].plot(range(1, num_epochs + 1), model_history.history[acc])
    axs[0].plot(range(1, num_epochs + 1), model_history.history[val_acc])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(tick_positions)
    axs[0].legend(['train', 'val'], loc='best')
    
    axs[1].plot(range(1, num_epochs + 1), model_history.history['loss'])
    axs[1].plot(range(1, num_epochs + 1), model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(tick_positions)
    axs[1].legend(['train', 'val'], loc='best')
    
    plt.show()


# data generator
# ////////////////////////////////////

# Create an ImageDataGenerator object for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,     # Normalize pixel values to [0,1]
    rotation_range=40,  # Randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # Randomly translate images horizontally
    height_shift_range=0.2, # Randomly translate images vertically
    shear_range=0.2,    # Randomly applying shearing transformations
    zoom_range=0.2,     # Randomly zooming inside pictures
    horizontal_flip=True,  # Randomly flipping half of the images horizontally
    fill_mode='nearest' # Strategy used for filling in newly created pixels
)

# Create an ImageDataGenerator object for validation data (No data augmentation)
validation_datagen = ImageDataGenerator(rescale=1./255)

# Flow training images in batches using train_datagen generator
train_generator = train_datagen.flow_from_directory(
        '/home/liam/git/Ai_Fungi_Finder/Data_V3/Training',  # This is the source directory for training images
        target_size=(224, 224),  # All images will be resized to 224x224
        batch_size=20,
        class_mode='categorical',  # Since we use categorical_crossentropy loss, we need categorical labels
        color_mode = 'rgb'
) 

# Flow validation images in batches using validation_datagen generator
validation_generator = validation_datagen.flow_from_directory(
        '/home/liam/git/Ai_Fungi_Finder/Data_V3/Testing',
        target_size=(224, 224),
        batch_size=20,
        class_mode='categorical',
        color_mode = 'rgb'
) 




# initialise model
# ///////////////////////////////////////////////////////////////////////////////////////////////////////

# Set the dimensions of the input image
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

pre_trained_model = VGG16(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), include_top=False, weights="imagenet")

# Freeze the layers except the last 4 layers
for layer in pre_trained_model.layers[:15]:
    layer.trainable = False

# Make the last 4 layers trainable
for layer in pre_trained_model.layers[15:]:
    layer.trainable = True

# Get the output of the last convolutional block
last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output

# Add new classifier layers
x = layers.Flatten()(last_output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(9, activation='softmax')(x)  # Change to 10 for the number of mushroom species

# Create the new model
vggmodel = tf.keras.models.Model(inputs=pre_trained_model.input, outputs=x)

# Compile the model
vggmodel.compile(loss='categorical_crossentropy',  # Change loss function
              optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# Print the model summary
vggmodel.summary()

print(train_generator.class_indices)
print(validation_generator.class_indices)


# Assuming 'dataset_train' and 'dataset_test' are properly set up data generators for your training and validation datasets
vgghist = vggmodel.fit(
    train_generator,  # use train_generator here
    epochs=20,
    validation_data=validation_generator  # use validation_generator here
)
vggmodel.save("/home/liam/git/Ai_Fungi_Finder/Models/vggmodelV3.keras")


plot_model_history(vgghist)

validation_generator.reset()
predictions = vggmodel.predict(validation_generator, steps=len(validation_generator))
predicted_classes = np.argmax(predictions, axis=1)

# Get true class indices
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())  # Get the list of class labels


# Compute the confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

print(classification_report(true_classes, predicted_classes, target_names=class_labels))

