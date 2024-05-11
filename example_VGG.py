import pandas as pd
import IPython.display as display
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

print(tf.__version__)

IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
BATCH_SIZE = 64

# image processing

def get_pathframe(path):
    '''
    Get all the images paths and its corresponding labels
    Store them in pandas dataframe
    '''
    filenames = os.listdir(path)
    categories = []
    paths = []
    for filename in filenames:
        paths.append(path + filename)
        category = filename.split('.')[0]
        if category == 'mushroom':
            categories.append(1)
        else:
            categories.append(0)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories,
        'paths': paths
    })
    return df

df = get_pathframe("/home/dan/git/Ai_Fungi_Finder/Data/Testing/")
df.tail(5)

def load_and_preprocess_image(path):
    '''
    Load each image and resize it to desired shape
    '''
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMAGE_WIDTH, IMAGE_HEIGHT])
    image /= 255.0  # normalize to [0,1] range
    return image

def convert_to_tensor(df):
    '''
    Convert each data and labels to tensor
    '''
    path_ds = tf.data.Dataset.from_tensor_slices(df['paths'])
    image_ds = path_ds.map(load_and_preprocess_image)
    onehot_label = tf.cast(df['category'], tf.int64)
    label_ds = tf.data.Dataset.from_tensor_slices(onehot_label)

    return image_ds, label_ds

X, Y = convert_to_tensor(df)
print("Shape of X in data:", X)
print("Shape of Y in data:", Y)

dataset = tf.data.Dataset.zip((X, Y)).shuffle(buffer_size=2000)
dataset_train = dataset.take(22500)
dataset_test = dataset.skip(22500)

dataset_train = dataset_train.batch(BATCH_SIZE, drop_remainder=True)
dataset_test = dataset_test.batch(BATCH_SIZE, drop_remainder=True)

def plotimages(imagesls):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for image, ax in zip(imagesls, axes):
        ax.imshow(image)
        ax.axis('off')

imagesls = []
for n, image in enumerate(X.take(5)):
    imagesls.append(image)

plotimages(imagesls)

# Transfer Learning - VGG16:

from tensorflow.keras.applications import VGG16

pre_trained_model = VGG16(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), include_top=False, weights="imagenet")

for layer in pre_trained_model.layers[:15]:
    layer.trainable = False

for layer in pre_trained_model.layers[15:]:
    layer.trainable = True

last_layer = pre_trained_model.get_layer('block5_pool')
last_output = last_layer.output
x = layers.Flatten()(last_output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(1, activation='sigmoid')(x)

vggmodel = tf.keras.models.Model(pre_trained_model.input, x)

vggmodel.compile(loss='binary_crossentropy',
                 optimizer=tf.keras.optimizers.SGD(lr=1e-4, momentum=0.9),
                 metrics=['accuracy'])

vggmodel.summary()

vgghist = vggmodel.fit_generator(dataset_train, epochs=20, validation_data=dataset_test)

vggmodel.save("/content/drive/My Drive/DPprojects/Mushroom_Classification/vggmodel.h5")

loss, accuracy = vggmodel.evaluate_generator(dataset_test)
print("Test: accuracy = %f  ;  loss = %f " % (accuracy, loss))

vgg_y_pred = vggmodel.predict_generator(dataset_test)
vgg_y_p = np.where(vgg_y_pred > 0.5, 1, 0)

# compute the confusion matrix
confusion_mtx = confusion_matrix(Y, vgg_y_p)
# plot the confusion matrix
f, ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01, cmap="Blues", linecolor="gray", fmt='.1f', ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

report = classification_report(Y, vgg_y_p, target_names=['0', '1'])
print(report)
