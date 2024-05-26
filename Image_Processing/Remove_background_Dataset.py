import os
import cv2
import numpy as np

def remove_background(image_path):
    # Load the image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to open image file {image_path}")
        return None
    
    # Create an initial mask
    mask = np.zeros(image.shape[:2], np.uint8)

    # Create temporary arrays for the GrabCut algorithm
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Define a rectangle around the object for the GrabCut algorithm
    rect = (10, 10, image.shape[1] - 20, image.shape[0] - 20)

    # Apply GrabCut algorithm
    cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Modify the mask to set the background pixels to 0 and the foreground pixels to 1
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # Apply the mask to the image
    foreground = image * mask2[:, :, np.newaxis]

    # Create a black background image
    black_background = np.zeros_like(image)

    # Combine the foreground with the black background
    result = cv2.add(foreground, black_background)

    return result

def process_images(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_folder, filename)
            result = remove_background(image_path)
            if result is not None:
                output_path = os.path.join(output_folder, filename.rsplit('.', 1)[0] + '_no_background.' + filename.rsplit('.', 1)[1])
                cv2.imwrite(output_path, result)
                print(f"Processed and saved: {output_path}")
            else:
                print(f"Failed to process: {image_path}")



base_input_folder = '/home/liam/git/Ai_Fungi_Finder/Data_V3/Training'
base_output_folder = '/home/liam/git/Ai_Fungi_Finder/Data_V3/Training_Full_background_gone'

folders = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus' ]

# 

for folder in folders:
    input_folder = os.path.join(base_input_folder, folder)
    output_folder = os.path.join(base_output_folder, folder + '_no_background')
    process_images(input_folder, output_folder)


