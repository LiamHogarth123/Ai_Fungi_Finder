import os
import shutil

def split_images_into_train_test(folders, base_input_folder, train_output_folder, test_output_folder):
    # Create output folders if they don't exist
    os.makedirs(train_output_folder, exist_ok=True)
    os.makedirs(test_output_folder, exist_ok=True)

    for folder in folders:
        # Get the list of image files in the current folder
        folder_path = os.path.join(base_input_folder, folder)
        image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        # Sort the image files alphabetically
        image_files.sort()

        # Iterate through the image files and copy them to training or testing folders
        for i, image_file in enumerate(image_files):
            src_path = os.path.join(folder_path, image_file)
            if i % 2 == 0:
                # Copy to training folder
                dst_path = os.path.join(train_output_folder, folder)
            else:
                # Copy to testing folder
                dst_path = os.path.join(test_output_folder, folder)

            # Create the destination folder if it doesn't exist
            os.makedirs(dst_path, exist_ok=True)

            # Copy the image file
            shutil.copy(src_path, dst_path)

# Example usage:
folders = ['Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula', 'Suillus']
base_input_folder = '/home/liam/git/Ai_Fungi_Finder/Verified_Data/Training_Testing'
train_output_folder = '/home/liam/git/Ai_Fungi_Finder/Data_V4/Testing/Training'
test_output_folder = '/home/liam/git/Ai_Fungi_Finder/Data_V4/Testing'

split_images_into_train_test(folders, base_input_folder, train_output_folder, test_output_folder)
