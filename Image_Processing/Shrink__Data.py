import os
import shutil

def create_limited_datasets(base_input_folder, base_output_folder, limit=160):
    # Create output folders if they don't exist
    train_limited_folder = os.path.join(base_output_folder, 'Training_Limited')
    test_limited_folder = os.path.join(base_output_folder, 'Testing_Limited')
    os.makedirs(train_limited_folder, exist_ok=True)
    os.makedirs(test_limited_folder, exist_ok=True)

    # Iterate through each class folder
    for folder_name in os.listdir(base_input_folder):
        class_folder = os.path.join(base_input_folder, folder_name)
        if os.path.isdir(class_folder):
            train_folder = os.path.join(class_folder, 'Training')
            test_folder = os.path.join(class_folder, 'Testing')
            train_limited_class_folder = os.path.join(train_limited_folder, folder_name)
            test_limited_class_folder = os.path.join(test_limited_folder, folder_name)
            os.makedirs(train_limited_class_folder, exist_ok=True)
            os.makedirs(test_limited_class_folder, exist_ok=True)
            copy_images(train_folder, train_limited_class_folder, limit)
            copy_images(test_folder, test_limited_class_folder, limit)

def copy_images(source_folder, destination_folder, limit):
    files = os.listdir(source_folder)
    count = 0
    for file in files:
        if count >= limit:
            break
        src_path = os.path.join(source_folder, file)
        if os.path.isfile(src_path):
            dst_path = os.path.join(destination_folder, file)
            shutil.copy(src_path, dst_path)
            count += 1

train_folder = '/home/liam/git/Ai_Fungi_Finder/Data_V4/Testing'
test_folder = '/home/liam/git/Ai_Fungi_Finder/Data_V4/Testing'
train_limited_folder = '/home/liam/git/Ai_Fungi_Finder/Data_V5/Traning'
test_limited_folder = '/home/liam/git/Ai_Fungi_Finder/Data_V5/Testing'

create_limited_datasets(train_folder, test_folder, train_limited_folder, test_limited_folder)
