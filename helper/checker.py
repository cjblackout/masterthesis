import os

def check_photo_file_presence(base_folder_path):
    for folder_name in os.listdir(base_folder_path):
        folder_path = os.path.join(base_folder_path, folder_name)
        if os.path.isdir(folder_path):
            files = os.listdir(folder_path)
            if "photo.jpg" not in files and "photo.jpeg" not in files:
                print(f"Folder '{folder_name}' does not have 'photo.jpg' or 'photo.jpeg'.")

if __name__ == "__main__":
    base_folder_path = "C:\\Users\\cjbla\\OneDrive\\Desktop\\Code\data\\images_depth\\geoPose3K_final_publish"
    check_photo_file_presence(base_folder_path)