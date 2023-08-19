import os
import csv

def check_files_presence(base_folder):
    folders_with_missing_files = []
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            photo_jpg_path = os.path.join(folder_path, "photo.jpg")
            photo_jpeg_path = os.path.join(folder_path, "photo.jpeg")
            cyl_pfm_path = os.path.join(folder_path, "cyl", "distance_crop.pfm")
            pinhole_pfm_path = os.path.join(folder_path, "pinhole", "distance_crop.pfm")
            
            missing_files = []
            if not (os.path.isfile(photo_jpg_path) or os.path.isfile(photo_jpeg_path)):
                missing_files.append("'photo.jpg' or 'photo.jpeg'")
            if not os.path.isfile(cyl_pfm_path):
                missing_files.append("'cyl/distance_crop.pfm'")
            if not os.path.isfile(pinhole_pfm_path):
                missing_files.append("'pinhole/distance_crop.pfm'")
            
            if missing_files:
                folders_with_missing_files.append([folder_name, ', '.join(missing_files)])

    if folders_with_missing_files:
        csv_file_path = "folders_with_missing_files.csv"
        with open(csv_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Folder Name", "Missing Files"])
            writer.writerows(folders_with_missing_files)

if __name__ == "__main__":
    base_folder = "C:\\Users\\cjbla\\OneDrive\\Desktop\\Code\\data\\images_depth\\geoPose3K_final_publish"
    check_files_presence(base_folder)