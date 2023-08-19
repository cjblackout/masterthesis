import os
import shutil

def delete_files_except_specified(folder_path):
    # List of file formats to exclude
    excluded_formats = {
        "photo.jpeg",
	    "photo.jpg",
        "cyl\distance_crop.pfm.gz",
        "pinhole\distance_crop.pfm.gz",
    }
    
    for folder_name in os.listdir(folder_path):
        folder_dir = os.path.join(folder_path, folder_name)
        if os.path.isdir(folder_dir):
            for root, dirs, files in os.walk(folder_dir):
                for filename in files:
                    file_path = os.path.join(root, filename)
                    rel_path = os.path.relpath(file_path, folder_path)
                    rel_path = "\\".join(rel_path.split("\\")[1:])  # Exclude the first directory (folder_name)
                    if rel_path not in excluded_formats:
                        #print("Deleting:", file_path)
                        os.remove(file_path)
                        #print ("Deleted:", file_path)

if __name__ == "__main__":
    base_folder_path = "C:\\Users\\cjbla\\OneDrive\\Desktop\\Code\\data\\images_depth\\geoPose3K_final_publish"
    delete_files_except_specified(base_folder_path)