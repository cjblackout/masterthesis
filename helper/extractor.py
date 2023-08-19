import os
import gzip
import shutil

def extract_gz_files(base_folder_path):
    for root, dirs, files in os.walk(base_folder_path):
        for filename in files:
            if filename.endswith(".gz"):
                gz_file_path = os.path.join(root, filename)
                extracted_file_path = gz_file_path.rstrip(".gz")
                print(f"Extracting: {gz_file_path} -> {extracted_file_path}")
                with gzip.open(gz_file_path, 'rb') as f_in, open(extracted_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(gz_file_path)

if __name__ == "__main__":
    base_folder_path = "C:\\Users\\cjbla\\OneDrive\\Desktop\\Code\data\\images_depth\\geoPose3K_final_publish"
    extract_gz_files(base_folder_path)