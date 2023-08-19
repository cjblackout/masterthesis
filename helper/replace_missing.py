import os
import csv
import shutil
import gzip

def copy_and_extract_missing_files(base_folder, dataset_folder):
    csv_file_path = "folders_with_missing_files.csv"
    if os.path.exists(csv_file_path):
        with open(csv_file_path, mode='r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                folder_name = row['Folder Name']
                missing_files = row['Missing Files'].split(', ')
                folder_path = os.path.join(base_folder, folder_name)
                for missing_file in missing_files:
                    src_file = os.path.join(dataset_folder,folder_name)
                    src_file = os.path.join(src_file, missing_file.strip())
                    dest_file = os.path.join(folder_path, missing_file.strip())
                    print(f"Copying {src_file} to {dest_file}")
                    shutil.copy(src_file.replace("/", "\\").replace("'", "") + ".gz", dest_file.replace("/", "\\").replace("'", "") + ".gz")

                    extracted_file = dest_file.replace("/", "\\").replace("'", "")
                    print(f"Extracting {dest_file} to {extracted_file}")
                    with gzip.open(dest_file.replace("/", "\\").replace("'", "") + ".gz", 'rb') as f_in, open(extracted_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(dest_file.replace("/", "\\").replace("'", "") + ".gz")

if __name__ == "__main__":
    base_folder = "C:\\Users\\cjbla\\OneDrive\\Desktop\\Code\\data\\images_depth\\geoPose3K_final_publish"
    dataset_folder = "C:\\Users\\cjbla\\Downloads\\dataset"

    copy_and_extract_missing_files(base_folder, dataset_folder)