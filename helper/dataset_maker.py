import os
import csv

def create_dataset_csv(base_folder, csv_filename):
    dataset = []
    missing_paths = []

    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)
        if os.path.isdir(folder_path):
            image_path = os.path.join(folder_path, "photo.jpg")
            if not os.path.isfile(image_path):
                image_path = os.path.join(folder_path, "photo.jpeg")

            label_path = os.path.join(folder_path, "cyl", "distance_crop.pfm")

            if os.path.isfile(label_path):
                dataset.append([image_path, label_path])
            else:
                missing_paths.append(folder_name)

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Image Path", "Label Path"])
        writer.writerows(dataset)

    if missing_paths:
        print("The following folders have missing image_path or label_path:")
        for folder in missing_paths:
            print(folder)

if __name__ == "__main__":
    base_folder = "C:\\Users\\cjbla\\OneDrive\\Desktop\\Code\\data\\images_depth\\geoPose3K_final_publish"
    csv_filename = "dataset.csv"
    create_dataset_csv(base_folder, csv_filename)