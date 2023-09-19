import os
import pandas as pd
import pymeanshift as pms
import cv2
import matplotlib.pyplot as plt
import numpy as np
import joblib
import json
import csv
import tensorflow.keras as keras

from ipywidgets import IntProgress
from sklearn.utils import shuffle
from IPython.display import display
from baseline import baseline_mean_shift, baseline_augmented, baseline_mean_shift_third, baseline_mean_shift_third_augmented
from constants import RELATIVE_IMAGES_CSV, RELATIVE_IMAGES_CSV_BD, EDEN_RGB_DIR, EDEN_DEPTH_DIR, RESULT_FOLDER

def make_image_csv(dataset_dir:str, output_csv_file:str):
    """
    Reads in a CSV file containing relative image paths and outputs a new CSV file containing the full image paths.

    Args:
        dataset_dir (str): The path to the input directory.
        output_csv_file (str): The path to the output CSV file.

    Returns:
        None
    """
    
    # read the CSV file
    df = pd.read_csv(RELATIVE_IMAGES_CSV)
    
    # add the dataset_dir to each element in the "image_path" column
    df['image_path'] = dataset_dir + "original\\" + df['image_path']
    df['label'] = dataset_dir + "original\\" + df['label']
    
    # output the result to "image_paths.csv"
    df.to_csv(output_csv_file, index=False, header=True, columns=['image_path', 'label'])

def make_image_csv_bd(dataset_dir:str, output_csv_file:str):
    """
    Reads in a CSV file containing relative image paths and outputs a new CSV file containing the full image paths.

    Args:
        dataset_dir (str): The path to the input directory.
        output_csv_file (str): The path to the output CSV file.

    Returns:
        None
    """
    
    # read the CSV file
    df = pd.read_csv(RELATIVE_IMAGES_CSV_BD)
    
    # add the dataset_dir to each element in the "image_path" column
    df['image_path'] = dataset_dir + "bachelor\\" + df['image_path']
    df['label'] = dataset_dir + "bachelor\\" + df['label']
    
    # output the result to "image_paths.csv"
    df.to_csv(output_csv_file, index=False, header=True, columns=['image_path', 'label'])

def make_cluster_separated_metadata(input_csv_file, columns, output_csv_file):
    """
    Reads in a CSV file containing image paths and labels, processes each image, and outputs a new CSV file with the
    processed data. The processed data includes the image path, label, average RGB values, number of pixels, centroid
    coordinates, ground truth, and predicted value for each region in the image.

    Args:
        input_csv_file (str): The path to the input CSV file.
        columns (list): A list of column names to use in the output CSV file.
        output_csv_file (str): The path to the output CSV file.
        regression (bool): Whether to use the regression model or the classification model.

    Returns:
        None
    """
    
    df_input = pd.read_csv(input_csv_file)
    results_df = pd.DataFrame(columns=columns)
    progress = IntProgress(min=0, max=len(df_input))
    display(progress)

    threshold = 0.5
    
    """scaler = joblib.load('scaler.joblib')

    if regression:
        model = keras.models.load_model('models/feature_regression_model.h5')
        threshold = 0.5
    else:
        model = keras.models.load_model('models/feature_classification_model.h5')
        threshold = 0.9"""

    for index, row in df_input.iterrows():
        print("Now processing image", index, "of", len(df_input), "...", end="\r")
        image = plt.imread(row['image_path'])
        label = plt.imread(row['label'], 0)  #type:ignore

        image = cv2.resize(image, (256, 256))
        label = cv2.resize(label, (256, 256))
        
        (segmented_image, labels_image, number_regions) = pms.segment(image, spatial_radius=2, range_radius=8, min_density=250)
        
        upper_labels = labels_image[0:labels_image.shape[0]//2, 0:labels_image.shape[1]]
        unique, counts = np.unique(upper_labels, return_counts=True)
        dominant_label = unique[np.argmax(counts)]
        
        for i in range(0, number_regions):
            mask = (labels_image == i).astype('uint8')
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            masked_label = cv2.bitwise_and(label, label, mask=mask)
            
            average_color = cv2.mean(masked_image, mask=mask)[:3]
            num_pixels = mask.sum()

            M = cv2.moments(mask)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            
            ground_truth = float((masked_label == 1).sum()/num_pixels)

            predicted_value = 1 if i == dominant_label else 0
            
            results_df = results_df.append({
                'image_path': row['image_path'],
                'label': i,
                'R' : average_color[0],
                'G' : average_color[1],
                'B' : average_color[2],
                'num_pixels': num_pixels,
                'cX': cX,
                'cY': cY,
                'ground_truth': ground_truth,
                'predicted_value': predicted_value
            }, ignore_index=True)  #type:ignore

        progress.value += 1
        
        if index > 0 and index % 10 == 0: #type:ignore
            results_df.to_csv(output_csv_file, mode='a', index=False, header=True, columns=columns)
            results_df = pd.DataFrame(columns=columns)
        
    results = pd.read_csv(output_csv_file)
    results = results[results['image_path'] != 'image_path']

    results.to_csv(output_csv_file, index=False, header=True, columns=columns)


def update_csv_with_paths_and_segmentation(input_csv_file, output_csv_file, spatial_radius=2, range_radius=8, min_density=250):
    """
    Reads in an input CSV file containing image paths, processes each image, and outputs a new CSV file with the
    processed data. The processed data includes the new image path and ground truth for each region in the image.

    Args:
        input_csv_file (str): The path to the input CSV file.
        output_csv_file (str): The path to the output CSV file.
        spatial_radius (int): The spatial radius parameter for the SLIC algorithm.
        range_radius (int): The range radius parameter for the SLIC algorithm.
        min_density (int): The minimum density parameter for the SLIC algorithm.

    Returns:
        None
    """
    
    df_input = pd.read_csv(input_csv_file)
    df_output = pd.read_csv(output_csv_file)

    new_df = pd.DataFrame(columns=['new_path', 'ground_truth'])
    new_csv_file = "updated_paths_and_segmentation.csv"
    progress = IntProgress(min=0, max=len(df_input))
    display(progress)
    
    for index, row in df_input.iterrows():
        image_path = row['image_path']
        new_path = image_path.replace('OriginalImages', 'LabelMaskedImages')

        save_dir = os.path.dirname(new_path)
        os.makedirs(save_dir, exist_ok=True)
        
        img = plt.imread(image_path)
        img = cv2.resize(img, (256, 256))
        
        _, labels_image, number_regions = pms.segment(img, spatial_radius=spatial_radius, range_radius=range_radius, min_density=min_density)
        
        for region_label in range(0, number_regions):
            mask = (labels_image == region_label).astype('uint8')
            masked_image = cv2.bitwise_and(img, img, mask=mask)

            save_path = os.path.splitext(new_path)[0] + f'_label_{region_label}.jpg'
            cv2.imwrite(save_path, masked_image)

            ground_truth = df_output.loc[(df_output['image_path'] == image_path) & (df_output['label'] == region_label), 'ground_truth'].values[0] #type:ignore
            new_df = new_df.append({'new_path': save_path, 'ground_truth': ground_truth}, ignore_index=True) #type:ignore

        if index > 0 and index % 10 == 0: #type:ignore
            new_df.to_csv(new_csv_file, mode='a', index=False, header=True, columns=['new_path', 'ground_truth'])
            new_df = pd.DataFrame(columns=['new_path', 'ground_truth'])

        progress.value += 1

    results = pd.read_csv(output_csv_file)
    results = results[results['new_path'] != 'new_path']
    results.to_csv(output_csv_file, index=False, header=True, columns=['new_path', 'ground_truth'])

#update_csv_with_paths_and_segmentation()

def check_for_corrupted_files(filepath):
    """
    Reads in a CSV file containing image paths and ground truth, checks each image to see if it is corrupted, and
    outputs a new CSV file with the processed data. The processed data includes the new image path and ground truth for
    each non-corrupted image.

    Args:
        None

    Returns:
        None
    """
    df = pd.read_csv(filepath)
    new_df = pd.DataFrame(columns=['new_path', 'ground_truth'])
    progress = IntProgress(min=0, max=len(df))
    display(progress)
    
    for _, row in df.iterrows():
        image_path = row['image_path']
        try:
            img = plt.imread(image_path)
        except:
            print(f"File {image_path} is corrupted")
            continue
        
        progress.value += 1

#check_for_corrupted_files()

def update_thresholds(df_output, threshold, lower_q, upper_q, threshold_file):
    """
    Update the threshold values in a JSON file with the 2.5th and 97.5th percentile values for the R, G, B, cX, and cY columns in a filtered dataframe.

    Args:
        df_output (pandas.DataFrame): The dataframe to filter and calculate the percentile values from.
        threshold (float): The threshold value to filter the dataframe by.
        lower_q (float): The lower quantile value to calculate the percentile values from.
        upper_q (float): The upper quantile value to calculate the percentile values from.
        threshold_file (str): The path to the JSON file to update the threshold values in.

    Returns:
        bool: True if the update was successful, False otherwise.
    """
    
    try:
        # Filter the dataframe to only include rows where ground_truth is greater than the threshold
        df_filtered = df_output[(df_output['ground_truth'] > threshold) & (df_output['cY'] < 128)]

        # Calculate the 2.5th and 97.5th percentile values for the R, G, B, cX, and cY columns in df_filtered
        percentiles = df_filtered[['label','R', 'G', 'B', 'cX', 'cY']].quantile(q=[lower_q, upper_q])

        # Create a dictionary of the threshold values
        thresholds = {
            "threshold": threshold,
            "lower_quantile": lower_q,
            "upper_quantile": upper_q,
            "label": percentiles['label'].tolist(),
            "R": percentiles['R'].tolist(),
            "G": percentiles['G'].tolist(),
            "B": percentiles['B'].tolist(),
            "cX": percentiles['cX'].tolist(),
            "cY": percentiles['cY'].tolist()
        }

        #print the percentage of cases in df_output that have ground_truth less than the threshold, cY greater than 128 and lie between the lower and upper quantiles
        percentage = (len(df_output[(df_output['ground_truth'] < threshold) & 
                            (df_output['cY'] > 128) & 
                            (df_output['R'] > thresholds['R'][0]) & 
                            (df_output['R'] < thresholds['R'][1]) & 
                            (df_output['G'] > thresholds['G'][0]) & 
                            (df_output['G'] < thresholds['G'][1]) & 
                            (df_output['B'] > thresholds['B'][0]) & 
                            (df_output['B'] < thresholds['B'][1]) & 
                            (df_output['cX'] > thresholds['cX'][0]) & 
                            (df_output['cX'] < thresholds['cX'][1])]) / 
              len(df_output[(df_output['R'] > thresholds['R'][0]) & 
                            (df_output['R'] < thresholds['R'][1]) & 
                            (df_output['G'] > thresholds['G'][0]) & 
                            (df_output['G'] < thresholds['G'][1]) & 
                            (df_output['B'] > thresholds['B'][0]) & 
                            (df_output['B'] < thresholds['B'][1]) & 
                            (df_output['cX'] > thresholds['cX'][0]) & 
                            (df_output['cX'] < thresholds['cX'][1])])) * 100

        print(f"{percentage}% of cases have ground_truth less than {threshold}, cY greater than 128, and lie between the lower and upper quantiles.")

        # Check if the file exists, and create it if it doesn't
        if not os.path.exists(threshold_file):
            with open(threshold_file, 'w') as f:
                json.dump({}, f)

        # Load the current thresholds from the file
        with open(threshold_file, 'r') as f:
            current_thresholds = json.load(f)

        # Update the current thresholds with the new values
        current_thresholds.update(thresholds)

        # Write the updated thresholds to the file
        with open(threshold_file, 'w') as f:
            json.dump(current_thresholds, f)

        # Print a message to confirm that the file was updated
        print(f"The {threshold_file} file has been updated with the label, R, G, B, cX, and cY values.")
        
        return True

    except FileNotFoundError:
        print(f"Error: The {threshold_file} file could not be found.")
        return False
    except json.JSONDecodeError:
        print(f"Error: The {threshold_file} file is not in valid JSON format.")
        return False
    except Exception as e:
        print(f"An error occurred while updating the {threshold_file} file: {e}")
        return False
    
def create_analysis_data(input_csv_file, output_csv_file, function):
    """
    Reads in a CSV file containing image paths and ground truth, and outputs a new CSV file with the metrics for that algorithm. 
    Only works with baseline variants.
    The processed data includes the new image path and the metrics for each non-corrupted image.

    Args:
        input_csv_file (str): The path to the CSV file containing the image paths and ground truth.
        output_csv_file (str): The path to the CSV file to output the processed data to.
        function (function): The function to use to process the data.

    Returns:
        None
    """
    df = pd.read_csv(input_csv_file)
    new_df = pd.DataFrame(columns=['image_path', 'precision', 'recall'])
    progress = IntProgress(min=0, max=len(df))
    display(progress)
    
    for _, row in df.iterrows():
        image = plt.imread(row['image_path'])
        label = cv2.cvtColor(cv2.imread(row['label']), cv2.COLOR_BGR2GRAY)
        try:
            if function == 'baseline':
                mask, _, _ = baseline_mean_shift(image)
            elif function == 'baseline_augmented':
                mask, _, _ = baseline_augmented(image)
            elif function == 'baseline_topthird':
                mask, _, _ = baseline_mean_shift_third(image)
            elif function == 'baseline_topthird_augmented':
                mask, _, _ = baseline_mean_shift_third_augmented(image)
            elif function == 'baseline_topthird_rr8':
                mask, _, _ = baseline_mean_shift_third(image, range_radius=8)

            # Calculate the precision and recall
            true_pos = np.sum(np.logical_and(label, mask))
            false_pos = np.sum(np.logical_and(np.logical_not(label), mask))
            false_neg = np.sum(np.logical_and(label, np.logical_not(mask)))
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)

            # Add the new data to the dataframe
            new_df = new_df.append({'image_path': row['image_path'], 'precision': precision, 'recall': recall}, ignore_index=True) #type: ignore
            
        except Exception as e:
            print(e)
            print(f"File {row['image_path']} is corrupted")
            continue
        
        progress.value += 1
        
    new_df.to_csv(output_csv_file, index=False, header=True, columns=['image_path', 'precision', 'recall'])

def create_depth_data_csv():
    """
    This function creates a CSV file containing the paths of RGB images and their corresponding depth maps.
    The paths are shuffled and saved to a new CSV file.

    Parameters:
    rgb_dir (str): The path to the directory containing the RGB images.
    depth_dir (str): The path to the directory containing the depth maps.
    results_folder (str): The path to the directory where the resulting CSV files will be saved.

    Returns:
    None
    """
    image_path = []
    label_path = []

    # Iterate through RGB directory and store paths of images with "_L" in their name
    for root, dirs, files in os.walk(EDEN_RGB_DIR):
        for file in files:
            if "_L" in file:
                image_path.append(os.path.join(root, file))

    # Iterate through Depth directory and store paths of .MAT files
    for root, dirs, files in os.walk(EDEN_DEPTH_DIR):
        for file in files:
            if file.endswith(".mat"):
                label_path.append(os.path.join(root, file))

    # Write paths to CSV file
    with open(os.path.join(RESULT_FOLDER, "depth_image_label_paths.csv"), mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "label_path"])
        for i in range(len(image_path)):
            writer.writerow([image_path[i], label_path[i]])

    #shuffle the values in depth_image_label_paths.csv and save the values in depth_image_label_paths_shuffled.csv
    df = pd.read_csv(os.path.join(RESULT_FOLDER, "depth_image_label_paths.csv"))
    df_shuffled = shuffle(df)
    df_shuffled.to_csv(os.path.join(RESULT_FOLDER, "depth_image_label_paths_shuffled.csv"), index=False)