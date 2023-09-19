import cv2
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import time
import seaborn as sns
import tensorflow as tf

from scipy.io import loadmat
from sklearn.metrics import mean_squared_error
from typing import List
from tensorflow import keras
from tensorflow.keras import layers, models #type: ignore
from transformers import DPTImageProcessor, DPTForDepthEstimation, logging
from PIL import Image
from tqdm import tqdm

from baseline import baseline_mean_shift_third
from constants import RESULT_FOLDER

def create_occlusion(image_path):
    """
    This function creates an occlusion map for a given RGB image using a pre-trained depth estimation model.
    The function also creates a baseline occlusion map using a mean shift algorithm.
    The function returns the occlusion maps and the time taken to create them.

    Parameters:
    image_path (str): The path to the RGB image.

    Returns:
    depth_masked_model (numpy.ndarray): The occlusion map created using the pre-trained depth estimation model.
    depth_masked_baseline (numpy.ndarray): The baseline occlusion map created using a mean shift algorithm.
    depth_inverted (PIL.Image.Image): The depth map of the image with inverted values.
    time_taken (float): The time taken to create the occlusion maps.
    """
    start = time.time()
    image = Image.open(image_path)

    processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
    model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)

    #invert the values of depth
    depth_inverted = np.invert(depth)
    depth_inverted = Image.fromarray(depth_inverted)

    # assuming `image` is a PIL Image object, convert it to a numpy array
    image = np.array(image)
    sky_mask_baseline, _, _ = baseline_mean_shift_third(image)

    model = keras.models.load_model("models/model_30000_5_optimized.h5")

    image_model = tf.io.read_file(image_path)
    image_model = tf.image.decode_jpeg(image_model, channels=3)

    #get the shape of the image
    original_shape = image_model.shape

    image_model = tf.image.resize(image_model, [256, 256], method='nearest')
    image_model = tf.image.convert_image_dtype(image_model, tf.float32)
    image_model = tf.expand_dims(image_model, axis=0)

    #reshape the prediction to the original shape
    prediction = model.predict(image_model, verbose = 0)
    prediction = tf.image.resize(prediction, [original_shape[0], original_shape[1]], method='nearest')

    #make a mask from the prediction where the value of the element is greater than 0.5 is 1 and 0 otherwise
    sky_mask_model = np.where(prediction[0] > 0.6, 1, 0)
    sky_mask_model = sky_mask_model.reshape(original_shape[0], original_shape[1])

    #assuming `mask` and `depth_inverted` are defined, use mask on the depth image
    depth_masked_model = np.ma.masked_array(depth_inverted, mask=sky_mask_model)
    depth_masked_model = depth_masked_model / np.max(depth_masked_model)
    depth_masked_model = depth_masked_model.filled(1.0)

    #assuming `mask` and `depth_inverted` are defined, use mask on the depth image
    depth_masked_baseline = np.ma.masked_array(depth_inverted, mask=sky_mask_baseline)
    depth_masked_baseline = depth_masked_baseline / np.max(depth_masked_baseline)
    depth_masked_baseline = depth_masked_baseline.filled(1.0)

    depth_inverted = depth_inverted / np.max(depth_inverted)

    end = time.time()

    return depth_masked_model, depth_masked_baseline, depth_inverted, end - start

def compare_rmse(image_path, label_path):
    """
    This function compares the RMSE of a model's depth estimation with the RMSE of a baseline and inverted depth map.

    Parameters:
    image_path (str): The path to the RGB image used for depth estimation.
    label_path (str): The path to the .MAT file containing the ground truth depth map.

    Returns:
    rmse_model (float): The RMSE of the model's depth estimation.
    rmse_baseline (float): The RMSE of the baseline depth estimation.
    rmse_inverted (float): The RMSE of the inverted depth estimation.
    time_taken (float): The time taken to perform the depth estimation.
    """
    tf.get_logger().setLevel('ERROR')
    logging.set_verbosity_error()

    # Load the .mat file
    label = loadmat(label_path)
    label = label['Depth']

    #normalize the values in depth_data between 0 and 1
    label = label / label.max()

    #changes the values where the depth is exactly 0.0 to 1.0
    label[label == 0.0] = 1.0
    label = tf.squeeze(label)

    depth_masked_model, depth_masked_baseline, depth_inverted, time_taken = create_occlusion(image_path)

    rmse_model = mean_squared_error(depth_masked_model, label, squared=False)
    rmse_baseline = mean_squared_error(depth_masked_baseline, label, squared=False)
    rmse_inverted = mean_squared_error(depth_inverted, label, squared=False)

    return rmse_model, rmse_baseline, rmse_inverted, time_taken

def compare_depth_estimation():
    """
    This function compares the RMSE of a model's depth estimation with the RMSE of a baseline and inverted depth map.
    The results are saved to a CSV file.

    Parameters:
    None

    Returns:
    None
    """
    df = pd.read_csv(RESULT_FOLDER + "depth_image_label_paths_shuffled.csv")

    # create empty dataframe to store results
    results_df = pd.DataFrame(columns=['image_path', 'label_path', 'rmse_model', 'rmse_baseline', 'rmse_inverted', 'time_taken'])

    # loop through rows of df
    for i, row in tqdm(df.iterrows()):
        # compare RMSE
        rmse_model, rmse_baseline, rmse_inverted, time_taken = compare_rmse(row['image_path'], row['label_path'])

        # add results to results_df
        results_df.loc[i] = [row['image_path'], row['label_path'], rmse_model, rmse_baseline, rmse_inverted, time_taken]

        if i % 10 == 0:
            # save results_df to depth_image_label_paths.csv
            results_df.to_csv(RESULT_FOLDER + 'dpt_evaluation.csv', mode="a", index=False, header=False)
            results_df = pd.DataFrame(columns=['image_path', 'label_path', 'rmse_model', 'rmse_baseline', 'rmse_inverted', 'time_taken'])

    results_df.to_csv(RESULT_FOLDER + "dpt_evaluation.csv", mode="a", index=False, header=False)