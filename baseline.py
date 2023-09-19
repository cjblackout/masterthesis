import time
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
import pymeanshift as pms

from constants import THRESHOLD, SPATIAL_RADIUS, RANGE_RADIUS, MIN_DENSITY, PROCESSING_SHAPE, LABEL_FOLDER

def baseline_mean_shift(img, processing_shape=PROCESSING_SHAPE, spatial_radius=SPATIAL_RADIUS, range_radius=RANGE_RADIUS, min_density=MIN_DENSITY):
    """
    Applies baseline mean shift segmentation to an input image and generates a mask for the dominant region on the top half of the image.

    Parameters:
        img (numpy.ndarray): The input image to be segmented.
        processing_shape (tuple, optional): The shape to resize the input image for processing. Defaults to PROCESSING_SHAPE.
        spatial_radius (int, optional): The spatial radius parameter for mean shift segmentation. Controls the size of the spatial neighborhood. Defaults to SPATIAL_RADIUS.
        range_radius (int, optional): The range radius parameter for mean shift segmentation. Controls the size of the color range neighborhood. Defaults to RANGE_RADIUS.
        min_density (int, optional): The minimum density parameter for mean shift segmentation. Controls the minimum number of pixels in a region. Defaults to MIN_DENSITY.

    Returns:
        mask (numpy.ndarray): The binary mask representing the dominant region (e.g., sky) in the original image shape.
        segmented_image (numpy.ndarray): The segmented image with labeled regions.
        labels_image (numpy.ndarray): The image with labels corresponding to each region.

    Raises:
        ValueError: If the input image is not a valid numpy.ndarray.
        Exception: If an error occurs during mean shift segmentation.

    Example usage:
        try:
            img = cv2.imread('image.jpg')
            mask, segmented_image, labels_image = baseline_mean_shift(img)
        except ValueError as ve:
            print(f"Invalid input image: {ve}")
        except Exception as e:
            print(f"Error occurred during mean shift segmentation: {e}")

    """

    try:
        # Check if the input image is a valid numpy.ndarray
        if not isinstance(img, np.ndarray):
            raise ValueError("Input image must be a numpy.ndarray")

        # Save the original shape of the image, resize it, and segment the image
        original_shape = img.shape
        img = cv2.resize(img, processing_shape)

        #perform mean shift segmentation
        try:
            (segmented_image, labels_image, number_regions) = pms.segment(img, spatial_radius=spatial_radius, range_radius=range_radius, min_density=min_density)
        except Exception as e:
            error_message1 = "Error occurred during sky detection: " + str(e)
            raise Exception(error_message1) from e

        # Take the upper half of labels_image and determine the most dominant label in the upper half
        upper_labels = labels_image[0:labels_image.shape[0]//2, 0:labels_image.shape[1]]
        unique, counts = np.unique(upper_labels, return_counts=True)
        dominant_label = unique[np.argmax(counts)]

        # Create a mask from labels_image where the dominant_label represents the dominant region (e.g., sky) and resize it to the original shape
        mask = np.zeros(labels_image.shape, dtype=np.uint8)
        mask[labels_image == dominant_label] = 1
        mask = cv2.resize(mask, (original_shape[1], original_shape[0]))

        return mask, segmented_image, labels_image

    except ValueError as ve:
        raise ve
    except Exception as e:
        error_message2 = "Error occurred during sky detection: " + str(e)
        raise Exception(error_message2) from e
    
def baseline_mean_shift_third(img, processing_shape=PROCESSING_SHAPE, spatial_radius=SPATIAL_RADIUS, range_radius=RANGE_RADIUS, min_density=MIN_DENSITY):
    """
    Applies baseline mean shift segmentation to an input image and generates a mask for the dominant region on the top third of the image.

    Parameters:
        img (numpy.ndarray): The input image to be segmented.
        processing_shape (tuple, optional): The shape to resize the input image for processing. Defaults to PROCESSING_SHAPE.
        spatial_radius (int, optional): The spatial radius parameter for mean shift segmentation. Controls the size of the spatial neighborhood. Defaults to SPATIAL_RADIUS.
        range_radius (int, optional): The range radius parameter for mean shift segmentation. Controls the size of the color range neighborhood. Defaults to RANGE_RADIUS.
        min_density (int, optional): The minimum density parameter for mean shift segmentation. Controls the minimum number of pixels in a region. Defaults to MIN_DENSITY.

    Returns:
        mask (numpy.ndarray): The binary mask representing the dominant region (e.g., sky) in the original image shape.
        segmented_image (numpy.ndarray): The segmented image with labeled regions.
        labels_image (numpy.ndarray): The image with labels corresponding to each region.

    Raises:
        ValueError: If the input image is not a valid numpy.ndarray.
        Exception: If an error occurs during mean shift segmentation.

    Example usage:
        try:
            img = cv2.imread('image.jpg')
            mask, segmented_image, labels_image = baseline_mean_shift(img)
        except ValueError as ve:
            print(f"Invalid input image: {ve}")
        except Exception as e:
            print(f"Error occurred during mean shift segmentation: {e}")

    """

    try:
        # Check if the input image is a valid numpy.ndarray
        if not isinstance(img, np.ndarray):
            raise ValueError("Input image must be a numpy.ndarray")

        # Save the original shape of the image, resize it, and segment the image
        original_shape = img.shape
        img = cv2.resize(img, processing_shape)

        #perform mean shift segmentation
        try:
            (segmented_image, labels_image, number_regions) = pms.segment(img, spatial_radius=spatial_radius, range_radius=range_radius, min_density=min_density)
        except Exception as e:
            raise Exception("Error occurred during mean shift segmentation") from e

        # Take the upper third of labels_image and determine the most dominant label in the upper third
        upper_labels = labels_image[0:labels_image.shape[0]//3, 0:labels_image.shape[1]]
        unique, counts = np.unique(upper_labels, return_counts=True)
        dominant_label = unique[np.argmax(counts)]

        # Create a mask from labels_image where the dominant_label represents the dominant region (e.g., sky) and resize it to the original shape
        mask = np.zeros(labels_image.shape, dtype=np.uint8)
        mask[labels_image == dominant_label] = 1
        mask = cv2.resize(mask, (original_shape[1], original_shape[0]))

        return mask, segmented_image, labels_image

    except ValueError as ve:
        raise ve
    except Exception as e:
        raise Exception("Error occurred during sky detection") from e
        
def baseline_augmented(img, processing_shape=PROCESSING_SHAPE, spatial_radius=SPATIAL_RADIUS, range_radius=RANGE_RADIUS, min_density=MIN_DENSITY):    
    """
    Applies baseline mean shift segmentation to an input image and generates a mask from the 96% quantile of R, G, B, cX and cY values.

    Parameters:
        img (numpy.ndarray): The input image to be segmented.
        processing_shape (tuple, optional): The shape to resize the input image for processing. Defaults to PROCESSING_SHAPE.
        spatial_radius (int, optional): The spatial radius parameter for mean shift segmentation. Controls the size of the spatial neighborhood. Defaults to SPATIAL_RADIUS.
        range_radius (int, optional): The range radius parameter for mean shift segmentation. Controls the size of the color range neighborhood. Defaults to RANGE_RADIUS.
        min_density (int, optional): The minimum density parameter for mean shift segmentation. Controls the minimum number of pixels in a region. Defaults to MIN_DENSITY.

    Returns:
        mask (numpy.ndarray): The binary mask representing the dominant region (e.g., sky) in the original image shape.
        segmented_image (numpy.ndarray): The segmented image with labeled regions.
        labels_image (numpy.ndarray): The image with labels corresponding to each region.

    Raises:
        ValueError: If the input image is not a valid numpy.ndarray.
        Exception: If an error occurs during mean shift segmentation.

    Example usage:
        try:
            img = cv2.imread('image.jpg')
            mask, segmented_image, labels_image = baseline_mean_shift(img)
        except ValueError as ve:
            print(f"Invalid input image: {ve}")
        except Exception as e:
            print(f"Error occurred during mean shift segmentation: {e}")

    """
    
    try:
        # Check if the input image is a valid numpy.ndarray
        if not isinstance(img, np.ndarray):
            raise ValueError("Input image must be a numpy.ndarray")

        # Save the original shape of the image, resize it, and segment the image
        original_shape = img.shape
        img = cv2.resize(img, processing_shape)

        with open('threshold.json', 'r') as f:
            threshold_data = json.load(f)

        # Create dictionaries for the upper and lower limits
        lower_limits = {'R': threshold_data['R'][0], 'G': threshold_data['G'][0], 'B': threshold_data['B'][0], 'cX': threshold_data['cX'][0], 'cY': threshold_data['cY'][0], 'label': threshold_data['label'][0]}
        upper_limits = {'R': threshold_data['R'][1], 'G': threshold_data['G'][1], 'B': threshold_data['B'][1], 'cX': threshold_data['cX'][1], 'cY': threshold_data['cY'][1], 'label': threshold_data['label'][1]}
        
        #perform mean shift segmentation
        try:
            (segmented_image, labels_image, number_regions) = pms.segment(img, spatial_radius=spatial_radius, range_radius=range_radius, min_density=min_density)
        except Exception as e:
            raise Exception("Error occurred during mean shift segmentation") from e
        
        final_mask = np.zeros(img.shape[:2], dtype='uint8')
        
        for i in range(0, number_regions):
            mask = (labels_image == i).astype('uint8')
            masked_image = cv2.bitwise_and(img, img, mask=mask)

            average_color = cv2.mean(masked_image, mask=mask)[:3]

            M = cv2.moments(mask)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            if (i >= int(lower_limits['label']) and i <= int(upper_limits['label']) and
                average_color[0] >= lower_limits['R'] and average_color[0] <= upper_limits['R'] and
                average_color[1] >= lower_limits['G'] and average_color[1] <= upper_limits['G'] and
                average_color[2] >= lower_limits['B'] and average_color[2] <= upper_limits['B'] and
                cX >= lower_limits['cX'] and cX <= upper_limits['cX'] and
                cY >= lower_limits['cY'] and cY <= upper_limits['cY']):
                final_mask = cv2.bitwise_or(final_mask, mask)
            else:
                pass

        mask = cv2.resize(final_mask, (original_shape[1], original_shape[0]))

        return mask, segmented_image, labels_image

    except ValueError as ve:
        raise ve
    except Exception as e:
        raise Exception("Error occurred during sky detection") from e
    
def baseline_mean_shift_third_augmented(img, processing_shape=PROCESSING_SHAPE, spatial_radius=SPATIAL_RADIUS, range_radius=RANGE_RADIUS, min_density=MIN_DENSITY):
    """
    Applies baseline mean shift segmentation to an input image and generates a mask for the dominant region on the top third of the image.

    Parameters:
        img (numpy.ndarray): The input image to be segmented.
        processing_shape (tuple, optional): The shape to resize the input image for processing. Defaults to PROCESSING_SHAPE.
        spatial_radius (int, optional): The spatial radius parameter for mean shift segmentation. Controls the size of the spatial neighborhood. Defaults to SPATIAL_RADIUS.
        range_radius (int, optional): The range radius parameter for mean shift segmentation. Controls the size of the color range neighborhood. Defaults to RANGE_RADIUS.
        min_density (int, optional): The minimum density parameter for mean shift segmentation. Controls the minimum number of pixels in a region. Defaults to MIN_DENSITY.

    Returns:
        mask (numpy.ndarray): The binary mask representing the dominant region (e.g., sky) in the original image shape.
        segmented_image (numpy.ndarray): The segmented image with labeled regions.
        labels_image (numpy.ndarray): The image with labels corresponding to each region.

    Raises:
        ValueError: If the input image is not a valid numpy.ndarray.
        Exception: If an error occurs during mean shift segmentation.

    Example usage:
        try:
            img = cv2.imread('image.jpg')
            mask, segmented_image, labels_image = baseline_mean_shift(img)
        except ValueError as ve:
            print(f"Invalid input image: {ve}")
        except Exception as e:
            print(f"Error occurred during mean shift segmentation: {e}")

    """

    try:
        # Check if the input image is a valid numpy.ndarray
        if not isinstance(img, np.ndarray):
            raise ValueError("Input image must be a numpy.ndarray")

        # Save the original shape of the image, resize it, and segment the image
        original_shape = img.shape
        img = cv2.resize(img, processing_shape)

        with open('threshold.json', 'r') as f:
            threshold_data = json.load(f)

        # Create dictionaries for the upper and lower limits
        lower_limits = {'R': threshold_data['R'][0], 'G': threshold_data['G'][0], 'B': threshold_data['B'][0], 'cX': threshold_data['cX'][0], 'cY': threshold_data['cY'][0], 'label': threshold_data['label'][0]}
        upper_limits = {'R': threshold_data['R'][1], 'G': threshold_data['G'][1], 'B': threshold_data['B'][1], 'cX': threshold_data['cX'][1], 'cY': threshold_data['cY'][1], 'label': threshold_data['label'][1]}
        
        #perform mean shift segmentation
        try:
            (segmented_image, labels_image, number_regions) = pms.segment(img, spatial_radius=spatial_radius, range_radius=range_radius, min_density=min_density)
        except Exception as e:
            raise Exception("Error occurred during mean shift segmentation") from e

        # Take the upper third of labels_image and determine the most dominant label in the upper third
        upper_labels = labels_image[0:labels_image.shape[0]//3, 0:labels_image.shape[1]]
        unique, counts = np.unique(upper_labels, return_counts=True)
        dominant_label = unique[np.argmax(counts)]

        final_mask = np.zeros(img.shape[:2], dtype='uint8')
        
        for i in range(0, number_regions):
            mask = (labels_image == i).astype('uint8')
            masked_image = cv2.bitwise_and(img, img, mask=mask)

            average_color = cv2.mean(masked_image, mask=mask)[:3]

            M = cv2.moments(mask)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            if (average_color[0] >= lower_limits['R'] and average_color[0] <= upper_limits['R'] and
                average_color[1] >= lower_limits['G'] and average_color[1] <= upper_limits['G'] and
                average_color[2] >= lower_limits['B'] and average_color[2] <= upper_limits['B'] and
                cX >= lower_limits['cX'] and cX <= upper_limits['cX'] and
                cY >= 85 and cY <= 128):
                final_mask = cv2.bitwise_or(final_mask, mask)
            else:
                pass

        # Create a mask from labels_image where the dominant_label represents the dominant region (e.g., sky) and resize it to the original shape
        mask = np.zeros(labels_image.shape, dtype=np.uint8)
        mask[labels_image == dominant_label] = 1

        #combine mask and final_mask
        mask = cv2.bitwise_or(mask, final_mask)
        mask = cv2.resize(mask, (original_shape[1], original_shape[0]))

        return mask, segmented_image, labels_image

    except ValueError as ve:
        raise ve
    except Exception as e:
        raise Exception("Error occurred during sky detection") from e

def display_results(filename, function, print_mode='display', dataset_mode='original', spatial_radius=SPATIAL_RADIUS, 
                    range_radius=RANGE_RADIUS, min_density=MIN_DENSITY):
    """
    Display the segmented image and original image side by side along with other information such as processing time,
    precision, recall, and F1-score. If `dataset_mode` is set to 'validate', the function also calculates and displays 
    the precision, recall, and F1-score. 

    Args:
    - filename (str): the filename of the image to be processed.
    - print_mode (str): 'display' to display the images or 'silent' to suppress the display.
    - dataset_mode (str): 'original' to process the image as part of the original dataset, or 'validate' to process the 
                          image as part of the validation dataset and calculate the precision, recall, and F1-score.
    - spatial_radius (int): the spatial radius used in the mean shift algorithm. Default is SPATIAL_RADIUS.
    - range_radius (int): the range radius used in the mean shift algorithm. Default is RANGE_RADIUS.
    - min_density (int): the minimum density used in the mean shift algorithm. Default is MIN_DENSITY.

    Returns:
    - If `dataset_mode` is set to 'validate', the function returns a list containing the precision, recall, F1-score, 
      filename, time taken, spatial radius, range radius, and minimum density.
    - If `dataset_mode` is set to 'original', the function returns a list containing 0s for precision, recall, and F1-score,
      filename, time taken, spatial radius, range radius, and minimum density.
    """

    try:
        start_time = time.time()

        img = plt.imread(filename)
        mask, segmented_image, labels_image = function(img, spatial_radius=spatial_radius, range_radius=range_radius, min_density=min_density)
        time_taken = time.time() - start_time

        after_img = cv2.bitwise_and(img, img, mask=mask)

        # Display the images in 2 subplots 
        if print_mode == 'display':
            print('------------------------------------------------------------------------------------------------------------------------')
            print("Processing: " + filename.split("\\")[-2] + ", spatial_radius: " + str(spatial_radius) + 
                  ", range_radius: " + str(range_radius) + ", min_density: " + str(min_density))
            print('Time taken: ', time_taken)

            fig, axs = plt.subplots(1, 3, figsize=(17, 10))
            fig.suptitle('Sky Segmentation', fontsize=22)

            axs[0].imshow(plt.imread(filename))
            axs[0].set_title('Original Image', fontsize=18)
            axs[0].axis('off')

            # Resize the segmented image to the same size as after_img
            segmented_image_resized = cv2.resize(segmented_image, after_img.shape[:2][::-1])

            axs[1].imshow(segmented_image_resized)
            axs[1].set_title('Segmented Image', fontsize=18)
            axs[1].axis('off')

            axs[2].imshow(after_img)
            axs[2].set_title('Result', fontsize=18)
            axs[2].axis('off')

            plt.show()

            if dataset_mode == 'validate':
                val_image = cv2.imread(LABEL_FOLDER + filename.split("\\")[-2] + ".png")
                val_image = cv2.cvtColor(val_image, cv2.COLOR_BGR2GRAY)

                # Calculate the precision and recall
                true_pos = np.sum(np.logical_and(val_image, mask))
                false_pos = np.sum(np.logical_and(np.logical_not(val_image), mask))
                false_neg = np.sum(np.logical_and(val_image, np.logical_not(mask)))
                precision = true_pos / (true_pos + false_pos)
                recall = true_pos / (true_pos + false_neg)
                f1 = 2 * (precision * recall) / (precision + recall)

                print("Precision: " + str(precision))
                print("Recall: " + str(recall))
                print("F1: " + str(f1))

                print('------------------------------------------------------------------------------------------------------------------------')
                return [precision, recall, f1, filename, time_taken, spatial_radius, range_radius, min_density]

            elif dataset_mode == 'original':
                print('------------------------------------------------------------------------------------------------------------------------')
                return [0, 0, 0, filename, time_taken, spatial_radius, range_radius, min_density]

        else:
            if dataset_mode == 'validate':
                val_image = cv2.imread(LABEL_FOLDER + filename.split("\\")[-2] + ".png")
                val_image = cv2.cvtColor(val_image, cv2.COLOR_BGR2GRAY)

                # Calculate the precision and recall
                true_pos = np.sum(np.logical_and(val_image, mask))
                false_pos = np.sum(np.logical_and(np.logical_not(val_image), mask))
                false_neg = np.sum(np.logical_and(val_image, np.logical_not(mask)))
                precision = true_pos / (true_pos + false_pos)
                recall = true_pos / (true_pos + false_neg)
                f1 = 2 * (precision * recall) / (precision + recall)

                return [precision, recall, f1, filename, time_taken, spatial_radius, range_radius, min_density]

            elif dataset_mode == 'original':
                return [0, 0, 0, filename, time_taken, spatial_radius, range_radius, min_density]

    except Exception as e:
        print("An error occurred:", str(e))