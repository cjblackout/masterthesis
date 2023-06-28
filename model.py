import cv2
import pandas as pd
import pymeanshift as pms
import matplotlib.pyplot as plt
import numpy as np

from typing import List
from tensorflow import keras
from tensorflow.keras import layers, models #type: ignore

def feature_regression_model(input_shape):
    # Define the input layer
    inputs = keras.Input(shape=input_shape)

    # Define the hidden layers
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(16, activation='relu')(x)

    # Define the output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def feature_classification_model(input_shape):
    # Define the input layer
    inputs = keras.Input(shape=input_shape)

    # Define the hidden layers
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(16, activation='relu')(x)

    # Define the output layer
    outputs = layers.Dense(1, activation='sigmoid')(x)

    # Create the model
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def predict_ground_truth(image_path: str, feature_regression_model: models, scaler, threshold: float, verbose=0) -> List[float]:

    image = cv2.resize(plt.imread(image_path), (256, 256))
    test_df = pd.DataFrame(columns=['num_pixels', 'R', 'G', 'B', 'cX', 'cY'])
    (segmented_image, labels_image, number_regions) = pms.segment(image, spatial_radius=2, range_radius=8, min_density=250)

    for i in range(0, number_regions):

        mask = (labels_image == i).astype('uint8')
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        average_color = cv2.mean(masked_image, mask=mask)[:3]
        num_pixels = mask.sum()

        M = cv2.moments(mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

        test_df = test_df.append({
            'num_pixels': num_pixels,
            'R': average_color[0],
            'G': average_color[1],
            'B': average_color[2],
            'cX': cX,
            'cY': cY,
        }, ignore_index=True) #type:ignore

    test_df[['num_pixels', 'R', 'G', 'B', 'cX', 'cY']] = scaler.fit_transform(test_df[['num_pixels', 'R', 'G', 'B', 'cX', 'cY']])
    prediction = feature_regression_model.predict(test_df.values)

    masked_image = image
    mask = np.zeros(image.shape[:2], dtype="uint8")
    for i in range(0, len(prediction)):
        if prediction[i][0] > threshold:
            if verbose == 2:
                print(f"Prediction: {prediction[i]}")
                print(f"Index: {i}")
                print(f"Number of pixels: {test_df['num_pixels'][i]}")
                print(f"Average color: {test_df['R'][i], test_df['G'][i], test_df['B'][i]}")
                print(f"Image: {image_path}")

            mask = cv2.bitwise_or(mask, (labels_image == i).astype('uint8'))
            
            # Apply the mask to the original image and the label
            masked_image = cv2.bitwise_and(image, image, mask=mask)

    if verbose==1 or verbose==2:
        plt.imshow(masked_image)
        plt.show()

    return prediction.tolist()