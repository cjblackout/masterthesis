import pandas as pd
import seaborn as sns
import numpy as np

from sklearn.utils import resample
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple

from constants import DATASET_FOLDER

def preprocess_data(file_path:str, columns:list, plot: bool=False) -> pd.DataFrame:
    """
    Preprocesses the data from the cluster data CSV file and returns a pandas DataFrame.

    Parameters:
    file_path (str): The path to the CSV file.
    columns (List[str]): The list of column names to use for the DataFrame.
    plot (bool, optional): Whether to plot the distribution of the 'ground_truth' column. Defaults to False.

    Returns:
    pandas.DataFrame: The preprocessed DataFrame.

    Raises:
    FileNotFoundError: If the file_path does not exist.
    ValueError: If the columns list is empty or if the CSV file does not have the same number of columns as the columns list.
    TypeError: If the file_path or columns is not a string or list of strings, respectively.
    """
    
    try:
        dtypes = {'image_path':str, 'label': int, 'num_pixels': int, 'cX': int, 'cY': int, 'R': float, 'G': float, 'B': float, 'ground_truth': float, 'predicted_value': int}
        usecols = ['image_path','label', 'num_pixels', 'cX', 'cY', 'R', 'G', 'B', 'ground_truth', 'predicted_value']
        df_output = pd.read_csv(file_path, header=1, names=columns, dtype=dtypes, usecols=usecols)

        df_output.dropna(inplace=True)

        df_output = df_output.astype({'image_path':str, 'label': int, 'num_pixels': int, 'cX': int, 'cY': int, 'R': float, 'G': float, 'B': float, 'ground_truth': float, 'predicted_value': int})

        if plot:
            df_plot = df_output.query('ground_truth > 0')
            sns.displot(df_output['ground_truth'], bins=100)

        return df_output
    
    except FileNotFoundError:
        print(f"File not found at {file_path}")
        return pd.DataFrame() 

    except ValueError:
        print(f"Columns list is empty or CSV file does not have the same number of columns as the columns list")
        return pd.DataFrame()

    except TypeError:
        print(f"File path should be a string and columns should be a list of strings")
        return pd.DataFrame()

def calculate_metrics(df_output:pd.DataFrame, threshold:float, print_metrics: bool=False, pixel_level_calculation: bool=False) -> pd.DataFrame:
    """
    Calculates the true positives, false positives, true negatives, and false negatives for each row of a DataFrame based on a given threshold.
    Also calculates the precision, recall, and f1 score.

    Parameters:
    df_output (pandas.DataFrame): The DataFrame containing the ground truth and predicted values.
    threshold (float): The threshold value to use for determining true positives and false positives.
    print_metrics (bool): Whether to print the calculated metrics. Default is False.
    pixel_level_calculation (bool): Whether to calculate the metrics at the pixel level or at the cluster level. Default is False.

    Returns:
    pandas.DataFrame: A DataFrame containing the true positives, false positives, true negatives, and false negatives for each row of the input DataFrame.

    Raises:
    KeyError: If one or more columns are not found in the input DataFrame.
    TypeError: If the threshold value is not a float.
    """
    
    try:
        df_cases = pd.DataFrame(columns=['true_pos', 'false_pos', 'true_neg', 'false_neg'])

        #find the true positives, false positives, true negatives, and false negatives for each row of df where if ground_truth > 0.8, predicted_value = 1
        df_cases.loc[:, 'true_pos'] = np.where((df_output['ground_truth'] > threshold) & (df_output['predicted_value'] == 1), df_output['num_pixels'] if pixel_level_calculation else 1, 0)
        df_cases.loc[:, 'false_pos'] = np.where((df_output['ground_truth'] < threshold) & (df_output['predicted_value'] == 1), df_output['num_pixels'] if pixel_level_calculation else 1, 0)
        df_cases.loc[:, 'true_neg'] = np.where((df_output['ground_truth'] < threshold) & (df_output['predicted_value'] == 0), df_output['num_pixels'] if pixel_level_calculation else 1, 0)
        df_cases.loc[:, 'false_neg'] = np.where((df_output['ground_truth'] > threshold) & (df_output['predicted_value'] == 0), df_output['num_pixels'] if pixel_level_calculation else 1, 0)

        #plot the values of true_pos, false_pos, true_neg, and false_neg in a confusion matrix
        confusion_matrix = [df_cases['true_pos'].sum(), df_cases['false_pos'].sum(), df_cases['true_neg'].sum(), df_cases['false_neg'].sum()]

        #calculate the precision, recall, and f1 score
        precision = df_cases['true_pos'].sum() / (df_cases['true_pos'].sum() + df_cases['false_pos'].sum())
        recall = df_cases['true_pos'].sum() / (df_cases['true_pos'].sum() + df_cases['false_neg'].sum())
        f1 = 2 * (precision * recall) / (precision + recall)

        if print_metrics:
            print("True positive: {} ({})".format(confusion_matrix[0], "pixels" if pixel_level_calculation else "number of clusters, added"))
            print("False positive: {} ({})".format(confusion_matrix[1], "pixels" if pixel_level_calculation else "number of clusters, added"))
            print("True negative: {} ({})".format(confusion_matrix[2], "pixels" if pixel_level_calculation else "number of clusters, added"))
            print("False negative: {} ({})".format(confusion_matrix[3], "pixels" if pixel_level_calculation else "number of clusters, added"))
            
            print("Precision: {}".format(precision))
            print("Recall: {}".format(recall))
            print("F1: {}".format(f1))

        return df_cases

    except KeyError:
        print("One or more columns not found in DataFrame")
        return pd.DataFrame()

    except TypeError:
        print("Threshold should be a float")
        return pd.DataFrame()

def resample_df(df_output, threshold):
    """
    Resamples a DataFrame to balance the number of rows above and below a given threshold.

    Parameters:
    df_output (pandas.DataFrame): The DataFrame to resample.
    threshold (float): The threshold value to use for resampling.

    Returns:
    pandas.DataFrame: A resampled DataFrame with the same number of rows above and below the threshold.

    Raises:
    KeyError: If one or more columns are not found in the input DataFrame.
    TypeError: If the threshold value is not a float.
    """
    
    try:
        df_below_thresh = df_output[df_output['ground_truth'] < threshold]
        df_above_thresh = df_output[df_output['ground_truth'] >= threshold]

        n_samples = len(df_above_thresh)
        df_below_thresh_resampled = df_below_thresh.sample(n=n_samples, replace=False)
        df_resampled = pd.concat([df_below_thresh_resampled, df_above_thresh])

        return df_resampled

    except KeyError:
        print("One or more columns not found in the input DataFrame")
        return None

    except TypeError:
        print("Threshold value should be a float")
        return None

def split_data(df_output: pd.DataFrame, threshold: float, regression: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Splits the input DataFrame into training and testing sets, normalizes the data using MinMaxScaler,
    and sets the labels to 1 if the ground_truth value is greater than the threshold, and 0 otherwise for regression cases.

    Args:
        df_output (pd.DataFrame): The input DataFrame to split.
        threshold (float): The threshold value to use for setting the labels.
        regression (bool, optional): Whether to perform regression or classification. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: A tuple containing the training features,
        training labels, testing features, and testing labels as numpy arrays.
    """
    try:
        scaler = MinMaxScaler()

        df_dropped = df_output.drop(['label', 'predicted_value'], axis=1)
        df_dropped[['num_pixels', 'R', 'G', 'B', 'cX', 'cY']] = scaler.fit_transform(df_dropped[['num_pixels', 'R', 'G', 'B', 'cX', 'cY']])
        df_dropped = df_dropped.sample(frac=1).reset_index(drop=True)

        train_data = df_dropped.sample(frac=0.8, random_state=1)
        test_data = df_dropped.drop(train_data.index)

        if regression:
            train_labels = train_data.pop('ground_truth').values
            train_features = train_data.values

            test_labels = test_data.pop('ground_truth').values
            test_features = test_data.values

        else:
            train_labels = np.where(train_data['ground_truth'] > threshold, 1, 0)
            test_labels = np.where(test_data['ground_truth'] > threshold, 1, 0)

            train_features = train_data.drop('ground_truth', axis=1).values
            test_features = test_data.drop('ground_truth', axis=1).values

        return train_features, train_labels, test_features, test_labels, scaler #type: ignore

    except Exception as e:
        print(f"Error in split_data: {e}")
        return None, None, None, None, None #type: ignore