THRESHOLD = 0.5
SPATIAL_RADIUS = 2
RANGE_RADIUS = 2
MIN_DENSITY = 150
PROCESSING_SHAPE = (256,256)

#Set this according to the place where the dataset is stored
DATASET_FOLDER = "C:\\Users\\cjbla\\OneDrive\\Desktop\\Code\\data\\dataset\\"
RESULT_FOLDER = "C:\\Users\\cjbla\\OneDrive\\Desktop\\Code\\data\\"
RELATIVE_IMAGES_CSV = "relative_image_paths_main.csv"
RELATIVE_IMAGES_CSV_BD = "relative_image_bd.csv"

ORIGINAL_IMAGES_FOLDER = DATASET_FOLDER + "OriginalImages\\"
LABEL_FOLDER = DATASET_FOLDER + "ValidationImages\\Skyfinder\\"

ORIGINAL_IMAGES_CSV = RESULT_FOLDER + "image_paths_original.csv"
BACHELOR_IMAGES_CSV = RESULT_FOLDER + "image_paths_bachelor.csv"

BASELINE_METRIC_CSV_FILE = RESULT_FOLDER + "baseline_metrics.csv"
BASELINE_AUGMENTED_METRIC_CSV_FILE = RESULT_FOLDER + "baseline_augmented_metrics.csv"
BASELINE_TOPTHIRD_METRIC_CSV_FILE = RESULT_FOLDER + "baseline_topthird_metrics.csv"
BASELINE_TOPTHIRD_RR8_METRIC_CSV_FILE = RESULT_FOLDER + "baseline_topthird_rr8_metrics.csv"
BASELINE_TOPTHIRD_AUGMENTED_METRIC_CSV_FILE = RESULT_FOLDER + "baseline_topthird_augmented_metrics.csv"

CLUSTER_SEPARATED_METADATA_CSV_FILE = RESULT_FOLDER + "cluster_separated_metadata.csv"
CLUSTER_SEPARATED_PATH_LABEL_CSV_FILE = RESULT_FOLDER + "cluster_separated_path_label.csv"
CLUSTER_REGRESSION_CSV_FILE = RESULT_FOLDER + "cluster_metadata_regression.csv"
CLUSTER_CLASSIFICATION_CSV_FILE = RESULT_FOLDER + "cluster_metadata_classification.csv"

COLUMNS = ['image_path', 'label', 'R','G','B', 'num_pixels', 'cX', 'cY', 'ground_truth', 'predicted_value']