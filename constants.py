THRESHOLD = 0.6
SPATIAL_RADIUS = 2
RANGE_RADIUS = 2
MIN_DENSITY = 150
PROCESSING_SHAPE = (256,256)

#############################################################################################################################################################

#Set this according to the place where the dataset is stored
DATASET_FOLDER = "C:\\Users\\cjbla\\OneDrive\\Desktop\\Code\\data\\dataset\\"  #set this to the folder where the dataset is stored
RESULT_FOLDER = "C:\\Users\\cjbla\\OneDrive\\Desktop\\Code\\masterthesis\\output\\analysis_data\\" #set this to the folder where the results should be stored

RELATIVE_IMAGES_CSV = "relative_image_paths_main.csv"
RELATIVE_IMAGES_CSV_BD = "relative_image_paths_bd.csv"

#############################################################################################################################################################

ORIGINAL_IMAGES_FOLDER = DATASET_FOLDER + "original\\OriginalImages\\"
LABEL_FOLDER = DATASET_FOLDER + "original\\ValidationImages\\Skyfinder\\"

ORIGINAL_IMAGES_CSV = RESULT_FOLDER + "image_paths_original.csv"
BACHELOR_IMAGES_CSV = RESULT_FOLDER + "image_paths_bachelor.csv"

BASELINE_METRIC_CSV_FILE = RESULT_FOLDER + "baseline_metrics.csv"
BASELINE_AUGMENTED_METRIC_CSV_FILE = RESULT_FOLDER + "baseline_augmented_metrics.csv"
BASELINE_TOPTHIRD_METRIC_CSV_FILE = RESULT_FOLDER + "baseline_topthird_metrics.csv"
BASELINE_TOPTHIRD_RR8_METRIC_CSV_FILE = RESULT_FOLDER + "baseline_topthird_rr8_metrics.csv"
BASELINE_TOPTHIRD_AUGMENTED_METRIC_CSV_FILE = RESULT_FOLDER + "baseline_topthird_augmented_metrics.csv"

CLUSTER_SEPARATED_METADATA_CSV_FILE = RESULT_FOLDER + "cluster_separated_metadata.csv"

EDEN_RGB_DIR = DATASET_FOLDER + "EDEN\\RGB\\"
EDEN_DEPTH_DIR = DATASET_FOLDER + "EDEN\\Depth\\"

COLUMNS = ['image_path', 'label', 'R','G','B', 'num_pixels', 'cX', 'cY', 'ground_truth', 'predicted_value']