# Masters Thesis: Data Scientific Approach to solving Occlusion in AR

### Author: Rishabh Saxena

### Objective : To create and compare algorithmic approaches to occlude a objects in long-distance images

### Dataset:

1. Create a folder "dataset" in the root directory of the project.

2. [Original Dataset](https://zenodo.org/record/2562396): Download the dataset from this [link](https://zenodo.org/record/2562396) for [OriginalImages.zip](https://zenodo.org/record/2562396/files/OriginalImages.zip?download=1) and [ValidationImages.zip](https://zenodo.org/record/2562396/files/ValidationImages.zip?download=1). Place the extracted folders in a folder named `dataset\original\` as: `dataset\original\OriginalImages\` and `dataset\original\ValidationImages\` respectively.

3. Bachelor Dataset: Included with the code. Create a folder `dataset\bachelor\` and place "images" as `dataset\bachelor\images\` and "labels" as `dataset\bachelor\labels\` respectively.

4. EDEN Dataset: Download RGB images from this [link](https://isis-data.science.uva.nl/hale/EDEN-samples/RGB.zip) and Depth images from this [link](https://isis-data.science.uva.nl/hale/EDEN-samples/Depth.zip). Create a folder `dataset\EDEN\` and place "RGB" as `dataset\EDEN\RGB\` and "Depth" as `dataset\EDEN\Depth\` respectively.

### Specific Requirements:

1. For mean-shift library, use follow the instructions in the following [repository](https://github.com/fjean/pymeanshift). The package has been included in the repository as pymeanshift.zip.

2. Set DATASET_FOLDER in constants.py to `dataset\\`

3. Set the RESULTS_FOLDER as `output\\analysis_data\\`

4. To install dependencies, run `pip install -r requirements.txt`

### Notes:

1. Output contains plots and analysis results
