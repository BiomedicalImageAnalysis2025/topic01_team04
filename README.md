# Read.me Topic01 - Team04: Biomedical Image Analysis - Implementation and evaluation of clustering methods
# Titel: Biocluster - A K-means approach
### Data Analysis project SS25, B.Sc.Molekulare Biotechnologie

#### **Supervisor:** Prof. Dr. Karl Rohr

#### **Teams:** David Lehmann, David Dulkies, Jonas Schenker, David Schroth

#### **Tutor:** Bastian Mucha


## Introduction
This project focuses on image segmentation using K-Means and other clustering methods.
Segmentation techniques are powerful tools for accurately identifying and separating different parts of an image.
We apply these methods to RGB images of cell nuclei and yeast cells, as well as grayscale images of HeLa cells, in order to highlight and analyze their structures in detail.
This approach is particularly useful in biological and medical research, where precise analysis of cellular images plays a crucial role.
To improve the clustering results, we applied several preprocessing techniques such as normalization and filtering to enhance image quality and prepare the data for segmentation.
We also experimented with different color models, including HSV, to evaluate whether they help the clustering algorithm better distinguish key image features.
As part of the project, we developed a custom K-Means algorithm from scratch and compared its performance to both the scikit-learn implementation of K-Means and Otsu thresholding, a classic image segmentation method.

## Data Sets:
For this project the data consists of three different datasets, showing cell nuclei, yeast cells and N2DL HeLa Cells. 
A ground truth is only available for the HeLa cells images.
For the Yeast cell image, we generated an artificial ground truth using an AI-based segmentation method with CellPose. 

Datasets can be viewed here: https://bmcv-nas01.bioquant.uni-heidelberg.de:5001/sharing/Dkx0iQWOP


## structure of the repository
Our repository has a modular structure and is organized by topic. Each subtopic has its own Jupyter Notebook and an associated Python file.
The Jupyter Notebooks contain the implementation and visualization of the individual topics with accompanying explanations.
The Python file contains the reusable functions that were used for the corresponding Jupyter Notebook. These files are stored in the functions folder.
In addition, there is also a separate folder containing all saved images in all different variations and applications.

## How to use the Repository

- Only Via Jupyter Notebooks - 

1. Start    with [Datenvorbereitung.ipynb](Code/Datenvorbereitung.ipynb)
2. Continue with [Colormodels.ipynb](Code/Colormodels.ipynb)
3. Continue with [sklearn_clustering_cellpose.ipynb](code/sklearn_clustering_cellpose.ipynb)
4. Continue with [Otsu_thresholding.ipynb](Code/Otsu_thresholding.ipynb)
5. Continue with [FinalKMeans.ipynb](Code/FinalKMeans.ipynb)
   Continue with [FilterEvaluation.ipynb](Code/FilterEvaluation.ipynb)
   Continue with [CellDistiction.ipynb](Code/CellDistinction.ipynb)				 
6. End      with [Final_Dice_Score.ipynb](Code/Final_Dice_Score.ipynb)


### 1. Datenvorbereitung
Here, the images were prepared for further use. The images were normalized, a z-transformation was performed, and different filters were applied to the images.

### 2. Colormodels
To check if different color models have an effect on segmentation, the images (yeast cell and cell nuclei) were converted from the RGB color model to the HSV color model and separated into individual channels (H, S, V).

### 3. k-means
A self-implemented k-means algorithm was used here and applied to the images as a segmentation method.

### 4. Otsu thresholding
For the N2DL HeLa cells, the Otsu method, an automatic threshold determination for image segmentation, was applied.

### 5. Dice score
The Dice score was used to compare the self-segmented images with the ground truth to evaluate the accuracy of the segmentation.

### 6. Cell distinction
??????

### 7. Filter Evaluation
?????

### 8. sklearn
?????




































































hhhhhhhhhh