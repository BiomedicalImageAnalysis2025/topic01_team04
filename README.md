# Read.me Topic01 - Team04: Biomedical Image Analysis - Implementation and evaluation of clustering methods
# Titel: Biocluster - A K-means approach
### Data Analysis project SS25, B.Sc.Molekulare Biotechnologie

#### **Supervisor:** Prof. Dr. Karl Rohr, Dr. Leonid Kostrykin

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
6. Continue with [FilterEvaluation.ipynb](Code/FilterEvaluation.ipynb)
7. Continue with [CellDistiction.ipynb](Code/CellDistinction.ipynb)				 
8. End      with [Final_Dice_Score.ipynb](Code/Final_Dice_Score.ipynb)


### 1. Datenvorbereitung
Here, the images were prepared for further use. The images were normalized, a z-transformation was performed, and different filters were applied to the images.

### 2. Colormodels
To check if different color models have an effect on segmentation, the images (yeast cell and cell nuclei) were converted from the RGB color model to the HSV color model and separated into individual channels (H, S, V).

### 3. sklearn and Cellpose
To create a reference mask, KMeans clustering algorithms from sklearn were used to divide the images into different clusters to see if our self implemented kmeans works correct. In addition, Cellpose was used, a deep learning-based tool specifically designed for precise cell segmentation.

### 4. Otsu thresholding
For the N2DL HeLa cells, the Otsu method, an automatic threshold determination for image segmentation, was applied.

### 5. k-means
A self-implemented k-means algorithm was implemented and applied to the images as a segmentation method.

### 6. Filter Evaluation
Here we apply k-Means on t13 with different filters to later evaluate, using the dice score, which filter improves image segmentation. Moreover we apply k-Means on a thresholded HSV image to evaluate if reducing halos via thresholding improves image segmentation.

### 7. Cell distinction
Additional features (coordinates) besides intensity were used to cluster N2DL HeLa image t13. Moreover we tried to find out how many clusters are "optimal" using the Elbow Method

### 8. Dice score
The Dice score was used to compare the self-segmented images with the reference masks to evaluate the accuracy of the segmentation.


## Used AI Tools:
Chat-GPT-4.1
DeepSeek R1

## References:
- Vassilvitskii, Sergei, and David Arthur. "k-means++: The advantages of careful seeding." Proceedings of the eighteenth annual ACM-SIAM
  symposium on Discrete algorithms. 2007.
- DSPA2: Data Science and Predictive Analytics (UMich HS650), VIII. Unsupervised Clustering.
- Stringer, C., Wang, T., Michaelos, M., & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. Nature    Methods, 18(1), 100-106.

## Installation of Packages

 **Virtual environment**  
   ```bash
   python -m venv .venv
   # macOS/Linux
   source .venv/bin/activate
   # Windows (PowerShell)
   .\.venv\Scripts\Activate.ps1

 
### Important Packages

pip install \
   numpy>=1.24.0,<2.0.0 \
   opencv-python>=4.7.0,<5.0.0 \
   matplotlib>=3.7.0,<4.0.0 \
   scikit-image>=0.21.0,<1.0.0 \
   scipy>=1.11.0,<2.0.0
   scikit-learn>=1.2.0,<2.0.0 \
   imageio>=2.27.0,<3.0.0 \
   cellpose>=2.2.0,<3.0.0 \
   seaborn>=0.12.0,<1.0.0 \
   pandas>=1.5.0,<2.0.0 \
   Pillow>=9.4.0,<11.0.0

































































