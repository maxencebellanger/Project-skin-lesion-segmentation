# Intro & Setup

This project was completed as part of a course titled [*Image and Pattern Recognition*](https://syllabus.emse.fr/cycle/ICM/gp/19). 


This project is divided into two parts: feature extraction from the dataset, followed by classification to determine whether a skin lesion is benign or indicative of melanoma.

## Environnement - setup

Use the file <em>Project_skin_lesion_segmentation_env.yml</em> to create the right environment

> conda create --file Project_skin_lesion_segmentation_env.yml

Activate it with 

> conda activate Project_Skin_lesion_segmentation


## Dataset

A dataset is provided, containing 200 images of skin lesions: 100 images labeled as benign (label 1) and 100 images labeled as melanoma (label 0), from which the following were extracted:

* 200 original images
* 200 mask of the segmented skin lesion
* 200 images of superpixels 

This dataset is stored in the *project_data* directory.

# Features extraction

Feature extraction:

* Geometrical features using the mask of the segmented skin lesion
* Color and texture features using the original image
* Average geometrical and color features within the skin lesion using the superpixel image

The extraction is performed by *extract_features.py* and the resulting dataset is stored in the *output* directory.

# Classification

Several models were tested, yielding various results:

* Neural networks models:
    * A dense neural network (using multiple activation functions)
    * A convolutionnal neural network (using multiple activation functions)
* Traditional machine learning models:
    * Support Vector Machine (SVM)
    * Decision Tree
    * Random Forest

These different models can be found in *classification_nn.ipynb* and *classification_traditional.ipynb*  
