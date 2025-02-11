# -*- coding: utf-8 -*-
import os
import glob
from sklearn.preprocessing import LabelEncoder
from skimage.io import imread
from skimage import util
from skimage.measure import label, regionprops, block_reduce
from skimage.feature import local_binary_pattern
import numpy as np
import h5py
import pandas
import scipy as scp

def check_if_directory_exists(name_folder):
    """
    check_if_directory_exists(name_folder)
    INPUT:
        name_folder: name of the directory to be checked
    OUTPUT:
        a message indicating that the directory does not exist and if it is
        created

    @author: Eduardo Fidalgo (EFF)
    """

    if not os.path.exists(name_folder):
        print(name_folder + " directory does not exist, created")
        os.makedirs(name_folder)
    else:
        print(name_folder + " directory exists, no action performed")


def geometrical_descriptors(image):
    """    
    Parameters
    ----------
    image_segmentation: ubyte array
        Black and white image with the region that is going to be described

    Returns
    -------
    Vector with the descriptor of the input image
    """

    print("[INFO] \t Calculating geometrical descriptors ...")
    
    def fractal_dimension(Z):
        """    
        Parameters
        ----------
        image_segmentation: ubyte array
            Black and white image with the region that is going to be described
    
        Returns
        -------
        Fractal dimension of the image (code found on campus)
        """

        def boxcount(Z, k):
            S = np.add.reduceat(
                np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                np.arange(0, Z.shape[1], k), axis=1)
            return len(np.where((S > 0) & (S < k * k))[0])

        Z = Z < 1
        p = min(Z.shape)
        n = 2**np.floor(np.log2(p))
        n = int(np.log2(n))
        sizes = 2**np.arange(n, 0, -1)
        counts = [boxcount(Z, size) for size in sizes]
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]

    shape_features = np.empty(shape=9)
    properties = regionprops(image)[0]

    shape_features[0] = properties.area_convex
    shape_features[1] = properties.axis_major_length
    shape_features[2] = properties.axis_minor_length
    shape_features[3] = properties.eccentricity
    shape_features[4] = properties.equivalent_diameter_area
    shape_features[5] = properties.feret_diameter_max
    shape_features[6] = properties.perimeter
    shape_features[7] = properties.perimeter_crofton
    shape_features[8] = fractal_dimension(image)

    print("[INFO] \t ... Finished")


    return shape_features

def image_original_descriptors(image, roi, P=16, R=2):
    """    
    Parameters
    ----------
    image:  int array
        RGB image that is going to be described

    Returns
    -------
    Vector with the descriptor of the input image
    """

    print("[INFO] \t Calculating image original descriptors ...")

    features = np.array([])

    mean_rgb = [0,0,0]
    std_rgb = [0,0,0]

    # Mean RGB value of ROI
    mean_rgb[0] = np.mean(image[::,::,0], where=roi)
    mean_rgb[1] = np.mean(image[::,::,1], where=roi)
    mean_rgb[2] = np.mean(image[::,::,2], where=roi)

    # std RGB value of ROI
    std_rgb[0] = np.std(image[::,::,0], where=roi)
    std_rgb[1] = np.std(image[::,::,1], where=roi)
    std_rgb[2] = np.std(image[::,::,2], where=roi)

    features = np.concatenate((features, mean_rgb))
    features = np.concatenate((features, std_rgb))

    # RGB LBP on ROI
    for i in range(3):
        lbp = local_binary_pattern(image[:,:,i], P, R, "uniform")
        n_bins = 256
        hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))

        hist[0] = hist[0] - (image.shape[0] * image.shape[1] - np.sum(roi))

        features = np.concatenate((features, hist))

    print("[INFO] \t ... Finished")

    return features


def image_superpixels_descriptors(image_superpixels, image_original, roi):
    """    
    Parameters
    ----------
    image_superpixels:  int array
        superpixels of the image that is going to be described

    image_original:  int array
        image that is going to be described    

    Returns
    -------
    Vector with the descriptor of the input image
    """

    print("[INFO] \t Calculating image superpixels descriptors ...")

    features = np.array([])

    # Getting superpixels in ROI and save the number of superpixels
    superpixels = image_superpixels[::,::,0] + image_superpixels[::,::,1] + image_superpixels[::,::,2]
    superpixels, nb_region = label(superpixels, return_num=True)
    features = np.concatenate((features, [nb_region]))

    superpixels_features = regionprops(superpixels)

    # Getting features of every superpixels in ROI
    shape_features = np.empty(shape=(8, nb_region))
    for i in range(nb_region):
        properties = superpixels_features[i]

        shape_features[0, i] = properties.area_convex
        shape_features[1, i] = properties.axis_major_length
        shape_features[2, i] = properties.axis_minor_length
        shape_features[3, i] = properties.eccentricity
        shape_features[4, i] = properties.equivalent_diameter_area
        shape_features[5, i] = properties.feret_diameter_max
        shape_features[6, i] = properties.perimeter
        shape_features[7, i] = properties.perimeter_crofton

    # Saving the average and std of these features
    features = np.concatenate((features, np.mean(shape_features, 1)))
    features = np.concatenate((features, np.std(shape_features, 1)))

    # Getting the weighted average of RGB value in the ROI based on the size of the superpixels

    weighted_rgb = np.zeros(3) 
    mean_rgb = np.zeros(3)
    #total_area = sum(region.area for region in superpixels_features)  
    total_area = np.sum(roi)

    for i, region in enumerate(superpixels_features):
        mask = superpixels == region.label 

        mean_rgb[0] = np.mean(image_original[:,:,0], where=mask, axis=(0,1))
        mean_rgb[1] = np.mean(image_original[:,:,1], where=mask, axis=(0,1))
        mean_rgb[2] = np.mean(image_original[:,:,2], where=mask, axis=(0,1))

        weight = region.area / total_area  

        weighted_rgb += mean_rgb * weight  

    features = np.concatenate((features, weighted_rgb))

    print("[INFO] \t ... Finished")

    return features


def get_features(image_original, image_segmentation, image_superpixels):
    """    
    Parameters
    ----------
    image_original:  int array
        RGB image that is going to be described (texture descriptors)

    image_segmentation: ubyte array
        Black and white image with the region that is going to be described (geometrical/morphological descriptors)

    image_superpixels: int array
        RGB image that is going to be described (intensity descriptors)

    Returns
    -------
    Vector with the descriptor of the input image
    """

    roi = image_segmentation > 0
    image_original = image_original * roi[:, :, np.newaxis]
    image_superpixels = image_superpixels * roi[:, :, np.newaxis]


    features = np.concatenate((geometrical_descriptors(image_segmentation), 
                             image_original_descriptors(image_original, roi),
                             image_superpixels_descriptors(image_superpixels, image_original, roi)))

    return features


def save_features():
    ### variables
    dir_base = "./project_data"
    dir_images = dir_base + "/images"
    dir_original = dir_images + "/original"
    dir_segmentation = dir_images + "/segmentation"
    dir_superpixels = dir_images + "/superpixels"
    dir_output = "Output"
    features_path = dir_output + "/features.h5"
    labels_path = dir_output + "/labels.h5"
    image_labels = pandas.read_csv(dir_base+"/label.csv")

    size_morphological_descriptors = 9
    size_image_original_descriptors = 6 + 3*256 #(mean + std of color + rgb-lbp) 
    size_image_superpixels_descriptors = 1 + 2*8 + 3 #(nb region + mean + std of geometrical descriptors of each region + 
                                                     # weighted average color of each region)

    # variables to hold features and labels
    X = np.empty((0, size_morphological_descriptors + size_image_original_descriptors + size_image_superpixels_descriptors))
    Y = np.array([])

    for image_path in glob.glob(dir_original + "/*g"):
        # Getting id
        id = image_path.split("\\")[-1].split(".")[0]

        # Getting features
        print("[INFO] Processing image " + id)
        img_ori = imread(dir_original + "/" + id + ".jpg")
        img_segmentation = imread(dir_segmentation + "/" + id + "_segmentation.png")
        img_superpixel = imread(dir_superpixels + "/" + id + "_superpixels.png")

        img_segmentation = util.img_as_ubyte(img_segmentation)

        features = get_features(img_ori, img_segmentation, img_superpixel)

        # Saving features
        print("[INFO] ...Storing desctiptors of image " + id)
        label = image_labels.loc[image_labels["image_id"] == id, "melanoma"].iat[0]

        X = np.append(X, np.array([np.transpose(features)]), axis=0)
        Y = np.append(Y, 1-label)

        print("[INFO] Finished")

    print("\n")
    print("\n")
    print("[INFO] Saving descriptors in folder " + dir_output)

    # Save the features and labels into a hdf5 file in the directory dir_output
    # If the directory does not exist, create it
    check_if_directory_exists(dir_output)

    # Save features and labels
    try:
        h5f_data = h5py.File(features_path, 'w')
    except:
        a = 1

    h5f_data.create_dataset("dataset_skin_lesion", data=X)

    h5f_label = h5py.File(labels_path, 'w')
    h5f_label.create_dataset("dataset_skin_lesion", data=Y)

    h5f_data.close()
    h5f_label.close()


save_features()