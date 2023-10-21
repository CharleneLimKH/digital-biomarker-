import numpy as np
import scipy
import pandas as pd 
from scipy.stats import entropy 
from skimage.feature import graycomatrix

def get_perimeter(seg_mask):
    
    filter_kernel = np.array([[-1, -1, -1], 
                              [-1,  8, -1],
                              [-1, -1, -1]])
    edges = scipy.signal.convolve2d(seg_mask, filter_kernel) > 0
    perimeter = np.sum(edges[:,:])
    return perimeter

def get_area(seg_mask):
    area = np.sum(seg_mask[:,:])
    return area

def get_circularity(area, perimeter):
    circularity = 4* np.pi*area/(perimeter**2)
    return circularity

def get_mean(img, seg_mask): 
    
    tumor = img * seg_mask
    mean = np.mean(tumor)
    
    return mean

def get_variance(img, seg_mask): 
    tumor = img * seg_mask
    variance = np.var(tumor)
    
    return variance

def get_entropy(img, seg_mask):
    tumor = img * seg_mask
    hist, _ = np.histogram(tumor, bins = 256, range = (0, 1))
    probability_distribution = hist / hist.sum()
    img_entropy = entropy(probability_distribution)
    
    return img_entropy

def get_cooccurence_matrix(img, seg_mask):
    indices = np.where(seg_mask == 1)
    min_x = np.min(indices[0])
    max_x = np.max(indices[0])
    min_y = np.min(indices[1])
    max_y = np.max(indices[1])
    
    tumor = img*seg_mask
    region_of_interest = tumor[min_x:max_x,min_y:max_y]
    
    bins = np.linspace(start = 0, stop = 1, num = 200)
    
    digitized = np.digitize(region_of_interest, bins)

    co_occurence_matrix = graycomatrix(image = digitized,
                                        distances = [2],
                                        angles = [3*np.pi/4],
                                        levels = 201)

    return co_occurence_matrix

def get_cooccurence_entropy(co_occurence_matrix):
    hist, _ = np.histogram(co_occurence_matrix)
    probability_distribution = hist / hist.sum()
    co_occurence_entropy = entropy(probability_distribution)
    
    return co_occurence_entropy


def get_feature_df(imgs, masks, tumor_classification):
    areas = []
    perimeters = []
    circularities = []
    means = []
    variances = [] 
    entropies = []
    co_occ_entropies = []
    tumor_class = []
    
    for this_img, this_mask in zip(imgs, masks): # iterate through ultrasound images and masks and extract features
        areas.append(get_area(seg_mask = this_mask))
        perimeters.append(get_perimeter(seg_mask = this_mask))    
        circularities.append(get_circularity(areas[-1], perimeters[-1])) # use the last calculated area and perimeter
        means.append(get_mean(this_img, this_mask))
        variances.append(get_variance(this_img,this_mask))
        entropies.append(get_entropy(this_img, this_mask))
        
        co_occurence_matrix = get_cooccurence_matrix(this_img, this_mask)[:,:,0,0]
        co_occurence_matrix[1,1] = 0
        co_occ_entropies.append(get_cooccurence_entropy(co_occurence_matrix))
        
        tumor_class.append(tumor_classification)
    
    # turn feature lists into dictionary and then into dataframe
    features_dict = {
        "area": areas, "perimeter": perimeters, "circularity": circularities,
        "mean": means, "variance": variances, "entropy": entropies,
        "co_occurence_entropy": co_occ_entropies,
        "diagnosis": tumor_class} 

    df = pd.DataFrame(features_dict)  
    
    return df